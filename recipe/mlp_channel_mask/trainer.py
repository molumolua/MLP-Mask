"""Independent GRPO trainer for clean/masked MLP-channel rollouts."""

from __future__ import annotations

import uuid
from pprint import pprint

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_advantage, compute_response_mask
from verl.trainer.ppo.reward import compute_reward
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.seqlen_balancing import (
    calculate_workload,
    get_seqlen_balanced_partitions,
    log_seqlen_unbalance,
)

from .intervention import (
    CLEAN_ROUTE,
    GLOBAL_RANDOM_SCOPE,
    MASKED_ROUTE,
    PER_LAYER_RANDOM_SCOPE,
    RANDOM_SELECTION,
    TOP_RELATIVE_ACTIVATION_SELECTION,
    WEIGHTED_RANDOM_SELECTION,
)


class MLPChannelMaskTrainer(RayPPOTrainer):
    """Synchronous, outcome-GRPO trainer with two conditional policy routes."""

    def _validate_recipe_contract(self) -> None:
        config = self.config
        intervention = config.actor_rollout_ref.mlp_intervention
        n_clean = int(intervention.n_clean)
        n_masked = int(intervention.n_masked)
        if n_clean < 2 or n_masked < 2:
            raise ValueError("GRPO needs at least two samples for each of clean and masked routes")
        if n_clean + n_masked != int(config.actor_rollout_ref.rollout.n):
            raise ValueError(
                "rollout.n must equal mlp_intervention.n_clean + mlp_intervention.n_masked "
                "so actor batch normalization uses the true total trajectory count"
            )
        if self.async_rollout_mode:
            raise NotImplementedError("dual MLP routes currently require synchronous rollout")
        if config.actor_rollout_ref.rollout.name != "vllm":
            raise NotImplementedError("dual MLP routes currently require vLLM")
        if int(config.actor_rollout_ref.rollout.pipeline_model_parallel_size) != 1:
            raise NotImplementedError("the recipe currently requires rollout pipeline_model_parallel_size=1")
        if int(config.actor_rollout_ref.rollout.data_parallel_size) != 1:
            raise NotImplementedError("the recipe currently requires vLLM data_parallel_size=1")
        if int(config.actor_rollout_ref.actor.ulysses_sequence_parallel_size) != 1:
            raise NotImplementedError(
                "relative-activation collection currently requires actor Ulysses SP size 1"
            )
        if self.use_critic:
            raise NotImplementedError("this recipe is outcome GRPO and intentionally has no critic")
        if self.use_reference_policy:
            raise NotImplementedError("reference KL is disabled until route-conditioned reference masks are added")
        if config.algorithm.adv_estimator != AdvantageEstimator.GRPO:
            raise NotImplementedError("this recipe currently supports only GRPO")
        if config.reward_model.launch_reward_fn_async:
            raise NotImplementedError("async reward functions are not supported by this focused recipe")
        if not config.actor_rollout_ref.rollout.calculate_log_probs:
            raise ValueError("vLLM calculate_log_probs must be true for route-correct behavior log-probs")
        if config.algorithm.rollout_correction.bypass_old_logprob_for_rollout:
            raise ValueError("bypass_old_logprob_for_rollout must be false so the actor recomputes old log-probs")
        if not config.actor_rollout_ref.actor.use_rollout_log_probs:
            raise ValueError("actor.use_rollout_log_probs must be true for exact clean/masked PPO ratios")
        if not config.actor_rollout_ref.actor.get("force_on_policy", False):
            raise ValueError("actor.force_on_policy must be true so every train step has one policy update")
        if int(config.actor_rollout_ref.actor.ppo_epochs) != 1:
            raise ValueError("actor.ppo_epochs must be 1 for forced-on-policy contribution statistics")
        refresh_freq = int(intervention.get("refresh_freq", config.trainer.test_freq))
        if refresh_freq != int(config.trainer.test_freq):
            raise ValueError("mlp_intervention.refresh_freq must equal trainer.test_freq in this recipe")
        selection_strategy = str(
            intervention.get("selection_strategy", TOP_RELATIVE_ACTIVATION_SELECTION)
        )
        if selection_strategy not in {
            TOP_RELATIVE_ACTIVATION_SELECTION,
            RANDOM_SELECTION,
            WEIGHTED_RANDOM_SELECTION,
        }:
            raise ValueError(f"unsupported mlp_intervention.selection_strategy={selection_strategy!r}")
        random_scope = str(intervention.get("random_scope", PER_LAYER_RANDOM_SCOPE))
        if random_scope not in {PER_LAYER_RANDOM_SCOPE, GLOBAL_RANDOM_SCOPE}:
            raise ValueError(f"unsupported mlp_intervention.random_scope={random_scope!r}")
        if random_scope == GLOBAL_RANDOM_SCOPE and selection_strategy != RANDOM_SELECTION:
            raise ValueError("mlp_intervention.random_scope=global requires selection_strategy=random")
        random_resample_every_step = bool(intervention.get("random_resample_every_step", False))
        if random_resample_every_step and selection_strategy not in {
            RANDOM_SELECTION,
            WEIGHTED_RANDOM_SELECTION,
        }:
            raise ValueError(
                "mlp_intervention.random_resample_every_step=true requires "
                "selection_strategy=random or weighted_random"
            )
        activation_update_every_step = bool(
            intervention.get("activation_update_every_step", False)
        )
        if activation_update_every_step and selection_strategy not in {
            TOP_RELATIVE_ACTIVATION_SELECTION,
            WEIGHTED_RANDOM_SELECTION,
        }:
            raise ValueError(
                "mlp_intervention.activation_update_every_step=true requires "
                "selection_strategy=top_relative_activation or weighted_random"
            )

    def _build_dual_route_batch(self, batch: DataProto, mask_version: int) -> DataProto:
        intervention = self.config.actor_rollout_ref.mlp_intervention
        n_clean = int(intervention.n_clean)
        n_masked = int(intervention.n_masked)

        clean = batch.repeat(repeat_times=n_clean, interleave=True)
        masked = batch.repeat(repeat_times=n_masked, interleave=True)
        clean_prompt_uid = np.asarray(clean.non_tensor_batch["uid"], dtype=object).copy()
        masked_prompt_uid = np.asarray(masked.non_tensor_batch["uid"], dtype=object).copy()
        clean.non_tensor_batch["prompt_uid"] = clean_prompt_uid
        masked.non_tensor_batch["prompt_uid"] = masked_prompt_uid
        clean.non_tensor_batch["uid"] = np.asarray(
            [f"{uid}::{CLEAN_ROUTE}" for uid in clean_prompt_uid], dtype=object
        )
        masked.non_tensor_batch["uid"] = np.asarray(
            [f"{uid}::{MASKED_ROUTE}" for uid in masked_prompt_uid], dtype=object
        )
        clean.non_tensor_batch["route_id"] = np.full(len(clean), CLEAN_ROUTE, dtype=object)
        masked.non_tensor_batch["route_id"] = np.full(len(masked), MASKED_ROUTE, dtype=object)
        clean.non_tensor_batch["mask_version"] = np.full(len(clean), mask_version, dtype=np.int64)
        masked.non_tensor_batch["mask_version"] = np.full(len(masked), mask_version, dtype=np.int64)
        clean.non_tensor_batch["loss_group_id"] = np.full(len(clean), CLEAN_ROUTE, dtype=object)
        masked.non_tensor_batch["loss_group_id"] = np.full(len(masked), MASKED_ROUTE, dtype=object)

        dual = DataProto.concat([clean, masked])
        order = _round_robin_indices(len(clean), len(masked))
        dual.reorder(torch.as_tensor(order, dtype=torch.long))
        dual.non_tensor_batch["dual_rollout_order"] = np.arange(len(dual), dtype=np.int64)
        dual.non_tensor_batch["loss_group_normalizer"] = np.full(len(dual), 2.0, dtype=np.float32)
        dual.non_tensor_batch["loss_multiplier"] = np.ones(len(dual), dtype=np.float32)
        return dual

    def _refresh_mask(self, metrics: dict, timing_raw: dict) -> int:
        outputs = self.actor_rollout_wg.refresh_mlp_mask()
        if not outputs:
            raise RuntimeError("MLP mask refresh returned no worker result")
        result = outputs[0]
        metrics.update(result["metrics"])
        timing_raw.update(result["timings"])
        return int(result["mask_version"])

    def _balance_dual_route_batch(self, batch: DataProto, metrics: dict) -> None:
        """Token-balance DP shards while preserving each route's per-rank quota."""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        workloads = calculate_workload(attention_mask.view(batch_size, -1).sum(-1))
        workloads = [int(value) for value in workloads]

        if "actor" not in self.actor_rollout_wg._dispatch_info:
            self.actor_rollout_wg._dispatch_info["actor"] = self.actor_rollout_wg._query_dispatch_info("actor")
        actor_dp_rank_mapping = self.actor_rollout_wg._dispatch_info["actor"]
        dp_size = max(actor_dp_rank_mapping) + 1
        route_values = np.asarray(batch.non_tensor_batch["route_id"], dtype=object)
        partitions: list[list[int]] = [[] for _ in range(dp_size)]
        expected_per_rank: dict[str, int] = {}

        for route in (CLEAN_ROUTE, MASKED_ROUTE):
            route_indices = np.flatnonzero(route_values == route)
            if route_indices.size % dp_size != 0:
                raise RuntimeError(
                    f"{route} rollout count {route_indices.size} must be divisible by actor DP size {dp_size}"
                )
            expected_per_rank[route] = int(route_indices.size // dp_size)
            route_workloads = [workloads[int(idx)] for idx in route_indices]
            route_partitions = get_seqlen_balanced_partitions(
                route_workloads,
                k_partitions=dp_size,
                equal_size=True,
            )
            for rank, local_indices in enumerate(route_partitions):
                partitions[rank].extend(int(route_indices[local_idx]) for local_idx in local_indices)

        # Keep every contiguous dispatch shard balanced and place shorter work at
        # both ends, matching the core trainer's bubble-reduction convention.
        for rank, partition in enumerate(partitions):
            partition.sort(key=lambda idx: (workloads[idx], idx))
            partitions[rank] = partition[::2] + partition[1::2][::-1]
            shard_routes = route_values[partitions[rank]]
            clean_count = int(np.sum(shard_routes == CLEAN_ROUTE))
            masked_count = int(np.sum(shard_routes == MASKED_ROUTE))
            if clean_count != expected_per_rank[CLEAN_ROUTE] or masked_count != expected_per_rank[MASKED_ROUTE]:
                raise RuntimeError(
                    f"route-aware balance produced rank {rank} clean={clean_count}, masked={masked_count}"
                )

        batch.reorder(torch.tensor([idx for partition in partitions for idx in partition], dtype=torch.long))
        metrics.update(log_seqlen_unbalance(workloads, partitions, prefix="global_seqlen"))
        metrics["route/clean_samples_per_actor_dp_rank"] = float(
            np.sum(route_values[partitions[0]] == CLEAN_ROUTE)
        )
        metrics["route/masked_samples_per_actor_dp_rank"] = float(
            np.sum(route_values[partitions[0]] == MASKED_ROUTE)
        )

    def fit(self):
        from verl.utils.tracking import Tracking

        self._validate_recipe_contract()
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        status = self.actor_rollout_wg.get_mlp_mask_status()[0]
        mask_version = int(status["mask_version"])
        current_mask_metrics = dict(status["metrics"])
        if self.global_steps > 0 and mask_version == 0:
            raise RuntimeError(
                "resumed a nonzero training step without mlp_channel_mask.pt; "
                "use a checkpoint produced by this recipe or start with resume_mode=disable"
            )

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial clean validation metrics: {val_metrics}")
            logger.log(data={**val_metrics, **current_mask_metrics}, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0
        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        intervention = self.config.actor_rollout_ref.mlp_intervention
        refresh_freq = int(intervention.get("refresh_freq", self.config.trainer.test_freq))
        warmup_steps = int(intervention.get("warmup_steps", 1))
        selection_strategy = str(
            intervention.get("selection_strategy", TOP_RELATIVE_ACTIVATION_SELECTION)
        )
        random_resample_every_step = bool(intervention.get("random_resample_every_step", False))
        activation_update_every_step = bool(
            intervention.get("activation_update_every_step", False)
        )
        weighted_random = selection_strategy == WEIGHTED_RANDOM_SELECTION
        # A weighted refresh performed after a checkpointed step prepares the mask
        # for the next step.  On resume, version==next global step preserves it.
        mask_prepared_for_current_step = (
            weighted_random
            and random_resample_every_step
            and mask_version == self.global_steps
        )

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics: dict[str, float] = {}
                timing_raw: dict[str, float] = {}
                metrics["mlp_mask/random_resample_every_step"] = float(random_resample_every_step)
                metrics["mlp_activation/update_every_step"] = float(
                    activation_update_every_step
                )
                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                if random_resample_every_step:
                    if mask_prepared_for_current_step:
                        mask_prepared_for_current_step = False
                    else:
                        # Sample before rollout and keep this version fixed for
                        # rollout, old-logprob, and the complete actor update.
                        with marked_timer("mlp_mask_resample_driver", timing_raw, color="cyan"):
                            mask_version = self._refresh_mask(metrics, timing_raw)
                        current_mask_metrics = {
                            key: value
                            for key, value in metrics.items()
                            if key.startswith("mlp_mask/")
                            and key != "mlp_mask/rollout_version_used"
                        }

                with marked_timer("dual_batch_build", timing_raw):
                    base_batch = DataProto.from_single_dict(batch_dict)
                    base_batch.non_tensor_batch["uid"] = np.asarray(
                        [str(uuid.uuid4()) for _ in range(len(base_batch.batch))], dtype=object
                    )
                    batch = self._build_dual_route_batch(base_batch, mask_version=mask_version)
                activation_refresh_due = self.global_steps == warmup_steps or (
                    refresh_freq > 0 and self.global_steps % refresh_freq == 0
                )
                activation_update_due = (
                    activation_update_every_step or activation_refresh_due
                )
                should_refresh = not random_resample_every_step and activation_update_due
                collect_mlp_activation = activation_update_due and selection_strategy in {
                    TOP_RELATIVE_ACTIVATION_SELECTION,
                    WEIGHTED_RANDOM_SELECTION,
                }
                # With every-step updates, step t's clean backward prepares the
                # relative-activation mask for step t+1.  Otherwise hooks are only
                # installed on the periodic refresh step.
                batch.meta_info["collect_mlp_activation"] = collect_mlp_activation
                metrics["mlp_mask/rollout_version_used"] = float(mask_version)
                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = self.global_steps
                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    with marked_timer("gen_dual", timing_raw, color="red"):
                        gen_batch_output = self.actor_rollout_wg.generate_dual_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info.pop("timing", {}))
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch:
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    if self.config.trainer.balance_batch:
                        with marked_timer("balance_dual_route", timing_raw):
                            self._balance_dual_route_batch(batch, metrics=metrics)
                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    with marked_timer("reward_dual", timing_raw, color="yellow"):
                        if self.use_rm and "rm_scores" not in batch.batch:
                            reward_scores = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_scores)
                        reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    from verl.trainer.ppo.rollout_corr_helper import (
                        compute_rollout_correction_and_add_to_batch,
                        maybe_apply_rollout_correction,
                    )

                    rollout_corr_config = self.config.algorithm.rollout_correction
                    need_recomputation = maybe_apply_rollout_correction(
                        batch=batch,
                        rollout_corr_config=rollout_corr_config,
                        policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                    )
                    if need_recomputation:
                        with marked_timer("old_log_prob_dual", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=batch.batch["response_mask"],
                                loss_agg_mode=self.config.actor_rollout_ref.actor.loss_agg_mode,
                            )
                            metrics["actor/entropy"] = entropy_agg.detach().item()
                            metrics.update(
                                compute_route_entropy_metrics(
                                    entropys=entropys,
                                    response_mask=batch.batch["response_mask"],
                                    route_values=batch.non_tensor_batch["route_id"],
                                    loss_agg_mode=self.config.actor_rollout_ref.actor.loss_agg_mode,
                                )
                            )
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                    if "old_log_probs" not in batch.batch:
                        raise RuntimeError("actor did not provide route-correct old_log_probs")

                    with marked_timer("adv_dual", timing_raw, color="brown"):
                        batch.batch["token_level_scores"] = reward_tensor
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update(
                                {key: np.asarray(value) for key, value in reward_extra_infos_dict.items()}
                            )
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                        if "rollout_log_probs" in batch.batch:
                            batch, correction_metrics = compute_rollout_correction_and_add_to_batch(
                                batch, rollout_corr_config
                            )
                            metrics.update(correction_metrics)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=self.config.algorithm.norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )
                        metrics.update(compute_route_metrics(batch))

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor_dual", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = False
                            # Keep the update RPC self-contained even though the
                            # preceding old-log-prob RPC also supplies this metadata.
                            batch.meta_info["temperature"] = float(
                                self.config.actor_rollout_ref.rollout.temperature
                            )
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))
                        if collect_mlp_activation:
                            # This RPC wall time includes the GPU hook work.  The
                            # controller's accumulate timing only measures host
                            # dispatch because CUDA kernels are asynchronous.
                            timing_raw["mlp_activation_enabled_actor_update"] = timing_raw[
                                "update_actor_dual"
                            ]

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                should_validate = (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                )
                if should_validate:
                    with marked_timer("testing_clean", timing_raw, color="green"):
                        val_metrics = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                metrics["mlp_activation/collection_enabled"] = float(
                    collect_mlp_activation
                )
                if should_refresh:
                    # Clean validation (when scheduled) intentionally happens before
                    # the new mask is selected; the refreshed mask starts next step.
                    with marked_timer("mlp_mask_refresh_driver", timing_raw, color="cyan"):
                        mask_version = self._refresh_mask(metrics, timing_raw)
                        current_mask_metrics = {
                            key: value
                            for key, value in metrics.items()
                            if key.startswith("mlp_mask/") and key != "mlp_mask/rollout_version_used"
                        }
                elif weighted_random and random_resample_every_step and collect_mlp_activation:
                    # Consume the newly collected relative activation before checkpointing and
                    # prepare the weighted mask for the next step.  The next loop
                    # must reuse it instead of sampling twice.
                    with marked_timer("mlp_mask_refresh_driver", timing_raw, color="cyan"):
                        mask_version = self._refresh_mask(metrics, timing_raw)
                    current_mask_metrics = {
                        key: value
                        for key, value in metrics.items()
                        if key.startswith("mlp_mask/") and key != "mlp_mask/rollout_version_used"
                    }
                    mask_prepared_for_current_step = True
                else:
                    metrics.update(current_mask_metrics)

                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                if self.config.trainer.save_freq > 0 and (
                    is_last_step
                    or self.global_steps % self.config.trainer.save_freq == 0
                    or esi_close_to_expiration
                ):
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                self.max_steps_duration = max(self.max_steps_duration, timing_raw["step"])
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                metrics.update(compute_data_metrics(batch=batch, use_critic=False))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                metrics.update(
                    compute_throughout_metrics(
                        batch=batch,
                        timing_raw=timing_raw,
                        n_gpus=self.resource_pool_manager.get_n_gpus(),
                    )
                )
                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final clean validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return


def _round_robin_indices(clean_size: int, masked_size: int) -> list[int]:
    indices: list[int] = []
    for offset in range(max(clean_size, masked_size)):
        if offset < clean_size:
            indices.append(offset)
        if offset < masked_size:
            indices.append(clean_size + offset)
    return indices


def compute_route_entropy_metrics(
    entropys: torch.Tensor,
    response_mask: torch.Tensor,
    route_values: np.ndarray,
    loss_agg_mode: str,
) -> dict[str, float]:
    """Aggregate entropy independently for clean and masked policy routes."""
    route_values = np.asarray(route_values, dtype=object)
    metrics: dict[str, float] = {}
    for route in (CLEAN_ROUTE, MASKED_ROUTE):
        idx_np = np.flatnonzero(route_values == route)
        if idx_np.size == 0:
            continue
        idx = torch.as_tensor(idx_np, device=entropys.device, dtype=torch.long)
        route_entropy = agg_loss(
            loss_mat=entropys.index_select(0, idx),
            loss_mask=response_mask.to(device=entropys.device).index_select(0, idx),
            loss_agg_mode=loss_agg_mode,
        )
        metrics[f"{route}_actor/entropy"] = float(route_entropy.detach().item())
    if "clean_actor/entropy" in metrics and "masked_actor/entropy" in metrics:
        metrics["route/entropy_gap_masked_minus_clean"] = (
            metrics["masked_actor/entropy"] - metrics["clean_actor/entropy"]
        )
    return metrics


def compute_route_metrics(batch: DataProto) -> dict[str, float]:
    """Exact driver-side route metrics after reward and advantage computation."""
    route_values = np.asarray(batch.non_tensor_batch["route_id"], dtype=object)
    metrics: dict[str, float] = {}
    for route in (CLEAN_ROUTE, MASKED_ROUTE):
        idx_np = np.flatnonzero(route_values == route)
        if idx_np.size == 0:
            continue
        idx = torch.as_tensor(idx_np, device=batch.batch.device, dtype=torch.long)
        response_mask = batch.batch["response_mask"].index_select(0, idx).float()
        token_count = response_mask.sum(dim=-1).clamp_min(1.0)
        reward = batch.batch["token_level_scores"].index_select(0, idx).sum(dim=-1)
        advantage = (
            batch.batch["advantages"].index_select(0, idx) * response_mask
        ).sum(dim=-1) / token_count
        prefix = f"{route}_actor"
        metrics.update(
            {
                f"{prefix}/samples": float(idx_np.size),
                f"{prefix}/reward_mean": float(reward.mean().item()),
                f"{prefix}/reward_std": float(reward.std(unbiased=False).item()),
                f"{prefix}/response_length_mean": float(token_count.mean().item()),
                f"{prefix}/response_length_max": float(token_count.max().item()),
                f"{prefix}/advantage_mean": float(advantage.mean().item()),
                f"{prefix}/advantage_std": float(advantage.std(unbiased=False).item()),
            }
        )
        for logprob_key in ("rollout_log_probs", "old_log_probs"):
            if logprob_key in batch.batch:
                logprob = batch.batch[logprob_key].index_select(0, idx)
                metrics[f"{prefix}/{logprob_key}_mean"] = float(
                    ((logprob * response_mask).sum() / response_mask.sum().clamp_min(1.0)).item()
                )
        for key, values in batch.non_tensor_batch.items():
            if key in {
                "route_id",
                "uid",
                "prompt_uid",
                "mask_version",
                "dual_rollout_order",
                "loss_group_id",
                "loss_group_normalizer",
                "loss_multiplier",
            }:
                continue
            array = np.asarray(values)
            if array.shape[:1] != (len(batch),) or array.dtype.kind not in "biuf":
                continue
            route_array = array[idx_np].astype(np.float64)
            metrics[f"{prefix}/{key}_mean"] = float(route_array.mean())
    if "clean_actor/reward_mean" in metrics and "masked_actor/reward_mean" in metrics:
        metrics["route/reward_gap_clean_minus_masked"] = (
            metrics["clean_actor/reward_mean"] - metrics["masked_actor/reward_mean"]
        )
    return metrics
