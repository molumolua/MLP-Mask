"""GRPO trainer contract and route-aware balancing for antithetic MLP rollouts."""

from __future__ import annotations

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.seqlen_balancing import (
    calculate_workload,
    get_seqlen_balanced_partitions,
    log_seqlen_unbalance,
)

from .intervention import NEGATIVE_ROUTE, POSITIVE_ROUTE, TRAINING_ROUTES
from .routing import copy_prompt_uids_for_generation


class MLPChannelAntitheticTrainer(RayPPOTrainer):
    """Standard GRPO over one prompt group containing both symmetric routes."""

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        """Preserve prompt identity required by the synchronous rollout worker."""
        if "uid" not in batch.non_tensor_batch:
            raise RuntimeError("antithetic trainer batch is missing the original prompt uid")
        prompt_uids = copy_prompt_uids_for_generation(
            batch.non_tensor_batch["uid"], expected_size=len(batch)
        )

        gen_batch = super()._get_gen_batch(batch)
        if len(gen_batch) != len(prompt_uids):
            raise RuntimeError(
                "generation batch size changed while preserving antithetic prompt uid: "
                f"uids={len(prompt_uids)}, generation_rows={len(gen_batch)}"
            )
        gen_batch.non_tensor_batch["uid"] = prompt_uids
        return gen_batch

    def _validate_recipe_contract(self) -> None:
        config = self.config
        intervention = config.actor_rollout_ref.mlp_channel_antithetic
        if not intervention.get("enabled", False):
            raise ValueError("mlp_channel_antithetic.enabled must be true")
        strength = float(intervention.get("perturbation_strength", 0.10))
        if not 0.0 < strength < 1.0:
            raise ValueError("perturbation_strength must be in (0, 1)")
        if not intervention.get("refresh_every_step", True):
            raise ValueError("this recipe requires refresh_every_step=true")

        rollout_n = int(config.actor_rollout_ref.rollout.n)
        if rollout_n < 2 or rollout_n % 2:
            raise ValueError("rollout.n must be an even integer >= 2")
        if self.async_rollout_mode:
            raise NotImplementedError("antithetic MLP routes require synchronous rollout")
        if config.actor_rollout_ref.rollout.name != "vllm":
            raise NotImplementedError("antithetic MLP routes currently require vLLM")
        if config.actor_rollout_ref.rollout.mode != "sync":
            raise NotImplementedError("antithetic MLP routes require rollout.mode=sync")
        if int(config.actor_rollout_ref.rollout.pipeline_model_parallel_size) != 1:
            raise NotImplementedError("rollout pipeline_model_parallel_size must be 1")
        if int(config.actor_rollout_ref.rollout.data_parallel_size) != 1:
            raise NotImplementedError(
                "the first implementation requires vLLM data_parallel_size=1 so all "
                "repetitions of a prompt receive an exact paired route assignment"
            )
        if self.use_critic:
            raise NotImplementedError("this focused recipe supports outcome GRPO without a critic")
        if self.use_reference_policy:
            raise NotImplementedError(
                "reference KL is disabled until its forwards are conditioned on the same route"
            )
        if config.algorithm.adv_estimator != AdvantageEstimator.GRPO:
            raise NotImplementedError("this recipe currently supports only GRPO")
        if config.reward_model.launch_reward_fn_async:
            raise NotImplementedError("async reward functions are not supported")
        if not config.actor_rollout_ref.rollout.calculate_log_probs:
            raise ValueError("rollout.calculate_log_probs must be true")
        if config.algorithm.rollout_correction.bypass_old_logprob_for_rollout:
            raise ValueError(
                "bypass_old_logprob_for_rollout must be false so old log-probs are "
                "recomputed with each trajectory's perturbation route"
            )
        if not config.actor_rollout_ref.actor.use_rollout_log_probs:
            raise ValueError("actor.use_rollout_log_probs must be true")
        if not config.actor_rollout_ref.actor.get("force_on_policy", False):
            raise ValueError("actor.force_on_policy must be true")
        if int(config.actor_rollout_ref.actor.ppo_epochs) != 1:
            raise ValueError("actor.ppo_epochs must be 1")
        if not config.trainer.balance_batch:
            raise ValueError(
                "trainer.balance_batch must be true so every actor DP shard keeps "
                "equal positive and negative route quotas"
            )

    def _balance_batch(
        self,
        batch: DataProto,
        metrics,
        logging_prefix="global_seqlen",
        keep_minibatch=False,
    ) -> None:
        """Token-balance actor shards while retaining equal +/- row counts."""
        if "route_id" not in batch.non_tensor_batch:
            return super()._balance_batch(
                batch,
                metrics,
                logging_prefix=logging_prefix,
                keep_minibatch=keep_minibatch,
            )
        if keep_minibatch:
            raise NotImplementedError(
                "route-aware antithetic balancing does not support keep_minibatch=true"
            )

        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        workloads = calculate_workload(attention_mask.view(batch_size, -1).sum(-1))
        workloads = [int(value) for value in workloads]

        if "actor" not in self.actor_rollout_wg._dispatch_info:
            self.actor_rollout_wg._dispatch_info["actor"] = (
                self.actor_rollout_wg._query_dispatch_info("actor")
            )
        actor_dp_rank_mapping = self.actor_rollout_wg._dispatch_info["actor"]
        dp_size = max(actor_dp_rank_mapping) + 1
        route_values = np.asarray(batch.non_tensor_batch["route_id"], dtype=object)
        unknown = set(str(value) for value in route_values) - set(TRAINING_ROUTES)
        if unknown:
            raise RuntimeError(f"unknown antithetic routes: {sorted(unknown)}")

        partitions: list[list[int]] = [[] for _ in range(dp_size)]
        expected_per_rank: dict[str, int] = {}
        for route in TRAINING_ROUTES:
            route_indices = np.flatnonzero(route_values == route)
            if route_indices.size % dp_size:
                raise RuntimeError(
                    f"{route} rollout count {route_indices.size} must be divisible "
                    f"by actor DP size {dp_size}"
                )
            expected_per_rank[route] = int(route_indices.size // dp_size)
            route_partitions = get_seqlen_balanced_partitions(
                [workloads[int(index)] for index in route_indices],
                k_partitions=dp_size,
                equal_size=True,
            )
            for rank, local_indices in enumerate(route_partitions):
                partitions[rank].extend(
                    int(route_indices[local_index]) for local_index in local_indices
                )

        for rank, partition in enumerate(partitions):
            partition.sort(key=lambda index: (workloads[index], index))
            partitions[rank] = partition[::2] + partition[1::2][::-1]
            shard_routes = route_values[partitions[rank]]
            for route in TRAINING_ROUTES:
                observed = int(np.sum(shard_routes == route))
                if observed != expected_per_rank[route]:
                    raise RuntimeError(
                        f"rank {rank} received {observed} {route} rows; "
                        f"expected {expected_per_rank[route]}"
                    )

        batch.reorder(
            torch.tensor(
                [index for partition in partitions for index in partition],
                dtype=torch.long,
            )
        )
        metrics.update(
            log_seqlen_unbalance(workloads, partitions, prefix=logging_prefix)
        )
        metrics["route/positive_samples_per_actor_dp_rank"] = float(
            expected_per_rank[POSITIVE_ROUTE]
        )
        metrics["route/negative_samples_per_actor_dp_rank"] = float(
            expected_per_rank[NEGATIVE_ROUTE]
        )
        versions = np.unique(
            np.asarray(batch.non_tensor_batch["perturbation_version"], dtype=np.int64)
        )
        if versions.size != 1:
            raise RuntimeError(f"training batch mixes perturbation versions: {versions.tolist()}")
        metrics["mlp_antithetic/version"] = float(versions[0])
        strength = float(
            self.config.actor_rollout_ref.mlp_channel_antithetic.perturbation_strength
        )
        one_bf16 = torch.ones((), dtype=torch.bfloat16)
        effective_bf16 = float(
            ((one_bf16 + torch.tensor(strength, dtype=torch.bfloat16)) - one_bf16)
            .to(torch.float32)
            .item()
        )
        metrics["mlp_antithetic/strength"] = strength
        metrics["mlp_antithetic/effective_strength_bfloat16"] = effective_bf16

    def fit(self):
        self._validate_recipe_contract()
        return super().fit()
