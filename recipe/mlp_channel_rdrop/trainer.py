"""GRPO trainer contract and route-aware balancing for rdrop MLP rollouts."""

from __future__ import annotations

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.seqlen_balancing import (
    calculate_workload,
    get_seqlen_balanced_partitions,
    log_seqlen_unbalance,
)

from .intervention import MASK_B_ROUTE, MASK_A_ROUTE, TRAINING_ROUTES
from .routing import copy_prompt_uids_for_generation
from .validation import validate_recipe_config


class MLPChannelRDropTrainer(RayPPOTrainer):
    """Standard GRPO over one prompt group containing both independent masks."""

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        """Preserve prompt identity required by the synchronous rollout worker."""
        if "uid" not in batch.non_tensor_batch:
            raise RuntimeError("rdrop trainer batch is missing the original prompt uid")
        prompt_uids = copy_prompt_uids_for_generation(
            batch.non_tensor_batch["uid"], expected_size=len(batch)
        )

        gen_batch = super()._get_gen_batch(batch)
        if len(gen_batch) != len(prompt_uids):
            raise RuntimeError(
                "generation batch size changed while preserving rdrop prompt uid: "
                f"uids={len(prompt_uids)}, generation_rows={len(gen_batch)}"
            )
        gen_batch.non_tensor_batch["uid"] = prompt_uids
        return gen_batch

    def _validate_recipe_contract(self) -> None:
        validate_recipe_config(self.config)
        if self.use_critic or self.use_reference_policy:
            raise NotImplementedError("this recipe uses outcome GRPO without critic or reference KL")

    def _validate(self):
        metrics = super()._validate()
        per_rank = self.actor_rollout_wg.compute_parameter_update_metrics()
        if not per_rank:
            raise RuntimeError("parameter diagnostics returned no results")
        metrics.update(per_rank[0])
        return metrics

    def _balance_batch(
        self,
        batch: DataProto,
        metrics,
        logging_prefix="global_seqlen",
        keep_minibatch=False,
    ) -> None:
        """Token-balance actor shards while retaining equal A/B row counts."""
        if "route_id" not in batch.non_tensor_batch:
            return super()._balance_batch(
                batch,
                metrics,
                logging_prefix=logging_prefix,
                keep_minibatch=keep_minibatch,
            )
        if keep_minibatch:
            raise NotImplementedError(
                "route-aware rdrop balancing does not support keep_minibatch=true"
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
            raise RuntimeError(f"unknown rdrop routes: {sorted(unknown)}")

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
        metrics["route/mask_a_samples_per_actor_dp_rank"] = float(
            expected_per_rank[MASK_A_ROUTE]
        )
        metrics["route/mask_b_samples_per_actor_dp_rank"] = float(
            expected_per_rank[MASK_B_ROUTE]
        )
        versions = np.unique(
            np.asarray(batch.non_tensor_batch["mask_version"], dtype=np.int64)
        )
        if versions.size != 1:
            raise RuntimeError(f"training batch mixes perturbation versions: {versions.tolist()}")
        metrics["mlp_rdrop/mask_version"] = float(versions[0])

    def fit(self):
        self._validate_recipe_contract()
        return super().fit()

    def _prepare_actor_update(self, batch: DataProto, metrics) -> None:
        routes = np.asarray(batch.non_tensor_batch["route_id"], dtype=object)
        rewards = batch.batch["token_level_scores"].sum(-1).detach().cpu().numpy()
        for route in TRAINING_ROUTES:
            values = rewards[routes == route]
            metrics[f"mlp_rdrop/{route}_reward_mean"] = float(values.mean())
            metrics[f"mlp_rdrop/{route}_positive_reward_fraction"] = float((values > 0).mean())
