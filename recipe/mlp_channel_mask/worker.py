"""Recipe-specific FSDP actor/rollout worker.

Two vLLM generations are executed under one actor->rollout weight synchronization:
clean first, then masked.  The prefix cache is reset between routes because their
hidden states and KV tensors are not interchangeable.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.distributed as dist

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.device import get_device_id, get_torch_device
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.profiler.performance import reduce_timing
from verl.utils.ray_utils import get_event_loop
from verl.workers.fsdp_workers import ActorRolloutRefWorker

from .intervention import (
    CLEAN_ROUTE,
    MASKED_ROUTE,
    MLPChannelInterventionController,
    install_hf_mlp_intervention,
    install_vllm_class_intervention,
    install_vllm_mlp_intervention,
)

_MASK_STATE_FILE = "mlp_channel_mask.pt"


class MLPChannelActorRolloutRefWorker(ActorRolloutRefWorker):
    """Dense-Qwen FSDP/vLLM worker with synchronized structured MLP routes."""

    def _intervention_config(self):
        config = self.config.get("mlp_intervention", None)
        if config is None or not config.get("enabled", False):
            raise RuntimeError("MLPChannelActorRolloutRefWorker requires mlp_intervention.enabled=true")
        return config

    def _new_controller(self, *, name: str, tp_rank: int = 0, tp_size: int = 1):
        config = self._intervention_config()
        return MLPChannelInterventionController(
            num_layers=int(self.actor_model_config.num_hidden_layers),
            intermediate_size=int(self.actor_model_config.intermediate_size),
            mask_ratio=float(config.get("mask_ratio", 0.10)),
            activation_ema_beta=float(config.get("activation_ema_beta", 0.95)),
            relative_activation_epsilon=float(
                config.get("relative_activation_epsilon", 1e-6)
            ),
            selection_strategy=str(
                config.get("selection_strategy", "soft_top")
            ),
            score_method=str(config.get("score_method", "relative_activation")),
            score_ema_beta=float(config.get("score_ema_beta", 0.0)),
            random_seed=int(config.get("random_seed", 42)),
            random_scope=str(config.get("random_scope", "per_layer")),
            weighted_max_ratio=float(config.get("weighted_max_ratio", 4.0)),
            weighted_rank_power=float(config.get("weighted_rank_power", 2.0)),
            tp_rank=tp_rank,
            tp_size=tp_size,
            name=name,
        )

    def _build_rollout(self, trust_remote_code=False):
        if self.config.rollout.name != "vllm" or self.config.rollout.mode != "sync":
            raise NotImplementedError("MLP-channel intervention currently supports only synchronous vLLM rollout")
        infer_tp = int(self.config.rollout.tensor_model_parallel_size)
        # ActorRolloutRefWorker creates the rollout mesh inside super().  Global rank
        # modulo TP is the local TP rank for the external-launcher layout used here.
        tp_rank = int(self.rank % infer_tp)
        self.rollout_mlp_controller = self._new_controller(
            name="vllm_rollout",
            tp_rank=tp_rank,
            tp_size=infer_tp,
        )
        # This must happen before LLM(...) constructs and captures the model.
        install_vllm_class_intervention(self.rollout_mlp_controller)
        super()._build_rollout(trust_remote_code=trust_remote_code)

        rollout_model = (
            self.rollout.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
        )
        # super()._build_rollout() returns with vLLM in sleep mode.  The masks were
        # installed by the class patch while the engine was awake, so this walk must
        # only validate/reuse those buffers and must not write their CUDA storage.
        install_vllm_mlp_intervention(rollout_model, self.rollout_mlp_controller)

    async def rollout_mode(self):
        await super().rollout_mode()
        # Weight/KV storage has been resumed, so deferred checkpoint, refresh, or
        # route changes can now be copied into the stable graph-captured buffers.
        self.rollout_mlp_controller.set_active_buffers_available(True)

    async def trainer_mode(self):
        # release() suspends vLLM-owned CUDA allocations.  Block controller writes
        # before that happens; CPU mask state may continue to change while asleep.
        self.rollout_mlp_controller.set_active_buffers_available(False)
        await super().trainer_mode()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if not self._is_actor:
            return
        self.actor_mlp_controller = self._new_controller(name="fsdp_actor")
        actor_model = getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        install_hf_mlp_intervention(actor_model, self.actor_mlp_controller)
        self.actor.intervention_controller = self.actor_mlp_controller

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    def generate_dual_sequences(self, prompts: DataProto) -> DataProto:
        """Generate clean and masked rows with one rollout-mode/weight sync."""
        assert self._is_rollout and self._is_actor
        prompts = prompts.to(get_device_id())
        prompts.meta_info.update(
            {
                "eos_token_id": self.generation_config.eos_token_id
                if self.generation_config is not None
                else self.tokenizer.eos_token_id,
                "pad_token_id": self.generation_config.pad_token_id
                if self.generation_config is not None
                else self.tokenizer.pad_token_id,
            }
        )
        route_values = np.asarray(prompts.non_tensor_batch.get("route_id", []), dtype=object)
        mask_versions = np.asarray(prompts.non_tensor_batch.get("mask_version", []), dtype=np.int64)
        unique_versions = np.unique(mask_versions)
        if unique_versions.tolist() != [self.rollout_mlp_controller.mask_version]:
            raise RuntimeError(
                "rollout mask version mismatch: "
                f"batch={unique_versions.tolist()}, worker={self.rollout_mlp_controller.mask_version}"
            )
        clean_indices = np.flatnonzero(route_values == CLEAN_ROUTE)
        masked_indices = np.flatnonzero(route_values == MASKED_ROUTE)
        if clean_indices.size == 0 or masked_indices.size == 0:
            raise RuntimeError(
                f"every rollout shard must contain both routes, got clean={clean_indices.size}, "
                f"masked={masked_indices.size}"
            )

        timings: dict[str, float] = {}
        enter_started = time.perf_counter()
        loop = get_event_loop()
        loop.run_until_complete(self.rollout_mode())
        timings["dual_rollout_weight_sync"] = time.perf_counter() - enter_started

        try:
            reset_started = time.perf_counter()
            # The previous operation may have left clean or masked KV blocks.
            # Start every dual rollout from an empty, route-neutral cache while
            # still allowing reuse among duplicate prompts inside one route.
            self.rollout.inference_engine.reset_prefix_cache()
            timings["mlp_prefix_cache_reset_before_clean"] = time.perf_counter() - reset_started

            switch_started = time.perf_counter()
            self.rollout_mlp_controller.set_route(CLEAN_ROUTE)
            timings["mlp_mask_switch_rollout_clean"] = time.perf_counter() - switch_started
            clean_started = time.perf_counter()
            clean_output = self.rollout.generate_sequences(prompts=prompts.select_idxs(clean_indices))
            timings["gen_clean"] = time.perf_counter() - clean_started

            reset_started = time.perf_counter()
            # Prefix-cache keys do not encode our forward route.  Reusing clean KV
            # for masked generation would silently invalidate the intervention.
            self.rollout.inference_engine.reset_prefix_cache()
            timings["mlp_prefix_cache_reset_between_routes"] = time.perf_counter() - reset_started

            switch_started = time.perf_counter()
            self.rollout_mlp_controller.set_route(MASKED_ROUTE)
            timings["mlp_mask_switch_rollout_masked"] = time.perf_counter() - switch_started
            masked_started = time.perf_counter()
            masked_output = self.rollout.generate_sequences(prompts=prompts.select_idxs(masked_indices))
            timings["gen_masked"] = time.perf_counter() - masked_started

            reset_started = time.perf_counter()
            # Validation uses the clean route through the standard single-route
            # worker method.  Never leave masked KV blocks available to it.
            self.rollout.inference_engine.reset_prefix_cache()
            timings["mlp_prefix_cache_reset_after_masked"] = time.perf_counter() - reset_started

            output = DataProto.concat([clean_output, masked_output])
            order = np.asarray(output.non_tensor_batch["dual_rollout_order"], dtype=np.int64)
            output.reorder(torch.from_numpy(np.argsort(order)).to(dtype=torch.long))
        finally:
            exit_started = time.perf_counter()
            loop.run_until_complete(self.trainer_mode())
            timings["dual_rollout_to_trainer"] = time.perf_counter() - exit_started

        timings = reduce_timing(timings)
        output.meta_info["timing"] = timings
        output = output.to("cpu")
        get_torch_device().empty_cache()
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    def generate_sequences(self, prompts: DataProto):
        """All single-route calls (validation) are forced to clean inference."""
        switch_started = time.perf_counter()
        self.rollout_mlp_controller.set_route(CLEAN_ROUTE)
        switch_elapsed = time.perf_counter() - switch_started
        output = super().generate_sequences(prompts)
        # DataProto.concat requires non-metric meta_info to be identical across
        # rollout workers.  The parent method reduces its generation timings for
        # that reason, so reduce the recipe-specific timing before adding it too.
        switch_timing = reduce_timing({"mlp_mask_switch_rollout_clean": switch_elapsed})
        output.meta_info.setdefault("timing", {}).update(switch_timing)
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def refresh_mlp_mask(self):
        assert self._is_actor
        result = self.actor_mlp_controller.refresh_mask()
        sync_started = time.perf_counter()
        self.rollout_mlp_controller.copy_mask_state_from(self.actor_mlp_controller)
        timings = dict(result.timings)
        timings["mlp_mask_sync_rollout"] = time.perf_counter() - sync_started
        return {
            "metrics": result.metrics,
            "timings": timings,
            "mask_version": self.actor_mlp_controller.mask_version,
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def observe_mlp_causal_effect(self, reward_gap_clean_minus_masked: float):
        """Feed one realized group-ablation outcome to the online estimator."""
        assert self._is_actor
        return self.actor_mlp_controller.observe_causal_ablation(
            reward_gap_clean_minus_masked
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_mlp_mask_status(self):
        return {
            "metrics": self.actor_mlp_controller.metrics(),
            "mask_version": self.actor_mlp_controller.mask_version,
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        super().save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)
        if dist.get_rank() == 0:
            os.makedirs(local_path, exist_ok=True)
            torch.save(self.actor_mlp_controller.state_dict(), os.path.join(local_path, _MASK_STATE_FILE))
        dist.barrier()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        state = None
        if local_path is not None:
            state_path = os.path.join(local_path, _MASK_STATE_FILE)
            if os.path.exists(state_path):
                state = torch.load(state_path, map_location="cpu", weights_only=False)
        super().load_checkpoint(local_path, hdfs_path, del_local_after_load)
        if state is not None:
            self.actor_mlp_controller.load_state_dict(state)
            self.rollout_mlp_controller.load_state_dict(state)
        aggressive_empty_cache(force_sync=True)
