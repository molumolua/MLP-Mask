"""FSDP actor/vLLM rollout worker for paired MLP-channel perturbations."""

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
    NEGATIVE_ROUTE,
    NEUTRAL_ROUTE,
    POSITIVE_ROUTE,
    MLPChannelAntitheticController,
    install_hf_mlp_intervention,
    install_vllm_class_intervention,
    install_vllm_mlp_intervention,
)
from .routing import assign_antithetic_routes
from .reward_update import REWARD_UPDATE_METADATA, RewardDifferenceUpdater, RewardUpdateConfig

_STATE_FILE = "mlp_channel_antithetic.pt"


class MLPChannelAntitheticActorRolloutRefWorker(ActorRolloutRefWorker):
    """Run +epsilon and -epsilon generations under one weight synchronization."""

    def _intervention_config(self):
        config = self.config.get("mlp_channel_antithetic", None)
        if config is None or not config.get("enabled", False):
            raise RuntimeError(
                "MLPChannelAntitheticActorRolloutRefWorker requires "
                "mlp_channel_antithetic.enabled=true"
            )
        return config

    def _new_controller(self, *, name: str, tp_rank: int = 0, tp_size: int = 1):
        config = self._intervention_config()
        return MLPChannelAntitheticController(
            num_layers=int(self.actor_model_config.num_hidden_layers),
            intermediate_size=int(self.actor_model_config.intermediate_size),
            perturbation_strength=float(config.get("perturbation_strength", 0.10)),
            random_seed=int(config.get("random_seed", 42)),
            tp_rank=tp_rank,
            tp_size=tp_size,
            name=name,
        )

    def _build_rollout(self, trust_remote_code=False):
        if self.config.rollout.name != "vllm" or self.config.rollout.mode != "sync":
            raise NotImplementedError(
                "antithetic MLP-channel rollout currently supports synchronous vLLM only"
            )
        infer_tp = int(self.config.rollout.tensor_model_parallel_size)
        tp_rank = int(self.rank % infer_tp)
        self.rollout_mlp_controller = self._new_controller(
            name="vllm_rollout", tp_rank=tp_rank, tp_size=infer_tp
        )
        # The class must be patched before vLLM builds/captures the model.
        install_vllm_class_intervention(self.rollout_mlp_controller)
        super()._build_rollout(trust_remote_code=trust_remote_code)

        rollout_model = (
            self.rollout.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
        )
        install_vllm_mlp_intervention(rollout_model, self.rollout_mlp_controller)

    async def rollout_mode(self):
        await super().rollout_mode()
        self.rollout_mlp_controller.set_active_buffers_available(True)

    async def trainer_mode(self):
        self.rollout_mlp_controller.set_active_buffers_available(False)
        await super().trainer_mode()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if not self._is_actor:
            return
        self.actor_mlp_controller = self._new_controller(name="fsdp_actor")
        actor_model = getattr(
            self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp
        )
        install_hf_mlp_intervention(actor_model, self.actor_mlp_controller)
        self.actor.intervention_controller = self.actor_mlp_controller
        update_config = RewardUpdateConfig.from_config(
            self._intervention_config().get("reward_difference_update", None)
        )
        if update_config.active:
            from verl.utils.torch_dtypes import PrecisionType

            if not self.config.actor.get("force_on_policy", False) or int(self.config.actor.ppo_epochs) != 1:
                raise ValueError("reward difference updates require one complete-batch optimizer step")
            if self._is_lora:
                raise NotImplementedError("reward difference updates require full down-projection weights, not LoRA")
            mixed_precision = self.config.actor.fsdp_config.get("mixed_precision", None) or {}
            compute_dtype = PrecisionType.to_dtype(mixed_precision.get("param_dtype", "bf16"))
            if compute_dtype != PrecisionType.to_dtype(self.config.rollout.dtype):
                raise ValueError("reward difference updates require identical actor/rollout compute dtypes")
            self.reward_difference_updater = RewardDifferenceUpdater(
                self.actor_module_fsdp, self.actor_optimizer, self.actor_mlp_controller,
                update_config, compute_dtype=compute_dtype,
            )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_actor(self, data: DataProto):
        updater = getattr(self, "reward_difference_updater", None)
        if updater is None:
            return super().update_actor(data)
        updater.begin_batch(
            data.meta_info.get(REWARD_UPDATE_METADATA),
            data.non_tensor_batch["perturbation_version"],
        )
        try:
            # Optimizer hooks run inside the base worker's loaded-parameter and
            # Ulysses contexts, before scheduler/offload/checkpoint operations.
            output = super().update_actor(data)
            output.meta_info.setdefault("metrics", {}).update(updater.last_metrics)
            return output
        finally:
            updater.end_batch()

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        assert self._is_rollout and self._is_actor

        if bool(prompts.meta_info.get("validate", False)):
            switch_started = time.perf_counter()
            self.rollout_mlp_controller.set_route(NEUTRAL_ROUTE)
            switch_timing = reduce_timing(
                {
                    "mlp_antithetic_switch_rollout_neutral": (
                        time.perf_counter() - switch_started
                    )
                }
            )
            output = super().generate_sequences(prompts)
            output.meta_info.setdefault("timing", {}).update(switch_timing)
            return output

        if "uid" not in prompts.non_tensor_batch:
            raise RuntimeError("antithetic rollout batch is missing the original prompt uid")
        if "global_steps" not in prompts.meta_info:
            raise RuntimeError("antithetic rollout batch is missing global_steps")

        version = int(prompts.meta_info["global_steps"])
        self.actor_mlp_controller.refresh_direction(version=version)
        self.rollout_mlp_controller.refresh_direction(version=version)
        routes = assign_antithetic_routes(prompts.non_tensor_batch["uid"])
        prompts.non_tensor_batch["route_id"] = routes
        prompts.non_tensor_batch["perturbation_version"] = np.full(
            len(prompts), version, dtype=np.int64
        )
        prompts.non_tensor_batch["antithetic_rollout_order"] = np.arange(
            len(prompts), dtype=np.int64
        )

        positive_indices = np.flatnonzero(routes == POSITIVE_ROUTE)
        negative_indices = np.flatnonzero(routes == NEGATIVE_ROUTE)
        if positive_indices.size != negative_indices.size:
            raise RuntimeError(
                "antithetic rollout shard is unbalanced: "
                f"positive={positive_indices.size}, negative={negative_indices.size}"
            )

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

        timings: dict[str, float] = {}
        loop = get_event_loop()
        enter_started = time.perf_counter()
        loop.run_until_complete(self.rollout_mode())
        timings["antithetic_rollout_weight_sync"] = time.perf_counter() - enter_started

        try:
            reset_started = time.perf_counter()
            self.rollout.inference_engine.reset_prefix_cache()
            timings["mlp_prefix_cache_reset_before_positive"] = (
                time.perf_counter() - reset_started
            )

            switch_started = time.perf_counter()
            self.rollout_mlp_controller.set_route(POSITIVE_ROUTE)
            timings["mlp_antithetic_switch_rollout_positive"] = (
                time.perf_counter() - switch_started
            )
            generation_started = time.perf_counter()
            positive_output = self.rollout.generate_sequences(
                prompts=prompts.select_idxs(positive_indices)
            )
            timings["gen_positive"] = time.perf_counter() - generation_started

            reset_started = time.perf_counter()
            # Prefix-cache keys do not encode the perturbation route.
            self.rollout.inference_engine.reset_prefix_cache()
            timings["mlp_prefix_cache_reset_between_routes"] = (
                time.perf_counter() - reset_started
            )

            switch_started = time.perf_counter()
            self.rollout_mlp_controller.set_route(NEGATIVE_ROUTE)
            timings["mlp_antithetic_switch_rollout_negative"] = (
                time.perf_counter() - switch_started
            )
            generation_started = time.perf_counter()
            negative_output = self.rollout.generate_sequences(
                prompts=prompts.select_idxs(negative_indices)
            )
            timings["gen_negative"] = time.perf_counter() - generation_started

            reset_started = time.perf_counter()
            self.rollout.inference_engine.reset_prefix_cache()
            timings["mlp_prefix_cache_reset_after_negative"] = (
                time.perf_counter() - reset_started
            )

            output = DataProto.concat([positive_output, negative_output])
            original_order = np.asarray(
                output.non_tensor_batch["antithetic_rollout_order"], dtype=np.int64
            )
            output.reorder(
                torch.from_numpy(np.argsort(original_order)).to(dtype=torch.long)
            )
        finally:
            exit_started = time.perf_counter()
            loop.run_until_complete(self.trainer_mode())
            timings["antithetic_rollout_to_trainer"] = (
                time.perf_counter() - exit_started
            )

        timings = reduce_timing(timings)
        output.meta_info["timing"] = timings
        output = output.to("cpu")
        get_torch_device().empty_cache()
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_mlp_antithetic_status(self):
        return {
            "metrics": self.actor_mlp_controller.metrics(),
            "perturbation_version": self.actor_mlp_controller.perturbation_version,
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(
        self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None
    ):
        super().save_checkpoint(
            local_path, hdfs_path, global_step, max_ckpt_to_keep
        )
        if dist.get_rank() == 0:
            os.makedirs(local_path, exist_ok=True)
            torch.save(
                self.actor_mlp_controller.state_dict(),
                os.path.join(local_path, _STATE_FILE),
            )
        dist.barrier()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        state = None
        if local_path is not None:
            state_path = os.path.join(local_path, _STATE_FILE)
            if os.path.exists(state_path):
                state = torch.load(state_path, map_location="cpu", weights_only=False)
        super().load_checkpoint(local_path, hdfs_path, del_local_after_load)
        if state is not None:
            self.actor_mlp_controller.load_state_dict(state)
            self.rollout_mlp_controller.load_state_dict(state)
        aggressive_empty_cache(force_sync=True)
