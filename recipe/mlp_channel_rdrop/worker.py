"""Two independent hard-mask rollouts and a shared R-Drop actor."""

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
    MASK_B_ROUTE,
    CLEAN_ROUTE,
    MASK_A_ROUTE,
    MLPChannelRDropController,
)
from .routing import assign_rdrop_routes
from .actor import RDropActorMixin
from verl.workers.actor.dp_actor import DataParallelPPOActor
from .diagnostics import ParameterUpdateTracker, SampledGradientTracker
from verl.utils.config import omega_conf_to_dataclass
from .backend import (install_hf_mlp_intervention, install_vllm_class_intervention,
                      install_vllm_mlp_intervention)

_STATE_FILE = "mlp_channel_rdrop.pt"


class MLPChannelRDropActor(RDropActorMixin, DataParallelPPOActor):
    pass


class MLPChannelRDropActorRolloutRefWorker(ActorRolloutRefWorker):
    """Generate both masked routes under one weight synchronization."""

    def _intervention_config(self):
        config = self.config.get("mlp_channel_rdrop", None)
        if config is None or not config.get("enabled", False):
            raise RuntimeError(
                "MLPChannelRDropActorRolloutRefWorker requires "
                "mlp_channel_rdrop.enabled=true"
            )
        return config

    def _new_controller(self, *, name: str, tp_rank: int = 0, tp_size: int = 1):
        config = self._intervention_config()
        return MLPChannelRDropController(
            num_layers=int(self.actor_model_config.num_hidden_layers),
            intermediate_size=int(self.actor_model_config.intermediate_size),
            mask_ratio=float(config.get("mask_ratio", 0.10)),
            random_seed=int(config.get("random_seed", 42)),
            tp_rank=tp_rank,
            tp_size=tp_size,
            name=name,
        )

    def _build_rollout(self, trust_remote_code=False):
        if self.config.rollout.name != "vllm" or self.config.rollout.mode != "sync":
            raise NotImplementedError(
                "rdrop MLP-channel rollout currently supports synchronous vLLM only"
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
        component = self._intervention_config()
        # Native dropout must stay off: the two explicit masks are the only
        # internal stochasticity and are reproducible on both backends.
        for module in actor_model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        self.actor = MLPChannelRDropActor(
            config=omega_conf_to_dataclass(self.config.actor),
            actor_module=self.actor_module_fsdp,
            actor_optimizer=self.actor_optimizer,
        )
        tracker = SampledGradientTracker(
            self.actor_module_fsdp,
            sample_size_per_rank=int(component.gradient_sample_size_per_rank),
            random_seed=int(component.random_seed) + 1_000_003 + self.rank,
        )
        self.actor.configure_rdrop(component, self.actor_mlp_controller, tracker)
        self.parameter_update_tracker = (
            ParameterUpdateTracker(self.actor_module_fsdp)
            if component.parameter_update_diagnostics_enabled else None
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def compute_parameter_update_metrics(self):
        if self.parameter_update_tracker is None:
            return {"val-aux/parameter_update/enabled": 0.0}
        metrics = self.parameter_update_tracker.distributed_metrics(atol=1e-5)
        metrics["val-aux/parameter_update/enabled"] = 1.0
        return metrics

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        assert self._is_rollout and self._is_actor

        if bool(prompts.meta_info.get("validate", False)):
            switch_started = time.perf_counter()
            self.rollout_mlp_controller.set_route(CLEAN_ROUTE)
            switch_timing = reduce_timing(
                {
                    "mlp_rdrop_switch_rollout_neutral": (
                        time.perf_counter() - switch_started
                    )
                }
            )
            output = super().generate_sequences(prompts)
            output.meta_info.setdefault("timing", {}).update(switch_timing)
            return output

        if "uid" not in prompts.non_tensor_batch:
            raise RuntimeError("rdrop rollout batch is missing the original prompt uid")
        if "global_steps" not in prompts.meta_info:
            raise RuntimeError("rdrop rollout batch is missing global_steps")

        version = int(prompts.meta_info["global_steps"])
        self.actor_mlp_controller.refresh_masks(version=version)
        self.rollout_mlp_controller.refresh_masks(version=version)
        routes = assign_rdrop_routes(prompts.non_tensor_batch["uid"])
        prompts.non_tensor_batch["route_id"] = routes
        prompts.non_tensor_batch["mask_version"] = np.full(
            len(prompts), version, dtype=np.int64
        )
        prompts.non_tensor_batch["rdrop_rollout_order"] = np.arange(
            len(prompts), dtype=np.int64
        )

        mask_a_indices = np.flatnonzero(routes == MASK_A_ROUTE)
        mask_b_indices = np.flatnonzero(routes == MASK_B_ROUTE)
        if mask_a_indices.size != mask_b_indices.size:
            raise RuntimeError(
                "rdrop rollout shard is unbalanced: "
                f"mask_a={mask_a_indices.size}, mask_b={mask_b_indices.size}"
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
        timings["rdrop_rollout_weight_sync"] = time.perf_counter() - enter_started

        try:
            reset_started = time.perf_counter()
            self.rollout.inference_engine.reset_prefix_cache()
            timings["mlp_prefix_cache_reset_before_mask_a"] = (
                time.perf_counter() - reset_started
            )

            switch_started = time.perf_counter()
            self.rollout_mlp_controller.set_route(MASK_A_ROUTE)
            timings["mlp_rdrop_switch_rollout_mask_a"] = (
                time.perf_counter() - switch_started
            )
            generation_started = time.perf_counter()
            mask_a_output = self.rollout.generate_sequences(
                prompts=prompts.select_idxs(mask_a_indices)
            )
            timings["gen_mask_a"] = time.perf_counter() - generation_started

            reset_started = time.perf_counter()
            # Prefix-cache keys do not encode the perturbation route.
            self.rollout.inference_engine.reset_prefix_cache()
            timings["mlp_prefix_cache_reset_between_routes"] = (
                time.perf_counter() - reset_started
            )

            switch_started = time.perf_counter()
            self.rollout_mlp_controller.set_route(MASK_B_ROUTE)
            timings["mlp_rdrop_switch_rollout_mask_b"] = (
                time.perf_counter() - switch_started
            )
            generation_started = time.perf_counter()
            mask_b_output = self.rollout.generate_sequences(
                prompts=prompts.select_idxs(mask_b_indices)
            )
            timings["gen_mask_b"] = time.perf_counter() - generation_started

            reset_started = time.perf_counter()
            self.rollout.inference_engine.reset_prefix_cache()
            timings["mlp_prefix_cache_reset_after_mask_b"] = (
                time.perf_counter() - reset_started
            )

            output = DataProto.concat([mask_a_output, mask_b_output])
            original_order = np.asarray(
                output.non_tensor_batch["rdrop_rollout_order"], dtype=np.int64
            )
            output.reorder(
                torch.from_numpy(np.argsort(original_order)).to(dtype=torch.long)
            )
        finally:
            self.rollout_mlp_controller.set_route(CLEAN_ROUTE)
            exit_started = time.perf_counter()
            loop.run_until_complete(self.trainer_mode())
            timings["rdrop_rollout_to_trainer"] = (
                time.perf_counter() - exit_started
            )

        timings = reduce_timing(timings)
        output.meta_info["timing"] = timings
        output = output.to("cpu")
        get_torch_device().empty_cache()
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_mlp_rdrop_status(self):
        return {
            "metrics": self.actor_mlp_controller.metrics(),
            "mask_version": self.actor_mlp_controller.mask_version,
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
                {**self.actor_mlp_controller.state_dict(), "initial_model_path": str(self.config.model.path)},
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
                if state.get("initial_model_path") != str(self.config.model.path):
                    raise ValueError("resume requires the same pre-RL model.path for parameter diagnostics")
            else:
                raise RuntimeError(f"checkpoint is missing {_STATE_FILE}")
        super().load_checkpoint(local_path, hdfs_path, del_local_after_load)
        if state is not None:
            self.actor_mlp_controller.load_state_dict(state)
            self.rollout_mlp_controller.load_state_dict(state)
        aggressive_empty_cache(force_sync=True)
