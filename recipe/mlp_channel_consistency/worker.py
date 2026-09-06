"""FSDP actor/rollout worker for hard MLP-channel KL consistency."""

from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist

from verl import DataProto
from verl.single_controller.base.decorator import (
    Dispatch,
    make_nd_compute_dataproto_dispatch_fn,
    register,
)
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.fsdp_utils import fsdp_version
from verl.workers.fsdp_workers import ActorRolloutRefWorker

from .actor import MLPChannelConsistencyActor
from .diagnostics import ParameterUpdateTracker, SampledGradientTracker
from .intervention import (
    MLPChannelConsistencyController,
    UPDATED_FRACTION_SCORE,
    install_hf_mlp_consistency_mask,
)


_STATE_FILE = "mlp_channel_consistency.pt"


class MLPChannelConsistencyActorRolloutRefWorker(ActorRolloutRefWorker):
    """Keep inference clean and apply hard masks only in the auxiliary actor pass."""

    def _consistency_config(self):
        config = self.config.get("mlp_channel_consistency", None)
        if config is None or not bool(config.get("enabled", False)):
            raise RuntimeError(
                "MLPChannelConsistencyActorRolloutRefWorker requires "
                "actor_rollout_ref.mlp_channel_consistency.enabled=true"
            )
        return config

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if not self._is_actor:
            return

        config = self._consistency_config()
        controller = MLPChannelConsistencyController(
            num_layers=int(self.actor_model_config.num_hidden_layers),
            intermediate_size=int(self.actor_model_config.intermediate_size),
            mask_ratio=float(config.get("mask_ratio", 0.10)),
            selection_strategy=str(config.get("selection_strategy", "random")),
            score_method=str(config.get("score_method", "none")),
            score_ema_beta=float(config.get("score_ema_beta", 0.0)),
            activation_ema_beta=float(config.get("activation_ema_beta", 0.95)),
            relative_activation_epsilon=float(
                config.get("relative_activation_epsilon", 1.0e-6)
            ),
            weighted_max_ratio=float(config.get("weighted_max_ratio", 4.0)),
            weighted_rank_power=float(config.get("weighted_rank_power", 2.0)),
            random_seed=int(config.get("random_seed", 42)),
        )
        actor_cfg = omega_conf_to_dataclass(self.config.actor)
        self.actor = MLPChannelConsistencyActor(
            config=actor_cfg,
            actor_module=self.actor_module_fsdp,
            actor_optimizer=self.actor_optimizer,
        )
        self.actor.consistency_controller = controller
        self.actor.consistency_auxiliary_enabled = bool(
            config.get("auxiliary_enabled", True)
        )
        self.actor.consistency_kl_coef = float(config.get("kl_coef", 0.01))
        self.actor.consistency_kl_top_k = int(config.get("kl_top_k", 64))
        self.actor.consistency_micro_batch_size_per_gpu = int(
            config.get("micro_batch_size_per_gpu", 1)
        )
        self.consistency_controller = controller

        actor_model = getattr(
            self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp
        )
        if self.actor.consistency_auxiliary_enabled:
            installed = install_hf_mlp_consistency_mask(actor_model, controller)
            if len(installed) != controller.num_layers:
                raise RuntimeError(
                    f"installed {len(installed)} MLP masks, "
                    f"expected {controller.num_layers}"
                )
        if fsdp_version(self.actor.actor_module) not in {1, 2}:
            raise RuntimeError("MLP-channel consistency requires an FSDP/FSDP2 actor")

        self.consistency_gradient_tracker = SampledGradientTracker(
            self.actor.actor_module,
            sample_size_per_rank=int(
                config.get("gradient_sample_size_per_rank", 262_144)
            ),
            random_seed=int(config.get("random_seed", 42)) + 1_000_003 + self.rank,
        )
        self.actor.consistency_gradient_tracker = self.consistency_gradient_tracker
        # Optional because the pre-RL BF16 CPU shard is useful diagnostically but
        # materially larger than the online channel-score state.
        self.parameter_update_diagnostics_enabled = bool(
            config.get("parameter_update_diagnostics_enabled", True)
        )
        self.parameter_update_tracker = (
            ParameterUpdateTracker(
                self.actor.actor_module,
                num_layers=(
                    controller.num_layers
                    if controller.score_method == UPDATED_FRACTION_SCORE
                    else None
                ),
                intermediate_size=(
                    controller.intermediate_size
                    if controller.score_method == UPDATED_FRACTION_SCORE
                    else None
                ),
            )
            if self.parameter_update_diagnostics_enabled
            else None
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def compute_parameter_update_metrics(self):
        if not self._is_actor:
            return {}
        self.consistency_controller.set_clean()
        if not self.parameter_update_diagnostics_enabled:
            return {"val-aux/parameter_update/enabled": 0.0}
        if self.parameter_update_tracker is None:
            raise RuntimeError("parameter-update diagnostics tracker is missing")
        atol = float(
            self._consistency_config().get("parameter_update_atol", 1.0e-5)
        )
        metrics = self.parameter_update_tracker.distributed_metrics(atol=atol)
        metrics["val-aux/parameter_update/enabled"] = 1.0
        return metrics

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_actor(self, data: DataProto):
        assert self._is_actor
        if self.actor.consistency_auxiliary_enabled:
            mask_metrics = self.consistency_controller.resample()
        else:
            mask_metrics = self.consistency_controller.metrics()
            mask_metrics.update(
                {
                    "mlp_consistency/mask_ratio_requested": 0.0,
                    "mlp_consistency/masked_per_layer": 0.0,
                    "mlp_consistency/hard_mask": 0.0,
                }
            )
        try:
            output = super().update_actor(data)
        finally:
            # Never let actor-side old-log-prob or later RPCs observe the hard mask.
            self.consistency_controller.set_clean()
        output_metrics = output.meta_info.setdefault("metrics", {})
        if self.consistency_controller.needs_updated_fraction_score:
            if self.parameter_update_tracker is None:
                raise RuntimeError("updated_fraction score requires its parameter tracker")
            atol = float(
                self._consistency_config().get("parameter_update_atol", 1.0e-5)
            )
            update_fraction_started = time.perf_counter()
            updated_fraction = (
                self.parameter_update_tracker.distributed_channel_updated_fraction(
                    atol=atol
                )
            )
            output_metrics.update(
                self.consistency_controller.update_updated_fraction_score(
                    updated_fraction,
                    atol=atol,
                )
            )
            output_metrics[
                "timing_s/mlp_consistency_updated_fraction"
            ] = time.perf_counter() - update_fraction_started
        # Actor-side score collection describes the just-finished clean update;
        # do not overwrite it with the pre-update score snapshot returned by
        # resample(). Mask-only keys are still added here.
        for name, value in mask_metrics.items():
            output_metrics.setdefault(name, value)
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(
        self,
        local_path,
        hdfs_path=None,
        global_step=0,
        max_ckpt_to_keep=None,
    ):
        super().save_checkpoint(
            local_path,
            hdfs_path,
            global_step,
            max_ckpt_to_keep,
        )
        if self._is_actor and dist.get_rank() == 0:
            os.makedirs(local_path, exist_ok=True)
            torch.save(
                self.consistency_controller.state_dict(),
                os.path.join(local_path, _STATE_FILE),
            )
        if self._is_actor:
            dist.barrier()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        state = None
        if self._is_actor and local_path is not None:
            state_path = os.path.join(local_path, _STATE_FILE)
            if os.path.exists(state_path):
                state = torch.load(state_path, map_location="cpu", weights_only=False)
            else:
                raise RuntimeError(
                    f"training checkpoint {local_path!r} is missing {_STATE_FILE}; "
                    "resume with a checkpoint produced by this recipe or disable resume"
                )
        super().load_checkpoint(local_path, hdfs_path, del_local_after_load)
        if state is not None:
            self.consistency_controller.load_state_dict(state)
