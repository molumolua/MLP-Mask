"""FSDP actor/rollout worker for hard MLP-channel KL consistency."""

from __future__ import annotations

import os

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
from .intervention import (
    MLPChannelConsistencyController,
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
            random_seed=int(config.get("random_seed", 42)),
        )
        actor_cfg = omega_conf_to_dataclass(self.config.actor)
        self.actor = MLPChannelConsistencyActor(
            config=actor_cfg,
            actor_module=self.actor_module_fsdp,
            actor_optimizer=self.actor_optimizer,
        )
        self.actor.consistency_controller = controller
        self.actor.consistency_kl_coef = float(config.get("kl_coef", 0.01))
        self.actor.consistency_kl_top_k = int(config.get("kl_top_k", 64))
        self.consistency_controller = controller

        actor_model = getattr(
            self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp
        )
        installed = install_hf_mlp_consistency_mask(actor_model, controller)
        if len(installed) != controller.num_layers:
            raise RuntimeError(
                f"installed {len(installed)} MLP masks, expected {controller.num_layers}"
            )
        if fsdp_version(self.actor.actor_module) not in {1, 2}:
            raise RuntimeError("MLP-channel consistency requires an FSDP/FSDP2 actor")

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_actor(self, data: DataProto):
        assert self._is_actor
        mask_metrics = self.consistency_controller.resample()
        try:
            output = super().update_actor(data)
        finally:
            # Never let actor-side old-log-prob or later RPCs observe the hard mask.
            self.consistency_controller.set_clean()
        output.meta_info.setdefault("metrics", {}).update(mask_metrics)
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
