"""FSDP2 actor/rollout worker for history-based MLP-channel updates."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.workers.fsdp_workers import ActorRolloutRefWorker

from .optimizer import ChannelRelativeUpdateAdamW
from .relative_update import MLPChannelRelativeUpdateController

_RELATIVE_UPDATE_STATE_FILE = "mlp_channel_relative_update.pt"


class MLPChannelRelativeUpdateActorRolloutRefWorker(ActorRolloutRefWorker):
    """Configure structured channel updates without changing rollout or loss."""

    def _relative_update_config(self):
        config = self.config.get("mlp_channel_relative_update", None)
        if config is None:
            raise RuntimeError(
                "MLPChannelRelativeUpdateActorRolloutRefWorker requires an "
                "actor_rollout_ref.mlp_channel_relative_update config"
            )
        return config

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if not self._is_actor:
            return
        config = self._relative_update_config()
        if not bool(config.get("enabled", False)):
            return
        if self.config.actor.strategy != "fsdp2":
            raise NotImplementedError(
                "MLP-channel relative updates require FSDP2 so logical 2-D parameter "
                "shapes remain available to the structured optimizer"
            )
        if not isinstance(self.actor_optimizer, ChannelRelativeUpdateAdamW):
            raise RuntimeError(
                "actor optimizer must be recipe.mlp_channel_relative_update.optimizer."
                "ChannelRelativeUpdateAdamW"
            )

        selected_layers = config.get("layers", None)
        if selected_layers is not None:
            selected_layers = [int(layer) for layer in selected_layers]
        controller = MLPChannelRelativeUpdateController(
            num_layers=int(self.actor_model_config.num_hidden_layers),
            intermediate_size=int(self.actor_model_config.intermediate_size),
            selected_layers=selected_layers,
            history_ema_beta=float(config.get("history_ema_beta", 0.99)),
            history_power=float(config.get("history_power", 0.5)),
            history_floor_ratio=float(config.get("history_floor_ratio", 0.1)),
            multiplier_ratio_cap=float(config.get("multiplier_ratio_cap", 10.0)),
            warmup_steps=int(config.get("warmup_steps", 16)),
            parameter_rms_epsilon=float(config.get("parameter_rms_epsilon", 1e-12)),
            history_epsilon=float(config.get("history_epsilon", 1e-12)),
        )
        installed = self.actor_optimizer.configure_channel_updates(
            controller=controller,
            named_parameters=self.actor_module_fsdp.named_parameters(),
        )
        expected = 3 * len(controller.selected_layers)
        if len(installed) != expected:
            raise RuntimeError(
                f"installed {len(installed)} structured MLP weights, expected {expected}"
            )
        self.relative_update_controller = controller

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_actor(self, data: DataProto):
        output = super().update_actor(data)
        if isinstance(self.actor_optimizer, ChannelRelativeUpdateAdamW):
            output.meta_info.setdefault("metrics", {}).update(
                self.actor_optimizer.get_last_relative_update_metrics()
            )
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        super().save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)
        controller = getattr(self, "relative_update_controller", None)
        if self._is_actor and controller is not None and dist.get_rank() == 0:
            os.makedirs(local_path, exist_ok=True)
            torch.save(
                controller.state_dict(),
                os.path.join(local_path, _RELATIVE_UPDATE_STATE_FILE),
            )
        if self._is_actor:
            dist.barrier()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        state = None
        controller = getattr(self, "relative_update_controller", None)
        if self._is_actor and controller is not None and local_path is not None:
            state_path = os.path.join(local_path, _RELATIVE_UPDATE_STATE_FILE)
            if os.path.exists(state_path):
                state = torch.load(state_path, map_location="cpu", weights_only=False)
            else:
                raise RuntimeError(
                    f"training checkpoint {local_path!r} is missing "
                    f"{_RELATIVE_UPDATE_STATE_FILE}; resume with a checkpoint produced "
                    "by this recipe or disable resume"
                )
        super().load_checkpoint(local_path, hdfs_path, del_local_after_load)
        if state is not None:
            controller.load_state_dict(state)
