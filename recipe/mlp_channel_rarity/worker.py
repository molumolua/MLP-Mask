"""FSDP actor/rollout worker for online MLP-channel rarity weighting."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.fsdp_utils import fsdp_version
from verl.workers.fsdp_workers import ActorRolloutRefWorker

from .actor import MLPChannelRarityActor
from .rarity import MLPChannelRarityController, install_hf_mlp_activation_observer

_RARITY_STATE_FILE = "mlp_channel_rarity.pt"


class MLPChannelRarityActorRolloutRefWorker(ActorRolloutRefWorker):
    """Collect rarity in the actor forward without changing rollout inference."""

    def _rarity_config(self):
        config = self.config.get("mlp_channel_rarity", None)
        if config is None or not bool(config.get("enabled", False)):
            raise RuntimeError(
                "MLPChannelRarityActorRolloutRefWorker requires "
                "actor_rollout_ref.mlp_channel_rarity.enabled=true"
            )
        return config

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if not self._is_actor:
            return

        config = self._rarity_config()
        selected_layers = config.get("layers", None)
        if selected_layers is not None:
            selected_layers = [int(layer) for layer in selected_layers]
        explicit_top_k = config.get("top_k", None)
        controller = MLPChannelRarityController(
            num_layers=int(self.actor_model_config.num_hidden_layers),
            intermediate_size=int(self.actor_model_config.intermediate_size),
            selected_layers=selected_layers,
            activation_ema_beta=float(config.get("activation_ema_beta", 0.95)),
            topk_ratio=float(config.get("topk_ratio", 0.01)),
            top_k=int(explicit_top_k) if explicit_top_k is not None else None,
            deviation_epsilon=float(config.get("deviation_epsilon", 1e-6)),
            frequency_epsilon=float(config.get("frequency_epsilon", 1e-8)),
            frequency_prior_strength=float(config.get("frequency_prior_strength", 64.0)),
            max_channel_rarity=float(config.get("max_channel_rarity", 8.0)),
            responses_per_question=int(self.config.rollout.n),
            use_frequency_prior=bool(config.get("use_frequency_prior", False)),
            min_loss_weight=float(config.get("min_loss_weight", 0.2)),
            max_loss_weight=float(config.get("max_loss_weight", 5.0)),
        )

        actor_cfg = omega_conf_to_dataclass(self.config.actor)
        self.actor = MLPChannelRarityActor(
            config=actor_cfg,
            actor_module=self.actor_module_fsdp,
            actor_optimizer=self.actor_optimizer,
        )
        self.actor.rarity_controller = controller
        self.rarity_controller = controller
        actor_model = getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        install_hf_mlp_activation_observer(actor_model, controller)
        self._last_rarity_metrics: dict[str, float] = {}

        # The checkpoint manager created by the parent retains the same model,
        # optimizer and scheduler objects, so replacing only the lightweight actor
        # wrapper is safe.
        if fsdp_version(self.actor.actor_module) not in {1, 2}:
            raise RuntimeError("MLP-channel rarity requires an FSDP/FSDP2 actor")

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_log_prob(self, data: DataProto):
        """Piggyback rarity collection on the mandatory old-log-prob forward."""
        assert self._is_actor
        if bool(data.meta_info.get("is_lora", False)):
            raise NotImplementedError("MLP-channel rarity does not support LoRA reference forwards")

        self.rarity_controller.begin_step()
        try:
            output = super().compute_log_prob(data)
            result = self.rarity_controller.finalize_step()
        except BaseException:
            self.rarity_controller.abort_step()
            raise

        if len(output) != result.loss_weights.numel():
            raise RuntimeError(
                f"rarity produced {result.loss_weights.numel()} weights for {len(output)} log-prob rows"
            )
        output.batch["rarity_scores"] = result.raw_scores.to(device="cpu", dtype=torch.float32)
        output.batch["rarity_loss_weights"] = result.loss_weights.to(
            device="cpu", dtype=torch.float32
        )
        output.meta_info["mlp_channel_rarity_metrics"] = result.metrics
        self._last_rarity_metrics = result.metrics
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_actor(self, data: DataProto):
        output = super().update_actor(data)
        output.meta_info.setdefault("metrics", {}).update(self._last_rarity_metrics)
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        super().save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)
        if self._is_actor and dist.get_rank() == 0:
            os.makedirs(local_path, exist_ok=True)
            torch.save(
                self.rarity_controller.state_dict(),
                os.path.join(local_path, _RARITY_STATE_FILE),
            )
        if self._is_actor:
            dist.barrier()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        state = None
        if self._is_actor and local_path is not None:
            state_path = os.path.join(local_path, _RARITY_STATE_FILE)
            if os.path.exists(state_path):
                state = torch.load(state_path, map_location="cpu", weights_only=False)
            else:
                raise RuntimeError(
                    f"training checkpoint {local_path!r} is missing {_RARITY_STATE_FILE}; "
                    "resume with a checkpoint produced by this recipe or disable resume"
                )
        super().load_checkpoint(local_path, hdfs_path, del_local_after_load)
        if state is not None:
            self.rarity_controller.load_state_dict(state)
