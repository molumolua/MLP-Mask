"""Recipe-specific data-parallel actor integration."""

from __future__ import annotations

import numpy as np
import torch

from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor

from .rarity import MLPChannelRarityController


class MLPChannelRarityActor(DataParallelPPOActor):
    """Collect prompt activations during the existing old-log-prob forward."""

    rarity_controller: MLPChannelRarityController

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False):
        controller = getattr(self, "rarity_controller", None)
        collecting = controller is not None and controller.collecting
        if collecting:
            if self.ulysses_sequence_parallel_size != 1:
                raise NotImplementedError(
                    "MLP-channel rarity collection currently requires "
                    "actor.ulysses_sequence_parallel_size=1"
                )
            if "multi_modal_inputs" in micro_batch:
                raise NotImplementedError("MLP-channel rarity currently supports text-only prompts")

            attention_mask = micro_batch["attention_mask"].to(dtype=torch.bool)
            response_length = int(micro_batch["responses"].shape[-1])
            sample_count, sequence_length = attention_mask.shape
            if response_length <= 0 or response_length >= sequence_length:
                raise RuntimeError(
                    f"invalid response length {response_length} for sequence length {sequence_length}"
                )
            prompt_mask = attention_mask.clone()
            prompt_mask[:, -response_length:] = False
            sample_ids = torch.arange(
                sample_count,
                device=attention_mask.device,
                dtype=torch.long,
            ).unsqueeze(1).expand_as(attention_mask)

            if self.use_remove_padding:
                # dp_actor.unpad_input preserves flattened row-major token order.
                valid = attention_mask
                prompt_mask = prompt_mask[valid].reshape(1, -1)
                sample_ids = sample_ids[valid].reshape(1, -1)
            controller.begin_micro_batch(
                prompt_mask=prompt_mask,
                sample_ids=sample_ids,
                sample_count=sample_count,
            )

        try:
            return super()._forward_micro_batch(
                micro_batch,
                temperature=temperature,
                calculate_entropy=calculate_entropy,
            )
        finally:
            if collecting:
                controller.end_micro_batch()

    def update_policy(self, data: DataProto):
        """Expose question rarity as one constant per-row actor-loss multiplier."""
        if "rarity_loss_weights" not in data.batch:
            raise RuntimeError(
                "MLP-channel rarity actor update is missing rarity_loss_weights; "
                "old-log-prob recomputation must run before every actor update"
            )
        weights = data.batch["rarity_loss_weights"].detach().cpu().numpy().astype(np.float32)
        if weights.shape != (len(data),):
            raise RuntimeError(
                f"rarity_loss_weights shape {weights.shape} does not match batch size {len(data)}"
            )
        data.non_tensor_batch["loss_multiplier"] = weights
        # One explicit group makes the core actor apply weights within a single
        # expected policy loss rather than interpreting every distinct weight as a
        # separate route/objective.
        data.non_tensor_batch["loss_group_id"] = np.full(
            len(data), "mlp_channel_rarity", dtype=object
        )
        data.non_tensor_batch["loss_group_normalizer"] = np.ones(len(data), dtype=np.float32)
        return super().update_policy(data)
