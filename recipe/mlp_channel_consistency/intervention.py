"""Random hard MLP-channel masks used only by an actor-side KL branch."""

from __future__ import annotations

import re
from types import MethodType
from typing import Any

import torch


_LAYER_RE = re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)(?:\.|$)")
_CHECKPOINT_FORMAT = "mlp_channel_consistency_v1"


class MLPChannelConsistencyController:
    """Hold one exact per-layer hard mask for a complete optimizer step.

    Rollout, old-log-prob, validation, and the GRPO backward always use the clean
    route.  The masked route is entered only for the auxiliary teacher-forced KL
    forward/backward.  Masks contain literal zeros and ones; no inverted-dropout
    rescaling is applied.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        intermediate_size: int,
        mask_ratio: float = 0.10,
        random_seed: int = 42,
    ) -> None:
        if num_layers <= 0 or intermediate_size <= 0:
            raise ValueError("num_layers and intermediate_size must be positive")
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in (0, 1)")
        masked_per_layer = int(round(mask_ratio * intermediate_size))
        if not 0 < masked_per_layer < intermediate_size:
            raise ValueError(
                "mask_ratio must mask at least one and retain at least one channel per layer"
            )

        self.num_layers = int(num_layers)
        self.intermediate_size = int(intermediate_size)
        self.mask_ratio = float(mask_ratio)
        self.masked_per_layer = masked_per_layer
        self.random_seed = int(random_seed)
        self.keep_mask = torch.ones(
            (self.num_layers, self.intermediate_size), dtype=torch.bool
        )
        self.mask_version = 0
        self.route = "clean"
        self._device_masks: dict[tuple[int, str, int, torch.dtype], torch.Tensor] = {}

    def set_clean(self) -> None:
        self.route = "clean"

    def set_masked(self) -> None:
        if self.mask_version <= 0:
            raise RuntimeError("masked consistency route requested before the first mask sample")
        self.route = "masked"

    def resample(self) -> dict[str, float]:
        """Sample exactly ``round(mask_ratio * d_ff)`` channels in every layer."""
        next_version = self.mask_version + 1
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.random_seed + next_version)
        keep = torch.ones_like(self.keep_mask)
        for layer_idx in range(self.num_layers):
            masked = torch.randperm(
                self.intermediate_size, generator=generator
            )[: self.masked_per_layer]
            keep[layer_idx, masked] = False
        self.keep_mask.copy_(keep)
        self.mask_version = next_version
        self._device_masks.clear()
        self.set_clean()
        return self.metrics()

    def apply(self, layer_idx: int, activation: torch.Tensor) -> torch.Tensor:
        if self.route == "clean":
            return activation
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(f"layer {layer_idx} outside [0, {self.num_layers})")
        if activation.shape[-1] != self.intermediate_size:
            raise RuntimeError(
                f"MLP layer {layer_idx} activation width {activation.shape[-1]} != "
                f"intermediate_size {self.intermediate_size}"
            )
        device = activation.device
        key = (layer_idx, device.type, device.index or 0, activation.dtype)
        mask = self._device_masks.get(key)
        if mask is None:
            mask = self.keep_mask[layer_idx].to(device=device, dtype=activation.dtype)
            self._device_masks[key] = mask
        return activation * mask

    def metrics(self) -> dict[str, float]:
        masked_per_layer = (~self.keep_mask).sum(dim=-1).to(dtype=torch.float32)
        return {
            "mlp_consistency/mask_version": float(self.mask_version),
            "mlp_consistency/mask_ratio_requested": float(self.mask_ratio),
            "mlp_consistency/masked_per_layer": float(self.masked_per_layer),
            "mlp_consistency/masked_per_layer_min": float(masked_per_layer.min().item()),
            "mlp_consistency/masked_per_layer_max": float(masked_per_layer.max().item()),
            "mlp_consistency/realized_mask_fraction": float(
                (~self.keep_mask).sum().item() / self.keep_mask.numel()
            ),
            "mlp_consistency/hard_mask": 1.0,
            "mlp_consistency/inverted_dropout_scaling": 0.0,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_format": _CHECKPOINT_FORMAT,
            "num_layers": self.num_layers,
            "intermediate_size": self.intermediate_size,
            "mask_ratio": self.mask_ratio,
            "random_seed": self.random_seed,
            "mask_version": self.mask_version,
            "keep_mask": self.keep_mask.cpu(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("checkpoint_format") != _CHECKPOINT_FORMAT:
            raise ValueError("incompatible MLP-channel consistency checkpoint")
        for name, expected in (
            ("num_layers", self.num_layers),
            ("intermediate_size", self.intermediate_size),
            ("random_seed", self.random_seed),
        ):
            if int(state[name]) != expected:
                raise ValueError(f"checkpoint {name}={state[name]} != configured {expected}")
        if float(state["mask_ratio"]) != self.mask_ratio:
            raise ValueError(
                f"checkpoint mask_ratio={state['mask_ratio']} != configured {self.mask_ratio}"
            )
        keep_mask = torch.as_tensor(state["keep_mask"], dtype=torch.bool)
        if keep_mask.shape != self.keep_mask.shape:
            raise ValueError(
                f"checkpoint keep_mask shape {tuple(keep_mask.shape)} != "
                f"{tuple(self.keep_mask.shape)}"
            )
        self.keep_mask.copy_(keep_mask)
        self.mask_version = int(state["mask_version"])
        self._device_masks.clear()
        self.set_clean()


def install_hf_mlp_consistency_mask(
    model: torch.nn.Module,
    controller: MLPChannelConsistencyController,
) -> list[str]:
    """Patch dense HF Qwen/Llama-style SwiGLU blocks in the actor only."""
    patched: list[str] = []
    seen_layers: set[int] = set()
    for name, module in model.named_modules():
        layer_idx = _layer_index(name)
        if layer_idx is None or layer_idx in seen_layers:
            continue
        if not all(
            hasattr(module, attr)
            for attr in ("gate_proj", "up_proj", "down_proj", "act_fn")
        ):
            continue
        if getattr(module, "_mlp_channel_consistency_patched", False):
            raise RuntimeError(f"MLP module {name!r} was already consistency-patched")

        def forward(this, hidden_state, *args, _layer_idx=layer_idx, **kwargs):
            if args or kwargs:
                raise TypeError("patched dense MLP expects only hidden_state")
            activation = this.act_fn(this.gate_proj(hidden_state)) * this.up_proj(
                hidden_state
            )
            activation = controller.apply(_layer_idx, activation)
            return this.down_proj(activation)

        module.forward = MethodType(forward, module)
        module._mlp_channel_consistency_patched = True
        module._mlp_channel_consistency_controller = controller
        patched.append(name)
        seen_layers.add(layer_idx)

    expected = set(range(controller.num_layers))
    if seen_layers != expected:
        raise RuntimeError(
            "HF actor: expected one dense MLP per transformer layer; "
            f"missing={sorted(expected - seen_layers)}, extra={sorted(seen_layers - expected)}"
        )
    return patched


def _layer_index(module_name: str) -> int | None:
    normalized = ".".join(
        part for part in module_name.split(".") if part != "_fsdp_wrapped_module"
    )
    match = _LAYER_RE.search(normalized)
    return int(match.group(1)) if match else None
