"""MLP-channel masks and clean-policy gradient x activation saliency.

The controller deliberately keeps masks outside the model state dict.  A fixed-size
active buffer is allocated lazily for every patched MLP.  Route switches update the
buffer in place, which is important for vLLM CUDA graphs: the captured graph keeps the
same pointer and reads the new clean/masked values without recapture.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch
import torch.distributed as dist

CLEAN_ROUTE = "clean"
MASKED_ROUTE = "masked"
_VALID_ROUTES = {CLEAN_ROUTE, MASKED_ROUTE}
_LAYER_RE = re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)\.mlp(?:\.|$)")


@dataclass(frozen=True)
class MaskRefreshResult:
    metrics: dict[str, float]
    timings: dict[str, float]


class MLPChannelInterventionController:
    """Owns per-block masks, saliency accumulators, and mask history.

    ``keep_mask[layer, channel]`` is True for an available channel and False for a
    channel removed by the structured intervention.  Saliency is accumulated only
    for clean policy-loss backward passes.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        intermediate_size: int,
        mask_ratio: float = 0.10,
        ema_beta: float = 0.95,
        tp_rank: int = 0,
        tp_size: int = 1,
        name: str = "actor",
    ) -> None:
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if intermediate_size <= 0:
            raise ValueError(f"intermediate_size must be positive, got {intermediate_size}")
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")
        if not 0.0 <= ema_beta < 1.0:
            raise ValueError(f"ema_beta must be in [0, 1), got {ema_beta}")
        if not 0 <= tp_rank < tp_size:
            raise ValueError(f"invalid tensor-parallel rank {tp_rank}/{tp_size}")
        if intermediate_size % tp_size != 0:
            raise ValueError(
                f"intermediate_size={intermediate_size} must be divisible by tensor-parallel size {tp_size}"
            )

        self.num_layers = int(num_layers)
        self.intermediate_size = int(intermediate_size)
        self.mask_ratio = float(mask_ratio)
        self.ema_beta = float(ema_beta)
        self.tp_rank = int(tp_rank)
        self.tp_size = int(tp_size)
        self.name = str(name)

        self.keep_mask = torch.ones((self.num_layers, self.intermediate_size), dtype=torch.bool)
        self.ever_masked = torch.zeros_like(self.keep_mask)
        self.ema_saliency = torch.zeros((self.num_layers, self.intermediate_size), dtype=torch.float32)
        self.ema_initialized = False
        self.mask_version = 0
        self.cumulative_mask_assignments = 0

        self.route = CLEAN_ROUTE
        self.collect_saliency = False
        self._response_token_mask: torch.Tensor | None = None
        self._saliency_sum: dict[int, torch.Tensor] = {}
        self._saliency_token_count: torch.Tensor | None = None
        self._saliency_accumulate_cpu_s = 0.0
        self._active_buffers: dict[tuple[int, str, int, torch.dtype, int], torch.Tensor] = {}

    @property
    def current_masked_channels(self) -> int:
        return int((~self.keep_mask).sum().item())

    @property
    def ever_masked_channels(self) -> int:
        return int(self.ever_masked.sum().item())

    @property
    def total_channels(self) -> int:
        return self.num_layers * self.intermediate_size

    def set_route(self, route: str, *, collect_saliency: bool = False) -> None:
        if route not in _VALID_ROUTES:
            raise ValueError(f"route must be one of {_VALID_ROUTES}, got {route!r}")
        if collect_saliency and route != CLEAN_ROUTE:
            raise ValueError("saliency may only be collected on the clean route")
        self.route = route
        self.collect_saliency = bool(collect_saliency)
        self._response_token_mask = None
        self._refresh_active_buffers()

    def set_response_token_mask(self, response_token_mask: torch.Tensor) -> None:
        """Set the response-token mask matching the leading activation dimensions."""
        self._response_token_mask = response_token_mask.detach()
        if self.collect_saliency:
            count = self._response_token_mask.to(dtype=torch.float32).sum()
            if self._saliency_token_count is None:
                self._saliency_token_count = torch.zeros((), device=count.device, dtype=torch.float32)
            self._saliency_token_count.add_(count)

    def end_batch(self) -> None:
        self.collect_saliency = False
        self._response_token_mask = None

    def apply(self, layer_idx: int, activation: torch.Tensor) -> torch.Tensor:
        """Apply the active route and optionally attach a clean saliency hook."""
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(f"layer {layer_idx} outside [0, {self.num_layers})")
        if activation.shape[-1] not in {
            self.intermediate_size,
            self.intermediate_size // self.tp_size,
        }:
            raise RuntimeError(
                f"{self.name} layer {layer_idx} has MLP width {activation.shape[-1]}, expected "
                f"{self.intermediate_size} or TP-local {self.intermediate_size // self.tp_size}"
            )

        if self.collect_saliency and activation.requires_grad:
            if self._response_token_mask is None:
                raise RuntimeError("clean saliency requested before response token mask was installed")
            token_mask = self._response_token_mask
            activation.register_hook(
                lambda grad, a=activation.detach(), m=token_mask, layer=layer_idx: self._accumulate_saliency(
                    layer, a, grad, m
                )
            )

        active_mask = self._get_active_buffer(layer_idx, activation)
        return activation * active_mask

    def _accumulate_saliency(
        self,
        layer_idx: int,
        activation: torch.Tensor,
        grad: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> None:
        accumulate_started = time.perf_counter()
        with torch.no_grad():
            expected_shape = activation.shape[:-1]
            if token_mask.numel() != _shape_numel(expected_shape):
                raise RuntimeError(
                    f"response mask has {token_mask.numel()} entries but layer {layer_idx} activation "
                    f"has leading shape {tuple(expected_shape)}"
                )
            weights = token_mask.reshape(expected_shape).to(device=activation.device, dtype=activation.dtype)
            contribution = (activation * grad).abs() * weights.unsqueeze(-1)
            reduce_dims = tuple(range(contribution.ndim - 1))
            score = contribution.sum(dim=reduce_dims, dtype=torch.float32)

            # Actor/HF activations are expected to be full-width.  Supporting a local
            # width here also makes the controller testable and future-proofs TP actors.
            if score.numel() != self.intermediate_size:
                full_score = torch.zeros(self.intermediate_size, device=score.device, dtype=torch.float32)
                start, stop = self._local_slice(score.numel())
                full_score[start:stop] = score
                score = full_score
            accumulator = self._saliency_sum.get(layer_idx)
            if accumulator is None or accumulator.device != score.device:
                accumulator = torch.zeros_like(score)
                self._saliency_sum[layer_idx] = accumulator
            accumulator.add_(score)
        # This is host dispatch time; GPU work remains included in update_actor.
        self._saliency_accumulate_cpu_s += time.perf_counter() - accumulate_started

    def _buffer_key(self, layer_idx: int, activation: torch.Tensor) -> tuple[int, str, int, torch.dtype, int]:
        device = activation.device
        return (layer_idx, device.type, device.index or 0, activation.dtype, activation.shape[-1])

    def _get_active_buffer(self, layer_idx: int, activation: torch.Tensor) -> torch.Tensor:
        key = self._buffer_key(layer_idx, activation)
        buffer = self._active_buffers.get(key)
        if buffer is None:
            buffer = torch.ones(activation.shape[-1], device=activation.device, dtype=activation.dtype)
            self._active_buffers[key] = buffer
            self._copy_route_to_buffer(layer_idx, buffer)
        return buffer

    def _local_slice(self, local_width: int) -> tuple[int, int]:
        if local_width == self.intermediate_size:
            return 0, self.intermediate_size
        if local_width * self.tp_size != self.intermediate_size:
            raise RuntimeError(
                f"cannot map local MLP width {local_width} to global width {self.intermediate_size} "
                f"with tp_size={self.tp_size}"
            )
        start = self.tp_rank * local_width
        return start, start + local_width

    def _copy_route_to_buffer(self, layer_idx: int, buffer: torch.Tensor) -> None:
        if self.route == CLEAN_ROUTE:
            buffer.fill_(1)
            return
        start, stop = self._local_slice(buffer.numel())
        source = self.keep_mask[layer_idx, start:stop].to(device=buffer.device, dtype=buffer.dtype)
        buffer.copy_(source)

    def _refresh_active_buffers(self) -> None:
        for key, buffer in self._active_buffers.items():
            self._copy_route_to_buffer(key[0], buffer)

    def refresh_mask(self) -> MaskRefreshResult:
        """All-reduce clean saliency and select the top ratio inside every block."""
        refresh_started = time.perf_counter()
        device = self._score_device()
        saliency = torch.stack(
            [
                self._saliency_sum.get(
                    layer_idx,
                    torch.zeros(self.intermediate_size, device=device, dtype=torch.float32),
                ).to(device)
                for layer_idx in range(self.num_layers)
            ],
            dim=0,
        )
        token_count = (
            self._saliency_token_count.to(device)
            if self._saliency_token_count is not None
            else torch.zeros((), device=device, dtype=torch.float32)
        )
        observed_layers = torch.tensor(
            [float(layer_idx in self._saliency_sum) for layer_idx in range(self.num_layers)],
            device=device,
            dtype=torch.float32,
        )

        reduce_started = time.perf_counter()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(saliency, op=dist.ReduceOp.SUM)
            dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(observed_layers, op=dist.ReduceOp.SUM)
        reduce_elapsed = time.perf_counter() - reduce_started
        if float(token_count.item()) <= 0:
            raise RuntimeError("cannot refresh MLP mask: no clean response-token saliency was accumulated")
        if bool((observed_layers == 0).any().item()):
            missing = torch.nonzero(observed_layers == 0).flatten().cpu().tolist()
            raise RuntimeError(f"cannot refresh MLP mask: no saliency hook fired for layers {missing}")
        interval_score = saliency / token_count.clamp_min(1.0)

        select_started = time.perf_counter()
        interval_cpu = interval_score.cpu()
        if self.ema_initialized:
            self.ema_saliency.mul_(self.ema_beta).add_(interval_cpu, alpha=1.0 - self.ema_beta)
        else:
            self.ema_saliency.copy_(interval_cpu)
            self.ema_initialized = True

        masked_per_layer = max(1, int(round(self.intermediate_size * self.mask_ratio)))
        old_masked = ~self.keep_mask
        new_keep = torch.ones_like(self.keep_mask)
        for layer_idx in range(self.num_layers):
            top_idx = torch.topk(
                self.ema_saliency[layer_idx], k=masked_per_layer, largest=True, sorted=False
            ).indices
            new_keep[layer_idx, top_idx] = False
        new_masked = ~new_keep
        overlap = int((old_masked & new_masked).sum().item())
        current = int(new_masked.sum().item())
        new_unique = int((new_masked & ~self.ever_masked).sum().item())

        self.keep_mask.copy_(new_keep)
        self.ever_masked.logical_or_(new_masked)
        self.mask_version += 1
        self.cumulative_mask_assignments += current
        self._refresh_active_buffers()
        select_elapsed = time.perf_counter() - select_started

        self._saliency_sum.clear()
        self._saliency_token_count = None

        metrics = self.metrics()
        metrics.update(
            {
                "mlp_mask/new_unique_channels": float(new_unique),
                "mlp_mask/overlap_with_previous": float(overlap),
                "mlp_mask/turnover_fraction": float(1.0 - overlap / max(current, 1)),
                "mlp_saliency/mean": float(self.ema_saliency.mean().item()),
                "mlp_saliency/max": float(self.ema_saliency.max().item()),
                "mlp_saliency/min": float(self.ema_saliency.min().item()),
                "mlp_saliency/response_tokens": float(token_count.item()),
                "mlp_saliency/layers_observed": float((observed_layers > 0).sum().item()),
            }
        )
        timings = {
            "mlp_saliency_accumulate_cpu": self._saliency_accumulate_cpu_s,
            "mlp_saliency_reduce": reduce_elapsed,
            "mlp_mask_select": select_elapsed,
            "mlp_mask_refresh": time.perf_counter() - refresh_started,
        }
        self._saliency_accumulate_cpu_s = 0.0
        return MaskRefreshResult(metrics=metrics, timings=timings)

    def metrics(self) -> dict[str, float]:
        current = self.current_masked_channels
        ever = self.ever_masked_channels
        current_per_layer = (~self.keep_mask).sum(dim=-1).to(dtype=torch.float32)
        ever_per_layer = self.ever_masked.sum(dim=-1).to(dtype=torch.float32)
        return {
            "mlp_mask/version": float(self.mask_version),
            "mlp_mask/initialized": float(self.mask_version > 0),
            "mlp_mask/layers": float(self.num_layers),
            "mlp_mask/channels_per_layer": float(self.intermediate_size),
            "mlp_mask/masked_per_layer": float(current / self.num_layers),
            "mlp_mask/masked_per_layer_min": float(current_per_layer.min().item()),
            "mlp_mask/masked_per_layer_max": float(current_per_layer.max().item()),
            "mlp_mask/current_channels": float(current),
            "mlp_mask/current_fraction": float(current / self.total_channels),
            "mlp_mask/ever_unique_channels": float(ever),
            "mlp_mask/ever_unique_fraction": float(ever / self.total_channels),
            "mlp_mask/ever_unique_per_layer_mean": float(ever_per_layer.mean().item()),
            "mlp_mask/ever_unique_per_layer_min": float(ever_per_layer.min().item()),
            "mlp_mask/ever_unique_per_layer_max": float(ever_per_layer.max().item()),
            "mlp_mask/cumulative_assignments": float(self.cumulative_mask_assignments),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "num_layers": self.num_layers,
            "intermediate_size": self.intermediate_size,
            "mask_ratio": self.mask_ratio,
            "ema_beta": self.ema_beta,
            "keep_mask": self.keep_mask.cpu(),
            "ever_masked": self.ever_masked.cpu(),
            "ema_saliency": self.ema_saliency.cpu(),
            "ema_initialized": self.ema_initialized,
            "mask_version": self.mask_version,
            "cumulative_mask_assignments": self.cumulative_mask_assignments,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = (self.num_layers, self.intermediate_size)
        keep_mask = torch.as_tensor(state["keep_mask"], dtype=torch.bool)
        if tuple(keep_mask.shape) != expected:
            raise ValueError(f"checkpoint mask shape {tuple(keep_mask.shape)} != {expected}")
        ever_masked = torch.as_tensor(state["ever_masked"], dtype=torch.bool)
        ema_saliency = torch.as_tensor(state["ema_saliency"], dtype=torch.float32)
        if tuple(ever_masked.shape) != expected or tuple(ema_saliency.shape) != expected:
            raise ValueError(
                "checkpoint history/saliency shapes must match controller shape "
                f"{expected}, got ever={tuple(ever_masked.shape)}, ema={tuple(ema_saliency.shape)}"
            )
        self.keep_mask.copy_(keep_mask)
        self.ever_masked.copy_(ever_masked)
        self.ema_saliency.copy_(ema_saliency)
        self.ema_initialized = bool(state.get("ema_initialized", True))
        self.mask_version = int(state.get("mask_version", 0))
        self.cumulative_mask_assignments = int(state.get("cumulative_mask_assignments", 0))
        self._refresh_active_buffers()

    def copy_mask_state_from(self, other: "MLPChannelInterventionController") -> None:
        self.load_state_dict(other.state_dict())

    def _score_device(self) -> torch.device:
        if self._saliency_sum:
            return next(iter(self._saliency_sum.values())).device
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")


def install_hf_mlp_intervention(model: torch.nn.Module, controller: MLPChannelInterventionController) -> list[str]:
    """Patch dense HF Qwen/Llama-style SwiGLU MLP instances."""
    patched: list[str] = []
    seen_layers: set[int] = set()
    for name, module in model.named_modules():
        layer_idx = _layer_index(name)
        if layer_idx is None or layer_idx in seen_layers:
            continue
        if not all(hasattr(module, attr) for attr in ("gate_proj", "up_proj", "down_proj", "act_fn")):
            continue
        if getattr(module, "_mlp_channel_intervention_patched", False):
            continue

        def forward(this, hidden_state, *args, _layer_idx=layer_idx, **kwargs):
            if args or kwargs:
                raise TypeError("patched dense MLP expects only hidden_state")
            activation = this.act_fn(this.gate_proj(hidden_state)) * this.up_proj(hidden_state)
            activation = controller.apply(_layer_idx, activation)
            return this.down_proj(activation)

        module.forward = MethodType(forward, module)
        module._mlp_channel_intervention_patched = True
        module._mlp_channel_intervention_controller = controller
        patched.append(name)
        seen_layers.add(layer_idx)

    _validate_patched_layers(controller, seen_layers, backend="HF actor")
    return patched


def install_vllm_mlp_intervention(
    model: torch.nn.Module, controller: MLPChannelInterventionController
) -> list[str]:
    """Patch dense vLLM Qwen/Llama-style fused SwiGLU MLP instances."""
    patched: list[str] = []
    seen_layers: set[int] = set()
    for name, module in model.named_modules():
        layer_idx = _layer_index(name)
        if layer_idx is None or layer_idx in seen_layers:
            continue
        if not all(hasattr(module, attr) for attr in ("gate_up_proj", "down_proj", "act_fn")):
            continue
        if getattr(module, "_mlp_channel_intervention_patched", False):
            if getattr(module, "_mlp_channel_intervention_controller", None) is not controller:
                raise RuntimeError(f"vLLM MLP {name} was patched with a different controller")
            patched.append(name)
            seen_layers.add(layer_idx)
            continue

        def forward(this, hidden_state, *args, _layer_idx=layer_idx, **kwargs):
            if args or kwargs:
                raise TypeError("patched vLLM dense MLP expects only hidden_state")
            gate_up = this.gate_up_proj(hidden_state)
            gate_up = gate_up[0] if isinstance(gate_up, tuple) else gate_up
            activation = this.act_fn(gate_up)
            activation = controller.apply(_layer_idx, activation)
            output = this.down_proj(activation)
            return output[0] if isinstance(output, tuple) else output

        module.forward = MethodType(forward, module)
        module._mlp_channel_intervention_patched = True
        module._mlp_channel_intervention_controller = controller
        patched.append(name)
        seen_layers.add(layer_idx)

    _validate_patched_layers(controller, seen_layers, backend="vLLM rollout")
    return patched


def install_vllm_class_intervention(controller: MLPChannelInterventionController) -> list[str]:
    """Patch vLLM's Qwen MLP class before engine construction/CUDA capture.

    vLLM captures model execution during ``LLM(...)`` initialization when CUDA graphs
    are enabled.  Patching only the already-built instances would therefore be too
    late.  Qwen3 reuses Qwen2MLP in current vLLM releases; both symbols are handled for
    compatibility with versions where they are distinct classes.
    """
    from vllm.model_executor.models.qwen2 import Qwen2MLP

    classes = [Qwen2MLP]
    try:
        from vllm.model_executor.models.qwen3 import Qwen3MLP

        if Qwen3MLP not in classes:
            classes.append(Qwen3MLP)
    except ImportError:
        pass

    patched_names: list[str] = []
    for cls in classes:
        existing_controller = getattr(cls, "_mlp_channel_intervention_controller", None)
        if existing_controller is not None:
            if existing_controller is not controller:
                raise RuntimeError(f"{cls.__name__} is already patched with a different controller")
            patched_names.append(cls.__name__)
            continue

        original_init = cls.__init__

        def patched_init(this, *args, __original_init=original_init, **kwargs):
            __original_init(this, *args, **kwargs)
            prefix = kwargs.get("prefix", args[4] if len(args) > 4 else "")
            layer_idx = _layer_index(str(prefix))
            if layer_idx is None:
                raise RuntimeError(f"cannot infer vLLM MLP layer from prefix {prefix!r}")
            this._mlp_channel_layer_idx = layer_idx
            this._mlp_channel_intervention_patched = True
            this._mlp_channel_intervention_controller = controller

        def patched_forward(this, hidden_state):
            gate_up = this.gate_up_proj(hidden_state)
            gate_up = gate_up[0] if isinstance(gate_up, tuple) else gate_up
            activation = this.act_fn(gate_up)
            activation = controller.apply(this._mlp_channel_layer_idx, activation)
            output = this.down_proj(activation)
            return output[0] if isinstance(output, tuple) else output

        cls.__init__ = patched_init
        cls.forward = patched_forward
        cls._mlp_channel_intervention_controller = controller
        patched_names.append(cls.__name__)
    return patched_names


def _layer_index(module_name: str) -> int | None:
    match = _LAYER_RE.search(module_name)
    return int(match.group(1)) if match else None


def _validate_patched_layers(
    controller: MLPChannelInterventionController, seen_layers: set[int], *, backend: str
) -> None:
    expected = set(range(controller.num_layers))
    if seen_layers != expected:
        missing = sorted(expected - seen_layers)
        extra = sorted(seen_layers - expected)
        raise RuntimeError(
            f"{backend}: expected dense MLPs for {controller.num_layers} layers; "
            f"missing={missing}, extra={extra}. MoE and unknown model layouts are intentionally unsupported."
        )


def _shape_numel(shape: torch.Size) -> int:
    result = 1
    for dim in shape:
        result *= int(dim)
    return result
