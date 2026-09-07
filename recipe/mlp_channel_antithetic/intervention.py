"""Symmetric multiplicative perturbations for dense MLP channels.

For one fixed Rademacher direction ``epsilon`` the two training routes use
``1 + sigma * epsilon`` and ``1 - sigma * epsilon``.  No channel is removed,
and the two gains average exactly to the unperturbed gain of one.
"""

from __future__ import annotations

import re
from types import MethodType
from typing import Any

import torch

POSITIVE_ROUTE = "positive"
NEGATIVE_ROUTE = "negative"
NEUTRAL_ROUTE = "neutral"
TRAINING_ROUTES = (POSITIVE_ROUTE, NEGATIVE_ROUTE)
_VALID_ROUTES = set(TRAINING_ROUTES) | {NEUTRAL_ROUTE}
_CHECKPOINT_FORMAT = "mlp_channel_antithetic_v1"
_LAYER_RE = re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)\.mlp(?:\.|$)")


class MLPChannelAntitheticController:
    """Own one step-stable, layer-wise antithetic channel direction."""

    valid_routes = TRAINING_ROUTES
    metric_prefix = "mlp_antithetic"
    batch_version_field = "perturbation_version"

    def __init__(
        self,
        *,
        num_layers: int,
        intermediate_size: int,
        perturbation_strength: float = 0.10,
        random_seed: int = 42,
        tp_rank: int = 0,
        tp_size: int = 1,
        name: str = "actor",
    ) -> None:
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if intermediate_size <= 0:
            raise ValueError(f"intermediate_size must be positive, got {intermediate_size}")
        if not 0.0 < perturbation_strength < 1.0:
            raise ValueError(
                "perturbation_strength must be in (0, 1), so every gain stays positive; "
                f"got {perturbation_strength}"
            )
        if random_seed < 0:
            raise ValueError(f"random_seed must be non-negative, got {random_seed}")
        if not 0 <= tp_rank < tp_size:
            raise ValueError(f"invalid tensor-parallel rank {tp_rank}/{tp_size}")
        if intermediate_size % tp_size != 0:
            raise ValueError(
                f"intermediate_size={intermediate_size} must be divisible by "
                f"tensor-parallel size {tp_size}"
            )

        self.num_layers = int(num_layers)
        self.intermediate_size = int(intermediate_size)
        self.perturbation_strength = float(perturbation_strength)
        self.random_seed = int(random_seed)
        self.tp_rank = int(tp_rank)
        self.tp_size = int(tp_size)
        self.name = str(name)

        self.direction = torch.empty(
            (self.num_layers, self.intermediate_size), dtype=torch.float32
        )
        self.perturbation_version = -1
        self.route = NEUTRAL_ROUTE
        self._active_buffers: dict[
            tuple[int, str, int | None, torch.dtype, int], torch.Tensor
        ] = {}
        self._active_buffers_available = True
        self.refresh_direction(version=0)

    def should_collect_activation(self, route: str) -> bool:
        """This recipe has no activation-statistics side channel."""
        if route not in _VALID_ROUTES:
            raise ValueError(f"unknown antithetic route {route!r}")
        return False

    def validate_batch_version(self, values: Any) -> None:
        versions = torch.as_tensor(values, dtype=torch.int64).view(-1)
        unique_versions = torch.unique(versions).tolist()
        if unique_versions != [self.perturbation_version]:
            raise RuntimeError(
                "antithetic perturbation version mismatch: "
                f"batch={unique_versions}, controller={self.perturbation_version}"
            )

    def refresh_direction(self, *, version: int) -> bool:
        """Install the deterministic direction for ``version``.

        Returns whether the direction changed.  Every even-width layer contains
        exactly as many +1 as -1 entries.  For an odd width the imbalance is one
        and its sign alternates by layer and version.
        """
        version = int(version)
        if version < 0:
            raise ValueError(f"version must be non-negative, got {version}")
        if version == self.perturbation_version:
            return False

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.random_seed + version)
        negative_count = self.intermediate_size // 2
        for layer_idx in range(self.num_layers):
            signs = torch.ones(self.intermediate_size, dtype=torch.float32)
            if self.intermediate_size % 2 and (layer_idx + version) % 2:
                negative_count_for_layer = negative_count + 1
            else:
                negative_count_for_layer = negative_count
            permutation = torch.randperm(self.intermediate_size, generator=generator)
            signs[permutation[:negative_count_for_layer]] = -1.0
            self.direction[layer_idx].copy_(signs)

        self.perturbation_version = version
        self._refresh_active_buffers()
        return True

    def set_route(self, route: str, *, collect_activation: bool = False) -> None:
        if route not in _VALID_ROUTES:
            raise ValueError(f"route must be one of {sorted(_VALID_ROUTES)}, got {route!r}")
        if collect_activation:
            raise ValueError("the antithetic recipe does not collect activation statistics")
        route_changed = route != self.route
        self.route = str(route)
        if route_changed:
            self._refresh_active_buffers()

    def end_batch(self) -> None:
        """Match the actor intervention-controller interface."""

    def set_active_buffers_available(self, available: bool) -> None:
        self._active_buffers_available = bool(available)
        if available:
            self._refresh_active_buffers()

    def _local_slice(self, width: int) -> tuple[int, int]:
        if width == self.intermediate_size:
            return 0, self.intermediate_size
        local_width = self.intermediate_size // self.tp_size
        if width != local_width:
            raise RuntimeError(
                f"{self.name} observed MLP width {width}; expected full width "
                f"{self.intermediate_size} or TP-local width {local_width}"
            )
        start = self.tp_rank * local_width
        return start, start + local_width

    def gain(self, layer_idx: int, route: str | None = None) -> torch.Tensor:
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(f"layer {layer_idx} outside [0, {self.num_layers})")
        route = self.route if route is None else str(route)
        if route == NEUTRAL_ROUTE:
            return torch.ones(self.intermediate_size, dtype=torch.float32)
        if route == POSITIVE_ROUTE:
            route_sign = 1.0
        elif route == NEGATIVE_ROUTE:
            route_sign = -1.0
        else:
            raise ValueError(f"unknown antithetic route {route!r}")
        return 1.0 + route_sign * self.perturbation_strength * self.direction[layer_idx]

    def _gain_for_tensor(self, layer_idx: int, tensor: torch.Tensor) -> torch.Tensor:
        start, stop = self._local_slice(tensor.shape[-1])
        unit = torch.ones((), device=tensor.device, dtype=tensor.dtype)
        if self.route == NEUTRAL_ROUTE:
            return torch.ones(tensor.shape[-1], device=tensor.device, dtype=tensor.dtype)
        # Quantize sigma to a delta representable on the coarser side of 1.0,
        # then build both routes from that same delta.  This avoids asymmetric
        # bf16 casts such as float32(0.90) and float32(1.10) rounding to gains
        # whose average is slightly different from one.
        strength = torch.as_tensor(
            self.perturbation_strength, device=tensor.device, dtype=tensor.dtype
        )
        effective_delta = (unit + strength) - unit
        direction = self.direction[layer_idx, start:stop].to(
            device=tensor.device, dtype=tensor.dtype
        )
        route_sign = 1.0 if self.route == POSITIVE_ROUTE else -1.0
        return unit + route_sign * effective_delta * direction

    def apply(self, layer_idx: int, activation: torch.Tensor, **_: Any) -> torch.Tensor:
        return activation * self._get_active_buffer(layer_idx, activation)

    def _buffer_key(
        self, layer_idx: int, tensor: torch.Tensor
    ) -> tuple[int, str, int | None, torch.dtype, int]:
        return (
            int(layer_idx),
            tensor.device.type,
            tensor.device.index,
            tensor.dtype,
            int(tensor.shape[-1]),
        )

    def _get_active_buffer(
        self, layer_idx: int, activation: torch.Tensor
    ) -> torch.Tensor:
        key = self._buffer_key(layer_idx, activation)
        buffer = self._active_buffers.get(key)
        if buffer is None:
            buffer = torch.ones(
                activation.shape[-1],
                device=activation.device,
                dtype=activation.dtype,
            )
            self.register_active_buffer(layer_idx, buffer)
        return buffer

    def register_active_buffer(
        self, layer_idx: int, buffer: torch.Tensor
    ) -> torch.Tensor:
        self._local_slice(buffer.numel())
        key = self._buffer_key(layer_idx, buffer)
        existing = self._active_buffers.get(key)
        if existing is not None and existing is not buffer:
            raise RuntimeError(f"duplicate active gain buffer for layer {layer_idx}: {key}")
        self._active_buffers[key] = buffer
        if self._active_buffers_available and buffer.device.type != "meta":
            buffer.copy_(self._gain_for_tensor(layer_idx, buffer))
        return buffer

    def _refresh_active_buffers(self) -> None:
        if not self._active_buffers_available:
            return
        with torch.no_grad():
            for (layer_idx, *_), buffer in self._active_buffers.items():
                if buffer.device.type != "meta":
                    buffer.copy_(self._gain_for_tensor(layer_idx, buffer))

    def metrics(self) -> dict[str, float]:
        positive_fraction = float((self.direction > 0).to(torch.float32).mean().item())
        one_bf16 = torch.ones((), dtype=torch.bfloat16)
        effective_bf16 = float(
            ((one_bf16 + torch.tensor(self.perturbation_strength, dtype=torch.bfloat16)) - one_bf16)
            .to(torch.float32)
            .item()
        )
        return {
            "mlp_antithetic/version": float(self.perturbation_version),
            "mlp_antithetic/strength": self.perturbation_strength,
            "mlp_antithetic/effective_strength_bfloat16": effective_bf16,
            "mlp_antithetic/gain_min": 1.0 - self.perturbation_strength,
            "mlp_antithetic/gain_max": 1.0 + self.perturbation_strength,
            "mlp_antithetic/direction_positive_fraction": positive_fraction,
            "mlp_antithetic/direction_mean": float(self.direction.mean().item()),
            "mlp_antithetic/layers": float(self.num_layers),
            "mlp_antithetic/channels_per_layer": float(self.intermediate_size),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_format": _CHECKPOINT_FORMAT,
            "num_layers": self.num_layers,
            "intermediate_size": self.intermediate_size,
            "perturbation_strength": self.perturbation_strength,
            "random_seed": self.random_seed,
            "perturbation_version": self.perturbation_version,
            "direction": self.direction.cpu(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("checkpoint_format") != _CHECKPOINT_FORMAT:
            raise ValueError(
                f"incompatible antithetic checkpoint format {state.get('checkpoint_format')!r}"
            )
        expected_scalars = {
            "num_layers": self.num_layers,
            "intermediate_size": self.intermediate_size,
            "perturbation_strength": self.perturbation_strength,
            "random_seed": self.random_seed,
        }
        for key, expected in expected_scalars.items():
            observed = state[key]
            if observed != expected:
                raise ValueError(f"checkpoint {key}={observed!r} does not match {expected!r}")
        direction = torch.as_tensor(state["direction"], dtype=torch.float32)
        if tuple(direction.shape) != tuple(self.direction.shape):
            raise ValueError(
                f"checkpoint direction shape {tuple(direction.shape)} does not match "
                f"{tuple(self.direction.shape)}"
            )
        if not bool(torch.all((direction == 1.0) | (direction == -1.0)).item()):
            raise ValueError("checkpoint direction must contain only -1 and +1")
        self.direction.copy_(direction)
        self.perturbation_version = int(state["perturbation_version"])
        self._refresh_active_buffers()

    def copy_state_from(self, other: "MLPChannelAntitheticController") -> None:
        self.load_state_dict(other.state_dict())


def install_hf_mlp_intervention(
    model: torch.nn.Module, controller: MLPChannelAntitheticController
) -> list[str]:
    """Patch dense HF Qwen/Llama-style SwiGLU MLP instances."""
    patched: list[str] = []
    seen_layers: set[int] = set()
    for name, module in model.named_modules():
        layer_idx = _layer_index(name)
        if layer_idx is None or layer_idx in seen_layers:
            continue
        if not all(hasattr(module, attr) for attr in ("gate_proj", "up_proj", "down_proj", "act_fn")):
            continue
        if getattr(module, "_mlp_channel_antithetic_patched", False):
            if getattr(module, "_mlp_channel_antithetic_controller", None) is not controller:
                raise RuntimeError(f"HF MLP {name} was patched with a different controller")
            patched.append(name)
            seen_layers.add(layer_idx)
            continue

        def forward(this, hidden_state, *args, _layer_idx=layer_idx, **kwargs):
            if args or kwargs:
                raise TypeError("patched dense MLP expects only hidden_state")
            activation = this.act_fn(this.gate_proj(hidden_state)) * this.up_proj(hidden_state)
            activation = controller.apply(_layer_idx, activation)
            return this.down_proj(activation)

        module.forward = MethodType(forward, module)
        module._mlp_channel_antithetic_patched = True
        module._mlp_channel_antithetic_controller = controller
        patched.append(name)
        seen_layers.add(layer_idx)

    _validate_patched_layers(controller, seen_layers, backend="HF actor")
    return patched


_VLLM_ACTIVE_GAIN_BUFFER = "_mlp_channel_antithetic_gain"


def _install_vllm_active_buffer(
    module: torch.nn.Module,
    controller: MLPChannelAntitheticController,
    layer_idx: int,
) -> torch.Tensor:
    existing = module._buffers.get(_VLLM_ACTIVE_GAIN_BUFFER)
    if existing is None:
        if hasattr(module, _VLLM_ACTIVE_GAIN_BUFFER):
            raise RuntimeError(
                f"vLLM MLP attribute {_VLLM_ACTIVE_GAIN_BUFFER!r} exists but is not a buffer"
            )
        reference = getattr(getattr(module, "down_proj", None), "weight", None)
        if reference is None:
            reference = next(module.parameters(), None)
        if reference is None:
            raise RuntimeError("cannot allocate vLLM gain: MLP has no parameter")
        local_width = controller.intermediate_size // controller.tp_size
        device = None if reference.device.type == "meta" else reference.device
        existing = torch.ones(local_width, device=device, dtype=reference.dtype)
        module.register_buffer(_VLLM_ACTIVE_GAIN_BUFFER, existing, persistent=False)
    return controller.register_active_buffer(layer_idx, existing)


def install_vllm_mlp_intervention(
    model: torch.nn.Module, controller: MLPChannelAntitheticController
) -> list[str]:
    """Patch already-built dense vLLM Qwen/Llama-style MLP instances."""
    patched: list[str] = []
    seen_layers: set[int] = set()
    for name, module in model.named_modules():
        layer_idx = _layer_index(name)
        if layer_idx is None or layer_idx in seen_layers:
            continue
        if not all(hasattr(module, attr) for attr in ("gate_up_proj", "down_proj", "act_fn")):
            continue
        if getattr(module, "_mlp_channel_antithetic_patched", False):
            if getattr(module, "_mlp_channel_antithetic_controller", None) is not controller:
                raise RuntimeError(f"vLLM MLP {name} was patched with a different controller")
            _install_vllm_active_buffer(module, controller, layer_idx)
            patched.append(name)
            seen_layers.add(layer_idx)
            continue

        _install_vllm_active_buffer(module, controller, layer_idx)

        def forward(this, hidden_state, *args, **kwargs):
            if args or kwargs:
                raise TypeError("patched vLLM dense MLP expects only hidden_state")
            gate_up = this.gate_up_proj(hidden_state)
            gate_up = gate_up[0] if isinstance(gate_up, tuple) else gate_up
            activation = this.act_fn(gate_up)
            activation = activation * getattr(this, _VLLM_ACTIVE_GAIN_BUFFER)
            output = this.down_proj(activation)
            return output[0] if isinstance(output, tuple) else output

        module.forward = MethodType(forward, module)
        module._mlp_channel_antithetic_patched = True
        module._mlp_channel_antithetic_controller = controller
        patched.append(name)
        seen_layers.add(layer_idx)

    _validate_patched_layers(controller, seen_layers, backend="vLLM rollout")
    return patched


def install_vllm_class_intervention(
    controller: MLPChannelAntitheticController,
) -> list[str]:
    """Patch vLLM Qwen MLP classes before model build and CUDA-graph capture."""
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
        existing_controller = getattr(cls, "_mlp_channel_antithetic_controller", None)
        if existing_controller is not None:
            if existing_controller is not controller:
                raise RuntimeError(f"{cls.__name__} is already patched with another controller")
            patched_names.append(cls.__name__)
            continue

        original_init = cls.__init__

        def patched_init(this, *args, __original_init=original_init, **kwargs):
            __original_init(this, *args, **kwargs)
            prefix = kwargs.get("prefix", args[4] if len(args) > 4 else "")
            layer_idx = _layer_index(str(prefix))
            if layer_idx is None:
                raise RuntimeError(f"cannot infer vLLM MLP layer from prefix {prefix!r}")
            this._mlp_channel_antithetic_layer_idx = layer_idx
            this._mlp_channel_antithetic_patched = True
            this._mlp_channel_antithetic_controller = controller
            _install_vllm_active_buffer(this, controller, layer_idx)

        def patched_forward(this, hidden_state):
            gate_up = this.gate_up_proj(hidden_state)
            gate_up = gate_up[0] if isinstance(gate_up, tuple) else gate_up
            activation = this.act_fn(gate_up)
            activation = activation * getattr(this, _VLLM_ACTIVE_GAIN_BUFFER)
            output = this.down_proj(activation)
            return output[0] if isinstance(output, tuple) else output

        cls.__init__ = patched_init
        cls.forward = patched_forward
        cls._mlp_channel_antithetic_controller = controller
        patched_names.append(cls.__name__)
    return patched_names


def _layer_index(module_name: str) -> int | None:
    normalized_name = ".".join(
        part for part in module_name.split(".") if part != "_fsdp_wrapped_module"
    )
    match = _LAYER_RE.search(normalized_name)
    return int(match.group(1)) if match else None


def _validate_patched_layers(
    controller: MLPChannelAntitheticController,
    seen_layers: set[int],
    *,
    backend: str,
) -> None:
    expected = set(range(controller.num_layers))
    if seen_layers != expected:
        missing = sorted(expected - seen_layers)
        extra = sorted(seen_layers - expected)
        raise RuntimeError(
            f"{backend}: expected dense MLPs for {controller.num_layers} layers; "
            f"missing={missing}, extra={extra}. MoE and unknown layouts are unsupported."
        )
