"""Two independent, step-stable hard masks over dense SwiGLU channels."""

from __future__ import annotations

from typing import Any

import torch

MASK_A_ROUTE = "mask_a"
MASK_B_ROUTE = "mask_b"
CLEAN_ROUTE = "clean"
TRAINING_ROUTES = (MASK_A_ROUTE, MASK_B_ROUTE)


class MLPChannelRDropController:
    valid_routes = TRAINING_ROUTES
    metric_prefix = "mlp_rdrop"
    batch_version_field = "mask_version"

    def __init__(self, *, num_layers: int, intermediate_size: int,
                 mask_ratio: float = 0.10, random_seed: int = 42,
                 tp_rank: int = 0, tp_size: int = 1, name: str = "actor"):
        if num_layers <= 0 or intermediate_size <= 0:
            raise ValueError("MLP dimensions must be positive")
        if not 0 <= mask_ratio < 1:
            raise ValueError("mask_ratio must be in [0, 1); zero is the clean control")
        if random_seed < 0 or not 0 <= tp_rank < tp_size or intermediate_size % tp_size:
            raise ValueError("invalid seed or tensor-parallel layout")
        self.num_layers = int(num_layers)
        self.intermediate_size = int(intermediate_size)
        self.mask_ratio = float(mask_ratio)
        self.random_seed = int(random_seed)
        self.tp_rank, self.tp_size = int(tp_rank), int(tp_size)
        self.name = name
        self.masked_per_layer = round(mask_ratio * intermediate_size)
        if mask_ratio > 0 and not 0 < self.masked_per_layer < intermediate_size:
            raise ValueError("mask_ratio rounds to zero or all channels")
        self.keep_masks = torch.ones((2, num_layers, intermediate_size), dtype=torch.bool)
        self.mask_version = -1
        self.route = CLEAN_ROUTE
        self._active_buffers = {}
        self._active_buffers_available = True
        self.refresh_masks(version=0)

    def refresh_masks(self, *, version: int) -> bool:
        if version < 0:
            raise ValueError("mask version must be non-negative")
        if version == self.mask_version:
            return False
        # Different streams for A/B, identical global masks on actor and rollout
        # ranks. Masks may overlap: the pair is independent, not complementary.
        self.keep_masks.fill_(True)
        for route_index in range(2):
            generator = torch.Generator(device="cpu").manual_seed(
                (self.random_seed + 2 * int(version) + route_index) % (2**63 - 1)
            )
            for layer in range(self.num_layers):
                indices = torch.randperm(self.intermediate_size, generator=generator)
                self.keep_masks[route_index, layer, indices[:self.masked_per_layer]] = False
        self.mask_version = int(version)
        self._refresh_active_buffers()
        return True

    def validate_batch_version(self, values: Any) -> None:
        versions = torch.unique(torch.tensor(values, dtype=torch.int64)).tolist()
        if versions != [self.mask_version]:
            raise RuntimeError(f"mask version mismatch: batch={versions}, actor={self.mask_version}")

    def should_collect_activation(self, route: str) -> bool:
        return False

    def set_route(self, route: str, *, collect_activation: bool = False) -> None:
        if route not in (*TRAINING_ROUTES, CLEAN_ROUTE):
            raise ValueError(f"unknown route {route!r}")
        if collect_activation:
            raise ValueError("random R-Drop masks do not collect channel scores")
        if self.route != route:
            self.route = route
            self._refresh_active_buffers()

    def end_batch(self) -> None:
        pass

    def _local_slice(self, width: int) -> tuple[int, int]:
        if width == self.intermediate_size:
            return 0, width
        if width != self.intermediate_size // self.tp_size:
            raise RuntimeError(f"unsupported MLP width {width}")
        return self.tp_rank * width, (self.tp_rank + 1) * width

    def gain(self, layer: int, route: str | None = None) -> torch.Tensor:
        if not 0 <= layer < self.num_layers:
            raise IndexError(layer)
        route = self.route if route is None else route
        if route == CLEAN_ROUTE:
            return torch.ones(self.intermediate_size, dtype=torch.bool)
        if route not in TRAINING_ROUTES:
            raise ValueError(f"unknown route {route!r}")
        return self.keep_masks[TRAINING_ROUTES.index(route), layer]

    def _buffer_key(self, layer: int, tensor: torch.Tensor):
        return layer, tensor.device.type, tensor.device.index, tensor.dtype, tensor.shape[-1]

    def _copy_to_buffer(self, layer: int, buffer: torch.Tensor) -> None:
        start, stop = self._local_slice(buffer.numel())
        buffer.copy_(self.gain(layer)[start:stop].to(device=buffer.device, dtype=buffer.dtype))

    def register_active_buffer(self, layer: int, buffer: torch.Tensor) -> torch.Tensor:
        self._local_slice(buffer.numel())
        key = self._buffer_key(layer, buffer)
        existing = self._active_buffers.get(key)
        if existing is not None and existing is not buffer:
            raise RuntimeError(f"duplicate active mask buffer: {key}")
        self._active_buffers[key] = buffer
        if self._active_buffers_available and buffer.device.type != "meta":
            self._copy_to_buffer(layer, buffer)
        return buffer

    def apply(self, layer: int, activation: torch.Tensor) -> torch.Tensor:
        key = self._buffer_key(layer, activation)
        buffer = self._active_buffers.get(key)
        if buffer is None:
            buffer = self.register_active_buffer(layer, activation.new_ones(activation.shape[-1]))
        return activation * buffer

    def set_active_buffers_available(self, available: bool) -> None:
        self._active_buffers_available = bool(available)
        if available:
            self._refresh_active_buffers()

    @torch.no_grad()
    def _refresh_active_buffers(self) -> None:
        if self._active_buffers_available:
            for (layer, *_), buffer in self._active_buffers.items():
                if buffer.device.type != "meta":
                    self._copy_to_buffer(layer, buffer)

    def metrics(self) -> dict[str, float]:
        dropped = ~self.keep_masks
        return {
            "mlp_rdrop/mask_version": float(self.mask_version),
            "mlp_rdrop/mask_ratio_requested": self.mask_ratio,
            "mlp_rdrop/masked_per_layer": float(self.masked_per_layer),
            "mlp_rdrop/mask_a_fraction": float(dropped[0].float().mean()),
            "mlp_rdrop/mask_b_fraction": float(dropped[1].float().mean()),
            "mlp_rdrop/both_masked_fraction": float((dropped[0] & dropped[1]).float().mean()),
            "mlp_rdrop/either_masked_fraction": float((dropped[0] | dropped[1]).float().mean()),
            "mlp_rdrop/inverted_dropout_scaling": 0.0,
        }

    def state_dict(self) -> dict:
        return dict(checkpoint_format="mlp_channel_rdrop_v1", num_layers=self.num_layers,
                    intermediate_size=self.intermediate_size, mask_ratio=self.mask_ratio,
                    random_seed=self.random_seed, mask_version=self.mask_version,
                    keep_masks=self.keep_masks.clone())

    def load_state_dict(self, state: dict) -> None:
        expected = self.state_dict()
        for key in ("checkpoint_format", "num_layers", "intermediate_size", "mask_ratio", "random_seed"):
            if state.get(key) != expected[key]:
                raise ValueError(f"checkpoint {key} differs from current configuration")
        masks = torch.as_tensor(state["keep_masks"])
        if masks.dtype != torch.bool or masks.shape != self.keep_masks.shape:
            raise ValueError("invalid checkpoint mask layout")
        if not bool(((~masks).sum(-1) == self.masked_per_layer).all()):
            raise ValueError("checkpoint masks violate the per-layer quota")
        if int(state["mask_version"]) < 0:
            raise ValueError("invalid checkpoint version")
        self.keep_masks.copy_(masks)
        self.mask_version = int(state["mask_version"])
        self.set_route(CLEAN_ROUTE)
        self._refresh_active_buffers()
