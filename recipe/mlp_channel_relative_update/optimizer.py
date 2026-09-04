"""AdamW with structured post-preconditioner MLP-channel multipliers."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

import torch

try:
    from torch.distributed.tensor import DTensor, Shard
except ImportError:  # pragma: no cover - older CPU-only torch builds
    DTensor = ()  # type: ignore[assignment,misc]
    Shard = ()  # type: ignore[assignment,misc]

from .relative_update import MLPChannelRelativeUpdateController

_MLP_WEIGHT_RE = re.compile(
    r"(?:^|\.)(?:layers|h)\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$"
)


@dataclass(frozen=True)
class _ChannelParameterLayout:
    name: str
    layer_slot: int
    projection: str
    channel_axis: int
    global_shape: tuple[int, int]
    local_channel_start: int


@dataclass
class _PreparedParameter:
    parameter: torch.Tensor
    parameter_local: torch.Tensor
    exp_avg_local: torch.Tensor
    exp_avg_sq_local: torch.Tensor
    step_size: float
    bias_correction2_sqrt: float
    eps: float
    lr: float
    weight_decay: float
    layout: _ChannelParameterLayout | None


class ChannelRelativeUpdateAdamW(torch.optim.AdamW):
    """Apply channel multipliers after AdamW moment normalization.

    The class deliberately supports the simple AdamW mode used by the recipe:
    no AMSGrad, fused/capturable/differentiable optimizer, sparse gradients, or
    complex parameters.  FSDP2 DTensors are updated through their local shards;
    ordinary tensors are supported to make the algorithm CPU-testable.
    """

    def __init__(self, params: Iterable[torch.Tensor], *args, **kwargs) -> None:
        super().__init__(params, *args, **kwargs)
        self.relative_update_controller: MLPChannelRelativeUpdateController | None = None
        self._channel_layouts: dict[int, _ChannelParameterLayout] = {}

    def configure_channel_updates(
        self,
        *,
        controller: MLPChannelRelativeUpdateController,
        named_parameters: Iterable[tuple[str, torch.Tensor]],
    ) -> list[str]:
        """Resolve the three weight slices belonging to every selected channel."""

        optimizer_parameter_ids = {
            id(parameter)
            for group in self.param_groups
            for parameter in group["params"]
        }
        layouts: dict[int, _ChannelParameterLayout] = {}
        found: dict[int, set[str]] = {
            layer: set() for layer in controller.selected_layers
        }
        installed_names: list[str] = []

        for name, parameter in named_parameters:
            normalized_name = ".".join(
                part for part in name.split(".") if part != "_fsdp_wrapped_module"
            )
            match = _MLP_WEIGHT_RE.search(normalized_name)
            if match is None:
                continue
            layer_idx = int(match.group(1))
            projection = match.group(2)
            if layer_idx not in controller.layer_to_slot:
                continue
            if id(parameter) not in optimizer_parameter_ids:
                raise RuntimeError(
                    f"selected MLP parameter {name!r} is absent from the actor optimizer"
                )
            if parameter.ndim != 2:
                raise RuntimeError(
                    f"selected MLP parameter {name!r} must be logically 2-D, "
                    f"got shape {tuple(parameter.shape)}"
                )

            global_shape = tuple(int(dim) for dim in parameter.shape)
            channel_axis = 0 if projection in {"gate_proj", "up_proj"} else 1
            if global_shape[channel_axis] != controller.intermediate_size:
                raise RuntimeError(
                    f"{name!r} channel dimension {global_shape[channel_axis]} != "
                    f"intermediate_size {controller.intermediate_size}"
                )
            local_channel_start = self._local_channel_start(
                parameter,
                channel_axis=channel_axis,
                global_shape=global_shape,
            )
            if id(parameter) in layouts:
                raise RuntimeError(f"selected MLP parameter {name!r} was resolved twice")
            layouts[id(parameter)] = _ChannelParameterLayout(
                name=name,
                layer_slot=controller.layer_to_slot[layer_idx],
                projection=projection,
                channel_axis=channel_axis,
                global_shape=global_shape,
                local_channel_start=local_channel_start,
            )
            found[layer_idx].add(projection)
            installed_names.append(name)

        required = {"gate_proj", "up_proj", "down_proj"}
        incomplete = {
            layer: sorted(required - projections)
            for layer, projections in found.items()
            if projections != required
        }
        if incomplete:
            raise RuntimeError(
                "could not resolve a complete dense SwiGLU channel parameter group; "
                f"missing projections by layer: {incomplete}"
            )

        self._validate_optimizer_mode()
        self.relative_update_controller = controller
        self._channel_layouts = layouts
        return installed_names

    @torch.no_grad()
    def step(self, closure=None):
        controller = self.relative_update_controller
        if controller is None or not self._channel_layouts:
            raise RuntimeError(
                "ChannelRelativeUpdateAdamW must be configured with named MLP parameters "
                "before the first optimizer step"
            )
        self._cuda_graph_capture_health_check()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        prepared: list[_PreparedParameter] = []
        for group in self.param_groups:
            params_with_grad: list[torch.Tensor] = []
            grads: list[torch.Tensor] = []
            exp_avgs: list[torch.Tensor] = []
            exp_avg_sqs: list[torch.Tensor] = []
            max_exp_avg_sqs: list[torch.Tensor] = []
            state_steps: list[torch.Tensor] = []
            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )
            if has_complex:
                raise NotImplementedError("relative-update AdamW does not support complex parameters")

            beta1, beta2 = group["betas"]
            lr = self._scalar_float(group["lr"], name="lr")
            for parameter, gradient, exp_avg, exp_avg_sq, state_step in zip(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
            ):
                parameter_local = self._local_tensor(parameter)
                gradient_local = self._local_tensor(gradient)
                exp_avg_local = self._local_tensor(exp_avg)
                exp_avg_sq_local = self._local_tensor(exp_avg_sq)
                if group["maximize"]:
                    gradient_local = -gradient_local

                state_step.add_(1)
                step_number = float(state_step.item())
                exp_avg_local.lerp_(gradient_local, 1.0 - beta1)
                exp_avg_sq_local.mul_(beta2).addcmul_(
                    gradient_local,
                    gradient_local,
                    value=1.0 - beta2,
                )
                bias_correction1 = 1.0 - beta1**step_number
                bias_correction2_sqrt = math.sqrt(1.0 - beta2**step_number)
                prepared.append(
                    _PreparedParameter(
                        parameter=parameter,
                        parameter_local=parameter_local,
                        exp_avg_local=exp_avg_local,
                        exp_avg_sq_local=exp_avg_sq_local,
                        step_size=lr / bias_correction1,
                        bias_correction2_sqrt=bias_correction2_sqrt,
                        eps=float(group["eps"]),
                        lr=lr,
                        weight_decay=float(group["weight_decay"]),
                        layout=self._channel_layouts.get(id(parameter)),
                    )
                )

        selected = [item for item in prepared if item.layout is not None]
        if len(selected) != len(self._channel_layouts):
            missing = sorted(
                layout.name
                for parameter_id, layout in self._channel_layouts.items()
                if all(id(item.parameter) != parameter_id for item in selected)
            )
            raise RuntimeError(
                "selected MLP parameters did not all receive gradients in this optimizer step: "
                f"{missing}"
            )

        stats_device = selected[0].parameter_local.device
        stats_shape = (len(controller.selected_layers), controller.intermediate_size)
        local_base_update_sq = torch.zeros(stats_shape, device=stats_device, dtype=torch.float32)
        local_parameter_sq = torch.zeros_like(local_base_update_sq)
        local_parameter_count = torch.zeros_like(local_base_update_sq)
        for item in selected:
            denominator = self._denominator(item)
            self._accumulate_channel_statistics(
                item,
                denominator,
                local_base_update_sq=local_base_update_sq,
                local_parameter_sq=local_parameter_sq,
                local_parameter_count=local_parameter_count,
            )
            del denominator

        relative_step = controller.prepare_step(
            local_base_update_sq=local_base_update_sq,
            local_parameter_sq=local_parameter_sq,
            local_parameter_count=local_parameter_count,
        )
        for item in prepared:
            if item.weight_decay != 0.0:
                item.parameter_local.mul_(1.0 - item.lr * item.weight_decay)
            denominator = self._denominator(item)
            if item.layout is None:
                item.parameter_local.addcdiv_(
                    item.exp_avg_local,
                    denominator,
                    value=-item.step_size,
                )
            else:
                direction = item.exp_avg_local / denominator
                direction.mul_(
                    self._local_multiplier_view(
                        relative_step.multipliers,
                        item.layout,
                        item.parameter_local,
                    )
                )
                item.parameter_local.add_(direction, alpha=-item.step_size)
            del denominator

        controller.commit_step(relative_step)
        return loss

    def get_last_relative_update_metrics(self) -> dict[str, float]:
        controller = self.relative_update_controller
        return {} if controller is None else dict(controller.last_metrics)

    def _validate_optimizer_mode(self) -> None:
        for group in self.param_groups:
            unsupported = [
                name
                for name in ("amsgrad", "capturable", "differentiable", "fused")
                if bool(group.get(name, False))
            ]
            if unsupported:
                raise NotImplementedError(
                    "relative-update AdamW does not support optimizer options "
                    f"{unsupported}"
                )

    @staticmethod
    def _scalar_float(value, *, name: str) -> float:
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError(f"optimizer {name} tensor must contain one value")
            return float(value.detach().item())
        return float(value)

    @staticmethod
    def _local_tensor(value: torch.Tensor) -> torch.Tensor:
        if DTensor and isinstance(value, DTensor):
            return value.to_local()
        return value

    @staticmethod
    def _local_channel_start(
        parameter: torch.Tensor,
        *,
        channel_axis: int,
        global_shape: tuple[int, int],
    ) -> int:
        if not DTensor or not isinstance(parameter, DTensor):
            return 0
        sharded_placements = [
            (mesh_dim, placement)
            for mesh_dim, placement in enumerate(parameter.placements)
            if isinstance(placement, Shard)
        ]
        if len(sharded_placements) > 1:
            raise NotImplementedError(
                "relative-update AdamW currently supports at most one FSDP2 Shard placement"
            )
        if not sharded_placements:
            return 0
        mesh_dim, placement = sharded_placements[0]
        if int(placement.dim) != channel_axis:
            return 0
        mesh_size = int(parameter.device_mesh.size(mesh_dim))
        mesh_rank = int(parameter.device_mesh.get_local_rank(mesh_dim))
        _, offset = Shard.local_shard_size_and_offset(
            global_shape[channel_axis],
            mesh_size,
            mesh_rank,
        )
        return int(offset)

    @staticmethod
    def _denominator(item: _PreparedParameter) -> torch.Tensor:
        return (
            item.exp_avg_sq_local.sqrt()
            .div_(item.bias_correction2_sqrt)
            .add_(item.eps)
        )

    @staticmethod
    def _accumulate_channel_statistics(
        item: _PreparedParameter,
        denominator: torch.Tensor,
        *,
        local_base_update_sq: torch.Tensor,
        local_parameter_sq: torch.Tensor,
        local_parameter_count: torch.Tensor,
    ) -> None:
        layout = item.layout
        assert layout is not None
        if item.parameter_local.ndim != 2:
            raise RuntimeError(
                f"local shard for {layout.name!r} must remain 2-D under FSDP2, "
                f"got {tuple(item.parameter_local.shape)}"
            )
        channel_axis = layout.channel_axis
        reduction_axis = 1 - channel_axis
        local_channel_count = int(item.parameter_local.shape[channel_axis])
        start = layout.local_channel_start
        end = start + local_channel_count
        if end > layout.global_shape[channel_axis]:
            raise RuntimeError(
                f"local channel range [{start}, {end}) exceeds {layout.name!r} shape "
                f"{layout.global_shape}"
            )
        if local_channel_count == 0:
            return

        direction = item.exp_avg_local / denominator
        update_sq = direction.to(torch.float32).square().sum(dim=reduction_axis)
        update_sq.mul_(item.step_size**2)
        parameter_sq = item.parameter_local.to(torch.float32).square().sum(
            dim=reduction_axis
        )
        count = float(item.parameter_local.shape[reduction_axis])
        slot = layout.layer_slot
        local_base_update_sq[slot, start:end].add_(update_sq)
        local_parameter_sq[slot, start:end].add_(parameter_sq)
        local_parameter_count[slot, start:end].add_(count)

    @staticmethod
    def _local_multiplier_view(
        multipliers: torch.Tensor,
        layout: _ChannelParameterLayout,
        parameter_local: torch.Tensor,
    ) -> torch.Tensor:
        local_channel_count = int(parameter_local.shape[layout.channel_axis])
        start = layout.local_channel_start
        values = multipliers[
            layout.layer_slot,
            start : start + local_channel_count,
        ].to(device=parameter_local.device, dtype=parameter_local.dtype)
        return values.unsqueeze(1 if layout.channel_axis == 0 else 0)
