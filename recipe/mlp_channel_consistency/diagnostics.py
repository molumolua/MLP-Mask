"""Low-overhead gradient and parameter-update diagnostics for the recipe."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

import torch
import torch.distributed as dist

try:
    from torch.distributed.tensor import DTensor, Shard
except ImportError:  # pragma: no cover - older PyTorch fallback
    DTensor = ()
    Shard = ()


_MLP_WEIGHT_RE = re.compile(
    r"(?:^|\.)(?:layers|h)\.(\d+)\.mlp\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    if DTensor and isinstance(value, DTensor):
        return value.to_local()
    return value


def _collective_device(reference: torch.device) -> torch.device:
    if not dist.is_available() or not dist.is_initialized():
        return reference
    backend = str(dist.get_backend()).lower()
    if "nccl" in backend:
        return torch.device("cuda", torch.cuda.current_device())
    return reference


def _all_reduce_sum(values: list[float], device: torch.device) -> list[float]:
    stats = torch.tensor(values, dtype=torch.float64, device=_collective_device(device))
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return [float(value) for value in stats.cpu().tolist()]


@dataclass
class _GradientSample:
    name: str
    parameter: torch.nn.Parameter
    expected_numel: int
    indices_cpu: torch.Tensor
    indices: torch.Tensor
    main: torch.Tensor
    auxiliary: torch.Tensor

    def move_accumulators(self, device: torch.device) -> None:
        if self.indices.device == device:
            return
        self.indices = self.indices_cpu.to(device=device)
        self.main = torch.zeros(self.indices.numel(), dtype=torch.float32, device=device)
        self.auxiliary = torch.zeros_like(self.main)


class SampledGradientTracker:
    """Track branch gradients on a fixed stratified sample of coordinates.

    The sampled vectors are accumulated across every backward in one optimizer
    update.  Their RMS ratio therefore includes the configured KL coefficient and
    remains valid when the KL backward is split into multiple micro-batches.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        *,
        sample_size_per_rank: int,
        random_seed: int,
    ) -> None:
        if sample_size_per_rank <= 0:
            raise ValueError("sample_size_per_rank must be positive")

        named_parameters = [
            (name, parameter)
            for name, parameter in module.named_parameters()
            if parameter.requires_grad and _local_tensor(parameter).numel() > 0
        ]
        total_numel = sum(_local_tensor(parameter).numel() for _, parameter in named_parameters)
        if total_numel <= 0:
            raise RuntimeError("gradient diagnostics found no trainable parameters")

        sample_count = min(int(sample_size_per_rank), total_numel)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(random_seed))
        # One random coordinate from every equal-width interval gives a fixed,
        # memory-bounded stratified sample without allocating a full permutation.
        positions = torch.floor(
            (
                torch.arange(sample_count, dtype=torch.float64)
                + torch.rand(sample_count, generator=generator, dtype=torch.float64)
            )
            * (float(total_numel) / float(sample_count))
        ).to(dtype=torch.int64)

        self.total_numel = int(total_numel)
        self.samples: list[_GradientSample] = []
        self._active = False
        self._awaiting_auxiliary = False
        offset = 0
        for name, parameter in named_parameters:
            local_parameter = _local_tensor(parameter)
            numel = int(local_parameter.numel())
            begin = int(torch.searchsorted(positions, offset, right=False).item())
            finish = int(torch.searchsorted(positions, offset + numel, right=False).item())
            local_indices_cpu = positions[begin:finish] - offset
            offset += numel
            if local_indices_cpu.numel() == 0:
                continue
            device = local_parameter.device
            indices = local_indices_cpu.to(device=device)
            sample = _GradientSample(
                name=name,
                parameter=parameter,
                expected_numel=numel,
                indices_cpu=local_indices_cpu,
                indices=indices,
                main=torch.zeros(indices.numel(), dtype=torch.float32, device=device),
                auxiliary=torch.zeros(indices.numel(), dtype=torch.float32, device=device),
            )
            self.samples.append(sample)

        self.sample_count = sum(sample.indices_cpu.numel() for sample in self.samples)
        if self.sample_count != sample_count:
            raise RuntimeError(
                f"gradient diagnostics sampled {self.sample_count} coordinates, "
                f"expected {sample_count}"
            )

    @staticmethod
    def _current_values(sample: _GradientSample) -> torch.Tensor:
        gradient = sample.parameter.grad
        if gradient is None:
            parameter_device = _local_tensor(sample.parameter.detach()).device
            sample.move_accumulators(parameter_device)
            return torch.zeros_like(sample.main)
        local_gradient = _local_tensor(gradient.detach())
        if local_gradient.numel() != sample.expected_numel:
            raise RuntimeError(
                f"gradient shard for {sample.name!r} changed size from "
                f"{sample.expected_numel} to {local_gradient.numel()}"
            )
        sample.move_accumulators(local_gradient.device)
        return (
            local_gradient.reshape(-1)
            .index_select(0, sample.indices)
            .to(dtype=torch.float32)
        )

    def start_update(self) -> None:
        for sample in self.samples:
            sample.main.zero_()
            sample.auxiliary.zero_()
        self._active = True
        self._awaiting_auxiliary = False

    def capture_main_gradient(self) -> None:
        """Capture cumulative main gradients immediately before auxiliary backward."""
        if not self._active:
            raise RuntimeError("gradient diagnostics are not tracking an update")
        if self._awaiting_auxiliary:
            raise RuntimeError("previous auxiliary gradient has not been captured")
        for sample in self.samples:
            # Before auxiliary backward, param.grad = cumulative_main + prior_aux.
            current = self._current_values(sample)
            sample.main.copy_(current - sample.auxiliary)
        self._awaiting_auxiliary = True

    def capture_auxiliary_gradient(self) -> None:
        """Capture cumulative weighted auxiliary gradients after its backward."""
        if not self._active:
            raise RuntimeError("gradient diagnostics are not tracking an update")
        if not self._awaiting_auxiliary:
            raise RuntimeError("main gradient must be captured before auxiliary gradient")
        for sample in self.samples:
            # After auxiliary backward, param.grad = cumulative_main + cumulative_aux.
            current = self._current_values(sample)
            sample.auxiliary.copy_(current - sample.main)
        self._awaiting_auxiliary = False

    def cancel_update(self) -> None:
        self._active = False
        self._awaiting_auxiliary = False

    def finish_update(self) -> dict[str, float]:
        if not self._active:
            raise RuntimeError("gradient diagnostics are not tracking an update")
        if self._awaiting_auxiliary:
            raise RuntimeError("auxiliary gradient capture is incomplete")
        self._active = False

        reference_device = self.samples[0].main.device
        # Concatenate the bounded sample once so metric collection introduces one
        # device synchronization instead of one per parameter tensor.
        main_vector = torch.cat([sample.main for sample in self.samples])
        auxiliary_vector = torch.cat([sample.auxiliary for sample in self.samples])
        main_sq = float(main_vector.double().square().sum().item())
        auxiliary_sq = float(auxiliary_vector.double().square().sum().item())
        dot = float((main_vector.double() * auxiliary_vector.double()).sum().item())
        main_sq, auxiliary_sq, dot, sample_count, total_numel = _all_reduce_sum(
            [main_sq, auxiliary_sq, dot, float(self.sample_count), float(self.total_numel)],
            reference_device,
        )

        main_rms = math.sqrt(main_sq / sample_count) if sample_count > 0 else 0.0
        auxiliary_rms = math.sqrt(auxiliary_sq / sample_count) if sample_count > 0 else 0.0
        ratio = auxiliary_rms / main_rms if main_rms > 0.0 else 0.0
        cosine_denom = math.sqrt(main_sq * auxiliary_sq)
        cosine = dot / cosine_denom if cosine_denom > 0.0 else 0.0
        return {
            "mlp_consistency/main_grad_rms_sampled": main_rms,
            "mlp_consistency/aux_grad_rms_sampled": auxiliary_rms,
            "mlp_consistency/aux_to_main_grad_ratio_sampled": ratio,
            "mlp_consistency/main_aux_grad_cosine_sampled": cosine,
            "mlp_consistency/gradient_sample_fraction": sample_count / total_numel,
            "mlp_consistency/gradient_sample_count": sample_count,
        }


@dataclass
class _ChannelParameterLayout:
    layer_idx: int
    projection: str
    channel_axis: int
    global_shape: tuple[int, int]
    local_channel_start: int


@dataclass
class _ParameterSnapshot:
    name: str
    parameter: torch.nn.Parameter
    initial_bfloat16_cpu: torch.Tensor
    channel_layout: _ChannelParameterLayout | None = None


class ParameterUpdateTracker:
    """Compare current shards with the pre-RL BF16 parameters on validation."""

    def __init__(
        self,
        module: torch.nn.Module,
        *,
        num_layers: int | None = None,
        intermediate_size: int | None = None,
    ) -> None:
        if (num_layers is None) != (intermediate_size is None):
            raise ValueError(
                "num_layers and intermediate_size must be provided together"
            )
        if num_layers is not None and (
            int(num_layers) <= 0 or int(intermediate_size) <= 0
        ):
            raise ValueError("channel tracking dimensions must be positive")
        self.channel_shape = (
            None
            if num_layers is None
            else (int(num_layers), int(intermediate_size))
        )
        self.snapshots: list[_ParameterSnapshot] = []
        found_projections: dict[int, set[str]] = (
            {}
            if self.channel_shape is None
            else {layer_idx: set() for layer_idx in range(self.channel_shape[0])}
        )
        with torch.no_grad():
            for name, parameter in module.named_parameters():
                local_parameter = _local_tensor(parameter.detach())
                if local_parameter.numel() == 0:
                    continue
                channel_layout = self._channel_layout(name, parameter)
                if channel_layout is not None:
                    found_projections[channel_layout.layer_idx].add(
                        channel_layout.projection
                    )
                self.snapshots.append(
                    _ParameterSnapshot(
                        name=name,
                        parameter=parameter,
                        initial_bfloat16_cpu=local_parameter.to(
                            device="cpu", dtype=torch.bfloat16
                        ).clone(),
                        channel_layout=channel_layout,
                    )
                )
        if not self.snapshots:
            raise RuntimeError("parameter-update diagnostics found no model parameters")
        if self.channel_shape is not None:
            required = {"gate_proj", "up_proj", "down_proj"}
            incomplete = {
                layer_idx: sorted(required - projections)
                for layer_idx, projections in found_projections.items()
                if projections != required
            }
            if incomplete:
                raise RuntimeError(
                    "could not resolve complete dense SwiGLU channel parameters; "
                    f"missing projections by layer: {incomplete}"
                )

    def _channel_layout(
        self, name: str, parameter: torch.nn.Parameter
    ) -> _ChannelParameterLayout | None:
        if self.channel_shape is None:
            return None
        normalized_name = ".".join(
            part for part in name.split(".") if part != "_fsdp_wrapped_module"
        )
        match = _MLP_WEIGHT_RE.search(normalized_name)
        if match is None:
            return None
        layer_idx = int(match.group(1))
        projection = match.group(2)
        num_layers, intermediate_size = self.channel_shape
        if not 0 <= layer_idx < num_layers:
            raise RuntimeError(
                f"MLP parameter {name!r} resolves to layer {layer_idx}, "
                f"outside [0, {num_layers})"
            )
        if parameter.ndim != 2:
            raise RuntimeError(
                f"MLP parameter {name!r} must remain logically 2-D under FSDP2"
            )
        global_shape = tuple(int(dim) for dim in parameter.shape)
        channel_axis = 0 if projection in {"gate_proj", "up_proj"} else 1
        if global_shape[channel_axis] != intermediate_size:
            raise RuntimeError(
                f"MLP parameter {name!r} channel dimension "
                f"{global_shape[channel_axis]} != {intermediate_size}"
            )
        return _ChannelParameterLayout(
            layer_idx=layer_idx,
            projection=projection,
            channel_axis=channel_axis,
            global_shape=global_shape,
            local_channel_start=self._local_channel_start(
                parameter,
                channel_axis=channel_axis,
                global_shape=global_shape,
            ),
        )

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
                "updated_fraction supports at most one FSDP2 Shard placement"
            )
        if not sharded_placements:
            return 0
        mesh_dim, placement = sharded_placements[0]
        if int(placement.dim) != channel_axis:
            return 0
        mesh_size = int(parameter.device_mesh.size(mesh_dim))
        mesh_rank = int(parameter.device_mesh.get_local_rank(mesh_dim))
        _, offset = Shard.local_shard_size_and_offset(
            global_shape[channel_axis], mesh_size, mesh_rank
        )
        return int(offset)

    @torch.no_grad()
    def local_statistics(self, *, atol: float) -> tuple[float, float, float, float]:
        if atol < 0.0:
            raise ValueError("parameter update atol must be non-negative")
        unchanged = 0
        total = 0
        absolute_delta_sum = 0.0
        squared_delta_sum = 0.0
        for snapshot in self.snapshots:
            current = _local_tensor(snapshot.parameter.detach()).to(
                device="cpu", dtype=torch.bfloat16
            )
            if current.shape != snapshot.initial_bfloat16_cpu.shape:
                raise RuntimeError(
                    f"parameter shard for {snapshot.name!r} changed shape from "
                    f"{tuple(snapshot.initial_bfloat16_cpu.shape)} to {tuple(current.shape)}"
                )
            # Match Mukherjee et al.: subtract BF16 checkpoints and regard a
            # coordinate as unchanged when its delta is close to zero at atol.
            delta_bfloat16 = current - snapshot.initial_bfloat16_cpu
            close_to_zero = torch.isclose(
                delta_bfloat16,
                torch.zeros((), dtype=torch.bfloat16),
                atol=float(atol),
            )
            delta_float = delta_bfloat16.float()
            unchanged += int(close_to_zero.sum().item())
            total += int(delta_bfloat16.numel())
            absolute_delta_sum += float(delta_float.abs().sum().item())
            squared_delta_sum += float(delta_float.double().square().sum().item())
        return float(unchanged), float(total), absolute_delta_sum, squared_delta_sum

    def distributed_metrics(self, *, atol: float) -> dict[str, float]:
        reference_device = _local_tensor(self.snapshots[0].parameter.detach()).device
        global_statistics = _all_reduce_sum(
            list(self.local_statistics(atol=atol)),
            reference_device,
        )
        return self.metrics_from_global_statistics(global_statistics, atol=atol)

    @torch.no_grad()
    def local_channel_statistics(
        self, *, atol: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Count pre-RL-updated BF16 coordinates for each complete MLP channel."""
        if self.channel_shape is None:
            raise RuntimeError("parameter tracker was not configured for channels")
        if atol < 0.0:
            raise ValueError("parameter update atol must be non-negative")
        updated = torch.zeros(self.channel_shape, dtype=torch.float64)
        total = torch.zeros_like(updated)
        zero = torch.zeros((), dtype=torch.bfloat16)
        for snapshot in self.snapshots:
            layout = snapshot.channel_layout
            if layout is None:
                continue
            current = _local_tensor(snapshot.parameter.detach()).to(
                device="cpu", dtype=torch.bfloat16
            )
            if current.shape != snapshot.initial_bfloat16_cpu.shape:
                raise RuntimeError(
                    f"parameter shard for {snapshot.name!r} changed shape"
                )
            delta = current - snapshot.initial_bfloat16_cpu
            changed = ~torch.isclose(delta, zero, atol=float(atol))
            reduction_axis = 1 - layout.channel_axis
            local_updated = changed.sum(dim=reduction_axis).to(torch.float64)
            local_channel_count = int(current.shape[layout.channel_axis])
            start = layout.local_channel_start
            stop = start + local_channel_count
            if stop > self.channel_shape[1]:
                raise RuntimeError(
                    f"local channel range [{start}, {stop}) exceeds "
                    f"intermediate_size={self.channel_shape[1]}"
                )
            updated[layout.layer_idx, start:stop].add_(local_updated)
            total[layout.layer_idx, start:stop].add_(
                float(current.shape[reduction_axis])
            )
        return updated, total

    def distributed_channel_updated_fraction(
        self, *, atol: float
    ) -> torch.Tensor:
        """Return globally reduced cumulative updated_fraction per MLP channel."""
        local_updated, local_total = self.local_channel_statistics(atol=atol)
        reference_device = _local_tensor(self.snapshots[0].parameter.detach()).device
        collective_device = _collective_device(reference_device)
        statistics = torch.stack((local_updated, local_total)).to(collective_device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
        updated, total = statistics.cpu()
        if bool((total <= 0).any().item()):
            missing = int((total <= 0).sum().item())
            raise RuntimeError(
                f"updated_fraction counted no parameters for {missing} channels"
            )
        return (updated / total).to(torch.float32)

    @staticmethod
    def metrics_from_global_statistics(
        statistics: list[float], *, atol: float
    ) -> dict[str, float]:
        unchanged, total, absolute_delta_sum, squared_delta_sum = statistics
        if total <= 0.0:
            raise RuntimeError("parameter-update diagnostics counted no parameters")
        sparsity = unchanged / total
        return {
            "val-aux/parameter_update/sparsity_atol_1e-5": sparsity,
            "val-aux/parameter_update/updated_fraction_atol_1e-5": 1.0 - sparsity,
            "val-aux/parameter_update/mean_abs_delta_bfloat16": absolute_delta_sum / total,
            "val-aux/parameter_update/rms_delta_bfloat16": math.sqrt(squared_delta_sum / total),
            "val-aux/parameter_update/atol": float(atol),
            "val-aux/parameter_update/parameter_count_billions": total / 1.0e9,
        }
