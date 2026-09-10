"""Low-overhead gradient and parameter-update diagnostics for the recipe."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.distributed as dist

try:
    from torch.distributed.tensor import DTensor
except ImportError:  # pragma: no cover - older PyTorch fallback
    DTensor = ()


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
        cosine = max(-1.0, min(1.0, cosine))
        return {
            "mlp_rdrop/main_grad_rms_sampled": main_rms,
            "mlp_rdrop/aux_grad_rms_sampled": auxiliary_rms,
            "mlp_rdrop/aux_to_main_grad_ratio_sampled": ratio,
            "mlp_rdrop/main_aux_grad_cosine_sampled": cosine,
            "mlp_rdrop/main_aux_grad_angle_degrees_sampled": (
                math.degrees(math.acos(cosine)) if cosine_denom > 0.0 else 0.0
            ),
            "mlp_rdrop/grad_angle_defined": float(cosine_denom > 0.0),
            "mlp_rdrop/grad_ratio_defined": float(main_rms > 0.0),
            "mlp_rdrop/main_grad_l2_sampled": math.sqrt(main_sq),
            "mlp_rdrop/aux_grad_l2_sampled": math.sqrt(auxiliary_sq),
            "mlp_rdrop/gradient_sample_fraction": sample_count / total_numel,
            "mlp_rdrop/gradient_sample_count": sample_count,
        }


class ParameterUpdateTracker:
    """Validation-time BF16 difference against the model loaded before RL.

    Created before resume loads RL weights; the initialization model must stay
    the same on resume. This is cumulative change, not a last-step gradient mask.
    """

    def __init__(self, module: torch.nn.Module):
        self.snapshots = []
        with torch.no_grad():
            for name, parameter in module.named_parameters():
                local = _local_tensor(parameter.detach())
                if local.numel():
                    self.snapshots.append((name, parameter, local.to("cpu", torch.bfloat16).clone()))
        if not self.snapshots:
            raise RuntimeError("parameter diagnostics found no model parameters")

    @torch.no_grad()
    def local_statistics(self, *, atol: float):
        if atol < 0 or not math.isfinite(atol):
            raise ValueError("atol must be finite and non-negative")
        unchanged = total = 0
        absolute = squared = 0.0
        for name, parameter, initial in self.snapshots:
            current = _local_tensor(parameter.detach()).to("cpu", torch.bfloat16)
            if current.shape != initial.shape:
                raise RuntimeError(f"parameter shard shape changed: {name}")
            delta = current - initial
            unchanged += int(torch.isclose(delta, torch.zeros((), dtype=torch.bfloat16), atol=atol).sum())
            total += delta.numel()
            absolute += float(delta.float().abs().sum())
            squared += float(delta.double().square().sum())
        return [float(unchanged), float(total), absolute, squared]

    def distributed_metrics(self, *, atol: float = 1e-5):
        device = _local_tensor(self.snapshots[0][1]).device
        unchanged, total, absolute, squared = _all_reduce_sum(self.local_statistics(atol=atol), device)
        return {
            "val-aux/parameter_update/sparsity_atol_1e-5": unchanged / total,
            "val-aux/parameter_update/updated_fraction_atol_1e-5": 1 - unchanged / total,
            "val-aux/parameter_update/updated_parameter_count": total - unchanged,
            "val-aux/parameter_update/mean_abs_delta_bfloat16": absolute / total,
            "val-aux/parameter_update/rms_delta_bfloat16": math.sqrt(squared / total),
            "val-aux/parameter_update/atol": atol,
            "val-aux/parameter_update/parameter_count_billions": total / 1e9,
        }
