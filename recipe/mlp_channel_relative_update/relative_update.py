"""History-based MLP-channel update allocation.

The controller does not inspect activations and does not alter the GRPO loss.
For every dense SwiGLU channel it tracks an EMA of the squared parameter update
relative to the squared parameter scale.  Channels with a small historical
relative update receive a larger multiplier on the *AdamW-preconditioned*
update, while channels with a large history receive a smaller multiplier.

The multiplier projection preserves the total squared MLP update norm for the
current optimizer step.  Consequently, the component redistributes a fixed
MLP update budget instead of silently increasing the global learning rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class RelativeUpdateStep:
    """Globally reduced sufficient statistics for one optimizer step."""

    base_update_sq: torch.Tensor
    parameter_sq: torch.Tensor
    parameter_count: torch.Tensor
    base_relative_update_sq: torch.Tensor
    history_rms_before: torch.Tensor
    multipliers: torch.Tensor
    warmup: bool


def project_multipliers_to_fixed_update_energy(
    raw_multipliers: torch.Tensor,
    update_energy: torch.Tensor,
    *,
    min_multiplier: float,
    max_multiplier: float,
) -> torch.Tensor:
    """Bound multipliers while preserving ``sum(m**2 * update_energy)``.

    A common positive scale is applied to the raw multipliers and the result is
    clipped into the configured box.  Bisection chooses the common scale that
    makes scaled update energy equal the unscaled update energy.  This preserves
    multiplier ordering and is feasible because the interval contains one.
    """

    if raw_multipliers.shape != update_energy.shape or raw_multipliers.numel() == 0:
        raise ValueError("raw_multipliers and update_energy must be non-empty equal shapes")
    if not 0.0 < min_multiplier <= 1.0 <= max_multiplier:
        raise ValueError("multiplier bounds must satisfy 0 < min <= 1 <= max")
    if not bool(torch.isfinite(raw_multipliers).all().item()) or bool(
        (raw_multipliers <= 0).any().item()
    ):
        return torch.ones_like(raw_multipliers, dtype=torch.float32)
    if not bool(torch.isfinite(update_energy).all().item()) or bool(
        (update_energy < 0).any().item()
    ):
        return torch.ones_like(raw_multipliers, dtype=torch.float32)

    raw = raw_multipliers.to(dtype=torch.float64)
    energy = update_energy.to(dtype=torch.float64)
    target = energy.sum()
    if float(target.item()) <= 0.0:
        return torch.ones_like(raw_multipliers, dtype=torch.float32)

    lower = torch.zeros((), device=raw.device, dtype=raw.dtype)
    upper = torch.as_tensor(
        2.0 * max_multiplier / float(raw.min().item()),
        device=raw.device,
        dtype=raw.dtype,
    )
    for _ in range(80):
        scale = (lower + upper) / 2.0
        candidate = torch.clamp(
            raw * scale,
            min=min_multiplier,
            max=max_multiplier,
        )
        candidate_energy = (candidate.square() * energy).sum()
        if float(candidate_energy.item()) < float(target.item()):
            lower = scale
        else:
            upper = scale

    result = torch.clamp(
        raw * ((lower + upper) / 2.0),
        min=min_multiplier,
        max=max_multiplier,
    ).to(dtype=torch.float32)
    relative_error = abs(
        float(((result.to(torch.float64).square() * energy).sum() / target).item()) - 1.0
    )
    if relative_error > 1e-6:
        raise RuntimeError(
            "relative-update multiplier projection failed to preserve update energy: "
            f"relative_error={relative_error}"
        )
    return result


class MLPChannelRelativeUpdateController:
    """Track and redistribute per-channel relative AdamW update energy."""

    def __init__(
        self,
        *,
        num_layers: int,
        intermediate_size: int,
        selected_layers: Iterable[int] | None = None,
        history_ema_beta: float = 0.99,
        history_power: float = 0.5,
        history_floor_ratio: float = 0.1,
        multiplier_ratio_cap: float = 10.0,
        warmup_steps: int = 16,
        parameter_rms_epsilon: float = 1e-12,
        history_epsilon: float = 1e-12,
    ) -> None:
        if num_layers <= 0 or intermediate_size <= 0:
            raise ValueError("num_layers and intermediate_size must be positive")
        if not 0.0 <= history_ema_beta < 1.0:
            raise ValueError("history_ema_beta must be in [0, 1)")
        if not math.isfinite(history_power) or history_power <= 0.0:
            raise ValueError("history_power must be finite and positive")
        if not math.isfinite(history_floor_ratio) or history_floor_ratio < 0.0:
            raise ValueError("history_floor_ratio must be finite and non-negative")
        if not math.isfinite(multiplier_ratio_cap) or multiplier_ratio_cap < 1.0:
            raise ValueError("multiplier_ratio_cap must be finite and at least one")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if parameter_rms_epsilon <= 0.0 or history_epsilon <= 0.0:
            raise ValueError("epsilon values must be positive")

        layers = (
            list(range(num_layers))
            if selected_layers is None
            else [int(layer) for layer in selected_layers]
        )
        if not layers or len(layers) != len(set(layers)):
            raise ValueError("selected_layers must be non-empty and contain no duplicates")
        if min(layers) < 0 or max(layers) >= num_layers:
            raise ValueError(f"selected_layers={layers} is outside [0, {num_layers})")

        self.num_layers = int(num_layers)
        self.intermediate_size = int(intermediate_size)
        self.selected_layers = tuple(sorted(layers))
        self.layer_to_slot = {layer: slot for slot, layer in enumerate(self.selected_layers)}
        self.history_ema_beta = float(history_ema_beta)
        self.history_power = float(history_power)
        self.history_floor_ratio = float(history_floor_ratio)
        self.multiplier_ratio_cap = float(multiplier_ratio_cap)
        self.min_multiplier = float(1.0 / math.sqrt(multiplier_ratio_cap))
        self.max_multiplier = float(math.sqrt(multiplier_ratio_cap))
        self.warmup_steps = int(warmup_steps)
        self.parameter_rms_epsilon = float(parameter_rms_epsilon)
        self.history_epsilon = float(history_epsilon)

        shape = (len(self.selected_layers), self.intermediate_size)
        self.history_relative_update_sq = torch.zeros(shape, dtype=torch.float32)
        self.step_count = 0
        self.last_metrics: dict[str, float] = {}

    def prepare_step(
        self,
        *,
        local_base_update_sq: torch.Tensor,
        local_parameter_sq: torch.Tensor,
        local_parameter_count: torch.Tensor,
    ) -> RelativeUpdateStep:
        """Reduce current statistics and compute multipliers from prior history."""

        expected = (len(self.selected_layers), self.intermediate_size)
        for name, value in (
            ("local_base_update_sq", local_base_update_sq),
            ("local_parameter_sq", local_parameter_sq),
            ("local_parameter_count", local_parameter_count),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} shape {tuple(value.shape)} != {expected}")

        base_update_sq = local_base_update_sq.detach().to(dtype=torch.float32).clone()
        parameter_sq = local_parameter_sq.detach().to(dtype=torch.float32).clone()
        parameter_count = local_parameter_count.detach().to(dtype=torch.float32).clone()
        self._all_reduce(base_update_sq)
        self._all_reduce(parameter_sq)
        self._all_reduce(parameter_count)
        if bool((parameter_count <= 0).any().item()):
            missing = int((parameter_count <= 0).sum().item())
            raise RuntimeError(f"relative-update statistics missed {missing} selected channels")

        self._move_state(base_update_sq.device)
        denominator = parameter_sq + parameter_count * (self.parameter_rms_epsilon**2)
        base_relative_update_sq = base_update_sq / denominator.clamp_min(
            self.history_epsilon**2
        )
        history_rms_before = self._history_rms()
        warmup = self.step_count < self.warmup_steps

        if warmup or float(base_update_sq.sum().item()) <= 0.0:
            multipliers = torch.ones_like(base_update_sq)
        else:
            reference = history_rms_before.median(dim=1, keepdim=True).values
            floor = torch.clamp(
                reference * self.history_floor_ratio,
                min=self.history_epsilon,
            )
            raw_multipliers = torch.pow(
                (reference + floor) / (history_rms_before + floor),
                self.history_power,
            )
            multipliers = project_multipliers_to_fixed_update_energy(
                raw_multipliers,
                base_update_sq,
                min_multiplier=self.min_multiplier,
                max_multiplier=self.max_multiplier,
            )

        return RelativeUpdateStep(
            base_update_sq=base_update_sq,
            parameter_sq=parameter_sq,
            parameter_count=parameter_count,
            base_relative_update_sq=base_relative_update_sq,
            history_rms_before=history_rms_before,
            multipliers=multipliers,
            warmup=warmup,
        )

    def commit_step(self, step: RelativeUpdateStep) -> dict[str, float]:
        """Record the actual scaled update after the optimizer applied it."""

        actual_update_sq = step.base_update_sq * step.multipliers.square()
        denominator = step.parameter_sq + step.parameter_count * (
            self.parameter_rms_epsilon**2
        )
        actual_relative_update_sq = actual_update_sq / denominator.clamp_min(
            self.history_epsilon**2
        )
        self.history_relative_update_sq.mul_(self.history_ema_beta).add_(
            actual_relative_update_sq,
            alpha=1.0 - self.history_ema_beta,
        )
        self.step_count += 1

        history_after = self._history_rms()
        base_relative = step.base_relative_update_sq.clamp_min(0.0).sqrt()
        actual_relative = actual_relative_update_sq.clamp_min(0.0).sqrt()
        multipliers = step.multipliers
        base_energy = step.base_update_sq.sum()
        actual_energy = actual_update_sq.sum()
        energy_ratio = actual_energy / base_energy.clamp_min(self.history_epsilon)

        layer_history_median = step.history_rms_before.median(dim=1, keepdim=True).values
        low_history_mask = step.history_rms_before < layer_history_median
        low_history_base_share = step.base_update_sq[low_history_mask].sum() / base_energy.clamp_min(
            self.history_epsilon
        )
        low_history_actual_share = actual_update_sq[low_history_mask].sum() / actual_energy.clamp_min(
            self.history_epsilon
        )
        base_effective_channels = self._effective_channel_count(step.base_update_sq)
        actual_effective_channels = self._effective_channel_count(actual_update_sq)

        flat_history = step.history_rms_before.flatten()
        flat_multiplier = multipliers.flatten()
        min_history_idx = int(flat_history.argmin().item())
        max_history_idx = int(flat_history.argmax().item())
        low_to_high_ratio = flat_multiplier[min_history_idx] / flat_multiplier[
            max_history_idx
        ].clamp_min(self.history_epsilon)
        correlation = self._log_history_multiplier_correlation(
            flat_history,
            flat_multiplier,
        )

        tolerance = 1e-6
        metrics = {
            "mlp_relative_update/step": float(self.step_count),
            "mlp_relative_update/warmup": float(step.warmup),
            "mlp_relative_update/history_ema_beta": self.history_ema_beta,
            "mlp_relative_update/history_power": self.history_power,
            "mlp_relative_update/multiplier_ratio_cap": self.multiplier_ratio_cap,
            "mlp_relative_update/selected_layers": float(len(self.selected_layers)),
            "mlp_relative_update/channel_count": float(multipliers.numel()),
            "mlp_relative_update/history_rms_before_mean": float(
                step.history_rms_before.mean().item()
            ),
            "mlp_relative_update/history_rms_before_min": float(
                step.history_rms_before.min().item()
            ),
            "mlp_relative_update/history_rms_before_max": float(
                step.history_rms_before.max().item()
            ),
            "mlp_relative_update/history_rms_after_mean": float(history_after.mean().item()),
            "mlp_relative_update/history_rms_after_min": float(history_after.min().item()),
            "mlp_relative_update/history_rms_after_max": float(history_after.max().item()),
            "mlp_relative_update/base_relative_rms_mean": float(base_relative.mean().item()),
            "mlp_relative_update/base_relative_rms_min": float(base_relative.min().item()),
            "mlp_relative_update/base_relative_rms_max": float(base_relative.max().item()),
            "mlp_relative_update/actual_relative_rms_mean": float(actual_relative.mean().item()),
            "mlp_relative_update/actual_relative_rms_min": float(actual_relative.min().item()),
            "mlp_relative_update/actual_relative_rms_max": float(actual_relative.max().item()),
            "mlp_relative_update/multiplier_mean": float(multipliers.mean().item()),
            "mlp_relative_update/multiplier_std": float(
                multipliers.to(torch.float64).std(unbiased=False).item()
            ),
            "mlp_relative_update/multiplier_min": float(multipliers.min().item()),
            "mlp_relative_update/multiplier_max": float(multipliers.max().item()),
            "mlp_relative_update/multiplier_max_to_min": float(
                (multipliers.max() / multipliers.min()).item()
            ),
            "mlp_relative_update/low_history_to_high_history_multiplier_ratio": float(
                low_to_high_ratio.item()
            ),
            "mlp_relative_update/boosted_fraction": float(
                (multipliers > 1.0 + tolerance).to(torch.float32).mean().item()
            ),
            "mlp_relative_update/damped_fraction": float(
                (multipliers < 1.0 - tolerance).to(torch.float32).mean().item()
            ),
            "mlp_relative_update/min_saturation_fraction": float(
                (multipliers <= self.min_multiplier + tolerance)
                .to(torch.float32)
                .mean()
                .item()
            ),
            "mlp_relative_update/max_saturation_fraction": float(
                (multipliers >= self.max_multiplier - tolerance)
                .to(torch.float32)
                .mean()
                .item()
            ),
            "mlp_relative_update/base_update_energy": float(base_energy.item()),
            "mlp_relative_update/actual_update_energy": float(actual_energy.item()),
            "mlp_relative_update/update_energy_ratio": float(energy_ratio.item()),
            "mlp_relative_update/update_energy_error": abs(float(energy_ratio.item()) - 1.0),
            "mlp_relative_update/low_history_base_energy_share": float(
                low_history_base_share.item()
            ),
            "mlp_relative_update/low_history_actual_energy_share": float(
                low_history_actual_share.item()
            ),
            "mlp_relative_update/energy_share_shift_to_low_history": float(
                (low_history_actual_share - low_history_base_share).item()
            ),
            "mlp_relative_update/base_effective_channel_fraction": float(
                (base_effective_channels / multipliers.numel()).item()
            ),
            "mlp_relative_update/actual_effective_channel_fraction": float(
                (actual_effective_channels / multipliers.numel()).item()
            ),
            "mlp_relative_update/log_history_multiplier_correlation": correlation,
        }
        self.last_metrics = metrics
        return metrics

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "num_layers": self.num_layers,
            "intermediate_size": self.intermediate_size,
            "selected_layers": self.selected_layers,
            "history_ema_beta": self.history_ema_beta,
            "history_power": self.history_power,
            "history_floor_ratio": self.history_floor_ratio,
            "multiplier_ratio_cap": self.multiplier_ratio_cap,
            "warmup_steps": self.warmup_steps,
            "parameter_rms_epsilon": self.parameter_rms_epsilon,
            "history_epsilon": self.history_epsilon,
            "step_count": self.step_count,
            "history_relative_update_sq": self.history_relative_update_sq.detach().cpu(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {
            "version": 1,
            "num_layers": self.num_layers,
            "intermediate_size": self.intermediate_size,
            "selected_layers": self.selected_layers,
            "history_ema_beta": self.history_ema_beta,
            "history_power": self.history_power,
            "history_floor_ratio": self.history_floor_ratio,
            "multiplier_ratio_cap": self.multiplier_ratio_cap,
            "warmup_steps": self.warmup_steps,
            "parameter_rms_epsilon": self.parameter_rms_epsilon,
            "history_epsilon": self.history_epsilon,
        }
        for key, expected_value in expected.items():
            actual = state.get(key)
            if isinstance(expected_value, tuple):
                actual = tuple(actual) if actual is not None else None
            if actual != expected_value:
                raise ValueError(
                    f"relative-update checkpoint {key}={actual!r} does not match "
                    f"current config {expected_value!r}"
                )
        history = torch.as_tensor(state["history_relative_update_sq"], dtype=torch.float32)
        if history.shape != self.history_relative_update_sq.shape:
            raise ValueError(
                "relative-update checkpoint history shape "
                f"{tuple(history.shape)} != {tuple(self.history_relative_update_sq.shape)}"
            )
        self.history_relative_update_sq = history.clone()
        self.step_count = int(state["step_count"])
        self.last_metrics = {}

    def _history_rms(self) -> torch.Tensor:
        if self.step_count <= 0:
            return torch.zeros_like(self.history_relative_update_sq)
        bias_correction = 1.0 - self.history_ema_beta**self.step_count
        return (
            self.history_relative_update_sq / max(bias_correction, self.history_epsilon)
        ).clamp_min(0.0).sqrt()

    def _move_state(self, device: torch.device) -> None:
        if self.history_relative_update_sq.device != device:
            self.history_relative_update_sq = self.history_relative_update_sq.to(device=device)

    @staticmethod
    def _all_reduce(value: torch.Tensor) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)

    def _log_history_multiplier_correlation(
        self,
        history: torch.Tensor,
        multipliers: torch.Tensor,
    ) -> float:
        if history.numel() < 2:
            return 0.0
        x = torch.log(history.to(torch.float64) + self.history_epsilon)
        y = torch.log(multipliers.to(torch.float64).clamp_min(self.history_epsilon))
        x = x - x.mean()
        y = y - y.mean()
        denominator = x.square().sum().sqrt() * y.square().sum().sqrt()
        if float(denominator.item()) <= self.history_epsilon:
            return 0.0
        return float(((x * y).sum() / denominator).item())

    def _effective_channel_count(self, update_energy: torch.Tensor) -> torch.Tensor:
        energy = update_energy.to(torch.float64).clamp_min(0.0)
        total = energy.sum()
        if float(total.item()) <= self.history_epsilon:
            return torch.zeros((), device=energy.device, dtype=energy.dtype)
        return total.square() / energy.square().sum().clamp_min(self.history_epsilon)
