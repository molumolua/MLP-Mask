"""Score-aware hard MLP-channel masks used only by an actor-side KL branch."""

from __future__ import annotations

import re
from types import MethodType
from typing import Any

import torch
import torch.distributed as dist


RANDOM_SELECTION = "random"
SOFT_TOP_SELECTION = "soft_top"
HARD_TOP_SELECTION = "hard_top"
_VALID_SELECTION_STRATEGIES = {
    RANDOM_SELECTION,
    SOFT_TOP_SELECTION,
    HARD_TOP_SELECTION,
}

NO_SCORE = "none"
RELATIVE_ACTIVATION_SCORE = "relative_activation"
OUTPUT_CONTRIBUTION_SCORE = "output_contribution"
GRADIENT_ACTIVATION_SCORE = "gradient_activation"
UPDATED_FRACTION_SCORE = "updated_fraction"
_VALID_SCORE_METHODS = {
    NO_SCORE,
    RELATIVE_ACTIVATION_SCORE,
    OUTPUT_CONTRIBUTION_SCORE,
    GRADIENT_ACTIVATION_SCORE,
    UPDATED_FRACTION_SCORE,
}
_FORWARD_SCORE_METHODS = {
    RELATIVE_ACTIVATION_SCORE,
    OUTPUT_CONTRIBUTION_SCORE,
    GRADIENT_ACTIVATION_SCORE,
}

_LAYER_RE = re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)(?:\.|$)")
_CHECKPOINT_FORMAT = "mlp_channel_consistency_v2"
_LEGACY_CHECKPOINT_FORMAT = "mlp_channel_consistency_v1"


def _shape_numel(shape: torch.Size) -> int:
    result = 1
    for value in shape:
        result *= int(value)
    return result


class MLPChannelConsistencyController:
    """Hold one exact per-layer hard mask for a complete optimizer step.

    Score-based masks consume statistics from the previous clean GRPO update.
    Thus the score collection piggybacks on an existing backward and never adds a
    model forward, while the selected mask remains fixed throughout the next
    optimizer step's auxiliary KL branch.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        intermediate_size: int,
        mask_ratio: float = 0.10,
        selection_strategy: str = RANDOM_SELECTION,
        score_method: str = NO_SCORE,
        score_ema_beta: float = 0.0,
        activation_ema_beta: float = 0.95,
        relative_activation_epsilon: float = 1.0e-6,
        weighted_max_ratio: float = 4.0,
        weighted_rank_power: float = 2.0,
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
        if selection_strategy not in _VALID_SELECTION_STRATEGIES:
            raise ValueError(
                "selection_strategy must be one of "
                f"{sorted(_VALID_SELECTION_STRATEGIES)}, got {selection_strategy!r}"
            )
        if score_method not in _VALID_SCORE_METHODS:
            raise ValueError(
                f"score_method must be one of {sorted(_VALID_SCORE_METHODS)}, "
                f"got {score_method!r}"
            )
        if selection_strategy == RANDOM_SELECTION and score_method != NO_SCORE:
            raise ValueError("selection_strategy=random requires score_method=none")
        if selection_strategy != RANDOM_SELECTION and score_method == NO_SCORE:
            raise ValueError(
                "score-based selection requires relative_activation, "
                "output_contribution, gradient_activation, or updated_fraction"
            )
        if not 0.0 <= score_ema_beta < 1.0:
            raise ValueError("score_ema_beta must be in [0, 1)")
        if not 0.0 <= activation_ema_beta < 1.0:
            raise ValueError("activation_ema_beta must be in [0, 1)")
        if relative_activation_epsilon <= 0.0:
            raise ValueError("relative_activation_epsilon must be positive")
        if weighted_max_ratio < 1.0:
            raise ValueError("weighted_max_ratio must be >= 1")
        if weighted_rank_power <= 0.0:
            raise ValueError("weighted_rank_power must be positive")
        if random_seed < 0:
            raise ValueError("random_seed must be non-negative")

        self.num_layers = int(num_layers)
        self.intermediate_size = int(intermediate_size)
        self.mask_ratio = float(mask_ratio)
        self.masked_per_layer = masked_per_layer
        self.selection_strategy = str(selection_strategy)
        self.score_method = str(score_method)
        self.score_ema_beta = float(score_ema_beta)
        self.activation_ema_beta = float(activation_ema_beta)
        self.relative_activation_epsilon = float(relative_activation_epsilon)
        self.weighted_max_ratio = float(weighted_max_ratio)
        self.weighted_rank_power = float(weighted_rank_power)
        self.random_seed = int(random_seed)

        shape = (self.num_layers, self.intermediate_size)
        self.keep_mask = torch.ones(shape, dtype=torch.bool)
        self.selection_score = torch.zeros(shape, dtype=torch.float32)
        self.selection_score_initialized = False
        self.activation_ema = torch.zeros(shape, dtype=torch.float32)
        self.activation_ema_initialized = False
        self.relative_activation_score = torch.zeros(shape, dtype=torch.float32)
        self.mask_version = 0
        self.route = "clean"
        self.collect_score = False

        self._response_token_mask: torch.Tensor | None = None
        self._response_sample_ids: torch.Tensor | None = None
        self._response_sample_count = 0
        self._score_level_sum: dict[int, torch.Tensor] = {}
        self._score_sample_count: torch.Tensor | None = None
        self._score_token_count: torch.Tensor | None = None
        self._device_masks: dict[tuple[int, str, int, torch.dtype], torch.Tensor] = {}
        self._last_selected_rank_mean = 0.0
        self._last_selection_used_score = False

    @property
    def needs_score(self) -> bool:
        return self.score_method != NO_SCORE

    @property
    def needs_forward_score(self) -> bool:
        return self.score_method in _FORWARD_SCORE_METHODS

    @property
    def needs_updated_fraction_score(self) -> bool:
        return self.score_method == UPDATED_FRACTION_SCORE

    def set_clean(self) -> None:
        self.route = "clean"

    def set_masked(self) -> None:
        if self.mask_version <= 0:
            raise RuntimeError("masked consistency route requested before the first mask sample")
        self.route = "masked"

    def start_score_collection(self) -> None:
        """Collect one clean optimizer update for the following step's mask."""
        self._clear_pending_score()
        self.collect_score = self.needs_forward_score

    def cancel_score_collection(self) -> None:
        self.collect_score = False
        self._clear_pending_score()

    def set_response_token_mask(
        self,
        response_token_mask: torch.Tensor,
        *,
        sample_ids: torch.Tensor | None = None,
        sample_count: int | None = None,
    ) -> None:
        """Install the response-logit layout used by clean activation hooks."""
        self._response_token_mask = response_token_mask.detach()
        if sample_ids is None:
            if self._response_token_mask.ndim == 0:
                raise ValueError("response_token_mask must have at least one dimension")
            inferred_samples = int(self._response_token_mask.shape[0])
            view_shape = (inferred_samples,) + (1,) * (
                self._response_token_mask.ndim - 1
            )
            sample_ids = torch.arange(
                inferred_samples,
                device=self._response_token_mask.device,
                dtype=torch.long,
            ).view(view_shape).expand_as(self._response_token_mask)
            sample_count = inferred_samples
        elif sample_ids.shape != self._response_token_mask.shape:
            raise ValueError("sample_ids must have the same shape as response_token_mask")
        if sample_count is None or sample_count <= 0:
            raise ValueError("sample_count must be positive")
        self._response_sample_ids = sample_ids.detach().to(dtype=torch.long)
        self._response_sample_count = int(sample_count)

        if self.collect_score and self.route == "clean":
            device = self._response_token_mask.device
            if self._score_sample_count is None:
                self._score_sample_count = torch.zeros(
                    (), device=device, dtype=torch.float32
                )
                self._score_token_count = torch.zeros(
                    (), device=device, dtype=torch.float32
                )
            self._score_sample_count.add_(float(self._response_sample_count))
            self._score_token_count.add_(
                self._response_token_mask.to(dtype=torch.float32).sum()
            )

    def end_batch(self) -> None:
        self._response_token_mask = None
        self._response_sample_ids = None
        self._response_sample_count = 0

    def resample(self) -> dict[str, float]:
        """Select an exact per-layer quota, using the previous clean update's score."""
        next_version = self.mask_version + 1
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.random_seed + next_version)
        keep = torch.ones_like(self.keep_mask)
        selected_ranks: list[torch.Tensor] = []
        used_score = self.needs_score and self.selection_score_initialized

        for layer_idx in range(self.num_layers):
            if self.selection_strategy == RANDOM_SELECTION or not used_score:
                selected = torch.randperm(
                    self.intermediate_size, generator=generator
                )[: self.masked_per_layer]
            else:
                score = self.selection_score[layer_idx]
                if self.selection_strategy == HARD_TOP_SELECTION:
                    selected = torch.topk(
                        score,
                        k=self.masked_per_layer,
                        largest=True,
                        sorted=False,
                    ).indices
                else:
                    rank = self._percentile_rank(score)
                    weights = 1.0 + (
                        self.weighted_max_ratio - 1.0
                    ) * rank.pow(self.weighted_rank_power)
                    selected = torch.multinomial(
                        weights,
                        num_samples=self.masked_per_layer,
                        replacement=False,
                        generator=generator,
                    )
                    selected_ranks.append(rank[selected])
            keep[layer_idx, selected] = False

        self.keep_mask.copy_(keep)
        self.mask_version = next_version
        self._device_masks.clear()
        self._last_selection_used_score = bool(used_score)
        self._last_selected_rank_mean = (
            float(torch.cat(selected_ranks).mean().item()) if selected_ranks else 0.0
        )
        self.set_clean()
        return self.metrics()

    @staticmethod
    def _percentile_rank(score: torch.Tensor) -> torch.Tensor:
        """Return average percentile ranks, assigning equal weights to ties."""
        score = score.detach().to(device="cpu", dtype=torch.float32).flatten()
        order = torch.argsort(score, stable=True)
        sorted_score = score[order]
        _, inverse, counts = torch.unique_consecutive(
            sorted_score,
            return_inverse=True,
            return_counts=True,
        )
        positions = torch.arange(score.numel(), dtype=torch.float32)
        rank_sums = torch.zeros(counts.numel(), dtype=torch.float32)
        rank_sums.scatter_add_(0, inverse, positions)
        average_rank = rank_sums / counts.to(dtype=torch.float32)
        rank = torch.empty_like(score)
        rank[order] = average_rank[inverse]
        return rank / float(max(score.numel() - 1, 1))

    def apply(
        self,
        layer_idx: int,
        activation: torch.Tensor,
        *,
        down_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(f"layer {layer_idx} outside [0, {self.num_layers})")
        if activation.shape[-1] != self.intermediate_size:
            raise RuntimeError(
                f"MLP layer {layer_idx} activation width {activation.shape[-1]} != "
                f"intermediate_size {self.intermediate_size}"
            )

        if self.collect_score and self.route == "clean" and activation.requires_grad:
            if self._response_token_mask is None or self._response_sample_ids is None:
                raise RuntimeError(
                    "clean score collection requested before response metadata was installed"
                )
            down_norm = None
            if self.score_method == OUTPUT_CONTRIBUTION_SCORE:
                down_norm = self._down_projection_column_norm(down_weight, activation)
            activation.register_hook(
                lambda grad, a=activation.detach(), m=self._response_token_mask,
                ids=self._response_sample_ids, count=self._response_sample_count,
                layer=layer_idx, dn=down_norm: self._accumulate_score(
                    layer, a, grad, m, ids, count, dn
                )
            )

        if self.route == "clean":
            return activation
        device = activation.device
        key = (layer_idx, device.type, device.index or 0, activation.dtype)
        mask = self._device_masks.get(key)
        if mask is None:
            mask = self.keep_mask[layer_idx].to(
                device=device, dtype=activation.dtype
            )
            self._device_masks[key] = mask
        return activation * mask

    @staticmethod
    def _down_projection_column_norm(
        down_weight: torch.Tensor | None,
        activation: torch.Tensor,
    ) -> torch.Tensor:
        if down_weight is None or down_weight.ndim != 2:
            raise RuntimeError(
                "output_contribution scoring requires the 2-D down_proj weight"
            )
        weight_for_norm = down_weight.detach()
        placements = getattr(weight_for_norm, "placements", ())
        if placements:
            if any(type(placement).__name__ != "Replicate" for placement in placements):
                raise RuntimeError(
                    "output_contribution requires down_proj to be unsharded during forward"
                )
            weight_for_norm = weight_for_norm.to_local()
        down_norm = weight_for_norm.to(dtype=torch.float32).square().sum(dim=0).sqrt()
        if down_norm.numel() != activation.shape[-1]:
            raise RuntimeError(
                "down_proj input width does not match the observed MLP activation"
            )
        return down_norm

    def _accumulate_score(
        self,
        layer_idx: int,
        activation: torch.Tensor,
        grad: torch.Tensor,
        token_mask: torch.Tensor,
        sample_ids: torch.Tensor,
        sample_count: int,
        down_weight_norm: torch.Tensor | None,
    ) -> None:
        with torch.no_grad():
            if token_mask.numel() != _shape_numel(activation.shape[:-1]):
                raise RuntimeError(
                    f"response mask does not match layer {layer_idx} activation layout"
                )
            flat_mask = token_mask.reshape(-1).to(
                device=activation.device, dtype=torch.bool
            )
            flat_ids = sample_ids.reshape(-1).to(
                device=activation.device, dtype=torch.long
            )
            valid_activation = activation.reshape(-1, activation.shape[-1])[
                flat_mask
            ].to(dtype=torch.float32)
            valid_ids = flat_ids[flat_mask]
            if valid_activation.numel() == 0:
                raise RuntimeError("clean score batch contains no valid response tokens")
            token_count = torch.bincount(
                valid_ids, minlength=sample_count
            ).to(device=activation.device, dtype=torch.float32)
            if bool((token_count == 0).any().item()):
                raise RuntimeError("clean score batch contains a sample without response tokens")

            per_sample_sum = torch.zeros(
                (sample_count, self.intermediate_size),
                device=activation.device,
                dtype=torch.float32,
            )
            if self.score_method == GRADIENT_ACTIVATION_SCORE:
                valid_grad = grad.detach().reshape(
                    -1, grad.shape[-1]
                )[flat_mask].to(dtype=torch.float32)
                per_sample_sum.index_add_(
                    0, valid_ids, (valid_activation * valid_grad).abs()
                )
                level_sum = (
                    per_sample_sum / token_count.unsqueeze(-1).clamp_min(1.0)
                ).sum(dim=0)
            else:
                per_sample_sum.index_add_(0, valid_ids, valid_activation.square())
                level_sum = torch.sqrt(
                    per_sample_sum / token_count.unsqueeze(-1).clamp_min(1.0)
                ).sum(dim=0)
                if self.score_method == OUTPUT_CONTRIBUTION_SCORE:
                    if down_weight_norm is None:
                        raise RuntimeError("output contribution is missing down_proj norms")
                    level_sum.mul_(down_weight_norm.to(device=level_sum.device))

            accumulator = self._score_level_sum.get(layer_idx)
            if accumulator is None or accumulator.device != level_sum.device:
                accumulator = torch.zeros_like(level_sum)
                self._score_level_sum[layer_idx] = accumulator
            accumulator.add_(level_sum)

    def finish_score_collection(self) -> dict[str, float]:
        """All-reduce the clean score and update the CPU EMA for the next step."""
        self.collect_score = False
        self.end_batch()
        if not self.needs_forward_score:
            return self._score_metrics(updated=False, samples=0.0, tokens=0.0)
        if self._score_sample_count is None or self._score_token_count is None:
            raise RuntimeError("score-based consistency collected no clean actor batch")

        device = self._score_sample_count.device
        score_sum = torch.stack(
            [
                self._score_level_sum.get(
                    layer_idx,
                    torch.zeros(
                        self.intermediate_size, device=device, dtype=torch.float32
                    ),
                ).to(device)
                for layer_idx in range(self.num_layers)
            ]
        )
        observed = torch.tensor(
            [float(layer_idx in self._score_level_sum) for layer_idx in range(self.num_layers)],
            device=device,
            dtype=torch.float32,
        )
        sample_count = self._score_sample_count
        token_count = self._score_token_count
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(score_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(observed, op=dist.ReduceOp.SUM)
            dist.all_reduce(sample_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        if bool((observed == 0).any().item()):
            missing = torch.nonzero(observed == 0).flatten().cpu().tolist()
            raise RuntimeError(f"no clean score hook fired for layers {missing}")
        samples = float(sample_count.item())
        tokens = float(token_count.item())
        if samples <= 0.0 or tokens <= 0.0:
            raise RuntimeError("score-based consistency observed no response samples/tokens")
        current = (score_sum / sample_count.clamp_min(1.0)).cpu()

        initialized_before = self.selection_score_initialized
        if self.score_method == RELATIVE_ACTIVATION_SCORE:
            if self.activation_ema_initialized:
                self.relative_activation_score.copy_(
                    (current - self.activation_ema)
                    / self.activation_ema.clamp_min(self.relative_activation_epsilon)
                )
                self._update_selection_score(self.relative_activation_score)
            if self.activation_ema_initialized:
                self.activation_ema.mul_(self.activation_ema_beta).add_(
                    current, alpha=1.0 - self.activation_ema_beta
                )
            else:
                self.activation_ema.copy_(current)
                self.activation_ema_initialized = True
        else:
            self._update_selection_score(current)

        metrics = self._score_metrics(updated=True, samples=samples, tokens=tokens)
        metrics.update(
            {
                "mlp_consistency/score_current_mean": float(current.mean().item()),
                "mlp_consistency/score_current_min": float(current.min().item()),
                "mlp_consistency/score_current_max": float(current.max().item()),
                "mlp_consistency/score_initialized_before_update": float(
                    initialized_before
                ),
            }
        )
        self._clear_pending_score()
        return metrics

    def update_updated_fraction_score(
        self, updated_fraction: torch.Tensor, *, atol: float = 1.0e-5
    ) -> dict[str, float]:
        """Use a cumulative pre-RL BF16 update fraction as the next mask score."""
        if not self.needs_updated_fraction_score:
            raise RuntimeError(
                "updated_fraction observations require score_method=updated_fraction"
            )
        current = updated_fraction.detach().to(device="cpu", dtype=torch.float32)
        if current.shape != self.selection_score.shape:
            raise ValueError(
                f"updated_fraction shape {tuple(current.shape)} != "
                f"{tuple(self.selection_score.shape)}"
            )
        if not bool(torch.isfinite(current).all().item()) or bool(
            ((current < 0.0) | (current > 1.0)).any().item()
        ):
            raise ValueError("updated_fraction values must be finite and in [0, 1]")
        if atol < 0.0:
            raise ValueError("updated_fraction atol must be non-negative")
        initialized_before = self.selection_score_initialized
        self._update_selection_score(current)
        metrics = self._score_metrics(updated=True, samples=0.0, tokens=0.0)
        metrics.update(
            {
                "mlp_consistency/score_current_mean": float(current.mean().item()),
                "mlp_consistency/score_current_min": float(current.min().item()),
                "mlp_consistency/score_current_max": float(current.max().item()),
                "mlp_consistency/score_initialized_before_update": float(
                    initialized_before
                ),
                "mlp_consistency/updated_fraction_atol": float(atol),
            }
        )
        return metrics

    def _update_selection_score(self, current: torch.Tensor) -> None:
        current = current.detach().to(device="cpu", dtype=torch.float32)
        if self.selection_score_initialized:
            self.selection_score.mul_(self.score_ema_beta).add_(
                current, alpha=1.0 - self.score_ema_beta
            )
        else:
            self.selection_score.copy_(current)
            self.selection_score_initialized = True

    def _clear_pending_score(self) -> None:
        self.end_batch()
        self._score_level_sum.clear()
        self._score_sample_count = None
        self._score_token_count = None

    def _score_metrics(
        self, *, updated: bool, samples: float, tokens: float
    ) -> dict[str, float]:
        return {
            "mlp_consistency/score_updated": float(updated),
            "mlp_consistency/score_response_samples": float(samples),
            "mlp_consistency/score_response_tokens": float(tokens),
            "mlp_consistency/score_initialized": float(
                self.selection_score_initialized
            ),
            "mlp_consistency/score_ema_beta": self.score_ema_beta,
            "mlp_consistency/activation_ema_beta": self.activation_ema_beta,
            "mlp_consistency/score_is_relative_activation": float(
                self.score_method == RELATIVE_ACTIVATION_SCORE
            ),
            "mlp_consistency/score_is_output_contribution": float(
                self.score_method == OUTPUT_CONTRIBUTION_SCORE
            ),
            "mlp_consistency/score_is_gradient_activation": float(
                self.score_method == GRADIENT_ACTIVATION_SCORE
            ),
            "mlp_consistency/score_is_updated_fraction": float(
                self.score_method == UPDATED_FRACTION_SCORE
            ),
        }

    def metrics(self) -> dict[str, float]:
        masked_per_layer = (~self.keep_mask).sum(dim=-1).to(dtype=torch.float32)
        result = {
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
            "mlp_consistency/selection_is_random": float(
                self.selection_strategy == RANDOM_SELECTION
            ),
            "mlp_consistency/selection_is_soft_top": float(
                self.selection_strategy == SOFT_TOP_SELECTION
            ),
            "mlp_consistency/selection_is_hard_top": float(
                self.selection_strategy == HARD_TOP_SELECTION
            ),
            "mlp_consistency/selection_used_score": float(
                self._last_selection_used_score
            ),
            "mlp_consistency/selected_rank_mean": self._last_selected_rank_mean,
            "mlp_consistency/weighted_max_ratio": self.weighted_max_ratio,
            "mlp_consistency/weighted_rank_power": self.weighted_rank_power,
        }
        result.update(self._score_metrics(updated=False, samples=0.0, tokens=0.0))
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_format": _CHECKPOINT_FORMAT,
            "num_layers": self.num_layers,
            "intermediate_size": self.intermediate_size,
            "mask_ratio": self.mask_ratio,
            "selection_strategy": self.selection_strategy,
            "score_method": self.score_method,
            "score_ema_beta": self.score_ema_beta,
            "activation_ema_beta": self.activation_ema_beta,
            "relative_activation_epsilon": self.relative_activation_epsilon,
            "weighted_max_ratio": self.weighted_max_ratio,
            "weighted_rank_power": self.weighted_rank_power,
            "random_seed": self.random_seed,
            "mask_version": self.mask_version,
            "keep_mask": self.keep_mask.cpu(),
            "selection_score": self.selection_score.cpu(),
            "selection_score_initialized": self.selection_score_initialized,
            "activation_ema": self.activation_ema.cpu(),
            "activation_ema_initialized": self.activation_ema_initialized,
            "relative_activation_score": self.relative_activation_score.cpu(),
            "last_selected_rank_mean": self._last_selected_rank_mean,
            "last_selection_used_score": self._last_selection_used_score,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        checkpoint_format = state.get("checkpoint_format")
        if checkpoint_format not in {_CHECKPOINT_FORMAT, _LEGACY_CHECKPOINT_FORMAT}:
            raise ValueError("incompatible MLP-channel consistency checkpoint")
        if checkpoint_format == _LEGACY_CHECKPOINT_FORMAT:
            if self.selection_strategy != RANDOM_SELECTION or self.score_method != NO_SCORE:
                raise ValueError(
                    "a v1 random consistency checkpoint cannot initialize score-based selection"
                )
        else:
            for name, expected in (
                ("selection_strategy", self.selection_strategy),
                ("score_method", self.score_method),
            ):
                if str(state[name]) != expected:
                    raise ValueError(
                        f"checkpoint {name}={state[name]!r} != configured {expected!r}"
                    )
            for name, expected in (
                ("score_ema_beta", self.score_ema_beta),
                ("activation_ema_beta", self.activation_ema_beta),
                ("relative_activation_epsilon", self.relative_activation_epsilon),
                ("weighted_max_ratio", self.weighted_max_ratio),
                ("weighted_rank_power", self.weighted_rank_power),
            ):
                if float(state[name]) != expected:
                    raise ValueError(
                        f"checkpoint {name}={state[name]} != configured {expected}"
                    )
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
        expected_shape = self.keep_mask.shape
        keep_mask = torch.as_tensor(state["keep_mask"], dtype=torch.bool)
        if keep_mask.shape != expected_shape:
            raise ValueError(
                f"checkpoint keep_mask shape {tuple(keep_mask.shape)} != "
                f"{tuple(expected_shape)}"
            )
        self.keep_mask.copy_(keep_mask)
        if checkpoint_format == _CHECKPOINT_FORMAT:
            for name, target in (
                ("selection_score", self.selection_score),
                ("activation_ema", self.activation_ema),
                ("relative_activation_score", self.relative_activation_score),
            ):
                value = torch.as_tensor(state[name], dtype=torch.float32)
                if value.shape != expected_shape:
                    raise ValueError(
                        f"checkpoint {name} shape {tuple(value.shape)} != "
                        f"{tuple(expected_shape)}"
                    )
                target.copy_(value)
            self.selection_score_initialized = bool(
                state["selection_score_initialized"]
            )
            self.activation_ema_initialized = bool(state["activation_ema_initialized"])
            self._last_selected_rank_mean = float(
                state.get("last_selected_rank_mean", 0.0)
            )
            self._last_selection_used_score = bool(
                state.get("last_selection_used_score", False)
            )
        self.mask_version = int(state["mask_version"])
        self._device_masks.clear()
        self.cancel_score_collection()
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
            activation = controller.apply(
                _layer_idx,
                activation,
                down_weight=this.down_proj.weight,
            )
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
