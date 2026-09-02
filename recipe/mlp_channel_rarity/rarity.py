"""Prompt-level MLP activation rarity with an online EMA baseline.

The controller observes the input to every selected MLP ``down_proj``.  That
tensor is the post-gate SwiGLU channel activation.  It reduces prompt tokens to
one RMS activation per question and channel, compares it with the EMA from
*previous* optimizer steps, and selects the largest relative deviations.

Exposure counts are cumulative.  By default, the frequency used by the
logarithm is the empirical cumulative frequency.  An optional uniform top-k
prior can be enabled:

    p0 = top_k / intermediate_size
    frequency[layer, channel] = (selected_questions + prior_strength * p0)
                                / (observed_questions + prior_strength)

The first observed training step only initializes the activation EMA and emits
unit weights.  From the second step onward, raw question rarity is the mean
self-information of its selected channels.  The global score vector is
projected into a configured box while enforcing a global mean of exactly one.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.distributed as dist

_LAYER_RE = re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)\.mlp(?:\.|$)")


def _layer_index(module_name: str) -> int | None:
    """Resolve an HF layer index through transparent FSDP1 wrappers."""
    # FSDP1 inserts this traversal-only module into each auto-wrapped decoder
    # layer: ``model.layers.0._fsdp_wrapped_module.mlp``.
    normalized_name = ".".join(
        part for part in module_name.split(".") if part != "_fsdp_wrapped_module"
    )
    match = _LAYER_RE.search(normalized_name)
    return int(match.group(1)) if match else None


@dataclass(frozen=True)
class RarityStepResult:
    raw_scores: torch.Tensor
    loss_weights: torch.Tensor
    metrics: dict[str, float]


def project_scores_to_bounded_mean_one(
    scores: torch.Tensor,
    *,
    min_weight: float,
    max_weight: float,
    amplification: float = 1.0,
) -> torch.Tensor:
    """Project non-negative scores to a box while enforcing mean(weight) == 1.

    Starting from ``v = scores / mean(scores)``, first amplify deviations from
    one as ``1 + amplification * (v - 1)``.  Then compute the Euclidean
    projection onto ``[min_weight, max_weight]`` intersected with the unit-mean
    hyperplane.  The solution has the form ``clip(v - lambda, low, high)``.
    """
    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError(f"scores must be a non-empty vector, got shape {tuple(scores.shape)}")
    if not 0.0 < min_weight <= 1.0 <= max_weight:
        raise ValueError("weight bounds must satisfy 0 < min_weight <= 1 <= max_weight")
    if not math.isfinite(amplification) or amplification <= 0.0:
        raise ValueError(f"amplification must be finite and positive, got {amplification}")
    if not bool(torch.isfinite(scores).all().item()) or bool((scores < 0).any().item()):
        return torch.ones_like(scores, dtype=torch.float32)

    values = scores.to(dtype=torch.float64)
    mean = values.mean()
    if float(mean.item()) <= 0.0:
        return torch.ones_like(scores, dtype=torch.float32)
    values = values / mean
    values = 1.0 + amplification * (values - 1.0)

    lower_shift = (values - max_weight).min()
    upper_shift = (values - min_weight).max()
    for _ in range(80):
        shift = (lower_shift + upper_shift) / 2.0
        candidate_mean = torch.clamp(
            values - shift,
            min=min_weight,
            max=max_weight,
        ).mean()
        if float(candidate_mean.item()) > 1.0:
            lower_shift = shift
        else:
            upper_shift = shift

    weights = torch.clamp(
        values - (lower_shift + upper_shift) / 2.0,
        min=min_weight,
        max=max_weight,
    )
    # Remove the final floating-point residual using available box capacity.
    residual = float(weights.numel()) - weights.sum()
    if float(residual.item()) > 0.0:
        capacity = max_weight - weights
        weights = weights + residual * capacity / capacity.sum().clamp_min(1e-12)
    elif float(residual.item()) < 0.0:
        capacity = weights - min_weight
        weights = weights + residual * capacity / capacity.sum().clamp_min(1e-12)

    weights = weights.to(dtype=torch.float32)
    mean_error = abs(float(weights.to(dtype=torch.float64).mean().item()) - 1.0)
    if mean_error > 1e-6:
        raise RuntimeError(
            f"bounded loss-weight projection failed to preserve mean one: error={mean_error}"
        )
    return weights


class MLPChannelRarityController:
    """Maintain the EMA activation baseline and cumulative channel exposure."""

    def __init__(
        self,
        *,
        num_layers: int,
        intermediate_size: int,
        selected_layers: Iterable[int] | None = None,
        activation_ema_beta: float = 0.95,
        topk_ratio: float = 0.01,
        top_k: int | None = None,
        deviation_epsilon: float = 1e-6,
        frequency_epsilon: float = 1e-8,
        frequency_prior_strength: float = 64.0,
        max_channel_rarity: float = 8.0,
        responses_per_question: int = 1,
        use_frequency_prior: bool = False,
        min_loss_weight: float = 0.2,
        max_loss_weight: float = 5.0,
        loss_weight_amplification: float = 1.0,
    ) -> None:
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if intermediate_size <= 0:
            raise ValueError(f"intermediate_size must be positive, got {intermediate_size}")
        if not 0.0 <= activation_ema_beta < 1.0:
            raise ValueError(
                f"activation_ema_beta must be in [0, 1), got {activation_ema_beta}"
            )
        if not 0.0 < topk_ratio <= 1.0:
            raise ValueError(f"topk_ratio must be in (0, 1], got {topk_ratio}")
        if deviation_epsilon <= 0.0 or frequency_epsilon <= 0.0:
            raise ValueError("deviation_epsilon and frequency_epsilon must be positive")
        if frequency_prior_strength < 0.0:
            raise ValueError(
                f"frequency_prior_strength must be non-negative, got {frequency_prior_strength}"
            )
        if max_channel_rarity <= 0.0:
            raise ValueError(
                f"max_channel_rarity must be positive, got {max_channel_rarity}"
            )
        if responses_per_question <= 0:
            raise ValueError(
                f"responses_per_question must be positive, got {responses_per_question}"
            )
        if not 0.0 < min_loss_weight <= 1.0 <= max_loss_weight:
            raise ValueError(
                "loss-weight bounds must satisfy 0 < min_loss_weight <= 1 <= max_loss_weight"
            )
        if not math.isfinite(loss_weight_amplification) or loss_weight_amplification <= 0.0:
            raise ValueError(
                "loss_weight_amplification must be finite and positive, "
                f"got {loss_weight_amplification}"
            )

        layers = (
            list(range(num_layers))
            if selected_layers is None
            else [int(x) for x in selected_layers]
        )
        if not layers:
            raise ValueError("selected_layers must contain at least one layer")
        if len(layers) != len(set(layers)):
            raise ValueError(f"selected_layers contains duplicates: {layers}")
        if min(layers) < 0 or max(layers) >= num_layers:
            raise ValueError(f"selected_layers={layers} is outside [0, {num_layers})")

        resolved_top_k = (
            int(top_k)
            if top_k is not None
            else max(1, round(intermediate_size * topk_ratio))
        )
        if not 1 <= resolved_top_k <= intermediate_size:
            raise ValueError(
                f"top_k must be in [1, {intermediate_size}], got {resolved_top_k}"
            )

        self.num_layers = int(num_layers)
        self.intermediate_size = int(intermediate_size)
        self.selected_layers = tuple(sorted(layers))
        self.layer_to_slot = {layer: slot for slot, layer in enumerate(self.selected_layers)}
        self.activation_ema_beta = float(activation_ema_beta)
        self.topk_ratio = float(topk_ratio)
        self.top_k = resolved_top_k
        self.deviation_epsilon = float(deviation_epsilon)
        self.frequency_epsilon = float(frequency_epsilon)
        self.frequency_prior_strength = float(frequency_prior_strength)
        self.max_channel_rarity = float(max_channel_rarity)
        self.prior_frequency = float(self.top_k / self.intermediate_size)
        self.responses_per_question = int(responses_per_question)
        self.use_frequency_prior = bool(use_frequency_prior)
        self.min_loss_weight = float(min_loss_weight)
        self.max_loss_weight = float(max_loss_weight)
        self.loss_weight_amplification = float(loss_weight_amplification)

        shape = (len(self.selected_layers), self.intermediate_size)
        self.normal_activation = torch.zeros(shape, dtype=torch.float32)
        self.exposure_count = torch.zeros(shape, dtype=torch.float32)
        self.exposure_questions = 0.0
        self.step_count = 0
        self.ema_initialized = False

        self.collecting = False
        self._prompt_mask: torch.Tensor | None = None
        self._sample_ids: torch.Tensor | None = None
        self._micro_batch_samples = 0
        self._seen_layers: set[int] = set()
        self._activation_sum: torch.Tensor | None = None
        self._topk_parts: dict[int, list[torch.Tensor]] = {}
        self._local_sample_count = 0
        self._selected_deviation_sum: torch.Tensor | None = None
        self._selected_deviation_count = 0
        self._selected_deviation_max: torch.Tensor | None = None
        self._hook_handles: list[Any] = []

    @property
    def first_step(self) -> bool:
        return not self.ema_initialized

    def begin_step(self) -> None:
        if self.collecting:
            raise RuntimeError("cannot begin a rarity step while another step is active")
        self.collecting = True
        self._prompt_mask = None
        self._sample_ids = None
        self._micro_batch_samples = 0
        self._seen_layers.clear()
        self._activation_sum = None
        self._topk_parts = {layer: [] for layer in self.selected_layers}
        self._local_sample_count = 0
        self._selected_deviation_sum = None
        self._selected_deviation_count = 0
        self._selected_deviation_max = None

    def begin_micro_batch(
        self,
        *,
        prompt_mask: torch.Tensor,
        sample_ids: torch.Tensor,
        sample_count: int,
    ) -> None:
        if not self.collecting:
            raise RuntimeError("begin_micro_batch called outside a rarity step")
        if self._prompt_mask is not None:
            raise RuntimeError("previous rarity micro-batch was not closed")
        if prompt_mask.shape != sample_ids.shape:
            raise ValueError(
                f"prompt_mask shape {tuple(prompt_mask.shape)} != sample_ids {tuple(sample_ids.shape)}"
            )
        if sample_count <= 0:
            raise ValueError(f"sample_count must be positive, got {sample_count}")
        self._prompt_mask = prompt_mask.detach().to(dtype=torch.bool)
        self._sample_ids = sample_ids.detach().to(dtype=torch.long)
        self._micro_batch_samples = int(sample_count)
        self._seen_layers.clear()

    def observe(self, layer_idx: int, activation: torch.Tensor) -> None:
        """Observe one selected layer's post-gate activation for a micro-batch."""
        if not self.collecting or layer_idx not in self.layer_to_slot:
            return
        if self._prompt_mask is None or self._sample_ids is None:
            raise RuntimeError("MLP activation observed before the prompt mask was installed")
        if layer_idx in self._seen_layers:
            raise RuntimeError(f"selected MLP layer {layer_idx} fired more than once in one forward")
        if activation.shape[-1] != self.intermediate_size:
            raise RuntimeError(
                f"layer {layer_idx} activation width {activation.shape[-1]} != "
                f"intermediate_size {self.intermediate_size}"
            )
        if activation.numel() // activation.shape[-1] != self._prompt_mask.numel():
            raise RuntimeError(
                f"layer {layer_idx} leading activation shape {tuple(activation.shape[:-1])} "
                f"does not match prompt mask {tuple(self._prompt_mask.shape)}"
            )

        with torch.no_grad():
            device = activation.device
            self._move_state(device)
            if self._activation_sum is None:
                self._activation_sum = torch.zeros_like(self.normal_activation, device=device)

            flat_activation = activation.detach().reshape(-1, self.intermediate_size)
            flat_mask = self._prompt_mask.reshape(-1).to(device=device)
            flat_sample_ids = self._sample_ids.reshape(-1).to(device=device)
            valid_activation = flat_activation[flat_mask].to(dtype=torch.float32)
            valid_sample_ids = flat_sample_ids[flat_mask]
            if valid_activation.numel() == 0:
                raise RuntimeError("rarity micro-batch contains no valid prompt tokens")
            if (
                int(valid_sample_ids.min().item()) < 0
                or int(valid_sample_ids.max().item()) >= self._micro_batch_samples
            ):
                raise RuntimeError("rarity sample IDs are outside the current micro-batch")

            squared_sum = torch.zeros(
                (self._micro_batch_samples, self.intermediate_size),
                device=device,
                dtype=torch.float32,
            )
            squared_sum.index_add_(0, valid_sample_ids, valid_activation.square())
            token_count = torch.bincount(
                valid_sample_ids,
                minlength=self._micro_batch_samples,
            ).to(device=device, dtype=torch.float32)
            if bool((token_count == 0).any().item()):
                missing = torch.nonzero(token_count == 0).flatten().tolist()
                raise RuntimeError(f"questions without prompt tokens in rarity batch: {missing}")
            activation_level = torch.sqrt(
                squared_sum / token_count.unsqueeze(-1).clamp_min(1.0)
            )

            slot = self.layer_to_slot[layer_idx]
            self._activation_sum[slot].add_(activation_level.sum(dim=0))
            if self.ema_initialized:
                normal = self.normal_activation[slot]
                relative_deviation = (activation_level - normal) / normal.clamp_min(
                    self.deviation_epsilon
                )
                top_values, top_indices = torch.topk(
                    relative_deviation,
                    k=self.top_k,
                    dim=-1,
                    largest=True,
                    sorted=False,
                )
                self._topk_parts[layer_idx].append(top_indices)
                selected_sum = top_values.sum(dtype=torch.float32)
                self._selected_deviation_sum = (
                    selected_sum
                    if self._selected_deviation_sum is None
                    else self._selected_deviation_sum + selected_sum
                )
                self._selected_deviation_count += int(top_values.numel())
                selected_max = top_values.max()
                self._selected_deviation_max = (
                    selected_max
                    if self._selected_deviation_max is None
                    else torch.maximum(self._selected_deviation_max, selected_max)
                )
            self._seen_layers.add(layer_idx)

    def end_micro_batch(self) -> None:
        if self._prompt_mask is None:
            raise RuntimeError("end_micro_batch called without an active micro-batch")
        missing = set(self.selected_layers) - self._seen_layers
        if missing:
            raise RuntimeError(f"selected MLP hooks did not fire for layers {sorted(missing)}")
        self._local_sample_count += self._micro_batch_samples
        self._prompt_mask = None
        self._sample_ids = None
        self._micro_batch_samples = 0
        self._seen_layers.clear()

    def abort_step(self) -> None:
        self.collecting = False
        self._prompt_mask = None
        self._sample_ids = None
        self._activation_sum = None
        self._topk_parts.clear()
        self._seen_layers.clear()

    def finalize_step(self) -> RarityStepResult:
        if not self.collecting:
            raise RuntimeError("finalize_step called without an active rarity step")
        if self._prompt_mask is not None:
            raise RuntimeError("cannot finalize rarity step with an open micro-batch")
        if self._activation_sum is None or self._local_sample_count <= 0:
            raise RuntimeError("rarity step observed no prompt activations")

        try:
            device = self._activation_sum.device
            global_activation_sum = self._activation_sum.clone()
            global_sample_count = torch.tensor(
                float(self._local_sample_count), device=device, dtype=torch.float32
            )
            self._all_reduce(global_activation_sum)
            self._all_reduce(global_sample_count)
            batch_normal = global_activation_sum / global_sample_count.clamp_min(1.0)

            if not self.ema_initialized:
                self.normal_activation.copy_(batch_normal)
                self.ema_initialized = True
                raw_scores = torch.ones(
                    self._local_sample_count, device=device, dtype=torch.float32
                )
                loss_weights = torch.ones_like(raw_scores)
                rarity_metrics = {
                    "mlp_rarity/exposure_questions": 0.0,
                    "mlp_rarity/exposed_channel_fraction": 0.0,
                    "mlp_rarity/raw_score_mean": 1.0,
                    "mlp_rarity/raw_score_std": 0.0,
                    "mlp_rarity/loss_weight_mean": 1.0,
                    "mlp_rarity/loss_weight_min": 1.0,
                    "mlp_rarity/loss_weight_max": 1.0,
                    "mlp_rarity/selected_deviation_mean": 0.0,
                    "mlp_rarity/selected_deviation_max": 0.0,
                }
            else:
                topk_by_layer = []
                for layer_idx in self.selected_layers:
                    parts = self._topk_parts[layer_idx]
                    if not parts:
                        raise RuntimeError(f"no top-k selections collected for layer {layer_idx}")
                    layer_topk = torch.cat(parts, dim=0)
                    if layer_topk.shape != (self._local_sample_count, self.top_k):
                        raise RuntimeError(
                            f"layer {layer_idx} top-k shape {tuple(layer_topk.shape)} != "
                            f"({self._local_sample_count}, {self.top_k})"
                        )
                    topk_by_layer.append(layer_topk)
                local_topk = torch.stack(topk_by_layer, dim=1)

                local_exposure = torch.zeros_like(self.exposure_count, device=device)
                for slot in range(len(self.selected_layers)):
                    indices = local_topk[:, slot, :].reshape(-1)
                    local_exposure[slot].scatter_add_(
                        0,
                        indices,
                        torch.ones_like(indices, dtype=torch.float32),
                    )
                self._all_reduce(local_exposure)
                # GRPO repeats each question the same number of times. Dividing
                # both selected counts and the denominator by that fixed group
                # size makes the stored frequency question-level. If tiny
                # numerical differences make responses choose different top-k
                # sets, the fractional count is their within-question average.
                self.exposure_count.add_(
                    local_exposure,
                    alpha=1.0 / self.responses_per_question,
                )
                self.exposure_questions += (
                    float(global_sample_count.item()) / self.responses_per_question
                )

                empirical_frequency = self.exposure_count / max(self.exposure_questions, 1.0)
                if self.use_frequency_prior:
                    frequency = (
                        self.exposure_count
                        + self.frequency_prior_strength * self.prior_frequency
                    ) / (self.exposure_questions + self.frequency_prior_strength)
                else:
                    frequency = empirical_frequency
                channel_rarity = (-torch.log(
                    frequency.clamp_min(self.frequency_epsilon)
                )).clamp(max=self.max_channel_rarity)
                expanded_rarity = channel_rarity.unsqueeze(0).expand(
                    self._local_sample_count, -1, -1
                )
                selected_rarity = torch.gather(expanded_rarity, 2, local_topk)
                raw_scores = selected_rarity.mean(dim=(1, 2), dtype=torch.float32)

                global_raw_scores, local_start = self._all_gather_vector(raw_scores)
                raw_mean = global_raw_scores.mean(dtype=torch.float32)
                raw_var = (
                    global_raw_scores.square().mean(dtype=torch.float32) - raw_mean.square()
                )
                global_loss_weights = project_scores_to_bounded_mean_one(
                    global_raw_scores,
                    min_weight=self.min_loss_weight,
                    max_weight=self.max_loss_weight,
                    amplification=self.loss_weight_amplification,
                )
                loss_weights = global_loss_weights[
                    local_start : local_start + raw_scores.numel()
                ]
                global_weight_mean = global_loss_weights.to(dtype=torch.float64).mean()
                global_weight_min = global_loss_weights.min()
                global_weight_max = global_loss_weights.max()

                deviation_sum = self._selected_deviation_sum
                if deviation_sum is None:
                    deviation_sum = torch.zeros((), device=device, dtype=torch.float32)
                deviation_count = torch.tensor(
                    float(self._selected_deviation_count), device=device, dtype=torch.float32
                )
                deviation_max = self._selected_deviation_max
                if deviation_max is None:
                    deviation_max = torch.full((), float("-inf"), device=device)
                self._all_reduce(deviation_sum)
                self._all_reduce(deviation_count)
                self._all_reduce(deviation_max, op=dist.ReduceOp.MAX)

                positive_empirical_frequency = empirical_frequency[empirical_frequency > 0]
                rarity_metrics = {
                    "mlp_rarity/exposure_questions": float(self.exposure_questions),
                    "mlp_rarity/exposed_channel_fraction": float(
                        (self.exposure_count > 0).to(dtype=torch.float32).mean().item()
                    ),
                    "mlp_rarity/effective_frequency_mean": float(frequency.mean().item()),
                    "mlp_rarity/effective_frequency_max": float(frequency.max().item()),
                    "mlp_rarity/effective_frequency_min": float(frequency.min().item()),
                    "mlp_rarity/empirical_frequency_max": float(
                        empirical_frequency.max().item()
                    ),
                    "mlp_rarity/empirical_frequency_min_positive": float(
                        positive_empirical_frequency.min().item()
                        if positive_empirical_frequency.numel()
                        else 0.0
                    ),
                    "mlp_rarity/channel_rarity_max_observed": float(
                        channel_rarity.max().item()
                    ),
                    "mlp_rarity/raw_score_mean": float(raw_mean.item()),
                    "mlp_rarity/raw_score_std": float(raw_var.clamp_min(0.0).sqrt().item()),
                    "mlp_rarity/loss_weight_mean": float(
                        global_weight_mean.item()
                    ),
                    "mlp_rarity/loss_weight_mean_error": abs(
                        float(global_weight_mean.item()) - 1.0
                    ),
                    "mlp_rarity/loss_weight_min": float(global_weight_min.item()),
                    "mlp_rarity/loss_weight_max": float(global_weight_max.item()),
                    "mlp_rarity/selected_deviation_mean": float(
                        (deviation_sum / deviation_count.clamp_min(1.0)).item()
                    ),
                    "mlp_rarity/selected_deviation_max": float(deviation_max.item()),
                }

                self.normal_activation.mul_(self.activation_ema_beta).add_(
                    batch_normal,
                    alpha=1.0 - self.activation_ema_beta,
                )

            self.step_count += 1
            metrics = {
                "mlp_rarity/step": float(self.step_count),
                "mlp_rarity/ema_initialized": float(self.ema_initialized),
                "mlp_rarity/first_step_unit_weights": float(self.step_count == 1),
                "mlp_rarity/activation_ema_beta": self.activation_ema_beta,
                "mlp_rarity/frequency_prior_strength": self.frequency_prior_strength,
                "mlp_rarity/prior_frequency": self.prior_frequency,
                "mlp_rarity/use_frequency_prior": float(self.use_frequency_prior),
                "mlp_rarity/max_channel_rarity": self.max_channel_rarity,
                "mlp_rarity/responses_per_question": float(self.responses_per_question),
                "mlp_rarity/normal_activation_mean": float(self.normal_activation.mean().item()),
                "mlp_rarity/normal_activation_max": float(self.normal_activation.max().item()),
                "mlp_rarity/selected_layers": float(len(self.selected_layers)),
                "mlp_rarity/top_k": float(self.top_k),
                "mlp_rarity/loss_weight_amplification": self.loss_weight_amplification,
                **rarity_metrics,
            }
            return RarityStepResult(
                raw_scores=raw_scores.detach(),
                loss_weights=loss_weights.detach(),
                metrics=metrics,
            )
        finally:
            self.abort_step()

    def state_dict(self) -> dict[str, Any]:
        return {
            "num_layers": self.num_layers,
            "intermediate_size": self.intermediate_size,
            "selected_layers": self.selected_layers,
            "activation_ema_beta": self.activation_ema_beta,
            "top_k": self.top_k,
            "frequency_prior_strength": self.frequency_prior_strength,
            "max_channel_rarity": self.max_channel_rarity,
            "responses_per_question": self.responses_per_question,
            "use_frequency_prior": self.use_frequency_prior,
            "min_loss_weight": self.min_loss_weight,
            "max_loss_weight": self.max_loss_weight,
            "loss_weight_amplification": self.loss_weight_amplification,
            "normal_activation": self.normal_activation.detach().cpu(),
            "exposure_count": self.exposure_count.detach().cpu(),
            "exposure_questions": self.exposure_questions,
            "step_count": self.step_count,
            "ema_initialized": self.ema_initialized,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = (len(self.selected_layers), self.intermediate_size)
        if int(state["num_layers"]) != self.num_layers:
            raise ValueError("rarity checkpoint num_layers does not match the current model")
        if int(state["intermediate_size"]) != self.intermediate_size:
            raise ValueError("rarity checkpoint intermediate_size does not match the current model")
        if tuple(int(x) for x in state["selected_layers"]) != self.selected_layers:
            raise ValueError("rarity checkpoint selected_layers does not match the recipe config")
        if int(state["top_k"]) != self.top_k:
            raise ValueError("rarity checkpoint top_k does not match the recipe config")
        if float(state["activation_ema_beta"]) != self.activation_ema_beta:
            raise ValueError(
                "rarity checkpoint activation_ema_beta does not match the recipe config"
            )
        if float(state["frequency_prior_strength"]) != self.frequency_prior_strength:
            raise ValueError(
                "rarity checkpoint frequency_prior_strength does not match the recipe config"
            )
        if float(state["max_channel_rarity"]) != self.max_channel_rarity:
            raise ValueError(
                "rarity checkpoint max_channel_rarity does not match the recipe config"
            )
        if int(state["responses_per_question"]) != self.responses_per_question:
            raise ValueError(
                "rarity checkpoint responses_per_question does not match the rollout config"
            )
        if bool(state["use_frequency_prior"]) != self.use_frequency_prior:
            raise ValueError(
                "rarity checkpoint use_frequency_prior does not match the recipe config"
            )
        if float(state.get("min_loss_weight", 0.2)) != self.min_loss_weight:
            raise ValueError(
                "rarity checkpoint min_loss_weight does not match the recipe config"
            )
        if float(state.get("max_loss_weight", 5.0)) != self.max_loss_weight:
            raise ValueError(
                "rarity checkpoint max_loss_weight does not match the recipe config"
            )
        if float(state.get("loss_weight_amplification", 1.0)) != self.loss_weight_amplification:
            raise ValueError(
                "rarity checkpoint loss_weight_amplification does not match the recipe config"
            )
        normal = torch.as_tensor(state["normal_activation"], dtype=torch.float32)
        exposure = torch.as_tensor(state["exposure_count"], dtype=torch.float32)
        if tuple(normal.shape) != expected or tuple(exposure.shape) != expected:
            raise ValueError(
                f"rarity checkpoint tensors must have shape {expected}, got "
                f"normal={tuple(normal.shape)}, exposure={tuple(exposure.shape)}"
            )
        self.normal_activation = normal.clone()
        self.exposure_count = exposure.clone()
        self.exposure_questions = float(state.get("exposure_questions", 0.0))
        self.step_count = int(state.get("step_count", 0))
        self.ema_initialized = bool(state.get("ema_initialized", self.step_count > 0))

    def _move_state(self, device: torch.device) -> None:
        if self.normal_activation.device != device:
            self.normal_activation = self.normal_activation.to(device=device)
            self.exposure_count = self.exposure_count.to(device=device)

    @staticmethod
    def _all_reduce(tensor: torch.Tensor, op: dist.ReduceOp = dist.ReduceOp.SUM) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=op)

    @staticmethod
    def _all_gather_vector(local_values: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Gather a small variable-length score vector in rank order."""
        if not dist.is_available() or not dist.is_initialized():
            return local_values, 0
        device = local_values.device
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_count = torch.tensor(local_values.numel(), device=device, dtype=torch.long)
        count_parts = [torch.zeros_like(local_count) for _ in range(world_size)]
        dist.all_gather(count_parts, local_count)
        counts = [int(part.item()) for part in count_parts]
        max_count = max(counts)
        padded = torch.zeros(max_count, device=device, dtype=local_values.dtype)
        padded[: local_values.numel()] = local_values
        gathered_parts = [torch.zeros_like(padded) for _ in range(world_size)]
        dist.all_gather(gathered_parts, padded)
        gathered = torch.cat(
            [part[:count] for part, count in zip(gathered_parts, counts)],
            dim=0,
        )
        return gathered, sum(counts[:rank])


def install_hf_mlp_activation_observer(
    model: torch.nn.Module,
    controller: MLPChannelRarityController,
) -> list[str]:
    """Attach read-only pre-hooks to dense HF SwiGLU ``down_proj`` modules."""
    installed: list[str] = []
    seen_layers: set[int] = set()
    for name, module in model.named_modules():
        layer_idx = _layer_index(name)
        if layer_idx is None:
            continue
        if layer_idx not in controller.layer_to_slot or layer_idx in seen_layers:
            continue
        if not all(hasattr(module, attr) for attr in ("gate_proj", "up_proj", "down_proj", "act_fn")):
            continue
        if getattr(module, "_mlp_channel_rarity_observer", None) is not None:
            raise RuntimeError(f"MLP module {name} already has a rarity observer")

        def observe_down_input(_down_proj, inputs, _layer_idx=layer_idx):
            if len(inputs) != 1:
                raise RuntimeError("dense MLP down_proj observer expected exactly one input")
            controller.observe(_layer_idx, inputs[0])

        handle = module.down_proj.register_forward_pre_hook(observe_down_input)
        module._mlp_channel_rarity_observer = controller
        controller._hook_handles.append(handle)
        installed.append(name)
        seen_layers.add(layer_idx)

    missing = set(controller.selected_layers) - seen_layers
    if missing:
        raise RuntimeError(
            "could not find dense HF SwiGLU MLP modules for selected layers "
            f"{sorted(missing)}"
        )
    return installed
