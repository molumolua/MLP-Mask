"""MLP-channel masks selected by configurable clean-policy scores.

The controller deliberately keeps masks outside the model state dict.  vLLM active
buffers are registered on their MLP modules before compilation/CUDA-graph capture;
other backends may allocate them lazily.  Route switches update each available buffer
in place (or defer the update while vLLM sleeps), so a captured graph keeps the same
pointer and reads new clean/masked values without recapture.

The preferred selector is soft-top sampling: score percentile ranks are converted to
bounded positive weights and an exact per-layer quota is sampled without replacement.
The controller supports relative RMS activation, scale-aware output contribution,
gradient x activation, and an online randomized grouped-ablation estimate.
"""

from __future__ import annotations

import math
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
TOP_RELATIVE_ACTIVATION_SELECTION = "top_relative_activation"
RANDOM_SELECTION = "random"
WEIGHTED_RANDOM_SELECTION = "weighted_random"
SOFT_TOP_SELECTION = "soft_top"
_VALID_SELECTION_STRATEGIES = {
    TOP_RELATIVE_ACTIVATION_SELECTION,
    RANDOM_SELECTION,
    WEIGHTED_RANDOM_SELECTION,
    SOFT_TOP_SELECTION,
}
RELATIVE_ACTIVATION_SCORE = "relative_activation"
OUTPUT_CONTRIBUTION_SCORE = "output_contribution"
GRADIENT_ACTIVATION_SCORE = "gradient_activation"
CAUSAL_ABLATION_SCORE = "causal_ablation"
_VALID_SCORE_METHODS = {
    RELATIVE_ACTIVATION_SCORE,
    OUTPUT_CONTRIBUTION_SCORE,
    GRADIENT_ACTIVATION_SCORE,
    CAUSAL_ABLATION_SCORE,
}
_FORWARD_SCORE_METHODS = {
    RELATIVE_ACTIVATION_SCORE,
    OUTPUT_CONTRIBUTION_SCORE,
    GRADIENT_ACTIVATION_SCORE,
}
PER_LAYER_RANDOM_SCOPE = "per_layer"
GLOBAL_RANDOM_SCOPE = "global"
_VALID_RANDOM_SCOPES = {PER_LAYER_RANDOM_SCOPE, GLOBAL_RANDOM_SCOPE}
_LAYER_RE = re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)\.mlp(?:\.|$)")
_LEGACY_ACTIVATION_SCORE_TYPE = "relative_activation_rms_v1"
_CHECKPOINT_FORMAT = "mlp_channel_soft_top_scores_v2"


@dataclass(frozen=True)
class MaskRefreshResult:
    metrics: dict[str, float]
    timings: dict[str, float]


class MLPChannelInterventionController:
    """Own per-block masks, channel scores, and mask history.

    ``keep_mask[layer, channel]`` is True for an available channel and False for a
    channel removed by the structured intervention.  The score definition is
    independent from the selection rule so all score variants can share soft-top.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        intermediate_size: int,
        mask_ratio: float = 0.10,
        activation_ema_beta: float = 0.95,
        relative_activation_epsilon: float = 1e-6,
        selection_strategy: str = TOP_RELATIVE_ACTIVATION_SELECTION,
        score_method: str = RELATIVE_ACTIVATION_SCORE,
        score_ema_beta: float = 0.0,
        random_seed: int = 42,
        random_scope: str = PER_LAYER_RANDOM_SCOPE,
        weighted_max_ratio: float = 4.0,
        weighted_rank_power: float = 2.0,
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
        if not 0.0 <= activation_ema_beta < 1.0:
            raise ValueError(
                f"activation_ema_beta must be in [0, 1), got {activation_ema_beta}"
            )
        if relative_activation_epsilon <= 0.0:
            raise ValueError(
                "relative_activation_epsilon must be positive, got "
                f"{relative_activation_epsilon}"
            )
        if selection_strategy not in _VALID_SELECTION_STRATEGIES:
            raise ValueError(
                f"selection_strategy must be one of {sorted(_VALID_SELECTION_STRATEGIES)}, "
                f"got {selection_strategy!r}"
            )
        if score_method not in _VALID_SCORE_METHODS:
            raise ValueError(
                f"score_method must be one of {sorted(_VALID_SCORE_METHODS)}, got {score_method!r}"
            )
        if not 0.0 <= score_ema_beta < 1.0:
            raise ValueError(f"score_ema_beta must be in [0, 1), got {score_ema_beta}")
        if (
            selection_strategy == TOP_RELATIVE_ACTIVATION_SELECTION
            and score_method != RELATIVE_ACTIVATION_SCORE
        ):
            raise ValueError("top_relative_activation requires score_method=relative_activation")
        if selection_strategy == WEIGHTED_RANDOM_SELECTION and score_method != RELATIVE_ACTIVATION_SCORE:
            raise ValueError(
                "legacy weighted_random requires score_method=relative_activation; "
                "use selection_strategy=soft_top for other scores"
            )
        if (
            score_method == CAUSAL_ABLATION_SCORE
            and selection_strategy != SOFT_TOP_SELECTION
        ):
            raise ValueError("causal_ablation requires selection_strategy=soft_top")
        if random_seed < 0:
            raise ValueError(f"random_seed must be non-negative, got {random_seed}")
        if random_scope not in _VALID_RANDOM_SCOPES:
            raise ValueError(
                f"random_scope must be one of {sorted(_VALID_RANDOM_SCOPES)}, got {random_scope!r}"
            )
        if selection_strategy != RANDOM_SELECTION and random_scope != PER_LAYER_RANDOM_SCOPE:
            raise ValueError("random_scope=global requires selection_strategy=random")
        if weighted_max_ratio < 1.0:
            raise ValueError(f"weighted_max_ratio must be >= 1, got {weighted_max_ratio}")
        if weighted_rank_power <= 0.0:
            raise ValueError(f"weighted_rank_power must be positive, got {weighted_rank_power}")
        if not 0 <= tp_rank < tp_size:
            raise ValueError(f"invalid tensor-parallel rank {tp_rank}/{tp_size}")
        if intermediate_size % tp_size != 0:
            raise ValueError(
                f"intermediate_size={intermediate_size} must be divisible by tensor-parallel size {tp_size}"
            )

        self.num_layers = int(num_layers)
        self.intermediate_size = int(intermediate_size)
        self.mask_ratio = float(mask_ratio)
        self.activation_ema_beta = float(activation_ema_beta)
        self.relative_activation_epsilon = float(relative_activation_epsilon)
        self.selection_strategy = str(selection_strategy)
        self.score_method = str(score_method)
        self.score_ema_beta = float(score_ema_beta)
        self.random_seed = int(random_seed)
        self.random_scope = str(random_scope)
        self.weighted_max_ratio = float(weighted_max_ratio)
        self.weighted_rank_power = float(weighted_rank_power)
        self.tp_rank = int(tp_rank)
        self.tp_size = int(tp_size)
        self.name = str(name)

        self.keep_mask = torch.ones((self.num_layers, self.intermediate_size), dtype=torch.bool)
        self.ever_masked = torch.zeros_like(self.keep_mask)
        self.activation_ema = torch.zeros(
            (self.num_layers, self.intermediate_size), dtype=torch.float32
        )
        self.relative_activation_score = torch.zeros_like(self.activation_ema)
        self.selection_score = torch.zeros_like(self.activation_ema)
        self.activation_ema_initialized = False
        self.relative_activation_initialized = False
        self.selection_score_initialized = False
        self.causal_reward_gap_ema = 0.0
        self.causal_reward_gap_initialized = False
        self.causal_observations = 0
        self._last_causal_reward_gap = 0.0
        self._last_causal_residual = 0.0
        self.causal_assignment_contrast = torch.zeros_like(self.activation_ema)
        self.causal_assignment_contrast_initialized = False
        self.mask_version = 0
        self.cumulative_mask_assignments = 0

        self.route = CLEAN_ROUTE
        self.collect_activation = False
        self._response_token_mask: torch.Tensor | None = None
        self._response_sample_ids: torch.Tensor | None = None
        self._response_sample_count = 0
        self._activation_level_sum: dict[int, torch.Tensor] = {}
        self._gradient_activation_sum: dict[int, torch.Tensor] = {}
        self._activation_sample_count: torch.Tensor | None = None
        self._activation_token_count: torch.Tensor | None = None
        self._activation_accumulate_cpu_s = 0.0
        self._active_buffers: dict[tuple[int, str, int, torch.dtype, int], torch.Tensor] = {}
        self._active_buffers_available = True

    @property
    def current_masked_channels(self) -> int:
        return int((~self.keep_mask).sum().item())

    @property
    def ever_masked_channels(self) -> int:
        return int(self.ever_masked.sum().item())

    @property
    def total_channels(self) -> int:
        return self.num_layers * self.intermediate_size

    def set_route(self, route: str, *, collect_activation: bool = False) -> None:
        if route not in _VALID_ROUTES:
            raise ValueError(f"route must be one of {_VALID_ROUTES}, got {route!r}")
        if collect_activation and route != CLEAN_ROUTE:
            raise ValueError("activation may only be collected on the clean route")
        if collect_activation and self.score_method not in _FORWARD_SCORE_METHODS:
            raise ValueError(
                f"score_method={self.score_method!r} does not use actor activation collection"
            )
        self.route = route
        self.collect_activation = bool(collect_activation)
        self._response_token_mask = None
        self._response_sample_ids = None
        self._response_sample_count = 0
        self._refresh_active_buffers()

    def set_response_token_mask(
        self,
        response_token_mask: torch.Tensor,
        *,
        sample_ids: torch.Tensor | None = None,
        sample_count: int | None = None,
    ) -> None:
        """Set valid response positions and their sample IDs for RMS reduction."""
        self._response_token_mask = response_token_mask.detach()
        if sample_ids is None:
            if self._response_token_mask.ndim == 0:
                raise ValueError("response_token_mask must have at least one dimension")
            inferred_samples = int(self._response_token_mask.shape[0])
            view_shape = (inferred_samples,) + (1,) * (self._response_token_mask.ndim - 1)
            sample_ids = torch.arange(
                inferred_samples,
                device=self._response_token_mask.device,
                dtype=torch.long,
            ).view(view_shape).expand_as(self._response_token_mask)
            sample_count = inferred_samples
        elif sample_ids.shape != self._response_token_mask.shape:
            raise ValueError(
                f"sample_ids shape {tuple(sample_ids.shape)} does not match response mask "
                f"{tuple(self._response_token_mask.shape)}"
            )
        if sample_count is None or sample_count <= 0:
            raise ValueError(f"sample_count must be positive, got {sample_count}")
        self._response_sample_ids = sample_ids.detach().to(dtype=torch.long)
        self._response_sample_count = int(sample_count)
        if self.collect_activation:
            device = self._response_token_mask.device
            if self._activation_sample_count is None:
                self._activation_sample_count = torch.zeros((), device=device, dtype=torch.float32)
                self._activation_token_count = torch.zeros((), device=device, dtype=torch.float32)
            self._activation_sample_count.add_(float(self._response_sample_count))
            self._activation_token_count.add_(
                self._response_token_mask.to(dtype=torch.float32).sum()
            )

    def end_batch(self) -> None:
        self.collect_activation = False
        self._response_token_mask = None
        self._response_sample_ids = None
        self._response_sample_count = 0

    def apply(
        self,
        layer_idx: int,
        activation: torch.Tensor,
        *,
        down_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the route and optionally collect the configured clean score."""
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

        if self.collect_activation and activation.requires_grad:
            if self._response_token_mask is None or self._response_sample_ids is None:
                raise RuntimeError("clean activation requested before response metadata was installed")
            token_mask = self._response_token_mask
            sample_ids = self._response_sample_ids
            sample_count = self._response_sample_count
            down_norm = None
            if self.score_method == OUTPUT_CONTRIBUTION_SCORE:
                if down_weight is None or down_weight.ndim != 2:
                    raise RuntimeError(
                        "output_contribution scoring requires the 2-D down_proj weight"
                    )
                weight_for_norm = down_weight.detach()
                placements = getattr(weight_for_norm, "placements", ())
                if placements:
                    if any(
                        type(placement).__name__ != "Replicate"
                        for placement in placements
                    ):
                        raise RuntimeError(
                            "output_contribution requires down_proj to be unsharded "
                            "during forward"
                        )
                    weight_for_norm = weight_for_norm.to_local()
                down_norm = (
                    weight_for_norm.to(dtype=torch.float32).square().sum(dim=0).sqrt()
                )
                if down_norm.numel() != activation.shape[-1]:
                    raise RuntimeError(
                        "down_proj input width does not match the observed MLP activation"
                    )
            # Waiting for backward avoids counting the checkpointed no-grad forward.
            # gradient_activation consumes the hook gradient; the two forward-only
            # scores use the hook solely as a once-per-training-forward trigger.
            activation.register_hook(
                lambda grad, a=activation.detach(), m=token_mask, ids=sample_ids,
                count=sample_count, layer=layer_idx, dn=down_norm: self._accumulate_score(
                    layer, a, grad, m, ids, count, dn
                )
            )

        active_mask = self._get_active_buffer(layer_idx, activation)
        return activation * active_mask

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
        accumulate_started = time.perf_counter()
        with torch.no_grad():
            expected_shape = activation.shape[:-1]
            if token_mask.numel() != _shape_numel(expected_shape):
                raise RuntimeError(
                    f"response mask has {token_mask.numel()} entries but layer {layer_idx} activation "
                    f"has leading shape {tuple(expected_shape)}"
                )
            if sample_ids.numel() != token_mask.numel():
                raise RuntimeError(
                    f"response sample IDs have {sample_ids.numel()} entries but token mask has "
                    f"{token_mask.numel()}"
                )
            flat_mask = token_mask.reshape(-1).to(device=activation.device, dtype=torch.bool)
            flat_sample_ids = sample_ids.reshape(-1).to(device=activation.device, dtype=torch.long)
            flat_activation = activation.reshape(-1, activation.shape[-1])
            valid_activation = flat_activation[flat_mask]
            valid_sample_ids = flat_sample_ids[flat_mask]
            if valid_activation.numel() == 0:
                raise RuntimeError("clean activation batch contains no valid response tokens")
            if (
                int(valid_sample_ids.min().item()) < 0
                or int(valid_sample_ids.max().item()) >= sample_count
            ):
                raise RuntimeError("response sample IDs are outside the current micro-batch")

            valid_activation = valid_activation.to(dtype=torch.float32)
            token_count = torch.bincount(
                valid_sample_ids,
                minlength=sample_count,
            ).to(device=activation.device, dtype=torch.float32)
            if bool((token_count == 0).any().item()):
                missing = torch.nonzero(token_count == 0).flatten().tolist()
                raise RuntimeError(f"clean samples without response tokens: {missing}")

            if self.score_method == GRADIENT_ACTIVATION_SCORE:
                flat_grad = grad.detach().reshape(-1, grad.shape[-1])
                valid_grad = flat_grad[flat_mask].to(dtype=torch.float32)
                per_sample_sum = torch.zeros(
                    (sample_count, activation.shape[-1]),
                    device=activation.device,
                    dtype=torch.float32,
                )
                per_sample_sum.index_add_(
                    0,
                    valid_sample_ids,
                    (valid_activation * valid_grad).abs(),
                )
                score_level_sum = (
                    per_sample_sum / token_count.unsqueeze(-1).clamp_min(1.0)
                ).sum(dim=0)
            else:
                squared_sum = torch.zeros(
                    (sample_count, activation.shape[-1]),
                    device=activation.device,
                    dtype=torch.float32,
                )
                squared_sum.index_add_(0, valid_sample_ids, valid_activation.square())
                score_level_sum = torch.sqrt(
                    squared_sum / token_count.unsqueeze(-1).clamp_min(1.0)
                ).sum(dim=0)
                if self.score_method == OUTPUT_CONTRIBUTION_SCORE:
                    if down_weight_norm is None:
                        raise RuntimeError("output contribution score is missing down_proj norms")
                    score_level_sum.mul_(down_weight_norm.to(device=score_level_sum.device))

            # Actor/HF activations are expected to be full-width.  Supporting a local
            # width here also makes the controller testable and future-proofs TP actors.
            if score_level_sum.numel() != self.intermediate_size:
                full_sum = torch.zeros(
                    self.intermediate_size,
                    device=score_level_sum.device,
                    dtype=torch.float32,
                )
                start, stop = self._local_slice(score_level_sum.numel())
                full_sum[start:stop] = score_level_sum
                score_level_sum = full_sum
            target = (
                self._gradient_activation_sum
                if self.score_method == GRADIENT_ACTIVATION_SCORE
                else self._activation_level_sum
            )
            accumulator = target.get(layer_idx)
            if accumulator is None or accumulator.device != score_level_sum.device:
                accumulator = torch.zeros_like(score_level_sum)
                target[layer_idx] = accumulator
            accumulator.add_(score_level_sum)
        # This is host dispatch time; GPU work remains included in update_actor.
        self._activation_accumulate_cpu_s += time.perf_counter() - accumulate_started

    def _buffer_key(self, layer_idx: int, activation: torch.Tensor) -> tuple[int, str, int, torch.dtype, int]:
        device = activation.device
        return (layer_idx, device.type, device.index or 0, activation.dtype, activation.shape[-1])

    def _get_active_buffer(self, layer_idx: int, activation: torch.Tensor) -> torch.Tensor:
        key = self._buffer_key(layer_idx, activation)
        buffer = self._active_buffers.get(key)
        if buffer is None:
            buffer = torch.ones(activation.shape[-1], device=activation.device, dtype=activation.dtype)
            self.register_active_buffer(layer_idx, buffer)
        return buffer

    def register_active_buffer(self, layer_idx: int, buffer: torch.Tensor) -> torch.Tensor:
        """Track a stable route buffer that was allocated outside model forward."""
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(f"layer {layer_idx} outside [0, {self.num_layers})")
        if buffer.ndim != 1 or buffer.numel() not in {
            self.intermediate_size,
            self.intermediate_size // self.tp_size,
        }:
            raise RuntimeError(
                f"{self.name} layer {layer_idx} active buffer has shape {tuple(buffer.shape)}, expected "
                f"({self.intermediate_size},) or TP-local "
                f"({self.intermediate_size // self.tp_size},)"
            )
        key = self._buffer_key(layer_idx, buffer)
        # vLLM's worker enters sleep mode before the recipe gets the completed
        # engine back.  The class-level constructor patch has already registered
        # this exact buffer while the engine was awake; the later instance walk is
        # validation only.  Rewriting the sleeping CUDA allocation here causes an
        # asynchronous illegal-memory-access failure.  Keep repeat registration of
        # the same tensor strictly read-only.
        if self._active_buffers.get(key) is buffer:
            return buffer
        self._active_buffers[key] = buffer
        if self._active_buffers_available:
            self._copy_route_to_buffer(layer_idx, buffer)
        return buffer

    def set_active_buffers_available(self, available: bool) -> None:
        """Gate writes to buffers whose CUDA storage may be suspended by vLLM."""
        available = bool(available)
        if available == self._active_buffers_available:
            return
        self._active_buffers_available = available
        if available:
            self._refresh_active_buffers()

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
        if not self._active_buffers_available:
            return
        for key, buffer in self._active_buffers.items():
            self._copy_route_to_buffer(key[0], buffer)

    @property
    def uses_soft_top(self) -> bool:
        return self.selection_strategy in {SOFT_TOP_SELECTION, WEIGHTED_RANDOM_SELECTION}

    @property
    def needs_forward_score(self) -> bool:
        return self.score_method in _FORWARD_SCORE_METHODS

    def _update_selection_score(self, current: torch.Tensor) -> None:
        current = current.detach().to(device="cpu", dtype=torch.float32)
        if tuple(current.shape) != tuple(self.selection_score.shape):
            raise ValueError(
                f"selection score shape {tuple(current.shape)} != {tuple(self.selection_score.shape)}"
            )
        if self.selection_score_initialized:
            self.selection_score.mul_(self.score_ema_beta).add_(
                current,
                alpha=1.0 - self.score_ema_beta,
            )
        else:
            self.selection_score.copy_(current)
            self.selection_score_initialized = True

    @staticmethod
    def _sampling_score_contrast(
        weights: torch.Tensor, selected_in_order: torch.Tensor
    ) -> torch.Tensor:
        """Derivative of an ordered weighted-without-replacement log probability.

        ``torch.multinomial(..., replacement=False)`` induces sequential sampling
        probabilities proportional to each remaining weight.  This derivative has
        expectation zero under that exact assignment policy, including after soft-top
        weights become nonuniform.
        """
        weights = weights.to(device="cpu", dtype=torch.float32)
        selected_in_order = selected_in_order.to(device="cpu", dtype=torch.long)
        selected_weights = weights[selected_in_order]
        removed_before = torch.cat(
            [torch.zeros(1), selected_weights.cumsum(dim=0)[:-1]]
        )
        remaining_totals = weights.sum() - removed_before
        inverse_total_prefix = remaining_totals.reciprocal().cumsum(dim=0)
        contrast = -weights * inverse_total_prefix[-1]
        contrast[selected_in_order] = (
            1.0 - selected_weights * inverse_total_prefix
        )
        return contrast

    def observe_causal_ablation(self, reward_gap_clean_minus_masked: float) -> dict[str, float]:
        """Update an online randomized grouped-ablation score.

        The dual rollout supplies one scalar outcome for the complete structured
        mask.  Its reward-gap residual multiplies the score-function contrast of the
        exact weighted-without-replacement assignment.  This estimates which changes
        in channel masking probability increase the realized clean-minus-masked gap;
        it is not an exact single-channel ablation.
        """
        if self.score_method != CAUSAL_ABLATION_SCORE:
            raise RuntimeError(
                "causal ablation observations require score_method=causal_ablation"
            )
        effect = float(reward_gap_clean_minus_masked)
        if not math.isfinite(effect):
            raise ValueError(f"causal reward gap must be finite, got {effect}")

        self._last_causal_reward_gap = effect
        residual = 0.0
        score_updated = False
        if (
            self.causal_reward_gap_initialized
            and self.causal_assignment_contrast_initialized
        ):
            residual = effect - self.causal_reward_gap_ema
            evidence = residual * self.causal_assignment_contrast
            self._update_selection_score(evidence)
            score_updated = True

        if self.causal_reward_gap_initialized:
            self.causal_reward_gap_ema = (
                self.score_ema_beta * self.causal_reward_gap_ema
                + (1.0 - self.score_ema_beta) * effect
            )
        else:
            self.causal_reward_gap_ema = effect
            self.causal_reward_gap_initialized = True
        self.causal_observations += 1
        self._last_causal_residual = residual
        return {
            "mlp_causal/group_reward_gap": effect,
            "mlp_causal/group_reward_gap_ema": self.causal_reward_gap_ema,
            "mlp_causal/group_reward_gap_residual": residual,
            "mlp_causal/observations": float(self.causal_observations),
            "mlp_causal/score_updated": float(score_updated),
        }

    def refresh_mask(self) -> MaskRefreshResult:
        """Update the configured score, then select the next structured mask."""
        refresh_started = time.perf_counter()
        reduce_elapsed = 0.0
        score_metrics: dict[str, float] = {}
        source_parts = (
            self._gradient_activation_sum
            if self.score_method == GRADIENT_ACTIVATION_SCORE
            else self._activation_level_sum
        )
        has_pending_score = self._activation_sample_count is not None or bool(source_parts)
        batch_activation_cpu: torch.Tensor | None = None
        if self.needs_forward_score and (
            self.selection_strategy == TOP_RELATIVE_ACTIVATION_SELECTION
            or has_pending_score
        ):
            device = self._score_device()
            score_level_sum = torch.stack(
                [
                    source_parts.get(
                        layer_idx,
                        torch.zeros(self.intermediate_size, device=device, dtype=torch.float32),
                    ).to(device)
                    for layer_idx in range(self.num_layers)
                ],
                dim=0,
            )
            sample_count = (
                self._activation_sample_count.to(device)
                if self._activation_sample_count is not None
                else torch.zeros((), device=device, dtype=torch.float32)
            )
            token_count = (
                self._activation_token_count.to(device)
                if self._activation_token_count is not None
                else torch.zeros((), device=device, dtype=torch.float32)
            )
            observed_layers = torch.tensor(
                [
                    float(layer_idx in self._activation_level_sum)
                    if self.score_method != GRADIENT_ACTIVATION_SCORE
                    else float(layer_idx in self._gradient_activation_sum)
                    for layer_idx in range(self.num_layers)
                ],
                device=device,
                dtype=torch.float32,
            )

            reduce_started = time.perf_counter()
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(score_level_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(sample_count, op=dist.ReduceOp.SUM)
                dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
                dist.all_reduce(observed_layers, op=dist.ReduceOp.SUM)
            reduce_elapsed = time.perf_counter() - reduce_started
            if float(sample_count.item()) <= 0 or float(token_count.item()) <= 0:
                raise RuntimeError(
                    "cannot refresh MLP mask: no clean response activation was accumulated"
                )
            if bool((observed_layers == 0).any().item()):
                missing = torch.nonzero(observed_layers == 0).flatten().cpu().tolist()
                raise RuntimeError(
                    f"cannot refresh MLP mask: no activation hook fired for layers {missing}"
                )
            batch_score_cpu = (score_level_sum / sample_count.clamp_min(1.0)).cpu()
            if self.score_method == RELATIVE_ACTIVATION_SCORE:
                batch_activation_cpu = batch_score_cpu
                baseline_was_initialized = self.activation_ema_initialized
                if baseline_was_initialized:
                    denominator = self.activation_ema.clamp_min(
                        self.relative_activation_epsilon
                    )
                    self.relative_activation_score.copy_(
                        (batch_activation_cpu - self.activation_ema) / denominator
                    )
                    self.relative_activation_initialized = True
                    self._update_selection_score(self.relative_activation_score)
                else:
                    self.relative_activation_score.zero_()
            else:
                baseline_was_initialized = self.selection_score_initialized
                self._update_selection_score(batch_score_cpu)
            score_metrics = {
                "mlp_score/current_mean": float(batch_score_cpu.mean().item()),
                "mlp_score/current_max": float(batch_score_cpu.max().item()),
                "mlp_score/current_min": float(batch_score_cpu.min().item()),
                "mlp_activation/response_samples": float(sample_count.item()),
                "mlp_activation/response_tokens": float(token_count.item()),
                "mlp_activation/layers_observed": float((observed_layers > 0).sum().item()),
                "mlp_score/updated_on_refresh": 1.0,
                "mlp_score/initialized_before_refresh": float(
                    baseline_was_initialized
                ),
            }
            if self.score_method == RELATIVE_ACTIVATION_SCORE:
                score_metrics.update(
                    {
                        "mlp_activation/current_rms_mean": float(batch_score_cpu.mean().item()),
                        "mlp_activation/current_rms_max": float(batch_score_cpu.max().item()),
                        "mlp_activation/current_rms_min": float(batch_score_cpu.min().item()),
                    }
                )
        elif self.uses_soft_top and self.needs_forward_score:
            score_metrics = {
                "mlp_activation/response_samples": 0.0,
                "mlp_activation/response_tokens": 0.0,
                "mlp_activation/layers_observed": 0.0,
                "mlp_score/updated_on_refresh": 0.0,
                "mlp_score/initialized_before_refresh": float(self.selection_score_initialized),
            }

        select_started = time.perf_counter()

        masked_per_layer = max(1, int(round(self.intermediate_size * self.mask_ratio)))
        old_masked = ~self.keep_mask
        new_keep = torch.ones_like(self.keep_mask)
        weighted_selected_ranks: list[torch.Tensor] = []
        causal_contrasts: list[torch.Tensor] = []
        if self.selection_strategy == TOP_RELATIVE_ACTIVATION_SELECTION:
            # The first observation only establishes the normal-activation EMA.
            # Masking begins once a later batch can be compared with that baseline.
            if self.relative_activation_initialized:
                for layer_idx in range(self.num_layers):
                    top_idx = torch.topk(
                        self.relative_activation_score[layer_idx],
                        k=masked_per_layer,
                        largest=True,
                        sorted=False,
                    ).indices
                    new_keep[layer_idx, top_idx] = False
        elif self.uses_soft_top:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.random_seed + self.mask_version)
            for layer_idx in range(self.num_layers):
                if not self.selection_score_initialized:
                    weights = torch.ones(self.intermediate_size, dtype=torch.float32)
                    selected = torch.randperm(
                        self.intermediate_size, generator=generator
                    )[:masked_per_layer]
                else:
                    order = torch.argsort(self.selection_score[layer_idx])
                    rank = torch.empty(self.intermediate_size, dtype=torch.float32)
                    rank[order] = torch.arange(self.intermediate_size, dtype=torch.float32)
                    rank.div_(max(self.intermediate_size - 1, 1))
                    weights = 1.0 + (self.weighted_max_ratio - 1.0) * rank.pow(
                        self.weighted_rank_power
                    )
                    selected = torch.multinomial(
                        weights,
                        num_samples=masked_per_layer,
                        replacement=False,
                        generator=generator,
                    )
                    weighted_selected_ranks.append(rank[selected])
                if self.score_method == CAUSAL_ABLATION_SCORE:
                    causal_contrasts.append(
                        self._sampling_score_contrast(weights, selected)
                    )
                new_keep[layer_idx, selected] = False
        elif self.random_scope == PER_LAYER_RANDOM_SCOPE:
            # Every actor worker reaches refresh with the same mask version.  A CPU
            # generator therefore produces identical masks without communication,
            # while adding mask_version resamples a new mask at every refresh.
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.random_seed + self.mask_version)
            for layer_idx in range(self.num_layers):
                random_idx = torch.randperm(self.intermediate_size, generator=generator)[:masked_per_layer]
                new_keep[layer_idx, random_idx] = False
        else:
            # Sample from the flattened (layer, channel) population.  The global
            # count is exact, while individual layers are intentionally allowed to
            # receive different numbers of masked channels.
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.random_seed + self.mask_version)
            masked_global = max(1, int(round(self.total_channels * self.mask_ratio)))
            random_idx = torch.randperm(self.total_channels, generator=generator)[:masked_global]
            new_keep.view(-1)[random_idx] = False
        new_masked = ~new_keep
        overlap = int((old_masked & new_masked).sum().item())
        current = int(new_masked.sum().item())
        new_unique = int((new_masked & ~self.ever_masked).sum().item())

        self.keep_mask.copy_(new_keep)
        if causal_contrasts:
            self.causal_assignment_contrast.copy_(torch.stack(causal_contrasts))
            self.causal_assignment_contrast_initialized = True
        self.ever_masked.logical_or_(new_masked)
        self.mask_version += 1
        self.cumulative_mask_assignments += current
        self._refresh_active_buffers()
        select_elapsed = time.perf_counter() - select_started

        # Match mlp_channel_rarity: score against the previous baseline first,
        # then update the baseline so the current batch cannot normalize itself.
        if batch_activation_cpu is not None:
            if self.activation_ema_initialized:
                self.activation_ema.mul_(self.activation_ema_beta).add_(
                    batch_activation_cpu,
                    alpha=1.0 - self.activation_ema_beta,
                )
            else:
                self.activation_ema.copy_(batch_activation_cpu)
                self.activation_ema_initialized = True

        self._activation_level_sum.clear()
        self._gradient_activation_sum.clear()
        self._activation_sample_count = None
        self._activation_token_count = None

        metrics = self.metrics()
        metrics.update(
            {
                "mlp_mask/new_unique_channels": float(new_unique),
                "mlp_mask/overlap_with_previous": float(overlap),
                "mlp_mask/turnover_fraction": float(
                    0.0 if current == 0 else 1.0 - overlap / current
                ),
            }
        )
        if weighted_selected_ranks:
            selected_ranks = torch.cat(weighted_selected_ranks)
            metrics.update(
                {
                    "mlp_mask/weighted_selected_rank_mean": float(selected_ranks.mean().item()),
                    "mlp_mask/weighted_selected_top_1pct_fraction": float(
                        (selected_ranks >= 0.99).to(dtype=torch.float32).mean().item()
                    ),
                    "mlp_mask/soft_top_used_score": 1.0,
                }
            )
        elif self.uses_soft_top:
            metrics["mlp_mask/soft_top_used_score"] = 0.0
        # Retain the legacy metric for dashboards that compare old runs.
        if self.uses_soft_top and self.score_method == RELATIVE_ACTIVATION_SCORE:
            metrics["mlp_mask/weighted_used_relative_activation"] = float(
                bool(weighted_selected_ranks)
            )
        metrics.update(score_metrics)
        if self.needs_forward_score:
            metrics.update(
                {
                    "mlp_activation/ema_mean": float(self.activation_ema.mean().item()),
                    "mlp_activation/ema_max": float(self.activation_ema.max().item()),
                    "mlp_activation/ema_min": float(self.activation_ema.min().item()),
                    "mlp_relative_activation/mean": float(
                        self.relative_activation_score.mean().item()
                    ),
                    "mlp_relative_activation/max": float(
                        self.relative_activation_score.max().item()
                    ),
                    "mlp_relative_activation/min": float(
                        self.relative_activation_score.min().item()
                    ),
                }
            )
        if self.selection_score_initialized:
            metrics.update(
                {
                    "mlp_score/ema_mean": float(self.selection_score.mean().item()),
                    "mlp_score/ema_max": float(self.selection_score.max().item()),
                    "mlp_score/ema_min": float(self.selection_score.min().item()),
                }
            )
        timings = {
            "mlp_activation_accumulate_cpu": self._activation_accumulate_cpu_s,
            "mlp_activation_reduce": reduce_elapsed,
            "mlp_mask_select": select_elapsed,
            "mlp_mask_refresh": time.perf_counter() - refresh_started,
        }
        self._activation_accumulate_cpu_s = 0.0
        return MaskRefreshResult(metrics=metrics, timings=timings)

    def metrics(self) -> dict[str, float]:
        current = self.current_masked_channels
        ever = self.ever_masked_channels
        current_per_layer = (~self.keep_mask).sum(dim=-1).to(dtype=torch.float32)
        ever_per_layer = self.ever_masked.sum(dim=-1).to(dtype=torch.float32)
        return {
            "mlp_mask/version": float(self.mask_version),
            "mlp_mask/initialized": float(current > 0),
            "mlp_mask/selection_is_relative_activation": float(
                self.selection_strategy == TOP_RELATIVE_ACTIVATION_SELECTION
            ),
            "mlp_mask/selection_is_random": float(self.selection_strategy == RANDOM_SELECTION),
            "mlp_mask/selection_is_weighted_random": float(
                self.selection_strategy == WEIGHTED_RANDOM_SELECTION
            ),
            "mlp_mask/selection_is_soft_top": float(self.uses_soft_top),
            "mlp_mask/random_scope_is_global": float(self.random_scope == GLOBAL_RANDOM_SCOPE),
            "mlp_mask/random_seed": float(self.random_seed),
            "mlp_mask/weighted_max_ratio": float(self.weighted_max_ratio),
            "mlp_mask/weighted_rank_power": float(self.weighted_rank_power),
            "mlp_activation/ema_beta": float(self.activation_ema_beta),
            "mlp_activation/relative_epsilon": float(self.relative_activation_epsilon),
            "mlp_activation/ema_initialized": float(self.activation_ema_initialized),
            "mlp_relative_activation/initialized": float(
                self.relative_activation_initialized
            ),
            "mlp_score/is_relative_activation": float(
                self.score_method == RELATIVE_ACTIVATION_SCORE
            ),
            "mlp_score/is_output_contribution": float(
                self.score_method == OUTPUT_CONTRIBUTION_SCORE
            ),
            "mlp_score/is_gradient_activation": float(
                self.score_method == GRADIENT_ACTIVATION_SCORE
            ),
            "mlp_score/is_causal_ablation": float(
                self.score_method == CAUSAL_ABLATION_SCORE
            ),
            "mlp_score/ema_beta": float(self.score_ema_beta),
            "mlp_score/initialized": float(self.selection_score_initialized),
            "mlp_causal/group_reward_gap_ema": float(self.causal_reward_gap_ema),
            "mlp_causal/group_reward_gap_initialized": float(
                self.causal_reward_gap_initialized
            ),
            "mlp_causal/observations": float(self.causal_observations),
            "mlp_causal/assignment_contrast_initialized": float(
                self.causal_assignment_contrast_initialized
            ),
            "mlp_causal/last_group_reward_gap": float(self._last_causal_reward_gap),
            "mlp_causal/last_group_reward_gap_residual": float(
                self._last_causal_residual
            ),
            "mlp_causal/estimator_is_randomized_group": float(
                self.score_method == CAUSAL_ABLATION_SCORE
            ),
            "mlp_causal/is_exact_single_channel_ablation": 0.0,
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
            "checkpoint_format": _CHECKPOINT_FORMAT,
            # Keep the v1 marker so old analysis scripts still recognize the
            # original relative-activation fields in a v2 checkpoint.
            "score_type": _LEGACY_ACTIVATION_SCORE_TYPE,
            "score_method": self.score_method,
            "score_ema_beta": self.score_ema_beta,
            "num_layers": self.num_layers,
            "intermediate_size": self.intermediate_size,
            "mask_ratio": self.mask_ratio,
            "activation_ema_beta": self.activation_ema_beta,
            "relative_activation_epsilon": self.relative_activation_epsilon,
            "selection_strategy": self.selection_strategy,
            "random_seed": self.random_seed,
            "random_scope": self.random_scope,
            "weighted_max_ratio": self.weighted_max_ratio,
            "weighted_rank_power": self.weighted_rank_power,
            "keep_mask": self.keep_mask.cpu(),
            "ever_masked": self.ever_masked.cpu(),
            "activation_ema": self.activation_ema.cpu(),
            "relative_activation_score": self.relative_activation_score.cpu(),
            "selection_score": self.selection_score.cpu(),
            "activation_ema_initialized": self.activation_ema_initialized,
            "relative_activation_initialized": self.relative_activation_initialized,
            "selection_score_initialized": self.selection_score_initialized,
            "causal_reward_gap_ema": self.causal_reward_gap_ema,
            "causal_reward_gap_initialized": self.causal_reward_gap_initialized,
            "causal_observations": self.causal_observations,
            "last_causal_reward_gap": self._last_causal_reward_gap,
            "last_causal_residual": self._last_causal_residual,
            "causal_assignment_contrast": self.causal_assignment_contrast.cpu(),
            "causal_assignment_contrast_initialized": (
                self.causal_assignment_contrast_initialized
            ),
            "mask_version": self.mask_version,
            "cumulative_mask_assignments": self.cumulative_mask_assignments,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        checkpoint_format = state.get("checkpoint_format")
        checkpoint_score_type = state.get("score_type")
        legacy_random_checkpoint = (
            checkpoint_format is None
            and checkpoint_score_type is None
            and self.selection_strategy == RANDOM_SELECTION
            and state.get("selection_strategy") == RANDOM_SELECTION
        )
        legacy_relative_checkpoint = (
            checkpoint_format is None
            and checkpoint_score_type == _LEGACY_ACTIVATION_SCORE_TYPE
        )
        if checkpoint_format not in {None, _CHECKPOINT_FORMAT}:
            raise ValueError(
                f"incompatible MLP mask checkpoint format {checkpoint_format!r}; "
                f"expected {_CHECKPOINT_FORMAT!r}"
            )
        if checkpoint_format is None and not (
            legacy_random_checkpoint or legacy_relative_checkpoint
        ):
            raise ValueError(
                "MLP mask checkpoint uses an incompatible score definition "
                f"{checkpoint_score_type!r}; expected {_LEGACY_ACTIVATION_SCORE_TYPE!r}. "
                "Start a new run instead of resuming a legacy saliency checkpoint."
            )
        if legacy_relative_checkpoint and self.score_method != RELATIVE_ACTIVATION_SCORE:
            raise ValueError(
                "a v1 relative-activation checkpoint cannot initialize "
                f"score_method={self.score_method!r}"
            )
        if checkpoint_format == _CHECKPOINT_FORMAT:
            checkpoint_score_method = str(state["score_method"])
            if checkpoint_score_method != self.score_method:
                raise ValueError(
                    f"checkpoint score_method={checkpoint_score_method!r} does not match "
                    f"controller score_method={self.score_method!r}"
                )
            checkpoint_score_ema_beta = float(state["score_ema_beta"])
            if checkpoint_score_ema_beta != self.score_ema_beta:
                raise ValueError(
                    f"checkpoint score_ema_beta={checkpoint_score_ema_beta} does not match "
                    f"controller score_ema_beta={self.score_ema_beta}"
                )
        checkpoint_strategy = str(state.get("selection_strategy", self.selection_strategy))
        if checkpoint_strategy != self.selection_strategy:
            raise ValueError(
                f"checkpoint selection_strategy={checkpoint_strategy!r} does not match "
                f"controller selection_strategy={self.selection_strategy!r}"
            )
        checkpoint_random_seed = int(state.get("random_seed", self.random_seed))
        if checkpoint_random_seed != self.random_seed:
            raise ValueError(
                f"checkpoint random_seed={checkpoint_random_seed} does not match "
                f"controller random_seed={self.random_seed}"
            )
        checkpoint_random_scope = str(state.get("random_scope", PER_LAYER_RANDOM_SCOPE))
        if checkpoint_random_scope != self.random_scope:
            raise ValueError(
                f"checkpoint random_scope={checkpoint_random_scope!r} does not match "
                f"controller random_scope={self.random_scope!r}"
            )
        checkpoint_weighted_max_ratio = float(
            state.get("weighted_max_ratio", self.weighted_max_ratio)
        )
        if checkpoint_weighted_max_ratio != self.weighted_max_ratio:
            raise ValueError(
                f"checkpoint weighted_max_ratio={checkpoint_weighted_max_ratio} does not match "
                f"controller weighted_max_ratio={self.weighted_max_ratio}"
            )
        checkpoint_weighted_rank_power = float(
            state.get("weighted_rank_power", self.weighted_rank_power)
        )
        if checkpoint_weighted_rank_power != self.weighted_rank_power:
            raise ValueError(
                f"checkpoint weighted_rank_power={checkpoint_weighted_rank_power} does not match "
                f"controller weighted_rank_power={self.weighted_rank_power}"
            )
        if not legacy_random_checkpoint:
            checkpoint_ema_beta = float(state["activation_ema_beta"])
            if checkpoint_ema_beta != self.activation_ema_beta:
                raise ValueError(
                    f"checkpoint activation_ema_beta={checkpoint_ema_beta} does not match "
                    f"controller activation_ema_beta={self.activation_ema_beta}"
                )
            checkpoint_epsilon = float(state["relative_activation_epsilon"])
            if checkpoint_epsilon != self.relative_activation_epsilon:
                raise ValueError(
                    f"checkpoint relative_activation_epsilon={checkpoint_epsilon} does not match "
                    f"controller relative_activation_epsilon={self.relative_activation_epsilon}"
                )
        expected = (self.num_layers, self.intermediate_size)
        keep_mask = torch.as_tensor(state["keep_mask"], dtype=torch.bool)
        if tuple(keep_mask.shape) != expected:
            raise ValueError(f"checkpoint mask shape {tuple(keep_mask.shape)} != {expected}")
        ever_masked = torch.as_tensor(state["ever_masked"], dtype=torch.bool)
        activation_ema = torch.as_tensor(
            state.get("activation_ema", torch.zeros(expected)), dtype=torch.float32
        )
        relative_activation_score = torch.as_tensor(
            state.get("relative_activation_score", torch.zeros(expected)),
            dtype=torch.float32,
        )
        selection_score_default = (
            relative_activation_score
            if legacy_relative_checkpoint
            else torch.zeros(expected)
        )
        selection_score = torch.as_tensor(
            state.get("selection_score", selection_score_default),
            dtype=torch.float32,
        )
        causal_assignment_contrast = torch.as_tensor(
            state.get("causal_assignment_contrast", torch.zeros(expected)),
            dtype=torch.float32,
        )
        if (
            tuple(ever_masked.shape) != expected
            or tuple(activation_ema.shape) != expected
            or tuple(relative_activation_score.shape) != expected
            or tuple(selection_score.shape) != expected
            or tuple(causal_assignment_contrast.shape) != expected
        ):
            raise ValueError(
                "checkpoint history/activation shapes must match controller shape "
                f"{expected}, got ever={tuple(ever_masked.shape)}, "
                f"ema={tuple(activation_ema.shape)}, "
                f"relative={tuple(relative_activation_score.shape)}, "
                f"selection={tuple(selection_score.shape)}, "
                f"causal_contrast={tuple(causal_assignment_contrast.shape)}"
            )
        self.keep_mask.copy_(keep_mask)
        self.ever_masked.copy_(ever_masked)
        self.activation_ema.copy_(activation_ema)
        self.relative_activation_score.copy_(relative_activation_score)
        self.selection_score.copy_(selection_score)
        self.causal_assignment_contrast.copy_(causal_assignment_contrast)
        self.activation_ema_initialized = bool(
            state.get("activation_ema_initialized", False if legacy_random_checkpoint else True)
        )
        self.relative_activation_initialized = bool(
            state.get("relative_activation_initialized", False)
        )
        self.selection_score_initialized = bool(
            state.get(
                "selection_score_initialized",
                legacy_relative_checkpoint and self.relative_activation_initialized,
            )
        )
        self.causal_reward_gap_ema = float(state.get("causal_reward_gap_ema", 0.0))
        self.causal_reward_gap_initialized = bool(
            state.get("causal_reward_gap_initialized", False)
        )
        self.causal_observations = int(state.get("causal_observations", 0))
        self._last_causal_reward_gap = float(state.get("last_causal_reward_gap", 0.0))
        self._last_causal_residual = float(state.get("last_causal_residual", 0.0))
        self.causal_assignment_contrast_initialized = bool(
            state.get("causal_assignment_contrast_initialized", False)
        )
        self.mask_version = int(state.get("mask_version", 0))
        self.cumulative_mask_assignments = int(state.get("cumulative_mask_assignments", 0))
        self._refresh_active_buffers()

    def copy_mask_state_from(self, other: "MLPChannelInterventionController") -> None:
        self.load_state_dict(other.state_dict())

    def _score_device(self) -> torch.device:
        if self._gradient_activation_sum:
            return next(iter(self._gradient_activation_sum.values())).device
        if self._activation_level_sum:
            return next(iter(self._activation_level_sum.values())).device
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
            activation = controller.apply(
                _layer_idx,
                activation,
                down_weight=this.down_proj.weight,
            )
            return this.down_proj(activation)

        module.forward = MethodType(forward, module)
        module._mlp_channel_intervention_patched = True
        module._mlp_channel_intervention_controller = controller
        patched.append(name)
        seen_layers.add(layer_idx)

    _validate_patched_layers(controller, seen_layers, backend="HF actor")
    return patched


_VLLM_ACTIVE_MASK_BUFFER = "_mlp_channel_active_mask"


def _install_vllm_active_buffer(
    module: torch.nn.Module,
    controller: MLPChannelInterventionController,
    layer_idx: int,
) -> torch.Tensor:
    """Create the vLLM mask before forward so Dynamo sees read-only module state."""
    existing = module._buffers.get(_VLLM_ACTIVE_MASK_BUFFER)
    if existing is None:
        if hasattr(module, _VLLM_ACTIVE_MASK_BUFFER):
            raise RuntimeError(
                f"vLLM MLP attribute {_VLLM_ACTIVE_MASK_BUFFER!r} exists but is not a registered buffer"
            )
        reference = getattr(getattr(module, "down_proj", None), "weight", None)
        if reference is None:
            reference = next(module.parameters(), None)
        if reference is None:
            raise RuntimeError("cannot allocate vLLM MLP mask: module has no parameter to infer dtype/device")
        local_width = controller.intermediate_size // controller.tp_size
        device = None if reference.device.type == "meta" else reference.device
        existing = torch.ones(local_width, device=device, dtype=reference.dtype)
        module.register_buffer(_VLLM_ACTIVE_MASK_BUFFER, existing, persistent=False)
    return controller.register_active_buffer(layer_idx, existing)


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
            _install_vllm_active_buffer(module, controller, layer_idx)
            patched.append(name)
            seen_layers.add(layer_idx)
            continue

        _install_vllm_active_buffer(module, controller, layer_idx)

        def forward(this, hidden_state, *args, _layer_idx=layer_idx, **kwargs):
            if args or kwargs:
                raise TypeError("patched vLLM dense MLP expects only hidden_state")
            gate_up = this.gate_up_proj(hidden_state)
            gate_up = gate_up[0] if isinstance(gate_up, tuple) else gate_up
            activation = this.act_fn(gate_up)
            activation = activation * getattr(this, _VLLM_ACTIVE_MASK_BUFFER)
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
            _install_vllm_active_buffer(this, controller, layer_idx)

        def patched_forward(this, hidden_state):
            gate_up = this.gate_up_proj(hidden_state)
            gate_up = gate_up[0] if isinstance(gate_up, tuple) else gate_up
            activation = this.act_fn(gate_up)
            activation = activation * getattr(this, _VLLM_ACTIVE_MASK_BUFFER)
            output = this.down_proj(activation)
            return output[0] if isinstance(output, tuple) else output

        cls.__init__ = patched_init
        cls.forward = patched_forward
        cls._mlp_channel_intervention_controller = controller
        patched_names.append(cls.__name__)
    return patched_names


def _layer_index(module_name: str) -> int | None:
    # FSDP1 inserts this transparent module into every auto-wrapped decoder
    # layer, e.g. ``model.layers.0._fsdp_wrapped_module.mlp``.  Normalize the
    # traversal-only segment before matching the underlying model layout.
    normalized_name = ".".join(
        part for part in module_name.split(".") if part != "_fsdp_wrapped_module"
    )
    match = _LAYER_RE.search(normalized_name)
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
