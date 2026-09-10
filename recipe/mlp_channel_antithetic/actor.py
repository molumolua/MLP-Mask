"""Route-conditioned PPO plus cross-route, detached-teacher forward KL."""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.distributed as dist

from verl import DataProto

from .batching import slice_model_inputs
from .diagnostics import _all_reduce_sum, _collective_device
from .intervention import NEUTRAL_ROUTE, POSITIVE_ROUTE, NEGATIVE_ROUTE, TRAINING_ROUTES
from .kl import build_distribution, forward_kl_sum, select_response_logits, slice_distribution_rows


def prepare_auxiliary_metadata(data: DataProto) -> tuple[float, int]:
    """Normalize cross KL globally without changing antithetic's PPO metadata.

    The existing PPO objective and shared prompt uid are left intact. FSDP
    averages gradients over ranks, so KL token sums need the world-size factor.
    """
    routes = np.asarray(data.non_tensor_batch["route_id"], dtype=object)
    if set(routes) != set(TRAINING_ROUTES):
        raise ValueError("every actor shard must contain both training routes")
    local_tokens = float(data.batch["response_mask"].sum())
    global_tokens = _all_reduce_sum([local_tokens], data.batch["response_mask"].device)[0]
    if global_tokens <= 0:
        raise ValueError("cross-route KL requires valid response tokens")
    world = dist.get_world_size() if dist.is_initialized() else 1
    return world / global_tokens, world


def synchronized_auxiliary_slots(batch_size: int, micro_batch_size: int,
                                 device: torch.device) -> int:
    """All ranks execute equal FSDP forward/backward counts with dynamic batches."""
    slots = math.ceil(batch_size / micro_batch_size)
    count = torch.tensor(slots, dtype=torch.int64, device=_collective_device(device))
    if dist.is_initialized():
        dist.all_reduce(count, op=dist.ReduceOp.MAX)
    return int(count.item())


class AntitheticCrossKLActorMixin:
    """Mixed into DataParallelPPOActor by the worker; tensor logic is CPU-testable."""
    def configure_cross_kl(self, config, controller, gradient_tracker):
        self.cross_kl_config = config
        self.intervention_controller = controller
        self.gradient_tracker = gradient_tracker
        self.kl_coef = float(config.kl_coef)
        self.auxiliary_enabled = bool(config.enabled) and self.kl_coef > 0
        self.kl_micro_batch_size = int(config.micro_batch_size_per_gpu)
        self.kl_token_chunk_size = int(config.kl_token_chunk_size)
        self.kl_top_k = int(config.kl_top_k)
        self._cross_kl_update_active = False

    def _clear_teacher(self):
        self._teacher_distribution = None
        self._teacher_row_token_counts = None
        self._teacher_route = None
        self._target_distribution = None
        self._kl_loss = None

    def update_policy(self, data: DataProto):
        self._aux_token_weight, world = prepare_auxiliary_metadata(data)
        # Each source contributes a single directed KL on its own trajectories.
        self._aux_by_source = {route: [0.0, 0, 0] for route in TRAINING_ROUTES}
        self._aux_padding_slots = 0
        self._aux_forward_calls = 0
        self._aux_backward_calls = 0
        self._aux_seconds = 0.0
        self._teacher_capture_seconds = 0.0
        self._teacher_cache_peak_tokens = 0
        self._cross_kl_update_active = self.auxiliary_enabled
        self._clear_teacher()
        self.gradient_tracker.start_update()
        try:
            metrics = super().update_policy(data)
            diagnostics = self.gradient_tracker.finish_update()
        except Exception:
            self.gradient_tracker.cancel_update()
            raise
        finally:
            self._response_logits_callback = None
            self._cross_kl_update_active = False
            self._clear_teacher()
            self.intervention_controller.set_route(NEUTRAL_ROUTE)

        # Sum contributions before cross-rank averaging. Ordinary micro-batch
        # metric means are not the complete objective when lengths differ.
        main = sum(float(v) for v in metrics.get("actor/pg_loss", []))
        stats = _all_reduce_sum(
            [main, self._aux_padding_slots, self._aux_forward_calls, self._aux_backward_calls,
             *self._aux_by_source[POSITIVE_ROUTE], *self._aux_by_source[NEGATIVE_ROUTE]],
            data.batch["response_mask"].device,
        )
        main, padding, forward_calls, backward_calls, a_sum, a_tokens, a_rows, b_sum, b_tokens, b_rows = stats
        main /= world
        tokens, rows = a_tokens + b_tokens, a_rows + b_rows
        raw = (a_sum + b_sum) * self._aux_token_weight / world
        weighted = self.kl_coef * raw
        maxima = torch.tensor([self._aux_seconds, self._teacher_capture_seconds, self._teacher_cache_peak_tokens],
                              dtype=torch.float64, device=_collective_device(data.batch["response_mask"].device))
        if dist.is_initialized():
            dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
        auxiliary_time, teacher_time, cache_tokens = maxima.cpu().tolist()
        diagnostics.update({
            "mlp_antithetic/cross_kl/main_pg_loss_step": main,
            "mlp_antithetic/cross_kl/kl": raw,
            "mlp_antithetic/cross_kl/weighted_kl_step": weighted,
            "mlp_antithetic/cross_kl/total_loss_step": main + weighted,
            "mlp_antithetic/cross_kl/aux_to_main_loss_abs_ratio": abs(weighted) / max(abs(main), 1e-12),
            "mlp_antithetic/cross_kl/loss_ratio_denominator_near_zero": float(abs(main) < 1e-8),
            "mlp_antithetic/cross_kl/kl_coef": self.kl_coef,
            "mlp_antithetic/cross_kl/auxiliary_enabled": float(self.auxiliary_enabled),
            "mlp_antithetic/cross_kl/response_tokens": tokens,
            "mlp_antithetic/cross_kl/aligned_response_rows": rows,
            "mlp_antithetic/cross_kl/auxiliary_padding_slots": padding,
            "mlp_antithetic/cross_kl/auxiliary_forward_calls": forward_calls,
            "mlp_antithetic/cross_kl/auxiliary_backward_calls": backward_calls,
            "mlp_antithetic/cross_kl/positive_to_negative_kl": a_sum / a_tokens if a_tokens else 0.0,
            "mlp_antithetic/cross_kl/negative_to_positive_kl": b_sum / b_tokens if b_tokens else 0.0,
            "mlp_antithetic/cross_kl/positive_to_negative_aligned_rows": a_rows,
            "mlp_antithetic/cross_kl/negative_to_positive_aligned_rows": b_rows,
            "mlp_antithetic/cross_kl/positive_to_negative_response_tokens": a_tokens,
            "mlp_antithetic/cross_kl/negative_to_positive_response_tokens": b_tokens,
            "mlp_antithetic/cross_kl/cross_route_forward_kl": 1.0,
            "mlp_antithetic/cross_kl/teacher_cache_peak_tokens": cache_tokens,
            "mlp_antithetic/cross_kl/full_vocabulary_kl": float(self.kl_top_k == 0),
            "mlp_antithetic/cross_kl/kl_top_k": float(self.kl_top_k),
            "timing_s/mlp_antithetic_cross_kl_auxiliary_step": auxiliary_time,
            "timing_s/mlp_antithetic_cross_kl_teacher_capture_step": teacher_time,
        })
        diagnostics.update(self.intervention_controller.metrics())
        for name, value in diagnostics.items():
            metrics[name] = [value]
        return metrics

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False):
        if not getattr(self, "_cross_kl_update_active", False):
            return super()._forward_micro_batch(micro_batch, temperature, calculate_entropy)

        routes = {str(route) for route in micro_batch["route_id"]}
        if len(routes) != 1 or not routes.issubset(TRAINING_ROUTES):
            raise RuntimeError("teacher capture requires a single source route")
        source = routes.pop()
        if self.intervention_controller.route != source:
            raise RuntimeError("PPO teacher route does not match the rollout source")
        self._clear_teacher()
        self._teacher_route = source
        self._teacher_row_token_counts = tuple(
            int(value) for value in micro_batch["response_mask"].sum(-1).detach().cpu().tolist()
        )
        self._response_logits_callback = self._capture_teacher
        try:
            result = super()._forward_micro_batch(micro_batch, temperature, calculate_entropy)
        finally:
            self._response_logits_callback = None
        if self._teacher_distribution is None:
            raise RuntimeError("PPO forward did not capture its teacher distribution")
        return result

    @torch.no_grad()
    def _capture_teacher(self, logits, mask):
        started = time.perf_counter()
        selected = select_response_logits(logits, mask)
        if selected.shape[0] != sum(self._teacher_row_token_counts):
            raise RuntimeError("teacher response layout does not match PPO row token counts")
        self._teacher_distribution = build_distribution(
            selected, top_k=self.kl_top_k, chunk_size=self.kl_token_chunk_size
        )
        self._teacher_cache_peak_tokens = max(self._teacher_cache_peak_tokens, selected.shape[0])
        self._teacher_capture_seconds += time.perf_counter() - started

    def _capture_student_loss(self, logits, mask):
        selected = select_response_logits(logits, mask)
        self._kl_loss = forward_kl_sum(
            selected, self._target_distribution.log_probs, token_ids=self._target_distribution.token_ids,
            chunk_size=self.kl_token_chunk_size
        )

    def _run_auxiliary_forward(self, inputs, *, route, callback, temperature):
        self.intervention_controller.set_route(route)
        self._response_logits_callback = callback
        try:
            super()._forward_micro_batch(inputs, temperature=temperature, calculate_entropy=False)
        finally:
            self._response_logits_callback = None

    def _backward_auxiliary_loss(self, *, model_inputs: dict, temperature: float,
                                 aggregation_scale: float):
        self.gradient_tracker.capture_main_gradient()
        if not self.auxiliary_enabled:
            self.gradient_tracker.capture_auxiliary_gradient()
            return {"timing_s/mlp_antithetic_cross_kl_auxiliary_micro_batch": 0.0}

        started = time.perf_counter()
        original_route = self.intervention_controller.route
        batch_size = int(model_inputs["responses"].shape[0])
        full_teacher = self._teacher_distribution
        counts = self._teacher_row_token_counts
        if full_teacher is None or counts is None or len(counts) != batch_size:
            raise RuntimeError("cross-route KL is missing its PPO teacher cache")
        if self._teacher_route != original_route:
            raise RuntimeError("cross-route KL teacher/source route mismatch")
        student_route = NEGATIVE_ROUTE if original_route == POSITIVE_ROUTE else POSITIVE_ROUTE
        slots = synchronized_auxiliary_slots(batch_size, self.kl_micro_batch_size,
                                             model_inputs["responses"].device)
        try:
            for slot in range(slots):
                start = slot * self.kl_micro_batch_size
                dummy = start >= batch_size
                end = min(start + self.kl_micro_batch_size, batch_size)
                # Zero-weight replay keeps all FSDP ranks in the same sequence
                # of collectives even when dynamic packing yields different bsz.
                if dummy:
                    start, end = 0, 1
                    self._aux_padding_slots += 1
                inputs = slice_model_inputs(model_inputs, start, end, batch_size)
                weight = 0.0 if dummy else self.kl_coef * self._aux_token_weight
                self._target_distribution = slice_distribution_rows(full_teacher, counts, start, end)
                self._kl_loss = None
                # The teacher is reused from the source PPO forward. Only the
                # opposite route is evaluated and differentiated for this KL.
                self._run_auxiliary_forward(inputs, route=student_route,
                                            callback=self._capture_student_loss, temperature=temperature)
                self._aux_forward_calls += 1
                if self._kl_loss is None:
                    raise RuntimeError("student forward did not produce cross-route KL")
                if not dummy:
                    stats = self._aux_by_source[original_route]
                    stats[0] += float(self._kl_loss.detach())
                    stats[1] += int(self._target_distribution.log_probs.shape[0])
                    stats[2] += end - start
                (self._kl_loss * weight).backward()
                self._aux_backward_calls += 1
                self._kl_loss = None
        finally:
            self._response_logits_callback = None
            self._clear_teacher()
            self.intervention_controller.end_batch()
            self.intervention_controller.set_route(original_route)
        self.gradient_tracker.capture_auxiliary_gradient()
        elapsed = time.perf_counter() - started
        self._aux_seconds += elapsed
        return {"timing_s/mlp_antithetic_cross_kl_auxiliary_micro_batch": elapsed}
