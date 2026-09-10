"""Route-conditioned PPO plus symmetric KL on both routes' responses."""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.distributed as dist

from verl import DataProto

from .batching import slice_model_inputs
from .diagnostics import _all_reduce_sum, _collective_device
from .intervention import CLEAN_ROUTE, MASK_A_ROUTE, MASK_B_ROUTE, TRAINING_ROUTES
from .kl import build_distribution, select_response_logits, symmetric_kl_sum


def prepare_loss_metadata(data: DataProto) -> tuple[float, int]:
    """Global token means despite unequal response lengths on different ranks.

    Main = 0.5 mean_A(PPO) + 0.5 mean_B(PPO). Auxiliary = token mean over
    all A/B trajectories. FSDP averages gradients over ranks, hence the DP factor.
    GRPO advantage still uses the original shared prompt uid (all 16 answers).
    """
    routes = np.asarray(data.non_tensor_batch["route_id"], dtype=object)
    if set(routes) != set(TRAINING_ROUTES):
        raise ValueError("every actor shard must contain both training routes")
    tokens = data.batch["response_mask"].sum(-1).detach().cpu().numpy()
    local = [float(tokens[routes == route].sum()) for route in TRAINING_ROUTES]
    global_counts = _all_reduce_sum(local, data.batch["response_mask"].device)
    if min(global_counts) <= 0:
        raise ValueError("each route must have valid response tokens")
    world = dist.get_world_size() if dist.is_initialized() else 1
    weights = np.empty(len(data), dtype=np.float32)
    for i, route in enumerate(TRAINING_ROUTES):
        weights[routes == route] = world * local[i] / global_counts[i]
    data.non_tensor_batch["loss_multiplier"] = weights
    data.non_tensor_batch["loss_group_id"] = routes.copy()
    data.non_tensor_batch["loss_group_normalizer"] = np.full(len(data), 2, dtype=np.int64)
    return world / sum(global_counts), world


def synchronized_auxiliary_slots(batch_size: int, micro_batch_size: int,
                                 device: torch.device) -> int:
    """All ranks execute equal FSDP forward/backward counts with dynamic batches."""
    slots = math.ceil(batch_size / micro_batch_size)
    count = torch.tensor(slots, dtype=torch.int64, device=_collective_device(device))
    if dist.is_initialized():
        dist.all_reduce(count, op=dist.ReduceOp.MAX)
    return int(count.item())


class RDropActorMixin:
    """Mixed into DataParallelPPOActor by the worker; tensor logic is CPU-testable."""
    def configure_rdrop(self, config, controller, gradient_tracker):
        self.rdrop_config = config
        self.intervention_controller = controller
        self.gradient_tracker = gradient_tracker
        self.kl_coef = float(config.kl_coef)
        self.auxiliary_enabled = bool(config.auxiliary_enabled) and self.kl_coef > 0
        self.kl_micro_batch_size = int(config.micro_batch_size_per_gpu)
        self.kl_token_chunk_size = int(config.kl_token_chunk_size)
        self.kl_top_k = int(config.kl_top_k)

    def update_policy(self, data: DataProto):
        self._aux_token_weight, world = prepare_loss_metadata(data)
        self._aux_symmetric_sum = 0.0
        self._aux_response_tokens = 0
        self._aux_response_rows = 0
        self._aux_padding_slots = 0
        self._aux_seconds = 0.0
        self.gradient_tracker.start_update()
        try:
            metrics = super().update_policy(data)
            diagnostics = self.gradient_tracker.finish_update()
        except Exception:
            self.gradient_tracker.cancel_update()
            raise
        finally:
            self._response_logits_callback = None
            self._target_log_probs = None
            self._next_target_log_probs = None
            self._kl_loss = None
            self.intervention_controller.set_route(CLEAN_ROUTE)

        # Sum contributions before cross-rank averaging. Ordinary micro-batch
        # metric means are not the complete objective when lengths differ.
        main = sum(float(v) for v in metrics.get("actor/pg_loss", []))
        raw = self._aux_symmetric_sum * self._aux_token_weight
        main, raw, tokens, rows, padding = _all_reduce_sum(
            [main, raw, self._aux_response_tokens, self._aux_response_rows, self._aux_padding_slots],
            data.batch["response_mask"].device,
        )
        main, raw = main / world, raw / world
        weighted = self.kl_coef * raw
        wall_time = torch.tensor(self._aux_seconds, dtype=torch.float64,
                                 device=_collective_device(data.batch["response_mask"].device))
        if dist.is_initialized():
            dist.all_reduce(wall_time, op=dist.ReduceOp.MAX)
        diagnostics.update({
            "mlp_rdrop/main_pg_loss_step": main,
            "mlp_rdrop/kl": raw,
            "mlp_rdrop/weighted_kl_step": weighted,
            "mlp_rdrop/total_loss_step": main + weighted,
            "mlp_rdrop/aux_to_main_loss_abs_ratio": abs(weighted) / max(abs(main), 1e-12),
            "mlp_rdrop/loss_ratio_denominator_near_zero": float(abs(main) < 1e-8),
            "mlp_rdrop/kl_coef": self.kl_coef,
            "mlp_rdrop/auxiliary_enabled": float(self.auxiliary_enabled),
            "mlp_rdrop/response_tokens": tokens,
            "mlp_rdrop/aligned_response_rows": rows,
            "mlp_rdrop/auxiliary_padding_slots": padding,
            "mlp_rdrop/full_vocabulary_kl": float(self.kl_top_k == 0),
            "mlp_rdrop/kl_top_k": float(self.kl_top_k),
            "timing_s/mlp_rdrop_auxiliary_step": float(wall_time.item()),
        })
        diagnostics.update(self.intervention_controller.metrics())
        for name, value in diagnostics.items():
            metrics[name] = [value]
        return metrics

    def _capture_reference(self, logits, mask):
        selected = select_response_logits(logits, mask)
        self._target_log_probs = build_distribution(selected, top_k=self.kl_top_k, chunk_size=self.kl_token_chunk_size)

    def _capture_partial_loss(self, logits, mask):
        selected = select_response_logits(logits, mask)
        self._kl_loss = symmetric_kl_sum(
            selected, self._target_log_probs.log_probs, token_ids=self._target_log_probs.token_ids,
            chunk_size=self.kl_token_chunk_size
        )
        if self._save_next_target:
            self._next_target_log_probs = build_distribution(
                selected, top_k=0, token_ids=self._target_log_probs.token_ids,
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
            return {"timing_s/mlp_rdrop_auxiliary_micro_batch": 0.0}

        started = time.perf_counter()
        original_route = self.intervention_controller.route
        batch_size = int(model_inputs["responses"].shape[0])
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
                self._target_log_probs = None
                self._next_target_log_probs = None
                self._kl_loss = None
                with torch.no_grad():
                    self._run_auxiliary_forward(inputs, route=MASK_A_ROUTE,
                                                callback=self._capture_reference, temperature=temperature)
                if self._target_log_probs is None:
                    raise RuntimeError("A forward did not capture its response distribution")

                self._save_next_target = True
                self._run_auxiliary_forward(inputs, route=MASK_B_ROUTE,
                                            callback=self._capture_partial_loss, temperature=temperature)
                if self._kl_loss is None or self._next_target_log_probs is None:
                    raise RuntimeError("B forward did not produce symmetric KL")
                if not dummy:
                    self._aux_symmetric_sum += float(self._kl_loss.detach())
                    self._aux_response_tokens += int(self._target_log_probs.log_probs.shape[0])
                    self._aux_response_rows += end - start
                (self._kl_loss * weight).backward()
                self._kl_loss = None

                self._target_log_probs = self._next_target_log_probs
                self._next_target_log_probs = None
                self._save_next_target = False
                self._run_auxiliary_forward(inputs, route=MASK_A_ROUTE,
                                            callback=self._capture_partial_loss, temperature=temperature)
                if self._kl_loss is None:
                    raise RuntimeError("A replay did not produce symmetric KL")
                (self._kl_loss * weight).backward()
                # The scalar objective is counted only once. These two backwards
                # add its two partial derivatives, NOT two copies of the loss.
                self._kl_loss = self._target_log_probs = None
        finally:
            self._response_logits_callback = None
            self._kl_loss = self._target_log_probs = self._next_target_log_probs = None
            self.intervention_controller.end_batch()
            self.intervention_controller.set_route(original_route)
        self.gradient_tracker.capture_auxiliary_gradient()
        elapsed = time.perf_counter() - started
        self._aux_seconds += elapsed
        return {"timing_s/mlp_rdrop_auxiliary_micro_batch": elapsed}
