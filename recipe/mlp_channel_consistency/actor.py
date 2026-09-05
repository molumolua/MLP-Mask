"""Actor integration for clean-to-hard-masked response-distribution KL."""

from __future__ import annotations

import time

import torch

from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor

from .batching import slice_model_inputs
from .diagnostics import SampledGradientTracker
from .intervention import MLPChannelConsistencyController
from .kl import (
    TeacherDistribution,
    build_teacher_distribution,
    forward_kl_sum,
    slice_teacher_rows,
)


class MLPChannelConsistencyActor(DataParallelPPOActor):
    """Add a second teacher-forced masked backward after each clean GRPO backward.

    ``kl_top_k > 0`` computes an exact KL after coarsening the vocabulary into the
    clean teacher's top-k tokens plus one aggregate tail bucket.  ``kl_top_k == 0``
    computes full categorical KL but retains a full detached teacher distribution
    and is therefore intended only for short-sequence or high-memory experiments.
    """

    consistency_controller: MLPChannelConsistencyController
    consistency_auxiliary_enabled: bool
    consistency_kl_coef: float
    consistency_kl_top_k: int
    consistency_micro_batch_size_per_gpu: int
    consistency_gradient_tracker: SampledGradientTracker

    def update_policy(self, data: DataProto):
        controller = self.consistency_controller
        gradient_tracker = self.consistency_gradient_tracker
        controller.set_clean()
        gradient_tracker.start_update()
        self._consistency_update_active = self.consistency_auxiliary_enabled
        self._inside_consistency_forward = False
        self._teacher_distribution: TeacherDistribution | None = None
        self._teacher_row_token_counts: tuple[int, ...] | None = None
        self._student_kl_sum: torch.Tensor | None = None
        completed = False
        try:
            metrics = super().update_policy(data)
            completed = True
        finally:
            self._response_logits_callback = None
            self._teacher_distribution = None
            self._teacher_row_token_counts = None
            self._student_kl_sum = None
            self._inside_consistency_forward = False
            self._consistency_update_active = False
            controller.set_clean()
            if not completed:
                gradient_tracker.cancel_update()

        gradient_metrics = gradient_tracker.finish_update()
        for name, value in gradient_metrics.items():
            metrics.setdefault(name, []).append(value)

        # The existing per-micro-batch values are contribution-scaled; summing
        # them reconstructs the two complete objectives for this optimizer step.
        main_pg_loss = sum(float(value) for value in metrics.get("actor/pg_loss", []))
        weighted_auxiliary_loss = sum(
            float(value) for value in metrics.get("mlp_consistency/weighted_kl", [])
        )
        metrics.setdefault("mlp_consistency/main_pg_loss_step", []).append(main_pg_loss)
        metrics.setdefault("mlp_consistency/weighted_kl_step", []).append(
            weighted_auxiliary_loss
        )
        metrics.setdefault("mlp_consistency/aux_to_main_loss_abs_ratio", []).append(
            abs(weighted_auxiliary_loss) / max(abs(main_pg_loss), 1.0e-12)
        )

        metrics.setdefault("mlp_consistency/kl_coef", []).append(
            float(self.consistency_kl_coef)
        )
        metrics.setdefault("mlp_consistency/auxiliary_enabled", []).append(
            float(self.consistency_auxiliary_enabled)
        )
        metrics.setdefault("mlp_consistency/kl_top_k", []).append(
            float(self.consistency_kl_top_k)
        )
        metrics.setdefault("mlp_consistency/micro_batch_size_per_gpu", []).append(
            float(self.consistency_micro_batch_size_per_gpu)
        )
        return metrics

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False):
        if not getattr(self, "_consistency_update_active", False) or getattr(
            self, "_inside_consistency_forward", False
        ):
            return super()._forward_micro_batch(
                micro_batch,
                temperature=temperature,
                calculate_entropy=calculate_entropy,
            )

        self.consistency_controller.set_clean()
        self._teacher_distribution = None
        self._teacher_row_token_counts = tuple(
            int(value)
            for value in micro_batch["response_mask"]
            .sum(dim=-1)
            .detach()
            .cpu()
            .tolist()
        )
        self._response_logits_callback = self._capture_teacher_distribution
        try:
            result = super()._forward_micro_batch(
                micro_batch,
                temperature=temperature,
                calculate_entropy=calculate_entropy,
            )
        finally:
            self._response_logits_callback = None
        if self._teacher_distribution is None:
            raise RuntimeError("clean actor forward did not capture a teacher distribution")
        return result

    def _capture_teacher_distribution(
        self, logits: torch.Tensor, response_token_mask: torch.Tensor
    ) -> None:
        self._teacher_distribution = build_teacher_distribution(
            logits,
            response_token_mask,
            top_k=self.consistency_kl_top_k,
            row_token_counts=self._teacher_row_token_counts,
        )

    def _capture_student_kl(
        self, logits: torch.Tensor, response_token_mask: torch.Tensor
    ) -> None:
        teacher = self._teacher_distribution
        if teacher is None:
            raise RuntimeError("masked forward started without a clean teacher distribution")
        self._student_kl_sum = forward_kl_sum(
            teacher,
            logits,
            response_token_mask,
        )

    def _backward_auxiliary_loss(
        self,
        *,
        model_inputs: dict,
        temperature: float,
        aggregation_scale: float,
    ):
        gradient_tracker = self.consistency_gradient_tracker
        # loss.backward() has returned, so FSDP/FSDP2 has already reduced and
        # resharded the clean gradients. Sample that stable local representation.
        gradient_tracker.capture_main_gradient()
        if not self.consistency_auxiliary_enabled:
            # Core actor calls this hook after every clean backward. Capture the
            # unchanged cumulative gradient a second time, which records an exact
            # zero auxiliary vector without doing a teacher or masked forward.
            gradient_tracker.capture_auxiliary_gradient()
            return {
                "mlp_consistency/kl": 0.0,
                "mlp_consistency/weighted_kl": 0.0,
                "mlp_consistency/response_tokens": float(
                    model_inputs["response_mask"].detach().sum().item()
                ),
                "mlp_consistency/micro_batches": 0.0,
                "timing_s/mlp_consistency_forward_backward": 0.0,
            }

        full_teacher = self._teacher_distribution
        if full_teacher is None:
            raise RuntimeError("consistency backward is missing its clean teacher")
        started = time.perf_counter()
        self._inside_consistency_forward = True
        self.consistency_controller.set_masked()
        self._response_logits_callback = self._capture_student_kl
        batch_size = int(model_inputs["responses"].shape[0])
        micro_batch_size = int(self.consistency_micro_batch_size_per_gpu)
        detached_kl_sum = 0.0
        sub_batch_count = 0
        try:
            for start in range(0, batch_size, micro_batch_size):
                end = min(start + micro_batch_size, batch_size)
                if sum(full_teacher.row_token_counts[start:end]) == 0:
                    continue
                self._teacher_distribution = slice_teacher_rows(full_teacher, start, end)
                self._student_kl_sum = None
                sub_inputs = slice_model_inputs(model_inputs, start, end, batch_size)

                # Reuse the sampled response verbatim. Splitting only changes the
                # masked teacher-forced memory schedule, not the KL objective.
                super()._forward_micro_batch(
                    sub_inputs,
                    temperature=temperature,
                    calculate_entropy=False,
                )
                if self._student_kl_sum is None:
                    raise RuntimeError("masked actor forward did not produce a KL loss")
                detached_kl_sum += float(self._student_kl_sum.detach().item())
                weighted_sub_kl = (
                    self._student_kl_sum
                    / float(full_teacher.token_count)
                    * float(self.consistency_kl_coef)
                    * float(aggregation_scale)
                )
                weighted_sub_kl.backward()
                self._student_kl_sum = None
                sub_batch_count += 1
        finally:
            self._response_logits_callback = None
            self._inside_consistency_forward = False
            self.consistency_controller.set_clean()

        gradient_tracker.capture_auxiliary_gradient()

        raw_kl = detached_kl_sum / float(full_teacher.token_count)
        weighted_kl = (
            raw_kl * float(self.consistency_kl_coef) * float(aggregation_scale)
        )
        metrics = {
            "mlp_consistency/kl": float(raw_kl),
            "mlp_consistency/weighted_kl": float(weighted_kl),
            "mlp_consistency/response_tokens": float(full_teacher.token_count),
            "mlp_consistency/micro_batches": float(sub_batch_count),
            "timing_s/mlp_consistency_forward_backward": time.perf_counter()
            - started,
        }
        self._teacher_distribution = None
        self._student_kl_sum = None
        return metrics
