"""Actor integration for clean-to-hard-masked response-distribution KL."""

from __future__ import annotations

import time

import torch

from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor

from .intervention import MLPChannelConsistencyController
from .kl import TeacherDistribution, build_teacher_distribution, forward_kl_sum


class MLPChannelConsistencyActor(DataParallelPPOActor):
    """Add a second teacher-forced masked backward after each clean GRPO backward.

    ``kl_top_k > 0`` computes an exact KL after coarsening the vocabulary into the
    clean teacher's top-k tokens plus one aggregate tail bucket.  ``kl_top_k == 0``
    computes full categorical KL but retains a full detached teacher distribution
    and is therefore intended only for short-sequence or high-memory experiments.
    """

    consistency_controller: MLPChannelConsistencyController
    consistency_kl_coef: float
    consistency_kl_top_k: int

    def update_policy(self, data: DataProto):
        controller = self.consistency_controller
        controller.set_clean()
        self._consistency_update_active = True
        self._inside_consistency_forward = False
        self._teacher_distribution: TeacherDistribution | None = None
        self._student_kl_sum: torch.Tensor | None = None
        try:
            metrics = super().update_policy(data)
        finally:
            self._response_logits_callback = None
            self._teacher_distribution = None
            self._student_kl_sum = None
            self._inside_consistency_forward = False
            self._consistency_update_active = False
            controller.set_clean()

        metrics.setdefault("mlp_consistency/kl_coef", []).append(
            float(self.consistency_kl_coef)
        )
        metrics.setdefault("mlp_consistency/kl_top_k", []).append(
            float(self.consistency_kl_top_k)
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
        teacher = self._teacher_distribution
        if teacher is None:
            raise RuntimeError("consistency backward is missing its clean teacher")
        started = time.perf_counter()
        self._student_kl_sum = None
        self._inside_consistency_forward = True
        self.consistency_controller.set_masked()
        self._response_logits_callback = self._capture_student_kl
        try:
            # The sampled response is reused verbatim.  This is teacher forcing,
            # not a second autoregressive rollout.
            super()._forward_micro_batch(
                model_inputs,
                temperature=temperature,
                calculate_entropy=False,
            )
            if self._student_kl_sum is None:
                raise RuntimeError("masked actor forward did not produce a KL loss")
            raw_kl = self._student_kl_sum / float(teacher.token_count)
            weighted_kl = (
                raw_kl
                * float(self.consistency_kl_coef)
                * float(aggregation_scale)
            )
            weighted_kl.backward()
        finally:
            self._response_logits_callback = None
            self._inside_consistency_forward = False
            self.consistency_controller.set_clean()

        metrics = {
            "mlp_consistency/kl": float(raw_kl.detach().item()),
            "mlp_consistency/weighted_kl": float(weighted_kl.detach().item()),
            "mlp_consistency/response_tokens": float(teacher.token_count),
            "timing_s/mlp_consistency_forward_backward": time.perf_counter()
            - started,
        }
        self._teacher_distribution = None
        self._student_kl_sum = None
        return metrics
