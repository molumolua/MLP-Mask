"""CPU tests for full and top-k-plus-tail KL calculations."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from .batching import slice_model_inputs
from .kl import build_teacher_distribution, forward_kl_sum, slice_teacher_rows


class MLPChannelConsistencyActorTest(unittest.TestCase):
    def test_topk_tail_kl_is_zero_for_identical_logits_and_backpropagates(self):
        teacher_logits = torch.tensor(
            [[[2.0, 1.0, 0.0, -1.0], [0.0, 2.0, 1.0, -1.0]]]
        )
        response_mask = torch.tensor([[1, 1]], dtype=torch.bool)
        teacher = build_teacher_distribution(
            teacher_logits, response_mask, top_k=2
        )

        student_logits = teacher_logits.clone().requires_grad_(True)
        kl_sum = forward_kl_sum(teacher, student_logits, response_mask)
        self.assertAlmostEqual(float(kl_sum.item()), 0.0, places=6)
        kl_sum.backward()
        self.assertIsNotNone(student_logits.grad)

    def test_topk_tail_kl_detects_distribution_change(self):
        teacher_logits = torch.tensor([[[3.0, 1.0, 0.0, -1.0]]])
        response_mask = torch.tensor([[1]], dtype=torch.bool)
        teacher = build_teacher_distribution(
            teacher_logits, response_mask, top_k=2
        )

        student_logits = torch.tensor(
            [[[0.0, 1.0, 3.0, -1.0]]], requires_grad=True
        )
        kl_sum = forward_kl_sum(teacher, student_logits, response_mask)
        self.assertGreater(float(kl_sum.item()), 0.1)
        kl_sum.backward()
        self.assertGreater(float(student_logits.grad.abs().sum().item()), 0.0)

    def test_full_kl_matches_torch_definition(self):
        teacher_logits = torch.tensor([[[1.5, 0.5, -0.5]]])
        response_mask = torch.tensor([[1]], dtype=torch.bool)
        teacher = build_teacher_distribution(
            teacher_logits, response_mask, top_k=0
        )

        student_logits = torch.tensor([[[0.0, 1.0, -1.0]]], requires_grad=True)
        kl_sum = forward_kl_sum(teacher, student_logits, response_mask)
        teacher_logp = torch.log_softmax(teacher_logits[0, 0], dim=-1)
        student_logp = torch.log_softmax(student_logits[0, 0], dim=-1)
        expected = (teacher_logp.exp() * (teacher_logp - student_logp)).sum()
        torch.testing.assert_close(kl_sum, expected)

    def test_response_mask_excludes_invalid_positions(self):
        logits = torch.randn(2, 3, 5)
        response_mask = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool)
        teacher = build_teacher_distribution(logits, response_mask, top_k=2)
        self.assertEqual(teacher.token_count, 3)

    def test_sliced_teacher_preserves_exact_chunked_kl(self):
        teacher_logits = torch.randn(3, 4, 7)
        student_logits = torch.randn(3, 4, 7)
        response_mask = torch.tensor(
            [[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]], dtype=torch.bool
        )
        teacher = build_teacher_distribution(teacher_logits, response_mask, top_k=3)
        full_student = student_logits.clone().requires_grad_(True)
        full = forward_kl_sum(teacher, full_student, response_mask) / teacher.token_count
        full.backward()

        chunked_student = student_logits.clone().requires_grad_(True)
        chunked_value = 0.0
        for row in range(3):
            row_teacher = slice_teacher_rows(teacher, row, row + 1)
            chunk_loss = forward_kl_sum(
                row_teacher,
                chunked_student[row : row + 1],
                response_mask[row : row + 1],
            ) / teacher.token_count
            chunked_value += float(chunk_loss.detach().item())
            chunk_loss.backward()
        self.assertAlmostEqual(chunked_value, float(full.detach().item()), places=6)
        torch.testing.assert_close(chunked_student.grad, full_student.grad)

    def test_model_input_slicing_only_slices_per_example_values(self):
        inputs = {
            "tokens": torch.arange(12).reshape(3, 4),
            "labels": np.asarray(["a", "b", "c"], dtype=object),
            "shared": torch.tensor(2.0),
            "constant": "keep",
        }
        sliced = slice_model_inputs(inputs, 1, 3, 3)
        torch.testing.assert_close(sliced["tokens"], inputs["tokens"][1:3])
        self.assertEqual(sliced["labels"].tolist(), ["b", "c"])
        self.assertIs(sliced["shared"], inputs["shared"])
        self.assertEqual(sliced["constant"], "keep")


if __name__ == "__main__":
    unittest.main()
