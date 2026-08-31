"""Focused numerical tests for validation pass@k and route entropy metrics."""

from __future__ import annotations

import math
import importlib.util
import pathlib
import unittest

VALIDATION_UTILS_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "verl"
    / "trainer"
    / "ppo"
    / "validation_utils.py"
)
SPEC = importlib.util.spec_from_file_location("validation_utils_under_test", VALIDATION_UTILS_PATH)
assert SPEC is not None and SPEC.loader is not None
VALIDATION_UTILS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATION_UTILS)
compute_pass_at_k_metrics = VALIDATION_UTILS.compute_pass_at_k_metrics
validation_prompt_uid = VALIDATION_UTILS.validation_prompt_uid


class ValidationPassAtKTest(unittest.TestCase):
    def test_prompt_uid_ignores_unicode_and_whitespace_serialization(self) -> None:
        self.assertEqual(
            validation_prompt_uid("Solve  x + 1 = 2\nnow"),
            validation_prompt_uid("Ｓｏｌｖｅ x + 1 = 2 now"),
        )

    def test_unbiased_pass_at_k_merges_sixteen_duplicate_rows(self) -> None:
        data_sources = ["aime"] * 32
        prompt_uids = ["question-a"] * 16 + ["question-b"] * 16
        correctness = [1] + [0] * 15 + [0] * 16

        result = compute_pass_at_k_metrics(data_sources, prompt_uids, correctness)["aime"]

        self.assertEqual(result["unique_prompts"], 2.0)
        self.assertEqual(result["samples_per_prompt_min"], 16.0)
        self.assertEqual(result["samples_per_prompt_max"], 16.0)
        self.assertAlmostEqual(result["pass@1"], 0.5 * (1.0 / 16.0))
        expected_question_a_pass2 = 1.0 - math.comb(15, 2) / math.comb(16, 2)
        self.assertAlmostEqual(result["pass@2"], 0.5 * expected_question_a_pass2)
        self.assertAlmostEqual(result["pass@16"], 0.5)
        self.assertEqual(result["prompts@16"], 2.0)

    def test_identical_prompt_ids_remain_separate_across_datasets(self) -> None:
        result = compute_pass_at_k_metrics(
            data_sources=["aime24", "aime25"],
            prompt_uids=["same-text", "same-text"],
            correctness_values=[1, 0],
        )

        self.assertEqual(result["aime24"]["pass@1"], 1.0)
        self.assertEqual(result["aime25"]["pass@1"], 0.0)
if __name__ == "__main__":
    unittest.main()
