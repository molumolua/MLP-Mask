"""CPU tests for the question-level MLP rarity JSONL records."""

from __future__ import annotations

import json
import unittest

import numpy as np
import torch

from verl import DataProto

from .diagnostics import build_question_rarity_records


class _Tokenizer:
    def decode(self, token_ids, skip_special_tokens=True) -> str:
        del skip_special_tokens
        return " ".join(str(int(token)) for token in token_ids if int(token) != 0)


class QuestionRarityRecordTest(unittest.TestCase):
    def _batch(self, *, include_accuracy: bool = True) -> DataProto:
        non_tensors = {
            # Deliberately interleave questions: grouping must use uid, not row chunks.
            "uid": np.array(["q-a", "q-b", "q-a", "q-b"], dtype=object),
            "data_source": np.array(["math", "aime", "math", "aime"], dtype=object),
            "index": np.array([10, 20, 10, 20], dtype=object),
            "reward_model": np.array(
                [
                    {"ground_truth": "A"},
                    {"ground_truth": "B"},
                    {"ground_truth": "A"},
                    {"ground_truth": "B"},
                ],
                dtype=object,
            ),
            "extra_info": np.array(
                [{"split": "train"}, {"split": "train"}] * 2,
                dtype=object,
            ),
            "format_score": np.array([1.0, 0.5, 0.0, 0.5], dtype=object),
        }
        if include_accuracy:
            non_tensors["acc"] = np.array([1.0, 0.0, 0.0, 1.0], dtype=object)

        return DataProto.from_dict(
            tensors={
                "prompts": torch.tensor(
                    [[0, 11, 12], [0, 21, 22], [0, 11, 12], [0, 21, 22]]
                ),
                "token_level_scores": torch.tensor(
                    [[0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0]]
                ),
                "rarity_scores": torch.tensor([2.0, 4.0, 2.0, 4.0]),
                "rarity_loss_weights": torch.tensor([0.5, 1.5, 0.5, 1.5]),
            },
            non_tensors=non_tensors,
            meta_info={
                "mlp_channel_rarity_metrics": {
                    "mlp_rarity/step": 7.0,
                    "mlp_rarity/exposure_questions": 32.0,
                }
            },
        )

    def test_aggregates_one_record_per_uid_with_accuracy_and_rarity(self) -> None:
        records = build_question_rarity_records(
            self._batch(),
            _Tokenizer(),
            global_step=9,
            rarity_config={"topk_ratio": 0.01},
        )

        self.assertEqual(len(records), 2)
        first, second = records
        self.assertEqual(first["question_uid"], "q-a")
        self.assertEqual(first["prompt"], "11 12")
        self.assertEqual(first["dataset_index"], 10)
        self.assertEqual(first["ground_truth"], "A")
        self.assertEqual(first["n_rollouts"], 2)
        self.assertEqual(first["average_accuracy"], 0.5)
        self.assertEqual(first["rollout_accuracies"], [1.0, 0.0])
        self.assertEqual(first["raw_s_q"], 2.0)
        self.assertEqual(first["s_q"], 0.5)
        self.assertEqual(first["rarity_step"], 7)
        self.assertEqual(first["rarity_config"], {"topk_ratio": 0.01})
        self.assertEqual(first["reward_extra_info"]["format_score"], [1.0, 0.0])
        self.assertEqual(second["question_uid"], "q-b")
        self.assertEqual(second["raw_s_q"], 4.0)
        self.assertEqual(second["s_q"], 1.5)
        json.dumps(first, ensure_ascii=False)

    def test_missing_acc_stays_null_instead_of_treating_reward_as_accuracy(self) -> None:
        records = build_question_rarity_records(
            self._batch(include_accuracy=False),
            _Tokenizer(),
            global_step=1,
        )

        self.assertIsNone(records[0]["average_accuracy"])
        self.assertEqual(records[0]["rollout_accuracies"], [])
        self.assertEqual(records[0]["reward_mean"], 0.5)

    def test_missing_required_rarity_tensor_is_an_explicit_error(self) -> None:
        batch = self._batch()
        batch.batch.pop("rarity_scores")

        with self.assertRaisesRegex(RuntimeError, "rarity_scores"):
            build_question_rarity_records(batch, _Tokenizer(), global_step=1)


if __name__ == "__main__":
    unittest.main()
