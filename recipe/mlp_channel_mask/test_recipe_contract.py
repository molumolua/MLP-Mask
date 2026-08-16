"""Dependency-light source/config contract tests for the standalone recipe."""

from __future__ import annotations

import pathlib
import unittest

import yaml


RECIPE_DIR = pathlib.Path(__file__).resolve().parent
WORKSPACE = RECIPE_DIR.parents[1]


class RecipeContractTest(unittest.TestCase):
    def test_default_budget_and_block_ratio(self) -> None:
        config = yaml.safe_load((RECIPE_DIR / "config" / "ppo_mlp_channel_mask.yaml").read_text())
        intervention = config["actor_rollout_ref"]["mlp_intervention"]
        rollout = config["actor_rollout_ref"]["rollout"]
        self.assertEqual(intervention["n_clean"], 8)
        self.assertEqual(intervention["n_masked"], 8)
        self.assertEqual(rollout["n"], 16)
        self.assertEqual(intervention["n_clean"] + intervention["n_masked"], rollout["n"])
        self.assertAlmostEqual(intervention["mask_ratio"], 0.10)
        self.assertEqual(intervention["refresh_freq"], "${trainer.test_freq}")
        self.assertTrue(rollout["enable_prefix_caching"])

    def test_offline_launcher_contains_all_requested_datasets(self) -> None:
        launcher = (RECIPE_DIR / "grpo_mlp_channel_mask_qwen3-4b_offline.sh").read_text()
        self.assertIn("WANDB_MODE=${WANDB_MODE:-offline}", launcher)
        for filename in (
            "MATH7500.with_wrong_boxed.qwen3-4b-base.parquet",
            "aime25_test.parquet",
            "bbeh_data.parquet",
            "MATH500-test.parquet",
            "amc23_test.parquet",
            "aime24_test.parquet",
            "MMLU-Pro-Valid.parquet",
        ):
            self.assertIn(filename, launcher)

    def test_recipe_does_not_import_another_recipe(self) -> None:
        for source_path in RECIPE_DIR.glob("*.py"):
            if source_path.name.startswith("test_"):
                continue
            source = source_path.read_text()
            for line in source.splitlines():
                stripped = line.strip()
                if stripped.startswith(("from recipe.", "import recipe.")):
                    self.assertIn("recipe.mlp_channel_mask", stripped, msg=f"{source_path}: {stripped}")

    def test_actor_saliency_uses_causal_response_loss_positions(self) -> None:
        actor_source = (WORKSPACE / "verl" / "workers" / "actor" / "dp_actor.py").read_text()
        self.assertIn(
            'response_token_mask[:, -response_length - 1 : -1] = micro_batch["response_mask"]',
            actor_source,
        )


if __name__ == "__main__":
    unittest.main()
