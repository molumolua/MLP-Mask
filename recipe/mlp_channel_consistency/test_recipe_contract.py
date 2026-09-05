"""Static contract tests for the standalone consistency recipe."""

from __future__ import annotations

import pathlib
import unittest

import yaml


RECIPE_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = RECIPE_DIR.parents[1]


class MLPChannelConsistencyRecipeContractTest(unittest.TestCase):
    def test_default_config_is_clean_grpo_with_hard_ten_percent_mask(self):
        config = yaml.safe_load(
            (RECIPE_DIR / "config" / "ppo_mlp_channel_consistency.yaml").read_text()
        )
        component = config["actor_rollout_ref"]["mlp_channel_consistency"]
        self.assertTrue(component["enabled"])
        self.assertTrue(component["auxiliary_enabled"])
        self.assertEqual(component["mask_ratio"], 0.10)
        self.assertEqual(component["kl_top_k"], 64)
        self.assertEqual(component["micro_batch_size_per_gpu"], 1)
        self.assertEqual(component["gradient_sample_size_per_rank"], 262144)
        self.assertEqual(component["parameter_update_atol"], 1.0e-5)
        self.assertEqual(config["actor_rollout_ref"]["rollout"]["n"], 16)
        self.assertFalse(config["algorithm"]["use_kl_in_reward"])

    def test_recipe_does_not_import_another_recipe(self):
        for path in RECIPE_DIR.glob("*.py"):
            if path.name.startswith("test_"):
                continue
            source = path.read_text()
            self.assertNotIn("from recipe.", source, path.name)
            self.assertNotIn("import recipe.", source, path.name)

    def test_launcher_uses_configurable_python_and_recipe_entrypoint(self):
        launcher = (
            RECIPE_DIR / "grpo_mlp_channel_consistency_qwen3-4b_offline.sh"
        ).read_text()
        self.assertIn("python_bin=${python_bin:-python}", launcher)
        self.assertIn('"${python_bin}" -m', launcher)
        self.assertIn("recipe.mlp_channel_consistency.main", launcher)
        self.assertIn("mask_ratio=${mask_ratio}", launcher)
        self.assertIn("auxiliary_enabled=${auxiliary_enabled}", launcher)
        self.assertIn("kl_coef=${kl_coef}", launcher)
        self.assertIn(
            "micro_batch_size_per_gpu=${kl_micro_batch_size_per_gpu}", launcher
        )
        self.assertIn(
            "gradient_sample_size_per_rank=${gradient_sample_size_per_rank}",
            launcher,
        )

    def test_core_actor_exposes_logits_and_auxiliary_backward_hooks(self):
        actor_source = (REPO_ROOT / "verl" / "workers" / "actor" / "dp_actor.py").read_text()
        self.assertIn('getattr(self, "_response_logits_callback", None)', actor_source)
        self.assertIn('getattr(self, "_backward_auxiliary_loss", None)', actor_source)
        self.assertLess(
            actor_source.index("loss.backward()"),
            actor_source.index("auxiliary_backward = getattr"),
        )

    def test_recipe_records_loss_gradient_and_validation_update_ratios(self):
        actor_source = (RECIPE_DIR / "actor.py").read_text()
        worker_source = (RECIPE_DIR / "worker.py").read_text()
        main_source = (RECIPE_DIR / "main.py").read_text()
        self.assertIn("aux_to_main_loss_abs_ratio", actor_source)
        self.assertIn("aux_to_main_grad_ratio_sampled", (RECIPE_DIR / "diagnostics.py").read_text())
        self.assertIn("compute_parameter_update_metrics", worker_source)
        self.assertIn("compute_parameter_update_metrics", main_source)

    def test_baseline_disables_auxiliary_work_but_keeps_recipe_diagnostics(self):
        baseline = (RECIPE_DIR / "baseline_grpo_qwen3-4b_offline.sh").read_text()
        self.assertIn("export auxiliary_enabled=False", baseline)
        self.assertIn("export kl_coef=0", baseline)
        self.assertIn("grpo_mlp_channel_consistency_qwen3-4b_offline.sh", baseline)

        actor_source = (RECIPE_DIR / "actor.py").read_text()
        self.assertIn("if not self.consistency_auxiliary_enabled", actor_source)
        self.assertIn('"mlp_consistency/weighted_kl": 0.0', actor_source)


if __name__ == "__main__":
    unittest.main()
