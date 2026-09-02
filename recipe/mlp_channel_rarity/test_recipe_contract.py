"""Static contract tests for the launch/config integration."""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml

RECIPE_DIR = Path(__file__).resolve().parent


class MLPChannelRarityRecipeContractTest(unittest.TestCase):
    def test_config_enables_online_ema_and_mandatory_old_logprob_forward(self) -> None:
        config = yaml.safe_load((RECIPE_DIR / "config" / "ppo_mlp_channel_rarity.yaml").read_text())
        rarity = config["actor_rollout_ref"]["mlp_channel_rarity"]
        rollout = config["actor_rollout_ref"]["rollout"]

        self.assertTrue(rarity["enabled"])
        self.assertAlmostEqual(rarity["activation_ema_beta"], 0.95)
        self.assertAlmostEqual(rarity["topk_ratio"], 0.01)
        self.assertEqual(rarity["frequency_prior_strength"], 64.0)
        self.assertEqual(rarity["max_channel_rarity"], 8.0)
        self.assertFalse(rarity["use_frequency_prior"])
        self.assertEqual(rarity["min_loss_weight"], 0.2)
        self.assertEqual(rarity["max_loss_weight"], 5.0)
        self.assertFalse(rollout["log_prob_use_dynamic_bsz"])
        self.assertFalse(config["algorithm"]["rollout_correction"]["bypass_old_logprob_for_rollout"])

    def test_worker_piggybacks_on_logprob_and_actor_applies_one_loss_group(self) -> None:
        worker = (RECIPE_DIR / "worker.py").read_text()
        actor = (RECIPE_DIR / "actor.py").read_text()

        self.assertIn("result = self.rarity_controller.finalize_step()", worker)
        self.assertIn('output.batch["rarity_loss_weights"]', worker)
        self.assertIn('data.non_tensor_batch["loss_multiplier"] = weights', actor)
        self.assertIn('"mlp_channel_rarity"', actor)

    def test_question_rarity_dump_is_enabled_by_the_launcher(self) -> None:
        trainer = (RECIPE_DIR / "trainer.py").read_text()
        diagnostics = (RECIPE_DIR / "diagnostics.py").read_text()
        launcher = (RECIPE_DIR / "grpo_mlp_channel_rarity_qwen3-4b_offline.sh").read_text()

        self.assertIn("class MLPChannelRarityTrainer", trainer)
        self.assertIn('Path(rollout_data_dir) / "question_rarity"', trainer)
        self.assertIn('"average_accuracy"', diagnostics)
        self.assertIn('"raw_s_q"', diagnostics)
        self.assertIn('"s_q"', diagnostics)
        self.assertIn('trainer.rollout_data_dir="${rollout_data_dir}"', launcher)

    def test_first_step_unit_weight_behavior_is_explicit(self) -> None:
        source = (RECIPE_DIR / "rarity.py").read_text()
        self.assertIn("if not self.ema_initialized:", source)
        self.assertIn("loss_weights = torch.ones_like(raw_scores)", source)
        self.assertIn("self.normal_activation.copy_(batch_normal)", source)

    def test_launcher_uses_recipe_and_configurable_python(self) -> None:
        launcher = (RECIPE_DIR / "grpo_mlp_channel_rarity_qwen3-4b_offline.sh").read_text()
        self.assertIn("recipe.mlp_channel_rarity.main", launcher)
        self.assertIn('python_bin=${python_bin:-python}', launcher)
        self.assertIn('"${python_bin}" -m recipe.mlp_channel_rarity.main', launcher)
        self.assertIn("actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False", launcher)
        self.assertIn("use_frequency_prior=${use_frequency_prior}", launcher)


if __name__ == "__main__":
    unittest.main()
