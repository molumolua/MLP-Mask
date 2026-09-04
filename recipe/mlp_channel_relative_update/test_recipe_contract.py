"""Static contract tests for the relative-update recipe."""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml

RECIPE_DIR = Path(__file__).resolve().parent


class RelativeUpdateRecipeContractTest(unittest.TestCase):
    def test_config_selects_fsdp2_custom_optimizer_and_tenfold_cap(self) -> None:
        config = yaml.safe_load(
            (RECIPE_DIR / "config" / "ppo_mlp_channel_relative_update.yaml").read_text()
        )
        actor = config["actor_rollout_ref"]["actor"]
        component = config["actor_rollout_ref"]["mlp_channel_relative_update"]

        self.assertEqual(actor["strategy"], "fsdp2")
        self.assertFalse(actor["fsdp_config"]["offload_policy"])
        self.assertEqual(
            actor["optim"]["optimizer_impl"],
            "recipe.mlp_channel_relative_update.optimizer",
        )
        self.assertEqual(actor["optim"]["optimizer"], "ChannelRelativeUpdateAdamW")
        self.assertTrue(component["enabled"])
        self.assertEqual(component["multiplier_ratio_cap"], 10.0)
        self.assertEqual(component["warmup_steps"], 16)

    def test_worker_saves_history_and_exports_metrics(self) -> None:
        worker = (RECIPE_DIR / "worker.py").read_text()
        self.assertIn("configure_channel_updates", worker)
        self.assertIn("get_last_relative_update_metrics", worker)
        self.assertIn('"mlp_channel_relative_update.pt"', worker)

    def test_launcher_uses_recipe_optimizer_and_molu_python(self) -> None:
        launcher = (
            RECIPE_DIR / "grpo_mlp_channel_relative_update_qwen3-4b_offline.sh"
        ).read_text()
        self.assertIn("recipe.mlp_channel_relative_update.main", launcher)
        self.assertIn("ChannelRelativeUpdateAdamW", launcher)
        self.assertIn("actor_rollout_ref.actor.strategy=fsdp2", launcher)
        self.assertIn("multiplier_ratio_cap=${multiplier_ratio_cap}", launcher)
        self.assertIn("/opt/homebrew/Caskroom/miniconda/base/envs/molu/bin/python", launcher)

    def test_control_launcher_forces_unit_multipliers(self) -> None:
        launcher = (
            RECIPE_DIR / "grpo_mlp_channel_relative_update_qwen3-4b_control_offline.sh"
        ).read_text()
        self.assertIn("multiplier_ratio_cap=${multiplier_ratio_cap:-1.0}", launcher)
        self.assertIn("grpo_mlp_channel_relative_update_qwen3-4b_offline.sh", launcher)

    def test_standard_control_disables_component_and_uses_native_adamw(self) -> None:
        launcher = (
            RECIPE_DIR
            / "grpo_mlp_channel_relative_update_qwen3-4b_standard_adamw_offline.sh"
        ).read_text()
        self.assertIn("relative_update_enabled=False", launcher)
        self.assertIn("optimizer_impl=torch.optim", launcher)
        self.assertIn("optimizer_name=AdamW", launcher)


if __name__ == "__main__":
    unittest.main()
