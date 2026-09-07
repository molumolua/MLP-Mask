from __future__ import annotations

import pathlib
import unittest

import yaml


RECIPE_DIR = pathlib.Path(__file__).resolve().parent
WORKSPACE = RECIPE_DIR.parents[1]


class AntitheticRecipeContractTest(unittest.TestCase):
    def test_default_config_keeps_original_rollout_budget(self) -> None:
        config = yaml.safe_load(
            (RECIPE_DIR / "config" / "ppo_mlp_channel_antithetic.yaml").read_text()
        )
        antithetic = config["actor_rollout_ref"]["mlp_channel_antithetic"]
        rollout = config["actor_rollout_ref"]["rollout"]
        self.assertEqual(rollout["n"], 16)
        self.assertEqual(rollout["n"] % 2, 0)
        self.assertAlmostEqual(antithetic["perturbation_strength"], 0.10)
        self.assertTrue(antithetic["refresh_every_step"])
        self.assertEqual(rollout["data_parallel_size"], 1)
        self.assertEqual(rollout["pipeline_model_parallel_size"], 1)
        self.assertTrue(config["trainer"]["balance_batch"])
        self.assertFalse(
            config["algorithm"]["rollout_correction"][
                "bypass_old_logprob_for_rollout"
            ]
        )

    def test_training_keeps_uid_group_and_uses_route_only_as_metadata(self) -> None:
        worker = (RECIPE_DIR / "worker.py").read_text()
        trainer = (RECIPE_DIR / "trainer.py").read_text()
        ray_trainer = (
            WORKSPACE / "verl" / "trainer" / "ppo" / "ray_trainer.py"
        ).read_text()
        self.assertIn('prompts.non_tensor_batch["route_id"] = routes', worker)
        self.assertNotIn('non_tensor_batch["uid"] =', worker)
        self.assertNotIn("loss_group_id", worker)
        self.assertNotIn("loss_group_id", trainer)
        self.assertIn(
            "num_repeat=self.config.actor_rollout_ref.rollout.n", ray_trainer
        )

    def test_route_is_reused_for_generation_logprob_and_actor_update(self) -> None:
        worker = (RECIPE_DIR / "worker.py").read_text()
        actor = (WORKSPACE / "verl" / "workers" / "actor" / "dp_actor.py").read_text()
        self.assertIn("set_route(POSITIVE_ROUTE)", worker)
        self.assertIn("set_route(NEGATIVE_ROUTE)", worker)
        self.assertIn("intervention_controller.set_route(route_name)", actor)
        self.assertIn("intervention_controller.set_route(\n                            route_name,", actor)
        self.assertIn("batch_version_field", actor)
        self.assertIn("valid_routes", actor)

    def test_validation_uses_neutral_route(self) -> None:
        worker = (RECIPE_DIR / "worker.py").read_text()
        self.assertIn('prompts.meta_info.get("validate", False)', worker)
        self.assertIn("set_route(NEUTRAL_ROUTE)", worker)

    def test_launcher_uses_paired_budget_and_local_python(self) -> None:
        launcher = (
            RECIPE_DIR / "grpo_mlp_channel_antithetic_qwen3-4b_offline.sh"
        ).read_text()
        self.assertIn("n_total=${n_total:-16}", launcher)
        self.assertIn("n_total % 2", launcher)
        self.assertIn("perturbation_strength=${perturbation_strength:-0.10}", launcher)
        self.assertIn(
            "/opt/homebrew/Caskroom/miniconda/base/envs/molu/bin/python", launcher
        )
        self.assertIn("actor_rollout_ref.rollout.data_parallel_size=1", launcher)
        self.assertIn("algorithm.adv_estimator=grpo", launcher)


if __name__ == "__main__":
    unittest.main()
