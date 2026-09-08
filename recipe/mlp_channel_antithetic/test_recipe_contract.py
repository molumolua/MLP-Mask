from __future__ import annotations

import pathlib
import os
import subprocess
import tempfile
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
        self.assertFalse(antithetic["reward_difference_update"]["enabled"])
        self.assertEqual(antithetic["reward_difference_update"]["max_update_ratio"], 0.05)
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
        self.assertIn("def _get_gen_batch(self, batch: DataProto)", trainer)
        self.assertIn('gen_batch.non_tensor_batch["uid"] = prompt_uids', trainer)
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
        self.assertIn(
            'callable(getattr(controller, "set_response_token_mask", None))', actor
        )

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

    def test_launcher_switch_and_experiment_names(self) -> None:
        # Execute the real shell launcher with an inert interpreter stand-in.
        # No Ray workers, models, or datasets are loaded.
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            capture = root / "arguments.txt"
            stub = root / "capture-arguments"
            stub.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE_ARGUMENTS"\n')
            stub.chmod(0o755)
            for enabled in ("False", "True"):
                env = dict(os.environ)
                for variable in ("experiment_name", "WANDB_RUN_ID", "CKPTS_DIR"):
                    env.pop(variable, None)
                env.update({
                    "python_bin": str(stub),
                    "CAPTURE_ARGUMENTS": str(capture),
                    "WANDB_DIR": str(root / "wandb"),
                    "reward_update_enabled": enabled,
                    "reward_update_lr": "0.002",
                    "reward_update_ratio": "0.03",
                })
                subprocess.run(
                    ["bash", str(RECIPE_DIR / "grpo_mlp_channel_antithetic_qwen3-4b_offline.sh")],
                    env=env, check=True, capture_output=True, text=True,
                )
                arguments = capture.read_text().splitlines()
                self.assertIn(f"actor_rollout_ref.mlp_channel_antithetic.reward_difference_update.enabled={enabled}", arguments)
                self.assertIn("actor_rollout_ref.mlp_channel_antithetic.reward_difference_update.learning_rate=0.002", arguments)
                self.assertIn("actor_rollout_ref.mlp_channel_antithetic.reward_difference_update.max_update_ratio=0.03", arguments)
                name = next(arg for arg in arguments if arg.startswith("trainer.experiment_name="))
                self.assertEqual("-reward-update-lr0.002-ratio0.03" in name, enabled == "True")


if __name__ == "__main__":
    unittest.main()
