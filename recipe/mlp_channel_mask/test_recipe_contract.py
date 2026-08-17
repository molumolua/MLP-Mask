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
        self.assertEqual(intervention["selection_strategy"], "top_saliency")
        self.assertEqual(intervention["random_seed"], 42)
        self.assertEqual(intervention["refresh_freq"], "${trainer.test_freq}")
        self.assertTrue(rollout["enable_prefix_caching"])
        self.assertFalse(
            config["algorithm"]["rollout_correction"]["bypass_old_logprob_for_rollout"]
        )

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

    def test_offline_launcher_disables_old_logprob_bypass(self) -> None:
        launcher = (RECIPE_DIR / "grpo_mlp_channel_mask_qwen3-4b_offline.sh").read_text()
        self.assertIn("algorithm.rollout_correction.bypass_old_logprob_for_rollout=False", launcher)

    def test_random_ten_percent_launcher_selects_seeded_random_masks(self) -> None:
        launcher = (RECIPE_DIR / "grpo_mlp_channel_mask_qwen3-4b_random10_offline.sh").read_text()
        base_launcher = (RECIPE_DIR / "grpo_mlp_channel_mask_qwen3-4b_offline.sh").read_text()

        self.assertIn("selection_strategy=${selection_strategy:-random}", launcher)
        self.assertIn("mask_ratio=${mask_ratio:-0.10}", launcher)
        self.assertIn("random_seed=${random_seed:-42}", launcher)
        self.assertIn("actor_rollout_ref.mlp_intervention.selection_strategy=${selection_strategy}", base_launcher)
        self.assertIn("actor_rollout_ref.mlp_intervention.random_seed=${random_seed}", base_launcher)

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

    def test_actor_update_receives_rollout_temperature(self) -> None:
        trainer_source = (RECIPE_DIR / "trainer.py").read_text()
        update_actor_body = trainer_source.split(
            'with marked_timer("update_actor_dual", timing_raw, color="red"):', 1
        )[1].split('metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))', 1)[0]

        temperature_assignment = 'batch.meta_info["temperature"] = float('
        actor_update = "self.actor_rollout_wg.update_actor(batch)"
        self.assertIn(temperature_assignment, update_actor_body)
        self.assertLess(update_actor_body.index(temperature_assignment), update_actor_body.index(actor_update))

    def test_old_log_prob_recomputation_is_route_aware(self) -> None:
        trainer_source = (RECIPE_DIR / "trainer.py").read_text()
        actor_source = (WORKSPACE / "verl" / "workers" / "actor" / "dp_actor.py").read_text()
        compute_log_prob_body = actor_source.split("    def compute_log_prob(", 1)[1].split(
            "    @GPUMemoryLogger", 1
        )[0]

        self.assertIn("self.actor_rollout_wg.compute_log_prob(batch)", trainer_source)
        self.assertIn('select_keys.append("response_mask")', compute_log_prob_body)
        self.assertIn('data.non_tensor_batch["route_id"]', compute_log_prob_body)
        self.assertIn("intervention_controller.set_route(route_name)", compute_log_prob_body)

    def test_rollout_mask_buffers_follow_vllm_sleep_lifecycle(self) -> None:
        worker_source = (RECIPE_DIR / "worker.py").read_text()
        rollout_body = worker_source.split("    async def rollout_mode(self):", 1)[1].split(
            "    async def trainer_mode(self):", 1
        )[0]
        trainer_body = worker_source.split("    async def trainer_mode(self):", 1)[1].split(
            "    @register", 1
        )[0]

        self.assertLess(
            rollout_body.index("await super().rollout_mode()"),
            rollout_body.index("set_active_buffers_available(True)"),
        )
        self.assertLess(
            trainer_body.index("set_active_buffers_available(False)"),
            trainer_body.index("await super().trainer_mode()"),
        )

    def test_validation_switch_timing_is_reduced_before_worker_concat(self) -> None:
        worker_source = (RECIPE_DIR / "worker.py").read_text()
        validation_body = worker_source.rsplit("    def generate_sequences(self, prompts: DataProto):", 1)[1].split(
            "    @register", 1
        )[0]

        reduce_call = 'reduce_timing({"mlp_mask_switch_rollout_clean": switch_elapsed})'
        timing_update = 'output.meta_info.setdefault("timing", {}).update(switch_timing)'
        self.assertIn(reduce_call, validation_body)
        self.assertIn(timing_update, validation_body)
        self.assertLess(validation_body.index(reduce_call), validation_body.index(timing_update))


if __name__ == "__main__":
    unittest.main()
