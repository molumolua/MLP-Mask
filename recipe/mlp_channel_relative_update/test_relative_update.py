"""CPU tests for history-based MLP-channel update allocation."""

from __future__ import annotations

import unittest

import torch

from .relative_update import (
    MLPChannelRelativeUpdateController,
    project_multipliers_to_fixed_update_energy,
)


class MultiplierProjectionTest(unittest.TestCase):
    def test_preserves_update_energy_bounds_and_order(self) -> None:
        raw = torch.tensor([[0.1, 1.0, 8.0, 2.0]])
        energy = torch.tensor([[1.0, 3.0, 0.5, 2.0]])
        result = project_multipliers_to_fixed_update_energy(
            raw,
            energy,
            min_multiplier=0.25,
            max_multiplier=4.0,
        )

        self.assertGreaterEqual(float(result.min()), 0.25)
        self.assertLessEqual(float(result.max()), 4.0)
        self.assertTrue(torch.equal(torch.argsort(raw.flatten()), torch.argsort(result.flatten())))
        before = energy.sum()
        after = (result.square() * energy).sum()
        torch.testing.assert_close(after, before, rtol=1e-6, atol=1e-7)


class RelativeUpdateControllerTest(unittest.TestCase):
    def _controller(self, **overrides) -> MLPChannelRelativeUpdateController:
        kwargs = {
            "num_layers": 1,
            "intermediate_size": 11,
            "history_ema_beta": 0.0,
            "history_power": 1.0,
            "history_floor_ratio": 0.0,
            "multiplier_ratio_cap": 10.0,
            "warmup_steps": 0,
        }
        kwargs.update(overrides)
        return MLPChannelRelativeUpdateController(**kwargs)

    def test_low_history_channel_can_receive_tenfold_multiplier(self) -> None:
        controller = self._controller()
        controller.history_relative_update_sq[0, :10] = 100.0**2
        controller.history_relative_update_sq[0, 10] = 0.01**2
        controller.step_count = 1
        ones = torch.ones((1, 11))

        step = controller.prepare_step(
            local_base_update_sq=ones,
            local_parameter_sq=ones,
            local_parameter_count=ones,
        )

        self.assertAlmostEqual(float(step.multipliers.max()), 10**0.5, places=5)
        self.assertAlmostEqual(float(step.multipliers.min()), 10**-0.5, places=5)
        self.assertAlmostEqual(
            float(step.multipliers[0, 10] / step.multipliers[0, 0]),
            10.0,
            places=5,
        )
        torch.testing.assert_close(
            (step.multipliers.square() * ones).sum(),
            ones.sum(),
        )
        metrics = controller.commit_step(step)
        self.assertGreater(
            metrics["mlp_relative_update/energy_share_shift_to_low_history"],
            0.0,
        )

    def test_warmup_uses_unit_multipliers_and_commits_actual_history(self) -> None:
        controller = self._controller(
            intermediate_size=3,
            history_ema_beta=0.5,
            warmup_steps=2,
        )
        base_update_sq = torch.tensor([[1.0, 4.0, 9.0]])
        parameter_sq = torch.tensor([[100.0, 100.0, 100.0]])
        count = torch.ones_like(base_update_sq)

        step = controller.prepare_step(
            local_base_update_sq=base_update_sq,
            local_parameter_sq=parameter_sq,
            local_parameter_count=count,
        )
        self.assertTrue(step.warmup)
        torch.testing.assert_close(step.multipliers, torch.ones_like(step.multipliers))
        metrics = controller.commit_step(step)

        torch.testing.assert_close(
            controller.history_relative_update_sq,
            0.5 * base_update_sq / parameter_sq,
        )
        self.assertEqual(metrics["mlp_relative_update/warmup"], 1.0)
        self.assertAlmostEqual(metrics["mlp_relative_update/update_energy_ratio"], 1.0)

    def test_checkpoint_round_trip_and_config_validation(self) -> None:
        controller = self._controller()
        controller.history_relative_update_sq.copy_(torch.arange(11).reshape(1, 11))
        controller.step_count = 7
        restored = self._controller()
        restored.load_state_dict(controller.state_dict())

        torch.testing.assert_close(
            restored.history_relative_update_sq,
            controller.history_relative_update_sq,
        )
        self.assertEqual(restored.step_count, 7)

        incompatible = self._controller(history_power=0.5)
        with self.assertRaisesRegex(ValueError, "history_power"):
            incompatible.load_state_dict(controller.state_dict())


if __name__ == "__main__":
    unittest.main()
