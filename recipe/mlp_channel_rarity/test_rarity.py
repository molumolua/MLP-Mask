"""CPU unit tests for online MLP-channel rarity statistics."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from .rarity import (
    MLPChannelRarityController,
    install_hf_mlp_activation_observer,
    project_scores_to_bounded_mean_one,
)


class _DenseMLP(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(width, width, bias=False)
        self.up_proj = nn.Linear(width, width, bias=False)
        self.down_proj = nn.Linear(width, width, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        hidden = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(hidden)


class _Block(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.mlp = _DenseMLP(width)


class _FSDPWrapper(nn.Module):
    """Dependency-free stand-in for FSDP1's named-module hierarchy."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self._fsdp_wrapped_module = module

    def forward(self, *args, **kwargs):
        return self._fsdp_wrapped_module(*args, **kwargs)


class _ToyModel(nn.Module):
    def __init__(self, layers: int, width: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Block(width) for _ in range(layers)])


class MLPChannelRarityControllerTest(unittest.TestCase):
    def _controller(
        self,
        *,
        beta: float = 0.9,
        prior_strength: float = 64.0,
        max_channel_rarity: float = 8.0,
        responses_per_question: int = 1,
        use_frequency_prior: bool = False,
    ) -> MLPChannelRarityController:
        return MLPChannelRarityController(
            num_layers=1,
            intermediate_size=4,
            selected_layers=[0],
            activation_ema_beta=beta,
            top_k=1,
            frequency_prior_strength=prior_strength,
            max_channel_rarity=max_channel_rarity,
            responses_per_question=responses_per_question,
            use_frequency_prior=use_frequency_prior,
            min_loss_weight=0.2,
            max_loss_weight=5.0,
        )

    @staticmethod
    def _run_step(controller: MLPChannelRarityController, levels: torch.Tensor):
        # One prompt token per question means RMS activation equals abs(levels).
        controller.begin_step()
        batch_size = levels.shape[0]
        controller.begin_micro_batch(
            prompt_mask=torch.ones((batch_size, 1), dtype=torch.bool),
            sample_ids=torch.arange(batch_size).unsqueeze(1),
            sample_count=batch_size,
        )
        controller.observe(0, levels.unsqueeze(1))
        controller.end_micro_batch()
        return controller.finalize_step()

    def test_first_step_initializes_ema_and_returns_exact_unit_weights(self) -> None:
        controller = self._controller()
        levels = torch.tensor([[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]])

        result = self._run_step(controller, levels)

        torch.testing.assert_close(result.loss_weights, torch.ones(2))
        torch.testing.assert_close(controller.normal_activation[0], torch.tensor([2.0, 3.0, 4.0, 5.0]))
        self.assertEqual(controller.step_count, 1)
        self.assertEqual(controller.exposure_questions, 0.0)
        self.assertEqual(result.metrics["mlp_rarity/first_step_unit_weights"], 1.0)

    def test_cumulative_frequency_makes_the_less_exposed_channel_heavier(self) -> None:
        controller = self._controller(beta=0.9, prior_strength=4.0)
        self._run_step(controller, torch.ones((2, 4)))

        # Step 2: both questions select channel 0, making it common.
        step2 = torch.tensor([[5.0, 1.0, 1.0, 1.0], [5.0, 1.0, 1.0, 1.0]])
        self._run_step(controller, step2)

        # Step 3: q0 selects common channel 0; q1 selects unseen channel 1.
        step3 = torch.tensor([[6.0, 1.0, 1.0, 1.0], [1.0, 6.0, 1.0, 1.0]])
        result = self._run_step(controller, step3)

        self.assertEqual(controller.exposure_questions, 4.0)
        torch.testing.assert_close(controller.exposure_count[0], torch.tensor([3.0, 1.0, 0.0, 0.0]))
        self.assertGreater(float(result.raw_scores[1]), float(result.raw_scores[0]))
        self.assertGreater(float(result.loss_weights[1]), float(result.loss_weights[0]))
        self.assertAlmostEqual(float(result.loss_weights.mean()), 1.0, places=6)

    def test_frequency_prior_and_rarity_cap_bound_early_self_information(self) -> None:
        controller = self._controller(
            prior_strength=4.0,
            max_channel_rarity=0.5,
            use_frequency_prior=True,
        )
        self._run_step(controller, torch.ones((1, 4)))

        result = self._run_step(controller, torch.tensor([[5.0, 1.0, 1.0, 1.0]]))

        # p0 = 1/4; after one selection f_tilde = (1 + 4*1/4)/(1+4) = 0.4.
        # -log(0.4) > 0.5, so the configured cap is active.
        self.assertAlmostEqual(result.metrics["mlp_rarity/effective_frequency_max"], 0.4)
        self.assertAlmostEqual(float(result.raw_scores[0]), 0.5)
        self.assertEqual(result.metrics["mlp_rarity/channel_rarity_max_observed"], 0.5)

    def test_frequency_prior_is_disabled_by_default(self) -> None:
        controller = self._controller(prior_strength=1_000.0)
        self._run_step(controller, torch.ones((2, 4)))

        result = self._run_step(
            controller,
            torch.tensor([[5.0, 1.0, 1.0, 1.0], [1.0, 5.0, 1.0, 1.0]]),
        )

        # Each selected channel has empirical C/N = 1/2. A very strong prior
        # would move this close to p0=1/4, so 1/2 also proves the switch is off.
        self.assertEqual(result.metrics["mlp_rarity/use_frequency_prior"], 0.0)
        self.assertAlmostEqual(result.metrics["mlp_rarity/effective_frequency_max"], 0.5)

    def test_loss_weight_projection_enforces_bounds_and_global_mean(self) -> None:
        scores = torch.tensor([0.0, 0.001, 1.0, 10.0, 1_000_000.0])

        weights = project_scores_to_bounded_mean_one(
            scores,
            min_weight=0.2,
            max_weight=5.0,
        )

        self.assertGreaterEqual(float(weights.min()), 0.2)
        self.assertLessEqual(float(weights.max()), 5.0)
        self.assertAlmostEqual(float(weights.to(torch.float64).mean()), 1.0, places=7)

    def test_exposure_is_counted_per_question_not_per_grpo_response(self) -> None:
        controller = self._controller(
            prior_strength=0.0,
            responses_per_question=2,
        )
        self._run_step(controller, torch.ones((2, 4)))

        self._run_step(controller, torch.tensor([[5.0, 1.0, 1.0, 1.0]] * 2))

        self.assertEqual(controller.exposure_questions, 1.0)
        torch.testing.assert_close(
            controller.exposure_count[0],
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
        )

    def test_prompt_mask_excludes_response_and_padding_tokens(self) -> None:
        controller = self._controller(beta=0.0)
        # Two samples packed into one unpadded token axis. IDs recover question boundaries.
        activation = torch.tensor(
            [[[100.0, 100.0, 100.0, 100.0], [3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 6.0, 8.0]]]
        )
        controller.begin_step()
        controller.begin_micro_batch(
            prompt_mask=torch.tensor([[False, True, True]]),
            sample_ids=torch.tensor([[0, 0, 1]]),
            sample_count=2,
        )
        controller.observe(0, activation)
        controller.end_micro_batch()
        controller.finalize_step()

        torch.testing.assert_close(
            controller.normal_activation[0],
            torch.tensor([1.5, 2.0, 3.0, 4.0]),
        )

    def test_state_round_trip_preserves_ema_and_exposure(self) -> None:
        controller = self._controller()
        self._run_step(controller, torch.ones((2, 4)))
        self._run_step(controller, torch.tensor([[5.0, 1.0, 1.0, 1.0]] * 2))
        restored = self._controller()

        restored.load_state_dict(controller.state_dict())

        torch.testing.assert_close(restored.normal_activation, controller.normal_activation.cpu())
        torch.testing.assert_close(restored.exposure_count, controller.exposure_count.cpu())
        self.assertEqual(restored.exposure_questions, controller.exposure_questions)
        self.assertEqual(restored.step_count, controller.step_count)

    def test_dense_mlp_down_projection_hook_observes_selected_layers(self) -> None:
        controller = MLPChannelRarityController(
            num_layers=2,
            intermediate_size=4,
            selected_layers=[1],
            top_k=1,
        )
        model = _ToyModel(layers=2, width=4)

        installed = install_hf_mlp_activation_observer(model, controller)

        self.assertEqual(installed, ["layers.1.mlp"])

    def test_dense_mlp_is_found_inside_fsdp1_wrapped_decoder_layers(self) -> None:
        controller = MLPChannelRarityController(
            num_layers=2,
            intermediate_size=4,
            top_k=1,
        )
        model = _ToyModel(layers=2, width=4)
        model.layers = nn.ModuleList([_FSDPWrapper(layer) for layer in model.layers])

        installed = install_hf_mlp_activation_observer(model, controller)

        self.assertEqual(
            installed,
            [
                "layers.0._fsdp_wrapped_module.mlp",
                "layers.1._fsdp_wrapped_module.mlp",
            ],
        )

        controller.begin_step()
        controller.begin_micro_batch(
            prompt_mask=torch.ones((1, 1), dtype=torch.bool),
            sample_ids=torch.zeros((1, 1), dtype=torch.long),
            sample_count=1,
        )
        model.layers[0]._fsdp_wrapped_module.mlp(torch.ones((1, 1, 4)))
        model.layers[1]._fsdp_wrapped_module.mlp(torch.ones((1, 1, 4)))
        controller.end_micro_batch()
        result = controller.finalize_step()
        torch.testing.assert_close(result.loss_weights, torch.ones(1))


if __name__ == "__main__":
    unittest.main()
