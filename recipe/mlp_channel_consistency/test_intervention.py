"""CPU tests for hard per-layer MLP-channel consistency masks."""

from __future__ import annotations

import unittest

import torch

from .intervention import (
    GRADIENT_ACTIVATION_SCORE,
    HARD_TOP_SELECTION,
    MLPChannelConsistencyController,
    OUTPUT_CONTRIBUTION_SCORE,
    RELATIVE_ACTIVATION_SCORE,
    SOFT_TOP_SELECTION,
    UPDATED_FRACTION_SCORE,
    install_hf_mlp_consistency_mask,
)


class _ToyMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(4, 10, bias=False)
        self.up_proj = torch.nn.Linear(4, 10, bias=False)
        self.down_proj = torch.nn.Linear(10, 4, bias=False)
        self.act_fn = torch.nn.SiLU()

    def forward(self, hidden_state):
        activation = self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(
            hidden_state
        )
        return self.down_proj(activation)


class _ToyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _ToyMLP()


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_ToyBlock(), _ToyBlock()])


class MLPChannelConsistencyControllerTest(unittest.TestCase):
    @staticmethod
    def _collect_score(
        controller: MLPChannelConsistencyController,
        activation: torch.Tensor,
        *,
        down_weight: torch.Tensor | None = None,
    ) -> dict[str, float]:
        controller.start_score_collection()
        controller.set_clean()
        controller.set_response_token_mask(
            torch.ones(activation.shape[:-1], dtype=torch.bool)
        )
        observed = controller.apply(0, activation, down_weight=down_weight)
        observed.sum().backward()
        controller.end_batch()
        return controller.finish_score_collection()

    def test_resample_masks_exact_fraction_per_layer_and_is_reproducible(self):
        first = MLPChannelConsistencyController(
            num_layers=3,
            intermediate_size=20,
            mask_ratio=0.10,
            random_seed=7,
        )
        second = MLPChannelConsistencyController(
            num_layers=3,
            intermediate_size=20,
            mask_ratio=0.10,
            random_seed=7,
        )
        first_metrics = first.resample()
        second.resample()

        torch.testing.assert_close(first.keep_mask, second.keep_mask)
        self.assertEqual((~first.keep_mask).sum(dim=-1).tolist(), [2, 2, 2])
        self.assertEqual(first_metrics["mlp_consistency/realized_mask_fraction"], 0.10)
        self.assertEqual(first.mask_version, 1)

        old_mask = first.keep_mask.clone()
        first.resample()
        self.assertFalse(torch.equal(old_mask, first.keep_mask))

    def test_clean_is_identity_and_masked_is_literal_zero_one(self):
        controller = MLPChannelConsistencyController(
            num_layers=1,
            intermediate_size=10,
            mask_ratio=0.20,
            random_seed=3,
        )
        activation = torch.randn(2, 4, 10)
        self.assertIs(controller.apply(0, activation), activation)

        controller.resample()
        controller.set_masked()
        masked = controller.apply(0, activation)
        expected = activation * controller.keep_mask[0].to(activation.dtype)
        torch.testing.assert_close(masked, expected)
        self.assertEqual(int((masked == 0).all(dim=(0, 1)).sum().item()), 2)

    def test_checkpoint_round_trip_preserves_next_random_mask(self):
        original = MLPChannelConsistencyController(
            num_layers=2,
            intermediate_size=20,
            mask_ratio=0.10,
            random_seed=11,
        )
        original.resample()
        state = original.state_dict()

        restored = MLPChannelConsistencyController(
            num_layers=2,
            intermediate_size=20,
            mask_ratio=0.10,
            random_seed=11,
        )
        restored.load_state_dict(state)
        torch.testing.assert_close(original.keep_mask, restored.keep_mask)
        original.resample()
        restored.resample()
        torch.testing.assert_close(original.keep_mask, restored.keep_mask)

    def test_hf_patch_changes_only_the_masked_route(self):
        torch.manual_seed(0)
        model = _ToyModel()
        controller = MLPChannelConsistencyController(
            num_layers=2,
            intermediate_size=10,
            mask_ratio=0.20,
            random_seed=13,
        )
        hidden = torch.randn(2, 3, 4)
        baseline = model.layers[0].mlp(hidden)
        installed = install_hf_mlp_consistency_mask(model, controller)
        self.assertEqual(installed, ["layers.0.mlp", "layers.1.mlp"])
        torch.testing.assert_close(model.layers[0].mlp(hidden), baseline)

        controller.resample()
        controller.set_masked()
        masked = model.layers[0].mlp(hidden)
        self.assertFalse(torch.equal(masked, baseline))
        controller.set_clean()
        torch.testing.assert_close(model.layers[0].mlp(hidden), baseline)

    def test_gradient_activation_score_reuses_clean_backward(self):
        controller = MLPChannelConsistencyController(
            num_layers=1,
            intermediate_size=4,
            mask_ratio=0.25,
            selection_strategy=HARD_TOP_SELECTION,
            score_method=GRADIENT_ACTIVATION_SCORE,
        )
        activation = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]],
            requires_grad=True,
        )
        metrics = self._collect_score(controller, activation)

        torch.testing.assert_close(
            controller.selection_score, torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        )
        self.assertEqual(metrics["mlp_consistency/score_response_samples"], 1.0)
        self.assertEqual(metrics["mlp_consistency/score_response_tokens"], 2.0)
        controller.resample()
        self.assertEqual(torch.nonzero(~controller.keep_mask[0]).item(), 3)

    def test_output_contribution_includes_down_projection_column_norm(self):
        controller = MLPChannelConsistencyController(
            num_layers=1,
            intermediate_size=4,
            mask_ratio=0.25,
            selection_strategy=HARD_TOP_SELECTION,
            score_method=OUTPUT_CONTRIBUTION_SCORE,
        )
        activation = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]],
            requires_grad=True,
        )
        down_weight = torch.tensor(
            [[1.0, 1.0, 1.0, 2.0], [0.0, 0.0, 0.0, 0.0]]
        )
        self._collect_score(controller, activation, down_weight=down_weight)

        torch.testing.assert_close(
            controller.selection_score, torch.tensor([[1.0, 2.0, 3.0, 8.0]])
        )

    def test_relative_activation_uses_prior_clean_update_as_baseline(self):
        controller = MLPChannelConsistencyController(
            num_layers=1,
            intermediate_size=4,
            mask_ratio=0.25,
            selection_strategy=SOFT_TOP_SELECTION,
            score_method=RELATIVE_ACTIVATION_SCORE,
            activation_ema_beta=0.0,
        )
        first = torch.ones((1, 1, 4), requires_grad=True)
        self._collect_score(controller, first)
        self.assertTrue(controller.activation_ema_initialized)
        self.assertFalse(controller.selection_score_initialized)

        second = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0]]], requires_grad=True
        )
        self._collect_score(controller, second)
        torch.testing.assert_close(
            controller.selection_score, torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        )

    def test_soft_top_is_seeded_and_uses_an_exact_quota(self):
        first = MLPChannelConsistencyController(
            num_layers=2,
            intermediate_size=20,
            mask_ratio=0.10,
            selection_strategy=SOFT_TOP_SELECTION,
            score_method=GRADIENT_ACTIVATION_SCORE,
            random_seed=17,
        )
        second = MLPChannelConsistencyController(
            num_layers=2,
            intermediate_size=20,
            mask_ratio=0.10,
            selection_strategy=SOFT_TOP_SELECTION,
            score_method=GRADIENT_ACTIVATION_SCORE,
            random_seed=17,
        )
        score = torch.arange(20, dtype=torch.float32).repeat(2, 1)
        for controller in (first, second):
            controller._update_selection_score(score)
            metrics = controller.resample()

        torch.testing.assert_close(first.keep_mask, second.keep_mask)
        self.assertEqual((~first.keep_mask).sum(dim=-1).tolist(), [2, 2])
        self.assertEqual(metrics["mlp_consistency/selection_used_score"], 1.0)

    def test_selection_and_score_must_be_configured_together(self):
        with self.assertRaisesRegex(ValueError, "requires score_method=none"):
            MLPChannelConsistencyController(
                num_layers=1,
                intermediate_size=10,
                selection_strategy="random",
                score_method=GRADIENT_ACTIVATION_SCORE,
            )
        with self.assertRaisesRegex(ValueError, "score-based selection requires"):
            MLPChannelConsistencyController(
                num_layers=1,
                intermediate_size=10,
                selection_strategy=SOFT_TOP_SELECTION,
                score_method="none",
            )

    def test_updated_fraction_score_selects_most_changed_channel(self):
        controller = MLPChannelConsistencyController(
            num_layers=1,
            intermediate_size=4,
            mask_ratio=0.25,
            selection_strategy=HARD_TOP_SELECTION,
            score_method=UPDATED_FRACTION_SCORE,
        )
        metrics = controller.update_updated_fraction_score(
            torch.tensor([[0.0, 0.25, 0.75, 0.5]])
        )

        self.assertEqual(metrics["mlp_consistency/score_is_updated_fraction"], 1.0)
        controller.resample()
        self.assertEqual(torch.nonzero(~controller.keep_mask[0]).item(), 2)

    def test_soft_top_assigns_equal_percentile_weight_to_tied_scores(self):
        rank = MLPChannelConsistencyController._percentile_rank(
            torch.tensor([0.0, 0.0, 1.0, 1.0])
        )
        torch.testing.assert_close(
            rank,
            torch.tensor([1.0 / 6.0, 1.0 / 6.0, 5.0 / 6.0, 5.0 / 6.0]),
        )


if __name__ == "__main__":
    unittest.main()
