"""CPU tests for hard per-layer MLP-channel consistency masks."""

from __future__ import annotations

import unittest

import torch

from .intervention import (
    MLPChannelConsistencyController,
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


if __name__ == "__main__":
    unittest.main()
