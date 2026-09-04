"""CPU tests for structured post-AdamW channel updates."""

from __future__ import annotations

import copy
import unittest

import torch
from torch import nn

from .optimizer import ChannelRelativeUpdateAdamW
from .relative_update import MLPChannelRelativeUpdateController


class _ToyMLP(nn.Module):
    def __init__(self, hidden_size: int = 2, intermediate_size: int = 3) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)


class _ToyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _ToyMLP()
        self.self_attn = nn.Linear(2, 2, bias=False)


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_ToyBlock()])


class ChannelRelativeUpdateAdamWTest(unittest.TestCase):
    def _controller(self, *, ratio: float = 1.0) -> MLPChannelRelativeUpdateController:
        return MLPChannelRelativeUpdateController(
            num_layers=1,
            intermediate_size=3,
            history_ema_beta=0.0,
            history_power=1.0,
            history_floor_ratio=0.0,
            multiplier_ratio_cap=ratio,
            warmup_steps=0,
        )

    def test_ratio_one_is_numerically_equivalent_to_torch_adamw(self) -> None:
        torch.manual_seed(7)
        reference_model = _ToyModel()
        recipe_model = copy.deepcopy(reference_model)
        reference = torch.optim.AdamW(
            reference_model.parameters(),
            lr=3e-3,
            betas=(0.8, 0.95),
            eps=1e-7,
            weight_decay=0.02,
        )
        recipe = ChannelRelativeUpdateAdamW(
            recipe_model.parameters(),
            lr=3e-3,
            betas=(0.8, 0.95),
            eps=1e-7,
            weight_decay=0.02,
        )
        recipe.configure_channel_updates(
            controller=self._controller(ratio=1.0),
            named_parameters=recipe_model.named_parameters(),
        )

        for step_idx in range(3):
            torch.manual_seed(100 + step_idx)
            gradients = [torch.randn_like(parameter) for parameter in reference_model.parameters()]
            for parameter, gradient in zip(reference_model.parameters(), gradients):
                parameter.grad = gradient.clone()
            for parameter, gradient in zip(recipe_model.parameters(), gradients):
                parameter.grad = gradient.clone()
            reference.step()
            recipe.step()

        for actual, expected in zip(recipe_model.parameters(), reference_model.parameters()):
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    def test_complete_channel_uses_one_multiplier_across_three_projections(self) -> None:
        model = _ToyModel()
        for parameter in model.parameters():
            parameter.data.fill_(1.0)
        controller = self._controller(ratio=10.0)
        controller.history_relative_update_sq[0] = torch.tensor([1e4, 1.0, 1e-4])
        controller.step_count = 1
        optimizer = ChannelRelativeUpdateAdamW(
            model.parameters(),
            lr=0.01,
            betas=(0.0, 0.0),
            weight_decay=0.0,
        )
        installed = optimizer.configure_channel_updates(
            controller=controller,
            named_parameters=model.named_parameters(),
        )
        self.assertEqual(len(installed), 3)
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)

        optimizer.step()

        gate_delta = before["layers.0.mlp.gate_proj.weight"] - model.layers[0].mlp.gate_proj.weight
        up_delta = before["layers.0.mlp.up_proj.weight"] - model.layers[0].mlp.up_proj.weight
        down_delta = before["layers.0.mlp.down_proj.weight"] - model.layers[0].mlp.down_proj.weight
        torch.testing.assert_close(gate_delta, up_delta)
        torch.testing.assert_close(gate_delta[:, 0], down_delta[0, :])
        torch.testing.assert_close(gate_delta[:, 1], down_delta[1, :])
        self.assertAlmostEqual(
            controller.last_metrics["mlp_relative_update/update_energy_ratio"],
            1.0,
            places=6,
        )

    def test_incomplete_mlp_group_is_rejected(self) -> None:
        model = _ToyModel()
        del model.layers[0].mlp.down_proj
        optimizer = ChannelRelativeUpdateAdamW(model.parameters(), lr=1e-3)
        with self.assertRaisesRegex(RuntimeError, "missing projections"):
            optimizer.configure_channel_updates(
                controller=self._controller(),
                named_parameters=model.named_parameters(),
            )


if __name__ == "__main__":
    unittest.main()
