from __future__ import annotations

import unittest
import warnings

import numpy as np
import torch
from torch import nn

from recipe.mlp_channel_antithetic.intervention import (
    NEGATIVE_ROUTE,
    NEUTRAL_ROUTE,
    POSITIVE_ROUTE,
    MLPChannelAntitheticController,
    install_hf_mlp_intervention,
    install_vllm_mlp_intervention,
)
from recipe.mlp_channel_antithetic.routing import (
    assign_antithetic_routes,
    copy_prompt_uids_for_generation,
)


class _DenseMLP(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.gate_proj = nn.Linear(width, width, bias=False)
        self.up_proj = nn.Linear(width, width, bias=False)
        self.down_proj = nn.Linear(width, width, bias=False)
        self.act_fn = nn.Identity()
        with torch.no_grad():
            self.gate_proj.weight.copy_(torch.eye(width))
            self.up_proj.weight.copy_(torch.eye(width))
            self.down_proj.weight.copy_(torch.eye(width))

    def forward(self, hidden_state):
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state)
        )


class _Layer(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.mlp = _DenseMLP(width)


class _InnerModel(nn.Module):
    def __init__(self, layers: int, width: int):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(width) for _ in range(layers)])


class _Model(nn.Module):
    def __init__(self, layers: int, width: int):
        super().__init__()
        self.model = _InnerModel(layers, width)


class _VLLMMLP(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.gate_up_proj = nn.Linear(width, width, bias=False)
        self.down_proj = nn.Linear(width, width, bias=False)
        self.act_fn = nn.Identity()
        with torch.no_grad():
            self.gate_up_proj.weight.copy_(torch.eye(width))
            self.down_proj.weight.copy_(torch.eye(width))


class _VLLMLayer(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.mlp = _VLLMMLP(width)


class _VLLMModel(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_VLLMLayer(width)])


class AntitheticControllerTest(unittest.TestCase):
    def test_gains_are_positive_and_exactly_complementary(self) -> None:
        controller = MLPChannelAntitheticController(
            num_layers=3,
            intermediate_size=8,
            perturbation_strength=0.2,
            random_seed=9,
        )
        for layer_idx in range(3):
            positive = controller.gain(layer_idx, POSITIVE_ROUTE)
            negative = controller.gain(layer_idx, NEGATIVE_ROUTE)
            self.assertTrue(torch.all(positive > 0))
            self.assertTrue(torch.all(negative > 0))
            torch.testing.assert_close((positive + negative) / 2, torch.ones(8))
            self.assertEqual(int((controller.direction[layer_idx] > 0).sum()), 4)
            self.assertEqual(int((controller.direction[layer_idx] < 0).sum()), 4)

    def test_direction_is_step_deterministic_and_changes(self) -> None:
        first = MLPChannelAntitheticController(
            num_layers=2, intermediate_size=10, random_seed=17
        )
        second = MLPChannelAntitheticController(
            num_layers=2, intermediate_size=10, random_seed=17
        )
        first.refresh_direction(version=12)
        second.refresh_direction(version=12)
        torch.testing.assert_close(first.direction, second.direction)
        old = first.direction.clone()
        self.assertFalse(first.refresh_direction(version=12))
        self.assertTrue(first.refresh_direction(version=13))
        self.assertFalse(torch.equal(old, first.direction))

    def test_hf_positive_negative_average_recovers_neutral_output(self) -> None:
        model = _Model(layers=2, width=6)
        controller = MLPChannelAntitheticController(
            num_layers=2,
            intermediate_size=6,
            perturbation_strength=0.3,
            random_seed=2,
        )
        self.assertEqual(
            install_hf_mlp_intervention(model, controller),
            ["model.layers.0.mlp", "model.layers.1.mlp"],
        )
        hidden = torch.tensor([[1.0, -2.0, 0.5, 3.0, -1.5, 2.5]])
        for layer in model.model.layers:
            controller.set_route(NEUTRAL_ROUTE)
            neutral = layer.mlp(hidden)
            controller.set_route(POSITIVE_ROUTE)
            positive = layer.mlp(hidden)
            controller.set_route(NEGATIVE_ROUTE)
            negative = layer.mlp(hidden)
            torch.testing.assert_close((positive + negative) / 2, neutral)

    def test_state_round_trip_and_version_validation(self) -> None:
        source = MLPChannelAntitheticController(
            num_layers=2,
            intermediate_size=8,
            perturbation_strength=0.1,
            random_seed=4,
        )
        source.refresh_direction(version=23)
        target = MLPChannelAntitheticController(
            num_layers=2,
            intermediate_size=8,
            perturbation_strength=0.1,
            random_seed=4,
        )
        target.load_state_dict(source.state_dict())
        self.assertEqual(target.perturbation_version, 23)
        torch.testing.assert_close(target.direction, source.direction)
        target.validate_batch_version(np.array([23, 23], dtype=np.int64))
        with self.assertRaisesRegex(RuntimeError, "version mismatch"):
            target.validate_batch_version(np.array([22], dtype=np.int64))

    def test_version_validation_accepts_read_only_ray_metadata(self) -> None:
        controller = MLPChannelAntitheticController(
            num_layers=1,
            intermediate_size=8,
        )
        versions = np.array([0, 0], dtype=np.int64)
        versions.flags.writeable = False
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            controller.validate_batch_version(versions)

    def test_actor_gain_buffer_is_reused_and_updated_in_place(self) -> None:
        controller = MLPChannelAntitheticController(
            num_layers=1,
            intermediate_size=8,
            perturbation_strength=0.25,
            random_seed=3,
        )
        activation = torch.ones(2, 8)
        controller.set_route(POSITIVE_ROUTE)
        controller.apply(0, activation)
        buffer = next(iter(controller._active_buffers.values()))
        pointer = buffer.data_ptr()
        positive = buffer.clone()
        controller.set_route(NEGATIVE_ROUTE)
        self.assertEqual(buffer.data_ptr(), pointer)
        torch.testing.assert_close((positive + buffer) / 2, torch.ones_like(buffer))

    def test_bfloat16_runtime_buffers_remain_exactly_complementary(self) -> None:
        controller = MLPChannelAntitheticController(
            num_layers=1,
            intermediate_size=8,
            perturbation_strength=0.05,
        )
        activation = torch.ones(1, 8, dtype=torch.bfloat16)
        controller.set_route(POSITIVE_ROUTE)
        positive = controller.apply(0, activation).clone()
        controller.set_route(NEGATIVE_ROUTE)
        negative = controller.apply(0, activation).clone()
        self.assertTrue(torch.equal(positive + negative, torch.full_like(positive, 2)))

    def test_vllm_hook_reuses_registered_gain_buffer(self) -> None:
        model = _VLLMModel(width=8)
        controller = MLPChannelAntitheticController(
            num_layers=1,
            intermediate_size=8,
            perturbation_strength=0.125,
        )
        self.assertEqual(
            install_vllm_mlp_intervention(model, controller),
            ["model.layers.0.mlp"],
        )
        mlp = model.model.layers[0].mlp
        buffer = mlp._buffers["_mlp_channel_antithetic_gain"]
        pointer = buffer.data_ptr()
        hidden = torch.arange(8, dtype=torch.float32).unsqueeze(0)
        controller.set_route(POSITIVE_ROUTE)
        positive = mlp(hidden)
        controller.set_route(NEGATIVE_ROUTE)
        negative = mlp(hidden)
        controller.set_route(NEUTRAL_ROUTE)
        neutral = mlp(hidden)
        self.assertEqual(buffer.data_ptr(), pointer)
        torch.testing.assert_close((positive + negative) / 2, neutral)

    def test_route_assignment_is_balanced_inside_each_uid(self) -> None:
        uids = np.array(["a", "a", "b", "a", "b", "b", "a", "b"], dtype=object)
        routes = assign_antithetic_routes(uids)
        for uid in ("a", "b"):
            selected = routes[uids == uid]
            self.assertEqual(int(np.sum(selected == POSITIVE_ROUTE)), 2)
            self.assertEqual(int(np.sum(selected == NEGATIVE_ROUTE)), 2)
        np.testing.assert_array_equal(uids, ["a", "a", "b", "a", "b", "b", "a", "b"])

    def test_prompt_uids_are_copied_for_sync_generation(self) -> None:
        uids = np.array(["a", "b"], dtype=object)
        copied = copy_prompt_uids_for_generation(uids, expected_size=2)
        np.testing.assert_array_equal(copied, uids)
        self.assertFalse(np.shares_memory(copied, uids))

    def test_prompt_uid_copy_rejects_wrong_batch_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "matching the generation batch size"):
            copy_prompt_uids_for_generation(
                np.array(["a", "b"], dtype=object), expected_size=3
            )

    def test_route_assignment_rejects_unpaired_prompt(self) -> None:
        with self.assertRaisesRegex(ValueError, "even count"):
            assign_antithetic_routes(np.array(["a", "a", "a"], dtype=object))


if __name__ == "__main__":
    unittest.main()
