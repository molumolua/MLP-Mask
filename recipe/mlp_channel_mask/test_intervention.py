"""CPU unit tests for the recipe's mask/relative-activation primitives.

Run without the training stack installed:
    python3 -m unittest recipe.mlp_channel_mask.test_intervention
"""

from __future__ import annotations

import unittest
from types import ModuleType
from unittest import mock

import torch
from torch import nn

from .intervention import (
    CLEAN_ROUTE,
    GLOBAL_RANDOM_SCOPE,
    MASKED_ROUTE,
    MLPChannelInterventionController,
    RANDOM_SELECTION,
    WEIGHTED_RANDOM_SELECTION,
    install_hf_mlp_intervention,
    install_vllm_class_intervention,
    install_vllm_mlp_intervention,
)


class _DenseHFMLP(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(width, width, bias=False)
        self.up_proj = nn.Linear(width, width, bias=False)
        self.down_proj = nn.Linear(width, width, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class _FusedAct(nn.Module):
    def forward(self, gate_up):
        gate, up = gate_up.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up


class _DenseVLLMMLP(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(width, 2 * width, bias=False)
        self.down_proj = nn.Linear(width, width, bias=False)
        self.act_fn = _FusedAct()

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_up_proj(hidden_state)))


class _FakeQwenMLP(_DenseVLLMMLP):
    def __init__(self, width: int, *, prefix: str) -> None:
        super().__init__(width)
        self.prefix = prefix


class _Block(nn.Module):
    def __init__(self, mlp: nn.Module) -> None:
        super().__init__()
        self.mlp = mlp


class _FSDPWrapper(nn.Module):
    """Dependency-free stand-in for FSDP1's named-module hierarchy."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self._fsdp_wrapped_module = module

    def forward(self, *args, **kwargs):
        return self._fsdp_wrapped_module(*args, **kwargs)


class _Backbone(nn.Module):
    def __init__(self, mlp_factory, layers: int, width: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Block(mlp_factory(width)) for _ in range(layers)])


class _Model(nn.Module):
    def __init__(self, mlp_factory, layers: int, width: int) -> None:
        super().__init__()
        self.model = _Backbone(mlp_factory, layers, width)


class MLPChannelInterventionTest(unittest.TestCase):
    def _controller(self) -> MLPChannelInterventionController:
        return MLPChannelInterventionController(
            num_layers=2,
            intermediate_size=10,
            mask_ratio=0.20,
            activation_ema_beta=0.0,
        )

    def _collect(self, controller, values: torch.Tensor) -> None:
        controller.set_route(CLEAN_ROUTE, collect_activation=True)
        controller.set_response_token_mask(torch.ones((1, 2)))
        loss = torch.zeros(())
        for layer_idx in range(controller.num_layers):
            activation = values.reshape(1, 1, -1).expand(1, 2, -1).clone().requires_grad_(True)
            loss = loss + controller.apply(layer_idx, activation).sum()
        loss.backward()
        controller.end_batch()

    def _collect_batch(
        self,
        controller: MLPChannelInterventionController,
        activation: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> None:
        controller.set_route(CLEAN_ROUTE, collect_activation=True)
        controller.set_response_token_mask(response_mask)
        loss = torch.zeros(())
        for layer_idx in range(controller.num_layers):
            layer_activation = activation.clone().requires_grad_(True)
            loss = loss + controller.apply(layer_idx, layer_activation).sum()
        loss.backward()
        controller.end_batch()

    def test_top_ratio_uses_relative_not_absolute_activation(self) -> None:
        controller = self._controller()
        baseline = torch.tensor([100.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        current = torch.tensor([110.0, 20.0, 19.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])

        self._collect(controller, baseline)
        first_result = controller.refresh_mask()
        self.assertEqual(controller.current_masked_channels, 0)
        self.assertEqual(first_result.metrics["mlp_relative_activation/initialized"], 0.0)

        self._collect(controller, current)
        result = controller.refresh_mask()

        self.assertEqual(controller.mask_version, 2)
        self.assertEqual((~controller.keep_mask).sum(dim=-1).tolist(), [2, 2])
        self.assertEqual(torch.nonzero(~controller.keep_mask[0]).flatten().tolist(), [1, 2])
        self.assertAlmostEqual(controller.relative_activation_score[0, 0].item(), 0.1)
        self.assertAlmostEqual(controller.relative_activation_score[0, 1].item(), 1.0)
        self.assertAlmostEqual(controller.relative_activation_score[0, 2].item(), 0.9)
        self.assertEqual(result.metrics["mlp_mask/current_channels"], 4.0)
        self.assertEqual(result.metrics["mlp_mask/ever_unique_channels"], 4.0)

        controller.set_route(MASKED_ROUTE)
        masked = controller.apply(0, torch.ones((1, 10)))
        self.assertEqual(masked[0].tolist(), [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    def test_first_collection_only_initializes_activation_ema(self) -> None:
        controller = self._controller()
        baseline = torch.arange(1, 11, dtype=torch.float32)

        self._collect(controller, baseline)
        result = controller.refresh_mask()

        self.assertTrue(torch.equal(controller.activation_ema[0], baseline))
        self.assertFalse(controller.relative_activation_initialized)
        self.assertEqual(controller.current_masked_channels, 0)
        self.assertEqual(result.metrics["mlp_mask/initialized"], 0.0)
        self.assertEqual(result.metrics["mlp_mask/current_channels"], 0.0)

    def test_zero_ema_beta_scores_before_replacing_activation_baseline(self) -> None:
        controller = self._controller()
        first = torch.arange(1, 11, dtype=torch.float32)
        second = torch.flip(first, dims=(0,))

        self._collect(controller, first)
        controller.refresh_mask()
        self.assertTrue(torch.equal(controller.activation_ema[0], first))

        self._collect(controller, second)
        result = controller.refresh_mask()
        expected_relative = (second - first) / first
        self.assertTrue(torch.allclose(controller.relative_activation_score[0], expected_relative))
        self.assertTrue(torch.equal(controller.activation_ema[0], second))
        self.assertEqual(result.metrics["mlp_activation/ema_beta"], 0.0)

    def test_relative_score_uses_previous_ema_then_updates_with_beta(self) -> None:
        controller = MLPChannelInterventionController(
            num_layers=2,
            intermediate_size=10,
            mask_ratio=0.20,
            activation_ema_beta=0.5,
        )
        baseline = torch.full((10,), 10.0)
        current = torch.full((10,), 20.0)

        self._collect(controller, baseline)
        controller.refresh_mask()
        self._collect(controller, current)
        controller.refresh_mask()

        self.assertTrue(
            torch.allclose(controller.relative_activation_score[0], torch.ones(10))
        )
        self.assertTrue(
            torch.allclose(controller.activation_ema[0], torch.full((10,), 15.0))
        )

    def test_relative_score_is_finite_when_previous_activation_is_zero(self) -> None:
        controller = MLPChannelInterventionController(
            num_layers=2,
            intermediate_size=10,
            mask_ratio=0.20,
            activation_ema_beta=0.0,
            relative_activation_epsilon=0.5,
        )
        baseline = torch.zeros(10)
        current = torch.zeros(10)
        current[0] = 1.0

        self._collect(controller, baseline)
        controller.refresh_mask()
        self._collect(controller, current)
        controller.refresh_mask()

        self.assertTrue(torch.isfinite(controller.relative_activation_score).all())
        self.assertEqual(controller.relative_activation_score[0, 0].item(), 2.0)
        self.assertEqual(controller.relative_activation_score[0, 1].item(), 0.0)

    def test_activation_baseline_averages_per_sample_rms_like_rarity(self) -> None:
        controller = self._controller()
        activation = torch.zeros((2, 9, 10), dtype=torch.float32)
        response_mask = torch.ones((2, 9), dtype=torch.float32)
        response_mask[0, 1:] = 0
        activation[0, 0, 0] = 10.0
        activation[:, :, 1] = 4.0

        self._collect_batch(controller, activation, response_mask)
        controller.refresh_mask()

        # Channel 0 has per-sample RMS values [10, 0], while channel 1 has [4, 4].
        # The baseline is their sample mean: [5, 4], not a token-weighted global RMS.
        self.assertEqual(controller.activation_ema[0, :2].tolist(), [5.0, 4.0])

    def test_default_ten_percent_masks_each_block(self) -> None:
        controller = MLPChannelInterventionController(
            num_layers=3,
            intermediate_size=10,
            mask_ratio=0.10,
            activation_ema_beta=0.0,
        )
        controller.set_route(CLEAN_ROUTE, collect_activation=True)
        controller.set_response_token_mask(torch.ones((1, 1)))
        loss = torch.zeros(())
        for layer_idx in range(3):
            activation = torch.arange(10, dtype=torch.float32).reshape(1, 1, -1).requires_grad_(True)
            loss = loss + controller.apply(layer_idx, activation).sum()
        loss.backward()
        controller.end_batch()
        controller.refresh_mask()
        self._collect(controller, torch.arange(10, 0, -1, dtype=torch.float32))
        controller.refresh_mask()
        self.assertEqual((~controller.keep_mask).sum(dim=-1).tolist(), [1, 1, 1])

    def test_random_selection_is_exact_and_reproducible_without_activation(self) -> None:
        def random_controller():
            return MLPChannelInterventionController(
                num_layers=3,
                intermediate_size=20,
                mask_ratio=0.10,
                selection_strategy=RANDOM_SELECTION,
                random_seed=123,
            )

        first = random_controller()
        second = random_controller()
        first_result = first.refresh_mask()
        second.refresh_mask()

        self.assertEqual(first.mask_version, 1)
        self.assertEqual((~first.keep_mask).sum(dim=-1).tolist(), [2, 2, 2])
        self.assertTrue(torch.equal(first.keep_mask, second.keep_mask))
        self.assertEqual(first_result.metrics["mlp_mask/selection_is_random"], 1.0)
        self.assertEqual(first_result.metrics["mlp_mask/random_scope_is_global"], 0.0)
        self.assertNotIn("mlp_activation/response_tokens", first_result.metrics)

        previous_mask = first.keep_mask.clone()
        first.refresh_mask()
        second.refresh_mask()
        self.assertTrue(torch.equal(first.keep_mask, second.keep_mask))
        self.assertFalse(torch.equal(first.keep_mask, previous_mask))

    def test_global_random_selection_has_exact_total_and_variable_layer_counts(self) -> None:
        def global_controller():
            return MLPChannelInterventionController(
                num_layers=4,
                intermediate_size=5,
                mask_ratio=0.20,
                selection_strategy=RANDOM_SELECTION,
                random_seed=1,
                random_scope=GLOBAL_RANDOM_SCOPE,
            )

        first = global_controller()
        second = global_controller()
        result = first.refresh_mask()
        second.refresh_mask()

        self.assertEqual(first.current_masked_channels, 4)
        self.assertEqual((~first.keep_mask).sum(dim=-1).tolist(), [1, 1, 2, 0])
        self.assertTrue(torch.equal(first.keep_mask, second.keep_mask))
        self.assertEqual(result.metrics["mlp_mask/current_fraction"], 0.20)
        self.assertEqual(result.metrics["mlp_mask/random_scope_is_global"], 1.0)

        per_layer = MLPChannelInterventionController(
            num_layers=4,
            intermediate_size=5,
            mask_ratio=0.20,
            selection_strategy=RANDOM_SELECTION,
            random_seed=1,
        )
        with self.assertRaisesRegex(ValueError, "random_scope"):
            per_layer.load_state_dict(first.state_dict())

    def test_legacy_random_checkpoint_remains_compatible(self) -> None:
        controller = MLPChannelInterventionController(
            num_layers=2,
            intermediate_size=10,
            mask_ratio=0.20,
            selection_strategy=RANDOM_SELECTION,
            random_seed=123,
        )
        controller.refresh_mask()
        legacy_state = controller.state_dict()
        for key in (
            "score_type",
            "activation_ema_beta",
            "relative_activation_epsilon",
            "activation_ema",
            "relative_activation_score",
            "activation_ema_initialized",
            "relative_activation_initialized",
        ):
            legacy_state.pop(key)
        legacy_state["ema_saliency"] = torch.zeros((2, 10))

        restored = MLPChannelInterventionController(
            num_layers=2,
            intermediate_size=10,
            mask_ratio=0.20,
            selection_strategy=RANDOM_SELECTION,
            random_seed=123,
        )
        restored.load_state_dict(legacy_state)

        self.assertTrue(torch.equal(restored.keep_mask, controller.keep_mask))
        self.assertEqual(restored.mask_version, controller.mask_version)

    def test_weighted_random_is_exact_reproducible_and_relative_activation_biased(self) -> None:
        def weighted_controller():
            return MLPChannelInterventionController(
                num_layers=2,
                intermediate_size=20,
                mask_ratio=0.10,
                activation_ema_beta=0.0,
                selection_strategy=WEIGHTED_RANDOM_SELECTION,
                random_seed=7,
                weighted_max_ratio=4.0,
                weighted_rank_power=2.0,
            )

        first = weighted_controller()
        second = weighted_controller()
        first_uniform = first.refresh_mask()
        second.refresh_mask()
        self.assertEqual((~first.keep_mask).sum(dim=-1).tolist(), [2, 2])
        self.assertTrue(torch.equal(first.keep_mask, second.keep_mask))
        self.assertEqual(
            first_uniform.metrics["mlp_mask/weighted_used_relative_activation"], 0.0
        )

        baseline = torch.ones(20, dtype=torch.float32)
        current = torch.cat([torch.ones(10), torch.full((10,), 2.0)])
        self._collect(first, baseline)
        self._collect(second, baseline)
        first.refresh_mask()
        second.refresh_mask()
        self._collect(first, current)
        self._collect(second, current)
        weighted_result = first.refresh_mask()
        second.refresh_mask()
        self.assertTrue(torch.equal(first.keep_mask, second.keep_mask))
        self.assertEqual(
            weighted_result.metrics["mlp_mask/weighted_used_relative_activation"], 1.0
        )
        self.assertEqual(weighted_result.metrics["mlp_mask/weighted_max_ratio"], 4.0)
        self.assertEqual(weighted_result.metrics["mlp_mask/weighted_rank_power"], 2.0)

        selected_counts = torch.zeros(20, dtype=torch.int64)
        for _ in range(200):
            first.refresh_mask()
            self.assertEqual((~first.keep_mask).sum(dim=-1).tolist(), [2, 2])
            selected_counts.add_((~first.keep_mask).sum(dim=0))
        self.assertGreater(int(selected_counts[10:].sum()), int(selected_counts[:10].sum()))

        restored = weighted_controller()
        restored.load_state_dict(first.state_dict())
        self.assertTrue(torch.equal(restored.keep_mask, first.keep_mask))
        self.assertTrue(torch.equal(restored.activation_ema, first.activation_ema))
        self.assertTrue(
            torch.equal(restored.relative_activation_score, first.relative_activation_score)
        )

        with self.assertRaisesRegex(ValueError, "random_scope=global"):
            MLPChannelInterventionController(
                num_layers=2,
                intermediate_size=20,
                selection_strategy=WEIGHTED_RANDOM_SELECTION,
                random_scope=GLOBAL_RANDOM_SCOPE,
            )

    def test_unique_history_and_checkpoint_round_trip(self) -> None:
        controller = self._controller()
        self._collect(controller, torch.ones(10, dtype=torch.float32))
        controller.refresh_mask()
        second = torch.ones(10, dtype=torch.float32)
        second[-2:] = 2.0
        self._collect(controller, second)
        controller.refresh_mask()
        third = second.clone()
        third[:2] = 2.0
        self._collect(controller, third)
        result = controller.refresh_mask()

        self.assertEqual(controller.current_masked_channels, 4)
        self.assertEqual(controller.ever_masked_channels, 8)
        self.assertEqual(result.metrics["mlp_mask/new_unique_channels"], 4.0)
        self.assertEqual(result.metrics["mlp_mask/turnover_fraction"], 1.0)

        restored = self._controller()
        restored.load_state_dict(controller.state_dict())
        self.assertTrue(torch.equal(restored.keep_mask, controller.keep_mask))
        self.assertTrue(torch.equal(restored.ever_masked, controller.ever_masked))
        self.assertTrue(torch.equal(restored.activation_ema, controller.activation_ema))
        self.assertTrue(
            torch.equal(restored.relative_activation_score, controller.relative_activation_score)
        )
        self.assertEqual(restored.mask_version, 3)

    def test_dense_hf_and_vllm_instance_layouts_are_patched(self) -> None:
        hf_controller = self._controller()
        hf_model = _Model(_DenseHFMLP, layers=2, width=10)
        self.assertEqual(
            install_hf_mlp_intervention(hf_model, hf_controller),
            ["model.layers.0.mlp", "model.layers.1.mlp"],
        )

        vllm_controller = self._controller()
        vllm_model = _Model(_DenseVLLMMLP, layers=2, width=10)
        self.assertEqual(
            install_vllm_mlp_intervention(vllm_model, vllm_controller),
            ["model.layers.0.mlp", "model.layers.1.mlp"],
        )
        first_mlp = vllm_model.model.layers[0].mlp
        self.assertIn("_mlp_channel_active_mask", dict(first_mlp.named_buffers()))
        active_buffer_count = len(vllm_controller._active_buffers)
        output = first_mlp(torch.randn(2, 10))
        self.assertEqual(tuple(output.shape), (2, 10))
        self.assertEqual(len(vllm_controller._active_buffers), active_buffer_count)
        self.assertIsNone(first_mlp.forward.__func__.__closure__)

        vllm_controller.keep_mask[0, -2:] = False
        vllm_controller.set_route(MASKED_ROUTE)
        self.assertTrue(
            torch.equal(
                first_mlp._mlp_channel_active_mask,
                torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 0, 0], dtype=torch.float32),
            )
        )

    def test_hf_mlp_is_found_inside_fsdp_wrapped_decoder_layers(self) -> None:
        controller = self._controller()
        model = _Model(_DenseHFMLP, layers=2, width=10)
        model.model.layers = nn.ModuleList([_FSDPWrapper(layer) for layer in model.model.layers])

        self.assertEqual(
            install_hf_mlp_intervention(model, controller),
            [
                "model.layers.0._fsdp_wrapped_module.mlp",
                "model.layers.1._fsdp_wrapped_module.mlp",
            ],
        )
        output = model.model.layers[0]._fsdp_wrapped_module.mlp(torch.randn(2, 10))
        self.assertEqual(tuple(output.shape), (2, 10))

    def test_vllm_class_patch_registers_mask_before_first_forward(self) -> None:
        vllm = ModuleType("vllm")
        model_executor = ModuleType("vllm.model_executor")
        models = ModuleType("vllm.model_executor.models")
        qwen2 = ModuleType("vllm.model_executor.models.qwen2")
        qwen3 = ModuleType("vllm.model_executor.models.qwen3")
        qwen2.Qwen2MLP = _FakeQwenMLP
        qwen3.Qwen3MLP = _FakeQwenMLP
        fake_modules = {
            "vllm": vllm,
            "vllm.model_executor": model_executor,
            "vllm.model_executor.models": models,
            "vllm.model_executor.models.qwen2": qwen2,
            "vllm.model_executor.models.qwen3": qwen3,
        }

        controller = self._controller()
        with mock.patch.dict("sys.modules", fake_modules):
            self.assertEqual(install_vllm_class_intervention(controller), ["_FakeQwenMLP"])
            mlps = [
                _FakeQwenMLP(10, prefix=f"model.layers.{layer_idx}.mlp")
                for layer_idx in range(controller.num_layers)
            ]

        mlp = mlps[0]
        model = _Model(_DenseVLLMMLP, layers=2, width=10)
        for layer_idx, patched_mlp in enumerate(mlps):
            model.model.layers[layer_idx].mlp = patched_mlp
        # This is the post-construction walk performed after vLLM has entered
        # sleep mode.  It must not refresh already registered CUDA buffers.
        with mock.patch.object(
            controller,
            "_copy_route_to_buffer",
            side_effect=AssertionError("post-build validation rewrote an active buffer"),
        ):
            self.assertEqual(
                install_vllm_mlp_intervention(model, controller),
                ["model.layers.0.mlp", "model.layers.1.mlp"],
            )

        self.assertIn("_mlp_channel_active_mask", dict(mlp.named_buffers()))
        active_buffer_count = len(controller._active_buffers)
        output = mlp(torch.randn(2, 10))
        self.assertEqual(tuple(output.shape), (2, 10))
        self.assertEqual(len(controller._active_buffers), active_buffer_count)
        self.assertIsNone(mlp.forward.__func__.__closure__)

    def test_repeat_registration_of_same_vllm_buffer_is_read_only(self) -> None:
        controller = self._controller()
        buffer = torch.ones(10)
        controller.register_active_buffer(0, buffer)

        with mock.patch.object(
            controller,
            "_copy_route_to_buffer",
            side_effect=AssertionError("repeat registration rewrote the active buffer"),
        ):
            returned = controller.register_active_buffer(0, buffer)

        self.assertIs(returned, buffer)

    def test_route_updates_are_deferred_while_vllm_buffers_are_asleep(self) -> None:
        controller = self._controller()
        buffer = controller.register_active_buffer(0, torch.ones(10))
        controller.keep_mask[0, -2:] = False

        controller.set_active_buffers_available(False)
        controller.set_route(MASKED_ROUTE)
        self.assertTrue(torch.equal(buffer, torch.ones(10)))

        controller.set_active_buffers_available(True)
        self.assertTrue(
            torch.equal(
                buffer,
                torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 0, 0], dtype=torch.float32),
            )
        )

    def test_refresh_without_clean_activation_fails_loudly(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "no clean response activation"):
            self._controller().refresh_mask()

    def test_refresh_requires_every_transformer_layer(self) -> None:
        controller = self._controller()
        controller.set_route(CLEAN_ROUTE, collect_activation=True)
        controller.set_response_token_mask(torch.ones((1, 1)))
        activation = torch.ones((1, 1, 10), requires_grad=True)
        controller.apply(0, activation).sum().backward()
        controller.end_batch()
        with self.assertRaisesRegex(RuntimeError, r"layers \[1\]"):
            controller.refresh_mask()

    def test_legacy_saliency_checkpoint_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "legacy saliency checkpoint"):
            self._controller().load_state_dict(
                {
                    "selection_strategy": "top_saliency",
                    "ema_saliency": torch.zeros((2, 10)),
                }
            )


if __name__ == "__main__":
    unittest.main()
