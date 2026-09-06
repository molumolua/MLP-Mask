"""CPU tests for auxiliary-gradient and parameter-update diagnostics."""

from __future__ import annotations

import math
import unittest

import torch

from .diagnostics import ParameterUpdateTracker, SampledGradientTracker


class SampledGradientTrackerTest(unittest.TestCase):
    def test_accumulates_main_and_weighted_auxiliary_branches(self):
        model = torch.nn.Linear(2, 1, bias=False)
        tracker = SampledGradientTracker(
            model,
            sample_size_per_rank=2,
            random_seed=7,
        )

        tracker.start_update()
        model(torch.tensor([[1.0, 2.0]])).sum().backward()
        tracker.capture_main_gradient()
        # This multiplier stands in for kl_coef: the tracker sees the actual
        # weighted gradient that will be added to the optimizer update.
        (0.5 * model(torch.tensor([[2.0, -1.0]])).sum()).backward()
        tracker.capture_auxiliary_gradient()
        metrics = tracker.finish_update()

        self.assertAlmostEqual(
            metrics["mlp_consistency/main_grad_rms_sampled"],
            math.sqrt(2.5),
        )
        self.assertAlmostEqual(
            metrics["mlp_consistency/aux_grad_rms_sampled"],
            math.sqrt(0.625),
        )
        self.assertAlmostEqual(
            metrics["mlp_consistency/aux_to_main_grad_ratio_sampled"],
            0.5,
        )
        self.assertAlmostEqual(
            metrics["mlp_consistency/main_aux_grad_cosine_sampled"],
            0.0,
        )
        self.assertEqual(metrics["mlp_consistency/gradient_sample_count"], 2.0)

    def test_fixed_sample_is_bounded(self):
        model = torch.nn.Linear(100, 10)
        first = SampledGradientTracker(model, sample_size_per_rank=17, random_seed=9)
        second = SampledGradientTracker(model, sample_size_per_rank=17, random_seed=9)

        self.assertEqual(first.sample_count, 17)
        self.assertEqual(second.sample_count, 17)
        self.assertEqual(
            [sample.indices_cpu.tolist() for sample in first.samples],
            [sample.indices_cpu.tolist() for sample in second.samples],
        )

    def test_branch_vectors_accumulate_across_micro_batches(self):
        model = torch.nn.Linear(2, 1, bias=False)
        tracker = SampledGradientTracker(
            model,
            sample_size_per_rank=2,
            random_seed=11,
        )

        tracker.start_update()
        model(torch.tensor([[1.0, 2.0]])).sum().backward()
        tracker.capture_main_gradient()
        (0.5 * model(torch.tensor([[2.0, -1.0]])).sum()).backward()
        tracker.capture_auxiliary_gradient()

        model(torch.tensor([[-1.0, 3.0]])).sum().backward()
        tracker.capture_main_gradient()
        (0.25 * model(torch.tensor([[4.0, 2.0]])).sum()).backward()
        tracker.capture_auxiliary_gradient()
        metrics = tracker.finish_update()

        # Accumulated main=[0, 5], auxiliary=[2, 0].
        self.assertAlmostEqual(
            metrics["mlp_consistency/aux_to_main_grad_ratio_sampled"],
            0.4,
        )


class ParameterUpdateTrackerTest(unittest.TestCase):
    def test_matches_bfloat16_atol_sparsity_definition(self):
        model = torch.nn.Linear(4, 1, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            model.weight.zero_()
        tracker = ParameterUpdateTracker(model)

        with torch.no_grad():
            model.weight.copy_(
                torch.tensor(
                    [[0.0, 5.0e-6, 2.0e-5, -4.0e-5]],
                    dtype=torch.bfloat16,
                )
            )
        metrics = tracker.distributed_metrics(atol=1.0e-5)

        self.assertEqual(
            metrics["val-aux/parameter_update/sparsity_atol_1e-5"],
            0.5,
        )
        self.assertEqual(
            metrics["val-aux/parameter_update/updated_fraction_atol_1e-5"],
            0.5,
        )
        self.assertEqual(metrics["val-aux/parameter_update/atol"], 1.0e-5)

    def test_computes_updated_fraction_for_complete_swiglu_channels(self):
        class ToyMLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = torch.nn.Linear(2, 3, bias=False, dtype=torch.bfloat16)
                self.up_proj = torch.nn.Linear(2, 3, bias=False, dtype=torch.bfloat16)
                self.down_proj = torch.nn.Linear(3, 2, bias=False, dtype=torch.bfloat16)

        class ToyBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = ToyMLP()

        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([ToyBlock()])

        model = ToyModel()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        tracker = ParameterUpdateTracker(
            model,
            num_layers=1,
            intermediate_size=3,
        )
        with torch.no_grad():
            model.layers[0].mlp.gate_proj.weight[1].fill_(2.0e-5)
            model.layers[0].mlp.up_proj.weight[1].fill_(2.0e-5)
            model.layers[0].mlp.down_proj.weight[:, 1].fill_(2.0e-5)

        fraction = tracker.distributed_channel_updated_fraction(atol=1.0e-5)

        torch.testing.assert_close(fraction, torch.tensor([[0.0, 1.0, 0.0]]))


if __name__ == "__main__":
    unittest.main()
