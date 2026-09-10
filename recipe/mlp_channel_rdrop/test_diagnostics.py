import math

import pytest
import torch

from .diagnostics import ParameterUpdateTracker, SampledGradientTracker


def test_bfloat16_update_fraction_count_and_cumulative_reference():
    model = torch.nn.Linear(4, 1, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        model.weight.zero_()
    tracker = ParameterUpdateTracker(model)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0, 5e-6, 2e-5, -4e-5]]))
    metrics = tracker.distributed_metrics()
    assert metrics["val-aux/parameter_update/updated_fraction_atol_1e-5"] == 0.5
    assert metrics["val-aux/parameter_update/updated_parameter_count"] == 2
    # A second observation still uses the pre-RL snapshot, not the first one.
    with torch.no_grad():
        model.weight[0, 1] = 3e-5
    assert tracker.distributed_metrics()["val-aux/parameter_update/updated_parameter_count"] == 3


def test_gradient_angle_weighted_auxiliary_and_accumulation():
    model = torch.nn.Linear(2, 1, bias=False)
    tracker = SampledGradientTracker(model, sample_size_per_rank=2, random_seed=3)
    tracker.start_update()
    for main, aux in [([1., 2.], [2., -1.]), ([-1., 3.], [2., 1.])]:
        model(torch.tensor([main])).sum().backward()
        tracker.capture_main_gradient()
        (0.5 * model(torch.tensor([aux])).sum()).backward()
        tracker.capture_auxiliary_gradient()
    metrics = tracker.finish_update()
    # Complete update main=[0,5], weighted aux=[2,0].
    assert metrics["mlp_rdrop/aux_to_main_grad_ratio_sampled"] == pytest.approx(0.4)
    assert metrics["mlp_rdrop/main_aux_grad_angle_degrees_sampled"] == pytest.approx(90)
    assert metrics["mlp_rdrop/main_grad_rms_sampled"] == pytest.approx(5 / math.sqrt(2))
    assert metrics["mlp_rdrop/grad_angle_defined"] == 1


def test_zero_gradients_are_flagged_and_sampling_is_repeatable():
    model = torch.nn.Linear(30, 30)
    tracker = SampledGradientTracker(model, sample_size_per_rank=17, random_seed=5)
    repeat = SampledGradientTracker(model, sample_size_per_rank=17, random_seed=5)
    assert tracker.sample_count == 17
    assert [s.indices.tolist() for s in tracker.samples] == [s.indices.tolist() for s in repeat.samples]
    tracker.start_update()
    tracker.capture_main_gradient()
    tracker.capture_auxiliary_gradient()
    metrics = tracker.finish_update()
    assert metrics["mlp_rdrop/grad_ratio_defined"] == 0
    assert metrics["mlp_rdrop/grad_angle_defined"] == 0
