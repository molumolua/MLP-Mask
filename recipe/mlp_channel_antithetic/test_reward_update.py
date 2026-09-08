"""Numerical tests for the reward signal, Adam isolation, and actual step cap."""

from __future__ import annotations

import copy
import itertools
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from torch import nn

from verl import DataProto

from .intervention import MLPChannelAntitheticController
from .reward_update import (
    REWARD_UPDATE_METADATA,
    DownProjectionShard,
    RewardDifferenceUpdater,
    RewardUpdateConfig,
    estimate_channel_direction,
    prepare_reward_difference,
    resolve_down_projection_shards,
)


class ToyLayer(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(4, 3, bias=False, dtype=dtype)
        self.mlp.up_proj = nn.Linear(3, 4, bias=False, dtype=dtype)

    def forward(self, inputs):
        return self.mlp.down_proj(torch.tanh(self.mlp.up_proj(inputs)))


class ToyModel(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.layers = nn.ModuleList([ToyLayer(dtype), ToyLayer(dtype)])

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


def make_controller():
    return MLPChannelAntitheticController(num_layers=2, intermediate_size=4)


def make_batch():
    # Different numbers/lengths of samples per prompt must not change prompt weights.
    rewards = torch.tensor([[1., 0.], [0., 0.], [1., 1.], [2., 0.], [0., 0.], [0., 0.]])
    return DataProto.from_dict(
        tensors={"token_level_scores": rewards},
        non_tensors={
            "uid": np.array(["a", "a", "b", "b", "b", "b"], dtype=object),
            "route_id": np.array(["positive", "negative", "positive", "positive", "negative", "negative"], dtype=object),
            "perturbation_version": np.full(6, 0),
        },
    )


def test_reward_means_are_equal_prompt_raw_scores_and_order_invariant():
    batch = make_batch()
    metrics = prepare_reward_difference(batch, RewardUpdateConfig(True))
    assert batch.meta_info[REWARD_UPDATE_METADATA] == {"reward_gap": 1.5, "version": 0}
    assert metrics["mlp_antithetic/reward_update/prompt_count"] == 2
    batch.reorder(torch.tensor([5, 0, 2, 4, 1, 3]))
    assert prepare_reward_difference(batch, RewardUpdateConfig(True)) == metrics
    # Global scalar metadata survives actor DP dispatch even with fragmented UIDs.
    for shard in batch.chunk(2):
        assert shard.meta_info[REWARD_UPDATE_METADATA]["reward_gap"] == 1.5


def test_disabled_and_zero_ratio_do_not_even_read_the_batch():
    for config in (RewardUpdateConfig(), RewardUpdateConfig(True, max_update_ratio=0), RewardUpdateConfig(True, learning_rate=0)):
        assert prepare_reward_difference(None, config) == {}
    assert not RewardUpdateConfig.from_config({"enabled": False, "learning_rate": "unused"}).active


@pytest.mark.parametrize("value", [-0.01, 1, float("nan"), float("inf")])
def test_invalid_ratio(value):
    with pytest.raises(ValueError, match="max_update_ratio"):
        RewardUpdateConfig.from_config({"enabled": True, "max_update_ratio": value})


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_learning_rate(value):
    with pytest.raises(ValueError, match="learning_rate"):
        RewardUpdateConfig.from_config({"enabled": True, "learning_rate": value})


def test_invalid_batches_are_rejected_before_training():
    batch = make_batch()
    batch.non_tensor_batch["route_id"][0] = "negative"
    with pytest.raises(ValueError, match="every prompt"):
        prepare_reward_difference(batch, RewardUpdateConfig(True))
    batch = make_batch()
    batch.batch["token_level_scores"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        prepare_reward_difference(batch, RewardUpdateConfig(True))
    batch = make_batch()
    batch.non_tensor_batch["perturbation_version"][0] = 1
    with pytest.raises(ValueError, match="one perturbation"):
        prepare_reward_difference(batch, RewardUpdateConfig(True))


def test_bfloat16_effective_delta_and_label_exchange_invariance():
    controller = make_controller()
    actual, delta, slope = estimate_channel_direction(controller, 0.25, torch.bfloat16)
    assert delta == 0.1015625
    assert slope == 0.25 / (2 * delta)
    torch.testing.assert_close(actual, controller.direction.double() * slope * 0.75)
    controller.direction.neg_()
    flipped, _, _ = estimate_channel_direction(controller, -0.25, torch.bfloat16)
    torch.testing.assert_close(actual, flipped)


def test_balanced_estimator_recovers_projected_gradient_of_quadratic_objective():
    controller = MLPChannelAntitheticController(num_layers=1, intermediate_size=4, perturbation_strength=0.125)
    gradient = torch.tensor([1., -3., 2., 5.], dtype=torch.float64)
    hessian = torch.diag(torch.tensor([2., 5., 7., 11.], dtype=torch.float64))
    estimates = []
    for positive in itertools.combinations(range(4), 2):
        direction = -torch.ones(4, dtype=torch.float64)
        direction[list(positive)] = 1
        controller.direction[0].copy_(direction)
        a = 0.125 * direction
        plus = gradient @ a + 0.5 * a @ hessian @ a
        minus = gradient @ (-a) + 0.5 * a @ hessian @ a
        estimates.append(estimate_channel_direction(controller, float(plus - minus), torch.float64)[0][0])
    torch.testing.assert_close(torch.stack(estimates).mean(0), gradient - gradient.mean())


def test_odd_width_and_rounded_zero_strength_fail():
    controller = MLPChannelAntitheticController(num_layers=1, intermediate_size=3)
    with pytest.raises(ValueError, match="even"):
        estimate_channel_direction(controller, 1, torch.bfloat16)
    controller = MLPChannelAntitheticController(num_layers=1, intermediate_size=4, perturbation_strength=1e-8)
    with pytest.raises(ValueError, match="effective strength"):
        estimate_channel_direction(controller, 1, torch.bfloat16)


@pytest.mark.parametrize("start,count", [(0, 12), (1, 9), (3, 2), (2, 1), (5, 7)])
def test_flat_shard_scaling_handles_partial_channel_rows(start, count):
    weight = torch.arange(12, dtype=torch.float32).reshape(3, 4) + 1
    direction = torch.tensor([0.5, -2., 3., -0.25])
    storage = nn.Parameter(torch.cat([torch.tensor([99.]), weight.flatten()[start : start + count], torch.tensor([88.])]))
    shard = DownProjectionShard(0, storage, (3, 4), (1, count), start)
    candidate = shard.tensor().detach().clone()
    shard.multiply_channels_(candidate, direction)
    torch.testing.assert_close(candidate, (weight * direction).flatten()[start : start + count])
    assert storage[0] == 99 and storage[-1] == 88


def test_flat_metadata_resolver_and_storage_guards():
    module = nn.Module()
    module.layers = nn.ModuleList([nn.Module(), nn.Module()])
    for layer in module.layers:
        layer.register_parameter("flat", nn.Parameter(torch.ones(16)))
        layer._handle = SimpleNamespace(flat_param=layer.flat)
        layer.flat._fqns = ("mlp.up_proj.weight", "mlp.down_proj.weight")
        layer.flat._shapes = ((1, 4), (3, 4))
        layer.flat._shard_param_infos = (
            SimpleNamespace(in_shard=True, offset_in_shard=0, numel_in_shard=4, intra_param_start_idx=0),
            SimpleNamespace(in_shard=True, offset_in_shard=4, numel_in_shard=12, intra_param_start_idx=0),
        )
        layer.flat._sharded_size = torch.Size([16])
    optimizer = torch.optim.AdamW(module.parameters())
    shards, sizes = resolve_down_projection_shards(module, optimizer, make_controller())
    assert sizes == [12, 12]
    assert len(shards) == 2 and shards[0].tensor().numel() == 12
    module.layers[0].flat.data = torch.ones(20)
    with pytest.raises(RuntimeError, match="sharded state"):
        shards[0].tensor()


@pytest.mark.parametrize("gap", [1.0, -1.0])
@pytest.mark.parametrize("aux_lr", [1e-7, 0.1])
def test_actual_post_adam_update_is_capped_and_momentum_untouched(gap, aux_lr):
    torch.manual_seed(51)
    model = ToyModel(dtype=torch.float64)
    reference = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.02)
    baseline = torch.optim.AdamW(reference.parameters(), lr=0.01, weight_decay=0.02)
    controller = make_controller()
    config = RewardUpdateConfig(True, aux_lr, 0.05)
    updater = RewardDifferenceUpdater(model, optimizer, controller, config)
    original = {name: p.detach().clone() for name, p in model.named_parameters()}
    for actual, expected in zip(model.parameters(), reference.parameters()):
        gradient = torch.randn_like(actual)
        actual.grad, expected.grad = gradient.clone(), gradient.clone()
    baseline.step()
    updater.begin_batch({"reward_gap": gap, "version": 0}, np.array([0, 0]))
    optimizer.step()
    for (name, actual), expected in zip(model.named_parameters(), reference.parameters()):
        if "down_proj" not in name:
            assert torch.equal(actual, expected)
            continue
        layer = int(name.split(".")[1])
        pg = expected.detach() - original[name]
        aux = actual.detach() - expected.detach()
        assert aux.norm() <= config.max_update_ratio * pg.norm() * (1 + 1e-12)
        estimated = estimate_channel_direction(controller, gap, torch.bfloat16)[0][layer]
        raw = aux_lr * original[name] * estimated
        assert (aux * raw).sum() > 0
        if raw.norm() < config.max_update_ratio * pg.norm():
            torch.testing.assert_close(aux, raw)
        for key, value in baseline.state[expected].items():
            assert torch.equal(optimizer.state[actual][key], value)
    assert updater.last_metrics["mlp_antithetic/reward_update/max_layer_ratio"] <= 0.05 * (1 + 1e-12)
    assert not updater.snapshots
    updater.end_batch()


def test_zero_gap_avoids_weight_snapshots_and_collectives():
    model = ToyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    updater = RewardDifferenceUpdater(model, optimizer, make_controller(), RewardUpdateConfig(True))
    updater.begin_batch({"reward_gap": 0., "version": 0}, np.array([0]))
    with mock.patch.object(DownProjectionShard, "tensor", side_effect=AssertionError("unexpected snapshot")), mock.patch.object(updater, "_sum", side_effect=AssertionError("unexpected collective")):
        optimizer.step()
    assert not updater.snapshots
    assert updater.last_metrics["mlp_antithetic/reward_update/applied"] == 0
    updater.end_batch()


def test_zero_main_step_suppresses_auxiliary_even_with_reward_gap():
    model = ToyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0)
    updater = RewardDifferenceUpdater(model, optimizer, make_controller(), RewardUpdateConfig(True, 1, 0.1))
    before = [p.detach().clone() for p in model.parameters()]
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    updater.begin_batch({"reward_gap": 1., "version": 0}, np.array([0]))
    optimizer.step()
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert updater.last_metrics["mlp_antithetic/reward_update/actual_ratio"] == 0
    updater.end_batch()


def test_float32_rounding_cannot_violate_actual_step_budget():
    model = ToyModel()
    for parameter in model.parameters():
        parameter.data.fill_(1)
        parameter.grad = torch.ones_like(parameter)
    reference = copy.deepcopy(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
    baseline = torch.optim.SGD(reference.parameters(), lr=1e-6)
    for parameter in reference.parameters():
        parameter.grad = torch.ones_like(parameter)
    original = model.layers[0].mlp.down_proj.weight.detach().clone()
    updater = RewardDifferenceUpdater(model, optimizer, make_controller(), RewardUpdateConfig(True, 1, 0.05))
    updater.begin_batch({"reward_gap": 1., "version": 0}, np.array([0]))
    baseline.step()
    optimizer.step()
    for layer, base in zip(model.layers, reference.layers):
        pg = base.mlp.down_proj.weight.detach() - original
        aux = layer.mlp.down_proj.weight.detach() - base.mlp.down_proj.weight.detach()
        assert aux.double().norm() <= 0.05 * pg.double().norm()
    updater.end_batch()


def test_stale_metadata_and_reused_step_rejected_and_cleanup_allows_next_batch():
    model = ToyModel()
    optimizer = torch.optim.AdamW(model.parameters())
    controller = make_controller()
    updater = RewardDifferenceUpdater(model, optimizer, controller, RewardUpdateConfig(True))
    with pytest.raises(RuntimeError, match="stale"):
        updater.begin_batch({"reward_gap": 1., "version": 1}, np.array([0]))
    with pytest.raises(RuntimeError, match="version mismatch"):
        updater.begin_batch({"reward_gap": 1., "version": 0}, np.array([1]))
    updater.begin_batch({"reward_gap": 0., "version": 0}, np.array([0]))
    optimizer.step()
    with pytest.raises(RuntimeError, match="exactly one"):
        optimizer.step()
    updater.end_batch()
    controller.refresh_direction(version=1)
    updater.begin_batch({"reward_gap": 0., "version": 1}, np.array([1]))
    optimizer.step()
    updater.close()


def test_checkpoint_resume_needs_no_additional_auxiliary_state():
    model = ToyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    controller = make_controller()
    config = RewardUpdateConfig(True, 0.01, 0.05)
    updater = RewardDifferenceUpdater(model, optimizer, controller, config)

    def step(network, opt, intervention, update, version):
        intervention.refresh_direction(version=version)
        for parameter in network.parameters():
            parameter.grad = torch.ones_like(parameter) * (version + 1)
        update.begin_batch({"reward_gap": 0.5, "version": version}, np.array([version]))
        opt.step()
        update.end_batch()

    step(model, optimizer, controller, updater, 0)
    resumed = ToyModel()
    resumed.load_state_dict(copy.deepcopy(model.state_dict()))
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.01)
    resumed_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    resumed_controller = make_controller()
    resumed_controller.load_state_dict(copy.deepcopy(controller.state_dict()))
    resumed_updater = RewardDifferenceUpdater(resumed, resumed_optimizer, resumed_controller, config)
    step(model, optimizer, controller, updater, 1)
    step(resumed, resumed_optimizer, resumed_controller, resumed_updater, 1)
    for actual, expected in zip(model.parameters(), resumed.parameters()):
        assert torch.equal(actual, expected)
        for key in optimizer.state[actual]:
            assert torch.equal(optimizer.state[actual][key], resumed_optimizer.state[expected][key])
