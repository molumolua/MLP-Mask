"""CPU gradient oracle for the production actor mixin and checkpointed masks.

The small base below supplies the PPO hook protocol without importing CUDA-only
training dependencies. The production antithetic cross-KL scheduling, KL and trackers run as-is.
"""

import copy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from verl import DataProto

from .actor import AntitheticCrossKLActorMixin
from .intervention import install_hf_mlp_intervention
from .diagnostics import SampledGradientTracker
from .intervention import NEUTRAL_ROUTE, POSITIVE_ROUTE, NEGATIVE_ROUTE, TRAINING_ROUTES, MLPChannelAntitheticController
from torch import nn
from torch.utils.checkpoint import checkpoint


class ToyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 20, bias=False)
        self.up_proj = nn.Linear(4, 20, bias=False)
        self.down_proj = nn.Linear(20, 4, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, hidden):
        return self.down_proj(self.act_fn(self.gate_proj(hidden)) * self.up_proj(hidden))


class ToyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(11, 4)
        self.layers = nn.ModuleList([nn.ModuleDict({"mlp": ToyMLP()})])
        self.head = nn.Linear(4, 11, bias=False)

    def forward(self, tokens):
        hidden = self.embed(tokens)
        hidden = hidden + checkpoint(self.layers[0]["mlp"], hidden, use_reentrant=False)
        return self.head(hidden)

    def explicit(self, tokens, gain):
        hidden = self.embed(tokens)
        mlp = self.layers[0]["mlp"]
        activation = mlp.act_fn(mlp.gate_proj(hidden)) * mlp.up_proj(hidden)
        return self.head(hidden + mlp.down_proj(activation * gain))
from .test_cross_kl_tensor import coarsened_reference, reference_kl


def clipped_ppo(logp, old, advantages, mask):
    ratio = (logp - old).exp()
    loss = torch.maximum(-advantages * ratio, -advantages * ratio.clamp(0.8, 1.2))
    return (loss * mask).sum() / mask.sum().clamp_min(1)


class ToyPPOBase:
    def _forward_micro_batch(self, inputs, temperature, calculate_entropy=False):
        callback = getattr(self, "_response_logits_callback", None)
        self.calls.append((self.intervention_controller.route, torch.is_grad_enabled(),
                           inputs["input_ids"].clone(), getattr(callback, "__name__", None)))
        logits = self.actor_module(inputs["input_ids"])[:, -3:-1] / temperature
        if callback:
            callback(logits, inputs["response_mask"])
        logp = logits.log_softmax(-1).gather(-1, inputs["responses"].unsqueeze(-1)).squeeze(-1)
        return None, logp

    def update_policy(self, data):
        self.actor_module.train()
        self.actor_optimizer.zero_grad()
        metrics = {"actor/pg_loss": []}
        for route in TRAINING_ROUTES:
            ids = np.flatnonzero(data.non_tensor_batch["route_id"] == route)
            whole = data.select_idxs(ids)
            for batch in whole.split(self.main_micro_batch_size):
                self.intervention_controller.set_route(route)
                inputs = {**batch.batch, **batch.non_tensor_batch}
                _, logp = self._forward_micro_batch(inputs, data.meta_info["temperature"])
                micro_tokens = float(inputs["response_mask"].sum())
                main = clipped_ppo(logp, inputs["old_log_probs"], inputs["advantages"], inputs["response_mask"])
                main = main * micro_tokens / float(data.batch["response_mask"].sum())
                main.backward()
                metrics["actor/pg_loss"].append(float(main.detach()))
                self._backward_auxiliary_loss(model_inputs=inputs, temperature=data.meta_info["temperature"],
                                              aggregation_scale=micro_tokens / float(data.batch["response_mask"].sum()))
        self.final_gradient = torch.cat([p.grad.reshape(-1) for p in self.actor_module.parameters()]).clone()
        self.actor_optimizer.step()
        self.actor_optimizer.zero_grad()
        return metrics


class TestableActor(AntitheticCrossKLActorMixin, ToyPPOBase):
    __test__ = False


def setup_actor(main_micro=2, aux_micro=1, coefficient=0.3, top_k=0):
    torch.manual_seed(24)
    actor = TestableActor()
    actor.actor_module = ToyLM()
    reference = copy.deepcopy(actor.actor_module)
    controller = MLPChannelAntitheticController(num_layers=1, intermediate_size=20)
    install_hf_mlp_intervention(actor.actor_module, controller)
    actor.actor_optimizer = torch.optim.SGD(actor.actor_module.parameters(), lr=0.01)
    actor.main_micro_batch_size = main_micro
    actor.calls = []
    actor.configure_cross_kl(SimpleNamespace(kl_coef=coefficient, enabled=coefficient > 0,
                                         micro_batch_size_per_gpu=aux_micro, kl_token_chunk_size=1, kl_top_k=top_k),
                          controller, SampledGradientTracker(actor.actor_module, sample_size_per_rank=10000, random_seed=2))
    tokens = torch.randint(0, 11, (16, 4))
    responses = tokens[:, -2:]
    mask = torch.tensor([[1, i % 3 != 0] for i in range(16)])
    routes = np.array([POSITIVE_ROUTE, NEGATIVE_ROUTE] * 8, dtype=object)
    temperature = 0.7
    old = torch.zeros(16, 2)
    with torch.no_grad():
        for route in TRAINING_ROUTES:
            ids = np.flatnonzero(routes == route)
            logits = reference.explicit(tokens[ids], controller.gain(0, route))[:, -3:-1] / temperature
            old[ids] = logits.log_softmax(-1).gather(-1, responses[ids].unsqueeze(-1)).squeeze(-1)
    data = DataProto.from_dict(tensors={"input_ids": tokens, "responses": responses,
                                       "response_mask": mask, "old_log_probs": old,
                                       "advantages": torch.linspace(-0.8, 1.2, 16)[:, None].expand(-1, 2)},
                               non_tensors={"route_id": routes, "perturbation_version": np.zeros(16, dtype=np.int64)},
                               meta_info={"temperature": temperature})
    return actor, reference, data


def oracle(actor, reference, data):
    tensors = data.batch
    logits = [reference.explicit(tensors["input_ids"], actor.intervention_controller.gain(0, route).clone())[:, -3:-1]
              / data.meta_info["temperature"] for route in TRAINING_ROUTES]
    main = 0
    for route, logit in zip(TRAINING_ROUTES, logits):
        ids = np.flatnonzero(data.non_tensor_batch["route_id"] == route)
        logp = logit.log_softmax(-1).gather(-1, tensors["responses"].unsqueeze(-1)).squeeze(-1)
        main += clipped_ppo(logp[ids], tensors["old_log_probs"][ids], tensors["advantages"][ids], tensors["response_mask"][ids]) * tensors["response_mask"][ids].sum() / tensors["response_mask"].sum()
    divergence = 0
    for source_index, route in enumerate(TRAINING_ROUTES):
        rows = np.flatnonzero(data.non_tensor_batch["route_id"] == route)
        mask = tensors["response_mask"][rows].bool()
        teacher = logits[source_index][rows][mask].detach()
        student = logits[1 - source_index][rows][mask]
        if actor.kl_top_k:
            ids = teacher.topk(actor.kl_top_k, -1).indices
            divergence += coarsened_reference(teacher, student, ids)
        else:
            divergence += reference_kl(teacher, student)
    auxiliary = actor.kl_coef * divergence / tensors["response_mask"].sum()
    gm = torch.autograd.grad(main, tuple(reference.parameters()), retain_graph=True)
    ga = torch.autograd.grad(auxiliary, tuple(reference.parameters()))
    return main.detach(), auxiliary.detach(), torch.cat([g.reshape(-1) for g in gm]), torch.cat([g.reshape(-1) for g in ga])


@pytest.mark.parametrize("main_micro,aux_micro", [(1, 1), (4, 1), (4, 3)])
@pytest.mark.parametrize("top_k", [0, 3])
def test_all_16_responses_cross_gradient_metrics_and_checkpoint_replay(main_micro, aux_micro, top_k):
    actor, reference, data = setup_actor(main_micro, aux_micro, top_k=top_k)
    main, auxiliary, gm, ga = oracle(actor, reference, data)
    metrics = actor.update_policy(data)
    torch.testing.assert_close(actor.final_gradient, gm + ga, atol=1e-7, rtol=2e-5)
    assert metrics["mlp_antithetic/cross_kl/main_pg_loss_step"][0] == pytest.approx(float(main), abs=1e-6)
    assert metrics["mlp_antithetic/cross_kl/weighted_kl_step"][0] == pytest.approx(float(auxiliary), abs=1e-7)
    assert metrics["mlp_antithetic/cross_kl/aligned_response_rows"] == [16]
    assert metrics["mlp_antithetic/cross_kl/response_tokens"] == [float(data.batch["response_mask"].sum())]
    assert metrics["mlp_antithetic/cross_kl/aux_to_main_grad_ratio_sampled"][0] == pytest.approx(float(ga.norm() / gm.norm()), rel=3e-5)
    cosine = float(torch.dot(gm, ga) / (gm.norm() * ga.norm()))
    assert metrics["mlp_antithetic/cross_kl/main_aux_grad_cosine_sampled"][0] == pytest.approx(cosine, abs=1e-5)
    assert actor.intervention_controller.route == NEUTRAL_ROUTE
    assert actor._response_logits_callback is None
    assert actor._teacher_distribution is None and not actor._cross_kl_update_active
    assert "loss_multiplier" not in data.non_tensor_batch
    assert "loss_group_id" not in data.non_tensor_batch
    # Teacher capture piggybacks on PPO. Each row gets exactly one source
    # forward and one opposite-route forward; no detached-teacher replay.
    assert all(grad for _, grad, _, _ in actor.calls)
    for source, student, prefix in [(POSITIVE_ROUTE, NEGATIVE_ROUTE, "positive_to_negative"), (NEGATIVE_ROUTE, POSITIVE_ROUTE, "negative_to_positive")]:
        expected_rows = data.batch["input_ids"][data.non_tensor_batch["route_id"] == source]
        main_rows = torch.cat([tokens for route, _, tokens, kind in actor.calls
                               if route == source and kind == "_capture_teacher"])
        student_rows = torch.cat([tokens for route, _, tokens, kind in actor.calls
                                  if route == student and kind == "_capture_student_loss"])
        torch.testing.assert_close(main_rows, expected_rows)
        torch.testing.assert_close(student_rows, expected_rows)
        assert metrics[f"mlp_antithetic/cross_kl/{prefix}_aligned_rows"] == [8]
    source_weighted_sum = sum(metrics[f"mlp_antithetic/cross_kl/{prefix}_kl"][0] * metrics[f"mlp_antithetic/cross_kl/{prefix}_response_tokens"][0]
                              for prefix in ("positive_to_negative", "negative_to_positive"))
    assert metrics["mlp_antithetic/cross_kl/kl"][0] == pytest.approx(source_weighted_sum / float(data.batch["response_mask"].sum()))
    extra_calls = sum(kind == "_capture_student_loss" for _, _, _, kind in actor.calls)
    assert metrics["mlp_antithetic/cross_kl/auxiliary_forward_calls"] == [extra_calls]
    assert metrics["mlp_antithetic/cross_kl/auxiliary_backward_calls"] == [extra_calls]


def test_no_auxiliary_baseline_and_zero_weight_padding():
    actor, reference, data = setup_actor(coefficient=0)
    _, _, gm, _ = oracle(actor, reference, data)
    metrics = actor.update_policy(data)
    torch.testing.assert_close(actor.final_gradient, gm, atol=1e-7, rtol=1e-5)
    assert metrics["mlp_antithetic/cross_kl/weighted_kl_step"] == [0]
    assert metrics["mlp_antithetic/cross_kl/grad_angle_defined"] == [0]
    assert all(grad and kind is None for _, grad, _, kind in actor.calls)
    assert metrics["mlp_antithetic/cross_kl/auxiliary_forward_calls"] == [0]
    assert metrics["timing_s/mlp_antithetic_cross_kl_teacher_capture_step"] == [0]

    actor, reference, data = setup_actor(main_micro=4, aux_micro=3)
    _, _, gm, ga = oracle(actor, reference, data)
    with patch("recipe.mlp_channel_antithetic.actor.synchronized_auxiliary_slots", return_value=3):
        metrics = actor.update_policy(data)
    torch.testing.assert_close(actor.final_gradient, gm + ga, atol=1e-7, rtol=2e-5)
    assert metrics["mlp_antithetic/cross_kl/aligned_response_rows"] == [16]
    assert metrics["mlp_antithetic/cross_kl/auxiliary_padding_slots"] == [4]


@pytest.mark.parametrize("callback", ["_capture_teacher", "_capture_student_loss"])
def test_forward_failure_restores_route_and_cancels_tracker(callback):
    actor, _, data = setup_actor()
    with patch.object(actor, callback, side_effect=RuntimeError("injected failure")):
        with pytest.raises(RuntimeError, match="injected failure"):
            actor.update_policy(data)
    assert actor.intervention_controller.route == NEUTRAL_ROUTE
    assert actor._response_logits_callback is None
    assert actor._teacher_distribution is None and not actor._cross_kl_update_active
    assert not actor.gradient_tracker._active


def test_logprob_forward_outside_update_does_not_capture_teacher():
    actor, _, data = setup_actor(top_k=3)
    with torch.no_grad(), patch.object(actor, "_capture_teacher", side_effect=AssertionError("unexpected capture")):
        actor._forward_micro_batch({**data.batch, **data.non_tensor_batch}, data.meta_info["temperature"])
    assert actor.intervention_controller.route == NEUTRAL_ROUTE
    assert actor.calls[0][-1] is None
