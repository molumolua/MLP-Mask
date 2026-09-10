"""CPU gradient oracle for the production actor mixin and checkpointed masks.

The small base below supplies the PPO hook protocol without importing CUDA-only
training dependencies. The production R-Drop scheduling, KL and trackers run as-is.
"""

import copy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from verl import DataProto

from .actor import RDropActorMixin
from .backend import install_hf_mlp_intervention
from .diagnostics import SampledGradientTracker
from .intervention import CLEAN_ROUTE, MASK_A_ROUTE, MASK_B_ROUTE, TRAINING_ROUTES, MLPChannelRDropController
from .test_intervention import ToyLM
from .test_kl import coarsened_reference, reference_kl


def clipped_ppo(logp, old, advantages, mask):
    ratio = (logp - old).exp()
    loss = torch.maximum(-advantages * ratio, -advantages * ratio.clamp(0.8, 1.2))
    return (loss * mask).sum() / mask.sum().clamp_min(1)


class ToyPPOBase:
    def _forward_micro_batch(self, inputs, temperature, calculate_entropy=False):
        self.calls.append((self.intervention_controller.route, torch.is_grad_enabled(),
                           inputs["input_ids"].clone()))
        logits = self.actor_module(inputs["input_ids"])[:, -3:-1] / temperature
        callback = getattr(self, "_response_logits_callback", None)
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
            route_tokens = float(whole.batch["response_mask"].sum())
            for batch in whole.split(self.main_micro_batch_size):
                self.intervention_controller.set_route(route)
                inputs = {**batch.batch, **batch.non_tensor_batch}
                _, logp = self._forward_micro_batch(inputs, data.meta_info["temperature"])
                micro_tokens = float(inputs["response_mask"].sum())
                main = clipped_ppo(logp, inputs["old_log_probs"], inputs["advantages"], inputs["response_mask"])
                main = main * micro_tokens / route_tokens * float(inputs["loss_multiplier"][0]) / 2
                main.backward()
                metrics["actor/pg_loss"].append(float(main.detach()))
                self._backward_auxiliary_loss(model_inputs=inputs, temperature=data.meta_info["temperature"],
                                              aggregation_scale=micro_tokens / float(data.batch["response_mask"].sum()))
        self.final_gradient = torch.cat([p.grad.reshape(-1) for p in self.actor_module.parameters()]).clone()
        self.actor_optimizer.step()
        self.actor_optimizer.zero_grad()
        return metrics


class TestableActor(RDropActorMixin, ToyPPOBase):
    __test__ = False


def setup_actor(main_micro=2, aux_micro=1, coefficient=0.3, top_k=0):
    torch.manual_seed(24)
    actor = TestableActor()
    actor.actor_module = ToyLM()
    reference = copy.deepcopy(actor.actor_module)
    controller = MLPChannelRDropController(num_layers=1, intermediate_size=20)
    install_hf_mlp_intervention(actor.actor_module, controller)
    actor.actor_optimizer = torch.optim.SGD(actor.actor_module.parameters(), lr=0.01)
    actor.main_micro_batch_size = main_micro
    actor.calls = []
    actor.configure_rdrop(SimpleNamespace(kl_coef=coefficient, auxiliary_enabled=coefficient > 0,
                                         micro_batch_size_per_gpu=aux_micro, kl_token_chunk_size=1, kl_top_k=top_k),
                          controller, SampledGradientTracker(actor.actor_module, sample_size_per_rank=10000, random_seed=2))
    tokens = torch.randint(0, 11, (16, 4))
    responses = tokens[:, -2:]
    mask = torch.tensor([[1, i % 3 != 0] for i in range(16)])
    routes = np.array([MASK_A_ROUTE, MASK_B_ROUTE] * 8, dtype=object)
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
                               non_tensors={"route_id": routes, "mask_version": np.zeros(16, dtype=np.int64)},
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
        main += 0.5 * clipped_ppo(logp[ids], tensors["old_log_probs"][ids], tensors["advantages"][ids], tensors["response_mask"][ids])
    mask = tensors["response_mask"].bool()
    if actor.kl_top_k:
        ids = logits[0][mask].detach().topk(actor.kl_top_k, -1).indices
        divergence = coarsened_reference(logits[0][mask], logits[1][mask], ids)
    else:
        divergence = reference_kl(logits[0][mask], logits[1][mask])
    auxiliary = actor.kl_coef * divergence / mask.sum()
    gm = torch.autograd.grad(main, tuple(reference.parameters()), retain_graph=True)
    ga = torch.autograd.grad(auxiliary, tuple(reference.parameters()))
    return main.detach(), auxiliary.detach(), torch.cat([g.reshape(-1) for g in gm]), torch.cat([g.reshape(-1) for g in ga])


@pytest.mark.parametrize("main_micro,aux_micro", [(1, 1), (4, 1), (4, 3)])
@pytest.mark.parametrize("top_k", [0, 3])
def test_all_16_responses_exact_joint_gradient_metrics_and_checkpoint_replay(main_micro, aux_micro, top_k):
    actor, reference, data = setup_actor(main_micro, aux_micro, top_k=top_k)
    main, auxiliary, gm, ga = oracle(actor, reference, data)
    metrics = actor.update_policy(data)
    torch.testing.assert_close(actor.final_gradient, gm + ga, atol=1e-7, rtol=2e-5)
    assert metrics["mlp_rdrop/main_pg_loss_step"][0] == pytest.approx(float(main), abs=1e-6)
    assert metrics["mlp_rdrop/weighted_kl_step"][0] == pytest.approx(float(auxiliary), abs=1e-7)
    assert metrics["mlp_rdrop/aligned_response_rows"] == [16]
    assert metrics["mlp_rdrop/response_tokens"] == [float(data.batch["response_mask"].sum())]
    assert metrics["mlp_rdrop/aux_to_main_grad_ratio_sampled"][0] == pytest.approx(float(ga.norm() / gm.norm()), rel=3e-5)
    cosine = float(torch.dot(gm, ga) / (gm.norm() * ga.norm()))
    assert metrics["mlp_rdrop/main_aux_grad_cosine_sampled"][0] == pytest.approx(cosine, abs=1e-5)
    assert actor.intervention_controller.route == CLEAN_ROUTE
    assert actor._response_logits_callback is None
    # Three auxiliary evaluations per row: detached A, differentiable B, replay A.
    reference_rows = sum(len(tokens) for route, grad, tokens in actor.calls if not grad)
    assert reference_rows == 16


def test_no_auxiliary_baseline_and_zero_weight_padding():
    actor, reference, data = setup_actor(coefficient=0)
    _, _, gm, _ = oracle(actor, reference, data)
    metrics = actor.update_policy(data)
    torch.testing.assert_close(actor.final_gradient, gm, atol=1e-7, rtol=1e-5)
    assert metrics["mlp_rdrop/weighted_kl_step"] == [0]
    assert metrics["mlp_rdrop/grad_angle_defined"] == [0]
    assert all(grad for _, grad, _ in actor.calls)

    actor, reference, data = setup_actor(main_micro=4, aux_micro=3)
    _, _, gm, ga = oracle(actor, reference, data)
    with patch("recipe.mlp_channel_rdrop.actor.synchronized_auxiliary_slots", return_value=3):
        metrics = actor.update_policy(data)
    torch.testing.assert_close(actor.final_gradient, gm + ga, atol=1e-7, rtol=2e-5)
    assert metrics["mlp_rdrop/aligned_response_rows"] == [16]
    assert metrics["mlp_rdrop/auxiliary_padding_slots"] == [4]


def test_auxiliary_failure_restores_route_and_cancels_tracker():
    actor, _, data = setup_actor()
    with patch.object(actor, "_capture_partial_loss", side_effect=RuntimeError("injected failure")):
        with pytest.raises(RuntimeError, match="injected failure"):
            actor.update_policy(data)
    assert actor.intervention_controller.route == CLEAN_ROUTE
    assert actor._response_logits_callback is None
    assert not actor.gradient_tracker._active
