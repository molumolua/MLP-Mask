import copy

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .backend import install_hf_mlp_intervention, install_vllm_mlp_intervention
from .intervention import CLEAN_ROUTE, MASK_A_ROUTE, MASK_B_ROUTE, MLPChannelRDropController
from .routing import assign_rdrop_routes, copy_prompt_uids_for_generation


class ToyMLP(nn.Module):
    def __init__(self, hidden=4, width=20):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, width, bias=False)
        self.up_proj = nn.Linear(hidden, width, bias=False)
        self.down_proj = nn.Linear(width, hidden, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


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

    def explicit(self, tokens, mask):
        # Captures each mask as a tensor in its own graph, a coupled reference
        # independent of the production controller's sequential route switches.
        hidden = self.embed(tokens)
        mlp = self.layers[0]["mlp"]
        z = mlp.act_fn(mlp.gate_proj(hidden)) * mlp.up_proj(hidden)
        return self.head(hidden + mlp.down_proj(z * mask))


def test_exact_independent_replay_and_checkpoint():
    controller = MLPChannelRDropController(num_layers=32, intermediate_size=100)
    assert ((~controller.keep_masks).sum(-1) == 10).all()
    assert not torch.equal(controller.keep_masks[0], controller.keep_masks[1])
    assert ((~controller.keep_masks[0]) & (~controller.keep_masks[1])).any()
    original = controller.keep_masks.clone()
    assert not controller.refresh_masks(version=0)
    torch.testing.assert_close(controller.keep_masks, original)
    controller.refresh_masks(version=10)
    state = controller.state_dict()
    other = MLPChannelRDropController(num_layers=32, intermediate_size=100)
    other.load_state_dict(state)
    torch.testing.assert_close(other.keep_masks, controller.keep_masks)
    other.validate_batch_version(np.array([10, 10]))
    with pytest.raises(RuntimeError, match="version mismatch"):
        other.validate_batch_version(np.array([9, 10]))
    state["keep_masks"][0].fill_(True)
    with pytest.raises(ValueError, match="quota"):
        other.load_state_dict(state)


def test_hf_routes_checkpoint_backward_and_clean_recovery():
    torch.manual_seed(6)
    model = ToyLM()
    reference = copy.deepcopy(model)
    controller = MLPChannelRDropController(num_layers=1, intermediate_size=20)
    install_hf_mlp_intervention(model, controller)
    tokens = torch.tensor([[1, 2, 3]])
    for route in (MASK_A_ROUTE, MASK_B_ROUTE, CLEAN_ROUTE):
        controller.set_route(route)
        actual = model(tokens)
        expected = reference.explicit(tokens, controller.gain(0).clone())
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        expected.sum().backward()
    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


def test_tp_slices_and_sleeping_buffers():
    controllers = [MLPChannelRDropController(num_layers=1, intermediate_size=20, tp_size=2, tp_rank=i)
                   for i in range(2)]
    outputs = []
    for c in controllers:
        c.set_route(MASK_B_ROUTE)
        outputs.append(c.apply(0, torch.ones(1, 10)))
    torch.testing.assert_close(torch.cat(outputs, -1).bool()[0], controllers[0].gain(0))
    c = controllers[0]
    buffer = next(iter(c._active_buffers.values()))
    address, old = buffer.data_ptr(), buffer.clone()
    c.set_active_buffers_available(False)
    c.set_route(CLEAN_ROUTE)
    torch.testing.assert_close(buffer, old)
    c.set_active_buffers_available(True)
    assert buffer.data_ptr() == address and buffer.eq(1).all()


def test_fused_vllm_layout_matches_hf():
    torch.manual_seed(7)
    model = ToyLM()
    mlp = model.layers[0]["mlp"]

    class FusedMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Linear(4, 40, bias=False)
            self.down_proj = copy.deepcopy(mlp.down_proj)
            self.act_fn = lambda x: torch.nn.functional.silu(x[..., :20]) * x[..., 20:]
            with torch.no_grad():
                self.gate_up_proj.weight.copy_(torch.cat((mlp.gate_proj.weight, mlp.up_proj.weight)))

    fused = nn.Module()
    fused.layers = nn.ModuleList([nn.ModuleDict({"mlp": FusedMLP()})])
    a = MLPChannelRDropController(num_layers=1, intermediate_size=20)
    b = MLPChannelRDropController(num_layers=1, intermediate_size=20)
    install_hf_mlp_intervention(model, a)
    install_vllm_mlp_intervention(fused, b)
    x = torch.randn(2, 3, 4)
    for route in (MASK_A_ROUTE, MASK_B_ROUTE, CLEAN_ROUTE):
        a.set_route(route)
        b.set_route(route)
        torch.testing.assert_close(mlp(x), fused.layers[0]["mlp"](x))


def test_route_quotas_preserve_prompt_identity_and_zero_mask_control():
    uids = np.array(["q1", "q2"] * 16, dtype=object)
    routes = assign_rdrop_routes(uids)
    for uid in ("q1", "q2"):
        assert sum(routes[uids == uid] == MASK_A_ROUTE) == 8
        assert sum(routes[uids == uid] == MASK_B_ROUTE) == 8
    saved = copy_prompt_uids_for_generation(uids, expected_size=32)
    saved[0] = "changed"
    assert uids[0] == "q1"
    with pytest.raises(ValueError):
        assign_rdrop_routes(np.array(["q"] * 3))
    c = MLPChannelRDropController(num_layers=2, intermediate_size=20, mask_ratio=0)
    assert c.keep_masks.all()
