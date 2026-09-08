"""Opt-in CPU Gloo check of real FSDP1 and DTensor parameter storage.

RUN_REWARD_UPDATE_DISTRIBUTED_TESTS=1 ./scripts/test-local \
    recipe/mlp_channel_antithetic/test_reward_update_distributed.py -q

Gloo opens local sockets; macOS sandboxed shells may need permission to run it.
"""

from __future__ import annotations

import copy
import os

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

from .reward_update import RewardDifferenceUpdater, RewardUpdateConfig, _normalized_name
from .test_reward_update import ToyModel, make_controller


def _run_distributed(rank, world_size, rendezvous):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size)
    try:
        mesh = init_device_mesh("cpu", (world_size,))
        for strategy in ("fsdp", "fsdp_orig", "dtensor_rows", "dtensor_columns", "dtensor_replicated"):
            torch.manual_seed(100)
            model = ToyModel()
            reference = copy.deepcopy(model)
            baseline = copy.deepcopy(model)
            if strategy.startswith("fsdp"):
                # Each layer's down projection lives only on rank 0. Rank 1
                # must still join every statistics reduction with empty slices.
                for i, layer in enumerate(model.layers):
                    model.layers[i] = FSDP(layer, device_id=torch.device("cpu"), use_orig_params=strategy == "fsdp_orig")
                model = FSDP(model, device_id=torch.device("cpu"), use_orig_params=strategy == "fsdp_orig")
            else:
                placement = {"dtensor_rows": Shard(0), "dtensor_columns": Shard(1), "dtensor_replicated": Replicate()}[strategy]
                for name, parameter in list(model.named_parameters()):
                    parent_name, leaf = name.rsplit(".", 1)
                    setattr(model.get_submodule(parent_name), leaf, torch.nn.Parameter(distribute_tensor(parameter.detach(), mesh, [placement])))
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.02)
            reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01, weight_decay=0.02)
            baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=0.01, weight_decay=0.02)
            config = RewardUpdateConfig(True, 0.1, 0.05)
            updater = RewardDifferenceUpdater(model, optimizer, make_controller(), config)
            # Replicated reference tensors also reduce their statistics, testing
            # the correction for replicated (rather than unique) element counts.
            reference_updater = RewardDifferenceUpdater(reference, reference_optimizer, make_controller(), config)
            if strategy.startswith("fsdp"):
                # Exercise real unshard/forward/backward/reshard before stepping.
                model(torch.ones(2, 3)).sum().backward()
                model.zero_grad(set_to_none=True)
            for network in (model, reference, baseline):
                for parameter in network.parameters():
                    parameter.grad = torch.ones_like(parameter)
            old = {name: p.detach().clone() for name, p in baseline.named_parameters()}
            baseline_optimizer.step()
            for update, opt in ((updater, optimizer), (reference_updater, reference_optimizer)):
                update.begin_batch({"reward_gap": 0.5, "version": 0}, np.array([0]))
                opt.step()
                update.end_batch()
            if strategy.startswith("fsdp"):
                with FSDP.summon_full_params(model, writeback=False):
                    actual = {_normalized_name(name): p.detach().clone() for name, p in model.named_parameters()}
            else:
                actual = {name: p.full_tensor().detach() if isinstance(p, DTensor) else p.detach().clone() for name, p in model.named_parameters()}
            for name, expected in reference.named_parameters():
                torch.testing.assert_close(actual[name], expected, atol=1e-7, rtol=1e-6, msg=lambda message: f"{strategy}: {message}")
                if "down_proj" in name:
                    base = dict(baseline.named_parameters())[name].detach()
                    aux_norm = (actual[name].double() - base.double()).norm()
                    main_norm = (base.double() - old[name].double()).norm()
                    assert aux_norm <= 0.05 * main_norm * (1 + 1e-10), strategy
            assert updater.last_metrics["mlp_antithetic/reward_update/max_layer_ratio"] <= 0.05
            updater.close()
            reference_updater.close()
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get("RUN_REWARD_UPDATE_DISTRIBUTED_TESTS") != "1", reason="opt-in test opens local Gloo sockets")
def test_two_rank_fsdp_and_dtensor_updates_match_unsharded_reference(tmp_path):
    mp.spawn(_run_distributed, args=(2, str(tmp_path / "gloo_store")), nprocs=2, join=True)
