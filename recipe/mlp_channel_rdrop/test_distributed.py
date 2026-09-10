"""Opt-in two-rank CPU Gloo check; no model downloads or CUDA required."""

from datetime import timedelta
import os

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor
from torch.nn.parallel import DistributedDataParallel

from .diagnostics import ParameterUpdateTracker
from .test_actor import oracle, setup_actor


def run_rank(rank, world, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world,
                            timeout=timedelta(seconds=60))
    try:
        actor, reference, full_data = setup_actor(main_micro=3 if rank == 0 else 2, aux_micro=1, top_k=3)
        main, auxiliary, gm, ga = oracle(actor, reference, full_data)
        actor.actor_module = DistributedDataParallel(actor.actor_module)
        # Unequal token counts and different inner shapes; still two main
        # micro-batches per route on both ranks (3+1 vs 2+2).
        local_data = full_data.select_idxs(np.arange(rank * 8, (rank + 1) * 8))
        metrics = actor.update_policy(local_data)
        torch.testing.assert_close(actor.final_gradient, gm + ga, atol=2e-7, rtol=3e-5)
        assert metrics["mlp_rdrop/main_pg_loss_step"][0] == pytest.approx(float(main), abs=1e-6)
        assert metrics["mlp_rdrop/weighted_kl_step"][0] == pytest.approx(float(auxiliary), abs=1e-7)
        assert metrics["mlp_rdrop/aligned_response_rows"] == [16]
        assert metrics["mlp_rdrop/a_to_b_aligned_rows"] == [8]
        assert metrics["mlp_rdrop/b_to_a_aligned_rows"] == [8]
        assert metrics["mlp_rdrop/auxiliary_forward_calls"] == metrics["mlp_rdrop/auxiliary_backward_calls"]
        assert metrics["mlp_rdrop/auxiliary_padding_slots"][0] > 0
        dist.barrier()

        # Parameter diagnostics use actual DTensor shards, not duplicated full
        # parameters. Three of sixteen coordinates differ from the pre-RL weights.
        mesh = init_device_mesh("cpu", (world,))
        module = torch.nn.Module()
        module.weight = torch.nn.Parameter(distribute_tensor(torch.zeros(8, 2), mesh, [Shard(0)]))
        tracker = ParameterUpdateTracker(module)
        changed = torch.zeros(8, 2)
        changed[0, 0] = 1
        changed[5, 1] = 2
        changed[6, 0] = 3
        with torch.no_grad():
            module.weight.copy_(distribute_tensor(changed, mesh, [Shard(0)]))
        measured = tracker.distributed_metrics()
        assert measured["val-aux/parameter_update/updated_parameter_count"] == 3
        assert measured["val-aux/parameter_update/updated_fraction_atol_1e-5"] == 3 / 16
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get("RUN_RDROP_DISTRIBUTED_TESTS") != "1", reason="opt-in Gloo test opens local sockets")
def test_two_rank_normalization_padding_and_parameter_shards(tmp_path):
    mp.spawn(run_rank, args=(2, str(tmp_path / "store")), nprocs=2, join=True)
