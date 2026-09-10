"""Opt-in Gloo check for unequal-rank batches and cross-route KL gradients."""

from datetime import timedelta
import os

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from .test_cross_kl_actor import oracle, setup_actor


def run_rank(rank, world, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world,
                            timeout=timedelta(seconds=60))
    try:
        actor, reference, full_data = setup_actor(main_micro=3 if rank == 0 else 2, aux_micro=1, top_k=3)
        _, auxiliary, _, ga = oracle(actor, reference, full_data)
        local_data = full_data.select_idxs(np.arange(rank * 8, (rank + 1) * 8))
        main, _, gm, _ = oracle(actor, reference, local_data)
        # Preserve the existing antithetic PPO mean over local tokens and DP
        # averaging; only the new KL uses a global token denominator.
        dist.all_reduce(main)
        dist.all_reduce(gm)
        main /= world
        gm /= world
        actor.actor_module = DistributedDataParallel(actor.actor_module)
        metrics = actor.update_policy(local_data)
        torch.testing.assert_close(actor.final_gradient, gm + ga, atol=2e-7, rtol=3e-5)
        prefix = "mlp_antithetic/cross_kl/"
        assert metrics[prefix + "main_pg_loss_step"][0] == pytest.approx(float(main), abs=1e-6)
        assert metrics[prefix + "weighted_kl_step"][0] == pytest.approx(float(auxiliary), abs=1e-7)
        assert metrics[prefix + "aligned_response_rows"] == [16]
        assert metrics[prefix + "positive_to_negative_aligned_rows"] == [8]
        assert metrics[prefix + "negative_to_positive_aligned_rows"] == [8]
        assert metrics[prefix + "auxiliary_padding_slots"][0] > 0
        assert metrics[prefix + "auxiliary_forward_calls"] == metrics[prefix + "auxiliary_backward_calls"]
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get("RUN_CROSS_KL_DISTRIBUTED_TESTS") != "1", reason="opt-in Gloo test opens local sockets")
def test_two_rank_cross_kl_and_unchanged_main_objective(tmp_path):
    mp.spawn(run_rank, args=(2, str(tmp_path / "store")), nprocs=2, join=True)
