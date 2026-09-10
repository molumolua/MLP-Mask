import ast
import os
from pathlib import Path
import subprocess

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from .cross_kl_config import validate_cross_kl_config, validate_cross_kl_worker_config
from .reward_update import RewardUpdateConfig

ROOT = Path(__file__).resolve().parent


def config(enabled=True):
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        return compose(config_name="ppo_mlp_channel_antithetic", overrides=[
            f"actor_rollout_ref.mlp_channel_antithetic.cross_route_kl.enabled={enabled}",
            "actor_rollout_ref.actor.strategy=fsdp2",
            "actor_rollout_ref.actor.entropy_coeff=0.0",
        ])


def test_defaults_keep_both_auxiliaries_off_and_disabled_path_unchanged():
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        baseline = compose(config_name="ppo_mlp_channel_antithetic")
    validate_cross_kl_config(baseline)
    component = baseline.actor_rollout_ref.mlp_channel_antithetic
    assert not component.cross_route_kl.enabled
    assert not component.reward_difference_update.enabled
    component.reward_difference_update.enabled = True
    # Original reward update can still use its original actor strategy/config.
    validate_cross_kl_config(baseline)
    validate_cross_kl_config(config())


@pytest.mark.parametrize("field,value", [
    ("mlp_channel_antithetic.reward_difference_update.enabled", True),
    ("mlp_channel_antithetic.cross_route_kl.kl_coef", -0.1),
    ("mlp_channel_antithetic.cross_route_kl.kl_coef", float("nan")),
    ("mlp_channel_antithetic.cross_route_kl.kl_top_k", -1),
    ("mlp_channel_antithetic.cross_route_kl.micro_batch_size_per_gpu", 0),
    ("actor.strategy", "fsdp"),
    ("actor.fsdp_config.fsdp_size", 2),
    ("actor.ppo_epochs", 2),
    ("actor.entropy_coeff", 0.01),
    ("actor.use_kl_loss", True),
    ("actor.ulysses_sequence_parallel_size", 2),
    ("model.use_fused_kernels", True),
    ("rollout.top_p", 0.9),
    ("rollout.temperature", 0.0),
])
def test_unsupported_config_fails_before_worker_allocation(field, value):
    c = config()
    OmegaConf.update(c.actor_rollout_ref, field, value)
    with pytest.raises((ValueError, NotImplementedError)):
        validate_cross_kl_worker_config(c.actor_rollout_ref)


@pytest.mark.parametrize("field,value", [
    ("algorithm.rollout_correction.bypass_old_logprob_for_rollout", True),
    ("algorithm.use_kl_in_reward", True),
    ("trainer.balance_batch", False),
])
def test_driver_rejects_unsupported_objectives(field, value):
    c = config()
    OmegaConf.update(c, field, value)
    with pytest.raises(ValueError):
        validate_cross_kl_config(c)


def test_real_launchers_compose_and_new_entrypoint_forces_reward_update_off(tmp_path):
    stub = tmp_path / "capture"
    stub.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE_ARGUMENTS"\n')
    stub.chmod(0o755)
    names = []
    for suffix in ("offline", "reward_update_offline", "cross_kl_offline"):
        env = dict(os.environ)
        for key in ("experiment_name", "WANDB_RUN_ID", "CKPTS_DIR", "reward_update_enabled", "cross_kl_enabled", "offload"):
            env.pop(key, None)
        capture = tmp_path / "arguments"
        env.update(python_bin=str(stub), CAPTURE_ARGUMENTS=str(capture), WANDB_DIR=str(tmp_path / "wandb"))
        extra = ["trainer.total_epochs=2"]
        if suffix == "cross_kl_offline":
            # Both inherited environment and accidental CLI flags must lose to
            # this dedicated entrypoint's explicit replacement of the old update.
            env.update(reward_update_enabled="True", cross_kl_coef="0.02", cross_kl_top_k="32")
            extra.append("actor_rollout_ref.mlp_channel_antithetic.reward_difference_update.enabled=True")
        subprocess.run(["bash", str(ROOT / f"grpo_mlp_channel_antithetic_qwen3-4b_{suffix}.sh"), *extra],
                       env=env, check=True, capture_output=True, text=True)
        args = capture.read_text().splitlines()
        assert args[:2] == ["-m", "recipe.mlp_channel_antithetic.main"]
        with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
            c = compose(config_name="ppo_mlp_channel_antithetic", overrides=args[2:])
        validate_cross_kl_config(c)
        assert c.trainer.total_epochs == 2
        assert c.actor_rollout_ref.rollout.n == 16
        component = c.actor_rollout_ref.mlp_channel_antithetic
        assert component.cross_route_kl.enabled == (suffix == "cross_kl_offline")
        assert RewardUpdateConfig.from_config(component.reward_difference_update).active == (suffix == "reward_update_offline")
        if suffix == "cross_kl_offline":
            assert component.cross_route_kl.kl_coef == 0.02
            assert component.cross_route_kl.kl_top_k == 32
            assert c.actor_rollout_ref.actor.fsdp_config.param_offload
            assert c.actor_rollout_ref.actor.fsdp_config.optimizer_offload
            assert "cross-kl0.02-top32" in c.trainer.experiment_name
        names.append(c.trainer.experiment_name)
    assert len(set(names)) == 3


def test_new_runtime_does_not_import_other_recipes():
    for name in ("actor.py", "kl.py", "batching.py", "diagnostics.py", "cross_kl_config.py"):
        for node in ast.walk(ast.parse((ROOT / name).read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("recipe.")
