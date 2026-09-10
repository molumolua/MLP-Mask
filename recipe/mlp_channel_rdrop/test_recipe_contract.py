import ast
import os
from pathlib import Path
import subprocess

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from .validation import validate_recipe_config

ROOT = Path(__file__).resolve().parent


def config():
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        return compose(config_name="ppo_mlp_channel_rdrop")


def test_composed_defaults_and_supported_controls():
    c = config()
    validate_recipe_config(c)
    assert c.actor_rollout_ref.rollout.n == 16
    assert c.actor_rollout_ref.mlp_channel_rdrop.mask_ratio == 0.1
    c.actor_rollout_ref.mlp_channel_rdrop.auxiliary_enabled = False
    c.actor_rollout_ref.mlp_channel_rdrop.kl_coef = 0
    validate_recipe_config(c)
    c.actor_rollout_ref.mlp_channel_rdrop.mask_ratio = 0
    validate_recipe_config(c)


@pytest.mark.parametrize("key,value", [
    ("actor_rollout_ref.rollout.n", 15),
    ("actor_rollout_ref.rollout.mode", "async"),
    ("actor_rollout_ref.rollout.top_p", 0.95),
    ("actor_rollout_ref.rollout.temperature", float("nan")),
    ("actor_rollout_ref.actor.ppo_epochs", 2),
    ("actor_rollout_ref.actor.entropy_coeff", 0.01),
    ("actor_rollout_ref.actor.use_fused_kernels", True),
    ("actor_rollout_ref.model.use_fused_kernels", True),
    ("actor_rollout_ref.actor.ulysses_sequence_parallel_size", 2),
    ("actor_rollout_ref.actor.strategy", "fsdp"),
    ("actor_rollout_ref.mlp_channel_rdrop.kl_coef", -1),
    ("actor_rollout_ref.mlp_channel_rdrop.random_seed", -1),
    ("actor_rollout_ref.mlp_channel_rdrop.micro_batch_size_per_gpu", 0),
    ("actor_rollout_ref.model.override_config.attention_dropout", 0.1),
])
def test_invalid_configuration_fails_early(key, value):
    c = config()
    OmegaConf.update(c, key, value)
    with pytest.raises((ValueError, NotImplementedError)):
        validate_recipe_config(c)


def test_launchers_resolve_real_hydra_arguments_and_keep_controls_distinct(tmp_path):
    stub = tmp_path / "capture"
    stub.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE_ARGUMENTS"\n')
    stub.chmod(0o755)
    names = []
    for filename in ("grpo_mlp_channel_rdrop_qwen3-4b_offline.sh", "baseline_no_aux_qwen3-4b_offline.sh",
                     "baseline_clean_qwen3-4b_offline.sh"):
        env = dict(os.environ)
        for key in ("experiment_name", "WANDB_RUN_ID", "CKPTS_DIR", "mask_ratio", "kl_coef", "auxiliary_enabled"):
            env.pop(key, None)
        capture = tmp_path / "arguments"
        env.update(python_bin=str(stub), CAPTURE_ARGUMENTS=str(capture), WANDB_DIR=str(tmp_path / "wandb"),
                   actor_ppo_max_token_len="4096", infer_ppo_max_token_len="8192")
        subprocess.run(["bash", str(ROOT / filename), "trainer.total_epochs=2"], env=env,
                       check=True, capture_output=True, text=True)
        args = capture.read_text().splitlines()
        assert args[:2] == ["-m", "recipe.mlp_channel_rdrop.main"]
        with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
            c = compose(config_name="ppo_mlp_channel_rdrop", overrides=args[2:])
            validate_recipe_config(c)
            assert c.trainer.total_epochs == 2
            assert c.actor_rollout_ref.actor.ppo_max_token_len_per_gpu == 4096
            assert c.actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu == 8192
            names.append(c.trainer.experiment_name)
    assert len(set(names)) == 3


def test_runtime_has_no_dependency_on_other_recipes():
    for path in ROOT.glob("*.py"):
        if path.name.startswith("test_"):
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("recipe."):
                assert node.module.startswith("recipe.mlp_channel_rdrop")
