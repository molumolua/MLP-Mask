#!/usr/bin/env bash
set -euo pipefail

# Reuse positive/negative PPO teachers; replace the post-Adam reward update.
export cross_kl_enabled=True
export reward_update_enabled=False
recipe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${recipe_dir}/grpo_mlp_channel_antithetic_qwen3-4b_offline.sh" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.use_fused_kernels=False \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    ++actor_rollout_ref.model.override_config.attn_pdrop=0.0 \
    "$@" \
    actor_rollout_ref.mlp_channel_antithetic.reward_difference_update.enabled=False \
    actor_rollout_ref.mlp_channel_antithetic.cross_route_kl.enabled=True
