#!/usr/bin/env bash
set -euo pipefail

# Online randomized grouped-ablation estimate from the realized reward gap.
# This is a tractable causal estimate, not exhaustive one-channel-at-a-time ablation.
model_name=${model_name:-Qwen3-4B-Base}
export selection_strategy=${selection_strategy:-soft_top}
export score_method=${score_method:-causal_ablation}
export score_ema_beta=${score_ema_beta:-0.95}
export mask_ratio=${mask_ratio:-0.10}
export weighted_max_ratio=${weighted_max_ratio:-4.0}
export weighted_rank_power=${weighted_rank_power:-2.0}
export random_resample_every_step=${random_resample_every_step:-True}
export activation_update_every_step=${activation_update_every_step:-False}
export experiment_name=${experiment_name:-"grpo-${model_name}-causal-group-ablation-softtop${mask_ratio}"}

exec bash "$(dirname "$0")/grpo_mlp_channel_mask_qwen3-4b_offline.sh"
