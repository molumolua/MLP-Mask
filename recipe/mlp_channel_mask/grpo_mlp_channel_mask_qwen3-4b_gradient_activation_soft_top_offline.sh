#!/usr/bin/env bash
set -euo pipefail

# Task-conditioned first-order importance: mean |z_j * dJ/dz_j|.
model_name=${model_name:-Qwen3-4B-Base}
export selection_strategy=${selection_strategy:-soft_top}
export score_method=${score_method:-gradient_activation}
export score_ema_beta=${score_ema_beta:-0.0}
export mask_ratio=${mask_ratio:-0.10}
export weighted_max_ratio=${weighted_max_ratio:-4.0}
export weighted_rank_power=${weighted_rank_power:-2.0}
export random_resample_every_step=${random_resample_every_step:-True}
export activation_update_every_step=${activation_update_every_step:-True}
export experiment_name=${experiment_name:-"grpo-${model_name}-gradient-activation-softtop${mask_ratio}"}

exec bash "$(dirname "$0")/grpo_mlp_channel_mask_qwen3-4b_offline.sh"
