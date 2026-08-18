#!/usr/bin/env bash
set -euo pipefail

# Per-layer weighted random intervention.  Every step masks exactly 10% of each
# MLP block without replacement.  Every clean backward collects current-step
# saliency; beta=0 replaces the previous score instead of applying an EMA.
# Saliency percentile rank sets a fixed weight in [1, weighted_max_ratio]; there
# is intentionally no weight warmup or ramp.
model_name=${model_name:-Qwen3-4B-Base}
export model_name
export selection_strategy=${selection_strategy:-weighted_random}
export mask_ratio=${mask_ratio:-0.10}
export random_seed=${random_seed:-42}
export random_scope=${random_scope:-per_layer}
export weighted_max_ratio=${weighted_max_ratio:-4.0}
export weighted_rank_power=${weighted_rank_power:-2.0}
export random_resample_every_step=${random_resample_every_step:-True}
export saliency_update_every_step=${saliency_update_every_step:-True}
export saliency_ema_beta=${saliency_ema_beta:-0.0}
export experiment_name=${experiment_name:-"grpo-${model_name}-clean8-masked8-weightedrandom-every-step${mask_ratio}-r${weighted_max_ratio}-p${weighted_rank_power}-seed${random_seed}"}

exec bash recipe/mlp_channel_mask/grpo_mlp_channel_mask_qwen3-4b_offline.sh "$@"
