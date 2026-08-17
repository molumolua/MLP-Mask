#!/usr/bin/env bash
set -euo pipefail

# Every step samples exactly 10% from the flattened population of all MLP
# (layer, channel) pairs.  Per-layer mask counts may vary; the global count is exact.
model_name=${model_name:-Qwen3-4B-Base}
export model_name
export selection_strategy=${selection_strategy:-random}
export mask_ratio=${mask_ratio:-0.10}
export random_seed=${random_seed:-42}
export random_scope=${random_scope:-global}
export random_resample_every_step=${random_resample_every_step:-True}
export experiment_name=${experiment_name:-"grpo-${model_name}-clean8-masked8-globalrandom-every-step${mask_ratio}-seed${random_seed}"}

exec bash recipe/mlp_channel_mask/grpo_mlp_channel_mask_qwen3-4b_offline.sh "$@"
