#!/usr/bin/env bash
set -euo pipefail

# Random-channel ablation baseline: every refresh independently samples exactly
# 1% of the intermediate channels in every Transformer MLP block.  The seed and
# mask version make the sampling reproducible while still changing across refreshes.
model_name=${model_name:-Qwen3-4B-Base}
export model_name
export selection_strategy=${selection_strategy:-random}
export mask_ratio=${mask_ratio:-0.01}
export random_seed=${random_seed:-42}
export random_scope=${random_scope:-per_layer}
export experiment_name=${experiment_name:-"grpo-${model_name}-clean8-masked8-blockrandom${mask_ratio}-seed${random_seed}"}

exec bash recipe/mlp_channel_mask/grpo_mlp_channel_mask_qwen3-4b_offline.sh "$@"
