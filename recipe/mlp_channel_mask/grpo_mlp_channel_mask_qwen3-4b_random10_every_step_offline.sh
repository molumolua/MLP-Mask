#!/usr/bin/env bash
set -euo pipefail

# Every training step independently resamples exactly 10% of the intermediate
# channels in every Transformer MLP block.  A mask is sampled before step 1 and
# remains fixed throughout that step's rollout, old-logprob, and actor update.
model_name=${model_name:-Qwen3-4B-Base}
export model_name
export selection_strategy=${selection_strategy:-random}
export mask_ratio=${mask_ratio:-0.10}
export random_seed=${random_seed:-42}
export random_scope=${random_scope:-per_layer}
export random_resample_every_step=${random_resample_every_step:-True}
export experiment_name=${experiment_name:-"grpo-${model_name}-clean8-masked8-blockrandom-every-step${mask_ratio}-seed${random_seed}"}

exec bash recipe/mlp_channel_mask/grpo_mlp_channel_mask_qwen3-4b_offline.sh "$@"
