#!/usr/bin/env bash
set -euo pipefail

# Stronger per-layer weighted-random intervention at the same 10% mask budget.
# With relative-activation percentile rank r in [0, 1], these defaults implement
#
#   weight(r) = 1 + 10 * r^2
#
# This is a moderate bridge between uniform random and deterministic hard-top:
# it increases top-rank sampling pressure without making a failed hard-top mask
# inevitable. All values remain environment-overridable for controlled sweeps.
model_name=${model_name:-Qwen3-4B-Base}
export model_name
export selection_strategy=${selection_strategy:-weighted_random}
export mask_ratio=${mask_ratio:-0.10}
export random_seed=${random_seed:-42}
export random_scope=${random_scope:-per_layer}
export weighted_max_ratio=${weighted_max_ratio:-11.0}
export weighted_rank_power=${weighted_rank_power:-2.0}
export random_resample_every_step=${random_resample_every_step:-True}
export activation_update_every_step=${activation_update_every_step:-True}
export activation_ema_beta=${activation_ema_beta:-0.0}
export experiment_name=${experiment_name:-"grpo-${model_name}-clean8-masked8-relativeactivation-weightedrandom-strong-every-step${mask_ratio}-r${weighted_max_ratio}-p${weighted_rank_power}-seed${random_seed}"}

exec bash recipe/mlp_channel_mask/grpo_mlp_channel_mask_qwen3-4b_offline.sh "$@"
