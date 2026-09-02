#!/usr/bin/env bash
set -euo pipefail

recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export loss_weight_amplification=${loss_weight_amplification:-10.0}
export min_loss_weight=${min_loss_weight:-0.2}
export max_loss_weight=${max_loss_weight:-5.0}
export experiment_name=${experiment_name:-"grpo-${model_name:-Qwen3-4B-Base}-ema-rarity-top${topk_ratio:-0.01}-amp10-0.2-5.0"}

exec bash "${recipe_dir}/grpo_mlp_channel_rarity_qwen3-4b_offline.sh"
