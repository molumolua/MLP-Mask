#!/usr/bin/env bash
set -euo pipefail

# Prefer channels with large RMS(z_j) * ||W_down[:, j]||_2 on clean responses.
export selection_strategy=${selection_strategy:-soft_top}
export score_method=${score_method:-output_contribution}
export score_ema_beta=${score_ema_beta:-0.0}
export weighted_max_ratio=${weighted_max_ratio:-4.0}
export weighted_rank_power=${weighted_rank_power:-2.0}

exec bash "$(dirname "$0")/grpo_mlp_channel_consistency_qwen3-4b_offline.sh" "$@"
