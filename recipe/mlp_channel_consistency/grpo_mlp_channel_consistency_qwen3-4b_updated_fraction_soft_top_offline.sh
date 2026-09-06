#!/usr/bin/env bash
set -euo pipefail

# Prefer channels with the largest cumulative BF16 updated-coordinate fraction
# relative to the pre-RL model. This FSDP2-only mode scans MLP parameters each step.
export selection_strategy=${selection_strategy:-soft_top}
export score_method=${score_method:-updated_fraction}
export score_ema_beta=${score_ema_beta:-0.0}
export weighted_max_ratio=${weighted_max_ratio:-4.0}
export weighted_rank_power=${weighted_rank_power:-2.0}
export parameter_update_diagnostics_enabled=True

exec bash "$(dirname "$0")/grpo_mlp_channel_consistency_qwen3-4b_offline.sh" "$@"
