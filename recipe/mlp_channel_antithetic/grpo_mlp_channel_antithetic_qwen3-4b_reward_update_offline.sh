#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Enable the auxiliary post-Adam channel update, capped at 5% per layer.
export reward_update_enabled=${reward_update_enabled:-True}
export reward_update_ratio=${reward_update_ratio:-0.05}
# Heuristic pre-cap channel-gain step size; this is not the main Adam LR.
# The actual auxiliary weight update is always bounded by reward_update_ratio.
export reward_update_lr=${reward_update_lr:-1e-3}

exec bash "${script_dir}/grpo_mlp_channel_antithetic_qwen3-4b_offline.sh" "$@"
