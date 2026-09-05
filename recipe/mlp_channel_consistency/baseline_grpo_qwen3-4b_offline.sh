#!/usr/bin/env bash
set -euo pipefail

# Matched clean-GRPO control. It reuses the same data, rollout, optimization,
# validation, checkpoint, and diagnostic paths, but performs no mask sampling,
# teacher-distribution capture, masked forward, auxiliary loss, or auxiliary
# backward. Auxiliary metric keys remain present and are exactly zero.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export auxiliary_enabled=False
export kl_coef=0
export experiment_name=${experiment_name:-"grpo-${model_name:-Qwen3-4B-Base}-baseline-no-aux"}

exec bash "${SCRIPT_DIR}/grpo_mlp_channel_consistency_qwen3-4b_offline.sh" "$@"
