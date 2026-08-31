#!/usr/bin/env bash
set -euo pipefail

# Pure GRPO baseline for the MLP-channel experiments. It reuses the common
# launcher so data, optimization, sampling, validation, and checkpoint settings
# stay identical. Each prompt receives the same total budget of 16 rollouts, but
# all rollouts use the standard clean policy and one GRPO advantage group.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export mlp_intervention_enabled=False
export n_total=${n_total:-16}
export experiment_name=${experiment_name:-"grpo-${model_name:-Qwen3-4B-Base}-baseline-clean${n_total}"}

exec bash "${SCRIPT_DIR}/grpo_mlp_channel_mask_qwen3-4b_offline.sh" "$@"
