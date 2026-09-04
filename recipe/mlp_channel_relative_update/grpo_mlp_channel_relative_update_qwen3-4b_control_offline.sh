#!/usr/bin/env bash
set -euxo pipefail

# Matched control: the custom optimizer and statistics remain enabled, but a
# ratio cap of one forces every channel multiplier to exactly one.
export multiplier_ratio_cap=${multiplier_ratio_cap:-1.0}
export experiment_name=${experiment_name:-"grpo-${model_name:-Qwen3-4B-Base}-relative-update-control-r1"}

exec bash recipe/mlp_channel_relative_update/grpo_mlp_channel_relative_update_qwen3-4b_offline.sh "$@"
