#!/usr/bin/env bash
set -euxo pipefail

# True standard control: identical FSDP2/GRPO launch settings, but the component
# is disabled and the actor uses the native torch.optim.AdamW implementation.
export relative_update_enabled=False
export optimizer_impl=torch.optim
export optimizer_name=AdamW
export experiment_name=${experiment_name:-"grpo-${model_name:-Qwen3-4B-Base}-standard-adamw-fsdp2"}

exec bash recipe/mlp_channel_relative_update/grpo_mlp_channel_relative_update_qwen3-4b_offline.sh "$@"
