#!/usr/bin/env bash
set -euo pipefail
export auxiliary_enabled=False
export kl_coef=0
export mask_ratio=0
recipe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${recipe_dir}/grpo_mlp_channel_rdrop_qwen3-4b_offline.sh" "$@"
