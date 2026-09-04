#!/usr/bin/env bash
set -euxo pipefail

export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-./wandb_offline}
export PYTHONUNBUFFERED=1
mkdir -p "${WANDB_DIR}"

model_name=${model_name:-Qwen3-4B-Base}
num_gpus=${num_gpus:-4}
tensor_model_parallel_size=${tensor_model_parallel_size:-1}

# History-based relative-update allocator.
relative_update_enabled=${relative_update_enabled:-True}
optimizer_impl=${optimizer_impl:-recipe.mlp_channel_relative_update.optimizer}
optimizer_name=${optimizer_name:-ChannelRelativeUpdateAdamW}
history_ema_beta=${history_ema_beta:-0.99}
history_power=${history_power:-0.5}
history_floor_ratio=${history_floor_ratio:-0.1}
multiplier_ratio_cap=${multiplier_ratio_cap:-10.0}
warmup_steps=${warmup_steps:-16}

# GRPO schedule. Keep these equal across baseline/rarity/relative-update runs.
n_rollouts=${n_rollouts:-16}
epoch=${epoch:-10000}
lr=${lr:-1e-6}
lr_warmup_steps=${lr_warmup_steps:-0}
test_and_save_freq=${test_and_save_freq:-40}
train_prompt_bsz=${train_prompt_bsz:-16}
train_prompt_mini_bsz=${train_prompt_mini_bsz:-16}
max_prompt_length=${max_prompt_length:-8192}
max_response_length=${max_response_length:-4096}
gpu_memory_utilization=${gpu_memory_utilization:-0.7}
use_dynamic_bsz=${use_dynamic_bsz:-True}
log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu:-1}
actor_ppo_max_token_len=$((2 * (max_prompt_length + max_response_length)))
infer_ppo_max_token_len=$((2 * (max_prompt_length + max_response_length)))

RAY_DATA_HOME=${RAY_DATA_HOME:-.}
MODEL_PATH=${MODEL_PATH:-../Model/Qwen/${model_name}}
TRAIN_FILE=${TRAIN_FILE:-./data/MATH7500-train.parquet}
TEST_FILE=${TEST_FILE:-'["./data/aime25_test.parquet","./data/bbeh_data.parquet","./data/MATH500-test.parquet","./data/amc23_test.parquet","./data/aime24_test.parquet","./data/MMLU-Pro-Valid.parquet"]'}

project_name=${project_name:-MLP-Channel-Relative-Update-4B}
experiment_name=${experiment_name:-"grpo-${model_name}-relative-update-r${multiplier_ratio_cap}-beta${history_ema_beta}-power${history_power}"}
export WANDB_RUN_ID=${WANDB_RUN_ID:-${experiment_name}}
CKPTS_DIR=${CKPTS_DIR:-${RAY_DATA_HOME}/ckpts/${project_name}/${experiment_name}}
rollout_data_dir=${rollout_data_dir:-${CKPTS_DIR}/rollout_data}

temperature=${temperature:-1.0}
top_p=${top_p:-1.0}
top_k=${top_k:--1}
val_temperature=${val_temperature:-0.6}
val_top_p=${val_top_p:-0.95}
python_bin=${python_bin:-/opt/homebrew/Caskroom/miniconda/base/envs/molu/bin/python}

"${python_bin}" -m recipe.mlp_channel_relative_update.main \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.shuffle=False \
    data.dataloader_num_workers=0 \
    data.truncation=left \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.val_batch_size=512 \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.attention_dropout=0.0 \
    +actor_rollout_ref.model.override_config.embd_pdrop=0.0 \
    +actor_rollout_ref.model.override_config.resid_pdrop=0.0 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.n=${n_rollouts} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_model_parallel_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.rollout_n=${n_rollouts} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.optim.optimizer_impl=${optimizer_impl} \
    actor_rollout_ref.actor.optim.optimizer=${optimizer_name} \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps} \
    actor_rollout_ref.actor.optim.weight_decay=0 \
    ++actor_rollout_ref.actor.force_on_policy=True \
    ++actor_rollout_ref.actor.use_rollout_log_probs=True \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.offload_policy=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.mlp_channel_relative_update.enabled=${relative_update_enabled} \
    actor_rollout_ref.mlp_channel_relative_update.history_ema_beta=${history_ema_beta} \
    actor_rollout_ref.mlp_channel_relative_update.history_power=${history_power} \
    actor_rollout_ref.mlp_channel_relative_update.history_floor_ratio=${history_floor_ratio} \
    actor_rollout_ref.mlp_channel_relative_update.multiplier_ratio_cap=${multiplier_ratio_cap} \
    actor_rollout_ref.mlp_channel_relative_update.warmup_steps=${warmup_steps} \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    ++algorithm.rollout_correction.bypass_old_logprob_for_rollout=False \
    ++algorithm.rollout_correction.rollout_is=null \
    ++algorithm.rollout_correction.rollout_rs=null \
    reward_model.reward_manager=naive \
    trainer.logger="['console','wandb']" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    ++trainer.merge_duplicate_val_prompts=True \
    ++trainer.validation_pass_reward_threshold=0.0 \
    trainer.n_gpus_per_node=${num_gpus} \
    trainer.nnodes=1 \
    trainer.balance_batch=True \
    trainer.val_before_train=False \
    trainer.test_freq=${test_and_save_freq} \
    trainer.save_freq=${test_and_save_freq} \
    trainer.total_epochs=${epoch} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.rollout_data_dir="${rollout_data_dir}" \
    trainer.resume_mode=auto \
    ++trainer.max_actor_ckpt_to_keep=1
