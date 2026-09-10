#!/usr/bin/env bash
set -euxo pipefail

export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-./wandb_offline}
export PYTHONUNBUFFERED=1
mkdir -p "${WANDB_DIR}"

model_name=${model_name:-Qwen3-4B-Base}
num_gpus=${num_gpus:-4}
tensor_model_parallel_size=${tensor_model_parallel_size:-1}
sp_size=${sp_size:-1}
offload=${offload:-True}
ref_offload=${ref_offload:-True}

# Total budget remains 16: each prompt receives 8 positive and 8 negative rows.
n_total=${n_total:-16}
perturbation_strength=${perturbation_strength:-0.10}
random_seed=${random_seed:-42}
reward_update_enabled=${reward_update_enabled:-False}
reward_update_lr=${reward_update_lr:-1e-3}
reward_update_ratio=${reward_update_ratio:-0.05}
cross_kl_enabled=${cross_kl_enabled:-False}
cross_kl_coef=${cross_kl_coef:-0.01}
cross_kl_top_k=${cross_kl_top_k:-64}
cross_kl_token_chunk_size=${cross_kl_token_chunk_size:-128}
cross_kl_micro_batch_size_per_gpu=${cross_kl_micro_batch_size_per_gpu:-1}
cross_kl_gradient_sample_size_per_rank=${cross_kl_gradient_sample_size_per_rank:-262144}
case "${reward_update_enabled}" in
    True|true|1) reward_update_enabled=True ;;
    False|false|0) reward_update_enabled=False ;;
    *) echo "reward_update_enabled must be True or False" >&2; exit 2 ;;
esac
case "${cross_kl_enabled}" in
    True|true|1) cross_kl_enabled=True ;;
    False|false|0) cross_kl_enabled=False ;;
    *) echo "cross_kl_enabled must be True or False" >&2; exit 2 ;;
esac
if [[ "${cross_kl_enabled}" == "True" && "${reward_update_enabled}" == "True" ]]; then
    echo "cross-route KL requires reward_update_enabled=False" >&2
    exit 2
fi

if (( n_total < 2 || n_total % 2 != 0 )); then
    echo "n_total must be an even integer >= 2, got: ${n_total}" >&2
    exit 2
fi

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
actor_ppo_max_token_len=${actor_ppo_max_token_len:-$((2 * (max_prompt_length + max_response_length)))}
infer_ppo_max_token_len=${infer_ppo_max_token_len:-$((2 * (max_prompt_length + max_response_length)))}

RAY_DATA_HOME=${RAY_DATA_HOME:-.}
MODEL_PATH=${MODEL_PATH:-../Model/Qwen/${model_name}}
TRAIN_FILE=${TRAIN_FILE:-./data/MATH7500-train.parquet}
TEST_FILE=${TEST_FILE:-'["./data/aime25_test.parquet","./data/bbeh_data.parquet","./data/MATH500-test.parquet","./data/amc23_test.parquet","./data/aime24_test.parquet","./data/MMLU-Pro-Valid.parquet"]'}

project_name=${project_name:-MLP-Channel-Antithetic-4B}
default_experiment_name="grpo-${model_name}-antithetic-sigma${perturbation_strength}-n${n_total}"
if [[ "${reward_update_enabled}" == "True" ]]; then
    default_experiment_name+="-reward-update-lr${reward_update_lr}-ratio${reward_update_ratio}"
fi
if [[ "${cross_kl_enabled}" == "True" ]]; then
    default_experiment_name+="-cross-kl${cross_kl_coef}-top${cross_kl_top_k}-seed${random_seed}"
fi
experiment_name=${experiment_name:-${default_experiment_name}}
export WANDB_RUN_ID=${WANDB_RUN_ID:-${experiment_name}}
CKPTS_DIR=${CKPTS_DIR:-${RAY_DATA_HOME}/ckpts/${project_name}/${experiment_name}}

temperature=${temperature:-1.0}
top_p=${top_p:-1.0}
top_k=${top_k:--1}
val_temperature=${val_temperature:-0.6}
val_top_p=${val_top_p:-0.95}
# Keep the Linux training host's Python default and this Mac's documented Conda
# interpreter. An explicit python_bin override takes precedence on either host.
if [[ -z "${python_bin:-}" ]]; then
    if [[ "$(uname -s)" == "Darwin" ]]; then
        python_bin=/opt/homebrew/Caskroom/miniconda/base/envs/molu/bin/python
    else
        python_bin=python
    fi
fi

"${python_bin}" -m recipe.mlp_channel_antithetic.main \
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
    actor_rollout_ref.rollout.n=${n_total} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.data_parallel_size=1 \
    actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
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
    actor_rollout_ref.actor.rollout_n=${n_total} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps} \
    actor_rollout_ref.actor.optim.weight_decay=0 \
    ++actor_rollout_ref.actor.force_on_policy=True \
    ++actor_rollout_ref.actor.use_rollout_log_probs=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${ref_offload} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.mlp_channel_antithetic.enabled=True \
    actor_rollout_ref.mlp_channel_antithetic.perturbation_strength=${perturbation_strength} \
    actor_rollout_ref.mlp_channel_antithetic.random_seed=${random_seed} \
    actor_rollout_ref.mlp_channel_antithetic.refresh_every_step=True \
    actor_rollout_ref.mlp_channel_antithetic.reward_difference_update.enabled=${reward_update_enabled} \
    actor_rollout_ref.mlp_channel_antithetic.reward_difference_update.learning_rate=${reward_update_lr} \
    actor_rollout_ref.mlp_channel_antithetic.reward_difference_update.max_update_ratio=${reward_update_ratio} \
    actor_rollout_ref.mlp_channel_antithetic.cross_route_kl.enabled=${cross_kl_enabled} \
    actor_rollout_ref.mlp_channel_antithetic.cross_route_kl.kl_coef=${cross_kl_coef} \
    actor_rollout_ref.mlp_channel_antithetic.cross_route_kl.kl_top_k=${cross_kl_top_k} \
    actor_rollout_ref.mlp_channel_antithetic.cross_route_kl.kl_token_chunk_size=${cross_kl_token_chunk_size} \
    actor_rollout_ref.mlp_channel_antithetic.cross_route_kl.micro_batch_size_per_gpu=${cross_kl_micro_batch_size_per_gpu} \
    actor_rollout_ref.mlp_channel_antithetic.cross_route_kl.gradient_sample_size_per_rank=${cross_kl_gradient_sample_size_per_rank} \
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
    trainer.balance_batch=True \
    ++trainer.merge_duplicate_val_prompts=True \
    ++trainer.validation_pass_reward_threshold=0.0 \
    trainer.n_gpus_per_node=${num_gpus} \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=${test_and_save_freq} \
    trainer.save_freq=${test_and_save_freq} \
    trainer.total_epochs=${epoch} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    ++trainer.max_actor_ckpt_to_keep=1 \
    "$@"
