"""Fail before allocating model workers for unsupported recipe combinations."""

import math


def validate_recipe_config(config):
    ar = config.actor_rollout_ref
    actor, rollout, model = ar.actor, ar.rollout, ar.model
    component = ar.mlp_channel_rdrop
    if not component.enabled:
        raise ValueError("mlp_channel_rdrop.enabled must be true")
    if actor.strategy != "fsdp2" or int(actor.fsdp_config.fsdp_size) != -1:
        raise NotImplementedError("R-Drop currently requires fsdp2 with fsdp_size=-1 (full sharding)")
    if config.algorithm.adv_estimator != "grpo":
        raise NotImplementedError("this recipe uses outcome GRPO with the standard clipped PPO policy loss")
    if actor.policy_loss.loss_mode != "vanilla" or actor.loss_agg_mode != "token-mean":
        raise ValueError("use vanilla clipped policy loss and token-mean aggregation")
    if not actor.get("force_on_policy", False) or int(actor.ppo_epochs) != 1:
        raise ValueError("one complete-batch optimizer step is required: force_on_policy=true, ppo_epochs=1")
    if float(actor.entropy_coeff) != 0 or actor.use_kl_loss or config.algorithm.use_kl_in_reward:
        raise ValueError("disable entropy/reference terms so main/aux diagnostics describe PPO and R-Drop exactly")
    if (int(actor.ulysses_sequence_parallel_size) != 1 or actor.get("use_fused_kernels", False)
            or model.get("use_fused_kernels", False)):
        raise NotImplementedError("full response KL requires SP=1 and use_fused_kernels=false")
    if int(model.get("lora_rank", 0)) > 0:
        raise NotImplementedError("this recipe currently requires full dense weights")
    if rollout.name != "vllm" or rollout.mode != "sync":
        raise NotImplementedError("two-mask generation requires synchronous vLLM")
    if int(rollout.pipeline_model_parallel_size) != 1 or int(rollout.data_parallel_size) != 1:
        raise NotImplementedError("vLLM requires PP=1 and DP=1; TP is supported")
    if int(rollout.n) < 2 or int(rollout.n) % 2:
        raise ValueError("rollout.n must be an even integer >= 2")
    if not rollout.calculate_log_probs or not actor.get("use_rollout_log_probs", False):
        raise ValueError("enable rollout log-probs and actor.use_rollout_log_probs")
    correction = config.algorithm.rollout_correction
    if correction.bypass_old_logprob_for_rollout or correction.rollout_is or correction.rollout_rs:
        raise ValueError("use actor-recomputed route-conditioned old log-probs without rollout correction")
    if not config.trainer.balance_batch or config.reward_model.launch_reward_fn_async:
        raise ValueError("enable route-aware balance_batch and disable async reward functions")
    if rollout.multi_turn.enable:
        raise NotImplementedError("this recipe currently supports single-turn text generation")
    temperature = float(rollout.temperature)
    if not math.isfinite(temperature) or temperature <= 0 or float(rollout.top_p) != 1 or int(rollout.top_k) != -1:
        raise ValueError("training requires temperature>0, top_p=1, top_k=-1 to match actor probabilities")
    if not 0 <= float(component.mask_ratio) < 1:
        raise ValueError("mask_ratio must be in [0, 1)")
    if int(component.random_seed) < 0:
        raise ValueError("random_seed must be non-negative")
    if not math.isfinite(float(component.kl_coef)) or float(component.kl_coef) < 0:
        raise ValueError("kl_coef must be finite and non-negative")
    if int(component.kl_top_k) < 0:
        raise ValueError("kl_top_k must be non-negative")
    if not component.auxiliary_enabled and float(component.kl_coef) != 0:
        raise ValueError("auxiliary_enabled=false requires kl_coef=0")
    for key in ("micro_batch_size_per_gpu", "kl_token_chunk_size", "gradient_sample_size_per_rank"):
        if int(component[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("attention_dropout", "embd_pdrop", "resid_pdrop", "attn_pdrop"):
        if float(model.override_config.get(key, 0)) != 0:
            raise ValueError("native dropout must be zero; use only the explicit channel masks")
