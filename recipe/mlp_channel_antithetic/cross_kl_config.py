"""Validate the optional cross-route KL before allocating training models."""

import math


def validate_cross_kl_worker_config(actor_rollout_ref):
    intervention = actor_rollout_ref.mlp_channel_antithetic
    component = intervention.get("cross_route_kl", None)
    if component is None or not component.get("enabled", False):
        return
    if not intervention.enabled:
        raise ValueError("cross-route KL requires antithetic perturbations")
    if intervention.get("reward_difference_update", {}).get("enabled", False):
        raise ValueError("cross_route_kl requires reward_difference_update.enabled=false")
    if not math.isfinite(float(component.kl_coef)) or float(component.kl_coef) < 0:
        raise ValueError("cross_route_kl.kl_coef must be finite and non-negative")
    if int(component.kl_top_k) < 0:
        raise ValueError("cross_route_kl.kl_top_k must be non-negative")
    for field in ("kl_token_chunk_size", "micro_batch_size_per_gpu", "gradient_sample_size_per_rank"):
        if int(component[field]) <= 0:
            raise ValueError(f"cross_route_kl.{field} must be positive")

    actor, model, rollout = actor_rollout_ref.actor, actor_rollout_ref.model, actor_rollout_ref.rollout
    if actor.strategy != "fsdp2" or int(actor.fsdp_config.fsdp_size) != -1:
        raise NotImplementedError("cross-route KL requires fsdp2 with full sharding (fsdp_size=-1)")
    if not actor.get("force_on_policy", False) or int(actor.ppo_epochs) != 1:
        raise ValueError("cross-route KL requires one complete-batch PPO update")
    if actor.policy_loss.loss_mode != "vanilla" or actor.loss_agg_mode != "token-mean":
        raise ValueError("cross-route KL requires vanilla PPO with token-mean aggregation")
    if float(actor.entropy_coeff) != 0 or actor.use_kl_loss:
        raise ValueError("cross-route KL requires entropy and reference-policy KL losses disabled")
    if (int(actor.ulysses_sequence_parallel_size) != 1 or actor.get("use_fused_kernels", False)
            or model.get("use_fused_kernels", False)):
        raise NotImplementedError("cross-route KL requires SP=1 and unfused response logits")
    if int(model.get("lora_rank", 0)) > 0:
        raise NotImplementedError("cross-route KL currently requires full dense weights")
    for field in ("attention_dropout", "embd_pdrop", "resid_pdrop", "attn_pdrop"):
        if float(model.override_config.get(field, 0)) != 0:
            raise ValueError("native dropout must be zero for the PPO teacher cache")
    if rollout.name != "vllm" or rollout.mode != "sync" or rollout.multi_turn.enable:
        raise NotImplementedError("cross-route KL requires synchronous single-turn vLLM")
    if int(rollout.data_parallel_size) != 1 or int(rollout.pipeline_model_parallel_size) != 1:
        raise NotImplementedError("cross-route KL requires rollout DP=1 and PP=1")
    if int(rollout.n) < 2 or int(rollout.n) % 2:
        raise ValueError("cross-route KL requires an even rollout count >= 2")
    if not rollout.calculate_log_probs or not actor.get("use_rollout_log_probs", False):
        raise ValueError("cross-route KL requires route-conditioned behavior log probabilities")
    temperature = float(rollout.temperature)
    if not math.isfinite(temperature) or temperature <= 0 or float(rollout.top_p) != 1 or int(rollout.top_k) != -1:
        raise ValueError("cross-route KL sampling requires temperature>0, top_p=1 and top_k=-1")


def validate_cross_kl_config(config):
    validate_cross_kl_worker_config(config.actor_rollout_ref)
    component = config.actor_rollout_ref.mlp_channel_antithetic.get("cross_route_kl", None)
    if component is None or not component.get("enabled", False):
        return
    if config.algorithm.adv_estimator != "grpo" or config.algorithm.use_kl_in_reward:
        raise ValueError("cross-route KL requires GRPO without reference KL in rewards")
    correction = config.algorithm.rollout_correction
    if correction.bypass_old_logprob_for_rollout or correction.rollout_is or correction.rollout_rs:
        raise ValueError("cross-route KL requires actor-recomputed old log-probs without rollout correction")
    if not config.trainer.balance_batch or config.reward_model.launch_reward_fn_async:
        raise ValueError("cross-route KL requires balanced batches and synchronous rewards")
