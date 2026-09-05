"""Entry point for clean GRPO with actor-side hard-channel KL consistency."""

from __future__ import annotations

import socket
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler, run_ppo
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config


class MLPChannelConsistencyTrainer(RayPPOTrainer):
    """Attach pre-RL parameter-update sparsity to every validation event."""

    def _validate(self):
        metrics = super()._validate()
        per_rank_metrics = self.actor_rollout_wg.compute_parameter_update_metrics()
        if not per_rank_metrics:
            raise RuntimeError("parameter-update diagnostics returned no actor metrics")
        # The worker performs an all-reduce, so every actor rank returns the same
        # global result. Keep one copy rather than averaging it a second time.
        metrics.update(per_rank_metrics[0])
        return metrics


@hydra.main(
    config_path="config",
    config_name="ppo_mlp_channel_consistency",
    version_base=None,
)
def main(config):
    runner = ray.remote(num_cpus=1)(MLPChannelConsistencyTaskRunner)
    run_ppo(config, task_runner_class=runner)


class MLPChannelConsistencyTaskRunner:
    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    @staticmethod
    def _validate_recipe_contract(config) -> None:
        component = config.actor_rollout_ref.mlp_channel_consistency
        actor = config.actor_rollout_ref.actor
        if not bool(component.enabled):
            raise ValueError("mlp_channel_consistency.enabled must be true")
        if not bool(component.auxiliary_enabled) and float(component.kl_coef) != 0.0:
            raise ValueError("the no-auxiliary baseline requires kl_coef=0")
        if actor.strategy not in {"fsdp", "fsdp2"}:
            raise NotImplementedError("MLP-channel consistency requires FSDP/FSDP2")
        if config.algorithm.adv_estimator != AdvantageEstimator.GRPO:
            raise NotImplementedError("this focused recipe supports outcome GRPO only")
        if config.actor_rollout_ref.rollout.mode != "sync":
            raise NotImplementedError("this focused recipe requires synchronous rollout")
        if not bool(config.actor_rollout_ref.rollout.calculate_log_probs):
            raise ValueError("rollout.calculate_log_probs must be true")
        if bool(
            config.algorithm.rollout_correction.bypass_old_logprob_for_rollout
        ):
            raise ValueError(
                "bypass_old_logprob_for_rollout must be false so PPO uses a clean "
                "actor-side old-policy forward"
            )
        if bool(config.algorithm.use_kl_in_reward) or bool(actor.use_kl_loss):
            raise NotImplementedError(
                "reference-policy KL is disabled so the logged KL is unambiguously "
                "the clean-to-masked consistency loss"
            )
        if int(actor.ulysses_sequence_parallel_size) != 1:
            raise NotImplementedError(
                "response-distribution KL currently requires Ulysses SP size 1"
            )
        if bool(actor.get("use_fused_kernels", False)):
            raise NotImplementedError(
                "response-distribution KL requires actor.use_fused_kernels=false"
            )
        if not bool(actor.get("force_on_policy", False)):
            raise ValueError("actor.force_on_policy must be true")
        if int(actor.ppo_epochs) != 1:
            raise ValueError("actor.ppo_epochs must be 1")
        if not bool(actor.get("use_rollout_log_probs", False)):
            raise ValueError("actor.use_rollout_log_probs must be true")
        if int(config.actor_rollout_ref.model.get("lora_rank", 0)) > 0:
            raise NotImplementedError(
                "LoRA is unsupported because the intervention targets dense MLP channels"
            )
        if actor.loss_agg_mode != "token-mean":
            raise NotImplementedError(
                "the first implementation fixes both GRPO and consistency aggregation "
                "to token-mean"
            )
        if not 0.0 < float(component.mask_ratio) < 1.0:
            raise ValueError("mask_ratio must be in (0, 1)")
        if float(component.kl_coef) < 0.0:
            raise ValueError("kl_coef must be non-negative")
        if int(component.micro_batch_size_per_gpu) <= 0:
            raise ValueError("micro_batch_size_per_gpu must be positive")
        if int(component.kl_top_k) < 0:
            raise ValueError("kl_top_k must be zero (full KL) or positive")
        if int(component.gradient_sample_size_per_rank) <= 0:
            raise ValueError("gradient_sample_size_per_rank must be positive")
        if float(component.parameter_update_atol) != 1.0e-5:
            raise ValueError(
                "parameter_update_atol must remain 1e-5 to match the published "
                "BF16 update-sparsity protocol"
            )

    def run(self, config):
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local
        from verl.workers.fsdp_workers import CriticWorker, RewardModelWorker

        from .worker import MLPChannelConsistencyActorRolloutRefWorker

        print(
            "MLPChannelConsistencyTaskRunner hostname: "
            f"{socket.gethostname()}"
        )
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        self._validate_recipe_contract(config)

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(
            MLPChannelConsistencyActorRolloutRefWorker
        )
        self.role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node]
            * config.trainer.nnodes,
        }
        self.mapping[Role.ActorRollout] = global_pool_id
        self.mapping[Role.Critic] = global_pool_id
        if config.reward_model.enable:
            self.role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            self.mapping[Role.RewardModel] = global_pool_id

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            use_critic=need_critic(config),
        )
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(
            local_path,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
        reward_fn = load_reward_manager(
            config,
            tokenizer,
            num_examine=0,
            **config.reward_model.get("reward_kwargs", {}),
        )
        val_reward_fn = load_reward_manager(
            config,
            tokenizer,
            num_examine=1,
            **config.reward_model.get("reward_kwargs", {}),
        )
        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=self.mapping,
        )
        trainer = MLPChannelConsistencyTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=RayWorkerGroup,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
