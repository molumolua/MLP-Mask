"""Entry point for online MLP-channel rarity-weighted GRPO."""

from __future__ import annotations

import os
import socket
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler, run_ppo
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config


@hydra.main(config_path="config", config_name="ppo_mlp_channel_rarity", version_base=None)
def main(config):
    runner = ray.remote(num_cpus=1)(MLPChannelRarityTaskRunner)
    run_ppo(config, task_runner_class=runner)


class MLPChannelRarityTaskRunner:
    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    @staticmethod
    def _validate_recipe_contract(config) -> None:
        rarity = config.actor_rollout_ref.mlp_channel_rarity
        if not bool(rarity.enabled):
            raise ValueError("mlp_channel_rarity.enabled must be true for this recipe")
        if config.actor_rollout_ref.actor.strategy not in {"fsdp", "fsdp2"}:
            raise NotImplementedError("MLP-channel rarity currently supports FSDP/FSDP2 actors only")
        if config.actor_rollout_ref.rollout.mode != "sync":
            raise NotImplementedError("MLP-channel rarity currently requires synchronous rollout")
        if config.algorithm.adv_estimator != AdvantageEstimator.GRPO:
            raise NotImplementedError("MLP-channel rarity currently supports outcome GRPO only")
        if bool(config.algorithm.use_kl_in_reward) or bool(config.actor_rollout_ref.actor.use_kl_loss):
            raise NotImplementedError("reference KL is not supported by this focused recipe")
        if not bool(config.actor_rollout_ref.rollout.calculate_log_probs):
            raise ValueError("rollout.calculate_log_probs must be true")
        if bool(config.algorithm.rollout_correction.bypass_old_logprob_for_rollout):
            raise ValueError(
                "bypass_old_logprob_for_rollout must be false because rarity is collected "
                "during old-log-prob recomputation"
            )
        if bool(config.actor_rollout_ref.rollout.log_prob_use_dynamic_bsz):
            raise ValueError(
                "rollout.log_prob_use_dynamic_bsz must be false so collected rarity rows "
                "preserve DataProto order"
            )
        if int(config.actor_rollout_ref.actor.ulysses_sequence_parallel_size) != 1:
            raise NotImplementedError("MLP-channel rarity currently requires Ulysses SP size 1")
        if not bool(config.actor_rollout_ref.actor.get("force_on_policy", False)):
            raise ValueError("actor.force_on_policy must be true")
        if int(config.actor_rollout_ref.actor.ppo_epochs) != 1:
            raise ValueError("actor.ppo_epochs must be 1")
        if not bool(config.actor_rollout_ref.actor.get("use_rollout_log_probs", False)):
            raise ValueError("actor.use_rollout_log_probs must be true")
        if config.reward_model.launch_reward_fn_async:
            raise NotImplementedError("async reward functions are not supported by this recipe")

    def run(self, config):
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local
        from verl.workers.fsdp_workers import CriticWorker, RewardModelWorker

        from .worker import MLPChannelRarityActorRolloutRefWorker
        from .trainer import MLPChannelRarityTrainer

        print(f"MLPChannelRarityTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        self._validate_recipe_contract(config)

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(
            MLPChannelRarityActorRolloutRefWorker
        )
        self.role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
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
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
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
        trainer = MLPChannelRarityTrainer(
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
