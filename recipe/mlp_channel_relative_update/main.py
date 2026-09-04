"""Entry point for GRPO with history-based MLP-channel update allocation."""

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


@hydra.main(config_path="config", config_name="ppo_mlp_channel_relative_update", version_base=None)
def main(config):
    runner = ray.remote(num_cpus=1)(MLPChannelRelativeUpdateTaskRunner)
    run_ppo(config, task_runner_class=runner)


class MLPChannelRelativeUpdateTaskRunner:
    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    @staticmethod
    def _validate_recipe_contract(config) -> None:
        relative_update = config.actor_rollout_ref.mlp_channel_relative_update
        actor = config.actor_rollout_ref.actor
        if actor.strategy != "fsdp2":
            raise NotImplementedError("MLP-channel relative updates require actor.strategy=fsdp2")
        if bool(actor.fsdp_config.offload_policy):
            raise NotImplementedError(
                "FSDP2 offload_policy is not yet supported because channel statistics "
                "must be all-reduced on the accelerator process group"
            )
        if config.algorithm.adv_estimator != AdvantageEstimator.GRPO:
            raise NotImplementedError("this focused recipe supports outcome GRPO only")
        if config.actor_rollout_ref.rollout.mode != "sync":
            raise NotImplementedError("this focused recipe requires synchronous rollout")
        if bool(config.algorithm.use_kl_in_reward) or bool(actor.use_kl_loss):
            raise NotImplementedError("reference KL is not supported by this focused recipe")
        if int(actor.ulysses_sequence_parallel_size) != 1:
            raise NotImplementedError("MLP-channel relative updates currently require Ulysses SP size 1")
        if not bool(actor.get("force_on_policy", False)):
            raise ValueError("actor.force_on_policy must be true")
        if int(actor.ppo_epochs) != 1:
            raise ValueError("actor.ppo_epochs must be 1")
        if not bool(actor.get("use_rollout_log_probs", False)):
            raise ValueError("actor.use_rollout_log_probs must be true")
        if int(config.actor_rollout_ref.model.get("lora_rank", 0)) > 0:
            raise NotImplementedError("LoRA is not supported because this recipe updates dense MLP channels")
        if bool(relative_update.enabled):
            if actor.optim.optimizer_impl != "recipe.mlp_channel_relative_update.optimizer":
                raise ValueError("actor optimizer_impl must select the recipe optimizer module")
            if actor.optim.optimizer != "ChannelRelativeUpdateAdamW":
                raise ValueError("actor optimizer must be ChannelRelativeUpdateAdamW")
        elif actor.optim.optimizer_impl != "torch.optim" or actor.optim.optimizer != "AdamW":
            raise ValueError(
                "the disabled standard control must use torch.optim.AdamW"
            )

    def run(self, config):
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local
        from verl.workers.fsdp_workers import CriticWorker, RewardModelWorker

        from .worker import MLPChannelRelativeUpdateActorRolloutRefWorker

        print(
            "MLPChannelRelativeUpdateTaskRunner hostname: "
            f"{socket.gethostname()}"
        )
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        self._validate_recipe_contract(config)

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(
            MLPChannelRelativeUpdateActorRolloutRefWorker
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
        trainer = RayPPOTrainer(
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
