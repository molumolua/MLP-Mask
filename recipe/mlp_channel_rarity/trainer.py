"""Trainer integration for question-level MLP-channel rarity diagnostics."""

from __future__ import annotations

import json
import os
from pathlib import Path

from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from .diagnostics import build_question_rarity_records


class MLPChannelRarityTrainer(RayPPOTrainer):
    """Ray PPO trainer that adds a compact, question-level rarity dump."""

    def _log_rollout_data(
        self,
        batch: DataProto,
        reward_extra_infos_dict: dict,
        timing_raw: dict,
        rollout_data_dir: str,
    ) -> None:
        super()._log_rollout_data(
            batch,
            reward_extra_infos_dict,
            timing_raw,
            rollout_data_dir,
        )

        rarity_config = OmegaConf.to_container(
            self.config.actor_rollout_ref.mlp_channel_rarity,
            resolve=True,
        )
        records = build_question_rarity_records(
            batch,
            self.tokenizer,
            global_step=self.global_steps,
            reward_extra_infos=reward_extra_infos_dict,
            rarity_config=rarity_config,
        )

        output_dir = Path(rollout_data_dir) / "question_rarity"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{self.global_steps}.jsonl"
        temporary_path = output_dir / f".{self.global_steps}.{os.getpid()}.tmp"
        with temporary_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(temporary_path, output_path)
        print(f"Dumped question rarity diagnostics to {output_path}")
