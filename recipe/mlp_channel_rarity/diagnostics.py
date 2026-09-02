"""Pure helpers for question-level MLP-channel rarity diagnostics."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from verl import DataProto


def _jsonable(value: Any) -> Any:
    """Convert common DataProto payload types to JSON-native values."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return _jsonable(value.item() if value.numel() == 1 else value.tolist())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _finite_float(value: Any) -> float | None:
    """Return a finite scalar float, or ``None`` for missing/non-numeric data."""
    value = _jsonable(value)
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _numeric_summary(values: Sequence[Any]) -> dict[str, Any]:
    converted = [_finite_float(value) for value in values]
    finite = [value for value in converted if value is not None]
    if len(finite) != len(converted) or not finite:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "values": converted,
        }
    array = np.asarray(finite, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "values": converted,
    }


def _group_rows_by_uid(batch: DataProto) -> list[tuple[str, list[int]]]:
    uids = batch.non_tensor_batch.get("uid")
    if uids is None:
        raise RuntimeError("question rarity dump requires batch.non_tensor_batch['uid']")
    if len(uids) != len(batch):
        raise RuntimeError(f"uid count {len(uids)} does not match batch size {len(batch)}")

    groups: OrderedDict[str, list[int]] = OrderedDict()
    for row, uid in enumerate(uids):
        groups.setdefault(str(_jsonable(uid)), []).append(row)
    return list(groups.items())


def _group_values(values: Any, rows: Sequence[int]) -> list[Any] | None:
    if values is None:
        return None
    try:
        if len(values) <= max(rows):
            return None
    except TypeError:
        return None
    return [_jsonable(values[row]) for row in rows]


def _constant_or_values(values: Sequence[Any]) -> Any:
    if not values:
        return None
    first = values[0]
    if all(value == first for value in values[1:]):
        return first
    return list(values)


def build_question_rarity_records(
    batch: DataProto,
    tokenizer: Any,
    *,
    global_step: int,
    reward_extra_infos: Mapping[str, Sequence[Any]] | None = None,
    rarity_config: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Aggregate response rows into one diagnostic record per unique question."""
    required_tensors = ("prompts", "token_level_scores", "rarity_scores", "rarity_loss_weights")
    missing = [key for key in required_tensors if key not in batch.batch]
    if missing:
        raise RuntimeError(f"question rarity dump is missing tensor fields: {missing}")

    reward_extra_infos = reward_extra_infos or {}
    groups = _group_rows_by_uid(batch)
    rewards = batch.batch["token_level_scores"].sum(dim=-1).detach().cpu().tolist()
    raw_scores = batch.batch["rarity_scores"].detach().cpu().tolist()
    loss_weights = batch.batch["rarity_loss_weights"].detach().cpu().tolist()
    rarity_step_metrics = _jsonable(batch.meta_info.get("mlp_channel_rarity_metrics", {}))
    rarity_step = _finite_float(rarity_step_metrics.get("mlp_rarity/step"))

    extra_fields: dict[str, Sequence[Any]] = dict(reward_extra_infos)
    for key, values in batch.non_tensor_batch.items():
        if key not in extra_fields and len(values) == len(batch):
            extra_fields[key] = values

    records: list[dict[str, Any]] = []
    for uid, rows in groups:
        first_row = rows[0]
        prompt = tokenizer.decode(
            batch.batch["prompts"][first_row],
            skip_special_tokens=True,
        )
        raw_summary = _numeric_summary([raw_scores[row] for row in rows])
        weight_summary = _numeric_summary([loss_weights[row] for row in rows])
        reward_summary = _numeric_summary([rewards[row] for row in rows])

        accuracy_values = _group_values(extra_fields.get("acc"), rows)
        accuracy_summary = (
            _numeric_summary(accuracy_values) if accuracy_values is not None else _numeric_summary([])
        )

        reward_model_values = _group_values(batch.non_tensor_batch.get("reward_model"), rows)
        reward_model = reward_model_values[0] if reward_model_values else {}
        ground_truth = reward_model.get("ground_truth") if isinstance(reward_model, Mapping) else None

        rollout_reward_extra: dict[str, Any] = {}
        ignored_extra_fields = {
            "uid",
            "acc",
            "reward_model",
            "raw_prompt",
            "data_source",
            "index",
            "extra_info",
            "multi_modal_data",
            "multi_modal_inputs",
        }
        for key, values in extra_fields.items():
            if key in ignored_extra_fields:
                continue
            grouped = _group_values(values, rows)
            if grouped is not None:
                rollout_reward_extra[key] = _constant_or_values(grouped)

        data_source_values = _group_values(batch.non_tensor_batch.get("data_source"), rows)
        index_values = _group_values(batch.non_tensor_batch.get("index"), rows)
        extra_info_values = _group_values(batch.non_tensor_batch.get("extra_info"), rows)

        records.append(
            {
                "schema_version": 1,
                "step": int(global_step),
                "rarity_step": int(rarity_step) if rarity_step is not None else None,
                "question_uid": uid,
                "dataset_index": _constant_or_values(index_values or []),
                "data_source": _constant_or_values(data_source_values or []),
                "prompt": prompt,
                "ground_truth": _jsonable(ground_truth),
                "n_rollouts": len(rows),
                "average_accuracy": accuracy_summary["mean"],
                "accuracy_std": accuracy_summary["std"],
                "accuracy_min": accuracy_summary["min"],
                "accuracy_max": accuracy_summary["max"],
                "rollout_accuracies": accuracy_summary["values"],
                "reward_mean": reward_summary["mean"],
                "reward_std": reward_summary["std"],
                "reward_min": reward_summary["min"],
                "reward_max": reward_summary["max"],
                "rollout_rewards": reward_summary["values"],
                # Project notation: raw_s_q is the pre-projection rarity score;
                # s_q is the bounded, mean-one actor loss multiplier.
                "raw_s_q": raw_summary["mean"],
                "s_q": weight_summary["mean"],
                "raw_s_q_std": raw_summary["std"],
                "s_q_std": weight_summary["std"],
                "raw_s_q_min": raw_summary["min"],
                "raw_s_q_max": raw_summary["max"],
                "s_q_min": weight_summary["min"],
                "s_q_max": weight_summary["max"],
                "rollout_raw_s_q": raw_summary["values"],
                "rollout_s_q": weight_summary["values"],
                "extra_info": _constant_or_values(extra_info_values or []),
                "reward_extra_info": rollout_reward_extra,
                "rarity_config": _jsonable(rarity_config or {}),
                "rarity_step_metrics": rarity_step_metrics,
            }
        )
    return records
