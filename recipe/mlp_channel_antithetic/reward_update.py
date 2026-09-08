"""Reward finite differences and bounded post-optimizer channel updates.

Only down-projection shards are copied. FSDP1 flat-parameter metadata and
FSDP2 DTensor layouts let us update local storage without gathering weights.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard

from .intervention import NEGATIVE_ROUTE, POSITIVE_ROUTE

REWARD_UPDATE_METADATA = "mlp_antithetic_reward_difference"
_WEIGHT_RE = re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)\.mlp\.down_proj\.weight$")
_PREFIX = "mlp_antithetic/reward_update"
_STEP_METRICS = (
    "update_metrics_available", "alignment_available", "main_down_proj_norm",
    "raw_aux_down_proj_norm", "aux_down_proj_norm", "raw_ratio", "actual_ratio",
    "max_layer_ratio", "budget_utilization", "active_layer_fraction",
    "clipped_layer_fraction", "rounding_backtrack_fraction", "aux_main_cosine",
    "opposing_layer_fraction", "aux_parallel_ratio", "aux_orthogonal_ratio",
)


def summarize_step_metrics(main_sq, raw_sq, actual_sq, main_aux_dot, initial_scales, scales, ratio):
    """Summarize globally reduced squared norms/dots in the down-proj subspace.

    Zero denominators use zero placeholders with explicit availability flags.
    Cosines use actual representable updates, including rounding/backtracking.
    """
    main_total, actual_total = main_sq.sum(), actual_sq.sum()
    main_norm, auxiliary_norm = main_total.sqrt(), actual_total.sqrt()
    valid_main = main_total > 0
    valid_alignment = valid_main & (actual_total > 0)
    actual_ratio = torch.where(valid_main, auxiliary_norm / main_norm.clamp_min(1e-150), 0)
    cosine = torch.where(
        valid_alignment,
        main_aux_dot.sum() / (main_norm * auxiliary_norm).clamp_min(1e-300),
        0,
    ).clamp(-1, 1)
    layer_ratios = torch.where(main_sq > 0, (actual_sq / main_sq.clamp_min(1e-300)).sqrt(), 0)
    active_layers = (main_sq > 0) & (actual_sq > 0)
    values = {
        "update_metrics_available": torch.ones_like(main_total),
        "alignment_available": valid_alignment.double(),
        "main_down_proj_norm": main_norm,
        "raw_aux_down_proj_norm": raw_sq.sum().sqrt(),
        "aux_down_proj_norm": auxiliary_norm,
        "raw_ratio": torch.where(valid_main, raw_sq.sum().sqrt() / main_norm.clamp_min(1e-150), 0),
        "actual_ratio": actual_ratio,
        "max_layer_ratio": layer_ratios.max(),
        "budget_utilization": actual_ratio / ratio,
        "active_layer_fraction": (actual_sq > 0).double().mean(),
        "clipped_layer_fraction": (raw_sq > ratio ** 2 * main_sq).double().mean(),
        "rounding_backtrack_fraction": (scales < initial_scales).double().mean(),
        "aux_main_cosine": cosine,
        "opposing_layer_fraction": ((main_aux_dot < 0) & active_layers).double().sum()
        / active_layers.double().sum().clamp_min(1),
        "aux_parallel_ratio": actual_ratio * cosine,
        "aux_orthogonal_ratio": actual_ratio * (1 - cosine.square()).clamp_min(0).sqrt(),
    }
    numbers = torch.stack(list(values.values())).cpu().tolist()
    return {f"{_PREFIX}/{name}": float(value) for name, value in zip(values, numbers)}


@dataclass(frozen=True)
class RewardUpdateConfig:
    enabled: bool = False
    learning_rate: float = 1e-3
    max_update_ratio: float = 0.05

    @classmethod
    def from_config(cls, config):
        if config is None or not config.get("enabled", False):
            return cls()
        rate = float(config.get("learning_rate", 1e-3))
        ratio = float(config.get("max_update_ratio", 0.05))
        if not math.isfinite(rate) or rate < 0:
            raise ValueError("reward_difference_update.learning_rate must be finite and >= 0")
        if not math.isfinite(ratio) or not 0 <= ratio < 1:
            raise ValueError("reward_difference_update.max_update_ratio must be in [0, 1)")
        return cls(True, rate, ratio)

    @property
    def active(self):
        return self.enabled and self.learning_rate > 0 and self.max_update_ratio > 0


def prepare_reward_difference(batch, config: RewardUpdateConfig) -> dict[str, float]:
    """Compute equal-prompt reward means on the driver, BEFORE actor DP dispatch.

    Do not use advantages, token-count weighting, or per-rank group fragments.
    The disabled path neither reads tensors nor allocates metadata.
    """
    if not config.active:
        return {}
    scores = batch.batch["token_level_scores"].detach().double().sum(dim=-1).cpu().numpy()
    uids = np.asarray(batch.non_tensor_batch["uid"], dtype=object)
    routes = np.asarray(batch.non_tensor_batch["route_id"], dtype=object)
    versions = np.unique(batch.non_tensor_batch["perturbation_version"])
    if scores.ndim != 1 or not (len(scores) == len(uids) == len(routes)) or not len(scores):
        raise ValueError("reward difference requires a nonempty aligned rollout batch")
    if len(versions) != 1 or not np.isfinite(scores).all():
        raise ValueError("reward difference requires finite rewards and one perturbation version")
    if set(routes) != {POSITIVE_ROUTE, NEGATIVE_ROUTE}:
        raise ValueError("reward difference requires positive and negative routes only")
    groups = {}
    for i, uid in enumerate(uids):
        groups.setdefault(uid, []).append(i)
    positive, negative = [], []
    split_gaps = [[], []]
    rollout_order = batch.non_tensor_batch.get("antithetic_rollout_order")
    for indices in groups.values():
        idx = np.asarray(indices)
        pos = scores[idx[routes[idx] == POSITIVE_ROUTE]]
        neg = scores[idx[routes[idx] == NEGATIVE_ROUTE]]
        if len(pos) == 0 or len(pos) != len(neg):
            raise ValueError("reward difference requires equal nonzero +/- counts for every prompt")
        positive.append(float(pos.mean()))
        negative.append(float(neg.mean()))
        if rollout_order is not None and len(pos) >= 2:
            # Use the original generation order, not the length-dependent DP
            # balancing order. Never split based on rewards or response length.
            ordered = idx[np.argsort(np.asarray(rollout_order)[idx])]
            pos_ordered = scores[ordered[routes[ordered] == POSITIVE_ROUTE]]
            neg_ordered = scores[ordered[routes[ordered] == NEGATIVE_ROUTE]]
            for half in (0, 1):
                split_gaps[half].append(float(pos_ordered[half::2].mean() - neg_ordered[half::2].mean()))
    mean_pos, mean_neg = float(np.mean(positive)), float(np.mean(negative))
    gap = mean_pos - mean_neg
    prompt_gaps = np.asarray(positive) - np.asarray(negative)
    abs_mean = float(np.abs(prompt_gaps).mean())
    se_available = len(groups) >= 2
    standard_error = float(prompt_gaps.std(ddof=1) / math.sqrt(len(groups))) if se_available else 0.0
    split_available = len(split_gaps[0]) == len(groups)
    split_first, split_second = (float(np.mean(part)) for part in split_gaps) if split_available else (0.0, 0.0)
    batch.meta_info[REWARD_UPDATE_METADATA] = {
        "reward_gap": gap,
        "version": int(versions[0]),
    }
    return {
        f"{_PREFIX}/reward_positive": mean_pos,
        f"{_PREFIX}/reward_negative": mean_neg,
        f"{_PREFIX}/reward_gap": gap,
        f"{_PREFIX}/prompt_count": float(len(groups)),
        f"{_PREFIX}/reward_gap_abs": abs(gap),
        f"{_PREFIX}/prompt_gap_abs_mean": abs_mean,
        f"{_PREFIX}/prompt_gap_nonzero_fraction": float(np.mean(prompt_gaps != 0)),
        f"{_PREFIX}/prompt_gap_cancellation": max(0.0, 1 - abs(gap) / abs_mean) if abs_mean else 0.0,
        f"{_PREFIX}/reward_gap_standard_error": standard_error,
        f"{_PREFIX}/reward_gap_standard_error_available": float(se_available),
        f"{_PREFIX}/split_half_available": float(split_available),
        f"{_PREFIX}/split_half_gap_first": split_first,
        f"{_PREFIX}/split_half_gap_second": split_second,
        f"{_PREFIX}/split_half_both_nonzero": float(split_first != 0 and split_second != 0),
        f"{_PREFIX}/split_half_same_sign": float(split_first * split_second > 0),
    }


def estimate_channel_direction(controller, reward_gap: float, compute_dtype: torch.dtype):
    """K=1 central difference, corrected for balanced even-width directions."""
    width = controller.intermediate_size
    if width < 2 or width % 2:
        raise ValueError("reward difference updates require an even intermediate_size >= 2")
    one = torch.ones((), dtype=compute_dtype)
    delta = float((one + torch.tensor(controller.perturbation_strength, dtype=compute_dtype) - one).item())
    if not math.isfinite(reward_gap) or not 0 < delta < 1:
        raise ValueError("reward difference requires a finite gap and effective strength in (0, 1)")
    slope = reward_gap / (2 * delta)
    direction = controller.direction.double() * (slope * (width - 1) / width)
    if not torch.isfinite(direction).all():
        raise ValueError("nonfinite reward difference direction")
    return direction, delta, slope


def _normalized_name(name):
    return ".".join(part for part in name.split(".") if part != "_fsdp_wrapped_module")


@dataclass
class DownProjectionShard:
    """A storage view resolved again after FSDP reshards/offloads parameters."""

    layer: int
    parameter: torch.Tensor
    global_shape: tuple[int, int]
    flat_slice: tuple[int, int] | None = None
    flat_parameter_start: int = 0
    channel_start: int = 0
    expected_storage_numel: int | None = None

    def tensor(self):
        value = self.parameter.to_local() if isinstance(self.parameter, DTensor) else self.parameter
        if self.expected_storage_numel is not None and value.numel() != self.expected_storage_numel:
            raise RuntimeError("FSDP down-projection storage is not in its expected sharded state")
        if self.flat_slice is not None:
            offset, count = self.flat_slice
            return value.view(-1).narrow(0, offset, count)
        if value.ndim != 2:
            raise RuntimeError("down-projection local tensor must be two-dimensional")
        return value

    def multiply_channels_(self, value, direction):
        """Scale columns, including FSDP1 shards beginning/ending inside a row."""
        width = self.global_shape[1]
        direction = direction.to(device=value.device, dtype=value.dtype)
        if self.flat_slice is None:
            value.mul_(direction[self.channel_start : self.channel_start + value.shape[1]].unsqueeze(0))
            return
        start = self.flat_parameter_start % width
        head = min(value.numel(), width - start) if start else 0
        if head:
            value[:head].mul_(direction[start : start + head])
        rows = (value.numel() - head) // width
        stop = head + rows * width
        if rows:
            value[head:stop].view(rows, width).mul_(direction.unsqueeze(0))
        if stop < value.numel():
            value[stop:].mul_(direction[: value.numel() - stop])


def resolve_down_projection_shards(module, optimizer, controller):
    """Resolve FSDP1 (including use_orig_params), FSDP2, and ordinary tensors."""
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    shards, found, consumed = [], {}, set()
    for prefix, child in module.named_modules():
        handle = getattr(child, "_handle", None)
        flat = getattr(handle, "flat_param", None)
        if flat is None or id(flat) in consumed:
            continue
        consumed.add(id(flat))
        # These metadata fields are required: never guess flat-parameter offsets.
        for name, shape, info in zip(flat._fqns, flat._shapes, flat._shard_param_infos, strict=True):
            match = _WEIGHT_RE.search(_normalized_name(f"{prefix}.{name}"))
            if match is None:
                continue
            layer = int(match.group(1))
            if layer in found:
                raise RuntimeError(f"duplicate down projection for layer {layer}")
            found[layer] = tuple(shape)
            if not flat.requires_grad:
                raise ValueError("reward difference updates require trainable down projections")
            if id(flat) not in optimizer_ids:
                originals = getattr(flat, "_params", None)
                if not originals or any(id(p) not in optimizer_ids for p in originals):
                    raise ValueError("FSDP flat/original parameters are missing from the optimizer")
            if info.in_shard:
                shards.append(DownProjectionShard(
                    layer, flat, tuple(shape),
                    (int(info.offset_in_shard), int(info.numel_in_shard)),
                    int(info.intra_param_start_idx),
                    expected_storage_numel=flat._sharded_size.numel(),
                ))
    if not consumed:
        for name, parameter in module.named_parameters():
            match = _WEIGHT_RE.search(_normalized_name(name))
            if match is None:
                continue
            layer = int(match.group(1))
            if layer in found:
                raise RuntimeError(f"duplicate down projection for layer {layer}")
            if id(parameter) not in optimizer_ids or not parameter.requires_grad:
                raise ValueError("reward difference updates require optimized, trainable down projections")
            found[layer] = tuple(parameter.shape)
            channel_start = 0
            if isinstance(parameter, DTensor):
                placements = parameter.placements
                if any(not isinstance(p, (Shard, Replicate)) for p in placements):
                    raise NotImplementedError("only Shard/Replicate DTensor placements are supported")
                sharded = [(i, p) for i, p in enumerate(placements) if isinstance(p, Shard)]
                if len(sharded) > 1:
                    raise NotImplementedError("at most one DTensor sharding dimension is supported")
                for mesh_dim, placement in sharded:
                    if placement.dim == 1:
                        _, channel_start = Shard.local_shard_size_and_offset(
                            parameter.shape[1], parameter.device_mesh.size(mesh_dim),
                            parameter.device_mesh.get_local_rank(mesh_dim),
                        )
            shards.append(DownProjectionShard(layer, parameter, tuple(parameter.shape), channel_start=channel_start))
    if set(found) != set(range(controller.num_layers)):
        raise ValueError("could not resolve exactly one down projection for every MLP layer")
    if any(len(shape) != 2 or shape[1] != controller.intermediate_size for shape in found.values()):
        raise ValueError("down projection shapes do not match the channel controller")
    if any(shard.parameter.dtype not in (torch.float32, torch.float64) for shard in shards):
        raise ValueError("reward difference updates require FP32/FP64 optimizer master weights")
    return shards, [math.prod(found[layer]) for layer in range(controller.num_layers)]


class RewardDifferenceUpdater:
    """Optional optimizer hooks. Momentum is untouched; only parameters change."""

    def __init__(self, module, optimizer, controller, config, compute_dtype=torch.bfloat16):
        if not config.active:
            raise ValueError("do not install reward update hooks when the feature is inactive")
        self.config, self.controller, self.compute_dtype = config, controller, compute_dtype
        estimate_channel_direction(controller, 0.0, compute_dtype)
        self.shards, self.global_numels = resolve_down_projection_shards(module, optimizer, controller)
        self.reference_parameter = next(module.parameters())
        self.direction = None
        self.snapshots = []
        self.last_metrics = {}
        self.step_count = 0
        self._hooks = [optimizer.register_step_pre_hook(self._before_step), optimizer.register_step_post_hook(self._after_step)]

    def begin_batch(self, metadata, versions):
        if self.direction is not None or self.snapshots:
            raise RuntimeError("reward difference update was not cleaned up")
        self.controller.validate_batch_version(versions)
        if metadata is None or int(metadata["version"]) != self.controller.perturbation_version:
            raise RuntimeError("missing or stale global reward difference metadata")
        self.direction, delta, slope = estimate_channel_direction(
            self.controller, float(metadata["reward_gap"]), self.compute_dtype
        )
        self.step_count = 0
        self.last_metrics = {
            **{f"{_PREFIX}/{name}": 0.0 for name in _STEP_METRICS},
            f"{_PREFIX}/effective_strength": delta,
            f"{_PREFIX}/directional_derivative": slope,
            f"{_PREFIX}/max_update_ratio": self.config.max_update_ratio,
            f"{_PREFIX}/applied": 0.0,
            f"{_PREFIX}/optimizer_step_executed": 0.0,
            f"{_PREFIX}/skipped_zero_gap": float(slope == 0),
            f"{_PREFIX}/time_s": 0.0,
        }

    def end_batch(self):
        self.direction = None
        self.snapshots.clear()

    def close(self):
        self.end_batch()
        for hook in self._hooks:
            hook.remove()

    def _device(self):
        # A rank may own no down-projection elements. It must still participate.
        if dist.is_initialized() and dist.get_backend() == "nccl":
            return torch.device("cuda", torch.cuda.current_device())
        return self.reference_parameter.device

    def _sum(self, stats):
        if dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        return stats

    def _raw_update(self, shard, old):
        # Keep the original snapshot to measure actual PG/auxiliary alignment.
        # Recompute one shard's raw update at a time instead of retaining a
        # second model-sized set of snapshots. Arithmetic matches the updater.
        raw = old.clone()
        shard.multiply_channels_(raw, self.direction[shard.layer])
        return raw.mul_(self.config.learning_rate)

    @torch.no_grad()
    def _before_step(self, optimizer, args, kwargs):
        if self.direction is None:
            raise RuntimeError("optimizer step is missing reward difference batch context")
        self.step_count += 1
        if self.step_count != 1:
            raise RuntimeError("reward difference updates require exactly one optimizer step per rollout batch")
        started = time.perf_counter()
        self._snapshot_time = 0.0
        if not torch.count_nonzero(self.direction):
            return
        self.snapshots = [(shard, shard.tensor().detach().clone()) for shard in self.shards]
        self._snapshot_time = time.perf_counter() - started

    @torch.no_grad()
    def _after_step(self, optimizer, args, kwargs):
        started = time.perf_counter()
        self.last_metrics[f"{_PREFIX}/optimizer_step_executed"] = 1.0
        if not torch.count_nonzero(self.direction):
            self.last_metrics[f"{_PREFIX}/time_s"] = time.perf_counter() - started
            return
        layers = self.controller.num_layers
        stats = torch.zeros((layers, 3), dtype=torch.float64, device=self._device())
        # Retain old weights for measuring the actual PG/auxiliary dot product.
        for shard, old in self.snapshots:
            current = shard.tensor()
            stats[shard.layer, 0] += torch.linalg.vector_norm(current - old, dtype=torch.float64).square().to(stats.device)
            raw = self._raw_update(shard, old)
            stats[shard.layer, 1] += torch.linalg.vector_norm(raw, dtype=torch.float64).square().to(stats.device)
            stats[shard.layer, 2] += old.numel()
            del raw
        self._sum(stats)
        if not torch.isfinite(stats).all() or (stats[:, 2] <= 0).any():
            raise RuntimeError("nonfinite or incomplete distributed reward update statistics")
        # Replicated/SP/hybrid shards repeat both squared norms equally.
        replication = stats[:, 2] / torch.tensor(self.global_numels, device=stats.device)
        main_sq, raw_sq = stats[:, 0] / replication, stats[:, 1] / replication
        budget_sq = self.config.max_update_ratio ** 2 * main_sq
        scales = torch.where(raw_sq > 0, (budget_sq / raw_sq.clamp_min(1e-300)).sqrt().clamp(max=1), 0)
        initial_scales = scales.clone()
        # Bound the ACTUAL representable addition too. Rounding can otherwise
        # exceed rho even with FP32 master weights. Backtrack before any write.
        for attempt in range(9):
            actual_stats = torch.zeros((layers, 2), dtype=torch.float64, device=stats.device)
            scale_values = scales.cpu().tolist()
            for shard, old in self.snapshots:
                if scale_values[shard.layer] == 0:
                    continue
                current = shard.tensor()
                raw = self._raw_update(shard, old)
                auxiliary = (current + raw * scale_values[shard.layer]) - current
                actual_stats[shard.layer, 0] += torch.linalg.vector_norm(auxiliary, dtype=torch.float64).square().to(stats.device)
                actual_stats[shard.layer, 1] += (auxiliary * (current - old)).sum(dtype=torch.float64).to(stats.device)
                del raw, auxiliary
            self._sum(actual_stats)
            actual_sq = actual_stats[:, 0] / replication
            main_aux_dot = actual_stats[:, 1] / replication
            excessive = actual_sq > budget_sq
            if not excessive.any():
                break
            scales[excessive] *= 0.5
        # If a representable step still cannot fit, leave that layer unchanged.
        scales[excessive] = 0
        actual_sq[excessive] = 0
        main_aux_dot[excessive] = 0
        scale_values = scales.cpu().tolist()
        for shard, old in self.snapshots:
            if scale_values[shard.layer] != 0:
                current = shard.tensor()
                raw = self._raw_update(shard, old)
                current.copy_(current + raw * scale_values[shard.layer])
                del raw
        self.snapshots.clear()
        self.last_metrics.update(summarize_step_metrics(
            main_sq, raw_sq, actual_sq, main_aux_dot, initial_scales, scales,
            self.config.max_update_ratio,
        ))
        self.last_metrics.update({
            f"{_PREFIX}/applied": float(self.last_metrics[f"{_PREFIX}/aux_down_proj_norm"] > 0),
            f"{_PREFIX}/time_s": self._snapshot_time + time.perf_counter() - started,
        })
