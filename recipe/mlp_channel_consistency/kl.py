"""Pure tensor helpers for response-token categorical consistency KL."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TeacherDistribution:
    token_count: int
    row_token_counts: tuple[int, ...]
    top_ids: torch.Tensor | None = None
    top_log_probs: torch.Tensor | None = None
    tail_log_probs: torch.Tensor | None = None
    full_log_probs: torch.Tensor | None = None


def select_response_logits(
    logits: torch.Tensor, response_token_mask: torch.Tensor
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_mask = response_token_mask.reshape(-1).to(
        device=logits.device, dtype=torch.bool
    )
    if flat_mask.numel() != flat_logits.shape[0]:
        raise RuntimeError(
            f"response mask has {flat_mask.numel()} positions but logits have "
            f"{flat_logits.shape[0]}"
        )
    return flat_logits[flat_mask]


def tail_log_probability(top_log_probs: torch.Tensor) -> torch.Tensor:
    top_mass = top_log_probs.exp().sum(dim=-1)
    top_mass = top_mass.clamp(min=0.0, max=1.0 - 1e-7)
    return torch.log1p(-top_mass)


def build_teacher_distribution(
    logits: torch.Tensor,
    response_token_mask: torch.Tensor,
    *,
    top_k: int,
    row_token_counts: tuple[int, ...] | None = None,
) -> TeacherDistribution:
    selected = select_response_logits(logits, response_token_mask)
    token_count = int(selected.shape[0])
    if token_count <= 0:
        raise RuntimeError("consistency micro-batch contains no valid response tokens")
    if row_token_counts is None:
        row_token_counts = tuple(
            int(value)
            for value in response_token_mask.reshape(
                response_token_mask.shape[0], -1
            )
            .sum(dim=-1)
            .detach()
            .cpu()
            .tolist()
        )
    if not row_token_counts or any(count < 0 for count in row_token_counts):
        raise RuntimeError("teacher row token counts must be non-negative")
    if sum(row_token_counts) != token_count:
        raise RuntimeError(
            f"teacher row token counts sum to {sum(row_token_counts)}, "
            f"but selected {token_count} response tokens"
        )

    selected_fp32 = selected.to(dtype=torch.float32)
    if top_k == 0:
        return TeacherDistribution(
            token_count=token_count,
            row_token_counts=row_token_counts,
            full_log_probs=torch.log_softmax(selected_fp32, dim=-1).detach(),
        )

    vocab_size = int(selected_fp32.shape[-1])
    effective_top_k = min(top_k, vocab_size - 1)
    if effective_top_k <= 0:
        raise RuntimeError(f"invalid vocabulary size for top-k KL: {vocab_size}")
    top_logits, top_ids = torch.topk(selected_fp32, k=effective_top_k, dim=-1)
    log_normalizer = torch.logsumexp(selected_fp32, dim=-1, keepdim=True)
    top_log_probs = top_logits - log_normalizer
    return TeacherDistribution(
        token_count=token_count,
        row_token_counts=row_token_counts,
        top_ids=top_ids.detach(),
        top_log_probs=top_log_probs.detach(),
        tail_log_probs=tail_log_probability(top_log_probs).detach(),
    )


def slice_teacher_rows(
    teacher: TeacherDistribution, start: int, end: int
) -> TeacherDistribution:
    """Select contiguous examples from the flattened clean distribution."""
    row_count = len(teacher.row_token_counts)
    if not 0 <= start < end <= row_count:
        raise IndexError(f"invalid teacher row slice [{start}:{end}] for {row_count} rows")
    token_start = sum(teacher.row_token_counts[:start])
    token_end = token_start + sum(teacher.row_token_counts[start:end])
    if token_end <= token_start:
        raise RuntimeError("consistency sub-batch contains no valid response tokens")

    kwargs = {
        "token_count": token_end - token_start,
        "row_token_counts": teacher.row_token_counts[start:end],
    }
    for name in ("top_ids", "top_log_probs", "tail_log_probs", "full_log_probs"):
        value = getattr(teacher, name)
        if value is not None:
            kwargs[name] = value[token_start:token_end]
    return TeacherDistribution(**kwargs)


def forward_kl_sum(
    teacher: TeacherDistribution,
    student_logits: torch.Tensor,
    response_token_mask: torch.Tensor,
) -> torch.Tensor:
    selected = select_response_logits(student_logits, response_token_mask).to(
        dtype=torch.float32
    )
    if int(selected.shape[0]) != teacher.token_count:
        raise RuntimeError(
            f"clean/masked response token counts differ: "
            f"{teacher.token_count} vs {selected.shape[0]}"
        )

    if teacher.full_log_probs is not None:
        student_log_probs = torch.log_softmax(selected, dim=-1)
        teacher_probs = teacher.full_log_probs.exp()
        token_kl = (
            teacher_probs * (teacher.full_log_probs - student_log_probs)
        ).sum(dim=-1)
    else:
        assert teacher.top_ids is not None
        assert teacher.top_log_probs is not None
        assert teacher.tail_log_probs is not None
        log_normalizer = torch.logsumexp(selected, dim=-1, keepdim=True)
        student_top_log_probs = selected.gather(-1, teacher.top_ids) - log_normalizer
        student_tail_log_probs = tail_log_probability(student_top_log_probs)
        token_kl = (
            teacher.top_log_probs.exp()
            * (teacher.top_log_probs - student_top_log_probs)
        ).sum(dim=-1)
        token_kl = token_kl + teacher.tail_log_probs.exp() * (
            teacher.tail_log_probs - student_tail_log_probs
        )

    return token_kl.clamp_min(0.0).sum()
