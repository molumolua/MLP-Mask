"""Full or top-k-plus-tail symmetric KL with bounded temporary tensors.

The target is held constant for one branch backward. Replaying the other branch
and differentiating this same symmetric objective supplies the other partial
derivative, exactly recovering the coupled R-Drop gradient on shared weights.
"""

from dataclasses import dataclass

import torch
from torch.autograd.function import once_differentiable


def select_response_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logits.numel() // logits.shape[-1] != mask.numel():
        raise ValueError(f"response logit/mask shapes differ: {logits.shape}, {mask.shape}")
    return logits.reshape(-1, logits.shape[-1])[mask.reshape(-1).to(logits.device).bool()]


@torch.no_grad()
def detached_log_probs(logits: torch.Tensor, *, chunk_size: int = 128) -> torch.Tensor:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    result = torch.empty_like(logits, dtype=dtype)
    for start in range(0, logits.shape[0], chunk_size):
        result[start:start + chunk_size] = logits[start:start + chunk_size].to(dtype).log_softmax(-1)
    return result


@dataclass
class Distribution:
    log_probs: torch.Tensor
    token_ids: torch.Tensor | None = None


def _coarsen(logits, token_ids):
    """Top token log-probs and an exact logsumexp tail, with stable derivatives."""
    top = logits.gather(-1, token_ids)
    tail_logits = logits.scatter(-1, token_ids, -torch.inf)
    tail = tail_logits.logsumexp(-1, keepdim=True)
    categories = torch.cat((top, tail), -1)
    return categories.log_softmax(-1), tail_logits, tail


@torch.no_grad()
def build_distribution(logits: torch.Tensor, *, top_k: int = 64,
                       token_ids: torch.Tensor | None = None, chunk_size: int = 128) -> Distribution:
    """Choose A's top-k once, or evaluate B on exactly the same partition."""
    if top_k < 0 or chunk_size <= 0:
        raise ValueError("top_k must be non-negative and chunk_size positive")
    if token_ids is None and 0 < top_k < logits.shape[-1]:
        token_ids = logits.topk(top_k, dim=-1).indices
    if token_ids is None:
        return Distribution(detached_log_probs(logits, chunk_size=chunk_size))
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    values = torch.empty((logits.shape[0], token_ids.shape[-1] + 1), device=logits.device, dtype=dtype)
    for start in range(0, logits.shape[0], chunk_size):
        end = start + chunk_size
        values[start:end] = _coarsen(logits[start:end].to(dtype), token_ids[start:end])[0]
    return Distribution(values, token_ids)


class _SymmetricKL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target_log_probs, chunk_size, token_ids):
        if logits.ndim != 2:
            raise ValueError("symmetric KL requires a two-dimensional token-by-vocabulary tensor")
        expected_shape = (logits.shape[0], logits.shape[1] if token_ids is None else token_ids.shape[1] + 1)
        if target_log_probs.shape != expected_shape:
            raise ValueError("symmetric KL requires matching token counts and categorical distributions")
        if target_log_probs.requires_grad:
            raise ValueError("this partial backward requires a detached target")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
        value = torch.zeros((), device=logits.device, dtype=dtype)
        derivative = torch.empty_like(logits)
        for start in range(0, logits.shape[0], chunk_size):
            end = start + chunk_size
            current = logits[start:end].to(dtype)
            if token_ids is None:
                logp = current.log_softmax(-1)
            else:
                logp, tail_logits, tail = _coarsen(current, token_ids[start:end])
            logq = target_log_probs[start:end].to(dtype)
            p, q = logp.exp(), logq.exp()
            delta = logp - logq
            value += (0.5 * (p - q) * delta).sum()
            # d [0.5 KL(p||q) + 0.5 KL(q||p)] / d logits(p).
            dlogp = 0.5 * (p * (delta + 1) - q)
            dcategories = dlogp - p * dlogp.sum(-1, keepdim=True)
            if token_ids is None:
                derivative[start:end] = dcategories
            else:
                # Differentiate the tail logsumexp back to every tail token.
                dlogits = (tail_logits - tail).exp() * dcategories[:, -1:]
                dlogits.scatter_(-1, token_ids[start:end], dcategories[:, :-1])
                derivative[start:end] = dlogits
        ctx.save_for_backward(derivative)
        return value

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (derivative,) = ctx.saved_tensors
        return derivative * grad_output, None, None, None


def symmetric_kl_sum(logits: torch.Tensor, target_log_probs: torch.Tensor,
                     *, chunk_size: int = 128, token_ids: torch.Tensor | None = None) -> torch.Tensor:
    """Exact symmetric KL on the supplied categories; logits gradients only.

    token_ids=None uses the full vocabulary; otherwise use those tokens + tail.
    Chunking changes memory scheduling, not the categorical objective.
    Empty response selections return a differentiable zero.
    """
    return _SymmetricKL.apply(logits, target_log_probs, chunk_size, token_ids)
