"""Dependency-light helpers for validation identity and pass@k aggregation."""

from __future__ import annotations

import hashlib
import math
import unicodedata
from collections import defaultdict
from typing import Any, Sequence

import numpy as np


def validation_prompt_uid(prompt_text: str) -> str:
    """Build a stable validation identity from prompt text instead of row metadata.

    Unicode normalization and whitespace collapsing make duplicate detection robust
    to harmless serialization differences while preserving the actual question text.
    Data-source separation is handled by the downstream metric grouping.
    """
    normalized = unicodedata.normalize("NFKC", str(prompt_text))
    normalized = " ".join(normalized.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def compute_pass_at_k_metrics(
    data_sources: Sequence[str] | np.ndarray,
    prompt_uids: Sequence[str],
    correctness_values: Sequence[Any],
    *,
    threshold: float = 0.0,
) -> dict[str, dict[str, float]]:
    """Compute unbiased pass@k over duplicate validation rows.

    Rows are grouped within each data source by ``prompt_uids``. For a prompt
    with ``n`` sampled responses and ``c`` correct responses, this uses the
    standard estimator ``1 - C(n-c, k) / C(n, k)``. Metrics are emitted for
    powers of two up to the largest available group size.
    """
    if not (len(data_sources) == len(prompt_uids) == len(correctness_values)):
        raise ValueError(
            "pass@k inputs must have identical lengths: "
            f"data_sources={len(data_sources)}, prompt_uids={len(prompt_uids)}, "
            f"correctness_values={len(correctness_values)}"
        )

    grouped: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for data_source, prompt_uid, value in zip(data_sources, prompt_uids, correctness_values):
        if isinstance(value, (bool, np.bool_)):
            is_correct = bool(value)
        else:
            numeric_value = float(value)
            is_correct = bool(np.isfinite(numeric_value) and numeric_value > threshold)
        grouped[str(data_source)][str(prompt_uid)].append(is_correct)

    output: dict[str, dict[str, float]] = {}
    for data_source, prompt2correct in grouped.items():
        group_sizes = [len(values) for values in prompt2correct.values()]
        source_metrics: dict[str, float] = {
            "unique_prompts": float(len(group_sizes)),
            "samples": float(sum(group_sizes)),
            "samples_per_prompt_min": float(min(group_sizes)),
            "samples_per_prompt_max": float(max(group_sizes)),
        }

        k = 1
        max_group_size = max(group_sizes)
        while k <= max_group_size:
            prompt_estimates = []
            for values in prompt2correct.values():
                n = len(values)
                if n < k:
                    continue
                c = int(sum(values))
                if n - c < k:
                    estimate = 1.0
                else:
                    estimate = 1.0 - math.comb(n - c, k) / math.comb(n, k)
                prompt_estimates.append(estimate)
            if prompt_estimates:
                source_metrics[f"pass@{k}"] = float(np.mean(prompt_estimates))
                source_metrics[f"prompts@{k}"] = float(len(prompt_estimates))
            k *= 2
        output[data_source] = source_metrics

    return output
