"""Batch slicing helpers for the memory-bounded consistency pass."""

from __future__ import annotations

import numpy as np
import torch


def slice_model_inputs(
    model_inputs: dict, start: int, end: int, batch_size: int
) -> dict:
    """Slice per-example inputs while preserving scalar/shared metadata."""
    sliced = {}
    for name, value in model_inputs.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch_size:
            sliced[name] = value[start:end]
        elif isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == batch_size:
            sliced[name] = value[start:end]
        elif isinstance(value, list) and len(value) == batch_size:
            sliced[name] = value[start:end]
        elif isinstance(value, tuple) and len(value) == batch_size:
            sliced[name] = value[start:end]
        else:
            sliced[name] = value
    return sliced
