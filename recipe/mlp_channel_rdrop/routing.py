"""Dependency-light route assignment helpers."""

from __future__ import annotations

import numpy as np

from .intervention import MASK_B_ROUTE, MASK_A_ROUTE


def copy_prompt_uids_for_generation(
    uids: np.ndarray, *, expected_size: int
) -> np.ndarray:
    """Copy prompt identities into the synchronous generation batch.

    The base PPO trainer keeps ``uid`` with reward-model metadata and therefore
    omits it from synchronous generation batches. Two-mask routing needs the
    original identity after rollout repetition so it can pair rows per prompt.
    """
    values = np.asarray(uids, dtype=object)
    if values.ndim != 1 or len(values) != expected_size:
        raise ValueError(
            "prompt uid must be a one-dimensional array matching the generation "
            f"batch size ({expected_size}); got shape {values.shape}"
        )
    return values.copy()


def assign_rdrop_routes(uids: np.ndarray) -> np.ndarray:
    """Assign an equal A/B quota inside every original prompt group."""
    values = np.asarray(uids, dtype=object)
    routes = np.empty(len(values), dtype=object)
    grouped: dict[str, list[int]] = {}
    for index, uid in enumerate(values):
        grouped.setdefault(str(uid), []).append(index)
    for uid, indices in grouped.items():
        if len(indices) < 2 or len(indices) % 2:
            raise ValueError(
                f"prompt uid {uid!r} has {len(indices)} rollout rows; "
                "rdrop routing requires a positive even count"
            )
        for local_index, row_index in enumerate(indices):
            routes[row_index] = (
                MASK_A_ROUTE if local_index % 2 == 0 else MASK_B_ROUTE
            )
    return routes
