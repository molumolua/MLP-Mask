"""Dependency-light route assignment helpers."""

from __future__ import annotations

import numpy as np

from .intervention import NEGATIVE_ROUTE, POSITIVE_ROUTE


def assign_antithetic_routes(uids: np.ndarray) -> np.ndarray:
    """Assign an equal +/- quota inside every original prompt group."""
    values = np.asarray(uids, dtype=object)
    routes = np.empty(len(values), dtype=object)
    grouped: dict[str, list[int]] = {}
    for index, uid in enumerate(values):
        grouped.setdefault(str(uid), []).append(index)
    for uid, indices in grouped.items():
        if len(indices) < 2 or len(indices) % 2:
            raise ValueError(
                f"prompt uid {uid!r} has {len(indices)} rollout rows; "
                "antithetic routing requires a positive even count"
            )
        for local_index, row_index in enumerate(indices):
            routes[row_index] = (
                POSITIVE_ROUTE if local_index % 2 == 0 else NEGATIVE_ROUTE
            )
    return routes
