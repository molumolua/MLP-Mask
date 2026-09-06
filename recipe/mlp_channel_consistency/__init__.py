"""Clean-policy GRPO with hard MLP-channel consistency regularization."""

from .intervention import (
    GRADIENT_ACTIVATION_SCORE,
    HARD_TOP_SELECTION,
    MLPChannelConsistencyController,
    NO_SCORE,
    OUTPUT_CONTRIBUTION_SCORE,
    RANDOM_SELECTION,
    RELATIVE_ACTIVATION_SCORE,
    SOFT_TOP_SELECTION,
    UPDATED_FRACTION_SCORE,
)

__all__ = [
    "GRADIENT_ACTIVATION_SCORE",
    "HARD_TOP_SELECTION",
    "MLPChannelConsistencyController",
    "NO_SCORE",
    "OUTPUT_CONTRIBUTION_SCORE",
    "RANDOM_SELECTION",
    "RELATIVE_ACTIVATION_SCORE",
    "SOFT_TOP_SELECTION",
    "UPDATED_FRACTION_SCORE",
]
