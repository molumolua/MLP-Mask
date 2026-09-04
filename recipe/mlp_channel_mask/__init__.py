"""Structured MLP-channel intervention recipe for GRPO."""

from .intervention import (
    CAUSAL_ABLATION_SCORE,
    CLEAN_ROUTE,
    GRADIENT_ACTIVATION_SCORE,
    GLOBAL_RANDOM_SCOPE,
    MASKED_ROUTE,
    MLPChannelInterventionController,
    OUTPUT_CONTRIBUTION_SCORE,
    PER_LAYER_RANDOM_SCOPE,
    RELATIVE_ACTIVATION_SCORE,
    SOFT_TOP_SELECTION,
    TOP_RELATIVE_ACTIVATION_SELECTION,
    WEIGHTED_RANDOM_SELECTION,
    install_hf_mlp_intervention,
    install_vllm_class_intervention,
    install_vllm_mlp_intervention,
)

__all__ = [
    "CAUSAL_ABLATION_SCORE",
    "CLEAN_ROUTE",
    "GRADIENT_ACTIVATION_SCORE",
    "GLOBAL_RANDOM_SCOPE",
    "MASKED_ROUTE",
    "MLPChannelInterventionController",
    "OUTPUT_CONTRIBUTION_SCORE",
    "PER_LAYER_RANDOM_SCOPE",
    "RELATIVE_ACTIVATION_SCORE",
    "SOFT_TOP_SELECTION",
    "TOP_RELATIVE_ACTIVATION_SELECTION",
    "WEIGHTED_RANDOM_SELECTION",
    "install_hf_mlp_intervention",
    "install_vllm_class_intervention",
    "install_vllm_mlp_intervention",
]
