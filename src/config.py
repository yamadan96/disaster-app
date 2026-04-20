"""Inference configuration for disaster building damage assessment model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceConfig:
    """Minimal configuration for inference-only usage.

    Defaults match the baseline_s42_aux checkpoint.
    """

    # Model architecture
    model_name: str = "vit_large_patch14_dinov2"
    image_size: int = 518
    num_classes: int = 6
    hidden_dim: int = 1024

    # LoRA parameters
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.1

    # Head configuration
    use_auxiliary_heads: bool = True
    ce_only: bool = False
