"""DINOv2 multi-head model for disaster building damage assessment.

Ported from hisaichi research code — simplified to inference-only,
original_only ROI mode (no two_stream, no CoVT, no cascade).
"""

import logging
from pathlib import Path

import timm
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from .config import InferenceConfig

logger = logging.getLogger(__name__)


class DINOv2MultiHeadModel(nn.Module):
    """Multi-head model with DINOv2 backbone and optional auxiliary heads.

    Architecture:
        backbone (DINOv2 ViT-L/14) -> feature_transform -> head_full (6-class)
        Optional auxiliary heads: head_damage (2), head_disaster_type (2), head_severity (3)
    """

    def __init__(self, config: InferenceConfig) -> None:
        super().__init__()
        self.config = config

        # Backbone
        self.backbone = timm.create_model(
            config.model_name,
            pretrained=True,
            num_classes=0,
            img_size=config.image_size,
        )

        # Feature transform
        self.feature_transform = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Main classification head
        self.head_full = nn.Linear(config.hidden_dim, config.num_classes)

        # Auxiliary heads (optional)
        if config.use_auxiliary_heads and not config.ce_only:
            self.head_damage = nn.Linear(config.hidden_dim, 2)
            self.head_disaster_type = nn.Linear(config.hidden_dim, 2)
            self.head_severity = nn.Linear(config.hidden_dim, 3)
        else:
            self.head_damage = None
            self.head_disaster_type = None
            self.head_severity = None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through backbone and classification heads.

        Parameters
        ----------
        x : torch.Tensor
            Input images of shape ``(B, 3, H, W)``.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with key ``"full"`` (always present) and optionally
            ``"damage"``, ``"disaster_type"``, ``"severity"``.
        """
        feat = self.backbone(x)
        transformed = self.feature_transform(feat)

        outputs: dict[str, torch.Tensor] = {
            "full": self.head_full(transformed),
        }

        if self.head_damage is not None:
            outputs["damage"] = self.head_damage(transformed)
            outputs["disaster_type"] = self.head_disaster_type(transformed)
            outputs["severity"] = self.head_severity(transformed)

        return outputs


def build_model(config: InferenceConfig, device: str) -> nn.Module:
    """Create DINOv2 model with LoRA adapters applied.

    Parameters
    ----------
    config : InferenceConfig
        Model configuration.
    device : str
        Target device (e.g. ``"cuda"`` or ``"cpu"``).

    Returns
    -------
    nn.Module
        Model with LoRA adapters, moved to ``device``.
    """
    model = DINOv2MultiHeadModel(config).to(device)

    modules_to_save = [
        "feature_transform",
        "head_full",
        "head_damage",
        "head_disaster_type",
        "head_severity",
    ]

    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=["qkv"],
        lora_dropout=config.lora_dropout,
        bias="none",
        modules_to_save=modules_to_save,
    )
    model = get_peft_model(model, lora_config)

    logger.info(
        "Built DINOv2 model with LoRA (r=%d, alpha=%d) on %s",
        config.lora_rank,
        config.lora_alpha,
        device,
    )
    return model


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: str) -> nn.Module:
    """Load trained weights from a checkpoint file.

    Parameters
    ----------
    model : nn.Module
        Model with LoRA adapters (from :func:`build_model`).
    checkpoint_path : Path
        Path to ``best_model.pth``.
    device : str
        Target device for weight mapping.

    Returns
    -------
    nn.Module
        Model in eval mode with loaded weights.

    Raises
    ------
    FileNotFoundError
        If ``checkpoint_path`` does not exist.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    logger.info("Loaded checkpoint from %s", checkpoint_path)
    return model
