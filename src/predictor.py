"""High-level predictor for disaster building damage assessment."""

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .config import InferenceConfig
from .model import build_model, load_checkpoint

logger = logging.getLogger(__name__)

CLASS_NAMES: list[str] = [
    "被害なし",
    "E1(地震大)",
    "E2(地震中)",
    "E3(地震小)",
    "T1(津波大)",
    "T3(津波小)",
]

REJECTION_THRESHOLD = 0.5


@dataclass(frozen=True)
class PredictionResult:
    """Immutable container for a single prediction."""

    class_id: int
    class_name: str
    confidence: float
    probabilities: list[float]
    rejected: bool


class Predictor:
    """Singleton predictor -- load once, infer many times.

    Usage::

        predictor = Predictor()
        predictor.initialize(checkpoint_dir=Path("..."), device="cuda")
        result = predictor.predict(image)
    """

    _instance: "Predictor | None" = None

    def __new__(cls) -> "Predictor":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def initialize(self, checkpoint_dir: Path, device: str = "cuda") -> None:
        """Load model and prepare transforms.

        Parameters
        ----------
        checkpoint_dir : Path
            Directory containing ``best_model.pth``.
        device : str
            Target device (``"cuda"`` or ``"cpu"``).
        """
        if self._initialized:
            logger.info("Predictor already initialized, skipping.")
            return

        config = InferenceConfig()
        model = build_model(config, device)
        checkpoint_path = checkpoint_dir / "best_model.pth"
        self.model = load_checkpoint(model, checkpoint_path, device)
        self.device = device
        self.transform = transforms.Compose(
            [
                transforms.Resize(570),
                transforms.CenterCrop(518),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self._initialized = True
        logger.info("Predictor initialized on %s", device)

    def predict(self, image: Image.Image) -> PredictionResult:
        """Run inference on a single PIL image.

        Parameters
        ----------
        image : Image.Image
            Input image (any mode -- will be converted to RGB).

        Returns
        -------
        PredictionResult
            Prediction with class, confidence, and rejection flag.

        Raises
        ------
        RuntimeError
            If :meth:`initialize` has not been called.
        """
        if not self._initialized:
            raise RuntimeError("Predictor not initialized. Call initialize() first.")

        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)

        probs = F.softmax(outputs["full"], dim=-1).squeeze(0).cpu().tolist()
        class_id = int(torch.argmax(torch.tensor(probs)).item())
        confidence = probs[class_id]
        rejected = confidence < REJECTION_THRESHOLD

        return PredictionResult(
            class_id=class_id,
            class_name=CLASS_NAMES[class_id],
            confidence=confidence,
            probabilities=probs,
            rejected=rejected,
        )
