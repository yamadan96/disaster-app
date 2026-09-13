"""API and predictor tests with the model stubbed out (no weight download)."""

import io
from collections.abc import Iterator

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app
from src.predictor import CLASS_NAMES, REJECTION_THRESHOLD, Predictor

NUM_CLASSES = len(CLASS_NAMES)


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), color=(120, 80, 40)).save(buffer, format="PNG")
    return buffer.getvalue()


def _stub_predictor(logits: list[float]) -> Predictor:
    """Mark the singleton as initialized with a fake model returning ``logits``."""
    predictor = Predictor()
    predictor.model = lambda tensor: {"full": torch.tensor([logits])}
    predictor.device = "cpu"
    predictor.transform = lambda image: torch.zeros(3, 8, 8)
    predictor._initialized = True
    return predictor


@pytest.fixture(autouse=True)
def reset_predictor() -> Iterator[None]:
    Predictor._instance = None
    yield
    Predictor._instance = None


@pytest.fixture
def client() -> TestClient:
    # Not used as a context manager, so the lifespan (checkpoint loading) is skipped
    return TestClient(app)


class TestApi:
    def test_health(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_predict_rejects_non_image_content_type(self, client: TestClient) -> None:
        response = client.post(
            "/predict", files={"file": ("a.txt", b"hello", "text/plain")}
        )
        assert response.status_code == 400

    def test_predict_rejects_undecodable_image(self, client: TestClient) -> None:
        response = client.post(
            "/predict", files={"file": ("a.png", b"not an image", "image/png")}
        )
        assert response.status_code == 400

    def test_predict_returns_503_when_model_not_loaded(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/predict", files={"file": ("a.png", _png_bytes(), "image/png")}
        )
        assert response.status_code == 503

    def test_predict_returns_prediction(self, client: TestClient) -> None:
        _stub_predictor([0.0, 5.0, 0.0, 0.0, 0.0, 0.0])
        response = client.post(
            "/predict", files={"file": ("a.png", _png_bytes(), "image/png")}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["class_id"] == 1
        assert body["class_name"] == CLASS_NAMES[1]
        assert body["rejected"] is False
        assert set(body["probabilities"]) == set(CLASS_NAMES)
        assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)


class TestSelectiveClassification:
    def test_confident_prediction_is_accepted(self) -> None:
        result = _stub_predictor([0.0, 0.0, 6.0, 0.0, 0.0, 0.0]).predict(
            Image.new("RGB", (8, 8))
        )
        assert result.class_id == 2
        assert result.confidence >= REJECTION_THRESHOLD
        assert not result.rejected

    def test_uniform_prediction_is_rejected(self) -> None:
        result = _stub_predictor([0.0] * NUM_CLASSES).predict(Image.new("RGB", (8, 8)))
        assert result.confidence == pytest.approx(1 / NUM_CLASSES)
        assert result.rejected

    def test_uninitialized_predictor_raises(self) -> None:
        with pytest.raises(RuntimeError, match="not initialized"):
            Predictor().predict(Image.new("RGB", (8, 8)))
