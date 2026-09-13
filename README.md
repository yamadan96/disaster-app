# Disaster Building Damage Assessment

[![CI](https://github.com/yamadan96/disaster-app/actions/workflows/ci.yml/badge.svg)](https://github.com/yamadan96/disaster-app/actions/workflows/ci.yml)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Space-demo-yellow)](https://huggingface.co/spaces/yuto090612/disaster-app)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

WebApp and API that classify **earthquake and tsunami building damage** from a single image.

The model is a **DINOv2 ViT-L/14 backbone fine-tuned with LoRA** from a research project, deployed with
**selective classification**: low-confidence predictions are rejected instead of being reported as a class.

- **Demo:** [Hugging Face Space](https://huggingface.co/spaces/yuto090612/disaster-app) (CPU; the first request after the Space wakes up can be slow)
- **Weights:** [yuto090612/disaster-app-model](https://huggingface.co/yuto090612/disaster-app-model) (`best_model.pth`)

## Pipeline

```
Building photo
      │  Resize(570) → CenterCrop(518) → ImageNet normalization
      ▼
DINOv2 ViT-L/14 (timm)  ── LoRA adapters on attention qkv (r=16, alpha=16)
      │
      ▼
Feature transform (Linear → ReLU → Dropout)
      │
      ▼
6-class head ──► softmax ──► max probability < 0.5 ? ──► rejected ("判定不能")
                                     │ no
                                     ▼
                              predicted damage class
```

## Damage Classes

| ID | Class | Meaning |
|---|---|---|
| 0 | 被害なし | No damage |
| 1 | E1(地震大) | Earthquake — large damage |
| 2 | E2(地震中) | Earthquake — medium damage |
| 3 | E3(地震小) | Earthquake — small damage |
| 4 | T1(津波大) | Tsunami — large damage |
| 5 | T3(津波小) | Tsunami — small damage |

## Model

| Item | Value (`src/config.py`, `src/model.py`, `src/predictor.py`) |
|---|---|
| Backbone | `vit_large_patch14_dinov2` (timm), base weights frozen by PEFT |
| Adapter | LoRA, rank 16, alpha 16, dropout 0.1, target `qkv` |
| Heads | 6-class head used for prediction; auxiliary heads in the checkpoint: damage (2), disaster type (2), severity (3) |
| Input | 518×518 RGB |
| Rejection | `REJECTION_THRESHOLD = 0.5` on the max softmax probability |

This repository is **inference-only**. Training code and evaluation results are not included.

## Quick Start

```bash
git clone https://github.com/yamadan96/disaster-app
cd disaster-app
uv sync   # Linux/Windows: CUDA 12.1 wheels, macOS: CPU/MPS wheels

# Gradio UI (http://localhost:7860)
# Uses CHECKPOINT_DIR if it exists, otherwise downloads best_model.pth (~1.2 GB) from the Hub
uv run python app.py

# FastAPI (CHECKPOINT_DIR is required)
CHECKPOINT_DIR=/path/to/dir/with/best_model.pth uv run uvicorn api.main:app --port 8000
curl -F "file=@building.jpg" http://localhost:8000/predict
curl http://localhost:8000/health
```

`DEVICE` (`cuda` / `cpu`) overrides device selection; by default CUDA is used when available.

### API

| Endpoint | Description |
|---|---|
| `GET /health` | Returns `{"status": "ok"}` |
| `POST /predict` | Multipart image upload → `class_id`, `class_name`, `confidence`, `probabilities`, `rejected`. Returns 400 for non-image or undecodable files, 503 if the model is not loaded |

## Tests

```bash
uv run pytest
```

`tests/test_api.py` stubs the model (no weight download) and checks the API status codes, the response
schema, and the rejection threshold logic.

## Project Structure

```
disaster-app/
├── src/
│   ├── config.py      # InferenceConfig (frozen dataclass)
│   ├── model.py       # DINOv2MultiHeadModel + build_model / load_checkpoint
│   └── predictor.py   # Singleton predictor with selective classification
├── api/
│   └── main.py        # FastAPI endpoints
├── tests/
│   └── test_api.py
├── app.py             # Gradio WebApp (also the Hugging Face Space entry point)
├── README_spaces.md   # README with Space metadata (used as README.md in the Space repo)
└── requirements.txt   # Dependencies installed by the Space
```

## Hugging Face Space

The Space repository contains `app.py`, `src/`, `api/`, `requirements.txt`, and `README_spaces.md` copied as
`README.md` (its front matter sets the Gradio SDK version). Keep `sdk_version` in `README_spaces.md` within the
`gradio` range in `pyproject.toml`.

## References

- Oquab et al. (2023). [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
- Hu et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

## License

MIT — see [LICENSE](LICENSE).
