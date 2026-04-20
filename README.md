# Disaster Building Damage Assessment App

WebApp for classifying **earthquake and tsunami building damage** from images.

Based on a research project using **DINOv2 + LoRA** fine-tuning with **Selective Classification** for reliable deployment.

## Demo

Upload a building image → get damage class + confidence score. Low-confidence predictions are automatically rejected to avoid misclassification.

```
Input: Building photo
         │
         ▼
   DINOv2 ViT-L/14
   (frozen backbone)
         │
      LoRA (r=16)
         │
         ▼
   6-class classifier
         │
  Selective threshold
   (reject if conf < 0.5)
         │
         ▼
  ┌──────────────────┐
  │ 被害なし  87.3%  │  ← result
  │ E1       5.1%   │
  │ E2       3.8%   │
  │ ...             │
  └──────────────────┘
```

## Damage Classes

| Class | Description |
|---|---|
| 被害なし | No damage |
| E1 | Earthquake — large damage |
| E2 | Earthquake — medium damage |
| E3 | Earthquake — small damage |
| T1 | Tsunami — large damage |
| T3 | Tsunami — small damage |

## Quick Start

```bash
git clone https://github.com/yamadan96/disaster-app
cd disaster-app
uv sync

# Set checkpoint path (DINOv2+LoRA weights)
export CHECKPOINT_DIR=/path/to/checkpoint

# Launch Gradio UI
uv run python app.py

# Or use FastAPI
uv run uvicorn api.main:app --port 8000
# POST /predict  (multipart image upload)
# GET  /health
```

## Project Structure

```
disaster-app/
├── src/
│   ├── config.py     # InferenceConfig (frozen dataclass)
│   ├── model.py      # DINOv2MultiHeadModel + build_model / load_checkpoint
│   └── predictor.py  # Singleton predictor with Selective Classification
├── api/
│   └── main.py       # FastAPI endpoints
└── app.py            # Gradio WebApp
```

## Model

- **Backbone**: DINOv2 ViT-L/14 (frozen)
- **Adapter**: LoRA (rank=16, alpha=16, target=qkv)
- **Input**: 518×518 RGB image
- **Selective Classification**: rejects predictions with max confidence < 0.5

## References

- Oquab et al. (2023). [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
- Hu et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

## License

MIT
