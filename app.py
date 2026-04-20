"""Gradio web application for disaster building damage assessment.

Launch with:
    CHECKPOINT_DIR=/path/to/checkpoint/dir uv run python app.py
"""

import logging
import os
from pathlib import Path

import gradio as gr
from PIL import Image

from src.predictor import CLASS_NAMES, REJECTION_THRESHOLD, Predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _resolve_checkpoint_dir() -> Path:
    """Resolve checkpoint directory from env var or HF Hub download."""
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR")
    if checkpoint_dir is not None:
        path = Path(checkpoint_dir)
        if path.exists():
            return path

    # Download from HF Hub (for HuggingFace Spaces deployment)
    from huggingface_hub import hf_hub_download
    cache_dir = Path("/tmp/disaster-app-checkpoint")
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "best_model.pth"
    if not model_path.exists():
        logger.info("Downloading model from HuggingFace Hub...")
        hf_hub_download(
            repo_id="yuto090612/disaster-app-model",
            filename="best_model.pth",
            local_dir=str(cache_dir),
        )
    return cache_dir


def _initialize_predictor() -> Predictor:
    """Initialize the singleton predictor."""
    checkpoint_path = _resolve_checkpoint_dir()
    device = os.environ.get("DEVICE", "cuda" if __import__("torch").cuda.is_available() else "cpu")
    predictor = Predictor()
    predictor.initialize(checkpoint_dir=checkpoint_path, device=device)
    return predictor


def predict_image(
    image: Image.Image | None,
) -> tuple[str, dict[str, float]]:
    """Run prediction and format results for Gradio.

    Parameters
    ----------
    image : Image.Image | None
        Input image from Gradio image component.

    Returns
    -------
    tuple[str, dict[str, float]]
        A status message and a dictionary of class probabilities for
        ``gr.Label``.
    """
    if image is None:
        return "画像をアップロードしてください。", {}

    predictor = Predictor()

    try:
        result = predictor.predict(image)
    except RuntimeError:
        logger.exception("Prediction failed")
        return "モデルが初期化されていません。", {}

    # Build label dict for gr.Label
    label_dict = {CLASS_NAMES[i]: prob for i, prob in enumerate(result.probabilities)}

    if result.rejected:
        status = (
            f"⚠ 判定不能（信頼度不足）\n"
            f"最も可能性の高いクラス: {result.class_name}\n"
            f"信頼度: {result.confidence:.1%} "
            f"(閾値: {REJECTION_THRESHOLD:.0%} 未満のため棄却)"
        )
    else:
        status = f"判定結果: {result.class_name}\n信頼度: {result.confidence:.1%}"

    return status, label_dict


def create_app() -> gr.Blocks:
    """Build and return the Gradio Blocks application."""
    with gr.Blocks(
        title="災害建物被害判定システム",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# 災害建物被害判定システム\n"
            "建物画像をアップロードすると被害クラスを判定します "
            "(DINOv2 + LoRA)\n\n"
            "**クラス**: 被害なし / E1(地震大) / E2(地震中) / "
            "E3(地震小) / T1(津波大) / T3(津波小)"
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="建物画像",
                    height=400,
                )
                submit_btn = gr.Button(
                    "判定する",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=1):
                status_output = gr.Textbox(
                    label="判定結果",
                    lines=3,
                    interactive=False,
                )
                label_output = gr.Label(
                    label="クラス別確率",
                    num_top_classes=NUM_DISPLAY_CLASSES,
                )

        submit_btn.click(
            fn=predict_image,
            inputs=[image_input],
            outputs=[status_output, label_output],
        )

        image_input.change(
            fn=predict_image,
            inputs=[image_input],
            outputs=[status_output, label_output],
        )

    return demo


NUM_DISPLAY_CLASSES = 6

# Initialize at module load (required for HF Spaces)
_initialize_predictor()
demo = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
    )
