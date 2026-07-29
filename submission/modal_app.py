"""Persistent, scale-to-zero Modal deployment for the Gradio demo.

Deploy from the repository root with::

    uvx modal setup
    uvx modal deploy submission/modal_app.py

The image contains only the evaluator-facing runtime and the promoted model.
No challenge audio, labels, local virtual environment, or API secrets are
uploaded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import modal


APP_NAME = "phone-accentedness-scorer"
CONTAINER_ROOT = "/app"
LOCAL_ROOT = Path(__file__).resolve().parent

# Install the CPU wheel explicitly. Resolving PyTorch from the default Linux
# index can pull large CUDA runtime packages that this CPU deployment cannot use.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .uv_pip_install(
        "torch==2.12.1",
        index_url="https://download.pytorch.org/whl/cpu",
    )
    .uv_pip_install(
        "gradio==6.20.0",
        "gruut==2.4.0",
        "numpy>=2.0,<3",
        "safetensors>=0.5,<1",
        "scipy>=1.14,<2",
        "soundfile>=0.13,<1",
        "transformers==5.14.1",
    )
    .env(
        {
            "HF_HUB_OFFLINE": "1",
            "PYTHONPATH": CONTAINER_ROOT,
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    .add_local_file(
        LOCAL_ROOT / "demo_app.py",
        f"{CONTAINER_ROOT}/demo_app.py",
        copy=True,
    )
    .add_local_file(
        LOCAL_ROOT / "inference.py",
        f"{CONTAINER_ROOT}/inference.py",
        copy=True,
    )
    .add_local_dir(
        LOCAL_ROOT / "accent_score",
        f"{CONTAINER_ROOT}/accent_score",
        copy=True,
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
    .add_local_dir(
        LOCAL_ROOT / "model",
        f"{CONTAINER_ROOT}/model",
        copy=True,
    )
    .workdir(CONTAINER_ROOT)
)

app = modal.App(APP_NAME)


def create_web_app() -> Any:
    """Mount the existing queued Gradio UI in a small FastAPI application."""

    import gradio as gr
    from fastapi import FastAPI

    from demo_app import DEMO_CSS, MAX_UPLOAD_SIZE, demo

    web_app = FastAPI(title="Phone Accentedness Scorer")

    @web_app.get("/healthz", include_in_schema=False)
    async def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    return gr.mount_gradio_app(
        app=web_app,
        blocks=demo,
        path="/",
        max_file_size=MAX_UPLOAD_SIZE,
        show_error=False,
        css=DEMO_CSS,
    )


@app.function(
    image=image,
    cpu=2.0,
    memory=2048,
    timeout=300,
    min_containers=0,
    max_containers=1,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web() -> Any:
    """Return the public ASGI application for the deployed Modal function."""

    return create_web_app()


__all__ = [
    "APP_NAME",
    "CONTAINER_ROOT",
    "LOCAL_ROOT",
    "app",
    "create_web_app",
    "image",
    "web",
]
