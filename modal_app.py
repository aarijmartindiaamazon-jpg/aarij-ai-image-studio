"""Deploy Aarij AI Image Studio as a scale-to-zero Modal web app.

Setup once:
  pip install modal
  modal setup
  modal secret create huggingface-secret HF_TOKEN=hf_your_token

Deploy:
  modal deploy modal_app.py
"""

from __future__ import annotations

import subprocess

import modal


APP_DIR = "/root/aarij-ai-image-studio"
CACHE_DIR = "/cache/huggingface"

app = modal.App("aarij-ai-image-studio")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.8.0",
        "diffusers==0.35.1",
        "transformers==4.55.4",
        "accelerate==1.10.1",
        "safetensors==0.6.2",
        "gradio==5.44.1",
        "Pillow==11.3.0",
        "rembg==2.0.67",
        "onnxruntime==1.22.1",
        "opencv-python-headless==4.12.0.88",
        "huggingface-hub==0.34.4",
        "sentencepiece==0.2.1",
        "protobuf==6.32.0",
        "spaces",
    )
    .env(
        {
            "HF_HOME": CACHE_DIR,
            "HF_HUB_CACHE": f"{CACHE_DIR}/hub",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "GRADIO_SERVER_NAME": "0.0.0.0",
            "PORT": "7860",
        }
    )
    .add_local_dir(
        ".",
        remote_path=APP_DIR,
        ignore=[".git", ".git/**", "__pycache__", "**/__pycache__/**", ".pytest_cache"],
    )
)

model_cache = modal.Volume.from_name("aarij-huggingface-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")


@app.function(
    image=image,
    gpu="L40S",
    timeout=20 * 60,
    startup_timeout=20 * 60,
    scaledown_window=5 * 60,
    max_containers=1,
    volumes={"/cache": model_cache},
    secrets=[hf_secret],
)
# Gradio makes several parallel HTTP requests for one browser session. Keeping
# those requests in a single concurrent container preserves its queue state and
# uploaded /tmp files. GPU generation remains serialized by demo.queue().
@modal.concurrent(max_inputs=100)
@modal.web_server(7860, startup_timeout=20 * 60)
def web():
    """Start Gradio on Modal's externally routed container port."""
    subprocess.Popen(["python", "hf_space_app.py"], cwd=APP_DIR)
