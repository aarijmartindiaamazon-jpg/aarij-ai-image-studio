from __future__ import annotations

from threading import RLock
from typing import Any

from .config import MODELS, Settings, settings
from .gpu_utils import cleanup_memory


class ModelManager:
    """Loads exactly one heavyweight pipeline at a time."""

    def __init__(self, app_settings: Settings = settings):
        self.settings = app_settings
        self.pipeline: Any | None = None
        self.kind: str | None = None
        self.task: str | None = None
        self._lock = RLock()

    def unload(self) -> None:
        with self._lock:
            if self.pipeline is not None:
                try:
                    self.pipeline.to("cpu")
                except Exception:
                    pass
            self.pipeline = None
            self.kind = None
            self.task = None
            cleanup_memory()

    def load(self, kind: str, task: str = "text2img"):
        kind = kind.lower()
        if kind not in MODELS:
            raise ValueError(f"Unknown model mode: {kind}")
        if task not in {"text2img", "img2img", "inpaint"}:
            raise ValueError(f"Unknown pipeline task: {task}")
        with self._lock:
            if self.pipeline is not None and self.kind == kind and self.task == task:
                return self.pipeline
            self.unload()
            try:
                import torch
                from diffusers import AutoPipelineForImage2Image, AutoPipelineForInpainting, AutoPipelineForText2Image

                if not torch.cuda.is_available():
                    raise RuntimeError("No CUDA GPU was found. In Colab, choose Runtime → Change runtime type → T4 GPU.")
                pipeline_class = {
                    "text2img": AutoPipelineForText2Image,
                    "img2img": AutoPipelineForImage2Image,
                    "inpaint": AutoPipelineForInpainting,
                }[task]
                pipe = pipeline_class.from_pretrained(
                    MODELS[kind].model_id,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16",
                )
                pipe.enable_attention_slicing()
                if hasattr(pipe, "enable_vae_slicing"):
                    pipe.enable_vae_slicing()
                if hasattr(pipe, "enable_vae_tiling"):
                    pipe.enable_vae_tiling()
                if kind == "quality" and self.settings.enable_cpu_offload:
                    pipe.enable_model_cpu_offload()
                else:
                    pipe.to("cuda")
                self.pipeline, self.kind, self.task = pipe, kind, task
                return pipe
            except Exception:
                self.unload()
                raise

    def status(self) -> str:
        if self.pipeline is None:
            return "No AI model loaded (models load on first use)."
        return f"{MODELS[self.kind].label} loaded for {self.task}."


model_manager = ModelManager()

