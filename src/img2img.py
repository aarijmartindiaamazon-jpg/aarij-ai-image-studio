from __future__ import annotations

from PIL import Image

from .config import MODELS
from .generation import _generator, resolve_seed
from .gpu_utils import cleanup_memory
from .image_utils import load_image
from .model_manager import ModelManager, model_manager


def transform_image(source: Image.Image, prompt: str, negative_prompt: str = "", strength: float = 0.35,
                    mode: str = "fast", width: int = 768, height: int = 768, seed: int | None = None,
                    randomize_seed: bool = True, manager: ModelManager = model_manager) -> tuple[Image.Image, int]:
    if source is None:
        raise ValueError("Please upload a source image.")
    if not prompt.strip():
        raise ValueError("Please describe the requested transformation.")
    used_seed = resolve_seed(seed, randomize_seed)
    image = load_image(source).convert("RGB").resize((int(width) // 8 * 8, int(height) // 8 * 8))
    pipe = manager.load(mode, "img2img")
    spec = MODELS[mode]
    args = dict(prompt=prompt, image=image, strength=float(strength), generator=_generator(used_seed),
                num_inference_steps=max(spec.default_steps, 4), guidance_scale=spec.default_guidance)
    if spec.supports_negative_prompt and negative_prompt:
        args["negative_prompt"] = negative_prompt
    try:
        import torch
        with torch.inference_mode():
            return pipe(**args).images[0], used_seed
    finally:
        cleanup_memory()

