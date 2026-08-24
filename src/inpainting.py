from __future__ import annotations

from PIL import Image, ImageOps

from .config import MODELS
from .generation import _generator, resolve_seed
from .gpu_utils import cleanup_memory
from .image_utils import load_image
from .model_manager import ModelManager, model_manager


def inpaint_image(source: Image.Image, mask: Image.Image, prompt: str, negative_prompt: str = "",
                  mode: str = "quality", seed: int | None = None, randomize_seed: bool = True,
                  manager: ModelManager = model_manager) -> tuple[Image.Image, int]:
    if source is None or mask is None:
        raise ValueError("Upload both a source image and a mask. White mask areas are changed.")
    if not prompt.strip():
        raise ValueError("Describe what should appear in the masked area.")
    base = load_image(source).convert("RGB")
    mask_image = ImageOps.grayscale(load_image(mask)).resize(base.size)
    used_seed = resolve_seed(seed, randomize_seed)
    pipe = manager.load(mode, "inpaint")
    spec = MODELS[mode]
    args = dict(prompt=prompt, image=base, mask_image=mask_image, generator=_generator(used_seed),
                num_inference_steps=spec.default_steps, guidance_scale=spec.default_guidance)
    if spec.supports_negative_prompt and negative_prompt:
        args["negative_prompt"] = negative_prompt
    try:
        import torch
        with torch.inference_mode():
            return pipe(**args).images[0], used_seed
    finally:
        cleanup_memory()

