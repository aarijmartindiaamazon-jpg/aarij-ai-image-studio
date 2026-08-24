from __future__ import annotations

import random

from PIL import Image

from .config import MODELS
from .gpu_utils import cleanup_memory
from .model_manager import ModelManager, model_manager


def _generator(seed: int):
    import torch
    return torch.Generator(device="cuda").manual_seed(seed)


def resolve_seed(seed: int | float | None, randomize: bool) -> int:
    if randomize or seed is None or int(seed) < 0:
        return random.SystemRandom().randint(0, 2**31 - 1)
    return int(seed)


def generate_image(
    prompt: str,
    negative_prompt: str = "",
    mode: str = "fast",
    width: int = 512,
    height: int = 512,
    steps: int | None = None,
    guidance_scale: float | None = None,
    seed: int | None = None,
    randomize_seed: bool = True,
    manager: ModelManager = model_manager,
) -> tuple[Image.Image, int]:
    if not prompt or not prompt.strip():
        raise ValueError("Please enter a description for the image.")
    spec = MODELS[mode]
    used_seed = resolve_seed(seed, randomize_seed)
    width, height = max(256, int(width) // 8 * 8), max(256, int(height) // 8 * 8)
    pipe = manager.load(mode, "text2img")
    args = dict(
        prompt=prompt.strip(), width=width, height=height,
        num_inference_steps=int(steps or spec.default_steps),
        guidance_scale=spec.default_guidance if guidance_scale is None else float(guidance_scale),
        generator=_generator(used_seed),
    )
    if spec.supports_negative_prompt and negative_prompt:
        args["negative_prompt"] = negative_prompt.strip()
    try:
        import torch
        with torch.inference_mode():
            return pipe(**args).images[0], used_seed
    finally:
        cleanup_memory()

