"""AI super-resolution for product-preserving photographic enhancement."""

from __future__ import annotations

import sys
import threading

import numpy as np
import torch
from PIL import Image


MODEL_PATH = "/models/RealESRGAN_x4plus.pth"
_lock = threading.Lock()
_upscaler = None


def _load_upscaler():
    global _upscaler
    if _upscaler is not None:
        return _upscaler
    with _lock:
        if _upscaler is not None:
            return _upscaler

        # BasicSR still imports the pre-0.17 torchvision compatibility module.
        # Modern torchvision exposes the same functions from ``functional``.
        import torchvision.transforms.functional as functional

        sys.modules.setdefault("torchvision.transforms.functional_tensor", functional)
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=4,
        )
        _upscaler = RealESRGANer(
            scale=4, model_path=MODEL_PATH, model=model,
            tile=512, tile_pad=24, pre_pad=0,
            half=torch.cuda.is_available(), gpu_id=0 if torch.cuda.is_available() else None,
        )
    return _upscaler


def ai_super_resolve(image: Image.Image) -> Image.Image:
    """Restore a photograph with Real-ESRGAN x4 and return an RGB image."""
    rgb = np.asarray(image.convert("RGB"))
    bgr = rgb[:, :, ::-1]
    output, _ = _load_upscaler().enhance(bgr, outscale=4)
    return Image.fromarray(output[:, :, ::-1].copy(), "RGB")
