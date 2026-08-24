from __future__ import annotations

from PIL import Image

from .image_utils import load_image


def remove_background(image: Image.Image) -> Image.Image:
    try:
        from rembg import remove
        return remove(load_image(image).convert("RGBA"))
    except ImportError as exc:
        raise RuntimeError("Background removal is not installed. Run: pip install rembg onnxruntime") from exc
    except Exception as exc:
        raise RuntimeError(f"Background removal failed: {exc}") from exc

