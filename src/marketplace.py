from __future__ import annotations

from PIL import Image

from .background_removal import remove_background
from .image_utils import add_natural_shadow, fit_inside, load_image


def create_white_background(
    source: Image.Image,
    canvas_size: int = 2000,
    product_size_percent: int = 85,
    padding: int = 40,
    shadow: bool = False,
    shadow_strength: float = 0.3,
) -> Image.Image:
    if source is None:
        raise ValueError("Please upload a product image first.")
    canvas_size = max(256, int(canvas_size))
    coverage = min(95, max(20, int(product_size_percent))) / 100
    available = max(1, int(canvas_size * coverage) - (2 * max(0, int(padding))))
    product = remove_background(load_image(source))
    product = fit_inside(product, available, available)
    if shadow:
        product = add_natural_shadow(product, shadow_strength)
    canvas = Image.new("RGBA", (canvas_size, canvas_size), "white")
    position = ((canvas_size - product.width) // 2, (canvas_size - product.height) // 2)
    canvas.alpha_composite(product.convert("RGBA"), position)
    return canvas.convert("RGB")

