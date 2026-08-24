from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageFilter, ImageOps

try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9
    RESAMPLE = Image.LANCZOS


def load_image(value: str | Path | Image.Image) -> Image.Image:
    image = value.copy() if isinstance(value, Image.Image) else Image.open(value)
    return ImageOps.exif_transpose(image)


def fit_inside(image: Image.Image, max_width: int, max_height: int, allow_upscale: bool = True) -> Image.Image:
    result = image.copy()
    if not allow_upscale and result.width <= max_width and result.height <= max_height:
        return result
    result.thumbnail((max_width, max_height), RESAMPLE)
    return result


def square_canvas(image: Image.Image, size: int = 2000, padding: int = 0, background=(255, 255, 255, 255)) -> Image.Image:
    available = max(1, size - (padding * 2))
    fitted = fit_inside(image, available, available)
    mode = "RGBA" if fitted.mode == "RGBA" or len(background) == 4 else "RGB"
    canvas = Image.new(mode, (size, size), background)
    x, y = (size - fitted.width) // 2, (size - fitted.height) // 2
    canvas.paste(fitted, (x, y), fitted if fitted.mode == "RGBA" else None)
    return canvas


def add_natural_shadow(product: Image.Image, strength: float = 0.35) -> Image.Image:
    rgba = product.convert("RGBA")
    alpha = rgba.getchannel("A")
    shadow = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    dark = Image.new("RGBA", rgba.size, (0, 0, 0, max(0, min(255, int(150 * strength)))))
    blurred = alpha.filter(ImageFilter.GaussianBlur(max(3, rgba.width // 80)))
    shadow.paste(dark, (max(1, rgba.width // 100), max(2, rgba.height // 80)), blurred)
    return Image.alpha_composite(shadow, rgba)


def export_image(image: Image.Image, path: str | Path, format_name: str = "PNG") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if format_name.upper() == "JPEG":
        image.convert("RGB").save(path, "JPEG", quality=95, subsampling=0)
    else:
        image.save(path, "PNG")
    return path

