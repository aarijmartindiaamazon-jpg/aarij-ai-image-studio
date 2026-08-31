"""Product-specific reference guidance and non-cropping canvas preparation."""

from math import ceil

from PIL import Image, ImageOps


PRODUCT_DETAILS = {
    "Smart door locks": (
        "lock body dimensions and silhouette, handle shape and handedness, keypad digits and layout, "
        "camera and sensor positions, fingerprint reader, screen, keyhole, door hardware, "
        "mounting position, metal finish, logo, model name and included accessories"
    ),
    "Watches": (
        "dial layout, hands, indices, date window, case, crown, bracelet links, "
        "movement details, colors, materials, proportions and engravings"
    ),
    "General product": "shape, proportions, visible features, colors, materials, logo and text",
}


def preservation_prompt(product_type: str) -> str:
    details = PRODUCT_DETAILS.get(product_type, PRODUCT_DETAILS["General product"])
    return (f"Preserve the reference product's visible design: {details}. "
            "Do not add, remove, replace or redesign product features; do not invent technical specifications. ")


def fit_product_reference(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Fit the entire product on the canvas; never crop handles or tall bodies."""
    source = ImageOps.exif_transpose(image).convert("RGB")
    return ImageOps.pad(source, size, method=Image.Resampling.LANCZOS, color=(240, 240, 240))


def reference_steps(requested: int, strength: float) -> int:
    """Keep at least one img2img timestep even with low reference strength."""
    return max(min(4, max(1, int(requested))), ceil(1 / max(.15, min(.95, strength))))
