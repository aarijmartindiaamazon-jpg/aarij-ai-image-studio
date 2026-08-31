"""Bounded, sequential smart-lock batch export with model-named downloads."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable
from uuid import uuid4
from zipfile import ZipFile, ZIP_DEFLATED

from PIL import Image, ImageOps

from src.lock_collage import detect_lock_panel_boxes
from src.reference_studio import PanelBox, extract_and_enhance_panels
from src.storage import StorageManager


ENHANCEMENT_MODES = [
    "AI Product Enhance (4x)", "AI Product Enhance + 4K",
    "Pixel-Safe Minimum 1024px", "Pixel-Safe 4K",
]


@dataclass
class LockSheet:
    path: Path
    name: str
    size: tuple[int, int]
    boxes: list[PanelBox]


def inspect_lock_batch(paths: list[str]) -> list[LockSheet]:
    if not paths or len(paths) > 10:
        raise ValueError("Upload between 1 and 10 smart-lock collages per batch.")
    sheets = []
    for path in paths:
        file = Path(path)
        try:
            with Image.open(file) as image:
                if image.width * image.height > 12_000_000:
                    raise ValueError("Please use a collage under 12 megapixels.")
                source = ImageOps.exif_transpose(image).convert("RGB")
                sheets.append(LockSheet(file, file.stem[:100], source.size, detect_lock_panel_boxes(source)))
        except (OSError, ValueError) as exc:
            raise ValueError(f"{file.name}: {exc}") from exc
    return sheets


def export_lock_batch(paths: list[str], storage: StorageManager, quality: str,
                      sharpen: float = .5, ai_upscaler: Callable | None = None,
                      progress: Callable | None = None):
    if quality not in ENHANCEMENT_MODES:
        raise ValueError("Choose a valid enhancement mode.")
    use_ai = quality.startswith("AI")
    if use_ai and ai_upscaler is None:
        raise ValueError("AI restoration is not available. Select Pixel-Safe mode or retry later.")
    # Validate every sheet before spending GPU time or creating partial output.
    sheets = inspect_lock_batch(paths)
    total = sum(len(sheet.boxes) for sheet in sheets)
    category = f"reference-panels/lock_batch_{uuid4().hex}"
    gallery, files, manifest = [], [], []
    archive_entries = []
    for source_index, sheet in enumerate(sheets, 1):
        folder = f"{source_index:02d}_{storage._slug(sheet.name)}"
        with Image.open(sheet.path) as uploaded:
            source = ImageOps.exif_transpose(uploaded).convert("RGB")
        for panel_index, box in enumerate(sheet.boxes, 1):
            if progress:
                progress(len(files) / total, desc=f"{sheet.name} · panel {panel_index}/{len(sheet.boxes)}")
            # Process one panel at a time instead of holding 61 4K images in RAM.
            panel = extract_and_enhance_panels(
                source, scale=1, sharpen=sharpen,
                target_long_side=3840 if "4K" in quality else None,
                ai_upscaler=ai_upscaler if use_ai else None, boxes=[box],
            )[0]
            filename = f"panel_{panel_index:02d}.png"
            record = {
                "source": sheet.path.name, "panel": panel_index,
                "source_size": sheet.size, "crop_box": [box.left, box.top, box.right, box.bottom],
                "output_size": panel.size, "mode": quality, "archive_path": f"{folder}/{filename}",
            }
            saved = storage.save_image(panel, f"{category}/{folder}", f"{sheet.name}_panel_{panel_index:02d}", record)
            thumb = panel.copy()
            thumb.thumbnail((420, 420))
            preview = storage.save_image(thumb, f"{category}/previews", f"{folder}_{panel_index:02d}")
            gallery.append((str(preview), f"{sheet.name} · {panel_index:02d} · {panel.width} × {panel.height}px"))
            files.append(str(saved))
            archive_entries.append((saved, record["archive_path"]))
            manifest.append(record)
            del panel, thumb
    zip_path = storage.unique_path(category, "velora_smart_lock_panels", "zip")
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        for saved, name in archive_entries:
            archive.write(saved, arcname=name)
        archive.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    if progress:
        progress(1, desc="Batch complete")
    status = f"Completed {len(sheets)} collage(s), {len(files)} panels. ZIP grouped by model; every short edge is at least 1024px."
    if use_ai:
        status += " AI restoration can alter tiny characters: inspect logos, keypad digits and sensors before use."
    return gallery, files, str(zip_path), status
