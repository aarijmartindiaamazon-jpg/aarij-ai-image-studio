"""Gutter-based layouts for two-row smart-lock product contact sheets."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from src.reference_studio import PanelBox, _runs


def detect_lock_panel_boxes(image: Image.Image) -> list[PanelBox]:
    """Find six/seven panels from actual white gutters, never product edges.

    Rows are analysed independently to support unequal column widths and the
    V60-style wide packaging panel. Unrecognised layouts require user review.
    """
    rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    height, width = rgb.shape[:2]
    white = rgb.min(axis=2) >= 245
    horizontal = _runs(white.mean(axis=1) >= 0.98)
    middle = [(a, b) for a, b in horizontal
              if height * .35 < a < b < height * .65 and b - a <= height * .04]
    if len(middle) != 1:
        raise ValueError("Expected a two-row smart-lock collage. Select General Auto for other layouts.")
    top, bottom = 0, height
    if horizontal and horizontal[0][0] == 0 and horizontal[0][1] < height * .06:
        top = horizontal[0][1]
    if horizontal and horizontal[-1][1] == height and horizontal[-1][0] > height * .94:
        bottom = horizontal[-1][0]

    result = []
    row_counts = []
    for y1, y2 in [(top, middle[0][0]), (middle[0][1], bottom)]:
        # Ignore rounded panel corners when projecting the vertical gutters.
        inset = max(1, round((y2 - y1) * .05))
        vertical = _runs(white[y1 + inset:y2 - inset].mean(axis=0) >= .98)
        left, right = 0, width
        if vertical and vertical[0][0] == 0 and vertical[0][1] < width * .06:
            left = vertical[0][1]
        if vertical and vertical[-1][1] == width and vertical[-1][0] > width * .94:
            right = vertical[-1][0]
        dividers = [(a, b) for a, b in vertical
                    if a > width * .1 and b < width * .9 and b - a <= width * .03]
        if len(dividers) not in (2, 3):
            raise ValueError("Could not confidently identify the lock panels. Check the layout preview or use General Auto.")
        row_boxes = []
        for a, b in dividers + [(right, right)]:
            if a - left < width * .12:
                raise ValueError("Detected an unusually narrow panel. Please check this collage layout.")
            row_boxes.append(PanelBox(left, y1, a, y2))
            left = b
        row_counts.append(len(row_boxes))
        result.extend(row_boxes)
    if row_counts not in ([3, 3], [4, 3]):
        raise ValueError("Smart-lock mode supports 3+3 or 4+3 panels. Use General Auto for this layout.")
    return result


def layout_preview(image: Image.Image, boxes: list[PanelBox]) -> Image.Image:
    """Draw numbered crop boundaries for inspection without altering exports."""
    preview = ImageOps.exif_transpose(image).convert("RGB").copy()
    draw = ImageDraw.Draw(preview)
    for number, box in enumerate(boxes, 1):
        draw.rectangle((box.left, box.top, box.right - 1, box.bottom - 1), outline="#16a34a", width=3)
        draw.rectangle((box.left + 5, box.top + 5, box.left + 38, box.top + 30), fill="#166534")
        draw.text((box.left + 13, box.top + 10), str(number), fill="white")
    preview.thumbnail((1000, 700))
    return preview
