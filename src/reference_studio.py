"""Pixel-preserving utilities for extracting and enhancing collage panels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


@dataclass(frozen=True)
class PanelBox:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


def _runs(values: np.ndarray, max_gap: int = 0) -> list[tuple[int, int]]:
    indexes = np.flatnonzero(values)
    if not len(indexes):
        return []
    starts = [int(indexes[0])]
    ends: list[int] = []
    for previous, current in zip(indexes, indexes[1:]):
        if current > previous + max_gap + 1:
            ends.append(int(previous) + 1)
            starts.append(int(current))
    ends.append(int(indexes[-1]) + 1)
    return list(zip(starts, ends))


def detect_panel_boxes(image: Image.Image, *, white_threshold: int = 230,
                       dark_threshold: int = 14,
                       coverage: float = 0.96, min_panel: int = 96,
                       max_divider: int = 16) -> list[PanelBox]:
    """Recursively split a collage along thin near-white separator lines.

    Recursive splitting supports asymmetric editorial collages rather than only
    fixed row/column grids. If no reliable divider exists, the full image is
    returned as one panel.
    """
    rgb = np.asarray(image.convert("RGB"))
    bright_pixels = np.min(rgb, axis=2) >= white_threshold
    dark_pixels = np.max(rgb, axis=2) <= dark_threshold
    height, width = bright_pixels.shape

    def split(box: PanelBox) -> list[PanelBox]:
        if box.width < min_panel * 2 or box.height < min_panel:
            return [box]
        margin = max(4, min(20, min(box.width, box.height) // 20))
        candidates: list[tuple[float, str, int, int]] = []

        def add_candidates(pixel_mask: np.ndarray, allowed_width: int, max_gap: int = 0) -> None:
            area = pixel_mask[box.top:box.bottom, box.left:box.right]
            vertical = area.mean(axis=0) >= coverage
            for start, end in _runs(vertical, max_gap=max_gap):
                if start < margin or end > box.width - margin or end - start > allowed_width:
                    continue
                if start < min_panel or box.width - end < min_panel:
                    continue
                balance = min(start, box.width - end) / max(start, box.width - end)
                candidates.append((balance, "v", start, end))

            horizontal = area.mean(axis=1) >= coverage
            for start, end in _runs(horizontal, max_gap=max_gap):
                if start < margin or end > box.height - margin or end - start > allowed_width:
                    continue
                if start < min_panel or box.height - end < min_panel:
                    continue
                balance = min(start, box.height - end) / max(start, box.height - end)
                candidates.append((balance, "h", start, end))

        add_candidates(bright_pixels, max_divider)
        # Dark editorial gutters are often wider because they blend into black
        # product-photo backgrounds. Very wide dark runs remain excluded.
        add_candidates(dark_pixels, max(64, max_divider), max_gap=3)

        if not candidates:
            return [box]
        _, axis, start, end = max(candidates, key=lambda item: item[0])
        if axis == "v":
            first = PanelBox(box.left, box.top, box.left + start, box.bottom)
            second = PanelBox(box.left + end, box.top, box.right, box.bottom)
        else:
            first = PanelBox(box.left, box.top, box.right, box.top + start)
            second = PanelBox(box.left, box.top + end, box.right, box.bottom)
        return split(first) + split(second)

    boxes = split(PanelBox(0, 0, width, height))
    return sorted(boxes, key=lambda box: (box.top, box.left))


def extract_and_enhance_panels(image: Image.Image, scale: int = 2,
                               sharpen: float = 1.0) -> list[Image.Image]:
    """Extract all detected panels, upscale with Lanczos, and lightly sharpen."""
    source = image.convert("RGB")
    scale = min(4, max(1, int(scale)))
    results: list[Image.Image] = []
    for box in detect_panel_boxes(source):
        panel = source.crop((box.left, box.top, box.right, box.bottom))
        if scale > 1:
            panel = panel.resize((panel.width * scale, panel.height * scale), Image.Resampling.LANCZOS)
        if sharpen > 0:
            radius = 1.2 if scale <= 2 else 1.8
            panel = panel.filter(ImageFilter.UnsharpMask(radius=radius, percent=int(90 * sharpen), threshold=3))
            panel = ImageEnhance.Contrast(panel).enhance(1.02)
        results.append(panel)
    return results
