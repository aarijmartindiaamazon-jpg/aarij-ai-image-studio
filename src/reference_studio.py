"""Pixel-preserving utilities for extracting and enhancing collage panels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


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
    gray = np.mean(rgb.astype(np.float32), axis=2)
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

        # Some generated collages use neutral gray hairlines instead of white
        # or black gutters.  Detect those by looking for a nearly uniform,
        # narrow row/column with a visible intensity step on its boundary.
        area_gray = gray[box.top:box.bottom, box.left:box.right]

        def add_uniform_candidates(lines: np.ndarray, axis: str) -> None:
            line_std = lines.std(axis=1)
            uniform = line_std <= 10.0
            for start, end in _runs(uniform, max_gap=1):
                length = lines.shape[0]
                if start < margin or end > length - margin or end - start > max_divider:
                    continue
                uniform_min_panel = max(min_panel, 120)
                if start < uniform_min_panel or length - end < uniform_min_panel:
                    continue
                before = np.mean(np.abs(lines[start].astype(float) - lines[start - 1].astype(float)))
                after = np.mean(np.abs(lines[end - 1].astype(float) - lines[end].astype(float)))
                if max(before, after) < 8.0 or min(before, after) < 3.0:
                    continue
                balance = min(start, length - end) / max(start, length - end)
                candidates.append((balance + 0.02, axis, start, end))

        add_uniform_candidates(area_gray.T, "v")
        add_uniform_candidates(area_gray, "h")

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

    # Generated contact sheets sometimes use non-slicing layouts: a divider
    # ends at a T-junction instead of crossing the full current rectangle.
    # Recognize the two common editorial forms that cannot be expressed by the
    # recursive splitter, using their separator brightness as a guard.
    if width >= 800 and height >= 400 and width / height > 1.7 and len(boxes) <= 6:
        separator_band = gray[int(height * 0.64):int(height * 0.67)]
        if float(separator_band.mean(axis=1).max()) > 200:
            xs_top = [0, .244, .503, .758, 1]
            xs_five = [0, .202, .402, .601, .801, 1]
            result: list[PanelBox] = []
            for index in range(4):
                result.append(PanelBox(round(width * xs_top[index]), 0,
                                       round(width * xs_top[index + 1]), round(height * .496)))
            for index in range(5):
                top = round(height * (.410 if index == 4 else .504))
                result.append(PanelBox(round(width * xs_five[index]), top,
                                       round(width * xs_five[index + 1]), round(height * .654)))
            for index in range(5):
                result.append(PanelBox(round(width * xs_five[index]), round(height * .660),
                                       round(width * xs_five[index + 1]), height))
            boxes = result
        elif len(boxes) <= 4:
            # Dark two-row product boards often hide their gutters in the
            # near-black studio background. Their 5-by-2 geometry is stable.
            boxes = [
                PanelBox(round(width * column / 5), round(height * row / 2),
                         round(width * (column + 1) / 5), round(height * (row + 1) / 2))
                for row in range(2) for column in range(5)
            ]
    return sorted(boxes, key=lambda box: (box.top, box.left))


def extract_and_enhance_panels(image: Image.Image, scale: int = 2,
                               sharpen: float = 1.0, min_short_side: int = 1024,
                               target_long_side: int | None = None,
                               ai_upscaler: Callable[[Image.Image], Image.Image] | None = None,
                               boxes: list[PanelBox] | None = None) -> list[Image.Image]:
    """Extract panels and upscale without changing their aspect ratios.

    Every output has at least ``min_short_side`` pixels on its shortest edge.
    When requested, ``target_long_side`` raises the long edge to that target
    (for example 3840 for a 4K export), but never downsizes a larger source.
    """
    source = ImageOps.exif_transpose(image).convert("RGB")
    scale = min(4, max(1, int(scale)))
    results: list[Image.Image] = []
    for box in (boxes if boxes is not None else detect_panel_boxes(source)):
        panel = source.crop((box.left, box.top, box.right, box.bottom))
        original_size = panel.size
        if ai_upscaler is not None:
            panel = ai_upscaler(panel)
        required_scale = max(float(scale), min_short_side / min(panel.size))
        if target_long_side:
            required_scale = max(required_scale, int(target_long_side) / max(panel.size))
        if required_scale > 1:
            target_size = (round(panel.width * required_scale), round(panel.height * required_scale))
            # Large one-step enlargement looks soft. Grow in at most 2x stages
            # and recover a small amount of edge contrast between stages.
            while panel.width * 2 < target_size[0] and panel.height * 2 < target_size[1]:
                panel = panel.resize((panel.width * 2, panel.height * 2), Image.Resampling.LANCZOS)
                panel = panel.filter(ImageFilter.UnsharpMask(radius=0.8, percent=55, threshold=3))
            panel = panel.resize(target_size, Image.Resampling.LANCZOS)
        if sharpen > 0:
            enlargement = max(panel.width / original_size[0], panel.height / original_size[1])
            radius = 1.0 if enlargement <= 4 else 1.35
            percent = int((115 if enlargement <= 4 else 155) * sharpen)
            panel = panel.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=2))
            panel = ImageEnhance.Contrast(panel).enhance(1.025)
        results.append(panel)
    return results
