import numpy as np
from PIL import Image

from src.reference_studio import detect_panel_boxes, extract_and_enhance_panels


def test_detects_asymmetric_collage_panels():
    canvas = np.full((300, 400, 3), 20, dtype=np.uint8)
    canvas[:, 198:202] = 255
    canvas[:198, 98:102] = 255
    canvas[198:202, :198] = 255
    image = Image.fromarray(canvas)

    boxes = detect_panel_boxes(image, min_panel=40)
    assert len(boxes) == 4
    assert all(box.width >= 40 and box.height >= 40 for box in boxes)


def test_extract_upscales_every_panel():
    canvas = Image.new("RGB", (204, 100), "black")
    for x in range(100, 104):
        for y in range(100):
            canvas.putpixel((x, y), (255, 255, 255))

    panels = extract_and_enhance_panels(canvas, scale=2, sharpen=0)
    assert len(panels) == 2
    assert panels[0].size == (200, 200)


def test_detects_thin_dark_dividers_without_splitting_wide_backgrounds():
    canvas = np.full((220, 320, 3), 80, dtype=np.uint8)
    canvas[:, 158:162] = 0
    canvas[108:112, :158] = 0
    canvas[:, 230:300] = 5  # wide dark image area must not become a divider
    image = Image.fromarray(canvas)

    boxes = detect_panel_boxes(image, min_panel=40)
    assert len(boxes) == 3
