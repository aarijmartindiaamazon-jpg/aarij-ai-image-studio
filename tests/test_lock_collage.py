import json
from pathlib import Path
from zipfile import ZipFile

import pytest
from PIL import Image, ImageDraw

from src.lock_batch import export_lock_batch, inspect_lock_batch
from src.lock_collage import detect_lock_panel_boxes
from src.product_reference import fit_product_reference, preservation_prompt, reference_steps
from src.storage import StorageManager


def make_sheet(top_cuts=(198, 398), bottom_cuts=(198, 398), background=238, rounded=False):
    image = Image.new("RGB", (600, 400), "white")
    draw = ImageDraw.Draw(image)
    for y1, y2, cuts in [(0, 196, top_cuts), (204, 400, bottom_cuts)]:
        left = 0
        for right in (*cuts, 600):
            bounds = (left, y1, right - 1, y2 - 1)
            if rounded:
                draw.rounded_rectangle(bounds, radius=8, fill=(background,) * 3)
            else:
                draw.rectangle(bounds, fill=(background,) * 3)
            # Fake body/door edges must not become dividers.
            draw.rectangle((left + 45, y1 + 10, left + 65, y2 - 10), fill="black")
            left = right + 6
    return image


@pytest.mark.parametrize("background,rounded", [(238, False), (230, True), (200, False)])
def test_lock_grid_ignores_bright_backgrounds_and_product_edges(background, rounded):
    boxes = detect_lock_panel_boxes(make_sheet(background=background, rounded=rounded))
    assert len(boxes) == 6
    assert [(b.left, b.right) for b in boxes[:3]] == [(0, 198), (204, 398), (404, 600)]
    assert all(b.height >= 196 for b in boxes)


@pytest.mark.parametrize("top,bottom,expected", [
    ((145, 295, 445), (145, 295), 7),
    ((198, 398), (198, 338), 6),
    ((168, 378), (168, 378), 6),
])
def test_lock_grid_keeps_unequal_and_wide_panels(top, bottom, expected):
    boxes = detect_lock_panel_boxes(make_sheet(top, bottom))
    assert len(boxes) == expected
    assert boxes[-1].left == bottom[-1] + 6
    assert boxes[-1].right == 600
    for i, first in enumerate(boxes):
        for second in boxes[i + 1:]:
            assert first.right <= second.left or second.right <= first.left or first.bottom <= second.top or second.bottom <= first.top


def test_unknown_sheet_is_not_arbitrarily_split():
    with pytest.raises(ValueError, match="two-row"):
        detect_lock_panel_boxes(Image.new("RGB", (600, 400), "gray"))


def test_batch_limit_and_invalid_image(tmp_path):
    with pytest.raises(ValueError, match="between 1 and 10"):
        inspect_lock_batch([])
    with pytest.raises(ValueError, match="between 1 and 10"):
        inspect_lock_batch(["unused"] * 11)
    with pytest.raises(ValueError, match="missing.png"):
        inspect_lock_batch([str(tmp_path / "missing.png")])


def test_batch_zip_groups_duplicate_model_names_without_overwriting(tmp_path):
    uploaded = tmp_path / "VELORA V50 ELITE-clean.png"
    make_sheet(bottom_cuts=(198, 338)).save(uploaded)
    storage = StorageManager(tmp_path / "output")
    gallery, files, archive, status = export_lock_batch([str(uploaded)] * 2, storage, "Pixel-Safe Minimum 1024px", sharpen=0)
    assert len(gallery) == len(files) == 12
    assert len(set(files)) == 12
    with ZipFile(archive) as zipped:
        metadata = json.loads(zipped.read("manifest.json"))
        assert len(zipped.namelist()) == 13
        assert len({p.split('/')[0] for p in zipped.namelist() if p.endswith('.png')}) == 2
        assert len(metadata) == 12
        assert all(min(item["output_size"]) >= 1024 for item in metadata)
        assert zipped.testzip() is None
    with Image.open(gallery[0][0]) as thumb:
        assert max(thumb.size) <= 420
    assert "12 panels" in status


def test_batch_ai_is_used_once_per_panel(tmp_path):
    uploaded = tmp_path / "lock.png"
    make_sheet().save(uploaded)
    calls = []
    def fake_ai(panel):
        calls.append(panel.size)
        return panel.resize((panel.width * 4, panel.height * 4))
    _, files, _, _ = export_lock_batch([str(uploaded)], StorageManager(tmp_path / "out"),
                                       "AI Product Enhance (4x)", sharpen=0, ai_upscaler=fake_ai)
    assert len(calls) == len(files) == 6


def test_smart_lock_guidance_does_not_describe_watch_parts():
    prompt = preservation_prompt("Smart door locks")
    assert "keypad" in prompt and "handedness" in prompt and "camera" in prompt
    assert "bracelet" not in prompt and "dial" not in prompt
    assert "dial" in preservation_prompt("Watches")


def test_reference_padding_keeps_top_and_bottom_features():
    image = Image.new("RGB", (100, 300), "black")
    ImageDraw.Draw(image).rectangle((0, 0, 99, 9), fill="red")
    ImageDraw.Draw(image).rectangle((0, 290, 99, 299), fill="blue")
    fitted = fit_product_reference(image, (300, 300))
    assert fitted.getpixel((150, 1)) == (255, 0, 0)
    assert fitted.getpixel((150, 298)) == (0, 0, 255)


@pytest.mark.parametrize("strength", [.15, .20, .30, .35, .95])
def test_low_strength_always_has_an_img2img_step(strength):
    assert int(reference_steps(1, strength) * strength) >= 1
