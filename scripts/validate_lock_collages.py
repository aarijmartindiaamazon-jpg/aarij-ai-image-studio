"""Validate local VELORA lock boards without uploading them or using GPU time."""

import argparse
from pathlib import Path
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.lock_collage import detect_lock_panel_boxes, layout_preview


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    total = 0
    previews = []
    for folder in args.folders:
        files = sorted(folder.glob("VELORA*.png"))
        if not files:
            raise ValueError(f"No VELORA PNG files in {folder}")
        for file in files:
            with Image.open(file) as image:
                boxes = detect_lock_panel_boxes(image)
                expected = 7 if "V60" in file.stem else 6
                assert len(boxes) == expected, file.name
                total += len(boxes)
                preview = layout_preview(image, boxes)
                preview.thumbnail((506, 340))
                previews.append((file.name, preview.copy()))
            print(f"PASS {file.name}: {len(boxes)} panels")
    if args.preview:
        sheet = Image.new("RGB", (1040, ((len(previews) + 1) // 2) * 376), "white")
        draw = ImageDraw.Draw(sheet)
        for index, (name, preview) in enumerate(previews):
            x, y = (index % 2) * 520, (index // 2) * 376
            draw.text((x + 8, y + 8), name, fill="black")
            sheet.paste(preview, (x + 8, y + 30))
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(args.preview)
    print(f"Validated {len(previews)} collages / {total} panels")


if __name__ == "__main__":
    main()
