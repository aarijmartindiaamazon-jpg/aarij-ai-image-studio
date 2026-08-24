from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from threading import Lock
from uuid import uuid4

from PIL import Image, PngImagePlugin


SUBDIRECTORIES = ("generated-images", "uploads", "marketplace", "lifestyle", "white-background", "advertisements", "temp")
_name_lock = Lock()


class StorageManager:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.ensure_directories()

    def ensure_directories(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for name in SUBDIRECTORIES:
            (self.root / name).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _slug(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "image"

    def unique_path(self, category: str, prefix: str = "image", extension: str = "png") -> Path:
        folder = self.root / category
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with _name_lock:
            for index in range(1, 10000):
                candidate = folder / f"aarij_{self._slug(prefix)}_{stamp}_{index:03d}.{extension.lstrip('.').lower()}"
                if not candidate.exists():
                    return candidate
        return folder / f"aarij_{self._slug(prefix)}_{stamp}_{uuid4().hex[:8]}.{extension.lstrip('.').lower()}"

    def save_image(self, image: Image.Image, category: str, prefix: str, metadata: dict | None = None, extension: str = "png") -> Path:
        path = self.unique_path(category, prefix, extension)
        if path.suffix == ".png":
            info = PngImagePlugin.PngInfo()
            if metadata:
                info.add_text("aarij_metadata", json.dumps(metadata, ensure_ascii=False))
            image.save(path, "PNG", pnginfo=info)
        else:
            image.convert("RGB").save(path, "JPEG", quality=95, subsampling=0)
        return path

