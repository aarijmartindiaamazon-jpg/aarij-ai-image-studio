from PIL import Image

from src.storage import SUBDIRECTORIES, StorageManager


def test_storage_creates_directories(tmp_path):
    manager = StorageManager(tmp_path / "studio")
    assert all((manager.root / name).is_dir() for name in SUBDIRECTORIES)


def test_unique_paths_do_not_overwrite(tmp_path):
    manager = StorageManager(tmp_path)
    first = manager.save_image(Image.new("RGB", (2, 2)), "generated-images", "Amazon Product")
    second = manager.save_image(Image.new("RGB", (2, 2)), "generated-images", "Amazon Product")
    assert first != second
    assert first.exists() and second.exists()

