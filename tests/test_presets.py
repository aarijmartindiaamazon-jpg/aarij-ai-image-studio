import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_marketplace_presets_exist():
    presets = json.loads((ROOT / "presets" / "marketplace.json").read_text(encoding="utf-8"))
    required = {"Amazon Main Image", "Flipkart Main Image", "Meesho Product Image", "Advertisement Image", "Fabric Close-Up"}
    assert required.issubset(presets)


def test_styles_are_nonempty():
    styles = json.loads((ROOT / "presets" / "styles.json").read_text(encoding="utf-8"))
    assert styles and all(isinstance(value, str) and value for value in styles.values())

