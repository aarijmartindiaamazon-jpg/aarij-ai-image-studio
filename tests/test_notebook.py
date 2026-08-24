import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_colab_launch_allows_drive_downloads():
    notebook = json.loads((ROOT / "notebooks" / "Aarij_AI_Image_Studio_Colab.ipynb").read_text(encoding="utf-8"))
    code = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code")
    assert "allowed_paths=['/content/drive/MyDrive/Aarij-AI-Image-Studio']" in code

