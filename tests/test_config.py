from src.config import MODELS, Settings


def test_models_are_registered():
    assert MODELS["fast"].model_id == "stabilityai/sdxl-turbo"
    assert MODELS["quality"].supports_negative_prompt


def test_invalid_default_falls_back(monkeypatch, tmp_path):
    monkeypatch.setenv("DEFAULT_MODEL", "unknown")
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(tmp_path))
    assert Settings().default_model == "fast"

