from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_hugging_face_space_metadata_and_entrypoint_exist():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    app = (ROOT / "hf_space_app.py").read_text(encoding="utf-8")
    assert "app_file: hf_space_app.py" in readme
    assert "black-forest-labs/FLUX.1-schnell" in app
    assert "@spaces.GPU" in app


def test_space_supports_multiple_images():
    app = (ROOT / "hf_space_app.py").read_text(encoding="utf-8")
    assert "num_images_per_prompt=count" in app
    assert "file_count=\"multiple\"" in app


def test_space_supports_reference_and_collage_workflows():
    app = (ROOT / "hf_space_app.py").read_text(encoding="utf-8")
    assert "FluxImg2ImgPipeline.from_pipe(flux)" in app
    assert "Optional reference image" in app
    assert "Collage Crop & Enhance" in app

