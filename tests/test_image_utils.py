from PIL import Image

from src.image_utils import fit_inside, square_canvas
from src.marketplace import create_white_background


def test_fit_inside_preserves_aspect_ratio():
    image = Image.new("RGB", (400, 200), "red")
    result = fit_inside(image, 100, 100)
    assert result.size == (100, 50)


def test_square_canvas_centers_image():
    image = Image.new("RGBA", (100, 50), (255, 0, 0, 255))
    result = square_canvas(image, size=200, padding=20)
    assert result.size == (200, 200)
    assert result.getpixel((100, 100))[:3] == (255, 0, 0)


def test_marketplace_composition_without_downloading_rembg(monkeypatch):
    transparent = Image.new("RGBA", (100, 50), (10, 20, 30, 255))
    monkeypatch.setattr("src.marketplace.remove_background", lambda image: transparent)
    result = create_white_background(transparent, canvas_size=500, product_size_percent=80)
    assert result.size == (500, 500)
    assert result.mode == "RGB"
    assert result.getpixel((0, 0)) == (255, 255, 255)

