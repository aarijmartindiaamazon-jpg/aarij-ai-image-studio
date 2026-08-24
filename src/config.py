from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COLAB_OUTPUT = Path("/content/drive/MyDrive/Aarij-AI-Image-Studio")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ModelSpec:
    label: str
    model_id: str
    default_steps: int
    default_guidance: float
    supports_negative_prompt: bool


MODELS = {
    "fast": ModelSpec("Fast (SDXL Turbo)", "stabilityai/sdxl-turbo", 3, 0.0, False),
    "quality": ModelSpec("Quality (SDXL 1.0)", "stabilityai/stable-diffusion-xl-base-1.0", 30, 7.0, True),
}


@dataclass
class Settings:
    debug: bool = field(default_factory=lambda: _env_bool("DEBUG", False))
    default_model: str = field(default_factory=lambda: os.getenv("DEFAULT_MODEL", "fast").lower())
    output_root: Path = field(default_factory=lambda: Path(os.getenv("OUTPUT_DIRECTORY", str(DEFAULT_COLAB_OUTPUT if Path("/content").exists() else PROJECT_ROOT / "outputs"))))
    enable_inpainting: bool = field(default_factory=lambda: _env_bool("ENABLE_INPAINTING", True))
    enable_controlnet: bool = field(default_factory=lambda: _env_bool("ENABLE_CONTROLNET", False))
    enable_ip_adapter: bool = field(default_factory=lambda: _env_bool("ENABLE_IP_ADAPTER", False))
    enable_lora: bool = field(default_factory=lambda: _env_bool("ENABLE_LORA", False))
    enable_cpu_offload: bool = field(default_factory=lambda: _env_bool("ENABLE_CPU_OFFLOAD", True))
    share: bool = field(default_factory=lambda: _env_bool("GRADIO_SHARE", Path("/content").exists()))

    def __post_init__(self) -> None:
        if self.default_model not in MODELS:
            self.default_model = "fast"


settings = Settings()

