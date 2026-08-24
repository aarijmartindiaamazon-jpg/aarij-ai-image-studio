from __future__ import annotations

from dataclasses import dataclass

from .config import Settings, settings


@dataclass(frozen=True)
class ExtensionStatus:
    name: str
    enabled: bool
    note: str


def extension_status(app_settings: Settings = settings) -> list[ExtensionStatus]:
    """Feature flags keep optional preservation modules out of the reliable base path."""
    return [
        ExtensionStatus("ControlNet", app_settings.enable_controlnet, "Requires a compatible SDXL control model and additional VRAM."),
        ExtensionStatus("IP-Adapter", app_settings.enable_ip_adapter, "Reference conditioning may exceed a free T4 budget with quality mode."),
        ExtensionStatus("LoRA", app_settings.enable_lora, "Only load trusted, SDXL-compatible adapters one at a time."),
    ]


def require_implemented(status: ExtensionStatus) -> None:
    if status.enabled:
        raise NotImplementedError(f"{status.name} is feature-gated but no adapter configuration was supplied.")

