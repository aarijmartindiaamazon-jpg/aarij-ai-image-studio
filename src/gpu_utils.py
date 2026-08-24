from __future__ import annotations

import gc


def cleanup_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except (ImportError, RuntimeError):
        pass


def gpu_status() -> str:
    try:
        import torch
        if not torch.cuda.is_available():
            return "GPU unavailable — image generation requires a Colab GPU runtime."
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / (1024 ** 3)
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
        return f"GPU: {props.name} · {total:.1f} GB VRAM · {allocated:.1f} GB currently used"
    except Exception as exc:
        return f"Could not read GPU status: {exc}"

