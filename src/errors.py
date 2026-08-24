from __future__ import annotations


def friendly_error(exc: Exception, debug: bool = False) -> str:
    text = str(exc)
    lower = text.lower()
    if "out of memory" in lower or "cuda" in lower and "memory" in lower:
        return "GPU memory is full. Try Fast mode, 512×512, fewer steps, then unload the model in Settings."
    if "cuda" in lower or "gpu" in lower:
        return "A supported GPU is unavailable. In Colab select Runtime → Change runtime type → T4 GPU."
    if "401" in lower or "gated" in lower or "token" in lower:
        return "Model access was denied. Accept the model license on Hugging Face and add an HF_TOKEN Colab secret if required."
    if "download" in lower or "connection" in lower:
        return "The model could not be downloaded. Check the internet connection and Hugging Face availability."
    return text if debug else f"The operation could not be completed: {text}"

