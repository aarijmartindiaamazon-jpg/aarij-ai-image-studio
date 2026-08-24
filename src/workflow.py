from __future__ import annotations


def next_chat_step(message: str, has_product: bool = False) -> str:
    """Small rule-based foundation for a future chat-style product workflow."""
    text = (message or "").lower()
    if not has_product:
        return "Upload your product image, then choose a white-background or lifestyle workflow."
    if any(word in text for word in ("amazon", "flipkart", "meesho", "white")):
        return "Use White Background to preserve the real product, then choose the matching marketplace preset."
    if any(word in text for word in ("lifestyle", "room", "scene", "advert")):
        return "Choose Product Lifestyle, select a scene, and start with low transformation strength."
    return "Choose White Background for accuracy or Product Lifestyle for an AI-assisted scene."

