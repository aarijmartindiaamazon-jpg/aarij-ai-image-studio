from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from src.config import MODELS, PROJECT_ROOT, settings
from src.errors import friendly_error
from src.generation import generate_image
from src.gpu_utils import gpu_status
from src.img2img import transform_image
from src.inpainting import inpaint_image
from src.marketplace import create_white_background
from src.model_manager import model_manager
from src.storage import StorageManager


def _load_json(name: str) -> dict:
    return json.loads((PROJECT_ROOT / "presets" / name).read_text(encoding="utf-8"))


STYLES = _load_json("styles.json")
MARKETPLACE = _load_json("marketplace.json")
storage = StorageManager(settings.output_root)


def _styled_prompt(prompt: str, style: str) -> str:
    suffix = STYLES.get(style, "")
    return f"{prompt.strip()}, {suffix}" if suffix else prompt.strip()


def run_generation(prompt, negative, style, mode_label, width, height, seed, randomize, steps, guidance):
    try:
        mode = "fast" if str(mode_label).startswith("Fast") else "quality"
        image, used_seed = generate_image(_styled_prompt(prompt, style), negative, mode, width, height,
                                          int(steps), float(guidance), int(seed), bool(randomize))
        path = storage.save_image(image, "generated-images", mode, {"prompt": prompt, "seed": used_seed, "mode": mode})
        return image, str(used_seed), str(path), f"Complete · {model_manager.status()}"
    except Exception as exc:
        return None, "—", None, friendly_error(exc, settings.debug)


def run_white_background(source, canvas, coverage, padding, shadow, strength, preset):
    try:
        selected = MARKETPLACE.get(preset, {})
        if selected.get("workflow") == "white_background":
            coverage = selected.get("product_size_percent", coverage)
            shadow = selected.get("shadow", shadow)
        image = create_white_background(source, int(canvas), int(coverage), int(padding), bool(shadow), float(strength))
        path = storage.save_image(image, "white-background", preset or "product")
        return image, str(path), "Complete. The original product pixels were preserved; only background removal and composition were applied."
    except Exception as exc:
        return None, None, friendly_error(exc, settings.debug)


def run_img2img(source, prompt, negative, preset, strength, mode_label, width, height, seed, randomize):
    try:
        mode = "fast" if str(mode_label).startswith("Fast") else "quality"
        preset_prompt = MARKETPLACE.get(preset, {}).get("prompt", "")
        full_prompt = ", ".join(part for part in (prompt.strip(), preset_prompt) if part)
        image, used_seed = transform_image(source, full_prompt, negative, float(strength), mode,
                                           int(width), int(height), int(seed), bool(randomize))
        category = "lifestyle" if preset in list(MARKETPLACE)[3:10] else "marketplace"
        path = storage.save_image(image, category, preset or "img2img", {"prompt": full_prompt, "seed": used_seed})
        return image, str(used_seed), str(path), "Complete. Check product details carefully before marketplace use."
    except Exception as exc:
        return None, "—", None, friendly_error(exc, settings.debug)


def run_inpainting(source, mask, prompt, negative, mode_label, seed, randomize):
    try:
        mode = "fast" if str(mode_label).startswith("Fast") else "quality"
        image, used_seed = inpaint_image(source, mask, prompt, negative, mode, int(seed), bool(randomize))
        path = storage.save_image(image, "marketplace", "inpaint", {"prompt": prompt, "seed": used_seed})
        return image, str(used_seed), str(path), "Complete. White areas of the uploaded mask were modified."
    except Exception as exc:
        return None, "—", None, friendly_error(exc, settings.debug)


def unload_models():
    model_manager.unload()
    return f"{model_manager.status()}\n\n{gpu_status()}"


def build_app() -> gr.Blocks:
    css = ".hero{text-align:center;margin-bottom:10px}.notice{border-radius:10px;padding:8px}"
    with gr.Blocks(title="Aarij AI Image Studio", theme=gr.themes.Soft(), css=css) as app:
        gr.Markdown("# Aarij AI Image Studio\n<div class='hero'>Free, Colab-first AI generation and product-image preparation</div>")
        gr.Markdown("Marketplace accuracy note: white-background mode preserves the uploaded product. AI transformations can alter color, logos, stitching, texture, and proportions.")

        with gr.Tab("AI Generator"):
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(label="Describe your image", lines=4, placeholder="A premium bean bag in a bright modern living room...")
                    negative = gr.Textbox(label="Avoid (Quality mode)", placeholder="text, watermark, distorted product")
                    style = gr.Dropdown(list(STYLES), value="Photorealistic Product", label="Style")
                    mode = gr.Radio([MODELS["fast"].label, MODELS["quality"].label], value=MODELS[settings.default_model].label, label="Generation mode")
                    with gr.Accordion("Advanced Settings", open=False):
                        width = gr.Slider(256, 1024, 512, step=64, label="Width")
                        height = gr.Slider(256, 1024, 512, step=64, label="Height")
                        seed = gr.Number(value=0, precision=0, label="Seed")
                        randomize = gr.Checkbox(value=True, label="Use a random seed")
                        steps = gr.Slider(1, 50, 3, step=1, label="Inference steps")
                        guidance = gr.Slider(0, 12, 0, step=0.5, label="Prompt guidance")
                    generate = gr.Button("Generate Image", variant="primary")
                with gr.Column():
                    output = gr.Image(label="Generated image", type="pil")
                    used_seed = gr.Textbox(label="Seed used", interactive=False)
                    download = gr.File(label="Download image")
                    status = gr.Textbox(label="Status", interactive=False)
            generate.click(run_generation, [prompt, negative, style, mode, width, height, seed, randomize, steps, guidance], [output, used_seed, download, status])
            mode.change(
                lambda choice: (3, 0.0) if str(choice).startswith("Fast") else (30, 7.0),
                inputs=mode,
                outputs=[steps, guidance],
            )

        with gr.Tab("White Background"):
            gr.Markdown("Best for marketplace main images. This removes the background without regenerating the product.")
            with gr.Row():
                white_source = gr.Image(label="Upload Product", type="pil")
                white_output = gr.Image(label="White-background result", type="pil")
            white_preset = gr.Dropdown(list(MARKETPLACE)[:3], value="Amazon Main Image", label="Marketplace preset")
            with gr.Accordion("Composition Settings", open=True):
                canvas = gr.Dropdown([1000, 1600, 2000, 2400], value=2000, label="Canvas size (square)")
                coverage = gr.Slider(20, 95, 85, step=1, label="Product size (%)")
                padding = gr.Slider(0, 300, 40, step=5, label="Extra padding")
                shadow = gr.Checkbox(False, label="Add natural shadow")
                shadow_strength = gr.Slider(0.05, 1, 0.3, step=0.05, label="Shadow strength")
            white_button = gr.Button("Create White Background", variant="primary")
            white_download = gr.File(label="Download")
            white_status = gr.Textbox(label="Status", interactive=False)
            white_button.click(run_white_background, [white_source, canvas, coverage, padding, shadow, shadow_strength, white_preset], [white_output, white_download, white_status])

        with gr.Tab("Marketplace Presets"):
            gr.Markdown("Main-image presets use the White Background tab. Lifestyle, fabric and advertisement presets are available below.")
            preset_summary = gr.JSON(MARKETPLACE, label="Included presets")

        with gr.Tab("Product Lifestyle"):
            gr.Markdown("AI can change real product details. Use low strength for better preservation and verify every result.")
            with gr.Row():
                life_source = gr.Image(label="Upload Product", type="pil")
                life_output = gr.Image(label="AI-assisted result", type="pil")
            life_preset = gr.Dropdown(list(MARKETPLACE)[3:], value="Modern living room", label="Scene or creative preset")
            life_prompt = gr.Textbox(label="Additional instructions", placeholder="Keep the product centered with natural window light")
            life_negative = gr.Textbox(label="Avoid", value="text, watermark, fake logo, distorted product")
            life_strength = gr.Slider(0.1, 0.9, 0.3, step=0.05, label="Transformation strength")
            life_mode = gr.Radio([MODELS["fast"].label, MODELS["quality"].label], value=MODELS["fast"].label, label="Mode")
            with gr.Accordion("Advanced Settings", open=False):
                life_width = gr.Slider(256, 1024, 768, step=64, label="Width")
                life_height = gr.Slider(256, 1024, 768, step=64, label="Height")
                life_seed = gr.Number(0, precision=0, label="Seed")
                life_random = gr.Checkbox(True, label="Use a random seed")
            life_button = gr.Button("Create Lifestyle Image", variant="primary")
            life_seed_out = gr.Textbox(label="Seed used", interactive=False)
            life_download = gr.File(label="Download")
            life_status = gr.Textbox(label="Status", interactive=False)
            life_button.click(run_img2img, [life_source, life_prompt, life_negative, life_preset, life_strength, life_mode, life_width, life_height, life_seed, life_random], [life_output, life_seed_out, life_download, life_status])

        with gr.Tab("Image-to-Image"):
            with gr.Row():
                i2i_source = gr.Image(label="Source image", type="pil")
                i2i_output = gr.Image(label="Result", type="pil")
            i2i_prompt = gr.Textbox(label="Transformation instructions")
            i2i_negative = gr.Textbox(label="Avoid", value="text, watermark, altered logo, distorted details")
            i2i_preset = gr.Radio(["Preserve Product", "Balanced", "Creative"], value="Preserve Product", label="Transformation preset")
            i2i_strength = gr.Slider(0.1, 0.9, 0.25, step=0.05, label="Strength")
            i2i_preset.change(lambda x: {"Preserve Product": 0.2, "Balanced": 0.45, "Creative": 0.7}[x], i2i_preset, i2i_strength)
            i2i_mode = gr.Radio([MODELS["fast"].label, MODELS["quality"].label], value=MODELS["fast"].label, label="Mode")
            i2i_width = gr.Slider(256, 1024, 768, step=64, label="Width")
            i2i_height = gr.Slider(256, 1024, 768, step=64, label="Height")
            i2i_seed = gr.Number(0, precision=0, label="Seed")
            i2i_random = gr.Checkbox(True, label="Use a random seed")
            i2i_button = gr.Button("Transform Image", variant="primary")
            i2i_seed_out, i2i_download, i2i_status = gr.Textbox(label="Seed used"), gr.File(label="Download"), gr.Textbox(label="Status")
            i2i_button.click(run_img2img, [i2i_source, i2i_prompt, i2i_negative, gr.State(""), i2i_strength, i2i_mode, i2i_width, i2i_height, i2i_seed, i2i_random], [i2i_output, i2i_seed_out, i2i_download, i2i_status])

        with gr.Tab("Inpainting"):
            gr.Markdown("Upload a black-and-white mask: white areas change; black areas remain protected. Quality mode is recommended.")
            with gr.Row():
                in_source = gr.Image(label="Source image", type="pil")
                in_mask = gr.Image(label="Mask (white = modify)", type="pil")
                in_output = gr.Image(label="Result", type="pil")
            in_prompt = gr.Textbox(label="Describe the change")
            in_negative = gr.Textbox(label="Avoid", value="text, watermark, distortion")
            in_mode = gr.Radio([MODELS["fast"].label, MODELS["quality"].label], value=MODELS["quality"].label, label="Mode")
            in_seed, in_random = gr.Number(0, precision=0, label="Seed"), gr.Checkbox(True, label="Use a random seed")
            in_button = gr.Button("Modify Selected Area", variant="primary")
            in_seed_out, in_download, in_status = gr.Textbox(label="Seed used"), gr.File(label="Download"), gr.Textbox(label="Status")
            in_button.click(run_inpainting, [in_source, in_mask, in_prompt, in_negative, in_mode, in_seed, in_random], [in_output, in_seed_out, in_download, in_status])

        with gr.Tab("Settings"):
            settings_status = gr.Markdown(f"**Output folder:** `{settings.output_root}`\n\n{gpu_status()}\n\n{model_manager.status()}")
            unload = gr.Button("Unload AI Model and Free GPU Memory")
            refresh = gr.Button("Refresh Status")
            unload.click(unload_models, outputs=settings_status)
            refresh.click(lambda: f"{model_manager.status()}\n\n{gpu_status()}", outputs=settings_status)
            gr.Markdown("Advanced preservation hooks (ControlNet, IP-Adapter and LoRA) are intentionally feature-gated for future releases so the base T4 workflow stays reliable.")
    return app
