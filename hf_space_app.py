"""Hugging Face ZeroGPU entry point for Aarij AI Image Studio."""

from __future__ import annotations

import random
import os
from zipfile import ZIP_DEFLATED, ZipFile
from pathlib import Path

import gradio as gr
import spaces
import torch
from diffusers import FluxImg2ImgPipeline, FluxPipeline
from PIL import Image

from src.errors import friendly_error
from src.marketplace import create_white_background
from src.reference_studio import extract_and_enhance_panels
from src.lock_collage import detect_lock_panel_boxes, layout_preview
from src.lock_batch import ENHANCEMENT_MODES, export_lock_batch, inspect_lock_batch
from src.product_reference import PRODUCT_DETAILS, fit_product_reference, preservation_prompt, reference_steps
from src.storage import StorageManager


MODEL_ID = "black-forest-labs/FLUX.1-schnell"
OUTPUT_ROOT = Path("/tmp/aarij-ai-image-studio")
storage = StorageManager(OUTPUT_ROOT)

STYLES = {
    "Ecommerce Product": "professional ecommerce product photography, accurate colors, realistic materials, clean commercial lighting",
    "Photorealistic": "high-end photorealistic photography, natural light, realistic detail, physically plausible materials",
    "Premium Studio": "premium advertising studio, elegant controlled light, refined shadows, luxury commercial composition",
    "Modern Lifestyle": "realistic modern lifestyle photography, believable scale, natural interior lighting",
    "Indian Interior": "contemporary Indian home interior, authentic tasteful decor, warm natural lighting",
    "Social Advertisement": "premium product advertisement, strong composition, empty copy space, no text and no unsupported claims",
}


# ZeroGPU optimizes CUDA placement performed during Space startup. This model is
# intentionally separate from the Colab lazy model manager.
flux = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
flux_img2img = FluxImg2ImgPipeline(**flux.components)
flux.to("cuda")
flux.set_progress_bar_config(disable=True)
flux_img2img.set_progress_bar_config(disable=True)


def _seed(value: int | float | None, randomize: bool) -> int:
    if randomize or value is None or int(value) < 0:
        return random.SystemRandom().randint(0, 2**31 - 5)
    return int(value)


@spaces.GPU(duration=120)
def generate_flux(prompt: str, reference, strength: float, product_lock: bool, style: str, width: int,
                  height: int, count: int, seed: int, randomize: bool, steps: int,
                  product_type: str = "General product",
                  progress=gr.Progress()):
    if not prompt or not prompt.strip():
        raise gr.Error("Please describe the image you want to create.")
    used_seed = _seed(seed, randomize)
    preservation = ""
    if reference is not None and product_lock:
        strength = min(float(strength), 0.35)
        preservation = preservation_prompt(product_type)
    full_prompt = f"{preservation}{prompt.strip()}, {STYLES.get(style, '')}".strip(", ")
    width = max(512, int(width) // 16 * 16)
    height = max(512, int(height) // 16 * 16)
    count = min(4, max(1, int(count)))
    generators = [torch.Generator(device="cuda").manual_seed(used_seed + index) for index in range(count)]
    progress(0.15, desc="ZeroGPU allocated · generating with FLUX")
    try:
        with torch.inference_mode():
            common = dict(
                prompt=full_prompt, width=width, height=height,
                num_inference_steps=reference_steps(steps, strength) if reference is not None else min(4, max(1, int(steps))),
                guidance_scale=0.0, num_images_per_prompt=count,
                generator=generators, max_sequence_length=256,
            )
            if reference is not None:
                init_image = fit_product_reference(reference, (width, height))
                images = flux_img2img(image=init_image, strength=float(strength), **common).images
            else:
                images = flux(**common).images
        paths = [
            str(storage.save_image(image, "generated-images", "flux", {
                "model": MODEL_ID, "prompt": full_prompt, "seed": used_seed + index,
            }))
            for index, image in enumerate(images)
        ]
        seeds = ", ".join(str(used_seed + index) for index in range(count))
        progress(1.0, desc="Complete")
        mode = "product-locked reference recreation" if reference is not None and product_lock else (
            "reference recreation" if reference is not None else "text generation"
        )
        return images, paths, seeds, f"Created {count} image(s) with FLUX.1-schnell · {mode}."
    except Exception as exc:
        raise gr.Error(friendly_error(exc, debug=True)) from exc


def white_background(source, canvas, coverage, padding, shadow, strength):
    try:
        image = create_white_background(source, int(canvas), int(coverage), int(padding), bool(shadow), float(strength))
        path = storage.save_image(image, "white-background", "marketplace")
        return image, str(path), "Complete · original product pixels preserved."
    except Exception as exc:
        raise gr.Error(friendly_error(exc, debug=True)) from exc


def split_collage(source, scale, quality, sharpen, layout="General Auto"):
    if source is None:
        raise gr.Error("Upload a collage or multi-image picture first.")
    try:
        use_ai = str(quality).startswith("AI")
        try:
            lock_boxes = detect_lock_panel_boxes(source)
        except ValueError:
            if layout.startswith("Smart locks"):
                raise
            lock_boxes = None
        target_long_side = 3840 if "4K" in str(quality) else None
        ai_upscaler = None
        if use_ai:
            from src.ai_upscale import ai_super_resolve
            ai_upscaler = ai_super_resolve
        panels = extract_and_enhance_panels(
            source, 1 if use_ai else int(scale), float(sharpen), min_short_side=1024,
            target_long_side=target_long_side, ai_upscaler=ai_upscaler,
            boxes=lock_boxes,
        )
        paths = [
            str(storage.save_image(panel, "reference-panels", f"panel_{index:02d}", {
                "workflow": "AI collage restoration" if use_ai else "collage enlargement", "scale": int(scale),
                "quality": str(quality), "minimum_short_side": 1024,
                "ai_super_resolution": use_ai,
            }))
            for index, panel in enumerate(panels, start=1)
        ]
        zip_path = storage.unique_path("reference-panels", "all_panels", "zip")
        with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
            for index, path in enumerate(paths, start=1):
                archive.write(path, arcname=f"aarij_enhanced_panel_{index:02d}.png")
        size_label = "4K long edge" if target_long_side else "minimum 1024px short edge"
        method = "Real-ESRGAN AI restoration" if use_ai else "pixel-safe enlargement"
        return panels, paths, str(zip_path), f"Extracted {len(panels)} panel(s) at {size_label} using {method}."
    except Exception as exc:
        raise gr.Error(friendly_error(exc, debug=True)) from exc


def preview_lock_batch(files):
    try:
        sheets = inspect_lock_batch(files)
        previews = []
        for sheet in sheets:
            with Image.open(sheet.path) as image:
                previews.append((layout_preview(image, sheet.boxes), f"{sheet.name} · {len(sheet.boxes)} panels"))
        counts = "; ".join(f"{sheet.name}: {len(sheet.boxes)}" for sheet in sheets)
        return previews, f"Ready: {len(sheets)} collages, {sum(len(s.boxes) for s in sheets)} panels. {counts}"
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def process_lock_batch(files, quality, sharpen, progress=gr.Progress()):
    try:
        upscaler = None
        if str(quality).startswith("AI"):
            from src.ai_upscale import ai_super_resolve
            upscaler = ai_super_resolve
        return export_lock_batch(files, storage, quality, float(sharpen), upscaler, progress)
    except Exception as exc:
        raise gr.Error(friendly_error(exc, debug=True)) from exc


def build_space() -> gr.Blocks:
    with gr.Blocks(title="Aarij AI Image Studio", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Aarij AI Image Studio\nProduct reference generation · Smart-lock batch extraction · AI restoration")
        gr.Markdown("AI jobs use the hosting account's available GPU quota or credits. Download results before the app stops or restarts.")

        with gr.Tab("High-Quality Generator"):
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(label="Describe your image", lines=5,
                                        placeholder="A premium green bean bag in a modern Indian living room, realistic scale...")
                    reference = gr.Image(label="Optional reference image", type="pil")
                    product_type = gr.Dropdown(list(PRODUCT_DETAILS), value="Smart door locks", label="Reference product type")
                    reference_strength = gr.Slider(
                        0.15, 0.95, 0.30, step=0.05,
                        label="Reference recreation strength (lower preserves more)",
                    )
                    product_lock = gr.Checkbox(
                        True, label="Product Detail Lock (reference guidance, not a guarantee)",
                    )
                    gr.Markdown("For smart locks, use one extracted panel as the reference. Handles and tall bodies are fitted without cropping. AI may still alter text or features.")
                    style = gr.Dropdown(list(STYLES), value="Ecommerce Product", label="Style")
                    with gr.Row():
                        width = gr.Dropdown([768, 1024, 1280], value=1024, label="Width")
                        height = gr.Dropdown([768, 1024, 1280], value=1024, label="Height")
                    count = gr.Slider(1, 4, 1, step=1, label="Number of images")
                    with gr.Accordion("Advanced", open=False):
                        seed = gr.Number(0, precision=0, label="Seed")
                        randomize = gr.Checkbox(True, label="Use random seed")
                        steps = gr.Slider(1, 4, 4, step=1, label="Quality steps")
                    button = gr.Button("Generate High-Quality Images", variant="primary")
                with gr.Column():
                    gallery = gr.Gallery(label="Generated images", columns=2, height="auto")
                    downloads = gr.File(label="Download images", file_count="multiple")
                    seeds = gr.Textbox(label="Seeds used", interactive=False)
                    status = gr.Textbox(label="Status", interactive=False)
            button.click(generate_flux, [prompt, reference, reference_strength, product_lock, style, width, height, count, seed, randomize, steps, product_type],
                         [gallery, downloads, seeds, status], concurrency_id="gpu", concurrency_limit=1)

        with gr.Tab("Collage Crop & Enhance"):
            gr.Markdown(
                "Upload a multi-panel image. White divider lines are detected automatically, "
                "and every panel is exported separately. High-clarity enlargement improves edges, but a small source panel "
                "cannot contain the same real detail as an original 4K photograph."
            )
            collage = gr.Image(label="Collage / contact sheet", type="pil")
            panel_layout = gr.Dropdown(
                ["General Auto", "Smart locks (6 / 7 panels)"], value="General Auto", label="Collage layout",
            )
            with gr.Row():
                panel_scale = gr.Dropdown([1, 2, 3, 4], value=2, label="Upscale factor")
                panel_quality = gr.Dropdown(
                    ENHANCEMENT_MODES,
                    value="AI Product Enhance (4x)", label="Enhancement mode",
                )
                panel_sharpen = gr.Slider(0, 2, 1, step=0.1, label="Detail sharpening")
            split_button = gr.Button("Extract All Panels", variant="primary")
            panel_gallery = gr.Gallery(label="Separate enhanced panels", columns=3, height="auto")
            panel_files = gr.File(label="Download all panels", file_count="multiple")
            panel_zip = gr.File(label="Download All Panels (ZIP)")
            panel_status = gr.Textbox(label="Status", interactive=False)
            split_button.click(split_collage, [collage, panel_scale, panel_quality, panel_sharpen, panel_layout],
                               [panel_gallery, panel_files, panel_zip, panel_status], concurrency_id="gpu", concurrency_limit=1)

        with gr.Tab("Smart Lock Batch"):
            gr.Markdown(
                "Upload up to **10 smart-lock collages** together. Supports six-panel boards, unequal columns, "
                "and the V60 seven-panel layout with its wide packaging image. Check the numbered crop preview first. "
                "AI enhancement is optional and may alter fine text; Pixel-Safe mode only resizes/sharpens. "
                "4K is an export size, not a guarantee of recovered detail."
            )
            lock_files = gr.File(label="Upload smart-lock collages (up to 10)", file_count="multiple", file_types=["image"], type="filepath")
            lock_check = gr.Button("Check panel layouts")
            lock_preview = gr.Gallery(label="Numbered crop previews", columns=2, height="auto")
            lock_check_status = gr.Textbox(label="Layout check", interactive=False)
            with gr.Row():
                lock_quality = gr.Dropdown(ENHANCEMENT_MODES, value="AI Product Enhance (4x)", label="Batch enhancement mode")
                lock_sharpen = gr.Slider(0, 2, .5, step=.1, label="Batch detail sharpening")
            lock_start = gr.Button("Enhance & Download All Lock Panels", variant="primary")
            lock_gallery = gr.Gallery(label="Enhanced lock panels (thumbnail previews)", columns=3, height="auto")
            lock_downloads = gr.File(label="Full-resolution individual images", file_count="multiple")
            lock_zip = gr.File(label="Download all models (ZIP)")
            lock_status = gr.Textbox(label="Batch status", interactive=False)
            lock_check.click(preview_lock_batch, [lock_files], [lock_preview, lock_check_status])
            lock_start.click(process_lock_batch, [lock_files, lock_quality, lock_sharpen],
                             [lock_gallery, lock_downloads, lock_zip, lock_status], concurrency_id="gpu", concurrency_limit=1)
            lock_files.change(lambda: ([], [], None, "", [], ""), [],
                              [lock_gallery, lock_downloads, lock_zip, lock_status, lock_preview, lock_check_status], queue=False)

        with gr.Tab("White Background"):
            gr.Markdown("This mode does not regenerate the product. It removes the background and preserves the uploaded product pixels.")
            with gr.Row():
                source = gr.Image(label="Upload Product", type="pil")
                result = gr.Image(label="Marketplace Result", type="pil")
            canvas = gr.Dropdown([1000, 1600, 2000, 2400], value=2000, label="Square canvas size")
            coverage = gr.Slider(20, 95, 85, step=1, label="Product size (%)")
            padding = gr.Slider(0, 300, 40, step=5, label="Extra padding")
            shadow = gr.Checkbox(False, label="Natural shadow")
            strength = gr.Slider(0.05, 1, 0.3, step=0.05, label="Shadow strength")
            white_button = gr.Button("Create White Background", variant="primary")
            white_file = gr.File(label="Download PNG")
            white_status = gr.Textbox(label="Status", interactive=False)
            white_button.click(white_background, [source, canvas, coverage, padding, shadow, strength],
                               [result, white_file, white_status])

        with gr.Tab("About"):
            gr.Markdown(
                "**Model:** `black-forest-labs/FLUX.1-schnell` · Apache-2.0 model license. "
                "AI-generated product details are not guaranteed to match a real product. "
                "Use White Background for marketplace main-image accuracy."
            )
    return demo


demo = build_space()

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("PORT", "7860")),
        root_path=os.getenv("GRADIO_ROOT_PATH") or None,
        allowed_paths=[str(OUTPUT_ROOT)],
    )

