# Aarij AI Image Studio

Aarij AI Image Studio is a browser-based image generation and ecommerce product-image toolkit designed for a free Google Colab T4 GPU. It runs downloadable Hugging Face models directly in Python—no paid image API or local high-end computer is required.

> Product accuracy comes first. Use **White Background** for marketplace main images because it preserves the uploaded product pixels. AI-assisted lifestyle, image-to-image, and inpainting workflows can alter colors, logos, stitching, texture, shape, and proportions; always inspect their output.

## What is included

- Fast text-to-image generation with SDXL Turbo
- Optional quality mode with SDXL Base 1.0 and CPU offload
- Lazy loading: only one large pipeline is kept at a time
- White-background product processing with `rembg`, pure-white 2000×2000 output, sizing, padding, and optional shadow
- Amazon, Flipkart, Meesho, lifestyle, fabric-detail, and advertisement presets
- Product image-to-image with Preserve, Balanced, and Creative strengths
- SDXL inpainting with an uploaded black/white mask
- EXIF orientation correction, high-quality resizing, PNG/JPEG helpers, metadata, and unique filenames
- Automatic Google Drive folder creation and permanent saving
- Friendly errors and a Settings action to release GPU memory
- Lightweight tests that do not download model weights

## Fastest start: Google Colab

Open `notebooks/Aarij_AI_Image_Studio_Colab.ipynb` in Colab. Before running it:

1. Choose **Runtime → Change runtime type → T4 GPU**.
2. Run the notebook from top to bottom (or choose **Run all**).
3. Approve Google Drive access when prompted.
4. Open the `gradio.live` link printed by the final cell.

The notebook installs dependencies, mounts Drive, clones this published project, and launches the app. No repository URL editing is required.

Outputs are stored permanently under:

```text
/content/drive/MyDrive/Aarij-AI-Image-Studio/
├── generated-images/
├── uploads/
├── marketplace/
├── lifestyle/
├── white-background/
├── advertisements/
└── temp/
```

Files receive timestamped, numbered names and existing images are never overwritten.

## First use

Models load only when an AI action is first requested, so the first generation is slower while weights download. SDXL Turbo defaults to 512×512, three steps, prompt guidance 0, and one image. Quality mode defaults to 30 steps and uses model CPU offload; switching workflows unloads the previous pipeline first.

No `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `REPLICATE_API_TOKEN`, or `STABILITY_API_KEY` is used. Some Hugging Face models may require accepting their license. If access requires authentication, create a Colab secret named `HF_TOKEN`; never paste it into source code.

## White-background workflow

Open **White Background**, upload the real product, choose a marketplace preset, then select **Create White Background**. The tool removes the original background, resizes and centers the cutout, optionally adds a soft shadow, and places it on pure white. It does not regenerate the product with diffusion.

Background removal is segmentation, so inspect fine hair, transparent materials, and very thin edges. Preserve a copy of the original upload.

## Local development (utilities/UI only)

The supplied local computers are not expected to run SDXL. They can still run tests or start the interface; generation will show a friendly GPU message.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Run lightweight tests with:

```bash
pytest -q
```

## Configuration

Environment variables are centralized in `src/config.py`:

| Variable | Default | Meaning |
|---|---|---|
| `DEBUG` | `false` | Show detailed application errors |
| `DEFAULT_MODEL` | `fast` | Initial `fast` or `quality` choice |
| `OUTPUT_DIRECTORY` | Drive path in Colab, `outputs/` elsewhere | Save root |
| `ENABLE_CPU_OFFLOAD` | `true` | Quality-mode memory reduction |
| `ENABLE_INPAINTING` | `true` | Reserved inpainting feature flag |
| `ENABLE_CONTROLNET` | `false` | Reserved advanced preservation hook |
| `ENABLE_IP_ADAPTER` | `false` | Reserved reference-conditioning hook |
| `ENABLE_LORA` | `false` | Reserved adapter hook |
| `GRADIO_SHARE` | `true` in Colab | Create a public Gradio link |

To change a model ID, edit the model registry in `src/config.py`. Future backends such as FLUX can be added behind the same manager, but are not enabled because their memory needs can make a free T4 unreliable. ControlNet, IP-Adapter, and LoRA extension points are likewise documented but not loaded in version 0.1.

## Troubleshooting

**CUDA out of memory:** Select Fast mode, use 512×512, reduce steps, click **Unload AI Model and Free GPU Memory** in Settings, then retry. Restart the Colab runtime if CUDA remains fragmented. Quality img2img/inpainting can be tight on a T4.

**GPU unavailable:** Confirm the Colab runtime says T4 under `!nvidia-smi`. Free GPU availability is controlled by Google and is not guaranteed.

**Hugging Face access denied:** Open the model page, accept its terms, and provide an `HF_TOKEN` through Colab Secrets if requested.

**Drive not mounted:** Rerun the mount cell and approve access. If Drive is intentionally unavailable, set `OUTPUT_DIRECTORY=/content/Aarij-AI-Image-Studio` before launch.

**`rembg` or ONNX error:** Restart the runtime after dependency installation. The first removal downloads a segmentation model and needs internet access.

**Stop Gradio:** Press the square stop icon beside the launch cell, choose **Runtime → Interrupt execution**, or restart the runtime. Public share links are temporary and expire when the runtime stops.

## Deployment later

Core services are not coupled to Colab. The same `app.py` can run on Kaggle, Hugging Face Spaces, RunPod, Vast.ai, or an NVIDIA server after installing requirements and choosing a persistent `OUTPUT_DIRECTORY`. Keep one worker per GPU unless memory measurements prove otherwise. Do not expose the public app without authentication and upload limits.

## Project map

```text
app.py                    Application entry point
ui/gradio_app.py          Beginner-friendly Gradio interface
src/model_manager.py      Lazy, single-pipeline GPU management
src/generation.py         Text-to-image workflow
src/img2img.py            AI-assisted transformations
src/inpainting.py         Masked editing
src/marketplace.py        Product-preserving white background
src/storage.py            Permanent directories and unique saving
presets/                  Editable JSON presets
notebooks/                One-click-oriented Colab launcher
tests/                    No-model-download unit tests
```

Screenshots can be added under `docs/screenshots/` after the first Colab launch.
