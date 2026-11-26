"""Stable Diffusion generation backend (Module 1).

Pipeline (matches the spec): validate -> check device/memory -> load model -> encode prompts
-> run diffusion -> decode latents -> return images -> save images + metadata.
Prompt encoding, the denoising loop and VAE decoding all happen inside the diffusers pipeline call.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from PIL import Image, ImageFilter

from config import settings
from config.settings import ASPECT_RATIOS, HIGH_QUALITY, MODEL_REGISTRY, SCHEDULER_NAMES
from services import gpu_manager, storage_manager
from services.model_manager import get_model_manager

SCHEDULER_FALLBACKS = ("DPM++ 2M", "Euler", "DDIM")   # tried in this order if the chosen scheduler crashes with an IndexError
_BROKEN_SCHEDULERS: set = set()                         # schedulers that crashed here but had a working replacement: skipped from then on
from modules import lora_registry


class GenerationError(Exception):
    """A problem the user can act on (bad settings, out of memory, missing model...)."""


@dataclass
class GenerationRequest:
    prompt: str
    model_key: str = "Stable Diffusion 1.5"
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 30
    guidance: float = 7.5
    seed: int = -1
    scheduler: str = "DPM++ 2M Karras"
    batch_size: int = 1
    lora_dir: str | None = None
    lora_scale: float = 1.0
    init_image: Image.Image | None = None
    strength: float = 0.6
    high_quality: bool = False
    save: bool = True
    gen_type: str = "txt2img"
    extra: dict = field(default_factory=dict)


@dataclass
class GenerationResult:
    images: list
    records: list
    seeds: list
    elapsed: float
    peak_gb: float | None
    warnings: list


def parse_resolution(text: str) -> tuple[int, int]:
    w, h = text.replace("x", "×").split("×")
    return int(w.strip()), int(h.strip())


def aspect_dims(ratio: str, base: int) -> tuple[int, int]:
    """Width/height for an aspect ratio keeping roughly base*base pixels, rounded to multiples of 8."""
    rw, rh = ASPECT_RATIOS[ratio]
    area = base * base
    w = round(math.sqrt(area * rw / rh) / 8) * 8
    h = round(math.sqrt(area * rh / rw) / 8) * 8
    return w, h


def validate_request(req: GenerationRequest) -> list[str]:
    errors = []
    if not req.prompt.strip():
        errors.append("Enter a prompt.")
    if len(req.prompt) > 1000:
        errors.append("The prompt is longer than 1000 characters.")
    if len(req.negative_prompt) > 500:
        errors.append("The negative prompt is longer than 500 characters.")
    if req.model_key not in MODEL_REGISTRY:
        errors.append("Unknown model.")
        return errors
    max_side = MODEL_REGISTRY[req.model_key]["max_side"]
    for label, v in (("Width", req.width), ("Height", req.height)):
        if v % 8 != 0:
            errors.append(f"{label} must be a multiple of 8.")
        if not 256 <= v <= max_side:
            errors.append(f"{label} must be between 256 and {max_side} for {req.model_key}.")
    if not 1 <= req.steps <= 150:
        errors.append("Steps must be between 1 and 150.")
    if not 0 <= req.guidance <= 30:
        errors.append("Guidance scale must be between 0 and 30.")
    if not 1 <= req.batch_size <= 8:
        errors.append("Batch size must be between 1 and 8.")
    if not -1 <= req.seed < 2 ** 32:
        errors.append("Seed must be -1 (random) or between 0 and 4294967295.")
    if req.scheduler not in SCHEDULER_NAMES:
        errors.append("Unknown scheduler.")
    if req.init_image is not None and not 0.05 <= req.strength <= 1.0:
        errors.append("Strength must be between 0.05 and 1.0.")
    if req.lora_dir:
        ok, msg = lora_registry.check_compatibility(req.lora_dir, req.model_key)
        if not ok:
            errors.append(msg)
    return errors


def apply_high_quality(req: GenerationRequest) -> GenerationRequest:
    """'High Quality Mode' = a fixed, tested configuration (see config.settings.HIGH_QUALITY)."""
    req.steps = max(req.steps, HIGH_QUALITY["steps"])
    req.guidance = HIGH_QUALITY["guidance"]
    req.scheduler = HIGH_QUALITY["scheduler"]
    return req


def resolve_seed(seed: int) -> int:
    return random.randint(0, 2 ** 32 - 1) if seed < 0 else int(seed)


def generate(req: GenerationRequest, progress_cb=None, status_cb=None) -> GenerationResult:
    """Run generation. `progress_cb(step, total, elapsed)` reports real denoising steps."""
    import torch

    if req.high_quality:
        apply_high_quality(req)
    errors = validate_request(req)
    if errors:
        raise GenerationError(" ".join(errors))
    cfg = MODEL_REGISTRY[req.model_key]
    info = gpu_manager.get_device_info()
    level, message = gpu_manager.check_generation_request(cfg["family"], req.width, req.height, req.batch_size, info)
    if level == "error":
        raise GenerationError(message)
    warnings = [message] if level == "warn" else []
    if not info["torch_installed"]:
        raise GenerationError("PyTorch is not installed. Install the requirements first.")

    mm = get_model_manager()
    mode = "img2img" if req.init_image is not None else "txt2img"
    started = time.time()
    try:
        pipe = mm.get_pipeline(req.model_key, mode, req.width, req.height, req.batch_size, status_cb)
        if req.lora_dir:
            mm.apply_lora(pipe, req.lora_dir, req.lora_scale)
        else:
            mm.clear_lora(pipe)
        chosen = req.scheduler
        if chosen in _BROKEN_SCHEDULERS:
            chosen = next((n for n in SCHEDULER_FALLBACKS if n not in _BROKEN_SCHEDULERS), "DDIM")
            warnings.append(f"Scheduler '{req.scheduler}' is known to fail with this diffusers version; using '{chosen}'.")
        mm.set_scheduler(pipe, chosen)
        if status_cb:
            status_cb("Running diffusion...")

        base_seed = resolve_seed(req.seed)
        seeds = [(base_seed + i) % 2 ** 32 for i in range(req.batch_size)]
        generators = [torch.Generator("cpu").manual_seed(s) for s in seeds]
        total_steps = req.steps if mode == "txt2img" else max(1, min(int(req.steps * req.strength), req.steps))
        t0 = time.time()

        def on_step(_pipe, step, _timestep, callback_kwargs):
            if progress_cb:
                progress_cb(step + 1, total_steps, time.time() - t0)
            return callback_kwargs

        kwargs = dict(prompt=[req.prompt] * req.batch_size, num_inference_steps=req.steps, guidance_scale=req.guidance,
                      generator=generators, callback_on_step_end=on_step)
        if req.negative_prompt.strip():
            kwargs["negative_prompt"] = [req.negative_prompt] * req.batch_size
        if mode == "txt2img":
            kwargs.update(width=req.width, height=req.height)
        else:
            kwargs.update(image=req.init_image.convert("RGB").resize((req.width, req.height), Image.LANCZOS), strength=req.strength)

        if info["cuda"]:
            torch.cuda.reset_peak_memory_stats()
        used_scheduler = chosen
        try:
            with torch.inference_mode():
                images = pipe(**kwargs).images
        except IndexError as first_error:
            # Some diffusers versions fail inside multistep/Karras schedulers ("index N is out of bounds for dimension 0 with size N").
            # Retry the SAME seed with simpler schedulers so the run still completes, and say so.
            images, failed = None, [chosen]
            for alt in [n for n in SCHEDULER_FALLBACKS if n != chosen and n not in _BROKEN_SCHEDULERS]:
                mm.set_scheduler(pipe, alt)
                kwargs["generator"] = [torch.Generator("cpu").manual_seed(sd) for sd in seeds]
                try:
                    with torch.inference_mode():
                        images = pipe(**kwargs).images
                    used_scheduler = alt
                    _BROKEN_SCHEDULERS.update(failed)
                    warnings.append(f"Scheduler '{chosen}' failed with this diffusers version ({first_error}); used '{alt}' instead.")
                    break
                except IndexError:
                    failed.append(alt)
            if images is None:
                raise
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3 if info["cuda"] else None
    except torch.cuda.OutOfMemoryError:
        mm.unload()
        raise GenerationError("The GPU ran out of memory. Lower the resolution or batch size, or choose Stable Diffusion 1.5.")
    except GenerationError:
        raise
    except Exception as exc:
        raise GenerationError(f"{type(exc).__name__}: {exc}") from exc

    elapsed = time.time() - started
    for img in images:
        if float(np.asarray(img.convert("L").resize((16, 16))).mean()) < 2.0:
            warnings.append("An output image is completely black (can happen with fp16 on some GPUs). Try another seed or scheduler.")
            break

    records = []
    for i, img in enumerate(images):
        meta = dict(prompt=req.prompt, negative_prompt=req.negative_prompt, model=req.model_key, scheduler=used_scheduler,
                    steps=req.steps, guidance=req.guidance, seed=seeds[i], gen_type=req.gen_type, elapsed=round(elapsed, 2),
                    lora=Path(req.lora_dir).name if req.lora_dir else "", lora_scale=req.lora_scale if req.lora_dir else None,
                    extra={"peak_gb": peak, "high_quality": req.high_quality, "batch_index": i, "offload": mm.offloaded, **req.extra})
        if req.save:
            if status_cb:
                status_cb("Saving image and metadata...")
            records.append(storage_manager.save_generated_image(img, meta))
        else:
            records.append({**meta, "id": None, "width": img.width, "height": img.height, "favorite": 0, "path": None})
    storage_manager.add_prompt_history(req.prompt, req.negative_prompt)
    return GenerationResult(images, records, seeds, elapsed, peak, warnings)


# --------------------------------------------------------------------------- upscaling
def upscale_lanczos(img: Image.Image, factor: int = 2) -> Image.Image:
    """Fast classical 2x/4x upscale with light sharpening (no new detail is invented)."""
    big = img.resize((img.width * factor, img.height * factor), Image.LANCZOS)
    return big.filter(ImageFilter.UnsharpMask(radius=1.6, percent=60, threshold=2))


def upscale_ai(img: Image.Image, prompt: str = "", status_cb=None) -> tuple[Image.Image, str]:
    """AI x4 upscale with Stable Diffusion's upscaler. The input is shrunk if it would not fit in VRAM."""
    import torch
    from diffusers import StableDiffusionUpscalePipeline

    info = gpu_manager.get_device_info()
    if not info["cuda"]:
        raise GenerationError("AI upscaling needs a GPU. Use the fast Lanczos upscaler instead.")
    limit = 512 if info["total_gb"] >= 22 else 256 if info["total_gb"] >= 12 else 160
    note = ""
    if max(img.size) > limit:
        scale = limit / max(img.size)
        img = img.resize((max(8, int(img.width * scale) // 8 * 8), max(8, int(img.height * scale) // 8 * 8)), Image.LANCZOS)
        note = f"Input was reduced to {img.width}x{img.height} to fit in GPU memory before the x4 upscale."
    mm = get_model_manager()
    mm.unload()
    try:
        if status_cb:
            status_cb("Loading the x4 upscaler...")
        pipe = StableDiffusionUpscalePipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", torch_dtype=torch.float16).to("cuda")
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        out = pipe(prompt=prompt or "high quality, detailed", image=img, num_inference_steps=20).images[0]
    except torch.cuda.OutOfMemoryError:
        raise GenerationError("Out of GPU memory during upscaling. Try the Lanczos upscaler.")
    finally:
        pipe = None
        gpu_manager.clear_memory()
    return out, note
