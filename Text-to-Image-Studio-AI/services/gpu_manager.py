"""Detects the compute device and checks whether a request will fit in memory.

PyTorch is imported lazily so the rest of the app (and the tests) still work on
machines where torch is not installed.
"""
from __future__ import annotations

import gc
from config.settings import MODEL_REGISTRY


def _torch():
    try:
        import torch
        return torch
    except Exception:  # torch missing or broken CUDA install
        return None


def get_device_info() -> dict:
    """Return a dict describing the device. Never assumes CUDA exists."""
    torch = _torch()
    info = {"torch_installed": torch is not None, "cuda": False, "device": "cpu", "name": "CPU",
            "total_gb": 0.0, "free_gb": 0.0, "used_gb": 0.0, "bf16": False,
            "torch_version": None, "cuda_version": None, "compute_capability": None}
    if torch is None:
        return info
    info["torch_version"] = torch.__version__
    info["cuda_version"] = torch.version.cuda
    try:
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            props = torch.cuda.get_device_properties(0)
            info.update(cuda=True, device="cuda", name=props.name, total_gb=total / 1024 ** 3,
                        free_gb=free / 1024 ** 3, used_gb=(total - free) / 1024 ** 3,
                        bf16=bool(torch.cuda.is_bf16_supported()),
                        compute_capability=f"{props.major}.{props.minor}")
    except Exception as exc:  # driver problems should not crash the UI
        info["error"] = str(exc)
    return info


def device_label(info: dict | None = None) -> str:
    info = info or get_device_info()
    if info["cuda"]:
        short = info["name"].replace("NVIDIA ", "").replace("Tesla ", "").replace("GeForce ", "")
        return f"GPU Connected ({short})"
    return "CPU only"


def clear_memory() -> None:
    gc.collect()
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def estimate_generation_vram_gb(family: str, width: int, height: int, batch: int, offload: bool = False) -> float:
    """Rough heuristic (not a measurement): weights + activations that scale with pixels and batch."""
    weights = MODEL_REGISTRY_WEIGHTS.get(family, 3.0)
    megapixels = width * height / 1e6
    per_mp = 3.5 if family == "sdxl" else 2.0
    resident = 1.5 if offload else weights
    return resident + per_mp * megapixels * max(1, batch) + 0.5


MODEL_REGISTRY_WEIGHTS = {v["family"]: v["weights_gb"] for v in MODEL_REGISTRY.values()}


def check_generation_request(family: str, width: int, height: int, batch: int, info: dict | None = None):
    """Return (level, message) where level is 'ok', 'warn' or 'error'."""
    info = info or get_device_info()
    if not info["cuda"]:
        if family == "sdxl":
            return "error", "SDXL needs a CUDA GPU. Switch the runtime to a GPU (Colab: Runtime > Change runtime type) or use Stable Diffusion 1.5."
        return "warn", "No GPU detected: generation runs on the CPU and can take several minutes per image."
    budget = info["total_gb"] * 0.92
    need = estimate_generation_vram_gb(family, width, height, batch)
    need_offload = estimate_generation_vram_gb(family, width, height, batch, offload=True)
    if need <= budget:
        return "ok", f"Estimated {need:.1f} GB of {info['total_gb']:.1f} GB VRAM."
    if need_offload <= budget:
        return "warn", (f"Estimated {need:.1f} GB exceeds {info['total_gb']:.1f} GB VRAM. "
                        "CPU offload will be used (slower but works).")
    return "error", (f"This request needs about {need_offload:.1f} GB even with CPU offload, "
                     f"but the GPU has {info['total_gb']:.1f} GB. Lower the resolution or batch size.")


def needs_offload(family: str, width: int, height: int, batch: int, info: dict, mode: str = "auto") -> bool:
    if not info["cuda"] or mode == "never":
        return False
    if mode == "always":
        return True
    return estimate_generation_vram_gb(family, width, height, batch) > info["total_gb"] * 0.92


def recommended_profile(info: dict | None = None) -> dict:
    """Suggest settings for the detected GPU (the notebook prints this)."""
    info = info or get_device_info()
    if not info["cuda"]:
        return {"label": "CPU only", "default_model": "Stable Diffusion 1.5", "max_batch": 1,
                "lora_training": False, "note": "Use a GPU runtime for real work."}
    gb = info["total_gb"]
    if gb >= 22:
        return {"label": "High-end GPU", "default_model": "Stable Diffusion XL Base 1.0", "max_batch": 4,
                "lora_training": True, "note": "SDXL at 1024x1024 and LoRA training run comfortably."}
    if gb >= 14:
        return {"label": "Colab T4 class (15 GB)", "default_model": "Stable Diffusion XL Base 1.0", "max_batch": 2,
                "lora_training": True, "note": "SDXL works (~30-60 s per image); SD 1.5 is much faster. LoRA training: batch size 1."}
    if gb >= 8:
        return {"label": "Mid-range GPU", "default_model": "Stable Diffusion 1.5", "max_batch": 2,
                "lora_training": True, "note": "SD 1.5 recommended; SDXL only with CPU offload."}
    return {"label": "Low-VRAM GPU", "default_model": "Stable Diffusion 1.5", "max_batch": 1,
            "lora_training": gb >= 6, "note": "Keep resolution at 512x512."}
