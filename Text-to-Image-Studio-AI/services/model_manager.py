"""Loads and caches diffusion pipelines (one at a time, to protect GPU memory).

Heavy imports (torch, diffusers) happen inside functions so this file can be imported
anywhere. Switching models unloads the previous one first.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from config import settings
from services import gpu_manager

SCHEDULERS = {
    "DPM++ 2M": ("DPMSolverMultistepScheduler", {}),
    "DPM++ 2M Karras": ("DPMSolverMultistepScheduler", {"use_karras_sigmas": True}),
    "Euler": ("EulerDiscreteScheduler", {}),
    "Euler a": ("EulerAncestralDiscreteScheduler", {}),
    "DDIM": ("DDIMScheduler", {}),
    "UniPC": ("UniPCMultistepScheduler", {}),
    "LMS": ("LMSDiscreteScheduler", {}),
}

_STATUS_CACHE: dict[str, tuple[float, dict]] = {}


def model_status(model_key: str) -> dict:
    """'cached' (already downloaded), 'downloadable' (reachable on the Hugging Face Hub) or 'unavailable'."""
    now = time.time()
    hit = _STATUS_CACHE.get(model_key)
    if hit and now - hit[0] < 90:
        return hit[1]
    repo = settings.MODEL_REGISTRY[model_key]["repo"]
    result = {"state": "unavailable", "detail": "Not downloaded and the Hugging Face Hub is unreachable."}
    try:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(repo, "model_index.json")
        if isinstance(cached, str):
            result = {"state": "cached", "detail": "Downloaded and ready."}
        else:
            from huggingface_hub import HfApi
            HfApi().model_info(repo, timeout=6)
            result = {"state": "downloadable", "detail": "Will download on first use (several GB)."}
    except Exception as exc:
        if result["state"] == "unavailable":
            result["detail"] = f"Not downloaded and Hub check failed ({type(exc).__name__})."
    _STATUS_CACHE[model_key] = (now, result)
    return result


class ModelManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._pipe = None
        self._cache_key: tuple | None = None
        self._base_scheduler_config = None
        self._lora_dir: str | None = None
        self.offloaded = False

    @property
    def loaded(self) -> str | None:
        return self._cache_key[0] if self._cache_key else None

    def unload(self) -> None:
        with self._lock:
            self._pipe = None
            self._cache_key = None
            self._lora_dir = None
            self._base_scheduler_config = None
            gpu_manager.clear_memory()

    def get_pipeline(self, model_key: str, mode: str = "txt2img", width: int = 512, height: int = 512,
                     batch: int = 1, status_cb=None):
        """Return a ready pipeline for `model_key` in 'txt2img' or 'img2img' mode (reusing the cache)."""
        import torch
        from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image

        with self._lock:
            key = (model_key, mode)
            if self._pipe is not None and self._cache_key == key:
                return self._pipe
            if self._pipe is not None and self._cache_key and self._cache_key[0] == model_key and not getattr(self, "offloaded", False):
                # Same model, other mode (e.g. Variation = img2img): share the weights already on the GPU instead of
                # unloading and re-reading several GB from disk, which is what made Variation take minutes.
                try:
                    cls = AutoPipelineForImage2Image if mode == "img2img" else AutoPipelineForText2Image
                    pipe = cls.from_pipe(self._pipe)
                    pipe.set_progress_bar_config(disable=True)
                    self._pipe, self._cache_key = pipe, key
                    return pipe
                except Exception:
                    pass                                   # fall back to a normal reload below
            self.unload()
            cfg = settings.MODEL_REGISTRY[model_key]
            prefs = settings.load_settings()
            info = gpu_manager.get_device_info()
            if status_cb:
                status_cb(f"Loading {model_key} (the first load downloads the weights and can take minutes)...")
            cls = AutoPipelineForImage2Image if mode == "img2img" else AutoPipelineForText2Image
            kwargs = {"torch_dtype": torch.float16 if info["cuda"] else torch.float32, "use_safetensors": True}
            if cfg["family"] == "sd15" and not prefs["safety_checker"]:
                kwargs.update(safety_checker=None, requires_safety_checker=False)
            try:
                pipe = cls.from_pretrained(cfg["repo"], **({"variant": "fp16"} if info["cuda"] else {}), **kwargs)
            except Exception:
                pipe = cls.from_pretrained(cfg["repo"], **kwargs)  # repo without an fp16 variant
            offload = gpu_manager.needs_offload(cfg["family"], width, height, batch, info, prefs["cpu_offload"])
            if info["cuda"]:
                pipe.enable_model_cpu_offload() if offload else pipe.to("cuda")
            if prefs["attention_slicing"] and (cfg["family"] == "sd15" or offload):
                try:
                    pipe.enable_attention_slicing()
                except Exception:
                    pass
            # Newer diffusers removed pipe.enable_vae_slicing(); the VAE method works in every version.
            try:
                pipe.vae.enable_slicing()
            except Exception:
                pass
            pipe.set_progress_bar_config(disable=True)
            self.offloaded = offload
            self._pipe, self._cache_key = pipe, key
            self._base_scheduler_config = pipe.scheduler.config
            return pipe

    def set_scheduler(self, pipe, name: str) -> None:
        import diffusers
        cls_name, extra = SCHEDULERS.get(name, SCHEDULERS["DPM++ 2M Karras"])
        cls = getattr(diffusers, cls_name)
        pipe.scheduler = cls.from_config(self._base_scheduler_config or pipe.scheduler.config, **extra)

    # ---- LoRA ----------------------------------------------------------------------
    def apply_lora(self, pipe, adapter_dir: str | Path, scale: float) -> None:
        adapter_dir = Path(adapter_dir)
        if self._lora_dir != str(adapter_dir):
            self.clear_lora(pipe)
            files = sorted(adapter_dir.glob("*.safetensors"))
            if not files:
                raise FileNotFoundError(f"No .safetensors LoRA weights found in {adapter_dir}")
            pipe.load_lora_weights(str(adapter_dir), weight_name=files[0].name, adapter_name="studio")
            self._lora_dir = str(adapter_dir)
        pipe.set_adapters(["studio"], adapter_weights=[float(scale)])

    def clear_lora(self, pipe=None) -> None:
        pipe = pipe or self._pipe
        if pipe is not None and self._lora_dir:
            try:
                pipe.unload_lora_weights()
            except Exception:
                pass
        self._lora_dir = None


_MANAGER = ModelManager()


def get_model_manager() -> ModelManager:
    return _MANAGER
