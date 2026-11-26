"""Central configuration: storage paths, model registry, defaults and user settings.

Everything that other files need to agree on (folder names, model repos, scheduler
names, style presets) lives here so there is exactly one place to change it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

APP_NAME = "Text-to-Image Studio AI"
APP_VERSION = "2.0"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SUBDIRS = ("generated_images", "lora_models", "checkpoints", "datasets", "reports", "exports", "cache")


def resolve_data_root() -> Path:
    """Where images, datasets, checkpoints and the SQLite DB are stored.

    Priority: STUDIO_DATA_DIR env var -> Google Drive (when mounted in Colab) -> ./data
    Using Drive means everything survives a Colab runtime disconnect.
    """
    env = os.environ.get("STUDIO_DATA_DIR")
    if env:
        return Path(env).expanduser()
    drive = Path("/content/drive/MyDrive")
    if drive.exists():
        return drive / "TextToImageStudio"
    return PROJECT_ROOT / "data"


DATA_ROOT: Path = resolve_data_root()
DB_PATH: Path = DATA_ROOT / "studio.db"
SETTINGS_PATH: Path = DATA_ROOT / "settings.json"


def set_data_root(path: str | Path) -> None:
    """Re-point all storage at another folder (used by tests and the Settings page)."""
    global DATA_ROOT, DB_PATH, SETTINGS_PATH
    DATA_ROOT = Path(path)
    DB_PATH = DATA_ROOT / "studio.db"
    SETTINGS_PATH = DATA_ROOT / "settings.json"


def dir_path(name: str) -> Path:
    return DATA_ROOT / name


def ensure_dirs() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    for sub in SUBDIRS:
        (DATA_ROOT / sub).mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- models
MODEL_REGISTRY = {
    "Stable Diffusion 1.5": {
        "repo": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "family": "sd15",
        "version": "1.5",
        "native": (512, 512),
        "resolutions": [(512, 512), (512, 768), (768, 512), (640, 640), (768, 768)],
        "weights_gb": 2.1,
        "min_vram_gb": 4.0,
        "max_side": 1024,
        "memory_note": "about 4 GB VRAM in fp16 (works well on a T4)",
    },
    "Stable Diffusion XL Base 1.0": {
        "repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "family": "sdxl",
        "version": "1.0",
        "native": (1024, 1024),
        "resolutions": [(1024, 1024), (1152, 896), (896, 1152), (1216, 832), (832, 1216), (1344, 768), (768, 1344)],
        "weights_gb": 6.9,
        "min_vram_gb": 10.0,
        "max_side": 1536,
        "memory_note": "10 GB+ VRAM recommended; 6 GB+ works with CPU offload (slower)",
    },
}

SCHEDULER_NAMES = ["DPM++ 2M Karras", "DPM++ 2M", "Euler a", "Euler", "DDIM", "UniPC", "LMS"]
ASPECT_RATIOS = {"1:1": (1, 1), "16:9": (16, 9), "9:16": (9, 16), "4:3": (4, 3), "3:4": (3, 4), "3:2": (3, 2), "2:3": (2, 3)}

HIGH_QUALITY = {"steps": 35, "guidance": 7.5, "scheduler": "DPM++ 2M Karras"}

STYLE_PRESETS = {
    "Realistic": "photorealistic, natural lighting, 85mm lens, shallow depth of field, ultra detailed, sharp focus",
    "Anime": "anime style, vibrant colors, clean line art, cel shading, studio quality illustration",
    "Fantasy": "epic fantasy art, magical atmosphere, dramatic lighting, intricate details, concept art",
    "Nature": "nature photography, golden hour light, lush environment, high dynamic range, detailed textures",
    "Cinematic": "cinematic composition, volumetric lighting, film grain, anamorphic lens, moody color grading",
    "Sci-Fi": "futuristic sci-fi scene, neon accents, advanced technology, highly detailed, atmospheric",
    "Watercolor": "soft watercolor painting, flowing pigments, textured paper, gentle gradients",
    "3D Render": "3D render, octane render, smooth materials, global illumination, high detail",
}

DEFAULT_SETTINGS = {
    "theme": "dark",
    "accent": "#8b5cf6",
    "default_model": "auto",
    "default_steps": 30,
    "default_guidance": 7.5,
    "default_scheduler": "DPM++ 2M Karras",
    "cpu_offload": "auto",          # auto | always | never
    "attention_slicing": True,
    "safety_checker": False,
    "cache_models": True,
    "retention_days": 0,            # 0 = keep forever
}


def load_settings() -> dict:
    data = dict(DEFAULT_SETTINGS)
    try:
        if SETTINGS_PATH.exists():
            data.update(json.loads(SETTINGS_PATH.read_text()))
    except (OSError, ValueError):
        pass
    return data


def save_settings(values: dict) -> None:
    ensure_dirs()
    merged = load_settings()
    merged.update(values)
    SETTINGS_PATH.write_text(json.dumps(merged, indent=2))
