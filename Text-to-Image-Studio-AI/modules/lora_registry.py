"""LoRA adapter bookkeeping that needs no GPU: listing, compatibility checks, rename/delete/export."""
from __future__ import annotations

import io
import json
import re
import shutil
import zipfile
from pathlib import Path

from config import settings


def adapters_dir() -> Path:
    path = settings.dir_path("lora_models")
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_name(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_")
    if not clean:
        raise ValueError("Use letters, numbers, dashes or underscores in the adapter name.")
    return clean[:60]


def read_meta(adapter_dir: str | Path) -> dict:
    path = Path(adapter_dir) / "adapter_meta.json"
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except (OSError, ValueError):
        return {}


def detect_family_from_weights(adapter_dir: str | Path) -> str | None:
    """Inspect the LoRA tensors: cross-attention K/V input width is 768 for SD 1.5 and 2048 for SDXL."""
    files = sorted(Path(adapter_dir).glob("*.safetensors"))
    if not files:
        return None
    try:
        from safetensors import safe_open
        with safe_open(str(files[0]), framework="numpy") as f:
            keys = list(f.keys())
            if any("text_encoder_2" in k or "lora_te2" in k for k in keys):
                return "sdxl"
            for k in keys:
                if "attn2" in k and "to_k" in k and ("lora_A" in k or "lora_down" in k):
                    width = f.get_slice(k).get_shape()[1]
                    return {768: "sd15", 2048: "sdxl"}.get(width)
    except Exception:
        return None
    return None


def check_compatibility(adapter_dir: str | Path, model_key: str) -> tuple[bool, str]:
    """A LoRA trained for SDXL must never be loaded into SD 1.5 (or vice versa)."""
    target = settings.MODEL_REGISTRY[model_key]["family"]
    meta = read_meta(adapter_dir)
    family = meta.get("family") or detect_family_from_weights(adapter_dir)
    if family is None:
        return False, "Could not determine which base model this adapter was trained for, so it is not loaded."
    if family != target:
        names = {v["family"]: k for k, v in settings.MODEL_REGISTRY.items()}
        return False, f"This adapter was trained for {names.get(family, family)} and cannot be used with {model_key}."
    return True, "Compatible."


def list_adapters() -> list[dict]:
    items = []
    for d in sorted(p for p in adapters_dir().iterdir() if p.is_dir()):
        if not list(d.glob("*.safetensors")):
            continue
        meta = read_meta(d)
        items.append({"name": d.name, "path": str(d), "family": meta.get("family") or detect_family_from_weights(d),
                      "base_model": meta.get("base_model_key", "unknown"), "rank": meta.get("rank"),
                      "steps": meta.get("steps"), "final_loss": meta.get("final_loss"),
                      "created": meta.get("created", ""), "meta": meta})
    return items


def rename_adapter(old: str, new: str) -> str:
    new = sanitize_name(new)
    src, dst = adapters_dir() / old, adapters_dir() / new
    if not src.exists():
        raise FileNotFoundError(old)
    if dst.exists():
        raise ValueError("An adapter with that name already exists.")
    src.rename(dst)
    meta = read_meta(dst)
    if meta:
        meta["adapter_name"] = new
        (dst / "adapter_meta.json").write_text(json.dumps(meta, indent=2))
    return new


def delete_adapter(name: str) -> None:
    shutil.rmtree(adapters_dir() / name, ignore_errors=True)


def export_adapter_zip(name: str) -> bytes:
    buf = io.BytesIO()
    root = adapters_dir() / name
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in root.rglob("*"):
            if f.is_file() and f.name != "resume.pt":
                zf.write(f, f.relative_to(root))
    return buf.getvalue()
