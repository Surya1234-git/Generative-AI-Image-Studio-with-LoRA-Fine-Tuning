"""Import a small COCO captions subset as a Dataset Explorer dataset (Task 4: a public dataset with real descriptions).

COCO has several human-written captions per image, unlike Oxford-102 which only has class labels. This module turns the
official annotation files into the folder layout the Dataset Explorer already understands:

    <dest>/<class>/<image file>     one folder per class (the object category with the largest area, or "coco")
    <dest>/captions.csv             columns filename, caption (one caption per image)
    <dest>/captions_all.json        every caption of every imported image

No GPU or PyTorch is needed. Images are copied from `images_dir` when given, otherwise downloaded from the `coco_url`
stored in the annotation file. The functions take the downloader as a parameter so tests can run without internet.
"""
from __future__ import annotations

import json
import random
import shutil
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Callable

from PIL import Image

from modules import dataset_explorer as dx


def _default_fetch(url: str, timeout: int = 30) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:          # noqa: S310 (official COCO image host)
        return response.read()


def _class_name(name: str) -> str:
    return "_".join(name.strip().lower().split()) or "coco"


def _main_categories(instances_json: str | Path) -> dict[int, str]:
    """image id -> name of the category whose annotations cover the largest total area in that image."""
    data = json.loads(Path(instances_json).read_text(encoding="utf-8"))
    names = {c["id"]: _class_name(c["name"]) for c in data.get("categories", [])}
    area: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for ann in data.get("annotations", []):
        area[ann["image_id"]][ann["category_id"]] += float(ann.get("area", 0.0))
    return {img: names[max(cats, key=cats.get)] for img, cats in area.items() if max(cats, key=cats.get) in names}


def build_coco_subset(captions_json: str | Path, dest: str | Path, images_dir: str | Path | None = None,
                      instances_json: str | Path | None = None, max_images: int = 500, seed: int = 42,
                      caption_index: int = 0, fetch: Callable[[str], bytes] | None = None,
                      progress_cb: Callable[[float], None] | None = None) -> dict:
    """Write a COCO subset to `dest`. Returns a summary: images written, skipped, class counts, caption length."""
    if max_images < 1:
        raise ValueError("max_images must be at least 1.")
    fetch = fetch or _default_fetch
    data = json.loads(Path(captions_json).read_text(encoding="utf-8"))
    captions: dict[int, list[str]] = defaultdict(list)
    for ann in sorted(data.get("annotations", []), key=lambda a: a.get("id", 0)):
        text = " ".join(str(ann.get("caption", "")).split())
        if text:
            captions[ann["image_id"]].append(text)
    images = [im for im in data.get("images", []) if im["id"] in captions]
    if not images:
        raise ValueError("The annotation file has no images with captions.")
    categories = _main_categories(instances_json) if instances_json else {}

    rng = random.Random(seed)
    rng.shuffle(images)                      # a seeded shuffle, so the same subset is chosen every time
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    rows, all_caps, skipped = [], {}, 0
    for k, info in enumerate(images):
        if len(rows) >= max_images:
            break
        cls = categories.get(info["id"], "coco")
        name = info["file_name"]
        target = dest / cls / name
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            local = Path(images_dir) / name if images_dir else None
            if local is not None and local.exists():
                shutil.copyfile(local, target)
            elif images_dir is None and info.get("coco_url"):
                target.write_bytes(fetch(info["coco_url"]))
            else:
                raise FileNotFoundError(name)
            with Image.open(target) as im:       # reject truncated or non-image downloads
                im.verify()
        except Exception:
            target.unlink(missing_ok=True)
            skipped += 1
            continue
        caps = captions[info["id"]]
        rows.append({"filename": str(Path(cls) / name), "caption": caps[min(caption_index, len(caps) - 1)]})
        all_caps[str(Path(cls) / name)] = caps
        if progress_cb:
            progress_cb(min(len(rows) / max_images, 1.0))
    if not rows:
        raise ValueError("No images could be imported. Check images_dir, or the internet connection when downloading.")
    dx.write_captions_csv(dest, rows)
    (dest / "captions_all.json").write_text(json.dumps(all_caps, indent=1), encoding="utf-8")
    lengths = [len(r["caption"].split()) for r in rows]
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[Path(r["filename"]).parts[0]] += 1
    return {"images": len(rows), "skipped": skipped, "classes": dict(sorted(counts.items())),
            "mean_caption_words": sum(lengths) / len(lengths), "min_caption_words": min(lengths), "max_caption_words": max(lengths)}
