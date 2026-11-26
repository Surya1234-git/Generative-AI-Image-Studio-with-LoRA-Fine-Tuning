"""Dataset analysis, cleaning, preprocessing and leakage-safe splitting (Module 2 / Task 4).

No deep-learning framework is needed here (Pillow + NumPy + pandas), except the optional
Oxford-102 Flowers download which uses torchvision.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
import shutil
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}
MAX_ZIP_BYTES = 4 * 1024 ** 3

OXFORD102_NAMES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea", "english marigold", "tiger lily",
    "moon orchid", "bird of paradise", "monkshood", "globe thistle", "snapdragon", "colt's foot", "king protea",
    "spear thistle", "yellow iris", "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
    "giant white arum lily", "fire lily", "pincushion flower", "fritillary", "red ginger", "grape hyacinth",
    "corn poppy", "prince of wales feathers", "stemless gentian", "artichoke", "sweet william", "carnation",
    "garden phlox", "love in the mist", "mexican aster", "alpine sea holly", "ruby-lipped cattleya", "cape flower",
    "great masterwort", "siam tulip", "lenten rose", "barbeton daisy", "daffodil", "sword lily", "poinsettia",
    "bolero deep blue", "wallflower", "marigold", "buttercup", "oxeye daisy", "common dandelion", "petunia",
    "wild pansy", "primula", "sunflower", "pelargonium", "bishop of llandaff", "gaura", "geranium", "orange dahlia",
    "pink-yellow dahlia", "cautleya spicata", "japanese anemone", "black-eyed susan", "silverbush",
    "californian poppy", "osteospermum", "spring crocus", "bearded iris", "windflower", "tree poppy", "gazania",
    "azalea", "water lily", "rose", "thorn apple", "morning glory", "passion flower", "lotus", "toad lily",
    "anthurium", "frangipani", "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow", "magnolia",
    "cyclamen", "watercress", "canna lily", "hippeastrum", "bee balm", "ball moss", "foxglove", "bougainvillea",
    "camellia", "mallow", "mexican petunia", "bromelia", "blanket flower", "trumpet creeper", "blackberry lily",
]


# --------------------------------------------------------------------------- ingestion
def safe_extract_zip(data: bytes | Path, dest: Path) -> int:
    """Extract image files (plus captions.csv / small .txt captions), blocking path traversal ('zip slip') and decompression bombs."""
    dest.mkdir(parents=True, exist_ok=True)
    zf = zipfile.ZipFile(io.BytesIO(data) if isinstance(data, bytes) else data)
    if sum(i.file_size for i in zf.infolist()) > MAX_ZIP_BYTES:
        raise ValueError("The ZIP expands to more than 4 GB and was rejected.")
    count = 0
    root = dest.resolve()
    for info in zf.infolist():
        name = info.filename
        if info.is_dir() or name.startswith("__MACOSX") or Path(name).name.startswith("."):
            continue
        is_image = Path(name).suffix.lower() in IMAGE_EXTS
        # captions travel with the images: a captions.csv and small .txt caption files are kept too
        is_caption = (Path(name).name.lower() == "captions.csv" and info.file_size <= 50 * 1024 ** 2) or (Path(name).suffix.lower() == ".txt" and info.file_size <= 20_000)
        if not (is_image or is_caption):
            continue
        target = (dest / name).resolve()
        if root not in target.parents:
            continue  # path traversal attempt
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, open(target, "wb") as out:
            shutil.copyfileobj(src, out)
        count += 1 if is_image else 0
    if count == 0:
        raise ValueError("No images were found in the ZIP.")
    return count


def save_uploaded_files(files: list[tuple[str, bytes]], dest: Path, class_name: str | None = None) -> int:
    folder = dest / class_name if class_name else dest
    folder.mkdir(parents=True, exist_ok=True)
    count = 0
    for name, data in files:
        if Path(name).suffix.lower() in IMAGE_EXTS:
            (folder / Path(name).name).write_bytes(data)
            count += 1
    return count


def flatten_single_root(dest: Path) -> None:
    """ZIPs often contain one wrapper folder ('dataset/cats/...'); step into it so classes are first-level folders."""
    while True:
        entries = [p for p in dest.iterdir() if not p.name.startswith(".")]
        if len(entries) == 1 and entries[0].is_dir() and not any(f.suffix.lower() in IMAGE_EXTS for f in entries[0].iterdir() if f.is_file()):
            for child in entries[0].iterdir():
                shutil.move(str(child), str(dest / child.name))
            entries[0].rmdir()
        else:
            return


def class_caption(class_name: str) -> str:
    """'003_canterbury_bells' -> 'a photo of canterbury bells' (templated caption, not human-written)."""
    clean = re.sub(r"^\d+[_\-\s]*", "", class_name).replace("_", " ").replace("-", " ").strip()
    return f"a photo of {clean or class_name}"


def load_oxford_flowers(dest: Path, max_per_class: int | None = None, progress_cb=None) -> int:
    """Download Oxford-102 Flowers (torchvision) and write it as class folders. Labels only: no human captions exist."""
    from torchvision.datasets import Flowers102

    tmp = dest.parent / "_oxford_download"
    items, per_class = [], {}
    for split in ("train", "val", "test"):
        ds = Flowers102(root=str(tmp), split=split, download=True)
        for img_path, label in zip(ds._image_files, ds._labels):
            items.append((img_path, int(label)))
    count = 0
    for i, (img_path, label) in enumerate(items):
        if max_per_class and per_class.get(label, 0) >= max_per_class:
            continue
        per_class[label] = per_class.get(label, 0) + 1
        slug = OXFORD102_NAMES[label].replace(" ", "_").replace("'", "")
        folder = dest / f"{label + 1:03d}_{slug}"
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy(img_path, folder / Path(img_path).name)
        count += 1
        if progress_cb and i % 200 == 0:
            progress_cb(i / len(items))
    shutil.rmtree(tmp, ignore_errors=True)
    return count


# --------------------------------------------------------------------------- analysis
def dhash(img: Image.Image, size: int = 8) -> int:
    """Difference hash: similar images get hashes that differ in only a few bits (near-duplicate detection)."""
    small = np.asarray(img.convert("L").resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_dataset(root: str | Path, labeled: bool = True, progress_cb=None) -> pd.DataFrame:
    """One row per image: class, size, format, bytes, md5, dhash and whether it is corrupted."""
    root = Path(root)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
                   and "_quarantine" not in p.parts and not p.name.startswith("."))
    rows = []
    for i, path in enumerate(files):
        rel = path.relative_to(root)
        cls = rel.parts[0] if labeled and len(rel.parts) > 1 else "unlabeled"
        row = {"path": str(path), "rel": str(rel), "cls": cls, "bytes": path.stat().st_size, "md5": _md5(path),
               "width": None, "height": None, "format": path.suffix.lower().lstrip("."), "mode": None, "dhash": None,
               "corrupted": False, "error": ""}
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:  # verify() invalidates the handle, reopen to read real pixels
                im.load()
                row.update(width=im.width, height=im.height, format=(im.format or row["format"]).lower(), mode=im.mode,
                           dhash=dhash(im))
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
            row.update(corrupted=True, error=f"{type(exc).__name__}: {exc}"[:120])
        rows.append(row)
        if progress_cb and (i % 50 == 0 or i == len(files) - 1):
            progress_cb((i + 1) / max(1, len(files)))
    df = pd.DataFrame(rows, columns=["path", "rel", "cls", "bytes", "md5", "width", "height", "format", "mode", "dhash", "corrupted", "error"])
    return mark_duplicates(df, hamming=0)


_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def mark_duplicates(df: pd.DataFrame, hamming: int = 0) -> pd.DataFrame:
    """Group exact copies (same bytes) and near-duplicates (dhash distance <= `hamming`).

    Adds `dup_group` (-1 = unique) and `is_duplicate` (True for every copy after the first in a group).
    """
    df = df.copy()
    n = len(df)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    good = [i for i in range(n) if not df.iloc[i]["corrupted"]]
    first_by_md5: dict[str, int] = {}
    for i in good:
        m = df.iloc[i]["md5"]
        if m in first_by_md5:
            union(first_by_md5[m], i)
        else:
            first_by_md5[m] = i
    if hamming > 0 and good:
        hashes = np.array([int(df.iloc[i]["dhash"]) for i in good], dtype=np.uint64)
        for k in range(len(good) - 1):
            xor = hashes[k + 1:] ^ hashes[k]
            dist = _POP[xor.view(np.uint8).reshape(-1, 8)].sum(axis=1)
            for off in np.nonzero(dist <= hamming)[0]:
                union(good[k], good[k + 1 + int(off)])
    groups: dict[int, list[int]] = {}
    for i in good:
        groups.setdefault(find(i), []).append(i)
    df["dup_group"] = -1
    df["is_duplicate"] = False
    gid = 0
    for members in groups.values():
        if len(members) > 1:
            for j, idx in enumerate(members):
                df.iat[idx, df.columns.get_loc("dup_group")] = gid
                df.iat[idx, df.columns.get_loc("is_duplicate")] = j > 0
            gid += 1
    return df


def summarize(df: pd.DataFrame) -> dict:
    ok = df[~df["corrupted"]]
    return {
        "total_images": int(len(df)),
        "num_classes": int(df["cls"].nunique()) if len(df) else 0,
        "avg_width": float(ok["width"].mean()) if len(ok) else 0.0,
        "avg_height": float(ok["height"].mean()) if len(ok) else 0.0,
        "corrupted": int(df["corrupted"].sum()),
        "duplicates": int(df["is_duplicate"].sum()) if "is_duplicate" in df else 0,
        "storage_bytes": int(df["bytes"].sum()),
        "formats": ok["format"].value_counts().to_dict(),
        "class_counts": df["cls"].value_counts().sort_index().to_dict(),
    }


def remove_corrupted(df: pd.DataFrame, root: str | Path) -> int:
    """Move corrupted files into <dataset>/_quarantine (reversible) instead of deleting them."""
    quarantine = Path(root) / "_quarantine"
    moved = 0
    for path in df.loc[df["corrupted"], "path"]:
        src = Path(path)
        if src.exists():
            quarantine.mkdir(exist_ok=True)
            shutil.move(str(src), str(quarantine / f"{moved}_{src.name}"))
            moved += 1
    return moved


# --------------------------------------------------------------------------- preprocessing
@dataclass
class PreprocessConfig:
    size: int = 64
    mode: str = "center_crop"      # resize | center_crop | random_crop
    flip: bool = False
    flip_probability: float = 0.5
    normalize: str = "-1..1"       # none | 0..1 | -1..1 | imagenet
    out_format: str = "PNG"        # PNG | JPEG | WEBP
    seed: int = 42


def preprocess_image(img: Image.Image, cfg: PreprocessConfig, rng: random.Random | None = None) -> Image.Image:
    rng = rng or random.Random(cfg.seed)
    img = img.convert("RGB")
    if cfg.mode == "resize":
        img = img.resize((cfg.size, cfg.size), Image.LANCZOS)
    else:
        scale = (cfg.size if cfg.mode == "center_crop" else int(cfg.size * 1.15)) / min(img.size)
        img = img.resize((max(cfg.size, round(img.width * scale)), max(cfg.size, round(img.height * scale))), Image.LANCZOS)
        if cfg.mode == "center_crop":
            left, top = (img.width - cfg.size) // 2, (img.height - cfg.size) // 2
        else:
            left, top = rng.randint(0, img.width - cfg.size), rng.randint(0, img.height - cfg.size)
        img = img.crop((left, top, left + cfg.size, top + cfg.size))
    if cfg.flip and rng.random() < cfg.flip_probability:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def normalize_array(img: Image.Image, mode: str) -> np.ndarray:
    """Pixel values as the model will see them (HxWx3 float array)."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    if mode == "-1..1":
        return arr * 2 - 1
    if mode == "imagenet":
        return (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return arr


def split_dataset(df: pd.DataFrame, ratios=(0.7, 0.15, 0.15), seed: int = 42) -> pd.Series:
    """Stratified, seeded split in which duplicate / near-duplicate images always stay together (no leakage)."""
    names = ["train", "val", "test"]
    if abs(sum(ratios) - 1) > 1e-6:
        raise ValueError("Split ratios must add up to 100%.")
    rng = random.Random(seed)
    out = pd.Series("", index=df.index, dtype=object)
    usable = df[~df["corrupted"]]
    for _cls, sub in usable.groupby("cls", sort=True):
        groups: dict[int, list] = {}
        for idx, g in sub["dup_group"].items():
            groups.setdefault(g if g >= 0 else -(idx + 2), []).append(idx)
        group_list = sorted(groups.values(), key=lambda m: m[0])
        rng.shuffle(group_list)
        total = len(sub)
        targets = [r * total for r in ratios]
        counts = [0, 0, 0]
        for members in group_list:
            deficits = [targets[i] - counts[i] for i in range(3)]
            choice = int(np.argmax(deficits))
            for idx in members:
                out.at[idx] = names[choice]
            counts[choice] += len(members)
    return out


def apply_preprocessing(df: pd.DataFrame, out_dir: Path, cfg: PreprocessConfig, splits: pd.Series | None = None,
                        progress_cb=None) -> int:
    """Write the processed dataset as class folders (inside train/val/test folders when a split is given)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.seed)
    ext = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}[cfg.out_format]
    usable = df[~df["corrupted"]]
    source_caps = _real_caption_map(df)            # only human/CSV/.txt captions; templated ones are regenerated on demand
    caption_rows = []
    written = 0
    for n, (idx, row) in enumerate(usable.iterrows()):
        if splits is not None and splits.get(idx, "") == "":
            continue
        folder = out_dir / (splits[idx] if splits is not None else "") / row["cls"]
        folder.mkdir(parents=True, exist_ok=True)
        with Image.open(row["path"]) as im:
            result = preprocess_image(im, cfg, rng)
        target = folder / (Path(row["path"]).stem + ext)
        result.save(target, quality=95) if cfg.out_format != "PNG" else result.save(target)
        if row["rel"] in source_caps:
            caption_rows.append({"filename": str(target.relative_to(out_dir)), "caption": source_caps[row["rel"]]})
        written += 1
        if progress_cb and n % 50 == 0:
            progress_cb((n + 1) / max(1, len(usable)))
    if caption_rows:
        write_captions_csv(out_dir, caption_rows)
    (out_dir / "preprocess_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    return written


def training_root(dataset_path: str | Path) -> Path:
    """Folder that holds class sub-folders: <dataset>/train if the dataset was split, else the dataset itself."""
    p = Path(dataset_path)
    return p / "train" if (p / "train").is_dir() else p


def class_names_from_root(root: str | Path) -> list[str]:
    return sorted(d.name for d in Path(root).iterdir() if d.is_dir() and not d.name.startswith(("_", ".")))


# --------------------------------------------------------------------------- export / report
def zip_directory(path: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for f in path.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(path))
    return buf.getvalue()


def build_report(df: pd.DataFrame, name: str = "dataset", caption_summary: dict | None = None) -> dict:
    s = summarize(df)
    ok = df[~df["corrupted"]]
    return {"dataset": name, "summary": s, "caption_stats": caption_summary or {},
            "size_stats": ok[["width", "height", "bytes"]].describe().round(2).to_dict() if len(ok) else {},
            "corrupted_files": df.loc[df["corrupted"], ["rel", "error"]].to_dict("records"),
            "duplicate_groups": {int(g): sub["rel"].tolist() for g, sub in df[df["dup_group"] >= 0].groupby("dup_group")}}


def report_markdown(report: dict) -> str:
    s = report["summary"]
    lines = [f"# Dataset report: {report['dataset']}", "",
             f"- Total images: **{s['total_images']}**", f"- Classes: **{s['num_classes']}**",
             f"- Average size: **{s['avg_width']:.0f} x {s['avg_height']:.0f}**", f"- Corrupted images: **{s['corrupted']}**",
             f"- Duplicate images (extra copies): **{s['duplicates']}**", f"- Storage: **{s['storage_bytes'] / 1024 ** 2:.1f} MB**", "",
             "## Class distribution", ""]
    lines += [f"- {k}: {v}" for k, v in s["class_counts"].items()]
    lines += ["", "## Formats", ""] + [f"- {k}: {v}" for k, v in s["formats"].items()]
    cs = report.get("caption_stats")
    if cs:
        lines += ["", "## Description (caption) length", "",
                  f"- Captions: **{cs['n_captions']}** ({cs['n_unique']} unique)",
                  f"- Words per caption: mean **{cs['words_mean']:.2f}**, median {cs['words_median']:.1f}, min {cs['words_min']}, max {cs['words_max']}",
                  f"- Characters per caption: mean **{cs['chars_mean']:.1f}**",
                  f"- Caption sources: {cs['sources']}"]
        if cs["templated_only"]:
            lines.append("- Note: every caption is templated from the class name, so these lengths say nothing about real descriptions.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- captions (Task 4: description statistics)
def write_captions_csv(root: str | Path, rows: list[dict]) -> None:
    """captions.csv with columns filename (path relative to the dataset root) and caption."""
    with open(Path(root) / "captions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "caption"])
        writer.writeheader()
        writer.writerows({"filename": r["filename"], "caption": r["caption"]} for r in rows)


def _read_caption_sources(root: Path) -> dict[str, str]:
    table: dict[str, str] = {}
    csv_path = root / "captions.csv"
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if (row.get("caption") or "").strip():
                    table[row["filename"]] = row["caption"].strip()
    return table


def load_captions(root: str | Path, df: pd.DataFrame) -> pd.DataFrame:
    """One row per readable image: rel, cls, caption, source.

    Priority (same as LoRA training): captions.csv > same-name .txt file > class-folder template ("a photo of <class>").
    `source` says which one was used, so templated captions are never mistaken for human descriptions.
    """
    root = Path(root)
    table = _read_caption_sources(root)
    rows = []
    for _, r in df[~df["corrupted"]].iterrows():
        rel, path = r["rel"], Path(r["path"])
        txt = path.with_suffix(".txt")
        if rel in table:
            cap, src = table[rel], "captions.csv"
        elif path.name in table:
            cap, src = table[path.name], "captions.csv"
        elif txt.exists() and txt.read_text(encoding="utf-8").strip():
            cap, src = txt.read_text(encoding="utf-8").strip(), ".txt file"
        else:
            cap, src = class_caption(r["cls"]), "template"
        rows.append({"rel": rel, "path": str(path), "cls": r["cls"], "caption": cap, "source": src})
    return pd.DataFrame(rows, columns=["rel", "path", "cls", "caption", "source"])


def _real_caption_map(df: pd.DataFrame) -> dict[str, str]:
    """rel path -> caption for images that have a real (non-template) caption. Needs the dataset root from df paths."""
    if df.empty:
        return {}
    first = df.iloc[0]
    root = Path(first["path"])
    for _ in Path(first["rel"]).parts:
        root = root.parent
    caps = load_captions(root, df)
    real = caps[caps["source"] != "template"]
    return dict(zip(real["rel"], real["caption"]))


def caption_stats(caps: pd.DataFrame) -> dict:
    """Description-length statistics (words and characters) overall and per class."""
    if caps.empty:
        return {"n_captions": 0, "n_unique": 0, "words_mean": 0.0, "words_median": 0.0, "words_min": 0, "words_max": 0,
                "chars_mean": 0.0, "sources": {}, "templated_only": True, "per_class": {}}
    words = caps["caption"].str.split().str.len()
    chars = caps["caption"].str.len()
    per_class = caps.assign(words=words).groupby("cls")["words"].mean().round(2).to_dict()
    return {"n_captions": int(len(caps)), "n_unique": int(caps["caption"].nunique()),
            "words_mean": float(words.mean()), "words_median": float(words.median()),
            "words_min": int(words.min()), "words_max": int(words.max()), "chars_mean": float(chars.mean()),
            "sources": caps["source"].value_counts().to_dict(),
            "templated_only": bool((caps["source"] == "template").all()), "per_class": per_class}


# --------------------------------------------------------------------------- synthetic shapes (Task 2 demo dataset)
SHAPE_NAMES = ("square", "circle", "triangle", "star")
_COLORS = {"red": (220, 40, 40), "green": (40, 170, 70), "blue": (50, 90, 220), "yellow": (240, 210, 40),
           "purple": (140, 60, 180), "orange": (240, 140, 30)}
_BACKGROUNDS = {"white": (255, 255, 255), "black": (20, 20, 20), "gray": (170, 170, 170)}
_CAPTION_TEMPLATES = (
    "a {color} {shape}",
    "a {size} {color} {shape}",
    "a {size} {color} {shape} on a {bg} background",
    "a {color} {shape} centered on a plain {bg} background",
    "a {size} {color} {shape} placed {where} on a {bg} background",
)


def _shape_points(shape: str, cx: float, cy: float, r: float) -> list[tuple[float, float]]:
    if shape == "square":
        return [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
    if shape == "triangle":
        return [(cx, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
    pts = []                                              # star: 5 outer + 5 inner points
    for k in range(10):
        ang = -np.pi / 2 + k * np.pi / 5
        rad = r if k % 2 == 0 else r * 0.45
        pts.append((cx + rad * np.cos(ang), cy + rad * np.sin(ang)))
    return pts


def draw_shape(shape: str, size: int, rng: random.Random) -> tuple[Image.Image, dict]:
    """Draw one shape with random colour, size, position and background. Returns (image, attributes)."""
    from PIL import ImageDraw
    if shape not in SHAPE_NAMES:
        raise ValueError(f"Unknown shape '{shape}'. Choose from {', '.join(SHAPE_NAMES)}.")
    color, bg = rng.choice(list(_COLORS)), rng.choice(list(_BACKGROUNDS))
    scale = rng.uniform(0.22, 0.42)
    r = size * scale
    margin = r + 2
    cx = rng.uniform(margin, size - margin)
    cy = rng.uniform(margin, size - margin)
    img = Image.new("RGB", (size, size), _BACKGROUNDS[bg])
    d = ImageDraw.Draw(img)
    if shape == "circle":
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=_COLORS[color])
    else:
        d.polygon(_shape_points(shape, cx, cy, r), fill=_COLORS[color])
    third = size / 3
    horiz = "left" if cx < third else "right" if cx > 2 * third else "center"
    vert = "top" if cy < third else "bottom" if cy > 2 * third else "middle"
    place = "in the middle" if (horiz, vert) == ("center", "middle") else " ".join(w for w in (vert, horiz) if w not in ("middle", "center"))
    attrs = {"color": color, "bg": bg, "size": "large" if scale > 0.33 else "small", "where": place}
    return img, attrs


def generate_shapes_dataset(dest: Path, shapes=("square", "circle", "triangle"), per_class: int = 100,
                            size: int = 64, seed: int = 42) -> int:
    """Create a labelled shapes dataset (one folder per shape) plus captions.csv with varied-length descriptions."""
    if len(shapes) < 2:
        raise ValueError("Choose at least 2 shapes: a conditional GAN needs at least 2 classes.")
    if not 8 <= per_class <= 5000:
        raise ValueError("Images per shape must be between 8 and 5000.")
    rng = random.Random(seed)
    dest.mkdir(parents=True, exist_ok=True)
    rows, count = [], 0
    for shape in shapes:
        folder = dest / shape
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            img, attrs = draw_shape(shape, size, rng)
            name = f"{shape}_{i:04d}.png"
            img.save(folder / name)
            caption = rng.choice(_CAPTION_TEMPLATES).format(shape=shape, **attrs)
            caption = re.sub(r"\ba (?=[aeiou])", "an ", caption)          # "an orange square", not "a orange square"
            rows.append({"filename": str(Path(shape) / name), "caption": caption})
            count += 1
    write_captions_csv(dest, rows)
    return count
