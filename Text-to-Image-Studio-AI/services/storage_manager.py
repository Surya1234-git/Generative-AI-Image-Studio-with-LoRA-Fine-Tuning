"""Persistent storage: image files on disk + metadata in SQLite.

Images are stored under generated_images/YYYY-MM/ with the generation parameters
embedded in the PNG itself *and* in the database, so the Gallery filters work on
real stored metadata and everything survives a restart (or a Colab disconnect when
the data folder lives on Google Drive).
"""
from __future__ import annotations

import json
import os
import random
import shutil
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT NOT NULL, filename TEXT, prompt TEXT, negative_prompt TEXT,
    model TEXT, scheduler TEXT, width INTEGER, height INTEGER, steps INTEGER, guidance REAL, seed INTEGER,
    lora TEXT, lora_scale REAL, gen_type TEXT, created_at TEXT, favorite INTEGER DEFAULT 0, tags TEXT DEFAULT '',
    elapsed REAL, extra TEXT DEFAULT '{}');
CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE, path TEXT, kind TEXT, num_images INTEGER,
    num_classes INTEGER, created_at TEXT, info TEXT DEFAULT '{}');
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, name TEXT, config TEXT, metrics TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS prompt_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT, prompt TEXT, negative_prompt TEXT, created_at TEXT);
"""


@contextmanager
def connection():
    settings.ensure_dirs()
    conn = sqlite3.connect(settings.DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.DatabaseError:
        pass  # some network drives do not support WAL; default mode still works
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connection() as c:
        c.executescript(SCHEMA)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _row(r) -> dict | None:
    if r is None:
        return None
    d = dict(r)
    for key in ("extra", "info", "config", "metrics"):
        if key in d and isinstance(d[key], str):
            try:
                d[key] = json.loads(d[key])
            except ValueError:
                d[key] = {}
    return d


# --------------------------------------------------------------------------- images
def save_generated_image(image: Image.Image, meta: dict, fmt: str = "PNG") -> dict:
    """Write the image to disk and insert its metadata row. Returns the stored record."""
    init_db()
    stamp = datetime.now()
    folder = settings.dir_path("generated_images") / stamp.strftime("%Y-%m")
    folder.mkdir(parents=True, exist_ok=True)
    name = f"img_{stamp.strftime('%Y%m%d_%H%M%S')}_{meta.get('seed', 0)}_{random.randint(0, 9999):04d}.{fmt.lower()}"
    path = folder / name
    info = PngInfo()
    info.add_text("parameters", json.dumps({k: v for k, v in meta.items() if k != "extra"}, default=str))
    image.save(path, pnginfo=info) if fmt.upper() == "PNG" else image.convert("RGB").save(path, quality=95)
    with connection() as c:
        cur = c.execute(
            """INSERT INTO images (path, filename, prompt, negative_prompt, model, scheduler, width, height, steps,
               guidance, seed, lora, lora_scale, gen_type, created_at, elapsed, extra)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (str(path), name, meta.get("prompt", ""), meta.get("negative_prompt", ""), meta.get("model", ""),
             meta.get("scheduler", ""), image.width, image.height, meta.get("steps"), meta.get("guidance"),
             meta.get("seed"), meta.get("lora", ""), meta.get("lora_scale"), meta.get("gen_type", "txt2img"),
             _now(), meta.get("elapsed"), json.dumps(meta.get("extra", {}), default=str)))
        new_id = cur.lastrowid
    return get_image(new_id)


def get_image(image_id: int) -> dict | None:
    with connection() as c:
        return _row(c.execute("SELECT * FROM images WHERE id=?", (image_id,)).fetchone())


def list_images(search: str = "", models=None, favorites_only: bool = False, date_from=None, date_to=None,
                min_width=None, max_width=None, tags=None, gen_types=None, order: str = "newest",
                limit: int = 500, offset: int = 0) -> list[dict]:
    init_db()
    where, args = [], []
    if search:
        like = f"%{search.lower()}%"
        where.append("(LOWER(prompt) LIKE ? OR LOWER(filename) LIKE ? OR LOWER(tags) LIKE ? OR LOWER(model) LIKE ?)")
        args += [like, like, like, like]
    if models:
        where.append(f"model IN ({','.join('?' * len(models))})"); args += list(models)
    if gen_types:
        where.append(f"gen_type IN ({','.join('?' * len(gen_types))})"); args += list(gen_types)
    if favorites_only:
        where.append("favorite=1")
    if date_from:
        where.append("created_at >= ?"); args.append(str(date_from))
    if date_to:
        where.append("created_at < ?"); args.append(str(date_to + timedelta(days=1)) if hasattr(date_to, "year") else str(date_to))
    if min_width:
        where.append("width >= ?"); args.append(int(min_width))
    if max_width:
        where.append("width <= ?"); args.append(int(max_width))
    for t in tags or []:
        where.append("tags LIKE ?"); args.append(f"%,{t.strip().lower()},%")
    orders = {"newest": "id DESC", "oldest": "id ASC", "favorites first": "favorite DESC, id DESC"}
    sql = "SELECT * FROM images" + (" WHERE " + " AND ".join(where) if where else "")
    sql += f" ORDER BY {orders.get(order, 'id DESC')} LIMIT ? OFFSET ?"
    with connection() as c:
        return [_row(r) for r in c.execute(sql, args + [limit, offset]).fetchall()]


def count_images(**filters) -> int:
    return len(list_images(limit=10 ** 6, **filters))


def distinct_values(column: str) -> list[str]:
    assert column in {"model", "gen_type"}
    init_db()
    with connection() as c:
        return [r[0] for r in c.execute(f"SELECT DISTINCT {column} FROM images WHERE {column} != '' ORDER BY 1")]


def all_tags() -> list[str]:
    tags = set()
    for rec in list_images(limit=10 ** 6):
        tags.update(t for t in rec["tags"].split(",") if t)
    return sorted(tags)


def set_favorite(image_id: int, value: bool) -> None:
    with connection() as c:
        c.execute("UPDATE images SET favorite=? WHERE id=?", (int(value), image_id))


def set_tags(image_id: int, tags: list[str]) -> None:
    clean = sorted({t.strip().lower().replace(",", " ") for t in tags if t.strip()})
    with connection() as c:
        c.execute("UPDATE images SET tags=? WHERE id=?", ("," + ",".join(clean) + "," if clean else "", image_id))


def rename_image(image_id: int, new_name: str) -> str:
    rec = get_image(image_id)
    stem = "".join(ch for ch in new_name if ch.isalnum() or ch in "-_ ").strip().replace(" ", "_")
    if not rec or not stem:
        raise ValueError("Enter a name using letters, numbers, dashes or underscores.")
    old = Path(rec["path"])
    new = old.with_name(stem + old.suffix)
    if new.exists() and new != old:
        raise ValueError("A file with that name already exists.")
    old.rename(new)
    with connection() as c:
        c.execute("UPDATE images SET path=?, filename=? WHERE id=?", (str(new), new.name, image_id))
    return new.name


def delete_image(image_id: int) -> bool:
    rec = get_image(image_id)
    if not rec:
        return False
    try:
        Path(rec["path"]).unlink(missing_ok=True)
    except OSError:
        pass
    with connection() as c:
        c.execute("DELETE FROM images WHERE id=?", (image_id,))
    return True


def load_pil(record: dict) -> Image.Image | None:
    try:
        return Image.open(record["path"]).convert("RGB")
    except (OSError, KeyError):
        return None


def purge_images(older_than_days: int, keep_favorites: bool = True) -> int:
    """Delete images older than N days (data-retention control). Returns how many were removed."""
    cutoff = (datetime.now() - timedelta(days=older_than_days)).isoformat(timespec="seconds")
    removed = 0
    for rec in list_images(limit=10 ** 6, order="oldest"):
        if rec["created_at"] < cutoff and not (keep_favorites and rec["favorite"]):
            removed += delete_image(rec["id"])
    return removed


# --------------------------------------------------------------------------- prompt history
def add_prompt_history(prompt: str, negative: str = "") -> None:
    if not prompt.strip():
        return
    init_db()
    with connection() as c:
        last = c.execute("SELECT prompt FROM prompt_history ORDER BY id DESC LIMIT 1").fetchone()
        if not last or last["prompt"] != prompt:
            c.execute("INSERT INTO prompt_history (prompt, negative_prompt, created_at) VALUES (?,?,?)",
                      (prompt, negative, _now()))


def list_prompt_history(limit: int = 30) -> list[dict]:
    init_db()
    with connection() as c:
        return [dict(r) for r in c.execute("SELECT * FROM prompt_history ORDER BY id DESC LIMIT ?", (limit,))]


# --------------------------------------------------------------------------- datasets
def register_dataset(name: str, path: str | Path, kind: str, num_images: int, num_classes: int, info: dict | None = None) -> dict:
    init_db()
    with connection() as c:
        c.execute("""INSERT INTO datasets (name, path, kind, num_images, num_classes, created_at, info)
                     VALUES (?,?,?,?,?,?,?)
                     ON CONFLICT(name) DO UPDATE SET path=excluded.path, kind=excluded.kind,
                     num_images=excluded.num_images, num_classes=excluded.num_classes, info=excluded.info""",
                  (name, str(path), kind, num_images, num_classes, _now(), json.dumps(info or {}, default=str)))
    return get_dataset(name)


def get_dataset(name: str) -> dict | None:
    init_db()
    with connection() as c:
        return _row(c.execute("SELECT * FROM datasets WHERE name=?", (name,)).fetchone())


def list_datasets() -> list[dict]:
    init_db()
    with connection() as c:
        return [_row(r) for r in c.execute("SELECT * FROM datasets ORDER BY id DESC")]


def delete_dataset(name: str, delete_files: bool = False) -> None:
    rec = get_dataset(name)
    if rec and delete_files:
        shutil.rmtree(rec["path"], ignore_errors=True)
    with connection() as c:
        c.execute("DELETE FROM datasets WHERE name=?", (name,))


# --------------------------------------------------------------------------- experiments
def add_experiment(kind: str, name: str, config: dict, metrics: dict) -> int:
    init_db()
    with connection() as c:
        cur = c.execute("INSERT INTO experiments (kind, name, config, metrics, created_at) VALUES (?,?,?,?,?)",
                        (kind, name, json.dumps(config, default=str), json.dumps(metrics, default=str), _now()))
        return cur.lastrowid


def list_experiments(kind: str | None = None, limit: int = 500) -> list[dict]:
    init_db()
    with connection() as c:
        if kind:
            rows = c.execute("SELECT * FROM experiments WHERE kind=? ORDER BY id DESC LIMIT ?", (kind, limit))
        else:
            rows = c.execute("SELECT * FROM experiments ORDER BY id DESC LIMIT ?", (limit,))
        return [_row(r) for r in rows]


def delete_experiment(exp_id: int) -> None:
    with connection() as c:
        c.execute("DELETE FROM experiments WHERE id=?", (exp_id,))


# --------------------------------------------------------------------------- search / maintenance
def global_search(query: str, limit: int = 8) -> dict:
    q = f"%{query.lower()}%"
    init_db()
    with connection() as c:
        return {
            "images": [_row(r) for r in c.execute("SELECT * FROM images WHERE LOWER(prompt) LIKE ? OR LOWER(tags) LIKE ? ORDER BY id DESC LIMIT ?", (q, q, limit))],
            "datasets": [_row(r) for r in c.execute("SELECT * FROM datasets WHERE LOWER(name) LIKE ? LIMIT ?", (q, limit))],
            "experiments": [_row(r) for r in c.execute("SELECT * FROM experiments WHERE LOWER(name) LIKE ? OR LOWER(kind) LIKE ? ORDER BY id DESC LIMIT ?", (q, q, limit))],
            "prompts": [dict(r) for r in c.execute("SELECT * FROM prompt_history WHERE LOWER(prompt) LIKE ? ORDER BY id DESC LIMIT ?", (q, limit))],
        }


def _folder_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def storage_usage() -> dict:
    usage = {sub: _folder_size(settings.dir_path(sub)) for sub in settings.SUBDIRS}
    usage["database"] = settings.DB_PATH.stat().st_size if settings.DB_PATH.exists() else 0
    try:
        usage["disk_free"] = shutil.disk_usage(settings.DATA_ROOT).free
    except OSError:
        usage["disk_free"] = 0
    return usage


def backup_database() -> Path:
    """Consistent copy of studio.db into reports/ using SQLite's backup API."""
    init_db()
    dest = settings.dir_path("reports") / f"backup_studio_{time.strftime('%Y%m%d_%H%M%S')}.db"
    with connection() as src:
        out = sqlite3.connect(dest)
        try:
            src.backup(out)
        finally:
            out.close()
    return dest
