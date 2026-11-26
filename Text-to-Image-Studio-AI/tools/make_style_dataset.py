"""Synthetic 'flat vector landscape' style dataset for testing LoRA fine-tuning (512x512, 24 images + captions.csv)."""
import csv, math, random, sys, zipfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

TRIGGER = "sks flat vector style"
SIZE = 512
# time of day: (sky_top, sky_bottom, sun_color, hill colours far->near, has_stars, word)
SCENES = {
    "sunrise": ((250, 190, 140), (255, 235, 190), (255, 245, 210), [(240, 170, 150), (205, 120, 130), (150, 85, 115), (95, 60, 95)], False),
    "sunset":  ((90, 50, 130),  (250, 140, 90),  (255, 220, 120), [(220, 110, 110), (170, 70, 110), (110, 45, 100), (60, 30, 75)], False),
    "day":     ((90, 170, 235), (200, 235, 250), (255, 240, 150), [(130, 190, 170), (90, 160, 140), (55, 120, 110), (30, 85, 85)], False),
    "night":   ((15, 20, 60),   (50, 60, 120),   (240, 240, 255), [(55, 65, 125), (40, 48, 105), (28, 34, 82), (15, 20, 55)], True),
}
COLOR_WORDS = {"sunrise": "peach and rose", "sunset": "purple and orange", "day": "teal and green", "night": "deep blue and indigo"}
SUN_WORD = {"sunrise": "a pale sun", "sunset": "a glowing sun", "day": "a bright sun", "night": "a full moon"}


def gradient(top, bottom):
    t = np.linspace(0, 1, SIZE)[:, None, None]
    arr = (np.array(top)[None, None, :] * (1 - t) + np.array(bottom)[None, None, :] * t)
    return Image.fromarray(np.repeat(arr, SIZE, axis=1).astype("uint8"))


def hill_layer(rng, base_y, amp, color):
    f1, f2, f3 = rng.uniform(1.0, 2.2), rng.uniform(2.5, 4.5), rng.uniform(5, 8)
    p1, p2, p3 = (rng.uniform(0, 6.28) for _ in range(3))
    pts = [(x, base_y - amp * (0.6 * math.sin(f1 * x / SIZE * 6.28 + p1) + 0.3 * math.sin(f2 * x / SIZE * 6.28 + p2) + 0.1 * math.sin(f3 * x / SIZE * 6.28 + p3)))
           for x in range(0, SIZE + 8, 8)]
    return pts + [(SIZE, SIZE), (0, SIZE)], color


def make_image(seed: int, scene: str):
    rng = random.Random(seed)
    top, bottom, sun_c, hills, stars = SCENES[scene]
    img = gradient(top, bottom)
    d = ImageDraw.Draw(img)
    if stars:
        for _ in range(rng.randint(40, 70)):
            x, y, r = rng.randint(0, SIZE), rng.randint(0, int(SIZE * .55)), rng.choice([1, 1, 2])
            d.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 235))
    sx, sy, sr = rng.randint(110, SIZE - 110), rng.randint(90, 230), rng.randint(36, 62)
    glow = Image.new("RGB", (SIZE, SIZE), (0, 0, 0)); gd = ImageDraw.Draw(glow)
    gd.ellipse((sx - sr * 2, sy - sr * 2, sx + sr * 2, sy + sr * 2), fill=tuple(int(c * .35) for c in sun_c))
    img = Image.fromarray(np.clip(np.asarray(img).astype(int) + np.asarray(glow.filter(ImageFilter.GaussianBlur(28))).astype(int), 0, 255).astype("uint8"))
    d = ImageDraw.Draw(img)
    d.ellipse((sx - sr, sy - sr, sx + sr, sy + sr), fill=sun_c)
    n_layers = rng.choice([3, 4])
    layers = hills[-n_layers:] if n_layers < 4 else hills
    for i, col in enumerate(layers):
        base = 270 + i * (230 // n_layers) + rng.randint(-10, 10)
        poly, c = hill_layer(rng, base, rng.randint(35, 70), col)
        d.polygon(poly, fill=c)
    tree_c = layers[-1]
    trees = 0
    if rng.random() < .7:
        for _ in range(rng.randint(3, 8)):
            x, base_y, h = rng.randint(10, SIZE - 10), rng.randint(430, 500), rng.randint(36, 70)
            d.polygon([(x, base_y - h), (x + h * .28, base_y), (x - h * .28, base_y)], fill=tree_c)
            trees += 1
    return img, {"sun": (sx, sy), "layers": n_layers, "trees": trees}


def caption(scene, info, rng):
    layers = {3: "three", 4: "four"}[info["layers"]]
    parts = [f"a {scene} landscape with {layers} layers of rolling hills and {SUN_WORD[scene]}",
             f"{COLOR_WORDS[scene]} palette"]
    if info["trees"]:
        parts.append("small pine trees in the foreground")
    if rng.random() < .5:
        parts.append("minimalist, smooth shapes, no outlines")
    return ", ".join(parts) + f", {TRIGGER}"


def build(out_dir: Path, n_per_scene: int = 6, seed: int = 7) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng, rows, k = random.Random(seed), [], 0
    for scene in SCENES:
        for i in range(n_per_scene):
            img, info = make_image(seed * 1000 + k, scene)
            name = f"landscape_{k:02d}_{scene}.png"
            img.save(out_dir / name)
            rows.append({"filename": name, "caption": caption(scene, info, rng)})
            k += 1
    with open(out_dir / "captions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "caption"]); w.writeheader(); w.writerows(rows)
    return k


if __name__ == "__main__":
    dst = Path(sys.argv[1]); print(build(dst))
