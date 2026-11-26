import random
import shutil
import tempfile
from pathlib import Path

from PIL import Image

from config import settings


def fresh_data_root() -> Path:
    path = Path(tempfile.mkdtemp(prefix="studio_test_"))
    settings.set_data_root(path)
    settings.ensure_dirs()
    return path


def cleanup(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def noise_image(seed: int, size=(48, 40), color_bias=0) -> Image.Image:
    rng = random.Random(seed)
    img = Image.new("RGB", size)
    img.putdata([(min(255, rng.randint(0, 255) + color_bias), rng.randint(0, 255), rng.randint(0, 255)) for _ in range(size[0] * size[1])])
    return img


def make_dataset(root: Path, classes=("cats", "dogs", "birds"), per_class=10, with_duplicate=True, with_corrupt=True) -> Path:
    for ci, cls in enumerate(classes):
        (root / cls).mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            noise_image(ci * 1000 + i).save(root / cls / f"{cls}_{i:03d}.png")
    if with_duplicate:
        shutil.copy(root / classes[0] / f"{classes[0]}_000.png", root / classes[0] / "copy_of_000.png")
        shutil.copy(root / classes[0] / f"{classes[0]}_001.png", root / classes[0] / "copy_of_001.png")
    if with_corrupt:
        (root / classes[1] / "broken.jpg").write_bytes(b"this is not an image")
    return root
