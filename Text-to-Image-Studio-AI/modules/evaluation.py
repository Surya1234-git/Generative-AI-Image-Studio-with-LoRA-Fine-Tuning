"""Evaluation Center logic (Module 5).

* Reference classifier  : a small CNN trained on the REAL dataset. Generated images are then classified
                          -> accuracy, confusion matrix, one-vs-rest ROC/AUC (the mentor's criteria).
* Diffusion comparison  : same prompt + seed on several models; time, VRAM and CLIPScore are measured.
* Reports               : PDF / CSV / JSON exports and a reproducibility report.
"""
from __future__ import annotations

import io
import json
import platform
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np

from config import settings
from modules import metrics


# --------------------------------------------------------------------------- reference classifier (needs torch)
def classifier_path(dataset_name: str, img_size: int) -> Path:
    path = settings.dir_path("checkpoints") / "classifiers"
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{dataset_name}_{img_size}.pt"


def _make_cnn(num_classes: int):
    import torch.nn as nn

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True), nn.MaxPool2d(2))
    return nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256), nn.AdaptiveAvgPool2d(1),
                         nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, num_classes))


def train_reference_classifier(dataset_path: str, dataset_name: str, img_size: int = 64, epochs: int = 8,
                               progress_cb=None) -> dict:
    """Train on real images (train split if present); report accuracy on held-out real images."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, random_split
    from torchvision import datasets, transforms
    from modules.dataset_explorer import training_root

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tf = transforms.Compose([transforms.Resize(img_size), transforms.CenterCrop(img_size), transforms.ToTensor(),
                             transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    root = Path(dataset_path)
    train_ds = datasets.ImageFolder(str(training_root(root)), transform=tf)
    test_dir = root / "test" if (root / "test").is_dir() else root / "val" if (root / "val").is_dir() else None
    if test_dir is not None:
        test_ds = datasets.ImageFolder(str(test_dir), transform=tf)
        if test_ds.classes != train_ds.classes:
            raise ValueError("Train and test folders contain different classes.")
    else:
        n_test = max(1, int(0.15 * len(train_ds)))
        train_ds, test_ds = random_split(train_ds, [len(train_ds) - n_test, n_test], generator=torch.Generator().manual_seed(0))
        train_ds.classes = test_ds.classes = train_ds.dataset.classes
    classes = list(train_ds.classes)
    model = _make_cnn(len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    for ep in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[3])        # light augmentation
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        if progress_cb:
            progress_cb((ep + 1) / epochs)
    model.eval()
    ys, probs = [], []
    with torch.no_grad():
        for x, y in DataLoader(test_ds, batch_size=128):
            probs.append(F.softmax(model(x.to(device)), dim=1).cpu().numpy())
            ys.append(y.numpy())
    real_metrics = metrics.classification_metrics(np.concatenate(ys), np.concatenate(probs), classes)
    torch.save({"state": model.state_dict(), "classes": classes, "img_size": img_size, "real_accuracy": real_metrics["accuracy"]},
               classifier_path(dataset_name, img_size))
    return {"classes": classes, "real_accuracy": real_metrics["accuracy"], "real_macro_auc": real_metrics["macro_auc"]}


def evaluate_gan_run(run_name: str, dataset_name: str, per_class: int = 50, seed: int = 0) -> dict:
    """Generate `per_class` images per class from a GAN run and classify them with the reference classifier."""
    import torch
    import torch.nn.functional as F
    from modules import gan_trainer

    G, classes, cfg = gan_trainer.load_generator(run_name)
    path = classifier_path(dataset_name, cfg["img_size"])
    if not path.exists():
        raise FileNotFoundError("Train the reference classifier for this dataset and image size first.")
    state = torch.load(path, map_location="cpu")
    if state["classes"] != classes:
        raise ValueError("The classifier and the GAN were trained on different class lists.")
    device = next(G.parameters()).device
    clf = _make_cnn(len(classes)).to(device)
    clf.load_state_dict(state["state"])
    clf.eval()
    gen = torch.Generator().manual_seed(seed)
    ys, probs = [], []
    with torch.no_grad():
        for c in range(len(classes)):
            labels = torch.full((per_class,), c, device=device)
            z = torch.randn(per_class, cfg["z_dim"], generator=gen).to(device)
            probs.append(F.softmax(clf(G(z, labels)), dim=1).cpu().numpy())
            ys.append(np.full(per_class, c))
    result = metrics.classification_metrics(np.concatenate(ys), np.concatenate(probs), classes)
    result.update(run=run_name, arch=cfg["arch"], cond_mode=cfg["cond_mode"], epochs=cfg["epochs"],
                  real_accuracy=state["real_accuracy"], classes=classes, per_class_generated=per_class, seed=seed)
    return result


# --------------------------------------------------------------------------- diffusion comparison
def compare_diffusion_models(prompts: list[str], model_keys: list[str], seed: int, steps: dict, scheduler: str,
                             guidance: float = 7.5, progress_cb=None, save: bool = True) -> list[dict]:
    """Same prompt + seed on each model. Every number comes from a real run on this machine."""
    from modules.image_generator import GenerationRequest, generate
    from services.model_manager import get_model_manager

    rows, total, done = [], len(prompts) * len(model_keys), 0
    for key in model_keys:                       # model-major order: each model is loaded once
        cfg = settings.MODEL_REGISTRY[key]
        w, h = cfg["native"]
        for prompt in prompts:
            req = GenerationRequest(prompt=prompt, model_key=key, width=w, height=h, steps=steps[key], guidance=guidance,
                                    seed=seed, scheduler=scheduler, gen_type="evaluation", save=save)
            res = generate(req)
            score = metrics.clip_scores(res.images[:1], [prompt])[0]
            rows.append({"model": key, "version": cfg["version"], "prompt": prompt, "resolution": f"{w}x{h}", "steps": steps[key],
                         "scheduler": (res.records[0].get("scheduler") or scheduler) if res.records else scheduler, "seed": seed, "time_s": round(res.elapsed, 2),
                         "peak_vram_gb": round(res.peak_gb, 2) if res.peak_gb is not None else None,
                         "clip_score": round(score, 2), "image": res.images[0], "record_id": res.records[0].get("id")})
            done += 1
            if progress_cb:
                progress_cb(done / total, f"{key}: {prompt[:40]}")
    get_model_manager().unload()
    return rows


def rank_models(rows: list[dict]) -> list[dict]:
    """Ranking protocol: highest mean CLIPScore first; ties (within 0.5) are broken by lower mean generation time."""
    agg = {}
    for r in rows:
        a = agg.setdefault(r["model"], {"model": r["model"], "clip": [], "time": []})
        a["clip"].append(r["clip_score"]); a["time"].append(r["time_s"])
    table = [{"model": k, "mean_clip_score": round(float(np.mean(v["clip"])), 2), "mean_time_s": round(float(np.mean(v["time"])), 2)}
             for k, v in agg.items()]
    table.sort(key=lambda t: (-round(t["mean_clip_score"] / 0.5), t["mean_time_s"]))
    for i, t in enumerate(table, 1):
        t["rank"] = i
    return table


# --------------------------------------------------------------------------- reports
def package_versions() -> dict:
    names = ["torch", "torchvision", "diffusers", "transformers", "peft", "accelerate", "sentence-transformers", "streamlit",
             "plotly", "numpy", "pandas", "scikit-learn", "scikit-image", "Pillow", "torchmetrics", "lpips", "umap-learn"]
    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = "not installed"
    return out


def reproducibility_report(experiment: dict | None = None) -> dict:
    from services import gpu_manager
    info = gpu_manager.get_device_info()
    return {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "python": sys.version.split()[0], "platform": platform.platform(),
            "device": {k: info[k] for k in ("name", "total_gb", "torch_version", "cuda_version", "compute_capability")},
            "packages": package_versions(),
            "model_repos": {k: v["repo"] for k, v in settings.MODEL_REGISTRY.items()},
            "metric_methodology": {
                "CLIPScore": "100 * max(cos(CLIP ViT-B/32 image embedding, CLIP text embedding), 0)",
                "FID": "torchmetrics FrechetInceptionDistance, 2048-d Inception features; unreliable below 2048 images per set",
                "classifier_accuracy": "small CNN trained on real images; generated images of class c are classified; accuracy = fraction labelled c",
                "AUC/ROC": "one-vs-rest per class on the classifier's softmax probabilities; macro AUC = mean over classes",
                "SSIM/PSNR": "only for corresponding image pairs (same content)", "LPIPS": "AlexNet LPIPS on 256x256 resized pairs"},
            "experiment": experiment or {}}


def experiments_to_csv(experiments: list[dict]) -> bytes:
    import pandas as pd
    rows = []
    for e in experiments:
        flat = {"id": e["id"], "kind": e["kind"], "name": e["name"], "created_at": e["created_at"]}
        for k, v in (e["metrics"] or {}).items():
            if isinstance(v, (int, float, str)) or v is None:
                flat[k] = v
        rows.append(flat)
    return pd.DataFrame(rows).to_csv(index=False).encode()


def build_pdf_report(title: str, experiments: list[dict], images: list | None = None, notes: str = "") -> bytes:
    """Evaluation report as PDF (reportlab): protocol, one table per experiment, optional images."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image as RLImage, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title=title, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Paragraph(time.strftime("Generated %Y-%m-%d %H:%M"), styles["Normal"]), Spacer(1, 12)]
    if notes:
        story += [Paragraph(notes, styles["BodyText"]), Spacer(1, 12)]
    for e in experiments:
        story.append(Paragraph(f"{e['name']} ({e['kind']}, {e['created_at']})", styles["Heading2"]))
        rows = [["Metric", "Value"]]
        for k, v in (e["metrics"] or {}).items():
            if isinstance(v, float):
                v = f"{v:.4f}"
            if isinstance(v, (int, float, str)) or v is None:
                rows.append([str(k), str(v)])
        table = Table(rows, colWidths=[7 * cm, 9 * cm])
        table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#6d28d9")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                                   ("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 9)]))
        story += [table, Spacer(1, 14)]
    for img in images or []:
        b = io.BytesIO(); img.convert("RGB").save(b, "PNG"); b.seek(0)
        story += [RLImage(b, width=7 * cm, height=7 * cm * img.height / img.width), Spacer(1, 8)]
    doc.build(story)
    return buf.getvalue()


def training_curves_png(history: list[dict], title: str = "Training curves") -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ep = [h["epoch"] for h in history]
    ax[0].plot(ep, [h["loss_g"] for h in history], label="Generator"); ax[0].plot(ep, [h["loss_d"] for h in history], label="Discriminator")
    ax[0].set_title("Losses"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(ep, [h["d_real"] for h in history], label="D(real)"); ax[1].plot(ep, [h["d_fake"] for h in history], label="D(fake)")
    ax[1].set_title("Discriminator confidence"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.suptitle(title); fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=140); plt.close(fig)
    return buf.getvalue()
