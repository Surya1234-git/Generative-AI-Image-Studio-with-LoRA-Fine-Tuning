"""Real GAN training loop with pause / stop / checkpoint / resume (Tasks 2, 5, 6). Requires PyTorch.

Each run lives in checkpoints/gan/<run_name>/ :
    config.json  history.json  latest.pt  ckpt_epoch_XXX.pt  samples/epoch_XXX.png
`latest.pt` stores generator, discriminator, both optimizers, epoch, history and the class list,
so training can be resumed exactly where it stopped.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils

from config import settings
from modules import gan_models
from modules.dataset_explorer import training_root
from services.job_manager import Job, StopRequested


@dataclass
class GANConfig:
    arch: str = "CGAN"
    cond_mode: str = "label"        # label | text
    epochs: int = 50
    batch_size: int = 32
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    z_dim: int = 100
    optimizer: str = "Adam"
    img_size: int = 64
    loss: str = "bce"               # bce | hinge
    base_channels: int = 64
    sample_every: int = 5
    ckpt_every: int = 5
    seed: int = 42
    max_per_class: int | None = None
    dataset_name: str = ""
    dataset_path: str = ""
    run_name: str = ""


def runs_dir() -> Path:
    path = settings.dir_path("checkpoints") / "gan"
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_runs() -> list[dict]:
    out = []
    for d in sorted(runs_dir().iterdir()):
        if (d / "config.json").exists():
            cfg = json.loads((d / "config.json").read_text())
            hist = json.loads((d / "history.json").read_text()) if (d / "history.json").exists() else []
            out.append({"name": d.name, "path": str(d), "config": cfg, "epochs_done": len(hist),
                        "has_checkpoint": (d / "latest.pt").exists(), "history": hist})
    return out


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_optimizer(name: str, params, lr: float):
    if name == "Adam":
        return torch.optim.Adam(params, lr=lr, betas=(0.5, 0.999))
    if name == "RMSprop":
        return torch.optim.RMSprop(params, lr=lr)
    if name == "SGD":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name}")


def build_dataset(dataset_path: str, img_size: int, max_per_class: int | None = None):
    tf = transforms.Compose([transforms.Resize(img_size), transforms.CenterCrop(img_size), transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    ds = datasets.ImageFolder(str(training_root(dataset_path)), transform=tf)
    if max_per_class:
        seen, keep = {}, []
        for i, (_p, y) in enumerate(ds.samples):
            if seen.get(y, 0) < max_per_class:
                keep.append(i)
                seen[y] = seen.get(y, 0) + 1
        ds = torch.utils.data.Subset(ds, keep)
        ds.classes = ds.dataset.classes
    return ds


def tensor_to_pil_grid(t: torch.Tensor, nrow: int) -> Image.Image:
    grid = vutils.make_grid(t.detach().cpu().clamp(-1, 1) * 0.5 + 0.5, nrow=nrow, padding=2)
    return Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8))


def _d_loss(real_logits, fake_logits, kind):
    if kind == "hinge":
        return F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean()
    return (F.binary_cross_entropy_with_logits(real_logits, torch.full_like(real_logits, 0.9)) +   # one-sided label smoothing
            F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits)))


def _g_loss(fake_logits, kind):
    if kind == "hinge":
        return -fake_logits.mean()
    return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))


def save_checkpoint(path: Path, G, D, opt_g, opt_d, epoch: int, cfg: GANConfig, classes, history, text_table):
    torch.save({"G": G.state_dict(), "D": D.state_dict(), "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                "epoch": epoch, "config": asdict(cfg), "classes": classes, "history": history,
                "text_table": text_table.cpu() if text_table is not None else None}, path)


def train_gan(job: Job, cfg: GANConfig, resume: bool = False) -> dict:
    """Runs inside a background thread. Call Job.check() each step so Pause/Stop act at a safe point."""
    device = _device()
    run_dir = runs_dir() / cfg.run_name
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    ds = build_dataset(cfg.dataset_path, cfg.img_size, cfg.max_per_class)
    classes = list(ds.classes)
    if len(classes) < 2:
        raise ValueError("The dataset needs at least 2 class folders for a conditional GAN.")
    workers = 0 if os.name == "nt" else 2   # Windows needs spawn, which does not mix well with Streamlit threads
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=workers,
                        pin_memory=device.type == "cuda", persistent_workers=workers > 0)
    if len(loader) == 0:
        raise ValueError("The batch size is larger than the number of training images.")

    text_table = None
    if cfg.cond_mode == "text":
        from modules.text_embeddings import class_text_table
        job.message = "Encoding class captions with CLIP..."
        text_table = torch.as_tensor(class_text_table(classes))
    G, D = gan_models.build_gan(cfg.arch, len(classes), cfg.z_dim, 128, cfg.base_channels, cfg.img_size, text_table)
    G, D = G.to(device), D.to(device)
    opt_g = make_optimizer(cfg.optimizer, G.parameters(), cfg.lr_g)
    opt_d = make_optimizer(cfg.optimizer, D.parameters(), cfg.lr_d)
    history, start_epoch = [], 0
    if resume and (run_dir / "latest.pt").exists():
        state = torch.load(run_dir / "latest.pt", map_location=device)
        G.load_state_dict(state["G"]); D.load_state_dict(state["D"])
        opt_g.load_state_dict(state["opt_g"]); opt_d.load_state_dict(state["opt_d"])
        history, start_epoch = state["history"], state["epoch"]
        job.log(f"Resumed from epoch {start_epoch}.")
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (run_dir / "classes.json").write_text(json.dumps(classes))
    job.history = list(history)
    job.extra.update(epochs=cfg.epochs, run_dir=str(run_dir), classes=classes, params_g=gan_models.count_parameters(G),
                     params_d=gan_models.count_parameters(D))

    n_show = min(len(classes), 10)
    show_idx = torch.linspace(0, len(classes) - 1, n_show).long()
    gen_fixed = torch.Generator().manual_seed(1234)
    fixed_z = torch.randn(n_show * 8, cfg.z_dim, generator=gen_fixed).to(device)
    fixed_y = show_idx.repeat_interleave(8).to(device)

    def checkpoint(epoch_done, tag=None):
        save_checkpoint(run_dir / "latest.pt", G, D, opt_g, opt_d, epoch_done, cfg, classes, history, text_table)
        if tag:
            save_checkpoint(run_dir / f"ckpt_epoch_{epoch_done:03d}.pt", G, D, opt_g, opt_d, epoch_done, cfg, classes, history, text_table)
            for old in sorted(run_dir.glob("ckpt_epoch_*.pt"))[:-3]:
                old.unlink(missing_ok=True)   # keep the 3 most recent epoch checkpoints
        job.log(f"Checkpoint saved at epoch {epoch_done}.")

    epoch = start_epoch
    try:
        for epoch in range(start_epoch, cfg.epochs):
            G.train(); D.train()
            sums = np.zeros(4)
            t0 = time.time()
            for i, (x, y) in enumerate(loader):
                job.check()
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                z = torch.randn(x.size(0), cfg.z_dim, device=device)
                fake = G(z, y)
                # --- discriminator step: real images should score high, generated images low
                opt_d.zero_grad(set_to_none=True)
                real_logits, fake_logits = D(x, y), D(fake.detach(), y)
                loss_d = _d_loss(real_logits, fake_logits, cfg.loss)
                loss_d.backward(); opt_d.step()
                # --- generator step: try to make the discriminator score generated images high
                opt_g.zero_grad(set_to_none=True)
                loss_g = _g_loss(D(fake, y), cfg.loss)
                loss_g.backward(); opt_g.step()
                sums += [loss_d.item(), loss_g.item(), torch.sigmoid(real_logits).mean().item(), torch.sigmoid(fake_logits).mean().item()]
                job.message = f"Epoch {epoch + 1}/{cfg.epochs} - batch {i + 1}/{len(loader)}"
                job.progress = (epoch + (i + 1) / len(loader)) / cfg.epochs
                if job.checkpoint_requested():
                    checkpoint(epoch)
            avg = sums / len(loader)
            if not np.isfinite(avg).all():
                raise RuntimeError("Training diverged (NaN loss). Lower the learning rate or switch the loss to hinge.")
            history.append({"epoch": epoch + 1, "loss_d": float(avg[0]), "loss_g": float(avg[1]), "d_real": float(avg[2]),
                            "d_fake": float(avg[3]), "seconds": round(time.time() - t0, 1)})
            job.history = list(history)
            (run_dir / "history.json").write_text(json.dumps(history))
            done = epoch + 1
            if done % cfg.sample_every == 0 or done == cfg.epochs:
                G.eval()
                with torch.no_grad():
                    grid = tensor_to_pil_grid(G(fixed_z, fixed_y), nrow=8)
                grid.save(run_dir / "samples" / f"epoch_{done:03d}.png")
                job.extra["latest_sample"] = str(run_dir / "samples" / f"epoch_{done:03d}.png")
            if done % cfg.ckpt_every == 0 or done == cfg.epochs:
                checkpoint(done, tag=True)
            else:
                checkpoint(done)
    except StopRequested:
        checkpoint(epoch)          # leave a resumable checkpoint at the last completed epoch
        raise
    return {"run_dir": str(run_dir), "epochs_done": len(history)}


# --------------------------------------------------------------------------- inference
def load_generator(run_name: str, which: str = "latest.pt"):
    """Return (generator in eval mode, classes, config dict) from a saved run."""
    device = _device()
    state = torch.load(runs_dir() / run_name / which, map_location=device)
    cfg = state["config"]
    table = state.get("text_table")
    G, _ = gan_models.build_gan(cfg["arch"], len(state["classes"]), cfg["z_dim"], 128, cfg["base_channels"], cfg["img_size"], table)
    G.load_state_dict(state["G"])
    return G.to(device).eval(), state["classes"], cfg


@torch.no_grad()
def generate_images(run_name: str, class_indices: list[int], per_class: int = 4, seed: int = 0, ckpt: str = "latest.pt"):
    """Images for the given class indices. Returns (list[PIL], list[class index])."""
    G, classes, cfg = load_generator(run_name, ckpt)
    device = next(G.parameters()).device
    gen = torch.Generator().manual_seed(seed)
    labels = torch.tensor([c for c in class_indices for _ in range(per_class)], device=device)
    z = torch.randn(len(labels), cfg["z_dim"], generator=gen).to(device)
    out = G(z, labels).cpu()
    images = [Image.fromarray(((im.clamp(-1, 1) * 0.5 + 0.5).permute(1, 2, 0).numpy() * 255).astype(np.uint8)) for im in out]
    return images, labels.cpu().tolist()


@torch.no_grad()
def text_to_image(run_name: str, text: str, n: int = 4, seed: int = 0) -> dict:
    """Task 6 pipeline: text -> tokenizer -> CLIP embedding -> generator -> image."""
    from modules.dataset_explorer import class_caption
    from modules.text_embeddings import EMBEDDING_MODELS, cosine_matrix, embed, tokenize
    G, classes, cfg = load_generator(run_name)
    if cfg["cond_mode"] != "text":
        raise ValueError("This run was trained with class labels. Train a run with 'text' conditioning to use free text.")
    model_name = next(iter(EMBEDDING_MODELS))
    tokens = tokenize([text], model_name)[0]
    emb = embed([text], model_name)
    device = next(G.parameters()).device
    gen = torch.Generator().manual_seed(seed)
    z = torch.randn(n, cfg["z_dim"], generator=gen).to(device)
    out = G.from_text_embedding(z, torch.as_tensor(emb).to(device).repeat(n, 1)).cpu()
    images = [Image.fromarray(((im.clamp(-1, 1) * 0.5 + 0.5).permute(1, 2, 0).numpy() * 255).astype(np.uint8)) for im in out]
    caps = [class_caption(c) for c in classes]
    sims = cosine_matrix(np.vstack([emb, embed(caps, model_name)]))[0, 1:]
    best = int(np.argmax(sims))
    return {"tokens": tokens["tokens"], "ids": tokens["ids"], "embedding_shape": list(emb.shape), "images": images,
            "nearest_class": classes[best], "nearest_similarity": float(sims[best])}
