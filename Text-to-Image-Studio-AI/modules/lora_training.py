"""Stable Diffusion 1.5 LoRA fine-tuning (Task 1). Requires torch, diffusers, transformers and peft.

What is trained: small low-rank matrices added to the UNet attention layers (to_q, to_k, to_v, to_out).
The 860M-parameter base model stays frozen, so a T4 GPU is enough and the adapter is only a few MB.

Training step (standard latent-diffusion recipe):
  image -> VAE -> latents ; add noise at a random timestep t ; UNet predicts the noise given the caption
  embedding ; loss = MSE(predicted noise, true noise).
SDXL is supported too: the same LoRA layers are trained, but the prompt goes through both SDXL text encoders and the
UNet also receives the pooled text embedding and the image-size "time ids". The frozen VAE runs in fp32 (the SDXL VAE
overflows in fp16). Use resolution 512 on a 15 GB T4 (768 may run out of memory).
"""
from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from config import settings
from modules import lora_registry
from services.job_manager import Job, StopRequested

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class LoRAConfig:
    adapter_name: str = "my_lora"
    base_model_key: str = "Stable Diffusion 1.5"
    rank: int = 8
    alpha: int = 16
    learning_rate: float = 1e-4
    epochs: int = 10
    batch_size: int = 1
    grad_accum: int = 4
    resolution: int = 512
    ckpt_every_epochs: int = 2
    seed: int = 42
    default_caption: str = "a photo"
    flip: bool = True
    gradient_checkpointing: bool = True


# --------------------------------------------------------------------------- data
def collect_items(dataset_root: str | Path, default_caption: str = "") -> list[tuple[str, str]]:
    """(image path, caption). Caption priority: captions.csv > same-name .txt file > class-folder template > default."""
    from modules.dataset_explorer import class_caption
    root = Path(dataset_root)
    table = {}
    csv_path = root / "captions.csv"
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                table[row["filename"]] = row["caption"]
    items = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMAGE_EXTS or "_quarantine" in p.parts:
            continue
        rel = str(p.relative_to(root))
        txt = p.with_suffix(".txt")
        if rel in table and table[rel].strip():
            cap = table[rel].strip()
        elif p.name in table and table[p.name].strip():
            cap = table[p.name].strip()
        elif txt.exists():
            cap = txt.read_text(encoding="utf-8").strip()
        elif p.parent != root:
            cap = class_caption(p.parent.name)
        else:
            cap = default_caption
        items.append((str(p), cap))
    return items


def save_captions(dataset_root: str | Path, rows: list[dict]) -> None:
    with open(Path(dataset_root) / "captions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "caption"])
        writer.writeheader()
        writer.writerows({"filename": r["filename"], "caption": r["caption"]} for r in rows)


def validate_training_data(items: list[tuple[str, str]], resolution: int) -> dict:
    errors, warnings = [], []
    if len(items) < 5:
        errors.append(f"Only {len(items)} images found. LoRA needs at least 5 (15-50 is typical).")
    bad, small, no_caption = 0, 0, 0
    for path, cap in items:
        if not cap.strip():
            no_caption += 1
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                if min(im.size) < resolution // 2:
                    small += 1
        except Exception:
            bad += 1
    if bad:
        errors.append(f"{bad} image(s) are corrupted. Remove them in the Dataset Explorer first.")
    if no_caption:
        errors.append(f"{no_caption} image(s) have no caption.")
    if small:
        warnings.append(f"{small} image(s) are less than half the training resolution and will look blurry after upscaling.")
    if len(items) > 400:
        warnings.append("Large dataset: training will take long. Consider max steps or fewer images for a first test.")
    return {"ok": not errors, "errors": errors, "warnings": warnings, "n_images": len(items)}


# --------------------------------------------------------------------------- SDXL helpers
def sdxl_encode_prompt(text_encoder, text_encoder_2, ids_1, ids_2):
    """SDXL conditioning: penultimate hidden states of both text encoders (concatenated) + pooled embedding of encoder 2."""
    import torch
    with torch.no_grad():
        out1 = text_encoder(ids_1, output_hidden_states=True)
        out2 = text_encoder_2(ids_2, output_hidden_states=True)
        prompt_embeds = torch.cat([out1.hidden_states[-2], out2.hidden_states[-2]], dim=-1)
        pooled = out2.text_embeds
    return prompt_embeds, pooled


def sdxl_time_ids(batch: int, resolution: int, device, dtype):
    """(original h, w, crop top, crop left, target h, w): images are centre-cropped to a square of `resolution`."""
    import torch
    ids = torch.tensor([[resolution, resolution, 0, 0, resolution, resolution]], dtype=dtype, device=device)
    return ids.repeat(batch, 1)


# --------------------------------------------------------------------------- training
def save_lora_adapter(pipe_cls, folder, unet, lora_config) -> None:
    """Write the trained adapter to `folder`.

    The adapter metadata (rank and alpha) is saved too, so a loaded adapter is scaled exactly like it was during
    training (alpha / rank). Without it, diffusers assumes alpha == rank, and an adapter trained with alpha != rank
    (the default here is rank 8, alpha 16) would load at the wrong strength.
    """
    import inspect

    from peft.utils import get_peft_model_state_dict
    kwargs = {}
    if "unet_lora_adapter_metadata" in inspect.signature(pipe_cls.save_lora_weights).parameters:
        kwargs["unet_lora_adapter_metadata"] = lora_config.to_dict()
    pipe_cls.save_lora_weights(save_directory=str(folder), unet_lora_layers=get_peft_model_state_dict(unet),
                               safe_serialization=True, **kwargs)


def train_lora(job: Job, cfg: LoRAConfig, dataset_root: str, resume: bool = False) -> dict:
    import torch
    import torch.nn.functional as F
    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline, UNet2DConditionModel
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from transformers import CLIPTextModel, CLIPTokenizer

    from services import gpu_manager
    if not torch.cuda.is_available():
        raise RuntimeError("LoRA training needs a CUDA GPU. In Colab choose Runtime > Change runtime type > T4 GPU.")
    model_cfg = settings.MODEL_REGISTRY[cfg.base_model_key]
    family = model_cfg["family"]
    if family not in ("sd15", "sdxl"):
        raise RuntimeError("LoRA training is implemented for Stable Diffusion 1.5 and Stable Diffusion XL.")
    sdxl = family == "sdxl"
    items = collect_items(dataset_root, cfg.default_caption)
    check = validate_training_data(items, cfg.resolution)
    if not check["ok"]:
        raise ValueError(" ".join(check["errors"]))

    out_dir = lora_registry.adapters_dir() / lora_registry.sanitize_name(cfg.adapter_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    gpu_manager.clear_memory()
    from services.model_manager import get_model_manager
    get_model_manager().unload()                       # free the generation pipeline before training
    random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    device, repo = "cuda", model_cfg["repo"]
    job.message = f"Loading {cfg.base_model_key} components..."
    tokenizer = CLIPTokenizer.from_pretrained(repo, subfolder="tokenizer")
    tokenizer_2 = text_encoder_2 = None
    if sdxl:
        from transformers import CLIPTextModelWithProjection
        v = {"variant": "fp16"}                        # the same fp16 files the generator downloads, so nothing is fetched twice
        tokenizer_2 = CLIPTokenizer.from_pretrained(repo, subfolder="tokenizer_2")
        text_encoder = CLIPTextModel.from_pretrained(repo, subfolder="text_encoder", torch_dtype=torch.float16, **v).to(device)
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(repo, subfolder="text_encoder_2", torch_dtype=torch.float16, **v).to(device)
        vae = AutoencoderKL.from_pretrained(repo, subfolder="vae", torch_dtype=torch.float32, **v).to(device)   # fp32: the SDXL VAE overflows in fp16
        unet = UNet2DConditionModel.from_pretrained(repo, subfolder="unet", torch_dtype=torch.float16, **v).to(device)
    else:
        text_encoder = CLIPTextModel.from_pretrained(repo, subfolder="text_encoder", torch_dtype=torch.float16).to(device)
        vae = AutoencoderKL.from_pretrained(repo, subfolder="vae", torch_dtype=torch.float16).to(device)
        unet = UNet2DConditionModel.from_pretrained(repo, subfolder="unet", torch_dtype=torch.float16).to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(repo, subfolder="scheduler")
    for m in (text_encoder, text_encoder_2, vae, unet):
        if m is not None:
            m.requires_grad_(False)
    lora_cfg = LoraConfig(r=cfg.rank, lora_alpha=cfg.alpha, init_lora_weights="gaussian",
                          target_modules=["to_k", "to_q", "to_v", "to_out.0"])
    unet.add_adapter(lora_cfg)
    for p in unet.parameters():
        if p.requires_grad:
            p.data = p.data.float()                    # LoRA weights train in fp32, frozen weights stay fp16
    if cfg.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda")

    tf = transforms.Compose([transforms.Resize(cfg.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                             transforms.CenterCrop(cfg.resolution),
                             transforms.RandomHorizontalFlip() if cfg.flip else transforms.Lambda(lambda x: x),
                             transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    class Data(Dataset):
        def __len__(self): return len(items)
        def __getitem__(self, i):
            path, cap = items[i]
            ids = tokenizer(cap, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids[0]
            row = {"pixel_values": tf(Image.open(path).convert("RGB")), "input_ids": ids}
            if sdxl:
                row["input_ids_2"] = tokenizer_2(cap, max_length=tokenizer_2.model_max_length, padding="max_length", truncation=True,
                                                 return_tensors="pt").input_ids[0]
            return row

    loader = DataLoader(Data(), batch_size=cfg.batch_size, shuffle=True, num_workers=0, drop_last=len(items) >= cfg.batch_size)
    start_epoch, step, history = 0, 0, []
    resume_file = out_dir / "resume.pt"
    if resume and resume_file.exists():
        state = torch.load(resume_file, map_location=device)
        set_peft_model_state_dict(unet, state["lora"])
        optimizer.load_state_dict(state["opt"])
        start_epoch, step, history = state["epoch"], state["step"], state["history"]
        job.log(f"Resumed from epoch {start_epoch}.")

    def save(final: bool, epoch_done: int, loss: float, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        pipe_cls = StableDiffusionXLPipeline if sdxl else StableDiffusionPipeline
        save_lora_adapter(pipe_cls, folder, unet, lora_cfg)
        meta = {"adapter_name": out_dir.name, "base_model_key": cfg.base_model_key, "base_repo": repo, "family": family,
                "rank": cfg.rank, "alpha": cfg.alpha, "steps": step, "epochs_done": epoch_done, "final_loss": loss,
                "n_images": len(items), "created": time.strftime("%Y-%m-%d %H:%M:%S"), "config": asdict(cfg), "final": final}
        (folder / "adapter_meta.json").write_text(json.dumps(meta, indent=2))
        (out_dir / "history.json").write_text(json.dumps(history))

    job.extra.update(out_dir=str(out_dir), n_images=len(items), trainable_params=sum(p.numel() for p in params))
    epoch, last_loss = start_epoch, float("nan")
    try:
        for epoch in range(start_epoch, cfg.epochs):
            unet.train()
            running, count = 0.0, 0
            for i, batch in enumerate(loader):
                job.check()
                with torch.no_grad():
                    if sdxl:
                        latents = (vae.encode(batch["pixel_values"].to(device, torch.float32)).latent_dist.sample() * vae.config.scaling_factor).to(torch.float16)
                        cond, pooled = sdxl_encode_prompt(text_encoder, text_encoder_2, batch["input_ids"].to(device), batch["input_ids_2"].to(device))
                    else:
                        latents = vae.encode(batch["pixel_values"].to(device, torch.float16)).latent_dist.sample() * vae.config.scaling_factor
                        cond = text_encoder(batch["input_ids"].to(device))[0]
                noise = torch.randn_like(latents)
                t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
                noisy = noise_scheduler.add_noise(latents, noise, t)
                target = noise if noise_scheduler.config.prediction_type == "epsilon" else noise_scheduler.get_velocity(latents, noise, t)
                with torch.autocast("cuda", dtype=torch.float16):
                    if sdxl:
                        extra = {"text_embeds": pooled, "time_ids": sdxl_time_ids(latents.shape[0], cfg.resolution, device, cond.dtype)}
                        pred = unet(noisy, t, cond, added_cond_kwargs=extra).sample
                    else:
                        pred = unet(noisy, t, cond).sample
                loss = F.mse_loss(pred.float(), target.float())
                scaler.scale(loss / cfg.grad_accum).backward()
                if (i + 1) % cfg.grad_accum == 0 or i + 1 == len(loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                    step += 1
                running += loss.item(); count += 1
                job.message = f"Epoch {epoch + 1}/{cfg.epochs} - batch {i + 1}/{len(loader)} - loss {loss.item():.4f}"
                job.progress = (epoch + (i + 1) / len(loader)) / cfg.epochs
            last_loss = running / max(1, count)
            history.append({"epoch": epoch + 1, "step": step, "loss": last_loss})
            job.history = list(history)
            done = epoch + 1
            if done % cfg.ckpt_every_epochs == 0 and done != cfg.epochs:
                save(False, done, last_loss, out_dir / f"checkpoint-epoch-{done}")
            torch.save({"lora": get_peft_model_state_dict(unet), "opt": optimizer.state_dict(), "epoch": done, "step": step,
                        "history": history}, resume_file)
        save(True, cfg.epochs, last_loss, out_dir)
    except StopRequested:
        torch.save({"lora": get_peft_model_state_dict(unet), "opt": optimizer.state_dict(), "epoch": epoch, "step": step,
                    "history": history}, resume_file)
        raise
    finally:
        del unet, vae, text_encoder, text_encoder_2, optimizer
        gpu_manager.clear_memory()
    write_training_report(out_dir, history, cfg, len(items))
    return {"adapter": out_dir.name, "final_loss": last_loss, "steps": step}


def write_training_report(out_dir: Path, history: list[dict], cfg: LoRAConfig, n_images: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if history:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.plot([h["epoch"] for h in history], [h["loss"] for h in history], marker="o")
        ax.set_xlabel("epoch"); ax.set_ylabel("mean MSE loss"); ax.set_title(f"LoRA training: {out_dir.name}")
        fig.tight_layout(); fig.savefig(out_dir / "loss_curve.png", dpi=140); plt.close(fig)
    lines = [f"# LoRA training report: {out_dir.name}", "", f"- Base model: {cfg.base_model_key}", f"- Images: {n_images}",
             f"- Rank / alpha: {cfg.rank} / {cfg.alpha}", f"- Learning rate: {cfg.learning_rate}", f"- Epochs: {cfg.epochs}",
             f"- Batch size x grad accumulation: {cfg.batch_size} x {cfg.grad_accum}", f"- Resolution: {cfg.resolution}", "",
             "| epoch | optimizer steps | mean loss |", "|---|---|---|"]
    lines += [f"| {h['epoch']} | {h['step']} | {h['loss']:.5f} |" for h in history]
    lines += ["", "A falling loss shows the adapter fits the training images; judge the result by generating test images "
              "(before/after comparison), not by the loss alone."]
    (out_dir / "training_report.md").write_text("\n".join(lines))
