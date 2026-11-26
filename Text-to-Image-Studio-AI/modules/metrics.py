"""Image-quality and classification metrics (Module 5).

Each metric states the input it needs; the UI only offers a metric when its input
requirements are met, and refuses to compute meaningless numbers.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

FID_MIN_IMAGES = 100          # hard minimum we accept
FID_RELIABLE_IMAGES = 2048    # below this the 2048-d covariance is rank-deficient: report as unreliable


def _arr(img: Image.Image, size=None) -> np.ndarray:
    img = img.convert("RGB")
    if size:
        img = img.resize(size, Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def psnr(a: Image.Image, b: Image.Image) -> float:
    """Pixel-level fidelity in dB. Requires two images of the SAME scene (e.g. original vs reconstruction)."""
    from skimage.metrics import peak_signal_noise_ratio
    b = b.resize(a.size, Image.LANCZOS) if a.size != b.size else b
    if np.array_equal(_arr(a), _arr(b)):
        return float("inf")          # identical images: zero error
    return float(peak_signal_noise_ratio(_arr(a), _arr(b), data_range=255))


def ssim(a: Image.Image, b: Image.Image) -> float:
    """Structural similarity in [-1, 1]. Same requirement as PSNR: corresponding images."""
    from skimage.metrics import structural_similarity
    b = b.resize(a.size, Image.LANCZOS) if a.size != b.size else b
    return float(structural_similarity(_arr(a), _arr(b), channel_axis=2, data_range=255))


def lpips_distance(a: Image.Image, b: Image.Image) -> float:
    """Learned perceptual distance (lower = more similar). Requires `pip install lpips`."""
    import lpips
    import torch
    if not hasattr(lpips_distance, "_net"):
        lpips_distance._net = lpips.LPIPS(net="alex", verbose=False)
    def prep(im):
        return torch.from_numpy(_arr(im, (256, 256))).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1
    with torch.no_grad():
        return float(lpips_distance._net(prep(a), prep(b)).item())


def clip_scores(images: list[Image.Image], texts: list[str], repo: str = "openai/clip-vit-base-patch32") -> list[float]:
    """CLIPScore = 100 * max(cosine(image_embedding, text_embedding), 0) per (image, prompt) pair.

    Measures prompt-image agreement only; it does not measure realism or aesthetic quality.
    """
    import torch
    from transformers import CLIPModel, CLIPProcessor
    if len(images) != len(texts):
        raise ValueError("Provide exactly one prompt per image.")
    if not hasattr(clip_scores, "_cache"):
        clip_scores._cache = {}
    if repo not in clip_scores._cache:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        clip_scores._cache[repo] = (CLIPModel.from_pretrained(repo).to(dev).eval(), CLIPProcessor.from_pretrained(repo), dev)
    model, proc, dev = clip_scores._cache[repo]
    batch = proc(text=texts, images=[i.convert("RGB") for i in images], return_tensors="pt", padding=True, truncation=True, max_length=77).to(dev)
    with torch.no_grad():
        img = model.get_image_features(pixel_values=batch["pixel_values"])
        txt = model.get_text_features(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        # transformers >= 5 returns an output object (embedding in .pooler_output); older versions return the tensor.
        img = img if torch.is_tensor(img) else img.pooler_output
        txt = txt if torch.is_tensor(txt) else txt.pooler_output
        img, txt = torch.nn.functional.normalize(img, dim=-1), torch.nn.functional.normalize(txt, dim=-1)
    return (100 * (img * txt).sum(-1).clamp(min=0)).cpu().tolist()


def fid_score(real: list[Image.Image], fake: list[Image.Image]) -> dict:
    """Frechet Inception Distance (torchmetrics). Needs a meaningful reference set and enough images."""
    if len(real) < FID_MIN_IMAGES or len(fake) < FID_MIN_IMAGES:
        raise ValueError(f"FID needs at least {FID_MIN_IMAGES} real and {FID_MIN_IMAGES} generated images "
                         f"(you have {len(real)} and {len(fake)}).")
    import torch
    from torchmetrics.image.fid import FrechetInceptionDistance
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    metric = FrechetInceptionDistance(feature=2048, normalize=False).to(dev)
    def to_batch(imgs):
        return torch.from_numpy(np.stack([_arr(i, (299, 299)) for i in imgs])).permute(0, 3, 1, 2)
    for imgs, is_real in ((real, True), (fake, False)):
        for k in range(0, len(imgs), 64):
            metric.update(to_batch(imgs[k:k + 64]).to(dev), real=is_real)
    n = min(len(real), len(fake))
    return {"fid": float(metric.compute().item()), "n_real": len(real), "n_fake": len(fake),
            "reliable": n >= FID_RELIABLE_IMAGES,
            "note": "" if n >= FID_RELIABLE_IMAGES else
            f"Fewer than {FID_RELIABLE_IMAGES} images per set: treat this FID as a rough comparison only, never against published numbers."}


# --------------------------------------------------------------------------- classification metrics
def classification_metrics(y_true, probs, class_names: list[str]) -> dict:
    """Accuracy, confusion matrix, one-vs-rest ROC curves and AUC for a multi-class classifier."""
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
    y_true = np.asarray(y_true)
    probs = np.asarray(probs, dtype=np.float64)
    y_pred = probs.argmax(axis=1)
    k = len(class_names)
    out = {"accuracy": float(accuracy_score(y_true, y_pred)), "n": int(len(y_true)),
           "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(k))).tolist(),
           "per_class_auc": {}, "roc": {}, "macro_auc": None}
    aucs = []
    for c in range(k):
        positive = (y_true == c).astype(int)
        if positive.sum() == 0 or positive.sum() == len(positive):
            continue                    # AUC is undefined when a class is absent or is the only class
        fpr, tpr, _ = roc_curve(positive, probs[:, c])
        auc = float(roc_auc_score(positive, probs[:, c]))
        out["per_class_auc"][class_names[c]] = auc
        out["roc"][class_names[c]] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        aucs.append(auc)
    if aucs:
        out["macro_auc"] = float(np.mean(aucs))
    return out
