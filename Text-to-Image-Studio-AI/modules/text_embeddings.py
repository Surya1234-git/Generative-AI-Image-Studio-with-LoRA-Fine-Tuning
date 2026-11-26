"""Text tokenization, embeddings and similarity analysis (Module 3 / Task 3).

The maths (cosine similarity, ranking, PCA, UMAP, export) needs only NumPy/scikit-learn.
The encoders are real pretrained models loaded with Hugging Face `transformers` / `sentence-transformers`.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

EMBEDDING_MODELS = {
    "CLIP ViT-B/32 text encoder (512-d)": {"kind": "clip", "repo": "openai/clip-vit-base-patch32"},
    "CLIP ViT-L/14 text encoder (768-d, same family as Stable Diffusion 1.x)": {"kind": "clip", "repo": "openai/clip-vit-large-patch14"},
    "Sentence-Transformers all-MiniLM-L6-v2 (384-d)": {"kind": "st", "repo": "sentence-transformers/all-MiniLM-L6-v2"},
    "Sentence-Transformers all-mpnet-base-v2 (768-d)": {"kind": "st", "repo": "sentence-transformers/all-mpnet-base-v2"},
}
MODEL_EXPLANATIONS = {
    "clip": "CLIP text embeddings live in a space shared with images, which is why image generators condition on them. "
            "Stable Diffusion feeds the *sequence* of 77 token vectors to the UNet; the single pooled vector shown here is "
            "what CLIP uses for image-text matching.",
    "st": "Sentence-Transformers embeddings are general-purpose semantic vectors trained for text-to-text similarity. "
          "They are not aligned with images and are not what image-generation models use.",
}
_CACHE: dict = {}


# --------------------------------------------------------------------------- pure maths
def cosine_matrix(emb: np.ndarray) -> np.ndarray:
    """S[i, j] = (a_i . a_j) / (|a_i| |a_j|)."""
    emb = np.asarray(emb, dtype=np.float64)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = emb / norms
    return np.clip(unit @ unit.T, -1.0, 1.0)


def rank_pairs(sim: np.ndarray, labels: list[str]) -> pd.DataFrame:
    """All unordered pairs sorted from most to least similar."""
    rows = [{"prompt A": labels[i], "prompt B": labels[j], "cosine similarity": float(sim[i, j])}
            for i in range(len(labels)) for j in range(i + 1, len(labels))]
    return pd.DataFrame(rows).sort_values("cosine similarity", ascending=False).reset_index(drop=True)


def pca_2d(emb: np.ndarray) -> np.ndarray:
    from sklearn.decomposition import PCA
    n, d = emb.shape
    if n < 2:
        raise ValueError("PCA needs at least 2 prompts.")
    k = min(2, n, d)
    coords = PCA(n_components=k, random_state=0).fit_transform(emb)
    return coords if k == 2 else np.hstack([coords, np.zeros((n, 1))])


def umap_2d(emb: np.ndarray, seed: int = 42) -> np.ndarray:
    n = emb.shape[0]
    if n < 5:
        raise ValueError("UMAP needs at least 5 prompts to build a neighbour graph. Add more prompts or use PCA.")
    import umap
    return umap.UMAP(n_components=2, n_neighbors=max(2, min(15, n - 1)), min_dist=0.1, random_state=seed).fit_transform(emb)


def kmeans_labels(emb: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    from sklearn.cluster import KMeans
    k = max(1, min(k, emb.shape[0]))
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(emb)


def similarity_agreement(sim_a: np.ndarray, sim_b: np.ndarray) -> float:
    """Spearman correlation between two models' pairwise similarities (do they rank prompt pairs alike?)."""
    from scipy.stats import spearmanr
    iu = np.triu_indices_from(sim_a, k=1)
    if len(iu[0]) < 2:
        return float("nan")
    return float(spearmanr(sim_a[iu], sim_b[iu])[0])


def embeddings_csv(emb: np.ndarray, texts: list[str]) -> bytes:
    df = pd.DataFrame(emb, columns=[f"d{i}" for i in range(emb.shape[1])])
    df.insert(0, "prompt", texts)
    return df.to_csv(index=False).encode()


def embeddings_npy(emb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, emb)
    return buf.getvalue()


def similarity_report_md(texts: list[str], sim: np.ndarray, model: str) -> str:
    pairs = rank_pairs(sim, texts)
    lines = [f"# Similarity report", "", f"Model: {model}", f"Prompts: {len(texts)}", "", "## Pairs (most to least similar)", ""]
    lines += [f"- {r['cosine similarity']:.4f}: \"{r['prompt A']}\" vs \"{r['prompt B']}\"" for _, r in pairs.iterrows()]
    return "\n".join(lines)


# --------------------------------------------------------------------------- real encoders
def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_clip(repo: str):
    if repo not in _CACHE:
        from transformers import CLIPModel, CLIPTokenizer
        model = CLIPModel.from_pretrained(repo).to(_device()).eval()
        _CACHE[repo] = (CLIPTokenizer.from_pretrained(repo), model)
    return _CACHE[repo]


def _load_st(repo: str):
    if repo not in _CACHE:
        from sentence_transformers import SentenceTransformer
        _CACHE[repo] = SentenceTransformer(repo, device=_device())
    return _CACHE[repo]


def tokenize(texts: list[str], model_name: str) -> list[dict]:
    """Show the real tokens and ids the model's tokenizer produces (CLIP pads/truncates to 77)."""
    cfg = EMBEDDING_MODELS[model_name]
    if cfg["kind"] == "clip":
        tok, _ = _load_clip(cfg["repo"])
        out = []
        for t in texts:
            ids = tok(t, truncation=True, max_length=77)["input_ids"]
            out.append({"text": t, "tokens": tok.convert_ids_to_tokens(ids), "ids": ids, "n_tokens": len(ids),
                        "truncated": len(tok(t)["input_ids"]) > 77})
        return out
    model = _load_st(cfg["repo"])
    out = []
    for t in texts:
        enc = model.tokenizer(t, truncation=True)
        out.append({"text": t, "tokens": model.tokenizer.convert_ids_to_tokens(enc["input_ids"]), "ids": enc["input_ids"],
                    "n_tokens": len(enc["input_ids"]), "truncated": False})
    return out


def embed(texts: list[str], model_name: str) -> np.ndarray:
    """Encode prompts -> array of shape [n_prompts, dim] using the selected real model."""
    cfg = EMBEDDING_MODELS[model_name]
    if cfg["kind"] == "clip":
        import torch
        tok, model = _load_clip(cfg["repo"])
        batch = tok(texts, padding=True, truncation=True, max_length=77, return_tensors="pt").to(model.device)
        with torch.no_grad():
            feats = model.get_text_features(**batch)
        # transformers >= 5 returns an output object (embedding in .pooler_output); older versions return the tensor.
        feats = feats if torch.is_tensor(feats) else feats.pooler_output
        return feats.float().cpu().numpy()
    return np.asarray(_load_st(cfg["repo"]).encode(texts, convert_to_numpy=True))


def clip_sequence_shape(text: str, model_name: str) -> dict:
    """The per-token tensor Stable Diffusion actually conditions on: [1, 77, hidden_size]."""
    import torch
    cfg = EMBEDDING_MODELS[model_name]
    if cfg["kind"] != "clip":
        raise ValueError("Only CLIP models have a token-sequence output.")
    tok, model = _load_clip(cfg["repo"])
    batch = tok([text], padding="max_length", truncation=True, max_length=77, return_tensors="pt").to(model.device)
    with torch.no_grad():
        hidden = model.text_model(**batch).last_hidden_state
    return {"shape": list(hidden.shape), "sample": hidden[0, 0, :6].float().cpu().numpy().round(3).tolist()}


def class_text_table(class_names: list[str], model_name: str | None = None) -> np.ndarray:
    """One CLIP embedding per class caption ('a photo of <class>'). Used to condition the GAN on text (Task 6 pipeline)."""
    from modules.dataset_explorer import class_caption
    name = model_name or next(iter(EMBEDDING_MODELS))
    emb = embed([class_caption(c) for c in class_names], name)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)
