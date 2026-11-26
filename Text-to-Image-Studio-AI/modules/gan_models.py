"""Conditional GAN architectures (Tasks 2, 5 and 6). Requires PyTorch.

  CGAN       : generator + projection discriminator, both conditioned on a class (or on a CLIP text embedding).
  AttnCGAN   : the same networks with a SAGAN self-attention block added at 16x16 resolution in BOTH
               the generator and the discriminator. Nothing else differs, so any change in results is
               attributable to attention.

Conditioning:
  "label" mode -> a learned embedding per class.
  "text"  mode -> a frozen CLIP caption embedding per class passed through a small trainable MLP.
                  Because the generator then consumes CLIP text vectors, new sentences can be fed in at
                  inference time (the text -> embedding -> GAN pipeline).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SN = nn.utils.spectral_norm


class SelfAttention2d(nn.Module):
    """Self-attention over all spatial positions (Zhang et al., SAGAN).

    attn[i, j] = how much output position i looks at input position j. `gamma` starts at 0 so the
    network first learns without attention, then learns how much to use it.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.q = SN(nn.Conv2d(channels, max(1, channels // 8), 1))
        self.k = SN(nn.Conv2d(channels, max(1, channels // 8), 1))
        self.v = SN(nn.Conv2d(channels, channels // 2, 1))
        self.out = SN(nn.Conv2d(channels // 2, channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.keep = False          # set True to store the attention matrix for visualisation
        self.last_attn = None

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        q = self.q(x).view(b, -1, n)
        k = self.k(x).view(b, -1, n)
        v = self.v(x).view(b, -1, n)
        attn = F.softmax(torch.bmm(q.transpose(1, 2), k), dim=-1)           # [b, n, n]
        out = torch.bmm(v, attn.transpose(1, 2)).view(b, c // 2, h, w)
        if self.keep:
            self.last_attn = attn.detach()
        return x + self.gamma * self.out(out)


class ConditionEncoder(nn.Module):
    def __init__(self, num_classes: int, dim: int, text_table: torch.Tensor | None = None):
        super().__init__()
        self.mode = "text" if text_table is not None else "label"
        if self.mode == "label":
            self.embed = nn.Embedding(num_classes, dim)
        else:
            self.register_buffer("text_table", text_table.float())
            self.proj = nn.Sequential(nn.Linear(text_table.shape[1], dim * 2), nn.LeakyReLU(0.2, True), nn.Linear(dim * 2, dim))

    def from_labels(self, labels):
        return self.embed(labels) if self.mode == "label" else self.proj(self.text_table[labels])

    def from_text_embedding(self, emb):
        if self.mode != "text":
            raise ValueError("This model was trained with class labels, not text embeddings.")
        return self.proj(F.normalize(emb.float(), dim=-1))


def _up(cin, cout):
    return nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(cin, cout, 3, padding=1),
                         nn.BatchNorm2d(cout), nn.ReLU(True))


class CGANGenerator(nn.Module):
    """z (noise) + condition -> image in [-1, 1]. Start at 4x4 and double the resolution until img_size."""
    use_attention = False

    def __init__(self, num_classes: int, z_dim=100, cond_dim=128, base=64, img_size=64, text_table=None):
        super().__init__()
        assert img_size in (32, 64, 128), "img_size must be 32, 64 or 128"
        self.z_dim, self.img_size = z_dim, img_size
        self.cond = ConditionEncoder(num_classes, cond_dim, text_table)
        self.base = base
        self.fc = nn.Linear(z_dim + cond_dim, base * 8 * 4 * 4)
        layers, res, ch = [], 4, base * 8
        for _ in range(int(math.log2(img_size // 4))):
            out = max(ch // 2, base)
            layers.append(_up(ch, out))
            res, ch = res * 2, out
            if self.use_attention and res == 16:
                layers.append(SelfAttention2d(ch))
        self.net = nn.Sequential(*layers)
        self.to_rgb = nn.Sequential(nn.BatchNorm2d(ch), nn.ReLU(True), nn.Conv2d(ch, 3, 3, padding=1), nn.Tanh())

    def _decode(self, z, cond):
        h = self.fc(torch.cat([z, cond], dim=1)).view(z.size(0), self.base * 8, 4, 4)
        return self.to_rgb(self.net(h))

    def forward(self, z, labels):
        return self._decode(z, self.cond.from_labels(labels))

    def from_text_embedding(self, z, text_emb):
        return self._decode(z, self.cond.from_text_embedding(text_emb))


class AttnCGANGenerator(CGANGenerator):
    use_attention = True


class CGANDiscriminator(nn.Module):
    """Projection discriminator: score = linear(features) + <class_embedding, features>."""
    use_attention = False

    def __init__(self, num_classes: int, base=64, img_size=64, text_table=None):
        super().__init__()
        layers = [SN(nn.Conv2d(3, base, 4, 2, 1)), nn.LeakyReLU(0.2, True)]
        res, ch = img_size // 2, base
        if self.use_attention and res == 16:
            layers.append(SelfAttention2d(ch))
        for _ in range(int(math.log2(img_size // 4)) - 1):
            out = min(ch * 2, base * 8)
            layers += [SN(nn.Conv2d(ch, out, 4, 2, 1)), nn.LeakyReLU(0.2, True)]
            res, ch = res // 2, out
            if self.use_attention and res == 16:
                layers.append(SelfAttention2d(ch))
        self.features = nn.Sequential(*layers)
        self.linear = SN(nn.Linear(ch, 1))
        self.cond = ConditionEncoder(num_classes, ch, text_table)

    def forward(self, x, labels):
        h = self.features(x).sum(dim=(2, 3))
        return (self.linear(h) + (self.cond.from_labels(labels) * h).sum(dim=1, keepdim=True)).squeeze(1)


class AttnCGANDiscriminator(CGANDiscriminator):
    use_attention = True


ARCHITECTURES = {"CGAN": (CGANGenerator, CGANDiscriminator), "Attention CGAN": (AttnCGANGenerator, AttnCGANDiscriminator)}


def build_gan(arch: str, num_classes: int, z_dim=100, cond_dim=128, base=64, img_size=64, text_table=None):
    g_cls, d_cls = ARCHITECTURES[arch]
    table = torch.as_tensor(text_table) if text_table is not None else None
    return (g_cls(num_classes, z_dim, cond_dim, base, img_size, table), d_cls(num_classes, base, img_size, table))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def attention_overlay(generator: CGANGenerator, z, labels, queries=((4, 4), (8, 8), (12, 12))):
    """Real attention activations from the generator's 16x16 attention layer.

    For each query pixel (row, col) returns the 16x16 map of positions that pixel attends to.
    Returns (images, maps[b, q, 16, 16], res) or None for a model without attention.
    """
    layers = [m for m in generator.modules() if isinstance(m, SelfAttention2d)]
    if not layers:
        return None
    layer = layers[0]
    was_training = generator.training
    generator.eval()
    layer.keep = True
    images = generator(z, labels)
    layer.keep = False
    generator.train(was_training)
    attn = layer.last_attn
    res = int(math.isqrt(attn.shape[-1]))
    idx = [r * res + c for r, c in queries]
    return images, attn[:, idx, :].view(attn.size(0), len(queries), res, res).cpu(), res
