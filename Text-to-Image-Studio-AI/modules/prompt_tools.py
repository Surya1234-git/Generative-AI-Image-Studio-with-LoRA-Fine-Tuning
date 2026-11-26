"""Prompt helpers: rule-based enhancer, random prompt generator, style presets.

The default enhancer is deterministic and runs anywhere (no model download). An optional
model-based enhancer (MagicPrompt, a GPT-2 fine-tuned on prompts) can be used when
`transformers` and internet access are available; the UI reports which one actually ran.
"""
from __future__ import annotations

import random

from config.settings import STYLE_PRESETS

CATEGORIES = {
    "architecture": (("house", "villa", "building", "castle", "temple", "pagoda", "city", "street", "cabin", "tower"),
                     ["architectural photography", "detailed materials and textures", "balanced composition", "crisp lines"]),
    "landscape": (("lake", "mountain", "forest", "river", "ocean", "beach", "valley", "sunset", "sunrise", "waterfall", "landscape", "field"),
                  ["lush scenery", "reflective water", "atmospheric depth", "wide-angle composition"]),
    "portrait": (("portrait", "woman", "man", "girl", "boy", "person", "face", "warrior", "knight", "wizard"),
                 ["expressive eyes", "detailed skin texture", "soft rim light", "shallow depth of field"]),
    "animal": (("dog", "cat", "bird", "horse", "puppy", "butterfly", "fox", "lion", "tiger", "wolf", "owl"),
               ["detailed fur and feathers", "natural pose", "bokeh background", "wildlife photography"]),
    "flower": (("flower", "rose", "tulip", "garden", "blossom", "petal", "daisy", "sunflower"),
               ["macro detail", "delicate petals", "dew drops", "soft natural light"]),
    "scifi": (("robot", "spaceship", "cyberpunk", "futuristic", "alien", "space", "galaxy", "neon"),
              ["intricate technology", "volumetric fog", "neon reflections", "ultra detailed"]),
}
LIGHTING = ["golden-hour lighting", "soft diffused light", "dramatic cinematic lighting", "warm sunset glow", "crisp morning light"]
QUALITY = ["highly detailed", "sharp focus", "professional quality", "8k resolution"]

_SUBJECTS = ["a lighthouse on a cliff", "an astronaut walking in a flower field", "a cozy cabin in a snowy forest",
             "a red fox in autumn woods", "a floating island with waterfalls", "a futuristic city at night",
             "a golden retriever playing in a garden", "an ancient temple surrounded by mist", "a sailing ship in a storm",
             "a steampunk robot reading a book"]
_STYLES = ["cinematic lighting", "oil painting", "studio ghibli inspired", "macro photography", "isometric illustration",
           "dramatic sunset colors", "soft pastel palette", "hyperrealistic"]


def apply_preset(prompt: str, preset: str) -> str:
    suffix = STYLE_PRESETS[preset]
    base = prompt.strip().rstrip(",. ")
    return f"{base}, {suffix}" if base else suffix


def enhance_prompt(prompt: str, rng: random.Random | None = None) -> str:
    """Add lighting, subject-specific detail and quality terms without repeating words already present."""
    rng = rng or random.Random()
    base = prompt.strip().rstrip(",. ")
    if not base:
        return ""
    lower = base.lower()
    extras: list[str] = []
    for words, details in CATEGORIES.values():
        if any(w in lower for w in words):
            extras += rng.sample(details, k=min(3, len(details)))
            break
    if not any(k in lower for k in ("light", "glow", "sunset", "sunrise")):
        extras.append(rng.choice(LIGHTING))
    extras += rng.sample(QUALITY, k=2)
    extras = [e for e in extras if e.lower() not in lower]
    return base + ", " + ", ".join(extras)


def enhance_with_model(prompt: str) -> tuple[str, str]:
    """Try the GPT-2 based enhancer; fall back to rules. Returns (prompt, method_used)."""
    try:
        from transformers import pipeline, set_seed
        gen = pipeline("text-generation", model="Gustavosta/MagicPrompt-Stable-Diffusion")
        set_seed(random.randint(0, 10 ** 6))
        out = gen(prompt, max_length=min(120, len(prompt) + 60), num_return_sequences=1, truncation=True)[0]["generated_text"]
        return out.strip(), "MagicPrompt model"
    except Exception:
        return enhance_prompt(prompt), "rule-based enhancer (model unavailable)"


def random_prompt(rng: random.Random | None = None) -> str:
    rng = rng or random.Random()
    return f"{rng.choice(_SUBJECTS)}, {rng.choice(_STYLES)}, {rng.choice(QUALITY)}"
