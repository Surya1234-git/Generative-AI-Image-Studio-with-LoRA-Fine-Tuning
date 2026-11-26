"""Home dashboard widgets: recent creations, generation stats, popular features, module shortcuts."""
from __future__ import annotations

import streamlit as st

from components.ui_components import go, panel_title, tile
from services import storage_manager


def creations_strip(limit: int = 6) -> None:
    with st.container(border=True):
        c1, c2 = st.columns([4, 1], vertical_alignment="center")
        with c1:
            panel_title("🖼️", "Your Creations")
        c2.button("View All →", key="home_viewall", on_click=go, args=("Image Gallery",))
        recs = storage_manager.list_images(limit=limit)
        if not recs:
            st.caption("Nothing here yet. Generate your first image and it appears here automatically.")
            return
        for col, rec in zip(st.columns(limit), recs):
            try:
                col.image(rec["path"], use_container_width=True)
                col.caption(("♥ " if rec["favorite"] else "") + (rec["prompt"][:28] or "-"))
            except Exception:
                col.caption("(file missing)")


def stats_panel() -> None:
    res = st.session_state.get("gen_result")
    with st.container(border=True):
        panel_title("📊", "Generation Stats", "Real-time", "ok")
        if not res:
            st.caption("Statistics appear after the first generation in this session.")
            return
        rec = res["records"][0]
        peak = f"{res['peak_gb']:.1f} GB" if res.get("peak_gb") is not None else "n/a"
        c = st.columns(3)
        c[0].markdown(tile("Time", f"{res['elapsed']:.1f}s"), unsafe_allow_html=True)
        c[1].markdown(tile("Size", f"{rec['width']}×{rec['height']}"), unsafe_allow_html=True)
        c[2].markdown(tile("Peak VRAM", peak), unsafe_allow_html=True)


def popular_features() -> None:
    with st.container(border=True):
        panel_title("⭐", "Popular Features")
        for item in ["AI Prompt Enhancer", "Image-to-image and variations", "SD 1.5 vs SDXL comparison", "AI and fast upscaling",
                     "LoRA fine-tuning with before/after", "CGAN vs Attention-CGAN with ROC/AUC"]:
            st.markdown(f"✅ {item}")
        st.button("Explore All →", key="home_explore", on_click=go, args=("About",), type="primary")


def module_cards() -> None:
    cards = [("Dataset Explorer", "Analyze & Explore", "🗃️"), ("Text Embeddings", "Visualize Similarity", "🧊"), ("GAN Studio", "Train & Generate", "🕸️"),
             ("Evaluation Center", "Compare & Analyze", "📈"), ("LoRA Fine-tuning", "Customize Models", "🧬")]
    for col, (name, sub, icon) in zip(st.columns(5), cards):
        with col:
            st.markdown(f"<div class='mod-card'><b>{icon} {name}</b><span>{sub}</span></div>", unsafe_allow_html=True)
            st.button("Open", key=f"card_{name}", on_click=go, args=(name,), use_container_width=True)
