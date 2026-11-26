"""Reusable Streamlit UI pieces: theme injection, titles, chips, tiles, notifications, shortcuts."""
from __future__ import annotations

import io
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from config import settings

STYLES = Path(__file__).resolve().parent.parent / "assets" / "styles"

PAGES = ["Home", "AI Image Generator", "Dataset Explorer", "Text Embeddings", "GAN Studio",
         "Evaluation Center", "LoRA Fine-tuning", "Image Gallery", "About"]
ICONS = {"Home": ":material/home:", "AI Image Generator": ":material/auto_awesome:", "Dataset Explorer": ":material/database:",
         "Text Embeddings": ":material/layers:", "GAN Studio": ":material/hub:", "Evaluation Center": ":material/analytics:",
         "LoRA Fine-tuning": ":material/memory:", "Image Gallery": ":material/photo_library:", "About": ":material/info:"}


def inject_theme(theme: str, accent: str) -> None:
    css = (STYLES / "theme.css").read_text()
    if theme == "light":
        css += (STYLES / "light.css").read_text()
    css += f":root{{--accent:{accent}}}"
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def go(page: str) -> None:
    """Callback for navigation buttons."""
    st.session_state["page"] = page


def page_title(title: str, subtitle: str = "") -> None:
    st.markdown(f"<div class='page-title'>{title}</div><div class='page-sub'>{subtitle}</div>", unsafe_allow_html=True)


def panel_title(icon: str, title: str, badge: str = "", tone: str = "gold") -> None:
    chip_html = f"<span class='chip {tone}'>{badge}</span>" if badge else ""
    st.markdown(f"<div class='panel-title'><span>{icon}</span>{title}{chip_html}</div>", unsafe_allow_html=True)


def chip(text: str, tone: str = "") -> str:
    return f"<span class='chip {tone}'>{text}</span>"


def tile(label: str, value) -> str:
    cls = "v long" if len(str(value)) > 6 else "v"   # long values (1024×1024, 10.3 GB) shrink to fit the box
    return f"<div class='tile'><div class='{cls}'>{value}</div><div class='l'>{label}</div></div>"


def tiles(items: list[tuple[str, object]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.markdown(tile(label, value), unsafe_allow_html=True)


def notify(message: str, kind: str = "info") -> None:
    """Toast now + keep in the notification list (header bell)."""
    st.session_state.setdefault("notifications", []).insert(0, {"kind": kind, "text": message})
    st.session_state["notifications"] = st.session_state["notifications"][:30]
    st.toast(message, icon={"success": "✅", "error": "⚠️", "warning": "⚠️"}.get(kind, "ℹ️"))


def confirm_button(label: str, key: str, on_confirm, help_text: str = "This cannot be undone.") -> None:
    """Destructive actions need two clicks: the first shows a confirmation, the second runs `on_confirm`."""
    flag = f"_confirm_{key}"
    if st.session_state.get(flag):
        st.warning(help_text)
        c1, c2 = st.columns(2)
        if c1.button("Yes, continue", key=f"{key}_yes", type="primary"):
            st.session_state[flag] = False
            on_confirm()
        if c2.button("Cancel", key=f"{key}_no"):
            st.session_state[flag] = False
            st.rerun()
    elif st.button(label, key=key):
        st.session_state[flag] = True
        st.rerun()


def to_bytes(img, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt, quality=95) if fmt == "JPEG" else img.save(buf, format=fmt)
    return buf.getvalue()


def style_fig(fig, height: int = 340):
    """Dark, transparent plotly layout that matches the app."""
    light = st.session_state.get("theme") == "light"
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#1d1540" if light else "#ece8ff"), margin=dict(l=10, r=10, t=40, b=10),
                      colorway=["#8b5cf6", "#f0abfc", "#38bdf8", "#34d399", "#fbbf24", "#fb7185"])
    fig.update_xaxes(gridcolor="rgba(139,92,246,.15)")
    fig.update_yaxes(gridcolor="rgba(139,92,246,.15)")
    return fig


def job_status_line(job) -> None:
    tone = {"running": "ok", "paused": "warn", "completed": "ok", "failed": "err", "stopped": "warn", "stopping": "warn"}.get(job.status, "")
    st.markdown(f"{chip(job.status.upper(), tone)} &nbsp; {job.message}", unsafe_allow_html=True)
    st.progress(min(1.0, max(0.0, job.progress)), text=f"{job.progress * 100:.0f}% - {job.elapsed:.0f}s elapsed")
    if job.error:
        st.error(job.error)


def empty_state(text: str) -> None:
    st.info(text)


def register_shortcuts() -> None:
    """Alt+1..9 jump between pages, '/' focuses the search box. Best effort: needs same-origin iframe access."""
    names = [p for p in PAGES]
    js = """<script>
    try{const d=window.parent.document; if(!d.__studioKeys){d.__studioKeys=true;
    d.addEventListener('keydown',e=>{
      const tag=(e.target.tagName||'').toLowerCase(); if(tag==='input'||tag==='textarea') return;
      if(e.key==='/'){e.preventDefault();const s=d.querySelector('input[aria-label="Global search"]'); if(s) s.focus(); return;}
      if(e.altKey && e.key>='1' && e.key<='9'){const names=%s; const label=names[parseInt(e.key)-1];
        const b=[...d.querySelectorAll('section[data-testid="stSidebar"] button')].find(x=>x.innerText.trim().endsWith(label)); if(b){e.preventDefault();b.click();}}
    });}}catch(err){}
    </script>""" % str(names)
    components.html(js, height=0)
