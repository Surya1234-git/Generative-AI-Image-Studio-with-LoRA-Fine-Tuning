"""Persistent sidebar, global header (search, GPU status, theme, notifications, settings) and breadcrumbs."""
from __future__ import annotations

import streamlit as st

from components.ui_components import ICONS, PAGES, go
from config import settings
from services import gpu_manager, storage_manager


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(f"""<div class='brand'><div class='logo'>✦</div>
            <div class='name'>Text-to-Image<br>Studio AI<span class='ver'>v{settings.APP_VERSION}</span></div>
            <div class='tag'>Create • Explore • Innovate</div></div>""", unsafe_allow_html=True)
        current = st.session_state.get("page", "Home")
        for page in PAGES:
            st.button(page, key=f"nav_{page}", icon=ICONS[page], on_click=go, args=(page,),
                      type="primary" if current == page else "secondary", use_container_width=True)
        st.markdown("<div class='side-card'>🏆<b>Generative AI<br>Creative Lab</b><span>Dream. Generate. Share.</span></div>",
                    unsafe_allow_html=True)


@st.cache_data(ttl=5, show_spinner=False)
def _device_info():
    return gpu_manager.get_device_info()


def _toggle_theme() -> None:
    new = "light" if st.session_state.get("theme") == "dark" else "dark"
    st.session_state["theme"] = new
    settings.save_settings({"theme": new})


def render_header() -> None:
    info = _device_info()
    c_search, c_gpu, c_bell, c_set = st.columns([5, 2.6, 0.9, 0.9], vertical_alignment="center")
    query = c_search.text_input("Global search", placeholder="Search prompts, images, datasets and experiments...  (press / to focus)",
                                label_visibility="collapsed", key="global_q", icon=":material/search:")
    label = gpu_manager.device_label(info)
    c_gpu.markdown(f"<div class='gpu-chip {'' if info['cuda'] else 'off'}'><i></i>{label}</div>", unsafe_allow_html=True)
    notes = st.session_state.get("notifications", [])
    count = "9+" if len(notes) > 9 else str(len(notes))
    with c_bell, st.container(key="bell_wrap"):   # the count is drawn as a red badge on the bell (CSS), like YouTube
        if notes:
            st.markdown(f"<style>.st-key-bell_wrap{{--bell-n:\"{count}\"}}</style>", unsafe_allow_html=True)
        bell = st.popover("🔔", help="Completed jobs and errors")
    with bell:
        if not notes:
            st.caption("No notifications yet.")
        for n in notes[:10]:
            st.markdown(f"{'⚠️' if n['kind'] in ('error', 'warning') else '✅' if n['kind'] == 'success' else 'ℹ️'} {n['text']}")
        if notes and st.button("Clear all", key="clear_notes"):
            st.session_state["notifications"] = []
            st.rerun()
    c_set.button("⚙️", key="settings_btn", on_click=go, args=("Settings",), help="Settings")
    if query.strip():
        _search_results(query.strip())


def _search_results(query: str) -> None:
    res = storage_manager.global_search(query)
    total = sum(len(v) for v in res.values())
    with st.expander(f"Search results for “{query}” ({total})", expanded=True):
        if not total:
            st.caption("Nothing matched. Try another keyword.")
        if res["images"]:
            st.markdown("**Images**")
            cols = st.columns(6)
            for col, rec in zip(cols, res["images"][:6]):
                try:
                    col.image(rec["path"], caption=rec["prompt"][:40], use_container_width=True)
                except Exception:
                    col.caption("(file missing)")
            st.button("Open in Image Gallery", key="gs_gallery", on_click=_open_gallery_search, args=(query,))
        for kind, label in (("datasets", "Datasets"), ("experiments", "Experiments"), ("prompts", "Prompts")):
            if res[kind]:
                st.markdown(f"**{label}**")
                for r in res[kind]:
                    st.caption(r.get("name") or r.get("prompt"))


def _open_gallery_search(query: str) -> None:
    st.session_state["gal_search"] = query
    go("Image Gallery")


def breadcrumbs(page: str) -> None:
    st.markdown(f"<div class='crumbs'>Studio / <b>{page}</b></div>", unsafe_allow_html=True)
