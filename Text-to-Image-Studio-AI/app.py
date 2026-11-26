"""Text-to-Image Studio AI 2.0 - entry point.   Run:  streamlit run app.py

Sidebar -> page router. Each page lives in views/ and calls the matching backend in modules/.
A failing page shows its error without taking the rest of the app down.
"""
import importlib
import sys
import traceback


def _hide_outdated_torchao(min_version: str = "0.16.0") -> None:
    """Colab ships an old optional 'torchao' that newer peft/diffusers reject with an ImportError when a LoRA
    adapter is attached. This project never uses torchao, so an outdated copy is hidden from them."""
    try:
        from importlib import metadata
        from packaging.version import Version
        if "torchao" not in sys.modules and Version(metadata.version("torchao")) < Version(min_version):
            sys.modules["torchao"] = None
    except Exception:
        pass


_hide_outdated_torchao()

import streamlit as st

st.set_page_config(page_title="Text-to-Image Studio AI", page_icon="✨", layout="wide", initial_sidebar_state="expanded")

from components import navigation
from components.ui_components import PAGES, inject_theme, register_shortcuts
from config import settings
from services import storage_manager

ROUTES = {"Home": "views.home", "AI Image Generator": "views.generator", "Dataset Explorer": "views.datasets",
          "Text Embeddings": "views.embeddings", "GAN Studio": "views.gan", "Evaluation Center": "views.evaluation",
          "LoRA Fine-tuning": "views.lora", "Image Gallery": "views.gallery", "About": "views.about",
          "Settings": "views.settings_page"}
# Widget state is dropped by Streamlit when a widget is not drawn; re-assigning keeps each page's inputs while you navigate.
PERSISTENT_PREFIXES = ("gen_", "ds_", "emb_", "gan_", "ev_", "lora_", "gal_")
# Buttons are also bools but cannot be assigned through session_state, so only these boolean widgets (toggles/checkboxes) are kept.
PERSISTENT_BOOLS = {"gen_hq", "gen_upscale", "gen_use_lora", "gen_save", "gen_enh_model", "ds_pp_flip", "ds_sp_use", "gal_fav", "ev_show_all"}
# st.data_editor keys hold an edit log that Streamlit forbids assigning (StreamlitValueAssignmentNotAllowedError), so they are never re-assigned.
NEVER_PERSIST = {"lora_caption_editor"}


def main() -> None:
    settings.ensure_dirs()
    storage_manager.init_db()
    prefs = settings.load_settings()
    st.session_state.setdefault("page", "Home")
    st.session_state.setdefault("theme", prefs["theme"])
    if prefs["retention_days"] and not st.session_state.get("_retention_done"):
        storage_manager.purge_images(int(prefs["retention_days"]))   # data-retention setting (favorites are kept)
        st.session_state["_retention_done"] = True
    for key in list(st.session_state.keys()):
        if key.startswith(PERSISTENT_PREFIXES) and key not in NEVER_PERSIST:
            value = st.session_state[key]
            if isinstance(value, dict) and "edited_rows" in value:   # any other data_editor state
                continue
            if not isinstance(value, bool) or key in PERSISTENT_BOOLS:
                st.session_state[key] = value
    inject_theme(st.session_state["theme"], prefs["accent"])
    navigation.render_sidebar()
    navigation.render_header()
    page = st.session_state["page"]
    navigation.breadcrumbs(page)
    try:
        importlib.import_module(ROUTES[page]).render()
    except Exception as exc:   # keep the shell alive; show what went wrong and where
        st.error(f"This page hit an error: {type(exc).__name__}: {exc}")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())
    register_shortcuts()


main()
