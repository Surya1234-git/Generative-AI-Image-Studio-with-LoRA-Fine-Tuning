"""Settings (opened from the header gear)."""
from __future__ import annotations

import streamlit as st

from components.ui_components import confirm_button, notify
from config import settings
from modules import evaluation as ev
from services import gpu_manager, storage_manager
from services.model_manager import get_model_manager


def render() -> None:
    st.markdown("<div class='page-title'>Settings</div><div class='page-sub'>Appearance, models, GPU, storage, defaults, performance and privacy.</div>", unsafe_allow_html=True)
    prefs = settings.load_settings()
    tabs = st.tabs(["Appearance", "Models", "GPU", "Storage", "Generation", "Performance", "Privacy", "About"])
    with tabs[0]:
        theme = st.radio("Theme", ["dark", "light"], index=["dark", "light"].index(prefs["theme"]), horizontal=True, key="set_theme")
        accent = st.color_picker("Accent color", prefs["accent"], key="set_accent")
        st.caption("The light theme is a best-effort restyle of the dark design.")
        if st.button("Save appearance", key="set_save_app"):
            settings.save_settings({"theme": theme, "accent": accent})
            st.session_state["theme"] = theme
            st.rerun()
    with tabs[1]:
        options = ["auto"] + list(settings.MODEL_REGISTRY)
        default = st.selectbox("Default model", options, index=options.index(prefs["default_model"]) if prefs["default_model"] in options else 0, key="set_model",
                               help="'auto' picks SDXL on GPUs with 14 GB+ and SD 1.5 otherwise.")
        import os
        st.text_input("Hugging Face cache directory (read-only)", os.environ.get("HF_HOME", "~/.cache/huggingface"), disabled=True)
        st.caption("Set the HF_HOME environment variable before launching to store model weights elsewhere (for example on Google Drive).")
        if st.button("Save model settings", key="set_save_model"):
            settings.save_settings({"default_model": default})
            notify("Saved.", "success")
    with tabs[2]:
        info = gpu_manager.get_device_info()
        c = st.columns(4)
        c[0].metric("Device", info["name"])
        c[1].metric("VRAM total", f"{info['total_gb']:.1f} GB")
        c[2].metric("VRAM in use", f"{info['used_gb']:.1f} GB")
        c[3].metric("CUDA / torch", f"{info['cuda_version'] or '-'} / {info['torch_version'] or '-'}")
        profile = gpu_manager.recommended_profile(info)
        st.info(f"{profile['label']}: {profile['note']}")
        b = st.columns(3)
        if b[0].button("Refresh", key="set_gpu_refresh"):
            st.rerun()
        if b[1].button("Unload models", key="set_unload"):
            get_model_manager().unload()
            notify("Models unloaded from GPU memory.", "success")
        if b[2].button("Clear GPU cache", key="set_clear"):
            gpu_manager.clear_memory()
            notify("GPU cache cleared.", "success")
    with tabs[3]:
        use = storage_manager.storage_usage()
        st.text_input("Data directory (read-only)", str(settings.DATA_ROOT), disabled=True)
        st.caption("Change it by setting STUDIO_DATA_DIR before launching (the Colab notebook does this for Google Drive).")
        for sub in settings.SUBDIRS:
            st.write(f"{sub}: {use[sub] / 1024 ** 2:,.1f} MB")
    with tabs[4]:
        steps = st.number_input("Default steps", 1, 150, int(prefs["default_steps"]), key="set_steps")
        guidance = st.number_input("Default guidance", 0.0, 30.0, float(prefs["default_guidance"]), step=0.5, key="set_guid")
        sched = st.selectbox("Default scheduler", settings.SCHEDULER_NAMES, index=settings.SCHEDULER_NAMES.index(prefs["default_scheduler"]), key="set_sched")
        if st.button("Save generation defaults", key="set_save_gen"):
            settings.save_settings({"default_steps": int(steps), "default_guidance": float(guidance), "default_scheduler": sched})
            notify("Saved. New sessions start with these values.", "success")
    with tabs[5]:
        off = st.selectbox("CPU offload", ["auto", "always", "never"], index=["auto", "always", "never"].index(prefs["cpu_offload"]), key="set_off",
                           help="Auto offloads model parts to the CPU only when the request would not fit in VRAM.")
        slicing = st.toggle("Attention slicing (less memory, slightly slower)", prefs["attention_slicing"], key="set_slice")
        safety = st.toggle("Stable Diffusion 1.5 safety checker", prefs["safety_checker"], key="set_safety", help="Off by default, as in the course project. It can black out harmless images.")
        if st.button("Save performance settings", key="set_save_perf"):
            settings.save_settings({"cpu_offload": off, "attention_slicing": slicing, "safety_checker": safety})
            get_model_manager().unload()
            notify("Saved. The model reloads with the new settings on the next generation.", "success")
    with tabs[6]:
        st.markdown("Everything is stored locally (or in your own Google Drive). Nothing is uploaded anywhere by this app.")
        days = st.number_input("Automatically keep images for (days, 0 = forever)", 0, 3650, int(prefs["retention_days"]), key="set_ret")
        if st.button("Save retention", key="set_save_ret"):
            settings.save_settings({"retention_days": int(days)})
        confirm_button("Delete ALL generated images and history now", "set_wipe", lambda: (
            [storage_manager.delete_image(r["id"]) for r in storage_manager.list_images(limit=10 ** 6)], notify("All generated images deleted.", "success"), st.rerun()),
            "This deletes every generated image, including favorites. Datasets and checkpoints are kept.")
    with tabs[7]:
        st.markdown(f"**{settings.APP_NAME} v{settings.APP_VERSION}**")
        st.dataframe([{"package": k, "version": v} for k, v in ev.package_versions().items()], use_container_width=True, hide_index=True)
