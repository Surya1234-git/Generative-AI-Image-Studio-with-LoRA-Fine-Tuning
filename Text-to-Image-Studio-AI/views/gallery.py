"""Image Gallery page (Module 7): search, filters, detail viewer, actions, storage controls."""
from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

from components.ui_components import confirm_button, go, notify, panel_title, to_bytes
from services import storage_manager

PAGE = 24


def _cb_fav(rec_id: int, value: bool) -> None:
    storage_manager.set_favorite(rec_id, value)


def _send_to_generator(rec: dict, kind: str, keep_seed: bool = False) -> None:
    s = st.session_state
    s["gen_apply"], s["gen_apply_keep_seed"] = rec, keep_seed
    if kind:
        s["gen_pending_run"] = kind
    if kind == "variation":
        s["gen_variation_source"] = rec["id"]
    go("AI Image Generator")


@st.dialog("Image details", width="large")
def _detail(rec_id: int) -> None:
    rec = storage_manager.get_image(rec_id)
    if not rec:
        st.error("This image no longer exists.")
        return
    img = storage_manager.load_pil(rec)
    a, b = st.columns([1.3, 1])
    with a:
        if img is not None:
            st.image(img, use_container_width=True)
        else:
            st.error("The image file is missing on disk.")
    with b:
        extra = rec.get("extra") or {}
        rows = [("Prompt", rec["prompt"]), ("Negative prompt", rec["negative_prompt"] or "-"), ("Model", rec["model"]), ("Seed", rec["seed"]),
                ("Resolution", f"{rec['width']} × {rec['height']}"), ("Steps", rec["steps"]), ("Guidance", rec["guidance"]), ("Scheduler", rec["scheduler"]),
                ("LoRA", f"{rec['lora']} × {rec['lora_scale']}" if rec["lora"] else "-"), ("Type", rec["gen_type"]), ("Created", rec["created_at"]),
                ("Generation time", f"{rec['elapsed']} s" if rec["elapsed"] else "-"), ("Peak VRAM", f"{extra['peak_gb']:.1f} GB" if extra.get("peak_gb") else "-"),
                ("File", rec["filename"])]
        st.markdown("\n".join(f"**{k}:** {v}  " for k, v in rows))
    c = st.columns(4)
    if img is not None:
        c[0].download_button("PNG", to_bytes(img, "PNG"), rec["filename"] if rec["filename"].endswith(".png") else rec["filename"] + ".png", "image/png", key=f"gd_png_{rec_id}", use_container_width=True)
        c[1].download_button("JPEG", to_bytes(img, "JPEG"), rec["filename"].rsplit(".", 1)[0] + ".jpg", "image/jpeg", key=f"gd_jpg_{rec_id}", use_container_width=True)
    if c[2].button("♥ Unfavorite" if rec["favorite"] else "♡ Favorite", key=f"gd_fav_{rec_id}", use_container_width=True):
        storage_manager.set_favorite(rec_id, not rec["favorite"])
        st.rerun()
    with c[3].popover("Copy prompt", use_container_width=True):
        st.code(rec["prompt"], language=None, wrap_lines=True)
    d = st.columns(3)
    if d[0].button("Regenerate", key=f"gd_regen_{rec_id}", use_container_width=True):
        _send_to_generator(rec, "regenerate")
        st.rerun()
    if d[1].button("Generate variations", key=f"gd_var_{rec_id}", use_container_width=True):
        _send_to_generator(rec, "variation")
        st.rerun()
    if d[2].button("Reuse settings", key=f"gd_reuse_{rec_id}", use_container_width=True):
        _send_to_generator(rec, "", keep_seed=True)
        st.rerun()
    e = st.columns(2)
    new_tags = e[0].text_input("Tags (comma separated)", ", ".join(t for t in rec["tags"].split(",") if t), key=f"gd_tags_{rec_id}")
    if e[0].button("Save tags", key=f"gd_savetags_{rec_id}"):
        storage_manager.set_tags(rec_id, new_tags.split(","))
        notify("Tags saved.", "success")
        st.rerun()
    new_name = e[1].text_input("Rename file", rec["filename"].rsplit(".", 1)[0], key=f"gd_name_{rec_id}")
    if e[1].button("Rename", key=f"gd_rename_{rec_id}"):
        try:
            storage_manager.rename_image(rec_id, new_name)
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    confirm_button("Delete image", f"gd_del_{rec_id}", lambda: (storage_manager.delete_image(rec_id), notify("Image deleted.", "success"), st.rerun()), "Delete this image file and its metadata permanently?")


def _date_range(choice: str):
    today = date.today()
    if choice == "Today":
        return today, today
    if choice == "Last 7 days":
        return today - timedelta(days=7), today
    if choice == "Last 30 days":
        return today - timedelta(days=30), today
    if choice == "Custom range":
        rng = st.date_input("From / to", (today - timedelta(days=7), today), key="gal_custom_dates")
        if isinstance(rng, tuple) and len(rng) == 2:
            return rng
    return None, None


def render() -> None:
    st.markdown("<div class='page-title'>Image Gallery</div><div class='page-sub'>Everything you generate is stored with its real settings and survives restarts.</div>", unsafe_allow_html=True)
    s = st.session_state
    s.setdefault("gal_search", "")
    with st.container(border=True):
        c = st.columns([2.2, 1.6, 1.4, 1.2])
        c[0].text_input("Search prompts, tags, models", key="gal_search", icon=":material/search:")
        models = c[1].multiselect("Model", storage_manager.distinct_values("model"), key="gal_models")
        types = c[2].multiselect("Generation type", storage_manager.distinct_values("gen_type"), key="gal_types")
        when = c[3].selectbox("Date", ["Any time", "Today", "Last 7 days", "Last 30 days", "Custom range"], key="gal_when")
        d = st.columns([1.4, 1, 1, 1, 1])
        tags = d[0].multiselect("Tags", storage_manager.all_tags(), key="gal_tags")
        minw = d[1].number_input("Min width", 0, 4096, 0, step=64, key="gal_minw")
        maxw = d[2].number_input("Max width (0 = any)", 0, 4096, 0, step=64, key="gal_maxw")
        order = d[3].selectbox("Sort", ["newest", "oldest", "favorites first"], key="gal_order")
        fav = d[4].toggle("Favorites only", key="gal_fav")
        d_from, d_to = _date_range(when)
    flt = dict(search=s.gal_search, models=models, gen_types=types, favorites_only=fav, date_from=d_from, date_to=d_to,
               min_width=minw or None, max_width=maxw or None, tags=tags)
    total = storage_manager.count_images(**flt)
    pages = max(1, -(-total // PAGE))
    top = st.columns([3, 1])
    top[0].caption(f"{total} image(s) match.")
    page = top[1].number_input("Page", 1, pages, 1, key="gal_page") if pages > 1 else 1
    recs = storage_manager.list_images(order=order, limit=PAGE, offset=(page - 1) * PAGE, **flt)
    if not recs:
        st.info("No images match. Generate something in the AI Image Generator, or relax the filters.")
    ncols = 4
    chosen = []
    for row in range(0, len(recs), ncols):
        cols = st.columns(ncols)
        for col, rec in zip(cols, recs[row:row + ncols]):
            with col, st.container(border=True):
                try:
                    st.image(rec["path"], use_container_width=True)
                except Exception:
                    st.warning("File missing")
                st.markdown(f"<div style='font-size:.85rem'>{rec['prompt'][:70] or '-'}</div><div style='color:var(--muted);font-size:.74rem'>{rec['model']} · {rec['created_at'][:10]} · {rec['width']}×{rec['height']}</div>", unsafe_allow_html=True)
                with st.container(key=f"gal_row_{rec['id']}"):
                    b = st.columns([.8, 1.1, 1.3], vertical_alignment="center")
                    b[0].button("♥" if rec["favorite"] else "♡", key=f"gal_f_{rec['id']}", on_click=_cb_fav, args=(rec["id"], not rec["favorite"]), use_container_width=True, help="Favorite")
                    if b[1].button("Open", key=f"gal_o_{rec['id']}", use_container_width=True):
                        _detail(rec["id"])
                    if b[2].checkbox("Cmp", key=f"gal_cmp_{rec['id']}", help="Select two images to compare"):
                        chosen.append(rec)
    if len(chosen) == 2:
        with st.container(border=True):
            panel_title("🆚", "Compare images")
            a, b = st.columns(2)
            for col, rec in zip((a, b), chosen):
                col.image(rec["path"], use_container_width=True)
                col.caption(f"#{rec['id']} · {rec['model']} · seed {rec['seed']} · {rec['steps']} steps · guidance {rec['guidance']} · {rec['scheduler']}")
                col.code(rec["prompt"], language=None, wrap_lines=True)
    elif len(chosen) > 2:
        st.warning("Select exactly two images to compare.")

    with st.expander("Storage, backup and cleanup"):
        use = storage_manager.storage_usage()
        mb = lambda n: f"{n / 1024 ** 2:,.1f} MB"
        st.markdown(f"Images **{mb(use['generated_images'])}** · datasets **{mb(use['datasets'])}** · checkpoints **{mb(use['checkpoints'])}** · LoRA **{mb(use['lora_models'])}** · database **{mb(use['database'])}** · free disk **{use['disk_free'] / 1024 ** 3:,.1f} GB**")
        from config import settings
        st.caption(f"Stored in: {settings.DATA_ROOT}" + ("  (Google Drive: survives Colab disconnects)" if "/content/drive" in str(settings.DATA_ROOT) else "  (local folder: mount Google Drive in Colab to persist it)"))
        if st.button("Back up database now", key="gal_backup"):
            path = storage_manager.backup_database()
            notify(f"Backup saved: {path.name}", "success")
        days = st.number_input("Delete non-favorite images older than (days)", 1, 3650, 90, key="gal_purge_days")
        confirm_button("Delete old images", "gal_purge", lambda: (notify(f"Deleted {storage_manager.purge_images(int(days))} image(s).", "success"), st.rerun()),
                       "Delete all non-favorite images older than the chosen number of days? Favorites are kept.")
