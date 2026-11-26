"""Dataset Explorer page (Module 2 / Task 4)."""
from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from components.ui_components import confirm_button, notify, panel_title, style_fig, tiles, to_bytes
from config import settings
from modules import dataset_explorer as dx
from services import storage_manager


def _ds_dir(name: str) -> Path:
    return settings.dir_path("datasets") / name


def _clean_name(name: str) -> str:
    clean = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip())
    if not clean:
        raise ValueError("Enter a dataset name (letters, numbers, dashes, underscores).")
    return clean


def _register(name: str, path: Path, kind: str, df: pd.DataFrame | None = None, info: dict | None = None) -> None:
    labeled = kind != "unlabeled"
    if df is None:
        df = dx.scan_dataset(path, labeled)
    s = dx.summarize(df)
    storage_manager.register_dataset(name, path, kind, s["total_images"], s["num_classes"], {**(info or {}), "summary": {k: v for k, v in s.items() if k != "class_counts"}})
    st.session_state["ds_df"], st.session_state["ds_active"] = df, name


def _upload_tab() -> None:
    with st.container(border=True):
        panel_title("⬆️", "Upload or import a dataset")
        name = st.text_input("Dataset name", key="ds_new_name", placeholder="flowers")
        source = st.radio("Source", ["ZIP of image folders", "Individual images", "Folder path on this machine / Drive", "Oxford-102 Flowers (built-in)", "Synthetic shapes (square / circle / triangle / star)"], key="ds_source", horizontal=True)
        labeled = st.radio("Labels", ["Labeled: each sub-folder is a class", "Unlabeled: no classes"], key="ds_labeled", horizontal=True)
        is_labeled = labeled.startswith("Labeled")
        if not is_labeled:
            st.caption("Unlabeled mode: all images are treated as one group. Class charts and class-conditional GAN training are not available for it.")
        payload = None
        if source.startswith("ZIP"):
            payload = st.file_uploader("ZIP file", type=["zip"], key="up_ds_zip")
        elif source.startswith("Individual"):
            payload = st.file_uploader("Images", type=["jpg", "jpeg", "png", "webp", "bmp"], accept_multiple_files=True, key="up_ds_files")
            cls = st.text_input("Class name for these images (labeled mode)", key="ds_files_class", placeholder="cats") if is_labeled else None
        elif source.startswith("Folder"):
            folder = st.text_input("Folder path", key="ds_folder_path", placeholder="/content/drive/MyDrive/my_dataset")
        elif source.startswith("Synthetic"):
            picked = st.multiselect("Shapes (one class each)", list(dx.SHAPE_NAMES), default=list(dx.SHAPE_NAMES[:3]), key="ds_shapes_pick")
            shp_n = st.slider("Images per shape", 20, 1000, 200, step=20, key="ds_shapes_n")
            shp_size = st.select_slider("Image size", [32, 64, 128], value=64, key="ds_shapes_size")
            st.caption("Generated locally in seconds (no download). Random colour, size, position and background; captions.csv holds a varied description for every image. "
                       "Ideal for Task 2: the GAN must learn what 'square' vs 'circle' means.")
        else:
            per = st.slider("Max images per class (smaller = faster)", 5, 80, 40, key="ds_oxford_n")
            st.caption("Downloads Oxford-102 Flowers with torchvision (about 330 MB) and writes it as class folders. The dataset has labels only; captions are templated ('a photo of <flower>').")
        if st.button("Upload / Import Dataset", type="primary", key="ds_upload_go"):
            try:
                ds_name = _clean_name(name)
                dest = _ds_dir(ds_name)
                kind = "labeled" if is_labeled else "unlabeled"
                bar = st.progress(0.0, text="Working...")
                if source.startswith("ZIP"):
                    if payload is None:
                        raise ValueError("Choose a ZIP file first.")
                    shutil.rmtree(dest, ignore_errors=True)
                    n = dx.safe_extract_zip(payload.getvalue(), dest)
                    dx.flatten_single_root(dest)
                elif source.startswith("Individual"):
                    if not payload:
                        raise ValueError("Choose at least one image.")
                    n = dx.save_uploaded_files([(f.name, f.getvalue()) for f in payload], dest, (cls or "images") if is_labeled else None)
                elif source.startswith("Folder"):
                    src = Path(folder).expanduser()
                    if not src.is_dir():
                        raise ValueError("That folder does not exist.")
                    dest = src
                    n = sum(1 for p in src.rglob("*") if p.suffix.lower() in dx.IMAGE_EXTS)
                elif source.startswith("Synthetic"):
                    shutil.rmtree(dest, ignore_errors=True)
                    n = dx.generate_shapes_dataset(dest, tuple(picked), int(shp_n), int(shp_size))
                    kind = "labeled"
                else:
                    shutil.rmtree(dest, ignore_errors=True)
                    n = dx.load_oxford_flowers(dest, per, lambda f: bar.progress(min(f, 1.0), text="Copying images..."))
                    kind = "labeled"
                bar.progress(1.0, text="Analyzing...")
                _register(ds_name, dest, kind)
                bar.empty()
                notify(f"Dataset '{ds_name}' ready with {n} images.", "success")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
                notify(str(exc), "error")


def _overview_tab(name: str, rec: dict) -> None:
    root = Path(rec["path"])
    labeled = rec["kind"] != "unlabeled"
    c1, c2 = st.columns([1, 3])
    if c1.button("Analyze Dataset", type="primary", key="ds_analyze"):
        bar = st.progress(0.0, text="Scanning images...")
        df = dx.scan_dataset(root, labeled, lambda f: bar.progress(min(f, 1.0), text="Scanning images..."))
        bar.empty()
        _register(name, root, rec["kind"], df)
        notify(f"Analyzed {len(df)} images.", "success")
    df = st.session_state.get("ds_df")
    if df is None or st.session_state.get("ds_active") != name:
        df = dx.scan_dataset(root, labeled)
        st.session_state["ds_df"], st.session_state["ds_active"] = df, name
    s = dx.summarize(df)
    tiles([("Total images", f"{s['total_images']:,}"), ("Classes", s["num_classes"]), ("Average size", f"{s['avg_width']:.0f} × {s['avg_height']:.0f}"),
           ("Corrupted", s["corrupted"]), ("Duplicates", s["duplicates"]), ("Storage", f"{s['storage_bytes'] / 1024 ** 2:.1f} MB")])
    st.write("")
    a, b, c = st.columns(3)
    with a:
        if s["corrupted"]:
            confirm_button("Remove Corrupted Files", "ds_rm_corrupt", lambda: (notify(f"Moved {dx.remove_corrupted(df, root)} file(s) to _quarantine.", "success"), st.session_state.pop("ds_df"), st.rerun()),
                           "Corrupted files are moved to a _quarantine folder inside the dataset (not deleted).")
        else:
            st.button("Remove Corrupted Files", disabled=True, key="ds_rm_none", help="No corrupted images found.")
    with b:
        ham = st.slider("Near-duplicate sensitivity (0 = exact copies only)", 0, 10, 0, key="ds_ham", help="Hamming distance between 64-bit perceptual hashes.")
    with c:
        st.write("")
        if st.button("Detect Duplicates", key="ds_dups"):
            st.session_state["ds_df"] = df = dx.mark_duplicates(df, ham)
            notify(f"{int(df['is_duplicate'].sum())} duplicate image(s) found.", "success")
            st.rerun()
    if s["duplicates"]:
        with st.expander(f"Duplicate groups ({s['duplicates']} extra copies)"):
            for g, sub in list(df[df["dup_group"] >= 0].groupby("dup_group"))[:10]:
                cols = st.columns(min(6, len(sub)))
                for col, (_, r) in zip(cols, sub.head(6).iterrows()):
                    col.image(r["path"], caption=r["rel"][:28], use_container_width=True)
    report = dx.build_report(df, name, dx.caption_stats(dx.load_captions(root, df)))
    d1, d2 = st.columns(2)
    d1.download_button("Download Dataset Report (JSON)", json.dumps(report, indent=2, default=str), f"{name}_report.json", "application/json", key="ds_rep_json")
    d2.download_button("Download Dataset Report (Markdown)", dx.report_markdown(report), f"{name}_report.md", "text/markdown", key="ds_rep_md")


def _visualize_tab(name: str) -> None:
    df = st.session_state.get("ds_df")
    if df is None or df.empty:
        st.info("Upload and analyze a dataset first.")
        return
    ok = df[~df["corrupted"]]
    c1, c2 = st.columns(2)
    counts = df["cls"].value_counts().sort_index().reset_index()
    counts.columns = ["class", "images"]
    c1.plotly_chart(style_fig(px.bar(counts, x="class", y="images", title="Class distribution")), use_container_width=True)
    fmt = ok["format"].value_counts().reset_index()
    fmt.columns = ["format", "images"]
    c2.plotly_chart(style_fig(px.pie(fmt, names="format", values="images", title="File formats", hole=.45)), use_container_width=True)
    c3, c4 = st.columns(2)
    c3.plotly_chart(style_fig(px.histogram(ok, x="width", nbins=30, title="Image width")), use_container_width=True)
    c4.plotly_chart(style_fig(px.histogram(ok, x="height", nbins=30, title="Image height")), use_container_width=True)
    st.markdown("**Size statistics**")
    st.dataframe(ok[["width", "height", "bytes"]].describe().round(1), use_container_width=True)
    st.markdown("**Sample viewer**")
    classes = ["All classes"] + sorted(df["cls"].unique())
    pick = st.selectbox("Show images of class", classes, key="ds_view_cls")
    pool = ok if pick == "All classes" else ok[ok["cls"] == pick]
    n = st.slider("Images to show", 6, 48, 18, step=6, key="ds_view_n")
    rows = pool.sample(min(n, len(pool)), random_state=int(st.session_state.get("ds_shuffle", 0))) if len(pool) else pool
    st.button("Shuffle", key="ds_shuffle_btn", on_click=lambda: st.session_state.update(ds_shuffle=random.randint(0, 10 ** 6)))
    cols = st.columns(6)
    for i, (_, r) in enumerate(rows.iterrows()):
        cols[i % 6].image(r["path"], caption=f"{r['cls']} · {int(r['width'])}×{int(r['height'])}", use_container_width=True)


def _captions_tab(name: str, rec: dict) -> None:
    """Task 4: description length statistics and images shown together with their text descriptions."""
    df = st.session_state.get("ds_df")
    if df is None or df.empty:
        st.info("Upload and analyze a dataset first.")
        return
    root = Path(rec["path"])
    caps = dx.load_captions(root, df)
    cs = dx.caption_stats(caps)
    tiles([("Captions", f"{cs['n_captions']:,}"), ("Unique captions", f"{cs['n_unique']:,}"), ("Words / caption (mean)", f"{cs['words_mean']:.2f}"),
           ("Words (min – max)", f"{cs['words_min']} – {cs['words_max']}"), ("Characters (mean)", f"{cs['chars_mean']:.1f}")])
    st.write("")
    src = ", ".join(f"{k}: {v}" for k, v in cs["sources"].items())
    st.caption(f"Caption sources: {src}. Priority: captions.csv, then a same-name .txt file, then the class-name template.")
    if cs["templated_only"]:
        st.warning("Every caption here is templated from the class name ('a photo of <class>'), so the length statistics are not meaningful. "
                   "Add a captions.csv (columns: filename, caption) or .txt files next to the images, or use the Synthetic shapes dataset, to analyse real descriptions.")
    caps = caps.assign(words=caps["caption"].str.split().str.len(), characters=caps["caption"].str.len())
    c1, c2 = st.columns(2)
    c1.plotly_chart(style_fig(px.histogram(caps, x="words", title="Caption length (words)")), use_container_width=True)
    per = caps.groupby("cls")["words"].mean().round(2).reset_index()
    per.columns = ["class", "mean words"]
    c2.plotly_chart(style_fig(px.bar(per, x="class", y="mean words", title="Mean caption length per class")), use_container_width=True)
    st.markdown("**Images with their descriptions**")
    pick = st.selectbox("Class", ["All classes"] + sorted(caps["cls"].unique()), key="ds_cap_cls")
    pool = caps if pick == "All classes" else caps[caps["cls"] == pick]
    n = st.slider("Images to show", 6, 24, 12, step=6, key="ds_cap_n")
    rows = pool.sample(min(n, len(pool)), random_state=int(st.session_state.get("ds_cap_shuffle", 0))) if len(pool) else pool
    st.button("Shuffle", key="ds_cap_shuffle_btn", on_click=lambda: st.session_state.update(ds_cap_shuffle=random.randint(0, 10 ** 6)))
    cols = st.columns(6)
    for i, (_, r) in enumerate(rows.iterrows()):
        cols[i % 6].image(r["path"], caption=f"{r['caption']} ({r['words']} words)", use_container_width=True)
    st.download_button("Download caption table (CSV)", caps[["rel", "cls", "caption", "source", "words", "characters"]].to_csv(index=False), f"{name}_captions.csv", "text/csv", key="ds_cap_csv")


def _preprocess_tab(name: str, rec: dict) -> None:
    df = st.session_state.get("ds_df")
    if df is None or df.empty:
        st.info("Upload and analyze a dataset first.")
        return
    c = st.columns(5)
    size = c[0].number_input("Output size", 16, 1024, 64, step=16, key="ds_pp_size")
    mode = c[1].selectbox("Resize mode", ["center_crop", "resize", "random_crop"], key="ds_pp_mode")
    norm = c[2].selectbox("Normalization", ["-1..1", "0..1", "imagenet", "none"], key="ds_pp_norm", help="Applied when training reads the images; recorded in preprocess_config.json.")
    fmt = c[3].selectbox("Format", ["PNG", "JPEG", "WEBP"], key="ds_pp_fmt")
    flip = c[4].toggle("Horizontal flip (50%)", key="ds_pp_flip")
    cfg = dx.PreprocessConfig(size=int(size), mode=mode, flip=flip, normalize=norm, out_format=fmt)
    ok = df[~df["corrupted"]]
    if len(ok):
        idx = st.slider("Preview image #", 0, len(ok) - 1, 0, key="ds_pp_idx")
        src = Image.open(ok.iloc[idx]["path"])
        after = dx.preprocess_image(src, cfg, random.Random(idx))
        p1, p2 = st.columns(2)
        p1.image(src, caption=f"Before: {src.width}×{src.height}", use_container_width=True)
        arr = dx.normalize_array(after, norm)
        p2.image(after, caption=f"After: {after.width}×{after.height} · {fmt} · values {arr.min():.2f}..{arr.max():.2f}", use_container_width=True)
    st.divider()
    panel_title("✂️", "Train / validation / test split")
    s1, s2, s3 = st.columns(3)
    tr = s1.slider("Training %", 50, 90, 70, key="ds_sp_tr")
    va = s2.slider("Validation %", 5, 30, 15, key="ds_sp_va")
    te = 100 - tr - va
    s3.metric("Test %", te)
    seed = st.number_input("Random seed", 0, 10 ** 6, 42, key="ds_sp_seed")
    st.caption("Splits are stratified per class and seeded. Exact and near-duplicate images are kept together in one split so test images never leak into training.")
    use_split = st.toggle("Include the split when applying preprocessing", value=True, key="ds_sp_use")
    if te < 0:
        st.error("Training % + validation % must not exceed 100.")
    if st.button("Split Dataset", key="ds_split_go", disabled=te < 0):
        work = df if df["dup_group"].ge(0).any() else dx.mark_duplicates(df, 4)
        st.session_state["ds_splits"] = dx.split_dataset(work, (tr / 100, va / 100, te / 100), int(seed))
        st.session_state["ds_df"] = work
        notify("Split created.", "success")
    splits = st.session_state.get("ds_splits")
    if splits is not None and len(splits) == len(st.session_state["ds_df"]):
        tab = pd.crosstab(st.session_state["ds_df"]["cls"], splits.where(splits != "", "excluded"))
        st.dataframe(tab, use_container_width=True)
    out_name = st.text_input("Name for the processed dataset", f"{name}_processed", key="ds_out_name")
    if st.button("Apply Preprocessing", type="primary", key="ds_apply"):
        try:
            out = _ds_dir(_clean_name(out_name))
            shutil.rmtree(out, ignore_errors=True)
            bar = st.progress(0.0, text="Processing images...")
            use = splits if (use_split and splits is not None and len(splits) == len(df)) else None
            n = dx.apply_preprocessing(st.session_state["ds_df"], out, cfg, use, lambda f: bar.progress(min(f, 1.0), text="Processing images..."))
            bar.empty()
            _register(_clean_name(out_name), out, "labeled", info={"preprocess": asdict(cfg), "split": use is not None, "source": name})
            notify(f"Wrote {n} processed images to '{out_name}'. It is now selectable in GAN Studio and LoRA Fine-tuning.", "success")
            st.session_state["ds_pick_next"] = _clean_name(out_name)   # applied in render() BEFORE the selectbox exists; assigning ds_pick here would raise
            st.rerun()
        except Exception as exc:
            st.error(str(exc))


def _export_tab(name: str, rec: dict) -> None:
    root = Path(rec["path"])
    st.caption(f"Dataset folder: {root}")
    if st.button("Prepare ZIP for download", key="ds_zip_go"):
        with st.spinner("Zipping..."):
            st.session_state["ds_zip"] = (name, dx.zip_directory(root))
    z = st.session_state.get("ds_zip")
    if z and z[0] == name:
        st.download_button("Export Processed Dataset (ZIP)", z[1], f"{name}.zip", "application/zip", key="ds_zip_dl", type="primary")
    cfg_file = root / "preprocess_config.json"
    if cfg_file.exists():
        st.download_button("Download preprocessing configuration", cfg_file.read_text(), "preprocess_config.json", "application/json", key="ds_cfg_dl")
    st.divider()
    confirm_button("Remove this dataset from the studio", "ds_remove", lambda: (storage_manager.delete_dataset(name, delete_files=False), st.session_state.update(ds_pick_next=""), st.rerun()),
                   "This removes the dataset from the list. Files stay on disk.")


def render() -> None:
    st.markdown("<div class='page-title'>Dataset Explorer</div><div class='page-sub'>Upload, inspect, clean, preprocess and split image datasets. Prepared datasets are reused by GAN Studio and LoRA Fine-tuning.</div>", unsafe_allow_html=True)
    tabs = st.tabs(["Upload", "Overview", "Visualize", "Descriptions", "Preprocess & Split", "Export"])
    with tabs[0]:
        _upload_tab()
    datasets = storage_manager.list_datasets()
    if not datasets:
        for t in tabs[1:]:
            with t:
                st.info("Add a dataset in the Upload tab to begin.")
        return
    names = [d["name"] for d in datasets]
    if "ds_pick_next" in st.session_state:                 # a page action asked to switch dataset: safe here, the widget is not created yet
        st.session_state["ds_pick"] = st.session_state.pop("ds_pick_next")
    if st.session_state.get("ds_pick") not in names:
        st.session_state["ds_pick"] = names[0]
    name = st.selectbox("Active dataset", names, key="ds_pick")
    rec = storage_manager.get_dataset(name)
    if st.session_state.get("ds_active") != name:
        st.session_state.pop("ds_df", None)
        st.session_state.pop("ds_splits", None)
    with tabs[1]:
        _overview_tab(name, rec)
    with tabs[2]:
        _visualize_tab(name)
    with tabs[3]:
        _captions_tab(name, rec)
    with tabs[4]:
        _preprocess_tab(name, rec)
    with tabs[5]:
        _export_tab(name, rec)
