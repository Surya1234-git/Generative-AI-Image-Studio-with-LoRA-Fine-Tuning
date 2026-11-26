"""LoRA Fine-tuning page (Task 1): dataset + captions, training, before/after test, adapter management."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go_
import streamlit as st

from components.ui_components import confirm_button, go, job_status_line, notify, panel_title, style_fig, tiles, to_bytes
from config import settings
from modules import dataset_explorer as dx
from modules import image_generator as ig
from modules import lora_registry, lora_training
from services import gpu_manager, storage_manager
from services.job_manager import JOBS, JobBusyError
from services.model_manager import get_model_manager


def _init() -> None:
    d = {"lora_name": "my_style", "lora_rank": 8, "lora_alpha": 16, "lora_lr": 1e-4, "lora_epochs": 10, "lora_batch": 1, "lora_accum": 4,
         "lora_res": 512, "lora_ckpt": 2, "lora_seed": 42, "lora_default_caption": "a photo", "lora_job": None, "lora_base": "Stable Diffusion 1.5",
         "lora_test_prompt": "a portrait of a cat, detailed", "lora_test_seed": 42, "lora_test_scale": 1.0, "lora_test_steps": 25}
    for k, v in d.items():
        st.session_state.setdefault(k, v)


def _cb_pause(): j = JOBS.get(st.session_state.lora_job); j and j.pause()
def _cb_resume(): j = JOBS.get(st.session_state.lora_job); j and j.resume()
def _cb_stop(): j = JOBS.get(st.session_state.lora_job); j and j.stop()


def _load_into_generator(name: str, model_key: str) -> None:
    s = st.session_state
    s.gen_use_lora, s.gen_lora, s.gen_model = True, name, model_key
    go("AI Image Generator")


def _dashboard_tab() -> None:
    info = gpu_manager.get_device_info()
    adapters = lora_registry.list_adapters()
    tiles([("Base model", "SD 1.5"), ("Adapters", len(adapters)), ("GPU", info["name"] if info["cuda"] else "none"),
           ("Last trained", adapters[-1]["name"] if adapters else "-")])
    if not info["cuda"]:
        st.warning("No CUDA GPU detected: LoRA training needs one (Colab free T4 is enough). You can still manage adapters here.")
    st.markdown("**How it works.** LoRA freezes Stable Diffusion's ~860 M parameters and trains small low-rank matrices inside the UNet attention layers (rank 8 ≈ 1.6 M trainable parameters). The result is a small adapter file you can switch on or off at generation time.")
    st.caption("Training works for Stable Diffusion 1.5 and Stable Diffusion XL. Pick the base model you generate with: an adapter only works with the model family it was trained on. SDXL needs more VRAM (use resolution 512 on a 15 GB T4).")
    if adapters:
        st.dataframe(pd.DataFrame([{"adapter": a["name"], "base": a["base_model"], "rank": a["rank"], "steps": a["steps"],
                                    "final loss": round(a["final_loss"], 4) if a["final_loss"] else None, "created": a["created"]} for a in adapters]),
                     use_container_width=True, hide_index=True)
    job = JOBS.latest("lora")
    if job:
        st.markdown(f"Latest training job: **{job.name}** · {job.status}")


def _dataset_tab():
    s = st.session_state
    datasets = storage_manager.list_datasets()
    with st.container(border=True):
        panel_title("🗂️", "Choose images")
        up = st.file_uploader("Upload new training images (or pick an existing dataset below)", type=["png", "jpg", "jpeg", "webp", "zip", "csv", "txt"], accept_multiple_files=True, key="up_lora_imgs",
                              help="Images, or one ZIP containing images + captions.csv (columns: filename, caption).")  # def_zip_upload_ok
        c = st.columns([2, 1])
        new_name = c[0].text_input("Name for the uploaded set", "style_images", key="lora_up_name")
        if c[1].button("Save upload as dataset", key="lora_up_go", disabled=not up):
            dest = settings.dir_path("datasets") / "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in new_name)
            import shutil
            shutil.rmtree(dest, ignore_errors=True)            # re-saving a name replaces the old set instead of mixing images
            dest.mkdir(parents=True, exist_ok=True)
            n = 0
            for f in up:
                ext = Path(f.name).suffix.lower()
                if ext == ".zip":
                    try:
                        n += dx.safe_extract_zip(f.getvalue(), dest)
                    except Exception as exc:
                        notify(f"Could not read '{f.name}': {exc}", "error")
                elif ext in dx.IMAGE_EXTS:
                    (dest / Path(f.name).name).write_bytes(f.getvalue()); n += 1
                elif f.name.lower() == "captions.csv" or ext == ".txt":
                    (dest / Path(f.name).name).write_bytes(f.getvalue())
            dx.flatten_single_root(dest)
            if n == 0:
                notify("No images found in the upload.", "error")
                st.stop()
            df = dx.scan_dataset(dest, labeled=False)
            sm = dx.summarize(df)
            storage_manager.register_dataset(dest.name, dest, "unlabeled", sm["total_images"], 0, {})
            s["lora_dataset"] = dest.name
            notify(f"Saved {n} images as '{dest.name}'.", "success")
            st.rerun()
        datasets = storage_manager.list_datasets()
        if not datasets:
            st.info("No datasets yet. Upload images above or use Dataset Explorer.")
            return None
        names = [d["name"] for d in datasets]
        if s.get("lora_dataset") not in names:
            s["lora_dataset"] = names[0]
        st.selectbox("Dataset (shared with Dataset Explorer)", names, key="lora_dataset")
    rec = storage_manager.get_dataset(s["lora_dataset"])
    root = Path(rec["path"])
    st.text_input("Default caption for images without one", key="lora_default_caption", help="For a style adapter use a consistent phrase such as 'in my-style'.")
    items = lora_training.collect_items(root, s.lora_default_caption)
    st.caption(f"{len(items)} images. Caption priority: captions.csv > same-name .txt > class-folder template > default caption.")
    cols = st.columns(6)
    for col, (path, cap) in zip(cols, items[:6]):
        col.image(path, caption=cap[:30], use_container_width=True)
    base = [(str(Path(p).relative_to(root)), c) for p, c in items]
    edited = st.data_editor(pd.DataFrame(base, columns=["filename", "caption"]), use_container_width=True, hide_index=True, num_rows="fixed", key=f"capedit_{s.lora_dataset}",   # not a persisted prefix; one editor state per dataset
                            
                            disabled=["filename"], height=min(420, 40 + 35 * min(len(base), 10)))
    c1, c2 = st.columns(2)
    if c1.button("Save captions", key="lora_save_caps"):
        lora_training.save_captions(root, edited.to_dict("records"))
        notify("captions.csv saved in the dataset folder.", "success")
    if c2.button("Validate dataset", key="lora_validate"):
        res = lora_training.validate_training_data(lora_training.collect_items(root, s.lora_default_caption), int(s.lora_res))
        for e in res["errors"]:
            st.error(e)
        for w in res["warnings"]:
            st.warning(w)
        if res["ok"]:
            st.success(f"{res['n_images']} images validated. Ready to train.")
    return rec


def _train_tab(rec: dict | None) -> None:
    s = st.session_state
    left, right = st.columns([1, 1.4])
    job = JOBS.get(s.lora_job)
    active = bool(job and job.is_active())
    with left, st.container(border=True):
        panel_title("⚙️", "Training configuration")
        if rec is None:
            st.info("Choose a dataset in the 'Dataset & captions' tab first.")
        st.text_input("Adapter name", key="lora_name")
        st.selectbox("Base model", list(settings.MODEL_REGISTRY), key="lora_base")
        sdxl = settings.MODEL_REGISTRY[s.lora_base]["family"] == "sdxl"
        if sdxl:
            st.info("SDXL training loads both text encoders and needs more VRAM than SD 1.5. Keep resolution at 512 and batch size at 1 on a 15 GB T4.")
        a = st.columns(3)
        a[0].number_input("Rank", 1, 128, key="lora_rank", help="Adapter capacity. 4-16 is typical.")
        a[1].number_input("Alpha", 1, 256, key="lora_alpha", help="LoRA scaling; alpha = 2 × rank is a common start.")
        a[2].number_input("Learning rate", 1e-6, 1e-2, step=1e-5, format="%.5f", key="lora_lr")
        b = st.columns(3)
        b[0].number_input("Epochs", 1, 200, key="lora_epochs")
        b[1].number_input("Batch size", 1, 8, key="lora_batch")
        b[2].number_input("Gradient accumulation", 1, 32, key="lora_accum")
        c = st.columns(3)
        c[0].selectbox("Resolution", [256, 384, 512, 640, 768], key="lora_res")
        c[1].number_input("Checkpoint every N epochs", 1, 50, key="lora_ckpt")
        c[2].number_input("Seed", 0, 10 ** 6, key="lora_seed")
        info = gpu_manager.get_device_info()
        if info["cuda"] and info["total_gb"] < 12 and (s.lora_res > 512 or s.lora_batch > 1):
            st.warning("With under 12 GB of VRAM keep resolution at 512 and batch size at 1 to avoid out-of-memory errors.")
        elif sdxl and info["cuda"] and info["total_gb"] < 20 and (s.lora_res > 512 or s.lora_batch > 1):
            st.warning("SDXL on a 15 GB GPU: resolution above 512 or batch size above 1 will probably run out of memory.")
        if st.button("Start Training", type="primary", key="lora_start", disabled=active or rec is None, use_container_width=True):
            try:
                name = lora_registry.sanitize_name(s.lora_name)
                if (lora_registry.adapters_dir() / name / "adapter_meta.json").exists():
                    st.error(f"An adapter named '{name}' already exists. Choose another name or delete it first.")
                else:
                    cfg = lora_training.LoRAConfig(adapter_name=name, base_model_key=s.lora_base, rank=int(s.lora_rank), alpha=int(s.lora_alpha),
                                                   learning_rate=float(s.lora_lr), epochs=int(s.lora_epochs), batch_size=int(s.lora_batch), grad_accum=int(s.lora_accum),
                                                   resolution=int(s.lora_res), ckpt_every_epochs=int(s.lora_ckpt), seed=int(s.lora_seed), default_caption=s.lora_default_caption)
                    job = JOBS.start("lora", f"LoRA training: {name}", lora_training.train_lora, cfg=cfg, dataset_root=rec["path"], resume=False)
                    s.lora_job = job.id
                    notify(f"LoRA training '{name}' started.", "success")
                    st.rerun()
            except (JobBusyError, ValueError) as exc:
                st.error(str(exc))
        c1, c2, c3 = st.columns(3)
        c1.button("Pause", key="lora_pause", on_click=_cb_pause, disabled=not active, use_container_width=True)
        c2.button("Resume", key="lora_resume_btn", on_click=_cb_resume, disabled=not active, use_container_width=True)
        c3.button("Stop", key="lora_stop", on_click=_cb_stop, disabled=not active, use_container_width=True)
        resumable = [a for a in lora_registry.list_adapters() if (Path(a["path"]) / "resume.pt").exists()]
        if resumable and not active and rec is not None:
            st.divider()
            pick = st.selectbox("Recover an interrupted training", [a["name"] for a in resumable], key="lora_resume_pick")
            if st.button("Resume from last saved epoch", key="lora_do_resume"):
                a = next(x for x in resumable if x["name"] == pick)
                saved = {**lora_training.LoRAConfig().__dict__, **(a["meta"].get("config") or {})}
                cfg = lora_training.LoRAConfig(**{k: v for k, v in saved.items() if k in lora_training.LoRAConfig.__dataclass_fields__})
                try:
                    job = JOBS.start("lora", f"LoRA training (resumed): {pick}", lora_training.train_lora, cfg=cfg, dataset_root=rec["path"], resume=True)
                    s.lora_job = job.id
                    st.rerun()
                except JobBusyError as exc:
                    st.error(str(exc))
    with right, st.container(border=True):
        panel_title("📉", "Training monitor")

        @st.fragment(run_every=3 if active else None)
        def live():
            j = JOBS.get(st.session_state.lora_job)
            if j is None:
                st.info("No LoRA training in this session yet.")
                return
            job_status_line(j)
            if j.extra:
                st.caption(f"{j.extra.get('n_images', '?')} images · {j.extra.get('trainable_params', 0) / 1e6:.2f} M trainable parameters")
            if j.history:
                fig = go_.Figure(go_.Scatter(x=[h["epoch"] for h in j.history], y=[h["loss"] for h in j.history], mode="lines+markers"))
                fig.update_layout(title="Mean training loss per epoch", xaxis_title="epoch")
                st.plotly_chart(style_fig(fig, 300), use_container_width=True)
            if j.logs:
                with st.expander("Job log"):
                    st.code("\n".join(j.logs[-12:]), language=None)
            was = st.session_state.get("_lora_prev_active")
            st.session_state["_lora_prev_active"] = j.is_active()
            if was and not j.is_active():
                notify(f"{j.name}: {j.status}.", "success" if j.status == "completed" else "warning" if j.status == "stopped" else "error")
                st.rerun()
        live()


def _test_tab() -> None:
    s = st.session_state
    adapters = [a for a in lora_registry.list_adapters() if lora_registry.check_compatibility(a["path"], "Stable Diffusion 1.5")[0]]
    if not adapters:
        st.info("Train or import an SD 1.5 adapter first. (An SDXL adapter can be tested from the AI Image Generator with SDXL selected.)")
        return
    c = st.columns([2, 1, 1])
    name = c[0].selectbox("Adapter", [a["name"] for a in adapters], key="lora_test_adapter")
    scale = c[1].slider("Adapter scale", 0.0, 2.0, step=0.05, key="lora_test_scale")
    seed = c[2].number_input("Seed", 0, 2 ** 31, key="lora_test_seed")
    st.text_area("Test prompt", key="lora_test_prompt", height=70)
    st.number_input("Steps", 10, 60, key="lora_test_steps")
    if st.button("Generate before / after", type="primary", key="lora_test_go"):
        path = next(a["path"] for a in adapters if a["name"] == name)
        try:
            with st.spinner("Generating with the base model, then with the adapter (same prompt and seed)..."):
                base_req = ig.GenerationRequest(prompt=s.lora_test_prompt, model_key="Stable Diffusion 1.5", seed=int(seed), steps=int(s.lora_test_steps), gen_type="lora_base")
                tuned_req = ig.GenerationRequest(prompt=s.lora_test_prompt, model_key="Stable Diffusion 1.5", seed=int(seed), steps=int(s.lora_test_steps),
                                                 lora_dir=path, lora_scale=float(scale), gen_type="lora_test")
                base, tuned = ig.generate(base_req), ig.generate(tuned_req)
            scores = None
            try:
                from modules import metrics
                scores = metrics.clip_scores([base.images[0], tuned.images[0]], [s.lora_test_prompt] * 2)
            except Exception:
                pass
            s["lora_cmp"] = {"base": base.images[0], "tuned": tuned.images[0], "scale": scale, "seed": int(seed), "name": name, "scores": scores}
            if scores:
                storage_manager.add_experiment("lora_compare", f"{name} vs base", {"adapter": name, "scale": scale, "seed": int(seed), "prompt": s.lora_test_prompt},
                                               {"clip_base": scores[0], "clip_lora": scores[1]})
        except Exception as exc:
            st.error(f"Generation failed: {exc}")
    cmp = s.get("lora_cmp")
    if cmp:
        a, b = st.columns(2)
        a.image(cmp["base"], caption="Original base model", use_container_width=True)
        b.image(cmp["tuned"], caption=f"With LoRA '{cmp['name']}' · scale {cmp['scale']} · seed {cmp['seed']}", use_container_width=True)
        if cmp["scores"]:
            st.caption(f"CLIPScore (prompt alignment): base {cmp['scores'][0]:.1f} · LoRA {cmp['scores'][1]:.1f}. A style adapter may lower prompt alignment while changing the look; judge visually as well.")


def _manage_tab() -> None:
    adapters = lora_registry.list_adapters()
    if not adapters:
        st.info("No adapters saved yet.")
        return
    for a in adapters:
        with st.container(border=True):
            ok, msg = (True, "") if a["family"] else (False, "Unknown base model")
            st.markdown(f"**{a['name']}** &nbsp; `{a['base_model']}` · rank {a['rank']} · {a['steps']} steps · loss {a['final_loss'] if a['final_loss'] is None else round(a['final_loss'], 4)} · {a['created']}")
            c = st.columns(6)
            model_key = next((k for k, v in settings.MODEL_REGISTRY.items() if v["family"] == a["family"]), None)
            c[0].button("Load into generator", key=f"lm_load_{a['name']}", on_click=_load_into_generator, args=(a["name"], model_key), disabled=model_key is None, use_container_width=True)
            if c[1].button("Unload", key=f"lm_unload_{a['name']}", help="Detach any adapter from the loaded pipeline", use_container_width=True):
                get_model_manager().clear_lora()
                notify("Adapter unloaded from the active pipeline.", "success")
            c[2].download_button("Export", lora_registry.export_adapter_zip(a["name"]), f"{a['name']}_lora.zip", "application/zip", key=f"lm_exp_{a['name']}", use_container_width=True)
            report = Path(a["path"]) / "training_report.md"
            if report.exists():
                c[3].download_button("Report", report.read_text(), f"{a['name']}_training_report.md", "text/markdown", key=f"lm_rep_{a['name']}", use_container_width=True)
            with c[4].popover("Rename", use_container_width=True):
                new = st.text_input("New name", key=f"lm_newname_{a['name']}")
                if st.button("Rename adapter", key=f"lm_rn_{a['name']}"):
                    try:
                        lora_registry.rename_adapter(a["name"], new)
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))
            with c[5]:
                confirm_button("Delete", f"lm_del_{a['name']}", lambda n=a["name"]: (lora_registry.delete_adapter(n), st.rerun()), f"Permanently delete adapter '{a['name']}' and its checkpoints?")
            curve = Path(a["path"]) / "loss_curve.png"
            if curve.exists():
                st.image(str(curve), width=380)


def render() -> None:
    st.markdown("<div class='page-title'>LoRA Fine-tuning</div><div class='page-sub'>Personalize Stable Diffusion 1.5 with your own images, then compare the base model and the adapter on the same prompt and seed.</div>", unsafe_allow_html=True)
    _init()
    try:
        import peft  # noqa: F401
        import diffusers  # noqa: F401
    except Exception:
        st.warning("peft/diffusers are not installed here, so training is disabled. Adapter management still works.")
    tabs = st.tabs(["Dashboard", "Dataset & captions", "Train", "Test: before / after", "Manage adapters"])
    with tabs[0]:
        _dashboard_tab()
    with tabs[1]:
        rec = _dataset_tab()
    if "rec" not in locals() or rec is None:
        sel = st.session_state.get("lora_dataset")
        rec = storage_manager.get_dataset(sel) if sel else None
    with tabs[2]:
        _train_tab(rec)
    with tabs[3]:
        _test_tab()
    with tabs[4]:
        _manage_tab()
