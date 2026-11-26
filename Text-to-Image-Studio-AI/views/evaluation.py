"""Evaluation Center page (Module 5): classifier metrics (accuracy/ROC/AUC), diffusion comparison, image metrics, history, reports."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go_
import streamlit as st
from PIL import Image

from components.ui_components import notify, panel_title, style_fig, tiles, to_bytes
from config import settings
from modules import evaluation as ev
from modules import metrics
from services import storage_manager


def _gan_runs():
    try:
        from modules import gan_trainer
        return [r for r in gan_trainer.list_runs() if r["has_checkpoint"]]
    except Exception:
        return []


# ----------------------------------------------------------------------------- tab 1: classifier metrics
def _classifier_tab() -> None:
    st.markdown("A **reference classifier** is trained on the *real* images. Generated images of class *c* are then classified: if the generator really produces class *c*, the classifier says *c*. This gives **accuracy, a confusion matrix and one-vs-rest ROC/AUC** for each GAN.")
    runs = _gan_runs()
    if not runs:
        st.info("Train at least one GAN in GAN Studio first.")
        return
    datasets = sorted({r["config"]["dataset_name"] for r in runs})
    ds_name = st.selectbox("Dataset", datasets, key="ev_ds")
    rec = storage_manager.get_dataset(ds_name)
    sizes = sorted({r["config"]["img_size"] for r in runs if r["config"]["dataset_name"] == ds_name})
    size = st.selectbox("Image size", sizes, key="ev_size")
    from modules.evaluation import classifier_path
    have = classifier_path(ds_name, size).exists()
    c = st.columns([1, 1, 2])
    epochs = c[0].number_input("Classifier epochs", 1, 50, 8, key="ev_clf_epochs")
    if c[1].button("Train reference classifier", key="ev_clf_train", disabled=rec is None):
        bar = st.progress(0.0, text="Training the classifier on real images...")
        try:
            out = ev.train_reference_classifier(rec["path"], ds_name, int(size), int(epochs), lambda f: bar.progress(min(f, 1.0), text="Training the classifier on real images..."))
            bar.empty()
            storage_manager.add_experiment("classifier", f"Reference classifier · {ds_name}", {"dataset": ds_name, "img_size": size, "epochs": int(epochs)},
                                           {"real_accuracy": out["real_accuracy"], "real_macro_auc": out["real_macro_auc"]})
            notify(f"Classifier trained. Accuracy on real held-out images: {out['real_accuracy']:.1%}", "success")
            have = True
        except Exception as exc:
            bar.empty()
            st.error(f"Training failed: {exc}")
    if have:
        import torch
        state = torch.load(classifier_path(ds_name, size), map_location="cpu")
        c[2].success(f"Classifier ready · accuracy on real held-out images: {state['real_accuracy']:.1%} (upper bound for generated images)")
    else:
        st.warning("No reference classifier for this dataset and image size yet. Train it first.")
        return
    candidates = [r["name"] for r in runs if r["config"]["dataset_name"] == ds_name and r["config"]["img_size"] == size]
    if "ev_run_pick" in st.session_state:
        st.session_state["ev_run_pick"] = [x for x in st.session_state["ev_run_pick"] if x in candidates]
        picks = st.multiselect("GAN runs to evaluate", candidates, key="ev_run_pick")
    else:
        picks = st.multiselect("GAN runs to evaluate", candidates, default=candidates[:2], key="ev_run_pick")
    c = st.columns(2)
    per = c[0].number_input("Generated images per class", 10, 500, 50, key="ev_per")
    seed = c[1].number_input("Seed", 0, 10 ** 6, 0, key="ev_seed")
    if st.button("Evaluate selected runs", type="primary", key="ev_run_go") and picks:
        results = {}
        for name in picks:
            with st.spinner(f"Evaluating {name}..."):
                try:
                    results[name] = ev.evaluate_gan_run(name, ds_name, int(per), int(seed))
                    r = results[name]
                    storage_manager.add_experiment("gan_classifier_eval", f"{name} ({r['arch']})",
                                                   {"run": name, "dataset": ds_name, "per_class": int(per), "seed": int(seed), "cond": r["cond_mode"], "epochs": r["epochs"]},
                                                   {"accuracy": r["accuracy"], "macro_auc": r["macro_auc"], "real_accuracy": r["real_accuracy"], "n": r["n"], "architecture": r["arch"]})
                except Exception as exc:
                    st.error(f"{name}: {exc}")
        st.session_state["ev_cls_results"] = results
        if results:
            notify(f"Evaluated {len(results)} run(s).", "success")
    results = st.session_state.get("ev_cls_results")
    if not results:
        return
    table = pd.DataFrame([{"run": k, "architecture": v["arch"], "epochs": v["epochs"], "accuracy": round(v["accuracy"], 4),
                           "macro AUC": round(v["macro_auc"], 4) if v["macro_auc"] is not None else None, "real-image accuracy (ceiling)": round(v["real_accuracy"], 4)}
                          for k, v in results.items()]).sort_values(["accuracy", "macro AUC"], ascending=False)
    st.markdown("**Ranking protocol:** highest accuracy first, macro AUC breaks ties. Same dataset, classifier, class set, image count and seed for every run.")
    st.dataframe(table, use_container_width=True, hide_index=True)
    if len(table) > 1:
        st.plotly_chart(style_fig(px.bar(table.melt(id_vars=["run", "architecture"], value_vars=["accuracy", "macro AUC"]), x="run", y="value", color="variable", barmode="group", title="Accuracy and macro AUC")), use_container_width=True)
    for name, r in results.items():
        with st.expander(f"{name} · {r['arch']} · accuracy {r['accuracy']:.1%} · macro AUC {r['macro_auc']:.3f}" if r["macro_auc"] is not None else name, expanded=len(results) == 1):
            a, b = st.columns(2)
            cm = np.array(r["confusion_matrix"])
            short = [c[:12] for c in r["classes"]]
            a.plotly_chart(style_fig(px.imshow(cm, x=short, y=short, color_continuous_scale="Purples", text_auto=cm.shape[0] <= 12, labels=dict(x="classified as", y="generated class"), title="Confusion matrix"), 420), use_container_width=True, key=f"ev_cm_{name}")
            fig = go_.Figure()
            for cls, curve in list(r["roc"].items())[:12]:
                fig.add_scatter(x=curve["fpr"], y=curve["tpr"], mode="lines", name=f"{cls[:14]} (AUC {r['per_class_auc'][cls]:.2f})")
            fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="chance", showlegend=False)
            fig.update_layout(title="One-vs-rest ROC curves" + (" (first 12 classes shown)" if len(r["roc"]) > 12 else ""), xaxis_title="False positive rate", yaxis_title="True positive rate")
            b.plotly_chart(style_fig(fig, 420), use_container_width=True, key=f"ev_roc_{name}")


# ----------------------------------------------------------------------------- tab 2: diffusion comparison
def _diffusion_tab() -> None:
    st.markdown("Run the **same prompts and seed** on both diffusion models. Time, GPU memory and CLIPScore are measured on this machine; nothing is estimated.")
    models = list(settings.MODEL_REGISTRY)
    sel = st.multiselect("Models", models, default=models, key="ev_models")
    prompts = st.text_area("Prompts (one per line, up to 5)", "A serene Japanese landscape with cherry blossoms and a lake at sunset\nA golden retriever puppy in a sunny garden", key="ev_prompts", height=90)
    c = st.columns(4)
    seed = c[0].number_input("Seed", 0, 2 ** 31, 42, key="ev_dseed")
    sched = c[1].selectbox("Scheduler", settings.SCHEDULER_NAMES, key="ev_dsched")
    steps = {m: c[2 + i % 2].number_input(f"Steps · {m.split(' ')[-1] if 'XL' not in m else 'SDXL'}", 5, 100, 30, key=f"ev_steps_{i}") for i, m in enumerate(sel[:2])}
    plist = [p.strip() for p in prompts.splitlines() if p.strip()][:5]
    if st.button("Run comparison", type="primary", key="ev_dgo") and sel and plist:
        bar = st.progress(0.0, text="Starting...")
        try:
            rows = ev.compare_diffusion_models(plist, sel, int(seed), steps, sched, 7.5, lambda f, t: bar.progress(min(f, 1.0), text=t))
        except Exception as exc:
            bar.empty()
            st.error(f"Comparison failed: {exc}")
            return
        bar.empty()
        st.session_state["ev_drows"] = rows
        storage_manager.add_experiment("diffusion_comparison", f"{' vs '.join(m.split(' ')[-1] for m in sel)} · {len(plist)} prompts",
                                       {"models": sel, "prompts": plist, "seed": int(seed), "scheduler": sched, "steps": steps},
                                       {"ranking": ev.rank_models(rows),
                                        **{f"{r['model']} | clip {i}": r["clip_score"] for i, r in enumerate(rows)},
                                        **{f"{r['model']} | time {i}": r["time_s"] for i, r in enumerate(rows)},
                                        **{f"{r['model']} | vram {i}": r["peak_vram_gb"] for i, r in enumerate(rows)}})
        notify("Comparison finished and saved to experiment history.", "success")
    rows = st.session_state.get("ev_drows")
    if not rows:
        return
    for prompt in dict.fromkeys(r["prompt"] for r in rows):
        st.markdown(f"**{prompt}**")
        subset = [r for r in rows if r["prompt"] == prompt]
        for col, r in zip(st.columns(len(subset)), subset):
            col.image(r["image"], use_container_width=True)
            col.caption(f"{r['model']} v{r['version']} · {r['resolution']} · {r['steps']} steps · {r['time_s']}s · VRAM {r['peak_vram_gb']} GB · CLIPScore {r['clip_score']}")
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("image", "record_id")} for r in rows])
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown("**Ranking** (protocol: mean CLIPScore, ties within 0.5 broken by lower time)")
    st.dataframe(pd.DataFrame(ev.rank_models(rows)), use_container_width=True, hide_index=True)
    a, b = st.columns(2)
    a.plotly_chart(style_fig(px.scatter(df, x="time_s", y="clip_score", color="model", hover_name="prompt", title="Prompt alignment vs generation time")), use_container_width=True)
    if df["peak_vram_gb"].notna().any():
        df["pixels (MP)"] = df["resolution"].map(lambda r: int(r.split("x")[0]) * int(r.split("x")[1]) / 1e6)
        b.plotly_chart(style_fig(px.scatter(df, x="pixels (MP)", y="peak_vram_gb", color="model", title="Resolution vs peak GPU memory")), use_container_width=True)
    st.caption("CLIPScore measures how well the image matches the prompt, not realism or aesthetics. With 1-2 prompts the ranking is anecdotal; use more prompts for a fair comparison.")


# ----------------------------------------------------------------------------- tab 3: image metrics
def _metrics_tab() -> None:
    st.markdown("Each metric is offered only for inputs that make it meaningful.")
    recs = storage_manager.list_images(limit=300)
    labels = {r["id"]: f"#{r['id']} · {r['model']} · {r['prompt'][:40]}" for r in recs}
    t1, t2, t3 = st.tabs(["CLIPScore (text ↔ image)", "FID (distribution vs reference)", "SSIM · PSNR · LPIPS (image pairs)"])
    with t1:
        st.caption("Uses each image's own stored prompt. Measures prompt alignment only.")
        ids = st.multiselect("Gallery images", list(labels), format_func=labels.get, key="ev_clip_ids")
        if st.button("Compute CLIPScore", key="ev_clip_go") and ids:
            try:
                sel = [storage_manager.get_image(i) for i in ids]
                imgs = [storage_manager.load_pil(r) for r in sel]
                scores = metrics.clip_scores(imgs, [r["prompt"] for r in sel])
                st.dataframe(pd.DataFrame({"image": ids, "prompt": [r["prompt"][:60] for r in sel], "CLIPScore": np.round(scores, 2)}), hide_index=True, use_container_width=True)
                st.metric("Mean CLIPScore", f"{np.mean(scores):.2f}")
                storage_manager.add_experiment("clip_score", f"CLIPScore · {len(ids)} images", {"image_ids": ids}, {"mean_clip_score": float(np.mean(scores))})
            except Exception as exc:
                st.error(f"CLIPScore failed: {exc}")
    with t2:
        st.caption("Needs a meaningful reference set (real images) and many generated images. Below 2048 per set the result is only a rough comparison.")
        ds = storage_manager.list_datasets()
        if not ds:
            st.info("Add a reference dataset in Dataset Explorer.")
        else:
            dname = st.selectbox("Reference dataset (real images)", [d["name"] for d in ds], key="ev_fid_ds")
            gen_types = storage_manager.distinct_values("gen_type")
            pick = st.multiselect("Generated images from gallery type(s)", gen_types, key="ev_fid_types")
            if st.button("Compute FID", key="ev_fid_go") and pick:
                try:
                    drec = storage_manager.get_dataset(dname)
                    from modules.dataset_explorer import IMAGE_EXTS
                    real_paths = [p for p in Path(drec["path"]).rglob("*") if p.suffix.lower() in IMAGE_EXTS and "_quarantine" not in p.parts][:2048]
                    real = [Image.open(p).convert("RGB") for p in real_paths]
                    fake = [storage_manager.load_pil(r) for r in storage_manager.list_images(gen_types=pick, limit=2048)]
                    fake = [f for f in fake if f is not None]
                    with st.spinner("Computing Inception features..."):
                        out = metrics.fid_score(real, fake)
                    st.metric("FID (lower is better)", f"{out['fid']:.2f}")
                    st.caption(f"{out['n_real']} real vs {out['n_fake']} generated images. {out['note']}")
                    storage_manager.add_experiment("fid", f"FID · {dname}", {"dataset": dname, "types": pick}, {"fid": out["fid"], "n_real": out["n_real"], "n_fake": out["n_fake"], "reliable": out["reliable"]})
                except Exception as exc:
                    st.error(str(exc))
    with t3:
        st.caption("Valid only for two images of the same content (for example an image and its upscaled/degraded version, or a variation with a low strength).")
        c = st.columns(2)
        a = c[0].selectbox("Image A", list(labels), format_func=labels.get, key="ev_pair_a")
        b = c[1].selectbox("Image B", list(labels), format_func=labels.get, key="ev_pair_b") if labels else None
        if st.button("Compute pair metrics", key="ev_pair_go") and a and b:
            ia, ib = (storage_manager.load_pil(storage_manager.get_image(i)) for i in (a, b))
            out = {"SSIM": metrics.ssim(ia, ib), "PSNR (dB)": metrics.psnr(ia, ib)}
            try:
                out["LPIPS"] = metrics.lpips_distance(ia, ib)
            except Exception as exc:
                st.caption(f"LPIPS unavailable ({type(exc).__name__}); install with `pip install lpips`.")
            tiles([(k, f"{v:.4f}") for k, v in out.items()])
            storage_manager.add_experiment("pair_metrics", f"Pair {a} vs {b}", {"a": a, "b": b}, out)


# ----------------------------------------------------------------------------- tab 4/5: history + reports
def _history_tab() -> None:
    exps = storage_manager.list_experiments()
    if not exps:
        st.info("Experiments you run (classifier evaluation, comparisons, metrics, embeddings) are recorded here.")
        return
    kinds = sorted({e["kind"] for e in exps})
    kind = st.multiselect("Experiment type", kinds, default=kinds, key="ev_kinds")
    shown = [e for e in exps if e["kind"] in kind]
    df = pd.DataFrame([{"id": e["id"], "type": e["kind"], "name": e["name"], "created": e["created_at"],
                        **{k: v for k, v in (e["metrics"] or {}).items() if isinstance(v, (int, float, str)) and "|" not in k}} for e in shown])
    st.dataframe(df, use_container_width=True, hide_index=True)
    gan = [e for e in shown if e["kind"] == "gan_classifier_eval"]
    if gan:
        g = pd.DataFrame([{"run": e["name"], "accuracy": e["metrics"].get("accuracy"), "macro AUC": e["metrics"].get("macro_auc")} for e in gan])
        st.plotly_chart(style_fig(px.bar(g.melt(id_vars="run"), x="run", y="value", color="variable", barmode="group", title="GAN evaluations over time")), use_container_width=True)
    delete = st.selectbox("Delete an experiment record", [None] + [e["id"] for e in shown], format_func=lambda i: "-" if i is None else f"#{i}", key="ev_del")
    if delete and st.button("Delete record", key="ev_del_go"):
        storage_manager.delete_experiment(delete)
        st.rerun()


def _reports_tab() -> None:
    exps = storage_manager.list_experiments()
    chosen = st.multiselect("Experiments to include", [e["id"] for e in exps], format_func=lambda i: next(f"#{e['id']} · {e['name']}" for e in exps if e["id"] == i), key="ev_rep_ids")
    sel = [e for e in exps if e["id"] in chosen]
    if sel:
        c = st.columns(4)
        c[0].download_button("PDF report", ev.build_pdf_report("Text-to-Image Studio AI: evaluation report", sel), "evaluation_report.pdf", "application/pdf", key="ev_pdf")
        c[1].download_button("Raw metrics (CSV)", ev.experiments_to_csv(sel), "metrics.csv", "text/csv", key="ev_csv")
        c[2].download_button("Configuration (JSON)", json.dumps([{"id": e["id"], "name": e["name"], "kind": e["kind"], "config": e["config"]} for e in sel], indent=2, default=str), "experiment_config.json", "application/json", key="ev_cfgjson")
        c[3].download_button("Reproducibility report", json.dumps(ev.reproducibility_report({"experiments": [e["id"] for e in sel]}), indent=2, default=str), "reproducibility_report.json", "application/json", key="ev_repro")
    else:
        st.download_button("Reproducibility report (environment + methodology)", json.dumps(ev.reproducibility_report(), indent=2, default=str), "reproducibility_report.json", "application/json", key="ev_repro0")
    rows = st.session_state.get("ev_drows")
    if rows:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for i, r in enumerate(rows):
                zf.writestr(f"{i:02d}_{r['model'].replace(' ', '_')}.png", to_bytes(r["image"]))
        st.download_button("Comparison images (ZIP)", buf.getvalue(), "comparison_images.zip", "application/zip", key="ev_imgzip")
    runs = _gan_runs()
    if runs:
        pick = st.selectbox("Training curves (PNG) for", [r["name"] for r in runs], key="ev_curve_run")
        run = next(r for r in runs if r["name"] == pick)
        if run["history"]:
            st.download_button("Download training curves", ev.training_curves_png(run["history"], pick), f"{pick}_curves.png", "image/png", key="ev_curve_dl")


def render() -> None:
    st.markdown("<div class='page-title'>Evaluation Center</div><div class='page-sub'>Which model generates better images, and why? Every number here comes from a real run.</div>", unsafe_allow_html=True)
    tabs = st.tabs(["GAN classifier metrics", "Diffusion comparison", "Image metrics", "Experiment history", "Reports & export"])
    for tab, fn in zip(tabs, (_classifier_tab, _diffusion_tab, _metrics_tab, _history_tab, _reports_tab)):
        with tab:
            fn()
