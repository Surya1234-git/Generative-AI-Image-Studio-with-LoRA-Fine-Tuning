"""GAN Studio page (Tasks 2, 5, 6): CGAN, Attention CGAN, training dashboard, samples, comparison, text -> GAN pipeline."""
from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go_
import streamlit as st
from PIL import Image

from components.ui_components import go, job_status_line, notify, panel_title, style_fig, tiles, to_bytes
from config.settings import dir_path
from modules import dataset_explorer as dx
from modules import evaluation as ev
from services import storage_manager
from services.job_manager import JOBS, JobBusyError

OPTIMIZERS = ["Adam", "RMSprop", "SGD"]


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _init_state() -> None:
    d = {"gan_arch": "CGAN", "gan_cond": "label", "gan_epochs": 50, "gan_batch": 32, "gan_lr": 0.0002, "gan_z": 100, "gan_opt": "Adam",
         "gan_img": 64, "gan_loss": "bce", "gan_sample_every": 5, "gan_ckpt_every": 5, "gan_seed": 42, "gan_maxper": 0, "gan_run": "",
         "gan_job": None, "gan_active_run": None, "gan_text_fixed": "a photo of a rose\na photo of a sunflower\na photo of a daffodil"}
    for k, v in d.items():
        st.session_state.setdefault(k, v)


def _runs():
    from modules import gan_trainer
    return gan_trainer.list_runs()


def _classes_of(run_path: str) -> list[str]:
    f = Path(run_path) / "classes.json"
    return json.loads(f.read_text()) if f.exists() else []


# ----------------------------------------------------------------------------- job callbacks
def _cb_pause(): j = JOBS.get(st.session_state.gan_job); j and j.pause()
def _cb_resume(): j = JOBS.get(st.session_state.gan_job); j and j.resume()
def _cb_stop(): j = JOBS.get(st.session_state.gan_job); j and j.stop()
def _cb_ckpt(): j = JOBS.get(st.session_state.gan_job); j and j.request_checkpoint()


def _make_config(dataset: dict):
    from modules.gan_trainer import GANConfig
    s = st.session_state
    return GANConfig(arch=s.gan_arch, cond_mode=s.gan_cond, epochs=int(s.gan_epochs), batch_size=int(s.gan_batch), lr_g=float(s.gan_lr),
                     lr_d=float(s.gan_lr), z_dim=int(s.gan_z), optimizer=s.gan_opt, img_size=int(s.gan_img), loss=s.gan_loss,
                     sample_every=int(s.gan_sample_every), ckpt_every=int(s.gan_ckpt_every), seed=int(s.gan_seed),
                     max_per_class=int(s.gan_maxper) or None, dataset_name=dataset["name"], dataset_path=dataset["path"],
                     run_name=(s.gan_run.strip() or f"{'attn' if 'Attention' in s.gan_arch else 'cgan'}_{time.strftime('%m%d_%H%M')}"))


def _start(cfg, resume: bool) -> None:
    from modules import gan_trainer
    run_dir = gan_trainer.runs_dir() / cfg.run_name
    if not resume and (run_dir / "latest.pt").exists():
        st.error(f"A run named '{cfg.run_name}' already exists. Choose another name or resume it.")
        return
    try:
        job = JOBS.start("gan", f"GAN training: {cfg.run_name}", gan_trainer.train_gan, cfg=cfg, resume=resume)
    except JobBusyError as exc:
        st.error(str(exc))
        return
    st.session_state.gan_job = job.id
    st.session_state.gan_active_run = cfg.run_name
    notify(f"Training '{cfg.run_name}' started.", "success")
    st.rerun()


# ----------------------------------------------------------------------------- live panel
def _live_panel() -> None:
    job = JOBS.get(st.session_state.gan_job)
    active = bool(job and job.is_active())

    @st.fragment(run_every=2 if active else None)
    def live():
        j = JOBS.get(st.session_state.gan_job)
        if j is None:
            st.info("No training job in this session yet. Configure a run and press Start Training.")
            return
        hist = j.history
        last = hist[-1] if hist else {}
        run_dir = Path(j.extra.get("run_dir", ""))
        tiles([("Status", str(j.status).capitalize()),
               ("Epoch", f"{last.get('epoch', 0)}/{j.extra.get('epochs', '?')}"),
               ("Generator loss", f"{last['loss_g']:.3f}" if last else "-"),
               ("Discriminator loss", f"{last['loss_d']:.3f}" if last else "-"),
               ("Checkpoint", "saved" if (run_dir / "latest.pt").exists() else "none")])
        job_status_line(j)
        if j.extra.get("params_g"):
            st.caption(f"Generator {j.extra['params_g'] / 1e6:.2f} M parameters · discriminator {j.extra['params_d'] / 1e6:.2f} M · {len(j.extra.get('classes', []))} classes")
        if hist:
            df = pd.DataFrame(hist)
            a, b = st.columns(2)
            fig = go_.Figure([go_.Scatter(x=df.epoch, y=df.loss_g, name="Generator"), go_.Scatter(x=df.epoch, y=df.loss_d, name="Discriminator")])
            fig.update_layout(title="Losses per epoch", xaxis_title="epoch")
            a.plotly_chart(style_fig(fig, 300), use_container_width=True)
            fig2 = go_.Figure([go_.Scatter(x=df.epoch, y=df.d_real, name="D(real)"), go_.Scatter(x=df.epoch, y=df.d_fake, name="D(fake)")])
            fig2.update_layout(title="Discriminator confidence (stable training stays near 0.5-0.9 / 0.1-0.5)", xaxis_title="epoch")
            b.plotly_chart(style_fig(fig2, 300), use_container_width=True)
            st.caption("A lower GAN loss does not automatically mean better images. Judge quality with the sample grid and the Evaluation Center (classifier accuracy, AUC, FID).")
        sample = j.extra.get("latest_sample")
        if sample and Path(sample).exists():
            st.image(sample, caption=f"Fixed-noise samples ({Path(sample).stem}); each row is one class", use_container_width=True)
        if j.logs:
            with st.expander("Job log"):
                st.code("\n".join(j.logs[-12:]), language=None)
        was = st.session_state.get("_gan_prev_active")
        st.session_state["_gan_prev_active"] = j.is_active()
        if was and not j.is_active():
            notify(f"Training '{j.name}' {j.status}.", "success" if j.status == "completed" else "warning" if j.status == "stopped" else "error")
            st.rerun()

    live()


# ----------------------------------------------------------------------------- tabs
def _train_tab() -> None:
    s = st.session_state
    datasets = [d for d in storage_manager.list_datasets() if d["kind"] != "unlabeled"]
    left, right = st.columns([1, 1.5])
    with left, st.container(border=True):
        panel_title("⚙️", "Training setup")
        if not datasets:
            st.warning("No labeled dataset yet. Open Dataset Explorer, import one (for example Oxford-102 Flowers), then come back.")
            return
        names = [d["name"] for d in datasets]
        if s.get("gan_dataset") not in names:
            s.gan_dataset = names[0]
        st.selectbox("Dataset (from Dataset Explorer)", names, key="gan_dataset")
        dataset = next(d for d in datasets if d["name"] == s.gan_dataset)
        classes = dx.class_names_from_root(dx.training_root(dataset["path"])) if Path(dataset["path"]).exists() else []
        st.caption(f"{len(classes)} classes · {dataset['num_images']} images · trains on the 'train' folder if the dataset was split")
        st.selectbox("Architecture", ["CGAN", "Attention CGAN"], key="gan_arch", help="Attention CGAN adds a self-attention block (SAGAN) to both networks.")
        st.radio("Condition", ["label", "text"], key="gan_cond", horizontal=True, format_func=lambda x: "Class label" if x == "label" else "CLIP text embedding of the class caption (enables free-text generation)")
        a = st.columns(3)
        a[0].number_input("Epochs", 1, 1000, key="gan_epochs")
        a[1].number_input("Batch size", 2, 256, key="gan_batch")
        a[2].number_input("Learning rate", 1e-6, 1e-2, step=5e-5, format="%.5f", key="gan_lr")
        b = st.columns(3)
        b[0].number_input("Latent dimension", 16, 512, key="gan_z")
        b[1].selectbox("Optimizer", OPTIMIZERS, key="gan_opt")
        b[2].selectbox("Image size", [32, 64, 128], key="gan_img")
        c = st.columns(3)
        c[0].selectbox("Loss", ["bce", "hinge"], key="gan_loss")
        c[1].number_input("Sample every N epochs", 1, 100, key="gan_sample_every")
        c[2].number_input("Checkpoint every N epochs", 1, 100, key="gan_ckpt_every")
        d = st.columns(3)
        d[0].number_input("Seed", 0, 10 ** 6, key="gan_seed")
        d[1].number_input("Max images per class (0 = all)", 0, 5000, key="gan_maxper")
        d[2].text_input("Run name (optional)", key="gan_run", placeholder="attn_flowers_v1")
        st.caption("Suggested starting values, not universal optima: 64×64 images, batch 32, lr 0.0002, Adam, latent 100.")

        job = JOBS.get(s.gan_job)
        active = bool(job and job.is_active())
        b1, b2 = st.columns(2)
        if b1.button("Initialize Model", key="gan_init", use_container_width=True, icon=":material/build:"):
            try:
                from modules import gan_models
                G, D = gan_models.build_gan(s.gan_arch, max(2, len(classes)), int(s.gan_z), 128, 64, int(s.gan_img))
                st.success(f"{s.gan_arch}: generator {gan_models.count_parameters(G) / 1e6:.2f} M, discriminator {gan_models.count_parameters(D) / 1e6:.2f} M parameters. Ready to train.")
            except Exception as exc:
                st.error(f"Could not build the model: {exc}")
        if b2.button("Start Training", key="gan_start", type="primary", use_container_width=True, disabled=active, icon=":material/play_arrow:"):
            if len(classes) < 2:
                st.error("The dataset needs at least 2 class folders.")
            else:
                _start(_make_config(dataset), resume=False)
        c1, c2, c3, c4 = st.columns(4)
        c1.button("Pause", key="gan_pause", on_click=_cb_pause, disabled=not active, use_container_width=True)
        c2.button("Resume", key="gan_resume_btn", on_click=_cb_resume, disabled=not active, use_container_width=True)
        c3.button("Stop", key="gan_stop", on_click=_cb_stop, disabled=not active, use_container_width=True)
        c4.button("Save Checkpoint", key="gan_save_ck", on_click=_cb_ckpt, disabled=not active, use_container_width=True)
        runs = [r for r in _runs() if r["has_checkpoint"]]
        if runs and not active:
            st.divider()
            pick = st.selectbox("Load Checkpoint / continue a run", [r["name"] for r in runs], key="gan_resume_pick")
            run = next(r for r in runs if r["name"] == pick)
            more = st.number_input("Train until epoch", run["epochs_done"] + 1, 2000, max(run["epochs_done"] + 10, int(run["config"]["epochs"])), key="gan_resume_to")
            r1, r2 = st.columns(2)
            if r1.button("Resume Training", key="gan_do_resume", use_container_width=True):
                from modules.gan_trainer import GANConfig
                cfg = GANConfig(**{**run["config"], "epochs": int(more)})
                _start(cfg, resume=True)
            if r2.button("Use for samples", key="gan_use_run", use_container_width=True):
                s.gan_active_run = pick
                notify(f"'{pick}' is now the active model for sampling and comparison.", "success")
    with right, st.container(border=True):
        panel_title("📈", "Training dashboard")
        _live_panel()


def _samples_tab() -> None:
    s = st.session_state
    runs = [r for r in _runs() if r["has_checkpoint"]]
    if not runs:
        st.info("Train a model first. Samples are generated from saved checkpoints.")
        return
    names = [r["name"] for r in runs]
    if s.get("gan_active_run") not in names:
        s.gan_active_run = names[0]
    run_name = st.selectbox("Model run", names, key="gan_active_run")
    run = next(r for r in runs if r["name"] == run_name)
    classes = _classes_of(run["path"])
    st.caption(f"{run['config']['arch']} · {run['config']['cond_mode']} conditioning · {run['epochs_done']} epochs · {run['config']['img_size']}px")
    ckpts = ["latest.pt"] + sorted(p.name for p in Path(run["path"]).glob("ckpt_epoch_*.pt"))
    c = st.columns([2, 1, 1, 1])
    chosen = c[0].multiselect("Classes", classes, default=classes[:4], key="gan_pick_classes")
    per = c[1].number_input("Images per class", 1, 16, 4, key="gan_per")
    seed = c[2].number_input("Seed", 0, 10 ** 6, 0, key="gan_gen_seed")
    ck = c[3].selectbox("Checkpoint", ckpts, key="gan_ckpt_pick")
    if st.button("Generate Samples", type="primary", key="gan_gen_go"):
        if not chosen:
            st.error("Choose at least one class.")
            return
        try:
            from modules import gan_trainer
            _G, cls, cfg = gan_trainer.load_generator(run_name, ck)
            idx = [cls.index(x) for x in chosen]
            imgs, labels = gan_trainer.generate_images(run_name, idx, int(per), int(seed), ck)
            s["gan_samples"] = {"images": imgs, "labels": [cls[i] for i in labels], "run": run_name, "arch": cfg["arch"]}
        except Exception as exc:
            st.error(f"Sampling failed: {exc}")
    smp = s.get("gan_samples")
    if smp:
        cols = st.columns(int(per) if int(per) <= 8 else 8)
        for i, (img, lab) in enumerate(zip(smp["images"], smp["labels"])):
            cols[i % len(cols)].image(img.resize((img.width * 2, img.height * 2), Image.NEAREST), caption=lab[:22], use_container_width=True)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for i, (img, lab) in enumerate(zip(smp["images"], smp["labels"])):
                zf.writestr(f"{i:03d}_{lab}.png", to_bytes(img))
        d1, d2 = st.columns(2)
        d1.download_button("Download samples (ZIP)", buf.getvalue(), f"{smp['run']}_samples.zip", "application/zip", key="gan_zip")
        if d2.button("Save samples to Image Gallery", key="gan_to_gallery"):
            for img, lab in zip(smp["images"], smp["labels"]):
                storage_manager.save_generated_image(img, {"prompt": dx.class_caption(lab), "model": f"{smp['arch']} ({smp['run']})", "seed": int(seed),
                                                           "gen_type": f"gan:{smp['arch']}", "scheduler": "-", "steps": None, "guidance": None})
            notify(f"Saved {len(smp['images'])} samples to the gallery.", "success")
    st.divider()
    panel_title("📦", "Training history and export")
    hist = run["history"]
    if hist:
        e1, e2 = st.columns(2)
        e1.download_button("Training curves (PNG)", ev.training_curves_png(hist, run_name), f"{run_name}_curves.png", "image/png", key="gan_curves")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for f in ["config.json", "history.json", "classes.json"]:
                if (Path(run["path"]) / f).exists():
                    zf.write(Path(run["path"]) / f, f)
            for f in (Path(run["path"]) / "samples").glob("*.png"):
                zf.write(f, f"samples/{f.name}")
            zf.writestr("training_curves.png", ev.training_curves_png(hist, run_name))
        e2.download_button("Export Results (ZIP)", buf.getvalue(), f"{run_name}_results.zip", "application/zip", key="gan_export")
    gallery = sorted((Path(run["path"]) / "samples").glob("*.png"))
    if gallery:
        st.markdown("**Samples across epochs** (same fixed noise each time)")
        pick = st.select_slider("Epoch", [p.stem.replace("epoch_", "") for p in gallery], key="gan_epoch_pick")
        st.image(str(Path(run["path"]) / "samples" / f"epoch_{pick}.png"), use_container_width=True)
    st.button("Evaluate this run (accuracy · ROC · AUC · FID)", key="gan_to_eval", on_click=lambda: (st.session_state.update(ev_run_pick=[run_name]), go("Evaluation Center")))


def _attention_figure(run_name: str, class_name: str, seed: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from modules import gan_models, gan_trainer
    G, classes, cfg = gan_trainer.load_generator(run_name)
    device = next(G.parameters()).device
    z = torch.randn(1, cfg["z_dim"], generator=torch.Generator().manual_seed(seed)).to(device)
    y = torch.tensor([classes.index(class_name)], device=device)
    queries = [(4, 4), (8, 8), (12, 12)]
    out = gan_models.attention_overlay(G, z, y, queries)
    if out is None:
        return None
    img, maps, res = out
    pic = ((img[0].detach().cpu().clamp(-1, 1) * .5 + .5).permute(1, 2, 0).numpy())
    fig, axes = plt.subplots(1, len(queries) + 1, figsize=(3 * (len(queries) + 1), 3.2))
    axes[0].imshow(pic); axes[0].set_title("Generated"); axes[0].axis("off")
    scale = pic.shape[0] / res
    for k, ((r, c), ax) in enumerate(zip(queries, axes[1:])):
        heat = np.asarray(Image.fromarray((maps[0, k].numpy() / maps[0, k].numpy().max() * 255).astype(np.uint8)).resize(pic.shape[:2][::-1], Image.BILINEAR)) / 255.0
        ax.imshow(pic); ax.imshow(heat, cmap="magma", alpha=.55)
        ax.plot((c + .5) * scale, (r + .5) * scale, "c*", markersize=14)
        ax.set_title(f"Where pixel ({r},{c}) of the\n16×16 map attends", fontsize=8); ax.axis("off")
    fig.tight_layout()
    return fig


def _compare_tab() -> None:
    runs = [r for r in _runs() if r["has_checkpoint"]]
    if len(runs) < 1:
        st.info("Train a CGAN and an Attention CGAN (same dataset and settings) to compare them here.")
        return
    names = [r["name"] for r in runs]
    c = st.columns(2)
    a = c[0].selectbox("Model A", names, key="gan_cmp_a")
    b = c[1].selectbox("Model B", names, index=min(1, len(names) - 1), key="gan_cmp_b")
    ra, rb = (next(r for r in runs if r["name"] == n) for n in (a, b))
    st.dataframe(pd.DataFrame([{"run": r["name"], "architecture": r["config"]["arch"], "condition": r["config"]["cond_mode"], "epochs done": r["epochs_done"],
                                "image size": r["config"]["img_size"], "batch": r["config"]["batch_size"], "lr": r["config"]["lr_g"], "dataset": r["config"]["dataset_name"]} for r in (ra, rb)]),
                 use_container_width=True, hide_index=True)
    fig = go_.Figure()
    for r in (ra, rb):
        if r["history"]:
            fig.add_scatter(x=[h["epoch"] for h in r["history"]], y=[h["loss_g"] for h in r["history"]], name=f"{r['name']} · G loss")
            fig.add_scatter(x=[h["epoch"] for h in r["history"]], y=[h["loss_d"] for h in r["history"]], name=f"{r['name']} · D loss", line=dict(dash="dot"))
    fig.update_layout(title="Training loss comparison", xaxis_title="epoch")
    st.plotly_chart(style_fig(fig, 340), use_container_width=True)
    st.caption("Loss curves of different architectures are not directly comparable measures of quality. Use the classifier metrics and FID in the Evaluation Center for that.")
    classes = _classes_of(ra["path"])
    if classes and classes == _classes_of(rb["path"]):
        picks = st.multiselect("Classes for the side-by-side grid", classes, default=classes[:3], key="gan_cmp_classes")
        seed = st.number_input("Seed", 0, 10 ** 6, 0, key="gan_cmp_seed")
        if st.button("Generate comparison grids", key="gan_cmp_go") and picks:
            try:
                from modules import gan_trainer
                cols = st.columns(2)
                for col, r in zip(cols, (ra, rb)):
                    imgs, labs = gan_trainer.generate_images(r["name"], [classes.index(p) for p in picks], 4, int(seed))
                    col.markdown(f"**{r['name']}** ({r['config']['arch']})")
                    sub = col.columns(4)
                    for i, im in enumerate(imgs):
                        sub[i % 4].image(im.resize((im.width * 2, im.height * 2), Image.NEAREST), caption=classes[labs[i]][:14], use_container_width=True)
            except Exception as exc:
                st.error(f"Could not generate: {exc}")
    elif classes:
        st.warning("The two runs use different class lists, so a fair side-by-side is not possible.")
    st.divider()
    panel_title("🔍", "Attention maps (real activations from the generator)")
    att = [r for r in runs if "Attention" in r["config"]["arch"]]
    if not att:
        st.info("Train an Attention CGAN to see its attention maps.")
        return
    c = st.columns(3)
    arun = c[0].selectbox("Attention model", [r["name"] for r in att], key="gan_att_run")
    cls = _classes_of(next(r for r in att if r["name"] == arun)["path"])
    cname = c[1].selectbox("Class", cls, key="gan_att_cls")
    seed = c[2].number_input("Seed", 0, 10 ** 6, 3, key="gan_att_seed")
    if st.button("Show attention maps", key="gan_att_go"):
        try:
            fig = _attention_figure(arun, cname, int(seed))
            st.pyplot(fig)
            st.caption("Each panel shows, for one query pixel (star), the attention weights over the 16×16 feature map, upsampled onto the image. Early in training the layer attends almost uniformly (its gate starts at zero).")
        except Exception as exc:
            st.error(f"Could not compute attention maps: {exc}")


def _pipeline_tab() -> None:
    runs = [r for r in _runs() if r["has_checkpoint"] and r["config"]["cond_mode"] == "text"]
    st.markdown("**End-to-end pipeline (Task 6):** text → tokenizer → CLIP embedding → conditional GAN → image")
    if not runs:
        st.info("Train a run with 'CLIP text embedding' conditioning (Training tab) to enable free-text generation.")
        return
    run = st.selectbox("Model run", [r["name"] for r in runs], key="gan_pipe_run")
    text = st.text_input("Describe an image", "a photo of a rose", key="gan_pipe_text")
    c = st.columns(2)
    n = c[0].number_input("Images", 1, 8, 4, key="gan_pipe_n")
    seed = c[1].number_input("Seed", 0, 10 ** 6, 0, key="gan_pipe_seed")
    if st.button("Run pipeline", type="primary", key="gan_pipe_go"):
        try:
            from modules import gan_trainer
            out = gan_trainer.text_to_image(run, text, int(n), int(seed))
            st.markdown("**1. Tokens**  " + " | ".join(out["tokens"]))
            st.markdown(f"**2. Embedding** shape `{out['embedding_shape']}` (CLIP ViT-B/32 text encoder)")
            st.markdown(f"**3. Generator** conditioned on that embedding · nearest training caption: *{out['nearest_class']}* (cosine {out['nearest_similarity']:.2f})")
            cols = st.columns(int(n))
            for col, im in zip(cols, out["images"]):
                col.image(im.resize((im.width * 2, im.height * 2), Image.NEAREST), use_container_width=True)
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
    with st.expander("End-to-end test on fixed prompts"):
        st.text_area("Prompts (one per line)", key="gan_text_fixed", height=90)
        if st.button("Run test", key="gan_pipe_test"):
            from modules import gan_trainer
            for p in [x.strip() for x in st.session_state.gan_text_fixed.splitlines() if x.strip()]:
                try:
                    out = gan_trainer.text_to_image(run, p, 4, 0)
                    st.markdown(f"**{p}** → {out['embedding_shape']} → nearest: {out['nearest_class']}")
                    cols = st.columns(4)
                    for col, im in zip(cols, out["images"]):
                        col.image(im.resize((im.width * 2, im.height * 2), Image.NEAREST), use_container_width=True)
                except Exception as exc:
                    st.error(f"'{p}' failed: {exc}")


def render() -> None:
    st.markdown("<div class='page-title'>GAN Studio</div><div class='page-sub'>Train conditional GANs and attention-based conditional GANs, watch real training progress, and generate from checkpoints.</div>", unsafe_allow_html=True)
    if not _torch_available():
        st.error("PyTorch is not installed in this environment. Install the requirements (or use the Colab notebook).")
        return
    _init_state()
    tabs = st.tabs(["Train", "Generate & export", "CGAN vs Attention CGAN", "Text → GAN pipeline"])
    with tabs[0]:
        _train_tab()
    with tabs[1]:
        _samples_tab()
    with tabs[2]:
        _compare_tab()
    with tabs[3]:
        _pipeline_tab()
