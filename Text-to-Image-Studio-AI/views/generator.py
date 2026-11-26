"""AI Image Generator page (Module 1). Also used, in compact form, by the Home page."""
from __future__ import annotations

import random
import time

import streamlit as st
from PIL import Image

from components.ui_components import chip, confirm_button, notify, panel_title, to_bytes
from config import settings
from config.settings import ASPECT_RATIOS, MODEL_REGISTRY, SCHEDULER_NAMES, STYLE_PRESETS
from modules import image_generator as ig
from modules import lora_registry, prompt_tools
from services import gpu_manager, storage_manager
from services.model_manager import model_status

QUICK = ["Anime", "Realistic", "Fantasy", "Nature"]


# ----------------------------------------------------------------------------- state helpers
def _available_models() -> dict:
    return {k: model_status(k) for k in MODEL_REGISTRY if model_status(k)["state"] != "unavailable"}


def _init_state(models: dict) -> None:
    prefs = settings.load_settings()
    wanted = prefs["default_model"]
    if wanted == "auto":
        wanted = gpu_manager.recommended_profile()["default_model"]
    if wanted not in models:
        wanted = next(iter(models))
    defaults = {"gen_prompt": "", "gen_neg": "", "gen_model": wanted, "gen_res": "", "gen_steps": int(prefs["default_steps"]),
                "gen_guidance": float(prefs["default_guidance"]), "gen_seed": -1, "gen_hq": False, "gen_upscale": False,
                "gen_up_mode": "Fast (Lanczos 2x)", "gen_use_lora": False, "gen_lora": "", "gen_lora_scale": 1.0,
                "gen_scheduler": prefs["default_scheduler"], "gen_batch": 1, "gen_aspect": "From resolution", "gen_save": True,
                "gen_w": 512, "gen_h": 512, "gen_strength": 0.6, "gen_enh_model": False, "gen_sel": 0}
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)
    if st.session_state.gen_model not in models:
        st.session_state.gen_model = wanted


def _res_options(model_key: str) -> list[str]:
    return [f"{w} × {h}" for w, h in MODEL_REGISTRY[model_key]["resolutions"]] + ["Custom"]


def _consume_apply() -> None:
    """Apply settings requested by another page/button. Must run before the widgets are drawn."""
    rec = st.session_state.pop("gen_apply", None)
    if not rec:
        return
    s = st.session_state
    keep_seed = s.pop("gen_apply_keep_seed", False)
    s.gen_prompt, s.gen_neg = rec["prompt"], rec.get("negative_prompt") or ""
    if rec.get("model") in MODEL_REGISTRY:
        s.gen_model = rec["model"]
    label = f"{rec['width']} × {rec['height']}"
    s.gen_res = label if label in _res_options(s.gen_model) else "Custom"
    s.gen_w, s.gen_h, s.gen_aspect = int(rec["width"]), int(rec["height"]), "From resolution"
    s.gen_steps = int(rec.get("steps") or 30)
    s.gen_guidance = float(rec.get("guidance") or 7.5)
    s.gen_scheduler = rec.get("scheduler") if rec.get("scheduler") in SCHEDULER_NAMES else s.gen_scheduler
    s.gen_seed = int(rec["seed"]) if keep_seed and rec.get("seed") is not None else -1
    if s.pop("gen_reuse_notice", False):
        st.toast(f"Settings restored from this image: {rec.get('model')}, {rec['width']}×{rec['height']}, {s.gen_steps} steps, "
                 f"guidance {s.gen_guidance}, {s.gen_scheduler}, seed {s.gen_seed}. The form above is filled in.", icon="✅")
        notify("Settings restored from the selected image. Press Generate to reproduce it.", "success")
        import streamlit.components.v1 as components      # bring the filled-in form into view (best effort)
        components.html("<script>const m=window.parent.document.querySelector('[data-testid=\"stMain\"]')||window.parent.document.querySelector('section.main');"
                        "if(m){m.scrollTo({top:0,behavior:'smooth'});}</script>", height=0)


def _dims(model_key: str) -> tuple[int, int]:
    s = st.session_state
    if s.gen_aspect != "From resolution":
        return ig.aspect_dims(s.gen_aspect, MODEL_REGISTRY[model_key]["native"][0])
    if s.gen_res == "Custom":
        return int(s.gen_w), int(s.gen_h)
    return ig.parse_resolution(s.gen_res)


# ----------------------------------------------------------------------------- callbacks (run before the script re-runs)
def _cb_preset(name: str) -> None:
    st.session_state.gen_prompt = prompt_tools.apply_preset(st.session_state.gen_prompt, name)


def _cb_enhance() -> None:
    text = st.session_state.gen_prompt
    if not text.strip():
        st.session_state.gen_enh_msg = "Type a short prompt first, then enhance it."
        return
    if st.session_state.gen_enh_model:
        new, how = prompt_tools.enhance_with_model(text)
    else:
        new, how = prompt_tools.enhance_prompt(text), "rule-based enhancer"
    st.session_state.gen_prompt = new[:1000]
    st.session_state.gen_enh_msg = f"Enhanced with the {how}. You can edit the result."


def _cb_random() -> None:
    st.session_state.gen_prompt = prompt_tools.random_prompt()


def _cb_clear() -> None:
    st.session_state.gen_prompt = ""
    st.session_state.gen_neg = ""


def _cb_history(prompt: str, neg: str) -> None:
    st.session_state.gen_prompt, st.session_state.gen_neg = prompt, neg or ""


def _cb_random_seed() -> None:
    st.session_state.gen_seed = -1


def _cb_favorite(sel: int) -> None:
    res = st.session_state.get("gen_result")
    rec = res["records"][sel] if res else None
    if rec and rec.get("id"):
        new = not rec["favorite"]
        storage_manager.set_favorite(rec["id"], new)
        rec["favorite"] = int(new)


def _request_run(rec: dict, kind: str, keep_seed: bool = False) -> None:
    st.session_state["gen_apply"] = rec
    st.session_state["gen_apply_keep_seed"] = keep_seed
    if kind:
        st.session_state["gen_pending_run"] = kind
    st.rerun()


# ----------------------------------------------------------------------------- building + running requests
def _selected_lora() -> str | None:
    s = st.session_state
    if not s.gen_use_lora or not s.gen_lora:
        return None
    return next((a["path"] for a in lora_registry.list_adapters() if a["name"] == s.gen_lora), None)


def _build_request(init_image=None, strength=None, gen_type="txt2img", extra=None) -> ig.GenerationRequest:
    s = st.session_state
    w, h = _dims(s.gen_model)
    return ig.GenerationRequest(prompt=s.gen_prompt, negative_prompt=s.gen_neg, model_key=s.gen_model, width=w, height=h,
                                steps=int(s.gen_steps), guidance=float(s.gen_guidance), seed=int(s.gen_seed), scheduler=s.gen_scheduler,
                                batch_size=int(s.gen_batch), lora_dir=_selected_lora(), lora_scale=float(s.gen_lora_scale),
                                init_image=init_image, strength=float(strength if strength is not None else s.gen_strength),
                                high_quality=bool(s.gen_hq), save=bool(s.gen_save), gen_type=gen_type, extra=extra or {})


def _show_busy(slot) -> None:
    """Replace the previous image at once, so a stale picture is never shown while a new one is being made."""
    with slot.container(border=True):
        panel_title("🖼️", "Generated Result")
        st.markdown("<div style='height:380px;display:flex;align-items:center;justify-content:center;text-align:center;color:var(--muted);border:1px dashed var(--line);border-radius:14px;padding:1rem'>Generating your image...<br>The result appears here when it is ready.</div>", unsafe_allow_html=True)


def _execute(req: ig.GenerationRequest, label: str = "Generating image", slot=None) -> None:
    if slot is not None:
        _show_busy(slot)
    box = st.status(label, expanded=True)
    bar = box.progress(0.0)
    line = box.empty()
    t0 = time.time()

    def status_cb(msg):
        line.write(f"{msg}  ·  {time.time() - t0:.0f}s")

    def progress_cb(step, total, elapsed):
        bar.progress(min(1.0, step / total))
        line.write(f"Denoising step {step}/{total}  ·  {elapsed:.1f}s")

    try:
        status_cb("Validating settings and checking memory...")
        res = ig.generate(req, progress_cb, status_cb)
        images, records = res.images, res.records
        if st.session_state.gen_upscale:
            status_cb("Upscaling...")
            images, records = _upscale_all(images, records, req)
    except ig.GenerationError as exc:
        box.update(label="Generation failed", state="error")
        st.error(str(exc))
        notify(str(exc), "error")
        return
    except Exception as exc:  # unexpected: show it, keep the app alive
        box.update(label="Generation failed", state="error")
        st.error(f"Unexpected error: {type(exc).__name__}: {exc}")
        notify(f"Generation failed: {exc}", "error")
        return
    box.update(label=f"Done in {res.elapsed:.1f}s", state="complete", expanded=False)
    st.session_state["gen_result"] = {"images": images, "records": records, "elapsed": res.elapsed, "peak_gb": res.peak_gb}
    st.session_state["gen_sel"] = 0
    st.session_state["gen_done_note"] = {"label": f"Done in {res.elapsed:.1f}s", "warnings": list(res.warnings)}
    notify(f"Generated {len(images)} image(s) in {res.elapsed:.1f}s.", "success")
    # Redraw the page from the saved result so the new image is shown immediately (no stale picture).
    st.rerun()


def _upscale_all(images, records, req):
    out_imgs, out_recs = [], []
    ai = st.session_state.gen_up_mode.startswith("AI")
    for img, rec in zip(images, records):
        if ai:
            big, note = ig.upscale_ai(img, req.prompt)
        else:
            big, note = ig.upscale_lanczos(img, 2), ""
        meta = {**{k: rec.get(k) for k in ("prompt", "negative_prompt", "model", "scheduler", "steps", "guidance", "seed")},
                "gen_type": "upscale", "extra": {"parent_id": rec.get("id"), "method": st.session_state.gen_up_mode, "note": note}}
        out_imgs.append(big)
        out_recs.append(storage_manager.save_generated_image(big, meta) if req.save else {**meta, "id": None, "width": big.width, "height": big.height, "favorite": 0})
    return out_imgs, out_recs


# ----------------------------------------------------------------------------- dialogs + actions
@st.dialog("Full-screen preview", width="large")
def _fullscreen(img):
    st.image(img, use_container_width=True)


def _delete_current(sel: int) -> None:
    res = st.session_state["gen_result"]
    rec = res["records"][sel]
    if rec.get("id"):
        storage_manager.delete_image(rec["id"])
    res["images"].pop(sel)
    res["records"].pop(sel)
    st.session_state.gen_sel = 0
    if not res["images"]:
        st.session_state.pop("gen_result")
    notify("Image deleted.", "success")
    st.rerun()


def _icon_row(res: dict, sel: int) -> None:
    img, rec = res["images"][sel], res["records"][sel]
    c = st.columns(4)
    c[0].download_button("", to_bytes(img, "PNG"), file_name=f"studio_{rec.get('seed', 'img')}.png", mime="image/png",
                         icon=":material/download:", key=f"dl_icon_{sel}", help="Download PNG", use_container_width=True)
    if c[1].button("", icon=":material/fullscreen:", key=f"fs_icon_{sel}", help="Full-screen preview", use_container_width=True):
        _fullscreen(img)
    c[2].button("", icon=":material/favorite:" if rec.get("favorite") else ":material/favorite_border:", key=f"fav_icon_{sel}",
                help="Favorite", on_click=_cb_favorite, args=(sel,), use_container_width=True, disabled=not rec.get("id"))
    if c[3].button("", icon=":material/delete:", key=f"del_icon_{sel}", help="Delete image", use_container_width=True):
        _delete_current(sel)


def _full_actions(res: dict, sel: int) -> None:
    img, rec = res["images"][sel], res["records"][sel]
    with st.container(border=True):
        panel_title("🛠️", "Image actions")
        with st.container(key="act_row"):   # lets the buttons flow onto a second line on narrow windows instead of wrapping their text
            c = st.columns(8)
            c[0].download_button("PNG", to_bytes(img, "PNG"), file_name=f"studio_{rec.get('seed', 'img')}.png", mime="image/png",
                                 key="act_png", icon=":material/download:", use_container_width=True)
            c[1].download_button("JPEG", to_bytes(img, "JPEG"), file_name=f"studio_{rec.get('seed', 'img')}.jpg", mime="image/jpeg",
                                 key="act_jpg", icon=":material/download:", use_container_width=True)
            if c[2].button("Regenerate", key="act_regen", icon=":material/refresh:", use_container_width=True):
                _request_run(rec, "regenerate")
            if c[3].button("Variation", key="act_var", icon=":material/shuffle:", use_container_width=True, disabled=not rec.get("id")):
                st.session_state["gen_variation_source"] = rec["id"]
                _request_run(rec, "variation")
            with c[4].popover("Copy prompt", icon=":material/content_copy:", use_container_width=True):
                st.code(rec.get("prompt", ""), language=None, wrap_lines=True)
            if c[5].button("Reuse settings", key="act_reuse", icon=":material/tune:", use_container_width=True):
                st.session_state["gen_reuse_notice"] = True
                _request_run(rec, "", keep_seed=True)
            with c[6].popover("Upscale", icon=":material/zoom_out_map:", use_container_width=True):
                mode = st.radio("Method", ["Fast (Lanczos 2x)", "AI (SD x4 upscaler)"], key="act_up_mode")
                st.caption("Lanczos is instant and invents no detail. The AI upscaler adds detail but needs a GPU and shrinks large inputs to fit memory.")
                if st.button("Upscale this image", key="act_up_go", type="primary"):
                    with st.spinner("Upscaling..."):
                        try:
                            big, note = ig.upscale_ai(img, rec.get("prompt", "")) if mode.startswith("AI") else (ig.upscale_lanczos(img, 2), "")
                            meta = {**{k: rec.get(k) for k in ("prompt", "negative_prompt", "model", "scheduler", "steps", "guidance", "seed")},
                                    "gen_type": "upscale", "extra": {"parent_id": rec.get("id"), "method": mode, "note": note}}
                            new_rec = storage_manager.save_generated_image(big, meta)
                            res["images"].append(big)
                            res["records"].append(new_rec)
                            st.session_state.gen_sel = len(res["images"]) - 1
                            notify(f"Upscaled to {big.width}×{big.height}. {note}", "success")
                            st.rerun()
                        except ig.GenerationError as exc:
                            st.error(str(exc))
            if c[7].button("Full screen", key="act_fs", icon=":material/fullscreen:", use_container_width=True):
                _fullscreen(img)
        c2 = st.columns([1, 2, 1])
        c2[0].button("Favorite" if not rec.get("favorite") else "Unfavorite", key="act_fav", icon=":material/favorite:",
                     on_click=_cb_favorite, args=(sel,), disabled=not rec.get("id"), use_container_width=True)
        with c2[1]:
            confirm_button("Delete image", "act_del", lambda: _delete_current(sel), "Delete this image and its metadata permanently?")
        st.caption(f"Seed {rec.get('seed')} · {rec.get('model')} · {rec.get('scheduler')} · {rec.get('steps')} steps · guidance {rec.get('guidance')}"
                   + (f" · LoRA {rec['lora']} ×{rec['lora_scale']}" if rec.get("lora") else ""))


# ----------------------------------------------------------------------------- main render
def render(home: bool = False) -> None:
    models = _available_models()
    if not models:
        st.error("No Stable Diffusion model is installed and the Hugging Face Hub is unreachable. Connect to the internet once so the weights can download.")
        return
    _init_state(models)
    _consume_apply()
    s = st.session_state
    info = gpu_manager.get_device_info()

    if not home:
        st.markdown("<div class='page-title'>AI Image Generator</div><div class='page-sub'>Describe an image, tune the diffusion settings, and generate with Stable Diffusion.</div>", unsafe_allow_html=True)

    left, right = st.columns([1.65, 1])
    slot = right.empty()   # fixed place for the result panel, so it can be replaced the moment a new run starts
    pending = s.pop("gen_pending_run", None)

    with left, st.container(border=True):
        # Same header as the original (title left, "Quick Presets:" + buttons right). The row may wrap to a second line on narrow windows.
        with st.container(key="preset_head"):
            t1, t2 = st.columns([1.1, 2.6], vertical_alignment="center")
            with t1:
                panel_title("✦", "AI Image Generator", MODEL_REGISTRY[s.gen_model]["family"].upper() if s.gen_model else "")
            with t2:
                qc = st.columns([1.1, .8, .9, .8, .8, .5], vertical_alignment="center")
                qc[0].markdown("<div class='preset-label'>Quick Presets:</div>", unsafe_allow_html=True)
                for col, name in zip(qc[1:5], QUICK):
                    col.button(name, key=f"qp_{name}", on_click=_cb_preset, args=(name,), use_container_width=True)
                with qc[5].popover("▾", help="More style presets"):
                    for name in STYLE_PRESETS:
                        if name not in QUICK:
                            st.button(name, key=f"qp2_{name}", on_click=_cb_preset, args=(name,), use_container_width=True)

        p1, p2 = st.columns([1.2, 1])
        p1.text_area("Enter your prompt ✨", key="gen_prompt", height=118, max_chars=1000, placeholder="A serene Japanese landscape with cherry blossoms, mountain and a beautiful lake at sunset")
        p2.text_area("Negative prompt (optional)", key="gen_neg", height=118, max_chars=500, placeholder="blurry, low quality, distorted...")
        b = st.columns(4)
        b[0].button("Enhance prompt", key="btn_enh", icon=":material/auto_fix_high:", on_click=_cb_enhance, use_container_width=True)
        b[1].button("Random prompt", key="btn_rand", icon=":material/casino:", on_click=_cb_random, use_container_width=True)
        with b[2].popover("History", icon=":material/history:", use_container_width=True):
            hist = storage_manager.list_prompt_history(15)
            if not hist:
                st.caption("Prompts you generate with are listed here.")
            for h in hist:
                st.button(h["prompt"][:60], key=f"hist_{h['id']}", on_click=_cb_history, args=(h["prompt"], h["negative_prompt"]), use_container_width=True)
        with b[3].popover("Clear", icon=":material/delete_sweep:", use_container_width=True):
            st.caption("Clear both prompt fields?")
            st.button("Yes, clear", key="btn_clear_yes", on_click=_cb_clear, type="primary")
        if s.get("gen_enh_msg"):
            st.caption(s.pop("gen_enh_msg"))

        # The Model dropdown gets a full-width row so its whole name is readable; Resolution, Steps, Guidance and Seed share the next row.
        # The widget owns st.session_state["gen_model"]; never assign to that key after the widget is created.
        st.selectbox("Model", list(models), key="gen_model", format_func=lambda k: f"{k} ({'ready' if models[k]['state'] == 'cached' else 'downloads on first use'})")
        options = _res_options(s.gen_model)
        if s.gen_res not in options:
            s.gen_res = options[0]
        r = st.columns([1.8, .9, 1, 1.1])
        r[0].selectbox("Resolution", options, key="gen_res")
        r[1].number_input("Steps", 1, 150, key="gen_steps")
        r[2].number_input("Guidance Scale", 0.0, 30.0, step=0.5, key="gen_guidance")
        r[3].number_input("Seed", -1, 2 ** 32 - 1, key="gen_seed", help="-1 picks a random seed. Re-use a seed to reproduce an image.")
        cfg = MODEL_REGISTRY[s.gen_model]
        st.caption(f"{s.gen_model} v{cfg['version']} · recommended {cfg['native'][0]}×{cfg['native'][1]} · {cfg['memory_note']} · {models[s.gen_model]['detail']}")

        t = st.columns(3)
        t[0].toggle("High Quality Mode", key="gen_hq", help="Applies a tested configuration: at least 35 steps, guidance 7.5, DPM++ 2M Karras.")
        t[1].toggle("Enable Upscaler", key="gen_upscale")
        t[2].toggle("Enable LoRA", key="gen_use_lora")
        if s.gen_upscale:
            st.radio("Upscale method", ["Fast (Lanczos 2x)", "AI (SD x4 upscaler)"], key="gen_up_mode", horizontal=True)
        if s.gen_use_lora:
            compat = [a for a in lora_registry.list_adapters() if lora_registry.check_compatibility(a["path"], s.gen_model)[0]]
            if not compat:
                other = sorted({(a["family"] or "unknown").upper() for a in lora_registry.list_adapters()})
                st.warning(f"No LoRA adapter compatible with {s.gen_model} was found."
                           + (f" Your saved adapters were trained for {', '.join(other)} and cannot be used with this model." if other else "")
                           + f" Open LoRA Fine-tuning, choose '{s.gen_model}' as the Base model and train one.")
            else:
                names = [a["name"] for a in compat]
                if s.gen_lora not in names:
                    s.gen_lora = names[0]
                lc = st.columns([2, 1])
                lc[0].selectbox("LoRA adapter", names, key="gen_lora")
                lc[1].slider("Adapter scale", 0.0, 2.0, step=0.05, key="gen_lora_scale")

        with st.expander("Advanced settings", expanded=not home):
            a = st.columns(4)
            a[0].selectbox("Scheduler", SCHEDULER_NAMES, key="gen_scheduler")
            a[1].number_input("Batch size", 1, 8, key="gen_batch")
            a[2].selectbox("Aspect ratio", ["From resolution"] + list(ASPECT_RATIOS), key="gen_aspect")
            a[3].toggle("Save to gallery", key="gen_save")
            if s.gen_res == "Custom" and s.gen_aspect == "From resolution":
                cc = st.columns(2)
                mx = cfg["max_side"]
                # a value carried over from the gallery or from another model can exceed this model's limit
                # (e.g. 2048 on an SDXL-size model): clamp it before the widgets are created, or Streamlit raises.
                for dim_key in ("gen_w", "gen_h"):
                    s[dim_key] = max(256, min(int(s[dim_key]) // 8 * 8, mx))
                cc[0].slider("Width", 256, mx, step=8, key="gen_w")
                cc[1].slider("Height", 256, mx, step=8, key="gen_h")
            st.toggle("Use the MagicPrompt model for 'Enhance prompt' (downloads ~500 MB, otherwise rules are used)", key="gen_enh_model")
            up = st.file_uploader("Image-to-image: start from this image (optional)", type=["png", "jpg", "jpeg", "webp"], key="up_init")
            if up is not None:
                st.slider("Strength (how far to move away from the image)", 0.05, 1.0, step=0.05, key="gen_strength")
                st.image(up, width=160)

        w, h = _dims(s.gen_model)
        level, msg = gpu_manager.check_generation_request(cfg["family"], w, h, int(s.gen_batch), info)
        {"ok": st.caption, "warn": st.warning, "error": st.error}[level](f"{w}×{h}, batch {int(s.gen_batch)}: {msg}")

        clicked = st.button("Generate Image  →", key="gen_go", type="primary", use_container_width=True, icon=":material/auto_awesome:")
        if clicked:
            init = Image.open(up).convert("RGB") if up is not None else None
            _execute(_build_request(init_image=init, gen_type="img2img" if init else "txt2img"), "Generating image", slot)
        elif pending == "regenerate":
            _execute(_build_request(), "Regenerating", slot)
        elif pending == "variation":
            src = storage_manager.get_image(s.pop("gen_variation_source", 0)) if s.get("gen_variation_source") else None
            base = storage_manager.load_pil(src) if src else None
            if base is None:
                st.error("The source image file could not be found.")
            else:
                _execute(_build_request(init_image=base, strength=0.55, gen_type="variation", extra={"parent_id": src["id"]}), "Creating a variation", slot)
        note = s.pop("gen_done_note", None)
        if note:
            st.success(note["label"], icon=":material/check_circle:")
            for w_msg in note["warnings"]:
                st.warning(w_msg)

    res = s.get("gen_result")
    with slot.container(border=True):
        panel_title("🖼️", "Generated Result")
        if not res:
            st.markdown("<div style='height:380px;display:flex;align-items:center;justify-content:center;text-align:center;color:var(--muted);border:1px dashed var(--line);border-radius:14px;padding:1rem'>Your image appears here.<br>Write a prompt and press Generate.</div>", unsafe_allow_html=True)
        else:
            n = len(res["images"])
            if s.gen_sel >= n:
                s.gen_sel = 0
            sel = s.gen_sel
            _icon_row(res, sel)
            rec = res["records"][sel]
            st.image(res["images"][sel], use_container_width=True, caption=f"{rec['width']} × {rec['height']} · {res['elapsed']:.1f}s · seed {rec.get('seed')}")
            if n > 1:
                st.radio("Image", list(range(n)), key="gen_sel", horizontal=True, format_func=lambda i: f"#{i + 1}")

    if not home and res:
        _full_actions(res, s.gen_sel)
    if not home:
        _comparison_section(models)


def _comparison_section(models: dict) -> None:
    s = st.session_state
    with st.expander("Compare Stable Diffusion 1.5 and SDXL on this prompt"):
        both = [k for k in MODEL_REGISTRY if k in models]
        if len(both) < 2:
            st.info("Both models must be available to compare them.")
            return
        st.caption("Runs the same prompt and seed on both models at their native resolution (the GPU loads one model at a time, so this takes a while).")
        if st.button("Run comparison", key="gen_cmp_go"):
            if not s.gen_prompt.strip():
                st.error("Enter a prompt first.")
                return
            from modules import evaluation as ev
            seed = random.randint(0, 2 ** 31) if int(s.gen_seed) < 0 else int(s.gen_seed)
            bar = st.progress(0.0, text="Starting...")
            try:
                rows = ev.compare_diffusion_models([s.gen_prompt], both, seed, {both[0]: 30, both[1]: 30}, s.gen_scheduler,
                                                   float(s.gen_guidance), lambda f, t: bar.progress(min(1.0, f), text=t))
            except Exception as exc:
                st.error(f"Comparison failed: {exc}")
                return
            bar.empty()
            for col, r in zip(st.columns(len(rows)), rows):
                col.image(r["image"], use_container_width=True)
                col.markdown(f"**{r['model']}**  \n{r['resolution']} · {r['time_s']}s · VRAM {r['peak_vram_gb']} GB · CLIPScore {r['clip_score']}")
