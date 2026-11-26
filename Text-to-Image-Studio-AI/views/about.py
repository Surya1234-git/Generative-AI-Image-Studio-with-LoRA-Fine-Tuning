"""About page: overview, technology actually installed, architecture, documentation, task matrix, credits."""
from __future__ import annotations

import streamlit as st

from components.ui_components import panel_title
from config import settings
from modules import evaluation as ev

TASK_MATRIX = [
    ("Task 1", "Fine-tune a pre-trained text-to-image model", "Stable Diffusion 1.5 LoRA fine-tuning with before/after comparison", "modules/lora_training.py, modules/lora_registry.py, views/lora.py"),
    ("Task 2", "CGAN generating images from text labels/categories", "Conditional GAN (class label or CLIP caption conditioning)", "modules/gan_models.py (CGANGenerator/Discriminator), modules/gan_trainer.py, views/gan.py"),
    ("Task 3", "Tokenize and encode descriptions (Hugging Face Transformers)", "CLIP / Sentence-Transformers embeddings, similarity, PCA/UMAP", "modules/text_embeddings.py, views/embeddings.py"),
    ("Task 4", "Analyse a public dataset (classes, description length, resolution, captioned images)", "Dataset Explorer with Oxford-102 Flowers built in; statistics, plots, sample viewer", "modules/dataset_explorer.py, views/datasets.py"),
    ("Task 5", "Improve a GAN with self- or cross-attention", "Attention CGAN (SAGAN self-attention in G and D), real attention maps, measured comparison", "modules/gan_models.py (SelfAttention2d, AttnCGAN*), views/gan.py, modules/evaluation.py"),
    ("Task 6", "End-to-end text preprocessing → embedding → GAN pipeline", "GAN Studio 'Text → GAN pipeline' tab (text-conditioned runs)", "modules/gan_trainer.py (text_to_image), views/gan.py"),
]
LICENSES = [("Stable Diffusion 1.5", "CreativeML OpenRAIL-M"), ("Stable Diffusion XL Base 1.0", "CreativeML OpenRAIL++-M"),
            ("CLIP (OpenAI)", "MIT"), ("Sentence-Transformers models", "Apache-2.0"), ("Stable Diffusion x4 upscaler", "CreativeML OpenRAIL++-M"),
            ("MagicPrompt-Stable-Diffusion (optional)", "MIT"), ("Oxford 102 Flowers", "Research use; see the dataset page")]


def render() -> None:
    st.markdown("<div class='page-title'>About</div><div class='page-sub'>What this platform is, how it is built, and how to run it.</div>", unsafe_allow_html=True)
    tabs = st.tabs(["Overview", "Technology", "Architecture", "Documentation", "Task matrix", "Credits & licenses"])
    with tabs[0]:
        st.markdown(f"### {settings.APP_NAME} {settings.APP_VERSION}")
        st.write("An integrated Generative AI platform that combines diffusion models, GAN architectures, text embeddings, dataset analysis, model evaluation and parameter-efficient fine-tuning in one workspace. Built for the ElevanceSkills 'Real Time Text-to-Image Generator' internship.")
        st.markdown("**Objectives:** generate images from text with Stable Diffusion; train and compare conditional GANs with and without attention; understand text embeddings; analyse datasets; fine-tune with LoRA; evaluate every model with real, reproducible metrics.")
        st.markdown("**Integrations:** Dataset Explorer → GAN Studio and LoRA · Generator → Gallery and Evaluation · GAN Studio → Gallery and Evaluation · LoRA → Generator · Gallery → Generator (reuse prompts and settings).")
    with tabs[1]:
        st.caption("Only packages actually installed in this environment are listed, with their real versions.")
        ver = ev.package_versions()
        cols = st.columns(4)
        for i, (name, v) in enumerate((k, v) for k, v in ver.items() if v != "not installed"):
            cols[i % 4].markdown(f"<div class='tile'><div class='v' style='font-size:1.05rem'>{name}</div><div class='l'>{v}</div></div>", unsafe_allow_html=True)
        missing = [k for k, v in ver.items() if v == "not installed"]
        if missing:
            st.caption("Optional or not installed here: " + ", ".join(missing))
    with tabs[2]:
        st.markdown("""<div class='arch'><div class='box'>Frontend: Streamlit studio (custom CSS, views/)</div><div class='arrow'>▼</div>
        <div class='box'>Python backend: validation, orchestration, background jobs (services/, modules/)</div><div class='arrow'>▼</div>
        <div class='row'><div class='box'>Diffusion<br>SD 1.5 / SDXL (diffusers)</div><div class='box'>GAN<br>CGAN / Attention CGAN (PyTorch)</div><div class='box'>Embeddings<br>CLIP / Sentence-Transformers</div><div class='box'>Fine-tuning<br>LoRA (PEFT)</div></div><div class='arrow'>▼</div>
        <div class='box'>Storage and evaluation: SQLite + images, datasets, checkpoints, metrics, reports</div></div>""", unsafe_allow_html=True)
    with tabs[3]:
        with st.expander("Installation (local)", expanded=True):
            st.code("python -m venv .venv && source .venv/bin/activate   # Windows: .venv\\Scripts\\activate\npip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # pick your CUDA build\npip install -r requirements.txt\nstreamlit run app.py", language="bash")
        with st.expander("Google Colab"):
            st.write("Open notebooks/Text_to_Image_Studio_Colab.ipynb, set Runtime → Change runtime type → GPU, and run the cells in order. The notebook checks the GPU, mounts Google Drive (so images, datasets and checkpoints persist), installs the requirements, starts Streamlit and prints a public link.")
        with st.expander("Model downloads"):
            st.write("Models download automatically from the Hugging Face Hub on first use (SD 1.5 about 2-4 GB, SDXL about 7-10 GB, CLIP 0.6-1.7 GB). Set HF_HOME to cache them elsewhere. The notebook can pre-download them.")
        with st.expander("How to use each module"):
            st.markdown("- **AI Image Generator:** write a prompt, pick a model, press Generate. Results are saved to the Gallery.\n- **Dataset Explorer:** import a dataset (ZIP, images, folder or Oxford-102), Analyze, clean, preprocess, split.\n- **Text Embeddings:** one prompt per line → Generate Embeddings.\n- **GAN Studio:** pick a dataset, architecture, press Start Training; watch the live charts; generate from checkpoints.\n- **Evaluation Center:** train the reference classifier, then evaluate GAN runs (accuracy, ROC/AUC); compare SD 1.5 and SDXL.\n- **LoRA Fine-tuning:** pick images, edit captions, train, test before/after, load the adapter in the generator.\n- **Image Gallery:** search, filter, open, tag, compare, reuse settings.")
        with st.expander("Supported models and GPU requirements"):
            st.table([{"Model": k, "Repo": v["repo"], "Native": f"{v['native'][0]}×{v['native'][1]}", "Memory": v["memory_note"]} for k, v in settings.MODEL_REGISTRY.items()])
            st.write("GAN training (64×64) runs on any CUDA GPU with 4 GB+ and slowly on CPU. LoRA training needs a CUDA GPU (a 15 GB T4 works at 512×512, batch size 1).")
        with st.expander("Known limitations"):
            st.markdown("- LoRA training supports Stable Diffusion 1.5 and SDXL (SDXL needs more VRAM); an adapter only works with the model family it was trained on.\n- Oxford-102 has labels but no human captions; captions are templated ('a photo of <flower>').\n- GAN images at 64×64 are small and, for a student-scale run, far less realistic than diffusion output. Evaluation uses classifier accuracy/AUC and (with enough images) FID.\n- FID below 2048 images per set is only a rough comparison.\n- The light theme is best-effort.\n- Keyboard shortcuts (Alt+1..9, '/') depend on the browser allowing iframe access.")
        with st.expander("Troubleshooting"):
            st.markdown("- **CUDA out of memory:** lower resolution/batch size, pick SD 1.5, or use Settings → Performance → CPU offload 'always'.\n- **Model download fails:** check the internet connection; some networks block huggingface.co.\n- **Black image:** try another seed or scheduler (fp16 issue on some GPUs).\n- **Upload fails through the Colab link:** the notebook disables XSRF protection for the tunnel; re-run the launch cell.\n- **Training stuck on 'paused':** press Resume.")
    with tabs[4]:
        st.table([{"Task": t, "Requirement": r, "Implementation": i, "Code": c} for t, r, i, c in TASK_MATRIX])
    with tabs[5]:
        st.markdown("**Developer:** Surya Prakash M · **Programme:** ElevanceSkills Technology Pvt Ltd, Data Science / Gen AI internship.")
        st.markdown("**References:** Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models* (2022) · Podell et al., *SDXL* (2023) · Hu et al., *LoRA* (2021) · Zhang et al., *Self-Attention GAN* (2019) · Miyato & Koyama, *cGANs with Projection Discriminator* (2018) · Radford et al., *CLIP* (2021).")
        st.table([{"Component": a, "License": b} for a, b in LICENSES])
        st.caption("Check each model's license before commercial use.")
