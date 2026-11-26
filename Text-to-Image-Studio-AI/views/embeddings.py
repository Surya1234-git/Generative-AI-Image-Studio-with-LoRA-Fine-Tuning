"""Text Embeddings page (Module 3 / Task 3)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from components.ui_components import notify, panel_title, style_fig, tiles
from modules import text_embeddings as te
from services import storage_manager

DEFAULT_TEXT = "A golden retriever playing in a garden.\nA brown dog running on grass.\nA futuristic city at night.\nA cyberpunk street with neon lights.\nA bowl of fresh fruit on a table."


def _texts() -> list[str]:
    return [t.strip() for t in st.session_state.emb_text.splitlines() if t.strip()]


def render() -> None:
    st.markdown("<div class='page-title'>Text Embeddings</div><div class='page-sub'>See how text becomes the vectors that image models understand: tokens, embeddings, similarity and maps.</div>", unsafe_allow_html=True)
    st.session_state.setdefault("emb_text", DEFAULT_TEXT)
    st.session_state.setdefault("emb_model", next(iter(te.EMBEDDING_MODELS)))
    with st.container(border=True):
        panel_title("✍️", "Prompts (one per line)")
        st.text_area("Prompts", key="emb_text", height=150, label_visibility="collapsed")
        c = st.columns([3, 2])
        c[0].selectbox("Embedding model", list(te.EMBEDDING_MODELS), key="emb_model")
        kind = te.EMBEDDING_MODELS[st.session_state.emb_model]["kind"]
        c[1].info("CLIP: image-aligned, used by image generators." if kind == "clip" else "Sentence embedding: general text similarity, not used by image generators.")
        st.caption(te.MODEL_EXPLANATIONS[kind])
        compare = st.multiselect("Also compare with (optional)", [m for m in te.EMBEDDING_MODELS if m != st.session_state.emb_model], key="emb_compare")
        b1, b2 = st.columns(2)
        do_tok = b1.button("Tokenize", key="emb_tok_go", use_container_width=True)
        do_emb = b2.button("Generate Embeddings", key="emb_go", type="primary", use_container_width=True)

    texts = _texts()
    if do_tok or do_emb:
        if len(texts) < 1:
            st.error("Enter at least one prompt.")
            return
    if do_tok:
        with st.spinner("Loading tokenizer..."):
            try:
                for item in te.tokenize(texts, st.session_state.emb_model):
                    with st.expander(f"{item['text']}  ·  {item['n_tokens']} tokens" + ("  (truncated to 77)" if item["truncated"] else ""), expanded=len(texts) <= 3):
                        st.code(" | ".join(item["tokens"]), language=None, wrap_lines=True)
                        st.caption("Token ids: " + ", ".join(map(str, item["ids"][:30])) + (" ..." if len(item["ids"]) > 30 else ""))
            except Exception as exc:
                st.error(f"Could not tokenize: {exc}")
    if do_emb:
        if len(texts) < 2:
            st.error("Enter at least two prompts to compare them.")
            return
        with st.spinner("Encoding prompts with the selected model(s)..."):
            try:
                models = [st.session_state.emb_model] + list(compare)
                results = {m: te.embed(texts, m) for m in models}
                seq = te.clip_sequence_shape(texts[0], st.session_state.emb_model) if kind == "clip" else None
            except Exception as exc:
                st.error(f"Embedding failed: {exc}")
                notify(f"Embedding failed: {exc}", "error")
                return
        st.session_state["emb_state"] = {"texts": texts, "results": results, "primary": st.session_state.emb_model, "seq": seq}
        notify(f"Embedded {len(texts)} prompts.", "success")

    state = st.session_state.get("emb_state")
    if not state:
        st.caption("Press Generate Embeddings to see vectors, similarity and plots.")
        return
    texts, primary, E = state["texts"], state["primary"], state["results"][state["primary"]]
    sim = te.cosine_matrix(E)
    tiles([("Prompts", len(texts)), ("Embedding shape", f"[{E.shape[0]}, {E.shape[1]}]"), ("Mean similarity", f"{sim[np.triu_indices_from(sim, 1)].mean():.3f}")])
    tabs = st.tabs(["Vectors", "Similarity", "Maps (PCA / UMAP)", "Model comparison", "Wording test", "Export"])

    with tabs[0]:
        for t, v in zip(texts, E):
            st.markdown(f"**{t}**  \nshape `[1, {E.shape[1]}]` · norm {np.linalg.norm(v):.3f}  \nvector preview: `[{', '.join(f'{x:.3f}' for x in v[:6])}, ...]`")
        if state["seq"]:
            st.info(f"Stable Diffusion conditions on the per-token tensor, not this single vector. For the first prompt it has shape {state['seq']['shape']} (77 tokens × hidden size); first values {state['seq']['sample']}.")
    with tabs[1]:
        st.latex(r"\text{Similarity}(A,B)=\frac{A\cdot B}{\|A\|\,\|B\|}")
        short = [t[:28] + ("…" if len(t) > 28 else "") for t in texts]
        fig = px.imshow(sim, x=short, y=short, zmin=-1, zmax=1, color_continuous_scale="Purples", text_auto=".2f", title="Cosine similarity matrix")
        st.plotly_chart(style_fig(fig, 460), use_container_width=True)
        pairs = te.rank_pairs(sim, texts)
        c1, c2 = st.columns(2)
        c1.markdown("**Most similar**"); c1.dataframe(pairs.head(3), use_container_width=True, hide_index=True)
        c2.markdown("**Least similar**"); c2.dataframe(pairs.tail(3).iloc[::-1], use_container_width=True, hide_index=True)
        st.download_button("Export similarity report", te.similarity_report_md(texts, sim, primary), "similarity_report.md", "text/markdown", key="emb_rep")
    with tabs[2]:
        method = st.radio("Projection", ["PCA", "UMAP"], horizontal=True, key="emb_proj")
        k = st.slider("Clusters (k-means, 1 = none)", 1, min(8, len(texts)), 1, key="emb_k")
        try:
            xy = te.pca_2d(E) if method == "PCA" else te.umap_2d(E)
            df = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "prompt": texts, "cluster": (te.kmeans_labels(E, k).astype(str) if k > 1 else "all")})
            fig = px.scatter(df, x="x", y="y", color="cluster", hover_name="prompt", text=[str(i + 1) for i in range(len(df))], title=f"{method} of {primary}")
            fig.update_traces(marker=dict(size=14), textposition="top center")
            st.plotly_chart(style_fig(fig, 480), use_container_width=True)
            st.caption("Hover a point to read its prompt. Numbers follow the order of your prompt list. Nearby points have similar meaning.")
        except Exception as exc:
            st.warning(str(exc))
    with tabs[3]:
        if len(state["results"]) < 2:
            st.info("Pick another model under 'Also compare with' and generate again.")
        else:
            rows, base_sim = [], sim
            for m, emb in state["results"].items():
                s2 = te.cosine_matrix(emb)
                rows.append({"model": m, "dimension": emb.shape[1], "mean pairwise similarity": round(float(s2[np.triu_indices_from(s2, 1)].mean()), 4),
                             "rank agreement with primary (Spearman)": round(te.similarity_agreement(base_sim, s2), 3)})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("Different models use different dimensions and similarity scales, so compare how they rank pairs rather than raw scores.")
    with tabs[4]:
        a = st.text_input("Prompt A", texts[0], key="emb_wa")
        b = st.text_input("Prompt B (reworded)", "A happy dog playing outside in a yard.", key="emb_wb")
        if st.button("Compare wording", key="emb_wgo"):
            try:
                e = te.embed([a, b], primary)
                st.metric("Cosine similarity", f"{te.cosine_matrix(e)[0, 1]:.4f}")
            except Exception as exc:
                st.error(str(exc))
    with tabs[5]:
        st.download_button("Download embeddings (CSV)", te.embeddings_csv(E, texts), "embeddings.csv", "text/csv", key="emb_csv")
        st.download_button("Download embeddings (NumPy .npy)", te.embeddings_npy(E), "embeddings.npy", key="emb_npy")
        if st.button("Save experiment", key="emb_save"):
            storage_manager.add_experiment("embeddings", f"{primary[:30]} · {len(texts)} prompts", {"model": primary, "prompts": texts},
                                           {"dimension": int(E.shape[1]), "mean_similarity": float(sim[np.triu_indices_from(sim, 1)].mean())})
            notify("Experiment saved. It appears in the Evaluation Center history.", "success")
