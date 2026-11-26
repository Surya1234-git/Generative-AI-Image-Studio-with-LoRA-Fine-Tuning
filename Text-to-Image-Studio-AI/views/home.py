"""Home: hero banner + the main generator (as in the reference design) + dashboard widgets."""
import streamlit as st

from components import dashboard
from views import generator


def render() -> None:
    st.markdown("""<div class='hero'><h1>Text-to-Image Studio <em>AI</em></h1>
        <p>Turn your imagination into stunning visuals with the power of Generative AI</p>
        <div class='ticks'><span>Stable Diffusion XL</span><span>High Quality Output</span><span>Advanced Controls</span><span>Multiple Features</span></div></div>""",
                unsafe_allow_html=True)
    generator.render(home=True)
    left, right = st.columns([1.55, 1])
    with left:
        dashboard.creations_strip()
    with right:
        dashboard.stats_panel()
        dashboard.popular_features()
    dashboard.module_cards()
