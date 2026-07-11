"""
Decoding Teacher Sentiment -- Streamlit Demo

Shows pre-computed LIME explanations for sample reviews
without needing TensorFlow or a trained model.
"""

import json
import os
import streamlit as st
import numpy as np

# --- Page config ---
st.set_page_config(
    page_title="Decoding Teacher Sentiment",
    page_icon="favicon",
    layout="centered",
)

# --- Load example data ---
demo_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(demo_dir, "example_data.json")

with open(data_path, "r") as f:
    examples = json.load(f)


# --- Helpers ---
def render_word_bars(pairs, color_positive="#4CAF50", color_negative="#F44336"):
    """Draw horizontal bars showing how much each word influenced the prediction."""
    items = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)
    max_abs = max(abs(w) for _, w in items) if items else 1

    html_parts = []
    for word, weight in items:
        bar_width = abs(weight) / max_abs * 100
        color = color_positive if weight > 0 else color_negative
        sign = "+" if weight > 0 else ""
        html_parts.append(f"""
        <div style="display:flex; align-items:center; margin:4px 0; gap:8px; font-size:14px;">
            <div style="width:100px; text-align:right; font-weight:500;">{word}</div>
            <div style="flex:1; max-width:200px;">
                <div style="
                    height:22px; width:{bar_width}%;
                    background:{color};
                    border-radius:4px;
                    opacity:0.85;
                    display:flex; align-items:center; padding-left:6px;
                    color:white; font-size:12px; font-weight:600;
                ">{sign}{weight:.2f}</div>
            </div>
        </div>
        """)
    return "".join(html_parts)


def render_prediction_bar(value, max_val=5.0, color="#4A90D9"):
    """Draw a small bar showing the score value."""
    pct = (value / max_val) * 100
    return f"""
    <div style="display:flex; align-items:center; gap:8px; margin:2px 0;">
        <div style="width:100px; text-align:right; font-size:14px;">
            <strong>{value:.1f}</strong> / {max_val:.0f}
        </div>
        <div style="flex:1; max-width:200px;">
            <div style="height:18px; width:{pct}%; background:{color};
                 border-radius:4px; opacity:0.8;"></div>
        </div>
    </div>
    """


# --- Header ---
st.title("Decoding Teacher Sentiment")
st.markdown(
    """
    A multi-task 1D CNN that predicts Quality and Difficulty scores
    from RateMyProfessor reviews, with LIME word-level explanations.

    Select a review below to see which words drove the model's predictions.
    """
)

st.divider()

# --- Review selector ---
review_labels = [
    f'{ex["review"][:70]}...' if len(ex["review"]) > 70 else ex["review"]
    for ex in examples
]
selected_idx = st.selectbox(
    "Choose a review to explore:",
    range(len(examples)),
    format_func=lambda i: f'Review #{i+1}: "{review_labels[i]}"',
)

ex = examples[selected_idx]

# --- Display review and scores ---
with st.container(border=True):
    st.markdown(f"**Review text:** _{ex['review']}_")
    st.markdown("#### Predicted vs. Actual Scores")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Quality**" + render_prediction_bar(ex["actual_quality"], color="#4CAF50"), unsafe_allow_html=True)
        st.markdown(render_prediction_bar(ex["predicted_quality"], color="#81C784"), unsafe_allow_html=True)
        st.caption("Actual (top) . Predicted (bottom)")
    with col2:
        st.markdown("**Difficulty**" + render_prediction_bar(ex["actual_difficulty"], color="#F44336"), unsafe_allow_html=True)
        st.markdown(render_prediction_bar(ex["predicted_difficulty"], color="#E57373"), unsafe_allow_html=True)
        st.caption("Actual (top) . Predicted (bottom)")

st.divider()

# --- LIME explanations ---
tab1, tab2 = st.tabs(["Quality Explanation", "Difficulty Explanation"])

with tab1:
    st.markdown(
        "Words in green pushed the quality score up; "
        "words in red pushed it down."
    )
    st.markdown(
        render_word_bars(ex["lime_quality"], color_positive="#4CAF50", color_negative="#F44336"),
        unsafe_allow_html=True,
    )

with tab2:
    st.markdown(
        "Words in green pushed the difficulty score up; "
        "words in red pushed it down."
    )
    st.markdown(
        render_word_bars(ex["lime_difficulty"], color_positive="#F44336", color_negative="#4CAF50"),
        unsafe_allow_html=True,
    )

st.divider()

# --- Footer ---
st.caption(
    "Built with Streamlit - "
    "[View on GitHub](https://github.com/jsmarkertjs/decoding-teacher-sentiment) - "
    "DATA-441 NLP Final Project, American University"
)