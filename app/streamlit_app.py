"""sarcasm-radar Streamlit demo.

Paste a tweet, get a sarcasm probability and a token-level LIME
explanation. Talks to the FastAPI service over HTTP — no model code
is imported here, so the demo container stays slim and the same
Streamlit app can target either ``http://localhost:8000`` or a
deployed API.
"""

from __future__ import annotations

import os
from typing import Any

import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("SARCASM_RADAR_API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 15.0  # /explain can take a few seconds on transformer + LIME

EXAMPLE_TEXTS = [
    "haa beta, very smart move",
    "mast plan, ab toh sab kuch theek ho hi jayega",
    "great, another Monday. My favorite day.",
    "amazing service, only took them 3 emails to respond",
    "Got my first chai of the day, feeling good",
    "Mumbai local was on time today, surprised but happy",
    "Made paneer butter masala at home, turned out great",
    "obviously the wifi works fine until I open Zoom",
]


# -------------------------------------------------------------- API helpers
def fetch_health() -> dict[str, Any] | None:
    try:
        r = requests.get(f"{API_URL}/health", timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            return dict(r.json())
    except requests.RequestException:
        pass
    return None


def call_predict(text: str) -> dict[str, Any] | None:
    try:
        r = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        st.error(f"Could not reach API at {API_URL}: {e}")
        return None
    if r.status_code != 200:
        st.error(f"API returned {r.status_code}: {r.text[:200]}")
        return None
    return dict(r.json())


def call_explain(text: str) -> dict[str, Any] | None:
    try:
        r = requests.post(f"{API_URL}/explain", json={"text": text}, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        st.error(f"Could not reach API at {API_URL}: {e}")
        return None
    if r.status_code != 200:
        st.error(f"API returned {r.status_code}: {r.text[:200]}")
        return None
    return dict(r.json())


# -------------------------------------------------------------- visuals
def score_gauge(probability: float, threshold: float) -> go.Figure:
    bar_color = "#dd8452" if probability >= threshold else "#4c72b0"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability,
            number={"valueformat": ".3f"},
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "P(sarcastic)"},
            gauge={
                "axis": {"range": [0, 1], "tickformat": ".1f"},
                "bar": {"color": bar_color},
                "steps": [
                    {"range": [0, threshold], "color": "#e8eef5"},
                    {"range": [threshold, 1.0], "color": "#fce8da"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 2},
                    "thickness": 0.85,
                    "value": threshold,
                },
            },
        )
    )
    fig.update_layout(height=260, margin={"t": 40, "b": 10, "l": 30, "r": 30})
    return fig


def token_weight_bar(tokens: list[dict[str, Any]]) -> go.Figure:
    sorted_tokens = sorted(tokens, key=lambda t: abs(t["weight"]), reverse=True)
    sorted_tokens = sorted_tokens[::-1]  # smallest at bottom for horizontal bar
    colors = ["#dd8452" if t["weight"] >= 0 else "#4c72b0" for t in sorted_tokens]
    fig = go.Figure(
        go.Bar(
            x=[t["weight"] for t in sorted_tokens],
            y=[t["token"] for t in sorted_tokens],
            orientation="h",
            marker_color=colors,
        )
    )
    fig.update_layout(
        title="LIME token contributions (orange pushes toward SARCASTIC, blue toward not)",
        xaxis_title="weight",
        height=400,
        margin={"t": 50, "b": 30, "l": 80, "r": 30},
    )
    return fig


# -------------------------------------------------------------- layout
st.set_page_config(page_title="sarcasm-radar", page_icon="🎯", layout="wide")

st.title("sarcasm-radar")
st.caption(
    "Sarcasm detection for Indian English tweets. Handles Hinglish "
    "code-switching (haa beta, kya baat hai, mast plan) thanks to "
    "XLM-RoBERTa's multilingual pretraining."
)

with st.sidebar:
    st.header("Service")
    health = fetch_health()
    if health is None:
        st.error("API unreachable")
        st.caption(f"Looked at: {API_URL}")
    else:
        st.success(f"Model {health['model_version']} live")
        st.metric("Active threshold", f"{health['threshold']:.3f}")
        st.caption(f"Backend: `{health['model_kind']}`")

    st.divider()
    st.caption(f"API: `{API_URL}`")

# -------------------------------------------------------------- main
st.subheader("Score a tweet")

if "text" not in st.session_state:
    st.session_state.text = EXAMPLE_TEXTS[0]

with st.expander("Pick an example", expanded=False):
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLE_TEXTS):
        with cols[i % 2]:
            if st.button(example, key=f"ex_{i}", use_container_width=True):
                st.session_state.text = example
                st.session_state.pop("last_predict", None)
                st.session_state.pop("last_explain", None)

text = st.text_area(
    "Text",
    value=st.session_state.text,
    height=100,
    label_visibility="collapsed",
)
st.session_state.text = text

action_col_a, action_col_b, action_col_c = st.columns([1, 1, 3])
with action_col_a:
    if st.button("Score", type="primary", use_container_width=True):
        st.session_state.last_predict = call_predict(text)
        st.session_state.pop("last_explain", None)
with action_col_b:
    if st.button("Explain", use_container_width=True):
        with st.spinner("Running LIME (a few seconds)..."):
            st.session_state.last_explain = call_explain(text)

# -------------------------------------------------------------- results
result: dict[str, Any] | None = st.session_state.get("last_explain") or st.session_state.get(
    "last_predict"
)
if result is None:
    st.info("Click *Score* for a probability, or *Explain* for token-level LIME weights.")
else:
    gauge_col, info_col = st.columns([1, 1])
    with gauge_col:
        st.plotly_chart(
            score_gauge(result["probability"], result["threshold"]),
            use_container_width=True,
        )
    with info_col:
        emoji = "🟠" if result["decision"] == "SARCASTIC" else "🔵"
        st.metric("Decision", f"{emoji} {result['decision']}")
        st.metric("Model", result["model_version"])

    if "tokens" in result:
        st.plotly_chart(token_weight_bar(result["tokens"]), use_container_width=True)
        with st.expander("Raw token weights"):
            st.dataframe(result["tokens"], hide_index=True, use_container_width=True)
