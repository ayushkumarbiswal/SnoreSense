"""Shared UI helpers: styling, cached resources, session, charts."""
from __future__ import annotations

from datetime import datetime, timezone

import plotly.graph_objects as go
import streamlit as st

from .. import db
from ..config import FLAG_THRESHOLD, LABELS
from ..models import ModelBundle

TEAL, AMBER, RED, BLUE, MUTED = "#5CD6C0", "#F5A524", "#EF6B73", "#7C9CFF", "#8A93B8"
MODEL_COLORS = {"rf": "#7C9CFF", "svm": "#C792EA", "cnn": "#F5A524", "ensemble": "#5CD6C0"}
ROLE_NAMES = {"home": "Home user", "caregiver": "Caregiver", "admin": "Administrator"}

CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1180px;}
h1, h2, h3 {letter-spacing: -0.01em;}
.eyebrow {color: #5CD6C0; font-size: 0.82rem; font-weight: 600; letter-spacing: .02em; margin-bottom: -0.4rem;}
.muted {color: #9AA3C7;}
.badge {display:inline-flex; align-items:center; gap:.35rem; padding:.18rem .7rem; border-radius:999px;
        font-weight:600; font-size:.88rem; border:1px solid;}
.badge-ok {color:#5CD6C0; background:rgba(92,214,192,.12); border-color:rgba(92,214,192,.45);}
.badge-warn {color:#F5A524; background:rgba(245,165,36,.12); border-color:rgba(245,165,36,.45);}
.badge-info {color:#9AA3C7; background:rgba(154,163,199,.10); border-color:rgba(154,163,199,.35);}
.st-key-hero {padding: 1.4rem 1.6rem 1.5rem; border-radius: 1rem; border:1px solid #2A3360;
       background: linear-gradient(135deg, #1B2350 0%, #151C3A 70%);}
.st-key-hero h2 {font-size: 1.6rem;}
.result-label {font-family: Fraunces, serif; font-size: 2rem; font-weight: 600; line-height: 1.2; margin: .2rem 0 .4rem;}
.disclaimer {border-left: 3px solid #F5A524; background: rgba(245,165,36,.08); padding:.7rem 1rem; border-radius:.4rem;}
.sidebar-note {font-size: .78rem; color:#8A93B8;}
div[data-testid="stMetricValue"] {font-family: Fraunces, serif;}
</style>
"""


def setup_page():
    st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models…")
def get_bundle(version_key: str) -> ModelBundle:
    return ModelBundle.active()


def bundle() -> ModelBundle:
    from ..models import load_registry
    reg = load_registry()
    active = next(v for v in reg["versions"] if v["id"] == reg["active"])
    return get_bundle(f"{active['id']}:{active.get('decision_model')}")


# --------------------------------------------------------------------------- session
def user() -> dict:
    return st.session_state.get("user")


def refresh_user():
    u = user()
    if u:
        fresh = db.get_user(u["id"])
        if fresh is None:
            st.session_state.clear()
            st.rerun()
        st.session_state.user = fresh


def nav_to(page_key: str):
    st.switch_page(st.session_state.pages[page_key])


# --------------------------------------------------------------------------- formatting
def badge(label: int | None, text: str | None = None) -> str:
    if label is None:
        return f'<span class="badge badge-info">{text or "—"}</span>'
    cls, icon = ("badge-warn", "⚠") if label == 1 else ("badge-ok", "✓")
    short = "Irregular pattern" if label == 1 else "Normal"
    return f'<span class="badge {cls}">{icon} {text or short}</span>'


def fmt_date(iso: str, with_time: bool = True) -> str:
    d = datetime.fromisoformat(iso)
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    d = d.astimezone()
    return d.strftime("%d %b %Y, %I:%M %p" if with_time else "%d %b %Y")


def fmt_duration(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 60} min {sec % 60:02d} s" if sec >= 60 else f"{sec} s"


def plotly_layout(fig: go.Figure, height: int = 300, **kw) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#C9CFEA", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), **kw)
    fig.update_xaxes(gridcolor="#232B52", zerolinecolor="#232B52")
    fig.update_yaxes(gridcolor="#232B52", zerolinecolor="#232B52")
    return fig


# --------------------------------------------------------------------------- result charts
def waveform_figure(result: dict) -> go.Figure:
    env = result["envelope"]
    fig = go.Figure()
    for w in result["windows"]:
        if w["flagged"]:
            fig.add_vrect(x0=w["start"], x1=w["end"], fillcolor=AMBER, opacity=0.13, line_width=0, layer="below")
    for p in result["pauses"]:
        fig.add_vrect(x0=p["start"], x1=p["end"], fillcolor=RED, opacity=0.28, line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=env["t"] + env["t"][::-1], y=env["max"] + env["min"][::-1], fill="toself",
                             line=dict(width=0.6, color=TEAL), fillcolor="rgba(92,214,192,0.55)",
                             hoverinfo="skip", name="Waveform"))
    # legend proxies
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=12, color=RED, symbol="square"),
                             name="Breathing pause ≥ 10 s"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=12, color=AMBER, symbol="square", opacity=.5),
                             name="Window flagged as irregular"))
    fig.update_yaxes(showticklabels=False, title=None)
    fig.update_xaxes(title="Time (seconds)")
    return plotly_layout(fig, 280)


def window_figure(result: dict) -> go.Figure:
    ws = result["windows"]
    mid = [(w["start"] + w["end"]) / 2 for w in ws]
    fig = go.Figure()
    names = {"rf": "Random Forest", "svm": "SVM", "cnn": "CNN", "ensemble": "Ensemble"}
    for k in ("rf", "svm", "cnn", "ensemble"):
        fig.add_trace(go.Scatter(x=mid, y=[w[k] for w in ws], mode="lines+markers", name=names[k],
                                 line=dict(color=MODEL_COLORS[k], width=3 if k == "ensemble" else 1.5,
                                           dash=None if k == "ensemble" else "dot")))
    fig.add_hline(y=FLAG_THRESHOLD, line=dict(color=MUTED, dash="dash"), annotation_text="flag threshold",
                  annotation_position="top left")
    fig.update_yaxes(range=[-0.02, 1.02], title="P(irregular)")
    fig.update_xaxes(title="Window centre (seconds)")
    return plotly_layout(fig, 260)


def label_text(label: int) -> str:
    return LABELS[label]
