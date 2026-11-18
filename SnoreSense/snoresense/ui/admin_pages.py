"""Administrator pages (Module 2 — Ajit Kumar Pradhan: US-10, 11; comparative evaluation)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import db
from ..config import UPLOADS_DIR
from ..models import MODEL_LABELS, delete_version, load_metrics, load_registry, set_active
from .common import AMBER, MODEL_COLORS, MUTED, TEAL, fmt_date, nav_to as go_page, plotly_layout, user

SHORT = {"rf": "Random Forest", "svm": "SVM", "cnn": "CNN", "ensemble": "Ensemble"}


def _usage() -> pd.DataFrame:
    df = pd.DataFrame(db.usage_rows())
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["created"], utc=True)
    df["day"] = df["date"].dt.tz_convert(None).dt.normalize()
    return df


def _active_version() -> dict:
    reg = load_registry()
    return next(v for v in reg["versions"] if v["id"] == reg["active"])


# =========================================================================== dashboard
def admin_dashboard():
    st.markdown('<p class="eyebrow">Overview</p>', unsafe_allow_html=True)
    st.title("Operations dashboard")
    st.markdown('<p class="muted">Aggregated, anonymised activity across all SnoreSense users.</p>', unsafe_allow_html=True)
    df = _usage()
    v = _active_version()
    week_ago = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)
    recent = df[df["date"] >= week_ago] if not df.empty else df
    dm = v.get("decision_model", "ensemble")
    acc = v["test"].get(dm, {}).get("accuracy")

    c = st.columns(4)
    c[0].metric("Total screenings", f"{len(df):,}", f"{len(recent)} this week", border=True)
    c[1].metric("Flagged rate", f"{df['label'].mean() * 100:.0f}%" if len(df) else "—", border=True)
    c[2].metric("Active users (7 days)", recent["uid"].nunique() if len(recent) else 0, border=True)
    c[3].metric("Model accuracy (test set)", f"{acc * 100:.1f}%" if acc else "—",
                f"{v['id']} · {SHORT[dm]}", delta_color="off", delta_arrow="off", border=True)

    c1, c2 = st.columns([2, 1])
    with c1.container(border=True):
        st.markdown("**Screenings in the last 7 days**")
        if len(df):
            days = pd.date_range(end=pd.Timestamp.now().normalize(), periods=7)
            g = recent.groupby(["day", "label"]).size().unstack(fill_value=0).reindex(days, fill_value=0)
            fig = go.Figure()
            for lab, name, col in ((0, "Normal", TEAL), (1, "Flagged", AMBER)):
                fig.add_bar(x=g.index, y=g.get(lab, pd.Series(0, index=g.index)), name=name, marker_color=col)
            fig.update_layout(barmode="stack")
            fig.update_xaxes(tickformat="%a %d")
            st.plotly_chart(plotly_layout(fig, 280), width="stretch")
    with c2.container(border=True):
        st.markdown("**Go to**")
        for key, label in (("analytics", "📊  Usage analytics"), ("models", "🧠  Model management"),
                           ("compare", "🧪  Model comparison")):
            if st.button(label, width="stretch", key=f"goto_{key}"):
                go_page(key)
        st.caption("Includes seeded demo activity for the review.")


# =========================================================================== analytics (US-10)
def analytics():
    st.markdown('<p class="eyebrow">Operations</p>', unsafe_allow_html=True)
    st.title("Usage analytics")
    st.markdown('<p class="muted">Anonymised aggregates only — no names, emails or audio are shown here.</p>',
                unsafe_allow_html=True)
    df = _usage()
    if df.empty:
        st.info("No screenings yet.")
        return
    span = st.segmented_control("Period", ["7 days", "14 days", "30 days", "All"], default="14 days", required=True)
    if span != "All":
        df = df[df["date"] >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=int(span.split()[0]))]
    if df.empty:
        st.info("No screenings in this period.")
        return

    c = st.columns(4)
    c[0].metric("Screenings", len(df), border=True)
    c[1].metric("Flagged", f"{df['label'].mean() * 100:.0f}%", border=True)
    c[2].metric("Median duration", f"{df['duration_sec'].median() / 60:.1f} min", border=True)
    c[3].metric("Mean confidence", f"{df['confidence'].mean() * 100:.0f}%", border=True)

    c1, c2 = st.columns([2, 1])
    with c1.container(border=True):
        st.markdown("**Screenings per day**")
        g = df.groupby(["day", "label"]).size().unstack(fill_value=0)
        fig = go.Figure()
        for lab, name, col in ((0, "Normal", TEAL), (1, "Flagged", AMBER)):
            if lab in g:
                fig.add_bar(x=g.index, y=g[lab], name=name, marker_color=col)
        fig.update_layout(barmode="stack")
        st.plotly_chart(plotly_layout(fig, 280), width="stretch")
    with c2.container(border=True):
        st.markdown("**Result distribution**")
        counts = df["label"].value_counts().reindex([0, 1], fill_value=0)
        fig = go.Figure(go.Pie(labels=["Normal", "Flagged"], values=counts.values, hole=.6,
                               marker=dict(colors=[TEAL, AMBER]), sort=False))
        st.plotly_chart(plotly_layout(fig, 280), width="stretch")

    c1, c2, c3 = st.columns(3)
    with c1.container(border=True):
        st.markdown("**Input source**")
        s = df["source"].value_counts()
        fig = go.Figure(go.Bar(x=s.values, y=[x.capitalize() for x in s.index], orientation="h", marker_color=TEAL))
        st.plotly_chart(plotly_layout(fig, 220), width="stretch")
    with c2.container(border=True):
        st.markdown("**Model confidence**")
        fig = go.Figure(go.Histogram(x=df["confidence"] * 100, nbinsx=12, marker_color="#7C9CFF"))
        fig.update_xaxes(title="%")
        st.plotly_chart(plotly_layout(fig, 220), width="stretch")
    with c3.container(border=True):
        st.markdown("**Pauses per hour by result**")
        fig = go.Figure()
        for lab, name, col in ((0, "Normal", TEAL), (1, "Flagged", AMBER)):
            fig.add_box(y=df[df["label"] == lab]["pauses_per_hour"], name=name, marker_color=col)
        st.plotly_chart(plotly_layout(fig, 220, showlegend=False), width="stretch")
    st.caption(f"Model versions in use: " + ", ".join(f"{k} ({v})" for k, v in df["model_version"].value_counts().items()))


# =========================================================================== model management (US-11)
def model_management():
    st.markdown('<p class="eyebrow">Operations</p>', unsafe_allow_html=True)
    st.title("Model management")
    reg = load_registry()

    st.subheader("Model versions")
    for v in reversed(reg["versions"]):
        active = v["id"] == reg["active"]
        with st.container(border=True):
            c1, c2 = st.columns([3, 2], vertical_alignment="center")
            with c1:
                tag = "🟢 Active" if active else ("🟡 Candidate" if v["status"] == "candidate" else "⚪ Previous")
                st.markdown(f"**{v['id']}** · {tag} · <span class='muted'>{fmt_date(v['created'])}</span>",
                            unsafe_allow_html=True)
                st.markdown(f"<span class='muted'>{v['description']}</span>", unsafe_allow_html=True)
                rows = [{"Model": SHORT[k], "Accuracy": f"{m.get('accuracy', 0) * 100:.1f}%",
                         "F1": f"{m.get('f1', 0):.3f}", "Sensitivity": f"{m.get('recall', 0) * 100:.1f}%",
                         "Specificity": f"{m.get('specificity', 0) * 100:.1f}%", "ROC-AUC": f"{m.get('roc_auc', 0):.3f}"}
                        for k, m in v["test"].items() if m]
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            with c2:
                dm = st.selectbox("Decision model", list(MODEL_LABELS), key=f"dm_{v['id']}",
                                  index=list(MODEL_LABELS).index(v.get("decision_model", "ensemble")),
                                  format_func=lambda k: MODEL_LABELS[k])
                if active:
                    if dm != v.get("decision_model") and st.button("Apply", key=f"apply_{v['id']}", type="primary"):
                        set_active(v["id"], dm)
                        st.toast(f"Decision model set to {SHORT[dm]}")
                        st.rerun()
                    st.caption("Used for all new screenings.")
                else:
                    if st.button("Set active", key=f"act_{v['id']}", type="primary"):
                        set_active(v["id"], dm)
                        st.toast(f"{v['id']} is now active")
                        st.rerun()
                    if v["id"] != "v1.0" and st.button("Delete version", key=f"del_{v['id']}"):
                        delete_version(v["id"])
                        st.rerun()

    st.subheader("Retrain with new data")
    st.markdown('<span class="muted">Add consented, anonymised recordings with a verified label (e.g. confirmed '
                'by a sleep study). Retraining refits the Random Forest and SVM on the original training set plus '
                'these clips and evaluates them on the same held-out test set; the result is saved as a candidate '
                'version for review. (Retraining the CNN needs the training toolkit — see README.)</span>',
                unsafe_allow_html=True)
    with st.form("upload_clips", clear_on_submit=True):
        files = st.file_uploader("Labelled recordings", type=["wav", "flac", "mp3", "ogg"], accept_multiple_files=True)
        label = st.radio("Verified label for these files", [0, 1], horizontal=True,
                         format_func=lambda l: "Normal" if l == 0 else "Irregular / apnea confirmed")
        if st.form_submit_button("Add to training pool") and files:
            UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
            for f in files:
                path = UPLOADS_DIR / f"{datetime.now(timezone.utc):%Y%m%d%H%M%S}_{Path(f.name).name}"
                path.write_bytes(f.getvalue())
                db.add_training_clip(f.name, label, str(path), user()["id"])
            st.toast(f"Added {len(files)} clip(s)")
            st.rerun()
    clips = db.training_clips()
    if clips:
        st.dataframe(pd.DataFrame([{"File": c["filename"], "Label": "Irregular" if c["label"] else "Normal",
                                    "Added": fmt_date(c["created"])} for c in clips]), hide_index=True, width="stretch")
    c1, c2 = st.columns([1, 3])
    if c1.button("Retrain models", type="primary"):
        from ..retrain import retrain
        with st.status("Retraining…", expanded=True) as status:
            log = st.empty()
            lines = []

            def logger(msg):
                lines.append(msg)
                log.code("\n".join(lines[-8:]))

            v = retrain([{"path": c["path"], "label": c["label"], "filename": c["filename"]} for c in clips], log=logger)
            status.update(label=f"Candidate {v['id']} created — review it above and set it active if it's better",
                          state="complete")
    c2.caption(f"{len(clips)} uploaded clip(s) in the pool.")


# =========================================================================== model comparison
def model_comparison():
    st.markdown('<p class="eyebrow">Research</p>', unsafe_allow_html=True)
    st.title("Model comparison")
    st.markdown('<p class="muted">Random Forest and SVM on handcrafted features (MFCC, spectral, pause statistics) '
                'vs. a CNN on mel-spectrograms — evaluated on the same held-out test set.</p>', unsafe_allow_html=True)
    m = load_metrics()
    if not m:
        st.warning("models/metrics.json not found — run `python -m snoresense.train`.")
        return
    t = m["test"]
    ds = m["dataset"]

    rows = [{"Model": MODEL_LABELS[k], "Accuracy": v["accuracy"], "Precision": v["precision"],
             "Sensitivity (recall)": v["recall"], "Specificity": v["specificity"], "F1": v["f1"], "ROC-AUC": v["roc_auc"]}
            for k, v in t.items()]
    best = max(t, key=lambda k: t[k]["f1"])
    st.dataframe(pd.DataFrame(rows).style.format({c: "{:.3f}" for c in rows[0] if c != "Model"})
                 .highlight_max(subset=[c for c in rows[0] if c != "Model"], color="#1F4D45"),
                 hide_index=True, width="stretch")
    st.caption(f"Held-out test set: {ds['test_windows']} windows of {ds['window_sec']:.0f} s built from ESC-50 fold "
               f"{ds['test_folds']} sound sources (training used folds {ds['train_folds']}, {ds['train_windows']} windows). "
               f"Best F1: **{SHORT[best]}**.")

    tab1, tab2, tab3, tab4 = st.tabs(["Noise robustness", "ROC & confusion", "Features", "Training & data"])
    with tab1:
        st.markdown("**How accuracy holds up as recordings get noisier** — consumer phone recordings are the "
                    "project's focus, so each model is tested at fixed signal-to-noise ratios.")
        metric = st.segmented_control("Metric", ["f1", "accuracy", "recall", "specificity"], default="f1",
                                      format_func=lambda x: {"f1": "F1", "accuracy": "Accuracy", "recall": "Sensitivity",
                                                             "specificity": "Specificity"}[x], required=True)
        fig = go.Figure()
        for k, curve in m["robustness"].items():
            snrs = sorted(curve, key=float)
            fig.add_trace(go.Scatter(x=[float(s) for s in snrs], y=[curve[s][metric] for s in snrs], mode="lines+markers",
                                     name=SHORT[k], line=dict(color=MODEL_COLORS[k], width=3 if k == "ensemble" else 2)))
        fig.update_xaxes(title="Signal-to-noise ratio (dB) — lower is noisier")
        lows = [c[s][metric] for c in m["robustness"].values() for s in c]
        fig.update_yaxes(title=metric, range=[max(0.0, min(lows) - 0.05), 1.01])
        st.plotly_chart(plotly_layout(fig, 340), width="stretch")
        st.caption(f"{ds['robustness_windows_per_snr']} test windows per SNR level.")
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            for k, v in t.items():
                fig.add_trace(go.Scatter(x=v["roc"]["fpr"], y=v["roc"]["tpr"], mode="lines",
                                         name=f"{SHORT[k]} (AUC {v['roc_auc']:.3f})", line=dict(color=MODEL_COLORS[k])))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color=MUTED), showlegend=False))
            fig.update_xaxes(title="False positive rate")
            fig.update_yaxes(title="True positive rate")
            st.plotly_chart(plotly_layout(fig, 360), width="stretch")
        with c2:
            k = st.selectbox("Confusion matrix", list(t), format_func=lambda k: MODEL_LABELS[k])
            cm = t[k]["confusion"]
            fig = go.Figure(go.Heatmap(z=cm, x=["Pred. normal", "Pred. irregular"], y=["True normal", "True irregular"],
                                       text=cm, texttemplate="%{text}", colorscale=[[0, "#171E3A"], [1, "#5CD6C0"]],
                                       showscale=False))
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(plotly_layout(fig, 300), width="stretch")
    with tab3:
        imp = m["feature_importance"][:15]
        fig = go.Figure(go.Bar(x=[v for _, v in imp][::-1], y=[n for n, _ in imp][::-1], orientation="h", marker_color=TEAL))
        fig.update_xaxes(title="Random Forest importance (mean decrease in impurity)")
        st.plotly_chart(plotly_layout(fig, 440), width="stretch")
        st.caption("Pause/rhythm features (longest gap, pauses per 30 s, rhythm periodicity) are designed from the "
                   "clinical definition of an apnea event (breathing stops ≥ 10 s); MFCC and spectral features describe "
                   "the sound of each snore.")
    with tab4:
        h = pd.DataFrame(m["cnn_history"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=h["epoch"], y=h["loss"], name="Training loss", line=dict(color=AMBER)))
        fig.add_trace(go.Scatter(x=h["epoch"], y=h["val_acc"], name="Validation accuracy", line=dict(color=TEAL), yaxis="y2"))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0.5, 1.0], title="validation accuracy",
                                      tickformat=".0%", showgrid=False))
        fig.update_layout(yaxis=dict(title="training loss"))
        fig.update_xaxes(title="Epoch")
        st.markdown("**CNN training curve**")
        st.plotly_chart(plotly_layout(fig, 280), width="stretch")
        cv = m["cv_f1"]
        st.markdown(f"**5-fold cross-validation F1 on the training set** — Random Forest "
                    f"{pd.Series(cv['rf']).mean():.3f} ± {pd.Series(cv['rf']).std():.3f}, SVM "
                    f"{pd.Series(cv['svm']).mean():.3f} ± {pd.Series(cv['svm']).std():.3f}")
        st.info(f"**Data source.** {ds['source']}. Real apnea-annotated audio (e.g. PSG-Audio) was not available for "
                "this prototype, so these figures measure how well each approach separates the two breathing "
                "patterns under realistic recording conditions — they are not clinical accuracy. Retraining on "
                "polysomnography-labelled recordings is the next step (see README).", icon="ℹ️")
