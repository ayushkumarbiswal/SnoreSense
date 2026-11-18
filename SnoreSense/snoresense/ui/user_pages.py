"""Pages for home users and caregivers (Module 3 — K. Dinesh Kumar Patro, calling Modules 1 & 2)."""
from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import db
from ..analysis import EXPLANATIONS, NEXT_STEPS, analyze
from ..audio import load_audio, to_wav_bytes
from ..config import APNEA_PAUSE_SEC, SAMPLES_DIR
from ..models import MODEL_LABELS
from .common import (AMBER, MUTED, RED, TEAL, badge, bundle, fmt_date, fmt_duration, nav_to as go_page, plotly_layout,
                     user, waveform_figure, window_figure)

MIME = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac", "ogg": "audio/ogg"}


def _profiles():
    return db.profiles(user()["id"])


def _greeting() -> str:
    h = datetime.now().hour
    return "Good morning" if h < 12 else "Good afternoon" if h < 17 else "Good evening"


def _samples() -> list[dict]:
    p = SAMPLES_DIR / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else []


# =========================================================================== dashboard
def dashboard():
    u = user()
    rows = db.screenings(u["id"])
    st.markdown('<p class="eyebrow">Tonight\'s check</p>', unsafe_allow_html=True)
    st.title(f"{_greeting()}, {u['name'].split()[0]}")
    st.markdown('<p class="muted">Run a quick screening on last night\'s sleep audio, or catch up on your history.</p>',
                unsafe_allow_html=True)

    with st.container(key="hero"):
        st.markdown('<h2 style="margin:0">Screen your sleep audio</h2><p class="muted">Record live or upload a clip, '
                    'and SnoreSense will flag any acoustic patterns worth a closer look.</p>', unsafe_allow_html=True)
        if st.button("Start a screening", type="primary", icon=":material/mic:"):
            go_page("screen")

    with st.container(border=True):
        if rows:
            r = rows[0]
            c1, c2 = st.columns([4, 1], vertical_alignment="center")
            c1.markdown(f'<span class="muted">Most recent result · {fmt_date(r["created"], False)} · '
                        f'{r["profile_name"]}</span><br>{badge(r["label"])} &nbsp; '
                        f'<span class="muted">{r["confidence"] * 100:.0f}% confidence</span>', unsafe_allow_html=True)
            if c2.button("View details", key="dash_view"):
                st.session_state.view_screening_id = r["id"]
                go_page("result")
        else:
            st.markdown('<span class="muted">No screenings yet — your first result will appear here.</span>',
                        unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1.container(border=True, height=170):
        st.markdown("**Your history**")
        st.markdown(f'<span class="muted">{len(rows)} screening{"s" if len(rows) != 1 else ""} recorded</span>',
                    unsafe_allow_html=True)
        if st.button("🕘  View history"):
            go_page("history")
    with c2.container(border=True, height=170):
        st.markdown("**Privacy controls**")
        st.markdown('<span class="muted">Review consent and manage stored recordings any time.</span>',
                    unsafe_allow_html=True)
        if st.button("🛡  Open consent settings"):
            go_page("consent")

    if u["role"] == "caregiver":
        st.subheader("Family overview")
        profs = _profiles()
        cols = st.columns(max(1, min(3, len(profs))))
        for i, p in enumerate(profs):
            pr = [r for r in rows if r["profile_id"] == p["id"]]
            with cols[i % len(cols)].container(border=True):
                st.markdown(f"**{p['name']}** <span class='muted'>· {p['relation']}</span>", unsafe_allow_html=True)
                if pr:
                    flagged = sum(r["label"] for r in pr)
                    st.markdown(f"{badge(pr[0]['label'])} <span class='muted'>latest, {fmt_date(pr[0]['created'], False)}"
                                f"</span><br><span class='muted'>{flagged} of {len(pr)} screenings flagged</span>",
                                unsafe_allow_html=True)
                else:
                    st.markdown("<span class='muted'>No screenings yet</span>", unsafe_allow_html=True)


# =========================================================================== consent gate (US-13)
def consent_form(key: str, intro: bool = True) -> bool:
    u = user()
    c = db.get_consent(u["id"])
    if intro:
        st.info("Before any recording is analysed or stored, we need your explicit consent. You can change "
                "these choices or delete your data at any time.", icon="🛡")
    with st.form(key):
        a = st.checkbox("I consent to SnoreSense analysing the sleep recordings I record or upload "
                        "(required for screening).", value=bool(c["analyse"]))
        h = st.checkbox("Save my screening results to my history so I can track changes over time.",
                        value=bool(c["store_history"]))
        au = st.checkbox("Also keep the audio recording with each saved result (optional — by default only "
                         "the result and summary numbers are kept, never the audio).", value=bool(c["store_audio"]))
        if u["role"] == "caregiver":
            st.caption("When recording a family member, make sure they know and agree to being recorded.")
        if st.form_submit_button("Save my choices", type="primary"):
            db.set_consent(u["id"], a, h, au)
            st.toast("Consent preferences saved", icon="🛡")
            st.rerun()
    return bool(c["analyse"])


# =========================================================================== screening (US-01, 02, 06, 08, 09)
def screen():
    u = user()
    st.markdown('<p class="eyebrow">New screening</p>', unsafe_allow_html=True)
    st.title("Screen my sleep")
    consent = db.get_consent(u["id"])
    if not consent["analyse"]:
        consent_form("consent_gate")
        return

    profs = _profiles()
    who = st.selectbox("Whose sleep is this?", profs, format_func=lambda p: f"{p['name']} ({p['relation']})",
                       disabled=len(profs) == 1)
    if u["role"] == "caregiver" and len(profs) == 1:
        st.caption("Add family members under **Family profiles** to keep their results separate.")

    mode = st.segmented_control("Audio source", ["🎙 Record", "📁 Upload a file", "🎧 Sample recordings"],
                                default="🎙 Record", label_visibility="collapsed", required=True)
    audio_bytes, filename, source, mime = None, None, None, "audio/wav"
    if mode == "🎙 Record":
        st.caption("Place the phone or laptop about 1 metre from the sleeper's head. Record at least 30 seconds — "
                   "a few minutes is better. Your browser will ask for microphone permission.")
        rec = st.audio_input("Record breathing / snoring sounds", sample_rate=16000)
        if rec is not None:
            audio_bytes, filename, source = rec.getvalue(), "live-recording.wav", "record"
    elif mode == "📁 Upload a file":
        up = st.file_uploader("Upload an overnight or sample recording", type=["wav", "mp3", "flac", "ogg"])
        st.caption("Your actual file's waveform will be analysed. Long recordings (up to ~1 hour) are "
                   "processed in 30-second windows.")
        if up is not None:
            audio_bytes, filename, source = up.getvalue(), up.name, "upload"
            mime = MIME.get(up.name.rsplit(".", 1)[-1].lower(), "audio/wav")
    elif mode == "🎧 Sample recordings":
        samples = _samples()
        if not samples:
            st.warning("No sample recordings found in the samples/ folder.")
        else:
            st.caption("Recordings prepared for the demo from held-out test audio (never seen in training).")
            pick = st.radio("Choose a sample", samples, format_func=lambda s: s["title"])
            st.markdown(f'<span class="muted">{pick["description"]}</span>', unsafe_allow_html=True)
            audio_bytes = (SAMPLES_DIR / pick["file"]).read_bytes()
            filename, source, mime = pick["file"], "sample", MIME["flac"]

    if audio_bytes is None:
        st.markdown('<p class="muted">If a recording is too quiet or too noisy to analyse, we\'ll ask you to try '
                    'again rather than guess.</p>', unsafe_allow_html=True)
        return

    st.markdown("**Review before analysing**")
    st.audio(audio_bytes, format=mime)
    if not st.button("Analyse this recording", type="primary"):
        return

    try:
        y = load_audio(audio_bytes)
    except Exception:
        st.error("We couldn't read this file. Please upload a WAV, MP3, FLAC or OGG audio file.")
        return
    if len(y) / 16000 > 3600 * 1.05:
        st.error("Recordings longer than one hour aren't supported in this prototype — please upload a shorter clip.")
        return

    with st.status("Analysing your recording…", expanded=True) as status:
        bar = st.progress(0.0)
        msgs = st.empty()

        def progress(text, frac):
            bar.progress(min(frac, 1.0))
            msgs.markdown(f"<span class='muted'>{text}</span>", unsafe_allow_html=True)

        result = analyze(y, bundle(), progress)
        if result["status"] != "ok":
            status.update(label="We couldn't analyse this recording", state="error")
        else:
            status.update(label="Analysis complete", state="complete")

    if result["status"] != "ok":
        st.error("**Please record again.**\n\n" + "\n".join(f"- {i}" for i in result["issues"]))
        q = result["quality"]
        st.caption(f"Duration {q['duration_sec']:.0f} s · peak level {q['peak_dbfs']:.0f} dBFS · "
                   f"estimated SNR {q['snr_db']:.0f} dB · clipping {q['clipped_pct']:.1f}%")
        return

    saved_id = None
    if consent["store_history"]:
        keep_audio = to_wav_bytes(y) if consent["store_audio"] else None
        saved_id = db.save_screening(u["id"], who["id"], source, filename, result, keep_audio)
    st.session_state.last_result = {"result": result, "audio": audio_bytes, "mime": mime, "filename": filename,
                                    "profile_name": who["name"], "created": db.now(), "saved_id": saved_id}
    st.session_state.pop("view_screening_id", None)
    go_page("result")


# =========================================================================== result (US-03, 04, 12)
def _result_from_row(row: dict) -> dict:
    d = json.loads(row["details"])
    return {**d, "status": "ok", "label": row["label"], "confidence": row["confidence"],
            "model_version": row["model_version"], "decision_model": row["decision_model"]}


def result_page():
    u = user()
    st.markdown('<p class="eyebrow">Screening result</p>', unsafe_allow_html=True)
    st.title("Latest result")
    data = None
    sid = st.session_state.get("view_screening_id")
    if sid:
        row = db.get_screening(u["id"], sid)
        if row:
            data = {"result": _result_from_row(row), "audio": row["audio"], "mime": "audio/wav",
                    "filename": row["filename"], "profile_name": row["profile_name"], "created": row["created"],
                    "saved_id": row["id"]}
    if data is None and "last_result" in st.session_state:
        data = st.session_state.last_result
    if data is None:
        rows = db.screenings(u["id"])
        if rows:
            row = db.get_screening(u["id"], rows[0]["id"])
            data = {"result": _result_from_row(row), "audio": row["audio"], "mime": "audio/wav",
                    "filename": row["filename"], "profile_name": row["profile_name"], "created": row["created"],
                    "saved_id": row["id"]}
    if data is None:
        st.info("No result to show yet — start a screening to see it here.")
        if st.button("Screen my sleep", type="primary"):
            go_page("screen")
        return
    render_result(data)


def render_result(data: dict):
    u = user()
    r = data["result"]
    s = r["summary"]
    label = r["label"]

    with st.container(border=True):
        c1, c2 = st.columns([3, 2], vertical_alignment="center")
        with c1:
            st.markdown(f'<span class="muted">{data["profile_name"]} · {fmt_date(data["created"])} · '
                        f'{data.get("filename") or ""}</span>', unsafe_allow_html=True)
            color = AMBER if label else TEAL
            text = "Irregular pattern — consult a physician" if label else "Normal snoring pattern"
            st.markdown(f'<div class="result-label" style="color:{color}">{text}</div>', unsafe_allow_html=True)
            st.markdown(f'{badge(label)} &nbsp; <span class="muted">{r["confidence"] * 100:.0f}% confidence · '
                        f'model {r["model_version"]} ({MODEL_LABELS[r["decision_model"]].split(" (")[0]})</span>',
                        unsafe_allow_html=True)
        with c2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=s["flagged_fraction"] * 100, number={"suffix": "%", "font": {"size": 30}},
                title={"text": "of the recording flagged", "font": {"size": 13}},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": AMBER if label else TEAL},
                       "bgcolor": "#1B2347", "threshold": {"line": {"color": MUTED, "width": 2}, "value": 25}}))
            fig = plotly_layout(fig, 190)
            fig.update_layout(margin=dict(l=25, r=25, t=45, b=5))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.markdown(f'<div class="disclaimer"><b>What this means.</b> {EXPLANATIONS[label]}</div>', unsafe_allow_html=True)
    st.write("")

    m = st.columns(5)
    m[0].metric("Duration", f"{s['duration_sec'] / 60:.1f} min", border=True, help="Analysed length after trimming silence at the edges")
    m[1].metric("Sounds / min", f"{s['events_per_min']:.0f}", border=True, help="Breathing and snoring sounds detected per minute")
    m[2].metric(f"Pauses ≥ {APNEA_PAUSE_SEC:.0f} s", s["n_pauses"], border=True)
    m[3].metric("Longest pause", f"{s['longest_pause_sec']:.0f} s" if s["n_pauses"] else "—", border=True)
    m[4].metric("Pauses per hour", f"{s['pauses_per_hour']:.0f}", border=True,
                help="Pauses ≥ 10 s scaled to one hour of sleep. Screening metric from audio only — not the clinical "
                     "apnea-hypopnea index (AHI), and only meaningful for longer (30 min +) recordings.")

    st.subheader("Waveform with flagged segments")
    st.plotly_chart(waveform_figure(r), width="stretch")
    if data.get("audio"):
        st.audio(data["audio"], format=data.get("mime", "audio/wav"))

    with st.expander("How each model scored this recording", expanded=False):
        st.markdown("Each 30-second window (50% overlap) is scored by three models. The decision uses "
                    f"**{MODEL_LABELS[r['decision_model']]}**; a recording is flagged when at least 25% of its "
                    "windows are above the threshold.")
        st.plotly_chart(window_figure(r), width="stretch")
        mp = s["mean_probs"]
        st.dataframe(pd.DataFrame([{"Model": MODEL_LABELS[k], "Mean P(irregular)": f"{mp[k] * 100:.0f}%",
                                    "Windows flagged": f"{sum(w[k] >= 0.5 for w in r['windows'])} / {len(r['windows'])}"}
                                   for k in ("rf", "svm", "cnn", "ensemble")]), hide_index=True, width="stretch")

    with st.expander("What should I do next?", expanded=bool(label)):
        for step in NEXT_STEPS[label]:
            st.markdown(f"- {step}")
        st.caption("SnoreSense is a first-level screening aid. It cannot diagnose sleep apnea or any other condition.")

    c1, c2, _ = st.columns([1.2, 1.4, 3])
    if data.get("saved_id"):
        c1.success("Saved to history", icon="✅")
        if c2.button("🗑 Delete this recording"):
            _confirm_delete_screening(data["saved_id"])
    else:
        consent = db.get_consent(u["id"])
        if consent["store_history"]:
            c1.info("Not saved")
        else:
            c1.info("Not saved — history is off", icon="🛡")
            st.caption("Turn on history under Consent & privacy to keep results. This result disappears when you sign out.")


@st.dialog("Delete this recording?")
def _confirm_delete_screening(sid: int):
    st.write("The audio and its result will be permanently removed from your history.")
    c1, c2 = st.columns(2)
    if c1.button("Delete", type="primary", width="stretch"):
        db.delete_screening(user()["id"], sid)
        st.session_state.pop("view_screening_id", None)
        if st.session_state.get("last_result", {}).get("saved_id") == sid:
            st.session_state.pop("last_result")
        st.rerun()
    if c2.button("Cancel", width="stretch"):
        st.rerun()


# =========================================================================== history (US-05, 07)
def history():
    u = user()
    st.markdown('<p class="eyebrow">Your history</p>', unsafe_allow_html=True)
    st.title("History")
    profs = _profiles()
    options = [None] + profs
    who = st.selectbox("Profile", options, format_func=lambda p: "All profiles" if p is None else f"{p['name']} ({p['relation']})",
                       disabled=len(profs) == 1) if len(profs) > 1 else None
    rows = db.screenings(u["id"], who["id"] if who else None)
    if not rows:
        st.info("No saved screenings yet.")
        return

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["created"], utc=True).dt.tz_convert(None)
    c = st.columns(4)
    c[0].metric("Screenings", len(df), border=True)
    c[1].metric("Flagged", int(df["label"].sum()), border=True)
    c[2].metric("Avg pauses / hour", f"{df['pauses_per_hour'].mean():.0f}", border=True)
    c[3].metric("Last screening", fmt_date(rows[0]["created"], False), border=True)

    fig = go.Figure()
    for prof, g in df.sort_values("date").groupby("profile_name"):
        fig.add_trace(go.Scatter(x=g["date"], y=g["pauses_per_hour"], mode="lines+markers", name=prof,
                                 marker=dict(size=11, color=[AMBER if l else TEAL for l in g["label"]],
                                             line=dict(width=1, color="#0D1226")),
                                 hovertemplate="%{x|%d %b}<br>%{y:.0f} pauses/hour<extra>" + prof + "</extra>"))
    fig.update_yaxes(title="Pauses ≥ 10 s per hour")
    st.markdown("**Trend** <span class='muted'>— amber points were flagged as irregular</span>", unsafe_allow_html=True)
    st.plotly_chart(plotly_layout(fig, 260), width="stretch")

    table = pd.DataFrame({
        "Date": [fmt_date(r["created"]) for r in rows],
        "Profile": df["profile_name"],
        "Result": ["⚠ Irregular" if l else "✓ Normal" for l in df["label"]],
        "Confidence": [f"{c * 100:.0f}%" for c in df["confidence"]],
        "Pauses/h": df["pauses_per_hour"].round(0).astype(int),
        "Duration": [fmt_duration(d) for d in df["duration_sec"]],
        "Source": df["source"].str.capitalize(),
    })
    st.caption("Select a row to view or delete it.")
    ev = st.dataframe(table, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row")
    sel = ev.selection.rows
    if sel:
        r = rows[sel[0]]
        c1, c2, _ = st.columns([1, 1, 4])
        if c1.button("View details", type="primary"):
            st.session_state.view_screening_id = r["id"]
            go_page("result")
        if c2.button("Delete"):
            _confirm_delete_screening(r["id"])


# =========================================================================== family (US-06, 07)
def family():
    u = user()
    st.markdown('<p class="eyebrow">Caregiver</p>', unsafe_allow_html=True)
    st.title("Family profiles")
    st.markdown('<p class="muted">Keep each family member\'s screenings separate so records never get mixed up.</p>',
                unsafe_allow_html=True)
    for p in _profiles():
        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 1, 1], vertical_alignment="center")
            c1.markdown(f"**{p['name']}** <span class='muted'>· {p['relation']} · {p['n_screenings']} screenings</span>",
                        unsafe_allow_html=True)
            if c2.button("Screen now", key=f"scr{p['id']}"):
                go_page("screen")
            if p["relation"] != "Self" and c3.button("Remove", key=f"rm{p['id']}"):
                _confirm_remove_profile(p)
    with st.form("add_profile", clear_on_submit=True):
        st.markdown("**Add family member**")
        c1, c2 = st.columns(2)
        name = c1.text_input("Name")
        rel = c2.selectbox("Relation", ["Parent", "Spouse / partner", "Grandparent", "Sibling", "Other"])
        if st.form_submit_button("Add family member", type="primary"):
            if name.strip():
                db.add_profile(u["id"], name, rel)
                st.toast(f"Added {name}", icon="👪")
                st.rerun()
            else:
                st.error("Please enter a name.")


@st.dialog("Remove family member?")
def _confirm_remove_profile(p):
    st.write(f"This removes **{p['name']}** and all of their {p['n_screenings']} saved screenings permanently.")
    c1, c2 = st.columns(2)
    if c1.button("Remove", type="primary", width="stretch"):
        db.delete_profile(user()["id"], p["id"])
        st.rerun()
    if c2.button("Cancel", width="stretch"):
        st.rerun()


# =========================================================================== consent & privacy (US-13)
def consent_page():
    u = user()
    st.markdown('<p class="eyebrow">Account</p>', unsafe_allow_html=True)
    st.title("Consent & privacy")
    consent_form("consent_page", intro=False)
    with st.container(border=True):
        st.markdown("**How your data is handled**")
        st.markdown(
            "- Audio is processed in memory to extract features and **not stored** unless you opt in above.\n"
            "- Saved results are linked only to your own profiles, are never shared, and are excluded from "
            "analytics except as anonymous counts.\n"
            "- You can delete individual recordings (History) or everything at once (below).")
    with st.container(border=True):
        st.markdown("**Delete all my data**")
        st.markdown('<span class="muted">Permanently removes every recording, result and history entry associated '
                    'with your profile.</span>', unsafe_allow_html=True)
        if st.button("Delete everything", type="primary"):
            _confirm_delete_all()


@st.dialog("Delete all your data?")
def _confirm_delete_all():
    st.write("This permanently removes every recording, result, and history entry tied to your profile. "
             "This action can't be undone.")
    c1, c2 = st.columns(2)
    if c1.button("Yes, delete everything", type="primary", width="stretch"):
        db.delete_all_data(user()["id"])
        st.session_state.pop("last_result", None)
        st.session_state.pop("view_screening_id", None)
        st.rerun()
    if c2.button("Cancel", width="stretch"):
        st.rerun()


# =========================================================================== settings & data
def settings_page():
    u = user()
    st.markdown('<p class="eyebrow">Account</p>', unsafe_allow_html=True)
    st.title("Settings & data")
    from .common import ROLE_NAMES
    with st.container(border=True):
        st.markdown("**Profile**")
        st.markdown(f"{u['name']} · {u['email']} · {ROLE_NAMES[u['role']]}")
    if u["role"] != "admin":
        with st.container(border=True):
            st.markdown("**Download my data**")
            st.markdown('<span class="muted">Export your screening history and profile as a JSON file.</span>',
                        unsafe_allow_html=True)
            st.download_button("Download", json.dumps(db.export_user(u["id"]), indent=1, default=str),
                               file_name="snoresense-export.json", mime="application/json")
    with st.container(border=True):
        st.markdown("**Delete account & data**")
        st.markdown('<span class="muted">Removes your profile and every recording permanently. '
                    'This can\'t be undone.</span>', unsafe_allow_html=True)
        if u.get("demo"):
            st.caption("Demo accounts can't be deleted (they're needed for the demo). Use *Delete all my data* "
                       "under Consent & privacy instead.")
        elif st.button("Delete everything", key="del_acct", type="primary"):
            _confirm_delete_account()


@st.dialog("Delete your account?")
def _confirm_delete_account():
    st.write("Your account, profiles and every recording will be permanently removed.")
    c1, c2 = st.columns(2)
    if c1.button("Delete account", type="primary", width="stretch"):
        db.delete_user(user()["id"])
        st.session_state.clear()
        st.rerun()
    if c2.button("Cancel", width="stretch"):
        st.rerun()
