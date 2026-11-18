"""Sign-in / register page."""
from __future__ import annotations

import streamlit as st

from .. import db
from .common import ROLE_NAMES

DEMO_ACCOUNTS = [
    ("home", "alex@snoresense.demo", "Continue as Home user", "Alex — screens their own sleep"),
    ("caregiver", "priya@snoresense.demo", "Continue as Caregiver", "Priya — screens family members"),
    ("admin", "admin@snoresense.demo", "Continue as Administrator", "Operations, analytics and models"),
]
DEMO_PASSWORD = "demo123"


def _login(u: dict):
    st.session_state.user = u
    st.session_state.pop("last_result", None)
    st.toast(f"Welcome back, {u['name'].split()[0]}", icon="✅")
    st.rerun()


def login_page():
    _, mid, _ = st.columns([1, 1.25, 1])
    with mid:
        st.image("assets/logo.svg", width=230)
        st.markdown('<p class="muted">Screen sleep audio and track results over time.</p>', unsafe_allow_html=True)
        with st.container(border=True):
            tab_in, tab_up = st.tabs(["Sign in", "Get access"])
            with tab_in:
                with st.form("login"):
                    email = st.text_input("Email", placeholder="you@example.com")
                    pw = st.text_input("Password", type="password")
                    if st.form_submit_button("Sign in", type="primary", width="stretch"):
                        u = db.authenticate(email, pw)
                        if u:
                            _login(u)
                        else:
                            st.error("Email or password is incorrect.")
                st.caption("or continue with a demo account")
                for role, email, label, help_ in DEMO_ACCOUNTS:
                    if st.button(label, key=f"demo_{role}", width="stretch", help=help_):
                        u = db.authenticate(email, DEMO_PASSWORD)
                        if u is None:
                            st.error("Demo account missing — restart the app to re-create demo data.")
                        else:
                            _login(u)
            with tab_up:
                with st.form("register"):
                    name = st.text_input("Full name")
                    email = st.text_input("Email ")
                    role = st.radio("I am a…", ["home", "caregiver"], format_func=lambda r: ROLE_NAMES[r],
                                    horizontal=True)
                    pw = st.text_input("Password (min. 6 characters)", type="password")
                    if st.form_submit_button("Create account", type="primary", width="stretch"):
                        if not name.strip() or "@" not in email or len(pw) < 6:
                            st.error("Please enter your name, a valid email and a password of at least 6 characters.")
                        elif db.get_user_by_email(email):
                            st.error("An account with this email already exists.")
                        else:
                            uid = db.create_user(email, name, role, pw)
                            _login(db.get_user(uid))
        st.markdown('<p class="sidebar-note" style="text-align:center">⚠ Screening aid only — not a medical '
                    'diagnosis.</p>', unsafe_allow_html=True)
