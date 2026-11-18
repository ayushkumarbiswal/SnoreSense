"""SnoreSense — audio-based screening for sleep-apnea risk patterns.

Run with:  streamlit run app.py      (or use start.bat / start.sh)
"""
import streamlit as st

from snoresense import db
from snoresense.ui import admin_pages, user_pages
from snoresense.ui.auth_page import login_page
from snoresense.ui.common import ROLE_NAMES, bundle, refresh_user, setup_page, user

st.set_page_config(page_title="SnoreSense", page_icon="assets/icon.svg", layout="wide")
setup_page()


@st.cache_resource
def _init_database():
    if db.init():
        from snoresense.demo import seed_database
        seed_database()
    return True


_init_database()
st.logo("assets/logo.svg", size="large", icon_image="assets/icon.svg")

if not user():
    st.navigation([st.Page(login_page, title="Sign in", url_path="login")], position="hidden").run()
    st.stop()

refresh_user()
u = user()
bundle()  # load the models once per server so the first screening starts immediately

if u["role"] == "admin":
    pages = {
        "dashboard": st.Page(admin_pages.admin_dashboard, title="Dashboard", icon=":material/home:", default=True),
        "analytics": st.Page(admin_pages.analytics, title="Usage analytics", icon=":material/bar_chart:"),
        "models": st.Page(admin_pages.model_management, title="Model management", icon=":material/memory:"),
        "compare": st.Page(admin_pages.model_comparison, title="Model comparison", icon=":material/science:"),
        "settings": st.Page(user_pages.settings_page, title="Settings & data", icon=":material/settings:"),
    }
    sections = {"Operations": ["dashboard", "analytics", "models", "compare"], "Account": ["settings"]}
else:
    pages = {
        "dashboard": st.Page(user_pages.dashboard, title="Dashboard", icon=":material/home:", default=True),
        "screen": st.Page(user_pages.screen, title="Screen my sleep", icon=":material/mic:"),
        "result": st.Page(user_pages.result_page, title="Latest result", icon=":material/description:"),
        "history": st.Page(user_pages.history, title="History", icon=":material/history:"),
        "family": st.Page(user_pages.family, title="Family profiles", icon=":material/group:"),
        "consent": st.Page(user_pages.consent_page, title="Consent & privacy", icon=":material/shield:"),
        "settings": st.Page(user_pages.settings_page, title="Settings & data", icon=":material/settings:"),
    }
    screening = ["dashboard", "screen", "result", "history"] + (["family"] if u["role"] == "caregiver" else [])
    sections = {"Screening": screening, "Account": ["consent", "settings"]}

st.session_state.pages = pages
nav = st.navigation({name: [pages[k] for k in keys] for name, keys in sections.items()})

with st.sidebar:
    st.markdown(f"**{u['name']}**  \n<span class='sidebar-note'>{ROLE_NAMES[u['role']]} · {u['email']}</span>",
                unsafe_allow_html=True)
    if st.button("Sign out", icon=":material/logout:"):
        st.session_state.clear()
        st.rerun()
    st.markdown("<p class='sidebar-note'>⚠ Screening aid only — not a medical diagnosis.</p>", unsafe_allow_html=True)

nav.run()
