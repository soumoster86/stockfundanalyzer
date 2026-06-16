"""
Login Gate (soft access control)
--------------------------------
A lightweight password gate for Streamlit Community Cloud. This keeps casual
visitors out — it is NOT strong security. Credentials are read from Streamlit
secrets (never committed to the repo) and compared as SHA-256 hashes.

How credentials are configured (in the Streamlit Cloud dashboard, or a local
.streamlit/secrets.toml that is gitignored):

    [auth]
    # map of username -> sha256(password) hex digest
    [auth.users]
    soumo = "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8"

Generate a hash locally:
    python -c "import hashlib; print(hashlib.sha256('YOURPASSWORD'.encode()).hexdigest())"

If no secrets are configured, the gate falls back to a single demo account
(demo / demo) so the app still runs — with a visible warning.
"""

import hashlib
import streamlit as st


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _get_users():
    """Return {username: sha256_hex} from secrets, or a demo fallback."""
    try:
        users = dict(st.secrets["auth"]["users"])
        if users:
            return users, False
    except Exception:
        pass
    # fallback demo account (warn the user it's insecure)
    return {"demo": _sha256("demo")}, True


def _check(username: str, password: str) -> bool:
    users, _ = _get_users()
    expected = users.get(username)
    if not expected:
        return False
    return _sha256(password) == expected


FEATURES = [
    ("📊", "Quality Score Engine", "22 metrics, percentile-ranked vs sector peers"),
    ("🚩", "Red-Flag Detection", "Earnings-quality, financial & governance forensics"),
    ("🛡️", "Data-Quality Guards", "Flags distorted or sparse fundamentals"),
    ("⚖️", "Configurable Weights", "Tilt toward value, quality, growth or safety"),
    ("📈", "Quality Trends", "See whether fundamentals are improving or declining"),
    ("🔍", "Compare & Rank", "Side-by-side radar profiles across 2,000+ stocks"),
]


def login_gate():
    """
    Render the login screen and stop the app unless authenticated.
    Returns the logged-in username when authenticated.
    """
    if st.session_state.get("auth_user"):
        return st.session_state["auth_user"]

    users, is_demo = _get_users()

    _LOGO_SVG = (
        "<svg width='52' height='52' viewBox='0 0 120 120' style='vertical-align:middle;margin-right:14px;'>"
        "<rect x='0' y='0' width='120' height='120' rx='26' fill='#0E1A14'/>"
        "<rect x='1' y='1' width='118' height='118' rx='25' fill='none' stroke='#1D9E75' stroke-width='2'/>"
        "<rect x='30' y='74' width='14' height='22' rx='3' fill='#2E6E55'/>"
        "<rect x='53' y='58' width='14' height='38' rx='3' fill='#26B583'/>"
        "<rect x='76' y='40' width='14' height='56' rx='3' fill='#1D9E75'/>"
        "<path d='M30 58 L52 42 L70 30 L94 26' fill='none' stroke='#7CF0C0' stroke-width='4' "
        "stroke-linecap='round' stroke-linejoin='round'/>"
        "<circle cx='94' cy='26' r='5.5' fill='#7CF0C0'/></svg>"
    )
    st.markdown(
        f"<div style='text-align:center;'>"
        f"<h1 style='margin-bottom:0;display:inline-flex;align-items:center;justify-content:center;'>"
        f"{_LOGO_SVG}<span>Fundamental Stock Analyzer</span></h1>"
        f"<p style='color:#888;margin-top:4px;'>"
        f"Fundamental quality scoring, ranking &amp; forensic red flags</p></div>",
        unsafe_allow_html=True,
    )
    st.divider()

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("What's inside")
        for icon, title, desc in FEATURES:
            st.markdown(
                f"<div style='border-left:3px solid #1D9E75;padding:6px 12px;"
                f"margin-bottom:10px;background:#11161c;border-radius:6px;'>"
                f"<b>{icon} {title}</b><br>"
                f"<span style='color:#9aa;font-size:0.9em;'>{desc}</span></div>",
                unsafe_allow_html=True,
            )

    with right:
        st.subheader("🔒 Secure Login")
        if is_demo:
            st.warning("Demo mode — no credentials configured. "
                       "Use **demo / demo**. Set `[auth.users]` in Streamlit "
                       "secrets to lock this down.")
        username = st.text_input("Username", placeholder="your username")
        password = st.text_input("Password", type="password", placeholder="your password")
        if st.button("Access Dashboard  ➜", use_container_width=True, type="primary"):
            if _check(username.strip(), password):
                st.session_state["auth_user"] = username.strip()
                st.rerun()
            else:
                st.error("Incorrect username or password.")
        st.markdown(
            "<p style='color:#888;font-size:0.85em;margin-top:8px;'>"
            "Access is restricted. Email "
            "<a href='mailto:soumoster@gmail.com'>soumoster@gmail.com</a> "
            "to request credentials.</p>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown(
        "<p style='text-align:center;color:#888;font-size:0.85em;'>"
        "⚠️ Educational purposes only — not financial advice"
        "&nbsp;&nbsp;·&nbsp;&nbsp; © 2026 Soumoster Analytics</p>",
        unsafe_allow_html=True,
    )
    st.stop()


def logout_button():
    """Render a sidebar logout control when authenticated."""
    if st.session_state.get("auth_user"):
        with st.sidebar:
            st.caption(f"Signed in as **{st.session_state['auth_user']}**")
            if st.button("Log out"):
                del st.session_state["auth_user"]
                st.rerun()
