"""
Login Gate (soft access control)
--------------------------------
A lightweight password gate for Streamlit Community Cloud. This keeps casual
visitors out — it is NOT strong security (fine for public-fundamentals data,
not for anything sensitive).

Credentials are read from Streamlit secrets (never committed) and compared as
PBKDF2-HMAC-SHA256 hashes with a per-password salt (constant-time check).

Configure in Streamlit Cloud → Settings → Secrets, or a local
`.streamlit/secrets.toml` (gitignored):

    [auth.users]
    soumo = "<salt$hash from the generator below>"

Generate a hash:

    python -m src.auth YOURPASSWORD

Fail-closed: if no users are configured, the app shows setup instructions and
refuses to load. For local exploration without secrets, set env:

    STOCKFUN_DEMO=1
    streamlit run app.py

That enables a single demo/demo account with a visible warning.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets as pysecrets
import time

PBKDF2_ITERATIONS = 200_000


def hash_password(password: str, salt: str | None = None) -> str:
    """Return 'salt$hash' using PBKDF2-HMAC-SHA256."""
    if salt is None:
        salt = pysecrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), PBKDF2_ITERATIONS
    ).hex()
    return f"{salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    """Constant-time check of a password against a 'salt$hash' string."""
    try:
        salt, _ = stored.split("$", 1)
    except (ValueError, AttributeError):
        return False
    return hmac.compare_digest(hash_password(password, salt), stored)


def _get_users() -> tuple[dict[str, str], bool]:
    """
    Return ({username: salt$hash}, is_demo).

    is_demo is True only when STOCKFUN_DEMO=1 and no secrets users exist.
    """
    try:
        import streamlit as st
        users = dict(st.secrets["auth"]["users"])
        if users:
            return users, False
    except Exception:
        pass

    if os.environ.get("STOCKFUN_DEMO", "").strip() in ("1", "true", "True", "yes"):
        return {"demo": hash_password("demo", salt="0" * 32)}, True

    return {}, False


def check_credentials(username: str, password: str, users: dict | None = None) -> bool:
    """Verify credentials. Burns a hash when the user is unknown (anti-probe)."""
    if users is None:
        users, _ = _get_users()
    stored = users.get(username)
    if stored is None:
        hash_password(password, "0" * 32)
        return False
    return verify_password(password, stored)


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
    import streamlit as st

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

    # Fail-closed setup gate
    if not users:
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            st.error("No users configured — the app is locked by default.")
            st.markdown(
                "**Setup (one time):**\n"
                "1. Generate a password hash:\n"
                "```bash\npython -m src.auth yourpassword\n```\n"
                "2. Create `.streamlit/secrets.toml` next to `app.py` and paste:\n"
                "```toml\n[auth.users]\nadmin = \"<the salt$hash it printed>\"\n```\n"
                "3. Restart the app. Add one line per user for more accounts.\n\n"
                "On Streamlit Cloud, paste the same TOML into **Settings → Secrets**.\n\n"
                "For local exploration without secrets: "
                "`STOCKFUN_DEMO=1 streamlit run app.py` (demo / demo)."
            )
        st.markdown(
            "<p style='text-align:center;color:#888;font-size:0.85em;'>"
            "⚠️ Educational purposes only — not financial advice"
            "&nbsp;&nbsp;·&nbsp;&nbsp; © 2026 Soumoster Analytics</p>",
            unsafe_allow_html=True,
        )
        st.stop()

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
            st.warning(
                "Demo mode (`STOCKFUN_DEMO=1`) — use **demo / demo**. "
                "Configure `[auth.users]` in Streamlit secrets before any public deploy."
            )
        with st.form("login"):
            username = st.text_input("Username", placeholder="your username")
            password = st.text_input(
                "Password", type="password", placeholder="your password"
            )
            submitted = st.form_submit_button(
                "Access Dashboard  ➜", use_container_width=True, type="primary"
            )

        if submitted:
            attempts = st.session_state.get("auth_attempts", 0)
            if attempts >= 3:
                time.sleep(min(attempts, 8))
            if check_credentials(username.strip(), password, users):
                st.session_state["auth_user"] = username.strip()
                st.session_state["auth_attempts"] = 0
                st.rerun()
            else:
                st.session_state["auth_attempts"] = attempts + 1
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
    import streamlit as st

    if st.session_state.get("auth_user"):
        with st.sidebar:
            st.caption(f"Signed in as **{st.session_state['auth_user']}**")
            if st.button("Log out"):
                st.session_state.pop("auth_user", None)
                st.rerun()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m src.auth <password>")
        sys.exit(1)
    print("\nAdd this to .streamlit/secrets.toml :\n")
    print("[auth.users]")
    print(f'admin = "{hash_password(sys.argv[1])}"')
    print("\n(change 'admin' to any username; one line per user)")
