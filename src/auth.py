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


def login_gate():
    """
    Render the login screen and stop the app unless authenticated.
    Returns the logged-in username when authenticated.
    """
    import streamlit as st

    from ui.landing import render_feature_sections, render_hero

    if st.session_state.get("auth_user"):
        return st.session_state["auth_user"]

    users, is_demo = _get_users()

    # st.title + caption avoid HTML clipping under Streamlit's top chrome
    render_hero(
        st,
        subtitle=(
            "Quality scoring · forensic red flags · institutional scores · "
            "sector ranking · watchlists · optional ML"
        ),
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

    # Login + What's new (collapsible features — no full Tutorial on landing)
    left, right = st.columns([1.15, 0.85], gap="large")
    with right:
        st.markdown("#### 🔒 Secure login")
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

        st.caption(
            "Access is restricted. Email "
            "[soumoster@gmail.com](mailto:soumoster@gmail.com) to request credentials."
        )

    with left:
        st.markdown("#### ✨ What's new")
        st.caption(
            "Key capabilities in this release. Expand a section to explore. "
            "After login, open **Tutorial** in the nav for a full how-to guide."
        )
        render_feature_sections(st, compact=True, expand_first=False)

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
