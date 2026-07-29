"""Watchlist dual-backend (session + mocked Supabase)."""
from src.watchlist import (
    SESSION_KEY,
    HYDRATED_KEY,
    add_ticker,
    get_watchlist,
    remove_ticker,
    set_watchlist,
    ensure_hydrated,
    backend_name,
)
from src.watchlist_supabase import normalize_supabase_url


def test_normalize_supabase_url():
    assert normalize_supabase_url("https://abc.supabase.co") == "https://abc.supabase.co"
    assert normalize_supabase_url("https://abc.supabase.co/") == "https://abc.supabase.co"
    assert normalize_supabase_url("https://abc.supabase.co/rest/v1") == "https://abc.supabase.co"
    assert normalize_supabase_url("https://abc.supabase.co/rest/v1/") == "https://abc.supabase.co"
    assert normalize_supabase_url('  "https://abc.supabase.co/"  ') == "https://abc.supabase.co"


class _SS(dict):
    pass


def test_session_backend_without_supabase(monkeypatch):
    monkeypatch.setattr("src.watchlist._use_supabase", lambda: False)
    ss = _SS()
    assert add_ticker(ss, "AAA.NS")
    assert get_watchlist(ss) == ["AAA.NS"]
    assert backend_name(ss) == "session"
    assert remove_ticker(ss, "AAA.NS")
    assert get_watchlist(ss) == []


def test_hydrate_from_supabase(monkeypatch):
    monkeypatch.setattr("src.watchlist._use_supabase", lambda: True)

    def fake_fetch(username, client=None):
        assert username == "alice"
        return ["TCS.NS", "INFY.NS"]

    monkeypatch.setattr("src.watchlist_supabase.fetch_tickers", fake_fetch)
    ss = _SS()
    ss["auth_user"] = "alice"
    ensure_hydrated(ss)
    assert get_watchlist(ss) == ["TCS.NS", "INFY.NS"]
    assert ss.get(HYDRATED_KEY) is True
    assert backend_name(ss) == "supabase"


def test_add_writes_through(monkeypatch):
    monkeypatch.setattr("src.watchlist._use_supabase", lambda: True)
    calls = []

    monkeypatch.setattr(
        "src.watchlist_supabase.fetch_tickers",
        lambda username, client=None: [],
    )

    def fake_insert(username, ticker, client=None):
        calls.append((username, ticker))

    monkeypatch.setattr("src.watchlist_supabase.insert_ticker", fake_insert)

    ss = _SS()
    ss["auth_user"] = "bob"
    assert add_ticker(ss, "RELIANCE.NS")
    assert calls == [("bob", "RELIANCE.NS")]
    assert get_watchlist(ss) == ["RELIANCE.NS"]


def test_set_watchlist_replace_all(monkeypatch):
    monkeypatch.setattr("src.watchlist._use_supabase", lambda: True)
    replaced = []

    monkeypatch.setattr(
        "src.watchlist_supabase.fetch_tickers",
        lambda username, client=None: ["OLD.NS"],
    )

    def fake_replace(username, tickers, client=None):
        replaced.append((username, list(tickers)))

    monkeypatch.setattr("src.watchlist_supabase.replace_all", fake_replace)

    ss = _SS()
    ss["auth_user"] = "carol"
    ensure_hydrated(ss)
    set_watchlist(ss, ["A.NS", "B.NS"])
    assert replaced and replaced[-1] == ("carol", ["A.NS", "B.NS"])
    assert get_watchlist(ss) == ["A.NS", "B.NS"]


def test_supabase_failure_falls_back(monkeypatch):
    monkeypatch.setattr("src.watchlist._use_supabase", lambda: True)

    def boom(username, client=None):
        raise RuntimeError("network down")

    monkeypatch.setattr("src.watchlist_supabase.fetch_tickers", boom)
    ss = _SS()
    ss["auth_user"] = "dave"
    ensure_hydrated(ss)
    assert ss.get(SESSION_KEY) == []
    assert add_ticker(ss, "X.NS")  # still works in session
    assert "X.NS" in get_watchlist(ss)
