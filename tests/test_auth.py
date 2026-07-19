"""Auth: PBKDF2 hashing, constant-time verify, credential checks."""
from src.auth import PBKDF2_ITERATIONS, check_credentials, hash_password, verify_password


def test_hash_verify_roundtrip():
    stored = hash_password("S3curePass!")
    assert "$" in stored
    salt, digest = stored.split("$", 1)
    assert len(salt) == 32
    assert len(digest) == 64
    assert verify_password("S3curePass!", stored)
    assert not verify_password("S3curePass", stored)
    assert not verify_password("", stored)


def test_salts_differ_for_same_password():
    a, b = hash_password("x"), hash_password("x")
    assert a != b
    assert verify_password("x", a) and verify_password("x", b)


def test_malformed_stored_hash_rejected():
    for bad in ("", "nosalt", None, 123):
        assert verify_password("x", bad) is False  # type: ignore[arg-type]


def test_check_credentials_known_user():
    users = {"admin": hash_password("admin123")}
    assert check_credentials("admin", "admin123", users)
    assert not check_credentials("admin", "wrong", users)
    assert not check_credentials("nobody", "admin123", users)


def test_pbkdf2_iteration_count_is_strong():
    assert PBKDF2_ITERATIONS >= 100_000
