"""The installation-key cache follows the key file, not just its path.

A long-lived process (an embedding runner running several runs) can see the key at the same
path rotated, replaced or deleted with its home. The cached secret must then be reloaded, so
the secret registered for checkpoint redaction is the key actually on disk.
"""
from __future__ import annotations

import os
import secrets
import shutil
from pathlib import Path

import kernel_request_auth_identity as auth


def _rotate(key: Path) -> bytes:
    new = secrets.token_bytes(32)
    replacement = key.with_name("identity.key.new")
    descriptor = os.open(replacement, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        os.write(descriptor, new)
    finally:
        os.close(descriptor)
    os.replace(replacement, key)
    return new


def test_unchanged_key_is_served_from_cache(tmp_path, monkeypatch):
    home = tmp_path / "home"
    first = auth._installation_secret(home)
    reads = []
    real = auth._read_secret
    monkeypatch.setattr(auth, "_read_secret", lambda path: reads.append(path) or real(path))
    assert auth._installation_secret(home) == first
    assert reads == []


def test_key_rotated_at_same_path_is_reloaded(tmp_path):
    home = tmp_path / "home"
    first = auth._installation_secret(home)
    rotated = _rotate(home / "identity.key")
    assert rotated != first
    assert auth._installation_secret(home) == rotated


def test_key_rewritten_in_place_is_reloaded(tmp_path):
    home = tmp_path / "home"
    key = home / "identity.key"
    first = auth._installation_secret(home)
    new = secrets.token_bytes(32)
    key.write_bytes(new)
    os.utime(key, ns=(1, 1))  # a rewrite must not hide behind an unchanged inode
    assert auth._installation_secret(home) == new != first


def test_home_recreated_between_runs_initializes_a_fresh_identity(tmp_path):
    home = tmp_path / "home"
    first = auth.protected_home_identity(home, initialize=True)
    shutil.rmtree(home)
    second = auth.protected_home_identity(home, initialize=True)
    assert (home / "identity.key").is_file()
    assert (second["key_st_dev"], second["key_st_ino"], second["key_st_ctime_ns"]) != (
        first["key_st_dev"], first["key_st_ino"], first["key_st_ctime_ns"])
    assert auth._installation_secret(home) == (home / "identity.key").read_bytes()


def test_fallback_salt_is_replaced_once_a_valid_key_appears(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setattr(auth, "_create_secret", lambda path: None)
    fallback = auth._installation_secret(home)
    assert fallback == auth._fallback_secret()
    assert auth._installation_secret(home) == fallback
    home.mkdir()
    created = _rotate(home / "identity.key")
    assert auth._installation_secret(home) == created
