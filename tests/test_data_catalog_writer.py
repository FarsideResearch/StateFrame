"""Data workspace catalog contract: SQLite WAL safety gate, single-writer lock, read-only readers, schema.

Red tests for stage 2. The module under test (kernel_data_catalog) is not implemented yet, so every test is a
strict xfail that fails on ImportError. Modules are imported inside each test so the file always collects.
All database paths live under tmp_path. Nothing here touches the real StateFrame home.
"""
from __future__ import annotations

import importlib
import sqlite3

import pytest

pytestmark = pytest.mark.xfail(strict=True, raises=ImportError, reason="DW2: module not implemented yet")

_REQUIRED_TABLES = {"objects", "resources", "snapshots", "datasets", "revisions", "results", "packets", "jobs"}
_SAFE_SQLITE = (3, 51, 3)
_UNSAFE_SQLITE = (3, 46, 1)


def _pin_sqlite_version(monkeypatch: pytest.MonkeyPatch, mod: object, version: tuple[int, int, int]) -> None:
    # The module reads SQLITE_VERSION at call time, so patching the module attribute is enough.
    monkeypatch.setattr(mod, "SQLITE_VERSION", version)


# --- sqlite_wal_safe --------------------------------------------------------

@pytest.mark.parametrize(
    ("version", "safe"),
    [
        ((3, 46, 1), False),
        ((3, 44, 5), False),
        ((3, 44, 6), True),
        ((3, 50, 6), False),
        ((3, 50, 7), True),
        ((3, 51, 2), False),
        ((3, 51, 3), True),
        ((3, 52, 0), True),
    ],
)
def test_sqlite_wal_safe_per_release_line(version, safe):
    mod = importlib.import_module("kernel_data_catalog")
    assert mod.sqlite_wal_safe(version) is safe


# --- open_catalog WAL gate --------------------------------------------------

def test_open_catalog_refuses_wal_on_unsafe_sqlite_without_single_writer(monkeypatch, tmp_path):
    mod = importlib.import_module("kernel_data_catalog")
    _pin_sqlite_version(monkeypatch, mod, _UNSAFE_SQLITE)
    with pytest.raises(mod.CatalogRefused) as exc_info:
        mod.open_catalog(tmp_path / "catalog.db", single_writer=False)
    assert exc_info.value.code == "wal_unpatched_sqlite"


def test_open_catalog_single_writer_override_opens_in_wal(monkeypatch, tmp_path):
    mod = importlib.import_module("kernel_data_catalog")
    _pin_sqlite_version(monkeypatch, mod, _UNSAFE_SQLITE)
    conn = mod.open_catalog(tmp_path / "catalog.db", single_writer=True)
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
    finally:
        conn.close()


# --- single-writer lock and readers -----------------------------------------

def test_second_writer_on_same_path_is_refused_while_first_is_open(monkeypatch, tmp_path):
    mod = importlib.import_module("kernel_data_catalog")
    _pin_sqlite_version(monkeypatch, mod, _SAFE_SQLITE)
    path = tmp_path / "catalog.db"
    first = mod.open_catalog(path, single_writer=False)
    try:
        with pytest.raises(mod.CatalogRefused) as exc_info:
            mod.open_catalog(path, single_writer=False)
        assert exc_info.value.code == "writer_busy"
    finally:
        first.close()


def test_reader_coexists_with_writer_and_cannot_write(monkeypatch, tmp_path):
    mod = importlib.import_module("kernel_data_catalog")
    _pin_sqlite_version(monkeypatch, mod, _SAFE_SQLITE)
    path = tmp_path / "catalog.db"
    writer = mod.open_catalog(path, single_writer=False)
    reader = None
    try:
        reader = mod.open_catalog_reader(path)
        # Reads must work through the reader while the writer is open. Proves the table exists before the write attempt,
        # so the write failure below cannot be a missing-table OperationalError.
        assert reader.execute("SELECT COUNT(*) FROM objects").fetchone()[0] == 0
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            reader.execute("INSERT INTO objects DEFAULT VALUES")
    finally:
        if reader is not None:
            reader.close()
        writer.close()


# --- schema -----------------------------------------------------------------

def test_catalog_creates_required_tables(monkeypatch, tmp_path):
    mod = importlib.import_module("kernel_data_catalog")
    _pin_sqlite_version(monkeypatch, mod, _SAFE_SQLITE)
    conn = mod.open_catalog(tmp_path / "catalog.db", single_writer=False)
    try:
        names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        missing = _REQUIRED_TABLES - names
        assert not missing, f"catalog missing tables: {sorted(missing)}"
    finally:
        conn.close()
