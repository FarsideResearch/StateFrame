from __future__ import annotations

from types import SimpleNamespace

import pytest

import kernel_unit_policy
from kernel_pilot import ACTIVE_RUNTIME
from probes.react_repair import FileSet, _materialize_public_tests


@pytest.mark.parametrize(("path", "source", "error", "message"), [
    ("", "", ValueError, "sandbox path must be a non-empty string path"),
    ("../test_escape.py", "", ValueError, "sandbox path must be repository-relative"),
    ("/test_absolute.py", "", ValueError, "sandbox path must be repository-relative"),
    ("module.py", "", ValueError, "public test must identify a declared test file: module.py"),
    (7, "", ValueError, "sandbox path must be a non-empty string path"),
    ("tests/test_bad.py", None, TypeError, "data must be str"),
])
def test_parent_materializer_rejects_invalid_public_tests(
        tmp_path, monkeypatch, path, source, error, message):
    monkeypatch.setattr(kernel_unit_policy, "_installed", None)
    fileset = FileSet(True, {}, {}, {}, public_tests={path: source})
    with pytest.raises(error, match=message):
        _materialize_public_tests(tmp_path, fileset, {})
    assert not any(path.is_file() for path in tmp_path.rglob("*"))


def test_parent_materializer_preserves_main_normalized_alias_order(tmp_path, monkeypatch):
    monkeypatch.setattr(kernel_unit_policy, "_installed", None)
    fileset = FileSet(True, {}, {}, {}, public_tests={
        "tests/test_ok.py": "VALUE = 2\n",
        "./tests/test_ok.py": "VALUE = 1\n",
    })
    assert _materialize_public_tests(tmp_path, fileset, {}) == (
        "tests/test_ok.py", "tests/test_ok.py")
    assert (tmp_path / "tests/test_ok.py").read_bytes() == b"VALUE = 2\n"


def test_child_materializer_skips_and_journals_r5_empty_test_path(tmp_path, monkeypatch):
    monkeypatch.setattr(kernel_unit_policy, "_installed", SimpleNamespace())
    # r5's child spec public_row: no public_tests, but empty legacy test_path/test_src fields.
    row = {
        "_skip_db": True,
        "allowed_product_paths": ["shop/invoicing.py"],
        "check_selectors": [],
        "create_paths": [],
        "entry": {"fn": "", "path": "shop/invoicing.py"},
        "files": {"shop/invoicing.py": {"parent_src": "def calculate_tax():\n    return 0\n"}},
        "path": "shop/invoicing.py",
        "public_tests": {},
        "task_kind": "bug",
        "test_path": "",
        "test_src": "",
    }
    fileset = FileSet.from_row(row)
    assert dict(fileset.public_tests) == {"": ""}

    events = []
    journal = SimpleNamespace(emit=lambda kind, **fields: events.append((kind, fields)))
    runtime_token = ACTIVE_RUNTIME.set(SimpleNamespace(journal=journal))
    try:
        materialized = _materialize_public_tests(tmp_path, fileset, row)
    finally:
        ACTIVE_RUNTIME.reset(runtime_token)

    assert materialized == ()
    assert list(tmp_path.iterdir()) == []
    assert events == [("public_test_path_skipped", {
        "path": "",
        "reason": "invalid_repo_path",
        "producer": "react_repair",
        "dispatch": False,
    })]
