"""Bound-target refusals are decided before the executor and the executor repeats the same prose."""
from __future__ import annotations

import hashlib
import pathlib

import pytest

import kernel_legacy_safe_edit as lse
from kernel_bound_edit import bind_target
from probes import react_repair as rr

SOURCE = "def alpha():\n    return 1\n"


def _setup(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_BOUND_EDIT", "on")
    for name in ("STATEFRAME_EMPIRICAL_FIXES", "STATEFRAME_EDIT_PROTOCOL", "STATEFRAME_WORKSPACE_TRUTH"):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path
    p = root / "mod.py"
    p.write_text(SOURCE, encoding="utf-8")
    raw = p.read_bytes()
    ref = bind_target(path="mod.py", symbol="", revision="sha256:" + hashlib.sha256(raw).hexdigest(),
                      source=raw.decode("utf-8"))
    p.write_text(SOURCE + "# drift\n", encoding="utf-8")
    return root, p, ref


def test_stale_target_refused_before_validator_and_prose_agrees(tmp_path, monkeypatch):
    root, p, ref = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(lse, "single_edit", lambda **kw: pytest.fail("stale target reached the validator"))
    pre = rr._bound_edit_refusal(root, p, ref)
    assert pre == "bound edit target rejected: stale revision; rebind the current source"
    obs = rr.str_replace_edit(p, "    return 1\n", "    return 10\n", target_ref=ref, root=root)
    assert obs == "str_replace REJECTED: " + pre
    assert p.read_text(encoding="utf-8") == SOURCE + "# drift\n"


def test_missing_target_refused_with_golden_prose(tmp_path, monkeypatch):
    root, p, _ref = _setup(tmp_path, monkeypatch)
    obs = rr.str_replace_edit(p, "    return 1\n", "    return 10\n", target_ref=None, root=root)
    assert obs == "str_replace REJECTED: bound edit target required: supply target_ref from the repair packet"
