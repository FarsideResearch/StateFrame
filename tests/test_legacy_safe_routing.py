"""legacy_safe routing: BOUND_EDIT=on takes the validator path, off keeps the inline path; prose identical."""
from __future__ import annotations

import pathlib

import pytest

import kernel_legacy_safe_edit as lse
from probes import react_repair as rr

SOURCE = "def alpha():\n    return 1\n\n\ndef beta():\n    return 2\n"


def _prep(tmp_path, monkeypatch, mode):
    for name in ("STATEFRAME_EDIT_GUARD", "STATEFRAME_EMPIRICAL_FIXES", "STATEFRAME_EFFECT_BOUNDARY",
                 "STATEFRAME_GREEN_LOCK", "STATEFRAME_DESTRUCTIVE_EDIT_GUARD", "STATEFRAME_WORKSPACE_TRUTH",
                 "STATEFRAME_EDIT_PROTOCOL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("STATEFRAME_BOUND_EDIT", mode)
    root = tmp_path / mode
    root.mkdir()
    p = root / "mod.py"
    p.write_text(SOURCE, encoding="utf-8")
    return root, p


def _bound(root, p):
    import hashlib
    from kernel_bound_edit import bind_target
    raw = p.read_bytes()
    return bind_target(path="mod.py", symbol="", revision="sha256:" + hashlib.sha256(raw).hexdigest(),
                       source=raw.decode("utf-8"))


def test_on_routes_through_single_edit(tmp_path, monkeypatch):
    root, p = _prep(tmp_path, monkeypatch, "on")
    calls = []
    real = lse.single_edit

    def spy(**kw):
        calls.append(kw["relative_path"])
        return real(**kw)

    monkeypatch.setattr(lse, "single_edit", spy)
    obs = rr.str_replace_edit(p, "    return 1\n", "    return 10\n", target_ref=_bound(root, p), root=root)
    assert obs == "str_replace applied." and calls == ["mod.py"]
    assert p.read_text(encoding="utf-8") == SOURCE.replace("    return 1\n", "    return 10\n", 1)


def test_off_never_touches_single_edit(tmp_path, monkeypatch):
    root, p = _prep(tmp_path, monkeypatch, "off")
    monkeypatch.setattr(lse, "single_edit", lambda **kw: pytest.fail("validator route taken with BOUND_EDIT=off"))
    assert rr.str_replace_edit(p, "    return 1\n", "    return 10\n", root=root) == "str_replace applied."


def test_handles_protocol_does_not_take_the_legacy_route(tmp_path, monkeypatch):
    root, p = _prep(tmp_path, monkeypatch, "on")
    monkeypatch.setenv("STATEFRAME_EDIT_PROTOCOL", "handles")
    assert rr._legacy_safe_validator_route(root) is False


def test_empirical_fixes_on_keeps_inline_path(tmp_path, monkeypatch):
    root, p = _prep(tmp_path, monkeypatch, "on")
    monkeypatch.setenv("STATEFRAME_EMPIRICAL_FIXES", "on")
    assert rr._legacy_safe_validator_route(root) is False
