"""The final diff guard scans the exact source bytes before text formatting."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from stateframe import runner, workcopy
from stateframe.engine import kernel_checkpoint as checkpoint
from stateframe.engine import kernel_request_auth_identity as auth_identity


_IDENTITY_KEY = b"A" * 15 + b"\n" + b"B" * 16
_SECRET_REFUSAL = "checkpoint refused: registered secret in checkpoint"


def _pin_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict[str, int | str]]:
    monkeypatch.setattr(checkpoint, "_SECRETS", [])
    monkeypatch.setattr(checkpoint, "_SECRET_GENERATION", 0)
    home = tmp_path / "stateframe-home"
    home.mkdir(mode=0o700)
    key_path = home / "identity.key"
    key_path.write_bytes(_IDENTITY_KEY)
    key_path.chmod(0o600)
    return home, auth_identity.protected_home_identity(home)


def _workcopy(tmp_path: Path, kind: str, after: bytes | None) -> tuple[Path, workcopy.WorkCopy]:
    repo = tmp_path / "repo"
    copy = tmp_path / "run" / "workspace"
    repo.mkdir()
    copy.mkdir(parents=True)
    before = _IDENTITY_KEY if kind == "deleted" else (b"before\n" if kind == "modified" else None)
    if before is not None:
        (repo / "payload.bin").write_bytes(before)
    if after is not None:
        (copy / "payload.bin").write_bytes(after)
    baseline = ({"payload.bin": hashlib.sha256(before).hexdigest()}
                if before is not None else {})
    return repo, workcopy.WorkCopy(original=repo, copy=copy, baseline=baseline)


def _run_with_home(run_dir: Path, repo: Path, home: Path,
                   identity: dict[str, int | str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "spec.json").write_text(json.dumps({
        "repo": str(repo),
        "environment": {"STATEFRAME_HOME": str(home)},
        "stateframe_home_identity": identity,
    }), encoding="utf-8")


def _reconstructed_changed_bytes(diff: str) -> bytes:
    pieces = []
    for line in diff.splitlines(keepends=True):
        if line.startswith(("--- ", "+++ ")):
            continue
        if line.startswith(("+", "-")):
            pieces.append(line[1:].encode("utf-8", errors="surrogateescape"))
    return b"".join(pieces)


@pytest.mark.parametrize("kind,after", [
    ("added", _IDENTITY_KEY),
    ("modified", _IDENTITY_KEY),
    ("deleted", None),
])
def test_final_diff_refuses_recoverable_identity_key_in_all_change_kinds(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, after: bytes | None) -> None:
    home, identity = _pin_home(tmp_path, monkeypatch)
    repo, work = _workcopy(tmp_path, kind, after)
    run_dir = tmp_path / "run"
    _run_with_home(run_dir, repo, home, identity)

    candidate_diff = workcopy.unified_diff(work)
    formatted_bytes = candidate_diff.encode("utf-8", errors="surrogateescape")
    assert _IDENTITY_KEY not in formatted_bytes
    assert _IDENTITY_KEY.hex().encode("ascii") not in formatted_bytes
    assert _IDENTITY_KEY in _reconstructed_changed_bytes(candidate_diff)

    reason = runner._export_final_diff(work, run_dir, identity)

    assert reason == _SECRET_REFUSAL
    assert not (run_dir / "final.diff").exists()
    assert not list(run_dir.glob(".final-diff-*"))


@pytest.mark.parametrize("ending", [b"\n", b"\r\n", b"\r", b""])
def test_final_diff_keeps_safe_export_bytes_unchanged(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ending: bytes) -> None:
    home, identity = _pin_home(tmp_path, monkeypatch)
    repo, work = _workcopy(tmp_path, "modified", b"after" + ending)
    run_dir = tmp_path / "run"
    _run_with_home(run_dir, repo, home, identity)
    expected = workcopy.unified_diff(work).encode("utf-8", errors="surrogateescape")
    assert expected == (
        b"--- a/payload.bin\n+++ b/payload.bin\n@@ -1 +1 @@\n-before\n+after"
        + (ending if ending else b"\n\\ No newline at end of file\n"))

    assert runner._export_final_diff(work, run_dir, identity) is None
    assert (run_dir / "final.diff").read_bytes() == expected


@pytest.mark.parametrize("layers", [1, 2, 8])
def test_final_diff_refuses_json_escaped_split_newline_identity_key(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layers: int) -> None:
    home, identity = _pin_home(tmp_path, monkeypatch)
    payload = _IDENTITY_KEY.decode()
    for _ in range(layers):
        payload = json.dumps({"key": payload})
    encoded = payload.encode()
    assert _IDENTITY_KEY not in encoded
    repo, work = _workcopy(tmp_path, "added", encoded)
    run_dir = tmp_path / "run"
    _run_with_home(run_dir, repo, home, identity)

    assert runner._export_final_diff(work, run_dir, identity) == _SECRET_REFUSAL
    assert not (run_dir / "final.diff").exists()
    assert not list(run_dir.glob(".final-diff-*"))


@pytest.mark.parametrize("boundary", ["source", "formatted-artifact"])
def test_final_diff_refuses_seven_unicode_layers_beyond_old_window(
        tmp_path, monkeypatch, boundary):
    from secret_scan_helpers import seven_layer_json

    home, identity = _pin_home(tmp_path, monkeypatch)
    payload = seven_layer_json(_IDENTITY_KEY)
    assert len(payload) > 9_000_000
    repo, work = _workcopy(tmp_path, "added", payload if boundary == "source" else b"safe\n")
    run_dir = tmp_path / "run"
    _run_with_home(run_dir, repo, home, identity)
    if boundary == "formatted-artifact":
        # Isolate the second boundary: no secret exists in the captured source.
        monkeypatch.setattr(workcopy, "unified_diff", lambda *_args, **_kwargs: payload.decode())

    expected_reason = (_SECRET_REFUSAL if boundary == "source"
                       else "checkpoint refused: registered secret in state")
    assert runner._export_final_diff(work, run_dir, identity) == expected_reason
    assert not (run_dir / "final.diff").exists()
    assert not list(run_dir.glob(".final-diff-*"))


@pytest.mark.parametrize("payload", [b"\x00opaque-data", b"\xffopaque-data"])
def test_final_diff_scans_binary_bytes_before_omitting_text_diff(tmp_path, monkeypatch, payload):
    home, identity = _pin_home(tmp_path, monkeypatch)
    repo, work = _workcopy(tmp_path, "added", payload + _IDENTITY_KEY)
    run_dir = tmp_path / "run"
    _run_with_home(run_dir, repo, home, identity)

    # Opaque inputs have no text diff; they must still pass through the source guard.
    assert "text diff omitted" in workcopy.unified_diff(work)
    assert runner._export_final_diff(work, run_dir, identity) == _SECRET_REFUSAL
    assert not (run_dir / "final.diff").exists()
    assert not list(run_dir.glob(".final-diff-*"))
