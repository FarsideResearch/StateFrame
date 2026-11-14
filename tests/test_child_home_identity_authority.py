"""Environment bootstrap must preserve the runner's pin without trusting the run spec."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import kernel_checkpoint as checkpoint
from stateframe import _child
from stateframe.engine import kernel_request_auth_identity as auth


class _StartupVerified(Exception):
    """Stop main at its identity check, before it installs the model loop."""


@pytest.fixture
def child_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_child.os, "environ", dict(os.environ))
    # Each test represents a fresh child process, including missing startup authority.
    monkeypatch.setattr(_child, "_RUNNER_HOME_IDENTITY_CAPTURED", False, raising=False)
    monkeypatch.setattr(_child, "_RUNNER_HOME_IDENTITY_PIN", None, raising=False)
    monkeypatch.setattr(checkpoint, "_RUNNER_HOME_IDENTITY_CONFIGURED", False)
    monkeypatch.setattr(checkpoint, "_RUNNER_HOME_IDENTITY_PIN", None)
    monkeypatch.setattr(checkpoint, "_SECRETS", [])
    monkeypatch.setattr(checkpoint, "_SECRET_GENERATION", 0)
    monkeypatch.setattr(_child, "watch_parent", lambda: None)
    monkeypatch.setattr(_child, "hold_engine_lock", lambda _run: None)
    monkeypatch.setattr(_child, "_install_direct_opener", lambda: None)
    monkeypatch.setattr(_child, "install_run_identity", lambda _seed: None)
    monkeypatch.setattr(_child.sys, "path", list(_child.sys.path))
    home, run, repo = (tmp_path / name for name in ("home", "run", "repo"))
    run.mkdir()
    repo.mkdir()
    assert not (home / "identity.key").exists()
    pin = auth.protected_home_identity(home, initialize=True)
    serialized = json.dumps(pin)
    monkeypatch.setenv(_child.RUNNER_HOME_IDENTITY_ENV, serialized)
    spec = {
        "repo": str(repo),
        "engine_dir": str(Path(checkpoint.__file__).parent),
        "environment": {
            "STATEFRAME_HOME": str(home),
            _child.RUNNER_HOME_IDENTITY_ENV: serialized,
        },
    }
    (run / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    return home, run, spec, pin


def _reach_identity_check(run: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def stop_at_check(_run: Path) -> None:
        raise _StartupVerified

    with monkeypatch.context() as patch:
        patch.setattr(checkpoint, "verify_stateframe_home_identity", stop_at_check)
        with pytest.raises(_StartupVerified):
            _child.main(["_child.py", str(run)])


@pytest.mark.parametrize("bootstrap_count", [0, 1, 2, 6])
def test_repeated_bootstrap_preserves_lazy_identity_for_six_checks(
        child_identity, monkeypatch: pytest.MonkeyPatch, bootstrap_count: int) -> None:
    home, run, spec, pin = child_identity
    key = home / "identity.key"
    initial_bytes = key.read_bytes()
    for _ in range(bootstrap_count):
        _child.install_environment(spec)
        assert _child.RUNNER_HOME_IDENTITY_ENV not in os.environ
    _reach_identity_check(run, monkeypatch)
    for _ in range(6):
        checkpoint.verify_stateframe_home_identity(run)
        assert auth.protected_home_identity(home) == pin
        assert key.read_bytes() == initial_bytes
    assert checkpoint._RUNNER_HOME_IDENTITY_PIN == pin
    assert _child.RUNNER_HOME_IDENTITY_ENV not in os.environ


@pytest.mark.parametrize("replacement", ["rewrite", "swap"])
def test_bootstrap_never_adopts_replacement_key_or_rewritten_spec(
        child_identity, monkeypatch: pytest.MonkeyPatch, replacement: str) -> None:
    home, run, spec, pin = child_identity
    _child.install_environment(spec)
    key = home / "identity.key"
    if replacement == "rewrite":
        key.write_bytes(b"a-replacement-installation-key!!")
    else:
        candidate = home / "replacement.key"
        candidate.write_bytes(b"a-replacement-installation-key!!")
        candidate.chmod(0o600)
        candidate.replace(key)
    changed = auth.protected_home_identity(home)
    assert changed != pin
    spec["environment"][_child.RUNNER_HOME_IDENTITY_ENV] = json.dumps(changed)
    (run / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    _child.install_environment(spec)
    _reach_identity_check(run, monkeypatch)
    assert checkpoint._RUNNER_HOME_IDENTITY_PIN == pin
    with pytest.raises(checkpoint.CheckpointError, match="STATEFRAME_HOME_IDENTITY_CHANGED"):
        checkpoint.verify_stateframe_home_identity(run)


@pytest.mark.parametrize("transport", [None, "null", "invalid json"])
def test_spec_cannot_supply_missing_or_invalid_startup_authority(
        child_identity, monkeypatch: pytest.MonkeyPatch, transport: str | None) -> None:
    _home, run, spec, _pin = child_identity
    if transport is None:
        monkeypatch.delenv(_child.RUNNER_HOME_IDENTITY_ENV)
    else:
        monkeypatch.setenv(_child.RUNNER_HOME_IDENTITY_ENV, transport)
    _child.install_environment(spec)
    _child.install_environment(spec)
    _reach_identity_check(run, monkeypatch)
    with pytest.raises(checkpoint.CheckpointError, match="STATEFRAME_HOME_IDENTITY_CHANGED"):
        checkpoint.verify_stateframe_home_identity(run)


def test_real_key_swap_after_six_successful_checks_is_refused(
        child_identity, monkeypatch: pytest.MonkeyPatch) -> None:
    home, run, spec, pin = child_identity
    _child.install_environment(spec)
    _reach_identity_check(run, monkeypatch)
    for _ in range(6):
        checkpoint.verify_stateframe_home_identity(run)
    candidate = home / "replacement.key"
    candidate.write_bytes(b"a-replacement-installation-key!!")
    candidate.chmod(0o600)
    candidate.replace(home / "identity.key")
    assert checkpoint._RUNNER_HOME_IDENTITY_PIN == pin
    with pytest.raises(checkpoint.CheckpointError, match="STATEFRAME_HOME_IDENTITY_CHANGED"):
        checkpoint.verify_stateframe_home_identity(run)
