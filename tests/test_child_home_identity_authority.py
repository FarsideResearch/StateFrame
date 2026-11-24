"""Environment bootstrap must preserve the runner's pin without trusting the run spec."""
from __future__ import annotations

import errno
import time
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
    monkeypatch.setattr(checkpoint, "_PRODUCT_HOST", False)
    monkeypatch.setattr(checkpoint, "_HOST_IDENTITY_BOUND", False)
    monkeypatch.setattr(checkpoint, "_FINGERPRINT", None)
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


def _prepare_checkpoint_save(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(checkpoint, "_STATE", {})
    monkeypatch.setattr(checkpoint, "_SIGNING_KEY", [])
    monkeypatch.setattr(checkpoint, "_VERIFIED_POOL_IDENTITIES", set())
    monkeypatch.setattr(checkpoint, "_POOL_KNOWN", {})
    context = {name: {} for name in ("hostfns", "participants", "framecells", "framefuncs", "cvars")}
    monkeypatch.setattr(checkpoint, "collect", lambda _frame: ({"frame": {}}, context))
    checkpoint.register_signing_key(b"child-home-boundary-test-key-0123456789")


def test_child_declares_product_host_at_startup(
        child_identity, monkeypatch: pytest.MonkeyPatch) -> None:
    _home, run, _spec, _pin = child_identity
    assert checkpoint._PRODUCT_HOST is False
    _reach_identity_check(run, monkeypatch)
    assert checkpoint._PRODUCT_HOST is True


@pytest.mark.parametrize("explicit_pin", [False, True])
def test_parent_validation_allows_ambient_home_different_from_run_home(
        child_identity, monkeypatch: pytest.MonkeyPatch, explicit_pin: bool) -> None:
    home, run, _spec, pin = child_identity
    ambient = home.parent / "parent-ambient-home"
    monkeypatch.setenv("STATEFRAME_HOME", str(ambient))
    assert checkpoint._RUNNER_HOME_IDENTITY_CONFIGURED is False
    if explicit_pin:
        checkpoint.verify_stateframe_home_identity(run, expected_identity=pin)
    else:
        checkpoint.verify_stateframe_home_identity(run)
    assert not ambient.exists()


@pytest.mark.parametrize("boundary", ["save", "capture", "fsync", "publication"])
def test_live_effective_home_switch_refuses_save_without_capturing_new_key(
        child_identity, monkeypatch: pytest.MonkeyPatch, boundary: str) -> None:
    _home, run, _spec, _pin = child_identity
    _reach_identity_check(run, monkeypatch)
    checkpoint.verify_stateframe_home_identity(run)
    _prepare_checkpoint_save(monkeypatch)
    selected = run / "new-active-home"
    selected.mkdir(mode=0o700)
    key = selected / "identity.key"
    secret = bytes(range(32))
    key.write_bytes(secret)
    key.chmod(0o600)
    # Do not register the new key: this must be refused by the live home pin itself.
    assert secret not in checkpoint._SECRETS
    switched = False

    def switch() -> None:
        nonlocal switched
        monkeypatch.setenv("STATEFRAME_HOME", str(selected))
        switched = True

    if boundary == "save":
        switch()
    else:
        hook = {"capture": "collect", "fsync": "_fsync_snapshot",
                "publication": "_publish_staged_artifacts"}[boundary]
        original = getattr(checkpoint, hook)

        def switch_after(*args, **kwargs):
            result = original(*args, **kwargs)
            switch()
            return result

        monkeypatch.setattr(checkpoint, hook, switch_after)
    try:
        with pytest.raises(checkpoint.CheckpointError, match="STATEFRAME_HOME_IDENTITY_CHANGED"):
            checkpoint.save(None, 1, run)
    finally:
        root = run / "checkpoints"
        assert switched
        assert not (root / "LATEST").exists()
        assert not list(root.rglob(checkpoint.META_FILE))
        assert all(secret not in path.read_bytes() for path in root.rglob("*") if path.is_file())



@pytest.mark.parametrize("links", [True, False])
@pytest.mark.parametrize("background", [False, True])
def test_refused_save_keeps_only_bytes_an_accepted_checkpoint_holds(
        child_identity, monkeypatch: pytest.MonkeyPatch, links: bool, background: bool) -> None:
    _home, run, _spec, _pin = child_identity
    _reach_identity_check(run, monkeypatch)
    checkpoint.verify_stateframe_home_identity(run)
    _prepare_checkpoint_save(monkeypatch)
    if not links:
        def no_link(*_args, **_kwargs):
            raise PermissionError(errno.EPERM, "no links here")
        monkeypatch.setattr(checkpoint.os, "link", no_link)
    (run / "kept.txt").write_bytes(b"accepted bytes")
    saved = checkpoint.save(None, 1, run)
    root = run / "checkpoints"
    (run / "late.txt").write_bytes(b"refused bytes")
    original = checkpoint._fsync_snapshot

    def refuse(*args, **kwargs):
        original(*args, **kwargs)
        raise checkpoint.CheckpointError("STATEFRAME_HOME_IDENTITY_CHANGED")

    monkeypatch.setattr(checkpoint, "_fsync_snapshot", refuse)
    if background:
        monkeypatch.setattr(checkpoint, "_note_failure", lambda *_args, **_kwargs: None)
        checkpoint.save(None, 2, run, background=True)
        while not checkpoint._reap():
            time.sleep(0.05)
    else:
        with pytest.raises(checkpoint.CheckpointError, match="STATEFRAME_HOME_IDENTITY_CHANGED"):
            checkpoint.save(None, 2, run)
    files = [path.read_bytes() for path in root.rglob("*") if path.is_file()]
    assert all(b"refused bytes" not in data for data in files)
    assert any(b"accepted bytes" in data for data in files)
    assert checkpoint.validate(saved)["step"] == 1
    monkeypatch.setattr(checkpoint, "_fsync_snapshot", original)
    assert checkpoint.validate(checkpoint.save(None, 3, run))["step"] == 3

def test_stale_checkpoint_transaction_prunes_only_unreferenced_objects(
        child_identity, monkeypatch: pytest.MonkeyPatch) -> None:
    _home, run, _spec, _pin = child_identity
    root = run / "checkpoints"
    objects = root / checkpoint.OBJECT_DIR
    objects.mkdir(parents=True)

    prior = root / "0001-1234"
    (prior / "run").mkdir(parents=True)
    (prior / checkpoint.META_FILE).write_text("{}", encoding="utf-8")
    referenced = objects / ("a" * 64 + "-600")
    referenced.write_bytes(b"kept by prior checkpoint")
    os.link(referenced, prior / "run" / "saved")

    stale = root / ".0002-987654321.tmp"
    (stale / "run").mkdir(parents=True)
    orphan = objects / ("b" * 64 + "-600")
    orphan.write_bytes(b"only the interrupted transaction")
    os.link(orphan, stale / "run" / "captured")
    assert orphan.stat().st_nlink == 2

    monkeypatch.setattr(checkpoint, "_pid_alive", lambda _pid: False)
    assert checkpoint._sweep_stale(root) == 1

    assert not stale.exists()
    assert not orphan.exists()
    assert referenced.read_bytes() == b"kept by prior checkpoint"
    assert referenced.stat().st_nlink == 2


@pytest.mark.parametrize("absolute_override", [False, True])
def test_home_change_only_refused_when_it_changes_effective_home(
        child_identity, monkeypatch: pytest.MonkeyPatch, absolute_override: bool) -> None:
    home, run, spec, _pin = child_identity
    if not absolute_override:
        home = home / ".stateframe"
        pin = auth.protected_home_identity(home, initialize=True)
        serialized = json.dumps(pin)
        monkeypatch.setenv(_child.RUNNER_HOME_IDENTITY_ENV, serialized)
        spec["environment"]["STATEFRAME_HOME"] = str(home)
        spec["environment"][_child.RUNNER_HOME_IDENTITY_ENV] = serialized
        (run / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    _reach_identity_check(run, monkeypatch)
    monkeypatch.setenv("HOME", str(home.parent))
    if not absolute_override:
        monkeypatch.delenv("STATEFRAME_HOME")
    checkpoint.verify_stateframe_home_identity(run)
    _prepare_checkpoint_save(monkeypatch)
    monkeypatch.setenv("HOME", str(run / "changed-user-home"))
    if absolute_override:
        saved = checkpoint.save(None, 1, run)
        assert checkpoint.validate(saved)["step"] == 1
    else:
        with pytest.raises(checkpoint.CheckpointError, match="STATEFRAME_HOME_IDENTITY_CHANGED"):
            checkpoint.save(None, 1, run)
        assert not (run / "checkpoints").exists()


def test_child_startup_refuses_effective_home_different_from_runner_pin(
        child_identity, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _home, run, _spec, _pin = child_identity
    install = _child.install_environment

    def install_then_switch(spec):
        install(spec)
        monkeypatch.setenv("STATEFRAME_HOME", str(run / "unexpected-home"))

    verify = checkpoint.verify_stateframe_home_identity

    def verify_then_stop(run):
        verify(run)
        raise _StartupVerified("startup accepted a different effective home")

    monkeypatch.setattr(_child, "install_environment", install_then_switch)
    monkeypatch.setattr(checkpoint, "verify_stateframe_home_identity", verify_then_stop)
    assert _child.main(["_child.py", str(run)]) == 1
    assert "STATEFRAME_HOME_IDENTITY_CHANGED" in capsys.readouterr().out
    assert not (run / "checkpoints").exists()


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("kind", ["symlink", "directory"])
def test_refused_save_leaves_files_the_store_did_not_write(
        child_identity, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: str, background: bool) -> None:
    _home, run, _spec, _pin = child_identity
    _reach_identity_check(run, monkeypatch)
    checkpoint.verify_stateframe_home_identity(run)
    _prepare_checkpoint_save(monkeypatch)
    folder = run / "checkpoints" / checkpoint.OBJECT_DIR
    folder.parent.mkdir(exist_ok=True)
    target = tmp_path / "elsewhere" if kind == "symlink" else folder
    target.mkdir()
    if kind == "symlink":
        folder.symlink_to(target, target_is_directory=True)
    kept = [target / "notes.tmp", target / "notes.txt"]
    if kind == "symlink":
        kept.append(target / ("a" * 64 + "-600"))
    for path in kept:
        path.write_bytes(b"not ours")
    (run / "late.txt").write_bytes(b"refused bytes")

    def refuse(*_args, **_kwargs):
        raise checkpoint.CheckpointError("STATEFRAME_HOME_IDENTITY_CHANGED")

    monkeypatch.setattr(checkpoint, "_fsync_snapshot", refuse)
    monkeypatch.setattr(checkpoint, "_note_failure", lambda *_args, **_kwargs: None)
    try:
        checkpoint.save(None, 1, run, background=background)
    except checkpoint.CheckpointError:
        pass
    while background and not checkpoint._reap():
        time.sleep(0.05)
    assert all(path.read_bytes() == b"not ours" for path in kept)
    assert {path.name for path in target.iterdir()} == {path.name for path in kept}


@pytest.mark.parametrize("links", ["on", "off", "other-device"])
@pytest.mark.parametrize("background", [False, True])
def test_refused_save_removes_published_pool_files(
        child_identity, monkeypatch: pytest.MonkeyPatch, links: str, background: bool) -> None:
    home, run, _spec, _pin = child_identity
    _reach_identity_check(run, monkeypatch)
    checkpoint.verify_stateframe_home_identity(run)
    _prepare_checkpoint_save(monkeypatch)
    (run / "kept.txt").write_bytes(b"accepted bytes")
    kept = run / "worker" / "candidates" / "kept.txt"
    kept.parent.mkdir(parents=True)
    kept.write_bytes(b"accepted pool bytes")
    accepted = checkpoint.save(None, 1, run)
    selected = home.parent / "replacement-home"
    selected.mkdir()
    key = bytes(range(32))
    for path in (selected / "identity.key", kept.parent / "identity.key"):
        path.write_bytes(key)
        path.chmod(0o600)
    if links != "on":
        def no_link(*_args, **_kwargs):
            raise PermissionError(errno.EPERM, "no links here")
        monkeypatch.setattr(checkpoint.os, "link", no_link)
    if links == "other-device":
        rename = os.rename

        def no_move(source, target, *args, **kwargs):
            if checkpoint.POOL_DIR in Path(target).parts:
                raise OSError(errno.EXDEV, "other device")
            return rename(source, target, *args, **kwargs)
        monkeypatch.setattr(checkpoint.os, "rename", no_move)
    publish = checkpoint._publish_staged_artifacts

    def publish_then_switch(root: Path, temporary: Path) -> None:
        publish(root, temporary)
        monkeypatch.setenv("STATEFRAME_HOME", str(selected))

    monkeypatch.setattr(checkpoint, "_publish_staged_artifacts", publish_then_switch)
    if background:
        monkeypatch.setattr(checkpoint, "_note_failure", lambda *_args, **_kwargs: None)
        checkpoint.save(None, 2, run, background=True)
        while not checkpoint._reap():
            time.sleep(0.05)
    else:
        with pytest.raises(checkpoint.CheckpointError, match="STATEFRAME_HOME_IDENTITY_CHANGED"):
            checkpoint.save(None, 2, run)
    files = [path for path in (run / "checkpoints").rglob("*") if path.is_file()]
    assert [path for path in files if key in path.read_bytes()] == []
    assert any(b"accepted pool bytes" in path.read_bytes() for path in files)
    assert checkpoint.latest(run) == accepted
    assert checkpoint.validate(accepted)["step"] == 1
