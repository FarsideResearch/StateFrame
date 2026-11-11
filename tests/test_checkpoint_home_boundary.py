"""Frozen regression: checkpoint run-file capture excludes a nested StateFrame home."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
import sys

import pytest

from _timeouts import scaled
from stateframe import runner, scripted_endpoint
from stateframe.endpoint import Endpoint
import stateframe.engine.kernel_request_auth_identity as auth_identity


ROOT = Path(__file__).resolve().parents[1]


def _repo(path: Path) -> Path:
    (path / "pkg").mkdir(parents=True)
    (path / "pkg" / "value.py").write_text("VALUE = 1\n", encoding="utf-8")
    return path


def _step(tool: str, **arguments: object) -> dict[str, object]:
    return {"tool": tool, "required": True, "arguments": arguments}


@pytest.mark.parametrize("home_relative", [Path("state"), Path("private/state/home")])
def test_real_run_checkpoint_omits_every_file_under_nested_stateframe_home(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, home_relative: Path) -> None:
    repo = _repo(tmp_path / "repo")
    run_dir = tmp_path / "run"
    state_home = run_dir / home_relative
    script = scripted_endpoint.Script({
        "model": "scripted",
        "steps": [
            _step("read", path="pkg/value.py"),
            _step("done", summary="Read the value module."),
        ],
        "tokenize": {"mode": "ok"},
    })

    # Let this real-run test exercise the snapshot defense independently of the
    # runner's separate fail-closed home/run/repository refusal.
    monkeypatch.setattr(
        runner, "_validate_stateframe_home_boundary", lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(runner, "_capture_stateframe_home_identity", lambda *_args: None,
                        raising=False)
    monkeypatch.setattr(runner, "_verify_stateframe_home_identity", lambda *_args: None,
                        raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "isolated-home"))
    monkeypatch.setenv("STATEFRAME_HOME", str(state_home))
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src"))
    _configure_checkpoint_child(tmp_path, monkeypatch, disable_identity_validation=True)
    server = scripted_endpoint.serve(script)

    original_prepare = runner.prepare

    def prepare_with_nested_home(options: runner.RunOptions, printer=print):
        prepared = original_prepare(options, printer)
        state_home.mkdir(parents=True, exist_ok=True)
        auth_identity._installation_secret()
        (state_home / "home-marker.dat").write_bytes(b"test marker, not a secret")
        return prepared

    monkeypatch.setattr(runner, "prepare", prepare_with_nested_home)
    try:
        options = runner.RunOptions(
            repo=repo,
            task="Read pkg/value.py and report its value.",
            endpoint=Endpoint(
                f"http://127.0.0.1:{server.server_port}/v1", "scripted",
            ),
            run_dir=run_dir,
            max_turns=4,
            test_python=sys.executable,
            tool_settings=runner.ToolSettings(shell="off", run_env=False),
            keep_checkpoints=True,
            no_project_hooks=True,
            instructions="off",
        )
        runner.run(options, printer=lambda _message: None)
        script.assert_consumed()
    finally:
        server.shutdown()
        server.server_close()

    key_path = state_home / "identity.key"
    marker_path = state_home / "home-marker.dat"
    assert key_path.is_file(), "the real run did not create identity.key in STATEFRAME_HOME"
    assert marker_path.is_file(), "the scripted run did not plant its non-secret home marker"

    checkpoint_dirs = [
        path for path in (run_dir / "checkpoints").iterdir()
        if path.is_dir() and (path / "checkpoint.json").is_file()
    ]
    assert checkpoint_dirs, "the real run did not retain a checkpoint"
    home_relative_to_run = state_home.relative_to(run_dir)
    for checkpoint_dir in checkpoint_dirs:
        captured_home = checkpoint_dir / "run" / home_relative_to_run
        captured_files = [
            path for path in captured_home.rglob("*")
            if path.is_file() or path.is_symlink()
        ] if captured_home.exists() else []
        if captured_files:
            pytest.fail(
                f"checkpoint {checkpoint_dir.name} captured {len(captured_files)} "
                "file(s) from STATEFRAME_HOME"
            )


_TEST_IDENTITY_KEY = b"0123456789abcdef0123456789abcdef"
_TEST_API_KEY_ENV = "SF_TEST_HOME_BOUNDARY_API_KEY"
_TEST_PROVIDER_KEY = "isolated-test-provider-credential"
_SECRET_GUARD_REASON = "checkpoint refused: registered secret in state"
_HOME_IDENTITY_REASON = "STATEFRAME_HOME_IDENTITY_CHANGED"
_FILE_SCAN_CHUNK_BYTES = 1 << 20


def _configure_checkpoint_child(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *,
        disable_identity_validation: bool = False,
        register_test_secret: bool = True) -> None:
    """Install a child-process control for the separate home-boundary validator."""
    hook_dir = tmp_path / "snapshot-control-hook"
    hook_dir.mkdir()
    engine_dir = ROOT / "src" / "stateframe" / "engine"
    hook = ["import sys\n", f"sys.path.insert(0, {str(engine_dir)!r})\n",
            "import kernel_checkpoint\n"]
    if register_test_secret:
        hook.append(
            f"kernel_checkpoint.register_secret(bytes.fromhex({_TEST_IDENTITY_KEY.hex()!r}))\n")
    if disable_identity_validation:
        hook.append("kernel_checkpoint._verify_stateframe_home_identity = lambda _run_dir, **_kwargs: None\n")
    (hook_dir / "sitecustomize.py").write_text("".join(hook), encoding="utf-8")
    monkeypatch.setattr(runner, "PASS_ENV", (*runner.PASS_ENV, "PYTHONPATH"))
    monkeypatch.setenv("PYTHONPATH", f"{hook_dir}:{ROOT / 'src'}")


def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, bytes]:
    home = tmp_path / "isolated-stateframe-home"
    home.mkdir(mode=0o700)
    key_path = home / "identity.key"
    key_path.write_bytes(_TEST_IDENTITY_KEY)
    key_path.chmod(0o600)
    monkeypatch.setenv("HOME", str(tmp_path / "isolated-user-home"))
    monkeypatch.setenv("STATEFRAME_HOME", str(home))
    monkeypatch.setenv(_TEST_API_KEY_ENV, _TEST_PROVIDER_KEY)
    # Register the same isolated file in the host process used by artifact-export checks.
    if auth_identity._installation_secret() != _TEST_IDENTITY_KEY:
        pytest.fail("isolated identity key setup failed")
    return home, _TEST_IDENTITY_KEY


def _scripted_real_run(repo: Path, run_dir: Path, after_prepare, *,
                       monkeypatch: pytest.MonkeyPatch,
                       profile: str | Path | None = None,
                       extra_env: dict[str, str] | None = None,
                       api_key_env: str | None = _TEST_API_KEY_ENV,
                       after_first_request=None,
                       expect_all_steps: bool = True) -> dict[str, object]:
    script = scripted_endpoint.Script({
        "model": "scripted",
        "steps": [
            _step("read", path="pkg/value.py"),
            _step("done", summary="Read the value module."),
        ],
        "tokenize": {"mode": "ok"},
    })
    original_next_call = script.next_call
    first_request = True

    def next_call(offered: list[str]) -> dict[str, object]:
        nonlocal first_request
        result = original_next_call(offered)
        if first_request and after_first_request is not None:
            first_request = False
            after_first_request()
        return result

    script.next_call = next_call
    server = scripted_endpoint.serve(script)
    original_prepare = runner.prepare

    def prepare_with_payload(options: runner.RunOptions, printer=print):
        prepared = original_prepare(options, printer)
        after_prepare(options, prepared)
        return prepared

    monkeypatch.setattr(runner, "prepare", prepare_with_payload)
    try:
        options = runner.RunOptions(
            repo=repo,
            task="Read pkg/value.py and report its value.",
            endpoint=Endpoint(
                f"http://127.0.0.1:{server.server_port}/v1", "scripted",
                api_key_env=api_key_env,
            ),
            profile=str(profile or runner.profiles.DEFAULT),
            run_dir=run_dir,
            max_turns=4,
            test_python=sys.executable,
            tool_settings=runner.ToolSettings(shell="off", run_env=False),
            keep_checkpoints=True,
            no_project_hooks=True,
            instructions="off",
            extra_env=extra_env or {},
        )
        result = runner.run(options, printer=lambda _message: None)
        if expect_all_steps:
            script.assert_consumed()
        return result
    finally:
        server.shutdown()
        server.server_close()


def _checkpoint_artifacts(run_dir: Path) -> list[Path]:
    checkpoint_root = run_dir / "checkpoints"
    return [path for path in checkpoint_root.rglob("*") if path.is_file()]


def _checkpoint_diagnostics(run_dir: Path) -> str:
    paths = [run_dir / "engine.log", run_dir / "checkpoints" / "events.jsonl"]
    return "\n".join(path.read_text(encoding="utf-8", errors="replace")
                      for path in paths if path.is_file())


def _checkpoint_events(run_dir: Path) -> list[dict[str, object]]:
    path = run_dir / "checkpoints" / "events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _assert_secret_absent(run_dir: Path, secret: bytes, message: str) -> None:
    if any(secret in artifact.read_bytes() for artifact in _checkpoint_artifacts(run_dir)):
        pytest.fail(message)


@pytest.mark.parametrize(
    "payload_kind",
    ["hardlink-in-repo", "copy-in-run", "chunk-boundary-pooled-blob"],
)
def test_real_run_checkpoint_refuses_registered_identity_key_bytes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload_kind: str) -> None:
    home, secret = _isolated_home(tmp_path, monkeypatch)
    repo = _repo(tmp_path / "repo")
    run_dir = tmp_path / "run"

    def add_payload(_options, prepared) -> None:
        if payload_kind == "hardlink-in-repo":
            os.link(home / "identity.key", prepared[1].copy / "linked-payload.bin")
        elif payload_kind == "copy-in-run":
            (prepared[0] / "copied-payload.bin").write_bytes(secret)
        elif payload_kind == "chunk-boundary-pooled-blob":
            wire = prepared[0] / "wire"
            wire.mkdir()
            prefix = b"x" * (_FILE_SCAN_CHUNK_BYTES - len(secret) // 2)
            (wire / "embedded-payload.bin").write_bytes(prefix + secret + b"y" * 64)

    _scripted_real_run(repo, run_dir, add_payload, monkeypatch=monkeypatch,
                       expect_all_steps=payload_kind != "hardlink-in-repo")

    diagnostics = _checkpoint_diagnostics(run_dir)
    reason = _HOME_IDENTITY_REASON if payload_kind == "hardlink-in-repo" else _SECRET_GUARD_REASON
    if reason not in diagnostics:
        pytest.fail("the real run did not refuse the changed identity or copied secret")
    _assert_secret_absent(run_dir, secret, "registered identity key bytes reached checkpoint artifacts")


def test_real_run_refuses_checkpoint_when_stateframe_home_moves_after_start(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home, secret = _isolated_home(tmp_path, monkeypatch)
    repo = _repo(tmp_path / "repo")
    run_dir = tmp_path / "run"

    def move_home(_options, prepared) -> None:
        home.rename(prepared[0] / "moved-stateframe-home")

    _scripted_real_run(repo, run_dir, move_home, monkeypatch=monkeypatch, expect_all_steps=False)

    diagnostics = _checkpoint_diagnostics(run_dir)
    if _HOME_IDENTITY_REASON not in diagnostics:
        pytest.fail("the real run did not fail closed after STATEFRAME_HOME identity changed")
    _assert_secret_absent(run_dir, secret, "a moved identity.key reached checkpoint artifacts")


def test_next_checkpoint_refuses_when_spec_and_both_settings_markers_disappear(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _home, secret = _isolated_home(tmp_path, monkeypatch)
    repo = _repo(tmp_path / "repo")
    run_dir = tmp_path / "run"
    _configure_checkpoint_child(tmp_path, monkeypatch, register_test_secret=False)

    def remove_live_metadata() -> None:
        for name in ("spec.json", "tool-settings.json", "skill-settings.json"):
            (run_dir / name).unlink()

    _scripted_real_run(repo, run_dir, lambda *_args: None, monkeypatch=monkeypatch,
                       after_first_request=remove_live_metadata)
    assert any(event.get("event") == "checkpoint_failed"
               and event.get("failure_reason") == _HOME_IDENTITY_REASON
               for event in _checkpoint_events(run_dir))
    _assert_secret_absent(run_dir, secret, "identity.key reached checkpoint artifacts after pin loss")


def test_next_checkpoint_refuses_valid_replacement_home_pin(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _home, secret = _isolated_home(tmp_path, monkeypatch)
    repo = _repo(tmp_path / "repo")
    run_dir = tmp_path / "run"
    alternate_home = tmp_path / "alternate-home"
    alternate_home.mkdir(mode=0o700)
    alternate_key = alternate_home / "identity.key"
    alternate_key.write_bytes(b"abcdef0123456789abcdef0123456789")
    alternate_key.chmod(0o600)
    home_info, key_info = alternate_home.stat(), alternate_key.stat()
    alternate_pin = {
        "realpath": str(alternate_home),
        "home_st_dev": home_info.st_dev, "home_st_ino": home_info.st_ino,
        "key_st_dev": key_info.st_dev, "key_st_ino": key_info.st_ino,
        "key_st_ctime_ns": key_info.st_ctime_ns,
    }
    _configure_checkpoint_child(tmp_path, monkeypatch, register_test_secret=False)

    def replace_live_pin() -> None:
        spec_path = run_dir / "spec.json"
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        spec["environment"]["STATEFRAME_HOME"] = str(alternate_home)
        spec["environment"][runner.RUNNER_HOME_IDENTITY_ENV] = json.dumps(alternate_pin)
        spec_path.write_text(json.dumps(spec), encoding="utf-8")

    result = _scripted_real_run(repo, run_dir, lambda *_args: None,
                                monkeypatch=monkeypatch, after_first_request=replace_live_pin)
    assert any(event.get("event") == "checkpoint_failed"
               and event.get("failure_reason") == _HOME_IDENTITY_REASON
               for event in _checkpoint_events(run_dir))
    assert result.get("final_diff_error") == _HOME_IDENTITY_REASON
    _assert_secret_absent(run_dir, secret, "identity.key reached checkpoint artifacts after pin replacement")


def test_no_api_key_checkpoint_refuses_copied_identity_key_bytes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _home, secret = _isolated_home(tmp_path, monkeypatch)
    monkeypatch.delenv(_TEST_API_KEY_ENV)
    repo = _repo(tmp_path / "repo")
    run_dir = tmp_path / "run"
    _configure_checkpoint_child(tmp_path, monkeypatch, register_test_secret=False)

    def copy_identity_key(_options, prepared) -> None:
        (prepared[0] / "copied-identity-key.bin").write_bytes(secret)

    _scripted_real_run(repo, run_dir, copy_identity_key,
                       monkeypatch=monkeypatch, api_key_env=None)
    assert any(event.get("event") == "checkpoint_failed"
               and event.get("failure_reason") == _SECRET_GUARD_REASON
               for event in _checkpoint_events(run_dir))
    _assert_secret_absent(run_dir, secret, "copied identity.key reached checkpoint artifacts")


@pytest.mark.parametrize("spec_change", ["unreadable", "missing-pin", "missing-spec"])
def test_runner_created_run_refuses_unavailable_home_pin(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spec_change: str) -> None:
    from stateframe.engine import kernel_checkpoint

    run_dir = tmp_path / "runner-created-run"
    run_dir.mkdir()
    (run_dir / "tool-settings.json").write_text("{}", encoding="utf-8")
    (run_dir / "skill-settings.json").write_text("{}", encoding="utf-8")
    spec_path = run_dir / "spec.json"
    if spec_change == "unreadable":
        spec_path.write_text("{invalid json", encoding="utf-8")
    elif spec_change == "missing-pin":
        spec_path.write_text(json.dumps({
            "repo": str(tmp_path / "repo"),
            "environment": {"STATEFRAME_HOME": str(tmp_path / "home")},
        }), encoding="utf-8")

    with pytest.raises(kernel_checkpoint.CheckpointError, match=_HOME_IDENTITY_REASON):
        kernel_checkpoint._snapshot_home_identity(run_dir)

    legacy_fixture = tmp_path / "legacy-direct-engine-fixture"
    legacy_fixture.mkdir()
    with pytest.raises(kernel_checkpoint.CheckpointError, match=_HOME_IDENTITY_REASON):
        kernel_checkpoint._snapshot_home_identity(legacy_fixture)
    monkeypatch.setenv("STATEFRAME_CHECKPOINT_LEGACY_DIRECT_ENGINE", "1")
    assert kernel_checkpoint._snapshot_home_identity(legacy_fixture) is None


def test_pool_capture_scans_private_bytes_before_atomic_publish(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from stateframe.engine import kernel_checkpoint

    run = tmp_path / "run"
    source = run / "wire" / "0001.request.json"
    source.parent.mkdir(parents=True)
    captured = b'{"model":"safe before capture"}'
    source.write_bytes(captured)
    secret = b"captured-identity-key-secret-2026"
    pool = tmp_path / "checkpoints" / kernel_checkpoint.POOL_DIR
    target = tmp_path / "checkpoint" / "run" / "wire"
    target.parent.mkdir(parents=True)
    monkeypatch.setattr(kernel_checkpoint, "_SECRETS", [])
    monkeypatch.setattr(kernel_checkpoint, "_SECRET_GENERATION", 0)
    monkeypatch.setattr(kernel_checkpoint, "_POOL_KNOWN", {})
    monkeypatch.setattr(kernel_checkpoint, "_VERIFIED_POOL_IDENTITIES", set())
    kernel_checkpoint.register_secret(secret)

    original_check = kernel_checkpoint._check_snapshot_file
    mutated = False

    def scan_then_mutate_live_source(path, *, digest_hint=None, cache_verified=True, force_scan=False):
        nonlocal mutated
        result = original_check(
            path, digest_hint=digest_hint, cache_verified=cache_verified, force_scan=force_scan)
        if not mutated:
            source.write_bytes(secret)
            mutated = True
        return result

    monkeypatch.setattr(kernel_checkpoint, "_check_snapshot_file", scan_then_mutate_live_source)
    manifests: dict[str, list] = {}
    kernel_checkpoint._snapshot_tree(source.parent, target, "wire", pool, manifests, [])

    pooled = pool / "wire" / source.name
    assert mutated, "the capture hook did not run between scanning and publication"
    assert pooled.read_bytes() == captured
    artifacts = [path for path in (tmp_path / "checkpoints").rglob("*") if path.is_file()]
    assert all(secret not in artifact.read_bytes() for artifact in artifacts)


@pytest.mark.parametrize("snapshotter", ["snapshot-tree", "copy-tree"])
def test_regular_snapshot_scans_private_bytes_before_atomic_publish(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, snapshotter: str) -> None:
    from stateframe.engine import kernel_checkpoint

    run = tmp_path / "run"
    source = run / "ordinary" / "payload.dat"
    source.parent.mkdir(parents=True)
    captured = b"safe regular-file contents"
    source.write_bytes(captured)
    secret = b"regular-capture-identity-key-secret-2026"
    checkpoint_root = tmp_path / "checkpoints"
    target = checkpoint_root / "run" / "ordinary"
    target.parent.mkdir(parents=True)
    monkeypatch.setattr(kernel_checkpoint, "_SECRETS", [])
    monkeypatch.setattr(kernel_checkpoint, "_SECRET_GENERATION", 0)
    kernel_checkpoint.register_secret(secret)

    original_check = kernel_checkpoint._check_snapshot_file
    mutated = False

    def scan_then_mutate_live_source(path, *, digest_hint=None, cache_verified=True, force_scan=False):
        nonlocal mutated
        result = original_check(
            path, digest_hint=digest_hint, cache_verified=cache_verified, force_scan=force_scan)
        if not mutated:
            source.write_bytes(secret)
            mutated = True
        return result

    monkeypatch.setattr(kernel_checkpoint, "_check_snapshot_file", scan_then_mutate_live_source)
    if snapshotter == "snapshot-tree":
        kernel_checkpoint._snapshot_tree(
            source.parent, target, "ordinary", checkpoint_root / kernel_checkpoint.POOL_DIR,
            {}, [])
    else:
        kernel_checkpoint._copy_tree(source.parent, target, None, set())

    captured_file = target / source.name
    assert mutated, "the capture hook did not run between scanning and publication"
    assert captured_file.read_bytes() == captured
    assert captured_file.stat().st_ino != source.stat().st_ino
    artifacts = [path for path in checkpoint_root.rglob("*") if path.is_file()]
    assert all(secret not in artifact.read_bytes() for artifact in artifacts)


@pytest.mark.parametrize("home_source", ["profile", "extra_env"])
def test_profile_and_extra_env_homes_allow_all_verified_exports(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, home_source: str) -> None:
    home, _secret = _isolated_home(tmp_path, monkeypatch)
    monkeypatch.delenv("STATEFRAME_HOME")
    repo = _repo(tmp_path / "repo")
    run_dir = tmp_path / "run"
    profile: Path | None = None
    extra_env: dict[str, str] = {}
    if home_source == "profile":
        switches = runner.profiles.load(runner.profiles.DEFAULT)
        switches["STATEFRAME_HOME"] = str(home)
        profile = tmp_path / "profile.json"
        profile.write_text(json.dumps({
            "schema": "stateframe-product-profile/1",
            "switches": switches,
        }), encoding="utf-8")
    else:
        extra_env["STATEFRAME_HOME"] = str(home)

    result = _scripted_real_run(
        repo, run_dir, lambda *_args: None, monkeypatch=monkeypatch,
        profile=profile, extra_env=extra_env,
    )
    assert (run_dir / "final.diff").is_file()
    assert not result.get("final_diff_error"), "final.diff export rejected the runner-pinned home"

    workspace = run_dir / "workspace"
    (workspace / "export-check.txt").write_text("verified export\n", encoding="utf-8")
    engine_dir = str(ROOT / "src" / "stateframe" / "engine")
    sys.path.insert(0, engine_dir)
    try:
        from stateframe.engine import kernel_checkpoint
        import kernel_git_commit as git_commits
        from kernel_effects import ContentStore, scan
        from kernel_git_view import excluded_path
        from stateframe.commit_export import CommitExportError, export_commits

        manifest = scan(workspace, exclude=lambda _path: False)
        committed = git_commits.commit(
            git_commits.CommitLog(run_dir), ContentStore(run_dir), manifest,
            workspace, "verified-home-export", "Add export fixture", excluded_path,
        )
        assert committed.startswith("commit ")
        assert export_commits(run_dir) is not None

        startup_identity = kernel_checkpoint.recorded_home_identity(
            json.loads((run_dir / "spec.json").read_text(encoding="utf-8")))
        replacement_home = tmp_path / "replacement-home"
        replacement_home.mkdir(mode=0o700)
        replacement_identity = auth_identity.protected_home_identity(
            replacement_home, initialize=True)
        spec_path = run_dir / "spec.json"
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        spec["environment"]["STATEFRAME_HOME"] = str(replacement_home)
        spec["environment"][runner.RUNNER_HOME_IDENTITY_ENV] = json.dumps(replacement_identity)
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        with pytest.raises(CommitExportError, match=_HOME_IDENTITY_REASON):
            export_commits(run_dir, startup_home_identity=startup_identity)
    finally:
        try:
            sys.path.remove(engine_dir)
        except ValueError:
            pass


def test_secret_scan_catches_identity_key_when_only_home_exclusion_is_disabled(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from stateframe.engine import kernel_checkpoint

    run_dir = tmp_path / "run"
    state_home = run_dir / "private" / "state" / "home"
    identity_key = state_home / "identity.key"
    identity_key.parent.mkdir(parents=True)
    identity_key.write_bytes(_TEST_IDENTITY_KEY)
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    monkeypatch.setattr(kernel_checkpoint, "_SECRETS", [])
    monkeypatch.setattr(kernel_checkpoint, "_SECRET_GENERATION", 0)
    monkeypatch.setattr(kernel_checkpoint, "_STATE", {"protected_home_identity": None})
    monkeypatch.setattr(kernel_checkpoint, "_VERIFIED_POOL_IDENTITIES", set())
    kernel_checkpoint.register_secret(_TEST_IDENTITY_KEY)

    # Disable only path exclusion. The private capture scan stays active.
    monkeypatch.setattr(kernel_checkpoint, "_snapshot_path_excluded",
                        lambda *_args, **_kwargs: False)
    with pytest.raises(kernel_checkpoint.CheckpointError, match="registered secret"):
        kernel_checkpoint._snapshot_tree(
            run_dir, checkpoint_root / "run", ".", checkpoint_root / kernel_checkpoint.POOL_DIR,
            {}, [], home=state_home, home_subtree=state_home, excluded=set(),
        )
    assert all(_TEST_IDENTITY_KEY not in path.read_bytes()
               for path in checkpoint_root.rglob("*") if path.is_file())


def test_planted_control_disabling_only_file_scan_exposes_run_payload(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from stateframe.engine import kernel_checkpoint

    secret = _TEST_IDENTITY_KEY
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "copied-payload.bin").write_bytes(secret)
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    monkeypatch.setattr(kernel_checkpoint, "_SECRETS", [])
    monkeypatch.setattr(kernel_checkpoint, "_SECRET_GENERATION", 0)
    monkeypatch.setattr(kernel_checkpoint, "_STATE", {"protected_home_identity": None})
    kernel_checkpoint.register_secret(secret)

    # Disable only the byte scan; the normal path-exclusion policy remains installed.
    monkeypatch.setattr(kernel_checkpoint, "_check_snapshot_file",
                        lambda _path, **_kwargs: (0, ""))
    kernel_checkpoint._snapshot_tree(
        run_dir, checkpoint_root / "run", ".", checkpoint_root / kernel_checkpoint.POOL_DIR,
        {}, [], home=tmp_path / "external-home", home_subtree=None, excluded=set(),
    )
    artifacts = [path for path in checkpoint_root.rglob("*") if path.is_file()]
    assert any(secret in artifact.read_bytes() for artifact in artifacts), (
        "disabling only the file scan did not expose the run payload"
    )


def test_export_refuses_patch_containing_registered_identity_key_bytes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _home, secret = _isolated_home(tmp_path, monkeypatch)
    repo = _repo(tmp_path / "repo")
    result = _scripted_real_run(repo, tmp_path / "run", lambda *_args: None,
                                monkeypatch=monkeypatch)
    run_dir = Path(str(result["run_dir"]))
    workspace = run_dir / "workspace"
    (workspace / "asset.txt").write_bytes(b"public\n" + secret + b"\n")

    engine_dir = str(ROOT / "src" / "stateframe" / "engine")
    sys.path.insert(0, engine_dir)
    try:
        import kernel_git_commit as git_commits
        from kernel_effects import ContentStore, scan
        from kernel_git_view import excluded_path
        from stateframe.commit_export import CommitExportError, export_commits

        manifest = scan(workspace, exclude=lambda _path: False)
        committed = git_commits.commit(
            git_commits.CommitLog(run_dir), ContentStore(run_dir), manifest,
            workspace, "isolated-key-export", "Add isolated test fixture", excluded_path,
        )
        assert committed.startswith("commit ")
        with pytest.raises(CommitExportError) as raised:
            export_commits(run_dir)
        if _SECRET_GUARD_REASON not in str(raised.value):
            pytest.fail("secret-bearing patch did not return KEY's fixed reason")
        assert not (run_dir / "commits").exists(), "a secret-bearing patch was published"
    finally:
        try:
            sys.path.remove(engine_dir)
        except ValueError:
            pass


@pytest.mark.parametrize("module_name", ["kernel_checkpoint", "engine.kernel_checkpoint", "stateframe.engine.kernel_checkpoint"])
def test_checkpoint_guard_state_is_never_collected_through_an_import_alias(
        monkeypatch: pytest.MonkeyPatch, module_name: str) -> None:
    from stateframe.engine import kernel_checkpoint

    monkeypatch.setattr(kernel_checkpoint, "_OWN_MODULE", "kernel_checkpoint")
    assert kernel_checkpoint._module_state(module_name, kernel_checkpoint) == {}


def test_explicit_legacy_mode_cannot_bypass_runner_startup_pin(tmp_path, monkeypatch):
    from stateframe.engine import kernel_checkpoint

    monkeypatch.setenv("STATEFRAME_CHECKPOINT_LEGACY_DIRECT_ENGINE", "1")
    monkeypatch.setattr(kernel_checkpoint, "_RUNNER_HOME_IDENTITY_CONFIGURED", True)
    monkeypatch.setattr(kernel_checkpoint, "_RUNNER_HOME_IDENTITY_PIN", None)
    with pytest.raises(kernel_checkpoint.CheckpointError, match=_HOME_IDENTITY_REASON):
        kernel_checkpoint._snapshot_home_identity(tmp_path)


def test_home_guard_runs_with_only_stdlib_and_engine_on_import_path(tmp_path):
    engine_dir = ROOT / "src" / "stateframe" / "engine"
    code = "\n".join([
        "import json, os, sys",
        "from pathlib import Path",
        f"sys.path.insert(0, {str(engine_dir)!r})",
        "import kernel_checkpoint as checkpoint",
        "from kernel_request_auth_identity import protected_home_identity",
        f"home, run, repo = [Path(p) for p in {list(map(str, (tmp_path / 'home', tmp_path / 'run', tmp_path / 'repo')))!r}]",
        "run.mkdir(); repo.mkdir()",
        "pin = protected_home_identity(home, initialize=True)",
        "(run / 'spec.json').write_text(json.dumps({'repo': str(repo), 'environment': {'STATEFRAME_HOME': str(home)}, 'stateframe_home_identity': pin}))",
        "checkpoint.configure_runner_home_identity_pin(json.dumps(pin))",
        "checkpoint.verify_stateframe_home_identity(run)",
        "assert (home / 'identity.key').read_bytes() in checkpoint._SECRETS",
        "checkpoint._snapshot_home_exclusions(run)",
        "assert 'stateframe' not in sys.modules",
    ])
    result = subprocess.run([sys.executable, "-I", "-S", "-c", code],
                            capture_output=True, text=True, timeout=scaled(30))
    assert result.returncode == 0, result.stderr


def test_child_environment_cannot_install_legacy_or_replace_startup_authority(monkeypatch):
    from stateframe import _child

    monkeypatch.setattr(_child.os, "environ", dict(os.environ))
    for name in ("STATEFRAME_CHECKPOINT_LEGACY_DIRECT_ENGINE", _child.RUNNER_HOME_IDENTITY_ENV):
        monkeypatch.setenv(name, "1")
    _child.install_environment({"environment": {
        "STATEFRAME_CHECKPOINT_LEGACY_DIRECT_ENGINE": "1",
        _child.RUNNER_HOME_IDENTITY_ENV: "untrusted spec pin",
    }})
    assert "STATEFRAME_CHECKPOINT_LEGACY_DIRECT_ENGINE" not in os.environ
    assert _child.RUNNER_HOME_IDENTITY_ENV not in os.environ


def test_identity_capture_refuses_key_replaced_during_read(tmp_path, monkeypatch):
    home, _secret = _isolated_home(tmp_path, monkeypatch)
    original_read = auth_identity._read_secret

    def read_then_replace(path):
        data = original_read(path)
        replacement = home / "replacement-key"
        replacement.write_bytes(b"abcdef0123456789abcdef0123456789")
        replacement.chmod(0o600)
        replacement.replace(path)
        return data

    monkeypatch.setattr(auth_identity, "_read_secret", read_then_replace)
    with pytest.raises(ValueError, match="changed during identity capture"):
        auth_identity.protected_home_identity(home)
