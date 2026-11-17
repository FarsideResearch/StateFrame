"""Snapshot-boundary symlink filtering and a planted negative control."""
from __future__ import annotations

import io
import json
import time
import tracemalloc
import zlib
from pathlib import Path

import pytest
from secret_scan_helpers import seven_layer_json

from stateframe.engine import kernel_checkpoint as checkpoint
from stateframe.engine.kernel_secret_scan import _JsonStringUnescaper

_TEST_SECRET = bytes(range(32))


_JSON_KEY = b"A" * 15 + b"\n" + b"B" * 16


def _register_json_key(monkeypatch):
    monkeypatch.setattr(checkpoint, "_SECRETS", [])
    monkeypatch.setattr(checkpoint, "_SECRET_GENERATION", 0)
    checkpoint.register_secret(_JSON_KEY)


def test_snapshot_refuses_json_escaped_split_newline_key(tmp_path, monkeypatch):
    _register_json_key(monkeypatch)
    run = tmp_path / "run"
    run.mkdir()
    payload = json.dumps({"key": _JSON_KEY.decode()}).encode()
    assert _JSON_KEY not in payload
    (run / "payload.json").write_bytes(payload)

    with pytest.raises(checkpoint.CheckpointError, match="registered secret"):
        checkpoint._copy_tree(run, tmp_path / "snapshot", None, set())
    assert not (tmp_path / "snapshot" / "payload.json").exists()


def test_index_chunk_refuses_json_escaped_split_newline_key(tmp_path, monkeypatch):
    _register_json_key(monkeypatch)
    monkeypatch.setattr(checkpoint, "_SAVE", {"root": tmp_path, "index": {}})
    payload = json.dumps({"key": _JSON_KEY.decode()})

    with pytest.raises(checkpoint.CheckpointError, match="registered secret"):
        checkpoint._chunk_list("escaped-key", [payload])
    assert not list(tmp_path.rglob("*"))


@pytest.mark.parametrize("split_prefix", [False, True])
@pytest.mark.parametrize("encoding", ["short", "unicode"])
def test_append_chunk_refuses_json_escaped_key_and_preserves_prior_chunks(
        tmp_path, monkeypatch, split_prefix, encoding):
    _register_json_key(monkeypatch)
    monkeypatch.setattr(checkpoint, "_SAVE", {})
    source = tmp_path / "source.jsonl"
    root = tmp_path / "checkpoints"
    root.mkdir()
    relative = "worker/measurement-runs/fixture/events.jsonl"
    payload = json.dumps({"key": _JSON_KEY.decode()}).encode()
    if encoding == "unicode":
        payload = ('{"key":"' + "".join(f"\\u{byte:04x}" for byte in _JSON_KEY) + '"}').encode()
    # A committed prefix may end halfway through the encoded key.
    split = len(payload) - 5 if split_prefix else 0
    source.write_bytes(payload[:split] if split_prefix else b'{"safe": true}\n')
    previous = {}
    assert checkpoint._snapshot_appended_file(
        source, tmp_path / "first", relative, root, [], previous, 1)
    committed = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    with source.open("ab") as stream:
        stream.write(payload[split:])

    with pytest.raises(checkpoint.CheckpointError, match="registered secret"):
        checkpoint._snapshot_appended_file(
            source, tmp_path / "second", relative, root, previous[relative], {}, 2)
    assert {path: path.read_bytes() for path in root.rglob("*") if path.is_file()} == committed
    assert not (tmp_path / "second").exists()


@pytest.mark.parametrize("kind", [
    "raw", "hex", "hex-upper", "short", "unicode", "nested", "surrogates",
])
def test_shared_guard_refuses_key_across_stream_read_boundary(tmp_path, monkeypatch, kind):
    secret = _JSON_KEY if kind != "surrogates" else ("A" * 14 + "\U0001f680" + "B" * 14).encode()
    monkeypatch.setattr(checkpoint, "_SECRETS", [])
    checkpoint.register_secret(secret)
    if kind == "raw":
        payload = secret
    elif kind.startswith("hex"):
        payload = secret.hex().encode()
        if kind == "hex-upper":
            payload = payload.upper()
    elif kind == "unicode":
        payload = ('"' + "".join(f"\\u{ord(char):04X}" for char in secret.decode()) + '"').encode()
    else:
        payload = json.dumps(secret.decode()).encode()
        if kind == "nested":
            for layer in range(7):
                # Escape every character, including backslashes from earlier layers.
                if layer < 3:
                    payload = ('"' + "".join(f"\\u{byte:04x}" for byte in payload) + '"').encode()
                else:
                    payload = json.dumps(payload.decode()).encode()
    assert len(payload) <= checkpoint._SECRET_JSON_MAX_BYTES
    # Put most of the encoded key in one read and its end in the next.
    cut = max(1, len(payload) - 5)
    prefix = b" " * ((1 << 20) - cut)
    source = tmp_path / "streamed-payload"
    source.write_bytes(prefix + payload)

    with pytest.raises(checkpoint.CheckpointError, match="registered secret"):
        checkpoint.check_registered_secret_file(source)


@pytest.mark.parametrize("input_mib", [1, 8])
def test_shared_guard_keeps_bounded_overlap_on_large_safe_input(monkeypatch, input_mib):
    _register_json_key(monkeypatch)
    # Allocate the caller's input before tracing: measure scanner work buffers,
    # including transient allocations that a retained-state check cannot see.
    unit = b'{"content":"ordinary text with an escape \\n"}\n' + b" " * 4096
    block = (unit * ((input_mib << 20) // len(unit) + 1))[:input_mib << 20]
    tracemalloc.start()
    try:
        tail = checkpoint._check_registered_secrets(block, final=False)
        checkpoint._check_registered_secrets(b"", tail)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert tail.retained_bytes <= 9 * 63 + 8 * 11
    assert peak < 1 << 20


def test_shared_guard_keeps_text_credentials_case_sensitive(monkeypatch):
    monkeypatch.setattr(checkpoint, "_SECRETS", [])
    checkpoint.register_secret("abcdef0123")
    checkpoint._check_registered_secrets(b'{"text": "ABCDEF0123"}')


def _copy_snapshot(run_dir: Path, target: Path) -> None:
    home, home_subtree, excluded = checkpoint._snapshot_home_exclusions(run_dir)
    checkpoint._copy_tree(
        run_dir, target, None, set(), home=home, home_subtree=home_subtree,
        excluded=excluded,
    )


def test_save_preserves_copy_hook_contract_and_home_filters_before_enospc(tmp_path, monkeypatch):
    import errno

    from stateframe.engine import kernel_request_auth_identity as auth

    run, home, repo = (tmp_path / name for name in ("run", "home", "repo"))
    workspace = run / "workspace"
    workspace.mkdir(parents=True)
    repo.mkdir()
    monkeypatch.setattr(checkpoint, "_SECRETS", [])
    monkeypatch.setattr(checkpoint, "_SECRET_GENERATION", 0)
    monkeypatch.setattr(checkpoint, "_VERIFIED_POOL_IDENTITIES", set())
    monkeypatch.setattr(checkpoint, "_POOL_KNOWN", {})
    monkeypatch.setattr(checkpoint, "_SAVE", {})
    monkeypatch.setattr(checkpoint, "_STATE", {})
    monkeypatch.setattr(checkpoint, "_PARTICIPANTS", {})
    monkeypatch.setattr(checkpoint, "_SIGNING_KEY", [b"copy-contract-signing-key-0123456789"])
    pin = auth.protected_home_identity(home, initialize=True)
    monkeypatch.setattr(checkpoint, "_RUNNER_HOME_IDENTITY_CONFIGURED", True)
    monkeypatch.setattr(checkpoint, "_RUNNER_HOME_IDENTITY_PIN", pin)
    monkeypatch.setenv("STATEFRAME_HOME", str(home))
    (run / "spec.json").write_text(json.dumps({
        "repo": str(repo), "environment": {"STATEFRAME_HOME": str(home)},
        "stateframe_home_identity": pin,
    }))
    (workspace / "value.py").write_text("VALUE = 1\n")
    (workspace / "installation").symlink_to(home, target_is_directory=True)
    context = {name: {} for name in ("hostfns", "participants", "framecells", "framefuncs", "cvars")}
    monkeypatch.setattr(checkpoint, "collect", lambda _frame: ({"frame": {}}, context))
    monkeypatch.setattr(checkpoint, "dump_to", lambda writer, *_args: writer.write(b"safe state"))
    copy_tree = checkpoint._copy_tree
    copied = []

    def copy_hook(source, target, previous, durable_inodes, *, seed=None):
        result = copy_tree(source, target, previous, durable_inodes, seed=seed)
        copied.append(target)
        assert (target / "value.py").read_text() == "VALUE = 1\n"
        assert not (target / "installation").is_symlink()
        return result

    entered = []

    def no_space(snapshot, *_args):
        entered.append(snapshot)
        raise OSError(errno.ENOSPC, "checkpoint storage full")

    monkeypatch.setattr(checkpoint, "_copy_tree", copy_hook)
    monkeypatch.setattr(checkpoint, "_fsync_snapshot", no_space)
    with pytest.raises(OSError) as error:
        checkpoint.save(object(), 1, run)
    assert error.value.errno == errno.ENOSPC
    assert copied and entered
    assert not (run / "checkpoints" / "LATEST").exists()
    assert checkpoint._SAVE == {}


def _assert_home_absent(snapshot: Path, relative_home: Path) -> None:
    captured_home = snapshot / relative_home
    captured_files = [
        path for path in captured_home.rglob("*")
        if path.is_file() or path.is_symlink()
    ] if captured_home.exists() else []
    if captured_files:
        raise AssertionError("StateFrame home was copied into the snapshot")


def _artifact_contents(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        stored = path.read_bytes()
        yield path, stored
        try:
            decoded = zlib.decompress(stored)
        except zlib.error:
            continue
        yield path, decoded


def test_snapshot_drops_run_symlink_into_external_stateframe_home(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    home = tmp_path / "stateframe-home"
    (run_dir / "workspace").mkdir(parents=True)
    home.mkdir()
    (run_dir / "workspace" / "ordinary.txt").write_text("safe", encoding="utf-8")
    (home / "identity.key").write_bytes(_TEST_SECRET)
    (run_dir / "home-link").symlink_to(home, target_is_directory=True)
    monkeypatch.setenv("STATEFRAME_HOME", str(home))

    snapshot = tmp_path / "snapshot"
    _copy_snapshot(run_dir, snapshot)

    assert (snapshot / "workspace" / "ordinary.txt").is_file()
    assert not (snapshot / "home-link").exists()
    assert not (snapshot / "home-link").is_symlink()


def test_snapshot_resolves_stateframe_home_symlink_before_filtering(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    home = run_dir / "private" / "state" / "home"
    (run_dir / "workspace").mkdir(parents=True)
    home.mkdir(parents=True)
    (run_dir / "workspace" / "ordinary.txt").write_text("safe", encoding="utf-8")
    (home / "identity.key").write_bytes(_TEST_SECRET)
    home_alias = tmp_path / "home-alias"
    home_alias.symlink_to(home, target_is_directory=True)
    monkeypatch.setenv("STATEFRAME_HOME", str(home_alias))

    snapshot = tmp_path / "snapshot"
    _copy_snapshot(run_dir, snapshot)

    assert (snapshot / "workspace" / "ordinary.txt").is_file()
    _assert_home_absent(snapshot, Path("private/state/home"))


def test_worker_snapshot_filters_nested_stateframe_home(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    home = run_dir / "worker" / "private" / "home"
    (run_dir / "worker" / "ordinary").mkdir(parents=True)
    home.mkdir(parents=True)
    (run_dir / "worker" / "ordinary" / "record.json").write_text("safe", encoding="utf-8")
    (home / "identity.key").write_bytes(_TEST_SECRET)
    monkeypatch.setenv("STATEFRAME_HOME", str(home))
    resolved_home, home_subtree, excluded = checkpoint._snapshot_home_exclusions(run_dir)
    assert resolved_home is not None

    snapshot = tmp_path / "snapshot"
    checkpoint._snapshot_tree(
        run_dir / "worker", snapshot, "worker", tmp_path / "pool", {}, [],
        home=resolved_home, home_subtree=home_subtree, excluded=excluded,
    )

    assert (snapshot / "ordinary" / "record.json").is_file()
    _assert_home_absent(snapshot, Path("private/home"))


def test_planted_control_fails_home_assertion_when_exclusion_is_disabled(
        tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    home = run_dir / "state"
    home.mkdir(parents=True)
    (home / "identity.key").write_bytes(_TEST_SECRET)
    monkeypatch.setenv("STATEFRAME_HOME", str(home))
    monkeypatch.setattr(
        checkpoint, "_snapshot_home_exclusions",
        lambda _run_dir: (home, None, set()),
    )

    snapshot = tmp_path / "planted-snapshot"
    _copy_snapshot(run_dir, snapshot)

    with pytest.raises(AssertionError, match="StateFrame home was copied"):
        _assert_home_absent(snapshot, Path("state"))


@pytest.mark.parametrize("payload", ["symlink-target", "filename"])
def test_snapshot_refuses_registered_key_in_filesystem_metadata(tmp_path, monkeypatch, payload):
    run = tmp_path / "run"
    run.mkdir()
    secret = b"metadata-identity-key-2026-fixture"
    monkeypatch.setattr(checkpoint, "_SECRETS", [])
    checkpoint.register_secret(secret)
    if payload == "symlink-target":
        (run / "asset").symlink_to("./" + secret.decode() + "/..")
    else:
        (run / secret.hex()).write_bytes(b"safe content")

    with pytest.raises(checkpoint.CheckpointError, match="registered secret"):
        checkpoint._copy_tree(run, tmp_path / "snapshot", None, set())
    snapshot = tmp_path / "snapshot"
    if snapshot.exists():
        assert not list(snapshot.rglob("*"))


def test_pooled_snapshot_refuses_registered_key_in_filename(tmp_path, monkeypatch):
    source = tmp_path / "wire"
    source.mkdir()
    secret = b"pooled-metadata-identity-key-2026"
    monkeypatch.setattr(checkpoint, "_SECRETS", [])
    checkpoint.register_secret(secret)
    (source / secret.hex()).write_bytes(b"safe content")
    with pytest.raises(checkpoint.CheckpointError, match="registered secret"):
        checkpoint._snapshot_tree(source, tmp_path / "snapshot", "wire", tmp_path / "pool", {}, [])
    assert not any(path.is_file() for path in (tmp_path / "pool").rglob("*"))


def test_symlink_capture_rechecks_target_after_boundary_filter(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    link = source / "asset"
    link.symlink_to("safe-target")
    secret = b"symlink-race-identity-key-fixture"
    monkeypatch.setattr(checkpoint, "_SECRETS", [])
    checkpoint.register_secret(secret)
    original = checkpoint._snapshot_path_excluded

    def filter_then_change(path, *args):
        result = original(path, *args)
        if path == link:
            link.unlink()
            link.symlink_to(secret.decode())
        return result

    monkeypatch.setattr(checkpoint, "_snapshot_path_excluded", filter_then_change)
    with pytest.raises(checkpoint.CheckpointError, match="registered secret"):
        checkpoint._copy_tree(source, tmp_path / "snapshot", None, set())
    assert not (tmp_path / "snapshot" / "asset").is_symlink()


@pytest.mark.parametrize("coarse_clock", [False, True])
def test_key_rewrite_before_publication_cannot_add_a_new_unscanned_secret(tmp_path, monkeypatch, coarse_clock):
    import json
    import types
    from detreason.world_knowledge import WorldKnowledgeStore
    from stateframe.engine import kernel_request_auth_identity as auth

    run, home, repo = (tmp_path / name for name in ("run", "home", "repo"))
    run.mkdir()
    repo.mkdir()
    monkeypatch.setattr(checkpoint, "_SECRETS", [])
    monkeypatch.setattr(checkpoint, "_SECRET_GENERATION", 0)
    monkeypatch.setattr(checkpoint, "_VERIFIED_POOL_IDENTITIES", set())
    pin = auth.protected_home_identity(home, initialize=True)
    if coarse_clock:
        original_identity = auth.protected_home_identity

        def coarse_identity(*args, **kwargs):
            current = original_identity(*args, **kwargs)
            if "key_st_ctime_ns" in pin:
                current["key_st_ctime_ns"] = pin["key_st_ctime_ns"]
            return current

        monkeypatch.setattr(auth, "protected_home_identity", coarse_identity)
    monkeypatch.setattr(checkpoint, "_RUNNER_HOME_IDENTITY_CONFIGURED", True)
    monkeypatch.setattr(checkpoint, "_RUNNER_HOME_IDENTITY_PIN", pin)
    monkeypatch.setattr(checkpoint, "_STATE", {})
    monkeypatch.setattr(checkpoint, "_SIGNING_KEY", [b"late-key-signing-fixture-0123456789"])
    monkeypatch.setenv("STATEFRAME_HOME", str(home))
    (run / "spec.json").write_text(json.dumps({
        "repo": str(repo), "environment": {"STATEFRAME_HOME": str(home)},
        "stateframe_home_identity": pin,
    }))
    replacement = bytes(range(32))
    (run / "wire").mkdir()
    (run / "wire" / "copied-key.bin").write_bytes(replacement)
    journal = run / "worker" / "measurement-runs" / "snapshot" / "events.jsonl"
    journal.parent.mkdir(parents=True)
    journal.write_bytes(replacement)
    index = WorldKnowledgeStore({"sandbox_id": "checkpoint-home-snapshot"})
    index._emit({"event_id": "key-bearing-index-event", "kind": "profile_sample",
                 "detail": replacement})
    context_names = ("hostfns", "participants", "framecells", "framefuncs", "cvars")
    context = {name: {} for name in context_names}
    monkeypatch.setattr(checkpoint, "collect", lambda _frame: (
        {"frame": {}, "index": index}, context))
    monkeypatch.setattr(checkpoint, "_PARTICIPANTS", {})
    original_inode = (home / "identity.key").stat().st_ino

    def rewrite_key_at_publication(snapshot_root, staged_pool_files, _durable_inodes):
        artifact_root = snapshot_root / checkpoint.ARTIFACT_STAGE_DIR
        staged_pool = list((artifact_root / checkpoint.POOL_DIR).rglob("*"))
        staged_pool = [path for path in staged_pool if path.is_file()]
        staged_chunks = list((artifact_root / checkpoint.CACHE_DIR).glob("*"))
        assert any(replacement in path.read_bytes() for path in staged_pool)
        assert any(path in staged_pool for path in staged_pool_files)
        decoded_chunks = []
        for path in staged_chunks:
            try:
                decoded_chunks.append((path.name, zlib.decompress(path.read_bytes())))
            except zlib.error:
                pass
        assert any(name.startswith("file-") and replacement in data
                   for name, data in decoded_chunks)
        assert any(not name.startswith("file-") and replacement in data
                   for name, data in decoded_chunks)
        (home / "identity.key").write_bytes(replacement)
        assert (home / "identity.key").stat().st_ino == original_inode

    monkeypatch.setattr(checkpoint, "_fsync_snapshot", rewrite_key_at_publication)
    with pytest.raises(checkpoint.CheckpointError, match="STATEFRAME_HOME_IDENTITY_CHANGED"):
        checkpoint.save(types.SimpleNamespace(), 1, run)
    assert (home / "identity.key").read_bytes() == replacement
    assert not (run / "checkpoints" / "LATEST").exists()
    assert all(replacement not in data for _path, data in
               _artifact_contents(run / "checkpoints"))


def test_capture_failure_discards_staged_pool_files(tmp_path, monkeypatch):
    import json
    import types
    from stateframe.engine import kernel_request_auth_identity as auth

    run, home, repo = (tmp_path / name for name in ("run", "home", "repo"))
    run.mkdir()
    repo.mkdir()
    monkeypatch.setattr(checkpoint, "_SECRETS", [])
    monkeypatch.setattr(checkpoint, "_SECRET_GENERATION", 0)
    monkeypatch.setattr(checkpoint, "_VERIFIED_POOL_IDENTITIES", set())
    monkeypatch.setattr(checkpoint, "_POOL_KNOWN", {})
    pin = auth.protected_home_identity(home, initialize=True)
    monkeypatch.setattr(checkpoint, "_RUNNER_HOME_IDENTITY_CONFIGURED", True)
    monkeypatch.setattr(checkpoint, "_RUNNER_HOME_IDENTITY_PIN", pin)
    monkeypatch.setattr(checkpoint, "_STATE", {})
    monkeypatch.setattr(checkpoint, "_SIGNING_KEY", [b"capture-test-signing-key-0123456789"])
    monkeypatch.setenv("STATEFRAME_HOME", str(home))
    (run / "spec.json").write_text(json.dumps({
        "repo": str(repo), "environment": {"STATEFRAME_HOME": str(home)},
        "stateframe_home_identity": pin,
    }))
    replacement = bytes(range(32))
    (run / "wire").mkdir()
    (run / "wire" / "copied-key.bin").write_bytes(replacement)
    context_names = ("hostfns", "participants", "framecells", "framefuncs", "cvars")
    context = {name: {} for name in context_names}
    monkeypatch.setattr(checkpoint, "collect", lambda _frame: ({"frame": {}}, context))
    monkeypatch.setattr(checkpoint, "_PARTICIPANTS", {})
    original_snapshot_tree = checkpoint._snapshot_tree

    def capture_then_fail(source, target, relative, pool, manifests, new_pool_files,
                          file_chunks=None, previous_files=None, seq=0, **kwargs):
        original_snapshot_tree(source, target, relative, pool, manifests, new_pool_files,
                               file_chunks, previous_files, seq, **kwargs)
        if relative == "wire":
            assert any(replacement in path.read_bytes() for path in new_pool_files)
            raise checkpoint.CheckpointError("injected checkpoint capture failure")

    monkeypatch.setattr(checkpoint, "_snapshot_tree", capture_then_fail)
    with pytest.raises(checkpoint.CheckpointError, match="injected checkpoint capture failure"):
        checkpoint.save(types.SimpleNamespace(), 1, run)

    root = run / "checkpoints"
    pool = root / checkpoint.POOL_DIR
    assert not pool.exists() or not any(path.is_file() for path in pool.rglob("*"))
    assert all(replacement not in data for _path, data in _artifact_contents(root))


@pytest.mark.parametrize("background", [False, True])
def test_failed_artifact_publication_rolls_back_partial_targets(tmp_path, monkeypatch, background):
    import json
    import types
    from stateframe.engine import kernel_request_auth_identity as auth

    run, home, repo = (tmp_path / name for name in ("run", "home", "repo"))
    run.mkdir()
    repo.mkdir()
    monkeypatch.setattr(checkpoint, "_SECRETS", [])
    monkeypatch.setattr(checkpoint, "_SECRET_GENERATION", 0)
    monkeypatch.setattr(checkpoint, "_VERIFIED_POOL_IDENTITIES", set())
    monkeypatch.setattr(checkpoint, "_POOL_KNOWN", {})
    pin = auth.protected_home_identity(home, initialize=True)
    monkeypatch.setattr(checkpoint, "_RUNNER_HOME_IDENTITY_CONFIGURED", True)
    monkeypatch.setattr(checkpoint, "_RUNNER_HOME_IDENTITY_PIN", pin)
    monkeypatch.setattr(checkpoint, "_STATE", {
        "failed": 0, "consecutive_failed": 0, "disabled": False,
    })
    monkeypatch.setattr(checkpoint, "_SIGNING_KEY", [b"publication-test-signing-key-0123456789"])
    monkeypatch.setenv("STATEFRAME_HOME", str(home))
    (run / "spec.json").write_text(json.dumps({
        "repo": str(repo), "environment": {"STATEFRAME_HOME": str(home)},
        "stateframe_home_identity": pin,
    }))
    replacement = bytes(range(32))
    (run / "wire").mkdir()
    (run / "wire" / "copied-key.bin").write_bytes(replacement)
    journal = run / "worker" / "measurement-runs" / "snapshot" / "events.jsonl"
    journal.parent.mkdir(parents=True)
    journal.write_bytes(replacement)
    context_names = ("hostfns", "participants", "framecells", "framefuncs", "cvars")
    context = {name: {} for name in context_names}
    monkeypatch.setattr(checkpoint, "collect", lambda _frame: ({"frame": {}}, context))
    monkeypatch.setattr(checkpoint, "_PARTICIPANTS", {})

    real_link = checkpoint.os.link
    link_count = 0
    failure_marker = tmp_path / "publication-failure-hit"

    def fail_after_first_link(source, target, *args, **kwargs):
        nonlocal link_count
        link_count += 1
        if link_count == 2:
            failure_marker.write_text("hit")
            raise OSError("injected checkpoint artifact publication failure")
        return real_link(source, target, *args, **kwargs)

    monkeypatch.setattr(checkpoint.os, "link", fail_after_first_link)
    if background:
        checkpoint.save(types.SimpleNamespace(), 1, run, background=True)
        deadline = time.monotonic() + 5
        while not checkpoint._reap() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert checkpoint._STATE.get("child") is None
        assert checkpoint.failure_summary()["checkpoint_failures"] == 1
        assert failure_marker.is_file()
    else:
        with pytest.raises(OSError, match="injected checkpoint artifact publication failure"):
            checkpoint.save(types.SimpleNamespace(), 1, run)
        assert link_count == 2

    assert checkpoint.latest(run) is None
    assert all(replacement not in data for _path, data in _artifact_contents(run / "checkpoints"))


def test_stale_transaction_cleanup_keeps_artifacts_referenced_by_checkpoints(tmp_path, monkeypatch):
    import json
    import os

    root = tmp_path / "checkpoints"
    checkpoint_folder = root / "0001-kept"
    checkpoint_folder.mkdir(parents=True)
    (checkpoint_folder / checkpoint.META_FILE).write_text(json.dumps(
        {"seq": 1, "chunks": ["kept.zpkl"]}))
    pool_artifact = root / checkpoint.POOL_DIR / "wire" / "copied-key.bin"
    pool_artifact.parent.mkdir(parents=True)
    kept_chunk = root / checkpoint.CACHE_DIR / "kept.zpkl"
    kept_chunk.parent.mkdir(parents=True)

    dead_pid = os.getpid() + 1_000_000
    tmp = root / f".0002-{dead_pid}.tmp"
    tmp.mkdir()
    staged_pool = (tmp / checkpoint.ARTIFACT_STAGE_DIR / checkpoint.POOL_DIR
                   / "wire" / "copied-key.bin")
    staged_chunk = tmp / checkpoint.ARTIFACT_STAGE_DIR / checkpoint.CACHE_DIR / "kept.zpkl"
    staged_pool.parent.mkdir(parents=True)
    staged_chunk.parent.mkdir(parents=True)
    staged_pool.write_bytes(_TEST_SECRET)
    staged_chunk.write_bytes(b"referenced")
    os.link(staged_pool, pool_artifact)
    os.link(staged_chunk, kept_chunk)
    checkpoint._write_artifact_transaction(tmp, [
        f"{checkpoint.POOL_DIR}/wire/copied-key.bin",
        f"{checkpoint.CACHE_DIR}/kept.zpkl",
    ])
    monkeypatch.setattr(checkpoint, "_pid_alive", lambda _pid: False)

    assert checkpoint._sweep_stale(root) == 1
    assert not tmp.exists()
    assert not pool_artifact.exists()
    assert kept_chunk.read_bytes() == b"referenced"


@pytest.mark.parametrize("mutation", ["escaped-secret", "safe-tampering"])
@pytest.mark.parametrize("boundary", ["link", "staging-removal"])
def test_artifact_publication_rescans_bytes_after_inode_mutation(
        tmp_path, monkeypatch, mutation, boundary):
    import os

    root = tmp_path / "checkpoints"
    pending = root / ".0001-test.tmp"
    staged_root = pending / checkpoint.ARTIFACT_STAGE_DIR
    source = tmp_path / "source.json"
    unsafe = json.dumps({"key": _JSON_KEY.decode()}).encode()
    source.write_bytes(b" " * len(unsafe))
    target = root / checkpoint.POOL_DIR / "wire" / "0001.request.json"
    _register_json_key(monkeypatch)
    monkeypatch.setattr(checkpoint, "_STATE", {})
    monkeypatch.setattr(checkpoint, "_VERIFIED_POOL_IDENTITIES", set())
    monkeypatch.setattr(checkpoint, "_POOL_KNOWN", {})
    monkeypatch.setattr(checkpoint, "_SAVE", {
        "artifact_root": staged_root, "pool_root": root / checkpoint.POOL_DIR,
    })
    created = []
    identity, _ = checkpoint._capture_snapshot_file(source, target, new_pool_files=created)
    checkpoint._POOL_KNOWN[str(target.parent)] = {target.name: identity}
    staged = created[0]
    assert checkpoint._snapshot_file_cache_key(
        staged.stat(), checkpoint._SECRET_GENERATION, identity[1]
    ) in checkpoint._VERIFIED_POOL_IDENTITIES
    real_link = checkpoint.os.link
    real_rmtree = checkpoint.shutil.rmtree
    mutations = []

    def mutate(path):
        before = path.stat()
        # Pooled files are read-only by design; this tamperer bypasses that with chmod.
        path.chmod(0o600)
        path.write_bytes(unsafe if mutation == "escaped-secret" else b"x" * len(unsafe))
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        mutations.append(path)

    def link_then_mutate(src, dst, *args, **kwargs):
        real_link(src, dst, *args, **kwargs)
        if boundary == "link":
            mutate(Path(dst))

    def remove_then_mutate(path, *args, **kwargs):
        real_rmtree(path, *args, **kwargs)
        if Path(path) == staged_root and boundary == "staging-removal":
            mutate(target)

    monkeypatch.setattr(checkpoint.os, "link", link_then_mutate)
    monkeypatch.setattr(checkpoint.shutil, "rmtree", remove_then_mutate)
    with pytest.raises(checkpoint.CheckpointError):
        checkpoint._publish_staged_artifacts(root, pending)

    assert mutations == [target]
    assert checkpoint._snapshot_file_cache_key(
        target.stat(), checkpoint._SECRET_GENERATION, identity[1]
    ) not in checkpoint._VERIFIED_POOL_IDENTITIES
    with pytest.raises(checkpoint.CheckpointError):
        checkpoint._check_snapshot_file(target, digest_hint=identity[1])
    checkpoint._cleanup_checkpoint_transaction(root, pending)
    assert not target.exists()


def test_snapshot_scan_refuses_path_replacement_while_fd_is_open(tmp_path, monkeypatch):
    import os

    source = tmp_path / "source"
    source.write_bytes(b"safe")
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"evil")
    monkeypatch.setattr(checkpoint, "_STATE", {})
    monkeypatch.setattr(checkpoint, "_SECRETS", [])
    monkeypatch.setattr(checkpoint, "_VERIFIED_POOL_IDENTITIES", set())
    original_read = checkpoint.os.read
    replaced = []

    def read_then_replace(fd, size):
        data = original_read(fd, size)
        if data and not replaced:
            os.replace(replacement, source)
            replaced.append(source)
        return data

    monkeypatch.setattr(checkpoint.os, "read", read_then_replace)
    with pytest.raises(checkpoint.CheckpointError, match="changed during checkpoint capture"):
        checkpoint._check_snapshot_file(source)
    assert replaced == [source]
    assert not checkpoint._VERIFIED_POOL_IDENTITIES


@pytest.mark.parametrize("failure", ["read-error", "invalid-json", "not-object", "missing"])
def test_rollback_preserves_all_artifacts_when_checkpoint_metadata_is_uncertain(
        tmp_path, monkeypatch, failure):
    import os

    root = tmp_path / "checkpoints"
    pending = root / ".0002-test.tmp"
    committed = root / "0001-kept"
    committed.mkdir(parents=True)
    metadata = committed / checkpoint.META_FILE
    metadata.write_text(json.dumps({"seq": 1, "chunks": []}))
    (committed / checkpoint.MANIFEST_FILE).write_text(json.dumps({
        "wire": [["0001.request.json", 4]],
    }))
    relatives = [f"{checkpoint.POOL_DIR}/wire/0001.request.json",
                 f"{checkpoint.CACHE_DIR}/unreferenced.zpkl"]
    for relative in relatives:
        staged = pending / checkpoint.ARTIFACT_STAGE_DIR / relative
        target = root / relative
        staged.parent.mkdir(parents=True, exist_ok=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        staged.write_bytes(b"safe")
        os.link(staged, target)
    checkpoint._write_artifact_transaction(pending, relatives)
    monkeypatch.setattr(checkpoint, "_VERIFIED_POOL_IDENTITIES", set())
    monkeypatch.setattr(checkpoint, "_POOL_KNOWN", {})
    original_read = Path.read_text
    failed_reads = []

    def transient_metadata_failure(path, *args, **kwargs):
        if path == metadata and not failed_reads:
            failed_reads.append(path)
            if failure == "read-error":
                raise OSError("injected transient metadata read failure")
            if failure == "missing":
                raise FileNotFoundError("injected transient metadata disappearance")
            return "{" if failure == "invalid-json" else "[]"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", transient_metadata_failure)
    checkpoint._cleanup_checkpoint_transaction(root, pending)

    assert failed_reads == [metadata]
    assert json.loads(metadata.read_text())["seq"] == 1
    assert not pending.exists()
    for relative in relatives:
        assert (root / relative).read_bytes() == b"safe"
    assert checkpoint._checkpoint_artifact_references(root) == {relatives[0]}


def test_legacy_pin_opt_in_still_guards_an_existing_installation_key(tmp_path, monkeypatch):
    run, home = tmp_path / "run", tmp_path / "home"
    run.mkdir()
    home.mkdir()
    key = bytes(range(32))
    (home / "identity.key").write_bytes(key)
    (home / "identity.key").chmod(0o600)
    (run / "copied-key.bin").write_bytes(key)
    monkeypatch.setenv("STATEFRAME_HOME", str(home))
    monkeypatch.setenv("STATEFRAME_CHECKPOINT_LEGACY_DIRECT_ENGINE", "1")
    monkeypatch.setattr(checkpoint, "_RUNNER_HOME_IDENTITY_CONFIGURED", False)
    monkeypatch.setattr(checkpoint, "_SECRETS", [])
    assert checkpoint._snapshot_home_identity(run) is None
    assert key in checkpoint._SECRETS
    with pytest.raises(checkpoint.CheckpointError, match="registered secret"):
        checkpoint._copy_tree(run, tmp_path / "snapshot", None, set())


@pytest.mark.parametrize("boundary", ["state-writer", "snapshot", "index"])
def test_checkpoint_refuses_seven_unicode_layers_beyond_old_window(
        tmp_path, monkeypatch, boundary):
    _register_json_key(monkeypatch)
    payload = seven_layer_json(_JSON_KEY)
    assert len(payload) > 9_000_000
    decoded = payload.decode()
    for _ in range(7):
        decoded = json.loads(decoded)
    assert decoded.encode() == _JSON_KEY

    with pytest.raises(checkpoint.CheckpointError, match="registered secret"):
        if boundary == "state-writer":
            writer = checkpoint._StateWriter(io.BytesIO())
            for offset in range(0, len(payload), 8191):
                writer.write(payload[offset:offset + 8191])
            writer.finish()
        elif boundary == "snapshot":
            run = tmp_path / "run"
            run.mkdir()
            (run / "payload.json").write_bytes(payload)
            checkpoint._copy_tree(run, tmp_path / "snapshot", None, set())
        else:
            monkeypatch.setattr(checkpoint, "_SAVE", {"root": tmp_path, "index": {}})
            checkpoint._chunk_list("deeply-escaped-key", [payload.decode()])
    if boundary == "snapshot":
        assert not (tmp_path / "snapshot" / "payload.json").exists()
    if boundary == "index":
        assert not list(tmp_path.rglob("*"))


def test_append_refuses_seven_unicode_layers_split_across_committed_prefix(
        tmp_path, monkeypatch):
    _register_json_key(monkeypatch)
    monkeypatch.setattr(checkpoint, "_SAVE", {})
    payload = seven_layer_json(_JSON_KEY)
    cut = len(payload) // 2
    source = tmp_path / "source.jsonl"
    source.write_bytes(payload[:cut])
    root = tmp_path / "checkpoints"
    root.mkdir()
    relative = "worker/measurement-runs/fixture/events.jsonl"
    previous = {}
    assert checkpoint._snapshot_appended_file(
        source, tmp_path / "first", relative, root, [], previous, 1)
    committed = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    with source.open("ab") as stream:
        stream.write(payload[cut:])

    with pytest.raises(checkpoint.CheckpointError, match="registered secret"):
        checkpoint._snapshot_appended_file(
            source, tmp_path / "second", relative, root, previous[relative], {}, 2)
    assert {path: path.read_bytes() for path in root.rglob("*") if path.is_file()} == committed
    assert not (tmp_path / "second").exists()


@pytest.mark.parametrize("value", [
    'prefix"\\/\b\f\n\r\tsuffix', "prefix\u263asuffix", "prefix\U0001f680suffix",
    "prefix\ud800suffix", pytest.param("prefix" + chr(0xDC00) + "suffix", id="lone-low"),
    "prefix\ud800", "prefix\ud800\u263a",
])
@pytest.mark.parametrize("layers", [1, 2, 8])
def test_streaming_guard_handles_every_escape_split(monkeypatch, value, layers):
    secret = value.encode("utf-8", errors="surrogatepass")
    monkeypatch.setattr(checkpoint, "_SECRETS", [])
    checkpoint.register_secret(secret)
    payload = value
    for _ in range(layers):
        payload = json.dumps(payload).replace("/", "\\/")
    encoded = payload.encode()
    # At eight layers, byte-at-a-time input exercises every boundary together
    # without making this regression quadratic in the escaped fixture size.
    if layers == 8:
        with pytest.raises(checkpoint.CheckpointError, match="registered secret"):
            scanner = None
            for byte in encoded:
                scanner = checkpoint._check_registered_secrets(bytes([byte]), scanner, final=False)
            checkpoint._check_registered_secrets(b"", scanner)
        return
    # Every two-chunk cut includes both halves of each short/Unicode escape and
    # each surrogate pair, including endpoints and final flushing.
    for cut in range(len(encoded) + 1):
        with pytest.raises(checkpoint.CheckpointError, match="registered secret"):
            scanner = checkpoint._check_registered_secrets(encoded[:cut], final=False)
            checkpoint._check_registered_secrets(encoded[cut:], scanner)


@pytest.mark.parametrize("value", [
    '"\\/\b\f\n\r\t', "\U0001f680", "\ud800", pytest.param(chr(0xDC00), id="lone-low"),
    "\ud800x", "\ud800\ud800",
])
def test_incremental_unescaper_matches_json_at_every_split(value):
    encoded = json.dumps(value).encode()[1:-1].replace(b"/", b"\\/")
    expected = value.encode("utf-8", errors="surrogatepass")
    for cut in range(len(encoded) + 1):
        stage = _JsonStringUnescaper()
        actual = stage.feed(encoded[:cut]) + stage.feed(encoded[cut:]) + stage.feed(b"", final=True)
        assert actual == expected
        assert stage.pending == b""
    stage = _JsonStringUnescaper()
    assert (b"".join(stage.feed(bytes([byte])) for byte in encoded)
            + stage.feed(b"", final=True)) == expected


def test_streaming_guard_flushes_lone_surrogate_at_actual_eof(monkeypatch):
    secret = ("prefix" + "\ud800").encode("utf-8", errors="surrogatepass")
    monkeypatch.setattr(checkpoint, "_SECRETS", [])
    checkpoint.register_secret(secret)
    scanner = checkpoint._check_registered_secrets(b"prefix\\ud800", final=False)
    with pytest.raises(checkpoint.CheckpointError, match="registered secret"):
        checkpoint._check_registered_secrets(b"", scanner)
    writer = checkpoint._StateWriter(io.BytesIO())
    writer.write(b"prefix\\ud800")
    with pytest.raises(checkpoint.CheckpointError, match="registered secret"):
        writer.finish()


def test_streaming_guard_scans_fifty_mib_benign_file_quickly(tmp_path, monkeypatch):
    _register_json_key(monkeypatch)
    source = tmp_path / "large-benign.jsonl"
    block = b'{"content":"ordinary text without credentials"}\n' * 1000
    remaining = 50 << 20
    with source.open("wb") as stream:
        while remaining:
            piece = block[:remaining]
            stream.write(piece)
            remaining -= len(piece)
    started = time.monotonic()
    checkpoint.check_registered_secret_file(source)
    elapsed = time.monotonic() - started
    assert elapsed < 15, f"50 MiB streaming secret scan took {elapsed:.2f}s"


@pytest.mark.parametrize("encoded,expected", [
    (b"\\q", b"\\q"), (b"\\u0_41", b"\\u0_41"), (b"\\u+041", b"\\u+041"),
    (b"\\u12", b"\\u12"), (b"tail\\", b"tail\\"),
    (b"\\ud800\\u00xz", chr(0xD800).encode("utf-8", errors="surrogatepass") + b"\\u00xz"),
])
def test_incremental_unescaper_preserves_invalid_and_truncated_escapes(encoded, expected):
    for cut in range(len(encoded) + 1):
        stage = _JsonStringUnescaper()
        first = stage.feed(encoded[:cut])
        assert len(stage.pending) <= 11
        second = stage.feed(encoded[cut:])
        assert len(stage.pending) <= 11
        assert first + second + stage.feed(b"", final=True) == expected
