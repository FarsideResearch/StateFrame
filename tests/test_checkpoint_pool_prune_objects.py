"""Pool pruning must ignore the shared object store and fail closed on bad checkpoints."""
from __future__ import annotations

import ast
import fcntl
import io
import json
import os
import pickle
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from fixtures.process_signals import record_identity, signal_run_verified
from stateframe.engine import kernel_checkpoint as checkpoint
from stateframe.engine import kernel_reason_log as reason_log


@pytest.fixture(autouse=True)
def _synthetic_checkpoint_environment(monkeypatch):
    monkeypatch.setenv(checkpoint.LEGACY_DIRECT_ENGINE_ENV, "1")
    monkeypatch.setenv(checkpoint._PUBLICATION_SWITCH, "off")
    monkeypatch.setenv(checkpoint.KEEP_ENV, "2")
    monkeypatch.setattr(checkpoint, "_SIGNING_KEY", [b"pool-prune-test-signing-key-012345"])
    monkeypatch.setattr(checkpoint, "_PRODUCT_HOST", True)
    monkeypatch.setattr(checkpoint, "_HOST_IDENTITY_BOUND", False)
    monkeypatch.setattr(checkpoint, "_FINGERPRINT", None)
    monkeypatch.setattr(checkpoint, "_POOL_KNOWN", {})
    monkeypatch.setattr(checkpoint, "_DIGEST_MEMO", {})
    context = {name: {} for name in ("hostfns", "participants", "framecells", "framefuncs", "cvars")}
    state = {"frame": {}, "modules": {}, "classes": {}, "patched": {}, "participants": {},
             "context": [], "random": random.getstate()}
    monkeypatch.setattr(checkpoint, "collect", lambda _frame: (dict(state), context))
    monkeypatch.setattr(checkpoint, "dump_to",
                        lambda stream, payload, _ctx, protocol=5: pickle.dump(payload, stream))


def _save(run: Path, step: int, *, background: bool = False) -> Path:
    checkpoint._STATE["setup"] = {"code": sys._getframe().f_code, "line": 0, "funcs": frozenset()}
    return checkpoint.save(sys._getframe(), step, run, background=background)


def _pool_path(run: Path, name: str) -> Path:
    return run / "checkpoints" / checkpoint.POOL_DIR / "worker__candidates" / name


def test_save_prunes_orphans_but_keeps_checkpoint_and_live_files(tmp_path, monkeypatch):
    monkeypatch.setenv(checkpoint.KEEP_ENV, "1")
    run = tmp_path / "run"
    candidates = run / "worker" / "candidates"
    candidates.mkdir(parents=True)
    (run / "payload.bin").write_bytes(b"content object")
    orphan = candidates / "orphan.bin"
    kept = candidates / "kept.bin"
    orphan.write_bytes(b"orphan")
    kept.write_bytes(b"kept")

    first = _save(run, 1)
    assert (run / "checkpoints" / checkpoint.OBJECT_DIR).is_dir()
    orphan.unlink()
    second = _save(run, 2)

    assert checkpoint.latest(run) == second
    assert not _pool_path(run, orphan.name).exists()
    assert _pool_path(run, kept.name).is_file()
    manifest = json.loads((second / checkpoint.MANIFEST_FILE).read_text())
    assert kept.name in {row[0] for row in manifest["worker/candidates"]}
    assert first.is_dir() is False  # keep=1 removed the only older checkpoint

    live_only = "live-only.bin"
    (candidates / live_only).write_bytes(b"written since the save")
    _pool_path(run, live_only).write_bytes(b"pool copy")
    checkpoint._prune_pool(run / "checkpoints", run)
    assert _pool_path(run, live_only).is_file()


def test_prune_memory_error_is_logged_without_replacing_rejection(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.setattr(checkpoint, "_STATE", {})
    monkeypatch.setattr(checkpoint, "_SAVE", {})
    prune_roots = []

    def fail_prune(root):
        prune_roots.append(root)
        raise MemoryError("simulated allocation failure")

    def reject_publication(*_args, **_kwargs):
        raise checkpoint.CheckpointError("original publication rejection")

    monkeypatch.setattr(checkpoint, "_prune_objects", fail_prune)
    monkeypatch.setattr(checkpoint, "_publish_staged_artifacts", reject_publication)
    reason_log.reset_for_tests()
    output = io.StringIO()
    reason_log.configure(level="WARN", stream=output)
    try:
        with pytest.raises(checkpoint.CheckpointError, match="original publication rejection"):
            _save(run, 1)

        assert prune_roots == [run / "checkpoints", run / "checkpoints"]
        logged = reason_log.run_summary()["by_cause"][
            "fallback:checkpoint_object_prune_failed"]
        assert logged["component"] == "kernel_checkpoint"
        assert logged["why"] == "transaction cleanup continued without object pruning"
        assert output.getvalue().count("checkpoint object pruning failed") == 1
    finally:
        reason_log.reset_for_tests()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="production writer cleanup requires fork")
def test_every_production_transaction_cleanup_holds_checkpoint_lock(tmp_path, monkeypatch):
    source = Path(checkpoint.__file__)
    module = ast.parse(source.read_text(encoding="utf-8"))
    production_sites = set()

    class CleanupCallVisitor(ast.NodeVisitor):
        def __init__(self):
            self.scopes = []

        def visit_FunctionDef(self, node):
            self.scopes.append(node.name)
            self.generic_visit(node)
            self.scopes.pop()

        def visit_AsyncFunctionDef(self, node):
            self.scopes.append(node.name)
            self.generic_visit(node)
            self.scopes.pop()

        def visit_Call(self, node):
            if (isinstance(node.func, ast.Name)
                    and node.func.id == "_cleanup_checkpoint_transaction"):
                assert self.scopes, "production cleanup call must be inside a function"
                production_sites.add((self.scopes[-1], node.lineno))
            self.generic_visit(node)

    CleanupCallVisitor().visit(module)
    assert production_sites
    assert {caller for caller, _line in production_sites} == {
        "save", "_save_into", "_reap", "_sweep_stale_locked", "_fork_writer",
    }
    save_sites = {site for site in production_sites if site[0] == "save"}
    assert len(save_sites) == 2, "cover both save cleanup branches, including reap leftovers"

    events_file = tmp_path / "cleanup-calls.jsonl"
    real_cleanup = checkpoint._cleanup_checkpoint_transaction

    def observe_cleanup(root, tmp):
        caller = sys._getframe(1)
        with (root / ".checkpoint.lock").open("a+b") as probe:
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock_held = True
            else:
                fcntl.flock(probe, fcntl.LOCK_UN)
                lock_held = False
        event = {"caller": caller.f_code.co_name, "line": caller.f_lineno,
                 "pid": os.getpid(), "lock_held": lock_held}
        with events_file.open("ab") as stream:
            stream.write(json.dumps(event).encode("utf-8") + b"\n")
        real_cleanup(root, tmp)

    monkeypatch.setattr(checkpoint, "_cleanup_checkpoint_transaction", observe_cleanup)
    monkeypatch.setattr(checkpoint, "_STATE", {
        "failed": 0, "consecutive_failed": 0, "disabled": False,
    })
    monkeypatch.setattr(checkpoint, "_SAVE", {})

    def reject_publication(*_args, **_kwargs):
        raise checkpoint.CheckpointError("cleanup caller coverage rejection")

    monkeypatch.setattr(checkpoint, "_publish_staged_artifacts", reject_publication)
    run = tmp_path / "in-process"
    run.mkdir()
    root = run / "checkpoints"
    root.mkdir()
    stale_tmp = root / f".9999-{os.getpid()}.tmp"
    stale_tmp.mkdir()
    with pytest.raises(checkpoint.CheckpointError, match="cleanup caller coverage rejection"):
        _save(run, 1)
    assert not stale_tmp.exists(), "save() must reap its same-process leftover under the lock"

    background_run = tmp_path / "background"
    background_run.mkdir()
    writer_launched_after = time.time()
    _save(background_run, 2, background=True)
    try:
        outcome = checkpoint.finish_background_writer()
    finally:
        child = checkpoint._STATE.get("child")
        if child:
            try:
                identity = record_identity(
                    child, launched_after=writer_launched_after,
                    command_fragment=Path(sys.executable).name,
                )
                signal_run_verified(
                    background_run, child, identity, signal.SIGKILL,
                    deadline=time.monotonic(), what="checkpoint writer cleanup",
                    require_checkpoint_saved=False,
                )
            except ProcessLookupError:
                pass
            checkpoint.finish_background_writer()
    assert outcome is not None and outcome["outcome"] == "failed"

    stale_root = tmp_path / "stale" / "checkpoints"
    stale_root.mkdir(parents=True)
    stale_sweep_tmp = stale_root / f".0003-{os.getpid() + 1_000_000_000}.tmp"
    stale_sweep_tmp.mkdir()
    monkeypatch.setattr(checkpoint, "_pid_alive", lambda _pid: False)
    assert checkpoint._sweep_stale(stale_root) == 1

    observed = [json.loads(line) for line in events_file.read_text(encoding="utf-8").splitlines()]
    observed_sites = {(event["caller"], event["line"]) for event in observed}
    assert observed_sites == production_sites
    assert len(observed) == len(production_sites)
    assert all(event["lock_held"] for event in observed), observed


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_cleanup_still_propagates_process_interrupts(tmp_path, monkeypatch, interrupt):
    root = tmp_path / "checkpoints"
    tmp = root / ".0001-interrupted.tmp"
    tmp.mkdir(parents=True)

    def interrupt_prune(_root):
        raise interrupt("preserve process control")

    monkeypatch.setattr(checkpoint, "_prune_objects", interrupt_prune)
    with pytest.raises(interrupt, match="preserve process control"):
        checkpoint._cleanup_checkpoint_transaction(root, tmp)


_REFUSAL_CASES = (
    "missing_metadata",
    "metadata_unreadable",
    "metadata_invalid_json",
    "metadata_non_object",
    "manifest_unreadable",
    "manifest_invalid_json",
    "manifest_non_object",
    "manifest_missing_but_required",
    "manifest_relative_not_string",
    "manifest_files_not_list",
    "manifest_row_not_list",
    "manifest_row_empty",
    "manifest_name_not_string",
)


@pytest.mark.parametrize("case", _REFUSAL_CASES)
def test_uncertain_checkpoint_refuses_all_pool_pruning(tmp_path, monkeypatch, case):
    root = tmp_path / "checkpoints"
    run = tmp_path / "run"
    object_dir = root / checkpoint.OBJECT_DIR
    bad = root / "damaged-checkpoint"
    store = root / checkpoint.POOL_DIR / "worker__candidates"
    object_dir.mkdir(parents=True)
    bad.mkdir(parents=True)
    store.mkdir(parents=True)
    orphan = store / "must-stay.bin"
    orphan.write_bytes(b"unreferenced")
    manifest = bad / checkpoint.MANIFEST_FILE
    meta = bad / checkpoint.META_FILE
    expected_error = None

    if case == "missing_metadata":
        pass
    elif case == "metadata_unreadable":
        original = Path.read_text

        def read_text(path, *args, **kwargs):
            if path == meta:
                raise PermissionError("simulated unreadable metadata")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", read_text)
    elif case == "metadata_invalid_json":
        meta.write_text("{")
    elif case == "metadata_non_object":
        meta.write_text("[]")
        expected_error = AttributeError  # malformed JSON value aborts before pruning
    elif case == "manifest_unreadable":
        manifest.write_text("{}")
        original = Path.read_text

        def read_text(path, *args, **kwargs):
            if path == manifest:
                raise PermissionError("simulated unreadable manifest")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", read_text)
    elif case == "manifest_invalid_json":
        manifest.write_text("{")
    elif case == "manifest_non_object":
        manifest.write_text("[]")
    elif case == "manifest_missing_but_required":
        meta.write_text(json.dumps({"blob_manifest_sha256": "expected"}))
    elif case == "manifest_relative_not_string":
        manifest.write_text("synthetic decoded mapping")
        original = checkpoint.json.loads

        def loads(document, *args, **kwargs):
            if document == "synthetic decoded mapping":
                return {None: []}
            return original(document, *args, **kwargs)

        monkeypatch.setattr(checkpoint.json, "loads", loads)
    elif case == "manifest_files_not_list":
        manifest.write_text(json.dumps({"worker/candidates": {}}))
    elif case == "manifest_row_not_list":
        manifest.write_text(json.dumps({"worker/candidates": ["name"]}))
    elif case == "manifest_row_empty":
        manifest.write_text(json.dumps({"worker/candidates": [[]]}))
    elif case == "manifest_name_not_string":
        manifest.write_text(json.dumps({"worker/candidates": [[3]]}))
    else:  # pragma: no cover - keeps the case list exhaustive
        raise AssertionError(case)

    if expected_error is None:
        checkpoint._prune_pool(root, run)
    else:
        with pytest.raises(expected_error):
            checkpoint._prune_pool(root, run)
    assert orphan.is_file(), case


def _assert_kept_checkpoint_loads(checkpoint_dir: Path, restored: Path) -> None:
    meta, _raw = checkpoint._validate(checkpoint_dir)
    manifest = json.loads((checkpoint_dir / checkpoint.MANIFEST_FILE).read_text())
    for relative, rows in manifest.items():
        store = checkpoint_dir.parent / checkpoint.POOL_DIR / relative.replace("/", "__")
        for row in rows:
            assert (store / row[0]).is_file(), (checkpoint_dir.name, relative, row[0])
    restored.mkdir(parents=True)
    checkpoint._restore_files(checkpoint_dir, restored, meta)
    assert (restored / "worker" / "candidates" / "kept.bin").read_bytes() == b"kept bytes"


def test_prune_objects_keeps_newest_quarantine_and_drops_dead_verifications(tmp_path, monkeypatch):
    root = tmp_path / "checkpoints"
    objects = root / checkpoint.OBJECT_DIR
    objects.mkdir(parents=True)
    object_name = f"{'a' * 64}-444"
    other_object_name = f"{'b' * 64}-444"
    live_name = f"{'c' * 64}-444"
    missing_name = f"{'d' * 64}-444"
    live = objects / live_name
    live.write_bytes(b"live")
    os.link(live, tmp_path / "snapshot-link")
    old = objects / f"quarantine-9-{object_name}"
    newest = objects / f"quarantine-10-{object_name}"
    old_symlink = objects / f"quarantine-8-{object_name}"
    separate = objects / f"quarantine-3-{other_object_name}"
    malformed = objects / "quarantine-11-not-an-object-444"
    leading_zero = objects / f"quarantine-010-{object_name}"
    unrelated = objects / "leave-this-alone"
    external = tmp_path / "external"
    external.write_bytes(b"outside the object folder")
    old_symlink.symlink_to(external)
    for path in (old, newest, separate, malformed, leading_zero, unrelated):
        path.write_bytes(b"quarantined")
    info = live.stat()
    live_identity = (os.getpid(), os.fspath(live), info.st_dev, info.st_ino,
                     info.st_size, info.st_mtime_ns, "live-digest")
    missing = (os.getpid(), os.fspath(objects / missing_name), 0, 0, 0, 0, "gone-digest")
    stale = (os.getpid(), os.fspath(live), 0, 0, info.st_size, info.st_mtime_ns, "stale-digest")
    monkeypatch.setattr(checkpoint, "_VERIFIED_OBJECTS", {live_identity, missing, stale})

    checkpoint._prune_objects(root)

    assert live.is_file() and newest.is_file() and separate.is_file()
    assert not old.exists()
    assert old_symlink.is_symlink() and external.read_bytes() == b"outside the object folder"
    assert malformed.is_file() and leading_zero.is_file()
    assert unrelated.read_bytes() == b"quarantined"
    assert checkpoint._VERIFIED_OBJECTS == {live_identity}


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="object-folder link check requires POSIX")
def test_prune_objects_does_not_follow_symlinked_object_folder(tmp_path):
    root = tmp_path / "checkpoints"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    orphan = outside / f"{'e' * 64}-444"
    orphan.write_bytes(b"must stay outside the store")
    (root / checkpoint.OBJECT_DIR).symlink_to(outside, target_is_directory=True)

    checkpoint._prune_objects(root)

    assert orphan.read_bytes() == b"must stay outside the store"


@pytest.mark.skipif(not hasattr(os, "fork"), reason="crash sweep requires POSIX save behavior")
def test_save_crash_after_each_pool_unlink_keeps_retained_checkpoints_loadable(tmp_path, monkeypatch):
    monkeypatch.setenv(checkpoint.KEEP_ENV, "1")
    src = Path(__file__).resolve().parents[1] / "src"
    child_script = tmp_path / "crash_save.py"
    child_script.write_text(f'''\
import os, pickle, random, sys
from pathlib import Path
sys.path.insert(0, {str(src)!r})
from stateframe.engine import kernel_checkpoint as checkpoint
run = Path(sys.argv[1])
kill_at = int(sys.argv[2])
step = int(sys.argv[3])
checkpoint._SIGNING_KEY[:] = [b"pool-prune-test-signing-key-012345"]
checkpoint.configure_product_host()
context = {{name: {{}} for name in ("hostfns", "participants", "framecells", "framefuncs", "cvars")}}
state = {{"frame": {{}}, "modules": {{}}, "classes": {{}}, "patched": {{}},
          "participants": {{}}, "context": [], "random": random.getstate()}}
checkpoint.collect = lambda _frame: (dict(state), context)
checkpoint.dump_to = lambda stream, payload, _ctx, protocol=5: pickle.dump(payload, stream)
checkpoint._STATE["setup"] = {{"code": sys._getframe().f_code, "line": 0, "funcs": frozenset()}}
target_pool = run / "checkpoints" / checkpoint.POOL_DIR
real_unlink = os.unlink
count = 0
def counted_unlink(path, *args, **kwargs):
    global count
    candidate = Path(path)
    if target_pool in candidate.parents and candidate.name.startswith("gone-"):
        if kill_at == 0 and count == 0:
            os._exit(86)
        result = real_unlink(path, *args, **kwargs)
        count += 1
        if count == kill_at:
            os._exit(86)
        return result
    return real_unlink(path, *args, **kwargs)
checkpoint.os.unlink = counted_unlink
checkpoint.save(sys._getframe(), step, run)
os._exit(0)
''')

    for kill_at in range(4):  # before the first delete, then after each of three deletes
        run = tmp_path / f"crash-{kill_at}" / "run"
        candidates = run / "worker" / "candidates"
        candidates.mkdir(parents=True)
        (run / "payload.bin").write_bytes(b"content object")
        (candidates / "kept.bin").write_bytes(b"kept bytes")
        for index in range(3):
            (candidates / f"gone-{index}.bin").write_bytes(f"gone {index}".encode())
        _save(run, 1)
        _save(run, 2)
        for index in range(3):
            (candidates / f"gone-{index}.bin").unlink()

        child_command = (
            'ulimit -v 16000000; exec "$1" "$2" "$3" "$4" "$5"'
        )
        env = os.environ.copy()
        result = subprocess.run(
            ["bash", "-c", child_command, "bash", sys.executable, str(child_script),
             str(run), str(kill_at), "3"],
            env=env, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        assert result.returncode == 86, result.stderr
        latest = checkpoint.latest(run)
        assert latest is not None
        _assert_kept_checkpoint_loads(latest, tmp_path / f"restore-{kill_at}")

        next_checkpoint = _save(run, 4)
        assert checkpoint.latest(run) == next_checkpoint
        _assert_kept_checkpoint_loads(next_checkpoint, tmp_path / f"restore-next-{kill_at}")


def _allocated_pool_bytes(root: Path) -> int:
    return sum(path.stat().st_blocks * 512 for path in root.rglob("*") if path.is_file())


def test_twenty_saves_report_pool_bytes_for_appearing_and_disappearing_file(tmp_path, capsys):
    run = tmp_path / "run"
    candidates = run / "worker" / "candidates"
    candidates.mkdir(parents=True)
    (run / "payload.bin").write_bytes(b"content object")
    live = None
    for step in range(1, 21):
        if step % 2:
            live = candidates / f"appearing-{step}.bin"
            live.write_bytes(bytes([step]) * (64 << 10))
        elif live is not None:
            live.unlink()
            live = None
        _save(run, step)
    root = run / "checkpoints"
    pool_bytes = _allocated_pool_bytes(root / checkpoint.POOL_DIR)
    quarantine_bytes = sum(path.stat().st_blocks * 512
                           for path in (root / checkpoint.OBJECT_DIR).glob("quarantine-*"))
    print(f"POOL_BYTES_AFTER_20_SAVES={pool_bytes + quarantine_bytes} "
          f"QUARANTINE_BYTES={quarantine_bytes}")


@pytest.mark.parametrize("next_step", [5, 6], ids=["same-step", "later-step"])
def test_failed_writer_tree_left_by_a_held_lock_is_removed_by_the_next_save(tmp_path, next_step):
    run = tmp_path / "run"
    root = run / "checkpoints"
    root.mkdir(parents=True)
    tree = root / f".0005-{os.getpid()}.tmp"
    tree.mkdir()
    (tree / "state.pkl").write_bytes(b"partial")
    other = root / f".0005-{os.getppid()}.tmp"
    other.mkdir()
    pid = os.fork()
    if pid == 0:
        os._exit(1)
    saved = dict(checkpoint._STATE)
    try:
        checkpoint._STATE.update(child=pid, child_tmp=tree, child_run_dir=run, child_step=5,
                                 child_writer_generation="5:0", child_journal_offset=0,
                                 child_journal_file_id=None, child_journal_missing=True)
        with (root / ".checkpoint.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            while not checkpoint._reap():
                time.sleep(0.01)
            assert tree.is_dir()
            assert checkpoint._sweep_stale(root) == 0
        assert checkpoint._sweep_stale(root) == 0
        assert tree.is_dir()

        final = _save(run, next_step)

        assert checkpoint.latest(run) == final
        assert not tree.exists()
        assert other.is_dir()
    finally:
        checkpoint._STATE.clear()
        checkpoint._STATE.update(saved)
