"""Checkpoint save cost: scandir snapshot walk, content-addressed object store, one save per send."""
from __future__ import annotations

import errno
import hashlib
import io
import json
import os
import pickle
import stat
import sys
from collections.abc import MutableMapping
from pathlib import Path

import pytest

from stateframe import resume
from stateframe.engine import kernel_checkpoint as checkpoint

posix_only = pytest.mark.skipif(os.name != "posix", reason="inode sharing requires POSIX")


def _tree(root: Path, directories: int = 4, files: int = 5) -> None:
    for d in range(directories):
        (root / f"d{d}").mkdir(parents=True)
        for f in range(files):
            (root / f"d{d}" / f"f{f}.py").write_bytes(f"{d}-{f}\n".encode() * 50)


def test_workspace_switch_reads_are_bounded_per_pickle_save(monkeypatch):
    setting = "STATEFRAME_WORKSPACE_OBJECTS"

    class CountingEnvironment(MutableMapping[str, str]):
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.reads = 0

        def __getitem__(self, key):
            return self.wrapped[key]

        def __setitem__(self, key, value):
            self.wrapped[key] = value

        def __delitem__(self, key):
            del self.wrapped[key]

        def __iter__(self):
            return iter(self.wrapped)

        def __len__(self):
            return len(self.wrapped)

        def get(self, key, default=None):
            if key == setting:
                self.reads += 1
            return self.wrapped.get(key, default)

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

    monkeypatch.setenv(setting, "off")
    environ = CountingEnvironment(os.environ)
    monkeypatch.setattr(checkpoint.os, "environ", environ)
    indexed = object()
    monkeypatch.setattr(checkpoint, "_SAVE", {"ids": {id(indexed): ("stable", "object-id")}})
    payload = [object() for _ in range(512)] + [indexed]
    ctx = {name: {} for name in ("hostfns", "participants", "framecells", "framefuncs", "cvars")}
    reads_per_save = []
    serialized = []

    for setting_value in ("off", "on"):
        environ[setting] = setting_value
        before = environ.reads
        stream = io.BytesIO()
        checkpoint.dump_to(stream, payload, ctx)
        reads_per_save.append(environ.reads - before)
        serialized.append(stream.getvalue())

    assert reads_per_save[0] <= 4
    assert reads_per_save[1] <= 4
    assert serialized[0] == serialized[1]
    assert b"object-id" in serialized[0]


def test_workspace_switch_is_refreshed_between_pickle_saves(monkeypatch, tmp_path):
    import sys
    import types

    module = types.ModuleType("kernel_shell_tool")

    class ShellTool:
        pass

    ShellTool.__module__ = module.__name__
    ShellTool.__qualname__ = "ShellTool"
    module.ShellTool = ShellTool
    tool = ShellTool()
    tool.root = tmp_path
    module.service = lambda: tool
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(checkpoint, "_SAVE", {})
    ctx = {name: {} for name in ("hostfns", "participants", "framecells", "framefuncs", "cvars")}

    serialized = []
    for setting_value in ("off", "on", "off"):
        monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", setting_value)
        serialized.append(checkpoint.dumps([tool], ctx))

    assert serialized[0] == serialized[2]
    assert serialized[0] != serialized[1]
    assert b"host-shell-tool" not in serialized[0]
    assert b"host-shell-tool" in serialized[1]


@posix_only
def test_unchanged_snapshot_costs_one_lstat_per_file(tmp_path, monkeypatch):
    run = tmp_path / "env"
    _tree(run)
    first, second = tmp_path / "first", tmp_path / "second"
    # ctime cannot be aged in a test, so the racy-timestamp window is lifted here; that guard has its
    # own test in test_checkpoint_disk_reuse.py.
    monkeypatch.setattr(checkpoint, "_RACY_WINDOW_NS", 0)
    checkpoint._copy_tree(run, first, None, set())
    calls = []
    real = os.lstat
    monkeypatch.setattr(os, "lstat", lambda *a, **k: calls.append(a[0]) or real(*a, **k))
    counts = checkpoint._copy_tree(run, second, first, set())
    assert counts == {"linked": 20, "copied": 0, "cloned": 0}
    assert len(calls) <= 20  # was 3 per file: lstat, islink, previous copy


@posix_only
def test_changed_file_is_stored_once_in_the_object_store(tmp_path):
    run, objects = tmp_path / "env", tmp_path / "objects"
    _tree(run)
    first, second = tmp_path / "first", tmp_path / "second"
    checkpoint._copy_tree(run, first, None, set(), objects=objects)
    # The first snapshot fills the store: every distinct file content is stored once.
    assert len(list(objects.iterdir())) == 20
    target = run / "d0" / "f0.py"
    target.write_bytes(b"changed\n" * 10)
    counts = checkpoint._copy_tree(run, second, first, set(), objects=objects)
    assert counts["copied"] == 1 and counts["linked"] == 19
    stored = list(objects.iterdir())
    assert len(stored) == 21
    assert (second / "d0" / "f0.py").stat().st_ino == next(
        item for item in stored if item.stat().st_size == len(b"changed\n" * 10)).stat().st_ino
    assert (second / "d0" / "f0.py").read_bytes() == b"changed\n" * 10
    assert (second / "d0" / "f0.py").stat().st_mtime_ns == target.stat().st_mtime_ns


@posix_only
def test_object_key_covers_content_and_mode_not_mtime(tmp_path):
    """Stale-value guard: a file differing in bytes or mode never reuses an object; an mtime-only change does."""
    run, objects = tmp_path / "env", tmp_path / "objects"
    run.mkdir()
    base = tmp_path / "base"
    checkpoint._copy_tree(run, base, None, set())
    target = run / "a.txt"
    seen = []
    for index, (data, mode, mtime) in enumerate([
            (b"aaaa", 0o644, 1_000_000_000), (b"bbbb", 0o644, 1_000_000_000),
            (b"aaaa", 0o755, 1_000_000_000), (b"aaaa", 0o644, 2_000_000_000)]):
        target.write_bytes(data)
        target.chmod(mode)
        os.utime(target, ns=(mtime, mtime))
        out = tmp_path / f"out{index}"
        checkpoint._copy_tree(run, out, base, set(), objects=objects)
        copy = (out / "a.txt").stat()
        assert (out / "a.txt").read_bytes() == data
        assert (copy.st_mode & 0o777) == mode & ~0o222  # stored read-only
        seen.append(copy.st_ino)
    # the mtime-only change (index 3) shares the object of the first capture of these bytes and mode
    assert seen[3] == seen[0]
    assert len(set(seen)) == 3
    assert len(list(objects.iterdir())) == 3


@posix_only
def test_prune_objects_keeps_linked_and_drops_orphans_and_temporaries(tmp_path):
    root = tmp_path / "checkpoints"
    objects = root / checkpoint.OBJECT_DIR
    objects.mkdir(parents=True)
    kept, orphan = objects / ("a" * 64 + "-600"), objects / ("b" * 64 + "-644")
    kept.write_text("k")
    orphan.write_text("o")
    os.link(kept, root / "snapshot-link")
    (objects / ".123-1.tmp").write_text("partial")
    checkpoint._prune_objects(root)
    assert sorted(p.name for p in objects.iterdir()) == [kept.name]


def test_save_point_saves_at_most_once_per_send(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    saves = []
    sends = {"n": 0}
    before = checkpoint._STATE.copy()
    monkeypatch.setattr(checkpoint, "mode", lambda: "save")
    monkeypatch.setattr(checkpoint, "_every", lambda: 1)
    monkeypatch.setattr(checkpoint, "_sends", lambda _run: sends["n"])
    monkeypatch.setattr(checkpoint, "_async_enabled", lambda: False)
    monkeypatch.setattr(checkpoint, "_low_disk_space", lambda _run: None)
    monkeypatch.setattr(checkpoint, "save",
                        lambda _f, step, _r, background=False: saves.append(step) or run / f"{step:04d}-1")
    try:
        checkpoint._STATE.update(disabled=None, restored_step=None, last_sends=None, saved=0)
        poc = run / "poc"
        for step, n in [(1, 0), (2, 1), (3, 1), (4, 1), (5, 2), (6, 2), (7, 3)]:
            sends["n"] = n
            checkpoint.save_point(None, step, poc)
        assert saves == [2, 5, 7]  # boundaries without a new model send never save
    finally:
        checkpoint._STATE.clear()
        checkpoint._STATE.update(before)


_SIGNING_KEY = b"checkpoint-save-cost-signing-key-0123456789"


@pytest.fixture
def _selection_context(monkeypatch):
    monkeypatch.setenv(checkpoint.LEGACY_DIRECT_ENGINE_ENV, "1")
    monkeypatch.setattr(checkpoint, "_SIGNING_KEY", [_SIGNING_KEY])
    monkeypatch.setattr(checkpoint, "_SECRETS", [])
    monkeypatch.setattr(checkpoint, "_PRODUCT_HOST", False)
    monkeypatch.setattr(checkpoint, "_FINGERPRINT", None)
    monkeypatch.setattr(checkpoint, "_HOST_IDENTITY_BOUND", False)
    checkpoint.configure_product_host()


def _resume_candidate(run: Path, name: str, seq: int, data: bytes = b"original",
                      *, file_digest: bool = True) -> Path:
    root = run / "checkpoints"
    checkpoint._private_dir(root)
    folder = root / name
    checkpoint._private_dir(folder)
    copied = folder / "run" / "env"
    copied.mkdir(parents=True)
    (copied / "model.py").write_bytes(data)
    (folder / "run" / "journal.jsonl").write_bytes(b'{"event":"sample"}\n')
    (folder / "run" / "request.json").write_bytes(b'{"request":"sample"}\n')
    raw = pickle.dumps({"seq": seq}, protocol=5)
    state = folder / checkpoint.STATE_FILE
    state.write_bytes(raw)
    state.chmod(0o600)
    manifest = b"{}"
    (folder / checkpoint.MANIFEST_FILE).write_bytes(manifest)
    (folder / checkpoint.MANIFEST_FILE).chmod(0o600)
    meta = {
        "schema": checkpoint.SCHEMA, "step": seq, "seq": seq, "sends": 0,
        "created": float(seq), "entries": ["env", "journal.jsonl", "request.json"],
        "python": list(sys.version_info[:3]),
        "engine_root": checkpoint._engine_root(),
        "engine_files_sha256": checkpoint.engine_fingerprint(bind_host=True),
        "pid": os.getpid(), "clocks": {"monotonic": 0, "perf_counter": 0},
        "files": {"linked": 0, "copied": 1, "cloned": 0},
        "blob_manifest_sha256": hashlib.sha256(manifest).hexdigest(), "pool_sha256": {},
        "file_chunks": {}, "file_modes": {}, "file_mtime_ns": {}, "link_fallback": 0,
        "bytes_new": len(raw) + len(manifest), "blob_count": 0,
        "state_bytes": len(raw), "state_sha256": hashlib.sha256(raw).hexdigest(),
        "chunk_index": {}, "chunks": [],
    }
    if file_digest:
        meta["file_sha256"] = {"env": {"model.py": hashlib.sha256(data).hexdigest()}}
    meta["signature"] = checkpoint._signature(meta, meta["state_sha256"])
    metadata = folder / checkpoint.META_FILE
    metadata.write_text(json.dumps(meta, sort_keys=True))
    metadata.chmod(0o600)
    return folder


def _select(run: Path):
    return resume.select_checkpoint(run, [{"send": 1}], run / "wire")


def test_same_size_run_file_damage_rejects_with_one_named_hash(tmp_path, monkeypatch,
                                                               _selection_context):
    folder = _resume_candidate(tmp_path / "run", "0001-1", 1)
    damaged = folder / "run" / "env" / "model.py"
    damaged.write_bytes(b"tampered")
    calls = []
    real_digest = checkpoint._file_sha256

    def track_digest(path, **kwargs):
        calls.append(Path(path))
        return real_digest(path, **kwargs)

    monkeypatch.setattr(checkpoint, "_file_sha256", track_digest)

    with pytest.raises(checkpoint.CheckpointError, match="env/model.py checksum mismatch"):
        checkpoint.validate(folder)
    assert calls == [damaged]


def test_bad_newest_file_falls_back_to_older_checkpoint(tmp_path, _selection_context):
    run = tmp_path / "run"
    older = _resume_candidate(run, "0001-1", 1)
    newest = _resume_candidate(run, "0002-1", 2)
    damaged = newest / "run" / "env" / "model.py"
    damaged.write_bytes(b"tampered")

    selected = _select(run)

    assert selected is not None and selected["dir"] == str(older)
    refusals = [json.loads(line) for line in (run / "checkpoints" / "events.jsonl").read_text().splitlines()]
    assert len(refusals) == 1
    assert "env/model.py checksum mismatch" in refusals[0]["detail"]


def test_bad_only_checkpoint_falls_back_to_replay(tmp_path, _selection_context):
    run = tmp_path / "run"
    folder = _resume_candidate(run, "0001-1", 1)
    (folder / "run" / "env" / "model.py").write_bytes(b"tampered")

    assert _select(run) is None
    refusal = json.loads((run / "checkpoints" / "events.jsonl").read_text())
    assert "env/model.py checksum mismatch" in refusal["detail"]


def test_checkpoint_without_file_digests_keeps_legacy_behavior(tmp_path, _selection_context):
    run = tmp_path / "run"
    folder = _resume_candidate(run, "0001-1", 1, file_digest=False)
    (folder / "run" / "env" / "model.py").write_bytes(b"tampered")

    selected = _select(run)

    assert selected is not None and selected["dir"] == str(folder)


def test_undamaged_journal_request_and_checkpoint_bytes_stay_equal(tmp_path, _selection_context):
    folder = _resume_candidate(tmp_path / "run", "0001-1", 1)
    paths = [folder / "run" / "journal.jsonl", folder / "run" / "request.json",
             folder / checkpoint.STATE_FILE, folder / checkpoint.MANIFEST_FILE,
             folder / checkpoint.META_FILE]
    before = {path: path.read_bytes() for path in paths}

    checkpoint.validate(folder)

    assert {path: path.read_bytes() for path in paths} == before


def test_unreadable_recorded_file_is_rejected_and_named(tmp_path, monkeypatch, _selection_context):
    run = tmp_path / "run"
    folder = _resume_candidate(run, "0001-1", 1)
    target = folder / "run" / "env" / "model.py"
    real_open = os.open

    def unreadable(path, *args, **kwargs):
        if os.fspath(path) == os.fspath(target):
            raise PermissionError(errno.EACCES, "unreadable", os.fspath(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(checkpoint.os, "open", unreadable)
    assert _select(run) is None
    refusal = json.loads((run / "checkpoints" / "events.jsonl").read_text())
    assert "env/model.py checksum mismatch" in refusal["detail"]


@pytest.mark.parametrize("corrupt", [False, True])
def test_existing_object_is_hashed_once_or_quarantined(tmp_path, monkeypatch, corrupt):
    source, objects = tmp_path / "source", tmp_path / checkpoint.OBJECT_DIR
    data = b"right" if corrupt else b"bytes"
    wrong = b"wrong"
    source.write_bytes(data)
    source.chmod(0o644)
    objects.mkdir()
    info = source.stat()
    digest = hashlib.sha256(data).hexdigest()
    final = objects / f"{digest}-{stat.S_IMODE(info.st_mode):o}"
    final.write_bytes(wrong if corrupt else data)
    final.chmod(0o444)
    monkeypatch.setattr(checkpoint, "_VERIFIED_OBJECTS", set())
    calls = []
    real_digest = checkpoint._file_sha256

    def track_digest(path, **kwargs):
        calls.append(Path(path))
        return real_digest(path, **kwargs)

    monkeypatch.setattr(checkpoint, "_file_sha256", track_digest)
    real_rename = os.rename
    rename_calls = []

    def track_rename(source, target, *args, **kwargs):
        rename_calls.append((source, target, kwargs))
        return real_rename(source, target, *args, **kwargs)

    monkeypatch.setattr(checkpoint.os, "rename", track_rename)

    written = None
    for index in range(3):  # each snapshot hard link changes the object's ctime
        stored, current_written = checkpoint._store_object(os.fspath(source), info, digest, objects)
        if index == 0:
            written = current_written
        os.link(stored, tmp_path / f"checkpoint-{index}")

    assert stored == final and final.read_bytes() == data
    assert len(calls) == (1 if not corrupt else 2)
    if corrupt:
        aside, = objects.glob("quarantine-*")
        assert aside.read_bytes() == wrong
        assert written == len(data)
        assert len(rename_calls) == 1
        source_name, target_name, rename_options = rename_calls[0]
        assert source_name == final.name and target_name == aside.name
        assert rename_options["src_dir_fd"] == rename_options["dst_dir_fd"]
    else:
        assert written == 0
        assert rename_calls == []


def test_enospc_while_quarantining_preserves_bad_object(tmp_path, monkeypatch):
    source, objects = tmp_path / "source", tmp_path / checkpoint.OBJECT_DIR
    data, wrong = b"right", b"wrong"
    source.write_bytes(data)
    info = source.stat()
    objects.mkdir()
    final = objects / f"{hashlib.sha256(data).hexdigest()}-{info.st_mode & 0o777:o}"
    final.write_bytes(wrong)
    monkeypatch.setattr(checkpoint, "_VERIFIED_OBJECTS", set())

    def full_disk(*_args, **_kwargs):
        raise OSError(errno.ENOSPC, "no space")

    monkeypatch.setattr(checkpoint.os, "rename", full_disk)
    with pytest.raises(OSError) as error:
        checkpoint._store_object(os.fspath(source), info, hashlib.sha256(data).hexdigest(), objects)
    assert error.value.errno == errno.ENOSPC
    assert final.read_bytes() == wrong
    assert not list(objects.glob("*.tmp")) and not list(objects.glob("quarantine-*"))


# ---------------------------------------------------------------- kill / resume replay

import subprocess  # noqa: E402

CHILD = Path(__file__).resolve().parent / "fixtures" / "checkpoint_kill_child.py"
TURNS = 6


def _child(action: str, run: Path, spec: str = "none") -> subprocess.CompletedProcess:
    # The child saves outside a runner, so it has no spec.json identity record to verify.
    env = dict(os.environ, STATEFRAME_CHECKPOINT_LEGACY_DIRECT_ENGINE="1", PYTHONPATH=os.pathsep.join(
        [str(Path(checkpoint.__file__).parents[2]), str(CHILD.parent.parent)]))
    return subprocess.run([sys.executable, str(CHILD), action, str(run), str(TURNS), spec],
                          env=env, capture_output=True, text=True, timeout=120)


@pytest.fixture(scope="module")
def control_digest(tmp_path_factory) -> str:
    done = _child("run", tmp_path_factory.mktemp("control") / "run")
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


@posix_only
@pytest.mark.parametrize("spec", ["mid:3", "mid:1", "prepublish:4", "between:2", "between:5"])
def test_sigkill_then_resume_is_byte_identical_to_no_kill_run(tmp_path, control_digest, spec):
    run = tmp_path / "run"
    killed = _child("run", run, spec)
    assert killed.returncode == -9, killed.stderr  # SIGKILL, not a clean exit or a crash
    root = run / "checkpoints"
    pointer = (root / "LATEST").read_text().strip() if (root / "LATEST").exists() else None
    if pointer is not None:  # LATEST only ever names a complete, published checkpoint
        assert (root / pointer / checkpoint.META_FILE).is_file()
    resumed = _child("resume", run)
    if pointer is None:  # killed before the first publish: nothing to resume, replay from scratch
        assert resumed.stdout.strip() == "NOCHECKPOINT"
        resumed = _child("run", tmp_path / "fresh")
    assert resumed.returncode == 0, resumed.stderr
    assert resumed.stdout.strip() == control_digest


@posix_only
def test_torn_state_file_is_rejected_not_loaded(tmp_path):
    run = tmp_path / "run"
    assert _child("run", run).returncode == 0
    newest = checkpoint.latest(run)
    state = newest / checkpoint.STATE_FILE
    state.write_bytes(state.read_bytes()[:-7])  # a partially written state image
    resumed = _child("resume", run)
    assert resumed.returncode != 0
    assert "CheckpointError" in resumed.stderr


@posix_only
def test_object_link_limit_falls_back_to_a_private_copy(tmp_path, monkeypatch):
    """Identical tiny files share one object inode; EMLINK must not fail the save."""
    import errno as _errno
    real_link = os.link
    refused = []

    def full_link(source, destination, *args, **kwargs):
        if "objects" in str(source):
            refused.append(destination)
            raise OSError(_errno.EMLINK, "Too many links")
        return real_link(source, destination, *args, **kwargs)

    run, objects = tmp_path / "env", tmp_path / "objects"
    _tree(run)
    first, second = tmp_path / "first", tmp_path / "second"
    checkpoint._copy_tree(run, first, None, set(), objects=objects)
    target = run / "d0" / "f0.py"
    target.write_bytes(b"changed\n")
    monkeypatch.setattr(checkpoint.os, "link", full_link)
    checkpoint._copy_tree(run, second, first, set(), objects=objects)
    assert refused, "the object-store link path was not exercised"
    assert (second / "d0" / "f0.py").read_bytes() == b"changed\n"
    assert (second / "d0" / "f0.py").stat().st_mtime_ns == target.stat().st_mtime_ns


@posix_only
def test_copy_tree_applies_home_exclusions_and_still_dedups_through_object_store(tmp_path):
    run, home = tmp_path / "env", tmp_path / "env" / "home"
    _tree(run, directories=1, files=2)
    home.mkdir()
    (home / "identity.key").write_bytes(b"installation-key-material")
    (run / "link").symlink_to(home / "identity.key")
    objects = tmp_path / "objects"
    first, second = tmp_path / "first", tmp_path / "second"
    checkpoint._copy_tree(run, first, None, set(), objects=objects, home=home,
                          home_subtree=home, excluded=set())
    checkpoint._copy_tree(run, second, first, set(), objects=objects, home=home,
                          home_subtree=home, excluded=set())
    for snapshot in (first, second):
        assert not (snapshot / "home").exists()  # boundary filter wins over the storage path
        assert not (snapshot / "link").exists() and not (snapshot / "link").is_symlink()
        assert (snapshot / "d0" / "f0.py").read_bytes() == b"0-0\n" * 50


@posix_only
def test_copy_tree_unchanged_file_is_rescanned_only_after_a_new_secret(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint, "_SECRETS", [])
    monkeypatch.setattr(checkpoint, "_SECRET_GENERATION", 0)
    monkeypatch.setattr(checkpoint, "_VERIFIED_POOL_IDENTITIES", set())
    monkeypatch.setattr(checkpoint, "_DIGEST_MEMO", {})
    monkeypatch.setattr(checkpoint, "_RACY_WINDOW_NS", 0)  # ctime of a fresh file cannot be aged here
    run = tmp_path / "env"
    _tree(run, directories=1, files=2)
    first, second, third = tmp_path / "first", tmp_path / "second", tmp_path / "third"
    checkpoint._copy_tree(run, first, None, set())
    scans = []
    real = checkpoint._check_snapshot_file
    monkeypatch.setattr(checkpoint, "_check_snapshot_file", lambda *a, **k: scans.append(a[0]) or real(*a, **k))
    checkpoint._copy_tree(run, second, first, set())
    assert scans == []  # captured clean once under this secret set: linked without another read
    checkpoint.register_secret(b"0-1\n" * 3)  # a secret that appears inside f1.py
    with pytest.raises(checkpoint.CheckpointError):
        checkpoint._copy_tree(run, third, second, set())
    assert scans  # the generation bump invalidated the memo, so the file was read again
