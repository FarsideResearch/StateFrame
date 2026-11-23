"""Process-kill points for checkpoint object, snapshot, and publication writes."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from stateframe.engine import kernel_checkpoint as checkpoint
from stateframe.engine import kernel_turn_publication as publication
from stateframe import resume


KILL_EXIT = 73
SIGNING_KEY = b"checkpoint-kill-points-signing-key-0123456789"


@pytest.fixture(autouse=True)
def _checkpoint_context(monkeypatch):
    monkeypatch.setattr(checkpoint, "_SIGNING_KEY", [SIGNING_KEY])
    monkeypatch.setattr(checkpoint, "_SECRETS", [])
    monkeypatch.setenv(checkpoint.LEGACY_DIRECT_ENGINE_ENV, "1")
    monkeypatch.delenv(checkpoint._PUBLICATION_SWITCH, raising=False)
    checkpoint.configure_product_host()


def _kill_child(operation: str, **payload) -> None:
    """Run a kill-point operation in a fresh, single-threaded Python process."""
    repository = Path(__file__).resolve().parents[1]
    source = str(repository / "src")
    environment = os.environ.copy()
    pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source + (os.pathsep + pythonpath if pythonpath else "")
    environment[checkpoint.LEGACY_DIRECT_ENGINE_ENV] = "1"
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--crash-worker", operation,
         json.dumps(payload)],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == KILL_EXIT, (
        f"{operation} exited {result.returncode}, expected {KILL_EXIT}; "
        f"stdout={result.stdout!r}; stderr={result.stderr!r}"
    )
    assert not result.stdout and not result.stderr, (
        f"{operation} emitted output; stdout={result.stdout!r}; stderr={result.stderr!r}"
    )


def _configure_crash_worker(*, indexed_publication: bool) -> None:
    checkpoint._SIGNING_KEY = [SIGNING_KEY]
    checkpoint._SECRETS = []
    os.environ[checkpoint.LEGACY_DIRECT_ENGINE_ENV] = "1"
    if indexed_publication:
        os.environ[checkpoint._PUBLICATION_SWITCH] = "on"
    else:
        os.environ.pop(checkpoint._PUBLICATION_SWITCH, None)
    checkpoint.configure_product_host()


def _crash_worker(operation: str, payload: dict[str, object]) -> None:
    """Dispatch one injected crash after the child has imported a clean runtime."""
    assert threading.active_count() == 1
    _configure_crash_worker(indexed_publication=operation == "indexed_publish")

    if operation == "object_store":
        source = Path(payload["source"])
        objects = Path(payload["objects"])
        point = payload["point"]
        data = source.read_bytes()
        info = os.stat(source)
        digest = hashlib.sha256(data).hexdigest()
        final = objects / f"{digest}-{stat.S_IMODE(info.st_mode):o}"
        real_fsync, real_replace, real_chmod = os.fsync, os.replace, os.chmod

        def stop_before_fsync(fd):
            if point == "before_fsync":
                os._exit(KILL_EXIT)
            real_fsync(fd)
            if point == "after_fsync":
                os._exit(KILL_EXIT)

        def replace_then_stop(src, dst):
            result = real_replace(src, dst)
            if point == "after_rename" and Path(dst) == final:
                os._exit(KILL_EXIT)
            return result

        def chmod_then_stop(path, mode, *args, **kwargs):
            result = real_chmod(path, mode, *args, **kwargs)
            if (point == "after_readonly" and Path(path).parent == objects
                    and stat.S_IMODE(os.stat(path).st_mode)
                    == stat.S_IMODE(info.st_mode) & ~0o222):
                os._exit(KILL_EXIT)
            return result

        checkpoint.os.fsync = stop_before_fsync
        checkpoint.os.replace = replace_then_stop
        checkpoint.os.chmod = chmod_then_stop
        checkpoint._store_object(str(source), info, digest, objects)
        return

    if operation == "copy_tree":
        source = Path(payload["source"])
        target = Path(payload["target"])
        previous = Path(payload["previous"])
        real_link = os.link

        def link_then_stop(src, dst, *args, **kwargs):
            result = real_link(src, dst, *args, **kwargs)
            os._exit(KILL_EXIT)

        checkpoint.os.link = link_then_stop
        checkpoint._copy_tree(source, target, previous, set())
        return

    if operation == "snapshot_tree":
        source = Path(payload["source"])
        target = Path(payload["target"])
        object_pool = Path(payload["object_pool"])
        real_replace = os.replace

        def replace_then_stop(src, dst):
            result = real_replace(src, dst)
            os._exit(KILL_EXIT)

        checkpoint.os.replace = replace_then_stop
        checkpoint._snapshot_tree(source, target, "custom", object_pool, {}, [])
        return

    if operation == "staged_pool_link":
        root = Path(payload["root"])
        temporary = Path(payload["temporary"])
        staged_root = Path(payload["staged_root"])
        checkpoint._POOL_KNOWN.clear()
        checkpoint._SAVE.clear()
        checkpoint._SAVE.update(artifact_root=staged_root,
                                pool_root=root / checkpoint.POOL_DIR)
        real_link = os.link

        def link_then_stop(src, dst, *args, **kwargs):
            result = real_link(src, dst, *args, **kwargs)
            os._exit(KILL_EXIT)

        checkpoint.os.link = link_then_stop
        checkpoint._publish_staged_artifacts(root, temporary)
        return

    if operation == "snapshot_fsync":
        snapshot = Path(payload["snapshot"])
        new_pool_file = Path(payload["new_pool_file"])
        fsync_number = payload["fsync_number"]
        real_fsync = os.fsync
        observed = [0]

        def fsync_then_stop(fd):
            real_fsync(fd)
            observed[0] += 1
            if observed[0] == fsync_number:
                os._exit(KILL_EXIT)

        checkpoint.os.fsync = fsync_then_stop
        checkpoint._fsync_snapshot(snapshot, [new_pool_file])
        return

    if operation == "legacy_publish":
        root = Path(payload["root"])
        pending = Path(payload["pending"])
        final = Path(payload["final"])
        run = Path(payload["run"])
        point = payload["point"]
        real_rename, real_fsync_dir = os.rename, checkpoint._fsync_dir
        real_fsync_file, real_replace = checkpoint._fsync_file, os.replace
        root_syncs = [0]

        def rename_then_stop(src, dst, *args, **kwargs):
            result = real_rename(src, dst, *args, **kwargs)
            if point == "after_rename" and Path(src) == pending and Path(dst) == final:
                os._exit(KILL_EXIT)
            return result

        def sync_dir_then_stop(path):
            result = real_fsync_dir(path)
            if Path(path) == root:
                root_syncs[0] += 1
                if ((point == "after_final_dir_fsync" and root_syncs[0] == 1)
                        or (point == "after_latest_dir_fsync" and root_syncs[0] == 2)):
                    os._exit(KILL_EXIT)
            return result

        def sync_file_then_stop(path):
            result = real_fsync_file(path)
            if point == "after_latest_file_fsync":
                os._exit(KILL_EXIT)
            return result

        def replace_latest(src, dst):
            if point == "before_latest_replace" and Path(dst) == root / "LATEST":
                os._exit(KILL_EXIT)
            result = real_replace(src, dst)
            if point == "after_latest_replace" and Path(dst) == root / "LATEST":
                os._exit(KILL_EXIT)
            return result

        checkpoint.os.rename = rename_then_stop
        checkpoint._fsync_dir = sync_dir_then_stop
        checkpoint._fsync_file = sync_file_then_stop
        checkpoint.os.replace = replace_latest
        checkpoint._publish_checkpoint(root, pending, final, run)
        return

    if operation == "indexed_publish":
        root = Path(payload["root"])
        pending = Path(payload["pending"])
        final = Path(payload["final"])
        run = Path(payload["run"])
        index_path = Path(payload["index_path"])
        replace_stage = payload["replace_stage"]
        real_replace, real_sync = os.replace, publication.sync_directory
        root_syncs = [0]

        def sync_then_stop(path):
            result = real_sync(path)
            if Path(path) == root:
                root_syncs[0] += 1
                if replace_stage == "after_index_dir_fsync" and root_syncs[0] == 2:
                    os._exit(KILL_EXIT)
            return result

        def replace_at_index(src, dst):
            if Path(dst) == index_path and replace_stage == "before_index_replace":
                os._exit(KILL_EXIT)
            result = real_replace(src, dst)
            if Path(dst) == index_path and replace_stage == "after_index_replace":
                os._exit(KILL_EXIT)
            return result

        publication.sync_directory = sync_then_stop
        checkpoint.os.replace = replace_at_index
        checkpoint._publish_checkpoint(root, pending, final, run)
        return

    raise ValueError(f"unknown crash operation: {operation}")


def _write_checkpoint(root: Path, name: str, seq: int, *, sends: int = 0) -> Path:
    """Write a small authenticated checkpoint that resume's real preflight accepts."""
    checkpoint._private_dir(root)
    folder = root / name
    checkpoint._private_dir(folder)
    raw = pickle.dumps({"saved_seq": seq}, protocol=5)
    state = folder / checkpoint.STATE_FILE
    state.write_bytes(raw)
    state.chmod(0o600)
    manifest = b"{}"
    (folder / checkpoint.MANIFEST_FILE).write_bytes(manifest)
    (folder / checkpoint.MANIFEST_FILE).chmod(0o600)
    meta = {
        "schema": checkpoint.SCHEMA,
        "step": seq,
        "seq": seq,
        "sends": sends,
        "created": float(seq),
        "entries": [],
        "python": list(sys.version_info[:3]),
        "engine_root": checkpoint._engine_root(),
        "engine_files_sha256": checkpoint.engine_fingerprint(bind_host=True),
        "pid": os.getpid(),
        "clocks": {"monotonic": time.monotonic(), "perf_counter": time.perf_counter()},
        "files": {"linked": 0, "copied": 0, "cloned": 0},
        "blob_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "pool_sha256": {},
        "file_chunks": {},
        "file_sha256": {},
        "file_modes": {},
        "file_mtime_ns": {},
        "link_fallback": 0,
        "bytes_new": len(raw) + len(manifest),
        "blob_count": 0,
        "state_bytes": len(raw),
        "state_sha256": hashlib.sha256(raw).hexdigest(),
        "chunk_index": {},
        "chunks": [],
    }
    meta["signature"] = checkpoint._signature(meta, meta["state_sha256"])
    metadata = folder / checkpoint.META_FILE
    metadata.write_text(json.dumps(meta, sort_keys=True))
    metadata.chmod(0o600)
    return folder


def _selected(run: Path, *, atomic: bool = False) -> dict[str, object] | None:
    return resume.select_checkpoint(run, [{"send": 1}], run / "wire",
                                    atomic_publication=atomic)


@pytest.mark.parametrize("point", ("before_fsync", "after_fsync", "after_readonly",
                                   "after_rename"))
def test_object_store_crash_keeps_partial_objects_unreferenced(tmp_path, point):
    source = tmp_path / "live.bin"
    data = b"object bytes survive only after the source copy is complete\n" * 8
    source.write_bytes(data)
    info = os.stat(source)
    digest = hashlib.sha256(data).hexdigest()
    objects = tmp_path / "checkpoints" / checkpoint.OBJECT_DIR
    objects.parent.mkdir()
    final = objects / f"{digest}-{stat.S_IMODE(info.st_mode):o}"

    _kill_child("object_store", source=str(source), objects=str(objects), point=point)
    temporaries = [path for path in objects.iterdir() if path.name.endswith(".tmp")]
    if point == "after_rename":
        assert final.read_bytes() == data
        assert stat.S_IMODE(final.stat().st_mode) == stat.S_IMODE(info.st_mode) & ~0o222
        assert temporaries == []
    else:
        assert not final.exists()
        assert len(temporaries) == 1
        assert temporaries[0].read_bytes() == data


def test_copy_tree_crash_after_hardlink_leaves_old_copy_intact_and_new_copy_hidden(tmp_path):
    run = tmp_path / "run"
    root = run / "checkpoints"
    old = _write_checkpoint(root, "0001-1", 1)
    (root / "LATEST").write_text(old.name + "\n")
    source = run / "env"
    source.mkdir(parents=True)
    live = source / "module.py"
    live.write_bytes(b"unchanged module\n")
    previous = old / "run" / "env"
    previous.parent.mkdir(parents=True)
    checkpoint._copy_tree(source, previous, None, set())
    target = root / ".0002-2.tmp" / "run" / "env"
    target.mkdir(parents=True)

    _kill_child("copy_tree", source=str(source), target=str(target), previous=str(previous))
    prior_file, interrupted_file = previous / live.name, target / live.name
    assert prior_file.read_bytes() == live.read_bytes()
    assert os.path.samestat(prior_file.stat(), interrupted_file.stat())
    assert checkpoint.latest(run) == old
    assert _selected(run)["dir"] == str(old)


def test_snapshot_tree_crash_after_file_replace_leaves_only_a_hidden_partial_snapshot(tmp_path):
    run = tmp_path / "run"
    root = run / "checkpoints"
    old = _write_checkpoint(root, "0001-1", 1)
    (root / "LATEST").write_text(old.name + "\n")
    source = run / "source-tree"
    source.mkdir(parents=True)
    (source / "payload.bin").write_bytes(b"captured file bytes\n" * 16)
    temporary = root / ".0002-2.tmp"
    target = temporary / "run" / "custom"
    target.parent.mkdir(parents=True)

    _kill_child("snapshot_tree", source=str(source), target=str(target),
                object_pool=str(root / checkpoint.POOL_DIR))
    assert (target / "payload.bin").read_bytes() == b"captured file bytes\n" * 16
    assert checkpoint.latest(run) == old
    assert _selected(run)["dir"] == str(old)


def test_staged_pool_link_crash_is_not_a_resume_candidate(tmp_path):
    run = tmp_path / "run"
    root = run / "checkpoints"
    old = _write_checkpoint(root, "0001-1", 1)
    (root / "LATEST").write_text(old.name + "\n")
    temporary = root / ".0002-2.tmp"
    staged_root = temporary / checkpoint.ARTIFACT_STAGE_DIR
    staged_file = staged_root / checkpoint.POOL_DIR / "wire" / "request.json"
    staged_file.parent.mkdir(parents=True)
    staged_file.write_bytes(b"staged pooled request\n")
    staged_file.chmod(0o444)
    published = root / checkpoint.POOL_DIR / "wire" / staged_file.name

    _kill_child("staged_pool_link", root=str(root), temporary=str(temporary),
                staged_root=str(staged_root))
    assert published.read_bytes() == staged_file.read_bytes()
    assert (temporary / checkpoint.ARTIFACT_TRANSACTION_FILE).is_file()
    assert not (root / "0002-2").exists()
    assert checkpoint.latest(run) == old
    assert _selected(run)["dir"] == str(old)


def test_snapshot_fsyncs_files_before_directories_and_flushes_staged_pool_directories(
        tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    nested = snapshot / "run" / "env"
    nested.mkdir(parents=True)
    files = [snapshot / "state.pkl", nested / "module.py"]
    for path in files:
        path.write_bytes(b"snapshot data")
    staged_pool = tmp_path / "staged" / checkpoint.POOL_DIR / "worker"
    staged_pool.mkdir(parents=True)
    new_pool_file = staged_pool / "blob"
    new_pool_file.write_bytes(b"pooled data")
    events: list[tuple[str, Path]] = []
    monkeypatch.setattr(checkpoint, "_fsync_file", lambda path: events.append(("file", Path(path))))
    monkeypatch.setattr(checkpoint, "_fsync_dir", lambda path: events.append(("dir", Path(path))))

    checkpoint._fsync_snapshot(snapshot, [new_pool_file])

    first_directory = next(index for index, (kind, _path) in enumerate(events) if kind == "dir")
    assert {path for kind, path in events[:first_directory] if kind == "file"} == {*files,
                                                                                   new_pool_file}
    directory_events = [path for kind, path in events if kind == "dir"]
    assert directory_events[:3] == [nested, snapshot / "run", snapshot]
    assert directory_events[-2:] == [staged_pool, staged_pool.parent]


@pytest.mark.parametrize("fsync_number", range(1, 8))
def test_kill_after_each_snapshot_fsync_keeps_checkpoint_unpublished(tmp_path, fsync_number):
    run = tmp_path / "run"
    root = run / "checkpoints"
    old = _write_checkpoint(root, "0001-1", 1)
    (root / "LATEST").write_text(old.name + "\n")
    temporary = root / ".0002-2.tmp"
    snapshot = temporary / "run"
    nested = snapshot / "env"
    nested.mkdir(parents=True)
    (snapshot / "state.pkl").write_bytes(b"state")
    (nested / "module.py").write_bytes(b"module")
    staged_pool = temporary / checkpoint.ARTIFACT_STAGE_DIR / checkpoint.POOL_DIR / "wire"
    staged_pool.mkdir(parents=True)
    new_pool_file = staged_pool / "request.json"
    new_pool_file.write_bytes(b"request")

    _kill_child("snapshot_fsync", snapshot=str(snapshot),
                new_pool_file=str(new_pool_file), fsync_number=fsync_number)
    assert checkpoint.latest(run) == old
    assert _selected(run)["dir"] == str(old)


@pytest.mark.parametrize("point", ("after_rename", "after_final_dir_fsync",
                                   "after_latest_file_fsync", "before_latest_replace",
                                   "after_latest_replace", "after_latest_dir_fsync"))
def test_legacy_resume_can_use_final_directory_before_latest_pointer_swap(tmp_path, point):
    run = tmp_path / "run"
    root = run / "checkpoints"
    old = _write_checkpoint(root, "0001-1", 1)
    (root / "LATEST").write_text(old.name + "\n")
    pending = _write_checkpoint(root, ".0002-2.tmp", 2)
    final = root / "0002-2"

    _kill_child("legacy_publish", root=str(root), pending=str(pending), final=str(final),
                run=str(run), point=point)
    if point in {"after_latest_replace", "after_latest_dir_fsync"}:
        assert checkpoint.latest(run) == final
    else:
        assert checkpoint.latest(run) == old
    selected = _selected(run)
    assert selected is not None and selected["dir"] == str(final)


@pytest.mark.parametrize("replace_stage", ("before_index_replace", "after_index_replace",
                                            "after_index_dir_fsync"))
def test_indexed_resume_changes_candidate_only_when_index_rename_happens(
        tmp_path, monkeypatch, replace_stage):
    monkeypatch.setenv(checkpoint._PUBLICATION_SWITCH, "on")
    run = tmp_path / "run"
    root = run / "checkpoints"
    old = _write_checkpoint(root, "0001-1", 1)
    (root / "LATEST").write_text(old.name + "\n")
    checkpoint._write_publication_index(root, 1, (old.name,))
    pending = _write_checkpoint(root, ".0002-2.tmp", 2)
    final = root / "0002-2"
    index_path = root / checkpoint._PUBLICATION_INDEX

    _kill_child("indexed_publish", root=str(root), pending=str(pending), final=str(final),
                run=str(run), index_path=str(index_path), replace_stage=replace_stage)
    if replace_stage == "before_index_replace":
        assert final.is_dir()
        assert (root / "LATEST").read_text().strip() == old.name
        assert checkpoint.publication_candidates(root) == (old,)
        assert _selected(run, atomic=True)["dir"] == str(old)
    else:
        assert checkpoint.publication_candidates(root) == (old, final)
        assert checkpoint.latest(run) == final
        selected = _selected(run, atomic=True)
        assert selected is not None and selected["dir"] == str(final)


def test_indexed_publication_fsyncs_renamed_directory_before_index_write(tmp_path, monkeypatch):
    monkeypatch.setenv(checkpoint._PUBLICATION_SWITCH, "on")
    run = tmp_path / "run"
    root = run / "checkpoints"
    old = _write_checkpoint(root, "0001-1", 1)
    (root / "LATEST").write_text(old.name + "\n")
    checkpoint._write_publication_index(root, 1, (old.name,))
    pending = _write_checkpoint(root, ".0002-2.tmp", 2)
    final = root / "0002-2"
    events: list[tuple[str, Path]] = []
    real_sync = publication.sync_directory
    real_write_index = checkpoint._write_publication_index

    def record_sync(path):
        events.append(("sync", Path(path)))
        real_sync(path)

    def record_index_write(path, seq, names):
        events.append(("index", Path(path)))
        real_write_index(path, seq, names)

    monkeypatch.setattr(publication, "sync_directory", record_sync)
    monkeypatch.setattr(checkpoint, "_write_publication_index", record_index_write)
    checkpoint._publish_checkpoint(root, pending, final, run)

    first_index_write = next(index for index, (kind, _path) in enumerate(events)
                             if kind == "index")
    assert ("sync", root) in events[:first_index_write]
    assert checkpoint.publication_candidates(root) == (old, final)


def test_resume_preflight_rejects_a_tampered_published_state(tmp_path):
    run = tmp_path / "run"
    root = run / "checkpoints"
    folder = _write_checkpoint(root, "0001-1", 1)
    (root / "LATEST").write_text(folder.name + "\n")
    state = folder / checkpoint.STATE_FILE
    state.write_bytes(state.read_bytes()[:-1] + b"x")

    assert checkpoint.latest(run) == folder
    assert _selected(run) is None


def test_resume_selection_ignores_valid_dot_prefixed_checkpoint(tmp_path):
    run = tmp_path / "run"
    root = run / "checkpoints"
    old = _write_checkpoint(root, "0001-1", 1)
    (root / "LATEST").write_text(old.name + "\n")
    hidden = _write_checkpoint(root, ".0002-2.tmp", 2)

    assert (hidden / checkpoint.META_FILE).is_file()
    selected = _selected(run)
    assert selected is not None and selected["dir"] == str(old)


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--crash-worker":
        raise SystemExit("expected --crash-worker OPERATION JSON_PAYLOAD")
    _crash_worker(sys.argv[2], json.loads(sys.argv[3]))
