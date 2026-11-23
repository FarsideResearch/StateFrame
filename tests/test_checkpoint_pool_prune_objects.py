"""Pool pruning must ignore the shared object store and fail closed on bad checkpoints."""
from __future__ import annotations

import json
import os
import pickle
import random
import subprocess
import sys
from pathlib import Path

import pytest

from stateframe.engine import kernel_checkpoint as checkpoint


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


def _save(run: Path, step: int) -> Path:
    checkpoint._STATE["setup"] = {"code": sys._getframe().f_code, "line": 0, "funcs": frozenset()}
    return checkpoint.save(sys._getframe(), step, run)


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
    pool_bytes = _allocated_pool_bytes(run / "checkpoints" / checkpoint.POOL_DIR)
    print(f"POOL_BYTES_AFTER_20_SAVES={pool_bytes}")
