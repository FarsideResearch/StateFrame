"""Per-turn editable product digests must not re-hash unchanged files on every call.

Covers the shared stat-identity digest memo used by kernel_green_lock's workspace
digests: perf (no re-reads when nothing changed), parity (digests and verdicts
match real file bytes across edit/replace/delete/create), and the racy-timestamp
rule (a file written less than 2 s ago is always re-hashed).
"""
from __future__ import annotations

import hashlib
import io
import builtins
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import kernel_green_lock as green_lock
import kernel_live_projection as projection

PERF_FILE_COUNT = 5000
RACY_SLEEP_S = 2.1


def _sha(path: Path) -> str | None:
    if not path.exists():
        return None
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime(root: Path, paths: tuple[str, ...], parent: dict[str, object]) -> SimpleNamespace:
    owner = SimpleNamespace(
        root=str(root),
        task_row={"editable_paths": list(paths), "editable_all": False},
    )
    lock = SimpleNamespace(
        _starting_editable_product_files=dict(parent),
        _starting_editable_product_paths=tuple(paths),
        _starting_binding=None,
        _starting_revision=None,
    )
    return SimpleNamespace(
        owner=owner,
        binding=lambda: {"product_files": {}},
        journal=None,
        green_lock=lock,
    )


@pytest.fixture
def open_spy(monkeypatch):
    """Count file opens under a root. Path.read_bytes goes through io.open."""
    state = {"root": "", "opens": []}
    real_open = io.open

    def spy(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and str(file).startswith(state["root"]):
            state["opens"].append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(io, "open", spy)
    monkeypatch.setattr(builtins, "open", spy)
    return state


def _write_tree(root: Path, count: int) -> tuple[str, ...]:
    paths = []
    for index in range(count):
        rel = f"src/pkg/m{index:04d}.py"
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"VALUE = {index}\n", encoding="utf-8")
        paths.append(rel)
    return tuple(paths)


def test_second_turn_does_not_reread_unchanged_product_files(tmp_path, open_spy):
    root = tmp_path / "repo"
    root.mkdir()
    paths = _write_tree(root, PERF_FILE_COUNT)
    # Files written this instant are inside the racy window and are never memoized.
    time.sleep(RACY_SLEEP_S)

    parent = green_lock._workspace_product_file_digests(
        _runtime(root, paths, {}), paths)
    assert parent is not None and len(parent) == PERF_FILE_COUNT
    runtime = _runtime(root, paths, parent)
    runtime.binding = lambda: projection.workspace_binding(root, "test")

    open_spy["root"] = str(root)
    open_spy["opens"].clear()
    start = time.perf_counter()
    first = green_lock.product_revision_change_from_parent(runtime)
    first_elapsed = time.perf_counter() - start
    first_opens = len(open_spy["opens"])

    open_spy["opens"].clear()
    start = time.perf_counter()
    second = green_lock.product_revision_change_from_parent(runtime)
    second_elapsed = time.perf_counter() - start
    second_opens = len(open_spy["opens"])

    assert first == second == (False, second[1], "product_file_digests_match_parent")
    assert second_opens < 50, (
        f"second turn re-read {second_opens} of {PERF_FILE_COUNT} unchanged files "
        f"(first turn read {first_opens})")
    assert second_elapsed < 5.0, f"second turn took {second_elapsed:.2f}s (first {first_elapsed:.2f}s)"


def test_racy_file_is_rehashed_every_call(tmp_path, open_spy):
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "src" / "fresh.py"
    target.parent.mkdir(parents=True)
    target.write_text("A = 1\n", encoding="utf-8")
    paths = ("src/fresh.py",)

    open_spy["root"] = str(root)
    open_spy["opens"].clear()
    first = green_lock._workspace_product_file_digests(_runtime(root, paths, {}), paths)
    second = green_lock._workspace_product_file_digests(_runtime(root, paths, {}), paths)
    # Written < 2 s ago: both calls must read it, never trust a memo entry.
    opens = len(open_spy["opens"])
    assert first == second == {"src/fresh.py": _sha(target)}
    assert opens == 2


def test_in_place_same_size_edit_inside_racy_window_is_detected(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "src" / "same.py"
    target.parent.mkdir(parents=True)
    target.write_text("A = 1\n", encoding="utf-8")
    paths = ("src/same.py",)
    before = green_lock._workspace_product_file_digests(_runtime(root, paths, {}), paths)
    target.write_text("A = 2\n", encoding="utf-8")
    after = green_lock._workspace_product_file_digests(_runtime(root, paths, {}), paths)
    assert before != after
    assert after == {"src/same.py": _sha(target)}


@pytest.mark.parametrize("scenario", [
    "unchanged",
    "in_place_edit_mtime_restored",
    "replaced_via_rename",
    "deleted",
    "created",
    "edited_inside_racy_window",
])
def test_digests_and_verdicts_match_real_bytes(tmp_path, scenario):
    root = tmp_path / "repo"
    root.mkdir()
    paths = _write_tree(root, 12)
    time.sleep(RACY_SLEEP_S)
    watched = paths[0]
    target = root / watched
    if scenario == "created":
        # Editable path absent at turn start, created during the turn.
        target.unlink()
    parent = green_lock._workspace_product_file_digests(_runtime(root, paths, {}), paths)
    assert parent is not None
    assert parent[watched] == _sha(target)

    stat_before = os.stat(target) if target.exists() else None
    if scenario == "in_place_edit_mtime_restored":
        # Same size as "VALUE = 0\n"; mtime restored, ctime still moves.
        target.write_text("VALUE = 9\n", encoding="utf-8")
        os.utime(target, ns=(stat_before.st_atime_ns, stat_before.st_mtime_ns))
    elif scenario == "replaced_via_rename":
        staged = root / "staged.tmp"
        staged.write_text("VALUE = 777\n", encoding="utf-8")
        os.replace(staged, target)
    elif scenario == "deleted":
        target.unlink()
    elif scenario == "created":
        target.write_text("VALUE = 42\n", encoding="utf-8")
    elif scenario == "edited_inside_racy_window":
        target.write_text("VALUE = 5\n", encoding="utf-8")
        target.write_text("VALUE = 6\n", encoding="utf-8")

    expected_files = {rel: _sha(root / rel) for rel in paths}
    runtime = _runtime(root, paths, parent)
    assert green_lock._workspace_product_file_digests(runtime, paths) == expected_files
    changed, _revision, reason = green_lock.product_revision_change_from_parent(runtime)
    if scenario == "unchanged":
        assert (changed, reason) == (False, "product_file_digests_match_parent")
    else:
        assert (changed, reason) == (True, "product_file_digests_differ_from_parent")


@pytest.mark.parametrize("scenario", ["ages_during_read", "clock_rollback"])
def test_projection_racy_identity_never_reuses_old_bytes(tmp_path, monkeypatch, scenario):
    target = tmp_path / "racy.py"
    target.write_bytes(b"old")
    frozen = os.lstat(target)
    stamp = max(frozen.st_mtime_ns, frozen.st_ctime_ns)
    now = [stamp + (1 if scenario == "ages_during_read" else 3_000_000_000)]
    real_lstat = os.lstat
    monkeypatch.setattr(os, "lstat", lambda path, *a, **kw:
                        frozen if str(path) == str(target) else real_lstat(path, *a, **kw))
    monkeypatch.setattr(time, "time_ns", lambda: now[0])
    real_digest = hashlib.sha256
    def advancing_digest(*args, **kwargs):
        result = real_digest(*args, **kwargs)
        now[0] = stamp + 3_000_000_000
        return result
    # Both the base streaming hasher and the port call hashlib.sha256.
    monkeypatch.setattr(hashlib, "sha256", advancing_digest)
    real_file_digest = hashlib.file_digest
    def advancing_file_digest(*args, **kwargs):
        result = real_file_digest(*args, **kwargs)
        now[0] = stamp + 3_000_000_000
        return result
    monkeypatch.setattr(hashlib, "file_digest", advancing_file_digest)
    assert projection._file_value(str(target)) == "sha256:" + real_digest(b"old").hexdigest()
    target.write_bytes(b"new")
    if scenario == "clock_rollback":
        now[0] = stamp + 1
    assert projection._file_value(str(target)) == "sha256:" + real_digest(b"new").hexdigest()


def test_projection_rejects_identity_change_during_hash(tmp_path, monkeypatch):
    target = tmp_path / "changing.py"
    target.write_bytes(b"old")
    real_digest = hashlib.sha256
    real_file_digest = hashlib.file_digest
    def mutate():
        target.write_bytes(b"new and longer")
    def racing_digest(*args, **kwargs):
        result = real_digest(*args, **kwargs)
        mutate()
        return result
    def racing_file_digest(*args, **kwargs):
        result = real_file_digest(*args, **kwargs)
        mutate()
        return result
    monkeypatch.setattr(hashlib, "sha256", racing_digest)
    monkeypatch.setattr(hashlib, "file_digest", racing_file_digest)
    with pytest.raises(OSError, match="changed"):
        projection._file_value(str(target))


def test_view_rejects_mutation_between_binding_and_signature_capture(tmp_path, monkeypatch):
    target = tmp_path / "value.py"
    target.write_text("old", encoding="utf-8")
    store = projection.WorkspaceViewStore(tmp_path, "test")
    real_binding = projection.workspace_binding
    def racing_binding(*args, **kwargs):
        result = real_binding(*args, **kwargs)
        target.write_text("new and longer", encoding="utf-8")
        return result
    monkeypatch.setattr(projection, "workspace_binding", racing_binding)
    with pytest.raises(OSError, match="changed"):
        store.current_view()
    monkeypatch.setattr(projection, "workspace_binding", real_binding)
    assert store.current_view().product_files["value.py"] == _sha(target)
