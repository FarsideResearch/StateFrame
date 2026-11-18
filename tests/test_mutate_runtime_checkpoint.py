"""A mutation run restores its engine checkpoint on resume without its process-bound executor.

The executor (supervisor process, launch ledger, runtime root) belongs to the process that saved
the checkpoint, and resume sweeps its runtime root before the engine restarts. A checkpoint must
therefore carry the model-visible mutation state but never the executor; the resuming host's
settings are rebound onto the restored runtime, and the child finishes the restored runtime.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

import kernel_checkpoint
import kernel_mutate_runtime
from kernel_mutate_policy import MutationPolicy
from kernel_mutate_records import Journal
from kernel_mutate_runtime import MutationRuntime
from stateframe import _child

_CTX = {"framecells": {}, "framefuncs": {}, "cvars": {}, "hostfns": {}, "participants": {}}


class _LiveExecutor:
    """Stands in for a live MutationExecutor; pickling it would be a bug."""

    def __init__(self, root: Path):
        self.root = root

    def __reduce__(self):
        raise AssertionError("a live mutation executor must not be pickled into a checkpoint")


def _runtime(tmp_path: Path, **overrides) -> MutationRuntime:
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    fields = dict(level="authored", journal=Journal(run), policy=MutationPolicy(level="authored"),
                  repo=tmp_path, runtime_root=tmp_path / "rt" / ("a" * 32))
    fields.update(overrides)
    return MutationRuntime(**fields)


def _round_trip(value):
    blob = kernel_checkpoint.dumps({"runtime": value}, _CTX)
    return kernel_checkpoint._Unpickler(io.BytesIO(blob), {"framecells": {}, "framefuncs": {}}).load()["runtime"]


@pytest.fixture
def installed():
    saved = (kernel_mutate_runtime._active, kernel_mutate_runtime._host)
    yield
    kernel_mutate_runtime._active, kernel_mutate_runtime._host = saved


def test_checkpoint_drops_a_live_executor_and_keeps_model_visible_state(tmp_path):
    runtime = _runtime(tmp_path)
    live = _LiveExecutor(runtime.runtime_root)
    runtime.executor = live
    runtime.ready["reader"] = {"manifest": {"name": "reader"}, "files": {"main.py": b"x"}}
    runtime.last = "use reader"
    restored = _round_trip(runtime)
    assert restored.executor is None
    # Without a factory the restored runtime recreates its own (swept) root on the next launch.
    assert restored._reopen_runtime_root is True
    assert restored.ready == runtime.ready and restored.last == "use reader"
    assert restored.runtime_root == runtime.runtime_root
    # Saving never changes the live runtime.
    assert runtime.executor is live and runtime._reopen_runtime_root is False


def test_checkpoint_of_a_factory_runtime_reopens_through_the_factory(tmp_path):
    runtime = _runtime(tmp_path, runtime_root=None, executor_factory=None)
    runtime.executor_factory = kernel_mutate_runtime.service  # any picklable callable
    runtime.runtime_root = tmp_path / "rt" / ("b" * 32)
    runtime.executor = _LiveExecutor(runtime.runtime_root)
    restored = _round_trip(runtime)
    assert restored.executor is None
    assert restored._reopen_runtime_root is False


def test_checkpoint_without_an_executor_keeps_reopen_state(tmp_path):
    runtime = _runtime(tmp_path)
    assert _round_trip(runtime)._reopen_runtime_root is False
    runtime._reopen_runtime_root = True  # a retired hook executor is still pending reopen
    assert _round_trip(runtime)._reopen_runtime_root is True


def test_restored_runtime_reopens_a_swept_root_on_next_launch(tmp_path, monkeypatch):
    import kernel_mutate_exec

    runtime = _runtime(tmp_path)
    runtime.executor = _LiveExecutor(runtime.runtime_root)
    restored = _round_trip(runtime)
    made = []

    class _Fresh:
        def __init__(self, root, *, network_envelope):
            assert Path(root).is_dir()  # recreated after the sweep removed it
            self.root = Path(root)
            made.append(network_envelope)

    monkeypatch.setattr(kernel_mutate_exec, "MutationExecutor", _Fresh)
    runtime.runtime_root.parent.mkdir(parents=True)
    executor = restored._ensure_executor()
    assert isinstance(executor, _Fresh) and made == ["none"]
    assert restored._reopen_runtime_root is False


def test_rebind_carries_resuming_host_settings_onto_the_restored_runtime(tmp_path, installed):
    saved = _runtime(tmp_path, replay_external_allowed=False)
    saved.attempt = 1
    saved.ready["reader"] = {"manifest": {"name": "reader"}, "files": {}}
    restored = _round_trip(saved)
    host = _runtime(tmp_path, replay_external_allowed=True, executor_factory=lambda: None)
    host.attempt = 2
    kernel_mutate_runtime.install(host)
    kernel_mutate_runtime._active = restored  # what a checkpoint restore does to module globals
    kernel_mutate_runtime.rebind_after_checkpoint()
    assert kernel_mutate_runtime.service() is restored
    assert restored.replay_external_allowed is True
    assert restored.executor_factory is host.executor_factory
    assert restored.attempt == 2
    assert "reader" in restored.ready
    assert kernel_mutate_runtime._host is restored
    # A second call (or a run without a restore) changes nothing.
    restored.replay_external_allowed = False
    kernel_mutate_runtime.rebind_after_checkpoint()
    assert restored.replay_external_allowed is False


def test_rebind_is_a_no_op_without_mutations(installed):
    kernel_mutate_runtime.install(None)
    kernel_mutate_runtime.rebind_after_checkpoint()
    assert kernel_mutate_runtime.service() is None and kernel_mutate_runtime._host is None


def test_host_runtime_global_is_never_saved_or_restored():
    assert ("kernel_mutate_runtime", "_host") in kernel_checkpoint.HOST_GLOBALS
    assert ("kernel_mutate_runtime", "_active") not in kernel_checkpoint.HOST_GLOBALS
    state = kernel_checkpoint._module_state("kernel_mutate_runtime", kernel_mutate_runtime)
    assert "_host" not in state and "_active" in state


def test_child_finishes_the_restored_runtime(installed, monkeypatch):
    host, restored = object(), object()
    monkeypatch.setitem(sys.modules, "kernel_mutate_runtime", kernel_mutate_runtime)
    kernel_mutate_runtime._active = restored
    assert _child._active_mutation_runtime(host) is restored
    kernel_mutate_runtime._active = None
    assert _child._active_mutation_runtime(host) is host
    monkeypatch.delitem(sys.modules, "kernel_mutate_runtime")
    assert _child._active_mutation_runtime(host) is host


def _commit(journal: Journal, call: str, path: str, pre: bytes | None, post: bytes | None) -> None:
    pre_blob = journal.put_blob(pre) if pre is not None else None
    post_blob = journal.put_blob(post) if post is not None else None
    journal.append("EFFECT_COMMITTED", call=call, files=[{
        "path": path, "pre_sha256": pre_blob, "post_sha256": post_blob,
        "pre_blob": pre_blob, "post_blob": post_blob}], root_identity=[0, 0])


def _workspace_runtime(tmp_path: Path) -> MutationRuntime:
    run = tmp_path / "run"
    workspace = run / "workspace"
    workspace.mkdir(parents=True)
    return _runtime(tmp_path, run_dir=run, repo=workspace)


def test_restore_rolls_forward_edits_committed_after_the_checkpoint(tmp_path, installed):
    saved = _workspace_runtime(tmp_path)
    workspace = saved.repo
    (workspace / "a.txt").write_bytes(b"B")
    _commit(saved.journal, "c1", "a.txt", b"A", b"B")
    restored = _round_trip(saved)  # the checkpoint holds a.txt == B
    _commit(saved.journal, "c2", "a.txt", b"B", b"C")
    (workspace / "sub").mkdir()
    _commit(saved.journal, "c3", "sub/new.txt", None, b"N")
    (workspace / "gone.txt").write_bytes(b"G")
    _commit(saved.journal, "c4", "gone.txt", b"G", None)
    # A later reconciled-but-already-applied file is left alone.
    (workspace / "same.txt").write_bytes(b"S2")
    _commit(saved.journal, "c5", "same.txt", b"S1", b"S2")
    host = _runtime(tmp_path, run_dir=saved.run_dir, repo=workspace, executor_factory=lambda: None)
    kernel_mutate_runtime.install(host)
    kernel_mutate_runtime._active = restored
    kernel_mutate_runtime.rebind_after_checkpoint()
    assert (workspace / "a.txt").read_bytes() == b"C"  # c1 was in the checkpoint, not applied twice
    assert (workspace / "sub" / "new.txt").read_bytes() == b"N"
    assert not (workspace / "gone.txt").exists()
    assert (workspace / "same.txt").read_bytes() == b"S2"
    assert "_checkpoint_committed_effects" not in vars(restored)


def test_restore_refuses_a_file_that_is_neither_pre_nor_postimage(tmp_path, installed):
    from kernel_mutate_edits import EditConflict

    saved = _workspace_runtime(tmp_path)
    restored = _round_trip(saved)
    (saved.repo / "a.txt").write_bytes(b"unexpected")
    _commit(saved.journal, "c1", "a.txt", b"A", b"B")
    kernel_mutate_runtime.install(_runtime(tmp_path, run_dir=saved.run_dir, repo=saved.repo))
    kernel_mutate_runtime._active = restored
    with pytest.raises(EditConflict):
        kernel_mutate_runtime.rebind_after_checkpoint()
    assert (saved.repo / "a.txt").read_bytes() == b"unexpected"


def test_mutation_journal_is_live_run_data_never_checkpointed():
    assert "mutations" in kernel_checkpoint.SKIP_RUN_ENTRIES


def _restore_with(saved: MutationRuntime, tmp_path: Path, restored: MutationRuntime) -> None:
    kernel_mutate_runtime.install(_runtime(tmp_path, run_dir=saved.run_dir, repo=saved.repo))
    kernel_mutate_runtime._active = restored
    kernel_mutate_runtime.rebind_after_checkpoint()


def test_roll_forward_is_all_or_nothing_when_a_later_row_conflicts(tmp_path, installed):
    from kernel_mutate_edits import EditConflict

    saved = _workspace_runtime(tmp_path)
    restored = _round_trip(saved)
    (saved.repo / "a.txt").write_bytes(b"A")
    (saved.repo / "b.txt").write_bytes(b"unexpected")
    _commit(saved.journal, "c1", "a.txt", b"A", b"B")
    _commit(saved.journal, "c2", "b.txt", b"X", b"Y")
    with pytest.raises(EditConflict):
        _restore_with(saved, tmp_path, restored)
    # The first row validated, but nothing is written unless every row does.
    assert (saved.repo / "a.txt").read_bytes() == b"A"
    assert (saved.repo / "b.txt").read_bytes() == b"unexpected"


def test_roll_forward_chains_rows_that_touch_the_same_file(tmp_path, installed):
    saved = _workspace_runtime(tmp_path)
    restored = _round_trip(saved)
    (saved.repo / "a.txt").write_bytes(b"A")
    _commit(saved.journal, "c1", "a.txt", b"A", b"B")
    _commit(saved.journal, "c2", "a.txt", b"B", b"C")  # its preimage is c1's postimage
    _restore_with(saved, tmp_path, restored)
    assert (saved.repo / "a.txt").read_bytes() == b"C"


def test_roll_forward_checks_a_later_row_against_the_earlier_rows_postimage(tmp_path, installed):
    from kernel_mutate_edits import EditConflict

    saved = _workspace_runtime(tmp_path)
    restored = _round_trip(saved)
    (saved.repo / "a.txt").write_bytes(b"A")
    _commit(saved.journal, "c1", "a.txt", b"A", b"B")
    # Valid against the disk (A), but after c1 the file holds B: neither pre nor postimage.
    _commit(saved.journal, "c2", "a.txt", b"A", b"C")
    with pytest.raises(EditConflict):
        _restore_with(saved, tmp_path, restored)
    assert (saved.repo / "a.txt").read_bytes() == b"A"


def test_roll_forward_chains_a_create_then_delete_of_one_file(tmp_path, installed):
    saved = _workspace_runtime(tmp_path)
    restored = _round_trip(saved)
    _commit(saved.journal, "c1", "t.txt", None, b"T")
    _commit(saved.journal, "c2", "t.txt", b"T", None)
    _restore_with(saved, tmp_path, restored)
    assert not (saved.repo / "t.txt").exists()


def test_roll_forward_undoes_earlier_writes_when_a_later_write_fails(tmp_path, installed, monkeypatch):
    import kernel_mutate_edits
    from kernel_mutate_edits import EditConflict

    saved = _workspace_runtime(tmp_path)
    restored = _round_trip(saved)
    (saved.repo / "a.txt").write_bytes(b"A")
    (saved.repo / "b.txt").write_bytes(b"X")
    _commit(saved.journal, "c1", "a.txt", b"A", b"B")
    _commit(saved.journal, "c2", "b.txt", b"X", b"Y")
    real_write, calls = kernel_mutate_edits._write_at, []

    def failing_write(parent_fd, name, data, mode):
        calls.append(name)
        if len(calls) == 2:
            raise OSError("injected write failure")
        real_write(parent_fd, name, data, mode)

    monkeypatch.setattr(kernel_mutate_edits, "_write_at", failing_write)
    with pytest.raises(EditConflict):
        _restore_with(saved, tmp_path, restored)
    assert (saved.repo / "a.txt").read_bytes() == b"A"
    assert (saved.repo / "b.txt").read_bytes() == b"X"
    assert sorted(path.name for path in saved.repo.iterdir()) == ["a.txt", "b.txt"]
