"""Bound workspace reads at the real post-green provider-request gate.

The reference takes the same gate with independent, uncached byte digests.
Only fixture timestamp settling uses a sleep; performance is measured in reads.
"""
from __future__ import annotations

import builtins
import hashlib
import io
import os
import time
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import kernel_green_lock as green_lock
import pytest
from kernel_completion_owner import postgreen_done_offer
from kernel_verdict import SATISFIED, evaluate_required_tests


def _uncached_digests(runtime: SimpleNamespace, paths: Sequence[str]) -> dict[str, object]:
    return {
        path: "sha256:" + hashlib.sha256(
            (Path(runtime.owner.root) / path).read_bytes()).hexdigest()
        for path in paths
    }


def _runtime(root: Path, paths: tuple[str, ...]) -> SimpleNamespace:
    runtime = SimpleNamespace(
        owner=SimpleNamespace(
            root=str(root), current_step=0,
            task_row={"editable_paths": list(paths), "editable_all": False}),
        max_steps=10, journal=None, revision="parent", test_receipts=[],
    )
    runtime.binding = lambda: {"product_revision": runtime.revision}
    runtime.public_test_verdict = lambda: evaluate_required_tests(
        runtime.binding(), ("tests/test_public.py::test_value",), runtime.test_receipts)
    return runtime


def _advance(runtime: SimpleNamespace, turn: int) -> None:
    runtime.owner.current_step = turn
    runtime.current_request_id = f"request-{turn}"
    # Deliberately distinct even when bytes return to the parent: workspace
    # identity, rather than a revision label alone, must govern the done offer.
    runtime.revision = f"revision-{turn}"
    binding = runtime.binding()
    runtime.test_receipts = [{
        "kind": "test_subprocess_completed", "producer": "test_subprocess",
        "receipt_id": f"public-{turn}", "sequence": turn,
        "selectors": ["tests/test_public.py::test_value"],
        "binding": binding, "after_binding": binding,
        "binding_stable": True, "exit_code": 0,
    }]
    assert runtime.public_test_verdict().state == SATISFIED


def test_green_lock_turn_reads_match_uncached_verdict(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Ambient profiles must not add unrelated completion policies to this gate.
    for key in tuple(os.environ):
        if key.startswith("STATEFRAME_") and key != "STATEFRAME_HOME":
            monkeypatch.delenv(key)
    root = tmp_path / "workspace"
    (root / "src").mkdir(parents=True)
    paths = tuple(f"src/module_{index:04d}.py" for index in range(3500))
    original = {path: f"VALUE = {index}\n" for index, path in enumerate(paths)}
    for path, content in original.items():
        (root / path).write_text(content, encoding="utf-8")
    # Real ctime cannot be backdated. Let initial files leave the memo's racy
    # window without altering its clock or weakening its admission predicate.
    time.sleep(2.1)

    measured = _runtime(root, paths)
    reference = _runtime(root, paths)
    with monkeypatch.context() as patch:
        patch.setattr(green_lock, "_workspace_product_file_digests", _uncached_digests)
        green_lock.GreenLock(reference, mode="terminal")

    reads: list[str] = []
    real_open = io.open

    def counting_open(file, mode="r", *args, **kwargs):
        if (isinstance(file, (str, os.PathLike)) and "r" in mode
                and Path(file).is_relative_to(root)):
            reads.append(os.fspath(file))
        return real_open(file, mode, *args, **kwargs)

    counts: list[int] = []
    verdicts: list[tuple[bool, str]] = []
    # One edit, then two edits, then two edits returning to the original bytes.
    edits = (
        {paths[0]: "VALUE = -1\n"},
        {paths[0]: "VALUE = -2\n", paths[1]: "VALUE = -3\n"},
        {paths[0]: original[paths[0]], paths[1]: original[paths[1]]},
    )
    with monkeypatch.context() as patch:
        patch.setattr(io, "open", counting_open)
        patch.setattr(builtins, "open", counting_open)
        green_lock.GreenLock(measured, mode="terminal")
        assert len(reads) == len(paths), "cold lock must capture all 3,500 product files"
        for turn, changes in enumerate(edits, start=1):
            if turn > 1:
                reads.clear()
            for path, content in changes.items():
                (root / path).write_text(content, encoding="utf-8")
                measured.green_lock.applied_product_edit(path)
                reference.green_lock.applied_product_edit(path)
            for runtime in (measured, reference):
                _advance(runtime, turn)
            verdict = postgreen_done_offer(measured)
            counts.append(len(reads))
            with monkeypatch.context() as reference_patch:
                reference_patch.setattr(
                    green_lock, "_workspace_product_file_digests", _uncached_digests)
                reference_reads = len(reads)
                expected = postgreen_done_offer(reference)
                assert len(reads) - reference_reads == len(paths)
            assert verdict == expected, f"turn {turn}: cached and uncached gates disagree"
            verdicts.append(verdict)

    assert [offered for offered, _ in verdicts] == [True, True, False]
    print(f"green-lock reads by turn: {counts}; done offers: {verdicts}")
    for turn, count in enumerate(counts[1:], start=2):
        assert count <= 5, f"turn {turn} read {count} product files; budget is 5"
