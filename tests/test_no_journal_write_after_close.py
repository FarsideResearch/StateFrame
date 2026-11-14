"""Class killer: no journal write may happen after the journal writer is closed.

Every live run used to end with ``final_product_no_candidate recorded to sidecar (journal unavailable
after finish: ...)`` because work()'s finally block recorded the fact after finish() had closed the
writer. The fact now travels through the open journal from a terminal-fact hook inside finish(); the
sidecar is a last resort for a genuinely broken writer, not the normal shutdown order.

The acceptance half drives the public CLI against a scripted endpoint with a sitecustomize that
instruments the real journal writer: any emit after close() is appended to a violation file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from _journal import measurement_events
from acceptance._support import run_case
from kernel_stateframe_runtime import StateFrameRuntime

POST_FINISH_NOTE = "journal unavailable after finish"

# Imported by the engine child at interpreter start-up (PYTHONPATH precedes site), so the
# instrumentation covers every writer the run opens, whatever module closes it.
SITECUSTOMIZE = '''
import os, traceback
import kernel_continuation_journal as _kcj

_VIOLATIONS = os.environ["JOURNAL_VIOLATION_LOG"]
_original_emit = _kcj.ContinuationJournal.emit
_original_close = _kcj.ContinuationJournal.close


def _emit(self, kind, **fields):
    if getattr(self, "_test_closed", False):
        with open(_VIOLATIONS, "a") as handle:
            handle.write(kind + "\\n" + "".join(traceback.format_stack(limit=12)) + "\\n---\\n")
    return _original_emit(self, kind, **fields)


def _close(self):
    try:
        return _original_close(self)
    finally:
        self._test_closed = True


_kcj.ContinuationJournal.emit = _emit
_kcj.ContinuationJournal.close = _close
'''


def _kinds(run_dir: Path) -> list[str]:
    return [str(event.get("kind")) for event in measurement_events(run_dir)]


def test_scripted_run_journals_terminal_fact_before_close(tmp_path):
    hook_dir = tmp_path / "hook"
    hook_dir.mkdir()
    (hook_dir / "sitecustomize.py").write_text(SITECUSTOMIZE)
    violations = tmp_path / "violations.log"
    case = run_case(
        tmp_path, "one_shell.script.json", substitutions={"COMMAND": "echo ok"},
        default_test_cmd=False,
        env_overrides={
            "JOURNAL_VIOLATION_LOG": str(violations),
            "PYTHONPATH": os.pathsep.join((
                str(hook_dir),
                str(Path(__file__).resolve().parents[1] / "src"),
                str(Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"))),
        })
    assert case.process.returncode == 0, case.process.stderr[-3000:]
    assert not violations.exists(), violations.read_text()
    kinds = _kinds(case.run_dir)
    assert "final_product_no_candidate" in kinds, kinds[-12:]
    assert "attempt_terminated" in kinds
    assert kinds.index("final_product_no_candidate") < kinds.index("attempt_terminated")
    assert POST_FINISH_NOTE not in case.process.stderr, case.process.stderr[-3000:]
    assert not list(case.run_dir.rglob("post_finish_facts.jsonl"))


class _Journal:
    def __init__(self) -> None:
        self.closed = False
        self.events: list[tuple[str, dict[str, object]]] = []
        self.late: list[str] = []

    def emit(self, kind: str, **fields: object) -> dict[str, object]:
        if self.closed:
            self.late.append(kind)
            raise RuntimeError("worker is closed, inherited or poisoned (closed)")
        self.events.append((kind, fields))
        return {"kind": kind, **fields}

    def close(self) -> None:
        self.closed = True


def _runtime(journal: _Journal) -> StateFrameRuntime:
    runtime = object.__new__(StateFrameRuntime)
    runtime.journal = journal
    runtime._terminal_fact_hooks = []
    return runtime


def test_hooks_run_once_with_the_terminal_result_and_are_fail_open():
    journal = _Journal()
    runtime = _runtime(journal)
    seen: list[dict[str, object]] = []

    def record(terminal_result):
        seen.append(dict(terminal_result))
        journal.emit("final_product_no_candidate", reason="not_completed")

    def broken(terminal_result):
        raise ValueError("bookkeeping exploded")

    runtime.register_terminal_fact_hook(record)
    runtime.register_terminal_fact_hook(broken)
    runtime._run_terminal_fact_hooks({"result": "SUBMITTED"})
    runtime._run_terminal_fact_hooks({"result": "SUBMITTED"})  # hooks are consumed: never twice
    journal.close()
    assert seen == [{"result": "SUBMITTED"}]
    assert [kind for kind, _ in journal.events] == [
        "final_product_no_candidate", "terminal_fact_hook_error"]
    error = journal.events[1][1]
    assert error["error_type"] == "ValueError" and error["hook"].endswith("broken")
    assert journal.late == []


def test_finish_runs_hooks_before_attempt_terminated_and_close():
    """The hook call sits ahead of the attempt_terminated row and of the continuation close."""
    import inspect

    source = inspect.getsource(StateFrameRuntime.finish)
    hook_at = source.index("self._run_terminal_fact_hooks(terminal_result)")
    assert hook_at < source.index('self.journal.emit("attempt_terminated"')
    assert hook_at < source.index("self.journal.close()")


@pytest.mark.parametrize("state", ["closed", "poisoned"])
def test_post_finish_recorder_still_falls_back_for_a_broken_writer(tmp_path, state, capsys):
    """The sidecar stays as the last-resort path for a writer that is genuinely unusable."""
    from probes.react_repair import _record_post_finish_fact

    class Broken:
        path = tmp_path / "events.jsonl"

        def emit(self, kind, **fields):
            raise RuntimeError(f"worker is closed, inherited or poisoned ({state})")

    destination = _record_post_finish_fact(SimpleNamespace(journal=Broken()), "final_product_no_candidate", reason="x")
    assert destination == "sidecar"
    rows = [json.loads(line) for line in (tmp_path / "post_finish_facts.jsonl").read_text().splitlines()]
    assert rows == [{"kind": "final_product_no_candidate", "reason": "x"}]
    assert POST_FINISH_NOTE in capsys.readouterr().err


def test_finish_keeps_its_host_phase_wrapper():
    # The hook methods were once inserted between @_host_phase("finalize") and finish(), which moved the
    # decorator onto register_terminal_fact_hook and silently dropped finish()'s host-phase timing.
    assert hasattr(StateFrameRuntime.finish, "__wrapped__")
    assert not hasattr(StateFrameRuntime.register_terminal_fact_hook, "__wrapped__")
