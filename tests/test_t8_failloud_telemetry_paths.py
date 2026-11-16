"""Fail-loud telemetry: broad except sites in the reason log, probe, safe probe, run vitals and fault disposition.

Scope: kernel_fault_disposition.py, kernel_probe.py, kernel_reason_log.py, kernel_run_vitals.py and
kernel_safe_probe.py (sites triaged at base 7e5c5598; reason_log lines as of the T8 telemetry fix). Every broad except in these modules is listed in the inventory guard
below with a verdict. The loudness bar: a raise, a WARNING-or-higher record from the owning module, or a typed
field or counter that a reader checks. Telemetry that must not fail its caller is allowed to return a typed
outcome (False, None, a status field) and count its own failure, so those sites are pinned to that outcome.

A strict xfail is used only for a CRASH-HIDER on a default (not switch-gated) path. A fix flips it to XPASS,
which fails the run until the test is promoted to a plain assertion. Tag convention: T8-<MODULE>-<line>.
"""
from __future__ import annotations

import ast
import io
import re
import sys
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"
_BROAD = re.compile(r"^\s*except\s*(?:\(|Exception\b|BaseException\b|:)")


class _Journal:
    """Journal stand-in that records emitted kwargs."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit(self, kind: str, **fields: object) -> None:
        self.events.append((kind, fields))


class _RaisingJournal:
    def emit(self, kind: str, **fields: object) -> None:
        raise RuntimeError("journal closed")


class _BrokenStream:
    def write(self, _text: str) -> int:
        raise OSError("stream closed")

    def flush(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _fresh_reason_log():
    import kernel_reason_log as rl

    rl.reset_for_tests()
    yield
    rl.reset_for_tests()


# --- kernel_fault_disposition -----------------------------------------------------------------------


def test_record_fault_reports_false_when_reason_log_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Site :104 (default path). The documented contract is False when the reason log could not record."""
    import kernel_reason_log as rl
    from kernel_fault_disposition import record_fault

    def boom(*_args: object, **_kwargs: object) -> bool:
        raise RuntimeError("forced reason-log failure")

    monkeypatch.setattr(rl, "emit_reason", boom)
    assert record_fault("t8.unregistered", ValueError("x"), disposition="boundary",
                        observed_at="2026-11-15T00:00:00Z") is False


def test_record_fault_still_propagates_interrupts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Passthrough set (:102-103) is never swallowed by the recording path."""
    import kernel_reason_log as rl
    from kernel_fault_disposition import record_fault

    def interrupted(*_args: object, **_kwargs: object) -> bool:
        raise KeyboardInterrupt

    monkeypatch.setattr(rl, "emit_reason", interrupted)
    with pytest.raises(KeyboardInterrupt):
        record_fault("t8.unregistered", ValueError("x"), disposition="boundary",
                     observed_at="2026-11-15T00:00:00Z")


def test_cause_code_reports_rejected_fault_code_when_attribute_read_raises() -> None:
    """Site :147 (default path). A hostile fault_code property yields the typed fault_code_rejected flag."""
    from kernel_fault_disposition import _cause_code

    class _Hostile(Exception):
        @property
        def fault_code(self) -> str:
            raise RuntimeError("property failed")

    assert _cause_code(_Hostile("x"), "_Hostile") == ("_Hostile", True)


# --- kernel_reason_log ------------------------------------------------------------------------------


def test_reason_journal_payload_renames_reserved_keys_without_the_helper(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Formerly a strict xfail (old site :135): without the journal module the rename set was a hardcoded
    six-key copy, so run_id, event_id, timestamp_ns, ... reached journal.emit unrenamed. The set is now
    read from the journal's own RESERVED (from its source when the import fails)."""
    import kernel_reason_log as rl

    reserved = {"schema", "kind", "run_id", "task_id", "event_id", "sequence",
                "timestamp_ns", "journal_format", "previous_hash", "record_hash"}
    monkeypatch.setitem(sys.modules, "kernel_continuation_journal", None)  # forces ImportError in the helper
    journal = _Journal()
    assert rl.emit_reason("refusal", "t8-telemetry:reserved", "t8", "forced", "forced", journal=journal,
                          run_id="r1", event_id="e1", timestamp_ns=1) is True
    assert len(journal.events) == 1, "the reason event must still reach the journal"
    for kind, fields in journal.events:
        assert not (reserved & set(fields)), f"{kind} carried unrenamed journal-reserved keys"
        assert (fields["receipt_run_id"], fields["receipt_event_id"], fields["receipt_timestamp_ns"]) == (
            "r1", "e1", 1)
    assert "self_errors" not in rl.run_summary()


def test_reason_journal_rename_set_is_the_journal_reserved_set() -> None:
    """Regression: the rename rule and set are the journal's own (journal_safe_fields), with or without the
    import, so they cannot drift again."""
    import kernel_continuation_journal as kcj
    import kernel_reason_log as rl

    assert rl._journal_reserved() == kcj.RESERVED
    assert rl._journal_reserved_from_source() == kcj.RESERVED
    sample = {name: index for index, name in enumerate(sorted(kcj.RESERVED))}
    sample.update({"seq": "kept", "ts": "kept", "dropped": "kept"})
    assert rl._safe_keys(sample) == kcj.journal_safe_fields(sample)


def test_unknown_reserved_names_withhold_the_journal_event_and_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: when neither the import nor the journal source yields the reserved set, the event is
    withheld (never sent with a key the journal owns) and the failure is counted; stderr still reports."""
    import kernel_reason_log as rl

    def unreadable(_source: str) -> frozenset[str]:
        raise LookupError("forced: RESERVED unreadable")

    monkeypatch.setitem(sys.modules, "kernel_continuation_journal", None)
    monkeypatch.setattr(rl, "_parse_journal_reserved", unreadable)
    journal = _Journal()
    assert rl.emit_reason("refusal", "t8:withheld", "t8", "what", "why", journal=journal, run_id="r1") is True
    assert journal.events == []
    assert rl.run_summary()["self_errors"] == 1
    # No journal target: nothing to withhold, nothing counted.
    assert rl.emit_reason("refusal", "t8:no-journal", "t8", "what", "why", journal=False, run_id="r1") is True
    assert rl.run_summary()["self_errors"] == 1


class _RaisingLock:
    def __init__(self, error: type[BaseException]) -> None:
        self.error = error

    def __enter__(self) -> None:
        raise self.error("forced lock failure")

    def __exit__(self, *_exc: object) -> bool:
        return False


def test_bind_journal_failure_is_counted_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    """Site :109 (default path; was a silent `pass` at old :105). Binding is telemetry: counted, not raised."""
    import kernel_reason_log as rl

    with monkeypatch.context() as patched:
        patched.setattr(rl, "_LOCK", _RaisingLock(RuntimeError))
        assert rl.bind_journal(_Journal()) is None
    assert rl.run_summary()["self_errors"] == 1


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_bind_journal_propagates_interrupts(monkeypatch: pytest.MonkeyPatch,
                                           interrupt: type[BaseException]) -> None:
    """Regression (old :105 swallowed KeyboardInterrupt/SystemExit, unlike its siblings)."""
    import kernel_reason_log as rl

    monkeypatch.setattr(rl, "_LOCK", _RaisingLock(interrupt))
    with pytest.raises(interrupt):
        rl.bind_journal(_Journal())


def test_configure_failure_is_counted_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    """Site :98 (default path). configure is telemetry: it returns None and bumps self_errors."""
    import kernel_reason_log as rl

    def boom(_raw: object) -> None:
        raise RuntimeError("forced level failure")

    monkeypatch.setattr(rl, "_apply_level", boom)
    assert rl.configure(level="ERROR") is None
    assert rl.run_summary()["self_errors"] == 1


def test_journal_write_failure_is_counted_and_emit_still_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    """Site :130 (default path). A raising journal is counted in self_errors; the reason row still emitted."""
    import kernel_reason_log as rl

    assert rl.emit_reason("refusal", "t8:journal", "t8", "what", "why", journal=_RaisingJournal()) is True
    assert rl.run_summary()["self_errors"] == 1


def test_emit_reason_returns_false_when_emit_internals_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """Site :220 (default path). Documented never-raises entry point returns False and counts the failure."""
    import kernel_reason_log as rl

    def boom(*_args: object) -> bool:
        raise RuntimeError("forced emit failure")

    monkeypatch.setattr(rl, "_emit", boom)
    assert rl.emit_reason("refusal", "t8:169", "t8", "what", "why") is False
    assert rl.run_summary()["self_errors"] == 1


def test_stderr_write_failure_is_counted_not_raised() -> None:
    """Site :252 (default path). A broken stream costs the stderr line, not the caller."""
    import kernel_reason_log as rl

    rl.configure(stream=_BrokenStream())
    assert rl.emit_reason("refusal", "t8:stream", "t8", "what", "why") is True
    assert rl.run_summary()["self_errors"] == 1


def test_problems_file_write_failure_is_counted_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    """Site :256 (default path). A failed problems.jsonl append is counted in self_errors."""
    import kernel_reason_log as rl

    def boom(_path: object, _row: dict) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(rl, "_append_row", boom)
    assert rl.emit_reason("refusal", "t8:append", "t8", "what", "why") is True
    assert rl.run_summary()["self_errors"] == 1


@pytest.mark.parametrize("helper", ["emit_kill", "emit_refusal", "emit_timeout"])
def test_typed_emitters_return_false_when_reason_log_raises(monkeypatch: pytest.MonkeyPatch,
                                                            helper: str) -> None:
    """Sites :290, :303, :321 (default path). Typed emitters are total: False plus a self-error count."""
    import kernel_reason_log as rl

    def boom(*_args: object, **_kwargs: object) -> bool:
        raise RuntimeError("forced reason-log failure")

    monkeypatch.setattr(rl, "emit_reason", boom)
    emitter = getattr(rl, helper)
    kwargs = {"target": "x", "why": "w"} if helper == "emit_kill" else {"what": "x", "why": "w"}
    assert emitter("t8", **kwargs) is False
    assert rl.run_summary()["self_errors"] == 1


def test_flush_summary_problems_row_failure_is_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Site :373 (default path). The summary still returns the summary; the failed row is counted."""
    import kernel_reason_log as rl

    rl.emit_reason("refusal", "t8:flush-row", "t8", "what", "why")

    def boom(_path: object, _row: dict) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(rl, "_append_row", boom)
    summary = rl.flush_summary()
    assert summary is not None and summary["total"] == 1
    assert rl.run_summary()["self_errors"] == 1


def test_flush_summary_stderr_failure_is_counted() -> None:
    """Site :383 (default path). A broken stream on the end-of-run line is counted, summary still returned."""
    import kernel_reason_log as rl

    rl.emit_reason("refusal", "t8:flush-stream", "t8", "what", "why")
    rl.configure(stream=_BrokenStream())
    summary = rl.flush_summary()
    assert summary is not None and summary["total"] == 1
    assert rl.run_summary()["self_errors"] == 1


def test_flush_summary_failure_returns_none_and_is_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Site :388 (default path). The outer guard returns None (no summary written) and counts the failure."""
    import kernel_reason_log as rl

    def boom() -> int:
        raise RuntimeError("forced threshold failure")

    monkeypatch.setattr(rl, "_threshold", boom)
    assert rl.flush_summary() is None
    assert rl.run_summary()["self_errors"] == 1


class _UnreadableCauses(dict):
    """Cause table whose iteration raises (run_summary's failure path); get/set still work."""

    error: type[BaseException] = RuntimeError

    def items(self):  # type: ignore[override]
        raise self.error("forced: cause table unreadable")

    def values(self):  # type: ignore[override]
        raise self.error("forced: cause table unreadable")


def test_failed_run_summary_is_not_read_as_zero_self_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Site T8-KERNEL_REASON_LOG-343 (default path; formerly a strict xfail at old :283). Two self-errors
    are already counted; an unreadable cause table must not turn that into zero for the reader record_fault
    uses."""
    import kernel_reason_log as rl
    from kernel_fault_disposition import _self_errors

    monkeypatch.setitem(rl._STATE, "self_errors", 2)
    monkeypatch.setitem(rl._STATE, "causes", _UnreadableCauses())
    try:
        count = _self_errors()
    except Exception:  # noqa: BLE001 - a raise is a loud outcome
        return
    assert count != 0, "a failed run_summary was read as zero self-errors"


def test_failed_run_summary_counts_itself_and_names_the_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: the failure is itself a self-error and the summary says why it is empty."""
    import kernel_reason_log as rl

    monkeypatch.setitem(rl._STATE, "self_errors", 2)
    monkeypatch.setitem(rl._STATE, "causes", _UnreadableCauses())
    summary = rl.run_summary()
    assert summary["self_errors"] == 3
    assert summary["summary_error"] == "RuntimeError"
    assert (summary["total"], summary["by_cause"]) == (0, {})


def test_failed_run_summary_propagates_interrupts(monkeypatch: pytest.MonkeyPatch) -> None:
    import kernel_reason_log as rl

    class _InterruptedCauses(_UnreadableCauses):
        error = KeyboardInterrupt

    monkeypatch.setitem(rl._STATE, "causes", _InterruptedCauses())
    with pytest.raises(KeyboardInterrupt):
        rl.run_summary()


def test_record_fault_reports_false_when_run_summary_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: record_fault diffs self_errors around its emit; an unreadable summary must make it report
    'not verified' (False), never True on a 0 == 0 comparison."""
    import kernel_reason_log as rl
    from kernel_fault_disposition import record_fault

    monkeypatch.setitem(rl._STATE, "causes", _UnreadableCauses())
    assert record_fault("t8.unregistered", ValueError("x"), disposition="boundary",
                        observed_at="2026-11-15T00:00:00Z") is False


def test_flush_summary_row_persists_self_errors(tmp_path: Path) -> None:
    """Regression: the problems.jsonl summary row carries self_errors, so a reader of the file sees them."""
    import json

    import kernel_reason_log as rl

    problems = tmp_path / "problems.jsonl"
    rl.configure(problems_path=problems, stream=_BrokenStream())  # the stderr line fails: one self-error
    assert rl.emit_reason("refusal", "t8:persist", "t8", "what", "why", journal=_RaisingJournal()) is True
    summary = rl.flush_summary()
    assert summary is not None and summary["self_errors"] == 2
    rows = [json.loads(line) for line in problems.read_text(encoding="utf-8").splitlines()]
    summary_rows = [row for row in rows if row.get("summary")]
    assert len(summary_rows) == 1 and summary_rows[0]["self_errors"] == 2


def test_flush_summary_row_records_zero_self_errors_on_a_clean_run(tmp_path: Path) -> None:
    import json

    import kernel_reason_log as rl

    problems = tmp_path / "problems.jsonl"
    rl.configure(problems_path=problems, stream=io.StringIO())
    assert rl.emit_reason("refusal", "t8:clean", "t8", "what", "why") is True
    assert rl.flush_summary() is not None
    rows = [json.loads(line) for line in problems.read_text(encoding="utf-8").splitlines()]
    assert [row["self_errors"] for row in rows if row.get("summary")] == [0]


# --- kernel_probe (switch-gated: STATEFRAME_PROBE, opt-in) -------------------------------------------


def test_probe_workspace_snapshot_failure_is_a_typed_runner_error(monkeypatch: pytest.MonkeyPatch,
                                                                  tmp_path: Path) -> None:
    """Site :741 (opt-in). A workspace revision failure yields RUNNER_ERROR with the exception class named."""
    import kernel_probe as probe

    monkeypatch.setenv(probe.MODE_ENV, "on")

    def boom(_root: object) -> str:
        raise RuntimeError("forced revision failure")

    monkeypatch.setattr(probe, "workspace_revision", boom)
    receipt = probe.ProbeRunner(tmp_path).run("print(1)")
    assert receipt.status is probe.ProbeStatus.RUNNER_ERROR
    assert receipt.reason == "workspace_snapshot:RuntimeError"


# --- kernel_run_vitals (switch-gated: STATEFRAME_RUN_VITALS, opt-in) --------------------------------


def test_run_vitals_emit_failure_never_fails_the_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    """Site :435 (opt-in). Telemetry returns None and marks the summary as attempted, never raising."""
    import kernel_run_vitals as vitals

    monkeypatch.setenv(vitals.FLAG_ENV, "on")

    class _Journal:
        _run_vitals_accumulator = vitals.RunVitalsAccumulator()

        def emit(self, kind: str, **fields: object) -> None:
            raise RuntimeError("journal closed")

    journal = _Journal()
    assert vitals.emit_run_vitals(journal) is None
    assert journal._run_vitals_emitted is True


# --- inventory and tag hygiene ----------------------------------------------------------------------


def test_t8_telemetry_tags_point_at_broad_except_sites() -> None:
    """Every T8 tag in this file names an existing broad except line, so a moved site is noticed."""
    source = Path(__file__).read_text(encoding="utf-8")
    tags = re.findall(r"T8-([A-Z0-9_]+)-(\d+)", source)
    assert tags
    for module, line in tags:
        lines = (ENGINE / (module.lower() + ".py")).read_text(encoding="utf-8").splitlines()
        assert _BROAD.match(lines[int(line) - 1]), (
            f"{module}:{line} is no longer a broad except; retag or promote the test")


def _enclosing_function(tree: ast.AST, line: int) -> str:
    """Dotted name of the innermost def/class whose span holds ``line`` (``<module>`` if none)."""
    best, best_span = "<module>", None

    def walk(node: ast.AST, prefix: str) -> None:
        nonlocal best, best_span
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}.{child.name}" if prefix else child.name
                end = getattr(child, "end_lineno", child.lineno)
                if child.lineno <= line <= end:
                    span = end - child.lineno
                    if best_span is None or span <= best_span:
                        best, best_span = name, span
                walk(child, name)
            else:
                walk(child, prefix)

    walk(tree, "")
    return best


def _broad_except_sites(stem: str) -> set[tuple[str, str, str, int]]:
    """Line-number-free keys for broad excepts: (module, enclosing function, except text, ordinal).

    The ordinal counts identical except lines within one function, so two equal handlers stay distinct
    while unrelated edits above a site never change its key.
    """
    source = (ENGINE / f"{stem}.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    seen: dict[tuple[str, str], int] = {}
    keys: set[tuple[str, str, str, int]] = set()
    for number, text in enumerate(source.splitlines(), 1):
        if not _BROAD.match(text):
            continue
        if re.match(r"^\s*except\s*\(", text) and not re.search(r"Exception|BaseException", text):
            continue
        clause = text.split("#", 1)[0].strip()
        function = _enclosing_function(tree, number)
        ordinal = seen.get((function, clause), 0) + 1
        seen[(function, clause)] = ordinal
        keys.add((stem, function, clause, ordinal))
    return keys


def test_t8_telemetry_modules_have_no_unaccounted_broad_excepts() -> None:
    """Inventory: every broad except in the five telemetry modules has a verdict in this file.

    Sites are keyed by enclosing function and except text, not line numbers, so code moving above a
    site does not invalidate the verdicts; a new or removed handler still fails the check.
    """
    modules = ("kernel_fault_disposition", "kernel_probe", "kernel_reason_log",
               "kernel_run_vitals", "kernel_safe_probe")
    triaged = {
        # kernel_fault_disposition: DEGRADE-OK x4. record_fault reports False instead of raising;
        # _cause_code flags the fault code as rejected; _detail returns the unrenderable placeholder
        # for a hostile __str__; _contains_registered_secret fails closed (withholds the text).
        ("kernel_fault_disposition", "record_fault", "except BaseException:", 1),
        ("kernel_fault_disposition", "_cause_code", "except Exception:", 1),
        ("kernel_fault_disposition", "_detail", "except Exception:", 1),
        ("kernel_fault_disposition", "_contains_registered_secret", "except Exception:", 1),
        # kernel_safe_probe: CLEANUP x2 (the raise follows each)
        ("kernel_safe_probe", "run_probe_session", "except BaseException as error:", 1),
        ("kernel_safe_probe", "run_probe_session", "except BaseException as original:", 1),
        # kernel_probe: ALREADY-COVERED (child script prints the traceback), DEAD x3 in
        # _collect_process, CLEANUP in _run_child, DEGRADE-OK + AMBIGUOUS in ProbeRunner.run
        ("kernel_probe", "_wrapper_source", "except BaseException:", 1),
        ("kernel_probe", "_collect_process", "except Exception:", 1),
        ("kernel_probe", "_collect_process", "except Exception:", 2),
        ("kernel_probe", "_collect_process", "except Exception:", 3),
        ("kernel_probe", "_run_child", "except BaseException:", 1),
        ("kernel_probe", "ProbeRunner.run", "except Exception as error:", 1),
        ("kernel_probe", "ProbeRunner.run", "except Exception:", 1),
        # kernel_reason_log: DEGRADE-OK configure/bind_journal/emit_reason/emit_kill/emit_refusal/
        # emit_timeout/flush_summary(final), DEAD _bump_self_error, ALREADY-COVERED _journal_write/
        # _emit x2/flush_summary x2, FIXED (was CRASH-HIDER) run_summary (counted + typed summary_error).
        # The old hardcoded rename set is gone: the set is the journal's RESERVED.
        ("kernel_reason_log", "configure", "except BaseException:", 1),
        ("kernel_reason_log", "bind_journal", "except BaseException:", 1),
        ("kernel_reason_log", "_bump_self_error", "except BaseException:", 1),
        ("kernel_reason_log", "_journal_write", "except Exception:", 1),
        ("kernel_reason_log", "emit_reason", "except BaseException:", 1),
        ("kernel_reason_log", "_emit", "except Exception:", 1),
        ("kernel_reason_log", "_emit", "except Exception:", 2),
        ("kernel_reason_log", "emit_kill", "except BaseException:", 1),
        ("kernel_reason_log", "emit_refusal", "except BaseException:", 1),
        ("kernel_reason_log", "emit_timeout", "except BaseException:", 1),
        ("kernel_reason_log", "run_summary", "except BaseException as exc:", 1),
        ("kernel_reason_log", "flush_summary", "except Exception:", 1),
        ("kernel_reason_log", "flush_summary", "except Exception:", 2),
        ("kernel_reason_log", "flush_summary", "except BaseException:", 1),
        # kernel_run_vitals: DEGRADE-OK
        ("kernel_run_vitals", "emit_run_vitals", "except Exception:", 1),
    }
    found: set[tuple[str, str, str, int]] = set()
    for stem in modules:
        found |= _broad_except_sites(stem)
    assert found == triaged, (
        f"untriaged: {sorted(found - triaged)}; stale: {sorted(triaged - found)}")
