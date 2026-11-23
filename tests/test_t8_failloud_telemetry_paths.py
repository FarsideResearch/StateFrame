"""Fail-loud telemetry: broad except sites in the reason log, probe, safe probe, run vitals and fault disposition.

Scope: kernel_fault_disposition.py, kernel_probe.py, kernel_reason_log.py, kernel_run_vitals.py and
kernel_safe_probe.py (sites triaged at base 7e5c5598, with the T8 telemetry fixes applied). The inventory below
lists every broad except in these modules with its verdict. The loudness bar is a raise, a WARNING-or-higher
record from the owning module, or a typed field or counter that a reader checks. Telemetry that must not fail
its caller is allowed to return a typed outcome (False, None, a status field) and count its own failure, so
those sites are pinned to that outcome.

A strict xfail is used only for a CRASH-HIDER on a default (not switch-gated) path. A fix flips it to XPASS,
which fails the run until the test is promoted to a plain assertion. T8 tags name the symbol keys of guarded handlers.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

from _except_sites import (
    UNREVIEWED_HANDLER,
    ExceptSite,
    _broad_except_sites,
    assert_site_tags,
    assert_triaged,
    edited_module_paths,
    shifted_module_paths,
)

ENGINE = Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"


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
    """Site T8-[kernel_fault_disposition | record_fault | except BaseException: | 1] (default path).
    The documented contract is False when the reason log could not record."""
    import kernel_reason_log as rl
    from kernel_fault_disposition import record_fault

    def boom(*_args: object, **_kwargs: object) -> bool:
        raise RuntimeError("forced reason-log failure")

    monkeypatch.setattr(rl, "emit_reason", boom)
    assert record_fault("t8.unregistered", ValueError("x"), disposition="boundary",
                        observed_at="2026-11-15T00:00:00Z") is False


def test_record_fault_still_propagates_interrupts(monkeypatch: pytest.MonkeyPatch) -> None:
    """The passthrough set in record_fault is never swallowed by the recording path."""
    import kernel_reason_log as rl
    from kernel_fault_disposition import record_fault

    def interrupted(*_args: object, **_kwargs: object) -> bool:
        raise KeyboardInterrupt

    monkeypatch.setattr(rl, "emit_reason", interrupted)
    with pytest.raises(KeyboardInterrupt):
        record_fault("t8.unregistered", ValueError("x"), disposition="boundary",
                     observed_at="2026-11-15T00:00:00Z")


def test_cause_code_reports_rejected_fault_code_when_attribute_read_raises() -> None:
    """Site T8-[kernel_fault_disposition | _cause_code | except Exception: | 1] (default path).
    A hostile fault_code property yields the typed fault_code_rejected flag."""
    from kernel_fault_disposition import _cause_code

    class _Hostile(Exception):
        @property
        def fault_code(self) -> str:
            raise RuntimeError("property failed")

    assert _cause_code(_Hostile("x"), "_Hostile") == ("_Hostile", True)


# --- kernel_reason_log ------------------------------------------------------------------------------


def test_reason_journal_payload_renames_reserved_keys_without_the_helper(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Formerly a strict xfail: without the journal module the rename set was a hardcoded
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
    """Site T8-[kernel_reason_log | bind_journal | except BaseException: | 1] (default path; was a silent `pass`).
    Binding is telemetry: counted, not raised."""
    import kernel_reason_log as rl

    with monkeypatch.context() as patched:
        patched.setattr(rl, "_LOCK", _RaisingLock(RuntimeError))
        assert rl.bind_journal(_Journal()) is None
    assert rl.run_summary()["self_errors"] == 1


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_bind_journal_propagates_interrupts(monkeypatch: pytest.MonkeyPatch,
                                           interrupt: type[BaseException]) -> None:
    """Regression for bind_journal, which had swallowed KeyboardInterrupt/SystemExit, unlike its siblings."""
    import kernel_reason_log as rl

    monkeypatch.setattr(rl, "_LOCK", _RaisingLock(interrupt))
    with pytest.raises(interrupt):
        rl.bind_journal(_Journal())


def test_configure_failure_is_counted_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    """Site T8-[kernel_reason_log | configure | except BaseException: | 1] (default path).
    configure is telemetry: it returns None and bumps self_errors."""
    import kernel_reason_log as rl

    def boom(_raw: object) -> None:
        raise RuntimeError("forced level failure")

    monkeypatch.setattr(rl, "_apply_level", boom)
    assert rl.configure(level="ERROR") is None
    assert rl.run_summary()["self_errors"] == 1


def test_journal_write_failure_is_counted_and_emit_still_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    """Site T8-[kernel_reason_log | _journal_write | except Exception: | 1] (default path).
    A raising journal is counted in self_errors; the reason row still emitted."""
    import kernel_reason_log as rl

    assert rl.emit_reason("refusal", "t8:journal", "t8", "what", "why", journal=_RaisingJournal()) is True
    assert rl.run_summary()["self_errors"] == 1


def test_emit_reason_returns_false_when_emit_internals_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """Site T8-[kernel_reason_log | emit_reason | except BaseException: | 1] (default path).
    Documented never-raises entry point returns False and counts the failure."""
    import kernel_reason_log as rl

    def boom(*_args: object) -> bool:
        raise RuntimeError("forced emit failure")

    monkeypatch.setattr(rl, "_emit", boom)
    assert rl.emit_reason("refusal", "t8:169", "t8", "what", "why") is False
    assert rl.run_summary()["self_errors"] == 1


def test_stderr_write_failure_is_counted_not_raised() -> None:
    """Site T8-[kernel_reason_log | _emit | except Exception: | 1] (default path).
    A broken stream costs the stderr line, not the caller."""
    import kernel_reason_log as rl

    rl.configure(stream=_BrokenStream())
    assert rl.emit_reason("refusal", "t8:stream", "t8", "what", "why") is True
    assert rl.run_summary()["self_errors"] == 1


def test_problems_file_write_failure_is_counted_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    """Site T8-[kernel_reason_log | _emit | except Exception: | 2] (default path).
    A failed problems.jsonl append is counted in self_errors."""
    import kernel_reason_log as rl

    def boom(_path: object, _row: dict) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(rl, "_append_row", boom)
    assert rl.emit_reason("refusal", "t8:append", "t8", "what", "why") is True
    assert rl.run_summary()["self_errors"] == 1


@pytest.mark.parametrize("helper", ["emit_kill", "emit_refusal", "emit_timeout"])
def test_typed_emitters_return_false_when_reason_log_raises(monkeypatch: pytest.MonkeyPatch,
                                                            helper: str) -> None:
    """Sites T8-[kernel_reason_log | emit_kill | except BaseException: | 1],
    T8-[kernel_reason_log | emit_refusal | except BaseException: | 1], and
    T8-[kernel_reason_log | emit_timeout | except BaseException: | 1] (default path).
    Typed emitters are total: False plus a self-error count."""
    import kernel_reason_log as rl

    def boom(*_args: object, **_kwargs: object) -> bool:
        raise RuntimeError("forced reason-log failure")

    monkeypatch.setattr(rl, "emit_reason", boom)
    emitter = getattr(rl, helper)
    kwargs = {"target": "x", "why": "w"} if helper == "emit_kill" else {"what": "x", "why": "w"}
    assert emitter("t8", **kwargs) is False
    assert rl.run_summary()["self_errors"] == 1


def test_flush_summary_problems_row_failure_is_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Site T8-[kernel_reason_log | flush_summary | except Exception: | 1] (default path).
    The summary still returns the summary; the failed row is counted."""
    import kernel_reason_log as rl

    rl.emit_reason("refusal", "t8:flush-row", "t8", "what", "why")

    def boom(_path: object, _row: dict) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(rl, "_append_row", boom)
    summary = rl.flush_summary()
    assert summary is not None and summary["total"] == 1
    assert rl.run_summary()["self_errors"] == 1


def test_flush_summary_stderr_failure_is_counted() -> None:
    """Site T8-[kernel_reason_log | flush_summary | except Exception: | 2] (default path).
    A broken stream on the end-of-run line is counted, summary still returned."""
    import kernel_reason_log as rl

    rl.emit_reason("refusal", "t8:flush-stream", "t8", "what", "why")
    rl.configure(stream=_BrokenStream())
    summary = rl.flush_summary()
    assert summary is not None and summary["total"] == 1
    assert rl.run_summary()["self_errors"] == 1


def test_flush_summary_failure_returns_none_and_is_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Site T8-[kernel_reason_log | flush_summary | except BaseException: | 1] (default path).
    The outer guard returns None (no summary written) and counts the failure."""
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
    """Site T8-[kernel_reason_log | run_summary | except BaseException as exc: | 1].
    Default path, formerly a strict xfail. Two self-errors are already counted; an unreadable cause table must
    not turn that into zero for the reader record_fault uses."""
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
    """Site T8-[kernel_probe | ProbeRunner.run | except Exception as error: | 1] (opt-in).
    A workspace revision failure yields RUNNER_ERROR with the exception class named."""
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
    """Site T8-[kernel_run_vitals | emit_run_vitals | except Exception: | 1] (opt-in).
    Telemetry returns None and marks the summary as attempted, never raising."""
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


_MODULE_PATHS = (
    ENGINE / "kernel_fault_disposition.py",
    ENGINE / "kernel_probe.py",
    ENGINE / "kernel_reason_log.py",
    ENGINE / "kernel_request_auth_identity.py",
    ENGINE / "kernel_run_vitals.py",
    ENGINE / "kernel_safe_probe.py",
)

_TRIAGED: set[ExceptSite] = {
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
    # kernel_request_auth_identity: DEGRADE-OK; key reads/creation return the safe fallback,
    # and diagnostic failure must not make request measurement unavailable.
    ("kernel_request_auth_identity", "_read_secret", "except Exception as error:", 1),
    ("kernel_request_auth_identity", "_create_secret", "except Exception as error:", 1),
    ("kernel_request_auth_identity", "_report_process_local_fallback", "except Exception:", 1),
    ("kernel_request_auth_identity", "_installation_secret", "except Exception as error:", 1),
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
    ("kernel_run_vitals", "emit_run_vitals", "except Exception as error:", 1),
    ("kernel_run_vitals", "emit_run_vitals", "except Exception:", 2),
}


def test_t8_telemetry_modules_have_no_unaccounted_broad_excepts() -> None:
    assert_triaged(_broad_except_sites(_MODULE_PATHS), _TRIAGED)


def test_t8_telemetry_tags_point_at_broad_except_sites() -> None:
    assert_site_tags(Path(__file__), _broad_except_sites(_MODULE_PATHS))


def test_telemetry_inventory_survives_line_shift(tmp_path: Path) -> None:
    found = _broad_except_sites(shifted_module_paths(_MODULE_PATHS, tmp_path))
    assert_triaged(found, _TRIAGED)
    assert_site_tags(Path(__file__), found)


def test_telemetry_inventory_rejects_missing_site(tmp_path: Path) -> None:
    paths = edited_module_paths(
        _MODULE_PATHS, tmp_path, "kernel_probe",
        "except BaseException:\n    traceback.print_exc()\n    _exit = 1\n", "",
    )
    stale = ("kernel_probe", "_wrapper_source", "except BaseException:", 1)
    found = _broad_except_sites(paths)
    with pytest.raises(AssertionError) as failure:
        assert_triaged(found, _TRIAGED)
    assert f"stale: [{stale!r}]" in str(failure.value)


def test_telemetry_inventory_rejects_untriaged_site(tmp_path: Path) -> None:
    paths = edited_module_paths(_MODULE_PATHS, tmp_path, "kernel_probe", None, UNREVIEWED_HANDLER)
    untriaged = ("kernel_probe", "_t8_unreviewed", "except Exception:", 1)
    with pytest.raises(AssertionError) as failure:
        assert_triaged(_broad_except_sites(paths), _TRIAGED)
    assert f"untriaged: [{untriaged!r}]; stale: []" in str(failure.value)
