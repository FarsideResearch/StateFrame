"""Fault disposition: registry validation, classification, event fields, once-per-run, self-errors."""
from __future__ import annotations

import io
import json
import threading
import time
from pathlib import Path

import pytest

import kernel_fault_disposition as disposition_mod
import kernel_reason_log as rl
from kernel_fault_disposition import record_fault
from kernel_fault_registry import HANDLERS, HandlerSpec, register_for_tests

PROBE = "synthetic.contract_probe"
UNCLASSIFIED_PROBE = "synthetic.unclassified_probe"
OBSERVED = "2026-11-15T00:00:00Z"
ENGINE_MODULE = "stateframe.engine.kernel_fault_disposition"

_VALID = dict(
    handler_id="x.probe", module=ENGINE_MODULE, function="record_fault",
    op="x_probe", taxonomy="D", tier="T3", requiredness="optional", expected=("ValueError",),
    dispositions=("boundary",), fallback_id=None, owner="engine", contract_tests=(), policy_version=1,
)
PROBE_SPEC = HandlerSpec(**{**_VALID, "handler_id": PROBE, "op": "synthetic_probe"})
UNCLASSIFIED_SPEC = HandlerSpec(**{**_VALID, "handler_id": UNCLASSIFIED_PROBE, "requiredness": "UNCLASSIFIED"})


class ProbeError(Exception):
    fault_code = "E42"


class _Journal:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit(self, kind: str, **fields: object) -> None:
        self.events.append((kind, fields))


class _BrokenStream:
    """Every write fails, so the reason log records a self-error."""

    def write(self, text: str) -> int:
        raise RuntimeError("disk gone")

    def flush(self) -> None:
        pass


class _RaisingStream:
    """Every write raises KeyboardInterrupt, which the recording path must not swallow."""

    def write(self, text: str) -> int:
        raise KeyboardInterrupt

    def flush(self) -> None:
        pass


class _RaceStream:
    """Writes naming BROKEN fail (self-error). Writes naming ok.race are slowed to widen the race window."""

    def write(self, text: str) -> int:
        if "BROKEN" in text:
            raise RuntimeError("disk gone")
        if "ok.race" in text:
            time.sleep(0.002)
        return len(text)

    def flush(self) -> None:
        pass


@pytest.fixture()
def fresh(monkeypatch, tmp_path):
    monkeypatch.delenv(rl.LEVEL_ENV, raising=False)
    monkeypatch.delenv(rl.PROBLEMS_ENV, raising=False)
    rl.reset_for_tests()
    journal, out = _Journal(), io.StringIO()
    problems = tmp_path / "problems.jsonl"
    rl.configure(level="WARN", journal=journal, problems_path=problems, stream=out)
    with register_for_tests(PROBE_SPEC):
        yield journal, problems
    rl.reset_for_tests()


def _rows(problems: Path) -> list[dict]:
    if not problems.exists():
        return []
    return [json.loads(line) for line in problems.read_text().splitlines()]


def _only_row(problems: Path) -> dict:
    rows = _rows(problems)
    assert len(rows) == 1, rows
    return rows[0]


def _emit(handler: str = PROBE, error: BaseException | None = None, **kwargs: object) -> bool:
    kwargs.setdefault("disposition", "boundary")
    kwargs.setdefault("observed_at", OBSERVED)
    return record_fault(handler, error if error is not None else ValueError("boom"), **kwargs)


# ---- registry ------------------------------------------------------------------------------------

def test_product_handlers_name_their_contract_tests():
    # Product rows are live sites: each names the tests that pin its contract (test-only entries never ship).
    for handler_id, spec in HANDLERS.items():
        assert not handler_id.startswith("synthetic."), handler_id
        assert spec.contract_tests, handler_id


def test_test_registration_is_visible_only_inside_the_context(fresh):
    _, problems = fresh
    with register_for_tests(UNCLASSIFIED_SPEC):
        assert record_fault(UNCLASSIFIED_PROBE, KeyError("k"), disposition="boundary",
                            observed_at=OBSERVED) is True
    assert _only_row(problems)["fields"]["expectation"] == "unclassified"
    # reset_for_tests also clears the problems path, so reconfigure before the post-exit call.
    rl.reset_for_tests()
    rl.configure(level="WARN", journal=_Journal(), problems_path=problems, stream=io.StringIO())
    assert record_fault(UNCLASSIFIED_PROBE, KeyError("k"), disposition="boundary",
                        observed_at=OBSERVED) is True
    assert _rows(problems)[-1]["fields"]["expectation"] == "unregistered"


def test_register_for_tests_refuses_product_collision(monkeypatch):
    monkeypatch.setitem(HANDLERS, PROBE, PROBE_SPEC)
    with pytest.raises(ValueError):
        with register_for_tests(PROBE_SPEC):
            pass


def test_register_for_tests_refuses_non_spec():
    with pytest.raises(TypeError):
        with register_for_tests({"handler_id": PROBE}):  # type: ignore[arg-type]
            pass


@pytest.mark.parametrize("overrides", [
    {"handler_id": ""},
    {"handler_id": "   "},
    {"taxonomy": "E"},
    {"tier": "T9"},
    {"requiredness": "maybe"},
    {"requiredness": "Optional"},
    {"dispositions": ()},
    {"dispositions": ["boundary"]},
    {"dispositions": ("explode",)},
    {"taxonomy": "B", "fallback_id": None},
    {"taxonomy": "B", "fallback_id": ""},
    {"taxonomy": "D", "fallback_id": "x.fallback"},
    {"taxonomy": "A", "fallback_id": "x.fallback"},
    {"expected": ("",)},
    {"expected": ["ValueError"]},
    {"contract_tests": ["tests/test_x.py::test_y"]},
    {"contract_tests": ("",)},
    {"owner": ""},
    {"owner": "   "},
    {"policy_version": 0},
    {"policy_version": True},
])
def test_handler_spec_rejects_bad_values(overrides):
    with pytest.raises(ValueError):
        HandlerSpec(**{**_VALID, **overrides})


def test_handler_spec_accepts_unclassified_requiredness():
    assert HandlerSpec(**{**_VALID, "requiredness": "UNCLASSIFIED"}).requiredness == "UNCLASSIFIED"


def test_handler_spec_accepts_taxonomy_b_with_fallback():
    spec = HandlerSpec(**{**_VALID, "taxonomy": "B", "fallback_id": "x.fallback"})
    assert spec.fallback_id == "x.fallback"


# ---- classification ------------------------------------------------------------------------------

def test_listed_fault_is_classified_expected(fresh):
    _, problems = fresh
    assert _emit(error=ValueError("boom")) is True
    row = _only_row(problems)
    assert row["fields"]["expectation"] == "expected"
    assert row["cause_key"] == f"{PROBE}:boundary:ValueError"


def test_unlisted_fault_is_classified_unexpected(fresh):
    _, problems = fresh
    assert _emit(error=KeyError("nope")) is True
    row = _only_row(problems)
    assert row["fields"]["expectation"] == "unexpected"
    assert row["cause_key"] == f"{PROBE}:boundary:KeyError"


def test_unclassified_requiredness_marks_unlisted_faults_unclassified(fresh):
    _, problems = fresh
    with register_for_tests(UNCLASSIFIED_SPEC):
        assert record_fault(UNCLASSIFIED_PROBE, KeyError("k"), disposition="boundary",
                            observed_at=OBSERVED) is True
    assert _only_row(problems)["fields"]["expectation"] == "unclassified"


def test_unclassified_requiredness_still_accepts_listed_faults(fresh):
    _, problems = fresh
    with register_for_tests(UNCLASSIFIED_SPEC):
        assert record_fault(UNCLASSIFIED_PROBE, ValueError("v"), disposition="boundary",
                            observed_at=OBSERVED) is True
    assert _only_row(problems)["fields"]["expectation"] == "expected"


def test_unregistered_handler_records_and_never_raises(fresh):
    _, problems = fresh
    assert record_fault("no.such.handler", ValueError("x"), disposition="fail_closed",
                        observed_at=OBSERVED) is True
    row = _only_row(problems)
    assert row["category"] == "refusal"
    assert row["fields"]["expectation"] == "unregistered"
    assert row["fields"]["policy_version"] is None
    assert row["fields"]["op"] is None
    assert row["fields"]["disposition_mismatch"] is False


def test_typed_code_attribute_is_appended_to_cause(fresh):
    _, problems = fresh
    assert _emit(error=ProbeError("coded")) is True
    row = _only_row(problems)
    assert row["fields"]["cause_code"] == "ProbeError:E42"
    assert row["cause_key"] == f"{PROBE}:boundary:ProbeError:E42"
    assert row["fields"]["fault_code_rejected"] is False


@pytest.mark.parametrize("code", ["E-42.x_1", "0", "A" * 40, "x_y.z-9"])
def test_fault_code_matching_pattern_is_accepted(fresh, code):
    _, problems = fresh
    error = ValueError("boom")
    error.fault_code = code  # type: ignore[attr-defined]
    assert _emit(error=error) is True
    row = _only_row(problems)
    assert row["fields"]["cause_code"] == f"ValueError:{code}"
    assert row["fields"]["fault_code_rejected"] is False


class _BadStr:
    def __str__(self) -> str:
        raise RuntimeError("no str for you")


class _HostileFaultCode(ValueError):
    @property
    def fault_code(self) -> str:
        raise RuntimeError("attribute read failed")


@pytest.mark.parametrize("code", [
    "A" * 5000,
    "A" * 41,
    42,
    True,
    "E42\n",
    "bad code",
    "a/b",
    "",
    _BadStr(),
])
def test_fault_code_outside_pattern_is_dropped(fresh, code):
    _, problems = fresh
    error = ValueError("boom")
    error.fault_code = code  # type: ignore[attr-defined]
    assert _emit(error=error) is True
    row = _only_row(problems)
    assert row["fields"]["cause_code"] == "ValueError"
    assert row["cause_key"] == f"{PROBE}:boundary:ValueError"
    assert row["fields"]["fault_code_rejected"] is True


def test_fault_code_that_raises_on_read_is_dropped_not_fatal(fresh):
    _, problems = fresh
    assert _emit(error=_HostileFaultCode("boom")) is True
    row = _only_row(problems)
    assert row["fields"]["cause_code"] == "_HostileFaultCode"
    assert row["fields"]["fault_code_rejected"] is True


# ---- disposition vs spec -------------------------------------------------------------------------

def test_disposition_outside_spec_still_records_with_mismatch_flag(fresh):
    _, problems = fresh
    assert _emit(disposition="fail_closed") is True
    row = _only_row(problems)
    assert row["category"] == "refusal"
    assert row["fields"]["disposition"] == "fail_closed"
    assert row["fields"]["disposition_mismatch"] is True


def test_disposition_inside_spec_is_not_a_mismatch(fresh):
    _, problems = fresh
    assert _emit(disposition="boundary") is True
    assert _only_row(problems)["fields"]["disposition_mismatch"] is False


def test_same_handler_and_cause_under_different_dispositions_are_distinct_rows(fresh):
    _, problems = fresh
    assert _emit(disposition="boundary") is True
    assert _emit(disposition="fail_closed") is True
    rows = _rows(problems)
    assert {row["cause_key"] for row in rows} == {
        f"{PROBE}:boundary:ValueError", f"{PROBE}:fail_closed:ValueError"}
    assert len(rows) == 2


# ---- event fields and journal --------------------------------------------------------------------

def test_event_fields_reach_reason_row_and_journal(fresh):
    journal, problems = fresh
    assert _emit(error=ValueError("boom"), origin="runner", fallback_id=None) is True
    expected_fields = {
        "handler_id": PROBE, "op": "synthetic_probe", "cause_code": "ValueError", "origin": "runner",
        "observed_at": OBSERVED, "expectation": "expected", "disposition": "boundary",
        "fallback_id": None, "policy_version": 1, "disposition_mismatch": False,
        "fault_code_rejected": False,
        "fault_id": disposition_mod._fault_id(PROBE, "boundary", "ValueError", OBSERVED, 0),
        "kind": "D", "exception_class": "ValueError", "exception_message": "boom", "site_id": PROBE,
        "required_work_id": None, "cause_chain": "[]",
        "primary_fault_id": disposition_mod._fault_id(PROBE, "boundary", "ValueError", OBSERVED, 0),
    }
    assert _only_row(problems)["fields"] == expected_fields
    assert [kind for kind, _ in journal.events] == ["reason_logged"]
    journal_fields = journal.events[0][1]
    # The journal event reserves "kind" for the event type, so the fault taxonomy letter travels as receipt_kind.
    journal_expected = {("receipt_kind" if k == "kind" else k): v for k, v in expected_fields.items()}
    assert {k: v for k, v in journal_fields.items() if k in journal_expected} == journal_expected


def test_journal_false_writes_problems_row_but_no_journal_event(fresh):
    journal, problems = fresh
    assert _emit(journal=False) is True
    assert journal.events == []
    assert _only_row(problems)["cause_key"] == f"{PROBE}:boundary:ValueError"


@pytest.mark.parametrize(("disposition", "category"), [
    ("fail_closed", "refusal"),
    ("degraded", "degrade"),
    ("narrowed", "fallback"),
    ("boundary", "invariant"),
])
def test_disposition_maps_to_reason_category(fresh, disposition, category):
    _, problems = fresh
    assert _emit(disposition=disposition) is True
    assert _only_row(problems)["category"] == category


def test_unknown_disposition_is_invariant_and_kept_verbatim(fresh):
    _, problems = fresh
    assert _emit(disposition="shrug") is True
    row = _only_row(problems)
    assert row["category"] == "invariant"
    assert row["fields"]["disposition"] == "shrug"
    assert row["fields"]["disposition_mismatch"] is True


# ---- once per run, empty messages ----------------------------------------------------------------

def test_repeat_is_counted_once_per_run_without_second_event(fresh):
    journal, problems = fresh
    assert _emit(error=ValueError("first")) is True
    assert _emit(error=ValueError("second")) is True
    _only_row(problems)
    assert [kind for kind, _ in journal.events] == ["reason_logged"]
    entry = rl.run_summary()["by_cause"].get(f"invariant:{PROBE}:boundary:ValueError")
    assert entry is not None and entry["count"] == 2


@pytest.mark.parametrize("error", [ValueError(), ValueError("   \n\t")])
def test_empty_message_keeps_type_as_cause(fresh, error):
    _, problems = fresh
    assert _emit(error=error) is True
    row = _only_row(problems)
    assert row["cause_key"] == f"{PROBE}:boundary:ValueError"
    assert row["why"] == "<empty message>"


# ---- failure modes ---------------------------------------------------------------------------------

def test_emit_self_error_returns_false(fresh):
    _, problems = fresh
    rl.configure(stream=_BrokenStream())
    assert _emit() is False
    assert rl.run_summary().get("self_errors") == 1
    assert _only_row(problems)["cause_key"] == f"{PROBE}:boundary:ValueError"


def test_keyboard_interrupt_from_recording_path_propagates(fresh):
    rl.configure(stream=_RaisingStream())
    with pytest.raises(KeyboardInterrupt):
        _emit()


def test_keyboard_interrupt_passed_as_error_is_recorded_not_raised(fresh):
    _, problems = fresh
    assert _emit(error=KeyboardInterrupt()) is True
    row = _only_row(problems)
    assert row["cause_key"] == f"{PROBE}:boundary:KeyboardInterrupt"
    assert row["fields"]["expectation"] == "unexpected"


def test_docstring_scopes_passthrough_to_recording_path():
    doc = disposition_mod.__doc__ or ""
    assert "raised by the recording path propagate" in doc
    assert "passed as the error argument are recorded" in doc


def test_concurrent_self_error_does_not_flip_a_successful_record(fresh):
    rl.configure(stream=_RaceStream())
    iterations = 200
    for i in range(iterations):
        barrier = threading.Barrier(2)
        results: dict[str, bool] = {}

        def broken() -> None:
            barrier.wait()
            results["bad"] = record_fault(f"BROKEN.{i}", ValueError("bad"), disposition="boundary",
                                          observed_at=OBSERVED)

        def healthy() -> None:
            barrier.wait()
            results["ok"] = record_fault(f"ok.race.{i}", ValueError("ok"), disposition="boundary",
                                         observed_at=OBSERVED)

        threads = [threading.Thread(target=broken), threading.Thread(target=healthy)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert results["ok"] is True, f"iteration {i}: successful record reported False"
        assert results["bad"] is False, f"iteration {i}: forced self-error reported True"
    assert rl.run_summary().get("self_errors") == iterations


def test_observed_at_is_required_keyword(fresh):
    with pytest.raises(TypeError):
        record_fault(PROBE, ValueError("x"), disposition="boundary")  # type: ignore[call-arg]


# ---- fault record schema (fault_id, kind, exception, site, work id, cause chain) ------------------

def _reset_reason_log(tmp_path: Path, name: str) -> Path:
    rl.reset_for_tests()
    problems = tmp_path / name
    rl.configure(level="WARN", journal=_Journal(), problems_path=problems, stream=io.StringIO())
    return problems


def _fields(problems: Path) -> dict:
    return _only_row(problems)["fields"]


def test_fault_id_is_sixteen_lowercase_hex_chars(fresh):
    _, problems = fresh
    _emit(error=ValueError("boom"))
    fault_id = _fields(problems)["fault_id"]
    assert len(fault_id) == 16 and all(c in "0123456789abcdef" for c in fault_id)


def test_fault_id_is_a_pure_function_of_its_inputs():
    base = ("h", "boundary", "ValueError", OBSERVED, 0)
    same = disposition_mod._fault_id(*base)
    assert same == disposition_mod._fault_id(*base)
    for index, replacement in enumerate(("h2", "fallback", "KeyError", "2026-11-16T00:00:00Z", 1)):
        changed = list(base)
        changed[index] = replacement
        assert disposition_mod._fault_id(*changed) != same, index


def test_fault_id_is_identical_across_two_fresh_reason_log_states(fresh, tmp_path):
    _, first_problems = fresh
    _emit(error=ProbeError("first"))
    first_id = _fields(first_problems)["fault_id"]
    second_problems = _reset_reason_log(tmp_path, "second.jsonl")
    _emit(error=ProbeError("first"))
    assert _fields(second_problems)["fault_id"] == first_id


def test_repeated_fault_in_one_run_consumes_an_occurrence_index(fresh):
    _, problems = fresh
    key = f"{PROBE}:boundary:ValueError"
    assert disposition_mod._prior_occurrences("invariant", key) == 0
    _emit(error=ValueError("boom"))
    assert disposition_mod._prior_occurrences("invariant", key) == 1
    _emit(error=ValueError("boom"))
    assert len(_rows(problems)) == 1  # repeat only counts; the next occurrence id is distinct
    assert disposition_mod._fault_id(PROBE, "boundary", "ValueError", OBSERVED, 1) != _fields(problems)["fault_id"]


def test_kind_is_registry_taxonomy_letter(fresh):
    _, problems = fresh
    _emit(error=ValueError("boom"))
    assert _fields(problems)["kind"] == "D"


def test_unregistered_handler_has_unregistered_kind_and_no_work_id(fresh):
    _, problems = fresh
    assert _emit("not.registered.handler", ValueError("boom")) is True
    fields = _fields(problems)
    assert fields["kind"] == "unregistered"
    assert fields["site_id"] == "not.registered.handler"
    assert fields["required_work_id"] is None
    assert fields["expectation"] == "unregistered"


def test_site_id_is_handler_id_and_exception_class_is_type_name(fresh):
    _, problems = fresh
    _emit(error=ProbeError("boom"))
    fields = _fields(problems)
    assert fields["site_id"] == fields["handler_id"] == PROBE
    assert fields["exception_class"] == "ProbeError"
    assert fields["cause_code"] == "ProbeError:E42"  # cause_code stays as before; class is separate


def test_exception_message_is_whitespace_normalized_and_capped(fresh):
    _, problems = fresh
    _emit(error=ValueError("alpha\n\n  beta\t" + "x" * 400))
    message = _fields(problems)["exception_message"]
    assert len(message) == disposition_mod.MAX_MESSAGE_CHARS
    assert message.startswith("alpha beta xxx") and message.endswith("...")
    assert "\n" not in message and "\t" not in message


def test_empty_exception_message_uses_placeholder(fresh):
    _, problems = fresh
    _emit(error=ValueError(""))
    assert _fields(problems)["exception_message"] == disposition_mod.EMPTY_DETAIL


def test_message_at_exactly_the_cap_is_not_truncated():
    text = "y" * disposition_mod.MAX_MESSAGE_CHARS
    assert disposition_mod._bounded(text) == text


def test_chained_exception_records_cause_chain_and_primary_id(fresh):
    _, problems = fresh
    try:
        try:
            raise KeyError("inner")
        except KeyError as inner:
            raise RuntimeError("outer") from inner
    except RuntimeError as outer:
        _emit(error=outer)
    fields = _fields(problems)
    assert json.loads(fields["cause_chain"]) == ["KeyError"]
    assert fields["primary_fault_id"] == disposition_mod._fault_id(PROBE, "boundary", "KeyError", OBSERVED, 0)
    assert fields["primary_fault_id"] != fields["fault_id"]


def test_implicit_context_is_followed_unless_suppressed(fresh):
    _, problems = fresh
    try:
        try:
            raise OSError("disk")
        except OSError:
            raise ValueError("wrapped")
    except ValueError as implicit:
        _emit(error=implicit)
    assert json.loads(_fields(problems)["cause_chain"]) == ["OSError"]


def test_suppressed_context_yields_empty_chain(fresh):
    _, problems = fresh
    try:
        try:
            raise OSError("disk")
        except OSError:
            raise ValueError("hidden") from None
    except ValueError as hidden:
        _emit(error=hidden)
    fields = _fields(problems)
    assert fields["cause_chain"] == "[]"
    assert fields["primary_fault_id"] == fields["fault_id"]


def test_cause_chain_is_capped_at_max_chain():
    error: BaseException = ValueError("leaf")
    for depth in range(disposition_mod.MAX_CHAIN + 3):
        outer = RuntimeError(f"level {depth}")
        outer.__cause__ = error
        error = outer
    chain = disposition_mod._cause_chain(error)
    assert len(chain) == disposition_mod.MAX_CHAIN
    assert chain[0] == "RuntimeError"  # outermost cause first


def test_cause_chain_terminates_on_a_cycle():
    first, second = ValueError("a"), KeyError("b")
    first.__cause__ = second
    second.__cause__ = first
    assert disposition_mod._cause_chain(first) == ["KeyError"]


def test_no_chain_primary_id_equals_fault_id(fresh):
    _, problems = fresh
    _emit(error=ValueError("plain"))
    fields = _fields(problems)
    assert fields["cause_chain"] == "[]"
    assert fields["primary_fault_id"] == fields["fault_id"]


def test_registry_required_work_id_defaults_to_none_and_rejects_blank():
    assert PROBE_SPEC.required_work_id is None
    with pytest.raises(ValueError, match="required_work_id"):
        HandlerSpec(**{**_VALID, "required_work_id": "   "})
    assert HANDLERS["react_repair.run_exception"].required_work_id == "engine_run"
