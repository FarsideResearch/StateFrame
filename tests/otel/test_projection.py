from __future__ import annotations

import hashlib
import json
import math

import pytest

from stateframe.otel.levels import Levels
from stateframe.otel.operations import epoch_seconds_ns
from stateframe.otel.projection import Projector
from stateframe.otel.types import Coverage, Cursor, Namespace, RunRef, SourceOccurrence


KEY = bytes(range(32))
RUN = RunRef(bytes.fromhex("10112233445566778899aabbccddeeff"))
TRACE_ID = bytes.fromhex("20112233445566778899aabbccddeeff")
SPACE = Namespace(RUN, 1, None, None)
EPOCH = 1_730_000_000_000_000_000


def _occurrence(
    family: str,
    row: dict[str, object],
    *,
    index: int = 1,
    epoch_ns: int | None = EPOCH,
    space: Namespace = SPACE,
    stream_id: bytes = b"s" * 16,
) -> SourceOccurrence:
    head = hashlib.sha256(json.dumps(
        row, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")).digest()
    return SourceOccurrence(
        space,
        family,
        index,
        epoch_ns,
        row,
        Cursor(stream_id, b"g" * 32, index, index, head),
    )


def _project(*occurrences: SourceOccurrence):
    projector = Projector(KEY, RUN, TRACE_ID)
    result = []
    rows = list(occurrences)
    needs_context = any(
        occurrence.family in {"summary", "engine_result"}
        or occurrence.family == "parallel" and occurrence.space.unit is not None
        for occurrence in rows
    )
    if needs_context and not any(occurrence.family == "spec" for occurrence in rows):
        rows.insert(0, _occurrence(
            "spec", {"schema": "stateframe-run/1", "created": "2024-11-01T00:00:00+00:00"},
            epoch_ns=EPOCH,
        ))
    for occurrence in rows:
        result.extend(projector.consume(occurrence))
    if rows:
        result.extend(projector.seal(rows[-1].space, Coverage(True, ())))
    return result


def _spans(projections):
    return [span for projection in projections for span in projection.spans]


def _attrs(span) -> dict[str, object]:
    return {attr.key: attr.value for attr in span.attrs}


def _summary(**values: object) -> SourceOccurrence:
    row: dict[str, object] = {
        "runtime_outcome": "completed",
        "check_outcome": "pass",
        "artifact": "saved",
        "engine_result": "SUBMITTED",
        "exit_code": 0,
        "child_exit_code": 0,
        "succeeded": True,
    }
    row.update(values)
    return _occurrence("summary", row)


def test_prompt_refusal_without_checks_or_provider():
    projections = _project(_summary(
        runtime_outcome="blocked",
        check_outcome="not_run",
        artifact="none",
        engine_result="AUTHORITY_FAILURE",
        exit_code=2,
        child_exit_code=2,
    ))
    spans = _spans(projections)
    assert not any(span.name in {"stateframe.model.request", "chat", "stateframe.check"} for span in spans)
    terminal = next(span for span in spans if span.name == "stateframe.attempt.outcome")
    attrs = _attrs(terminal)
    assert attrs["stateframe.runtime"] == "blocked"
    assert attrs["stateframe.check"] == "not_run"


@pytest.mark.parametrize(
    ("runtime", "check", "artifact", "exit_code", "expected_error"),
    [
        ("completed", "pass", "saved", 0, False),
        ("completed", "fail", "saved", 1, True),
        ("engine_error", "not_run", "partial", 2, True),
        ("suspended", "incomplete", "partial", 1, False),
        ("budget_stop", "pass", "none", 1, False),
        ("unknown", "unknown", "unknown", 255, False),
    ],
)
def test_runtime_check_artifact_exit_cross_product(runtime, check, artifact, exit_code, expected_error):
    projections = _project(_summary(
        runtime_outcome=runtime,
        check_outcome=check,
        artifact=artifact,
        exit_code=exit_code,
        child_exit_code=exit_code,
        engine_result="ENGINE_ERROR" if runtime == "engine_error" else "SUBMITTED",
    ))
    terminal = next(span for span in _spans(projections) if span.name == "stateframe.attempt.outcome")
    attrs = _attrs(terminal)
    assert attrs["stateframe.runtime"] == runtime
    assert attrs["stateframe.check"] == check
    assert attrs["stateframe.artifact"] == artifact
    assert attrs["stateframe.exit_code"] == exit_code
    assert terminal.error is expected_error


@pytest.mark.parametrize(
    ("runtime", "engine_result", "resumable", "expected_runtime", "expected_outcome"),
    [
        ("blocked", "BLOCKED", True, "suspended", "blocked"),
        ("blocked", "BLOCKED", False, "blocked", "blocked"),
        ("completed", "DELIVERY_REFUSED", False, "completed", "refused"),
        ("unknown", "AUTHORITY_FAILURE", False, "unknown", "error"),
    ],
)
def test_suspended_blocked_refused_are_distinct(runtime, engine_result, resumable, expected_runtime,
                                                 expected_outcome):
    projections = _project(_summary(
        runtime_outcome=runtime,
        engine_result=engine_result,
        engine_terminal={"result": engine_result, "resumable": resumable},
    ))
    terminal = next(span for span in _spans(projections) if span.name == "stateframe.attempt.outcome")
    assert _attrs(terminal)["stateframe.runtime"] == expected_runtime
    assert _attrs(terminal)["stateframe.outcome"] == expected_outcome


@pytest.mark.parametrize(
    "runtime",
    [
        "completed", "hook_blocked", "continuation_blocked", "suspended", "cancelled",
        "budget_stop", "recovery_required", "hook_containment", "engine_error",
        "context_too_small", "incomplete", "missing_result", "startup_error",
        "mutation_containment",
    ],
)
def test_all_terminal_outcomes_have_explicit_mapping(runtime):
    projections = _project(_summary(runtime_outcome=runtime, check_outcome="unknown"))
    terminal = next(span for span in _spans(projections) if span.name == "stateframe.attempt.outcome")
    assert _attrs(terminal)["stateframe.runtime"] == runtime


def test_unknown_and_seal_recovery_failure_are_never_green():
    projections = _project(_summary(
        runtime_outcome="unknown",
        engine_result="RECOVERY_REQUIRED",
        artifact="unknown",
        check_outcome="unknown",
        succeeded=True,
    ))
    terminal = next(span for span in _spans(projections) if span.name == "stateframe.attempt.outcome")
    attrs = _attrs(terminal)
    assert attrs["stateframe.outcome"] != "ok"
    assert attrs["stateframe.runtime"] == "unknown"


def test_uncaught_exception_marks_error_without_promoting_exit_code_alone():
    plain = _project(_summary(
        runtime_outcome="unknown", check_outcome="unknown", engine_result="UNKNOWN",
        exit_code=2, child_exit_code=2,
    ))
    plain_terminal = next(span for span in _spans(plain) if span.name == "stateframe.attempt.outcome")
    assert not plain_terminal.error

    exceptional = _project(_summary(
        runtime_outcome="unknown", check_outcome="unknown", engine_result="UNKNOWN",
        engine_terminal={"result": "UNKNOWN", "uncaught_exception": True},
        exit_code=2, child_exit_code=2,
    ))
    exceptional_terminal = next(span for span in _spans(exceptional)
                                if span.name == "stateframe.attempt.outcome")
    assert exceptional_terminal.error


def test_engine_result_document_projects_its_fixed_terminal():
    projections = _project(_occurrence(
        "engine_result", {"result": {"result": "SUBMITTED"}, "exit_status": 0, "seconds": 1.5}, index=2,
    ))
    terminal = next(span for span in _spans(projections) if span.name == "stateframe.attempt.outcome")
    attrs = _attrs(terminal)
    assert attrs["stateframe.runtime"] == "completed"
    assert attrs["stateframe.outcome"] == "ok"
    assert attrs["stateframe.exit_code"] == 0


def test_host_summary_supersedes_engine_result_on_resumed_attempt():
    space = Namespace(RUN, 2, None, None)
    spec = _occurrence("spec", {"resume": {"attempt": 2, "resumed": "2024-11-02T00:00:00+00:00"}},
                       space=space)
    engine = _occurrence("engine_result", {"result": {"result": "SUBMITTED"}, "exit_status": 0},
                         space=space)
    summary = _occurrence("summary", {
        "engine_result": "SUBMITTED", "runtime_outcome": "completed", "check_outcome": "fail",
        "exit_code": 1, "resume": {"attempt": 2},
    }, space=space, epoch_ns=None)
    outcomes = [span for span in _spans(_project(spec, engine, summary))
                if span.name == "stateframe.attempt.outcome"]
    assert len(outcomes) == 1
    attrs = _attrs(outcomes[0])
    assert attrs["stateframe.outcome"] == "error"
    assert attrs["stateframe.exit_code"] == 1
    assert attrs["stateframe.resume"] == "resumed"


def test_epoch_conversion_rejects_finite_float_overflow():
    assert epoch_seconds_ns(1e308) is None
    projections = _project(_occurrence("parallel", {
        "kind": "unit_finished", "call_id": "bounded-call",
        "started_epoch": 1e308, "finished_epoch": 1e308,
    }))
    assert not _spans(projections)


@pytest.mark.parametrize("late_failure", [
    {"engine_result": "DELIVERY_REFUSED", "delivery_refusal": {"reason_code": "CHECKS_STALE"}},
    {"engine_result": "SUBMITTED", "apply_error": "private failure marker"},
])
def test_delivery_refusal_overrides_submission_success(late_failure):
    projections = _project(_summary(**late_failure))
    terminal = next(span for span in _spans(projections) if span.name == "stateframe.attempt.outcome")
    attrs = _attrs(terminal)
    assert attrs["stateframe.delivery"] in {"refused", "apply_failed"}
    assert attrs["stateframe.outcome"] in {"refused", "error"}


def test_cleanup_and_end_hook_final_precedence():
    projections = _project(_summary(
        runtime_outcome="hook_containment",
        hook_containment="private exception detail",
        engine_result="DELIVERY_REFUSED",
        delivery_refusal={"reason_code": "CANDIDATE_NOT_DELIVERED"},
    ))
    terminal = next(span for span in _spans(projections) if span.name == "stateframe.attempt.outcome")
    attrs = _attrs(terminal)
    assert attrs["stateframe.runtime"] == "hook_containment"
    assert attrs["stateframe.delivery"] == "refused"
    assert terminal.error


def test_provisioning_without_evidence_is_not_timed():
    projections = _project(_occurrence("journal", {"kind": "provisioning_started", "sequence": 1}))
    assert not _spans(projections)


def test_prepare_and_provider_are_separate():
    projections = _project(
        _occurrence("journal", {"kind": "request_prepare_timing", "request_id": "private-id",
                                 "host_seconds": 0.25, "sequence": 1}, index=1),
        _occurrence("journal", {"kind": "request_submitted", "request_id": "private-id",
                                 "event_id": "transport-private-id", "step": 0, "sequence": 2}, index=2),
        _occurrence("journal", {"kind": "response_received", "request_id": "private-id",
                                 "step": 0, "transport_attempt_id": "transport-private-id",
                                 "elapsed_ns": 500_000_000, "sequence": 3}, index=3,
                    epoch_ns=EPOCH + 500_000_000),
    )
    spans = _spans(projections)
    exchange = next(span for span in spans if span.name == "chat")
    assert not any(span.name == "stateframe.request.prepare" for span in spans)
    assert _attrs(exchange)["stateframe.duration_ns"] == 500_000_000
    assert math.isclose((exchange.end_ns - exchange.start_ns) / 1e9, 0.5, abs_tol=1e-9)


def test_journal_request_ids_do_not_pair_across_streams():
    projections = _project(
        _occurrence("journal", {"kind": "request_submitted", "request_id": "same-id", "step": 1,
                                 "event_id": "first-event", "sequence": 1}, stream_id=b"a" * 16),
        _occurrence("journal", {"kind": "response_received", "request_id": "same-id", "step": 1,
                                 "transport_attempt_id": "other-event", "elapsed_ns": 10_000_000,
                                 "sequence": 1}, epoch_ns=EPOCH + 10_000_000, stream_id=b"b" * 16),
    )
    spans = _spans(projections)
    calls = [span for span in spans if span.name == "chat"]
    assert all("stateframe.duration_ns" not in _attrs(span) for span in calls)


def test_conflicting_request_and_transport_ids_do_not_cross_pair():
    projections = _project(
        _occurrence("journal", {"kind": "request_submitted", "request_id": "request-a", "step": 1,
                                 "event_id": "event-a", "sequence": 1}, index=1),
        _occurrence("journal", {"kind": "request_submitted", "request_id": "request-b", "step": 2,
                                 "event_id": "event-b", "sequence": 2}, index=2),
        _occurrence("journal", {"kind": "response_received", "request_id": "request-a", "step": 1,
                                 "transport_attempt_id": "event-b", "elapsed_ns": 50_000_000,
                                 "sequence": 3}, index=3, epoch_ns=EPOCH + 50_000_000),
    )
    spans = _spans(projections)
    assert any(span.name == "chat" for span in spans)
    assert all("stateframe.duration_ns" not in _attrs(span) for span in spans if span.name == "chat")
    assert not projections[-1].coverage.complete


def test_plain_stream_pairs_single_inflight_without_request_ids():
    projections = _project(
        _occurrence("plain", {"kind": "request_submitted", "request_id": "private-id", "step": 1,
                               "sequence": 1}, stream_id=b"p" * 16),
        _occurrence("plain", {"kind": "response_received", "elapsed_ns": 50_000_000,
                               "sequence": 2}, index=2, epoch_ns=EPOCH + 50_000_000, stream_id=b"p" * 16),
    )
    assert any(span.name == "chat" for span in _spans(projections))


def test_action_refusal_not_physical_execution():
    projections = _project(_occurrence("tool", {
        "kind": "refused", "tool": "shell", "dispatch": False, "outcome": "refused",
    }))
    span = next(span for span in _spans(projections) if span.name == "stateframe.action")
    dims = _attrs(span)
    assert dims["stateframe.execution"] == "not_dispatched"
    assert dims["stateframe.outcome"] == "refused"


@pytest.mark.parametrize("kind", ["baseline", "inner", "authored", "unit", "final", "unrecognized"])
def test_five_check_kinds_distinct_or_other(kind):
    projections = _project(_occurrence("journal", {
        "kind": "check_receipt", "check_kind": kind, "execution_id": "execution-opaque",
        "status": "PASS", "elapsed_seconds": 0.5,
    }))
    check = next(span for span in _spans(projections) if span.name == "stateframe.check")
    assert check.start_ns == check.end_ns
    assert _attrs(check)["stateframe.duration_s"] == 0.5
    assert _attrs(check)["stateframe.check.kind"] == (kind if kind in {
        "baseline", "inner", "authored", "unit", "final",
    } else "other")


def test_reexecuted_check_is_new_execution():
    first = _occurrence("journal", {"kind": "check_receipt", "check_kind": "unit",
                                     "execution_id": "first", "status": "PASS"}, index=1)
    second = _occurrence("journal", {"kind": "check_receipt_unchanged", "check_kind": "unit",
                                      "execution_id": "second", "status": "PASS",
                                      "execution": "reexecuted"}, index=2)
    checks = [span for span in _spans(_project(first, second)) if span.name == "stateframe.check"]
    assert len(checks) == 2
    assert {_attrs(check)["stateframe.execution"] for check in checks} == {"executed", "reexecuted"}


def test_worker_exchange_reuses_the_admitted_subframe_context():
    unit = Namespace(RUN, 1, b"a" * 16, 2)
    projections = _project(
        _occurrence("parallel", {"kind": "unit_context", "call_id": "private-call"},
                    space=unit),
        _occurrence("journal", {"kind": "request_submitted", "request_id": "private-request",
                                "step": 0, "event_id": "private-transport"}, space=unit, index=2),
        _occurrence("journal", {"kind": "response_received", "request_id": "private-request",
                                "step": 0, "transport_attempt_id": "private-transport",
                                "elapsed_ns": 10_000_000}, space=unit, index=3,
                    epoch_ns=EPOCH + 20_000_000),
    )
    by_name = {span.name: span for span in _spans(projections)}
    assert by_name["stateframe.subframe.context"].parent_id == by_name["stateframe.parallel.context"].span_id
    assert by_name["chat"].parent_id == by_name["stateframe.subframe.context"].span_id


def test_parallel_launch_refusal_and_relaunch_identity():
    first = Namespace(RUN, 1, b"u" * 16, 1)
    second = Namespace(RUN, 1, b"u" * 16, 2)
    projections = _project(
        _occurrence("parallel", {"kind": "unit_finished", "unit_id": "u", "incarnation": 1,
                                  "disposition": "REFUSED"}, index=1, space=first),
        _occurrence("parallel", {"kind": "unit_finished", "unit_id": "u", "incarnation": 2,
                                  "disposition": "COMMITTED"}, index=2, space=second),
    )
    contexts = [span for span in _spans(projections) if span.name == "stateframe.subframe.context"]
    assert len({span.span_id for span in contexts}) == len(contexts) == 2
    assert { _attrs(span)["stateframe.incarnation"] for span in contexts } == {1, 2}


def test_parent_conflict_maps_to_fixed_refusal_without_replacing_worker_outcome():
    unit = Namespace(RUN, 1, b"u" * 16, 1)
    projections = _project(_occurrence("parallel", {
        "kind": "unit_finished", "call_id": "call-a", "terminal": "SUBMITTED",
        "parent_disposition": "CONFLICT", "started_epoch": EPOCH / 1e9,
        "finished_epoch": EPOCH / 1e9 + 1,
    }, space=unit))
    span = next(span for span in _spans(projections) if span.name == "stateframe.subframe")
    attrs = _attrs(span)
    assert attrs["stateframe.outcome"] == "ok"
    assert attrs["stateframe.parallel.disposition"] == "refused"


def test_parallel_execution_and_incarnation_namespace():
    first = Namespace(RUN, 1, b"u" * 16, 1)
    second = Namespace(RUN, 1, b"u" * 16, 2)
    other_unit = Namespace(RUN, 1, b"v" * 16, 1)
    projections = _project(
        _occurrence("parallel", {"kind": "unit_registered", "incarnation": 1},
                    index=1, space=first),
        _occurrence("parallel", {"kind": "unit_registered", "incarnation": 2},
                    index=2, space=second),
        _occurrence("parallel", {"kind": "unit_registered", "incarnation": 1},
                    index=3, space=other_unit),
    )
    contexts = [span for span in _spans(projections) if span.name == "stateframe.subframe.context"]
    assert len(contexts) == 3
    assert len({span.span_id for span in contexts}) == 3
    assert all(span.trace_id == TRACE_ID for span in contexts)


def test_parallel_contexts_include_call_namespace_for_reused_unit():
    unit = Namespace(RUN, 1, b"u" * 16, 1)
    projections = _project(
        _occurrence("parallel", {"kind": "unit_registered", "call_id": "first-call",
                                  "incarnation": 1}, index=2, space=unit),
        _occurrence("parallel", {"kind": "unit_registered", "call_id": "second-call",
                                  "incarnation": 1}, index=3, space=unit),
    )
    contexts = [span for span in _spans(projections) if span.name == "stateframe.subframe.context"]
    assert len(contexts) == 2
    assert len({span.span_id for span in contexts}) == 2
    assert len({span.parent_id for span in contexts}) == 2


def test_stopped_boundary_tracks_last_evidence_without_inventing_time():
    spec = _occurrence("spec", {"schema": "stateframe-run/1"})
    first = _occurrence("journal", {"kind": "engine_start"}, epoch_ns=EPOCH + 10)
    later = _occurrence("journal", {"kind": "engine_progress"}, index=2, epoch_ns=EPOCH + 20)
    stopped = _occurrence("stopped", {}, index=3, epoch_ns=None)

    old = _project(spec, first, stopped)
    newer = _project(spec, first, later, stopped)

    def boundary(projections):
        return next(span for span in _spans(projections) if span.name == "stateframe.attempt.observation_end")

    assert boundary(old).end_ns == EPOCH + 10
    assert boundary(newer).start_ns == boundary(newer).end_ns == EPOCH + 20
    assert boundary(old).span_id != boundary(newer).span_id
    assert _attrs(boundary(newer))["stateframe.complete"] is False
    assert _attrs(boundary(newer))["stateframe.runtime"] == "unknown"


def test_killed_run_exports_honest_boundary():
    projector = Projector(KEY, RUN, TRACE_ID)
    beginning = _occurrence("spec", {"schema": "stateframe-run/1"})
    last = _occurrence("journal", {"kind": "request_prepare_timing", "host_seconds": 0.25},
                       index=2, epoch_ns=EPOCH + 1_000_000_000)
    stopped = _occurrence("stopped", {"interrupted": True}, index=3, epoch_ns=EPOCH + 8_000_000_000)
    rows = [item for row in (beginning, last, stopped) for item in projector.consume(row)]
    boundary = next(span for item in rows for span in item.spans
                    if span.name == "stateframe.attempt.observation_end")
    assert boundary.start_ns == boundary.end_ns == last.epoch_ns
    values = {str(attr.key): attr.value for attr in boundary.attrs}
    assert values["stateframe.complete"] is False
    assert values["stateframe.runtime"] == "interrupted"
    assert "stateframe.exit_code" not in values
    assert all(span.name != "stateframe.attempt.outcome" for item in rows for span in item.spans)
