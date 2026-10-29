"""Host-only R-201 instrumentation validation for grading/accounting evidence.

The validator deliberately does not alter request construction or model-facing
messages.  It consumes public evidence after (or immediately before) a run,
checks four deterministic boundary faults, and supplies the A/A identity and
worker-nonce checks needed before comparisons are trusted.  The feature is
strictly opt-in through ``STATEFRAME_INSTRUMENT_VALIDATION=on``.
"""

from __future__ import annotations

import copy
import json
import os
import secrets
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from kernel_harness_scanners import audit_serialized_requests, scan_repeated_blockers
from kernel_measurement import digest

MODE_ENV = "STATEFRAME_INSTRUMENT_VALIDATION"
SCHEMA = "stateframe-instrument-validation/1"
VALID_MODES = frozenset({"off", "on"})

WRONG_REVISION = "wrong_revision"
DROPPED_REQUIRED_SOURCE = "dropped_required_source"
LOST_EDIT_TOOL = "lost_edit_tool"
INTERRUPTED_EFFECT_RECEIPT = "interrupted_effect_receipt"
NONCE_NOT_REACHED = "nonce_not_reached"
AA_CONFIG_MISMATCH = "aa_config_mismatch"
AA_REQUEST_MISMATCH = "aa_request_mismatch"
AA_TASK_MISMATCH = "aa_task_mismatch"
AA_SEED_MISMATCH = "aa_seed_mismatch"
AA_RUN_ID_COLLISION = "aa_run_id_collision"
NONCE_REUSED = "nonce_reused"
MISSING_EVIDENCE = "missing_evidence"
MALFORMED_EVIDENCE = "malformed_evidence"

_FAULTS = (
    WRONG_REVISION,
    DROPPED_REQUIRED_SOURCE,
    LOST_EDIT_TOOL,
    INTERRUPTED_EFFECT_RECEIPT,
)
_EFFECT_OK = frozenset({"APPLIED", "COMPLETED", "PRESENT", "RECEIVED"})
_BOUNDARIES = (
    "construction",
    "delivery",
    "parsing",
    "prerequisite",
    "execution",
    "persistence",
    "observation",
    "completion",
)


class InstrumentValidationError(ValueError):
    """Raised when an opt-in evidence gate cannot authorize a comparison."""

    def __init__(self, report: "ValidationReport"):
        self.report = report
        super().__init__(json.dumps(report.as_dict(), sort_keys=True))


class JournalEmitter(Protocol):
    """The only journal capability needed by the nonce producer."""

    def emit(self, kind: str, **fields: object) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class ValidationReport:
    """Small, serializable result shared by the run and grading gates."""

    valid: bool
    checked: bool
    fault_codes: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    request_digest: str | None = None
    config_digest: str | None = None
    nonce_reached: bool = False
    runner_phase: str = "not_run"
    boundary_reached: tuple[str, ...] = ()
    raw_outcome: Mapping[str, object] | None = None
    scanner_findings: tuple[Mapping[str, object], ...] = ()

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly receipt without exposing mutable internals."""
        return {
            "schema": SCHEMA,
            "valid": self.valid,
            "checked": self.checked,
            "fault_codes": list(self.fault_codes),
            "errors": list(self.errors),
            "request_digest": self.request_digest,
            "config_digest": self.config_digest,
            "nonce_reached": self.nonce_reached,
            "runner_phase": self.runner_phase,
            "boundary_reached": list(self.boundary_reached),
            "raw_outcome": copy.deepcopy(self.raw_outcome),
            "scanner_findings": copy.deepcopy(list(self.scanner_findings)),
        }


def mode(value: str | None = None) -> str:
    """Read the strict feature switch; the absent value is the byte-safe OFF path."""
    raw = os.environ.get(MODE_ENV, "off") if value is None else value
    raw = str(raw).strip().lower()
    if raw not in VALID_MODES:
        raise ValueError(f"{MODE_ENV} must be 'off' or 'on'")
    return raw


def enabled() -> bool:
    return mode() == "on"


def request_digest(request: object) -> str:
    """Digest the canonical request object, including messages and tool schema."""
    return digest(request)


def config_digest(config: object) -> str:
    """Digest the canonical effective configuration used for a comparison."""
    return digest(config)


def _safe_digest(value: object) -> str | None:
    """Digest public evidence without exposing serializer details in a report."""
    try:
        return digest(value)
    except (TypeError, ValueError, OverflowError):
        return None


def new_nonce(prefix: str = "r201") -> str:
    """Create an unguessable worker marker for a live run's sink event."""
    prefix = str(prefix).strip()
    if not prefix or any(character.isspace() for character in prefix):
        raise ValueError("nonce prefix must be non-empty and contain no whitespace")
    return f"{prefix}-{secrets.token_urlsafe(18)}"


def _unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values))


def _as_names(value: object) -> set[str]:
    """Normalize public source/tool lists without trusting presentation text."""
    if isinstance(value, Mapping):
        value = value.keys()
    if isinstance(value, (str, bytes)):
        return {str(value)}
    if not isinstance(value, Iterable):
        return set()
    names: set[str] = set()
    for item in value:
        if isinstance(item, Mapping):
            function = item.get("function")
            if isinstance(function, Mapping) and function.get("name") is not None:
                names.add(str(function["name"]))
            elif item.get("name") is not None:
                names.add(str(item["name"]))
            elif item.get("path") is not None:
                names.add(str(item["path"]))
        elif item is not None:
            names.add(str(item))
    return names


def _effect(case: Mapping[str, object]) -> Mapping[str, object]:
    value = case.get("effect")
    return value if isinstance(value, Mapping) else case


def _raw_outcome(case: Mapping[str, object]) -> Mapping[str, object] | None:
    value = case.get("raw_outcome")
    if isinstance(value, Mapping):
        return dict(value)
    value = case.get("probe_receipt")
    if isinstance(value, Mapping):
        return {
            key: value[key]
            for key in ("exit_code", "timeout", "result")
            if key in value
        }
    return None


def _report(
    *,
    faults: Sequence[str] = (),
    errors: Sequence[str] = (),
    request: object | None = None,
    config: object | None = None,
    nonce_reached: bool = False,
    runner_phase: str = "not_run",
    boundary_reached: Sequence[str] = (),
    raw_outcome: Mapping[str, object] | None = None,
    scanner_findings: Sequence[Mapping[str, object]] = (),
    checked: bool = True,
) -> ValidationReport:
    fault_codes = _unique(faults)
    return ValidationReport(
        valid=not fault_codes,
        checked=checked,
        fault_codes=fault_codes,
        errors=tuple(str(error) for error in errors),
        request_digest=_safe_digest(request) if request is not None else None,
        config_digest=_safe_digest(config) if config is not None else None,
        nonce_reached=nonce_reached,
        runner_phase=str(runner_phase),
        boundary_reached=tuple(str(item) for item in boundary_reached),
        raw_outcome=dict(raw_outcome) if raw_outcome is not None else None,
        scanner_findings=tuple(dict(item) for item in scanner_findings),
    )


def _nonce_state(
    case: Mapping[str, object], *, require_worker_id: bool = True
) -> tuple[bool, list[str]]:
    expected = case.get("expected_nonce")
    observed = case.get("nonce_sink", case.get("nonce_observed"))
    worker = case.get("worker_nonce")
    expected_worker = case.get("expected_worker_id")
    worker_id = case.get("worker_id")
    values = (expected, observed, worker)
    if require_worker_id:
        values += (expected_worker, worker_id)
    if any(value is None or not str(value).strip() for value in values):
        return False, [NONCE_NOT_REACHED]
    if str(expected) != str(observed) or str(worker) != str(expected):
        return False, [NONCE_NOT_REACHED]
    if require_worker_id and str(expected_worker) != str(worker_id):
        return False, [NONCE_NOT_REACHED]
    return True, []


def _add_missing_evidence(
    case: Mapping[str, object], faults: list[str], errors: list[str]
) -> None:
    """Require stable public identities before accepting a run row."""
    missing = [
        name
        for name in ("task_id", "run_id", "seed")
        if case.get(name) is None or not str(case.get(name)).strip()
    ]
    if missing:
        faults.append(MISSING_EVIDENCE)
        errors.append("missing case identity: " + ", ".join(missing))

    for name in ("request", "config"):
        value = case.get(name)
        if not isinstance(value, Mapping):
            faults.append(MISSING_EVIDENCE if value is None else MALFORMED_EVIDENCE)
            errors.append(f"{name} evidence must be a JSON object")

    expected_revision = case.get("expected_revision")
    observed_revision = case.get("observed_revision", case.get("revision"))
    if expected_revision is None or observed_revision is None:
        faults.append(MISSING_EVIDENCE)
        errors.append("revision evidence is incomplete")


def validate_case(
    case: Mapping[str, object], *, require_nonce: bool = True
) -> ValidationReport:
    """Validate one public evidence case without assigning blame to a model.

    Each rule checks a boundary fact: revision equality, source delivery, edit
    affordance, and effect receipt.  A missing/invalid fact makes the case
    unusable; it does not become a candidate failure or a successful run.
    """
    if not enabled():
        return _report(checked=False)
    if not isinstance(case, Mapping):
        raise TypeError("instrument validation case must be a mapping")

    faults: list[str] = []
    errors: list[str] = []

    _add_missing_evidence(case, faults, errors)

    expected_revision = case.get("expected_revision")
    observed_revision = case.get("observed_revision", case.get("revision"))
    if (
        expected_revision is not None
        and observed_revision is not None
        and str(expected_revision) != str(observed_revision)
    ):
        faults.append(WRONG_REVISION)
        errors.append("expected revision does not match observed revision")

    required_sources = _as_names(
        case.get("required_sources", case.get("required_source", ()))
    )
    delivered_sources = _as_names(
        case.get(
            "delivered_sources",
            case.get("source_paths", case.get("delivered_source", ())),
        )
    )
    missing_sources = sorted(required_sources - delivered_sources)
    if missing_sources:
        faults.append(DROPPED_REQUIRED_SOURCE)
        errors.append("required source missing: " + ", ".join(missing_sources))

    required_tools = _as_names(case.get("required_tools", ()))
    if required_tools:
        offered_tools = _as_names(case.get("offered_tools", case.get("tools", ())))
        dispatched_tools = _as_names(
            case.get("dispatched_tools", case.get("dispatch_tools", ()))
        )
        missing_tools = sorted(required_tools - offered_tools)
        undispatched_tools = sorted(required_tools - dispatched_tools)
        if missing_tools or undispatched_tools:
            faults.append(LOST_EDIT_TOOL)
            details = []
            if missing_tools:
                details.append("not offered: " + ", ".join(missing_tools))
            if undispatched_tools:
                details.append("not dispatchable: " + ", ".join(undispatched_tools))
            errors.append(
                "required edit route unavailable (" + "; ".join(details) + ")"
            )

    effect_present = "effect" in case or any(
        key in case
        for key in (
            "intent_id",
            "effect_intent_id",
            "dispatch_id",
            "effect_dispatch_id",
            "receipt",
            "effect_receipt",
        )
    )
    effect = _effect(case)
    if not effect_present:
        faults.append(MISSING_EVIDENCE)
        errors.append("effect intent/dispatch/receipt evidence is incomplete")
    intent_id = effect.get("intent_id", effect.get("effect_intent_id"))
    dispatch_id = effect.get("dispatch_id", effect.get("effect_dispatch_id"))
    receipt = effect.get("receipt", effect.get("effect_receipt"))
    if effect_present:
        status = receipt.get("status") if isinstance(receipt, Mapping) else None
        receipt_id = (
            receipt.get("receipt_id", receipt.get("id"))
            if isinstance(receipt, Mapping)
            else None
        )
        if (
            not intent_id
            or not dispatch_id
            or not receipt_id
            or str(status).upper() not in _EFFECT_OK
        ):
            faults.append(INTERRUPTED_EFFECT_RECEIPT)
            errors.append("effect intent/dispatch has no completed receipt")

    nonce_reached = True
    if require_nonce:
        nonce_reached, nonce_faults = _nonce_state(case)
        faults.extend(nonce_faults)

    outcome = _raw_outcome(case)
    runner_phase = case.get("runner_phase", "not_run")
    boundaries = case.get("boundary_reached", ())
    if isinstance(case.get("probe_receipt"), Mapping):
        annotated = annotate_probe_receipt(
            case["probe_receipt"],
            expected_nonce=case.get("expected_nonce"),
            worker_nonce=case.get("worker_nonce"),
            nonce_sink=case.get("nonce_sink"),
            worker_id=case.get("worker_id"),
            expected_worker_id=case.get("expected_worker_id"),
        )
        runner_phase = annotated.get("runner_phase", runner_phase)
        boundaries = annotated.get("boundary_reached", boundaries)
        outcome = annotated.get("raw_outcome", outcome)

    return _report(
        faults=faults,
        errors=errors,
        request=case.get("request"),
        config=case.get("config"),
        nonce_reached=nonce_reached,
        runner_phase=str(runner_phase),
        boundary_reached=boundaries
        if isinstance(boundaries, Sequence) and not isinstance(boundaries, (str, bytes))
        else (),
        raw_outcome=outcome,
    )


def assert_valid(
    case: Mapping[str, object], *, require_nonce: bool = True
) -> ValidationReport:
    """Raise the explicit host gate error for an invalid comparison input."""
    report = validate_case(case, require_nonce=require_nonce)
    if enabled() and not report.valid:
        raise InstrumentValidationError(report)
    return report


def compare_aa(
    left: Mapping[str, object], right: Mapping[str, object]
) -> ValidationReport:
    """Require matched A/A request/config identities and distinct worker markers."""
    if not enabled():
        return _report(checked=False)
    left_report = validate_case(left)
    right_report = validate_case(right)
    faults = [*left_report.fault_codes, *right_report.fault_codes]
    errors = [
        *(f"A: {error}" for error in left_report.errors),
        *(f"B: {error}" for error in right_report.errors),
    ]
    left_request = _safe_digest(left.get("request"))
    right_request = _safe_digest(right.get("request"))
    left_config = _safe_digest(left.get("config"))
    right_config = _safe_digest(right.get("config"))
    if (
        left.get("task_id") is None
        or right.get("task_id") is None
        or left.get("task_id") != right.get("task_id")
    ):
        faults.append(AA_TASK_MISMATCH)
        errors.append("A/A task identity differs")
    if (
        left.get("seed") is None
        or right.get("seed") is None
        or left.get("seed") != right.get("seed")
    ):
        faults.append(AA_SEED_MISMATCH)
        errors.append("A/A seed differs")
    if left_request is None or right_request is None or left_request != right_request:
        faults.append(AA_REQUEST_MISMATCH)
        errors.append("A/A request digest is missing or differs")
    if left_config is None or right_config is None or left_config != right_config:
        faults.append(AA_CONFIG_MISMATCH)
        errors.append("A/A config digest is missing or differs")
    if (
        left.get("run_id") is None
        or right.get("run_id") is None
        or left.get("run_id") == right.get("run_id")
    ):
        faults.append(AA_RUN_ID_COLLISION)
        errors.append("A/A run IDs are missing or collide")
    left_nonce = left.get("worker_nonce")
    right_nonce = right.get("worker_nonce")
    if left_nonce is not None and right_nonce is not None and left_nonce == right_nonce:
        faults.append(NONCE_REUSED)
        errors.append("A/A worker nonce was reused")
    return _report(
        faults=faults,
        errors=errors,
        request=left.get("request"),
        config=left.get("config"),
        nonce_reached=left_report.nonce_reached and right_report.nonce_reached,
        runner_phase=left_report.runner_phase,
        boundary_reached=left_report.boundary_reached,
        raw_outcome=left_report.raw_outcome,
    )


def validate_grading_batch(
    cases: Sequence[Mapping[str, object]],
    *,
    aa_pairs: Sequence[tuple[int, int]] | None = None,
) -> ValidationReport:
    """Gate grading/accounting rows before their results are compared."""
    if not enabled():
        return _report(checked=False)
    if not cases:
        return _report(
            faults=(MISSING_EVIDENCE,),
            errors=("grading batch must contain at least one case",),
        )
    reports = [validate_case(case) for case in cases]
    faults: list[str] = [fault for report in reports for fault in report.fault_codes]
    errors = [
        f"case {index}: {error}"
        for index, report in enumerate(reports)
        for error in report.errors
    ]
    # A grading batch is not implicitly an A/A pair: candidate/control rows
    # often share task and seed while intentionally differing in config.
    pairs = list(aa_pairs) if aa_pairs is not None else []
    for left_index, right_index in pairs:
        if (
            left_index < 0
            or right_index < 0
            or left_index >= len(cases)
            or right_index >= len(cases)
        ):
            faults.append(AA_TASK_MISMATCH)
            errors.append("A/A pair index is outside the grading batch")
            continue
        pair = compare_aa(cases[left_index], cases[right_index])
        faults.extend(pair.fault_codes)
        errors.extend(
            f"A/A {left_index},{right_index}: {error}" for error in pair.errors
        )
    first = cases[0] if cases else {}
    return _report(
        faults=faults,
        errors=errors,
        request=first.get("request") if isinstance(first, Mapping) else None,
        config=first.get("config") if isinstance(first, Mapping) else None,
        nonce_reached=bool(reports) and all(report.nonce_reached for report in reports),
    )


def preflight_grading_batch(
    cases: Sequence[Mapping[str, object]],
    *,
    aa_pairs: Sequence[tuple[int, int]] | None = None,
) -> ValidationReport:
    """Raise before grading/accounting consumes an invalid comparison batch."""
    report = validate_grading_batch(cases, aa_pairs=aa_pairs)
    if enabled() and not report.valid:
        raise InstrumentValidationError(report)
    return report


def annotate_probe_receipt(
    receipt: Mapping[str, object],
    *,
    expected_nonce: object | None = None,
    worker_nonce: object | None = None,
    nonce_sink: object | None = None,
    worker_id: object | None = None,
    expected_worker_id: object | None = None,
) -> Mapping[str, object]:
    """Add the R-201 phase/validity fields to a public probe receipt when enabled.

    The OFF path returns the original object, preserving the pre-R-201 receipt
    shape and identity.  A timeout or setup result is kept distinct from a
    candidate ``PROBE_RED`` result.
    """
    if not enabled():
        return receipt
    if not isinstance(receipt, Mapping):
        raise TypeError("probe receipt must be a mapping")
    out = copy.deepcopy(dict(receipt))
    result = str(out.get("result", "")).upper()
    timed_out = out.get("timeout") is True
    exit_code = out.get("exit_code")
    candidate_result = (
        result == "PROBE_GREEN" and type(exit_code) is int and exit_code == 0
    ) or (result == "PROBE_RED" and type(exit_code) is int and exit_code == 1)
    if candidate_result and not timed_out:
        validity = "VALID"
        phase = "candidate"
        boundaries = _BOUNDARIES
    elif timed_out or result == "PROBE_UNAVAILABLE":
        validity = "UNAVAILABLE"
        phase = "runner"
        boundaries = _BOUNDARIES[:5]
    elif result:
        validity = "INVALID"
        phase = "runner"
        boundaries = _BOUNDARIES[:3]
    else:
        validity = "INVALID"
        phase = "admission"
        boundaries = _BOUNDARIES[:2]
    expected = expected_nonce if expected_nonce is not None else worker_nonce
    nonce_reached = (
        expected is not None
        and worker_nonce is not None
        and nonce_sink is not None
        and str(expected) == str(worker_nonce) == str(nonce_sink)
        and (expected_worker_id is None or str(expected_worker_id) == str(worker_id))
    )
    out.update(
        {
            "validity": validity,
            "runner_phase": phase,
            "boundary_reached": list(boundaries),
            "raw_outcome": {
                key: out.get(key)
                for key in ("exit_code", "timeout", "result")
                if key in out
            },
            "nonce_reached": nonce_reached,
        }
    )
    if worker_nonce is not None:
        out["worker_nonce"] = worker_nonce
    if worker_id is not None:
        out["worker_id"] = worker_id
    return out


def emit_worker_nonce(
    journal: JournalEmitter, nonce: str, worker_id: str
) -> Mapping[str, object] | None:
    """Write the nonce sink marker through the existing journal only when ON."""
    if not enabled():
        return None
    if (
        not isinstance(nonce, str)
        or not nonce
        or not isinstance(worker_id, str)
        or not worker_id
    ):
        raise ValueError("worker nonce and worker ID must be non-empty strings")
    event = journal.emit(
        "instrument_validation_nonce",
        worker_nonce=nonce,
        worker_id=worker_id,
        sink="journal",
    )
    if not isinstance(event, Mapping):
        raise TypeError("journal emit must return a mapping")
    return dict(event)


def _journal_records(path: str | os.PathLike[str]) -> list[dict[str, object]]:
    from kernel_measurement import resolve_journal_blob_refs

    journal_path = Path(path)
    records: list[dict[str, object]] = []
    blob_cache: dict[str, object] = {}
    with journal_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(
                    f"journal event must be an object at line {line_number}"
                )
            resolved = resolve_journal_blob_refs(record, journal_path, blob_cache)
            if not isinstance(resolved, dict):
                raise ValueError(f"journal event did not resolve to an object at line {line_number}")
            records.append(resolved)
    return records


def validate_journal(
    path: str | os.PathLike[str],
    *,
    expected_nonce: str | None = None,
    expected_worker_id: str | None = None,
) -> ValidationReport:
    """Run the existing public request scanners and gate obvious journal faults."""
    if not enabled():
        return _report(checked=False)
    try:
        scanner = audit_serialized_requests(path)
        blockers = scan_repeated_blockers(path)
        findings = [*scanner, *blockers]
        records = _journal_records(path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return _report(
            faults=(MALFORMED_EVIDENCE,),
            errors=(f"journal scan failed: {type(error).__name__}",),
        )
    faults: list[str] = []
    errors: list[str] = []
    for finding in scanner:
        if finding.get("current_source") is False:
            faults.append(DROPPED_REQUIRED_SOURCE)
            errors.append("request scanner found no current source bytes")
        if finding.get("edit_tool") is False:
            faults.append(LOST_EDIT_TOOL)
            errors.append("request scanner found no edit route for the named file")
    nonce_events = [
        record
        for record in records
        if record.get("kind") == "instrument_validation_nonce"
    ]
    request_events = [
        record
        for record in records
        if record.get("kind") in {"request_prepared", "request_submitted"}
    ]
    nonce_reached = bool(nonce_events)
    if request_events and not nonce_events:
        faults.append(NONCE_NOT_REACHED)
        errors.append("request evidence has no worker nonce sink event")
    elif request_events and expected_nonce is None:
        faults.append(NONCE_NOT_REACHED)
        errors.append("request evidence has no expected worker nonce")
    elif request_events and expected_nonce is not None:
        observed = nonce_events[0]
        reached, nonce_faults = _nonce_state(
            {
                "expected_nonce": expected_nonce,
                "worker_nonce": observed.get("worker_nonce"),
                "nonce_sink": observed.get("worker_nonce"),
                "expected_worker_id": expected_worker_id,
                "worker_id": observed.get("worker_id"),
            },
            require_worker_id=expected_worker_id is not None,
        )
        nonce_reached = reached
        faults.extend(nonce_faults)
        if nonce_faults:
            errors.append("worker nonce sink does not match the intended worker")
        if any(
            event.get("worker_nonce") != observed.get("worker_nonce")
            or event.get("worker_id") != observed.get("worker_id")
            for event in nonce_events[1:]
        ):
            faults.append(NONCE_REUSED)
            errors.append("journal contains conflicting worker nonce events")
    first_request = request_events[0] if request_events else {}
    request_object = {
        key: first_request[key]
        for key in ("messages", "tools", "payload", "context")
        if key in first_request
    } or None
    started = next(
        (record for record in records if record.get("kind") == "run_started"), {}
    )
    config = (
        started.get("manifest")
        if isinstance(started.get("manifest"), Mapping)
        else None
    )
    return _report(
        faults=faults,
        errors=errors,
        request=request_object,
        config=config,
        nonce_reached=nonce_reached
        and not any(fault in {NONCE_NOT_REACHED, NONCE_REUSED} for fault in faults),
        runner_phase="observation" if records else "not_run",
        boundary_reached=("construction", "delivery", "observation") if records else (),
        scanner_findings=findings,
    )


def _control_case() -> dict[str, object]:
    nonce = "nonce-r201-control"
    return {
        "schema": SCHEMA,
        "task_id": "r201-public-accounting",
        "run_id": "control",
        "arm": "A",
        "seed": 201,
        "config": {"model": "fixture-model", "temperature": 0.0, "max_tokens": 16},
        "request": {
            "messages": [{"role": "user", "content": "public R-201 fixture"}],
            "tools": [{"function": {"name": "edit"}}],
        },
        "expected_revision": "revision-201-a",
        "observed_revision": "revision-201-a",
        "required_sources": ["app/target.py"],
        "delivered_sources": ["app/target.py"],
        "required_tools": ["edit"],
        "offered_tools": ["edit", "run_tests"],
        "dispatched_tools": ["edit"],
        "effect": {
            "intent_id": "effect-intent-1",
            "dispatch_id": "effect-dispatch-1",
            "receipt": {"receipt_id": "effect-receipt-1", "status": "APPLIED"},
        },
        "expected_nonce": nonce,
        "worker_nonce": nonce,
        "nonce_sink": nonce,
        "worker_id": "worker-r201",
        "expected_worker_id": "worker-r201",
        "raw_outcome": {"exit_code": 0, "timeout": False, "result": "PROBE_GREEN"},
        "runner_phase": "candidate",
        "boundary_reached": list(_BOUNDARIES),
    }


def fault_injection_cases() -> dict[str, dict[str, object]]:
    """Return the control and exactly four deterministic public mutations."""
    control = _control_case()
    cases = {"control": copy.deepcopy(control)}
    wrong_revision = copy.deepcopy(control)
    wrong_revision["observed_revision"] = "revision-201-old"
    cases[WRONG_REVISION] = wrong_revision
    dropped_source = copy.deepcopy(control)
    dropped_source["delivered_sources"] = []
    cases[DROPPED_REQUIRED_SOURCE] = dropped_source
    lost_tool = copy.deepcopy(control)
    lost_tool["offered_tools"] = ["run_tests"]
    lost_tool["dispatched_tools"] = ["run_tests"]
    cases[LOST_EDIT_TOOL] = lost_tool
    interrupted = copy.deepcopy(control)
    effect = interrupted["effect"]
    assert isinstance(effect, dict)
    effect["receipt"] = None
    cases[INTERRUPTED_EFFECT_RECEIPT] = interrupted
    return cases


def run_fault_injections() -> dict[str, ValidationReport]:
    """Execute the four mutations through the same validator as a live gate."""
    return {name: validate_case(case) for name, case in fault_injection_cases().items()}


# Names kept intentionally small for callers in grading/accounting adapters.
validate_run = validate_case
validate_instrumentation = validate_case
run_fault_injection_suite = run_fault_injections
aa_digest_check = compare_aa


__all__ = [
    "AA_CONFIG_MISMATCH",
    "AA_REQUEST_MISMATCH",
    "AA_RUN_ID_COLLISION",
    "AA_SEED_MISMATCH",
    "AA_TASK_MISMATCH",
    "DROPPED_REQUIRED_SOURCE",
    "INTERRUPTED_EFFECT_RECEIPT",
    "InstrumentValidationError",
    "LOST_EDIT_TOOL",
    "MALFORMED_EVIDENCE",
    "MODE_ENV",
    "MISSING_EVIDENCE",
    "NONCE_NOT_REACHED",
    "NONCE_REUSED",
    "SCHEMA",
    "ValidationReport",
    "WRONG_REVISION",
    "aa_digest_check",
    "annotate_probe_receipt",
    "assert_valid",
    "compare_aa",
    "config_digest",
    "emit_worker_nonce",
    "enabled",
    "fault_injection_cases",
    "mode",
    "new_nonce",
    "preflight_grading_batch",
    "request_digest",
    "run_fault_injection_suite",
    "run_fault_injections",
    "validate_case",
    "validate_grading_batch",
    "validate_instrumentation",
    "validate_journal",
    "validate_run",
]
