"""Language-neutral benchmark evidence, routing, and release receipts.

This module is a host-side bridge, not a task solver or completion authority.
It consumes public source/revision identities, executes only an explicitly
selected fixture through a bounded language adapter, and returns observation
records that existing receipt/completion owners can consume.  Private callable
names are allowed when their containing source path is public; hidden and test
paths are never admitted.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence

from kernel_observation_contract import (
    CallableIdentity,
    EffectObservation,
    EvidenceObservation,
    EvidenceScope,
    ObservationStatus,
    RelationResult,
    Revision,
    RevisionBoundReceipt,
    SerializedObservation,
    TraceEvent,
    logical_timer_event_trace,
    state_equality,
)


SCHEMA = "aha11-native-honesty/1"
_FORBIDDEN_PATH_PARTS = frozenset({".git", "private", "hidden", "hidden_tests", "oracle", "gold", "grader", "reference"})
_TEST_NAMES = frozenset({"test", "tests", "testdata", "fixtures", "fixture"})
_STATUS_ALIASES = {
    "PASS": ObservationStatus.OBSERVED_PASS,
    "PASSED": ObservationStatus.OBSERVED_PASS,
    "GREEN": ObservationStatus.OBSERVED_PASS,
    "FAIL": ObservationStatus.COUNTEREXAMPLE,
    "FAILED": ObservationStatus.COUNTEREXAMPLE,
    "RED": ObservationStatus.COUNTEREXAMPLE,
}


def _text(value: object, label: str, *, allow_empty: bool = False) -> str:
    result = str(value or "").strip()
    if not result and not allow_empty:
        raise ValueError(f"{label} must be non-empty")
    return result


def _canonical(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def _digest(value: object) -> str:
    encoded = json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _public_path(value: object) -> str:
    raw = _text(value, "public path").replace("\\", "/")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or not path.parts or any(part in {"", "."} for part in path.parts):
        raise ValueError("path is outside the declared public source scope")
    for part in path.parts[:-1]:
        lower = part.casefold()
        if part.startswith("_") or part.startswith(".") or lower in _FORBIDDEN_PATH_PARTS or lower in _TEST_NAMES:
            raise ValueError("hidden or test source path is not public")
    filename = path.name.casefold()
    if ((filename.startswith(".") or filename.startswith("_")) and filename != "__init__.py") \
            or filename.startswith("test_") or filename.endswith("_test.py") \
            or filename in {"conftest.py", "test.py"}:
        raise ValueError("test source path is not public")
    return path.as_posix()


def _public_function(value: object) -> str:
    function = _text(value, "function", allow_empty=True)
    if not function:
        return ""
    if any(part in {"", ".", ".."} for part in function.split(".")):
        raise ValueError("invalid public function name")
    if any(part.casefold() in _TEST_NAMES or part.startswith("__") for part in function.split(".")):
        raise ValueError("hidden test entrypoint is not admissible")
    # A single leading underscore is intentionally allowed: privacy is not a
    # universal rejection when the source path and declaration are public.
    if not all(part.replace("_", "a").isalnum() and not part[0].isdigit() for part in function.split(".")):
        raise ValueError("function name is not an identifier path")
    return function


@dataclass(frozen=True)
class PublicLocalization:
    path: str
    function: str = ""
    source: str = "public"

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _public_path(self.path))
        object.__setattr__(self, "function", _public_function(self.function))
        _text(self.source, "localization source")

    def as_dict(self) -> dict[str, object]:
        return {"path": self.path, "function": self.function, "source": self.source}


_SOLVER_FORBIDDEN_KEYS = frozenset({
    "evaluator", "reference", "reference_fix", "grader", "gold", "oracle",
    "changed_functions", "fixed_functions", "fixed_location", "hidden_location",
})


def public_localization(row: Mapping[str, object], *, discovered: Mapping[str, object] | None = None) -> PublicLocalization:
    """Resolve only declared public row/discovery fields; never inspect evaluator data."""
    entry = row.get("entry") if isinstance(row.get("entry"), Mapping) else {}
    path_value = entry.get("path") or row.get("path") or row.get("rel")
    if not path_value and isinstance(row.get("files"), Mapping):
        path_value = sorted(str(item) for item in row["files"])[0] if row["files"] else ""
    path = _public_path(path_value)
    function = entry.get("fn") or row.get("fn") or ""
    if isinstance(discovered, Mapping):
        discovered_path = discovered.get("path")
        if discovered_path and _public_path(discovered_path) == path:
            function = discovered.get("function", discovered.get("fn", function))
    return PublicLocalization(path, _public_function(function), "native_discovery" if discovered else "public")


def public_solver_row(public_row: Mapping[str, object], *, discovered: Mapping[str, object] | None = None) -> dict[str, object]:
    """Build the solver row from public material and erase evaluator routing."""
    def scrub(value: object) -> object:
        if isinstance(value, Mapping):
            cleaned: dict[str, object] = {}
            for key, item in value.items():
                normalized = str(key).casefold().replace("-", "_")
                if (normalized in _SOLVER_FORBIDDEN_KEYS
                        or any(marker in normalized for marker in (
                            "evaluator", "grader", "oracle", "reference_fix", "fixed_location",
                            "hidden_location", "changed_functions", "fixed_functions"))):
                    continue
                scrubbed = scrub(item)
                if isinstance(scrubbed, Mapping) and not scrubbed:
                    continue
                cleaned[str(key)] = scrubbed
            return cleaned
        if isinstance(value, (list, tuple)):
            return [scrub(item) for item in value]
        return copy.deepcopy(value)

    row = scrub(public_row)
    if not isinstance(row, dict):
        raise ValueError("public solver row must be a mapping")
    localization = public_localization(row, discovered=discovered)
    row["path"] = localization.path
    row["rel"] = localization.path
    row["fn"] = localization.function
    row["entry"] = {"path": localization.path, "fn": localization.function}
    row["localization_source"] = localization.source
    row["solver_input_digest"] = _digest(row)
    return row


def public_solver_identity(public_row: Mapping[str, object]) -> str:
    return str(public_solver_row(public_row).get("solver_input_digest"))


class TemporalPartition(str, Enum):
    ENTRY_STOP = "entry_stop"
    ACTIVE_FIRST = "active_first"
    REPEATED_CADENCE = "repeated_cadence"
    COMPLETION_DURING_WAIT = "completion_during_wait"
    FALSE_ACK = "false_ack"
    EXCEPTION = "exception"


@dataclass(frozen=True)
class TemporalExpectation:
    clause_id: str
    revision: Revision
    target: CallableIdentity
    first_action: str
    wait_event: str
    interval_seconds: float
    grant_argument: str
    grant_value: object
    required_partitions: tuple[str, ...] = tuple(item.value for item in TemporalPartition)

    def __post_init__(self) -> None:
        _text(self.clause_id, "clause id")
        _text(self.first_action, "first action")
        _text(self.wait_event, "wait event")
        _text(self.grant_argument, "grant argument")
        if self.interval_seconds < 0:
            raise ValueError("temporal interval cannot be negative")
        unknown = set(self.required_partitions) - {item.value for item in TemporalPartition}
        if unknown:
            raise ValueError(f"unknown temporal partitions: {sorted(unknown)}")
        if not self.required_partitions:
            raise ValueError("temporal expectation requires a bounded partition set")


def _observation_status(value: object) -> ObservationStatus:
    if isinstance(value, ObservationStatus):
        return value
    text = str(value or "UNKNOWN").strip().upper()
    return _STATUS_ALIASES.get(text, ObservationStatus(text) if text in ObservationStatus._value2member_map_ else ObservationStatus.UNKNOWN)


def _event(value: object, index: int) -> TraceEvent:
    if not isinstance(value, Mapping):
        raise ValueError("trace event must be an object")
    args = value.get("arguments", {})
    if not isinstance(args, Mapping):
        raise ValueError("trace event arguments must be an object")
    interval = value.get("interval_seconds")
    return TraceEvent(
        str(value.get("event_id", f"event-{index}")),
        str(value.get("kind", "observation")),
        _text(value.get("name"), "trace event name"),
        dict(args),
        int(value.get("logical_time", index)),
        float(interval) if interval is not None else None,
    )


def _effect(value: object, index: int) -> EffectObservation:
    if not isinstance(value, Mapping):
        raise ValueError("effect observation must be an object")
    args = value.get("arguments", {})
    if not isinstance(args, Mapping):
        raise ValueError("effect arguments must be an object")
    return EffectObservation(
        str(value.get("effect_id", f"effect-{index}")),
        _text(value.get("kind"), "effect kind"),
        _text(value.get("scope"), "effect scope"),
        int(value.get("count", 1)),
        dict(args),
    )


@dataclass(frozen=True)
class TemporalTrace:
    clause_id: str
    partition: str
    revision: Revision
    target: CallableIdentity
    adapter_id: str
    status: ObservationStatus
    events: tuple[TraceEvent, ...]
    effects: tuple[EffectObservation, ...]
    outcome: str
    state_before: SerializedObservation | None
    state_after: SerializedObservation | None
    receipt_id: str

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        expectation: TemporalExpectation,
        adapter_id: str,
        receipt_id: str,
    ) -> "TemporalTrace":
        partition = _text(value.get("partition"), "temporal partition")
        if partition not in expectation.required_partitions:
            raise ValueError("trace partition is not part of the admitted clause")
        events = tuple(_event(item, index) for index, item in enumerate(value.get("events", ())))
        effects = tuple(_effect(item, index) for index, item in enumerate(value.get("effects", ())))
        before_payload = value.get("state_before")
        after_payload = value.get("state_after")
        if before_payload is None or after_payload is None:
            raise ValueError("temporal trace must include before and after preservation state")
        return cls(
            expectation.clause_id,
            partition,
            expectation.revision,
            expectation.target,
            _text(adapter_id, "adapter id"),
            _observation_status(value.get("status", "OBSERVED_PASS")),
            events,
            effects,
            _text(value.get("outcome"), "temporal outcome"),
            SerializedObservation(f"{receipt_id}:before", before_payload, f"{adapter_id}:before"),
            SerializedObservation(f"{receipt_id}:after", after_payload, f"{adapter_id}:after"),
            _text(receipt_id, "receipt id"),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "clause_id": self.clause_id,
            "partition": self.partition,
            "revision": self.revision.as_dict(),
            "target": self.target.as_dict(),
            "adapter_id": self.adapter_id,
            "status": self.status.value,
            "events": [item.as_dict() for item in self.events],
            "effects": [item.as_dict() for item in self.effects],
            "outcome": self.outcome,
            "state_before": self.state_before.as_dict() if self.state_before else None,
            "state_after": self.state_after.as_dict() if self.state_after else None,
            "receipt_id": self.receipt_id,
        }


def _preserved_state(trace: TemporalTrace) -> RelationResult:
    if trace.state_before is None or trace.state_after is None:
        return RelationResult("temporal_state_preservation", ObservationStatus.UNCOVERED, "preservation state is missing")
    return state_equality(trace.state_before, trace.state_after, scope="")


def assess_temporal_trace(expectation: TemporalExpectation, trace: TemporalTrace) -> RelationResult:
    """Assess one executed partition; no scalar return comparison is used."""
    if trace.clause_id != expectation.clause_id or trace.revision != expectation.revision:
        return RelationResult("temporal_partition", ObservationStatus.UNCOVERED, "trace clause or revision is stale")
    if trace.status is not ObservationStatus.OBSERVED_PASS:
        return RelationResult("temporal_partition", trace.status, f"adapter settled partition as {trace.status.value}")
    partition = trace.partition
    if partition == TemporalPartition.ENTRY_STOP.value:
        if trace.outcome != "stopped_at_entry" or any(event.name == expectation.first_action for event in trace.events):
            return RelationResult("temporal_partition", ObservationStatus.COUNTEREXAMPLE, "entry stop performed an action")
    elif partition == TemporalPartition.ACTIVE_FIRST.value:
        relation = logical_timer_event_trace(
            trace.events,
            first_action=expectation.first_action,
            wait_event=expectation.wait_event,
            interval_seconds=expectation.interval_seconds,
        )
        if relation.status is not ObservationStatus.OBSERVED_PASS:
            return RelationResult("temporal_partition", relation.status, relation.detail, relation.value)
        first = next(event for event in trace.events if event.name == expectation.first_action)
        if first.arguments.get(expectation.grant_argument) != expectation.grant_value:
            return RelationResult("temporal_partition", ObservationStatus.COUNTEREXAMPLE, "first action grant differs from the public expectation")
    elif partition == TemporalPartition.REPEATED_CADENCE.value:
        actions = [event for event in trace.events if event.name == expectation.first_action]
        waits = [event for event in trace.events if event.name == expectation.wait_event]
        if len(actions) < 2 or len(waits) < 2:
            return RelationResult("temporal_partition", ObservationStatus.UNCOVERED, "repeated cadence has fewer than two actions or waits")
        if any(event.interval_seconds != expectation.interval_seconds for event in waits):
            return RelationResult("temporal_partition", ObservationStatus.COUNTEREXAMPLE, "repeated cadence interval differs")
        if any(event.arguments.get(expectation.grant_argument) != expectation.grant_value for event in actions):
            return RelationResult("temporal_partition", ObservationStatus.COUNTEREXAMPLE, "repeated action grant differs")
    elif partition == TemporalPartition.COMPLETION_DURING_WAIT.value:
        waits = [index for index, event in enumerate(trace.events) if event.name == expectation.wait_event]
        completions = [index for index, event in enumerate(trace.events) if event.kind in {"completion", "stop"}]
        if not waits or not completions or completions[0] < waits[0] or any(
                event.name == expectation.first_action for event in trace.events[waits[0] + 1:]):
            return RelationResult("temporal_partition", ObservationStatus.COUNTEREXAMPLE, "completion during wait was followed by renewal")
    elif partition == TemporalPartition.FALSE_ACK.value:
        if trace.outcome not in {"false_ack", "lease_lost"}:
            return RelationResult("temporal_partition", ObservationStatus.UNCOVERED, "false-ack outcome was not observed")
    elif partition == TemporalPartition.EXCEPTION.value:
        if trace.outcome not in {"exception", "lease_lost"}:
            return RelationResult("temporal_partition", ObservationStatus.UNCOVERED, "exception outcome was not observed")
    preserved = _preserved_state(trace)
    if preserved.status is not ObservationStatus.OBSERVED_PASS:
        return RelationResult("temporal_partition", preserved.status, "primary preservation state was not proven", preserved.value)
    return RelationResult("temporal_partition", ObservationStatus.OBSERVED_PASS, "temporal partition observed", {"partition": partition})


def temporal_evidence(
    expectation: TemporalExpectation,
    trace: TemporalTrace,
    *,
    ticket_id: str,
    source_id: str,
    candidate_id: str,
    check_id: str,
) -> EvidenceObservation:
    relation = assess_temporal_trace(expectation, trace)
    receipt = RevisionBoundReceipt(
        ticket_id, source_id, candidate_id, check_id, trace.adapter_id, trace.receipt_id, trace.revision,
    )
    return EvidenceObservation(
        relation.status,
        EvidenceScope.CASE,
        trace.partition,
        trace.clause_id,
        receipt,
        trace.as_dict(),
        (trace.partition,) if relation.status is ObservationStatus.OBSERVED_PASS else (),
        () if relation.status is ObservationStatus.OBSERVED_PASS else (relation.detail,),
    )


def assess_temporal_clause(expectation: TemporalExpectation, evidence: Sequence[EvidenceObservation]) -> RelationResult:
    """Require all current bounded partitions and all receipt identities."""
    current = [item for item in evidence if item.receipt.revision == expectation.revision]
    if not current:
        return RelationResult("temporal_clause", ObservationStatus.UNCOVERED, "no current temporal evidence")
    if any(item.case_id != expectation.clause_id for item in current):
        return RelationResult("temporal_clause", ObservationStatus.UNCOVERED, "evidence clause identity differs")
    identity = tuple(getattr(current[0].receipt, field) for field in ("ticket_id", "source_id", "candidate_id", "check_id"))
    if any(tuple(getattr(item.receipt, field) for field in ("ticket_id", "source_id", "candidate_id", "check_id")) != identity for item in current[1:]):
        return RelationResult("temporal_clause", ObservationStatus.UNCOVERED, "receipt identity differs across partitions")
    for status in (ObservationStatus.COUNTEREXAMPLE, ObservationStatus.ENVIRONMENT, ObservationStatus.UNRUNNABLE, ObservationStatus.UNKNOWN):
        if any(item.status is status for item in current):
            return RelationResult("temporal_clause", status, f"current evidence contains {status.value}")
    covered = {item.partition for item in current if item.status is ObservationStatus.OBSERVED_PASS}
    missing = sorted(set(expectation.required_partitions) - covered)
    if missing:
        return RelationResult("temporal_clause", ObservationStatus.UNCOVERED, "bounded temporal partitions remain open", {"missing": missing})
    return RelationResult("temporal_clause", ObservationStatus.OBSERVED_PASS, "all bounded temporal partitions have current receipts", {"partitions": sorted(covered)}, True)


def temporal_completion_authorized(runtime: object, current_revision: object) -> bool | None:
    """Return None when the benchmark bridge is not active; otherwise require its full proof."""
    expectation = getattr(runtime, "aha11_temporal_expectation", None)
    if not isinstance(expectation, TemporalExpectation):
        return None
    if isinstance(current_revision, Revision):
        revision_matches = expectation.revision == current_revision
    elif isinstance(current_revision, Mapping):
        current_value = current_revision.get("current", current_revision.get("product_revision"))
        source_value = current_revision.get("source", current_revision.get("source_revision"))
        candidate_value = current_revision.get("candidate", current_revision.get("candidate_revision"))
        check_value = current_revision.get("check", current_revision.get("check_identity"))
        revision_matches = (
            str(current_value or "") == expectation.revision.current
            and (source_value in (None, "") or str(source_value) == expectation.revision.source)
            and (candidate_value in (None, "") or str(candidate_value) == expectation.revision.candidate)
            and (check_value in (None, "") or str(check_value) == expectation.revision.check)
        )
    else:
        revision_matches = str(current_revision or "") == expectation.revision.current
    if not revision_matches:
        return False
    evidence = getattr(runtime, "aha11_temporal_evidence", ())
    if not isinstance(evidence, Sequence):
        return False
    return assess_temporal_clause(expectation, evidence).certifies_clause


def compare_parent_candidate(parent: TemporalTrace, candidate: TemporalTrace) -> RelationResult:
    """Keep parent failure and candidate pass separate while binding shared identity."""
    if parent.clause_id != candidate.clause_id or parent.target != candidate.target:
        return RelationResult("parent_candidate", ObservationStatus.UNCOVERED, "parent/candidate target identity differs")
    if parent.revision.source != candidate.revision.source or parent.revision.check != candidate.revision.check:
        return RelationResult("parent_candidate", ObservationStatus.UNCOVERED, "parent/candidate source or check identity differs")
    if parent.status is not ObservationStatus.COUNTEREXAMPLE:
        return RelationResult("parent_candidate", ObservationStatus.UNCOVERED, "parent behavior failure was not reached")
    if candidate.status is not ObservationStatus.OBSERVED_PASS:
        return RelationResult("parent_candidate", candidate.status, "candidate did not pass the same observation")
    return RelationResult("parent_candidate", ObservationStatus.OBSERVED_PASS, "parent failure and candidate observation are bound", {"parent": parent.receipt_id, "candidate": candidate.receipt_id}, False)


@dataclass(frozen=True)
class SourceReservation:
    """A bounded source reservation made before advisory material is added."""

    path: str
    revision: str
    total_lines: int
    selected_lines: tuple[int, ...]
    source_text: str
    complete: bool
    omitted_lines: tuple[int, ...]
    continuation: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "path": self.path,
            "revision": self.revision,
            "total_lines": self.total_lines,
            "selected_lines": list(self.selected_lines),
            "source_text": self.source_text,
            "complete": self.complete,
            "omitted_lines": list(self.omitted_lines),
            "continuation": dict(self.continuation),
        }


def _line_numbers(ranges: Sequence[object], total: int) -> set[int]:
    selected: set[int] = set()
    for value in ranges:
        if isinstance(value, Mapping):
            start, end = value.get("start", value.get("start_line")), value.get("end", value.get("end_line"))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and value:
            start, end = value[0], value[-1]
        else:
            continue
        try:
            selected.update(range(max(1, int(start)), min(total, int(end)) + 1))
        except (TypeError, ValueError):
            continue
    return selected


def reserve_source_unit(
    source: str,
    *,
    path: str,
    revision: str,
    max_bytes: int,
    non_source_bytes: int = 0,
    required_ranges: Sequence[object] = (),
    signature_ranges: Sequence[object] = (),
    caller_ranges: Sequence[object] = (),
    cursor: int = 0,
) -> SourceReservation:
    """Reserve complete source when possible, otherwise emit bounded continuation metadata."""
    public_path = _public_path(path)
    revision_text = _text(revision, "source revision")
    if max_bytes <= 0 or non_source_bytes < 0 or cursor < 0:
        raise ValueError("source reservation budgets and cursor must be non-negative")
    lines = source.splitlines()
    total = len(lines)
    numbered = [f"{index:06d}: {line}" for index, line in enumerate(lines, 1)]
    available = max(1, max_bytes - non_source_bytes)
    full = "\n".join(numbered)
    if len(full.encode("utf-8")) <= available:
        return SourceReservation(public_path, revision_text, total, tuple(range(1, total + 1)), full,
                                 True, (), {"kind": "complete", "cursor": cursor, "next_cursor": None})
    required = _line_numbers((*required_ranges, *signature_ranges, *caller_ranges), total)
    ordered = sorted(required) + [index for index in range(1, total + 1) if index not in required]
    selected: list[int] = []
    used = 0
    for index in ordered:
        rendered = numbered[index - 1] + ("\n" if selected else "")
        size = len(rendered.encode("utf-8"))
        if selected and used + size > available:
            continue
        if not selected and size > available:
            break
        selected.append(index)
        used += size
    selected_set = set(selected)
    omitted = tuple(index for index in range(1, total + 1) if index not in selected_set)
    next_cursor = cursor + 1 if omitted else None
    continuation = {
        "kind": "omission" if omitted else "complete",
        "cursor": cursor,
        "next_cursor": next_cursor,
        "omitted_line_count": len(omitted),
        "omitted_lines": list(omitted[:128]),
        "receipt": _digest({"path": public_path, "revision": revision_text, "cursor": cursor, "omitted": omitted}),
    }
    return SourceReservation(public_path, revision_text, total, tuple(selected),
                             "\n".join(numbered[index - 1] for index in selected),
                             not omitted, omitted, continuation)


@dataclass(frozen=True)
class FixtureExecution:
    adapter_id: str
    status: ObservationStatus
    trace: TemporalTrace | None
    stdout: str = ""
    stderr: str = ""
    reason: str = ""

    def as_dict(self) -> dict[str, object]:
        return {"schema": SCHEMA, "adapter_id": self.adapter_id, "status": self.status.value, "trace": self.trace.as_dict() if self.trace else None, "stdout": self.stdout[-2000:], "stderr": self.stderr[-2000:], "reason": self.reason}


def _fixture_environment() -> dict[str, str]:
    return {"PATH": os.environ.get("PATH", "")}


_PYTHON_LAUNCHER = r'''import importlib.util, json, sys
source, entry = sys.argv[1], sys.argv[2]
payload = json.loads(sys.stdin.read() or "null")
spec = importlib.util.spec_from_file_location("public_fixture", source)
if spec is None or spec.loader is None:
    raise RuntimeError("fixture loader unavailable")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
value = module
for part in entry.split("."):
    value = getattr(value, part)
if isinstance(payload, dict) and "args" in payload:
    result = value(*(payload.get("args") or ()), **(payload.get("kwargs") or {}))
else:
    result = value(payload)
print(json.dumps(result, sort_keys=True, ensure_ascii=False))
'''

_NODE_LAUNCHER = r'''const fs = require("fs");
const source = process.argv[2];
const entry = process.argv[3];
const payload = JSON.parse(fs.readFileSync(0, "utf8") || "null");
const mod = require(source);
let fn = mod;
for (const part of entry.split(".")) fn = fn[part];
let result;
if (payload && typeof payload === "object" && Object.prototype.hasOwnProperty.call(payload, "args")) {
  result = fn(...(payload.args || []));
} else {
  result = fn(payload);
}
Promise.resolve(result).then(value => process.stdout.write(JSON.stringify(value))).catch(error => { console.error(error.name + ": " + error.message); process.exit(2); });
'''


def execute_fixture(
    *,
    root: Path,
    path: str,
    entrypoint: str,
    expectation: TemporalExpectation,
    adapter_id: str,
    payload: object = None,
    timeout_seconds: float = 5.0,
) -> FixtureExecution:
    """Execute one public fixture through Python or Node without shell dispatch."""
    try:
        relative = _public_path(path)
        entry = _public_function(entrypoint)
    except ValueError as error:
        return FixtureExecution(adapter_id, ObservationStatus.UNRUNNABLE, None, reason=str(error))
    source = (Path(root) / relative).resolve()
    try:
        source.relative_to(Path(root).resolve())
    except ValueError:
        return FixtureExecution(adapter_id, ObservationStatus.UNRUNNABLE, None, reason="fixture escaped public root")
    if not source.is_file() or source.is_symlink():
        return FixtureExecution(adapter_id, ObservationStatus.UNRUNNABLE, None, reason="fixture source is unavailable")
    language = adapter_id.casefold()
    executable = shutil.which("node") if language in {"javascript", "typescript", "javascript-module-process", "typescript-module-process"} else shutil.which("python3")
    if executable is None:
        return FixtureExecution(adapter_id, ObservationStatus.UNRUNNABLE, None, reason="runtime executable is unavailable")
    with tempfile.TemporaryDirectory(prefix="aha11-fixture-") as directory:
        launcher = Path(directory) / ("launcher.js" if executable.endswith("node") else "launcher.py")
        launcher.write_text(_NODE_LAUNCHER if launcher.suffix == ".js" else _PYTHON_LAUNCHER, encoding="utf-8")
        command = [executable, str(launcher), str(source), entry] if launcher.suffix == ".js" else [executable, "-B", str(launcher), str(source), entry]
        try:
            completed = subprocess.run(
                command,
                input=json.dumps(payload, sort_keys=True),
                text=True,
                capture_output=True,
                cwd=str(Path(root).resolve()),
                env=_fixture_environment(),
                timeout=max(0.1, float(timeout_seconds)),
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            return FixtureExecution(adapter_id, ObservationStatus.ENVIRONMENT, None, stderr=str(error), reason="fixture timeout")
        except OSError as error:
            return FixtureExecution(adapter_id, ObservationStatus.ENVIRONMENT, None, reason=type(error).__name__)
    if completed.returncode != 0:
        return FixtureExecution(adapter_id, ObservationStatus.ENVIRONMENT, None, completed.stdout, completed.stderr, "fixture process failed")
    try:
        value = json.loads(completed.stdout.strip().splitlines()[-1])
        trace = TemporalTrace.from_mapping(value, expectation=expectation, adapter_id=adapter_id, receipt_id=_digest({"path": relative, "entry": entry, "output": value})[:32])
    except (IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
        return FixtureExecution(adapter_id, ObservationStatus.UNRUNNABLE, None, completed.stdout, completed.stderr, f"trace protocol invalid: {type(error).__name__}")
    return FixtureExecution(adapter_id, trace.status, trace, completed.stdout, completed.stderr)


@dataclass(frozen=True)
class DocumentExecution:
    status: ObservationStatus
    operation: str
    before: SerializedObservation
    after: SerializedObservation
    events: tuple[TraceEvent, ...]
    effects: tuple[EffectObservation, ...]


def execute_document_operation(*, path: Path, document: Mapping[str, object], operation: str = "read") -> DocumentExecution:
    """Exercise one real JSON/document persistence boundary with independent read-back."""
    operation = _text(operation, "document operation")
    if operation not in {"read", "replace"}:
        raise ValueError("unsupported document operation")
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = copy.deepcopy(dict(document))
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    before = SerializedObservation("document-before", payload, "document-writer", str(path))
    if operation == "replace":
        payload = {**payload, "_replacement": True}
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    readback = json.loads(path.read_text(encoding="utf-8"))
    after = SerializedObservation("document-after", readback, "document-reader", str(path), True)
    event = TraceEvent("document-read", "document", "read_document", {"path": path.name}, 0)
    effect = EffectObservation("document-read", "document_read", "document", 1, {"path": path.name})
    return DocumentExecution(ObservationStatus.OBSERVED_PASS, operation, before, after, (event,), (effect,))


def _route_key(clause_id: str, revision: str, tool: str) -> str:
    return _digest({"clause_id": clause_id, "revision": revision, "tool": tool})


def project_open_clause_actions(
    clauses: Sequence[Mapping[str, object]],
    available_tools: Sequence[str],
    *,
    revision: str,
    attempted: Sequence[str] = (),
) -> tuple[dict[str, object], ...]:
    """Return one concrete route per clause or settle it UNRUNNABLE once."""
    names = {str(item) for item in available_tools}
    attempted_keys = set(str(item) for item in attempted)
    result: list[dict[str, object]] = []
    for clause in clauses:
        clause_id = _text(clause.get("clause_ref", clause.get("clause_id", "")), "clause id")
        path = clause.get("path") or clause.get("target_path")
        entrypoint = clause.get("entrypoint")
        check = clause.get("check_id") or clause.get("check")
        receipt = clause.get("receipt_id") or clause.get("receipt_handle")
        candidates: list[tuple[str, dict[str, object], str]] = []
        if "probe" in names and path and entrypoint:
            try:
                safe_path = _public_path(path)
                safe_entry = _public_function(entrypoint)
            except ValueError:
                safe_path = safe_entry = ""
            if safe_path and safe_entry:
                candidates.append(("probe", {"mode": "temporal_trace", "path": safe_path, "entrypoint": safe_entry, "clause_id": clause_id, "revision": revision}, "execute the admitted temporal observation"))
        if "run_tests" in names and path and check:
            try:
                safe_path = _public_path(path)
            except ValueError:
                safe_path = ""
            if safe_path:
                candidates.append(("run_tests", {"path": safe_path, "check": _text(check, "check"), "clause_id": clause_id, "revision": revision}, "run the existing sandbox/source check"))
        if "read_check_receipt" in names and receipt:
            candidates.append(("read_check_receipt", {"handle": _text(receipt, "receipt handle"), "stream": "assertion_context", "clause_id": clause_id, "revision": revision}, "read the current check receipt without rerunning it"))
        if "read" in names and path:
            try:
                safe_path = _public_path(path)
            except ValueError:
                safe_path = ""
            if safe_path:
                candidates.append(("read", {"path": safe_path, "query": clause_id, "revision": revision}, "read the current public source boundary"))
        selected = candidates[0] if candidates else None
        if selected is None:
            result.append({"clause_id": clause_id, "status": ObservationStatus.UNRUNNABLE.value, "action": None, "reason": "no admitted execution or receipt route"})
            continue
        tool, arguments, reason = selected
        key = _route_key(clause_id, revision, tool)
        if key in attempted_keys:
            result.append({"clause_id": clause_id, "status": ObservationStatus.UNRUNNABLE.value, "action": None, "attempt_key": key, "reason": "route already settled unavailable; no unchanged retry"})
            continue
        result.append({"clause_id": clause_id, "status": ObservationStatus.UNCOVERED.value, "attempt_key": key, "action": {"tool": tool, "arguments": arguments}, "reason": reason})
    return tuple(result)


def build_honest_release_receipt(
    public_row: Mapping[str, object],
    *,
    source_revision: str,
    candidate_revision: str,
    parent_status: str,
    candidate_status: str,
    terminal_status: str,
    clause_ids: Sequence[str] = (),
    limitations: Sequence[str] = (),
) -> dict[str, object]:
    """Create an auditable release receipt with no evaluator/reference fields."""
    solver_row = public_solver_row(public_row)
    limits = tuple(dict.fromkeys(str(item) for item in limitations if str(item).strip()))
    statuses = {str(parent_status).upper(), str(candidate_status).upper()}
    eligible = (
        not limits
        and "ENVIRONMENT" not in statuses
        and "UNRUNNABLE" not in statuses
        and "UNCOVERED" not in statuses
        and str(parent_status).upper() == "COUNTEREXAMPLE"
        and str(candidate_status).upper() == "OBSERVED_PASS"
        and str(terminal_status).upper() in {"VERIFIED_FIXED", "PASS", "ACCEPTED"}
    )
    return {
        "schema": "aha11-honest-release-receipt/1",
        "eligible": eligible,
        "public_input_digest": _digest(solver_row),
        "solver_row_digest": str(solver_row["solver_input_digest"]),
        "solver_localization": solver_row.get("entry"),
        "source_revision": _text(source_revision, "source revision"),
        "candidate_revision": _text(candidate_revision, "candidate revision"),
        "parent_status": str(parent_status).upper(),
        "candidate_status": str(candidate_status).upper(),
        "terminal_status": str(terminal_status).upper(),
        "clause_ids": list(dict.fromkeys(str(item) for item in clause_ids if str(item).strip())),
        "limitations": list(limits),
        "evaluator_fields": "ABSENT",
        "reference_fix_fields": "ABSENT",
    }


__all__ = [
    "FixtureExecution", "DocumentExecution", "PublicLocalization", "SCHEMA", "SourceReservation", "TemporalExpectation",
    "TemporalPartition", "TemporalTrace", "assess_temporal_clause", "assess_temporal_trace",
    "build_honest_release_receipt", "compare_parent_candidate", "execute_document_operation",
    "execute_fixture", "project_open_clause_actions", "public_localization", "public_solver_identity",
    "public_solver_row", "reserve_source_unit", "temporal_completion_authorized", "temporal_evidence",
]
