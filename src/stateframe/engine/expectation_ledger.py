"""Typed enumeration expectations and a bounded full-suite runner.

The ledger records observations from deterministic adapters only.  It does not
scrape model prose, silently rebaseline after edits, or retain raw test output.
World facts and action receipts remain the durable evidence layers; this module
keeps the comparison index and emits typed references into those layers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Protocol, TypeAlias

from kernel_liveness import Evidence, OrganCensusSpec, TriState


logger = logging.getLogger(__name__)

SCHEMA = "poc.expectation_ledger/1"
ORGAN_ID = "expectation_ledger"
MAX_VISIBLE_CELLS = 5
MAX_VISIBLE_UNKNOWN = 1
MAX_OUTPUT_TAIL = 4_000

Record: TypeAlias = Mapping[str, object]
ReceiptSink: TypeAlias = Callable[[Mapping[str, object]], object]
SubprocessRun: TypeAlias = Callable[..., subprocess.CompletedProcess[str]]


class EnumerationKind(str, Enum):
    """Supported typed enumeration producers."""

    TEST_COLLECTION = "test_collection"
    TEST_EXECUTED = "test_executed"
    TEST_PASSED = "test_passed"
    TEST_FAILED = "test_failed"
    TEST_SKIPPED = "test_skipped"
    TEST_DESELECTED = "test_deselected"
    TEST_ERROR = "test_error"
    TEST_XFAILED = "test_xfailed"
    TEST_XPASSED = "test_xpassed"
    FILE_COUNT = "file_count"
    GREP_COUNT = "grep_count"
    ROW_COUNT = "row_count"
    ENUMERATION_UNKNOWN = "enumeration_unknown"


class ExpectationStatus(str, Enum):
    """State produced when an observation is compared with its history."""

    RECORDED = "RECORDED"
    UNCHANGED = "UNCHANGED"
    DELTA = "DELTA"
    UNKNOWN = "UNKNOWN"


class SuiteStatus(str, Enum):
    """Fail-closed suite result vocabulary."""

    PASS = "PASS"
    ASSERTION_FAIL = "ASSERTION_FAIL"
    NO_TESTS = "NO_TESTS"
    COLLECTION_ERROR = "COLLECTION_ERROR"
    HARNESS_ERROR = "HARNESS_ERROR"
    INFRA_ERROR = "INFRA_ERROR"
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"


class WorldFactWriter(Protocol):
    """Small structural interface for the existing world-cache store."""

    def observe_fact(self, **kwargs: object) -> object:
        ...


def _text(value: object) -> str:
    return str(value or "").strip()


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _clip(value: object, limit: int = 180) -> str:
    return _text(value)[:limit]


def _positive_int(value: object, field_name: str, *, allow_zero: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _kind(value: EnumerationKind | str) -> EnumerationKind:
    try:
        return value if isinstance(value, EnumerationKind) else EnumerationKind(value)
    except ValueError as exc:
        raise ValueError(f"unsupported enumeration kind: {value!r}") from exc


@dataclass(frozen=True)
class ExpectationKey:
    """Revision-independent identity for one comparable enumeration."""

    scope: str
    environment: str
    runner: str
    subject: str
    selector: str
    metric: str
    unit: str

    def __post_init__(self) -> None:
        for name in ("scope", "environment", "runner", "subject", "metric", "unit"):
            if not _text(getattr(self, name)):
                raise ValueError(f"{name} must not be empty")
        object.__setattr__(self, "scope", _text(self.scope))
        object.__setattr__(self, "environment", _text(self.environment))
        object.__setattr__(self, "runner", _text(self.runner))
        object.__setattr__(self, "subject", _text(self.subject))
        object.__setattr__(self, "selector", _text(self.selector))
        object.__setattr__(self, "metric", _text(self.metric))
        object.__setattr__(self, "unit", _text(self.unit))

    def to_dict(self) -> dict[str, str]:
        return {
            "scope": self.scope,
            "environment": self.environment,
            "runner": self.runner,
            "subject": self.subject,
            "selector": self.selector,
            "metric": self.metric,
            "unit": self.unit,
        }

    @property
    def identity(self) -> str:
        """Stable key; workspace revision intentionally is not included."""

        return _digest(self.to_dict())


@dataclass(frozen=True)
class ObservationProvenance:
    """Evidence pointer for one observation, without duplicating raw output."""

    turn: int | None = None
    workspace_revision: int = 0
    action_id: str = ""
    receipt_id: str = ""
    command: str = ""
    source_span: str = ""
    output_digest: str = ""

    def __post_init__(self) -> None:
        if self.turn is not None:
            _positive_int(self.turn, "turn")
        _positive_int(self.workspace_revision, "workspace_revision")
        object.__setattr__(self, "action_id", _clip(self.action_id))
        object.__setattr__(self, "receipt_id", _clip(self.receipt_id))
        object.__setattr__(self, "command", _clip(self.command, 500))
        object.__setattr__(self, "source_span", _clip(self.source_span, 240))
        object.__setattr__(self, "output_digest", _clip(self.output_digest, 120))

    def to_dict(self) -> dict[str, object]:
        return {
            "turn": self.turn,
            "workspace_revision": self.workspace_revision,
            "action_id": self.action_id,
            "receipt_id": self.receipt_id,
            "command": self.command,
            "source_span": self.source_span,
            "output_digest": self.output_digest,
        }


@dataclass(frozen=True)
class EnumerationObservation:
    """One typed count observation; ``None`` is an explicit unknown, never zero."""

    key: ExpectationKey
    kind: EnumerationKind
    value: int | None
    provenance: ObservationProvenance
    unknown_reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _kind(self.kind))
        if self.value is not None:
            _positive_int(self.value, "value")
            if self.unknown_reason:
                raise ValueError("known observations cannot have unknown_reason")
        elif not _text(self.unknown_reason):
            raise ValueError("unknown observations need unknown_reason")
        object.__setattr__(self, "unknown_reason", _clip(self.unknown_reason))

    @classmethod
    def count(
        cls,
        *,
        kind: EnumerationKind | str,
        key: ExpectationKey,
        value: int | None,
        provenance: ObservationProvenance,
        unknown_reason: str = "",
    ) -> "EnumerationObservation":
        return cls(key=key, kind=_kind(kind), value=value, provenance=provenance,
                   unknown_reason=unknown_reason)

    @property
    def known(self) -> bool:
        return self.value is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "key": self.key.to_dict(),
            "expectation_id": "expectation:" + self.key.identity.removeprefix("sha256:"),
            "value": self.value,
            "unknown_reason": self.unknown_reason,
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True)
class ExpectationRecord:
    """Current expectation plus immutable observation history."""

    expectation_id: str
    key: ExpectationKey
    current_value: int | None
    status: ExpectationStatus
    first_seen: ObservationProvenance
    last_seen: ObservationProvenance
    observations: tuple[EnumerationObservation, ...] = ()

    @property
    def value(self) -> int | None:
        return self.current_value

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "expectation_id": self.expectation_id,
            "key": self.key.to_dict(),
            "value": self.current_value,
            "status": self.status.value,
            "first_seen": self.first_seen.to_dict(),
            "last_seen": self.last_seen.to_dict(),
            "observations": [item.to_dict() for item in self.observations],
        }


@dataclass(frozen=True)
class ObservationResult:
    """Comparison result and the one bounded advisory/challenge decision."""

    record: ExpectationRecord
    observation: EnumerationObservation
    previous_value: int | None
    delta: int | None
    event: str
    challenge: bool
    advisory: str | None
    errors: tuple[str, ...] = ()

    @property
    def status(self) -> ExpectationStatus:
        return self.record.status


@dataclass(frozen=True)
class TypedSuiteResult:
    """Structural result for target or full pytest execution."""

    scope: str
    runner: str
    status: SuiteStatus | str
    exit_code: int | None
    collected: int | None
    passed: int | None
    failed: int | None
    skipped: int | None
    deselected: int | None
    errors: int | None
    xfailed: int | None
    xpassed: int | None
    selected: tuple[str, ...]
    command: tuple[str, ...]
    output_digest: str
    output_tail: str
    primary_failure: str | None = None

    def __post_init__(self) -> None:
        try:
            status = self.status if isinstance(self.status, SuiteStatus) else SuiteStatus(self.status)
        except ValueError as exc:
            raise ValueError(f"unsupported suite status: {self.status!r}") from exc
        object.__setattr__(self, "status", status)
        if self.exit_code is not None and not isinstance(self.exit_code, int):
            raise TypeError("exit_code must be an integer or null")
        for name in ("collected", "passed", "failed", "skipped", "deselected", "errors", "xfailed", "xpassed"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        object.__setattr__(self, "scope", _text(self.scope))
        object.__setattr__(self, "runner", _text(self.runner))
        object.__setattr__(self, "selected", tuple(_text(item) for item in self.selected if _text(item)))
        object.__setattr__(self, "command", tuple(_text(item) for item in self.command if _text(item)))
        object.__setattr__(self, "output_digest", _clip(self.output_digest, 120))
        object.__setattr__(self, "output_tail", _clip(self.output_tail, MAX_OUTPUT_TAIL))
        object.__setattr__(self, "primary_failure", _clip(self.primary_failure, 500) or None)

    @property
    def executed(self) -> int | None:
        if self.status is SuiteStatus.COLLECTION_ERROR:
            return None
        counts = (self.passed, self.failed, self.skipped, self.errors, self.xfailed, self.xpassed)
        return sum(counts) if all(value is not None for value in counts) else None

    @property
    def passed_authoritative(self) -> bool:
        return self.status is SuiteStatus.PASS and self.collected is not None and self.collected > 0

    def as_dict(self) -> dict[str, object]:
        return {
            "scope": self.scope,
            "runner": self.runner,
            "status": self.status.value,
            "exit_code": self.exit_code,
            "collected": self.collected,
            "executed": self.executed,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "deselected": self.deselected,
            "errors": self.errors,
            "xfailed": self.xfailed,
            "xpassed": self.xpassed,
            "selected": list(self.selected),
            "command": list(self.command),
            "output_digest": self.output_digest,
            "output_tail": self.output_tail,
            "primary_failure": self.primary_failure,
        }


def _provenance(
    *,
    turn: int | None,
    workspace_revision: int,
    action_id: str,
    receipt_id: str,
    command: str,
    source_span: str,
    output_digest: str,
) -> ObservationProvenance:
    return ObservationProvenance(
        turn=turn,
        workspace_revision=workspace_revision,
        action_id=action_id,
        receipt_id=receipt_id,
        command=command,
        source_span=source_span,
        output_digest=output_digest,
    )


def _count_key(
    *,
    kind: EnumerationKind,
    scope: str,
    environment: str,
    runner: str,
    subject: str,
    selector: str,
    unit: str,
) -> ExpectationKey:
    return ExpectationKey(
        scope=scope,
        environment=environment,
        runner=runner,
        subject=subject,
        selector=selector,
        metric=kind.value,
        unit=unit,
    )


class ExpectationLedger:
    """Compare typed enumerations while preserving all prior evidence."""

    def __init__(
        self,
        *,
        receipt_sink: ReceiptSink | None = None,
        world_store: WorldFactWriter | None = None,
        max_visible_cells: int = MAX_VISIBLE_CELLS,
    ) -> None:
        if max_visible_cells < 1:
            raise ValueError("max_visible_cells must be positive")
        self._records: dict[str, ExpectationRecord] = {}
        self._challenged: set[str] = set()
        self._cards: list[ObservationResult] = []
        self._receipt_sink = receipt_sink
        self._world_store = world_store
        self._max_visible_cells = min(max_visible_cells, MAX_VISIBLE_CELLS)
        self._sequence = 0

    @property
    def records(self) -> tuple[ExpectationRecord, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

    @property
    def expectations(self) -> tuple[ExpectationRecord, ...]:
        return self.records

    def get(self, key: ExpectationKey) -> ExpectationRecord | None:
        return self._records.get(key.identity)

    def observe(self, observation: EnumerationObservation) -> ObservationResult:
        """Record one observation and emit typed receipt/fact references."""

        self._sequence += 1
        if not observation.provenance.receipt_id:
            generated_receipt = "expectation-receipt:" + _digest(
                {
                    "sequence": self._sequence,
                    "expectation_id": observation.key.identity,
                    "value": observation.value,
                    "provenance": observation.provenance.to_dict(),
                }
            ).removeprefix("sha256:")[:24]
            observation = replace(
                observation,
                provenance=replace(observation.provenance, receipt_id=generated_receipt),
            )
        identity = observation.key.identity
        current = self._records.get(identity)
        known_previous = current.current_value if current and current.current_value is not None else None
        errors: list[str] = []

        if not observation.known:
            status = ExpectationStatus.UNKNOWN
            event = "ENUMERATION_UNKNOWN"
            challenge = False
            delta = None
            advisory = self._unknown_advisory(observation)
            current_value = known_previous
        elif current is None or current.current_value is None:
            status = ExpectationStatus.RECORDED
            event = "EXPECTATION_RECORDED"
            challenge = False
            delta = None
            advisory = self._recorded_advisory(observation)
            current_value = observation.value
        elif observation.value == current.current_value:
            status = ExpectationStatus.UNCHANGED
            event = "EXPECTATION_UNCHANGED"
            challenge = False
            delta = 0
            advisory = None
            current_value = current.current_value
        else:
            status = ExpectationStatus.DELTA
            event = "EXPECTATION_DELTA"
            delta = observation.value - current.current_value
            challenge = identity not in self._challenged
            if challenge:
                self._challenged.add(identity)
            advisory = self._delta_advisory(
                current=current,
                observation=observation,
                delta=delta,
                challenge=challenge,
            )
            current_value = observation.value

        expectation_id = "expectation:" + identity.removeprefix("sha256:")
        history = (current.observations if current else ()) + (observation,)
        first_seen = current.first_seen if current else observation.provenance
        record = ExpectationRecord(
            expectation_id=expectation_id,
            key=observation.key,
            current_value=current_value,
            status=status,
            first_seen=first_seen,
            last_seen=observation.provenance,
            observations=history,
        )
        self._records[identity] = record
        result = ObservationResult(
            record=record,
            observation=observation,
            previous_value=known_previous,
            delta=delta,
            event=event,
            challenge=challenge,
            advisory=advisory,
        )
        if status is ExpectationStatus.UNKNOWN or (status is ExpectationStatus.DELTA and challenge):
            self._cards.append(result)
        errors.extend(self._emit_receipt(result))
        errors.extend(self._emit_world_fact(result))
        if errors:
            result = replace(result, errors=tuple(errors))
        return result

    def observe_count(
        self,
        *,
        kind: EnumerationKind | str,
        value: int | None,
        scope: str,
        environment: str,
        runner: str,
        subject: str,
        selector: str = "",
        unit: str = "rows",
        turn: int | None = None,
        workspace_revision: int = 0,
        action_id: str = "",
        receipt_id: str = "",
        command: str = "",
        source_span: str = "",
        output_digest: str = "",
        unknown_reason: str = "",
    ) -> ObservationResult:
        enum_kind = _kind(kind)
        key = _count_key(
            kind=enum_kind,
            scope=scope,
            environment=environment,
            runner=runner,
            subject=subject,
            selector=selector,
            unit=unit,
        )
        return self.observe(
            EnumerationObservation.count(
                kind=enum_kind,
                key=key,
                value=value,
                unknown_reason=unknown_reason,
                provenance=_provenance(
                    turn=turn,
                    workspace_revision=workspace_revision,
                    action_id=action_id,
                    receipt_id=receipt_id,
                    command=command,
                    source_span=source_span,
                    output_digest=output_digest,
                ),
            )
        )

    def observe_file_count(self, value: int | None, **kwargs: object) -> ObservationResult:
        return self.observe_count(kind=EnumerationKind.FILE_COUNT, value=value, unit="files", **kwargs)

    def observe_grep_count(self, value: int | None, **kwargs: object) -> ObservationResult:
        return self.observe_count(kind=EnumerationKind.GREP_COUNT, value=value, unit="matches", **kwargs)

    def observe_row_count(self, value: int | None, **kwargs: object) -> ObservationResult:
        return self.observe_count(kind=EnumerationKind.ROW_COUNT, value=value, unit="rows", **kwargs)

    def observe_suite_result(
        self,
        result: TypedSuiteResult,
        *,
        environment: str,
        subject: str = "suite",
        selector: str = "",
        turn: int | None = None,
        workspace_revision: int = 0,
        action_id: str = "",
        receipt_id: str = "",
    ) -> tuple[ObservationResult, ...]:
        return tuple(
            self.observe(observation)
            for observation in suite_observations(
                result,
                environment=environment,
                subject=subject,
                selector=selector,
                turn=turn,
                workspace_revision=workspace_revision,
                action_id=action_id,
                receipt_id=receipt_id,
            )
        )

    def render(self, *, max_cells: int | None = None) -> str:
        """Render highest-signal cards under E394/E398 limits."""

        limit = self._max_visible_cells if max_cells is None else min(max_cells, MAX_VISIBLE_CELLS)
        if limit < 1:
            return ""
        selected: list[ObservationResult] = []
        unknown_seen = False
        ranked = sorted(
            enumerate(self._cards),
            key=lambda pair: (0 if pair[1].status is ExpectationStatus.DELTA else 1, pair[0]),
        )
        for _, result in ranked:
            if result.status is ExpectationStatus.UNKNOWN:
                if unknown_seen:
                    continue
                unknown_seen = True
            if len(selected) >= limit:
                continue
            selected.append(result)
        return "\n\n".join(self._render_result(result) for result in selected)

    @property
    def visible_cells(self) -> tuple[str, ...]:
        rendered = self.render()
        return tuple(part for part in rendered.split("\n\n") if part)

    def _emit_receipt(self, result: ObservationResult) -> list[str]:
        if self._receipt_sink is None:
            return []
        observation = result.observation
        receipt_id = observation.provenance.receipt_id or self._receipt_id(result)
        receipt: dict[str, object] = {
            "schema": SCHEMA,
            "kind": "expectation",
            "intent": "expectation-observation",
            "organ_id": ORGAN_ID,
            "event": result.event,
            "expectation_id": result.record.expectation_id,
            "key": observation.key.to_dict(),
            "value": observation.value,
            "previous_value": result.previous_value,
            "delta": result.delta,
            "status": result.status.value,
            "challenge": result.challenge,
            "receipt_id": receipt_id,
            "provenance": observation.provenance.to_dict(),
        }
        try:
            self._receipt_sink(receipt)
            if result.status is ExpectationStatus.DELTA:
                follow_up = dict(receipt)
                if result.challenge:
                    follow_up["event"] = "EXPECTATION_DELTA_CHALLENGE_ONCE"
                else:
                    follow_up["event"] = "EXPECTATION_DEDUPED"
                follow_up["receipt_id"] = f"{receipt_id}:follow-up"
                self._receipt_sink(follow_up)
        except Exception as exc:  # noqa: BLE001 - evidence sink cannot break the index
            logger.warning("[kernel-wiring] expectation receipt sink failed: %s", type(exc).__name__)
            return [f"receipt_sink:{type(exc).__name__}"]
        return []

    def _emit_world_fact(self, result: ObservationResult) -> list[str]:
        if self._world_store is None:
            return []
        observation = result.observation
        evidence_ref = observation.provenance.receipt_id or self._receipt_id(result)
        output_digest = observation.provenance.output_digest or _digest(observation.to_dict())
        try:
            self._world_store.observe_fact(
                path="expectation_ledger",
                predicate="enumeration_observed",
                subject_ref=result.record.expectation_id,
                value={
                    "key": observation.key.to_dict(),
                    "value": observation.value,
                    "status": result.status.value,
                    "event": result.event,
                },
                revision=observation.provenance.workspace_revision,
                evidence_ref=evidence_ref,
                content_digest=output_digest,
                spans=(),
                dependencies=(),
                dependencies_complete=True,
            )
        except Exception as exc:  # noqa: BLE001 - cache write is advisory to comparison
            logger.warning("[kernel-wiring] expectation world fact failed: %s", type(exc).__name__)
            return [f"world_store:{type(exc).__name__}"]
        return []

    def _receipt_id(self, result: ObservationResult) -> str:
        return "expectation-receipt:" + _digest(
            {
                "sequence": self._sequence,
                "expectation_id": result.record.expectation_id,
                "value": result.observation.value,
                "provenance": result.observation.provenance.to_dict(),
            }
        ).removeprefix("sha256:")[:24]

    @staticmethod
    def _recorded_advisory(observation: EnumerationObservation) -> str:
        return (
            f"RECORDED: {observation.key.subject} {observation.key.metric} "
            f"= {_format_value(observation.value)} {observation.key.unit}. "
            "Future observations for this exact scope and selector will be compared."
        )

    @staticmethod
    def _unknown_advisory(observation: EnumerationObservation) -> str:
        return (
            f"ENUMERATION UNKNOWN: could not parse {observation.key.metric} for "
            f"{observation.key.subject} ({observation.unknown_reason}); "
            "this was not recorded as zero."
        )

    @staticmethod
    def _delta_advisory(
        *,
        current: ExpectationRecord,
        observation: EnumerationObservation,
        delta: int,
        challenge: bool,
    ) -> str | None:
        if not challenge:
            return None
        prior = current.observations[-1]
        unit = observation.key.unit
        noun = unit if abs(delta) != 1 else unit.rstrip("s")
        return (
            "EXPECTATION DELTA:\n"
            f"You saw {_format_value(prior.value)} at {_where(prior.provenance)}.\n"
            f"Now you saw {_format_value(observation.value)} at {_where(observation.provenance)}.\n"
            f"Delta: {_format_delta(delta)} {noun}.\n"
            "Investigate before proceeding."
        )

    @staticmethod
    def _render_result(result: ObservationResult) -> str:
        if result.advisory:
            return result.advisory
        return f"{result.status.value}: {result.record.expectation_id}"


def _format_value(value: int | None) -> str:
    return "UNKNOWN" if value is None else f"{value:,}"


def _format_delta(value: int) -> str:
    return f"{value:+,}" if value else "0"


def _where(provenance: ObservationProvenance) -> str:
    turn = f"turn {provenance.turn}" if provenance.turn is not None else "an unknown turn"
    receipt = f", receipt {provenance.receipt_id}" if provenance.receipt_id else ""
    return turn + receipt


_COUNT_RE = re.compile(
    r"(?<!\w)(?P<count>\d+)\s+(?P<label>passed|failed|errors?|skipped|xfailed|xpassed|deselected)\b",
    re.IGNORECASE,
)
_COLLECTED_RE = re.compile(r"\bcollected\s+(?P<count>\d+)\s+items?\b", re.IGNORECASE)


def _summary_line(text: str) -> str:
    lines = text.splitlines()
    for line in reversed(lines[-80:]):
        if _COUNT_RE.search(line) and (" in " in line or "deselected" in line.lower()):
            return line
    for line in reversed(lines[-80:]):
        if _COUNT_RE.search(line):
            return line
    return ""


def _pytest_counts(text: str) -> dict[str, int | None]:
    collected_match = _COLLECTED_RE.search(text)
    summary = _summary_line(text)
    if not summary and not re.search(r"\bno tests ran\b", text, re.IGNORECASE):
        return {name: (int(collected_match.group("count"))
                       if name == "collected" and collected_match else None)
                for name in ("collected", "passed", "failed", "skipped", "deselected",
                             "errors", "xfailed", "xpassed")}
    counts = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "deselected": 0,
        "errors": 0,
        "xfailed": 0,
        "xpassed": 0,
    }
    for match in _COUNT_RE.finditer(summary):
        label = match.group("label").lower()
        key = "errors" if label.startswith("error") else label
        counts[key] += int(match.group("count"))
    summary_total = sum(counts.values())
    counts["collected"] = int(collected_match.group("count")) if collected_match else summary_total
    return counts


def _suite_status(text: str, exit_code: int) -> SuiteStatus:
    counts = _pytest_counts(text)
    if re.search(
        r"ERROR collecting|collected\s+0\s+items?.{0,500}(?:ImportError|ModuleNotFoundError|SyntaxError)",
        text,
        re.IGNORECASE | re.DOTALL,
    ):
        return SuiteStatus.COLLECTION_ERROR
    if exit_code == 5 or re.search(r"no tests ran", text, re.IGNORECASE):
        return SuiteStatus.NO_TESTS
    if exit_code == 0:
        return SuiteStatus.PASS if (counts["collected"] or 0) > 0 else SuiteStatus.UNKNOWN
    if exit_code == 1:
        return SuiteStatus.ASSERTION_FAIL if (counts["collected"] or 0) > 0 else SuiteStatus.UNKNOWN
    if exit_code == 4:
        return SuiteStatus.HARNESS_ERROR
    if exit_code in {2, 3}:
        return SuiteStatus.HARNESS_ERROR
    return SuiteStatus.UNKNOWN


def parse_pytest_result(
    stdout: str,
    stderr: str = "",
    exit_code: int = 0,
    *,
    scope: str = "full",
    selected: Sequence[str] = (),
    command: Sequence[str] = (),
) -> TypedSuiteResult:
    """Parse only structural pytest output into all ledger-relevant counts."""

    text = (stdout or "") + (("\n" + stderr) if stderr else "")
    counts = _pytest_counts(text)
    status = _suite_status(text, exit_code)
    primary = None
    if status in {SuiteStatus.ASSERTION_FAIL, SuiteStatus.COLLECTION_ERROR, SuiteStatus.HARNESS_ERROR}:
        primary = "pytest reported a non-passing suite"
    return TypedSuiteResult(
        scope=scope,
        runner="pytest",
        status=status,
        exit_code=exit_code,
        collected=counts["collected"],
        passed=counts["passed"],
        failed=counts["failed"],
        skipped=counts["skipped"],
        deselected=counts["deselected"],
        errors=counts["errors"],
        xfailed=counts["xfailed"],
        xpassed=counts["xpassed"],
        selected=tuple(selected),
        command=tuple(command),
        output_digest=_digest(text),
        output_tail=text[-MAX_OUTPUT_TAIL:],
        primary_failure=primary,
    )


def _error_result(*, scope: str, status: SuiteStatus, selected: Sequence[str], command: Sequence[str], detail: str) -> TypedSuiteResult:
    return TypedSuiteResult(
        scope=scope,
        runner="pytest",
        status=status,
        exit_code=None,
        collected=None,
        passed=None,
        failed=None,
        skipped=None,
        deselected=None,
        errors=None,
        xfailed=None,
        xpassed=None,
        selected=tuple(selected),
        command=tuple(command),
        output_digest=_digest(detail),
        output_tail=detail,
        primary_failure=detail,
    )


def _safe_targets(root: Path, values: Iterable[str | Path]) -> tuple[str, ...]:
    root_resolved = root.resolve()
    targets: list[str] = []
    seen: set[str] = set()
    for value in values:
        raw = Path(value)
        candidate = raw if raw.is_absolute() else root / raw
        resolved = candidate.resolve()
        try:
            relative = resolved.relative_to(root_resolved)
        except ValueError as exc:
            raise ValueError(f"suite target escapes root: {value}") from exc
        target = relative.as_posix() or "."
        if target not in seen:
            seen.add(target)
            targets.append(target)
    return tuple(targets)


def _safe_selectors(root: Path, values: Iterable[str]) -> tuple[str, ...]:
    selectors: list[str] = []
    for value in values:
        selector = _text(value)
        if not selector or selector.startswith("-"):
            raise ValueError("pytest selectors must be non-empty and cannot start with '-'")
        path_part, separator, node_id = selector.partition("::")
        safe_path = _safe_targets(root, (path_part,))[0]
        selectors.append(safe_path + (separator + node_id if separator else ""))
    return tuple(dict.fromkeys(selectors))


def run_typed_suite(
    root: Path,
    *,
    scope: str = "full",
    module_roots: Sequence[str | Path] = (),
    test_roots: Sequence[str | Path] = (),
    selectors: Sequence[str] = (),
    run: SubprocessRun | None = None,
    timeout: float = 300,
    env: Mapping[str, str] | None = None,
) -> TypedSuiteResult:
    """Run pytest over all declared roots; full mode never uses ``-x``."""

    if scope not in {"target", "full"}:
        raise ValueError("scope must be 'target' or 'full'")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    root = Path(root)
    try:
        if scope == "full":
            targets = _safe_targets(root, (*module_roots, *test_roots) or (".",))
        else:
            target_selectors = _safe_selectors(root, selectors)
            targets = (*target_selectors, *_safe_targets(root, test_roots)) or (".",)
    except ValueError as exc:
        return _error_result(scope=scope, status=SuiteStatus.HARNESS_ERROR, selected=(), command=(), detail=str(exc))
    from kernel_test_python import test_python
    command = (test_python(), "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider", *targets)
    process_env = os.environ.copy()
    if env:
        process_env.update({str(key): str(value) for key, value in env.items()})
    subprocess_run = run or subprocess.run
    try:
        completed = subprocess_run(
            list(command),
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=process_env,
        )
    except subprocess.TimeoutExpired:
        return _error_result(scope=scope, status=SuiteStatus.TIMEOUT, selected=targets, command=command,
                             detail=f"pytest timed out after {timeout}s")
    except (FileNotFoundError, OSError) as exc:
        return _error_result(scope=scope, status=SuiteStatus.INFRA_ERROR, selected=targets, command=command,
                             detail=f"pytest unavailable: {type(exc).__name__}")
    except Exception as exc:  # noqa: BLE001 - runner boundary fails closed
        logger.warning("[kernel-wiring] typed suite runner failed: %s", type(exc).__name__)
        return _error_result(scope=scope, status=SuiteStatus.INFRA_ERROR, selected=targets, command=command,
                             detail=f"pytest subprocess failed: {type(exc).__name__}")
    return parse_pytest_result(
        completed.stdout or "",
        completed.stderr or "",
        completed.returncode,
        scope=scope,
        selected=targets,
        command=command,
    )


run_full_suite = run_typed_suite
run_tests = run_typed_suite


_SUITE_METRICS: tuple[tuple[str, EnumerationKind], ...] = (
    ("collected", EnumerationKind.TEST_COLLECTION),
    ("executed", EnumerationKind.TEST_EXECUTED),
    ("passed", EnumerationKind.TEST_PASSED),
    ("failed", EnumerationKind.TEST_FAILED),
    ("skipped", EnumerationKind.TEST_SKIPPED),
    ("deselected", EnumerationKind.TEST_DESELECTED),
    ("errors", EnumerationKind.TEST_ERROR),
    ("xfailed", EnumerationKind.TEST_XFAILED),
    ("xpassed", EnumerationKind.TEST_XPASSED),
)


def suite_observations(
    result: TypedSuiteResult,
    *,
    environment: str,
    subject: str = "suite",
    selector: str = "",
    turn: int | None = None,
    workspace_revision: int = 0,
    action_id: str = "",
    receipt_id: str = "",
) -> tuple[EnumerationObservation, ...]:
    """Adapt every suite count into a separate comparable expectation."""

    command = " ".join(result.command)
    provenance = _provenance(
        turn=turn,
        workspace_revision=workspace_revision,
        action_id=action_id,
        receipt_id=receipt_id,
        command=command,
        source_span="pytest terminal summary",
        output_digest=result.output_digest,
    )
    values = {name: getattr(result, name) for name, _ in _SUITE_METRICS}
    return tuple(
        EnumerationObservation.count(
            kind=kind,
            key=_count_key(
                kind=kind,
                scope=result.scope,
                environment=environment,
                runner=result.runner,
                subject=subject,
                selector=selector,
                unit="tests",
            ),
            value=values[name],
            unknown_reason="pytest count not observed" if values[name] is None else "",
            provenance=provenance,
        )
        for name, kind in _SUITE_METRICS
    )


_CENSUS_EVENTS = frozenset({
    "EXPECTATION_RECORDED",
    "EXPECTATION_DELTA",
    "ENUMERATION_UNKNOWN",
    "EXPECTATION_UNCHANGED",
    "EXPECTATION_NOT_APPLICABLE",
    "EXPECTATION_DEDUPED",
    "EXPECTATION_DELTA_CHALLENGE_ONCE",
})


def _is_ledger_record(record: Record) -> bool:
    organ = _text(record.get("organ_id") or record.get("organ") or record.get("cylinder"))
    event = _text(record.get("event") or record.get("event_name"))
    if organ == ORGAN_ID:
        return event in _CENSUS_EVENTS
    return not organ and event in {
        *_CENSUS_EVENTS,
    }


def _expectation_ledger_trigger(evidence: Evidence) -> TriState:
    """Census trigger kept beside its declaration, as required by the registry law."""

    record = evidence.record
    if not isinstance(record, Mapping):
        return TriState.UNKNOWN
    organ = _text(record.get("organ_id") or record.get("organ") or record.get("cylinder"))
    event = _text(record.get("event") or record.get("event_name"))
    if organ == ORGAN_ID and event not in _CENSUS_EVENTS:
        return TriState.UNKNOWN
    if _is_ledger_record(record):
        return TriState.YES
    return TriState.NO


CENSUS_SPECS = (
    OrganCensusSpec(
        organ_id=ORGAN_ID,
        trigger=_expectation_ledger_trigger,
        fire_events=frozenset({"EXPECTATION_RECORDED", "EXPECTATION_DELTA", "ENUMERATION_UNKNOWN"}),
        suppression_events=frozenset({"EXPECTATION_UNCHANGED", "EXPECTATION_NOT_APPLICABLE"}),
        dedup_events=frozenset({"EXPECTATION_DEDUPED"}),
        challenge_once_events=frozenset({"EXPECTATION_DELTA_CHALLENGE_ONCE"}),
    ),
)


__all__ = [
    "CENSUS_SPECS",
    "EnumerationKind",
    "EnumerationObservation",
    "ExpectationKey",
    "ExpectationLedger",
    "ExpectationRecord",
    "ExpectationStatus",
    "MAX_VISIBLE_CELLS",
    "ObservationProvenance",
    "ObservationResult",
    "ORGAN_ID",
    "SuiteStatus",
    "TypedSuiteResult",
    "parse_pytest_result",
    "run_full_suite",
    "run_tests",
    "run_typed_suite",
    "suite_observations",
]
