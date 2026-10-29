"""Bounded ground-truth organs for the component parity slice.

This module is deliberately a shadow organ. It does not reject actions or
perform probes; it identifies the one cheap measurement that should happen
next, records whether evidence was later consumed, and mirrors nearby source
conventions before a new file or function is created. The live ReAct wire
points remain documented seams for ``react_repair`` and ``kernel_bolt``.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import TypeAlias

from kernel_liveness import Evidence, OrganCensusSpec, TriState


SCHEMA = "poc.kernel_groundtruth/1"
MAX_RENDER_CHARS = 300
Record: TypeAlias = Mapping[str, object]
PathSource: TypeAlias = tuple[str, str]


class AssertionKind(StrEnum):
    """Workspace facts that can be asserted before a tool action."""

    SYMBOL_EXISTS = "SYMBOL_EXISTS"
    FIXTURE_ROWS = "FIXTURE_ROWS"
    IMPORT_RESOLVES = "IMPORT_RESOLVES"
    ENUM_MEMBER = "ENUM_MEMBER"
    CONSTANT = "CONSTANT"
    SCHEMA_KEY = "SCHEMA_KEY"
    LITERAL_SHAPE = "LITERAL_SHAPE"


class FactKind(StrEnum):
    """Evidence families tracked by the consumption organ."""

    WORLD = "world"
    ORGAN = "organ"
    READ = "read"


def _text(value: object) -> str:
    return str(value or "").strip()


def _clip(value: object, limit: int = MAX_RENDER_CHARS) -> str:
    return _text(value)[:limit]


def _int(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


def _path(value: object) -> str:
    return _text(value).replace("\\", "/").lstrip("./")


def _mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(value)
    return ()


def _action_text(record: Record) -> str:
    parts: list[str] = []
    for key in (
        "kind",
        "intent",
        "subject_ref",
        "target_path",
        "path",
        "file",
        "target",
        "test",
        "nodeid",
        "symbol",
        "why",
        "detail",
    ):
        value = record.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            parts.extend(_text(item) for item in value)
        elif value is not None:
            parts.append(_text(value))
    return " ".join(item for item in parts if item)


_PATH_WITH_SPAN = re.compile(
    r"(?P<path>[A-Za-z0-9_.-][A-Za-z0-9_./-]*\.[A-Za-z0-9_]+)"
    r"(?:@(?P<start>\d+)(?:-(?P<end>\d+))?|:(?P<colon>\d+))?"
)


def _path_span(value: object) -> tuple[str, int | None, int | None]:
    match = _PATH_WITH_SPAN.search(_text(value))
    if match is None:
        return "", None, None
    start_text = match.group("start") or match.group("colon")
    start = int(start_text) if start_text else None
    end = int(match.group("end")) if match.group("end") else start
    return _path(match.group("path")), start, end


def _same_path(left: str, right: str) -> bool:
    a, b = _path(left).casefold(), _path(right).casefold()
    return bool(a and b and (a == b or a.endswith("/" + b) or b.endswith("/" + a)))


def _span_overlap(
    left: tuple[int | None, int | None], right: tuple[int | None, int | None]
) -> bool:
    if None in left or None in right:
        return True
    return left[0] <= right[1] and right[0] <= left[1]  # type: ignore[operator]


@dataclass(frozen=True)
class PlanAssertion:
    """A typed claim made by a plan, never inferred from general prose."""

    subject_ref: str
    kind: AssertionKind | str
    asserted_value: object = True
    target_path: str = ""
    assertion_id: str = ""
    observed_fact_id: str = ""
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        subject = _text(self.subject_ref)
        if not subject:
            raise ValueError("PlanAssertion.subject_ref must not be empty")
        try:
            kind = self.kind if isinstance(self.kind, AssertionKind) else AssertionKind(str(self.kind).upper())
        except ValueError as exc:
            raise ValueError(f"unsupported assertion kind: {self.kind!r}") from exc
        object.__setattr__(self, "subject_ref", subject[:180])
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "target_path", _path(self.target_path)[:240])
        object.__setattr__(self, "assertion_id", _text(self.assertion_id)[:120])
        object.__setattr__(self, "observed_fact_id", _text(self.observed_fact_id)[:180])
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def identity(self) -> str:
        identity = {
            "subject_ref": self.subject_ref,
            "kind": self.kind.value,
            "asserted_value": self.asserted_value,
            "target_path": self.target_path,
        }
        # Untyped legacy edit receipts are separate observations when their
        # receipt fingerprints differ; typed plan assertions retain semantic
        # challenge-once identity independent of transport metadata.
        if self.provenance.get("legacy_edit"):
            identity["receipt_fingerprint"] = self.provenance.get("receipt_fingerprint", "")
        return _digest(identity)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "PlanAssertion":
        return cls(
            subject_ref=str(value.get("subject_ref") or value.get("subject") or value.get("symbol") or ""),
            kind=str(value.get("kind") or value.get("assertion_kind") or AssertionKind.SYMBOL_EXISTS),
            asserted_value=value.get("asserted_value", value.get("value", True)),
            target_path=str(value.get("target_path") or value.get("path") or ""),
            assertion_id=str(value.get("assertion_id") or ""),
            observed_fact_id=str(value.get("observed_fact_id") or value.get("fact_id") or ""),
            provenance=value.get("provenance") if isinstance(value.get("provenance"), Mapping) else {},
        )


_SOURCE_PATH_SUFFIXES = frozenset({
    ".c", ".cc", ".cpp", ".css", ".go", ".h", ".html", ".java", ".js", ".json",
    ".jsx", ".md", ".php", ".py", ".rb", ".rs", ".sql", ".ts", ".tsx", ".vue",
    ".xml", ".yaml", ".yml",
})


def _looks_like_source_path(value: object) -> bool:
    path, _, _ = _path_span(value)
    if not path:
        return False
    return "/" in path or Path(path).suffix.casefold() in _SOURCE_PATH_SUFFIXES


def _assertion_target_path(assertion: PlanAssertion) -> str:
    if assertion.target_path:
        return _path_span(assertion.target_path)[0] or _path(assertion.target_path)
    provenance_path = assertion.provenance.get("path")
    return _path_span(provenance_path)[0] if provenance_path else ""


def _has_real_subject(assertion: PlanAssertion) -> bool:
    """Reject file paths as subjects; probes need a symbol or source literal."""

    subject = _text(assertion.subject_ref)
    if not subject or _looks_like_source_path(subject):
        return False
    target = _assertion_target_path(assertion)
    if not target:
        return True
    subject_normalized = subject.replace("\\", "/").casefold()
    target_normalized = target.replace("\\", "/").casefold()
    if target_normalized in subject_normalized:
        return False
    subject_path, _, _ = _path_span(subject)
    if not subject_path:
        return True
    if _same_path(subject_path, target):
        return False
    return Path(subject_path).name.casefold() != Path(target).name.casefold()


def _legacy_edit_subject(action: Record) -> str:
    """Keep old file-only edit tapes measurable without probing their filename."""

    text = _text(action.get("detail") or action.get("intent"))
    match = re.search(r"\b(str_replace|replace|edit)\b", text, re.IGNORECASE)
    return match.group(1) if match else "edit"


@dataclass(frozen=True)
class Measurement:
    """One receipt-backed workspace measurement."""

    subject_ref: str
    kind: AssertionKind | str
    value: object
    target_path: str = ""
    turn: int = 0
    receipt_id: str = ""
    source_span: str = ""
    workspace_revision: int = 0
    fact_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject_ref", _text(self.subject_ref)[:180])
        try:
            kind = self.kind if isinstance(self.kind, AssertionKind) else AssertionKind(str(self.kind).upper())
        except ValueError:
            kind = AssertionKind.SYMBOL_EXISTS
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "target_path", _path(self.target_path)[:240])
        object.__setattr__(self, "turn", max(0, _int(self.turn)))
        object.__setattr__(self, "workspace_revision", max(0, _int(self.workspace_revision)))
        object.__setattr__(self, "receipt_id", _text(self.receipt_id)[:120])
        object.__setattr__(self, "source_span", _clip(self.source_span, 180))
        if not self.fact_id:
            object.__setattr__(self, "fact_id", "fact:" + _digest(self.to_dict()).removeprefix("sha256:"))

    def to_dict(self) -> dict[str, object]:
        return {
            "fact_id": self.fact_id,
            "subject_ref": self.subject_ref,
            "kind": self.kind.value,
            "value": self.value,
            "target_path": self.target_path,
            "turn": self.turn,
            "receipt_id": self.receipt_id,
            "source_span": self.source_span,
            "workspace_revision": self.workspace_revision,
        }


@dataclass(frozen=True)
class MeasureResult:
    """Non-blocking B decision and its exact next probe, when needed."""

    assertion: PlanAssertion
    event: str
    challenge: bool
    advisory: str | None
    probe: str | None
    observed: Measurement | None = None
    allowed: bool = True
    turn: int = 0

    @property
    def grounded(self) -> bool:
        return self.event in {"MEASURE_PREANSWER_MATCH", "MEASURE_PROBE_OBSERVED"}


def _probe_for(assertion: PlanAssertion) -> str | None:
    if not _has_real_subject(assertion):
        return None
    path = assertion.target_path or _path_span(assertion.provenance.get("path"))[0]
    subject = assertion.subject_ref.replace("'", "\\'")
    if assertion.kind in {AssertionKind.FIXTURE_ROWS}:
        return f"pytest --collect-only -q {path}" if path else None
    if assertion.kind in {AssertionKind.IMPORT_RESOLVES}:
        return f"grep -n 'import {subject}' {path}" if path else f"grep -R -n 'import {subject}' ."
    if assertion.kind in {AssertionKind.SYMBOL_EXISTS, AssertionKind.ENUM_MEMBER, AssertionKind.CONSTANT, AssertionKind.SCHEMA_KEY}:
        return f"grep -n '{subject}' {path}" if path else f"grep -R -n '{subject}' ."
    if assertion.kind is AssertionKind.LITERAL_SHAPE:
        return f"read {path}" if path else None
    return None


def _value_matches(assertion: PlanAssertion, measurement: Measurement) -> bool:
    if assertion.target_path and measurement.target_path and not _same_path(assertion.target_path, measurement.target_path):
        return False
    if assertion.kind != measurement.kind and assertion.kind not in {
        AssertionKind.SYMBOL_EXISTS,
        AssertionKind.IMPORT_RESOLVES,
    }:
        return False
    asserted = assertion.asserted_value
    measured = measurement.value
    if asserted is True and measured in (True, "OK", "HITS", "EXISTS"):
        return True
    if isinstance(asserted, str) or isinstance(measured, str):
        return str(asserted) == str(measured)
    return asserted == measured


def _assertion_from_action(action: Record) -> tuple[PlanAssertion, ...]:
    raw_values: list[object] = []
    for key in ("assertions", "plan_assertions"):
        raw_values.extend(_sequence(action.get(key)))
    for key in ("assertion", "plan_assertion"):
        if isinstance(action.get(key), Mapping):
            raw_values.append(action[key])
    if isinstance(action.get("asserted_value"), (str, int, float, bool)):
        raw_values.append(action)
    had_typed_assertions = bool(raw_values)
    assertions: list[PlanAssertion] = []
    for raw in raw_values:
        if isinstance(raw, PlanAssertion):
            assertions.append(raw)
        elif isinstance(raw, Mapping):
            try:
                assertions.append(PlanAssertion.from_mapping(raw))
            except ValueError:
                continue
    if assertions:
        return tuple(item for item in assertions if _has_real_subject(item))

    if had_typed_assertions:
        return ()

    if action.get('assertion_policy') == 'explicit_only':
        return ()

    # An explicit subject with no valid typed assertion is not safe to infer
    # from the operation name or path; drop it rather than inventing a probe.
    if action.get("subject_ref") is not None or action.get("symbol") is not None:
        return ()

    # Receipt tapes predate typed plans.  An applied edit still asserts that
    # its target exists; use the operation literal, never the target filename,
    # so legacy tapes retain a safe measurement without self-grepping.
    if _text(action.get("kind")).lower() in {"edit", "str_replace"}:
        target, _, _ = _path_span(action.get("target_path") or action.get("path") or action.get("intent"))
        if target:
            return (PlanAssertion(
                subject_ref=_legacy_edit_subject(action),
                kind=AssertionKind.SYMBOL_EXISTS,
                target_path=target,
                provenance={
                    "path": target,
                    "legacy_edit": True,
                    "receipt_fingerprint": _text(action.get("fingerprint")),
                },
            ),)
    return ()


class MeasureBeforeInfer:
    """Receipt-backed, challenge-once measure-first organ."""

    organ_id = "measure_first"

    def __init__(self) -> None:
        self._measurements: list[Measurement] = []
        self._challenged: set[str] = set()
        self._events: list[Record] = []

    @property
    def measurements(self) -> tuple[Measurement, ...]:
        return tuple(self._measurements)

    @property
    def events(self) -> tuple[Record, ...]:
        return tuple(self._events)

    def observe_measurement(self, measurement: Measurement) -> None:
        if not any(item.fact_id == measurement.fact_id for item in self._measurements):
            self._measurements.append(measurement)

    observe_fact = observe_measurement

    def observe_receipt(self, receipt: Record) -> None:
        kind = _text(receipt.get("kind")).lower()
        result = _text(receipt.get("result_class") or receipt.get("result")).upper()
        if result in {"RED", "REJECTED", "ERROR"}:
            return
        if kind == "read" and result != "OK":
            return
        source = (receipt["target_path"] if "target_path" in receipt
                  else receipt.get("path") or receipt.get("file") or receipt.get("intent"))
        path, _, _ = _path_span(source)
        subject = _text(receipt.get("subject_ref") or receipt.get("symbol"))
        if not subject:
            subject = _text(receipt.get("intent"))
            if kind == "read":
                subject = path
        if not subject:
            return
        if kind == "read":
            measurement = Measurement(subject, AssertionKind.SYMBOL_EXISTS, True, path, _int(receipt.get("turn")), _receipt_id(receipt), _text(source), _int(receipt.get("workspace_revision")))
        elif kind == "grep":
            value = _text(receipt.get("detail")) or result
            measurement = Measurement(subject, AssertionKind.SYMBOL_EXISTS, value, path, _int(receipt.get("turn")), _receipt_id(receipt), _text(source), _int(receipt.get("workspace_revision")))
        elif kind == "test" and "collect" in _action_text(receipt).lower():
            measurement = Measurement(subject, AssertionKind.FIXTURE_ROWS, receipt.get("collected", result), path, _int(receipt.get("turn")), _receipt_id(receipt), _text(source), _int(receipt.get("workspace_revision")))
        else:
            return
        self.observe_measurement(measurement)
        self._events.append({
            "organ_id": self.organ_id,
            "event": "MEASURE_PROBE_OBSERVED",
            "turn": _int(receipt.get("turn")),
            "emitted": False,
            "detail": measurement.subject_ref,
        })

    def check(
        self,
        assertion: PlanAssertion,
        *,
        turn: int = 0,
        workspace_revision: int | None = None,
    ) -> MeasureResult:
        if not _has_real_subject(assertion):
            return self._result(assertion, "MEASURE_NOT_A_VALUE_ASSERTION", False, None, None, None, turn)
        observed = next(
            (
                item
                for item in reversed(self._measurements)
                if (workspace_revision is None or item.workspace_revision == workspace_revision)
                and (
                    assertion.observed_fact_id
                    and item.fact_id == assertion.observed_fact_id
                    or not assertion.observed_fact_id
                    and (
                    item.subject_ref.casefold() == assertion.subject_ref.casefold()
                    or (
                        assertion.target_path
                        and _same_path(item.target_path, assertion.target_path)
                        and _same_path(item.subject_ref, assertion.target_path)
                    )
                    )
                )
            ),
            None,
        )
        if observed is not None and _value_matches(assertion, observed):
            return self._result(assertion, "MEASURE_PREANSWER_MATCH", False, None, None, observed, turn)
        if observed is not None and not _value_matches(assertion, observed):
            advisory = _clip(f"MEASURE-FIRST: observed {observed.value!r} for {assertion.subject_ref}; revise before acting.")
            return self._result(assertion, "MEASURE_PREANSWER_CONTRADICTION", False, advisory, None, observed, turn)

        key = assertion.identity
        if key in self._challenged:
            return self._result(assertion, "MEASURE_DEDUPED", False, None, None, observed, turn)
        self._challenged.add(key)
        probe = _probe_for(assertion)
        if probe is None:
            return self._result(assertion, "MEASURE_UNKNOWN", True, _clip("MEASURE-FIRST: no safe deterministic probe; keep the claim ungrounded."), None, observed, turn)
        advisory = _clip(f"MEASURE-FIRST: verify {assertion.subject_ref} first; next action: {probe}")
        return self._result(assertion, "MEASURE_FIRST_CHALLENGE", True, advisory, probe, observed, turn)

    def check_action(self, action: Record) -> tuple[MeasureResult, ...]:
        turn = _int(action.get("turn"))
        assertions = _assertion_from_action(action)
        if not assertions:
            return ()
        revision = _int(action.get("workspace_revision"))
        return tuple(
            self.check(assertion, turn=turn, workspace_revision=revision)
            for assertion in assertions
        )

    def _result(
        self,
        assertion: PlanAssertion,
        event: str,
        challenge: bool,
        advisory: str | None,
        probe: str | None,
        observed: Measurement | None,
        turn: int,
    ) -> MeasureResult:
        result = MeasureResult(assertion, event, challenge, advisory, probe, observed, True, turn)
        self._events.append({
            "organ_id": self.organ_id,
            "event": event,
            "turn": turn,
            "emitted": bool(advisory),
            "detail": advisory or assertion.subject_ref,
        })
        if challenge:
            self._events.append({
                "organ_id": self.organ_id,
                "event": "MEASURE_CHALLENGE_ONCE",
                "turn": turn,
                "emitted": bool(advisory),
                "detail": assertion.subject_ref,
            })
        return result

    def render(self, result: MeasureResult | None = None) -> str:
        if result is None:
            result = next((item for item in reversed(self._events) if item.get("detail")), None)
            return _clip(result.get("detail") if result else "")
        return _clip(result.advisory or "")


def _receipt_id(record: Record) -> str:
    return _text(record.get("receipt_id") or record.get("id") or record.get("fingerprint"))


@dataclass(frozen=True)
class ObservedFact:
    """Consumption index row; field names mirror WorldFact/FactProvenance."""

    fact_id: str
    kind: FactKind | str
    subject_ref: str
    path: str = ""
    spans: tuple[tuple[int, int], ...] = ()
    evidence_ref: str = ""
    turn: int = 0
    used_by: tuple[str, ...] = ()
    dynamic: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "fact_id", _text(self.fact_id) or _digest(self.subject_ref))
        try:
            kind = self.kind if isinstance(self.kind, FactKind) else FactKind(str(self.kind).lower())
        except ValueError as exc:
            raise ValueError(f"unsupported fact kind: {self.kind!r}") from exc
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "subject_ref", _text(self.subject_ref)[:180])
        object.__setattr__(self, "path", _path(self.path)[:240])
        object.__setattr__(self, "turn", max(0, _int(self.turn)))
        spans: list[tuple[int, int]] = []
        for raw in self.spans:
            if isinstance(raw, Sequence) and len(raw) >= 2:
                start, end = _int(raw[0]), _int(raw[1], _int(raw[0]))
                if start > 0 and end >= start:
                    spans.append((start, end))
        object.__setattr__(self, "spans", tuple(dict.fromkeys(spans)))
        object.__setattr__(self, "evidence_ref", _text(self.evidence_ref)[:160])
        object.__setattr__(self, "used_by", tuple(dict.fromkeys(_text(item) for item in self.used_by if _text(item))))

    @property
    def consumed(self) -> bool:
        return bool(self.used_by)

    def to_dict(self) -> dict[str, object]:
        return {
            "fact_id": self.fact_id,
            "kind": self.kind.value,
            "subject_ref": self.subject_ref,
            "path": self.path,
            "spans": [{"start": start, "end": end} for start, end in self.spans],
            "evidence_ref": self.evidence_ref,
            "turn": self.turn,
            "used_by": list(self.used_by),
            "dynamic": self.dynamic,
        }


def _spans_from_mapping(value: object) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    for raw in _sequence(value):
        if isinstance(raw, Mapping):
            start, end = raw.get("start"), raw.get("end", raw.get("start"))
        elif isinstance(raw, Sequence) and len(raw) >= 2:
            start, end = raw[0], raw[1]
        else:
            continue
        start_i, end_i = _int(start), _int(end, _int(start))
        if start_i > 0 and end_i >= start_i:
            result.append((start_i, end_i))
    return tuple(result)


class ConsumptionTracker:
    """Track observed facts until a later edit/test targets each fact."""

    organ_id = "consumption_judgment"

    def __init__(self) -> None:
        self._facts: dict[str, ObservedFact] = {}
        self._events: list[Record] = []

    @property
    def facts(self) -> tuple[ObservedFact, ...]:
        return tuple(self._facts[key] for key in sorted(self._facts))

    @property
    def events(self) -> tuple[Record, ...]:
        return tuple(self._events)

    def record_world_fact(self, fact: object, *, turn: int = 0) -> ObservedFact | None:
        value = fact.to_dict() if hasattr(fact, "to_dict") else fact
        mapping = _mapping(value)
        if mapping is None:
            return None
        key = _mapping(mapping.get("key")) or mapping
        source = _mapping(mapping.get("source")) or mapping
        row = ObservedFact(
            fact_id=_text(mapping.get("fact_id")) or _digest(mapping),
            kind=FactKind.WORLD,
            subject_ref=_text(mapping.get("subject_ref") or key.get("subject_ref") or mapping.get("predicate")),
            path=_text(mapping.get("path") or key.get("path")),
            spans=_spans_from_mapping(mapping.get("spans", source.get("spans"))),
            evidence_ref=_text(mapping.get("evidence_ref") or source.get("evidence_ref")),
            turn=turn or _int(mapping.get("turn") or mapping.get("workspace_revision")),
            dynamic=bool(mapping.get("dynamic")),
        )
        return self.record(row)

    def record_read_span(
        self,
        path: str,
        start: int | None,
        end: int | None,
        *,
        turn: int = 0,
        evidence_ref: str = "",
    ) -> ObservedFact:
        normalized_path = _path(path)
        spans = ((start, end or start),) if start and start > 0 else ()
        row = ObservedFact(
            fact_id=_digest({"kind": FactKind.READ.value, "path": normalized_path, "spans": spans}),
            kind=FactKind.READ,
            subject_ref=normalized_path,
            path=normalized_path,
            spans=spans,
            evidence_ref=evidence_ref,
            turn=turn,
        )
        return self.record(row)

    def record_organ_line(self, name: str, line: str, *, turn: int = 0, evidence_ref: str = "") -> tuple[ObservedFact, ...]:
        targets = tuple(dict.fromkeys(_path_span(match.group(0))[0] for match in _PATH_WITH_SPAN.finditer(line) if _path_span(match.group(0))[0]))
        if not targets:
            targets = (_text(line)[:180],)
        return tuple(self.record(ObservedFact(
            fact_id=_digest({"kind": FactKind.ORGAN.value, "name": name, "target": target, "turn": turn}),
            kind=FactKind.ORGAN,
            subject_ref=target,
            path=target if "." in target else "",
            evidence_ref=evidence_ref,
            turn=turn,
        )) for target in targets)

    def record(self, fact: ObservedFact) -> ObservedFact:
        existing = self._facts.get(fact.fact_id)
        if existing is not None:
            return existing
        self._facts[fact.fact_id] = fact
        self._events.append({"organ_id": self.organ_id, "event": "CONSUMPTION_OBSERVED", "detail": fact.subject_ref, "turn": fact.turn})
        return fact

    def observe_receipt(self, receipt: Record) -> None:
        kind = _text(receipt.get("kind")).lower()
        turn = _int(receipt.get("turn"))
        embedded = receipt.get("world_fact") or receipt.get("fact")
        if embedded is not None:
            self.record_world_fact(embedded, turn=turn)
        if kind == "read" and _text(receipt.get("result_class") or "OK").upper() in {"OK", "HITS"}:
            path, start, end = _path_span(receipt.get("path") or receipt.get("intent"))
            if path:
                self.record_read_span(path, start, end, turn=turn, evidence_ref=_receipt_id(receipt))
        event = _text(receipt.get("event") or receipt.get("event_name")).upper()
        if kind in {"organ_out", "organ"} or event == "ORGAN_DELIVERED":
            name = _text(receipt.get("organ_id") or receipt.get("organ") or "organ")
            line = _text(receipt.get("line") or receipt.get("detail"))
            if line:
                self.record_organ_line(name, line, turn=turn, evidence_ref=_receipt_id(receipt))
        if kind in {"edit", "str_replace", "test"}:
            self.observe_action(receipt)

    def observe_action(self, action: Record) -> tuple[ObservedFact, ...]:
        kind = _text(action.get("kind")).lower()
        if kind not in {"edit", "str_replace", "test"}:
            return ()
        turn = _int(action.get("turn"))
        action_id = _receipt_id(action) or f"turn:{turn}"
        action_path, start, end = _path_span(action.get("target_path") or action.get("path") or action.get("intent"))
        action_span = (start, end)
        text = _action_text(action).casefold()
        consumed: list[ObservedFact] = []
        for fact_id, fact in tuple(self._facts.items()):
            if turn <= fact.turn:
                continue
            path_match = bool(action_path and fact.path and _same_path(action_path, fact.path))
            span_match = path_match and _span_overlap(action_span, (fact.spans[0] if fact.spans else (None, None)))
            subject_match = bool(fact.subject_ref and fact.subject_ref.casefold() in text)
            if path_match and (not fact.spans or span_match) or subject_match:
                if action_id not in fact.used_by:
                    updated = replace(fact, used_by=(*fact.used_by, action_id))
                    self._facts[fact_id] = updated
                    consumed.append(updated)
                    self._events.append({"organ_id": self.organ_id, "event": "CONSUMPTION_FOUND", "detail": f"{fact.subject_ref} used by {action_id}", "turn": turn})
        return tuple(consumed)

    def summary(self) -> dict[str, object]:
        counts = Counter(fact.kind.value for fact in self.facts)
        consumed = sum(fact.consumed for fact in self.facts)
        dead = len(self._facts) - consumed
        return {"schema": SCHEMA, "observed": len(self._facts), "consumed": consumed, "dead": dead, "by_kind": dict(sorted(counts.items()))}

    def render(self) -> str:
        summary = self.summary()
        dead_facts = [fact.subject_ref for fact in self.facts if not fact.consumed][:3]
        suffix = f"; dead: {', '.join(dead_facts)}" if dead_facts else ""
        return _clip(f"CONSUMPTION: consumed={summary['consumed']} dead={summary['dead']}{suffix}")

    render_trim_digest = render


def _profile(source: str) -> tuple[str, ...]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ()
    labels: list[str] = []
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    if imports:
        from_count = sum(isinstance(node, ast.ImportFrom) for node in imports)
        labels.append("from-imports" if from_count >= len(imports) / 2 else "import-statements")
    if any(isinstance(node, ast.ImportFrom) and node.module == "__future__" and any(alias.name == "annotations" for alias in node.names) for node in imports):
        labels.append("future-annotations")
    functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    test_functions = [node for node in functions if node.name.startswith("test")]
    if test_functions:
        labels.append("test_snake_case" if all(re.fullmatch(r"test_[a-z0-9_]+", node.name) for node in test_functions) else "test_non_snake_case")
    named = [node.name for node in functions]
    if named:
        labels.append("functions_snake_case" if all(re.fullmatch(r"[a-z][a-z0-9_]*", name) for name in named) else "functions_mixed_case")
    if ast.get_docstring(tree):
        labels.append("module-docstring")
    first_definition = next((node.lineno for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))), None)
    if imports and first_definition is not None and max(node.end_lineno or node.lineno for node in imports) < first_definition:
        labels.append("imports-before-definitions")
    return tuple(labels)


def _sibling_pair(item: object, root: Path | None) -> PathSource | None:
    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], str):
        candidate = _path(item[0])
        return (candidate, item[1]) if candidate else None
    if isinstance(item, Mapping):
        candidate = _path(item.get("path") or item.get("file"))
        source = item.get("source")
        if candidate and isinstance(source, str):
            return candidate, source
        return None
    candidate = Path(_text(item))
    if root is not None and not candidate.is_absolute():
        candidate = root / candidate
    try:
        source = candidate.read_text(encoding="utf-8")
    except OSError:
        return None
    return _path(candidate), source


def _nearest_siblings(target_path: str, siblings: Iterable[object], root: Path | None) -> tuple[PathSource, ...]:
    target = Path(_path(target_path))
    source_items: Iterable[object] = siblings.items() if isinstance(siblings, Mapping) else siblings
    pairs = [pair for item in source_items if (pair := _sibling_pair(item, root)) is not None]
    unique = {path: source for path, source in pairs if _path(path) != _path(target)}
    def rank(pair: PathSource) -> tuple[int, int, str]:
        path = Path(pair[0])
        same_parent = 0 if path.parent == target.parent else 1
        distance = abs(len(path.parts) - len(target.parts)) + sum(a != b for a, b in zip(path.parts, target.parts))
        return same_parent, distance, pair[0]
    return tuple(sorted(unique.items(), key=rank)[:3])


@dataclass(frozen=True)
class ConventionResult:
    target_path: str
    sibling_paths: tuple[str, ...]
    conventions: tuple[str, ...]
    event: str
    advisory: str | None
    violation: bool
    challenge: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "target_path": self.target_path,
            "sibling_paths": list(self.sibling_paths),
            "conventions": list(self.conventions),
            "event": self.event,
            "advisory": self.advisory,
            "violation": self.violation,
            "challenge": self.challenge,
        }


class ConventionMirror:
    """Select at most three nearest siblings and compare a new artifact."""

    organ_id = "convention_mirror"

    def __init__(self) -> None:
        self._challenged: set[str] = set()
        self._events: list[Record] = []

    @property
    def events(self) -> tuple[Record, ...]:
        return tuple(self._events)

    def check(
        self,
        target_path: str,
        siblings: Iterable[object],
        *,
        proposed_source: str | None = None,
        proposed_name: str = "",
        root: str | Path | None = None,
        new_kind: str = "file",
        turn: int = 0,
    ) -> ConventionResult:
        root_path = Path(root) if root is not None else None
        selected = _nearest_siblings(target_path, siblings, root_path)
        profiles = [_profile(source) for _, source in selected]
        counts = Counter(label for profile in profiles for label in profile)
        majority = len(profiles) / 2
        conventions = tuple(sorted(label for label, count in counts.items() if count > majority))
        if not selected:
            return self._result(target_path, (), (), "CONVENTION_NO_SAFE_EXEMPLAR", None, False, False, turn)
        proposed = set(_profile(proposed_source or ""))
        if proposed_name:
            if "test" in new_kind.lower():
                proposed.add("test_snake_case" if proposed_name.startswith("test_") else "test_non_snake_case")
            elif "function" in new_kind.lower():
                proposed.add("functions_snake_case" if re.fullmatch(r"[a-z][a-z0-9_]*", proposed_name) else "functions_mixed_case")
        relevant_prefixes = ("test_", "functions_") if "function" in new_kind.lower() else (
            "from-imports", "import-statements", "future-annotations", "test_", "functions_", "module-docstring", "imports-before-definitions"
        )
        has_proposal = proposed_source is not None or bool(proposed_name)
        violations = sorted(label for label in conventions if label.startswith(relevant_prefixes) and has_proposal and label not in proposed)
        target_stem = Path(_path(target_path)).stem
        if "test_snake_case" in conventions and "test" in new_kind.lower() and not re.fullmatch(r"test_[a-z0-9_]+", target_stem):
            violations.append("test_snake_case")
        if proposed_source is None and not proposed_name:
            violations.append("unmeasured-proposal")
        key = _digest({"target": _path(target_path), "kind": new_kind, "siblings": [path for path, _ in selected], "violations": violations})
        challenge = bool(violations) and key not in self._challenged
        if challenge:
            self._challenged.add(key)
        event = "CONVENTION_EXEMPLAR_DELIVERED" if challenge else "CONVENTION_SPLIT" if violations else "CONVENTION_CONFORMING"
        advisory = None
        if challenge:
            examples = ", ".join(path for path, _ in selected[:3])
            labels = ", ".join(conventions[:4]) or "the nearest sibling structure"
            advisory = _clip(f"siblings do {labels} ({examples}) - mirror it")
        return self._result(target_path, tuple(path for path, _ in selected), conventions, event, advisory, bool(violations), challenge, turn)

    def _result(
        self,
        target_path: str,
        siblings: tuple[str, ...],
        conventions: tuple[str, ...],
        event: str,
        advisory: str | None,
        violation: bool,
        challenge: bool,
        turn: int,
    ) -> ConventionResult:
        result = ConventionResult(_path(target_path), siblings, conventions, event, advisory, violation, challenge)
        self._events.append({"organ_id": self.organ_id, "event": event, "turn": turn, "emitted": bool(advisory), "detail": advisory or result.target_path})
        if challenge:
            self._events.append({"organ_id": self.organ_id, "event": "CONVENTION_CHALLENGE_ONCE", "turn": turn, "emitted": bool(advisory), "detail": result.target_path})
        return result


def load_receipts(path: str | Path) -> tuple[Record, ...]:
    """Load valid JSON object rows, matching the liveness replay idiom."""

    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ()
    records: list[Record] = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            records.append(value)
    return tuple(records)


def replay_measure_receipts(records: Iterable[Record], organ: MeasureBeforeInfer | None = None) -> tuple[MeasureResult, ...]:
    """Replay action receipts in order, checking each action before measuring it."""

    current = organ or MeasureBeforeInfer()
    results: list[MeasureResult] = []
    for record in records:
        results.extend(current.check_action(record))
        current.observe_receipt(record)
    return tuple(results)


def measure_before_action(
    assertion: PlanAssertion,
    measurements: Iterable[Measurement] = (),
    *,
    turn: int = 0,
    workspace_revision: int = 0,
    organ: MeasureBeforeInfer | None = None,
) -> MeasureResult:
    """Documented B wire seam: evaluate one plan claim before its action."""

    current = organ or MeasureBeforeInfer()
    for measurement in measurements:
        current.observe_measurement(measurement)
    return current.check(assertion, turn=turn, workspace_revision=workspace_revision)


def _trigger_for(organ_id: str, events: frozenset[str]):
    def trigger(evidence: Evidence) -> TriState:
        record = evidence.record
        if not isinstance(record, Mapping):
            return TriState.UNKNOWN
        owner = _text(record.get("organ_id") or record.get("organ") or record.get("cylinder"))
        event = _text(record.get("event") or record.get("event_name")).upper()
        if owner.casefold() != organ_id.casefold():
            return TriState.NO
        return TriState.YES if event in events else TriState.UNKNOWN if not event else TriState.NO
    trigger.__name__ = f"{organ_id.replace('-', '_')}_trigger"
    return trigger


_MEASURE_EVENTS = frozenset({"MEASURE_FIRST_CHALLENGE", "MEASURE_PREANSWER_MATCH", "MEASURE_PREANSWER_CONTRADICTION", "MEASURE_PROBE_OBSERVED", "MEASURE_DEDUPED", "MEASURE_CHALLENGE_ONCE"})
_CONSUMPTION_EVENTS = frozenset({"CONSUMPTION_OBSERVED", "CONSUMPTION_FOUND", "CONSUMPTION_CLEAR", "CONSUMPTION_DEAD", "CONSUMPTION_UNKNOWN", "CONSUMPTION_DEDUPED", "CONSUMPTION_CHALLENGE_ONCE"})
_CONVENTION_EVENTS = frozenset({"CONVENTION_EXEMPLAR_DELIVERED", "CONVENTION_NO_SAFE_EXEMPLAR", "CONVENTION_CONFORMING", "CONVENTION_SPLIT", "CONVENTION_DEDUPED", "CONVENTION_CHALLENGE_ONCE"})

CENSUS_SPECS = (
    OrganCensusSpec(
        organ_id="measure_first",
        trigger=_trigger_for("measure_first", _MEASURE_EVENTS),
        fire_events=frozenset({"MEASURE_FIRST_CHALLENGE", "MEASURE_PREANSWER_CONTRADICTION", "MEASURE_PROBE_OBSERVED"}),
        suppression_events=frozenset({"MEASURE_PREANSWER_MATCH", "MEASURE_NOT_A_VALUE_ASSERTION"}),
        dedup_events=frozenset({"MEASURE_DEDUPED"}),
        challenge_once_events=frozenset({"MEASURE_CHALLENGE_ONCE"}),
    ),
    OrganCensusSpec(
        organ_id="consumption_judgment",
        trigger=_trigger_for("consumption_judgment", _CONSUMPTION_EVENTS),
        fire_events=frozenset({"CONSUMPTION_FOUND", "CONSUMPTION_CLEAR", "CONSUMPTION_DEAD", "CONSUMPTION_UNKNOWN"}),
        suppression_events=frozenset({"CONSUMPTION_NOT_DESTRUCTIVE"}),
        dedup_events=frozenset({"CONSUMPTION_DEDUPED"}),
        challenge_once_events=frozenset({"CONSUMPTION_CHALLENGE_ONCE"}),
    ),
    OrganCensusSpec(
        organ_id="convention_mirror",
        trigger=_trigger_for("convention_mirror", _CONVENTION_EVENTS),
        fire_events=frozenset({"CONVENTION_EXEMPLAR_DELIVERED", "CONVENTION_NO_SAFE_EXEMPLAR"}),
        suppression_events=frozenset({"CONVENTION_CONFORMING", "CONVENTION_SPLIT"}),
        dedup_events=frozenset({"CONVENTION_DEDUPED"}),
        challenge_once_events=frozenset({"CONVENTION_CHALLENGE_ONCE"}),
    ),
)


ConsumptionAwareJudgment = ConsumptionTracker
ConventionMirrorOrgan = ConventionMirror
render_consumption_digest = ConsumptionTracker.render
mirror_convention = ConventionMirror.check


__all__ = [
    "AssertionKind",
    "CENSUS_SPECS",
    "ConsumptionTracker",
    "ConsumptionAwareJudgment",
    "ConventionMirror",
    "ConventionMirrorOrgan",
    "ConventionResult",
    "FactKind",
    "MAX_RENDER_CHARS",
    "MeasureBeforeInfer",
    "MeasureResult",
    "Measurement",
    "ObservedFact",
    "PlanAssertion",
    "SCHEMA",
    "load_receipts",
    "measure_before_action",
    "mirror_convention",
    "render_consumption_digest",
    "replay_measure_receipts",
]
