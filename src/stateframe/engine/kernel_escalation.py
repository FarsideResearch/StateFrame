"""Deterministic escalation packets for attempts that exhaust their ceiling.

This module is intentionally outside the live runner.  It consumes sealed
agent-world attempt memories, the existing cross-run capsule renderer, and an
optional close-look dossier.  It never reads a fixed source or asks a model to
infer facts; a later decider may use the JSON sidecar to return direction only.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from detreason.attempt_capsule import render_capsule
from detreason.attempt_memory import AttemptMemoryV1, discover_tags, distill_attempt


SCHEMA = "poc.escalation_packet/1"
MAX_RENDER_CHARS = 1500
MAX_ATTEMPTS = 16
MAX_TRIED_APPROACH_CHARS = 220
MAX_PROVEN_FACTS = 8
MAX_UNKNOWN_FACTS = 6
MAX_VIOLATED_ATOMS = 6
MAX_DOSSIER_TAIL_CHARS = 420
MAX_RECEIPT_REFS = 8

JsonMapping: TypeAlias = Mapping[str, object]

_GREEN_CLASSES = frozenset({"GREEN", "PRODUCT_GREEN", "PASS", "SUCCESS", "CERTIFIED"})
_FORBIDDEN_MARKERS = frozenset(
    {
        "answer_key",
        "fixed_source",
        "fixed_src",
        "gold_source",
        "gold_src",
        "hidden_overlay",
        "patch_text",
        "solution_patch",
    }
)


class EscalationError(ValueError):
    """Raised when a packet input violates its bounded contract."""


def _mapping(value: object) -> JsonMapping:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        candidate = to_dict()
        if isinstance(candidate, Mapping):
            return candidate
    raise EscalationError("attempt must be a mapping or expose to_dict()")


def _records(value: object) -> tuple[JsonMapping, ...]:
    if isinstance(value, Mapping):
        return (value,)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _text(value: object, limit: int = 240) -> str:
    rendered = " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())
    if len(rendered) <= limit:
        return rendered
    return rendered[: max(1, limit - 1)].rstrip() + "…"


def _canonical(value: object) -> object:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _canonical(to_dict())
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_canonical(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return _text(value)


def canonical_json(value: object) -> str:
    """Serialize a sidecar payload with stable keys and no timestamp noise."""

    return json.dumps(
        _canonical(value),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise EscalationError(f"{field_name} must be positive")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise EscalationError(f"{field_name} must be positive") from exc
    if result < 1:
        raise EscalationError(f"{field_name} must be positive")
    return result


def _attempt_no(record: JsonMapping) -> int:
    key = record.get("key")
    source = key if isinstance(key, Mapping) else record
    value = source.get("attempt_no")
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _tag(record: JsonMapping) -> str:
    key = record.get("key")
    source = key if isinstance(key, Mapping) else record
    return _text(source.get("tag"), 100)


def _terminal(record: JsonMapping) -> JsonMapping:
    value = record.get("terminal")
    return value if isinstance(value, Mapping) else {}


def _receipt_refs(record: JsonMapping) -> tuple[str, ...]:
    refs: list[str] = []
    for leg in _records(record.get("failed_legs")):
        raw_refs = leg.get("evidence_refs")
        values = raw_refs if isinstance(raw_refs, Sequence) and not isinstance(raw_refs, (str, bytes)) else ()
        for raw_ref in values:
            ref = _text(raw_ref, 160)
            if ref.startswith("receipt:") and ref not in refs:
                refs.append(ref)
    audit = record.get("audit")
    if isinstance(audit, Mapping):
        raw_refs = audit.get("source_refs")
        values = raw_refs if isinstance(raw_refs, Sequence) and not isinstance(raw_refs, (str, bytes)) else ()
        for raw_ref in values:
            ref = _text(raw_ref, 160)
            if ref.startswith("receipts/") and ref not in refs:
                refs.append(ref)
    return tuple(refs[:MAX_RECEIPT_REFS])


def _failure_legs(record: JsonMapping) -> tuple[JsonMapping, ...]:
    return tuple(
        leg
        for leg in _records(record.get("failed_legs"))
        if _text(leg.get("fp"), 160) or _text(leg.get("test_selector"), 160)
    )


def _result_classes(record: JsonMapping) -> tuple[str, ...]:
    terminal = _terminal(record)
    values = [terminal.get("result"), terminal.get("result_class")]
    values.extend(leg.get("result_class") for leg in _failure_legs(record))
    return tuple(_text(value, 50).upper() for value in values if _text(value, 50))


def _is_converged(record: JsonMapping) -> bool:
    terminal = _terminal(record)
    values = _result_classes(record)
    if any(value in _GREEN_CLASSES for value in values):
        return True
    if _text(terminal.get("shape"), 60).upper() in {"CONVERGED", "SUCCESS", "PASS"}:
        return True
    for key in ("result", "outcome", "status"):
        if _text(record.get(key), 50).upper() in {"PASS", "SUCCESS", "CONVERGED", "GREEN"}:
            return True
    return False


def _explicit_autopsy(record: JsonMapping) -> str:
    for key in ("autopsy_class", "failure_class", "autopsy_shape", "autopsy"):
        value = record.get(key)
        if isinstance(value, Mapping):
            value = value.get("primary", value.get("class", value.get("label")))
        label = _text(value, 80).upper()
        if label:
            return label
    terminal = _terminal(record)
    for key in ("autopsy_class", "failure_class", "autopsy_shape"):
        value = terminal.get(key)
        if isinstance(value, Mapping):
            value = value.get("primary", value.get("class", value.get("label")))
        label = _text(value, 80).upper()
        if label:
            return label
    return ""


def classify_autopsy(record: object) -> str | None:
    """Classify a sealed attempt from its trajectory without source inference."""

    payload = _mapping(record)
    explicit = _explicit_autopsy(payload)
    if explicit:
        return explicit
    terminal = _terminal(payload)
    shape = _text(terminal.get("shape"), 80).upper()
    legs = _failure_legs(payload)
    edits = _records(payload.get("edits"))
    if shape in {"PARTIAL", "EXCEPTION", "HARNESS_SUSPECT"}:
        return shape
    if legs and len(edits) > 1:
        return "THRASH"
    if legs:
        return "UNRESOLVED_RED"
    if shape in {"BUDGET_EXHAUSTED", "FAIL_FAST_NO_ACTION", "EXPLICIT_DONE"}:
        return shape
    return None


def _safe_claim(value: object, limit: int = 240) -> str:
    claim = _text(value, limit)
    lowered = claim.lower()
    if any(marker in lowered for marker in _FORBIDDEN_MARKERS) or claim.startswith("diff "):
        return "redacted unsupported model detail"
    return claim


def _approach(record: JsonMapping) -> str:
    theories = [_safe_claim(item.get("claim"), 120) for item in _records(record.get("theories"))]
    theories = [claim for claim in theories if claim and claim != "redacted unsupported model detail"]
    if theories:
        return _safe_claim(" / ".join(sorted(dict.fromkeys(theories))[:3]), MAX_TRIED_APPROACH_CHARS)
    edits = sorted(
        {
            _safe_claim(item.get("operation") or item.get("path"), 100)
            for item in _records(record.get("edits"))
            if item.get("operation") or item.get("path")
        }
    )
    if edits:
        return _safe_claim("edits: " + ", ".join(edits[:3]), MAX_TRIED_APPROACH_CHARS)
    capsule = render_capsule((record,), max_tokens=42, max_chars=MAX_TRIED_APPROACH_CHARS)
    for line in capsule.splitlines():
        if line.startswith("attempt ") and " tried " in line:
            return _safe_claim(line.rstrip("."), MAX_TRIED_APPROACH_CHARS)
    return "no recorded product approach"


def _failure_summary(record: JsonMapping) -> tuple[str, str, tuple[str, ...]]:
    legs = sorted(
        _failure_legs(record),
        key=lambda item: (_text(item.get("turn"), 20), _text(item.get("fp"), 160)),
    )
    if not legs:
        return "no failing test leg recorded", "", _receipt_refs(record)
    leg = legs[0]
    selector = _safe_claim(leg.get("test_selector"), 180) or "test"
    fingerprint = _text(leg.get("fp"), 160) or _digest({"selector": selector})
    anchor = _safe_claim(leg.get("anchor"), 180)
    detail = f"{selector} RED [{fingerprint}]"
    if anchor:
        detail += f" {anchor}"
    raw_refs = leg.get("evidence_refs")
    leg_refs = (
        tuple(_text(item, 160) for item in raw_refs)
        if isinstance(raw_refs, Sequence) and not isinstance(raw_refs, (str, bytes))
        else ()
    )
    refs = tuple(ref for ref in (*leg_refs, *_receipt_refs(record)) if ref.startswith("receipt:"))
    return detail, fingerprint, tuple(dict.fromkeys(refs))[:MAX_RECEIPT_REFS]


@dataclass(frozen=True, slots=True)
class AttemptSummary:
    """One bounded description of an exhausted attempt."""

    attempt_no: int
    tag: str
    approach: str
    failure: str
    failing_fingerprint: str
    autopsy_class: str
    receipt_refs: tuple[str, ...] = ()
    capsule_summary: str = ""

    def __post_init__(self) -> None:
        if self.attempt_no < 1:
            raise EscalationError("attempt_no must be positive")
        if not self.autopsy_class.strip():
            raise EscalationError("autopsy_class is required")
        object.__setattr__(self, "tag", _text(self.tag, 100) or f"attempt-{self.attempt_no}")
        object.__setattr__(self, "approach", _safe_claim(self.approach, MAX_TRIED_APPROACH_CHARS))
        object.__setattr__(self, "failure", _safe_claim(self.failure, 280))
        object.__setattr__(self, "failing_fingerprint", _text(self.failing_fingerprint, 160))
        object.__setattr__(self, "autopsy_class", _text(self.autopsy_class, 80).upper())
        object.__setattr__(self, "receipt_refs", tuple(dict.fromkeys(_text(ref, 160) for ref in self.receipt_refs if _text(ref, 160))))
        object.__setattr__(self, "capsule_summary", _safe_claim(self.capsule_summary, MAX_TRIED_APPROACH_CHARS))

    @classmethod
    def from_record(cls, record: object) -> "AttemptSummary":
        payload = _mapping(record)
        number = _attempt_no(payload)
        if number < 1:
            raise EscalationError("each attempt needs a positive attempt_no")
        failure, fingerprint, refs = _failure_summary(payload)
        autopsy = classify_autopsy(payload)
        if not autopsy:
            raise EscalationError(f"attempt {number} has no autopsy classification")
        capsule = render_capsule((payload,), max_tokens=48, max_chars=MAX_TRIED_APPROACH_CHARS)
        return cls(
            attempt_no=number,
            tag=_tag(payload),
            approach=_approach(payload),
            failure=failure,
            failing_fingerprint=fingerprint,
            autopsy_class=autopsy,
            receipt_refs=refs,
            capsule_summary=capsule,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "attempt_no": self.attempt_no,
            "tag": self.tag,
            "approach": self.approach,
            "failure": self.failure,
            "failing_fingerprint": self.failing_fingerprint,
            "autopsy_class": self.autopsy_class,
            "receipt_refs": list(self.receipt_refs),
            "capsule_summary": self.capsule_summary,
        }


@dataclass(frozen=True, slots=True)
class ProvenFact:
    claim: str
    receipt_refs: tuple[str, ...]
    fact_id: str = ""

    def __post_init__(self) -> None:
        claim = _safe_claim(self.claim, 280)
        refs = tuple(dict.fromkeys(_text(ref, 160) for ref in self.receipt_refs if _text(ref, 160).startswith("receipt:")))
        if not claim or not refs:
            raise EscalationError("proven facts require a claim and receipt provenance")
        object.__setattr__(self, "claim", claim)
        object.__setattr__(self, "receipt_refs", refs[:MAX_RECEIPT_REFS])
        object.__setattr__(self, "fact_id", self.fact_id or _digest({"claim": claim, "refs": refs}))

    def to_dict(self) -> dict[str, object]:
        return {"id": self.fact_id, "claim": self.claim, "receipt_refs": list(self.receipt_refs)}


@dataclass(frozen=True, slots=True)
class UnknownFact:
    question: str
    receipt_refs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {"question": _safe_claim(self.question, 280), "receipt_refs": list(self.receipt_refs)}


def _dossier_text(dossier: object | None) -> str:
    if dossier is None:
        return ""
    if isinstance(dossier, str):
        rendered = dossier
    else:
        render = getattr(dossier, "render", None)
        rendered = render() if callable(render) else _text(dossier, MAX_DOSSIER_TAIL_CHARS)
    return _safe_claim(_text(rendered, 100000)[-MAX_DOSSIER_TAIL_CHARS:], MAX_DOSSIER_TAIL_CHARS)


def _all_evidence_text(records: Sequence[JsonMapping], summaries: Sequence[AttemptSummary]) -> str:
    parts: list[str] = []
    for record, summary in zip(records, summaries):
        parts.append(summary.failure)
        for leg in _failure_legs(record):
            parts.extend((_text(leg.get("test_selector"), 180), _text(leg.get("anchor"), 180)))
        for atom in _records(record.get("atoms")):
            parts.extend((_text(atom.get("expected"), 180), _text(atom.get("observed"), 180)))
    return " ".join(parts).lower()


def _decider_question(records: Sequence[JsonMapping], summaries: Sequence[AttemptSummary]) -> str:
    evidence = _all_evidence_text(records, summaries)
    selectors = sorted(
        {
            _safe_claim(leg.get("test_selector"), 140)
            for record in records
            for leg in _failure_legs(record)
            if _safe_claim(leg.get("test_selector"), 140)
        }
    )
    selector = selectors[0] if selectors else "the repeated RED receipt"
    if ("ordering" in evidence or "order" in evidence) and (
        "cap" in evidence or "limit" in evidence or "window" in evidence
    ):
        return (
            "Does the source-level query apply descending/recent ordering before the 100-row cap, "
            f"which would explain repeated RED in {selector}?"
        )
    fingerprint = next((summary.failing_fingerprint for summary in summaries if summary.failing_fingerprint), "unknown")
    return f"Which source-level boundary explains repeated RED for {selector} (fingerprint {fingerprint})?"


def _proven_facts(records: Sequence[JsonMapping], summaries: Sequence[AttemptSummary]) -> tuple[ProvenFact, ...]:
    facts: list[ProvenFact] = []
    for record, summary in zip(records, summaries):
        refs = summary.receipt_refs
        if refs:
            facts.append(ProvenFact(f"{summary.tag} recorded {summary.failure}.", refs))
        for atom in _records(record.get("atoms")):
            if _text(atom.get("state"), 40).upper() != "SUPPORTED":
                continue
            atom_refs = tuple(
                _text(ref, 160)
                for ref in atom.get("evidence_refs", ())
                if isinstance(atom.get("evidence_refs"), Sequence) and _text(ref, 160).startswith("receipt:")
            )
            if atom_refs:
                expected = _safe_claim(atom.get("expected"), 120)
                observed = _safe_claim(atom.get("observed"), 120)
                facts.append(ProvenFact(f"{_text(atom.get('kind'), 80)}: expected {expected}; observed {observed}.", atom_refs))
    deduped: dict[str, ProvenFact] = {}
    for fact in facts:
        deduped.setdefault(fact.claim.lower(), fact)
    return tuple(deduped.values())[:MAX_PROVEN_FACTS]


def _unknowns(records: Sequence[JsonMapping], summaries: Sequence[AttemptSummary], question: str) -> tuple[UnknownFact, ...]:
    unknowns = [UnknownFact(question, tuple(ref for summary in summaries for ref in summary.receipt_refs))]
    for record, summary in zip(records, summaries):
        audit = record.get("audit")
        gaps = audit.get("gaps") if isinstance(audit, Mapping) else ()
        if isinstance(gaps, Sequence) and not isinstance(gaps, (str, bytes)):
            for gap in gaps:
                rendered = _safe_claim(gap, 180)
                if rendered:
                    unknowns.append(UnknownFact(f"Receipt gap remains unresolved: {rendered}.", summary.receipt_refs))
    return tuple(unknowns[:MAX_UNKNOWN_FACTS])


def _violated(records: Sequence[JsonMapping]) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for record in records:
        for atom in _records(record.get("atoms")):
            if _text(atom.get("state"), 40).upper() != "CONTRADICTED":
                continue
            rows.append(
                {
                    "kind": _safe_claim(atom.get("kind"), 80),
                    "expected": _safe_claim(atom.get("expected"), 180),
                    "observed": _safe_claim(atom.get("observed"), 180),
                    "evidence_refs": [
                        _text(ref, 160)
                        for ref in atom.get("evidence_refs", ())
                        if isinstance(atom.get("evidence_refs"), Sequence) and _text(ref, 160).startswith("receipt:")
                    ],
                }
            )
    return tuple(rows[:MAX_VIOLATED_ATOMS])


@dataclass(frozen=True, slots=True)
class EscalationPacket:
    """One content-addressed handoff packet; no packet means no escalation."""

    task_id: str
    attempt_ceiling: int
    attempts: tuple[AttemptSummary, ...]
    proven: tuple[ProvenFact, ...]
    unknown: tuple[UnknownFact, ...]
    dossier_tail: str
    decider_question: str
    terminal_reason: str = "ATTEMPT_CEILING"
    violated: tuple[dict[str, object], ...] = ()
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        _positive_int(self.attempt_ceiling, "attempt_ceiling")
        if not self.attempts:
            raise EscalationError("packet needs attempts")
        if len(self.attempts) > MAX_ATTEMPTS:
            raise EscalationError("packet exceeds attempt bound")
        if not self.decider_question.strip().endswith("?"):
            raise EscalationError("decider_question must be a question")
        object.__setattr__(self, "task_id", _text(self.task_id, 180) or "task")
        object.__setattr__(self, "dossier_tail", _text(self.dossier_tail, MAX_DOSSIER_TAIL_CHARS))
        object.__setattr__(self, "decider_question", _safe_claim(self.decider_question, 420))

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "task_id": self.task_id,
            "attempt_ceiling": self.attempt_ceiling,
            "terminal_reason": self.terminal_reason,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "proven": [fact.to_dict() for fact in self.proven],
            "unknown": [fact.to_dict() for fact in self.unknown],
            "violated": list(self.violated),
            "dossier_tail": self.dossier_tail,
            "decider_question": self.decider_question,
        }

    @property
    def packet_id(self) -> str:
        return _digest(self._payload())

    def to_dict(self) -> dict[str, object]:
        return {"packet_id": self.packet_id, **self._payload()}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "EscalationPacket":
        """Rehydrate and verify a sidecar without source or model access."""

        if value.get("schema") != SCHEMA:
            raise EscalationError(f"schema must be {SCHEMA}")
        attempts = tuple(
            AttemptSummary(
                attempt_no=int(item["attempt_no"]),
                tag=str(item.get("tag", "")),
                approach=str(item.get("approach", "")),
                failure=str(item.get("failure", "")),
                failing_fingerprint=str(item.get("failing_fingerprint", "")),
                autopsy_class=str(item["autopsy_class"]),
                receipt_refs=tuple(str(ref) for ref in item.get("receipt_refs", ())),
                capsule_summary=str(item.get("capsule_summary", "")),
            )
            for item in _records(value.get("attempts", ()))
        )
        proven = tuple(
            ProvenFact(
                claim=str(item["claim"]),
                receipt_refs=tuple(str(ref) for ref in item.get("receipt_refs", ())),
                fact_id=str(item.get("id", "")),
            )
            for item in _records(value.get("proven", ()))
        )
        unknown = tuple(
            UnknownFact(
                question=str(item["question"]),
                receipt_refs=tuple(str(ref) for ref in item.get("receipt_refs", ())),
            )
            for item in _records(value.get("unknown", ()))
        )
        packet = cls(
            task_id=str(value.get("task_id", "task")),
            attempt_ceiling=int(value["attempt_ceiling"]),
            attempts=attempts,
            proven=proven,
            unknown=unknown,
            dossier_tail=str(value.get("dossier_tail", "")),
            decider_question=str(value["decider_question"]),
            terminal_reason=str(value.get("terminal_reason", "ATTEMPT_CEILING")),
            violated=_records(value.get("violated", ())),
        )
        supplied = value.get("packet_id")
        if supplied not in (None, "") and str(supplied) != packet.packet_id:
            raise EscalationError("packet_id does not match its content")
        return packet

    def sidecar_json(self) -> str:
        """Return the machine-readable future-decider sidecar."""

        return canonical_json(self.to_dict())

    serialize = sidecar_json
    json_sidecar = sidecar_json

    def render(self, *, max_chars: int = MAX_RENDER_CHARS) -> str:
        """Render a deterministic human handoff bounded by ``max_chars``."""

        limit = _positive_int(max_chars, "max_chars")
        lines = [
            f"ESCALATION {self.task_id} | ceiling {self.attempt_ceiling}/{len(self.attempts)} | {self.terminal_reason}",
            "TRIED:",
        ]
        lines.extend(
            f"A{attempt.attempt_no} {attempt.autopsy_class}: {attempt.approach}; fail={attempt.failure}"
            for attempt in self.attempts
        )
        lines.append("PROVEN:")
        lines.extend(f"- {fact.claim} ({', '.join(fact.receipt_refs[:2])})" for fact in self.proven)
        lines.append("UNKNOWN:")
        lines.extend(f"- {fact.question}" for fact in self.unknown)
        if self.violated:
            lines.append("VIOLATED:")
            lines.extend(f"- {row.get('kind')}: {row.get('expected')} vs {row.get('observed')}" for row in self.violated)
        if self.dossier_tail:
            lines.append("DOSSIER TAIL:")
            lines.append(self.dossier_tail)
        lines.append(f"DECIDER QUESTION: {self.decider_question}")
        rendered = "\n".join(lines)
        if len(rendered) <= limit:
            return rendered
        suffix = f"\nDECIDER QUESTION: {self.decider_question}"
        if len(suffix) >= limit:
            return suffix[-limit:]
        return rendered[: limit - len(suffix)].rstrip() + suffix


def _normalise_records(attempts: Iterable[object]) -> tuple[JsonMapping, ...]:
    records = tuple(_mapping(attempt) for attempt in attempts)
    ordered = sorted(records, key=lambda record: (_attempt_no(record), _tag(record)))
    return tuple(record for record in ordered if _attempt_no(record) > 0)[:MAX_ATTEMPTS]


def build_escalation_packet(
    attempts: Iterable[object],
    *,
    attempt_ceiling: int,
    task_id: str = "task",
    dossier: object | None = None,
    converged: bool | None = None,
) -> EscalationPacket | None:
    """Compile one packet, or ``None`` when the task should not escalate."""

    ceiling = _positive_int(attempt_ceiling, "attempt_ceiling")
    records = _normalise_records(attempts)
    if converged is True or any(_is_converged(record) for record in records):
        return None
    if len(records) < ceiling:
        return None
    selected = records[:ceiling]
    summaries = tuple(AttemptSummary.from_record(record) for record in selected)
    question = _decider_question(selected, summaries)
    return EscalationPacket(
        task_id=task_id,
        attempt_ceiling=ceiling,
        attempts=summaries,
        proven=_proven_facts(selected, summaries),
        unknown=_unknowns(selected, summaries, question),
        dossier_tail=_dossier_text(dossier),
        decider_question=question,
        violated=_violated(selected),
    )


compile_escalation_packet = build_escalation_packet


def load_collect_attempts(
    artifact_root: str | Path,
    *,
    task_fn: str = "_collect_metrics",
) -> tuple[AttemptMemoryV1, ...]:
    """Load the real ``receipts_*collect*``/transcript history as capsules."""

    tags = sorted(tag for tag in discover_tags(artifact_root, task_fn) if "collect" in tag.lower())
    records: list[AttemptMemoryV1] = []
    for number, tag in enumerate(tags, start=1):
        records.append(
            distill_attempt(
                artifact_root,
                task_fn,
                tag=tag,
                attempt_no=number,
                campaign="",
            )
        )
    return tuple(records)


__all__ = [
    "AttemptSummary",
    "EscalationError",
    "EscalationPacket",
    "MAX_RENDER_CHARS",
    "ProvenFact",
    "SCHEMA",
    "UnknownFact",
    "build_escalation_packet",
    "canonical_json",
    "classify_autopsy",
    "compile_escalation_packet",
    "load_collect_attempts",
]
