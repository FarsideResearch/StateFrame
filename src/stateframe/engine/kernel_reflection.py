"""Bounded, event-driven micro-reflection and prediction capture.

This organ emits typed concerns for the kernel to route.  It never edits a
plan, chooses a repair, or returns prose steering.  Reflection is invoked only
after a deterministic trigger is observed, and every external-facing result
has a bounded shape.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import OrderedDict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

from kernel_liveness import Evidence, OrganCensusSpec, TriState


SCHEMA = "poc.kernel_reflection/1"
MAX_REQUEST_TOKENS = 60
MAX_REPLY_CHARS = 4096
MAX_SIGNALS = 3
MAX_TARGET_REF_CHARS = 120
MAX_RECEIPTS = 256
MAX_TRACKED_FINGERPRINTS = 256

Record: TypeAlias = Mapping[str, object]
Reply: TypeAlias = str | Mapping[str, object] | Sequence[object]


class ReflectionTrigger(StrEnum):
    """Deterministic boundaries at which micro-reflection may run."""

    ABOUT_TO_CLAIM_DONE = "about_to_claim_done"
    EDIT_AFTER_RED_TEST = "edit_after_red_test"
    EXPECTATION_CONTRADICTION = "expectation_contradiction"
    CONFIDENCE_COLLAPSE = "confidence_collapse"
    HANDOFF_UNIT_BOUNDARY = "handoff_unit_boundary"


class ConcernKind(StrEnum):
    """Closed concern vocabulary; no free-form steering channel exists."""

    CONTRACT = "contract"
    ASSUMPTION = "assumption"
    OBSERVATION = "observation"
    ROOT_CAUSE = "root_cause"
    INVALIDATED_WORK = "invalidated_work"
    CONSEQUENCE = "consequence"
    NONE = "none"


class Severity(StrEnum):
    """Bounded concern severity."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ParseStatus(StrEnum):
    """Receipt status for one non-retried micro-reflection call."""

    PARSED = "parsed"
    MALFORMED = "malformed"
    NO_CONCERN = "no_concern"
    SUPPRESSED = "suppressed"


class PredictionMismatch(StrEnum):
    """Deterministic prediction mismatch reasons."""

    EXPECTED_GREEN_FAILED = "expected_green_failed"
    EXPECTED_SUBSTRING_ABSENT = "expected_substring_absent"


class RoutingOrgan(StrEnum):
    """Existing kernel organ destinations; delivery belongs to the harness."""

    NIT = "nit"
    CLOSELOOK = "closelook"
    COMPOSED = "composed"
    GOVERNOR = "governor"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class ReflectionSignal:
    """One typed concern, intentionally without advice or steering prose."""

    concern_kind: ConcernKind | str
    target_ref: str = ""
    severity: Severity | str = Severity.NONE

    def __post_init__(self) -> None:
        try:
            concern = ConcernKind(_normalize_enum(self.concern_kind))
            severity = Severity(_normalize_enum(self.severity))
        except ValueError as exc:
            raise ValueError("reflection signal has an unknown typed field") from exc
        target = _clip_ref(self.target_ref)
        if concern is not ConcernKind.NONE and not target:
            raise ValueError("non-empty reflection concern requires target_ref")
        if concern is ConcernKind.NONE:
            severity = Severity.NONE
            target = ""
        object.__setattr__(self, "concern_kind", concern)
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "target_ref", target)

    def to_dict(self) -> dict[str, str]:
        return {
            "concern_kind": self.concern_kind.value,
            "target_ref": self.target_ref,
            "severity": self.severity.value,
        }


@dataclass(frozen=True, slots=True)
class PredictionExpectation:
    """Expected observation captured before an action is dispatched."""

    action_kind: str
    fingerprint: str
    expected: str

    def __post_init__(self) -> None:
        action = _clip_text(self.action_kind, 64).lower().replace(" ", "_")
        fingerprint = _clip_text(self.fingerprint, 128)
        expected = _clip_text(self.expected, 240)
        if not action or not fingerprint or not expected:
            raise ValueError("prediction expectation fields are required")
        object.__setattr__(self, "action_kind", action)
        object.__setattr__(self, "fingerprint", fingerprint)
        object.__setattr__(self, "expected", expected)


@dataclass(frozen=True, slots=True)
class PredictionMiss:
    """A deterministic mismatch between a recorded expectation and observation."""

    action_kind: str
    fingerprint: str
    expected: str
    observed: str
    mismatch: PredictionMismatch | str

    def __post_init__(self) -> None:
        try:
            mismatch = PredictionMismatch(_normalize_enum(self.mismatch))
        except ValueError as exc:
            raise ValueError("prediction miss has an unknown mismatch") from exc
        object.__setattr__(self, "action_kind", _clip_text(self.action_kind, 64))
        object.__setattr__(self, "fingerprint", _clip_text(self.fingerprint, 128))
        object.__setattr__(self, "expected", _clip_text(self.expected, 240))
        object.__setattr__(self, "observed", _clip_text(self.observed, 240))
        object.__setattr__(self, "mismatch", mismatch)

    def to_dict(self) -> dict[str, str]:
        return {
            "schema": SCHEMA,
            "kind": "reflection_gate",
            "organ": "reflection_gate",
            "event": "PREDICTION_MISS",
            "action_kind": self.action_kind,
            "fingerprint": self.fingerprint,
            "expected": self.expected,
            "observed": self.observed,
            "mismatch": self.mismatch.value,
        }


@dataclass(frozen=True, slots=True)
class ReflectionReceipt:
    """Compact receipt proving parse, suppression, or malformed handling."""

    event: str
    status: ParseStatus | str
    trigger_kinds: tuple[ReflectionTrigger | str, ...]
    target_ref: str
    signal_count: int
    request_token_count: int
    suppressed: bool = False
    fingerprint: str = ""

    def __post_init__(self) -> None:
        status = ParseStatus(_normalize_enum(self.status))
        triggers = tuple(
            ReflectionTrigger(_normalize_enum(trigger)) for trigger in self.trigger_kinds
        )
        if self.signal_count < 0 or self.signal_count > MAX_SIGNALS:
            raise ValueError("reflection signal count is out of bounds")
        if self.request_token_count > MAX_REQUEST_TOKENS:
            raise ValueError("reflection request exceeds token bound")
        object.__setattr__(self, "event", _clip_text(self.event, 48))
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "trigger_kinds", triggers)
        object.__setattr__(self, "target_ref", _clip_ref(self.target_ref))
        object.__setattr__(self, "fingerprint", _clip_text(self.fingerprint, 128))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "kind": "reflection_gate",
            "organ": "reflection_gate",
            "event": self.event,
            "receipt_id": "reflection:" + _digest(
                {
                    "event": self.event,
                    "status": self.status.value,
                    "triggers": [trigger.value for trigger in self.trigger_kinds],
                    "target_ref": self.target_ref,
                    "signal_count": self.signal_count,
                    "request_token_count": self.request_token_count,
                    "suppressed": self.suppressed,
                    "fingerprint": self.fingerprint,
                }
            )[:24],
            "status": self.status.value,
            "trigger_kinds": tuple(trigger.value for trigger in self.trigger_kinds),
            "target_ref": self.target_ref,
            "signal_count": self.signal_count,
            "request_token_count": self.request_token_count,
            "suppressed": self.suppressed,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True, slots=True)
class ReflectionResult:
    """Bounded output from one explicitly triggered reflection attempt."""

    request: str
    signals: tuple[ReflectionSignal, ...]
    receipt: ReflectionReceipt


@dataclass(frozen=True, slots=True)
class ReflectionParseResult:
    """Parser output without retaining the original model reply."""

    signals: tuple[ReflectionSignal, ...]
    status: ParseStatus


def _clip_text(value: object, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _clip_ref(value: object) -> str:
    return _clip_text(value, MAX_TARGET_REF_CHARS)


def _normalize_enum(value: object) -> str:
    return str(value.value if isinstance(value, StrEnum) else value).strip().lower().replace("-", "_")


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _normal_kind(value: object) -> str:
    return _clip_text(value, 64).lower().replace("-", "_").replace(" ", "_")


def _event_kind(event: Record) -> str:
    return _normal_kind(event.get("kind", event.get("action_kind", event.get("type", ""))))


def _event_fingerprint(event: Record) -> str:
    supplied = _clip_text(event.get("fingerprint", ""), 128)
    if supplied:
        return supplied
    material = {
        "kind": _event_kind(event),
        "intent": _clip_text(event.get("intent", ""), 180),
        "path": _clip_text(event.get("path", event.get("file", "")), 120),
        "result_class": _clip_text(event.get("result_class", ""), 48),
    }
    return "sha256:" + _digest(material)


def _result_class(event: Record) -> str:
    return _clip_text(event.get("result_class", event.get("result", "")), 48).upper()


def _is_red_test(event: Record) -> bool:
    return _event_kind(event) in {"test", "pytest", "test_run"} and _result_class(event) in {
        "RED",
        "FAILED",
        "FAILURE",
    }


def _is_failed_attempt(event: Record) -> bool:
    return _result_class(event) in {
        "RED",
        "FAILED",
        "FAILURE",
        "ERROR",
        "REJECTED",
        "BLOCKED",
        "TIMEOUT",
        "NO_OP",
    }


def _event_matches(event: Record, *names: str) -> bool:
    values = {
        _normal_kind(event.get("event", "")),
        _normal_kind(event.get("kind", "")),
        _normal_kind(event.get("action_kind", "")),
    }
    return bool(values.intersection(names))


def compose_self_check(
    trigger: ReflectionTrigger | str,
    *,
    target_ref: str = "",
) -> str:
    """Compose the fixed six-question micro request and enforce its token cap."""

    trigger_value = ReflectionTrigger(_normalize_enum(trigger)).value
    target = _clip_ref(target_ref)
    request = (
        "Return JSON only with signals [{concern_kind,target_ref,severity}]. "
        "Check: contract satisfied? unmeasured assumption? ignored observation? "
        "simpler root cause? invalidated earlier work? consequence exceeds certainty? "
        f"trigger={trigger_value} target_ref={target or 'unknown'}."
    )
    if len(request.split()) > MAX_REQUEST_TOKENS:  # pragma: no cover - defensive invariant
        raise AssertionError("micro-reflection request exceeded token bound")
    return request


def _decode_scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _parse_yaml_reply(reply: str) -> object:
    """Parse only the tiny YAML subset needed for compact signal replies."""

    rows: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    saw_signals = False
    for raw_line in reply.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "signals:":
            saw_signals = True
            continue
        if line == "signals: []":
            saw_signals = True
            continue
        if line.startswith("-"):
            if current is not None:
                rows.append(current)
            current = {}
            line = line[1:].strip()
            if line:
                if ":" not in line:
                    raise ValueError("malformed yaml signal")
                key, value = line.split(":", 1)
                current[key.strip()] = _decode_scalar(value)
            continue
        if ":" not in line or current is None:
            raise ValueError("malformed yaml reply")
        key, value = line.split(":", 1)
        if key.strip() in current:
            raise ValueError("duplicate yaml signal field")
        current[key.strip()] = _decode_scalar(value)
    if current is not None:
        rows.append(current)
    if not saw_signals and not rows:
        raise ValueError("yaml reply must contain signals")
    return {"signals": rows}


def _reply_object(reply: Reply) -> object:
    if isinstance(reply, str):
        if len(reply) > MAX_REPLY_CHARS:
            raise ValueError("reflection reply exceeds bound")
        stripped = reply.strip()
        if not stripped:
            raise ValueError("reflection reply is empty")
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return _parse_yaml_reply(stripped)
    return reply


def _signal_rows(value: object) -> list[object]:
    if isinstance(value, Mapping):
        if set(value) != {"signals"}:
            raise ValueError("reflection reply has unknown top-level fields")
        rows = value.get("signals")
    else:
        rows = value
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise ValueError("signals must be an array")
    return list(rows)


def parse_reflection_reply(reply: Reply) -> ReflectionParseResult:
    """Parse compact JSON/tiny-YAML into typed signals; malformed means one NONE."""

    try:
        rows = _signal_rows(_reply_object(reply))
        if len(rows) > MAX_SIGNALS:
            raise ValueError("too many reflection signals")
        signals: list[ReflectionSignal] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("signal must be an object")
            if set(row) - {"concern_kind", "target_ref", "severity"}:
                raise ValueError("signal has unknown fields")
            signals.append(
                ReflectionSignal(
                    concern_kind=row.get("concern_kind", "none"),
                    target_ref=str(row.get("target_ref", "")),
                    severity=row.get("severity", "none"),
                )
            )
        status = ParseStatus.NO_CONCERN if not signals else ParseStatus.PARSED
        return ReflectionParseResult(tuple(signals), status)
    except (TypeError, ValueError, json.JSONDecodeError):
        return ReflectionParseResult(
            (ReflectionSignal(ConcernKind.NONE),),
            ParseStatus.MALFORMED,
        )


_ROUTING: dict[ConcernKind, RoutingOrgan] = {
    ConcernKind.CONTRACT: RoutingOrgan.NIT,
    ConcernKind.ASSUMPTION: RoutingOrgan.CLOSELOOK,
    ConcernKind.OBSERVATION: RoutingOrgan.COMPOSED,
    ConcernKind.ROOT_CAUSE: RoutingOrgan.CLOSELOOK,
    ConcernKind.INVALIDATED_WORK: RoutingOrgan.COMPOSED,
    ConcernKind.CONSEQUENCE: RoutingOrgan.GOVERNOR,
    ConcernKind.NONE: RoutingOrgan.NONE,
}


def route_concern(concern_kind: ConcernKind | str) -> RoutingOrgan:
    """Return routing only; the harness owns delivery and state transitions."""

    try:
        kind = ConcernKind(_normalize_enum(concern_kind))
    except ValueError:
        return RoutingOrgan.NONE
    return _ROUTING[kind]


def _observed_exit_code(observed: str) -> int | None:
    try:
        value = json.loads(observed)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, Mapping):
        for key in ("exit_code", "returncode", "exit", "code"):
            candidate = value.get(key)
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                return candidate
    match = re.search(
        r"(?:exit(?:[\s_]?code)?|returncode|code)\s*[:=]\s*(-?\d+)",
        observed,
        re.I,
    )
    return int(match.group(1)) if match else None


def _is_green_expectation(expected: str) -> bool:
    normalized = expected.lower().replace("_", " ")
    return bool(
        re.search(r"\b(?:expected\s+)?green\b", normalized)
        or re.search(r"\b(?:exit|returncode)\s*[:=]\s*0\b", normalized)
        or normalized.strip() in {"success", "successful"}
    )


class ReflectionGate:
    """Bounded trigger detector, prediction ledger, and micro-reflection parser."""

    def __init__(self) -> None:
        self._red_test_pending = False
        self._failed_attempts: OrderedDict[str, int] = OrderedDict()
        self._expectations: dict[str, PredictionExpectation] = {}
        self._pending_expectation_fingerprints: set[str] = set()
        self._seen_reflections: set[str] = set()
        self._seen_order: deque[str] = deque(maxlen=MAX_RECEIPTS)
        self._receipts: deque[dict[str, object]] = deque(maxlen=MAX_RECEIPTS)
        self._trigger_count = 0

    @property
    def receipts(self) -> tuple[Record, ...]:
        return tuple(dict(receipt) for receipt in self._receipts)

    @property
    def pending_triggers(self) -> tuple[ReflectionTrigger, ...]:
        return (
            (ReflectionTrigger.EXPECTATION_CONTRADICTION,)
            if self._pending_expectation_fingerprints
            else ()
        )

    @property
    def trigger_count(self) -> int:
        return self._trigger_count

    def _append_receipt(self, receipt: Mapping[str, object]) -> None:
        self._receipts.append(dict(receipt))

    def inspect(self, event: Record) -> tuple[ReflectionTrigger, ...]:
        """Inspect one observed event and return triggers without invoking reflection."""

        triggers: list[ReflectionTrigger] = []
        fingerprint = _event_fingerprint(event)
        kind = _event_kind(event)
        if fingerprint in self._pending_expectation_fingerprints:
            triggers.append(ReflectionTrigger.EXPECTATION_CONTRADICTION)
            self._pending_expectation_fingerprints.remove(fingerprint)
        if _event_matches(event, "about_to_claim_done", "claim_done", "done_gate"):
            triggers.append(ReflectionTrigger.ABOUT_TO_CLAIM_DONE)
        if kind in {"handoff", "unit_boundary", "unit_planner", "unit_planner_gate"} or _event_matches(
            event, "handoff_unit_boundary", "unit_planner_gate"
        ):
            triggers.append(ReflectionTrigger.HANDOFF_UNIT_BOUNDARY)
        if kind == "edit" and self._red_test_pending:
            triggers.append(ReflectionTrigger.EDIT_AFTER_RED_TEST)
            self._red_test_pending = False
        if _is_red_test(event):
            self._red_test_pending = True
        elif kind in {"test", "pytest", "test_run"} and _result_class(event) in {
            "GREEN",
            "PASSED",
            "OK",
        }:
            self._red_test_pending = False
        if _is_failed_attempt(event):
            if fingerprint not in self._failed_attempts and len(self._failed_attempts) >= MAX_TRACKED_FINGERPRINTS:
                self._failed_attempts.popitem(last=False)
            count = self._failed_attempts.get(fingerprint, 0) + 1
            self._failed_attempts[fingerprint] = count
            self._failed_attempts.move_to_end(fingerprint)
            if count >= 3:
                triggers.append(ReflectionTrigger.CONFIDENCE_COLLAPSE)
        unique = tuple(dict.fromkeys(triggers))
        if unique:
            self._trigger_count += len(unique)
            self._append_receipt(
                {
                    "schema": SCHEMA,
                    "kind": "reflection_gate",
                    "organ": "reflection_gate",
                    "event": "REFLECTION_TRIGGERED",
                    "trigger_kinds": tuple(trigger.value for trigger in unique),
                    "fingerprint": fingerprint,
                }
            )
        return unique

    def record_expectation(
        self,
        action_kind: str,
        fingerprint: str,
        expected: str,
    ) -> PredictionExpectation:
        """Capture expected observation before tool/action dispatch."""

        expectation = PredictionExpectation(action_kind, fingerprint, expected)
        if len(self._expectations) >= MAX_TRACKED_FINGERPRINTS and expectation.fingerprint not in self._expectations:
            oldest = next(iter(self._expectations))
            del self._expectations[oldest]
        self._expectations[expectation.fingerprint] = expectation
        return expectation

    def compare_outcome(self, fingerprint: str, observed: str) -> PredictionMiss | None:
        """Compare one outcome deterministically; no model call or retry occurs."""

        key = _clip_text(fingerprint, 128)
        expectation = self._expectations.pop(key, None)
        if expectation is None:
            return None
        observed_text = _clip_text(observed, 240)
        if _is_green_expectation(expectation.expected):
            exit_code = _observed_exit_code(observed_text)
            if exit_code is None or exit_code == 0:
                return None
            mismatch = PredictionMismatch.EXPECTED_GREEN_FAILED
        else:
            expected_text = expectation.expected
            if expected_text.startswith("substring:"):
                expected_text = expected_text.removeprefix("substring:").strip()
            if expected_text in observed_text:
                return None
            mismatch = PredictionMismatch.EXPECTED_SUBSTRING_ABSENT
        miss = PredictionMiss(
            action_kind=expectation.action_kind,
            fingerprint=expectation.fingerprint,
            expected=expectation.expected,
            observed=observed_text,
            mismatch=mismatch,
        )
        self._pending_expectation_fingerprints.add(miss.fingerprint)
        self._append_receipt(miss.to_dict())
        return miss

    def reflect(
        self,
        trigger: ReflectionTrigger | str | Sequence[ReflectionTrigger | str],
        *,
        target_ref: str = "",
        reply: Reply,
    ) -> ReflectionResult:
        """Parse exactly one already-triggered reply; malformed output is never retried."""

        raw_triggers = (trigger,) if isinstance(trigger, (str, ReflectionTrigger)) else tuple(trigger)
        triggers = tuple(dict.fromkeys(ReflectionTrigger(_normalize_enum(item)) for item in raw_triggers))
        if not triggers:
            raise ValueError("reflection requires a deterministic trigger")
        request = compose_self_check(triggers[0], target_ref=target_ref)
        reflection_key = _digest(
            {
                "triggers": [item.value for item in triggers],
                "target_ref": _clip_ref(target_ref),
                "reply": reply,
            }
        )
        if reflection_key in self._seen_reflections:
            receipt = ReflectionReceipt(
                event="REFLECTION_SUPPRESSED",
                status=ParseStatus.SUPPRESSED,
                trigger_kinds=triggers,
                target_ref=target_ref,
                signal_count=0,
                request_token_count=len(request.split()),
                suppressed=True,
                fingerprint=reflection_key,
            )
            self._append_receipt(receipt.to_dict())
            return ReflectionResult(request, (), receipt)
        self._remember_reflection(reflection_key)
        parsed = parse_reflection_reply(reply)
        event = "REFLECTION_MALFORMED" if parsed.status is ParseStatus.MALFORMED else (
            "REFLECTION_NO_CONCERN" if parsed.status is ParseStatus.NO_CONCERN else "REFLECTION_FIRED"
        )
        receipt = ReflectionReceipt(
            event=event,
            status=parsed.status,
            trigger_kinds=triggers,
            target_ref=target_ref,
            signal_count=len(parsed.signals),
            request_token_count=len(request.split()),
            fingerprint=reflection_key,
        )
        self._append_receipt(receipt.to_dict())
        return ReflectionResult(request, parsed.signals, receipt)

    def _remember_reflection(self, fingerprint: str) -> None:
        if len(self._seen_order) == self._seen_order.maxlen:
            expired = self._seen_order[0]
            self._seen_reflections.discard(expired)
        self._seen_order.append(fingerprint)
        self._seen_reflections.add(fingerprint)

    def evaluate(
        self,
        event: Record,
        *,
        reply: Reply | None = None,
        target_ref: str = "",
    ) -> ReflectionResult | None:
        """Convenience path: inspect event, then parse only if caller supplies a reply."""

        triggers = self.inspect(event)
        if not triggers or reply is None:
            return None
        return self.reflect(triggers, target_ref=target_ref, reply=reply)


def replay_receipts(receipts: Sequence[Record]) -> ReflectionGate:
    """Replay observed receipts through deterministic trigger detection only."""

    gate = ReflectionGate()
    for receipt in receipts:
        gate.inspect(receipt)
    return gate


def _reflection_trigger(evidence: Evidence) -> TriState:
    event = str(evidence.event.get("event", "")).upper()
    if event in CENSUS_SPEC.fire_events:
        return TriState.YES
    if event in CENSUS_SPEC.suppression_events:
        return TriState.NO
    return TriState.UNKNOWN if not evidence.complete else TriState.NO


CENSUS_SPEC = OrganCensusSpec(
    organ_id="reflection_gate",
    trigger=_reflection_trigger,
    fire_events=frozenset({"REFLECTION_TRIGGERED", "REFLECTION_FIRED", "REFLECTION_MALFORMED", "PREDICTION_MISS"}),
    suppression_events=frozenset({"REFLECTION_NO_CONCERN", "REFLECTION_SUPPRESSED"}),
    dedup_events=frozenset({"REFLECTION_SUPPRESSED"}),
    challenge_once_events=frozenset({"REFLECTION_MALFORMED"}),
)
CENSUS_SPECS = (CENSUS_SPEC,)


__all__ = [
    "CENSUS_SPEC",
    "CENSUS_SPECS",
    "ConcernKind",
    "MAX_REQUEST_TOKENS",
    "MAX_SIGNALS",
    "ParseStatus",
    "PredictionExpectation",
    "PredictionMismatch",
    "PredictionMiss",
    "ReflectionGate",
    "ReflectionParseResult",
    "ReflectionReceipt",
    "ReflectionResult",
    "ReflectionSignal",
    "ReflectionTrigger",
    "RoutingOrgan",
    "SCHEMA",
    "Severity",
    "compose_self_check",
    "parse_reflection_reply",
    "replay_receipts",
    "route_concern",
]
