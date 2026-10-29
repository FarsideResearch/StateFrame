"""Deterministic difficulty dosing for the kernel's organ deliveries.

This slice is deliberately shadow-only.  It folds action receipts into a small
difficulty state, chooses one of three delivery postures, and gives the
``KernelBolt.organ_out`` adapter a class-aware dosing policy.  Unknown organs
are conservative advisory deliveries, so cruise can dose them without creating
a fail-open path.  It never calls an LLM, executes a tool, edits a plan, or
vetoes an action.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import TypeAlias

from kernel_liveness import Evidence, OrganCensusSpec, TriState


SCHEMA = "poc.kernel_cog_governor/1"
MAX_RECEIPTS = 512
SAME_FINGERPRINT_RED_STUCK = 3
IDLE_NON_PRODUCT_STUCK = 8
PREDICTION_MISS_STUCK = 1
CRUISE_GREEN_STREAK = 1
CRUISE_MAX_REDS = 1
CRUISE_MAX_IDLE = 3
CRUISE_MAX_TURN_EDIT_RATIO = 16.0
BASE_DELIVERY_CAP = 1
STUCK_DELIVERY_BONUS = 1

Record: TypeAlias = Mapping[str, object]


class CognitiveGovernorError(ValueError):
    """Raised when governor configuration or typed input is invalid."""


class Posture(StrEnum):
    """How much non-enforcement organ context should reach the model."""

    CRUISE = "CRUISE"
    STANDARD = "STANDARD"
    STUCK = "STUCK"


GovernorPosture = Posture


class OrganClass(StrEnum):
    """Delivery safety class for an organ name at the ``organ_out`` seam."""

    ENFORCEMENT = "enforcement"
    DIAGNOSIS = "diagnosis"
    ADVISORY = "advisory"


# Keep this table explicit.  The names below are the complete current literal
# first arguments to organ_out in kernel_bolt.py and probes/react_repair.py.
# Enforcement includes safety messages that must survive a quiet cruise even
# when their delivery is rendered as text rather than a hard dispatcher veto.
_ORGAN_CLASS_TABLE: dict[str, OrganClass] = {
    # kernel_bolt.py
    "frontier": OrganClass.ADVISORY,
    "hypotheses": OrganClass.ADVISORY,
    "invalidation": OrganClass.ADVISORY,
    "composed": OrganClass.ADVISORY,
    "framebreak": OrganClass.DIAGNOSIS,
    "closelook": OrganClass.DIAGNOSIS,
    "retry_gate": OrganClass.ENFORCEMENT,
    "obligation_regression": OrganClass.ENFORCEMENT,
    # probes/react_repair.py boot and action adapters
    "semantic": OrganClass.DIAGNOSIS,
    "fact_compiler": OrganClass.DIAGNOSIS,
    "crossrun": OrganClass.ADVISORY,
    "unit_plan": OrganClass.ADVISORY,
    "edit_relevance": OrganClass.DIAGNOSIS,
    "self_oracle": OrganClass.ENFORCEMENT,
    "pivot": OrganClass.ADVISORY,
    "expectation": OrganClass.ADVISORY,
    "call_path": OrganClass.DIAGNOSIS,
    "get_sure": OrganClass.DIAGNOSIS,
    "navigation": OrganClass.ADVISORY,
    "admissibility": OrganClass.ENFORCEMENT,
    "guard_tracer": OrganClass.DIAGNOSIS,
    "contract_card": OrganClass.DIAGNOSIS,
    "idle_governor": OrganClass.ENFORCEMENT,
    "reflection": OrganClass.ADVISORY,
    "groundtruth_b": OrganClass.ADVISORY,
    "advice_ledger": OrganClass.ADVISORY,
    "nitpicker": OrganClass.ENFORCEMENT,
    # Named safety aliases used by the adapter and tests.
    "done_gate": OrganClass.ENFORCEMENT,
    "target_gone": OrganClass.ENFORCEMENT,
    "red_confrontation": OrganClass.ENFORCEMENT,
    "governor_denial": OrganClass.ENFORCEMENT,
}
ORGAN_CLASS_TABLE: Mapping[str, OrganClass] = MappingProxyType(_ORGAN_CLASS_TABLE)

_GENERIC_DELIVERY_ORGAN_NAMES = frozenset({"", "cognitive_governor", "liveness", "organ"})
_MEASURE_ORGAN_NAMES = frozenset({"groundtruth_b", "measure", "measure_first"})

ENFORCEMENT_ORGANS = frozenset(
    name for name, organ_class in ORGAN_CLASS_TABLE.items()
    if organ_class is OrganClass.ENFORCEMENT
)
DIAGNOSIS_ORGANS = frozenset(
    name for name, organ_class in ORGAN_CLASS_TABLE.items()
    if organ_class is OrganClass.DIAGNOSIS
)
ADVISORY_ORGANS = frozenset(
    name for name, organ_class in ORGAN_CLASS_TABLE.items()
    if organ_class is OrganClass.ADVISORY
)

_TEST_KINDS = frozenset({"test", "pytest", "test_run"})
_EDIT_KINDS = frozenset({"edit", "str_replace", "file_create"})
_PRODUCT_KINDS = frozenset(
    {"edit", "str_replace", "file_create", "test", "pytest", "test_run",
     "contract_fact", "new_contract_fact", "new_contract", "fact"}
)
_IDLE_KINDS = frozenset({"read", "grep", "list", "collect", "collect_only"})
_NON_PRODUCT_RESULTS = frozenset(
    {
        "NO_VERDICT",
        "SELECTOR_ERROR",
        "INFRA",
        "INFRA_FAILURE",
        "ENVIRONMENT",
        "UNKNOWN",
        "NO_TEST",
        "NO_TESTS",
        "NO_OP",
        "REJECTED",
        "BLOCKED",
        "TIMEOUT",
    }
)
_GREEN_RESULTS = frozenset(
    {"GREEN", "PRODUCT_GREEN", "PASS", "PASSED", "SUCCESS", "CERTIFIED", "OK"}
)
_RED_RESULTS = frozenset(
    {"RED", "PRODUCT_RED", "FAILED", "FAILURE", "ERROR", "CONTRADICTION"}
)
_REPLAN_PIVOT_FIRE_EVENTS = frozenset(
    {
        "PIVOT_TRIGGERED",
        "REPLAN_TRIGGERED",
        "REPLAN_FIRED",
        "SNITCH_REPLANNER_TRIGGERED",
    }
)
_PREDICTION_MISS_EVENTS = frozenset(
    {"PREDICTION_MISS", "PREDICTION_MISSED", "PREDICTION_FAILURE"}
)


def _text(value: object) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())


def _normal(value: object) -> str:
    return _text(value).lower().replace("-", "_").replace(" ", "_")


def _upper(value: object) -> str:
    return _normal(value).upper()


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _record_view(record: Record) -> Record:
    """Flatten PivotObservation-style nested receipts without losing the parent event."""

    nested = record.get("receipt")
    if not isinstance(nested, Mapping):
        return record
    merged = dict(nested)
    merged.update({key: value for key, value in record.items() if value is not None})
    return merged


def _kind(record: Record) -> str:
    return _normal(record.get("kind", record.get("action_kind", record.get("type", ""))))


def _result(record: Record) -> str:
    return _upper(record.get("result_class", record.get("result", "")))


def _event(record: Record) -> str:
    return _upper(record.get("event", record.get("event_name", record.get("lifecycle_event", ""))))


def _fingerprint(record: Record) -> str:
    supplied = _text(record.get("fingerprint"))
    if supplied:
        return supplied[:160]
    return "derived:" + _digest(
        {
            "kind": _kind(record),
            "result_class": _result(record),
            "intent": _text(record.get("intent"))[:240],
            "anchor": _text(record.get("anchor", record.get("primary_anchor", "")))[:180],
        }
    )[:32]


def _turn(record: Record, ordinal: int) -> int:
    raw = record.get("turn", record.get("step", ordinal))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = ordinal
    return max(0, value)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise CognitiveGovernorError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise CognitiveGovernorError(f"{name} must be a positive integer") from exc
    if result < 1 or (isinstance(value, float) and not value.is_integer()):
        raise CognitiveGovernorError(f"{name} must be a positive integer")
    return result


def _is_summary(record: Record) -> bool:
    return _kind(record) in {"census", "summary", "report"} or _event(record) in {
        "CENSUS",
        "LIVENESS_CENSUS",
    }


def _is_green_test(record: Record) -> bool:
    return _kind(record) in _TEST_KINDS and _result(record) in _GREEN_RESULTS


def _is_red_test(record: Record) -> bool:
    return _kind(record) in _TEST_KINDS and _result(record) in _RED_RESULTS


def _is_productive(record: Record) -> bool:
    kind = _kind(record)
    return kind in _PRODUCT_KINDS and _result(record) not in _NON_PRODUCT_RESULTS


def _is_prediction_miss(record: Record) -> bool:
    event = _event(record)
    kind = _kind(record)
    result = _result(record)
    return bool(
        event in _PREDICTION_MISS_EVENTS
        or "prediction_miss" in _normal(event)
        or kind in {"prediction_miss", "prediction_missed"}
        or (kind == "prediction" and result in {"MISS", "MISSED", "FAILURE"})
        or record.get("prediction_miss") is True
        or record.get("prediction_missed") is True
    )


def _is_replan_or_pivot_fire(record: Record) -> bool:
    event = _event(record)
    if event in _REPLAN_PIVOT_FIRE_EVENTS:
        return True
    if record.get("fired") is not True:
        return False
    identity = " ".join((_kind(record), _normal(record.get("organ")), _normal(event)))
    return "pivot" in identity or "replan" in identity


def _normal_organ_name(value: object) -> str:
    name = _normal(value)
    if "." in name:
        name = name.rsplit(".", 1)[-1]
    return name


def _declared_organ_class(organ_name: object) -> OrganClass | None:
    return ORGAN_CLASS_TABLE.get(_normal_organ_name(organ_name))


def organ_class(organ_name: object) -> OrganClass:
    """Return an organ class, defaulting unknown names to dose-able advisory."""

    return _declared_organ_class(organ_name) or OrganClass.ADVISORY


organ_class_for = organ_class


def _coerce_posture(value: object, *, fail_open: bool = False) -> Posture:
    if isinstance(value, Posture):
        return value
    normalized = _upper(value)
    try:
        return Posture(normalized)
    except ValueError as exc:
        if fail_open:
            return Posture.STANDARD
        raise CognitiveGovernorError(f"unknown posture: {value!r}") from exc


def should_deliver(organ_name: object, posture: Posture | str) -> bool:
    """Return the pure delivery decision for one organ/posture pair.

    Unknown names are dose-able advisory organs.  Only advisory organs are
    quieted in CRUISE; diagnosis and enforcement always pass.
    """

    organ = organ_class(organ_name)
    selected = _coerce_posture(posture, fail_open=True)
    return organ is not OrganClass.ADVISORY or selected is not Posture.CRUISE


def _delivery_text(record: Record) -> str:
    parts = []
    for key in ("line", "detail", "message", "reason"):
        value = _text(record.get(key))
        if value:
            parts.append(value)
    return " ".join(parts)


def _delivery_organ_name(record: Record) -> str:
    """Extract the organ from cognitive and KernelBolt ``ORGAN_DELIVERED`` shapes."""

    fallback = ""
    for key in ("organ_name", "organ_id", "organ"):
        candidate = _normal_organ_name(record.get(key))
        if not candidate:
            continue
        if candidate not in _GENERIC_DELIVERY_ORGAN_NAMES:
            return candidate
        fallback = candidate
    for token in _delivery_text(record).split():
        if token.startswith("organ="):
            candidate = _normal_organ_name(token.partition("=")[2])
            if candidate:
                return candidate
    return fallback


def _delivery_requests_observation(record: Record) -> bool:
    """Recognize only delivered organ lines that explicitly ask for read/grep work."""

    if _event(record) != "ORGAN_DELIVERED":
        return False
    organ = _delivery_organ_name(record)
    text = _delivery_text(record).casefold()
    if organ == "invalidation":
        return "re-observe" in text or "reobserve" in text
    if organ in _MEASURE_ORGAN_NAMES:
        return ("probe:" in text or "measure-first" in text) and (
            "grep" in text or "read" in text
        )
    return organ == "advice_ledger" and "second notice" in text


@dataclass(frozen=True, slots=True)
class DifficultySignals:
    """Receipt-folded difficulty measurements used by the posture policy."""

    consecutive_same_fingerprint_reds: int = 0
    idle_non_product_streak: int = 0
    replan_pivot_fire_count: int = 0
    prediction_misses: int = 0
    turn_count: int = 0
    edit_count: int = 0
    turn_edit_ratio: float = 0.0
    green_streak: int = 0
    green_count: int = 0
    red_count: int = 0
    test_count: int = 0

    def __post_init__(self) -> None:
        integer_fields = (
            "consecutive_same_fingerprint_reds",
            "idle_non_product_streak",
            "replan_pivot_fire_count",
            "prediction_misses",
            "turn_count",
            "edit_count",
            "green_streak",
            "green_count",
            "red_count",
            "test_count",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CognitiveGovernorError(f"{name} must be a non-negative integer")
        if not isinstance(self.turn_edit_ratio, (int, float)) or isinstance(self.turn_edit_ratio, bool):
            raise CognitiveGovernorError("turn_edit_ratio must be numeric")
        ratio = float(self.turn_edit_ratio)
        if ratio < 0 or not math.isfinite(ratio):
            raise CognitiveGovernorError("turn_edit_ratio must be finite and non-negative")
        object.__setattr__(self, "turn_edit_ratio", round(ratio, 3))

    @property
    def same_fingerprint_red_streak(self) -> int:
        return self.consecutive_same_fingerprint_reds

    @property
    def idle_streak(self) -> int:
        return self.idle_non_product_streak

    @property
    def replan_pivot_fires(self) -> int:
        return self.replan_pivot_fire_count

    @property
    def turns(self) -> int:
        return self.turn_count

    @property
    def edits(self) -> int:
        return self.edit_count

    @property
    def turns_per_edit(self) -> float:
        return self.turn_edit_ratio

    def to_dict(self) -> dict[str, object]:
        return {
            "consecutive_same_fingerprint_reds": self.consecutive_same_fingerprint_reds,
            "idle_non_product_streak": self.idle_non_product_streak,
            "replan_pivot_fire_count": self.replan_pivot_fire_count,
            "prediction_misses": self.prediction_misses,
            "turn_count": self.turn_count,
            "edit_count": self.edit_count,
            "turn_edit_ratio": self.turn_edit_ratio,
            "green_streak": self.green_streak,
            "green_count": self.green_count,
            "red_count": self.red_count,
            "test_count": self.test_count,
        }

    as_dict = to_dict

    @classmethod
    def from_receipts(cls, receipts: Iterable[Record]) -> "DifficultySignals":
        return fold_difficulty_signals(receipts)


@dataclass(slots=True)
class _SignalState:
    consecutive_same_fingerprint_reds: int = 0
    last_red_fingerprint: str = ""
    idle_non_product_streak: int = 0
    neutral_next_observation: bool = False
    replan_pivot_fire_count: int = 0
    prediction_misses: int = 0
    turn_count: int = 0
    edit_count: int = 0
    green_streak: int = 0
    green_count: int = 0
    red_count: int = 0
    test_count: int = 0

    def observe(self, record: Record, ordinal: int) -> int:
        view = _record_view(record)
        turn = _turn(view, ordinal)
        if _is_summary(view):
            return turn
        self.turn_count = max(self.turn_count, turn + 1)
        if _event(view) == "ORGAN_DELIVERED":
            self.neutral_next_observation = _delivery_requests_observation(view)
            return turn
        kind = _kind(view)
        result = _result(view)
        neutral_observation = self.neutral_next_observation and kind in {"read", "grep"}
        self.neutral_next_observation = False

        if kind in _IDLE_KINDS:
            if not neutral_observation:
                self.idle_non_product_streak += 1
        elif kind in _PRODUCT_KINDS:
            if _is_productive(view):
                self.idle_non_product_streak = 0
            else:
                self.idle_non_product_streak += 1

        if kind in _EDIT_KINDS and result == "APPLIED":
            self.edit_count += 1
        if kind in _TEST_KINDS:
            self.test_count += 1
            if result in _GREEN_RESULTS:
                self.green_count += 1
                self.green_streak += 1
                self.consecutive_same_fingerprint_reds = 0
                self.last_red_fingerprint = ""
            elif result in _RED_RESULTS:
                self.red_count += 1
                fingerprint = _fingerprint(view)
                if fingerprint == self.last_red_fingerprint:
                    self.consecutive_same_fingerprint_reds += 1
                else:
                    self.consecutive_same_fingerprint_reds = 1
                    self.last_red_fingerprint = fingerprint
                self.green_streak = 0
            else:
                self.green_streak = 0
                self.consecutive_same_fingerprint_reds = 0
                self.last_red_fingerprint = ""

        if _is_replan_or_pivot_fire(view):
            self.replan_pivot_fire_count += 1
        if _is_prediction_miss(view):
            self.prediction_misses += 1
        return turn

    def snapshot(self) -> DifficultySignals:
        return DifficultySignals(
            consecutive_same_fingerprint_reds=self.consecutive_same_fingerprint_reds,
            idle_non_product_streak=self.idle_non_product_streak,
            replan_pivot_fire_count=self.replan_pivot_fire_count,
            prediction_misses=self.prediction_misses,
            turn_count=self.turn_count,
            edit_count=self.edit_count,
            turn_edit_ratio=self.turn_count / max(self.edit_count, 1),
            green_streak=self.green_streak,
            green_count=self.green_count,
            red_count=self.red_count,
            test_count=self.test_count,
        )


def fold_difficulty_signals(receipts: Iterable[Record]) -> DifficultySignals:
    """Fold a receipt stream without delivery side effects."""

    if isinstance(receipts, Mapping):
        nested = receipts.get("receipts", receipts.get("records", ()))
        receipts = nested if isinstance(nested, Iterable) and not isinstance(nested, (str, bytes)) else ()
    state = _SignalState()
    for ordinal, receipt in enumerate(receipts):
        if isinstance(receipt, Mapping):
            state.observe(receipt, ordinal)
    return state.snapshot()


fold_signals = fold_difficulty_signals


def posture_for(signals: DifficultySignals) -> Posture:
    """Choose posture with hard difficulty triggers taking precedence."""

    if (
        signals.consecutive_same_fingerprint_reds >= SAME_FINGERPRINT_RED_STUCK
        or signals.idle_non_product_streak >= IDLE_NON_PRODUCT_STUCK
        or signals.prediction_misses >= PREDICTION_MISS_STUCK
    ):
        return Posture.STUCK
    if (
        signals.green_streak >= CRUISE_GREEN_STREAK
        and signals.red_count <= CRUISE_MAX_REDS
        and signals.idle_non_product_streak <= CRUISE_MAX_IDLE
        and signals.replan_pivot_fire_count == 0
        and signals.prediction_misses == 0
        and signals.turn_edit_ratio <= CRUISE_MAX_TURN_EDIT_RATIO
    ):
        return Posture.CRUISE
    return Posture.STANDARD


derive_posture = posture_for


def _difficulty_reason(signals: DifficultySignals) -> str:
    reasons: list[str] = []
    if signals.consecutive_same_fingerprint_reds >= SAME_FINGERPRINT_RED_STUCK:
        reasons.append("same-fingerprint-reds")
    if signals.idle_non_product_streak >= IDLE_NON_PRODUCT_STUCK:
        reasons.append("idle-streak")
    if signals.prediction_misses >= PREDICTION_MISS_STUCK:
        reasons.append("prediction-miss")
    if reasons:
        return ",".join(reasons)
    if signals.green_streak >= CRUISE_GREEN_STREAK and signals.red_count <= CRUISE_MAX_REDS:
        return "green-progress"
    return "insufficient-convergence-evidence"


@dataclass(frozen=True, slots=True)
class CognitiveGovernorReceipt:
    """Append-only governor receipt for a fold, transition, or delivery decision."""

    event: str
    turn: int
    posture: Posture
    previous_posture: Posture | None = None
    organ_name: str = ""
    organ_class: OrganClass | None = None
    decision: str = "observe"
    delivery_cap: int = BASE_DELIVERY_CAP
    signals: DifficultySignals = field(default_factory=DifficultySignals)
    reason: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.posture, str) and not isinstance(self.posture, Posture):
            object.__setattr__(self, "posture", _coerce_posture(self.posture))
        if self.previous_posture is not None and isinstance(self.previous_posture, str):
            object.__setattr__(self, "previous_posture", _coerce_posture(self.previous_posture))
        if self.organ_class is not None and isinstance(self.organ_class, str):
            object.__setattr__(self, "organ_class", OrganClass(self.organ_class))
        if self.turn < 0 or self.delivery_cap < 1:
            raise CognitiveGovernorError("receipt turn must be non-negative and cap positive")
        object.__setattr__(self, "event", _upper(self.event))
        object.__setattr__(self, "organ_name", _normal_organ_name(self.organ_name))
        object.__setattr__(self, "decision", _normal(self.decision) or "observe")
        object.__setattr__(self, "reason", _text(self.reason)[:240])

    @property
    def receipt_id(self) -> str:
        payload = self.to_dict(include_id=False)
        return "cog:" + _digest(payload)[:24]

    def to_dict(self, *, include_id: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": SCHEMA,
            "kind": "cognitive_governor",
            "organ": "cognitive_governor",
            "event": self.event,
            "turn": self.turn,
            "posture": self.posture.value,
            "previous_posture": self.previous_posture.value if self.previous_posture else None,
            "organ_name": self.organ_name,
            "organ_class": self.organ_class.value if self.organ_class else "",
            "decision": self.decision,
            "delivery_cap": self.delivery_cap,
            "signals": self.signals.to_dict(),
            "reason": self.reason,
        }
        if include_id:
            payload["receipt_id"] = self.receipt_id
        return payload


class CognitiveGovernor:
    """Receipt-backed posture reducer and class-aware organ delivery ledger."""

    def __init__(
        self,
        *,
        base_delivery_cap: int = BASE_DELIVERY_CAP,
        max_receipts: int = MAX_RECEIPTS,
    ) -> None:
        self.base_delivery_cap = _positive_int(base_delivery_cap, "base_delivery_cap")
        self.max_receipts = _positive_int(max_receipts, "max_receipts")
        self._state = _SignalState()
        self._posture = Posture.STANDARD
        self._sequence = 0
        self._receipts: deque[dict[str, object]] = deque(maxlen=self.max_receipts)
        self._posture_history: list[dict[str, object]] = []
        self._transition_counts: Counter[str] = Counter()
        self._posture_counts: Counter[str] = Counter()
        self._suppressed_by_posture: Counter[str] = Counter()
        self._suppressed_by_organ: Counter[str] = Counter()
        self._delivered = 0
        self._suppressed = 0
        self._unknown_defaulted_advisory = 0
        self._last_receipt: CognitiveGovernorReceipt | None = None

    @property
    def posture(self) -> Posture:
        return self._posture

    @property
    def signals(self) -> DifficultySignals:
        return self._state.snapshot()

    @property
    def delivery_cap(self) -> int:
        # A cap without an organ identity uses the conservative advisory class.
        return self._delivery_cap_for(self._posture, OrganClass.ADVISORY)

    @property
    def delivery_caps(self) -> dict[str, int]:
        return {
            organ_class.value: self._delivery_cap_for(self._posture, organ_class)
            for organ_class in OrganClass
        }

    @property
    def receipts(self) -> tuple[Record, ...]:
        return tuple(dict(row) for row in self._receipts)

    @property
    def trace(self) -> tuple[Record, ...]:
        return self.receipts

    @property
    def posture_history(self) -> tuple[Record, ...]:
        return tuple(
            {
                "turn": row["turn"],
                "posture": row["posture"],
                "signals": dict(row["signals"]),
            }
            for row in self._posture_history
        )

    @property
    def history(self) -> tuple[Record, ...]:
        return self.posture_history

    @property
    def last_receipt(self) -> CognitiveGovernorReceipt | None:
        return self._last_receipt

    @staticmethod
    def _delivery_cap_for_posture(
        posture: Posture,
        base: int,
        organ_class: OrganClass = OrganClass.ADVISORY,
    ) -> int:
        if posture is Posture.STUCK and organ_class in {
            OrganClass.ENFORCEMENT,
            OrganClass.DIAGNOSIS,
        }:
            return base + STUCK_DELIVERY_BONUS
        return base

    def _delivery_cap_for(
        self,
        posture: Posture,
        organ_class: OrganClass = OrganClass.ADVISORY,
    ) -> int:
        return self._delivery_cap_for_posture(posture, self.base_delivery_cap, organ_class)

    def delivery_cap_for(
        self,
        organ_name: object,
        posture: Posture | str | None = None,
    ) -> int:
        """Return the class-specific cap, with unknown organs treated as advisory."""

        selected = self._posture if posture is None else _coerce_posture(posture, fail_open=True)
        return self._delivery_cap_for(selected, organ_class(organ_name))

    def _append(self, receipt: CognitiveGovernorReceipt) -> CognitiveGovernorReceipt:
        self._last_receipt = receipt
        self._receipts.append(receipt.to_dict())
        return receipt

    def observe_receipt(self, receipt: Record) -> CognitiveGovernorReceipt:
        """Fold one raw action receipt and record any posture transition."""

        if not isinstance(receipt, Mapping):
            raise CognitiveGovernorError("receipt must be a mapping")
        view = _record_view(receipt)
        turn = self._state.observe(view, self._sequence)
        self._sequence += 1
        signals = self.signals
        previous = self._posture
        next_posture = posture_for(signals)
        self._posture = next_posture
        self._posture_counts[next_posture.value] += 1
        self._posture_history.append(
            {"turn": turn, "posture": next_posture.value, "signals": signals.to_dict()}
        )
        event = "SUMMARY_IGNORED" if _is_summary(view) else "SIGNALS_FOLDED"
        folded = self._append(
            CognitiveGovernorReceipt(
                event=event,
                turn=turn,
                posture=next_posture,
                previous_posture=previous,
                decision="observe",
                delivery_cap=self._delivery_cap_for(next_posture),
                signals=signals,
                reason=_difficulty_reason(signals),
            )
        )
        if next_posture is not previous:
            transition = f"{previous.value}->{next_posture.value}"
            self._transition_counts[transition] += 1
            self._append(
                CognitiveGovernorReceipt(
                    event="POSTURE_TRANSITION",
                    turn=turn,
                    posture=next_posture,
                    previous_posture=previous,
                    decision="transition",
                    delivery_cap=self._delivery_cap_for(next_posture),
                    signals=signals,
                    reason=transition,
                )
            )
        return folded

    observe = observe_receipt
    ingest_receipt = observe_receipt
    record_receipt = observe_receipt

    def should_deliver(
        self,
        organ_name: object,
        posture: Posture | str | None = None,
    ) -> bool:
        """Decide and receipt-log delivery at the ``organ_out`` weld."""

        selected = self._posture if posture is None else _coerce_posture(posture, fail_open=True)
        declared = _declared_organ_class(organ_name)
        organ = declared or OrganClass.ADVISORY
        deliver = should_deliver(organ_name, selected)
        if deliver:
            self._delivered += 1
        else:
            self._suppressed += 1
            self._suppressed_by_posture[selected.value] += 1
            self._suppressed_by_organ[_normal_organ_name(organ_name)] += 1
        if declared is None:
            event = "ORGAN_UNKNOWN_DEFAULTED_ADVISORY"
            self._unknown_defaulted_advisory += 1
        else:
            event = "ORGAN_DELIVERED" if deliver else "ORGAN_SUPPRESSED"
        self._append(
            CognitiveGovernorReceipt(
                event=event,
                turn=max(0, self._sequence - 1),
                posture=selected,
                organ_name=_normal_organ_name(organ_name),
                organ_class=organ,
                decision="deliver" if deliver else "suppress",
                delivery_cap=self._delivery_cap_for(selected, organ),
                signals=self.signals,
                reason=("unknown-organ-defaulted-advisory" if declared is None else "posture-policy"),
            )
        )
        return deliver

    decide_delivery = should_deliver

    def census(self) -> dict[str, object]:
        """Return deterministic posture, suppression, and receipt counts."""

        transitions = {
            key: self._transition_counts[key]
            for key in sorted(self._transition_counts)
        }
        suppressed_by_posture = {
            posture.value: self._suppressed_by_posture[posture.value]
            for posture in Posture
        }
        suppressed_by_organ = {
            key: self._suppressed_by_organ[key]
            for key in sorted(self._suppressed_by_organ)
        }
        posture_counts = {
            posture.value: self._posture_counts[posture.value]
            for posture in Posture
        }
        return {
            "schema": SCHEMA,
            "posture": self._posture.value,
            "posture_transitions": transitions,
            "posture_transition_count": sum(transitions.values()),
            "suppressed_by_posture": suppressed_by_posture,
            "suppressed_by_organ": suppressed_by_organ,
            "posture_counts": posture_counts,
            "delivered": self._delivered,
            "suppressed": self._suppressed,
            # Retain the old key as a zero-valued compatibility field; unknown
            # organs no longer use fail-open delivery.
            "unknown_fail_open": 0,
            "unknown_defaulted_advisory": self._unknown_defaulted_advisory,
            "delivery_cap": self.delivery_cap,
            "delivery_caps": self.delivery_caps,
            "receipt_count": len(self._receipts),
            "signals": self.signals.to_dict(),
        }

    summary = census


def load_receipts(path: str | Path) -> tuple[Record, ...]:
    """Load valid JSON object rows from a receipt JSONL tape."""

    records: list[Record] = []
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ()
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            records.append(value)
    return tuple(records)


def replay_receipts(
    receipts: Iterable[Record],
    *,
    base_delivery_cap: int = BASE_DELIVERY_CAP,
    max_receipts: int = MAX_RECEIPTS,
) -> CognitiveGovernor:
    """Replay a receipt stream through the same deterministic fold."""

    governor = CognitiveGovernor(
        base_delivery_cap=base_delivery_cap,
        max_receipts=max_receipts,
    )
    for receipt in receipts:
        if isinstance(receipt, Mapping):
            governor.observe_receipt(receipt)
    return governor


def _cog_governor_trigger(evidence: Evidence) -> TriState:
    event = _event(evidence.event)
    if event in {
        "SIGNALS_FOLDED",
        "POSTURE_TRANSITION",
        "ORGAN_DELIVERED",
        "ORGAN_SUPPRESSED",
        "ORGAN_UNKNOWN_FAIL_OPEN",
        "ORGAN_UNKNOWN_DEFAULTED_ADVISORY",
    }:
        return TriState.YES
    return TriState.UNKNOWN if not evidence.complete else TriState.NO


CENSUS_SPEC = OrganCensusSpec(
    organ_id="cognitive_governor",
    trigger=_cog_governor_trigger,
    fire_events=frozenset(
        {
            "SIGNALS_FOLDED",
            "POSTURE_TRANSITION",
            "ORGAN_DELIVERED",
            "ORGAN_SUPPRESSED",
            "ORGAN_UNKNOWN_FAIL_OPEN",
            "ORGAN_UNKNOWN_DEFAULTED_ADVISORY",
        }
    ),
    suppression_events=frozenset({"SUMMARY_IGNORED"}),
    dedup_events=frozenset(),
    challenge_once_events=frozenset({"POSTURE_TRANSITION"}),
)
CENSUS_SPECS = (CENSUS_SPEC,)


__all__ = [
    "ADVISORY_ORGANS",
    "BASE_DELIVERY_CAP",
    "CENSUS_SPEC",
    "CENSUS_SPECS",
    "CRUISE_GREEN_STREAK",
    "CRUISE_MAX_IDLE",
    "CRUISE_MAX_REDS",
    "CRUISE_MAX_TURN_EDIT_RATIO",
    "CognitiveGovernor",
    "CognitiveGovernorError",
    "CognitiveGovernorReceipt",
    "DIAGNOSIS_ORGANS",
    "DifficultySignals",
    "ENFORCEMENT_ORGANS",
    "GovernorPosture",
    "IDLE_NON_PRODUCT_STUCK",
    "MAX_RECEIPTS",
    "OrganClass",
    "ORGAN_CLASS_TABLE",
    "PREDICTION_MISS_STUCK",
    "Posture",
    "SCHEMA",
    "SAME_FINGERPRINT_RED_STUCK",
    "STUCK_DELIVERY_BONUS",
    "derive_posture",
    "fold_difficulty_signals",
    "fold_signals",
    "load_receipts",
    "organ_class",
    "organ_class_for",
    "posture_for",
    "replay_receipts",
    "should_deliver",
]
