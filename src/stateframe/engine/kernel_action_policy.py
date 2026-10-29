"""Pure acquire/act/ask/stop/escalate policy for O10.

The policy consumes host-owned evidence state and a versioned calibration table.
It never reads files, calls a model, mutates state, or dispatches an effect while
deciding.  File loading is kept in the explicit ``load_calibration_table`` API.

Calibration is deliberately conservative: a probability is usable for ``act``
only when its record is explicitly marked calibrated.  A high model-reported
confidence, successful schema parse, or an unmeasured default is not a
competence estimate.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping


SCHEMA = "kernel-action-policy-calibration/1"
DEFAULT_CALIBRATION_PATH = Path(__file__).with_name(
    "kernel_action_policy_calibration.json"
)
DEFAULT_MINIMUM_ACT_PROBABILITY = 0.80
DEFAULT_MAX_FAILURES = 2


class Disposition(str, Enum):
    """The only next-step dispositions produced by this policy."""

    ACQUIRE = "acquire"
    ACT = "act"
    ASK = "ask"
    STOP = "stop"
    ESCALATE = "escalate"


class ConfidenceSource(str, Enum):
    """Common confidence origins; callers may also provide a stable custom key."""

    VERIFIED_OUTCOME_HISTORY = "verified_outcome_history"
    MODEL_SELF_REPORT = "model_self_report"
    SCHEMA_PARSE = "schema_parse"
    UNKNOWN = "unknown"


def _finite_non_negative(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field_name} must be a finite non-negative number")
    return result


def _probability(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number or null")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return result


def _text(value: object, field_name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    result = value.strip()
    if not result and not allow_empty:
        raise ValueError(f"{field_name} must not be empty")
    return result


def _normalise_strings(values: Iterable[object], field_name: str) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = _text(value, field_name)
        if item not in seen:
            result.append(item)
            seen.add(item)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class FailureRecord:
    """One host-observed failed attempt, not a model-authored assertion."""

    operation: str
    reason: str
    made_progress: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation", _text(self.operation, "operation").lower())
        object.__setattr__(self, "reason", _text(self.reason, "reason"))
        if not isinstance(self.made_progress, bool):
            raise TypeError("made_progress must be a bool")

    @classmethod
    def from_value(cls, value: object) -> "FailureRecord":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(operation="unknown", reason=value)
        if isinstance(value, Mapping):
            return cls(
                operation=value.get("operation", "unknown"),
                reason=value.get("reason", "unspecified failure"),
                made_progress=bool(value.get("made_progress", False)),
            )
        raise TypeError("failure_history entries must be FailureRecord, string, or mapping")


@dataclass(frozen=True, slots=True)
class ActionCosts:
    """Caller-supplied costs in the shared budget's units."""

    acquire: float = 1.0
    act: float = 1.0
    ask: float = 1.0
    escalate: float = 1.0

    def __post_init__(self) -> None:
        for name in ("acquire", "act", "ask", "escalate"):
            object.__setattr__(self, name, _finite_non_negative(getattr(self, name), name))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ActionCosts":
        return cls(
            acquire=value.get("acquire", 1.0),
            act=value.get("act", 1.0),
            ask=value.get("ask", 1.0),
            escalate=value.get("escalate", 1.0),
        )

    def for_disposition(self, disposition: Disposition) -> float:
        if disposition is Disposition.STOP:
            return 0.0
        return float(getattr(self, disposition.value))


@dataclass(frozen=True, slots=True)
class EvidenceState:
    """Immutable host evidence presented to :func:`decide`.

    ``supported_operations`` expresses available host capabilities.  Human and
    stronger-model routes additionally require their explicit availability flag;
    the policy never assumes either route is authorized merely because it exists
    in the enum.
    """

    task_family: str
    unknown_required_facts: tuple[str, ...] = ()
    budget_remaining: float = 0.0
    failure_history: tuple[FailureRecord, ...] = ()
    confidence_source: str = ConfidenceSource.UNKNOWN.value
    model_profile: str = "default"
    confidence: float | None = None
    supported_operations: frozenset[str] = frozenset(
        {Disposition.ACQUIRE.value, Disposition.ACT.value}
    )
    human_available: bool = False
    stronger_route_available: bool = False
    observed_progress: bool = False
    disagreement: bool = False
    action_costs: ActionCosts = field(default_factory=ActionCosts)
    max_failures: int = DEFAULT_MAX_FAILURES

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_family", _text(self.task_family, "task_family"))
        object.__setattr__(self, "model_profile", _text(self.model_profile, "model_profile"))
        facts = self.unknown_required_facts
        if isinstance(facts, str):
            raise TypeError("unknown_required_facts must be an iterable of strings")
        object.__setattr__(
            self,
            "unknown_required_facts",
            _normalise_strings(facts, "unknown_required_facts"),
        )
        object.__setattr__(
            self,
            "budget_remaining",
            _finite_non_negative(self.budget_remaining, "budget_remaining"),
        )
        history = self.failure_history
        if isinstance(history, (str, bytes)):
            raise TypeError("failure_history must be an iterable of failure records")
        object.__setattr__(
            self,
            "failure_history",
            tuple(FailureRecord.from_value(item) for item in history),
        )
        object.__setattr__(
            self,
            "confidence_source",
            _text(self.confidence_source, "confidence_source").lower(),
        )
        object.__setattr__(
            self,
            "confidence",
            _probability(self.confidence, "confidence"),
        )
        operations = self.supported_operations
        if isinstance(operations, str):
            raise TypeError("supported_operations must be an iterable of operation names")
        normalised_operations = frozenset(
            _text(item, "supported_operations").lower() for item in operations
        )
        allowed = {item.value for item in Disposition} - {Disposition.STOP.value}
        unknown = normalised_operations - allowed
        if unknown:
            raise ValueError(f"unsupported operation names: {sorted(unknown)!r}")
        object.__setattr__(self, "supported_operations", normalised_operations)
        for name in ("human_available", "stronger_route_available", "observed_progress", "disagreement"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        if not isinstance(self.action_costs, ActionCosts):
            if not isinstance(self.action_costs, Mapping):
                raise TypeError("action_costs must be ActionCosts or a mapping")
            object.__setattr__(self, "action_costs", ActionCosts.from_mapping(self.action_costs))
        if isinstance(self.max_failures, bool) or not isinstance(self.max_failures, int):
            raise TypeError("max_failures must be a non-negative integer")
        if self.max_failures < 0:
            raise ValueError("max_failures must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class CalibrationRecord:
    """Outcome calibration for one task-family/confidence-source pair."""

    task_family: str = "*"
    confidence_source: str = "*"
    model_profile: str = "*"
    calibrated: bool = False
    sample_size: int = 0
    act_success_probability: float | None = None
    acquire_success_probability: float | None = None
    escalate_success_probability: float | None = None
    minimum_act_probability: float = DEFAULT_MINIMUM_ACT_PROBABILITY
    provenance: str = "uncalibrated default"

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_family", _text(self.task_family, "task_family"))
        object.__setattr__(self, "model_profile", _text(self.model_profile, "model_profile"))
        object.__setattr__(
            self,
            "confidence_source",
            _text(self.confidence_source, "confidence_source").lower(),
        )
        if not isinstance(self.calibrated, bool):
            raise TypeError("calibrated must be a bool")
        if isinstance(self.sample_size, bool) or not isinstance(self.sample_size, int):
            raise TypeError("sample_size must be a non-negative integer")
        if self.sample_size < 0:
            raise ValueError("sample_size must be a non-negative integer")
        for name in (
            "act_success_probability",
            "acquire_success_probability",
            "escalate_success_probability",
        ):
            object.__setattr__(self, name, _probability(getattr(self, name), name))
        threshold = _probability(self.minimum_act_probability, "minimum_act_probability")
        assert threshold is not None
        object.__setattr__(self, "minimum_act_probability", threshold)
        object.__setattr__(self, "provenance", _text(self.provenance, "provenance"))

    @property
    def calibration_status(self) -> str:
        return "calibrated" if self.calibrated else "uncalibrated"

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "CalibrationRecord":
        def first(*names: str) -> object:
            for name in names:
                if name in value:
                    return value[name]
            return None

        return cls(
            task_family=value.get("task_family", "*"),
            model_profile=value.get("model_profile", "*"),
            confidence_source=value.get("confidence_source", "*"),
            calibrated=value.get("calibrated", False),
            sample_size=value.get("sample_size", 0),
            act_success_probability=first("act_success_probability", "act_success"),
            acquire_success_probability=first(
                "acquire_success_probability", "acquire_success"
            ),
            escalate_success_probability=first(
                "escalate_success_probability", "escalate_success"
            ),
            minimum_act_probability=value.get(
                "minimum_act_probability", DEFAULT_MINIMUM_ACT_PROBABILITY
            ),
            provenance=value.get("provenance", "calibration data"),
        )

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["calibration_status"] = self.calibration_status
        return result


@dataclass(frozen=True, slots=True)
class CalibrationTable:
    """Immutable lookup table with explicit wildcard/default behavior."""

    defaults: CalibrationRecord = field(default_factory=CalibrationRecord)
    entries: tuple[CalibrationRecord, ...] = ()
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.defaults, CalibrationRecord):
            raise TypeError("defaults must be a CalibrationRecord")
        if not isinstance(self.entries, tuple):
            object.__setattr__(self, "entries", tuple(self.entries))
        if not isinstance(self.schema, str) or not self.schema:
            raise ValueError("schema must be a non-empty string")
        keys: set[tuple[str, str, str]] = set()
        for entry in self.entries:
            if not isinstance(entry, CalibrationRecord):
                raise TypeError("entries must contain CalibrationRecord values")
            key = (entry.task_family, entry.model_profile, entry.confidence_source)
            if key in keys:
                raise ValueError(f"duplicate calibration entry: {key!r}")
            keys.add(key)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "CalibrationTable":
        raw_defaults = value.get("defaults", {})
        if not isinstance(raw_defaults, Mapping):
            raise TypeError("calibration defaults must be a mapping")
        raw_entries = value.get("entries", ())
        if isinstance(raw_entries, Mapping):
            converted: list[Mapping[str, object]] = []
            for key, raw_entry in raw_entries.items():
                if not isinstance(raw_entry, Mapping):
                    raise TypeError("calibration entries must be mappings")
                parts = str(key).split("::")
                if len(parts) == 3:
                    family, profile, source = parts
                elif len(parts) == 2:
                    family, source = parts
                    profile = "*"
                else:
                    family, profile, source = str(key), "*", "*"
                entry = dict(raw_entry)
                entry.setdefault("task_family", family)
                entry.setdefault("model_profile", profile)
                entry.setdefault("confidence_source", source)
                converted.append(entry)
            raw_entries = converted
        if isinstance(raw_entries, (str, bytes)) or not isinstance(raw_entries, Iterable):
            raise TypeError("calibration entries must be an iterable of mappings")
        entries: list[CalibrationRecord] = []
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, Mapping):
                raise TypeError("calibration entries must be mappings")
            entries.append(CalibrationRecord.from_mapping(raw_entry))
        return cls(
            defaults=CalibrationRecord.from_mapping(raw_defaults),
            entries=tuple(entries),
            schema=str(value.get("schema", SCHEMA)),
        )

    def lookup(
        self,
        task_family: str,
        confidence_source: str,
        model_profile: str = "default",
    ) -> CalibrationRecord:
        family = _text(task_family, "task_family")
        profile = _text(model_profile, "model_profile")
        source = _text(confidence_source, "confidence_source").lower()
        candidates = (
            (family, profile, source),
            (family, profile, "*"),
            (family, "*", source),
            (family, "*", "*"),
            ("*", profile, source),
            ("*", profile, "*"),
            ("*", "*", source),
        )
        indexed = {
            (entry.task_family, entry.model_profile, entry.confidence_source): entry
            for entry in self.entries
        }
        for key in candidates:
            if key in indexed:
                return indexed[key]
        return self.defaults

    @classmethod
    def from_prior_registry(
        cls,
        path: str | Path,
        *,
        context_kind: str,
        confidence_source: str = ConfidenceSource.VERIFIED_OUTCOME_HISTORY.value,
        model_profile: str = "*",
    ) -> "CalibrationTable":
        """Adapt existing detreason priors without reimplementing posterior math."""

        try:
            from detreason.priors import CompetenceTable, PriorRegistry
        except ImportError:
            from .detreason.priors import CompetenceTable, PriorRegistry

        registry = PriorRegistry.load(str(path))
        competence = CompetenceTable(registry)
        entries: list[CalibrationRecord] = []
        for family in competence.families(context_kind):
            row = competence.competence(context_kind, family)
            mean = row.get("mean")
            sample_size = row.get("n_effective", 0)
            if not isinstance(sample_size, int):
                raise TypeError("prior registry n_effective must be an integer")
            known = bool(row.get("known"))
            calibrated = known and row.get("recommendation") != "no_data"
            entries.append(
                CalibrationRecord(
                    task_family=family,
                    model_profile=model_profile,
                    confidence_source=confidence_source,
                    calibrated=calibrated,
                    sample_size=sample_size,
                    act_success_probability=mean if isinstance(mean, (int, float)) else None,
                    acquire_success_probability=mean if isinstance(mean, (int, float)) else None,
                    escalate_success_probability=mean if isinstance(mean, (int, float)) else None,
                    provenance=f"detreason.priors:{context_kind}",
                )
            )
        return cls(entries=tuple(entries))


def load_calibration_table(path: str | Path) -> CalibrationTable:
    """Load and validate one calibration JSON document."""

    with Path(path).open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping):
        raise TypeError("calibration document must be a JSON object")
    return CalibrationTable.from_mapping(raw)


def load_default_calibration_table() -> CalibrationTable:
    """Load the checked-in data file; defaults remain explicitly uncalibrated."""

    return load_calibration_table(DEFAULT_CALIBRATION_PATH)


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    disposition: Disposition
    reason_code: str
    reason: str
    calibration: CalibrationRecord
    budget_remaining: float
    required_facts_remaining: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "disposition": self.disposition.value,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "calibration": self.calibration.as_dict(),
            "budget_remaining": self.budget_remaining,
            "required_facts_remaining": list(self.required_facts_remaining),
        }


def _failure_count(state: EvidenceState, operation: str) -> int:
    return sum(1 for item in state.failure_history if item.operation == operation)


def _available(state: EvidenceState, disposition: Disposition) -> bool:
    if disposition.value not in state.supported_operations:
        return False
    if disposition is Disposition.ASK:
        return state.human_available
    if disposition is Disposition.ESCALATE:
        return state.stronger_route_available
    return True


def _fits(state: EvidenceState, disposition: Disposition) -> bool:
    return state.action_costs.for_disposition(disposition) <= state.budget_remaining


def _decision(
    disposition: Disposition,
    reason_code: str,
    reason: str,
    state: EvidenceState,
    calibration: CalibrationRecord,
) -> PolicyDecision:
    return PolicyDecision(
        disposition=disposition,
        reason_code=reason_code,
        reason=reason,
        calibration=calibration,
        budget_remaining=state.budget_remaining,
        required_facts_remaining=state.unknown_required_facts,
    )


def _fallback_for_missing_facts(
    state: EvidenceState, calibration: CalibrationRecord
) -> PolicyDecision:
    if _available(state, Disposition.ASK) and _fits(state, Disposition.ASK):
        return _decision(
            Disposition.ASK,
            "missing_facts_ask",
            "required facts remain unknown after acquisition attempts; ask the authorized human",
            state,
            calibration,
        )
    if _available(state, Disposition.ESCALATE) and _fits(state, Disposition.ESCALATE):
        return _decision(
            Disposition.ESCALATE,
            "missing_facts_escalate",
            "required facts remain unknown and the authorized stronger route is the only remaining bounded option",
            state,
            calibration,
        )
    return _decision(
        Disposition.STOP,
        "missing_facts_stop",
        "required facts remain unknown and no authorized, affordable resolution route is available",
        state,
        calibration,
    )


def _fallback_after_untrusted_confidence(
    state: EvidenceState, calibration: CalibrationRecord, *, reason_code: str, reason: str
) -> PolicyDecision:
    if _available(state, Disposition.ESCALATE) and _fits(state, Disposition.ESCALATE):
        return _decision(Disposition.ESCALATE, reason_code, reason, state, calibration)
    if _available(state, Disposition.ASK) and _fits(state, Disposition.ASK):
        return _decision(
            Disposition.ASK,
            "confidence_ask",
            f"{reason}; ask an authorized human because escalation is unavailable",
            state,
            calibration,
        )
    return _decision(
        Disposition.STOP,
        "confidence_stop",
        f"{reason}; no authorized, affordable confidence-improvement route is available",
        state,
        calibration,
    )


def decide(state: EvidenceState, calibration: CalibrationTable) -> PolicyDecision:
    """Return one deterministic next disposition from host evidence state.

    The order is safety-first: resolve missing facts, honor hard budget limits,
    reject repeated non-progress, then require calibrated competence before act.
    No branch performs I/O or changes either input.
    """

    if not isinstance(state, EvidenceState):
        raise TypeError("state must be an EvidenceState")
    if not isinstance(calibration, CalibrationTable):
        raise TypeError("calibration must be a CalibrationTable")
    record = calibration.lookup(
        state.task_family, state.confidence_source, state.model_profile
    )

    if state.budget_remaining <= 0.0:
        return _decision(
            Disposition.STOP,
            "budget_exhausted",
            "shared budget is exhausted before another bounded disposition can be funded",
            state,
            record,
        )

    if state.unknown_required_facts:
        acquire_failures = _failure_count(state, Disposition.ACQUIRE.value)
        if (
            _available(state, Disposition.ACQUIRE)
            and _fits(state, Disposition.ACQUIRE)
            and acquire_failures < state.max_failures
            and not (record.calibrated and record.acquire_success_probability == 0.0)
        ):
            return _decision(
                Disposition.ACQUIRE,
                "missing_facts_acquire",
                "required facts are unknown; acquire bounded evidence before reasoning or acting",
                state,
                record,
            )
        return _fallback_for_missing_facts(state, record)

    act_failures = _failure_count(state, Disposition.ACT.value)
    if act_failures >= state.max_failures and not state.observed_progress:
        return _fallback_after_untrusted_confidence(
            state,
            record,
            reason_code="repeated_failure_escalate",
            reason="repeated act failures produced no observed progress",
        )

    if state.disagreement:
        return _fallback_after_untrusted_confidence(
            state,
            record,
            reason_code="disagreement_escalate",
            reason="applicable evidence disagrees; acting would conceal an unresolved conflict",
        )

    if not record.calibrated:
        return _fallback_after_untrusted_confidence(
            state,
            record,
            reason_code="uncalibrated_confidence_escalate",
            reason=(
                f"confidence source {state.confidence_source!r} has no calibrated outcome history; "
                "a default or model confidence cannot authorize act"
            ),
        )

    probability = record.act_success_probability
    if probability is None or probability < record.minimum_act_probability:
        return _fallback_after_untrusted_confidence(
            state,
            record,
            reason_code="low_calibrated_confidence_escalate",
            reason=(
                f"calibrated act success probability {probability!r} is below "
                f"the minimum {record.minimum_act_probability:.2f}"
            ),
        )

    if not _available(state, Disposition.ACT):
        return _fallback_after_untrusted_confidence(
            state,
            record,
            reason_code="act_unsupported_escalate",
            reason="calibrated evidence is sufficient but the act operation is not supported",
        )
    if not _fits(state, Disposition.ACT):
        return _fallback_after_untrusted_confidence(
            state,
            record,
            reason_code="act_over_budget_escalate",
            reason="calibrated evidence is sufficient but act does not fit the remaining budget",
        )

    return _decision(
        Disposition.ACT,
        "calibrated_act",
        (
            f"required facts are known and calibrated act success probability "
            f"{probability:.2f} meets the {record.minimum_act_probability:.2f} threshold"
        ),
        state,
        record,
    )


def next_disposition(state: EvidenceState, calibration: CalibrationTable) -> PolicyDecision:
    """Descriptive alias for integrations that name the output operation."""

    return decide(state, calibration)


__all__ = [
    "ActionCosts",
    "CalibrationRecord",
    "CalibrationTable",
    "ConfidenceSource",
    "DEFAULT_CALIBRATION_PATH",
    "Disposition",
    "EvidenceState",
    "FailureRecord",
    "PolicyDecision",
    "decide",
    "load_calibration_table",
    "load_default_calibration_table",
    "next_disposition",
]
