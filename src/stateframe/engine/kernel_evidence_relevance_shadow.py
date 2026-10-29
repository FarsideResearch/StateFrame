"""Shadow-only System One site for evidence relevance.

This module observes the items already selected by :mod:`frame_selector` and
asks an injected ``kernel_decision_api`` BOOLEAN scorer about each item.  It
does not select, reorder, remove, or rewrite frame content.  Labels are joined
after scoring from host journal citations, so labels never enter the model
request.

The concrete ``kernel_decision_api.py`` function is intentionally injected so
replay fixtures stay provider-free.  Its verified ``score_boolean`` signature
is represented by the small protocol below.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

try:  # Supports the v46 flat-module loader and package imports.
    from kernel_continuation_journal import journal_safe_fields
    from kernel_decision_calibration import calibrate_probabilities
    from kernel_decision_registry import (
        AuthorityBoundary,
        CostPair as RegistryCostPair,
        DecisionSite,
        DecisionSiteRegistry,
        LabelSource,
        OptionSpec,
        QuestionType,
        emit_decision_labelled,
        emit_decision_recorded,
    )
except ImportError:  # pragma: no cover - exercised by package importers.
    from .kernel_continuation_journal import journal_safe_fields
    from .kernel_decision_calibration import calibrate_probabilities
    from .kernel_decision_registry import (
        AuthorityBoundary,
        CostPair as RegistryCostPair,
        DecisionSite,
        DecisionSiteRegistry,
        LabelSource,
        OptionSpec,
        QuestionType,
        emit_decision_labelled,
        emit_decision_recorded,
    )


SCHEMA = "sys1-evidence-relevance-shadow/1"
SITE_ID = "frame_selector.evidence_relevance"
QUESTION_VERSION = "evidence-relevance/v1"
ANSWERABLE = "ANSWERABLE"
INSUFFICIENT = "INSUFFICIENT"
UNAVAILABLE = "UNAVAILABLE"
DEFAULT_CANDIDATE_TOP_K = 8
SHADOW_BUDGET_SCOPE = "sys1_shadow"
SHADOW_ADMISSION_EVENT = "shadow_admission_receipt"
SHADOW_RESPONSE_EVENT = "shadow_response_receipt"
SHADOW_CONSUMPTION_EVENT = "shadow_consumption_receipt"
_USAGE_KNOWN = "KNOWN"
_USAGE_UNKNOWN = "UNKNOWN"
_USAGE_NOT_OBSERVED = "NOT_OBSERVED"

_EVALUATOR_KEYS = frozenset({
    "accepted",
    "attention",
    "evaluator",
    "evaluator_fix",
    "fix",
    "fixes",
    "gold",
    "grader",
    "ground_truth",
    "label",
    "labels",
    "model_attention",
    "reference",
    "target_label",
    "verdict",
})

_EDIT_KINDS = frozenset(
    {
        "edit_accepted",
        "edit_applied",
        "edit_completed",
        "edit_outcome_closed",
    }
)
_CHECK_KINDS = frozenset(
    {
        "check_passed",
        "check_completed",
        "check_outcome_closed",
        "test_passed",
    }
)
_CITATION_FIELDS = (
    "cited_receipt_id",
    "cited_receipt_ids",
    "evidence_receipt_id",
    "evidence_receipt_ids",
    "supporting_receipt_id",
    "supporting_receipt_ids",
)
_SUCCESS_STATUSES = frozenset({"ACCEPTED", "APPLIED", "PASS", "PASSED", "OK"})


class BooleanDecisionAPI(Protocol):
    """SYS1_API dependency required by :func:`score_evidence_relevance`.

    This matches :func:`kernel_decision_api.score_boolean`; the request object
    stays on this adapter's boundary so the provider API never sees host labels.
    """

    def score_boolean(
        self,
        question: str,
        *,
        evidence: Sequence[object],
        decision_id: str,
        site_id: str,
        question_version: str,
    ) -> object:
        """Return the SYS1 BOOLEAN ``DecisionRecord`` for one candidate."""


class CalibrationAPI(Protocol):
    """Minimal site-calibration seam used by the observational adapter."""

    site_id: str

    def evaluate(
        self,
        probabilities: Mapping[str, float],
        *,
        selected: str | None = None,
        legal_mass: float = 1.0,
    ) -> Mapping[str, object]:
        """Return raw/calibrated probabilities and disposition metadata."""


@dataclass(frozen=True, slots=True)
class ShadowReservation:
    """An aggregate lease that is structurally unable to spend solver budget."""

    reservation_id: str
    resource_account_id: str
    endpoint_id: str
    reserved_units: int
    budget_scope: str = SHADOW_BUDGET_SCOPE
    primary_solver_units: int = 0

    def __post_init__(self) -> None:
        for name in ("reservation_id", "resource_account_id", "endpoint_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty text")
        if type(self.reserved_units) is not int or self.reserved_units < 1:
            raise ValueError("reserved_units must be a positive integer")
        if self.budget_scope != SHADOW_BUDGET_SCOPE:
            raise ValueError("reservation is not in the shadow budget scope")
        if self.primary_solver_units != 0:
            raise ValueError("shadow reservation cannot consume primary solver units")


class ShadowResourceAccount(Protocol):
    """Host-owned shared endpoint account with a separate shadow namespace.

    A generic ``reserve`` method is deliberately not accepted.  The explicit
    shadow methods make it impossible for this adapter to accidentally spend
    the primary solver's budget while still allowing the host to enforce one
    endpoint-wide capacity account.
    """

    def reserve_shadow(
        self,
        units: int,
        *,
        reservation_key: str,
        site_id: str,
        candidate_set_id: str,
    ) -> ShadowReservation | None:
        """Reserve all candidate-item units as one bounded admission."""

    def settle_shadow(
        self,
        reservation: ShadowReservation,
        *,
        observed_usage: int | None,
        cache_hits: int,
        timeouts: int,
    ) -> None:
        """Settle the aggregate reservation exactly once."""


@dataclass(frozen=True, slots=True)
class ShadowAccounting:
    """Aggregate endpoint/resource facts for one shadow candidate batch."""

    reservation_id: str | None
    resource_account_id: str | None
    endpoint_id: str | None
    budget_scope: str
    requested_items: int
    admitted_items: int
    admission_status: str
    cache_hits: int
    timeouts: int
    observed_usage: int | None
    usage_status: str
    settled: bool
    primary_solver_units: int
    admission_receipt_id: str
    response_receipt_ids: tuple[str, ...]
    consumption_receipt_id: str

    def to_dict(self) -> dict[str, object]:
        return {
            "reservation_id": self.reservation_id,
            "resource_account_id": self.resource_account_id,
            "endpoint_id": self.endpoint_id,
            "budget_scope": self.budget_scope,
            "requested_items": self.requested_items,
            "admitted_items": self.admitted_items,
            "admission_status": self.admission_status,
            "cache_hits": self.cache_hits,
            "timeouts": self.timeouts,
            "observed_usage": self.observed_usage,
            "usage_status": self.usage_status,
            "settled": self.settled,
            "primary_solver_units": self.primary_solver_units,
            "admission_receipt_id": self.admission_receipt_id,
            "response_receipt_ids": list(self.response_receipt_ids),
            "consumption_receipt_id": self.consumption_receipt_id,
        }


def evidence_relevance_site(*, candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K) -> DecisionSite:
    """Return the declarative BOOLEAN site for the host-selected evidence menu."""

    return DecisionSite(
        site_id=SITE_ID,
        question_type=QuestionType.BOOLEAN,
        options=(
            OptionSpec("true", "Y", "The evidence item is relevant."),
            OptionSpec("false", "N", "The evidence item is not relevant."),
        ),
        label_source=LabelSource(
            "accepted_edit_or_passing_check",
            "Host accepted edit or passing check cites the evidence receipt.",
        ),
        cost_pair=RegistryCostPair(wrong_action=10, fallback=1),
        authority_boundary=AuthorityBoundary("host", "host", True),
        fallback="host_selector",
        question_version=QUESTION_VERSION,
        candidate_top_k=candidate_top_k,
    )


def evidence_relevance_registry(
    *, candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K
) -> DecisionSiteRegistry:
    """Create an isolated registry containing only the SYS1 shadow site."""

    return DecisionSiteRegistry((evidence_relevance_site(candidate_top_k=candidate_top_k),))


@dataclass(frozen=True, slots=True)
class BooleanScoreRequest:
    """Provider-neutral request recorded at the SYS1 BOOLEAN boundary."""

    decision_id: str
    site_id: str
    question_version: str
    statement: str
    run_id: str
    step: int
    frame_context: Mapping[str, object] = field(default_factory=dict)
    evidence: tuple[Mapping[str, object], ...] = ()
    evidence_handles: tuple[Mapping[str, str], ...] = ()
    candidate_set_id: str = ""
    candidate_ids: tuple[str, ...] = ()
    menu_option_ids: tuple[str, ...] = ("true", "false")

    def to_dict(self) -> dict[str, object]:
        """Return an isolated payload without exposing caller-owned objects."""

        return {
            "schema": "kernel-decision-api/boolean-request/1",
            "decision_id": self.decision_id,
            "site_id": self.site_id,
            "question_version": self.question_version,
            "statement": self.statement,
            "run_id": self.run_id,
            "step": self.step,
            "frame_context": _clone(self.frame_context),
            "evidence": _clone(self.evidence),
            "evidence_handles": _clone(self.evidence_handles),
            "candidate_set_id": self.candidate_set_id,
            "candidate_ids": list(self.candidate_ids),
            "menu_option_ids": list(self.menu_option_ids),
        }


@dataclass(frozen=True, slots=True)
class LabelJoin:
    """Host-derived labels for the candidate receipt identities."""

    outcome_available: bool
    labels: Mapping[str, bool | None]
    sources: Mapping[str, tuple[str, ...]]
    label_kinds: Mapping[str, tuple[str, ...]]
    source_timestamps: Mapping[str, str | int | float]
    positive_receipt_ids: tuple[str, ...]
    source_event_ids: tuple[str, ...]
    outcome_state: str


@dataclass(frozen=True, slots=True)
class EvidenceRelevanceRecord:
    """One advisory BOOLEAN score and its optional host outcome label."""

    decision_id: str
    site_id: str
    question_version: str
    item_id: str
    receipt_id: str
    run_id: str
    step: int
    candidate_set_id: str
    candidate_rank: int
    menu_option_ids: tuple[str, ...]
    status: str
    answerability: str
    decision_record: Mapping[str, object] | None
    raw_p: Mapping[str, float]
    calibrated_p: Mapping[str, float] | None
    legal_mass: float | None
    temperature: float | None
    threshold: float | None
    disposition: str
    label: bool | None
    label_sources: tuple[str, ...]
    label_source_kinds: tuple[str, ...]
    request: BooleanScoreRequest
    error_type: str | None = None
    response_received: bool = False
    cache_hit: bool = False
    timeout: bool = False
    observed_usage: int | None = None
    usage_status: str = _USAGE_NOT_OBSERVED
    admission_receipt_id: str = ""
    response_receipt_id: str = ""
    consumption_receipt_id: str = ""

    def to_dict(self) -> dict[str, object]:
        """Serialize the shadow receipt without adding authority to it."""

        return {
            "schema": SCHEMA,
            "decision_id": self.decision_id,
            "site_id": self.site_id,
            "question_version": self.question_version,
            "item_id": self.item_id,
            "receipt_id": self.receipt_id,
            "run_id": self.run_id,
            "step": self.step,
            "candidate_set_id": self.candidate_set_id,
            "candidate_rank": self.candidate_rank,
            "menu_option_ids": list(self.menu_option_ids),
            "status": self.status,
            "answerability": self.answerability,
            "decision_record": _clone(self.decision_record),
            "raw_p": _clone(self.raw_p),
            "calibrated_p": _clone(self.calibrated_p),
            "legal_mass": self.legal_mass,
            "T_site": self.temperature,
            "tau_site": self.threshold,
            "disposition": self.disposition,
            "label": self.label,
            "label_sources": list(self.label_sources),
            "label_source_kinds": list(self.label_source_kinds),
            "request": self.request.to_dict(),
            "error_type": self.error_type,
            "response_received": self.response_received,
            "cache_hit": self.cache_hit,
            "timeout": self.timeout,
            "observed_usage": self.observed_usage,
            "usage_status": self.usage_status,
            "admission_receipt_id": self.admission_receipt_id,
            "response_receipt_id": self.response_receipt_id,
            "consumption_receipt_id": self.consumption_receipt_id,
            "shadow": True,
            "applied": False,
        }


@dataclass(frozen=True, slots=True)
class ShadowEvaluation:
    """Complete observation; ``rendered_frame`` is returned exactly as given."""

    rendered_frame: object
    records: tuple[EvidenceRelevanceRecord, ...]
    labels: LabelJoin
    candidate_set_id: str
    candidate_ids: tuple[str, ...]
    candidate_receipt_ids: tuple[str, ...]
    menu_option_ids: tuple[str, ...]
    accounting: ShadowAccounting
    omitted_candidate_ids: tuple[str, ...] = ()
    journal_events: tuple[Mapping[str, object], ...] = ()
    journal_errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return only shadow metadata; frame bytes stay outside this payload."""

        return {
            "schema": SCHEMA,
            "record_count": len(self.records),
            "records": [record.to_dict() for record in self.records],
            "candidate_set_id": self.candidate_set_id,
            "candidate_ids": list(self.candidate_ids),
            "candidate_receipt_ids": list(self.candidate_receipt_ids),
            "menu_option_ids": list(self.menu_option_ids),
            "accounting": self.accounting.to_dict(),
            "omitted_candidate_ids": list(self.omitted_candidate_ids),
            "label_outcome_available": self.labels.outcome_available,
            "label_outcome_state": self.labels.outcome_state,
            "positive_receipt_ids": list(self.labels.positive_receipt_ids),
            "journal_event_ids": [
                str(event["event_id"])
                for event in self.journal_events
                if "event_id" in event
            ],
            "journal_errors": list(self.journal_errors),
            "shadow": True,
            "applied": False,
        }


def _clone(value: object) -> object:
    """Deep-copy request/receipt data so a scorer cannot mutate host inputs."""

    return copy.deepcopy(value)


def _public_copy(value: object) -> object:
    """Copy public evidence while excluding evaluator and attention artifacts."""

    if isinstance(value, Mapping):
        result: dict[object, object] = {}
        for key, nested in value.items():
            normalized = str(key).casefold().replace("-", "_").replace(" ", "_")
            if normalized in _EVALUATOR_KEYS:
                continue
            result[key] = _public_copy(nested)
        return result
    if isinstance(value, list):
        return [_public_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_public_copy(item) for item in value)
    return copy.deepcopy(value)


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _step(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("step must be a non-negative integer")
    return value


def _canonical_digest(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        payload = repr(value).encode("utf-8", "replace")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _receipt_identity(kind: str, value: object) -> str:
    """Create a stable receipt handle without pretending it is host authority."""

    return f"shadow-{kind}:" + _canonical_digest(value).removeprefix("sha256:")


def _nested_mapping(value: object, key: str) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    nested = value.get(key)
    return nested if isinstance(nested, Mapping) else None


def _explicit_bool(value: Mapping[str, object], names: Sequence[str]) -> bool:
    for name in names:
        raw = value.get(name)
        if isinstance(raw, bool):
            return raw
    for container_name in ("metadata", "cost", "usage"):
        nested = _nested_mapping(value, container_name)
        if nested is not None:
            for name in names:
                raw = nested.get(name)
                if isinstance(raw, bool):
                    return raw
    return False


def _observed_usage(value: Mapping[str, object], *, response_received: bool) -> tuple[int | None, str]:
    """Extract host-reported units; absence is never converted to zero."""

    candidates: list[Mapping[str, object]] = [value]
    for name in ("usage", "observed_usage", "cost"):
        nested = _nested_mapping(value, name)
        if nested is not None:
            candidates.append(nested)
    for candidate in candidates:
        for name in ("total_tokens", "tokens", "observed_units", "completion_tokens"):
            raw = candidate.get(name)
            if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
                return raw, _USAGE_KNOWN
    return (None, _USAGE_UNKNOWN if response_received else _USAGE_NOT_OBSERVED)


def _timeout_observed(value: Mapping[str, object]) -> bool:
    if _explicit_bool(value, ("timeout", "timed_out")):
        return True
    status = str(value.get("status", value.get("transport_status", ""))).upper()
    return "TIMEOUT" in status


def _emit_shadow_receipt(
    journal: object,
    kind: str,
    payload: Mapping[str, object],
) -> dict[str, object]:
    if kind not in {SHADOW_ADMISSION_EVENT, SHADOW_RESPONSE_EVENT, SHADOW_CONSUMPTION_EVENT}:
        raise ValueError("unsupported shadow receipt event")
    emit = getattr(journal, "emit", None)
    if not callable(emit):
        raise TypeError("journal must provide emit(kind, **fields)")
    emitted = emit(kind, **journal_safe_fields(payload))
    if not isinstance(emitted, Mapping):
        raise TypeError("journal receipt emission must return a mapping")
    return dict(emitted)


def _try_reserve_shadow(
    resource_account: ShadowResourceAccount | None,
    *,
    units: int,
    reservation_key: str,
    site_id: str,
    candidate_set_id: str,
) -> tuple[ShadowReservation | None, str | None]:
    if resource_account is None:
        return None, "SHADOW_RESOURCE_ACCOUNT_REQUIRED"
    reserve = getattr(resource_account, "reserve_shadow", None)
    settle = getattr(resource_account, "settle_shadow", None)
    if not callable(reserve) or not callable(settle):
        return None, "SHADOW_RESOURCE_ACCOUNT_INVALID"
    try:
        reservation = reserve(
            units,
            reservation_key=reservation_key,
            site_id=site_id,
            candidate_set_id=candidate_set_id,
        )
    except Exception as error:  # account errors are observational in shadow mode
        return None, f"SHADOW_RESOURCE_ACCOUNT_{type(error).__name__}"
    if reservation is None:
        return None, "SHADOW_ADMISSION_REFUSED"
    if not isinstance(reservation, ShadowReservation):
        return None, "SHADOW_RESERVATION_INVALID"
    if reservation.reserved_units != units:
        return None, "SHADOW_RESERVATION_BOUND_MISMATCH"
    return reservation, None


def _item_id(item: Mapping[str, object]) -> str:
    value = item.get("id", item.get("item_id"))
    return _text(value, "evidence item id")


def _receipt_id(item: Mapping[str, object]) -> str:
    value = item.get("receipt_id")
    if value in (None, ""):
        payload = item.get("payload")
        if isinstance(payload, Mapping):
            value = payload.get("receipt_id")
    return _text(value, "evidence receipt_id")


def _evidence_view(item: Mapping[str, object], item_id: str, receipt_id: str) -> dict[str, object]:
    """Expose typed item facts while excluding selector/evaluator labels."""

    result: dict[str, object] = {
        "item_id": item_id,
        "receipt_id": receipt_id,
    }
    for key in (
        "kind",
        "seq",
        "payload",
        "links",
        "source",
        "span",
        "gist",
    ):
        if key in item:
            result[key] = _public_copy(item[key])
    return result


def _decision_id(site_id: str, run_id: str, step: int, item_id: str, receipt_id: str) -> str:
    identity = {
        "site_id": site_id,
        "run_id": run_id,
        "step": step,
        "item_id": item_id,
        "receipt_id": receipt_id,
    }
    return "decision:" + _canonical_digest(identity).removeprefix("sha256:")


def _candidate_set_id(site_id: str, run_id: str, step: int, items: Sequence[Mapping[str, object]]) -> str:
    identity = {
        "site_id": site_id,
        "run_id": run_id,
        "step": step,
        "candidates": [
            {"item_id": _item_id(item), "receipt_id": _receipt_id(item)}
            for item in items
        ],
    }
    return "candidates:" + _canonical_digest(identity).removeprefix("sha256:")


def _score_mapping(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(_clone(value))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, Mapping):
            return dict(_clone(converted))
    if dataclasses.is_dataclass(value):
        converted = dataclasses.asdict(value)
        if isinstance(converted, Mapping):
            return dict(_clone(converted))
    raise TypeError("SYS1_API score_boolean must return a mapping or to_dict record")


def _probabilities(score: Mapping[str, object]) -> dict[str, float]:
    value = score.get("normalized_probabilities", score.get("probabilities"))
    if value is None:
        true_value = score.get("p_true")
        false_value = score.get("p_false")
        if true_value is not None or false_value is not None:
            value = {"true": true_value or 0.0, "false": false_value or 0.0}
    if not isinstance(value, Mapping):
        return {}
    probabilities: dict[str, float] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or isinstance(raw, bool) or not isinstance(raw, (int, float)):
            continue
        probability = float(raw)
        if math.isfinite(probability) and 0.0 <= probability <= 1.0:
            probabilities[key] = probability
    total = sum(probabilities.values())
    return {key: value / total for key, value in probabilities.items()} if total > 0 else {}


def _legal_mass(score: Mapping[str, object]) -> float | None:
    value = score.get("legal_mass")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and 0.0 <= value <= 1.0 else None


def _selected_option(score: Mapping[str, object], probabilities: Mapping[str, float]) -> str | None:
    value = score.get("selected_option", score.get("selected"))
    if isinstance(value, str) and value in probabilities:
        return value
    value = score.get("argmax_option")
    if isinstance(value, str) and value in probabilities:
        return value
    return max(probabilities, key=probabilities.__getitem__) if probabilities else None


def _answerability(score: Mapping[str, object], probabilities: Mapping[str, float]) -> str:
    legal_mass = _legal_mass(score)
    if not probabilities or legal_mass is None or legal_mass < 0.9:
        return INSUFFICIENT
    if score.get("abstained") is True or score.get("abstain") is True:
        return INSUFFICIENT
    return ANSWERABLE


def _calibrated(
    calibration: CalibrationAPI,
    probabilities: Mapping[str, float],
    *,
    selected: str | None,
    legal_mass: float | None,
) -> tuple[dict[str, float] | None, float | None, float | None, str]:
    if not probabilities:
        return None, None, None, INSUFFICIENT
    evaluated = calibration.evaluate(
        probabilities,
        selected=selected,
        legal_mass=legal_mass if legal_mass is not None else 0.0,
    )
    temperature = evaluated.get("T_site", evaluated.get("temperature"))
    threshold = evaluated.get("tau_site", evaluated.get("tau"))
    calibrated = evaluated.get("calibrated_p")
    if isinstance(calibrated, Mapping):
        calibrated_p = {key: float(value) for key, value in calibrated.items()}
    elif isinstance(temperature, (int, float)) and not isinstance(temperature, bool):
        calibrated_p = calibrate_probabilities(probabilities, float(temperature))
    else:
        calibrated_p = None
    return (
        calibrated_p,
        float(temperature) if isinstance(temperature, (int, float)) and not isinstance(temperature, bool) else None,
        float(threshold) if isinstance(threshold, (int, float)) and not isinstance(threshold, bool) else None,
        str(evaluated.get("disposition", "UNCALIBRATED_PROPOSE")),
    )


def _event_kind(record: Mapping[str, object]) -> str:
    value = record.get("kind", record.get("event_type", ""))
    return str(value).strip().lower().replace("-", "_")


def _label_record(record: Mapping[str, object]) -> Mapping[str, object]:
    """Flatten a journal envelope without broadening its citation fields."""

    nested = record.get("record")
    if not isinstance(nested, Mapping):
        return record
    merged = dict(nested)
    for key in ("kind", "event_type", "run_id", "step", "event_id", "sequence", "timestamp_ns"):
        if key in record:
            merged[key] = record[key]
    return merged


def _event_applies(
    record: Mapping[str, object], *, run_id: str | None, step: int | None
) -> bool:
    if run_id is not None and record.get("run_id") != run_id:
        return False
    if step is not None and record.get("step") != step:
        return False
    return True


def _sequence_values(value: object) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _citation_ids(record: Mapping[str, object]) -> tuple[str, ...]:
    """Read citation fields only; item IDs and action receipt IDs are ignored."""

    found: list[str] = []
    for field_name in _CITATION_FIELDS:
        for value in _sequence_values(record.get(field_name)):
            if isinstance(value, Mapping):
                value = value.get("receipt_id")
            if isinstance(value, str) and value.strip() and value.strip() not in found:
                found.append(value.strip())
    for field_name in ("evidence", "citations", "supporting_evidence"):
        nested = record.get(field_name)
        for value in _sequence_values(nested):
            if isinstance(value, Mapping):
                receipt_id = value.get("receipt_id")
                if isinstance(receipt_id, str) and receipt_id.strip() and receipt_id.strip() not in found:
                    found.append(receipt_id.strip())
    return tuple(found)


def _is_successful_edit(record: Mapping[str, object], kind: str) -> bool:
    operation = str(record.get("operation", record.get("action", record.get("tool", "")))).lower()
    if kind not in _EDIT_KINDS and not (
        kind == "action_outcome_closed" and operation in {"edit", "str_replace"}
    ):
        return False
    accepted = record.get("accepted")
    if accepted is not None:
        return accepted is True
    status = str(record.get("status", record.get("outcome", ""))).upper()
    if status:
        return status in _SUCCESS_STATUSES
    return kind in {"edit_accepted", "edit_applied"}


def _is_successful_check(record: Mapping[str, object], kind: str) -> bool:
    operation = str(record.get("operation", record.get("action", record.get("tool", "")))).lower()
    if kind not in _CHECK_KINDS and not (
        kind == "action_outcome_closed" and operation in {"check", "test", "run_tests"}
    ):
        return False
    if record.get("passed") is not None:
        return record.get("passed") is True
    status = str(record.get("status", record.get("outcome", ""))).upper()
    if status:
        return status in {"PASS", "PASSED", "OK", "GREEN"}
    return kind in {"check_passed", "test_passed"}


def _event_id(record: Mapping[str, object]) -> str:
    value = record.get("event_id", record.get("sequence"))
    if value not in (None, ""):
        return str(value)
    return _canonical_digest(record)


def join_labels(
    evidence_items: Sequence[Mapping[str, object]],
    journal_records: Iterable[Mapping[str, object]],
    *,
    run_id: str | None = None,
    step: int | None = None,
) -> LabelJoin:
    """Join host outcome labels to items by receipt ID only.

    A positive accepted-edit or passing-check event makes the outcome available
    for the scoped candidate set.  Candidate receipts cited by that event are
    ``True`` and the other candidate receipts are ``False``.  Without such an
    outcome all labels remain ``None`` rather than pretending that absence is a
    negative observation.
    """

    item_receipts = tuple(_receipt_id(item) for item in evidence_items)
    candidate_receipts = set(item_receipts)
    positive: set[str] = set()
    sources: dict[str, list[str]] = {}
    label_kinds: dict[str, list[str]] = {}
    source_timestamps: dict[str, str | int | float] = {}
    source_event_ids: list[str] = []
    scoped_outcome_seen = False
    for raw_record in journal_records:
        if not isinstance(raw_record, Mapping):
            continue
        record = _label_record(raw_record)
        if not _event_applies(record, run_id=run_id, step=step):
            continue
        kind = _event_kind(record)
        successful_edit = _is_successful_edit(record, kind)
        successful_check = _is_successful_check(record, kind)
        if not (successful_edit or successful_check):
            continue
        scoped_outcome_seen = True
        cited = tuple(receipt for receipt in _citation_ids(record) if receipt in candidate_receipts)
        if not cited:
            continue
        event_id = _event_id(record)
        source_event_ids.append(event_id)
        source_kind = "accepted_edit" if successful_edit else "passing_check"
        timestamp = record.get("timestamp_ns", event_id)
        for receipt in cited:
            positive.add(receipt)
            sources.setdefault(receipt, []).append(event_id)
            label_kinds.setdefault(receipt, []).append(source_kind)
            if isinstance(timestamp, (str, int, float)) and not isinstance(timestamp, bool):
                source_timestamps.setdefault(receipt, timestamp)
    outcome_available = bool(source_event_ids)
    outcome_state = (
        "LABELLED" if outcome_available else "CENSORED" if scoped_outcome_seen else "MISSING"
    )
    labels = {
        receipt: (receipt in positive if outcome_available else None)
        for receipt in dict.fromkeys(item_receipts)
    }
    return LabelJoin(
        outcome_available=outcome_available,
        labels=labels,
        sources={key: tuple(value) for key, value in sources.items()},
        label_kinds={key: tuple(dict.fromkeys(value)) for key, value in label_kinds.items()},
        source_timestamps=source_timestamps,
        positive_receipt_ids=tuple(sorted(positive)),
        source_event_ids=tuple(dict.fromkeys(source_event_ids)),
        outcome_state=outcome_state,
    )


def load_journal_records(path: str | Path) -> tuple[dict[str, object], ...]:
    """Load a bounded JSONL fixture or journal without importing journal owners."""

    records: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except (UnicodeError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid journal JSON at line {line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"journal line {line_number} must be an object")
            records.append(value)
    return tuple(records)


def _call_decision_api(decision_api: BooleanDecisionAPI, request: BooleanScoreRequest) -> object:
    score_boolean = getattr(decision_api, "score_boolean", None)
    if not callable(score_boolean):
        raise TypeError("decision_api must expose score_boolean(question, ..., evidence=...)")
    return score_boolean(
        request.statement,
        evidence=request.evidence,
        decision_id=request.decision_id,
        site_id=request.site_id,
        question_version=request.question_version,
    )


def _site_options(site: DecisionSite) -> tuple[dict[str, object], ...]:
    return tuple(option.to_dict() for option in site.options)


def _journal_record_payload(
    score: Mapping[str, object],
    request: BooleanScoreRequest,
    site: DecisionSite,
    *,
    raw_p: Mapping[str, float],
    calibrated_p: Mapping[str, float] | None,
    legal_mass: float | None,
    temperature: float | None,
    threshold: float | None,
    disposition: str,
    answerability: str,
    admission_status: str,
    admission_receipt_id: str,
    response_receipt_id: str,
    consumption_receipt_id: str,
    response_received: bool,
    cache_hit: bool,
    timeout: bool,
    observed_usage: int | None,
    usage_status: str,
) -> dict[str, object]:
    options = score.get("options")
    if not isinstance(options, Sequence) or isinstance(options, (str, bytes, bytearray)):
        options = _site_options(site)
    return {
        "decision_id": request.decision_id,
        "site_id": request.site_id,
        "question_version": request.question_version,
        "type": "BOOLEAN",
        "options": [dict(option) for option in options if isinstance(option, Mapping)],
        "evidence_handles": [dict(handle) for handle in request.evidence_handles],
        "served_model_identity": score.get("served_model_identity"),
        "server_build": score.get("server_build"),
        "raw_top_logprobs": _clone(score.get("raw_top_logprobs", ())),
        "legal_mass": legal_mass,
        "raw_p": dict(raw_p),
        "calibrated_p": dict(calibrated_p or {}),
        "T_site": temperature,
        "tau_site": threshold,
        "disposition": disposition,
        "applied": False,
        "cost": _clone(score.get("cost", {})),
        "run_id": request.run_id,
        "step": request.step,
        "candidate_set_id": request.candidate_set_id,
        "candidate_ids": list(request.candidate_ids),
        "menu_option_ids": list(request.menu_option_ids),
        "answerability": answerability,
        "admission_status": admission_status,
        "admission_receipt_id": admission_receipt_id,
        "response_receipt_id": response_receipt_id,
        "consumption_receipt_id": consumption_receipt_id,
        "response_received": response_received,
        "cache_hit": cache_hit,
        "timeout": timeout,
        "observed_usage": observed_usage,
        "usage_status": usage_status,
        "shadow": True,
    }


def _emit_shadow_events(
    journal: object,
    payload: Mapping[str, object],
    *,
    label: bool | None,
    label_kinds: Sequence[str],
    source_event_ids: Sequence[str],
    labelled_at: str | int | float | None,
    receipt_id: str,
) -> tuple[dict[str, object], ...]:
    """Emit through the registry adapter; never write around the journal owner."""

    emitted: list[dict[str, object]] = [
        emit_decision_recorded(journal, dict(payload)),
    ]
    if label is not None:
        emitted.append(
            emit_decision_labelled(
                journal,
                {
                    "decision_id": payload["decision_id"],
                    "label": "true" if label else "false",
                    "label_source": label_kinds[0] if label_kinds else "accepted_edit_or_passing_check",
                    "labelled_at": labelled_at if labelled_at is not None else (source_event_ids[0] if source_event_ids else "unknown"),
                    "candidate_receipt_id": receipt_id,
                    "source_event_ids": list(source_event_ids),
                    "shadow": True,
                },
            )
        )
    return tuple(emitted)


def score_evidence_relevance(
    rendered_frame: object,
    evidence_items: Sequence[Mapping[str, object]],
    *,
    decision_api: BooleanDecisionAPI,
    site_registry: DecisionSiteRegistry,
    run_id: str,
    step: int,
    frame_context: Mapping[str, object] | None = None,
    journal_records: Iterable[Mapping[str, object]] = (),
    journal: object | None = None,
    resource_account: ShadowResourceAccount | None = None,
    calibration: CalibrationAPI | None = None,
    site_id: str = SITE_ID,
    question_version: str | None = None,
) -> ShadowEvaluation:
    """Score selected evidence in shadow and return the untouched frame.

    The scorer receives the selector's public items in their existing order,
    capped by the registered site's ``candidate_top_k``.  It never receives
    ``rendered_frame`` or caller-owned item mappings. One explicit shadow
    reservation covers the entire bounded candidate batch before any score
    call. Per-item API and journal failures become observational errors so an
    advisory lane cannot interrupt or alter frame rendering.
    """

    run_id = _text(run_id, "run_id")
    site_id = _text(site_id, "site_id")
    step = _step(step)
    if not isinstance(site_registry, DecisionSiteRegistry):
        raise TypeError("site_registry must be a DecisionSiteRegistry")
    site = site_registry.require(site_id)
    if site.question_type is not QuestionType.BOOLEAN:
        raise ValueError("evidence relevance site must be BOOLEAN")
    if question_version is not None and _text(question_version, "question_version") != site.question_version:
        raise ValueError("question_version does not match the registered site")
    question_version = site.question_version
    if not isinstance(evidence_items, Sequence) or isinstance(
        evidence_items, (str, bytes, bytearray)
    ):
        raise TypeError("evidence_items must be a sequence of mappings")
    if frame_context is not None and not isinstance(frame_context, Mapping):
        raise TypeError("frame_context must be a mapping or None")
    copied_items: list[dict[str, object]] = []
    for item in evidence_items:
        if not isinstance(item, Mapping):
            raise TypeError("evidence_items entries must be mappings")
        copied_items.append(dict(_clone(item)))
    all_items = tuple(copied_items)
    items = all_items[: site.candidate_top_k]
    for item in items:
        _item_id(item)
        _receipt_id(item)
    omitted_ids = tuple(_item_id(item) for item in all_items[site.candidate_top_k:])
    candidate_ids = tuple(_item_id(item) for item in items)
    candidate_receipt_ids = tuple(_receipt_id(item) for item in items)
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("registered candidate set contains duplicate item ids")
    if len(set(candidate_receipt_ids)) != len(candidate_receipt_ids):
        raise ValueError("registered candidate set contains duplicate receipt ids")
    candidate_set_id = _candidate_set_id(site_id, run_id, step, items)
    menu_option_ids = tuple(option.option_id for option in site.options)
    admission_receipt_id = _receipt_identity(
        "admission",
        {"candidate_set_id": candidate_set_id, "units": len(items)},
    )
    consumption_receipt_id = _receipt_identity(
        "consumption",
        {"candidate_set_id": candidate_set_id, "units": len(items)},
    )
    if items:
        reservation, admission_error = _try_reserve_shadow(
            resource_account,
            units=len(items),
            reservation_key=candidate_set_id,
            site_id=site_id,
            candidate_set_id=candidate_set_id,
        )
    else:
        reservation, admission_error = None, None
    admission_status = (
        "NOT_REQUESTED" if not items else
        "ADMITTED" if reservation is not None else
        "REFUSED" if admission_error == "SHADOW_ADMISSION_REFUSED" else
        "UNAVAILABLE"
    )
    journal_snapshot = tuple(journal_records)
    labels = join_labels(items, journal_snapshot, run_id=run_id, step=step)
    context = _public_copy(frame_context or {})
    if not isinstance(context, Mapping):
        context = {}
    records: list[EvidenceRelevanceRecord] = []
    emitted_events: list[Mapping[str, object]] = []
    journal_errors: list[str] = []
    if journal is not None:
        try:
            emitted_events.append(
                _emit_shadow_receipt(
                    journal,
                    SHADOW_ADMISSION_EVENT,
                    {
                        "schema": "sys1-shadow-admission-receipt/1",
                        "receipt_id": admission_receipt_id,
                        "site_id": site_id,
                        "question_version": question_version,
                        "candidate_set_id": candidate_set_id,
                        "requested_items": len(items),
                        "admitted_items": len(items) if reservation is not None else 0,
                        "admission_status": admission_status,
                        "reservation_id": reservation.reservation_id if reservation else None,
                        "resource_account_id": reservation.resource_account_id if reservation else None,
                        "endpoint_id": reservation.endpoint_id if reservation else None,
                        "budget_scope": SHADOW_BUDGET_SCOPE,
                        "primary_solver_units": 0,
                        "reason": admission_error,
                        "shadow": True,
                        "applied": False,
                    },
                )
            )
        except Exception as error:  # journal failure stays observational
            journal_errors.append(type(error).__name__)
    for candidate_rank, item in enumerate(items, start=1):
        item_id = _item_id(item)
        receipt_id = _receipt_id(item)
        evidence = _evidence_view(item, item_id, receipt_id)
        request = BooleanScoreRequest(
            decision_id=_decision_id(site_id, run_id, step, item_id, receipt_id),
            site_id=site_id,
            question_version=question_version,
            statement=f"Evidence item {item_id} is relevant to the current frame decision.",
            run_id=run_id,
            step=step,
            frame_context=_clone(context),
            evidence=(evidence,),
            evidence_handles=(
                {
                    "receipt_id": receipt_id,
                    "kind": str(item.get("kind", "evidence")),
                    "digest": _canonical_digest(evidence),
                },
            ),
            candidate_set_id=candidate_set_id,
            candidate_ids=candidate_ids,
            menu_option_ids=menu_option_ids,
        )
        raw_p: dict[str, float] = {}
        calibrated_p: dict[str, float] | None = None
        legal_mass: float | None = None
        temperature: float | None = None
        threshold: float | None = None
        disposition = UNAVAILABLE
        answerability = UNAVAILABLE
        score: dict[str, object] | None = None
        error_type: str | None = None
        response_received = False
        cache_hit = False
        timeout = False
        observed_usage: int | None = None
        usage_status = _USAGE_NOT_OBSERVED
        if reservation is None:
            status = UNAVAILABLE
            answerability = UNAVAILABLE
            error_type = admission_error or "SHADOW_ADMISSION_REFUSED"
        else:
            try:
                raw_response = _call_decision_api(decision_api, request)
                response_received = True
                score = _score_mapping(raw_response)
                timeout = _timeout_observed(score)
                response_received = not timeout
                cache_hit = _explicit_bool(score, ("cache_hit", "cached"))
                observed_usage, usage_status = _observed_usage(
                    score, response_received=response_received
                )
                if timeout:
                    raise TimeoutError("shadow score response timed out")
                raw_p = _probabilities(score)
                legal_mass = _legal_mass(score)
                selected = _selected_option(score, raw_p)
                answerability = _answerability(score, raw_p)
                disposition = str(score.get("disposition", "UNCALIBRATED_PROPOSE"))
                if calibration is not None:
                    if getattr(calibration, "site_id", site_id) != site_id:
                        raise ValueError("calibration site_id does not match registered site")
                    calibrated_p, temperature, threshold, disposition = _calibrated(
                        calibration,
                        raw_p,
                        selected=selected,
                        legal_mass=legal_mass,
                    )
                    if answerability == INSUFFICIENT:
                        disposition = str(score.get("disposition", disposition))
                else:
                    calibrated_p = dict(raw_p) if raw_p else None
                    disposition = (
                        "UNCALIBRATED_PROPOSE"
                        if answerability == ANSWERABLE
                        else str(score.get("disposition", INSUFFICIENT))
                    )
                status = "SCORED" if answerability == ANSWERABLE else INSUFFICIENT
            except Exception as error:  # shadow must not become a live-path failure
                status = UNAVAILABLE
                answerability = UNAVAILABLE
                if isinstance(error, TimeoutError):
                    timeout = True
                    response_received = False
                    observed_usage = None
                    usage_status = _USAGE_NOT_OBSERVED
                error_type = type(error).__name__
        record = EvidenceRelevanceRecord(
            decision_id=request.decision_id,
            site_id=site_id,
            question_version=question_version,
            item_id=item_id,
            receipt_id=receipt_id,
            run_id=run_id,
            step=step,
            candidate_set_id=candidate_set_id,
            candidate_rank=candidate_rank,
            menu_option_ids=menu_option_ids,
            status=status,
            answerability=answerability,
            decision_record=score,
            raw_p=raw_p,
            calibrated_p=calibrated_p,
            legal_mass=legal_mass,
            temperature=temperature,
            threshold=threshold,
            disposition=disposition,
            label=labels.labels.get(receipt_id),
            label_sources=labels.sources.get(receipt_id, ()),
            label_source_kinds=labels.label_kinds.get(receipt_id, ()),
            request=request,
            error_type=error_type,
            response_received=response_received,
            cache_hit=cache_hit,
            timeout=timeout,
            observed_usage=observed_usage,
            usage_status=usage_status,
            admission_receipt_id=admission_receipt_id,
            response_receipt_id=_receipt_identity(
                "response", {"decision_id": request.decision_id}
            ),
            consumption_receipt_id=consumption_receipt_id,
        )
        records.append(record)
        if journal is not None:
            response_receipt_id = record.response_receipt_id
            payload = _journal_record_payload(
                score or {},
                request,
                site,
                raw_p=raw_p,
                calibrated_p=calibrated_p,
                legal_mass=legal_mass,
                temperature=temperature,
                threshold=threshold,
                disposition=disposition,
                answerability=answerability,
                admission_status=admission_status,
                admission_receipt_id=admission_receipt_id,
                response_receipt_id=response_receipt_id,
                consumption_receipt_id=consumption_receipt_id,
                response_received=response_received,
                cache_hit=cache_hit,
                timeout=timeout,
                observed_usage=observed_usage,
                usage_status=usage_status,
            )
            try:
                emitted_events.extend(
                    _emit_shadow_events(
                        journal,
                        payload,
                        label=record.label,
                        label_kinds=record.label_source_kinds,
                        source_event_ids=record.label_sources,
                        labelled_at=labels.source_timestamps.get(receipt_id),
                        receipt_id=receipt_id,
                    )
                )
                emitted_events.append(
                    _emit_shadow_receipt(
                        journal,
                        SHADOW_RESPONSE_EVENT,
                        {
                            "schema": "sys1-shadow-response-receipt/1",
                            "receipt_id": response_receipt_id,
                            "decision_id": request.decision_id,
                            "site_id": site_id,
                            "candidate_set_id": candidate_set_id,
                            "admission_receipt_id": admission_receipt_id,
                            "response_received": response_received,
                            "cache_hit": cache_hit,
                            "timeout": timeout,
                            "observed_usage": observed_usage,
                            "usage_status": usage_status,
                            "error_type": error_type,
                            "shadow": True,
                            "applied": False,
                        },
                    )
                )
            except Exception as error:  # journal failure stays observational
                journal_errors.append(type(error).__name__)
    cache_hits = sum(1 for record in records if record.cache_hit)
    timeouts = sum(1 for record in records if record.timeout)
    admitted_records = records if reservation is not None else []
    usage_values = [record.observed_usage for record in admitted_records]
    usage_status = (
        _USAGE_NOT_OBSERVED
        if not admitted_records
        else _USAGE_KNOWN
        if all(record.response_received and value is not None for record, value in zip(admitted_records, usage_values))
        else _USAGE_UNKNOWN
    )
    observed_usage = sum(value for value in usage_values if value is not None) if usage_status == _USAGE_KNOWN else None
    settled = False
    if reservation is not None:
        settle = getattr(resource_account, "settle_shadow", None)
        if not callable(settle):
            journal_errors.append("SHADOW_RESOURCE_ACCOUNT_INVALID")
        else:
            try:
                settle(
                    reservation,
                    observed_usage=observed_usage,
                    cache_hits=cache_hits,
                    timeouts=timeouts,
                )
                settled = True
            except Exception as error:  # shadow settlement cannot change the frame
                journal_errors.append(type(error).__name__)
    accounting = ShadowAccounting(
        reservation_id=reservation.reservation_id if reservation else None,
        resource_account_id=reservation.resource_account_id if reservation else None,
        endpoint_id=reservation.endpoint_id if reservation else None,
        budget_scope=SHADOW_BUDGET_SCOPE,
        requested_items=len(items),
        admitted_items=len(items) if reservation else 0,
        admission_status=admission_status,
        cache_hits=cache_hits,
        timeouts=timeouts,
        observed_usage=observed_usage,
        usage_status=usage_status,
        settled=settled,
        primary_solver_units=0,
        admission_receipt_id=admission_receipt_id,
        response_receipt_ids=tuple(record.response_receipt_id for record in records),
        consumption_receipt_id=consumption_receipt_id,
    )
    if journal is not None:
        try:
            emitted_events.append(
                _emit_shadow_receipt(
                    journal,
                    SHADOW_CONSUMPTION_EVENT,
                    {
                        "schema": "sys1-shadow-consumption-receipt/1",
                        "receipt_id": consumption_receipt_id,
                        "site_id": site_id,
                        "candidate_set_id": candidate_set_id,
                        "admission_receipt_id": admission_receipt_id,
                        "response_receipt_ids": list(accounting.response_receipt_ids),
                        "requested_items": accounting.requested_items,
                        "admitted_items": accounting.admitted_items,
                        "cache_hits": accounting.cache_hits,
                        "timeouts": accounting.timeouts,
                        "observed_usage": accounting.observed_usage,
                        "usage_status": accounting.usage_status,
                        "settled": accounting.settled,
                        "consumed": False,
                        "applied": False,
                        "budget_scope": SHADOW_BUDGET_SCOPE,
                        "primary_solver_units": 0,
                        "shadow": True,
                    },
                )
            )
        except Exception as error:  # journal failure stays observational
            journal_errors.append(type(error).__name__)
    return ShadowEvaluation(
        rendered_frame=rendered_frame,
        records=tuple(records),
        labels=labels,
        candidate_set_id=candidate_set_id,
        candidate_ids=candidate_ids,
        candidate_receipt_ids=candidate_receipt_ids,
        menu_option_ids=menu_option_ids,
        accounting=accounting,
        omitted_candidate_ids=omitted_ids,
        journal_events=tuple(emitted_events),
        journal_errors=tuple(journal_errors),
    )


__all__ = [
    "ANSWERABLE",
    "BooleanDecisionAPI",
    "BooleanScoreRequest",
    "CalibrationAPI",
    "DEFAULT_CANDIDATE_TOP_K",
    "EvidenceRelevanceRecord",
    "INSUFFICIENT",
    "LabelJoin",
    "QUESTION_VERSION",
    "SCHEMA",
    "SITE_ID",
    "SHADOW_BUDGET_SCOPE",
    "SHADOW_ADMISSION_EVENT",
    "SHADOW_RESPONSE_EVENT",
    "SHADOW_CONSUMPTION_EVENT",
    "ShadowAccounting",
    "ShadowEvaluation",
    "ShadowReservation",
    "ShadowResourceAccount",
    "UNAVAILABLE",
    "evidence_relevance_registry",
    "evidence_relevance_site",
    "join_labels",
    "load_journal_records",
    "score_evidence_relevance",
]
