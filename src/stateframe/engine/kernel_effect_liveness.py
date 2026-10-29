"""Host-owned effect liveness evidence.

This module is deliberately an observation kernel, not a model guidance
scaffold.  It does not infer an effect from a registration, a job id, a
non-empty response, or a model assertion.  A positive control and a negative
control both need an independently observed, correlated receipt before an
obligation can be assessed as true.

The public functions are usable by a thin domain pack.  The runtime only
loads this module when ``STATEFRAME_EFFECT_LIVENESS=on`` so the default path
does not add fields, events, or serialization output.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field


MODE_ENV = "STATEFRAME_EFFECT_LIVENESS"
SCHEMA = "stateframe-effect-liveness/1"
OBLIGATION_SCHEMA = "stateframe-effect-obligation/1"
EXPERIMENT_SCHEMA = "stateframe-effect-experiment/1"
RECEIPT_SCHEMA = "stateframe-effect-receipt/1"
ASSESSMENT_SCHEMA = "stateframe-effect-assessment/1"

OBSERVATION_ONLY = "OBSERVATION_ONLY"
TRUE = "TRUE"
FALSE = "FALSE"
UNKNOWN = "UNKNOWN"
INVALID = "INVALID"
NOT_APPLICABLE = "NOT_APPLICABLE"

APPLIES = "APPLIES"
DOES_NOT_APPLY = "DOES_NOT_APPLY"
APPLICABILITY_UNKNOWN = "UNKNOWN"

POSITIVE = "positive"
NEGATIVE = "negative"
CONTROLS = (POSITIVE, NEGATIVE)

_UNKNOWN = "UNKNOWN"
_MISSING = object()


def mode() -> str:
    """Return the strict opt-in mode for the effect probe."""
    value = os.environ.get(MODE_ENV, "off")
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str).encode("utf-8")


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _copy_json(value: object) -> object:
    """Copy a receipt field while keeping record serialization deterministic."""
    try:
        return json.loads(_canonical(value).decode("utf-8"))
    except (TypeError, ValueError, UnicodeError):
        return str(value)


def _first(mapping: Mapping[str, object], *names: str, default: object = _MISSING) -> object:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _optional_bool(value: object, name: str) -> bool | None:
    if value is None or value is _MISSING:
        return None
    if type(value) is not bool:
        raise TypeError(f"{name} must be bool or None")
    return value


def _optional_count(value: object, name: str) -> int | None:
    if value is None or value is _MISSING:
        return None
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer or None")
    return value


def _result_count(mapping: Mapping[str, object]) -> int | None:
    """Normalize optional cardinality evidence without treating it as effect proof."""
    for name in (
        "result_count", "result_rows", "row_count", "match_count",
        "effect_count", "matches_count",
    ):
        if name in mapping:
            return _optional_count(mapping[name], name)
    for name in ("rows", "results", "matches"):
        value = mapping.get(name, _MISSING)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return len(value)
    return None


def _known(value: object) -> bool:
    if value is None or value is _MISSING:
        return False
    if isinstance(value, str) and (not value.strip() or value.strip().upper() in {"UNKNOWN", "MISSING"}):
        return False
    return True


def _identity(value: object) -> str:
    if not _known(value):
        return _UNKNOWN
    if isinstance(value, str):
        return value
    return _digest(value)


def _invoke(function: Callable[..., object], *args: object) -> object:
    """Call a pack callback without guessing from its returned value."""
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(*args)
    positional = [parameter for parameter in signature.parameters.values()
                  if parameter.kind in (parameter.POSITIONAL_ONLY,
                                        parameter.POSITIONAL_OR_KEYWORD)]
    has_varargs = any(parameter.kind == parameter.VAR_POSITIONAL
                      for parameter in signature.parameters.values())
    if has_varargs or len(positional) >= len(args):
        return function(*args)
    if positional:
        return function(*args[:len(positional)])
    return function()


def _emit(journal: object | None, event: str, **fields: object) -> object | None:
    """Emit only payload fields; journal-owned reserved fields stay reserved."""
    emitter = getattr(journal, "emit", None)
    if callable(emitter):
        return emitter(event, **fields)
    return None


def inspect_runtime_binding(
    binding: Mapping[str, object] | None = None,
    *,
    process: object = _MISSING,
    loaded_artifact: object = _MISSING,
    config: object = _MISSING,
    dependency_snapshot: object = _MISSING,
) -> dict[str, object]:
    """Normalize the host binding needed to interpret an effect receipt.

    These fields are receipts about which candidate ran.  They are not claims
    that the candidate is correct.  An empty dependency mapping is a known
    snapshot; an omitted snapshot is not.
    """
    raw = dict(binding) if isinstance(binding, Mapping) else {}
    candidate_value = _first(raw, "candidate_digest", "candidate", "artifact_digest",
                             "artifact", "revision", "product_revision", default=_MISSING)
    process_value = process if process is not _MISSING else _first(
        raw, "exercised_process", "process", "worker", "pid", default=_MISSING)
    loaded_value = loaded_artifact if loaded_artifact is not _MISSING else _first(
        raw, "loaded_artifact", "loaded", "loaded_revision", default=_MISSING)
    config_value = config if config is not _MISSING else _first(
        raw, "config_digest", "config", "configuration", default=_MISSING)
    dependencies = (dependency_snapshot if dependency_snapshot is not _MISSING
                    else _first(raw, "dependency_snapshot", "dependencies", default=_MISSING))

    normalized = {
        "candidate_digest": _identity(candidate_value),
        "exercised_process": str(process_value) if _known(process_value) else _UNKNOWN,
        "loaded_artifact": _copy_json(loaded_value) if _known(loaded_value) else _UNKNOWN,
        "config_digest": _identity(config_value),
        "dependency_snapshot": _copy_json(dependencies) if _known(dependencies) else _UNKNOWN,
    }
    normalized["binding_digest"] = _digest(normalized)
    return normalized


def _merge_binding(mapping: Mapping[str, object], binding: Mapping[str, object] | None) -> dict[str, object]:
    result = inspect_runtime_binding(binding)
    receipt_binding = _first(mapping, "binding", default=_MISSING)
    if isinstance(receipt_binding, Mapping):
        result = inspect_runtime_binding({**result, **dict(receipt_binding)})
    # Explicit receipt fields are authoritative observations, but never infer
    # one ladder level from another.
    result = inspect_runtime_binding({
        **result,
        **{name: mapping[name] for name in (
            "candidate_digest", "exercised_process", "loaded_artifact",
            "config_digest", "dependency_snapshot",
        ) if name in mapping},
    })
    return result


@dataclass(frozen=True, slots=True)
class EffectObligation:
    """A host-admitted obligation whose meaning is an externally visible effect."""

    obligation_id: str
    effect_kind: str
    description: str
    observer: str
    positive_stimulus: object
    negative_stimulus: object
    deadline_s: float | None = None
    applicability: str = APPLIES
    source: str = "domain_pack"
    authority: str = OBSERVATION_ONLY

    def __post_init__(self) -> None:
        object.__setattr__(self, "obligation_id", _required_text(self.obligation_id, "obligation_id"))
        object.__setattr__(self, "effect_kind", _required_text(self.effect_kind, "effect_kind"))
        object.__setattr__(self, "description", _required_text(self.description, "description"))
        object.__setattr__(self, "observer", _required_text(self.observer, "observer"))
        if self.applicability not in {APPLIES, DOES_NOT_APPLY, APPLICABILITY_UNKNOWN}:
            raise ValueError("applicability must be APPLIES, DOES_NOT_APPLY, or UNKNOWN")
        if self.deadline_s is not None and (type(self.deadline_s) not in {int, float} or self.deadline_s < 0):
            raise ValueError("deadline_s must be a nonnegative number or None")
        if self.authority != OBSERVATION_ONLY:
            raise ValueError("effect obligations are observation-only records")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "EffectObligation":
        if not isinstance(value, Mapping):
            raise TypeError("effect obligation must be a mapping")
        positive = _first(value, "positive_stimulus", "positive", "stimulus", default={})
        negative = _first(value, "negative_stimulus", "negative", default={})
        return cls(
            obligation_id=_first(value, "obligation_id", "id", default=""),
            effect_kind=_first(value, "effect_kind", "effect", "kind", default=""),
            description=_first(value, "description", "clause", default="effect obligation"),
            observer=_first(value, "observer", "readback", default="independent observer"),
            positive_stimulus=_copy_json(positive),
            negative_stimulus=_copy_json(negative),
            deadline_s=_first(value, "deadline_s", "deadline", default=None),
            applicability=_first(value, "applicability", default=APPLIES),
            source=_first(value, "source", default="domain_pack"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": OBLIGATION_SCHEMA,
            "record_type": "obligation",
            "authority": self.authority,
            "obligation_id": self.obligation_id,
            "effect_kind": self.effect_kind,
            "description": self.description,
            "observer": self.observer,
            "positive_stimulus": _copy_json(self.positive_stimulus),
            "negative_stimulus": _copy_json(self.negative_stimulus),
            "deadline_s": self.deadline_s,
            "applicability": self.applicability,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class EffectExperiment:
    experiment_id: str
    obligation_id: str
    control: str
    stimulus: object
    stimulus_correlation_id: str
    authority: str = OBSERVATION_ONLY
    eligible: bool | None = None
    executed: bool | None = None
    status: str = "CREATED"
    substitutions: tuple[str, ...] = ()
    detail: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": EXPERIMENT_SCHEMA,
            "record_type": "experiment",
            "authority": self.authority,
            "experiment_id": self.experiment_id,
            "obligation_id": self.obligation_id,
            "control": self.control,
            "stimulus": _copy_json(self.stimulus),
            "stimulus_correlation_id": self.stimulus_correlation_id,
            "eligible": self.eligible,
            "executed": self.executed,
            "status": self.status,
            "substitutions": list(self.substitutions),
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class EffectReceipt:
    receipt_id: str
    experiment_id: str
    obligation_id: str
    control: str
    candidate_digest: str
    exercised_process: str
    loaded_artifact: object
    config_digest: str
    dependency_snapshot: object
    stimulus_correlation_id: str
    observer_completeness: bool | None = None
    substitutions: tuple[str, ...] = ()
    eligible: bool | None = None
    executed: bool | None = None
    effect_observed: bool | None = None
    result_count: int | None = None
    deadline_reached: bool | None = None
    execution_status: str = "UNKNOWN"
    detail: str = ""
    authority: str = OBSERVATION_ONLY
    observed_at_ns: int | None = None

    def __post_init__(self) -> None:
        for name in ("receipt_id", "experiment_id", "obligation_id", "stimulus_correlation_id"):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        if self.control not in CONTROLS:
            raise ValueError("control must be positive or negative")
        if self.authority != OBSERVATION_ONLY:
            raise ValueError("effect receipts are observation-only records")
        for name in ("observer_completeness", "eligible", "executed",
                     "effect_observed", "deadline_reached"):
            value = getattr(self, name)
            if value is not None and type(value) is not bool:
                raise TypeError(f"{name} must be bool or None")
        if self.result_count is not None:
            _optional_count(self.result_count, "result_count")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object], *, binding: Mapping[str, object] | None = None,
                     default_obligation_id: str | None = None) -> "EffectReceipt":
        if not isinstance(value, Mapping):
            raise TypeError("effect receipt must be a mapping")
        normalized = _merge_binding(value, binding)
        effect_value = _first(value, "effect_observed", "effect_present", "observed", default=None)
        return cls(
            receipt_id=_first(value, "receipt_id", "id", default="receipt-" + uuid.uuid4().hex),
            experiment_id=_first(value, "experiment_id", "experiment", default="experiment-" + uuid.uuid4().hex),
            obligation_id=_first(value, "obligation_id", "obligation", default=default_obligation_id or ""),
            control=_first(value, "control", "arm", default="positive"),
            candidate_digest=normalized["candidate_digest"],
            exercised_process=normalized["exercised_process"],
            loaded_artifact=normalized["loaded_artifact"],
            config_digest=normalized["config_digest"],
            dependency_snapshot=normalized["dependency_snapshot"],
            stimulus_correlation_id=_first(value, "stimulus_correlation_id", "correlation_id",
                                            "correlation", default="correlation-" + uuid.uuid4().hex),
            observer_completeness=_optional_bool(_first(
                value, "observer_completeness", "observer_complete", "observer_sufficient", default=None),
                "observer_completeness"),
            substitutions=tuple(str(item) for item in (_first(value, "substitutions", default=()) or ())),
            eligible=_optional_bool(_first(value, "eligible", default=None), "eligible"),
            executed=_optional_bool(_first(value, "executed", "invoked", default=None), "executed"),
            effect_observed=_optional_bool(effect_value, "effect_observed"),
            result_count=_result_count(value),
            deadline_reached=_optional_bool(
                _first(value, "deadline_reached", "authorized_deadline", default=None),
                "deadline_reached"),
            execution_status=str(_first(value, "execution_status", "status", default="UNKNOWN")),
            detail=str(_first(value, "detail", "observation_detail", default="")),
            observed_at_ns=_first(value, "observed_at_ns", default=None),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": RECEIPT_SCHEMA,
            "record_type": "receipt",
            "authority": self.authority,
            "receipt_id": self.receipt_id,
            "experiment_id": self.experiment_id,
            "obligation_id": self.obligation_id,
            "control": self.control,
            "candidate_digest": self.candidate_digest,
            "exercised_process": self.exercised_process,
            "loaded_artifact": _copy_json(self.loaded_artifact),
            "config_digest": self.config_digest,
            "dependency_snapshot": _copy_json(self.dependency_snapshot),
            "stimulus_correlation_id": self.stimulus_correlation_id,
            "observer_completeness": self.observer_completeness,
            "substitutions": list(self.substitutions),
            "eligible": self.eligible,
            "executed": self.executed,
            "effect_observed": self.effect_observed,
            "result_count": self.result_count,
            "deadline_reached": self.deadline_reached,
            "execution_status": self.execution_status,
            "detail": self.detail,
            "observed_at_ns": self.observed_at_ns,
        }


@dataclass(frozen=True, slots=True)
class EffectAssessment:
    obligation_id: str
    applicability: str
    predicate: str
    support: dict[str, object] = field(default_factory=dict)
    reason: str = ""
    receipt_ids: tuple[str, ...] = ()
    positive_receipt_id: str | None = None
    negative_receipt_id: str | None = None
    authority: str = OBSERVATION_ONLY

    @property
    def blocking(self) -> bool:
        # Unknown applicability is itself an evidence gap.  Only an explicit
        # DOES_NOT_APPLY disposition can discharge the obligation without
        # effect evidence.
        return self.applicability != DOES_NOT_APPLY and self.predicate != TRUE

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": ASSESSMENT_SCHEMA,
            "record_type": "assessment",
            "authority": self.authority,
            "obligation_id": self.obligation_id,
            "applicability": self.applicability,
            "predicate": self.predicate,
            "support": _copy_json(self.support),
            "reason": self.reason,
            "receipt_ids": list(self.receipt_ids),
            "positive_receipt_id": self.positive_receipt_id,
            "negative_receipt_id": self.negative_receipt_id,
        }


def _coerce_obligation(value: EffectObligation | Mapping[str, object]) -> EffectObligation:
    return value if isinstance(value, EffectObligation) else EffectObligation.from_mapping(value)


def _coerce_receipt(value: EffectReceipt | Mapping[str, object], *, binding: Mapping[str, object] | None = None,
                    default_obligation_id: str | None = None) -> EffectReceipt:
    return value if isinstance(value, EffectReceipt) else EffectReceipt.from_mapping(
        value, binding=binding, default_obligation_id=default_obligation_id)


def _correlation_matches(receipt: EffectReceipt,
                         experiments: Sequence[EffectExperiment] | None) -> bool:
    """Require a receipt to match a host-recorded experiment when available."""
    if not experiments:
        return True
    return any(
        experiment.experiment_id == receipt.experiment_id
        and experiment.obligation_id == receipt.obligation_id
        and experiment.control == receipt.control
        and experiment.stimulus_correlation_id == receipt.stimulus_correlation_id
        for experiment in experiments
    )


def _binding_complete(receipt: EffectReceipt) -> bool:
    return (
        _known(receipt.candidate_digest)
        and _known(receipt.exercised_process)
        and _known(receipt.loaded_artifact)
        and _known(receipt.config_digest)
        and _known(receipt.dependency_snapshot)
        and _known(receipt.stimulus_correlation_id)
    )


def _binding_matches(receipt: EffectReceipt, current: Mapping[str, object] | None) -> bool:
    """Reject a receipt from a different candidate/configuration as stale."""
    if current is None:
        return True
    for name in ("candidate_digest", "exercised_process", "config_digest",
                 "dependency_snapshot"):
        expected = current.get(name, _UNKNOWN)
        observed = getattr(receipt, name)
        if _known(expected) and _known(observed) and expected != observed:
            return False
    expected_loaded = current.get("loaded_artifact", _UNKNOWN)
    if _known(expected_loaded) and _known(receipt.loaded_artifact):
        if _canonical(expected_loaded) != _canonical(receipt.loaded_artifact):
            return False
    return True


def _ladder(receipt: EffectReceipt, *, deadline_reached: bool | None) -> dict[str, object]:
    deadline = receipt.deadline_reached if receipt.deadline_reached is not None else deadline_reached
    return {
        "built": _known(receipt.candidate_digest),
        "loaded_routed": _known(receipt.exercised_process) and _known(receipt.loaded_artifact),
        "eligible": receipt.eligible,
        "executed": receipt.executed,
        "effect_observed": receipt.effect_observed,
        "result_count": receipt.result_count,
        "outcome_validated": (
            receipt.effect_observed is not None
            and receipt.eligible is True
            and receipt.executed is True
            and receipt.observer_completeness is True
            and (receipt.effect_observed is True or deadline is True)
        ),
        "observer_complete": receipt.observer_completeness,
        "authorized_deadline": deadline,
    }


def _control_assessment(receipt: EffectReceipt, *, deadline_reached: bool | None) -> tuple[str, str, dict[str, object], bool]:
    ladder = _ladder(receipt, deadline_reached=deadline_reached)
    if not _binding_complete(receipt):
        return UNKNOWN, "RECEIPT_BINDING_INCOMPLETE", ladder, False
    if receipt.eligible is not True:
        return UNKNOWN, "ELIGIBILITY_NOT_ESTABLISHED", ladder, False
    if receipt.executed is not True:
        return UNKNOWN, "EXECUTION_NOT_ESTABLISHED", ladder, False
    if receipt.observer_completeness is not True:
        return UNKNOWN, "OBSERVER_INSUFFICIENT", ladder, False
    if receipt.result_count == 0:
        # A bare empty result can be a swallowed failure or an unexercised
        # path.  It is not an independently observed effect, even if a
        # producer also supplied an optimistic boolean.
        return UNKNOWN, "EFFECT_RESULT_EMPTY", ladder, False
    if receipt.effect_observed is None:
        return UNKNOWN, "EFFECT_OBSERVATION_MISSING", ladder, False
    if receipt.control == NEGATIVE and receipt.effect_observed is True:
        return FALSE, "NEGATIVE_CONTROL_EFFECT_OBSERVED", ladder, False
    if receipt.control == POSITIVE and receipt.effect_observed is False:
        authorized = (receipt.deadline_reached if receipt.deadline_reached is not None
                      else deadline_reached) is True
        if not authorized:
            return UNKNOWN, "NEGATIVE_EVIDENCE_DEADLINE_NOT_AUTHORIZED", ladder, False
        return FALSE, "POSITIVE_CONTROL_EFFECT_MISSING", ladder, True
    if receipt.control == NEGATIVE and receipt.effect_observed is False:
        authorized = (receipt.deadline_reached if receipt.deadline_reached is not None
                      else deadline_reached) is True
        if not authorized:
            return UNKNOWN, "NEGATIVE_EVIDENCE_DEADLINE_NOT_AUTHORIZED", ladder, False
        return TRUE, "NEGATIVE_CONTROL_EFFECT_ABSENT", ladder, True
    return TRUE, "POSITIVE_CONTROL_EFFECT_OBSERVED", ladder, False


def assess_effect_obligation(
    obligation: EffectObligation | Mapping[str, object],
    receipts: Sequence[EffectReceipt | Mapping[str, object]] = (),
    *,
    binding: Mapping[str, object] | None = None,
    experiments: Sequence[EffectExperiment] | None = None,
    deadline_reached: bool | None = None,
) -> EffectAssessment:
    """Assess one effect obligation without promoting incomplete evidence.

    Missing effect evidence is UNKNOWN unless eligibility, execution,
    observer completeness, and an authorized deadline make the absence
    meaningful.  This is intentionally asymmetric: an incomplete observer
    cannot establish a violation merely because it saw no effect.
    """
    item = _coerce_obligation(obligation)
    if item.applicability == DOES_NOT_APPLY:
        return EffectAssessment(item.obligation_id, item.applicability, NOT_APPLICABLE,
                                reason="OBLIGATION_NOT_APPLICABLE")
    if item.applicability != APPLIES:
        return EffectAssessment(item.obligation_id, item.applicability, UNKNOWN,
                                reason="APPLICABILITY_UNKNOWN")

    if isinstance(receipts, Mapping):
        receipts = (receipts,)
    current_binding = inspect_runtime_binding(binding) if binding is not None else None
    coerced: list[EffectReceipt] = []
    invalid: list[str] = []
    for value in receipts:
        try:
            current = _coerce_receipt(value, binding=binding, default_obligation_id=item.obligation_id)
        except (TypeError, ValueError) as error:
            invalid.append(type(error).__name__)
            continue
        if current.obligation_id == item.obligation_id:
            coerced.append(current)
    by_control: dict[str, list[EffectReceipt]] = {POSITIVE: [], NEGATIVE: []}
    for current in coerced:
        if current.control in by_control:
            by_control[current.control].append(current)
    selected: dict[str, EffectReceipt | None] = {}
    for control in CONTROLS:
        # A host-generated observation receipt is append-only.  The last one
        # for a control is the current observation, while all ids remain in
        # the assessment for auditability.
        selected[control] = by_control[control][-1] if by_control[control] else None

    support: dict[str, object] = {}
    predicates: list[str] = []
    receipt_ids: list[str] = []
    for control in CONTROLS:
        current = selected[control]
        if current is None:
            predicates.append(UNKNOWN)
            support[control] = {
                "predicate": UNKNOWN,
                "reason": "CONTROL_RECEIPT_MISSING",
                "ladder": None,
                "negative_evidence_authorized": False,
            }
            continue
        receipt_ids.append(current.receipt_id)
        if not _correlation_matches(current, experiments):
            predicate, reason, ladder, negative_authorized = (
                UNKNOWN, "RECEIPT_CORRELATION_UNAUTHENTICATED",
                _ladder(current, deadline_reached=deadline_reached), False)
        elif not _binding_matches(current, current_binding):
            predicate, reason, ladder, negative_authorized = (
                UNKNOWN, "RECEIPT_BINDING_STALE", _ladder(current, deadline_reached=deadline_reached), False)
        else:
            predicate, reason, ladder, negative_authorized = _control_assessment(
                current, deadline_reached=deadline_reached)
        predicates.append(predicate)
        support[control] = {
            "receipt_id": current.receipt_id,
            "predicate": predicate,
            "reason": reason,
            "ladder": ladder,
            "negative_evidence_authorized": negative_authorized,
        }

    if invalid:
        predicate = INVALID
        reason = "MALFORMED_RECEIPT"
    elif FALSE in predicates:
        predicate = FALSE
        reason = next((support[control]["reason"] for control in CONTROLS
                       if support[control].get("predicate") == FALSE), "EFFECT_LIVENESS_VIOLATION")
    elif all(value == TRUE for value in predicates):
        predicate = TRUE
        reason = "POSITIVE_AND_NEGATIVE_CONTROLS_OBSERVED"
    else:
        predicate = UNKNOWN
        reason = next((support[control]["reason"] for control in CONTROLS
                       if support[control].get("predicate") == UNKNOWN), "EFFECT_EVIDENCE_INCOMPLETE")
    if invalid:
        support["invalid_receipts"] = invalid
    return EffectAssessment(
        obligation_id=item.obligation_id,
        applicability=item.applicability,
        predicate=predicate,
        support=support,
        reason=reason,
        receipt_ids=tuple(receipt_ids),
        positive_receipt_id=selected[POSITIVE].receipt_id if selected[POSITIVE] else None,
        negative_receipt_id=selected[NEGATIVE].receipt_id if selected[NEGATIVE] else None,
    )


def run_behavior_experiment(
    obligation: EffectObligation | Mapping[str, object],
    *,
    control: str = POSITIVE,
    stimulus: object = _MISSING,
    eligible: bool | None = None,
    exercise: Callable[..., object] | None = None,
    observe: Callable[..., object] | None = None,
    binding: Mapping[str, object] | None = None,
    process: object = _MISSING,
    loaded_artifact: object = _MISSING,
    config: object = _MISSING,
    dependency_snapshot: object = _MISSING,
    deadline_reached: bool | None = None,
    substitutions: Sequence[object] = (),
    journal: object | None = None,
    correlation_id: str | None = None,
    experiment_id: str | None = None,
    receipt_id: str | None = None,
    experiment_sink: list[EffectExperiment] | None = None,
) -> EffectReceipt:
    """Run one host experiment and record a receipt for the independent observer."""
    item = _coerce_obligation(obligation)
    if control not in CONTROLS:
        raise ValueError("control must be positive or negative")
    if stimulus is _MISSING:
        stimulus = item.positive_stimulus if control == POSITIVE else item.negative_stimulus
    correlation = correlation_id or "corr-" + uuid.uuid4().hex
    experiment = experiment_id or "experiment-" + uuid.uuid4().hex
    receipt_name = receipt_id or "receipt-" + uuid.uuid4().hex
    host_binding = inspect_runtime_binding(
        binding, process=process, loaded_artifact=loaded_artifact,
        config=config, dependency_snapshot=dependency_snapshot)
    effective_eligible = _optional_bool(eligible, "eligible")
    executed: bool | None = False if effective_eligible is False else None
    effect_observed: bool | None = None
    result_count: int | None = None
    observer_completeness: bool | None = None
    status = "INELIGIBLE" if effective_eligible is False else "NOT_EXECUTED"
    detail = ""
    execution_value: object = None

    _emit(journal, "effect_liveness_probe_fired", producer="kernel_effect_liveness",
          record_type="experiment", obligation_id=item.obligation_id,
          experiment_id=experiment, control=control,
          stimulus_correlation_id=correlation, authority=OBSERVATION_ONLY)
    if effective_eligible is True and exercise is not None:
        try:
            execution_value = _invoke(exercise, stimulus, correlation)
            executed = True
            status = "EXECUTED"
        except Exception as error:  # an exercised failure is evidence of no completed effect, not a pass
            executed = True
            status = "EXECUTION_ERROR"
            detail = f"exercise_error:{type(error).__name__}"
    elif effective_eligible is True:
        status = "NO_EXERCISE_CALLBACK"
    if executed is True and status == "EXECUTED" and observe is not None:
        try:
            # The host token is available to the independent observer.  The
            # arity shim keeps two-argument domain callbacks compatible while
            # allowing a real readback to bind its query to this experiment.
            observed = _invoke(observe, stimulus, execution_value, correlation)
            if isinstance(observed, Mapping):
                effect_observed = _optional_bool(_first(
                    observed, "effect_observed", "effect_present", "observed", default=None),
                    "effect_observed")
                observer_completeness = _optional_bool(_first(
                    observed, "observer_completeness", "observer_complete", "observer_sufficient",
                    default=None), "observer_completeness")
                result_count = _result_count(observed)
                detail = str(_first(observed, "detail", default=detail))
            elif type(observed) is bool:
                effect_observed = observed
                result_count = None
            else:
                detail = "observer_returned_no_admitted_predicate"
        except Exception as error:
            effect_observed = None
            result_count = None
            observer_completeness = None
            detail = f"observer_error:{type(error).__name__}"
    elif executed is True and status == "EXECUTED":
        detail = "independent_observer_missing"

    experiment_record = EffectExperiment(
        experiment_id=experiment, obligation_id=item.obligation_id, control=control,
        stimulus=stimulus, stimulus_correlation_id=correlation,
        eligible=effective_eligible, executed=executed,
        status=status, substitutions=tuple(str(value) for value in substitutions),
        detail=detail,
    )
    if experiment_sink is not None:
        experiment_sink.append(experiment_record)
    _emit(journal, "effect_liveness_experiment", producer="kernel_effect_liveness",
          record_type="experiment", payload_schema=EXPERIMENT_SCHEMA,
          experiment=experiment_record.to_dict())

    receipt = EffectReceipt(
        receipt_id=receipt_name, experiment_id=experiment,
        obligation_id=item.obligation_id, control=control,
        candidate_digest=host_binding["candidate_digest"],
        exercised_process=host_binding["exercised_process"],
        loaded_artifact=host_binding["loaded_artifact"],
        config_digest=host_binding["config_digest"],
        dependency_snapshot=host_binding["dependency_snapshot"],
        stimulus_correlation_id=correlation,
        observer_completeness=observer_completeness,
        substitutions=tuple(str(value) for value in substitutions),
        eligible=effective_eligible, executed=executed,
        effect_observed=effect_observed, result_count=result_count,
        deadline_reached=deadline_reached,
        execution_status=status, detail=detail, observed_at_ns=time.time_ns(),
    )
    _emit(journal, "effect_liveness_receipt", producer="kernel_effect_liveness",
          record_type="receipt", payload_schema=RECEIPT_SCHEMA,
          receipt=receipt.to_dict())
    return receipt


def effect_obligations_from(value: object) -> tuple[EffectObligation, ...]:
    """Read explicit effect checks from a task/domain pack without heuristics."""
    candidates: object = value
    if isinstance(value, Mapping):
        candidates = _first(value, "effect_obligations", "effect_checks", "effect_liveness", default=())
    if candidates is None:
        return ()
    if isinstance(candidates, Mapping):
        candidates = (candidates,)
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        return ()
    result: list[EffectObligation] = []
    for candidate in candidates:
        if isinstance(candidate, EffectObligation):
            result.append(candidate)
        elif isinstance(candidate, Mapping):
            result.append(EffectObligation.from_mapping(candidate))
    return tuple(result)


class EffectLivenessProbe:
    """Small host-owned registry used by the runtime and coding domain pack."""

    def __init__(self, obligations: Sequence[EffectObligation | Mapping[str, object]] = (), *,
                 journal: object | None = None, binding_provider: Callable[[], Mapping[str, object]] | None = None,
                 enabled: bool | None = None):
        self.enabled = globals()["enabled"]() if enabled is None else bool(enabled)
        self.journal = journal
        self.binding_provider = binding_provider
        self.obligations: list[EffectObligation] = []
        self.receipts: list[EffectReceipt] = []
        self.experiments: list[EffectExperiment] = []
        self.assessments: list[EffectAssessment] = []
        if self.enabled:
            if isinstance(obligations, Mapping) or isinstance(obligations, EffectObligation):
                obligations = (obligations,)
            for item in obligations:
                self.register(item)
        else:
            # The object exists as a stable hook, but OFF performs no work.
            self.obligations = []

    def _binding(self, binding: Mapping[str, object] | None) -> Mapping[str, object] | None:
        if binding is not None:
            return binding
        if self.binding_provider is None:
            return None
        try:
            value = self.binding_provider()
        except Exception:
            return None
        return value if isinstance(value, Mapping) else None

    def _fire(self, action: str, **fields: object) -> None:
        if self.enabled:
            _emit(self.journal, "effect_liveness_probe_fired", producer="kernel_effect_liveness",
                  record_type=action, authority=OBSERVATION_ONLY, **fields)

    def register(self, obligation: EffectObligation | Mapping[str, object]) -> EffectObligation | None:
        if not self.enabled:
            return None
        item = _coerce_obligation(obligation)
        self.obligations.append(item)
        self._fire("obligation_registered", obligation_id=item.obligation_id)
        _emit(self.journal, "effect_liveness_obligation_registered",
              producer="kernel_effect_liveness", record_type="obligation",
              payload_schema=OBLIGATION_SCHEMA, obligation=item.to_dict())
        return item

    def run(self, obligation: EffectObligation | Mapping[str, object], **kwargs: object) -> EffectReceipt | None:
        if not self.enabled:
            return None
        item = _coerce_obligation(obligation)
        receipt = run_behavior_experiment(
            item, binding=self._binding(kwargs.pop("binding", None)), journal=self.journal,
            experiment_sink=self.experiments, **kwargs)
        self.receipts.append(receipt)
        return receipt

    def ingest_receipt(self, value: EffectReceipt | Mapping[str, object], *,
                       binding: Mapping[str, object] | None = None) -> EffectReceipt | None:
        if not self.enabled:
            return None
        if isinstance(value, Mapping):
            nested = _first(value, "effect_receipt", "effect_observation", "effect", default=_MISSING)
            if isinstance(nested, Mapping):
                value = nested
        try:
            receipt = _coerce_receipt(value, binding=self._binding(binding))
        except (TypeError, ValueError) as error:
            self._fire("receipt_invalid", error_type=type(error).__name__)
            _emit(self.journal, "effect_liveness_receipt_invalid",
                  producer="kernel_effect_liveness", record_type="receipt",
                  payload_schema=RECEIPT_SCHEMA, error_type=type(error).__name__)
            return None
        self.receipts.append(receipt)
        self._fire("receipt_ingested", obligation_id=receipt.obligation_id,
                   receipt_id=receipt.receipt_id, stimulus_correlation_id=receipt.stimulus_correlation_id)
        _emit(self.journal, "effect_liveness_receipt_ingested",
              producer="kernel_effect_liveness", record_type="receipt",
              payload_schema=RECEIPT_SCHEMA, receipt=receipt.to_dict())
        return receipt

    def assess(self, obligation: EffectObligation | Mapping[str, object], *,
               receipts: Sequence[EffectReceipt | Mapping[str, object]] | None = None,
               binding: Mapping[str, object] | None = None,
               deadline_reached: bool | None = None) -> EffectAssessment | None:
        if not self.enabled:
            return None
        item = _coerce_obligation(obligation)
        current_binding = self._binding(binding)
        self._fire("assessment", obligation_id=item.obligation_id)
        assessment = assess_effect_obligation(
            item, self.receipts if receipts is None else receipts,
            binding=current_binding, experiments=self.experiments,
            deadline_reached=deadline_reached)
        self.assessments.append(assessment)
        _emit(self.journal, "effect_liveness_assessment", producer="kernel_effect_liveness",
              record_type="assessment", payload_schema=ASSESSMENT_SCHEMA,
              assessment=assessment.to_dict())
        return assessment

    def assess_registered(self, *, receipts: Sequence[EffectReceipt | Mapping[str, object]] | None = None,
                          binding: Mapping[str, object] | None = None,
                          deadline_reached: bool | None = None) -> tuple[EffectAssessment, ...]:
        if not self.enabled:
            return ()
        return tuple(self.assess(item, receipts=receipts, binding=binding,
                                 deadline_reached=deadline_reached)
                     for item in self.obligations)

    def completion_blockers(self, assessments: Sequence[EffectAssessment] | None = None) -> tuple[EffectAssessment, ...]:
        current = self.assessments if assessments is None else assessments
        return tuple(item for item in current if item.blocking)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "enabled": self.enabled,
            "obligations": [item.to_dict() for item in self.obligations],
            "experiments": [item.to_dict() for item in self.experiments],
            "receipts": [item.to_dict() for item in self.receipts],
            "assessments": [item.to_dict() for item in self.assessments],
        }


# Compatibility aliases for thin packs that use the shorter names.
EffectCheck = EffectObligation
EffectExperimentReceipt = EffectReceipt
assess = assess_effect_obligation
