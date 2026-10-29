"""Host-owned System One policy selection.

This module consumes the immutable distribution produced by
``sys1.calibration`` and returns an advisory proposal or a typed refusal.  It
does not call a provider, lower a threshold, retry, grant permission, mutate a
decision graph, or execute an effect.  Existing host owners must recheck and
consume a proposal at the point of use.

The policy uses the complete calibrated outcome distribution.  In particular,
``p_selected_acceptable`` is retained as a diagnostic for its separately
labelled target; it is never substituted for a multiclass posterior.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import ClassVar, TypeAlias, Union

try:  # Package import path.
    from .calibration import CALIBRATED, CalibratedDistribution
except ImportError:  # pragma: no cover - flat v46 module loading.
    from calibration import CALIBRATED, CalibratedDistribution


_TIE_TOLERANCE = 1e-12
_QUESTION_TYPES = frozenset({"CHOICE", "BOOLEAN", "SCORE"})


class PolicyError(ValueError):
    """A loss specification or host snapshot is malformed."""


class PolicyStatus(str, Enum):
    """The only outcomes this policy layer may return."""

    PROPOSE = "PROPOSE"
    ABSTAIN = "ABSTAIN"
    UNAVAILABLE = "UNAVAILABLE"
    STALE = "STALE"


PROPOSE = PolicyStatus.PROPOSE
ABSTAIN = PolicyStatus.ABSTAIN
UNAVAILABLE = PolicyStatus.UNAVAILABLE
STALE = PolicyStatus.STALE


class ReasonCode(str, Enum):
    """Stable, host-readable reasons for selection or consumption refusal."""

    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    MISSING_PROBABILITY = "MISSING_PROBABILITY"
    INCOMPLETE_PROBABILITY_VECTOR = "INCOMPLETE_PROBABILITY_VECTOR"
    INVALID_PROBABILITY = "INVALID_PROBABILITY"
    INVALID_SELECTED_ACCEPTABILITY = "INVALID_SELECTED_ACCEPTABILITY"
    UNCALIBRATED = "UNCALIBRATED"
    CALIBRATOR_MISMATCH = "CALIBRATOR_MISMATCH"
    OUT_OF_SUPPORT = "OUT_OF_SUPPORT"
    TRANSPORT_UNAVAILABLE = "TRANSPORT_UNAVAILABLE"
    SHARED_BUDGET_REFUSED = "SHARED_BUDGET_REFUSED"
    NO_ADMISSIBLE_ACTION = "NO_ADMISSIBLE_ACTION"
    TIE_FALLBACK = "TIE_FALLBACK"
    UNDECLARED_TIE = "UNDECLARED_TIE"
    FALLBACK_LOWER_LOSS = "FALLBACK_LOWER_LOSS"
    RISK_CAP = "RISK_CAP"
    STALE_RECEIPT = "STALE_RECEIPT"
    STALE_STATE = "STALE_STATE"
    FRESHNESS_CHANGED = "FRESHNESS_CHANGED"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    PERMISSION_CHANGED = "PERMISSION_CHANGED"
    SCOPE_REFUSED = "SCOPE_REFUSED"
    SCOPE_CHANGED = "SCOPE_CHANGED"
    BUDGET_CHANGED = "BUDGET_CHANGED"


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or "\x00" in value:
        raise PolicyError(f"{field} must be a non-empty stable string")
    if len(value) > 256:
        raise PolicyError(f"{field} exceeds 256 characters")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicyError(f"{field} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise PolicyError(f"{field} must be finite")
    return result


def _nonnegative(value: object, field: str) -> float:
    result = _finite(value, field)
    if result < 0.0:
        raise PolicyError(f"{field} must be non-negative")
    return result


def _unique(values: Sequence[str], field: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise PolicyError(f"{field} must be a sequence")
    result = tuple(_text(value, f"{field}[]") for value in values)
    if not result:
        raise PolicyError(f"{field} must not be empty")
    if len(set(result)) != len(result):
        raise PolicyError(f"{field} must not contain duplicates")
    return result


def _question_type(value: object) -> str:
    result = _text(value, "question_type").upper()
    if result not in _QUESTION_TYPES:
        raise PolicyError("question_type must be CHOICE, BOOLEAN, or SCORE")
    return result


def _frozen_float_map(value: Mapping[str, object], field: str, keys: Sequence[str]) -> Mapping[str, float]:
    if not isinstance(value, Mapping):
        raise PolicyError(f"{field} must be a mapping")
    if set(value) != set(keys):
        raise PolicyError(f"{field} must cover the declared outcomes exactly")
    converted = {key: _nonnegative(value[key], f"{field}[{key}]") for key in keys}
    return MappingProxyType(converted)


@dataclass(frozen=True, slots=True)
class LossSpec:
    """Immutable host-owned action/outcome losses and calibration binding.

    ``action_losses[action][outcome]`` is the loss if that action is selected
    and the registered outcome is true.  ``fallback_cost`` plus the expected
    ``fallback_residual_losses`` is the comparison baseline.  There are no
    probability thresholds in this object: the strict comparison is derived
    from the declared losses and the complete distribution.
    """

    policy_id: str
    question_type: str
    outcomes: tuple[str, ...]
    actions: tuple[str, ...]
    action_losses: Mapping[str, Mapping[str, object]]
    fallback_id: str
    fallback_cost: float
    calibration_key_digest: str
    calibration_artifact_id: str
    fallback_residual_losses: Mapping[str, object] | None = None
    outcome_values: Mapping[str, object] | None = None
    declared_ties: Sequence[Sequence[str]] = ()
    risk_cap: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))
        object.__setattr__(self, "question_type", _question_type(self.question_type))
        outcomes = _unique(self.outcomes, "outcomes")
        actions = _unique(self.actions, "actions")
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(self, "actions", actions)
        if self.question_type == "BOOLEAN" and len(outcomes) != 2:
            raise PolicyError("BOOLEAN loss specs require exactly two outcomes")
        if self.question_type == "SCORE":
            if self.outcome_values is None:
                raise PolicyError("SCORE loss specs require ordered outcome values")
            if not isinstance(self.outcome_values, Mapping):
                raise PolicyError("SCORE outcome values must be a mapping")
            if set(self.outcome_values) != set(outcomes):
                raise PolicyError("SCORE outcome values must cover all levels")
            values = tuple(_finite(self.outcome_values[item], f"outcome_values[{item}]") for item in outcomes)
            if any(left >= right for left, right in zip(values, values[1:])):
                raise PolicyError("SCORE outcome values must be strictly increasing")
            object.__setattr__(self, "outcome_values", MappingProxyType(dict(zip(outcomes, values))))
        elif self.outcome_values is not None:
            if not isinstance(self.outcome_values, Mapping):
                raise PolicyError("outcome values must be a mapping")
            if set(self.outcome_values) != set(outcomes):
                raise PolicyError("outcome values must cover outcomes exactly")
            object.__setattr__(
                self,
                "outcome_values",
                MappingProxyType({item: _finite(self.outcome_values[item], f"outcome_values[{item}]") for item in outcomes}),
            )
        action_rows: dict[str, Mapping[str, float]] = {}
        if not isinstance(self.action_losses, Mapping) or set(self.action_losses) != set(actions):
            raise PolicyError("action_losses must cover every declared action exactly")
        for action in actions:
            action_rows[action] = _frozen_float_map(
                self.action_losses[action], f"action_losses[{action}]", outcomes
            )
        object.__setattr__(self, "action_losses", MappingProxyType(action_rows))
        residual = self.fallback_residual_losses
        if residual is None:
            residual = {outcome: 0.0 for outcome in outcomes}
        object.__setattr__(self, "fallback_residual_losses", _frozen_float_map(residual, "fallback_residual_losses", outcomes))
        object.__setattr__(self, "fallback_id", _text(self.fallback_id, "fallback_id"))
        object.__setattr__(self, "fallback_cost", _nonnegative(self.fallback_cost, "fallback_cost"))
        object.__setattr__(self, "calibration_key_digest", _text(self.calibration_key_digest, "calibration_key_digest"))
        object.__setattr__(self, "calibration_artifact_id", _text(self.calibration_artifact_id, "calibration_artifact_id"))
        ties: list[tuple[str, ...]] = []
        seen: set[str] = set()
        for index, group in enumerate(self.declared_ties):
            members = _unique(group, f"declared_ties[{index}]")
            if len(members) < 2:
                raise PolicyError("declared tie groups require at least two actions")
            if not set(members).issubset(actions):
                raise PolicyError("declared tie group contains an undeclared action")
            if seen.intersection(members):
                raise PolicyError("declared tie groups must not overlap")
            seen.update(members)
            ties.append(members)
        object.__setattr__(self, "declared_ties", tuple(ties))
        if self.risk_cap is not None:
            object.__setattr__(self, "risk_cap", _nonnegative(self.risk_cap, "risk_cap"))

    def fallback_loss(self, probabilities: Mapping[str, float]) -> float:
        """Return fallback cost plus its expected residual loss."""

        return self.fallback_cost + sum(
            probabilities[outcome] * self.fallback_residual_losses[outcome] for outcome in self.outcomes
        )

    def declared_tie_for(self, actions: Sequence[str]) -> tuple[str, ...] | None:
        selected = frozenset(actions)
        for group in self.declared_ties:
            if selected == frozenset(group):
                return group
        return None

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "question_type": self.question_type,
            "outcomes": list(self.outcomes),
            "actions": list(self.actions),
            "action_losses": {action: dict(losses) for action, losses in self.action_losses.items()},
            "fallback_id": self.fallback_id,
            "fallback_cost": self.fallback_cost,
            "fallback_residual_losses": dict(self.fallback_residual_losses),
            "calibration_key_digest": self.calibration_key_digest,
            "calibration_artifact_id": self.calibration_artifact_id,
            "outcome_values": None if self.outcome_values is None else dict(self.outcome_values),
            "declared_ties": [list(group) for group in self.declared_ties],
            "risk_cap": self.risk_cap,
        }


ImmutableLossSpec = LossSpec


@dataclass(frozen=True, slots=True)
class GateSnapshot:
    """Host-owned point-in-time identities and gate values.

    Tokens are opaque host identities.  Policy compares them at consumption;
    it does not resolve, refresh, or override the owners that issued them.
    """

    freshness_token: str
    permission_token: str
    scope_token: str
    state_token: str
    budget_token: str
    receipt_fresh: bool = True
    permission_granted: bool = True
    scope_allowed: bool = True
    state_current: bool = True
    shared_budget_available: bool = True

    def __post_init__(self) -> None:
        for field_name in ("freshness_token", "permission_token", "scope_token", "state_token", "budget_token"):
            object.__setattr__(self, field_name, _text(getattr(self, field_name), field_name))
        for field_name in (
            "receipt_fresh", "permission_granted", "scope_allowed", "state_current", "shared_budget_available"
        ):
            if type(getattr(self, field_name)) is not bool:
                raise PolicyError(f"{field_name} must be boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "freshness_token": self.freshness_token,
            "permission_token": self.permission_token,
            "scope_token": self.scope_token,
            "state_token": self.state_token,
            "budget_token": self.budget_token,
            "receipt_fresh": self.receipt_fresh,
            "permission_granted": self.permission_granted,
            "scope_allowed": self.scope_allowed,
            "state_current": self.state_current,
            "shared_budget_available": self.shared_budget_available,
        }


@dataclass(frozen=True, slots=True)
class SelectionContext:
    """Host admissions supplied for one policy evaluation."""

    admissible_actions: Sequence[str]
    gates: GateSnapshot
    evidence_sufficient: bool = True
    input_in_support: bool = True
    transport_available: bool = True

    def __post_init__(self) -> None:
        actions = tuple(_text(item, "admissible_actions[]") for item in self.admissible_actions)
        if len(set(actions)) != len(actions):
            raise PolicyError("admissible_actions must not contain duplicates")
        object.__setattr__(self, "admissible_actions", frozenset(actions))
        if not isinstance(self.gates, GateSnapshot):
            raise PolicyError("gates must be a GateSnapshot")
        for field_name in ("evidence_sufficient", "input_in_support", "transport_available"):
            if type(getattr(self, field_name)) is not bool:
                raise PolicyError(f"{field_name} must be boolean")


@dataclass(frozen=True, slots=True)
class PolicyDiagnostics:
    """Auditable diagnostics; none of these fields grants action authority."""

    probabilities: tuple[tuple[str, float], ...] = ()
    expected_losses: tuple[tuple[str, float], ...] = ()
    fallback_loss: float | None = None
    legal_mass: float | None = None
    selected_action_acceptability: float | None = None
    score_mean: float | None = None
    score_variance: float | None = None
    score_tails: tuple[tuple[str, float], ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "probabilities": dict(self.probabilities),
            "expected_losses": dict(self.expected_losses),
            "fallback_loss": self.fallback_loss,
            "legal_mass": self.legal_mass,
            "selected_action_acceptability": self.selected_action_acceptability,
            "score_mean": self.score_mean,
            "score_variance": self.score_variance,
            "score_tails": dict(self.score_tails),
        }


@dataclass(frozen=True, slots=True)
class Proposal:
    """An advisory selected action for an existing host owner to consume."""

    status: ClassVar[PolicyStatus] = PolicyStatus.PROPOSE
    action_id: str
    expected_loss: float
    fallback_loss: float
    policy_id: str
    calibration_key_digest: str
    calibration_artifact_id: str
    gates: GateSnapshot
    diagnostics: PolicyDiagnostics
    declared_tie: tuple[str, ...] = ()
    authoritative: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_id", _text(self.action_id, "action_id"))
        for field_name in ("expected_loss", "fallback_loss"):
            object.__setattr__(self, field_name, _finite(getattr(self, field_name), field_name))
        if self.authoritative is not False:
            raise PolicyError("SYS1 proposals are never authoritative")

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "action_id": self.action_id,
            "expected_loss": self.expected_loss,
            "fallback_loss": self.fallback_loss,
            "policy_id": self.policy_id,
            "calibration_key_digest": self.calibration_key_digest,
            "calibration_artifact_id": self.calibration_artifact_id,
            "gates": self.gates.to_dict(),
            "diagnostics": self.diagnostics.to_dict(),
            "declared_tie": list(self.declared_tie),
            "authoritative": self.authoritative,
        }


@dataclass(frozen=True, slots=True)
class Abstention:
    status: ClassVar[PolicyStatus] = PolicyStatus.ABSTAIN
    reason_codes: tuple[ReasonCode, ...]
    policy_id: str
    diagnostics: PolicyDiagnostics = PolicyDiagnostics()
    fallback_id: str | None = None

    def __post_init__(self) -> None:
        if not self.reason_codes:
            raise PolicyError("ABSTAIN requires a reason")
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "reason_codes": [reason.value for reason in self.reason_codes],
            "policy_id": self.policy_id,
            "diagnostics": self.diagnostics.to_dict(),
            "fallback_id": self.fallback_id,
        }


@dataclass(frozen=True, slots=True)
class Unavailable:
    status: ClassVar[PolicyStatus] = PolicyStatus.UNAVAILABLE
    reason_codes: tuple[ReasonCode, ...]
    policy_id: str
    diagnostics: PolicyDiagnostics = PolicyDiagnostics()

    def __post_init__(self) -> None:
        if not self.reason_codes:
            raise PolicyError("UNAVAILABLE requires a reason")
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "reason_codes": [reason.value for reason in self.reason_codes],
            "policy_id": self.policy_id,
            "diagnostics": self.diagnostics.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class Stale:
    status: ClassVar[PolicyStatus] = PolicyStatus.STALE
    reason_codes: tuple[ReasonCode, ...]
    policy_id: str
    diagnostics: PolicyDiagnostics = PolicyDiagnostics()

    def __post_init__(self) -> None:
        if not self.reason_codes:
            raise PolicyError("STALE requires a reason")
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "reason_codes": [reason.value for reason in self.reason_codes],
            "policy_id": self.policy_id,
            "diagnostics": self.diagnostics.to_dict(),
        }


PolicyResult: TypeAlias = Union[Proposal, Abstention, Unavailable, Stale]


@dataclass(frozen=True, slots=True)
class ConsumptionCheck:
    """Result of host-owner gate rechecks; acceptance is not authorization."""

    accepted: bool
    result: PolicyResult
    owner_authorized: bool = False

    def __post_init__(self) -> None:
        if self.owner_authorized is not False:
            raise PolicyError("policy cannot authorize effects")


def _reason(value: ReasonCode) -> tuple[ReasonCode, ...]:
    return (value,)


def _empty_diagnostics(legal_mass: float | None = None) -> PolicyDiagnostics:
    return PolicyDiagnostics(legal_mass=legal_mass)


def _distribution_values(
    distribution: object,
    spec: LossSpec,
    legal_mass: float | None,
) -> tuple[dict[str, float] | None, PolicyDiagnostics, ReasonCode | None]:
    if legal_mass is not None:
        legal_mass = _finite(legal_mass, "legal_mass")
        if legal_mass < 0.0:
            raise PolicyError("legal_mass must be non-negative")
    if not isinstance(distribution, CalibratedDistribution):
        return None, _empty_diagnostics(legal_mass), ReasonCode.MISSING_PROBABILITY
    selected = distribution.p_selected_acceptable
    if selected is not None and (not math.isfinite(selected) or not 0.0 <= selected <= 1.0):
        return None, _empty_diagnostics(legal_mass), ReasonCode.INVALID_SELECTED_ACCEPTABILITY
    diagnostics = PolicyDiagnostics(
        legal_mass=legal_mass,
        selected_action_acceptability=selected,
    )
    if distribution.status != CALIBRATED or distribution.shadow_only:
        return None, diagnostics, ReasonCode.UNCALIBRATED
    if distribution.key_digest != spec.calibration_key_digest or distribution.artifact_id != spec.calibration_artifact_id:
        return None, diagnostics, ReasonCode.CALIBRATOR_MISMATCH
    if str(distribution.question_type).upper() != spec.question_type:
        return None, diagnostics, ReasonCode.CALIBRATOR_MISMATCH
    raw = distribution.probabilities
    if set(raw) != set(spec.outcomes):
        return None, diagnostics, ReasonCode.INCOMPLETE_PROBABILITY_VECTOR
    values: dict[str, float] = {}
    for outcome in spec.outcomes:
        value = raw.get(outcome)
        if value is None:
            return None, diagnostics, ReasonCode.MISSING_PROBABILITY
        try:
            number = _finite(value, f"probabilities[{outcome}]")
        except PolicyError:
            return None, diagnostics, ReasonCode.INVALID_PROBABILITY
        if not 0.0 <= number <= 1.0:
            return None, diagnostics, ReasonCode.INVALID_PROBABILITY
        values[outcome] = number
    total = sum(values.values())
    if not math.isclose(total, 1.0, rel_tol=1e-8, abs_tol=1e-8):
        return None, diagnostics, ReasonCode.INVALID_PROBABILITY
    if spec.question_type == "BOOLEAN":
        first, second = spec.outcomes
        complement = 1.0 - values[first]
        if not math.isclose(values[second], complement, rel_tol=1e-8, abs_tol=1e-8):
            return None, diagnostics, ReasonCode.INVALID_PROBABILITY
        values[second] = complement
    probabilities = tuple((outcome, values[outcome]) for outcome in spec.outcomes)
    score_mean: float | None = None
    score_variance: float | None = None
    score_tails: tuple[tuple[str, float], ...] = ()
    if spec.question_type == "SCORE":
        assert spec.outcome_values is not None  # validated by LossSpec
        score_mean = sum(values[outcome] * spec.outcome_values[outcome] for outcome in spec.outcomes)
        score_variance = sum(
            values[outcome] * (spec.outcome_values[outcome] - score_mean) ** 2 for outcome in spec.outcomes
        )
        score_tails = tuple(
            (outcome, sum(values[item] for item in spec.outcomes[index:]))
            for index, outcome in enumerate(spec.outcomes)
        )
    diagnostics = PolicyDiagnostics(
        probabilities=probabilities,
        legal_mass=legal_mass,
        selected_action_acceptability=selected,
        score_mean=score_mean,
        score_variance=score_variance,
        score_tails=score_tails,
    )
    return values, diagnostics, None


def _block(
    status: PolicyStatus,
    reasons: Sequence[ReasonCode],
    spec: LossSpec,
    diagnostics: PolicyDiagnostics,
) -> PolicyResult:
    if status is PolicyStatus.ABSTAIN:
        return Abstention(tuple(reasons), spec.policy_id, diagnostics, spec.fallback_id)
    if status is PolicyStatus.STALE:
        return Stale(tuple(reasons), spec.policy_id, diagnostics)
    if status is PolicyStatus.UNAVAILABLE:
        return Unavailable(tuple(reasons), spec.policy_id, diagnostics)
    raise PolicyError("PROPOSE is not a blocker status")


def select_action(
    distribution: CalibratedDistribution,
    loss_spec: LossSpec,
    context: SelectionContext,
    *,
    legal_mass: float | None = None,
) -> PolicyResult:
    """Select one admissible action, or return one typed refusal.

    This is a single bounded calculation.  It does not accept thresholds,
    retry counts, callbacks, or effect handles, so callers cannot turn an
    unsuccessful result into an unlimited policy ladder here.
    """

    if not isinstance(loss_spec, LossSpec):
        raise PolicyError("loss_spec must be a LossSpec")
    if not isinstance(context, SelectionContext):
        raise PolicyError("context must be a SelectionContext")
    values, diagnostics, distribution_reason = _distribution_values(distribution, loss_spec, legal_mass)
    if not context.transport_available:
        return _block(PolicyStatus.UNAVAILABLE, _reason(ReasonCode.TRANSPORT_UNAVAILABLE), loss_spec, diagnostics)
    if not context.gates.shared_budget_available:
        return _block(PolicyStatus.UNAVAILABLE, _reason(ReasonCode.SHARED_BUDGET_REFUSED), loss_spec, diagnostics)
    if not context.gates.receipt_fresh:
        return _block(PolicyStatus.STALE, _reason(ReasonCode.STALE_RECEIPT), loss_spec, diagnostics)
    if not context.gates.state_current:
        return _block(PolicyStatus.STALE, _reason(ReasonCode.STALE_STATE), loss_spec, diagnostics)
    if not context.gates.permission_granted:
        return _block(PolicyStatus.UNAVAILABLE, _reason(ReasonCode.PERMISSION_DENIED), loss_spec, diagnostics)
    if not context.gates.scope_allowed:
        return _block(PolicyStatus.UNAVAILABLE, _reason(ReasonCode.SCOPE_REFUSED), loss_spec, diagnostics)
    if not context.input_in_support:
        return _block(PolicyStatus.UNAVAILABLE, _reason(ReasonCode.OUT_OF_SUPPORT), loss_spec, diagnostics)
    if not context.evidence_sufficient:
        return _block(PolicyStatus.ABSTAIN, _reason(ReasonCode.INSUFFICIENT_EVIDENCE), loss_spec, diagnostics)
    if distribution_reason is not None:
        status = PolicyStatus.ABSTAIN if distribution_reason in {
            ReasonCode.MISSING_PROBABILITY,
            ReasonCode.INCOMPLETE_PROBABILITY_VECTOR,
            ReasonCode.INVALID_PROBABILITY,
            ReasonCode.INVALID_SELECTED_ACCEPTABILITY,
        } else PolicyStatus.UNAVAILABLE
        return _block(status, _reason(distribution_reason), loss_spec, diagnostics)
    assert values is not None
    admissible = tuple(action for action in loss_spec.actions if action in context.admissible_actions)
    if not admissible:
        return _block(PolicyStatus.ABSTAIN, _reason(ReasonCode.NO_ADMISSIBLE_ACTION), loss_spec, diagnostics)
    fallback_loss = loss_spec.fallback_loss(values)
    expected = {
        action: sum(values[outcome] * loss_spec.action_losses[action][outcome] for outcome in loss_spec.outcomes)
        for action in admissible
    }
    diagnostics = PolicyDiagnostics(
        probabilities=diagnostics.probabilities,
        expected_losses=tuple((action, expected[action]) for action in admissible),
        fallback_loss=fallback_loss,
        legal_mass=diagnostics.legal_mass,
        selected_action_acceptability=diagnostics.selected_action_acceptability,
        score_mean=diagnostics.score_mean,
        score_variance=diagnostics.score_variance,
        score_tails=diagnostics.score_tails,
    )
    minimum = min(expected.values())
    tied = tuple(action for action in admissible if math.isclose(expected[action], minimum, rel_tol=0.0, abs_tol=_TIE_TOLERANCE))
    if len(tied) > 1:
        tie_reason = ReasonCode.TIE_FALLBACK if loss_spec.declared_tie_for(tied) is not None else ReasonCode.UNDECLARED_TIE
        return _block(PolicyStatus.ABSTAIN, _reason(tie_reason), loss_spec, diagnostics)
    selected = tied[0]
    if not minimum < fallback_loss - _TIE_TOLERANCE:
        return _block(PolicyStatus.ABSTAIN, _reason(ReasonCode.FALLBACK_LOWER_LOSS), loss_spec, diagnostics)
    if loss_spec.risk_cap is not None and minimum > loss_spec.risk_cap + _TIE_TOLERANCE:
        return _block(PolicyStatus.ABSTAIN, _reason(ReasonCode.RISK_CAP), loss_spec, diagnostics)
    return Proposal(
        action_id=selected,
        expected_loss=minimum,
        fallback_loss=fallback_loss,
        policy_id=loss_spec.policy_id,
        calibration_key_digest=loss_spec.calibration_key_digest,
        calibration_artifact_id=loss_spec.calibration_artifact_id,
        gates=context.gates,
        diagnostics=diagnostics,
        declared_tie=loss_spec.declared_tie_for(tied) or (),
    )


def _consumption_block(
    proposal: Proposal,
    status: PolicyStatus,
    reasons: Sequence[ReasonCode],
) -> ConsumptionCheck:
    if status is PolicyStatus.STALE:
        result: PolicyResult = Stale(tuple(reasons), proposal.policy_id, proposal.diagnostics)
    elif status is PolicyStatus.UNAVAILABLE:
        result = Unavailable(tuple(reasons), proposal.policy_id, proposal.diagnostics)
    else:
        raise PolicyError("consumption block must be STALE or UNAVAILABLE")
    return ConsumptionCheck(False, result)


def recheck_consumption(proposal: Proposal, current: GateSnapshot) -> ConsumptionCheck:
    """Recheck all host gates immediately before an owner consumes a proposal."""

    if not isinstance(proposal, Proposal):
        raise PolicyError("proposal must be a Proposal")
    if not isinstance(current, GateSnapshot):
        raise PolicyError("current must be a GateSnapshot")
    captured = proposal.gates
    stale_reasons: list[ReasonCode] = []
    unavailable_reasons: list[ReasonCode] = []
    if not current.receipt_fresh:
        stale_reasons.append(ReasonCode.STALE_RECEIPT)
    elif current.freshness_token != captured.freshness_token:
        stale_reasons.append(ReasonCode.FRESHNESS_CHANGED)
    if not current.state_current:
        stale_reasons.append(ReasonCode.STALE_STATE)
    elif current.state_token != captured.state_token:
        stale_reasons.append(ReasonCode.STALE_STATE)
    if not current.permission_granted:
        unavailable_reasons.append(ReasonCode.PERMISSION_DENIED)
    elif current.permission_token != captured.permission_token:
        unavailable_reasons.append(ReasonCode.PERMISSION_CHANGED)
    if not current.scope_allowed:
        unavailable_reasons.append(ReasonCode.SCOPE_REFUSED)
    elif current.scope_token != captured.scope_token:
        unavailable_reasons.append(ReasonCode.SCOPE_CHANGED)
    if not current.shared_budget_available:
        unavailable_reasons.append(ReasonCode.SHARED_BUDGET_REFUSED)
    elif current.budget_token != captured.budget_token:
        unavailable_reasons.append(ReasonCode.BUDGET_CHANGED)
    if stale_reasons:
        return _consumption_block(proposal, PolicyStatus.STALE, stale_reasons)
    if unavailable_reasons:
        return _consumption_block(proposal, PolicyStatus.UNAVAILABLE, unavailable_reasons)
    return ConsumptionCheck(True, proposal)


check_consumption = recheck_consumption


__all__ = [
    "ABSTAIN",
    "PROPOSE",
    "STALE",
    "UNAVAILABLE",
    "Abstention",
    "ConsumptionCheck",
    "GateSnapshot",
    "ImmutableLossSpec",
    "LossSpec",
    "PolicyDiagnostics",
    "PolicyError",
    "PolicyResult",
    "PolicyStatus",
    "Proposal",
    "ReasonCode",
    "SelectionContext",
    "Stale",
    "Unavailable",
    "check_consumption",
    "recheck_consumption",
    "select_action",
]
