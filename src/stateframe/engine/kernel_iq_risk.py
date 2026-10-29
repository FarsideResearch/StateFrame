"""Selective, host-settled two-risk pre-mortem records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


SCHEMA = "iq-procedural-risk/1"


class RiskContractError(ValueError):
    """Raised for ungrounded or over-broad risk records."""


class RiskStatus(str, Enum):
    ACTIVE = "ACTIVE"
    RESOLVED = "RESOLVED"
    FIRED = "FIRED"


class RiskObservationStatus(str, Enum):
    FIRED = "FIRED"
    IGNORED_RESOLVED = "IGNORED_RESOLVED"
    IGNORED_IRRELEVANT = "IGNORED_IRRELEVANT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class RiskRecord:
    risk_id: str
    basis_refs: tuple[str, ...]
    affected_goal: str
    early_observable: str
    mitigation_or_acquisition: str
    status: RiskStatus = RiskStatus.ACTIVE

    def __post_init__(self) -> None:
        for field_name in (
            "risk_id",
            "affected_goal",
            "early_observable",
            "mitigation_or_acquisition",
        ):
            _text(getattr(self, field_name), field_name)
        object.__setattr__(
            self, "basis_refs", _text_tuple(self.basis_refs, "basis ref")
        )
        if not self.basis_refs:
            raise RiskContractError(
                "risk must name at least one current claim or dependency"
            )
        object.__setattr__(self, "status", RiskStatus(self.status))


@dataclass(frozen=True, slots=True)
class RiskObservation:
    risk_id: str
    status: RiskObservationStatus
    observable: str
    receipt_id: str
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "iq-procedural-risk-observation/1",
            "risk_id": self.risk_id,
            "status": self.status.value,
            "observable": self.observable,
            "receipt_id": self.receipt_id,
            "detail": self.detail,
        }


class RiskRegister:
    """A bounded register that consumes only current, goal-relevant risks."""

    def __init__(
        self, *, current_basis: Sequence[str], active_goals: Sequence[str]
    ) -> None:
        self.current_basis = frozenset(
            _text_tuple(tuple(current_basis), "current basis")
        )
        self.active_goals = frozenset(_text_tuple(tuple(active_goals), "active goal"))
        self._risks: dict[str, RiskRecord] = {}
        self._observations: list[RiskObservation] = []

    def add(self, risk: RiskRecord) -> None:
        if risk.risk_id in self._risks:
            raise RiskContractError(f"duplicate risk: {risk.risk_id}")
        if len(self._risks) >= 2:
            raise RiskContractError("pre-mortem may contain at most two risks")
        self._risks[risk.risk_id] = risk

    def active(self, goal: str | None = None) -> tuple[RiskRecord, ...]:
        if goal is not None:
            _text(goal, "goal")
        return tuple(
            risk
            for risk in self._risks.values()
            if risk.status is RiskStatus.ACTIVE
            and risk.basis_refs
            and set(risk.basis_refs) & self.current_basis
            and (goal is None or risk.affected_goal == goal)
            and risk.affected_goal in self.active_goals
        )

    def resolve(self, risk_id: str) -> None:
        risk = self._risks.get(_text(risk_id, "risk_id"))
        if risk is None:
            raise RiskContractError("unknown risk")
        self._risks[risk_id] = RiskRecord(
            risk.risk_id,
            risk.basis_refs,
            risk.affected_goal,
            risk.early_observable,
            risk.mitigation_or_acquisition,
            RiskStatus.RESOLVED,
        )

    def observe(
        self, risk_id: str, observable: str, *, receipt_id: str
    ) -> RiskObservation:
        risk_id = _text(risk_id, "risk_id")
        observable = _text(observable, "observable")
        receipt_id = _text(receipt_id, "receipt_id")
        risk = self._risks.get(risk_id)
        if risk is None:
            result = RiskObservation(
                risk_id,
                RiskObservationStatus.UNKNOWN,
                observable,
                receipt_id,
                "risk is not registered",
            )
        elif risk.status is RiskStatus.RESOLVED:
            result = RiskObservation(
                risk_id,
                RiskObservationStatus.IGNORED_RESOLVED,
                observable,
                receipt_id,
                "resolved risk is suppressed",
            )
        elif risk.affected_goal not in self.active_goals or not (
            set(risk.basis_refs) & self.current_basis
        ):
            result = RiskObservation(
                risk_id,
                RiskObservationStatus.IGNORED_IRRELEVANT,
                observable,
                receipt_id,
                "risk is not current or goal-relevant",
            )
        elif observable != risk.early_observable:
            result = RiskObservation(
                risk_id,
                RiskObservationStatus.UNKNOWN,
                observable,
                receipt_id,
                "observable does not match the registered early signal",
            )
        else:
            result = RiskObservation(
                risk_id,
                RiskObservationStatus.FIRED,
                observable,
                receipt_id,
                "host observed the registered early signal",
            )
            self._risks[risk_id] = RiskRecord(
                risk.risk_id,
                risk.basis_refs,
                risk.affected_goal,
                risk.early_observable,
                risk.mitigation_or_acquisition,
                RiskStatus.FIRED,
            )
        self._observations.append(result)
        return result

    @property
    def observations(self) -> tuple[RiskObservation, ...]:
        return tuple(self._observations)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "current_basis": sorted(self.current_basis),
            "active_goals": sorted(self.active_goals),
            "risks": [
                {
                    "risk_id": risk.risk_id,
                    "basis_refs": list(risk.basis_refs),
                    "affected_goal": risk.affected_goal,
                    "early_observable": risk.early_observable,
                    "mitigation_or_acquisition": risk.mitigation_or_acquisition,
                    "status": risk.status.value,
                }
                for risk in self._risks.values()
            ],
            "observations": [
                observation.as_dict() for observation in self._observations
            ],
        }


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RiskContractError(f"{field_name} must be non-empty")
    return value.strip()


def _text_tuple(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise RiskContractError(f"{field_name} must be a sequence")
    return tuple(dict.fromkeys(_text(value, field_name) for value in values))


__all__ = [
    "RiskContractError",
    "RiskObservation",
    "RiskObservationStatus",
    "RiskRecord",
    "RiskRegister",
    "RiskStatus",
    "SCHEMA",
]
