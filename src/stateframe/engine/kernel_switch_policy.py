"""Deterministic, host-owned course switching policy.

This module is an adapter-sized policy reducer, not a second completion owner.
It records course identity, basis freshness, settled progress, and bounded
comparisons.  It never dispatches an action, changes a workspace, or asks a
model to choose a course.  A later live owner can consume :class:`Decision`
as a proposal and perform its own admission/effect reconciliation.

The data model is deliberately language-neutral: course and evidence
identities are opaque strings, and no source-language parser or Python AST is
used here.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum


SCHEMA = "stateframe-switch-policy/1"


class SwitchPolicyError(ValueError):
    """Raised when a host cannot admit a malformed policy record."""


class BasisStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    SUSPECT = "SUSPECT"
    UNSUPPORTED = "UNSUPPORTED"


class ReviewTrigger(str, Enum):
    ALTERNATIVE = "ALTERNATIVE"
    STALL = "STALL"
    CHECKPOINT = "CHECKPOINT"
    BASIS_BROKEN = "BASIS_BROKEN"
    OWNER_OVERRIDE = "OWNER_OVERRIDE"


class DecisionOutcome(str, Enum):
    CONTINUE = "CONTINUE"
    RECOMMIT = "RECOMMIT"
    SWITCH = "SWITCH"
    PROBE = "PROBE"
    ABSTAIN = "ABSTAIN"


def _text(value: object, field: str, *, limit: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise SwitchPolicyError(f"{field} must be a non-empty string of at most {limit} characters")
    return value.strip()


def _finite_nonnegative(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SwitchPolicyError(f"{field} must be a finite non-negative number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise SwitchPolicyError(f"{field} must be a finite non-negative number")
    return number


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LossBounds:
    """Comparable future-loss bounds in one declared unit and horizon."""

    low: float
    high: float
    unit: str = "seconds"
    horizon: str = "goal"

    def __post_init__(self) -> None:
        low = _finite_nonnegative(self.low, "loss low")
        high = _finite_nonnegative(self.high, "loss high")
        if high < low:
            raise SwitchPolicyError("loss high must be greater than or equal to loss low")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)
        object.__setattr__(self, "unit", _text(self.unit, "loss unit", limit=64))
        object.__setattr__(self, "horizon", _text(self.horizon, "loss horizon", limit=128))

    def to_dict(self) -> dict[str, object]:
        return {"low": self.low, "high": self.high, "unit": self.unit, "horizon": self.horizon}


@dataclass(frozen=True)
class Course:
    """A host-registered intention with a revision-bound positive basis."""

    course_id: str
    goal_id: str
    basis_digest: str
    status: BasisStatus = BasisStatus.SUPPORTED
    admitted: bool = True
    checkpoint_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "course_id", _text(self.course_id, "course_id"))
        object.__setattr__(self, "goal_id", _text(self.goal_id, "goal_id"))
        object.__setattr__(self, "basis_digest", _text(self.basis_digest, "basis_digest"))
        object.__setattr__(self, "status", _enum_value(self.status, BasisStatus, "basis status"))
        if not isinstance(self.admitted, bool):
            raise SwitchPolicyError("course admitted must be boolean")
        if self.checkpoint_id:
            object.__setattr__(self, "checkpoint_id", _text(self.checkpoint_id, "checkpoint_id"))

    def to_dict(self) -> dict[str, object]:
        return {
            "course_id": self.course_id,
            "goal_id": self.goal_id,
            "basis_digest": self.basis_digest,
            "status": self.status.value,
            "admitted": self.admitted,
            "checkpoint_id": self.checkpoint_id,
        }


@dataclass(frozen=True)
class Candidate:
    """An admitted comparison option with future, not sunk, costs."""

    course: Course
    remaining_loss: LossBounds
    transition_loss: LossBounds = LossBounds(0.0, 0.0)

    def __post_init__(self) -> None:
        if self.remaining_loss.unit != self.transition_loss.unit:
            raise SwitchPolicyError("loss bounds must use the same unit")
        if self.remaining_loss.horizon != self.transition_loss.horizon:
            raise SwitchPolicyError("loss bounds must use the same horizon")

    def to_dict(self) -> dict[str, object]:
        return {
            "course": self.course.to_dict(),
            "remaining_loss": self.remaining_loss.to_dict(),
            "transition_loss": self.transition_loss.to_dict(),
        }


@dataclass(frozen=True)
class ProgressReceipt:
    """One settled host observation; transport IDs alone are not progress."""

    receipt_id: str
    course_id: str
    effect_key: str
    settled: bool = True
    product_progress: bool = False
    information_progress: bool = False

    def __post_init__(self) -> None:
        for field in ("receipt_id", "course_id", "effect_key"):
            object.__setattr__(self, field, _text(getattr(self, field), field, limit=512))
        for field in ("settled", "product_progress", "information_progress"):
            if not isinstance(getattr(self, field), bool):
                raise SwitchPolicyError(f"{field} must be boolean")


@dataclass(frozen=True)
class PolicyEvent:
    kind: str
    course_id: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "course_id": self.course_id, "detail": self.detail}


@dataclass(frozen=True)
class Decision:
    """A cached policy result for a host owner to admit or reject."""

    outcome: DecisionOutcome
    reason_code: str
    current_course_id: str
    destination_course_id: str | None
    trigger: ReviewTrigger
    comparison_key: str
    review_required: bool
    no_progress_steps: int
    steps_since_checkpoint: int
    events: tuple[PolicyEvent, ...] = ()
    model_calls: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "outcome": self.outcome.value,
            "reason_code": self.reason_code,
            "current_course_id": self.current_course_id,
            "destination_course_id": self.destination_course_id,
            "trigger": self.trigger.value,
            "comparison_key": self.comparison_key,
            "review_required": self.review_required,
            "no_progress_steps": self.no_progress_steps,
            "steps_since_checkpoint": self.steps_since_checkpoint,
            "events": [event.to_dict() for event in self.events],
            "model_calls": self.model_calls,
        }


def _enum_value(value: object, enum_type: type[Enum], field: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except (TypeError, ValueError) as error:
        raise SwitchPolicyError(f"{field} is unsupported") from error


def _zero_loss(unit: str, horizon: str) -> LossBounds:
    return LossBounds(0.0, 0.0, unit=unit, horizon=horizon)


class SwitchPolicy:
    """A bounded reducer for optional course changes and basis invalidation."""

    def __init__(
        self,
        goal_id: str,
        current: Course,
        *,
        stall_k: int = 5,
        lease_steps: int = 10,
        soft_dwell: int = 3,
        absolute_margin: float = 15.0,
        relative_margin: float = 0.10,
    ) -> None:
        self.goal_id = _text(goal_id, "goal_id")
        if current.goal_id != self.goal_id:
            raise SwitchPolicyError("current course belongs to a different goal")
        if current.status is not BasisStatus.SUPPORTED:
            raise SwitchPolicyError("current course must start SUPPORTED")
        if not current.admitted:
            raise SwitchPolicyError("current course must be admitted")
        for field, value, minimum in (
            ("stall_k", stall_k, 1),
            ("lease_steps", lease_steps, 1),
            ("soft_dwell", soft_dwell, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise SwitchPolicyError(f"{field} must be an integer >= {minimum}")
        if isinstance(relative_margin, bool) or not isinstance(relative_margin, (int, float)):
            raise SwitchPolicyError("relative_margin must be finite and non-negative")
        if not math.isfinite(float(relative_margin)) or float(relative_margin) < 0:
            raise SwitchPolicyError("relative_margin must be finite and non-negative")
        self.stall_k = stall_k
        self.lease_steps = lease_steps
        self.soft_dwell = soft_dwell
        self.absolute_margin = _finite_nonnegative(absolute_margin, "absolute_margin")
        self.relative_margin = float(relative_margin)
        self._courses: dict[str, Course] = {current.course_id: current}
        self._current_id = current.course_id
        self._departed: set[str] = set()
        self._seen_receipts: dict[str, bool] = {}
        self._seen_effects: set[str] = set()
        self._comparison_cache: dict[str, Decision] = {}
        self._events: list[PolicyEvent] = []
        self._no_progress_steps = 0
        self._steps_since_checkpoint = 0
        self._dwell_until_step = 0

    @property
    def current(self) -> Course:
        return self._courses[self._current_id]

    @property
    def no_progress_steps(self) -> int:
        return self._no_progress_steps

    @property
    def steps_since_checkpoint(self) -> int:
        return self._steps_since_checkpoint

    @property
    def stall_due(self) -> bool:
        return self._no_progress_steps >= self.stall_k

    @property
    def checkpoint_due(self) -> bool:
        return self._steps_since_checkpoint >= self.lease_steps

    @property
    def departed_courses(self) -> tuple[str, ...]:
        return tuple(sorted(self._departed))

    def _register(self, course: Course) -> None:
        if course.goal_id != self.goal_id:
            raise SwitchPolicyError("candidate course belongs to a different goal")
        previous = self._courses.get(course.course_id)
        if previous is not None and previous.goal_id != course.goal_id:
            raise SwitchPolicyError("course identity was reused across goals")
        if previous is None:
            self._courses[course.course_id] = course
        # Existing basis/status is host-owned.  A caller-provided candidate is
        # cost input only; it cannot resurrect a SUSPECT course during review.

    def observe_progress(self, receipt: ProgressReceipt) -> bool:
        """Apply one settled receipt; duplicate transport/effect IDs count once."""
        if receipt.course_id not in self._courses:
            raise SwitchPolicyError("progress receipt references an unknown course")
        previous_settled = self._seen_receipts.get(receipt.receipt_id)
        if previous_settled is True or (previous_settled is False and not receipt.settled):
            return False
        self._seen_receipts[receipt.receipt_id] = receipt.settled
        if not receipt.settled:
            return True
        self._steps_since_checkpoint += 1
        effect_seen = receipt.effect_key in self._seen_effects
        if not effect_seen:
            self._seen_effects.add(receipt.effect_key)
        if (receipt.product_progress or receipt.information_progress) and not effect_seen:
            self._no_progress_steps = 0
        else:
            self._no_progress_steps += 1
        self._comparison_cache.clear()
        return True

    def observe_basis(
        self,
        course_id: str,
        *,
        basis_digest: str,
        status: BasisStatus = BasisStatus.SUPPORTED,
    ) -> tuple[PolicyEvent, ...]:
        """Refresh one basis and immediately invalidate dependent current work."""
        if course_id not in self._courses:
            raise SwitchPolicyError("cannot refresh an unknown course")
        basis = _text(basis_digest, "basis_digest")
        normalized = _enum_value(status, BasisStatus, "basis status")
        course = self._courses[course_id]
        changed = course.basis_digest != basis or course.status is not normalized
        updated_status = normalized
        if course_id == self._current_id and changed and normalized is BasisStatus.SUPPORTED:
            updated_status = BasisStatus.SUSPECT
        updated = replace(course, basis_digest=basis, status=updated_status)
        self._courses[course_id] = updated
        if changed:
            detail = "current basis invalidated" if course_id == self._current_id else "alternative basis refreshed"
            event = PolicyEvent(
                "BASIS_INVALIDATED" if course_id == self._current_id else "ALTERNATIVE_BASIS_CHANGED",
                course_id,
                detail,
            )
            self._events.append(event)
            self._comparison_cache.clear()
            return (event,)
        return ()

    def recommit_current(self, basis_digest: str) -> PolicyEvent:
        """Allow a fresh host-supported basis to renew the same course."""
        course = self.current
        updated = replace(course, basis_digest=_text(basis_digest, "basis_digest"), status=BasisStatus.SUPPORTED)
        self._courses[course.course_id] = updated
        event = PolicyEvent("RECOMMIT", course.course_id, "fresh supporting basis admitted")
        self._events.append(event)
        self._comparison_cache.clear()
        return event

    def commit(self, course_id: str, *, automatic: bool = False) -> PolicyEvent:
        """Commit a host-admitted destination while preserving departed history."""
        course = self._courses.get(course_id)
        if course is None:
            raise SwitchPolicyError("cannot commit an unknown course")
        if not course.admitted or course.status is not BasisStatus.SUPPORTED:
            raise SwitchPolicyError("course is not currently admitted and supported")
        if automatic and course_id in self._departed:
            raise SwitchPolicyError("automatic return to a departed course is forbidden")
        if course_id == self._current_id:
            return PolicyEvent("RECOMMIT", course_id, "same course remains current")
        previous = self._current_id
        self._departed.add(previous)
        self._current_id = course_id
        self._steps_since_checkpoint = 0
        self._no_progress_steps = 0
        self._dwell_until_step = self._steps_since_checkpoint + self.soft_dwell
        event = PolicyEvent("SWITCH_COMMITTED", course_id, f"departed {previous}")
        self._events.append(event)
        self._comparison_cache.clear()
        return event

    def _comparison_key(
        self,
        current: Candidate,
        alternatives: Sequence[Candidate],
        trigger: ReviewTrigger,
    ) -> str:
        payload = {
            "goal_id": self.goal_id,
            "current": current.to_dict(),
            "alternatives": [candidate.to_dict() for candidate in sorted(alternatives, key=lambda item: item.course.course_id)],
            "trigger": trigger.value,
            "progress": {"no_progress": self._no_progress_steps, "steps": self._steps_since_checkpoint},
            "departed": sorted(self._departed),
            "dwell_until": self._dwell_until_step,
        }
        return _digest(payload)

    def evaluate(
        self,
        current: Candidate,
        alternatives: Sequence[Candidate],
        *,
        trigger: ReviewTrigger = ReviewTrigger.ALTERNATIVE,
    ) -> Decision:
        """Evaluate an optional change without performing the change."""
        normalized_trigger = _enum_value(trigger, ReviewTrigger, "review trigger")
        self._register(current.course)
        for candidate in alternatives:
            self._register(candidate.course)
        current_course = self.current
        key = self._comparison_key(current, alternatives, normalized_trigger)
        cached = self._comparison_cache.get(key)
        if cached is not None:
            return cached

        if current.course.course_id != current_course.course_id:
            decision = self._decision(
                DecisionOutcome.ABSTAIN, "CURRENT_CANDIDATE_NOT_CURRENT", key,
                normalized_trigger, review_required=True,
            )
        elif current_course.status is not BasisStatus.SUPPORTED:
            decision = self._decision(
                DecisionOutcome.ABSTAIN, "CURRENT_BASIS_INVALIDATED", key,
                ReviewTrigger.BASIS_BROKEN, review_required=True,
            )
        elif not current_course.admitted:
            decision = self._decision(
                DecisionOutcome.ABSTAIN, "CURRENT_COURSE_NOT_ADMITTED", key,
                normalized_trigger, review_required=True,
            )
        elif normalized_trigger is ReviewTrigger.STALL and not self.stall_due:
            decision = self._decision(
                DecisionOutcome.CONTINUE, "STALL_THRESHOLD_NOT_REACHED", key,
                normalized_trigger, review_required=False,
            )
        elif normalized_trigger is ReviewTrigger.CHECKPOINT and not self.checkpoint_due:
            decision = self._decision(
                DecisionOutcome.CONTINUE, "CHECKPOINT_NOT_DUE", key,
                normalized_trigger, review_required=False,
            )
        else:
            decision = self._compare_admitted(current, alternatives, key, normalized_trigger)
        self._comparison_cache[key] = decision
        return decision

    def _compare_admitted(
        self,
        current: Candidate,
        alternatives: Sequence[Candidate],
        key: str,
        trigger: ReviewTrigger,
    ) -> Decision:
        eligible = [
            item for item in alternatives
            if item.course.status is BasisStatus.SUPPORTED
            and item.course.admitted
            and item.course.course_id not in self._departed
        ]
        if not eligible:
            return self._decision(DecisionOutcome.ABSTAIN, "NO_ADMITTED_ALTERNATIVE", key, trigger, review_required=True)
        if trigger is ReviewTrigger.ALTERNATIVE:
            return self._decision(DecisionOutcome.CONTINUE, "ALTERNATIVE_NOTE_ONLY", key, trigger, review_required=False)
        if self._steps_since_checkpoint < self._dwell_until_step:
            return self._decision(DecisionOutcome.CONTINUE, "SOFT_DWELL_ACTIVE", key, trigger, review_required=False)

        if any(item.remaining_loss.unit != current.remaining_loss.unit for item in eligible):
            return self._decision(DecisionOutcome.ABSTAIN, "LOSS_UNIT_MISMATCH", key, trigger, review_required=True)
        if any(item.remaining_loss.horizon != current.remaining_loss.horizon for item in eligible):
            return self._decision(DecisionOutcome.ABSTAIN, "LOSS_HORIZON_MISMATCH", key, trigger, review_required=True)
        ranked: list[tuple[float, Candidate]] = []
        for candidate in eligible:
            robust_gain = current.remaining_loss.low - candidate.remaining_loss.high - candidate.transition_loss.high
            ranked.append((robust_gain, candidate))
        if not ranked:
            return self._decision(DecisionOutcome.ABSTAIN, "NO_COMPARABLE_ALTERNATIVE", key, trigger, review_required=True)
        ranked.sort(key=lambda item: (-item[0], item[1].course.course_id))
        robust_gain, best = ranked[0]
        margin = max(self.absolute_margin, self.relative_margin * current.remaining_loss.low)
        if robust_gain > margin:
            return self._decision(DecisionOutcome.SWITCH, "ROBUST_GAIN_EXCEEDS_MARGIN", key, trigger,
                                  destination=best.course.course_id, review_required=True)
        if robust_gain > 0:
            return self._decision(DecisionOutcome.PROBE, "GAIN_WITHIN_UNCERTAINTY_MARGIN", key, trigger,
                                  review_required=True)
        return self._decision(DecisionOutcome.CONTINUE, "NO_ROBUST_GAIN", key, trigger, review_required=True)

    def _decision(
        self,
        outcome: DecisionOutcome,
        reason_code: str,
        key: str,
        trigger: ReviewTrigger,
        *,
        destination: str | None = None,
        review_required: bool,
    ) -> Decision:
        return Decision(
            outcome=outcome,
            reason_code=reason_code,
            current_course_id=self._current_id,
            destination_course_id=destination,
            trigger=trigger,
            comparison_key=key,
            review_required=review_required,
            no_progress_steps=self._no_progress_steps,
            steps_since_checkpoint=self._steps_since_checkpoint,
            events=tuple(self._events[-4:]),
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "goal_id": self.goal_id,
            "current_course_id": self._current_id,
            "courses": [self._courses[key].to_dict() for key in sorted(self._courses)],
            "departed_courses": list(self.departed_courses),
            "no_progress_steps": self._no_progress_steps,
            "steps_since_checkpoint": self._steps_since_checkpoint,
            "stall_k": self.stall_k,
            "lease_steps": self.lease_steps,
            "soft_dwell": self.soft_dwell,
            "dwell_until_step": self._dwell_until_step,
        }


__all__ = [
    "BasisStatus", "Candidate", "Course", "Decision", "DecisionOutcome",
    "LossBounds", "PolicyEvent", "ProgressReceipt", "ReviewTrigger", "SCHEMA",
    "SwitchPolicy", "SwitchPolicyError",
]
