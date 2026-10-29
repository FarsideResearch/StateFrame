"""Durable settled product-edit budget with a verification/recovery reserve."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from kernel_step_limits import reached, remaining, runtime_unbounded


MODE_ENV = "STATEFRAME_EDIT_CHURN_BUDGET"
SCHEMA = "stateframe-edit-churn-budget/1"
ALLOW = "ALLOW"
VERIFY_REQUIRED = "VERIFY_REQUIRED"
RECOVERY_REQUIRED = "RECOVERY_REQUIRED"
STOP_INCOMPLETE = "STOP_INCOMPLETE"
MAX_REQUESTS = 150
DEFAULT_DEADLINE_SECONDS = 3600.0
DEFAULT_RESERVE_SLOTS = 3


def enabled(runtime: object | None = None) -> bool:
    configured = getattr(runtime, "stateframe_edit_churn_budget_mode", None) if runtime is not None else None
    value = str(configured if configured is not None else os.environ.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value == "on"


def _field(value: object, name: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _as_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        return tuple(str(key) for key in value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(str(item) for item in value)
    return ()


@dataclass(frozen=True)
class ChurnState:
    run_id: str
    requests_consumed: int = 0
    elapsed_seconds: float = 0.0
    product_transactions: int = 0
    speculative_transactions: int = 0
    no_op_count: int = 0
    refused_count: int = 0
    read_count: int = 0
    evidence_increases: int = 0
    public_green_candidate: bool = False
    grounded_evidence: tuple[str, ...] = ()
    settled_effects: tuple[str, ...] = ()
    settled_transactions: tuple[str, ...] = ()
    last_decision: str = ALLOW
    deadline_seconds: float = DEFAULT_DEADLINE_SECONDS
    configured_limit: int | None = MAX_REQUESTS


@dataclass(frozen=True)
class ChargeReceipt:
    schema: str
    effect_id: str
    product_changed: bool
    evidence_increased: bool
    transaction_counted: bool
    status: str


def _evidence_keys(delta: object) -> tuple[str, ...]:
    if isinstance(delta, Mapping):
        keys: list[str] = []
        for key, value in delta.items():
            identifier = str(key)
            if isinstance(value, Mapping):
                if value.get("observed") is False or value.get("observation_count", 1) == 0:
                    continue
                definition = str(value.get("definition_hash", value.get("check_hash", "")) or "")
                keys.append(identifier + "@" + definition)
            elif value not in (None, "", False, "UNKNOWN", "SKIPPED", "REFUSED", "TIMEOUT"):
                keys.append(identifier)
        return tuple(dict.fromkeys(keys))
    if isinstance(delta, (list, tuple, set, frozenset)):
        return tuple(dict.fromkeys(str(item) for item in delta if str(item)))
    return ()


class EditChurnBudget:
    """State survives restart, checkpoint, and recovery by explicit snapshot."""

    def __init__(self, run_id: object, *, configured_limit: int | None = MAX_REQUESTS,
                 deadline_seconds: float = DEFAULT_DEADLINE_SECONDS, reserve_slots: int = DEFAULT_RESERVE_SLOTS):
        limit = None if configured_limit is None else int(configured_limit)
        if limit is not None and limit < 0:
            raise ValueError("configured_limit must be non-negative")
        if float(deadline_seconds) <= 0:
            raise ValueError("deadline_seconds must be positive")
        self.reserve_slots = max(0, int(reserve_slots))
        self._state = ChurnState(
            str(run_id or ""), deadline_seconds=float(deadline_seconds),
            configured_limit=None if limit is None else min(limit, MAX_REQUESTS),
        )

    @property
    def state(self) -> ChurnState:
        return self._state

    @property
    def effective_limit(self) -> int | None:
        return self._state.configured_limit

    @property
    def remaining_requests(self) -> int | None:
        return remaining(self.effective_limit, self._state.requests_consumed)

    def mark_public_green_candidate(self) -> None:
        self._state = self._replace(public_green_candidate=True)

    def _replace(self, **changes: object) -> ChurnState:
        values = {
            "run_id": self._state.run_id,
            "requests_consumed": self._state.requests_consumed,
            "elapsed_seconds": self._state.elapsed_seconds,
            "product_transactions": self._state.product_transactions,
            "speculative_transactions": self._state.speculative_transactions,
            "no_op_count": self._state.no_op_count,
            "refused_count": self._state.refused_count,
            "read_count": self._state.read_count,
            "evidence_increases": self._state.evidence_increases,
            "public_green_candidate": self._state.public_green_candidate,
            "grounded_evidence": self._state.grounded_evidence,
            "settled_effects": self._state.settled_effects,
            "settled_transactions": self._state.settled_transactions,
            "last_decision": self._state.last_decision,
            "deadline_seconds": self._state.deadline_seconds,
            "configured_limit": self._state.configured_limit,
        }
        values.update(changes)
        return ChurnState(**values)

    def charge(
            self,
            effect_id: object,
            product_changed: bool,
            evidence_delta: object,
            elapsed: object,
            *,
            transaction_id: object = "",
            action: str = "",
            refused: bool = False,
    ) -> ChargeReceipt:
        effect = str(effect_id or "")
        if not effect:
            raise ValueError("settled charge requires effect_id")
        if effect in self._state.settled_effects:
            return ChargeReceipt(SCHEMA, effect, bool(product_changed), False, False, "DUPLICATE")
        keys = _evidence_keys(evidence_delta)
        old_keys = set(self._state.grounded_evidence)
        new_keys = tuple(key for key in keys if key not in old_keys)
        evidence_increased = bool(new_keys)
        tx = str(transaction_id or effect)
        count_transaction = bool(product_changed) and tx not in self._state.settled_transactions
        no_op = not product_changed and str(action).lower() not in {"read", "inspect", "grep", "list_files", ""}
        read = str(action).lower() in {"read", "inspect", "grep", "list_files"}
        refused = bool(refused) or str(action).lower() in {"refused", "refusal", "denied", "deny"}
        decision = self._state.last_decision
        self._state = self._replace(
            elapsed_seconds=self._state.elapsed_seconds + max(0.0, float(elapsed or 0.0)),
            product_transactions=self._state.product_transactions + int(count_transaction),
            speculative_transactions=(
                self._state.speculative_transactions + int(count_transaction and self._state.public_green_candidate and not evidence_increased)
            ),
            no_op_count=self._state.no_op_count + int(no_op),
            refused_count=self._state.refused_count + int(refused),
            read_count=self._state.read_count + int(read),
            evidence_increases=self._state.evidence_increases + int(evidence_increased),
            grounded_evidence=tuple(dict.fromkeys((*self._state.grounded_evidence, *new_keys))),
            settled_effects=(*self._state.settled_effects, effect),
            settled_transactions=(
                (*self._state.settled_transactions, tx) if count_transaction else self._state.settled_transactions
            ),
            last_decision=decision,
        )
        return ChargeReceipt(SCHEMA, effect, bool(product_changed), evidence_increased, count_transaction,
                             "CHARGED" if count_transaction or evidence_increased else "SETTLED")

    def _reserve_feasible(self, reserves: object) -> bool:
        remaining = _field(reserves, "response_slots_remaining", _field(reserves, "remaining_requests", self.remaining_requests))
        try:
            remaining_slots = None if remaining is None and self.effective_limit is None else int(remaining)
        except (TypeError, ValueError):
            return False
        required_slots = int(_field(reserves, "required_response_reserve", self.reserve_slots))
        remaining_seconds = _field(reserves, "remaining_seconds", None)
        p95 = sum(float(_field(reserves, name, 0.0) or 0.0)
                  for name in ("p95_verification_seconds", "p95_recovery_seconds", "p95_export_seconds"))
        if remaining_seconds is None:
            remaining_seconds = self._state.deadline_seconds - self._state.elapsed_seconds
        try:
            return (remaining_slots is None or remaining_slots >= required_slots) and float(remaining_seconds) >= p95
        except (TypeError, ValueError):
            return False

    def admit(self, action: object, transaction: object, reserves: object) -> str:
        name = str(action or _field(transaction, "action", "")).strip().lower()
        current_failure = bool(_field(transaction, "current_public_failure", False))
        recovery_available = bool(_field(transaction, "recovery_available", False))
        evidence_delta = _field(transaction, "evidence_delta", {})
        grounded = bool(_evidence_keys(evidence_delta))
        recovery_action = name in {"restore_candidate", "recover_candidate", "recovery"} or bool(_field(transaction, "recovery", False))
        settled = bool(_field(transaction, "settled", True))
        if not settled:
            decision = VERIFY_REQUIRED
        elif current_failure and recovery_available and not recovery_action:
            decision = RECOVERY_REQUIRED
        elif recovery_action:
            decision = ALLOW
        elif reached(self.effective_limit, self._state.requests_consumed):
            decision = STOP_INCOMPLETE
        elif self._state.elapsed_seconds >= self._state.deadline_seconds:
            decision = STOP_INCOMPLETE
        elif self._state.public_green_candidate and self._state.speculative_transactions >= 2 and not grounded:
            decision = VERIFY_REQUIRED
        elif self._state.public_green_candidate and not self._reserve_feasible(reserves) and not grounded:
            decision = STOP_INCOMPLETE
        else:
            decision = ALLOW
        self._state = self._replace(last_decision=decision)
        return decision

    def note_request(self, count: int = 1) -> None:
        self._state = self._replace(requests_consumed=self._state.requests_consumed + max(0, int(count)))

    def snapshot(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "reserve_slots": self.reserve_slots,
            "state": {
                "run_id": self._state.run_id,
                "requests_consumed": self._state.requests_consumed,
                "elapsed_seconds": self._state.elapsed_seconds,
                "product_transactions": self._state.product_transactions,
                "speculative_transactions": self._state.speculative_transactions,
                "no_op_count": self._state.no_op_count,
                "refused_count": self._state.refused_count,
                "read_count": self._state.read_count,
                "evidence_increases": self._state.evidence_increases,
                "public_green_candidate": self._state.public_green_candidate,
                "grounded_evidence": list(self._state.grounded_evidence),
                "settled_effects": list(self._state.settled_effects),
                "settled_transactions": list(self._state.settled_transactions),
                "last_decision": self._state.last_decision,
                "deadline_seconds": self._state.deadline_seconds,
                "configured_limit": self._state.configured_limit,
            },
        }

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, object]) -> "EditChurnBudget":
        state = snapshot.get("state") if isinstance(snapshot, Mapping) else None
        if not isinstance(state, Mapping):
            raise ValueError("churn snapshot is missing state")
        budget = cls(
            state.get("run_id", ""),
            configured_limit=state.get("configured_limit", MAX_REQUESTS),
            deadline_seconds=float(state.get("deadline_seconds", DEFAULT_DEADLINE_SECONDS)),
            reserve_slots=int(snapshot.get("reserve_slots", DEFAULT_RESERVE_SLOTS)),
        )
        budget._state = ChurnState(
            run_id=str(state.get("run_id", "")),
            requests_consumed=int(state.get("requests_consumed", 0)),
            elapsed_seconds=float(state.get("elapsed_seconds", 0.0)),
            product_transactions=int(state.get("product_transactions", 0)),
            speculative_transactions=int(state.get("speculative_transactions", 0)),
            no_op_count=int(state.get("no_op_count", 0)),
            refused_count=int(state.get("refused_count", 0)),
            read_count=int(state.get("read_count", 0)),
            evidence_increases=int(state.get("evidence_increases", 0)),
            public_green_candidate=bool(state.get("public_green_candidate", False)),
            grounded_evidence=tuple(str(item) for item in state.get("grounded_evidence", ()) or ()),
            settled_effects=tuple(str(item) for item in state.get("settled_effects", ()) or ()),
            settled_transactions=tuple(str(item) for item in state.get("settled_transactions", ()) or ()),
            last_decision=str(state.get("last_decision", ALLOW)),
            deadline_seconds=float(state.get("deadline_seconds", DEFAULT_DEADLINE_SECONDS)),
            configured_limit=budget.effective_limit,
        )
        return budget


def budget_for(runtime: object) -> EditChurnBudget:
    existing = getattr(runtime, "edit_churn_budget", None)
    if isinstance(existing, EditChurnBudget):
        return existing
    run_id = getattr(getattr(runtime, "journal", None), "run_id", None) or getattr(runtime, "run_id", "runtime")
    limit = (None if runtime_unbounded(runtime) else
             int(getattr(runtime, "max_steps", MAX_REQUESTS) or MAX_REQUESTS))
    budget = EditChurnBudget(run_id, configured_limit=limit)
    try:
        runtime.edit_churn_budget = budget
    except AttributeError:
        pass
    return budget


def charge(runtime: object, effect_id: object, product_changed: bool, evidence_delta: object, elapsed: object,
           *, transaction_id: object = "", action: str = "", refused: bool = False) -> ChargeReceipt:
    return budget_for(runtime).charge(effect_id, product_changed, evidence_delta, elapsed,
                                      transaction_id=transaction_id, action=action, refused=refused)


def admit(action: object, transaction: object, reserves: object, *, runtime: object | None = None) -> str:
    budget = budget_for(runtime) if runtime is not None else None
    if budget is None:
        return STOP_INCOMPLETE
    return budget.admit(action, transaction, reserves)


def observe_settled_action(runtime: object, record: Mapping[str, object]) -> str | None:
    """Consume one settled action and hand its single decision to completion owner."""
    if not enabled(runtime) or not isinstance(record, Mapping):
        return None
    tool = str(record.get("tool", record.get("action", "")) or "").strip().lower()
    result = str(record.get("result", record.get("result_class", "")) or "").upper()
    product_changed = tool in {"edit", "str_replace", "file_create", "replace_module", "write", "replace"} and result in {
        "APPLIED", "PASS", "SUCCESS", "OK",
    }
    evidence_delta = record.get("evidence_delta", record.get("new_evidence", {}))
    transaction = {
        "action": tool,
        "settled": True,
        "product_changed": product_changed,
        "evidence_delta": evidence_delta,
        "current_public_failure": str(record.get("public_verdict", "")).upper() in {"RED", "FAIL"},
        "recovery_available": bool(record.get("recovery_available", False)),
        "recovery": tool in {"restore_candidate", "recover_candidate"},
    }
    budget = budget_for(runtime)
    if tool in {"run_tests", "test", "pytest"} and result in {"PASS", "PASSED", "GREEN", "SUCCESS", "OK"}:
        if bool(record.get("model_owned_candidate", False)) or bool(getattr(runtime, "_candidate_last_captured", None)):
            budget.mark_public_green_candidate()
    effect_id = record.get("effect_id", record.get("step_record_id", ""))
    charge_receipt = budget.charge(
        effect_id, product_changed, evidence_delta,
        record.get("elapsed", record.get("elapsed_seconds", 0.0)),
        transaction_id=record.get("transaction_id", ""), action=tool,
        refused=result in {"REFUSED", "REJECTED", "DENIED"},
    )
    reserves = getattr(runtime, "edit_churn_reserves", None) or {
        "response_slots_remaining": budget.remaining_requests,
        "remaining_seconds": budget.state.deadline_seconds - budget.state.elapsed_seconds,
    }
    decision = budget.admit(tool, transaction, reserves)
    setattr(runtime, "_edit_churn_last_charge", charge_receipt)
    setattr(runtime, "_edit_churn_last_decision", decision)
    try:
        from kernel_completion_owner import handoff_churn_decision
        handoff_churn_decision(runtime, decision)
    except (ImportError, AttributeError, TypeError, ValueError):
        pass
    return decision


__all__ = [
    "ALLOW", "ChurnState", "ChargeReceipt", "EditChurnBudget", "MAX_REQUESTS",
    "RECOVERY_REQUIRED", "SCHEMA", "STOP_INCOMPLETE", "VERIFY_REQUIRED", "admit", "budget_for",
    "charge", "enabled", "observe_settled_action",
]
