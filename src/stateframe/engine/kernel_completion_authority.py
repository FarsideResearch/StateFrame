"""Opt-in completion authority completion authority.

The older completion reducer answers one question: *may this done attempt be
admitted?* This module owns the separate question of *what must happen next,
and when must the host stop waiting?* It is intentionally self contained so
the legacy reducer and its serialized OFF shape remain unchanged.

``STATEFRAME_COMPLETION_V2=on`` enables this protocol. The host, rather than
the model, owns every terminal transition. A blocker is useful only when its
registered progress contract says which action can produce a result that is
actually capable of resolving that blocker.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Union

try:  # Python 3.9 compatibility for the type-only alias marker.
    from typing import TypeAlias
except ImportError:  # pragma: no cover - exercised by the repository's 3.9 runner.
    TypeAlias = object


MODE_ENV = "STATEFRAME_COMPLETION_V2"
SCHEMA = "completion-authority/1"
POLICY_VERSION = "completion-authority-policy/1"


def mode() -> str:
    """Return the strict opt-in mode for completion authority."""
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


class TerminalState(str, Enum):
    """The complete terminal vocabulary owned by the host."""

    ACTIVE = "ACTIVE"
    ACCEPTED = "ACCEPTED"
    FAILED_UNVERIFIED = "FAILED_UNVERIFIED"
    FAILED_UNRESOLVED_DEFECT = "FAILED_UNRESOLVED_DEFECT"
    FAILED_BUDGET = "FAILED_BUDGET"
    FAILED_CONTROL_CONFLICT = "FAILED_CONTROL_CONFLICT"
    FAILED_DEADLINE = "FAILED_DEADLINE"

    @property
    def terminal(self) -> bool:
        return self is not TerminalState.ACTIVE


ACTIVE = TerminalState.ACTIVE.value
ACCEPTED = TerminalState.ACCEPTED.value
FAILED_UNVERIFIED = TerminalState.FAILED_UNVERIFIED.value
FAILED_UNRESOLVED_DEFECT = TerminalState.FAILED_UNRESOLVED_DEFECT.value
FAILED_BUDGET = TerminalState.FAILED_BUDGET.value
FAILED_CONTROL_CONFLICT = TerminalState.FAILED_CONTROL_CONFLICT.value
FAILED_DEADLINE = TerminalState.FAILED_DEADLINE.value
TERMINAL_STATES = frozenset(item.value for item in TerminalState if item.terminal)
NO_HOST_CHECKS = "NO_HOST_CHECKS"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False, default=str,
    ).encode("utf-8")


def _digest(value: object) -> str:
    if isinstance(value, str) and value.startswith("sha256:"):
        return value
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _tuple_strings(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if not isinstance(value, Iterable) or isinstance(value, (bytes, bytearray)):
        return ()
    return tuple(dict.fromkeys(str(item) for item in value if str(item).strip()))


def _base_reason_code(value: object) -> str:
    """Strip a selector/detail suffix from a host reason code."""
    text = str(value or "").strip().upper()
    return text.split(":", 1)[0]


@dataclass(frozen=True)
class ProductViolation:
    """A witnessed defect in the candidate product."""

    reason_code: str = "PRODUCT_VIOLATION"
    obligation_id: str = "product"
    detail: str = ""
    candidate_digest: str | None = None
    evidence_refs: tuple[str, ...] = ()

    @property
    def kind(self) -> str:
        return "ProductViolation"

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason_code", _base_reason_code(self.reason_code))
        object.__setattr__(self, "obligation_id", str(self.obligation_id or "product"))
        object.__setattr__(self, "detail", str(self.detail or ""))
        object.__setattr__(self, "evidence_refs", _tuple_strings(self.evidence_refs))


@dataclass(frozen=True)
class RequiredEvidenceGap:
    """Evidence is missing, stale, failed, or must be rechecked."""

    reason_code: str = "REQUIRED_EVIDENCE_GAP"
    obligation_id: str = "evidence"
    detail: str = ""
    candidate_digest: str | None = None
    evidence_refs: tuple[str, ...] = ()

    @property
    def kind(self) -> str:
        return "RequiredEvidenceGap"

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason_code", _base_reason_code(self.reason_code))
        object.__setattr__(self, "obligation_id", str(self.obligation_id or "evidence"))
        object.__setattr__(self, "detail", str(self.detail or ""))
        object.__setattr__(self, "evidence_refs", _tuple_strings(self.evidence_refs))


@dataclass(frozen=True)
class ControlFault:
    """A host/control-plane fault, not a product defect."""

    reason_code: str = "CONTROL_FAULT"
    obligation_id: str = "control"
    detail: str = ""
    candidate_digest: str | None = None
    evidence_refs: tuple[str, ...] = ()

    @property
    def kind(self) -> str:
        return "ControlFault"

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason_code", _base_reason_code(self.reason_code))
        object.__setattr__(self, "obligation_id", str(self.obligation_id or "control"))
        object.__setattr__(self, "detail", str(self.detail or ""))
        object.__setattr__(self, "evidence_refs", _tuple_strings(self.evidence_refs))


BlockingCause: TypeAlias = Union[ProductViolation, RequiredEvidenceGap, ControlFault]
CAUSE_TYPES = (ProductViolation, RequiredEvidenceGap, ControlFault)


# These are the host action effects, not claims that an action succeeded.  An
# action result must be observed and ingested before the matching effect can
# resolve a contract.
ACTION_RESULT_EFFECTS: dict[str, frozenset[str]] = {
    "edit": frozenset({"product_fixed", "candidate_revision"}),
    "str_replace": frozenset({"product_fixed", "candidate_revision"}),
    "file_create": frozenset({"product_fixed", "candidate_revision"}),
    "file_delete": frozenset({"product_fixed", "candidate_revision"}),
    "file_move": frozenset({"product_fixed", "candidate_revision"}),
    "replace_module": frozenset({"product_fixed", "candidate_revision"}),
    "write": frozenset({"product_fixed", "candidate_revision"}),
    "replace": frozenset({"product_fixed", "candidate_revision"}),
    "run_tests": frozenset({
        "current_pass", "evidence_current", "recheck_cleared", "receipt_current",
    }),
    "work_update": frozenset({"work_closed", "work_record_current"}),
    "work": frozenset({"work_closed", "work_record_current"}),
    "read": frozenset({"source_current", "source_observed"}),
    "compile_checks": frozenset({"checks_compiled", "control_reconciled"}),
    "host_finalize": frozenset({"terminalized"}),
}
_LEGACY_ACTION_ALIASES = {
    "control_reconcile": "run_tests",
    "answer_question": "read",
}


@dataclass(frozen=True)
class ProgressContract:
    """The only permitted route from one refusal reason to progress.

    ``resolution_predicate`` is a named host predicate.  The registry's
    static validator checks that the named predicate is among the effects the
    permitted action can produce.  Runtime ingestion additionally checks the
    concrete action result and its status.
    """

    reason_code: str
    cause_type: type[BlockingCause]
    resolution_predicate: str
    recovery_owner: str
    permitted_action: str
    preconditions: tuple[str, ...] = ()
    result_ingestion_path: str = "action_results"
    root_budget_key: str = "repairs"
    attempt_limit: int = 1
    expiration: float | None = None
    terminal_fallback: str = FAILED_UNRESOLVED_DEFECT
    result_kinds: tuple[str, ...] = ()
    advisory: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason_code", _base_reason_code(self.reason_code))
        object.__setattr__(self, "permitted_action", str(self.permitted_action).strip().lower())
        object.__setattr__(self, "resolution_predicate", str(self.resolution_predicate).strip())
        object.__setattr__(self, "recovery_owner", str(self.recovery_owner).strip())
        object.__setattr__(self, "preconditions", _tuple_strings(self.preconditions))
        object.__setattr__(self, "result_ingestion_path", str(self.result_ingestion_path).strip())
        object.__setattr__(self, "root_budget_key", str(self.root_budget_key).strip())
        object.__setattr__(self, "attempt_limit", int(self.attempt_limit))
        object.__setattr__(self, "terminal_fallback", str(self.terminal_fallback))
        object.__setattr__(self, "result_kinds", _tuple_strings(self.result_kinds))

    def result_feeds_resolution(self, result: object) -> bool:
        """Whether this concrete result can change this contract's predicate."""
        if isinstance(result, Mapping):
            action = str(result.get("action", result.get("tool", ""))).strip().lower()
            if action != self.permitted_action:
                return False
            explicit = result.get("feeds", result.get("effects", ()))
            if isinstance(explicit, str):
                explicit = (explicit,)
            if isinstance(explicit, Iterable) and not isinstance(explicit, (bytes, bytearray)):
                if (self.resolution_predicate in {str(item) for item in explicit}
                        and str(result.get("status", result.get("result_class", ""))).upper()
                        not in {"FAIL", "FAILED", "TIMEOUT", "TIMED_OUT", "ERROR", "RED", "REFUSED"}):
                    return True
            kind = str(result.get("result_kind", result.get("kind", ""))).strip().lower()
            status = str(result.get("status", result.get("result_class", result.get("result", "")))).upper()
            effects = _observed_effects(action, kind, status, result)
            if self.result_kinds and kind not in {item.lower() for item in self.result_kinds}:
                return False
            return self.resolution_predicate in effects
        return False


def _observed_effects(action: str, kind: str, status: str, result: Mapping[str, object]) -> frozenset[str]:
    if status in {"FAIL", "FAILED", "TIMEOUT", "TIMED_OUT", "ERROR", "RED", "REFUSED", "UNRESOLVED"}:
        return frozenset()
    if kind in {"edit_applied", "candidate_revision"} or (
        action in {"edit", "str_replace", "file_create", "file_delete", "file_move", "replace_module", "write", "replace"}
        and status in {"APPLIED", "SUCCESS", "PASS", "OK"}
    ):
        return ACTION_RESULT_EFFECTS.get(action, frozenset())
    if kind in {"current_pass", "receipt_current", "recheck_cleared"} or (
        action == "run_tests" and (status in {"PASS", "PASSED", "GREEN", "SUCCESS", "OK"}
                                    or result.get("exit_code") == 0)
    ):
        return ACTION_RESULT_EFFECTS["run_tests"]
    if kind in {"work_closed", "work_record_current"} or (
        action in {"work", "work_update"} and status in {"APPLIED", "CLOSED", "SUCCESS", "OK"}
    ):
        return ACTION_RESULT_EFFECTS.get(action, ACTION_RESULT_EFFECTS["work_update"])
    if kind in {"control_reconciled", "checks_compiled"} or (
        action in {"control_reconcile", "compile_checks"}
        and status in {"APPLIED", "SUCCESS", "OK", "PASS", "PASSED"}
    ):
        return ACTION_RESULT_EFFECTS.get(action, frozenset())
    if kind in {"source_current", "source_observed"}:
        return ACTION_RESULT_EFFECTS.get(action, frozenset())
    if kind == "question_recorded":
        return ACTION_RESULT_EFFECTS.get(action, frozenset())
    return frozenset()


_PRODUCT_REASONS = {
    "PRODUCT_VIOLATION", "UNRESOLVED_DEFECT", "TESTS_NOT_GREEN", "NONZERO_EXIT",
    "COUNTEREXAMPLE", "OPEN_BLOCKER", "DECLARED_WORK", "INWORLD_OBLIGATION",
    "STUB_GATE", "PROBE_GATE", "CONTRACT_AUDIT", "ORACLE_TESTS_RED",
}
_EVIDENCE_REASONS = {
    "REQUIRED_EVIDENCE_GAP", "RECHECK_REQUIRED", "STALE_EVIDENCE", "STALE",
    "MISSING_SELECTOR", "MISSING_CHECKS", "NOT_EXECUTED", "NO_VERIFIER",
    "AMBIGUOUS_CONTRACT", "UNBOUND_ENTRYPOINT", "UNBOUND_INVOCATION", "SKIPPED",
    "CONFLICTING_RECEIPTS", "CONFLICTING_RESULTS", "SELECTOR_ERROR", "VERIFY",
    "TESTS_NOT_RUN", "SOURCE_EVIDENCE", "READ_EVIDENCE", "EVIDENCE_GAP",
    "QUESTION_CLAUSE", "BOLT_GATE", "CONTINUATION_BLOCKER", "SIDEKICK_CHALLENGE",
    "ORACLE_UNAVAILABLE", "V2_DISPOSITION", "V2_TERMINAL", "LATE_VETO",
    "BASIS_CHANGED", "COMPLETION_SEALED", "NEEDS_WORK", "CLOSURE_UNAFFORDABLE",
}
_CONTROL_REASONS = {
    "CONTROL_FAULT", "CONTROL_CONFLICT", "NO_FEASIBLE_REPAIR_ACTION",
    "NO_FEASIBLE_RECOVERY", "COMMIT", "COMMIT_FAILED", "CONTINUATION_BLOCKED",
    "HARNESS_ERROR", "SETUP_ERROR", "FUEL_EXHAUSTED", "BUDGET_EXHAUSTED",
    "DEADLINE", "UNVERIFIED", "HOST_OBLIGATION", "PUBLIC_REVIEW", "TURTLE_DEFECT",
    "COMPLETION_BLOCKED",
    "ADMITTED", "ADMITTED_BASIS_UNCHANGED",
}


def _contract_for_reason(reason: str) -> ProgressContract:
    base = _base_reason_code(reason)
    if base in _PRODUCT_REASONS:
        return ProgressContract(
            base, ProductViolation, "product_fixed", "repair", "edit",
            preconditions=("product_path_admitted",), root_budget_key="repairs",
            terminal_fallback=FAILED_UNRESOLVED_DEFECT,
        )
    if base in {"OPEN_WORK", "DECLARED_WORK"}:
        return ProgressContract(
            base, ProductViolation, "work_closed", "work-record", "work_update",
            root_budget_key="repairs", terminal_fallback=FAILED_UNRESOLVED_DEFECT,
        )
    if base == "QUESTION_CLAUSE":
        return ProgressContract(
            base, RequiredEvidenceGap, "source_observed", "evidence", "read",
            root_budget_key="model_calls", terminal_fallback=FAILED_UNVERIFIED, advisory=True,
        )
    if base in _EVIDENCE_REASONS:
        return ProgressContract(
            base, RequiredEvidenceGap, "evidence_current", "verification", "run_tests",
            root_budget_key="verifications", terminal_fallback=FAILED_UNVERIFIED,
        )
    return ProgressContract(
        base, ControlFault, "evidence_current", "host-check", "run_tests",
        root_budget_key="repairs", terminal_fallback=FAILED_CONTROL_CONFLICT,
    )


# This list is deliberately explicit.  It is the static audit surface for
# reason codes emitted by the opt-in authority and includes the legacy names
# that can be translated into a typed cause at the boundary.
_REFUSAL_REASON_CODES = frozenset(
    _PRODUCT_REASONS | _EVIDENCE_REASONS | _CONTROL_REASONS
    | {"OPEN_WORK", "REQUIRED_EVIDENCE_GAP", "PRODUCT_VIOLATION", "CONTROL_FAULT"}
)


class ProgressContractRegistry(Mapping[str, ProgressContract]):
    """Registry with a static proof that every refusal has a live route."""

    def __init__(self, contracts: Iterable[ProgressContract] = ()) -> None:
        self._contracts: dict[str, ProgressContract] = {}
        for contract in contracts:
            self.register(contract)

    @classmethod
    def default(cls) -> "ProgressContractRegistry":
        return cls(_contract_for_reason(code) for code in sorted(_REFUSAL_REASON_CODES))

    def register(self, contract: ProgressContract) -> None:
        if not isinstance(contract, ProgressContract):
            raise TypeError("progress registry accepts ProgressContract values")
        self._contracts[contract.reason_code] = contract

    def contract_for(self, reason_code: object) -> ProgressContract | None:
        return self._contracts.get(_base_reason_code(reason_code))

    def require(self, reason_code: object) -> ProgressContract:
        contract = self.contract_for(reason_code)
        if contract is None:
            raise KeyError(_base_reason_code(reason_code))
        return contract

    def __getitem__(self, key: str) -> ProgressContract:
        return self._contracts[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._contracts)

    def __len__(self) -> int:
        return len(self._contracts)

    def reason_codes(self) -> tuple[str, ...]:
        """Return the registry's statically auditable refusal vocabulary."""
        return tuple(sorted(self._contracts))

    def validate(self) -> tuple[str, ...]:
        errors: list[str] = []
        for code in sorted(_REFUSAL_REASON_CODES):
            contract = self._contracts.get(code)
            if contract is None:
                errors.append(f"{code}: missing progress contract")
                continue
            if not isinstance(contract.cause_type, type) or contract.cause_type not in CAUSE_TYPES:
                errors.append(f"{code}: cause type is not a BlockingCause type")
            if not contract.permitted_action:
                errors.append(f"{code}: missing permitted action")
            effects = ACTION_RESULT_EFFECTS.get(contract.permitted_action, frozenset())
            if contract.resolution_predicate not in effects:
                errors.append(
                    f"{code}: action {contract.permitted_action!r} cannot produce "
                    f"predicate {contract.resolution_predicate!r}"
                )
            if not contract.result_ingestion_path:
                errors.append(f"{code}: missing RESULT_INGESTION_PATH")
            if contract.attempt_limit < 1:
                errors.append(f"{code}: attempt limit must be positive")
            if contract.terminal_fallback not in TERMINAL_STATES:
                errors.append(f"{code}: terminal fallback is not terminal")
        return tuple(errors)


def refusal_reason_codes() -> tuple[str, ...]:
    return tuple(sorted(_REFUSAL_REASON_CODES))


def validate_progress_contracts(
    registry: ProgressContractRegistry | Mapping[str, ProgressContract] | None = None,
) -> tuple[str, ...]:
    if registry is None:
        registry = ProgressContractRegistry.default()
    if isinstance(registry, ProgressContractRegistry):
        return registry.validate()
    return ProgressContractRegistry(registry.values()).validate()


def assert_progress_contracts(
    registry: ProgressContractRegistry | Mapping[str, ProgressContract] | None = None,
) -> None:
    errors = validate_progress_contracts(registry)
    if errors:
        raise AssertionError("invalid progress contracts: " + "; ".join(errors))


DEFAULT_PROGRESS_CONTRACTS = ProgressContractRegistry.default()
PROGRESS_CONTRACTS = DEFAULT_PROGRESS_CONTRACTS


class FuelExhausted(RuntimeError):
    """Raised only when a dispatch cannot preserve terminalization reserve."""


_FUEL_ALIASES = {
    "model": "model_calls", "model_call": "model_calls", "model_calls": "model_calls",
    "verification": "verifications", "verify": "verifications", "verifications": "verifications",
    "repair": "repairs", "repairs": "repairs", "edit": "repairs",
    "compile": "check_compilation", "check_compile": "check_compilation",
    "check_compilation": "check_compilation", "compilation": "check_compilation",
    "terminal": "terminalization", "terminalization": "terminalization",
}
_FUEL_KEYS = ("model_calls", "verifications", "repairs", "check_compilation")


class RootFuelBudget:
    """A single monotone fuel ledger for one logical run.

    The per-kind counters are accounting labels; ``total_remaining`` is the
    authority.  ``terminalization_reserve`` cannot be spent by a model,
    verifier, repair, or compiler dispatch. Explicit unbounded fuel uses None
    for absent limits while retaining spent counters and terminal reserves.
    """

    def __init__(
        self,
        limits: Mapping[str, int] | None = None,
        *,
        total: int | None = None,
        unbounded: bool = False,
        model_calls: int | None = None,
        verifications: int | None = None,
        repairs: int | None = None,
        check_compilation: int | None = None,
        terminalization_reserve: int = 1,
        root_budget_id: str | None = None,
    ) -> None:
        if isinstance(limits, int):
            if total is not None:
                raise TypeError("fuel total was supplied twice")
            total = limits
            limits = None
        reserve = int(terminalization_reserve)
        if reserve < 0:
            raise ValueError("terminalization reserve must be non-negative")
        self.root_budget_id = root_budget_id or _digest({"created": time.time_ns()})
        self._terminalization_reserve_remaining = reserve
        self._shared_limit = total is not None
        self._unbounded = unbounded
        if unbounded:
            if total is not None or limits or any(value is not None for value in (
                    model_calls, verifications, repairs, check_compilation)):
                raise ValueError("unbounded fuel cannot also have finite allocations")
            self._initial = {key: None for key in _FUEL_KEYS}
            self._remaining = dict(self._initial)
            self._total_initial = None
        elif total is not None:
            total_value = int(total)
            if total_value < reserve or total_value < 0:
                raise ValueError("root total must cover terminalization reserve")
            self._initial = {key: total_value for key in _FUEL_KEYS}
            self._remaining = dict(self._initial)
            self._total_initial = total_value
        else:
            supplied: dict[str, int] = {}
            if limits:
                for raw_key, value in limits.items():
                    key = _FUEL_ALIASES.get(str(raw_key).strip().lower(), str(raw_key).strip())
                    if key not in _FUEL_KEYS:
                        raise KeyError(f"unknown root fuel key: {raw_key!r}")
                    supplied[key] = value
            for key, value in {
                "model_calls": model_calls,
                "verifications": verifications,
                "repairs": repairs,
                "check_compilation": check_compilation,
            }.items():
                if value is not None:
                    supplied[key] = value
            self._initial = {key: int(supplied.get(key, 0)) for key in _FUEL_KEYS}
            if any(value < 0 for value in self._initial.values()):
                raise ValueError("fuel limits must be non-negative")
            self._remaining = dict(self._initial)
            self._total_initial = sum(self._initial.values())
            if self._total_initial < reserve:
                raise ValueError("fuel limits must cover terminalization reserve")
        self._total_remaining = self._total_initial
        self._spent = {key: 0 for key in _FUEL_KEYS}
        self._spent_terminalization = 0

    @classmethod
    def from_total(cls, total: int, *, terminalization_reserve: int = 1) -> "RootFuelBudget":
        return cls(total=int(total), terminalization_reserve=terminalization_reserve)

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, object]) -> "RootFuelBudget":
        """Restore a root ledger without recomputing or replenishing it."""
        if not isinstance(snapshot, Mapping):
            raise TypeError("root fuel snapshot must be a mapping")
        initial_raw = snapshot.get("initial")
        remaining_raw = snapshot.get("remaining_by_kind")
        spent_raw = snapshot.get("spent_by_kind")
        if not isinstance(initial_raw, Mapping) or not isinstance(remaining_raw, Mapping):
            raise ValueError("root fuel snapshot is missing its ledger")
        if snapshot.get("unbounded") is True:
            if (any(initial_raw.get(key, 0) is not None or remaining_raw.get(key, 0) is not None
                    for key in _FUEL_KEYS)
                    or snapshot.get("total_initial") is not None
                    or snapshot.get("total_remaining") is not None
                    or not isinstance(spent_raw, Mapping)):
                raise ValueError("invalid unbounded root fuel snapshot")
            reserve = snapshot.get("terminalization_reserve")
            terminal_spent = snapshot.get("terminalization_spent")
            values = [reserve, terminal_spent, *(spent_raw.get(key) for key in _FUEL_KEYS)]
            if any(type(value) is not int or value < 0 for value in values):
                raise ValueError("root fuel snapshot is not monotone")
            restored = cls(unbounded=True, terminalization_reserve=reserve,
                           root_budget_id=str(snapshot.get("root_budget_id", "")) or _digest(snapshot))
            restored._spent = {key: spent_raw[key] for key in _FUEL_KEYS}
            restored._spent_terminalization = terminal_spent
            return restored
        initial = {key: int(initial_raw.get(key, 0)) for key in _FUEL_KEYS}
        remaining = {key: int(remaining_raw.get(key, 0)) for key in _FUEL_KEYS}
        spent = ({key: int(spent_raw.get(key, 0)) for key in _FUEL_KEYS}
                 if isinstance(spent_raw, Mapping) else
                 {key: initial[key] - remaining[key] for key in _FUEL_KEYS})
        total_initial = int(snapshot.get("total_initial", sum(initial.values())))
        total_remaining = int(snapshot.get("total_remaining", sum(remaining.values())))
        reserve = int(snapshot.get("terminalization_reserve", 0))
        if any(value < 0 for value in (*initial.values(), *remaining.values(), *spent.values())) \
                or total_remaining < reserve:
            raise ValueError("root fuel snapshot is not monotone")
        restored = object.__new__(cls)
        restored.root_budget_id = str(snapshot.get("root_budget_id", "")) or _digest(snapshot)
        restored._terminalization_reserve_remaining = reserve
        restored._unbounded = False
        restored._shared_limit = bool(snapshot.get("shared_limit", False))
        restored._initial = initial
        restored._remaining = remaining
        restored._total_initial = total_initial
        restored._total_remaining = total_remaining
        restored._spent = spent
        restored._spent_terminalization = int(snapshot.get("terminalization_spent", 0))
        return restored

    @property
    def total_initial(self) -> int | None:
        return self._total_initial

    @property
    def total_remaining(self) -> int | None:
        return self._total_remaining

    @property
    def remaining(self) -> int | None:
        return self._total_remaining

    @property
    def root_remaining(self) -> int | None:
        return self._total_remaining

    @property
    def remaining_by_kind(self) -> Mapping[str, int | None]:
        return dict(self._remaining)

    @property
    def spent_by_kind(self) -> Mapping[str, int]:
        return dict(self._spent)

    @property
    def spendable_remaining(self) -> int | None:
        return (None if self._unbounded else
                max(0, self._total_remaining - self._terminalization_reserve_remaining))

    @property
    def terminalization_reserve_remaining(self) -> int:
        return self._terminalization_reserve_remaining

    @property
    def fuel_exhausted(self) -> bool:
        return self.spendable_remaining == 0

    @property
    def exhausted(self) -> bool:
        return self.fuel_exhausted

    def _key(self, kind: object) -> str:
        key = _FUEL_ALIASES.get(str(kind or "").strip().lower())
        if key is None:
            raise KeyError(f"unknown root fuel key: {kind!r}")
        return key

    def can_charge(self, kind: object, amount: int = 1) -> bool:
        key = self._key(kind)
        amount = int(amount)
        if amount < 0:
            raise ValueError("fuel charge must be non-negative")
        if key == "terminalization":
            return amount <= self._terminalization_reserve_remaining
        if self._unbounded:
            return True
        if amount > self._remaining.get(key, 0):
            return False
        return amount <= self.spendable_remaining

    def charge(self, kind: object, amount: int = 1) -> bool:
        """Charge before dispatch; a failed charge dispatches nothing."""
        key = self._key(kind)
        amount = int(amount)
        if key == "terminalization":
            return self.consume_terminalization(amount)
        if not self.can_charge(key, amount):
            return False
        if not self._unbounded:
            self._remaining[key] -= amount
            self._total_remaining -= amount
        self._spent[key] += amount
        return True

    def charge_before_dispatch(self, kind: object, amount: int = 1) -> bool:
        return self.charge(kind, amount)

    def consume_terminalization(self, amount: int = 1) -> bool:
        amount = int(amount)
        if amount < 0 or amount > self._terminalization_reserve_remaining:
            return False
        self._terminalization_reserve_remaining -= amount
        self._spent_terminalization += amount
        if not self._unbounded:
            self._total_remaining -= amount
        return True

    def refill(self, *args: object, **kwargs: object) -> None:
        raise RuntimeError("root fuel is nonreplenishing")

    def snapshot(self) -> dict[str, object]:
        return {
            **({"unbounded": True} if self._unbounded else {}),
            "schema": "root-fuel/1",
            "root_budget_id": self.root_budget_id,
            "initial": dict(self._initial),
            "remaining_by_kind": dict(self._remaining),
            "spent_by_kind": dict(self._spent),
            "total_initial": self._total_initial,
            "total_remaining": self._total_remaining,
            "spendable_remaining": self.spendable_remaining,
            "terminalization_reserve": self._terminalization_reserve_remaining,
            "terminalization_spent": self._spent_terminalization,
            "shared_limit": self._shared_limit,
        }

    to_dict = snapshot

    def _carry(self) -> "RootFuelBudget":
        clone = object.__new__(RootFuelBudget)
        clone.root_budget_id = self.root_budget_id
        clone._terminalization_reserve_remaining = self._terminalization_reserve_remaining
        clone._unbounded = self._unbounded
        clone._shared_limit = self._shared_limit
        clone._initial = dict(self._initial)
        clone._remaining = dict(self._remaining)
        clone._total_initial = self._total_initial
        clone._total_remaining = self._total_remaining
        clone._spent = dict(self._spent)
        clone._spent_terminalization = self._spent_terminalization
        return clone

    def restart(self) -> "RootFuelBudget":
        return self._carry()

    def checkpoint(self) -> "RootFuelBudget":
        return self._carry()

    def fork(self) -> "RootFuelBudget":
        return self._carry()

    def new_revision(self) -> "RootFuelBudget":
        return self._carry()


@dataclass(frozen=True)
class VerificationReceipt:
    """A current/stale observation used to discharge evidence debt."""

    receipt_id: str
    obligation_id: str
    status: str
    candidate_digest: str | None = None
    revision: str | None = None
    sequence: int | float = 0
    stale: bool = False
    evidence: tuple[str, ...] = ()
    identity: Mapping[str, object] | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not str(self.receipt_id).strip():
            raise ValueError("receipt requires receipt_id")
        object.__setattr__(self, "receipt_id", str(self.receipt_id))
        object.__setattr__(self, "obligation_id", str(self.obligation_id))
        object.__setattr__(self, "status", str(self.status).upper())
        object.__setattr__(self, "evidence", _tuple_strings(self.evidence))
        if self.identity is not None:
            if not isinstance(self.identity, Mapping):
                raise ValueError("receipt identity must be a mapping")
            object.__setattr__(self, "identity", copy.deepcopy(dict(self.identity)))

    @property
    def passed(self) -> bool:
        return self.status in {"PASS", "PASSED", "GREEN", "SUCCESS", "OK"}

    @property
    def failed(self) -> bool:
        return not self.passed

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "VerificationReceipt":
        status = value.get("status", value.get("result_class", value.get("result", "")))
        if not status and value.get("exit_code") == 0:
            status = "PASS"
        return cls(
            receipt_id=str(value.get("receipt_id", value.get("id", "receipt"))),
            obligation_id=str(value.get("obligation_id", value.get("selector", value.get("target", "evidence")))),
            status=str(status),
            candidate_digest=(str(value["candidate_digest"]) if value.get("candidate_digest") is not None else None),
            revision=(str(value["revision"]) if value.get("revision") is not None else
                      str(value["product_revision"]) if value.get("product_revision") is not None else None),
            sequence=value.get("sequence", 0),
            stale=bool(value.get("stale", False)),
            evidence=_tuple_strings(value.get("evidence", value.get("evidence_refs", ()))),
            identity=(value.get("identity") if isinstance(value.get("identity"), Mapping)
                      else {key: value[key] for key in (
                          "run_id", "code_hash", "release_id", "harness_version")
                      if key in value} or None),
        )

    def to_dict(self, *, identity: object = None) -> dict[str, object]:
        result = {
            "receipt_id": self.receipt_id,
            "obligation_id": self.obligation_id,
            "status": self.status,
            "candidate_digest": self.candidate_digest,
            "revision": self.revision,
            "sequence": self.sequence,
            "stale": self.stale,
            "evidence": list(self.evidence),
        }
        attached = identity if identity is not None else self.identity
        if attached is not None:
            from kernel_verdict_identity import stamp_record
            stamped = stamp_record(result, attached)
            if isinstance(stamped, Mapping):
                return dict(stamped)
        return result


Receipt = VerificationReceipt
TestReceipt = VerificationReceipt


@dataclass(frozen=True)
class AcceptanceSeal:
    candidate_digest: str
    policy_digest: str
    evidence_digest: str
    state_version: int
    state: str = ACCEPTED

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "acceptance-seal/1",
            "candidate_digest": self.candidate_digest,
            "policy_digest": self.policy_digest,
            "evidence_digest": self.evidence_digest,
            "state_version": self.state_version,
            "state": self.state,
        }


@dataclass(frozen=True)
class AuthorityDecision(Mapping[str, object]):
    state: str
    accepted: bool = False
    reason_codes: tuple[str, ...] = ()
    detail: str = ""
    action: Mapping[str, object] | None = None
    seal: AcceptanceSeal | None = None
    ready_to_commit: bool = False
    fuel_remaining: int | None = 0
    ignored: bool = False
    state_version: int = 0
    identity: Mapping[str, object] | None = field(default=None, compare=False, repr=False)

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def outcome(self) -> str:
        return self.state

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    def to_dict(self, *, identity: object = None) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": SCHEMA,
            "state": self.state,
            "accepted": self.accepted,
            "terminal": self.terminal,
            "reason_codes": list(self.reason_codes),
            "detail": self.detail,
            "action": copy.deepcopy(dict(self.action)) if isinstance(self.action, Mapping) else None,
            "ready_to_commit": self.ready_to_commit,
            "fuel_remaining": self.fuel_remaining,
            "ignored": self.ignored,
            "state_version": self.state_version,
        }
        if self.seal is not None:
            result["seal"] = self.seal.to_dict()
        attached = identity if identity is not None else self.identity
        if attached is not None:
            from kernel_verdict_identity import stamp_record
            stamped = stamp_record(result, attached)
            if isinstance(stamped, Mapping):
                return dict(stamped)
        return result


def _action_mapping(action: str, *, reason_code: str = "", arguments: Mapping[str, object] | None = None) -> dict[str, object]:
    return {
        "tool": str(action),
        "reason_code": _base_reason_code(reason_code),
        "arguments": copy.deepcopy(dict(arguments or {})),
    }


class CompletionAuthority:
    """Host-owned bounded completion state machine."""

    def __init__(
        self,
        candidate: object = None,
        policy: object = None,
        evidence: object = None,
        *,
        budget: RootFuelBudget | Mapping[str, int] | None = None,
        fuel: RootFuelBudget | Mapping[str, int] | None = None,
        root_budget: RootFuelBudget | Mapping[str, int] | None = None,
        required_evidence: Sequence[str] = (),
        available_actions: Sequence[str] | None = None,
        current_revision: object = None,
        deadline: float | None = None,
        deadline_at: float | None = None,
        now: float | None = None,
        registry: ProgressContractRegistry | None = None,
        identity: object = None,
    ) -> None:
        self.registry = registry or ProgressContractRegistry.default()
        assert_progress_contracts(self.registry)
        if budget is None:
            budget = fuel
        if budget is None:
            budget = root_budget
        if budget is None:
            budget = RootFuelBudget.from_total(8, terminalization_reserve=1)
        if isinstance(budget, RootFuelBudget):
            self.fuel = budget
        elif isinstance(budget, Mapping):
            self.fuel = RootFuelBudget(budget, terminalization_reserve=1)
        else:
            raise TypeError("budget must be RootFuelBudget or a mapping")
        self.candidate_digest = _digest(candidate)
        self.policy_digest = _digest(policy)
        self.current_revision = str(current_revision) if current_revision is not None else self.candidate_digest
        self.verdict_identity = identity
        self.deadline_at = deadline_at if deadline_at is not None else deadline
        self.now = float(time.monotonic() if now is None else now)
        self._state = ACTIVE
        self._state_version = 0
        self._evidence_seed = copy.deepcopy(evidence)
        self._receipts: list[VerificationReceipt] = []
        self._current_evidence: dict[str, VerificationReceipt] = {}
        self._recheck_required: set[str] = set()
        self._required_evidence = set(_tuple_strings(required_evidence))
        self._blockers: dict[str, BlockingCause] = {}
        self._attempts: dict[str, int] = {}
        self._inflight: dict[str, object] | None = None
        self._available_actions = (
            None if available_actions is None else set(str(item).strip().lower() for item in available_actions)
        )
        self._seal: AcceptanceSeal | None = None
        self._terminal_decision: AuthorityDecision | None = None
        self._history: list[dict[str, object]] = []

    @property
    def state(self) -> str:
        return self._state

    @property
    def terminal(self) -> bool:
        return self._state in TERMINAL_STATES

    @property
    def seal(self) -> AcceptanceSeal | None:
        return self._seal

    @property
    def state_version(self) -> int:
        return self._state_version

    @property
    def blockers(self) -> tuple[BlockingCause, ...]:
        return tuple(self._blockers.values())

    @property
    def admitted_blockers(self) -> tuple[BlockingCause, ...]:
        return self.blockers

    @property
    def has_admitted_blocker(self) -> bool:
        return bool(self._blockers)

    @property
    def recheck_required(self) -> frozenset[str]:
        return frozenset(self._recheck_required)

    @property
    def receipts(self) -> tuple[VerificationReceipt, ...]:
        return tuple(self._receipts)

    @property
    def required_evidence(self) -> frozenset[str]:
        return frozenset(self._required_evidence)

    @property
    def host_check_state(self) -> str:
        return "REQUIRED_PUBLIC_CHECKS" if self._required_evidence else NO_HOST_CHECKS

    @property
    def ready_to_commit(self) -> bool:
        if self.terminal:
            return self._state == ACCEPTED
        if self._blockers or self._recheck_required:
            return False
        return all(item in self._current_evidence and self._current_evidence[item].passed
                   for item in self._required_evidence)

    @property
    def may_accept(self) -> bool:
        return self.ready_to_commit and not self.terminal

    @property
    def inflight(self) -> Mapping[str, object] | None:
        return copy.deepcopy(self._inflight) if self._inflight is not None else None

    def _decision(
        self,
        *,
        state: str | None = None,
        accepted: bool = False,
        reason_codes: Sequence[str] = (),
        detail: str = "",
        action: Mapping[str, object] | None = None,
        seal: AcceptanceSeal | None = None,
        ignored: bool = False,
    ) -> AuthorityDecision:
        identity = None
        if self.verdict_identity is not None:
            from kernel_verdict_identity import enabled as identity_enabled
            if identity_enabled():
                current = (self.verdict_identity.current()
                           if callable(getattr(self.verdict_identity, "current", None))
                           else self.verdict_identity)
                if callable(getattr(current, "to_dict", None)):
                    value = current.to_dict()
                    identity = dict(value) if isinstance(value, Mapping) else None
                elif isinstance(current, Mapping):
                    identity = dict(current)
        return AuthorityDecision(
            state or self._state, accepted=accepted,
            reason_codes=tuple(dict.fromkeys(_base_reason_code(code) for code in reason_codes if str(code))),
            detail=str(detail), action=copy.deepcopy(dict(action)) if isinstance(action, Mapping) else None,
            seal=seal, ready_to_commit=self.ready_to_commit, fuel_remaining=self.fuel.total_remaining,
            ignored=ignored, state_version=self._state_version, identity=identity,
        )

    def _terminal_or(self, *, ignored: bool = False) -> AuthorityDecision | None:
        if self._terminal_decision is None:
            return None
        if not ignored:
            return self._terminal_decision
        return replace(self._terminal_decision, ignored=True)

    def _record(self, kind: str, **fields: object) -> None:
        self._history.append({"kind": kind, "state_version": self._state_version, **copy.deepcopy(fields)})

    def _cause_for(self, reason_code: str, *, obligation_id: str = "", detail: str = "") -> BlockingCause:
        contract = self.registry.require(reason_code)
        values = {
            "reason_code": contract.reason_code,
            "obligation_id": obligation_id or contract.reason_code.lower(),
            "detail": detail,
            "candidate_digest": self.candidate_digest,
        }
        return contract.cause_type(**values)

    def require_evidence(self, obligation_id: str, *, reason_code: str = "REQUIRED_EVIDENCE_GAP") -> AuthorityDecision:
        self._required_evidence.add(str(obligation_id))
        if str(obligation_id) not in self._current_evidence:
            self._recheck_required.add(str(obligation_id))
        self._state_version += 1
        self._record("evidence_required", obligation_id=str(obligation_id), reason_code=reason_code)
        return self._decision(reason_codes=(reason_code,), detail=f"current PASS required for {obligation_id}")

    def admit_blocker(self, cause: BlockingCause) -> AuthorityDecision:
        terminal = self._terminal_or()
        if terminal is not None:
            return terminal
        if not isinstance(cause, CAUSE_TYPES):
            raise TypeError("admitted blocker must be ProductViolation, RequiredEvidenceGap, or ControlFault")
        contract = self.registry.contract_for(cause.reason_code)
        if contract is None:
            raise ValueError(f"no progress contract for {cause.reason_code}")
        if contract.cause_type is not type(cause) and not contract.advisory:
            raise TypeError(f"{cause.reason_code} has the wrong typed BlockingCause")
        if contract.advisory:
            # Questions/plans are facts, not blockers.  Keep the contract for
            # static coverage, but do not let it veto completion.
            self._record("advisory_refusal", reason_code=contract.reason_code)
            return self._decision(reason_codes=(contract.reason_code,), detail=cause.detail)
        key = str(cause.obligation_id or cause.reason_code)
        self._blockers[key] = cause
        self._state_version += 1
        self._record("blocker_admitted", reason_code=cause.reason_code, obligation_id=key)
        return self._decision(
            reason_codes=(cause.reason_code,),
            detail=cause.detail or f"admitted blocker: {key}",
            action=_action_mapping(contract.permitted_action, reason_code=cause.reason_code),
        )

    def refuse(
        self,
        reason_code: str,
        *,
        cause: BlockingCause | None = None,
        obligation_id: str = "",
        detail: str = "",
    ) -> AuthorityDecision:
        """Admit a typed refusal, never a free-floating reason string."""
        self.registry.require(reason_code)
        if cause is None:
            cause = self._cause_for(reason_code, obligation_id=obligation_id, detail=detail)
        elif _base_reason_code(cause.reason_code) != _base_reason_code(reason_code):
            raise ValueError("cause and refusal reason_code disagree")
        return self.admit_blocker(cause)

    def _receipt_current(self, receipt: VerificationReceipt) -> bool:
        if receipt.stale:
            return False
        if self.verdict_identity is not None:
            from kernel_verdict_identity import IdentityError, assert_current, enabled as identity_enabled
            if identity_enabled():
                try:
                    assert_current(receipt.to_dict(), self.verdict_identity)
                except IdentityError:
                    return False
        if receipt.candidate_digest is not None and receipt.candidate_digest != self.candidate_digest:
            return False
        if receipt.revision is not None and self.current_revision is not None \
                and receipt.revision != str(self.current_revision):
            return False
        return True

    def record_receipt(self, receipt: VerificationReceipt | Mapping[str, object]) -> AuthorityDecision:
        terminal = self._terminal_or(ignored=True)
        if terminal is not None:
            return terminal
        if not isinstance(receipt, VerificationReceipt):
            receipt = VerificationReceipt.from_mapping(receipt)
        current = self._receipt_current(receipt)
        self._receipts.append(receipt)
        if receipt.failed:
            # A failed receipt never proves anything.  If it is stale, the debt
            # is intentionally retained rather than being erased as history.
            self._recheck_required.add(receipt.obligation_id)
        elif current:
            self._current_evidence[receipt.obligation_id] = receipt
            self._recheck_required.discard(receipt.obligation_id)
        self._state_version += 1
        self._record("receipt_ingested", receipt=receipt.to_dict(), current=current)
        return self._decision(
            reason_codes=("RECHECK_REQUIRED",) if self._recheck_required else (),
            detail="current PASS ingested" if receipt.passed and current else
            "failed or stale receipt ingested; current PASS remains required",
        )

    def _resolve_blocker_for_result(self, result: Mapping[str, object]) -> list[str]:
        resolved: list[str] = []
        for key, cause in tuple(self._blockers.items()):
            contract = self.registry.require(cause.reason_code)
            if contract.result_feeds_resolution(result):
                resolved.append(key)
        for key in resolved:
            self._blockers.pop(key, None)
        return resolved

    def record_action_result(self, result: Mapping[str, object]) -> AuthorityDecision:
        terminal = self._terminal_or(ignored=True)
        if terminal is not None:
            return terminal
        result = dict(result)
        resolved = self._resolve_blocker_for_result(result)
        receipt = result.get("receipt")
        if isinstance(receipt, (VerificationReceipt, Mapping)):
            self.record_receipt(receipt)
        if resolved or receipt is not None:
            self._state_version += 1
        if self._inflight is not None:
            expected = str(self._inflight.get("action", ""))
            actual = str(result.get("action", result.get("tool", "")))
            if not actual or actual == expected:
                self._inflight = None
        self._record("action_result_ingested", result=result, resolved=resolved)
        return self._decision(
            reason_codes=() if not self._blockers else tuple(item.reason_code for item in self._blockers.values()),
            detail=("progress result resolved: " + ", ".join(resolved)) if resolved
            else "action result did not resolve the admitted blocker",
        )

    ingest_action_result = record_action_result

    def _available(self, action: str) -> bool:
        if self._available_actions is None:
            return True
        normalized = action.lower()
        return (normalized in self._available_actions
                or any(alias in self._available_actions and target == normalized
                       for alias, target in _LEGACY_ACTION_ALIASES.items()))

    def dispatch_action(
        self,
        action: str,
        *,
        reason_code: str | None = None,
        arguments: Mapping[str, object] | None = None,
        expires_at: float | None = None,
    ) -> AuthorityDecision:
        terminal = self._terminal_or()
        if terminal is not None:
            return terminal
        action = str(action).strip().lower()
        action = _LEGACY_ACTION_ALIASES.get(action, action)
        action = {"verify": "run_tests", "verification": "run_tests", "test": "run_tests",
                  "repair": "edit"}.get(action, action)
        if reason_code is None and action in {
                "model", "model_call", "model_calls", "verification", "verifications",
                "repair", "repairs", "check_compilation", "compile_checks", "compile",
        }:
            return self.dispatch_resource(action, arguments=arguments, expires_at=expires_at)
        code = _base_reason_code(reason_code or self._active_reason_code() or "REQUIRED_EVIDENCE_GAP")
        contract = self.registry.require(code)
        if action != contract.permitted_action:
            return self._fail(
                FAILED_CONTROL_CONFLICT, "CONTROL_CONFLICT",
                f"action {action!r} cannot resolve {code}; expected {contract.permitted_action!r}",
            )
        if not self._available(action):
            return self._fail(
                FAILED_CONTROL_CONFLICT, "CONTROL_CONFLICT",
                f"permitted recovery action {action!r} is not on the current capability menu",
            )
        attempts = self._attempts.get(code, 0) + 1
        if attempts > contract.attempt_limit:
            return self._fail(contract.terminal_fallback, "NO_FEASIBLE_RECOVERY",
                              f"attempt limit exhausted for {code}")
        if not self.fuel.charge(contract.root_budget_key):
            return self._fail(FAILED_BUDGET, "FUEL_EXHAUSTED",
                              f"root fuel cannot afford {action} while preserving terminalization")
        self._attempts[code] = attempts
        self._inflight = {
            "action": action,
            "reason_code": code,
            "arguments": copy.deepcopy(dict(arguments or {})),
            "dispatched_at": self.now,
            "expires_at": expires_at if expires_at is not None else (
                self.now + contract.expiration if contract.expiration is not None else None),
            "attempt": attempts,
        }
        self._state_version += 1
        self._record("action_dispatched", action=action, reason_code=code, attempt=attempts)
        return self._decision(
            reason_codes=(code,), detail=f"dispatched bounded recovery action {action}",
            action=_action_mapping(action, reason_code=code, arguments=arguments),
        )

    dispatch = dispatch_action

    def dispatch_resource(
        self,
        resource: str,
        *,
        arguments: Mapping[str, object] | None = None,
        expires_at: float | None = None,
    ) -> AuthorityDecision:
        """Charge a non-contract dispatch (model/check compilation/etc.) first."""
        terminal = self._terminal_or()
        if terminal is not None:
            return terminal
        raw = str(resource).strip().lower()
        key = {
            "model": "model_calls", "model_call": "model_calls", "model_calls": "model_calls",
            "verification": "verifications", "verifications": "verifications",
            "run_tests": "verifications", "test": "verifications",
            "repair": "repairs", "repairs": "repairs", "edit": "repairs",
            "check_compilation": "check_compilation", "compile_checks": "check_compilation",
            "compile": "check_compilation",
        }.get(raw)
        if key is None:
            return self._fail(FAILED_CONTROL_CONFLICT, "CONTROL_CONFLICT",
                              f"unknown dispatch resource {resource!r}")
        if not self.fuel.charge(key):
            return self._fail(FAILED_BUDGET, "FUEL_EXHAUSTED",
                              f"root fuel cannot afford {key} while preserving terminalization")
        self._inflight = {
            "action": raw, "resource": key, "arguments": copy.deepcopy(dict(arguments or {})),
            "dispatched_at": self.now, "expires_at": expires_at,
        }
        self._state_version += 1
        self._record("resource_dispatched", resource=key)
        return self._decision(
            detail=f"charged root fuel before dispatching {key}",
            action=_action_mapping(raw, arguments=arguments),
        )

    charge_before_dispatch = dispatch_resource

    def charge_model_call(self, **kwargs: object) -> AuthorityDecision:
        return self.dispatch_resource("model_calls", **kwargs)

    def charge_verification(self, **kwargs: object) -> AuthorityDecision:
        return self.dispatch_resource("verifications", **kwargs)

    def charge_repair(self, **kwargs: object) -> AuthorityDecision:
        return self.dispatch_resource("repairs", **kwargs)

    def charge_check_compilation(self, **kwargs: object) -> AuthorityDecision:
        return self.dispatch_resource("check_compilation", **kwargs)

    def _active_reason_code(self) -> str | None:
        if self._blockers:
            return next(iter(self._blockers.values())).reason_code
        if self._recheck_required:
            return "RECHECK_REQUIRED"
        missing = self._required_evidence - set(self._current_evidence)
        if missing:
            return "REQUIRED_EVIDENCE_GAP"
        return None

    def _fail(self, state: str, reason_code: str, detail: str) -> AuthorityDecision:
        if self._terminal_decision is not None:
            return self._terminal_decision
        if state not in TERMINAL_STATES:
            raise ValueError(f"not a terminal state: {state}")
        self._inflight = None
        self._state_version += 1
        # Best effort is deliberate: terminalization must remain possible even
        # when an adversarial budget has no reserve configured.
        self.fuel.consume_terminalization()
        self._state = state
        self._record("terminalized", state=state, reason_code=reason_code)
        decision = self._decision(state=state, reason_codes=(reason_code,), detail=detail)
        self._terminal_decision = decision
        return decision

    def _evidence_digest(self) -> str:
        return _digest({
            "seed": self._evidence_seed,
            "receipts": [item.to_dict() for item in self._receipts],
            "current": {key: value.receipt_id for key, value in sorted(self._current_evidence.items())},
            "recheck_required": sorted(self._recheck_required),
        })

    def _accept(self) -> AuthorityDecision:
        if self._terminal_decision is not None:
            return self._terminal_decision
        # Construct the complete seal before publishing any terminal field.
        next_version = self._state_version + 1
        seal = AcceptanceSeal(
            candidate_digest=self.candidate_digest,
            policy_digest=self.policy_digest,
            evidence_digest=self._evidence_digest(),
            state_version=next_version,
        )
        self.fuel.consume_terminalization()
        self._state_version = next_version
        self._state = ACCEPTED
        self._seal = seal
        self._record("accepted", seal=seal.to_dict())
        decision = self._decision(
            state=ACCEPTED, accepted=True, reason_codes=("ACCEPTED",),
            detail=(
                "host finalized ready candidate with no host checks; limitation disclosed; "
                "requirement evidence or model-owned checks remain the completion basis"
                if self.host_check_state == NO_HOST_CHECKS else
                "host finalized ready candidate; no extra model turn required"
            ), seal=seal,
        )
        self._terminal_decision = decision
        return decision

    def finalize(self) -> AuthorityDecision:
        terminal = self._terminal_or()
        if terminal is not None:
            return terminal
        if self.ready_to_commit:
            return self._accept()
        return self._decision(
            reason_codes=(self._active_reason_code() or "UNVERIFIED",),
            detail="candidate is not ready_to_commit; host did not accept",
        )

    def reduce(self, event: Mapping[str, object] | None = None, *, now: float | None = None) -> AuthorityDecision:
        """Purely ordered host reduction with an always-available terminal path."""
        terminal = self._terminal_or()
        if terminal is not None:
            return terminal
        if now is not None:
            self.now = float(now)
        if event is not None:
            kind = str(event.get("kind", ""))
            if kind in {"receipt", "verification_receipt"}:
                self.record_receipt(event.get("receipt", event))
            elif kind in {"action_result", "result"}:
                self.record_action_result(event.get("result", event))
        # Deadline precedes acceptance: a host cannot claim a timed-out run was
        # accepted merely because a late reducer call noticed readiness.
        if self.deadline_at is not None and self.now >= float(self.deadline_at):
            return self._fail(FAILED_DEADLINE, "DEADLINE", "logical run deadline expired")
        if self.ready_to_commit:
            return self._accept()
        if self._inflight is not None:
            expires_at = self._inflight.get("expires_at")
            if expires_at is not None and self.now >= float(expires_at):
                return self._fail(FAILED_DEADLINE, "DEADLINE", "in-flight recovery action expired")
            if self.fuel.fuel_exhausted:
                return self._fail(FAILED_BUDGET, "FUEL_EXHAUSTED", "fuel exhausted while awaiting a result")
            return self._decision(
                reason_codes=(self._active_reason_code() or "UNVERIFIED",),
                detail="awaiting the in-flight action result",
            )
        reason = self._active_reason_code()
        if reason is None:
            # With no blockers, no missing evidence, and no deadline, the
            # candidate is ready; the branch above should have accepted it.
            return self._fail(FAILED_UNVERIFIED, "UNVERIFIED", "completion readiness could not be established")
        contract = self.registry.require(reason)
        if self._available_actions is not None and not self._available(contract.permitted_action):
            return self._fail(
                FAILED_CONTROL_CONFLICT, "CONTROL_CONFLICT",
                f"no executable action for progress contract {reason}",
            )
        if self.fuel.fuel_exhausted:
            fallback = FAILED_BUDGET
            return self._fail(fallback, "FUEL_EXHAUSTED", "root fuel exhausted before resolution")
        return self.dispatch_action(contract.permitted_action, reason_code=reason)

    tick = reduce
    step = reduce

    accept = finalize
    finalize_acceptance = finalize
    record_verification = record_receipt
    ingest_receipt = record_receipt

    def set_candidate(self, candidate: object, *, revision: object = None) -> AuthorityDecision:
        terminal = self._terminal_or()
        if terminal is not None:
            return terminal
        new_digest = _digest(candidate)
        if (new_digest == self.candidate_digest
                and (revision is None or str(revision) == str(self.current_revision))):
            return self._decision(detail="candidate unchanged")
        self.candidate_digest = new_digest
        if revision is not None:
            self.current_revision = str(revision)
        self._current_evidence.clear()
        self._recheck_required.update(self._required_evidence)
        self._state_version += 1
        self._record("candidate_revision", candidate_digest=new_digest, revision=self.current_revision)
        return self._decision(reason_codes=("RECHECK_REQUIRED",) if self._required_evidence else ())

    def new_revision(self, candidate: object | None = None, *, revision: object = None) -> AuthorityDecision:
        return self.set_candidate(self.candidate_digest if candidate is None else candidate, revision=revision)

    def restart(self) -> "CompletionAuthority":
        self.fuel = self.fuel.restart()
        return self

    def checkpoint(self) -> "CompletionAuthority":
        self.fuel = self.fuel.checkpoint()
        return self

    def fork(self) -> "CompletionAuthority":
        clone = copy.copy(self)
        clone.fuel = self.fuel.fork()
        clone._receipts = list(self._receipts)
        clone._current_evidence = dict(self._current_evidence)
        clone._recheck_required = set(self._recheck_required)
        clone._required_evidence = set(self._required_evidence)
        clone._blockers = dict(self._blockers)
        clone._attempts = dict(self._attempts)
        clone._history = copy.deepcopy(self._history)
        return clone

    def render(self) -> dict[str, object]:
        """One host projection; all completion renderings share this version."""
        return {
            "schema": SCHEMA,
            "state": self._state,
            "decision_version": self._state_version,
            "state_version": self._state_version,
            "ready_to_commit": self.ready_to_commit,
            "may_accept": self.may_accept,
            "host_check_state": self.host_check_state,
            "no_host_checks": self.host_check_state == NO_HOST_CHECKS,
            "candidate_digest": self.candidate_digest,
            "policy_digest": self.policy_digest,
            "recheck_required": sorted(self._recheck_required),
            "blockers": [
                {"kind": cause.kind, "reason_code": cause.reason_code,
                 "obligation_id": cause.obligation_id, "detail": cause.detail}
                for cause in self._blockers.values()
            ],
            "inflight": self.inflight,
            "fuel": self.fuel.snapshot(),
            "seal": self._seal.to_dict() if self._seal is not None else None,
        }

    def snapshot(self) -> dict[str, object]:
        """Durable state used by restart/checkpoint replay."""
        result = self.render()
        result.update({
            "candidate_digest": self.candidate_digest,
            "policy_digest": self.policy_digest,
            "current_revision": self.current_revision,
            "deadline_at": self.deadline_at,
            "now": self.now,
            "required_evidence": sorted(self._required_evidence),
            "host_check_state": self.host_check_state,
            "recheck_required": sorted(self._recheck_required),
            "receipts": [item.to_dict() for item in self._receipts],
            "current_evidence": {
                key: value.receipt_id for key, value in self._current_evidence.items()
            },
            "attempts": dict(self._attempts),
            "history": copy.deepcopy(self._history),
            "evidence_seed": copy.deepcopy(self._evidence_seed),
        })
        return result

    @classmethod
    def from_snapshot(
        cls,
        snapshot: Mapping[str, object],
        *,
        available_actions: Sequence[str] | None = None,
        registry: ProgressContractRegistry | None = None,
        identity: object = None,
    ) -> "CompletionAuthority":
        """Rehydrate a run without resetting fuel, debt, or terminal state."""
        if not isinstance(snapshot, Mapping):
            raise TypeError("completion authority snapshot must be a mapping")
        fuel_snapshot = snapshot.get("fuel")
        fuel = RootFuelBudget.from_snapshot(fuel_snapshot) if isinstance(fuel_snapshot, Mapping) else None
        authority = cls(
            candidate=str(snapshot.get("candidate_digest", "")),
            policy=str(snapshot.get("policy_digest", "")),
            evidence=snapshot.get("evidence_seed"),
            budget=fuel,
            required_evidence=_tuple_strings(snapshot.get("required_evidence", ())),
            available_actions=available_actions,
            current_revision=snapshot.get("current_revision"),
            deadline_at=snapshot.get("deadline_at"),
            now=float(snapshot.get("now", time.monotonic())),
            registry=registry,
            identity=identity,
        )
        authority._state_version = int(snapshot.get("state_version", snapshot.get("decision_version", 0)))
        authority._recheck_required = set(_tuple_strings(snapshot.get("recheck_required", ())))
        authority._attempts = {
            str(key): int(value) for key, value in (snapshot.get("attempts", {}) or {}).items()
        } if isinstance(snapshot.get("attempts", {}), Mapping) else {}
        authority._history = copy.deepcopy(list(snapshot.get("history", ()))) \
            if isinstance(snapshot.get("history", ()), Sequence) \
            and not isinstance(snapshot.get("history", ()), (str, bytes)) else []
        authority._receipts = []
        by_id: dict[str, VerificationReceipt] = {}
        raw_receipts = snapshot.get("receipts", ())
        if isinstance(raw_receipts, Sequence) and not isinstance(raw_receipts, (str, bytes)):
            for raw in raw_receipts:
                if not isinstance(raw, Mapping):
                    continue
                receipt = VerificationReceipt.from_mapping(raw)
                authority._receipts.append(receipt)
                by_id[receipt.receipt_id] = receipt
        current_ids = snapshot.get("current_evidence", {})
        if isinstance(current_ids, Mapping):
            authority._current_evidence = {
                str(key): by_id[str(receipt_id)]
                for key, receipt_id in current_ids.items() if str(receipt_id) in by_id
            }
        authority._blockers = {}
        raw_blockers = snapshot.get("blockers", ())
        if isinstance(raw_blockers, Sequence) and not isinstance(raw_blockers, (str, bytes)):
            classes = {item.__name__: item for item in CAUSE_TYPES}
            for raw in raw_blockers:
                if not isinstance(raw, Mapping):
                    continue
                kind = str(raw.get("kind", "RequiredEvidenceGap"))
                cause_class = classes.get(kind, RequiredEvidenceGap)
                try:
                    cause = cause_class(
                        reason_code=str(raw.get("reason_code", "REQUIRED_EVIDENCE_GAP")),
                        obligation_id=str(raw.get("obligation_id", "evidence")),
                        detail=str(raw.get("detail", "")),
                    )
                except (TypeError, ValueError):
                    continue
                authority._blockers[cause.obligation_id] = cause
        authority._inflight = copy.deepcopy(snapshot.get("inflight")) \
            if isinstance(snapshot.get("inflight"), Mapping) else None
        authority._state = str(snapshot.get("state", ACTIVE))
        raw_seal = snapshot.get("seal")
        if isinstance(raw_seal, Mapping):
            authority._seal = AcceptanceSeal(
                candidate_digest=str(raw_seal.get("candidate_digest", authority.candidate_digest)),
                policy_digest=str(raw_seal.get("policy_digest", authority.policy_digest)),
                evidence_digest=str(raw_seal.get("evidence_digest", "")),
                state_version=int(raw_seal.get("state_version", authority._state_version)),
                state=str(raw_seal.get("state", ACCEPTED)),
            )
        if authority._state in TERMINAL_STATES:
            authority._terminal_decision = authority._decision(
                state=authority._state,
                accepted=authority._state == ACCEPTED,
                reason_codes=("ACCEPTED" if authority._state == ACCEPTED else "RESTORED_TERMINAL",),
                detail="terminal state restored from the same logical run",
                seal=authority._seal,
            )
        return authority

    @property
    def history(self) -> tuple[Mapping[str, object], ...]:
        return tuple(copy.deepcopy(self._history))


def green_capability_menu(
    tools: Sequence[Mapping[str, object]],
    *,
    green: bool,
    authority: CompletionAuthority | None = None,
) -> list[Mapping[str, object]]:
    """Narrow only evidence-free green work; preserve tools for open blockers."""
    source = list(tools)
    if not green or (authority is not None and authority.has_admitted_blocker):
        return source
    return [
        tool for tool in source
        if str(tool.get("function", {}).get("name", "")) in {"done", "run_tests"}
    ]


# Friendly aliases used by host adapters and offline probes.
CompletionState = TerminalState
CompletionAuthorityState = CompletionAuthority
ProgressContracts = ProgressContractRegistry
RootBudget = RootFuelBudget


__all__ = [
    "ACCEPTED", "ACTIVE", "ACTION_RESULT_EFFECTS", "AcceptanceSeal", "AuthorityDecision",
    "BlockingCause", "CAUSE_TYPES", "CompletionAuthority", "CompletionAuthorityState",
    "CompletionState", "ControlFault", "DEFAULT_PROGRESS_CONTRACTS", "PROGRESS_CONTRACTS", "FAILED_BUDGET",
    "FAILED_CONTROL_CONFLICT", "FAILED_DEADLINE", "FAILED_UNRESOLVED_DEFECT",
    "FAILED_UNVERIFIED", "FuelExhausted", "MODE_ENV", "POLICY_VERSION", "ProductViolation",
    "ProgressContract", "ProgressContractRegistry", "ProgressContracts", "Receipt",
    "RequiredEvidenceGap", "RootBudget", "RootFuelBudget", "SCHEMA", "TERMINAL_STATES",
    "TerminalState", "TestReceipt", "VerificationReceipt", "enabled", "green_capability_menu",
    "mode", "refusal_reason_codes", "assert_progress_contracts", "validate_progress_contracts",
]
