"""Host-owned candidate preservation and public-regression recovery.

The reducer in this module stores facts only: immutable product manifests,
settled mutation identity, executed check receipts, and numerical counters.  It
does not grade candidates, select a completion outcome, or generate steering
text.  Recovery is a forward activation of an earlier immutable overlay; it is
never a PASS, done, or certification transition.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from kernel_candidate_checkpoint_store import (
    ActivationReceipt,
    CheckpointConflictError,
    CheckpointError,
    CheckpointStore,
    ProductManifest,
    StoredCheckpoint,
)


MODE_ENV = "STATEFRAME_CANDIDATE_PRESERVATION"
SCHEMA = "stateframe-candidate-preservation/1"
UNKNOWN = "UNKNOWN"
PASS = "PASS"
FAIL = "FAIL"
RECOVERABLE_RESULTS = frozenset({PASS})
UNKNOWN_RESULTS = frozenset({"", UNKNOWN, "ABSENT", "SKIPPED", "REFUSED", "TIMEOUT", "UNAVAILABLE"})
PUBLIC_FAILURE_RESULTS = frozenset({FAIL, "RED", "COUNTEREXAMPLE", "ERROR"})


def enabled(runtime: object | None = None) -> bool:
    configured = getattr(runtime, "stateframe_candidate_preservation_mode", None) if runtime is not None else None
    raw = configured if configured is not None else os.environ.get(MODE_ENV, "off")
    value = str(raw or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value == "on"


def _digest(value: object) -> str:
    if isinstance(value, bytes):
        data = value
    else:
        data = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _field(value: object, name: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _normal_result(value: object) -> str:
    result = str(value or "").strip().upper()
    if result in {PASS, FAIL}:
        return result
    if result in {"PASSED", "GREEN", "SUCCESS", "OK"}:
        return PASS
    if result in {"FAILED", "RED", "COUNTEREXAMPLE", "ERROR"}:
        return FAIL if result != "COUNTEREXAMPLE" else "COUNTEREXAMPLE"
    return result if result in UNKNOWN_RESULTS else UNKNOWN


def _observation_count(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, Mapping):
        for name in ("count", "tests_run", "observed_count", "observations"):
            if isinstance(value.get(name), int) and not isinstance(value.get(name), bool):
                return max(0, int(value[name]))
    return 0


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze(item) for item in value)
    return value


def _manifest(value: object) -> ProductManifest:
    if isinstance(value, ProductManifest):
        return value
    if isinstance(value, Mapping):
        return ProductManifest.from_mapping(value)
    raise TypeError("candidate product manifest is required")


@dataclass(frozen=True)
class CheckDefinition:
    check_id: str
    definition_hash: str
    environment_id: str
    requirement_ids: tuple[str, ...] = ()
    public: bool = True

    @classmethod
    def from_value(cls, value: object, *, check_id: str | None = None) -> "CheckDefinition":
        identifier = str(check_id or _field(value, "check_id", _field(value, "id", "")) or "").strip()
        if not identifier:
            raise ValueError("check definition requires a stable check_id")
        requirements = tuple(dict.fromkeys(
            str(item) for item in _sequence(_field(value, "requirement_ids", _field(value, "requirements", ())))
            if str(item).strip()
        ))
        definition_hash = str(_field(value, "definition_hash", "") or "").strip()
        if not definition_hash:
            definition_hash = _digest({
                "check_id": identifier,
                "requirements": requirements,
                "definition": _field(value, "definition", _field(value, "source", value)),
            })
        return cls(
            check_id=identifier,
            definition_hash=definition_hash,
            environment_id=str(_field(value, "environment_id", _field(value, "environment", "default")) or "default"),
            requirement_ids=requirements,
            public=bool(_field(value, "public", True)),
        )


@dataclass(frozen=True)
class CheckInventory:
    definitions: tuple[CheckDefinition, ...]

    @classmethod
    def from_value(cls, value: object) -> "CheckInventory":
        if isinstance(value, cls):
            return value
        rows = value.items() if isinstance(value, Mapping) else enumerate(_sequence(value))
        definitions = tuple(CheckDefinition.from_value(row, check_id=str(key)) for key, row in rows)
        return cls(tuple(item for item in definitions if item.public))

    def get(self, check_id: str) -> CheckDefinition | None:
        return next((item for item in self.definitions if item.check_id == check_id), None)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(item.check_id for item in self.definitions)


@dataclass(frozen=True)
class CheckReceipt:
    check_id: str
    result: str
    candidate_digest: str
    definition_hash: str
    environment_id: str
    execution_id: str
    observation_count: int
    run_id: str = ""
    task_id: str = ""
    receipt_id: str = ""
    semantic_eligible: bool = True
    wrapper: bool = False

    @classmethod
    def from_value(cls, value: object) -> "CheckReceipt":
        check_id = str(_field(value, "check_id", _field(value, "id", _field(value, "obligation_id", ""))) or "").strip()
        raw_result = _field(value, "result", _field(value, "result_class", _field(value, "status", "")))
        result = _normal_result(raw_result)
        observation = _field(value, "observation_count", _field(value, "observed_count", _field(value, "observed", 0)))
        count = _observation_count(observation)
        wrapper = bool(_field(value, "wrapper", False)) or str(_field(value, "receipt_type", "")).upper() in {
            "BOOKKEEPING", "WRAPPER", "PROJECTION",
        }
        semantic = bool(_field(value, "semantic_eligible", True)) and not wrapper
        if count <= 0 or not semantic or result in UNKNOWN_RESULTS:
            result = UNKNOWN
        return cls(
            check_id=check_id,
            result=result,
            candidate_digest=str(_field(value, "candidate_digest", _field(value, "after_digest", _field(value, "revision", ""))) or ""),
            definition_hash=str(_field(value, "definition_hash", _field(value, "check_hash", "")) or ""),
            environment_id=str(_field(value, "environment_id", _field(value, "environment", "default")) or "default"),
            execution_id=str(_field(value, "execution_id", _field(value, "execution", _field(value, "id", ""))) or ""),
            observation_count=count,
            run_id=str(_field(value, "run_id", "") or ""),
            task_id=str(_field(value, "task_id", "") or ""),
            receipt_id=str(_field(value, "receipt_id", _field(value, "id", "")) or ""),
            semantic_eligible=semantic,
            wrapper=wrapper,
        )

    @property
    def binding(self) -> tuple[str, str, str, str]:
        return self.check_id, self.definition_hash, self.environment_id, self.candidate_digest

    @property
    def observed(self) -> bool:
        return self.observation_count > 0 and self.semantic_eligible and not self.wrapper


def _receipt_values(receipt: CheckReceipt) -> dict[str, object]:
    return {
        "check_id": receipt.check_id,
        "result": receipt.result,
        "candidate_digest": receipt.candidate_digest,
        "definition_hash": receipt.definition_hash,
        "environment_id": receipt.environment_id,
        "execution_id": receipt.execution_id,
        "observation_count": receipt.observation_count,
        "run_id": receipt.run_id,
        "task_id": receipt.task_id,
        "receipt_id": receipt.receipt_id,
        "semantic_eligible": receipt.semantic_eligible,
        "wrapper": receipt.wrapper,
    }


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    checkpoint_id: str
    run_id: str
    task_id: str
    parent_revision: str
    revision: str
    effect_id: str
    branch_id: str
    manifest: ProductManifest
    created_index: int
    model_owned: bool = True
    settled: bool = True
    baseline: bool = False
    transaction_id: str = ""
    changed_lines: int = 0
    open_requirements: tuple[str, ...] = ()
    authored_checks: tuple[CheckDefinition, ...] = ()


@dataclass(frozen=True)
class RecoveryPlan:
    checkpoint_id: str
    candidate_id: str
    expected_current_revision: str
    reason_receipt_id: str
    run_id: str
    task_id: str
    generation_id: str
    invalidates: tuple[str, ...] = (
        "revision_bound_handles", "read_spans", "bindings", "completion_projections",
    )


@dataclass(frozen=True)
class RecoveryReceipt:
    schema: str
    status: str
    checkpoint_id: str
    candidate_id: str
    old_digest: str
    new_digest: str
    generation_id: str = ""
    preserved_branch_id: str = ""
    check_binding: str = ""
    open_requirements: tuple[str, ...] = ()
    reason: str = ""
    counters_retained: bool = True
    completion_accepted: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "status": self.status,
            "checkpoint_id": self.checkpoint_id,
            "candidate_id": self.candidate_id,
            "old_digest": self.old_digest,
            "new_digest": self.new_digest,
            "generation_id": self.generation_id,
            "preserved_branch_id": self.preserved_branch_id,
            "check_binding": self.check_binding,
            "open_requirements": list(self.open_requirements),
            "reason": self.reason,
            "counters_retained": self.counters_retained,
            "completion_accepted": self.completion_accepted,
        }


@dataclass(frozen=True)
class PreservationState:
    run_id: str
    task_id: str
    current_revision: str = ""
    active_candidate_id: str = ""
    candidates: tuple[Candidate, ...] = ()
    receipts: tuple[CheckReceipt, ...] = ()
    recovery_count: int = 0
    generation_counter: int = 0
    requests_consumed: int = 0
    elapsed_seconds: float = 0.0
    check_spend: int = 0
    invalidation_epoch: int = 0


def _receipt_rows(receipts: object) -> tuple[CheckReceipt, ...]:
    if isinstance(receipts, Mapping):
        receipts = receipts.values()
    return tuple(item if isinstance(item, CheckReceipt) else CheckReceipt.from_value(item)
                 for item in _sequence(receipts))


def _best_receipt(rows: Iterable[CheckReceipt], definition: CheckDefinition, candidate_digest: str) -> CheckReceipt | None:
    matches = [row for row in rows if (
        row.check_id == definition.check_id
        and row.candidate_digest == candidate_digest
        and row.definition_hash == definition.definition_hash
        and row.environment_id == definition.environment_id
    )]
    if not matches:
        return None
    return max(matches, key=lambda row: (
        int(row.observed), int(row.result == PASS), int(row.result in PUBLIC_FAILURE_RESULTS),
        int(bool(row.execution_id)), row.receipt_id,
    ))


def recoverable(
        candidate: Candidate,
        public_receipts: object,
        current_check_inventory: object,
) -> bool:
    """Return true only for a settled, model-owned, fully observed candidate."""
    if not isinstance(candidate, Candidate) or candidate.baseline or not candidate.model_owned or not candidate.settled:
        return False
    if not candidate.run_id or not candidate.task_id or not candidate.effect_id:
        return False
    if not isinstance(candidate.manifest, ProductManifest) or not candidate.manifest.complete:
        return False
    inventory = CheckInventory.from_value(current_check_inventory)
    rows = tuple(item for item in _receipt_rows(public_receipts)
                 if item.run_id == candidate.run_id and item.task_id == candidate.task_id)
    if not inventory.definitions:
        return False
    return all(
        (receipt := _best_receipt(rows, definition, candidate.revision)) is not None
        and receipt.result == PASS
        and receipt.observed
        for definition in inventory.definitions
    )


def evidence_vector(candidate: Candidate, receipts: object, inventory: object) -> tuple[tuple[str, str], ...]:
    rows = tuple(item for item in _receipt_rows(receipts)
                 if item.run_id == candidate.run_id and item.task_id == candidate.task_id)
    definitions = CheckInventory.from_value(inventory).definitions
    result: list[tuple[str, str]] = []
    for definition in definitions:
        receipt = _best_receipt(rows, definition, candidate.revision)
        result.append((definition.check_id, receipt.result if receipt is not None else UNKNOWN))
    return tuple(result)


def evidence_dominates(left: Candidate, right: Candidate, receipts: object, inventory: object) -> bool | None:
    """Compare only comparable evidence; ``None`` means incomparable."""
    left_vector = dict(evidence_vector(left, receipts, inventory))
    right_vector = dict(evidence_vector(right, receipts, inventory))
    if set(left_vector) != set(right_vector):
        return None
    rank = {UNKNOWN: 0, "COUNTEREXAMPLE": 1, FAIL: 1, PASS: 2}
    values = [(rank.get(left_vector[key], 0), rank.get(right_vector[key], 0)) for key in left_vector]
    left_better = any(a > b for a, b in values)
    right_better = any(a < b for a, b in values)
    if left_better and right_better:
        return None
    if right_better:
        return False
    return left_better


def eligible_fallback(
        current: Candidate,
        inventory: object,
        candidates: Iterable[Candidate] = (),
        public_receipts: object = (),
) -> Candidate | None:
    """Choose the latest applicable pre-transaction model candidate.

    Evidence dominance is used where possible.  Ties retain the recent-green
    candidate and small-diff candidate as separate stored facts; recovery uses
    the latest eligible candidate, never the unchanged parent.
    """
    if not isinstance(current, Candidate):
        return None
    options = [item for item in candidates if (
        isinstance(item, Candidate)
        and item.candidate_id != current.candidate_id
        and item.run_id == current.run_id
        and item.task_id == current.task_id
        and item.created_index < current.created_index
        and recoverable(item, public_receipts, inventory)
    )]
    if not options:
        return None
    non_dominated: list[Candidate] = []
    for item in options:
        dominated = False
        for other in options:
            relation = evidence_dominates(other, item, public_receipts, inventory)
            if relation is True:
                dominated = True
                break
        if not dominated:
            non_dominated.append(item)
    return max(non_dominated or options, key=lambda item: (item.created_index, -item.changed_lines, item.candidate_id))


def editable(current_revision: object, transaction: object, remaining_budget: object) -> bool:
    """Independent mutation predicate; completion state is intentionally absent."""
    revision = str(current_revision or "")
    if not revision or not isinstance(transaction, Mapping):
        return False
    if bool(transaction.get("settled", False)) is False and transaction.get("active") is False:
        return False
    remaining = remaining_budget
    if isinstance(remaining_budget, Mapping):
        remaining = remaining_budget.get("model_calls", remaining_budget.get("remaining", 0))
    return isinstance(remaining, (int, float)) and not isinstance(remaining, bool) and remaining > 0


def certifiable(current_revision: object, existing_completion_authority: object) -> bool:
    """Ask the existing completion authority; never create another decision model."""
    for name in ("certifiable", "is_certifiable", "can_certify", "is_ready"):
        method = getattr(existing_completion_authority, name, None)
        if callable(method):
            try:
                return bool(method(current_revision))
            except (AttributeError, TypeError, ValueError):
                return False
    accepted = bool(getattr(existing_completion_authority, "completion_accepted", False))
    authority_revision = getattr(existing_completion_authority, "current_revision", None)
    return accepted and (authority_revision in (None, current_revision))


def _candidate_values(candidate: Candidate) -> dict[str, object]:
    return {
        "candidate_id": candidate.candidate_id,
        "checkpoint_id": candidate.checkpoint_id,
        "run_id": candidate.run_id,
        "task_id": candidate.task_id,
        "parent_revision": candidate.parent_revision,
        "revision": candidate.revision,
        "effect_id": candidate.effect_id,
        "branch_id": candidate.branch_id,
        "manifest": candidate.manifest,
        "created_index": candidate.created_index,
        "model_owned": candidate.model_owned,
        "settled": candidate.settled,
        "baseline": candidate.baseline,
        "transaction_id": candidate.transaction_id,
        "changed_lines": candidate.changed_lines,
        "open_requirements": candidate.open_requirements,
        "authored_checks": candidate.authored_checks,
    }


class PreservationReducer:
    """Mutable reducer whose snapshots are immutable ``PreservationState`` values."""

    def __init__(self, run_id: object, task_id: object, store: CheckpointStore | None = None):
        self.run_id = str(run_id or "")
        self.task_id = str(task_id or "")
        if not self.run_id or not self.task_id:
            raise ValueError("candidate preservation requires run_id and task_id")
        self.store = store or CheckpointStore()
        self._state = PreservationState(self.run_id, self.task_id)
        self._receipt_keys: set[tuple[str, str, str, str, str]] = set()

    @property
    def state(self) -> PreservationState:
        return self._state

    def snapshot(self) -> PreservationState:
        return self._state

    def export(self) -> dict[str, object]:
        """Serialize reducer facts for continuation/restart reconciliation."""
        return {
            "schema": SCHEMA,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "current_revision": self._state.current_revision,
            "active_candidate_id": self._state.active_candidate_id,
            "recovery_count": self._state.recovery_count,
            "generation_counter": self._state.generation_counter,
            "requests_consumed": self._state.requests_consumed,
            "elapsed_seconds": self._state.elapsed_seconds,
            "check_spend": self._state.check_spend,
            "invalidation_epoch": self._state.invalidation_epoch,
            "candidates": [
                {
                    "candidate_id": item.candidate_id,
                    "checkpoint_id": item.checkpoint_id,
                    "parent_revision": item.parent_revision,
                    "revision": item.revision,
                    "effect_id": item.effect_id,
                    "branch_id": item.branch_id,
                    "created_index": item.created_index,
                    "model_owned": item.model_owned,
                    "settled": item.settled,
                    "baseline": item.baseline,
                    "transaction_id": item.transaction_id,
                    "changed_lines": item.changed_lines,
                    "open_requirements": list(item.open_requirements),
                    "authored_checks": [
                        {
                            "check_id": definition.check_id,
                            "definition_hash": definition.definition_hash,
                            "environment_id": definition.environment_id,
                            "requirement_ids": list(definition.requirement_ids),
                            "public": definition.public,
                        }
                        for definition in item.authored_checks
                    ],
                }
                for item in self._state.candidates
            ],
            "receipts": [_receipt_values(item) for item in self._state.receipts],
        }

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, object], store: CheckpointStore) -> "PreservationReducer":
        if not isinstance(snapshot, Mapping):
            raise TypeError("preservation snapshot must be a mapping")
        reducer = cls(snapshot.get("run_id", ""), snapshot.get("task_id", ""), store)
        candidates: list[Candidate] = []
        for row in snapshot.get("candidates", ()) or ():
            if not isinstance(row, Mapping):
                continue
            checkpoint = store.get(row.get("checkpoint_id", ""))
            if checkpoint is None:
                continue
            candidates.append(Candidate(
                candidate_id=str(row.get("candidate_id", checkpoint.checkpoint_id)),
                checkpoint_id=checkpoint.checkpoint_id,
                run_id=reducer.run_id, task_id=reducer.task_id,
                parent_revision=str(row.get("parent_revision", checkpoint.parent_revision)),
                revision=str(row.get("revision", checkpoint.manifest.digest)),
                effect_id=str(row.get("effect_id", checkpoint.effect_id)),
                branch_id=str(row.get("branch_id", checkpoint.branch_id)),
                manifest=checkpoint.manifest,
                created_index=int(row.get("created_index", checkpoint.created_index)),
                model_owned=bool(row.get("model_owned", checkpoint.model_owned)),
                settled=bool(row.get("settled", checkpoint.settled)),
                baseline=bool(row.get("baseline", checkpoint.baseline)),
                transaction_id=str(row.get("transaction_id", "")),
                changed_lines=int(row.get("changed_lines", 0)),
                open_requirements=tuple(str(item) for item in row.get("open_requirements", ()) or ()),
                authored_checks=tuple(
                    CheckDefinition.from_value(item)
                    for item in row.get("authored_checks", ()) or ()
                    if isinstance(item, Mapping)
                ),
            ))
        receipts = tuple(CheckReceipt.from_value(row) for row in snapshot.get("receipts", ()) or ()
                         if isinstance(row, Mapping))
        reducer._receipt_keys = {(*item.binding, item.execution_id) for item in receipts}
        reducer._state = PreservationState(
            reducer.run_id, reducer.task_id,
            str(snapshot.get("current_revision", "")),
            str(snapshot.get("active_candidate_id", "")), tuple(candidates), receipts,
            int(snapshot.get("recovery_count", 0)), int(snapshot.get("generation_counter", 0)),
            int(snapshot.get("requests_consumed", 0)), float(snapshot.get("elapsed_seconds", 0.0)),
            int(snapshot.get("check_spend", 0)), int(snapshot.get("invalidation_epoch", 0)),
        )
        return reducer

    def carry_counters(self, *, requests: int = 0, elapsed: float = 0.0, check_spend: int = 0) -> None:
        self._state = PreservationState(
            self.run_id, self.task_id, self._state.current_revision,
            self._state.active_candidate_id, self._state.candidates,
            self._state.receipts, self._state.recovery_count,
            self._state.generation_counter,
            self._state.requests_consumed + max(0, int(requests)),
            self._state.elapsed_seconds + max(0.0, float(elapsed)),
            self._state.check_spend + max(0, int(check_spend)),
            self._state.invalidation_epoch,
        )

    def record_check(self, executed_receipt: object) -> CheckReceipt:
        receipt = executed_receipt if isinstance(executed_receipt, CheckReceipt) else CheckReceipt.from_value(executed_receipt)
        key = (*receipt.binding, receipt.execution_id)
        if key not in self._receipt_keys:
            self._receipt_keys.add(key)
            self._state = PreservationState(
                self.run_id, self.task_id, self._state.current_revision,
                self._state.active_candidate_id, self._state.candidates,
                (*self._state.receipts, receipt), self._state.recovery_count,
                self._state.generation_counter, self._state.requests_consumed,
                self._state.elapsed_seconds, self._state.check_spend,
                self._state.invalidation_epoch,
            )
        return receipt

    def observe_settled_mutation(
            self,
            effect_id: object,
            before_manifest: object,
            after_manifest: object,
            *,
            transaction_id: object = "",
            settled: bool = True,
            model_owned: bool = True,
            baseline: bool = False,
            changed_lines: int = 0,
            open_requirements: Iterable[object] = (),
            check_inventory: object = (),
    ) -> Candidate | None:
        before = _manifest(before_manifest)
        after = _manifest(after_manifest)
        if before.digest == after.digest or not settled or not model_owned or baseline:
            self._state = PreservationState(
                self.run_id, self.task_id, after.digest,
                self._state.active_candidate_id, self._state.candidates,
                self._state.receipts, self._state.recovery_count,
                self._state.generation_counter, self._state.requests_consumed,
                self._state.elapsed_seconds, self._state.check_spend,
                self._state.invalidation_epoch,
            )
            return None
        candidate_id = f"candidate-{len(self._state.candidates) + 1:04d}"
        checkpoint_id = candidate_id
        branch_id = f"branch-{candidate_id}"
        checkpoint = self.store.capture(
            after,
            checkpoint_id=checkpoint_id,
            run_id=self.run_id,
            task_id=self.task_id,
            parent_revision=before.digest,
            effect_id=str(effect_id),
            branch_id=branch_id,
            model_owned=model_owned,
            settled=settled,
            baseline=baseline,
        )
        candidate = Candidate(
            candidate_id=candidate_id, checkpoint_id=checkpoint.checkpoint_id,
            run_id=self.run_id, task_id=self.task_id,
            parent_revision=before.digest, revision=after.digest,
            effect_id=str(effect_id), branch_id=branch_id, manifest=after,
            created_index=checkpoint.created_index, model_owned=model_owned,
            settled=settled, baseline=baseline,
            transaction_id=str(transaction_id or ""), changed_lines=max(0, int(changed_lines)),
            open_requirements=tuple(dict.fromkeys(str(item) for item in open_requirements if str(item))),
            authored_checks=CheckInventory.from_value(check_inventory).definitions,
        )
        self._state = PreservationState(
            self.run_id, self.task_id, after.digest, candidate_id,
            (*self._state.candidates, candidate), self._state.receipts,
            self._state.recovery_count, self._state.generation_counter,
            self._state.requests_consumed, self._state.elapsed_seconds,
            self._state.check_spend, self._state.invalidation_epoch,
        )
        return candidate

    def candidate(self, candidate_id: object) -> Candidate | None:
        return next((item for item in self._state.candidates if item.candidate_id == str(candidate_id)), None)

    def observe_current_revision(self, revision: object) -> None:
        """Bind the live revision after a settled action without creating a candidate."""
        self._state = PreservationState(
            self.run_id, self.task_id, str(revision or ""),
            self._state.active_candidate_id, self._state.candidates,
            self._state.receipts, self._state.recovery_count,
            self._state.generation_counter, self._state.requests_consumed,
            self._state.elapsed_seconds, self._state.check_spend,
            self._state.invalidation_epoch,
        )

    def settle_effect(self, effect_id: object) -> None:
        effect = str(effect_id or "")
        candidates = tuple(
            Candidate(
                **{**_candidate_values(item), "settled": True}
            ) if item.effect_id == effect else item
            for item in self._state.candidates
        )
        self._state = PreservationState(
            self.run_id, self.task_id, self._state.current_revision,
            self._state.active_candidate_id, candidates, self._state.receipts,
            self._state.recovery_count, self._state.generation_counter,
            self._state.requests_consumed, self._state.elapsed_seconds,
            self._state.check_spend, self._state.invalidation_epoch,
        )

    def settle_latest(self) -> None:
        if self._state.candidates:
            self.settle_effect(self._state.candidates[-1].effect_id)

    def eligible_fallback(self, current: Candidate | None, inventory: object) -> Candidate | None:
        current = current or next((item for item in reversed(self._state.candidates)
                                   if item.revision == self._state.current_revision), None)
        if current is None:
            return None
        return eligible_fallback(current, inventory, self._state.candidates, self._state.receipts)

    def plan_recovery(self, expected_current: object, checkpoint_id: object, reason_receipt: object,
                      inventory: object) -> RecoveryPlan:
        expected = str(expected_current or "")
        if expected != self._state.current_revision:
            raise CheckpointConflictError("recovery expected revision is stale")
        reason = reason_receipt if isinstance(reason_receipt, CheckReceipt) else CheckReceipt.from_value(reason_receipt)
        if reason.result not in PUBLIC_FAILURE_RESULTS or not reason.observed:
            raise CheckpointError("recovery reason is not an observed public failure")
        target = self.candidate(str(checkpoint_id))
        if target is None or not recoverable(target, self._state.receipts, inventory):
            raise CheckpointError("checkpoint is not eligible under current check inventory")
        if target.run_id != self.run_id or target.task_id != self.task_id or target.baseline:
            raise CheckpointError("checkpoint lineage is not eligible")
        generation = f"generation-{self._state.generation_counter + 1:08d}"
        return RecoveryPlan(
            checkpoint_id=target.checkpoint_id, candidate_id=target.candidate_id,
            expected_current_revision=expected, reason_receipt_id=reason.receipt_id,
            run_id=self.run_id, task_id=self.task_id, generation_id=generation,
        )

    def commit_recovery(
            self,
            plan: RecoveryPlan,
            *,
            workspace: object | None = None,
            open_requirements: Iterable[object] = (),
    ) -> RecoveryReceipt:
        if plan.run_id != self.run_id or plan.task_id != self.task_id:
            return RecoveryReceipt(SCHEMA, "REFUSED_LINEAGE", plan.checkpoint_id, plan.candidate_id,
                                   plan.expected_current_revision, "", reason="run/task lineage mismatch")
        target = self.candidate(plan.candidate_id)
        if target is None or target.checkpoint_id != plan.checkpoint_id:
            return RecoveryReceipt(SCHEMA, "REFUSED_MISSING_CHECKPOINT", plan.checkpoint_id, plan.candidate_id,
                                   plan.expected_current_revision, "", reason="checkpoint missing")
        try:
            if workspace is None:
                activation = ActivationReceipt(
                    "stateframe-candidate-checkpoint/1", "ACTIVATED", target.checkpoint_id,
                    plan.generation_id, plan.expected_current_revision, target.revision,
                )
            else:
                activation = self.store.activate(
                    self.store.get(target.checkpoint_id) or _stored_from_candidate(target),
                    workspace,
                    expected_current_digest=plan.expected_current_revision,
                    generation_id=plan.generation_id,
                )
        except (CheckpointError, OSError, ValueError) as error:
            return RecoveryReceipt(
                SCHEMA, "RECOVERY_INCOMPLETE", target.checkpoint_id, target.candidate_id,
                plan.expected_current_revision, "", generation_id=plan.generation_id,
                preserved_branch_id=target.branch_id, reason=type(error).__name__,
                open_requirements=tuple(str(item) for item in open_requirements if str(item)),
            )
        self._state = PreservationState(
            self.run_id, self.task_id, target.revision, target.candidate_id,
            self._state.candidates, self._state.receipts,
            self._state.recovery_count + 1, self._state.generation_counter + 1,
            self._state.requests_consumed, self._state.elapsed_seconds,
            self._state.check_spend, self._state.invalidation_epoch + 1,
        )
        return RecoveryReceipt(
            SCHEMA, "RECOVERED", target.checkpoint_id, target.candidate_id,
            activation.old_digest, activation.new_digest,
            generation_id=activation.generation_id, preserved_branch_id=target.branch_id,
            check_binding="preserved-executed-receipts",
            open_requirements=tuple(str(item) for item in open_requirements if str(item)),
        )


def _stored_from_candidate(candidate: Candidate) -> StoredCheckpoint:
    return StoredCheckpoint(
        checkpoint_id=candidate.checkpoint_id, manifest=candidate.manifest,
        run_id=candidate.run_id, task_id=candidate.task_id,
        parent_revision=candidate.parent_revision, effect_id=candidate.effect_id,
        branch_id=candidate.branch_id, created_index=candidate.created_index,
        model_owned=candidate.model_owned, settled=candidate.settled,
        baseline=candidate.baseline,
    )


def invalidate_revision_handles(runtime: object) -> None:
    """Invalidate revision-bound handles without changing completion status."""
    for name in (
        "prepared_binding", "current_source_handle", "source_handle", "read_span",
        "read_spans", "target_ref", "_completion_owner_dispatch_tools",
        "_completion_owner_done_selection", "_completion_owner_reused_verification",
    ):
        if hasattr(runtime, name):
            try:
                setattr(runtime, name, None if name != "_completion_owner_dispatch_tools" else [])
            except AttributeError:
                pass
    for owner_name in ("continuation", "decision_continuity"):
        owner = getattr(runtime, owner_name, None)
        for method_name in ("invalidate_revision", "invalidate_bindings", "on_revision_changed"):
            method = getattr(owner, method_name, None)
            if callable(method):
                try:
                    method()
                except (AttributeError, TypeError, ValueError):
                    pass
                break
    setattr(runtime, "_candidate_invalidation_epoch",
            int(getattr(runtime, "_candidate_invalidation_epoch", 0) or 0) + 1)


def reducer_for(runtime: object) -> PreservationReducer | None:
    if runtime is None:
        return None
    existing = getattr(runtime, "candidate_preservation", None)
    if isinstance(existing, PreservationReducer):
        return existing
    run_id = getattr(getattr(runtime, "journal", None), "run_id", None) or getattr(runtime, "run_id", None)
    task_id = getattr(runtime, "task_id", None) or getattr(getattr(runtime, "owner", None), "task_id", None)
    if not run_id or not task_id:
        return None
    root = getattr(runtime, "candidate_checkpoint_root", None)
    if root is None:
        owner = getattr(runtime, "owner", None)
        root = Path(getattr(owner, "root", Path.cwd())) / ".stateframe-candidate-checkpoints"
    store = CheckpointStore(root)
    persisted = getattr(runtime, "candidate_preservation_snapshot", None)
    if isinstance(persisted, Mapping):
        try:
            reducer = PreservationReducer.from_snapshot(persisted, store)
        except (TypeError, ValueError):
            reducer = PreservationReducer(run_id, task_id, store)
    else:
        reducer = PreservationReducer(run_id, task_id, store)
    try:
        runtime.candidate_preservation = reducer
    except AttributeError:
        return reducer
    return reducer


def restore_candidate_for_runtime(runtime: object, checkpoint_id: object, expected_current_revision: object) -> dict[str, object]:
    from kernel_done_is_submit_authority import done_sealed

    if done_sealed(runtime) is not None:
        return {"schema": SCHEMA, "status": "SEALED", "reason": "done submission is immutable"}
    if not enabled(runtime):
        return {"schema": SCHEMA, "status": "REFUSED_UNAVAILABLE", "reason": "preservation unavailable"}
    reducer = reducer_for(runtime)
    if reducer is None:
        return {"schema": SCHEMA, "status": "REFUSED_UNAVAILABLE", "reason": "preservation unavailable"}
    current = str(expected_current_revision or "")
    if current != reducer.state.current_revision:
        return {"schema": SCHEMA, "status": "REFUSED_STALE_REVISION", "reason": "current revision compare-and-swap failed"}
    inventory = getattr(runtime, "candidate_check_inventory", None) or getattr(runtime, "public_check_inventory", None) or ()
    reason = next((item for item in reversed(reducer.state.receipts)
                   if item.result in PUBLIC_FAILURE_RESULTS and item.candidate_digest == current), None)
    if reason is None:
        return {"schema": SCHEMA, "status": "REFUSED_NO_PUBLIC_FAILURE", "reason": "no observed current public failure"}
    try:
        plan = reducer.plan_recovery(current, checkpoint_id, reason, inventory)
    except (CheckpointError, ValueError, TypeError) as error:
        return {"schema": SCHEMA, "status": "REFUSED_INELIGIBLE", "reason": type(error).__name__}
    root = getattr(runtime, "root", None) or getattr(getattr(runtime, "owner", None), "root", None)
    receipt = reducer.commit_recovery(
        plan,
        workspace=root,
        open_requirements=getattr(runtime, "open_requirement_ids", ()) or (),
    )
    if receipt.status == "RECOVERED":
        invalidate_revision_handles(runtime)
        runtime._candidate_last_recovery = receipt.as_dict()
        runtime.candidate_preservation_snapshot = reducer.export()
        runtime.completion_accepted = False
        runtime.system_outcome = getattr(runtime, "system_outcome", None)
    return receipt.as_dict()


def restore_available(runtime: object | None) -> bool:
    if runtime is None or not enabled(runtime):
        return False
    reducer = reducer_for(runtime) if runtime is not None else None
    if reducer is None:
        return False
    inventory = getattr(runtime, "candidate_check_inventory", None) or getattr(runtime, "public_check_inventory", None) or ()
    current = next((item for item in reversed(reducer.state.candidates)
                    if item.revision == reducer.state.current_revision), None)
    if current is None:
        return False
    reason = next((item for item in reversed(reducer.state.receipts)
                   if item.result in PUBLIC_FAILURE_RESULTS and item.candidate_digest == current.revision), None)
    return reason is not None and reducer.eligible_fallback(current, inventory) is not None


def observe_settled_mutation(
        runtime: object,
        effect_id: object,
        before_manifest: object,
        after_manifest: object,
        **kwargs: object,
) -> Candidate | None:
    reducer = reducer_for(runtime)
    if reducer is None:
        return None
    return reducer.observe_settled_mutation(effect_id, before_manifest, after_manifest, **kwargs)


def record_check(runtime: object, executed_receipt: object) -> CheckReceipt | None:
    reducer = reducer_for(runtime)
    return reducer.record_check(executed_receipt) if reducer is not None else None


def plan_recovery(runtime: object, expected_current: object, checkpoint_id: object,
                  reason_receipt: object, inventory: object) -> RecoveryPlan:
    reducer = reducer_for(runtime)
    if reducer is None:
        raise CheckpointError("preservation reducer unavailable")
    return reducer.plan_recovery(expected_current, checkpoint_id, reason_receipt, inventory)


def commit_recovery(runtime: object, plan: RecoveryPlan, *, workspace: object | None = None,
                    open_requirements: Iterable[object] = ()) -> RecoveryReceipt:
    reducer = reducer_for(runtime)
    if reducer is None:
        return RecoveryReceipt(SCHEMA, "REFUSED_UNAVAILABLE", plan.checkpoint_id, plan.candidate_id,
                               plan.expected_current_revision, reason="preservation reducer unavailable")
    receipt = reducer.commit_recovery(plan, workspace=workspace, open_requirements=open_requirements)
    if receipt.status == "RECOVERED":
        invalidate_revision_handles(runtime)
    return receipt


__all__ = [
    "Candidate", "CheckDefinition", "CheckInventory", "CheckReceipt", "FAIL", "PASS",
    "PreservationReducer", "PreservationState", "RecoveryPlan", "RecoveryReceipt", "SCHEMA",
    "UNKNOWN", "certifiable", "commit_recovery", "evidence_dominates", "evidence_vector", "editable",
    "eligible_fallback", "enabled", "invalidate_revision_handles", "observe_settled_mutation",
    "plan_recovery", "record_check", "recoverable", "reducer_for", "restore_available",
    "restore_candidate_for_runtime",
]
