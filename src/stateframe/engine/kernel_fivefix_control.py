"""Receipt-backed, language-neutral control primitives for the five-fix lane.

This module deliberately does not own IDEALPATH stages, menu composition,
completion, decisions, localization, verification, or terminal delivery.  It
only gives those existing owners immutable identity, bounded accounting, and
receipt validation helpers so their composed path cannot silently drift.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field


SCHEMA = "stateframe-fivefix-control/1"
HARNESS_CONTRACT = "HARNESS_CONTRACT"
STALE_REPLAN = "STALE_REPLAN"
UNRUNNABLE = "UNRUNNABLE"

_PROGRESS_KINDS = frozenset({
    "host_fact",
    "admitted_stage_record",
    "planned_edit",
    "coverage_transition",
    "behavioral_observation",
})


def digest(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str,
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _text(value: object, *, name: str, limit: int = 512) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} must be a non-empty string of at most {limit} characters")
    return value.strip()


def _optional_text(value: object, *, limit: int = 512) -> str:
    return value.strip() if isinstance(value, str) and value.strip() and len(value) <= limit else ""


def _unique_strings(
        values: object, *, name: str, limit: int = 128,
        reject_duplicates: bool = False,
) -> tuple[str, ...]:
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes, bytearray, Mapping)):
        raise ValueError(f"{name} must be a bounded sequence")
    iterable = sorted(values, key=str) if isinstance(values, (set, frozenset)) else values
    result: list[str] = []
    for value in iterable:
        item = _text(value, name=name, limit=256)
        if item not in result:
            result.append(item)
        elif reject_duplicates:
            raise ValueError(f"{name} contains duplicate action {item!r}")
        if len(result) > limit:
            raise ValueError(f"{name} exceeds {limit} items")
    return tuple(result)


class HarnessContractError(ValueError):
    """A host-composed surface cannot be sent to a provider."""

    code = HARNESS_CONTRACT

    def __init__(self, detail: str, *, stage: str = "", actions: Sequence[str] = ()) -> None:
        self.detail = str(detail)
        self.stage = str(stage)
        self.actions = tuple(str(item) for item in actions)
        super().__init__(self.detail)


@dataclass(frozen=True)
class CapabilitySnapshot:
    """The one request-bound identity shared by menu, dispatch, and continuation."""

    request_id: str
    stage: str
    epoch: int
    source_revision: str
    product_revision: str
    capability_digest: str
    budget_generation: str
    admitted_actions: tuple[str, ...]
    offered_actions: tuple[str, ...]
    final_actions: tuple[str, ...]
    exit_action_ids: tuple[str, ...]
    recovery_action_ids: tuple[str, ...]
    satisfying_action_ids: tuple[str, ...]

    @property
    def identity(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "request_id": self.request_id,
            "stage": self.stage,
            "epoch": self.epoch,
            "source_revision": self.source_revision,
            "product_revision": self.product_revision,
            "capability_digest": self.capability_digest,
            "budget_generation": self.budget_generation,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self.identity,
            "admitted_actions": list(self.admitted_actions),
            "offered_actions": list(self.offered_actions),
            "final_actions": list(self.final_actions),
            "exit_action_ids": list(self.exit_action_ids),
            "recovery_action_ids": list(self.recovery_action_ids),
            "satisfying_action_ids": list(self.satisfying_action_ids),
            "liveness": bool(self.satisfying_action_ids),
        }


def build_capability_snapshot(
        *,
        request_id: object,
        stage: object,
        epoch: object,
        source_revision: object,
        product_revision: object,
        budget_generation: object,
        admitted_actions: object,
        offered_actions: object,
        exit_action_ids: object,
        recovery_action_ids: object = (),
        capability_facts: Mapping[str, object] | None = None,
) -> CapabilitySnapshot:
    """Intersect the final offered surface with host admission and prove liveness."""
    request = _text(request_id, name="request_id")
    stage_value = _text(stage, name="stage", limit=16)
    if type(epoch) is not int or epoch < 1:
        raise ValueError("epoch must be a positive integer")
    source = _text(source_revision, name="source_revision")
    product = _text(product_revision, name="product_revision")
    budget = _text(budget_generation, name="budget_generation")
    try:
        admitted = _unique_strings(admitted_actions, name="admitted_actions")
        offered = _unique_strings(
            offered_actions, name="offered_actions", reject_duplicates=True,
        )
        exits = _unique_strings(exit_action_ids, name="exit_action_ids")
        recovery = _unique_strings(recovery_action_ids, name="recovery_action_ids")
    except ValueError as error:
        raise HarnessContractError(
            f"malformed capability menu: {error}", stage=stage_value,
        ) from error
    final = tuple(action for action in offered if action in set(admitted))
    if not final:
        raise HarnessContractError(
            "final menu has no admitted action", stage=stage_value, actions=offered,
        )
    satisfying = tuple(action for action in final if action in set(exits) | set(recovery))
    if not satisfying:
        raise HarnessContractError(
            "final menu has no satisfiable exit or admitted recovery route",
            stage=stage_value,
            actions=final,
        )
    capability_digest = digest({
        "stage": stage_value,
        "epoch": epoch,
        "source_revision": source,
        "product_revision": product,
        "admitted_actions": admitted,
        "offered_actions": offered,
        "final_actions": final,
        "exit_action_ids": exits,
        "recovery_action_ids": recovery,
        "capability_facts": dict(capability_facts or {}),
    })
    return CapabilitySnapshot(
        request_id=request,
        stage=stage_value,
        epoch=epoch,
        source_revision=source,
        product_revision=product,
        capability_digest=capability_digest,
        budget_generation=budget,
        admitted_actions=admitted,
        offered_actions=offered,
        final_actions=final,
        exit_action_ids=exits,
        recovery_action_ids=recovery,
        satisfying_action_ids=satisfying,
    )


@dataclass(frozen=True)
class BatchDecision:
    status: str
    batch_id: str
    reason: str
    replan: bool
    cancel: bool


class StaleBatchGate:
    """Emit one typed replan for each stale native batch, then cancel duplicates."""

    def __init__(self) -> None:
        self._settled: set[str] = set()

    def validate(self, snapshot: CapabilitySnapshot, batch: Mapping[str, object]) -> BatchDecision:
        batch_id = _text(batch.get("batch_id"), name="batch_id")
        expected = snapshot.identity
        fields = (
            "request_id", "stage", "epoch", "source_revision", "product_revision",
            "capability_digest", "budget_generation",
        )
        # Native adapters historically called this origin_request_id. Accept
        # that spelling, but a missing request identity is never current.
        request_id = batch.get("request_id", batch.get("origin_request_id"))
        origin_request_id = batch.get("origin_request_id")
        stale = any(
            (request_id if field == "request_id" else batch.get(field))
            != expected[field]
            for field in fields
        )
        if origin_request_id is not None and origin_request_id != expected["request_id"]:
            stale = True
        if not stale:
            return BatchDecision("ADMITTED", batch_id, "snapshot_matches", False, False)
        if batch_id in self._settled:
            return BatchDecision("STALE_CANCEL_DUPLICATE", batch_id, "replan_already_issued", False, True)
        self._settled.add(batch_id)
        return BatchDecision(STALE_REPLAN, batch_id, "request_snapshot_changed", True, True)


@dataclass
class ResourceAccount:
    """Separate measured work from requested reservations in one shared account."""

    provider_calls: int = 0
    retries: int = 0
    requested_reservations: int = 0
    generated_tokens: int = 0
    reasoning_tokens: int = 0
    main_provider_calls: int = 0
    sidecall_provider_calls: int = 0
    charges: list[dict[str, object]] = field(default_factory=list)

    def charge(
            self,
            *,
            purpose: object,
            owner: object,
            actual_provider_calls: int = 0,
            requested_reservations: int = 0,
            generated_tokens: int = 0,
            reasoning_tokens: int = 0,
            retries: int = 0,
    ) -> dict[str, object]:
        values = {
            "actual_provider_calls": actual_provider_calls,
            "requested_reservations": requested_reservations,
            "generated_tokens": generated_tokens,
            "reasoning_tokens": reasoning_tokens,
            "retries": retries,
        }
        if any(type(value) is not int or value < 0 for value in values.values()):
            raise ValueError("resource charges must be non-negative integers")
        purpose_text = _text(purpose, name="purpose", limit=120)
        owner_text = _text(owner, name="owner", limit=120)
        row = {
            "purpose": purpose_text,
            "owner": owner_text,
            **values,
        }
        self.provider_calls += actual_provider_calls
        self.retries += retries
        self.requested_reservations += requested_reservations
        self.generated_tokens += generated_tokens
        self.reasoning_tokens += reasoning_tokens
        if purpose_text == "sidecall":
            self.sidecall_provider_calls += actual_provider_calls
        else:
            self.main_provider_calls += actual_provider_calls
        self.charges.append(row)
        return dict(row)

    def to_dict(self) -> dict[str, object]:
        return {
            "provider_calls": self.provider_calls,
            "retries": self.retries,
            "requested_reservations": self.requested_reservations,
            "generated_tokens": self.generated_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "main_provider_calls": self.main_provider_calls,
            "sidecall_provider_calls": self.sidecall_provider_calls,
            "charges": [dict(item) for item in self.charges],
        }


class ProgressLedger:
    """Count request opportunities and only unseen host-backed semantic progress."""

    def __init__(self, *, global_response_cap: int = 0) -> None:
        self.global_response_cap = global_response_cap
        self._requests: set[str] = set()
        self._facts: set[tuple[str, str]] = set()
        self.counters = {
            "responses": 0,
            "tool_attempts": 0,
            "sidecalls": 0,
            "automatic_checks": 0,
            "wall_time_ms": 0,
        }
        self.local_no_progress_streak = 0
        self.recovery_used = False

    def observe_request(self, request_id: object) -> bool:
        ident = _text(request_id, name="request_id")
        if ident in self._requests:
            return False
        self._requests.add(ident)
        self.counters["responses"] += 1
        return True

    def record_resource(self, kind: str, amount: int = 1) -> None:
        if kind not in {"tool_attempts", "sidecalls", "automatic_checks", "wall_time_ms"}:
            raise ValueError("unknown progress resource counter")
        if type(amount) is not int or amount < 0:
            raise ValueError("resource counter amount must be a non-negative integer")
        self.counters[kind] += amount

    def record_progress(self, kind: object, identity: object, *, stage: object = "") -> bool:
        kind_text = _text(kind, name="progress_kind", limit=64)
        if kind_text not in _PROGRESS_KINDS:
            self.local_no_progress_streak += 1
            return False
        identity_text = _text(identity, name="progress_identity", limit=512)
        key = (kind_text, identity_text)
        if key in self._facts:
            self.local_no_progress_streak += 1
            return False
        self._facts.add(key)
        self.local_no_progress_streak = 0
        return True

    def record_action(self, *, result: object, progress: bool = False) -> bool:
        """Record an attempt; only ``record_progress`` can admit progress.

        ``progress`` remains a compatibility argument, but a caller cannot
        certify its own semantic progress by setting it.
        """
        self.record_resource("tool_attempts")
        del result, progress
        self.local_no_progress_streak += 1
        return False

    def remaining_s5_allowance(
            self, *, remaining_admitted_work: int, verification_reservation: int = 1,
            global_remaining: int | None = None,
    ) -> int:
        if type(remaining_admitted_work) is not int or remaining_admitted_work < 0:
            raise ValueError("remaining_admitted_work must be non-negative")
        if type(verification_reservation) is not int or verification_reservation < 0:
            raise ValueError("verification_reservation must be non-negative")
        allowance = remaining_admitted_work + verification_reservation
        if global_remaining is not None:
            if type(global_remaining) is not int or global_remaining < 0:
                raise ValueError("global_remaining must be non-negative")
            allowance = min(allowance, global_remaining)
        return allowance

    def select_recovery(
            self,
            candidates: Sequence[Mapping[str, object]],
            *,
            evidence_refs: Sequence[object],
            remaining_admitted_work: int,
    ) -> dict[str, object] | None:
        """Admit at most one evidence-selected recovery before an incomplete stop."""
        if self.recovery_used or remaining_admitted_work < 1:
            return None
        evidence = {
            _text(item, name="evidence_ref", limit=256)
            for item in evidence_refs
            if isinstance(item, str) and item.strip()
        }
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            action = _optional_text(candidate.get("action"), limit=120)
            raw_refs = candidate.get("evidence_refs")
            candidate_refs = (
                raw_refs if isinstance(raw_refs, Sequence)
                and not isinstance(raw_refs, (str, bytes, bytearray)) else ()
            )
            refs = {
                _optional_text(item, limit=256)
                for item in candidate_refs
                if isinstance(item, str) and item.strip()
            }
            if action and refs & evidence:
                self.recovery_used = True
                return {
                    "status": "ADMITTED",
                    "action": action,
                    "evidence_refs": sorted(refs & evidence),
                    "remaining_admitted_work": remaining_admitted_work,
                }
        return None

    def snapshot(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "counters": dict(self.counters),
            "local_no_progress_streak": self.local_no_progress_streak,
            "recovery_used": self.recovery_used,
        }


def decision_key(
        *, unresolved_predicate: object, evidence_digest: object,
        capability_digest: object, revision: object,
) -> str:
    return digest({
        "unresolved_predicate": _text(unresolved_predicate, name="unresolved_predicate"),
        "evidence_digest": _text(evidence_digest, name="evidence_digest"),
        "capability_digest": _text(capability_digest, name="capability_digest"),
        "revision": _text(revision, name="revision"),
    })


class DecisionEpisodeLedger:
    """Deduplicate only unchanged consequential ambiguity episodes."""

    def __init__(self, *, account: ResourceAccount | None = None) -> None:
        self.account = account or ResourceAccount()
        self.closed_keys: set[str] = set()
        self.active_key: str | None = None
        self.last_reason = ""

    def open(
            self,
            *,
            unresolved_predicate: object,
            evidence_digest: object,
            capability_digest: object,
            revision: object,
            options: Sequence[object],
            deterministic: bool = False,
    ) -> dict[str, object]:
        offered = _unique_strings(options, name="options", limit=32)
        key = decision_key(
            unresolved_predicate=unresolved_predicate,
            evidence_digest=evidence_digest,
            capability_digest=capability_digest,
            revision=revision,
        )
        if deterministic or len(offered) <= 1:
            self.last_reason = "deterministic_or_single_exit"
            return {"status": "SKIPPED", "key": key, "options": list(offered), "reason": self.last_reason}
        if key in self.closed_keys:
            self.last_reason = "unchanged_episode"
            return {"status": "DEDUPED", "key": key, "options": list(offered), "reason": self.last_reason}
        if self.active_key == key:
            self.last_reason = "active_episode"
            return {"status": "DEDUPED", "key": key, "options": list(offered), "reason": self.last_reason}
        self.active_key = key
        self.last_reason = "consequential_ambiguity"
        return {"status": "OPEN", "key": key, "options": list(offered), "reason": self.last_reason}

    def close(self, *, key: object, reason: object = "settled") -> None:
        key_text = _text(key, name="decision_key", limit=200)
        self.closed_keys.add(key_text)
        if self.active_key == key_text:
            self.active_key = None
        self.last_reason = _text(reason, name="reason", limit=120)

    def close_contract_error(self, *, key: object, error_digest: object) -> None:
        self.close(key=key, reason=f"contract_error:{_text(error_digest, name='error_digest')}")


def classify_negative_lookup(
        *, status: object, complete: bool, query: object, scope: object,
        product_revision: object,
) -> dict[str, object]:
    """Turn a revision/scope-bound negative fact into invalidation or UNKNOWN."""
    if type(complete) is not bool:
        raise ValueError("complete must be a host-classified boolean")
    status_text = _text(status, name="status", limit=64).upper()
    query_text = _text(query, name="query", limit=256)
    scope_text = _text(scope, name="scope", limit=512)
    revision = _text(product_revision, name="product_revision")
    normalized_query = " ".join(query_text.split()).casefold()
    normalized_scope = "/".join(
        part.strip().casefold()
        for part in scope_text.replace("\\", "/").split("/")
        if part.strip()
    )
    if status_text in {"NOT_FOUND", "REFERENCED_ONLY"} and complete:
        effect = "TARGET_INVALIDATED"
        next_route = "rebind_existing_boundary"
    else:
        effect = "UNKNOWN"
        next_route = "expand_lookup_scope"
    return {
        "schema": SCHEMA,
        "status": status_text,
        "complete": bool(complete),
        "query": query_text,
        "scope": scope_text,
        "product_revision": revision,
        "effect": effect,
        "next_route": next_route,
        "dedupe_key": digest({
            "status": status_text,
            "complete": complete,
            "query": normalized_query,
            "scope": normalized_scope,
            "product_revision": revision,
        }),
    }


class NegativeLookupLedger:
    def __init__(self) -> None:
        self._seen: set[str] = set()

    def record(self, finding: Mapping[str, object]) -> bool:
        key = _text(finding.get("dedupe_key"), name="dedupe_key", limit=200)
        if key in self._seen:
            return False
        self._seen.add(key)
        return True


def adapter_result(language: object, *, available: bool, detail: str = "") -> dict[str, object]:
    language_text = _text(language, name="language", limit=80)
    return {
        "language": language_text,
        "status": "READY" if available else UNRUNNABLE,
        "detail": _optional_text(detail, limit=240),
    }


def _receipt(receipts: Mapping[str, Mapping[str, object]], ident: object) -> Mapping[str, object] | None:
    value = receipts.get(str(ident))
    return value if isinstance(value, Mapping) else None


def _check_facts(
        receipt: Mapping[str, object] | None, *,
        outcome: str, revision: str, check_id: str,
) -> bool:
    if receipt is None:
        return False
    return (
        str(receipt.get("outcome", "")).upper() == outcome
        and receipt.get("reached_observation") is True
        and str(receipt.get("product_revision", "")) == revision
        and str(receipt.get("check_hash", receipt.get("check_id", ""))) == check_id
        and receipt.get("integrity") is True
    )


def verify_obligations(
        *,
        obligations: Mapping[str, Mapping[str, object]],
        receipts: Mapping[str, Mapping[str, object]],
        base_revision: object,
        candidate_revision: object,
        integrity: bool,
) -> dict[str, object]:
    """Verify separate REPAIR and PRESERVE rows over multiple sites."""
    base = _text(base_revision, name="base_revision")
    candidate = _text(candidate_revision, name="candidate_revision")
    if type(integrity) is not bool:
        raise ValueError("integrity must be host-classified")
    results: dict[str, dict[str, object]] = {}
    all_ok = bool(obligations) and integrity
    for row_id, obligation in obligations.items():
        if not isinstance(obligation, Mapping):
            all_ok = False
            results[str(row_id)] = {"role": "UNKNOWN", "certified": False, "reason": "invalid_obligation"}
            continue
        role = str(obligation.get("role", "REPAIR")).upper()
        baseline_id = obligation.get("baseline")
        candidate_id = obligation.get("candidate")
        check_id = _text(obligation.get("check_hash", obligation.get("check_id")), name="check_id")
        baseline = _receipt(receipts, baseline_id)
        candidate_receipt = _receipt(receipts, candidate_id)
        same_check = bool(
            baseline is not None and candidate_receipt is not None
            and str(baseline.get("check_hash", baseline.get("check_id", ""))) == check_id
            and str(candidate_receipt.get("check_hash", candidate_receipt.get("check_id", ""))) == check_id
        )
        if role == "REPAIR":
            ok = (
                _check_facts(baseline, outcome="BEHAVIOR_FAIL", revision=base, check_id=check_id)
                and _check_facts(candidate_receipt, outcome="PASS", revision=candidate, check_id=check_id)
                and same_check
            )
            reason = "repair_pair" if ok else "repair_requires_reached_parent_failure_and_candidate_pass"
        elif role == "PRESERVE":
            ok = (
                _check_facts(baseline, outcome="PASS", revision=base, check_id=check_id)
                and _check_facts(candidate_receipt, outcome="PASS", revision=candidate, check_id=check_id)
                and same_check
            )
            reason = "preserve_pair" if ok else "preserve_requires_pass_to_pass_on_same_check"
        else:
            ok = False
            reason = "unknown_obligation_role"
        results[str(row_id)] = {"role": role, "certified": ok and integrity, "reason": reason}
        all_ok = all_ok and ok
    return {
        "schema": SCHEMA,
        "certified": all_ok,
        "integrity": integrity,
        "obligations": results,
        "repair_rows": [row for row, item in results.items() if item["role"] == "REPAIR"],
        "preserve_rows": [row for row, item in results.items() if item["role"] == "PRESERVE"],
    }


__all__ = [
    "CapabilitySnapshot", "BatchDecision", "DecisionEpisodeLedger", "HarnessContractError",
    "NegativeLookupLedger", "ProgressLedger", "ResourceAccount", "SCHEMA", "STALE_REPLAN",
    "StaleBatchGate", "UNRUNNABLE", "adapter_result",
    "build_capability_snapshot", "classify_negative_lookup", "decision_key", "digest",
    "verify_obligations",
]
