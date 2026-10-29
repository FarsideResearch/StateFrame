"""Shared retry policy, parent accounting and typed completed-action diagnostics."""
from __future__ import annotations

from kernel_step_limits import minimum_limit, reached, remaining as limit_remaining

import contextvars
import copy
import math
import threading
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass

from kernel_measurement import digest

ACTIVE_PARENT = contextvars.ContextVar("pilot_parent", default=None)
ACTIVE_RUNTIME = contextvars.ContextVar("pilot_runtime", default=None)


class ParentBudgetExhausted(RuntimeError):
    """No further work may start under this parent allocation.

    Provider admission failures are deliberately typed as incomplete.  Callers
    may surface the failure, but must not turn it into a successful certificate.
    """

    outcome = "INCOMPLETE"
    certifiable = False

    def __init__(self, category: str, reason: str = "budget_exhausted") -> None:
        self.category = category
        self.reason = reason
        super().__init__(f"{category}: {reason}")


@dataclass(frozen=True)
class RetryPolicy:
    transport_attempts: int = 3
    repeat_threshold: int = 6
    retry_temperature: float = 0.6
    transport_backoff_seconds: tuple[float, ...] = (2.0, 5.0, 10.0)

    def attempt_temperature(self, index: int) -> float:
        return 0.0 if index == 0 else self.retry_temperature

    def repeat_temperature(self, repeats: int, current: float | None) -> float | None:
        return self.retry_temperature if repeats >= self.repeat_threshold and current is None else current

    def transport_delay(self, retry_number: int) -> float:
        """Return bounded delay before one-based transport retry number."""
        if retry_number < 1:
            raise ValueError("transport retry number must be positive")
        if not self.transport_backoff_seconds:
            return 0.0
        return self.transport_backoff_seconds[
            min(retry_number - 1, len(self.transport_backoff_seconds) - 1)
        ]


POLICY = RetryPolicy()


@dataclass(frozen=True)
class ProviderPermit:
    """One durable authorization for a logical request and its transport retries."""

    permit_id: str
    parent_id: str
    sequence: int
    reason: str
    inner_limit: int | None
    outer_limit: int | None
    optional: bool

    def mapping(self) -> dict[str, object]:
        return {
            "permit_id": self.permit_id,
            "parent_id": self.parent_id,
            "sequence": self.sequence,
            "reason": self.reason,
            "inner_limit": self.inner_limit,
            "outer_limit": self.outer_limit,
            "optional": self.optional,
            "charged": True,
        }


class ParentMeter:
    """One parent allocation for all work and logical provider requests.

    ``model_calls`` is the compatibility name for logical provider requests.
    Transport retries reuse the request's durable permit; side calls, repairs,
    and child requests consume their own permits. The permit event is fsynced by
    the supplied continuation journal before the first transport attempt.
    """

    _COUNTER_KINDS = frozenset({
        "attempts", "model_calls", "responses", "challenges", "extraction_calls",
        "tool_work", "tools", "tests",
    })

    def __init__(self, *, attempts: int, steps: int | None, enforce: bool,
                 provider_limit: int | None = None, deadline: float | None = None,
                 journal: object | None = None):
        if (type(attempts) is not int or attempts < 1
                or (steps is not None and (type(steps) is not int or steps < 1))):
            raise ValueError("positive parent allocation required")
        if provider_limit is not None and (
                type(provider_limit) is not int or provider_limit < 1):
            raise ValueError("provider limit must be a positive integer")
        if deadline is not None and (
                isinstance(deadline, bool) or not isinstance(deadline, (int, float))
                or not math.isfinite(float(deadline))):
            raise ValueError("deadline must be a finite monotonic timestamp")
        inner_limit = None if steps is None else steps * POLICY.transport_attempts
        outer_limit = inner_limit if provider_limit is None else provider_limit
        effective_limit = minimum_limit(inner_limit, outer_limit)
        self.parent_id = uuid.uuid4().hex
        self.enforce = enforce
        self.provider_limits = {
            "inner": inner_limit, "outer": outer_limit, "effective": effective_limit,
        }
        self.limits = {
            "attempts": attempts,
            "model_calls": effective_limit,
            "responses": effective_limit,
            "challenges": 1,
            "extraction_calls": steps,
            "tool_work": None if steps is None else steps * 2,
            "tools": None if steps is None else steps * 2,
            "tests": steps,
            # StateFrame already copies ``limits`` into its run manifest.  Keep
            # these measured allocation facts there without making them charge
            # categories.
            "provider_inner_limit": inner_limit,
            "provider_outer_limit": outer_limit,
        }
        self.limits["total"] = (None if steps is None else
                                sum(self.limits[key] for key in self._COUNTER_KINDS))
        self.used = Counter()
        self.events: list[dict] = []
        self.completed_attempts: list[dict] = []
        self.deadline = float(deadline) if deadline is not None else None
        self._journal = journal
        self._durability_failed = False
        self._permits: dict[str, ProviderPermit] = {}
        self._pending_legacy_permits: deque[str] = deque()
        self._lock = threading.RLock()

    def bind_journal(self, journal: object) -> None:
        """Bind the existing durable journal; never create a second ledger."""
        if journal is None or not callable(getattr(journal, "emit", None)):
            raise ValueError("provider budget requires a durable journal")
        with self._lock:
            if self._journal is not None and self._journal is not journal:
                raise ValueError("parent budget cannot switch journals")
            self._journal = journal

    def _active_journal(self, journal: object | None = None) -> object | None:
        if journal is not None:
            return journal
        if self._journal is not None:
            return self._journal
        runtime = ACTIVE_RUNTIME.get()
        return getattr(runtime, "journal", None) if runtime is not None else None

    @staticmethod
    def _validate_permit_id(permit_id: object) -> str:
        if not isinstance(permit_id, str) or not permit_id.strip():
            raise ValueError("provider permit ID must be a non-empty string")
        return permit_id.strip()

    def _closure_profile(self, journal: object | None) -> dict[str, object]:
        """Derive closure reserve from already-recorded host timing receipts."""
        snapshot = getattr(journal, "host_span_snapshot", None)
        spans = snapshot() if callable(snapshot) else ()
        if not isinstance(spans, (tuple, list)):
            spans = ()
        closure_names = ("verification", "certification", "export", "finalize",
                         "observe", "settle")
        closure_elapsed = [
            int(row["elapsed_ns"])
            for row in spans
            if isinstance(row, dict)
            and isinstance(row.get("name"), str)
            and any(token in row["name"].lower() for token in closure_names)
            and type(row.get("elapsed_ns")) is int
            and row["elapsed_ns"] >= 0
        ]
        provider_elapsed = [
            int(row["elapsed_ns"])
            for row in spans
            if isinstance(row, dict) and row.get("name") == "provider"
            and type(row.get("elapsed_ns")) is int and row["elapsed_ns"] > 0
        ]
        reserve_ns = sum(closure_elapsed)
        provider_ns = max(provider_elapsed, default=0)
        reserve_slots = (
            int(math.ceil(reserve_ns / provider_ns))
            if reserve_ns and provider_ns else 0
        )
        return {
            "measured": bool(closure_elapsed),
            "source": "host_span_receipts" if closure_elapsed else "unmeasured",
            "reserve_ns": reserve_ns,
            "reserve_seconds": reserve_ns / 1_000_000_000,
            "provider_elapsed_ns": provider_ns,
            "provider_slots": reserve_slots,
        }

    def _append_event(self, event: dict[str, object], *, journal: object | None,
                      require_durable: bool = False) -> None:
        target = self._active_journal(journal)
        if target is None:
            if require_durable and self.enforce:
                self._durability_failed = True
                raise ParentBudgetExhausted(event["category"], "durability_unavailable")
            self.events.append(copy.deepcopy(event))
            return
        if self._journal is not None and target is not self._journal:
            self._durability_failed = True
            raise ParentBudgetExhausted(event["category"], "journal_identity_changed")
        try:
            from kernel_continuation_journal import journal_safe_fields
            target.emit("parent_budget", **journal_safe_fields(event))
        except Exception as error:
            self._durability_failed = True
            raise ParentBudgetExhausted(
                event["category"], "durability_failed") from error
        self._journal = target
        self.events.append(copy.deepcopy(event))

    def _blocked_event(self, *, permit_id: str, reason: str, journal: object | None,
                       optional: bool, closure: dict[str, object]) -> None:
        event = {
            "parent_id": self.parent_id,
            "category": "model_calls",
            "reason": reason,
            "blocked": True,
            "used_before": dict(self.used),
            "permit_id": permit_id,
            "inner_limit": self.provider_limits["inner"],
            "outer_limit": self.provider_limits["outer"],
            "optional": optional,
            "charged": False,
            "closure_reserve": closure,
        }
        self._append_event(event, journal=journal, require_durable=True)

    def admit_provider(self, *, permit_id: str | None = None, reason: str = "",
                       journal: object | None = None, optional: bool = True,
                       deadline: float | None = None,
                       preview: bool = False) -> ProviderPermit | None:
        """Durably admit one logical provider request before its first transport attempt."""
        if type(optional) is not bool:
            raise ValueError("optional must be boolean")
        if preview:
            return None
        target = self._active_journal(journal)
        effective_deadline = self.deadline if deadline is None else deadline
        if effective_deadline is not None and (
                isinstance(effective_deadline, bool)
                or not isinstance(effective_deadline, (int, float))
                or not math.isfinite(float(effective_deadline))):
            raise ValueError("deadline must be a finite monotonic timestamp")
        if permit_id is None:
            permit_id = f"{self.parent_id}:{uuid.uuid4().hex}"
        permit_id = self._validate_permit_id(permit_id)
        with self._lock:
            if self._durability_failed:
                raise ParentBudgetExhausted("model_calls", "durability_failed")
            now = time.monotonic()
            closure = self._closure_profile(target)
            remaining = (None if effective_deadline is None
                         else float(effective_deadline) - now)
            if remaining is not None and remaining <= 0 and self.enforce:
                self._blocked_event(permit_id=permit_id, reason="deadline_expired",
                                    journal=target, optional=optional, closure=closure)
                raise ParentBudgetExhausted("model_calls", "deadline_expired")
            existing = self._permits.get(permit_id)
            if existing is not None:
                if existing.optional != optional or (reason and existing.reason != reason):
                    raise ValueError("provider permit identity changed")
                return existing
            remaining_slots = minimum_limit(
                limit_remaining(self.limits["model_calls"], self.used["model_calls"]),
                limit_remaining(self.limits["total"], self.used["total"]),
            )
            reserve_slots = int(closure["provider_slots"])
            reserve_seconds = float(closure["reserve_seconds"])
            reserve_blocked = optional and (
                (remaining_slots is not None and remaining_slots <= reserve_slots)
                or (remaining is not None and remaining <= reserve_seconds))
            over_limit = remaining_slots == 0
            if (over_limit or reserve_blocked) and self.enforce:
                reason_code = "provider_cap_exhausted" if over_limit else "closure_reserve"
                self._blocked_event(permit_id=permit_id, reason=reason_code,
                                    journal=target, optional=optional, closure=closure)
                raise ParentBudgetExhausted("model_calls", reason_code)
            event = {
                "parent_id": self.parent_id,
                "category": "model_calls",
                "reason": reason,
                "blocked": False,
                "used_before": dict(self.used),
                "permit_id": permit_id,
                "inner_limit": self.provider_limits["inner"],
                "outer_limit": self.provider_limits["outer"],
                "optional": optional,
                "charged": True,
                "closure_reserve": closure,
            }
            self._append_event(event, journal=target, require_durable=True)
            self.used["model_calls"] += 1
            self.used["total"] += 1
            permit = ProviderPermit(
                permit_id, self.parent_id, self.used["model_calls"], reason,
                self.provider_limits["inner"], self.provider_limits["outer"], optional)
            self._permits[permit_id] = permit
            return permit

    # Explicit aliases keep the boundary discoverable to adapters without
    # introducing another accounting owner.
    provider_permit = admit_provider
    charge_provider = admit_provider

    def take_pending_provider_permit(self) -> str | None:
        with self._lock:
            return self._pending_legacy_permits.popleft() if self._pending_legacy_permits else None

    def provider_permit_for_id(self, permit_id: str) -> ProviderPermit | None:
        with self._lock:
            return self._permits.get(self._validate_permit_id(permit_id))

    def record_response(self, *, permit_id: str | None = None,
                       journal: object | None = None) -> None:
        """Persist the response counter without charging provider capacity again."""
        with self._lock:
            if permit_id is not None and permit_id not in self._permits:
                raise ValueError("response references an unknown provider permit")
            event = {
                "parent_id": self.parent_id,
                "category": "responses",
                "reason": "provider response received",
                "blocked": False,
                "used_before": dict(self.used),
                "permit_id": permit_id,
                "charged": True,
            }
            if reached(self.limits["responses"], self.used["responses"]):
                if self.enforce:
                    event["blocked"] = True
                    self._append_event(event, journal=journal)
                    raise ParentBudgetExhausted("responses", "response_cap_exhausted")
            self._append_event(event, journal=journal)
            self.used["responses"] += 1
            self.used["total"] += 1

    def charge(self, kind: str, *, reason: str = "", journal: object | None = None,
               optional: bool = True) -> ProviderPermit | None:
        if kind == "model_calls":
            permit = self.admit_provider(
                reason=reason, journal=journal, optional=optional)
            # Existing callers charge immediately before calling RequestCapture;
            # let that adapter consume this durable permit instead of charging a
            # second time at the wire boundary.
            with self._lock:
                self._pending_legacy_permits.append(permit.permit_id)
            return permit
        if kind not in self._COUNTER_KINDS:
            raise ValueError(f"unknown budget category: {kind}")
        with self._lock:
            blocked = (reached(self.limits[kind], self.used[kind])
                       or reached(self.limits["total"], self.used["total"]))
            event = {"parent_id": self.parent_id, "category": kind, "reason": reason,
                     "blocked": blocked and self.enforce, "used_before": dict(self.used)}
            self._append_event(event, journal=journal)
            if blocked and self.enforce:
                raise ParentBudgetExhausted(kind)
            self.used[kind] += 1
            self.used["total"] += 1


    @classmethod
    def restore(cls, journal, *, attempts: int, steps: int | None, enforce: bool,
                provider_limit: int | None = None) -> ParentMeter:
        """Fold a validated live journal, without charging or authorizing execution.

        A durable successful charge consumes its allocation even if the process
        died before updating RAM. Accounting that began outside this journal is
        not reconstructible and is rejected rather than assigned a fresh budget.
        """
        from kernel_continuation_journal import ContinuationJournal
        if not isinstance(journal, ContinuationJournal):
            raise ValueError("budget recovery requires a durable continuation journal")
        if (type(attempts) is not int or (steps is not None and type(steps) is not int)
                or type(enforce) is not bool):
            raise ValueError("invalid expected parent allocation")
        configured = journal.manifest.get("parent_budget")
        configured_limits = configured.get("limits") if isinstance(configured, dict) else None
        if provider_limit is None and isinstance(configured_limits, dict):
            candidate = configured_limits.get("provider_outer_limit")
            if type(candidate) is int:
                provider_limit = candidate
        parent = cls(attempts=attempts, steps=steps, enforce=enforce,
                     provider_limit=provider_limit)
        expected = {"limits": dict(parent.limits), "enforced": enforce}

        def counters(value):
            if (not isinstance(value, dict) or set(value) - (set(parent._COUNTER_KINDS) | {"total"})
                    or any(type(n) is not int or n < 0 for n in value.values())
                    or value.get("total", 0) != sum(
                        value.get(key, 0) for key in parent._COUNTER_KINDS)):
                raise ValueError("invalid parent counters")
            return Counter(value)

        def identify(value):
            if (not isinstance(value, str) or len(value) != 32
                    or any(c not in "0123456789abcdef" for c in value)):
                raise ValueError("invalid parent identity")
            return value

        identity = None
        attached = False
        with journal.ownership():
            if digest(journal.manifest.get("parent_budget")) != digest(expected):
                raise ValueError("parent allocation differs from current configuration")
            for record in journal.records():
                kind = record["kind"]
                if kind == "run_started" and (
                        digest(record.get("manifest")) != digest(journal.manifest)
                        or record.get("config_hash") != digest(journal.manifest)
                        or record.get("config_hash") != journal.config_hash):
                    raise ValueError("parent journal configuration identity changed")
                if kind == "attempt_terminated":
                    raise ValueError("terminated attempt cannot recover execution budget")
                if kind not in {"parent_budget", "parent_attached"}:
                    continue
                body = record if kind == "parent_budget" else record.get("budget")
                if not isinstance(body, dict):
                    raise ValueError("invalid parent accounting record")
                candidate = identify(body.get("parent_id"))
                if identity is not None and candidate != identity:
                    raise ValueError("mixed parent identities")
                identity = candidate
                if kind == "parent_attached":
                    if (digest(body.get("limits")) != digest(parent.limits)
                            or type(body.get("enforced")) is not bool
                            or body["enforced"] != enforce
                            or counters(body.get("used")) != parent.used
                            or (body.get("provider_limits") is not None
                                and body.get("provider_limits") != parent.provider_limits)):
                        raise ValueError("parent snapshot disagrees with durable charges")
                    attached = True
                    continue
                category = body.get("category")
                if not isinstance(category, str) or category not in parent._COUNTER_KINDS:
                    raise ValueError("invalid parent budget category")
                if counters(body.get("used_before")) != parent.used:
                    raise ValueError("parent charge sequence is missing or inconsistent")
                if type(body.get("blocked")) is not bool:
                    raise ValueError("invalid parent charge blocking flag")
                if category == "model_calls":
                    cap_blocked = (
                        reached(parent.limits[category], parent.used[category])
                        or reached(parent.limits["total"], parent.used["total"]))
                    blocked = bool(body["blocked"])
                    allowed_block_reasons = {
                        "provider_cap_exhausted", "closure_reserve", "deadline_expired",
                    }
                    if blocked and body.get("reason") not in allowed_block_reasons:
                        raise ValueError("invalid provider denial reason")
                    if not blocked and cap_blocked and enforce:
                        raise ValueError("provider charge crossed allocation")
                else:
                    blocked = enforce and (
                        reached(parent.limits[category], parent.used[category])
                        or reached(parent.limits["total"], parent.used["total"]))
                if type(body.get("blocked")) is not bool or body["blocked"] != blocked:
                    raise ValueError("parent charge blocking disagrees with allocation")
                if not isinstance(body.get("reason"), str):
                    raise ValueError("invalid parent charge reason")
                if category == "model_calls":
                    if body.get("inner_limit") != parent.provider_limits["inner"] \
                            or body.get("outer_limit") != parent.provider_limits["outer"]:
                        raise ValueError("provider permit limits changed")
                    if (type(body.get("charged")) is not bool
                            or body["charged"] != (not body["blocked"])
                            or type(body.get("optional")) is not bool):
                        raise ValueError("invalid provider permit charge fields")
                    permit_id = body.get("permit_id")
                    if not isinstance(permit_id, str):
                        raise ValueError("provider charge is missing permit identity")
                    if isinstance(permit_id, str) and permit_id in parent._permits:
                        raise ValueError("duplicate durable provider permit")
                event = copy.deepcopy({key: body[key] for key in body
                                       if key not in {"kind", "schema", "run_id", "task_id",
                                                      "event_id", "sequence", "timestamp_ns",
                                                      "journal_format", "previous_hash", "record_hash"}})
                parent.events.append(event)
                if not blocked:
                    parent.used[category] += 1
                    parent.used["total"] += 1
                    if category == "model_calls":
                        permit_id = body.get("permit_id")
                        if not isinstance(permit_id, str):
                            permit_id = f"legacy:{parent.parent_id}:{record.get('sequence')}"
                        parent._permits[permit_id] = ProviderPermit(
                            permit_id, parent.parent_id, parent.used[category],
                            str(body.get("reason", "")),
                            parent.provider_limits["inner"], parent.provider_limits["outer"],
                            bool(body.get("optional", False)))
            if identity is None or not attached:
                raise ValueError("missing original parent attachment")
        parent.parent_id = identity
        # A resumed meter must keep the exact journal it was folded from.  This
        # makes a post-restart admission durable even when the caller has not
        # yet rebound ACTIVE_RUNTIME or passed ``journal=`` explicitly.
        parent._journal = journal
        parent._permits = {
            permit_id: ProviderPermit(
                permit.permit_id, identity, permit.sequence, permit.reason,
                permit.inner_limit, permit.outer_limit, permit.optional)
            for permit_id, permit in parent._permits.items()
        }
        return parent

    def snapshot(self) -> dict:
        return {"parent_id": self.parent_id, "limits": dict(self.limits),
                "used": dict(self.used), "enforced": self.enforce,
                "provider_limits": dict(self.provider_limits),
                "closure_reserve": self._closure_profile(self._active_journal(None))}

    def incomplete_result(self, reason: str) -> dict[str, object]:
        return {
            "result": "INCOMPLETE", "outcome": "INCOMPLETE",
            "accepted": False, "certified": False, "certifiable": False,
            "reason": str(reason), "parent_budget": self.snapshot(),
        }


def charge(kind: str, reason: str = "") -> None:
    parent = ACTIVE_PARENT.get()
    if parent is not None:
        parent.charge(kind, reason=reason)


def provider_permit(*, permit_id: str | None = None, reason: str = "",
                    journal: object | None = None, optional: bool = True,
                    deadline: float | None = None, preview: bool = False) -> ProviderPermit | None:
    parent = ACTIVE_PARENT.get()
    if parent is None:
        return None
    return parent.admit_provider(
        permit_id=permit_id, reason=reason, journal=journal,
        optional=optional, deadline=deadline, preview=preview)


class ActionJournal:
    """Bounded facts, not transcript prose or cosmetic whole-state changes."""
    def __init__(self, bound: int = 64):
        self.actions: deque[dict] = deque(maxlen=bound)

    def append(self, *, kind: str, arguments: object, resources: dict, result: str,
               observation: object, artifact_change: bool = False,
               obligation_transition: dict | None = None, pending: str = "completed") -> dict:
        record = {"action_kind": kind, "canonical_arguments": arguments,
                  "resource_versions": resources, "result_kind": result,
                  "observation_id": digest(observation), "observation": copy.deepcopy(observation),
                  "artifact_change": artifact_change,
                  "obligation_transition": obligation_transition, "pending_status": pending}
        self.actions.append(copy.deepcopy(record))
        return record

    def diagnose(self) -> dict:
        actions = list(self.actions)
        keys = [digest({key: row[key] for key in (
            "action_kind", "canonical_arguments", "resource_versions", "result_kind",
            "observation_id")}) for row in actions]
        repeated = max(Counter(keys).values(), default=0)
        progress = [row for row in actions if row["obligation_transition"]
                    and row["obligation_transition"].get("authority") == "existing_grader"
                    and row["obligation_transition"].get("from") != "satisfied"
                    and row["obligation_transition"].get("to") == "satisfied"]
        return {
            "execution_liveness": "outcome_unknown" if any(
                row["pending_status"] in {"pending", "timeout", "unknown"} for row in actions)
                else "completed_actions_observed" if actions else "no_completed_actions",
            "repetition": {"status": "repeated" if repeated >= 3 else "no_repeated_cycle_observed",
                           "max_same_action_result": repeated, "window": len(actions)},
            "observed_progress": {"count": len(progress),
                                  "reason": "Only authoritative obligation satisfaction counts; reads and edits do not."}}


def retry_capsule(actions: list[dict], dependencies: dict) -> dict:
    """Carry evidence references only; no executable action or control state."""
    evidence = []
    for row in actions:
        if row.get("pending_status") != "completed":
            continue
        resources = row.get("resource_versions", {})
        if not resources or any(dependencies.get(k) != v for k, v in resources.items()):
            continue
        evidence.append({key: copy.deepcopy(row[key]) for key in (
            "action_kind", "canonical_arguments", "resource_versions", "result_kind", "observation_id", "observation")})
    return {"schema": "retry-evidence/1", "evidence": evidence,
            "verification": "unverified", "pending": None, "approvals": [],
            "prohibitions": [], "replay_actions": [],
            "reset": ["worktree", "database_identity", "verification", "pending", "leases",
                      "grants", "approvals", "attempt_budget"]}
