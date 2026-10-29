"""Crash-safe, keyed effect sequences for the opt-in R-040 slice.

The journal records the local protocol, while the destination owns the
effect/idempotency transaction.  A local journal cannot prove whether an
arbitrary external call happened between two fsyncs; therefore a destination
must expose ``submit(key)`` and ``result_for(key)`` and bind both operations to
one explicit durable namespace.  Lookup failures stay UNKNOWN and never turn
into a blind retry.
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping

from kernel_measurement import digest


MODE_ENV = "STATEFRAME_GUARDED_EFFECTS"
SCHEMA = "guarded-effect-sequence/1"
_FALLBACK_OUTCOMES = frozenset({"UNKNOWN", "PRESENT", "ABSENT"})
PRODUCER = "kernel_guarded_effects"
INTENT_EVENT = "guarded_effect_intent"
DISPATCH_EVENT = "guarded_effect_dispatch"
RECEIPT_EVENT = "guarded_effect_receipt"
ACCEPTANCE_EVENT = "guarded_effect_acceptance"


class GuardedEffectError(RuntimeError):
    """The local effect protocol cannot safely authorize or reconcile work."""


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = (source.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(runtime: object | None = None) -> bool:
    if runtime is not None:
        value = getattr(runtime, "stateframe_guarded_effects_mode", None)
        if value is not None:
            normalized = str(value).strip().lower()
            if normalized not in {"off", "on"}:
                raise ValueError(f"{MODE_ENV} must be off or on")
            return normalized == "on"
    return mode() == "on"


def _effect_id(effect_id: object) -> str:
    if not isinstance(effect_id, str) or not effect_id.strip():
        raise GuardedEffectError("guarded effect id must be a non-empty string")
    return effect_id.strip()


def _outcome_vocabulary() -> frozenset[str]:
    """Use R-041's tri-state vocabulary when the merged module is available."""
    try:
        from kernel_unknown_outcomes import OUTCOMES
    except ImportError:
        return _FALLBACK_OUTCOMES
    return OUTCOMES


def _destination_name(destination: object) -> str:
    name = getattr(destination, "name", None)
    if (not isinstance(name, str) or not name.strip()
            or name.strip().lower() in {"destination", "default"}):
        raise GuardedEffectError(
            "guarded effects require a destination with an explicit stable identity")
    if not callable(getattr(destination, "submit", None)):
        raise GuardedEffectError("guarded effect destination needs submit(key)")
    if not callable(getattr(destination, "result_for", None)):
        raise GuardedEffectError("guarded effect destination needs result_for(key)")
    return name.strip()


def _json_value(value: object, *, label: str) -> object:
    """Validate and freeze values before they become durable event fields."""
    try:
        return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False))
    except (TypeError, ValueError, UnicodeError) as error:
        raise GuardedEffectError(f"{label} must be JSON-compatible") from error


class GuardedEffectSequence:
    """One journal-bound intent/dispatch/receipt sequence.

    ``effect_id`` is minted by the caller at the action boundary and is reused
    on recovery.  The durable idempotency key is ``(journal.run_id,
    effect_id)``.  ``destination`` is optional for an audit-only live hook;
    without it, recovery is UNKNOWN and no external call is attempted.
    """

    def __init__(self, journal, *, destination: object | None = None,
                 destination_provider: Callable[[str | None], object | None] | None = None,
                 enabled_override: bool | None = None):
        self.journal = journal
        self.destination = destination
        self.destination_provider = destination_provider
        self.enabled = enabled() if enabled_override is None else bool(enabled_override)

    def idempotency_key(self, effect_id: object) -> tuple[str, str]:
        """Return the stable destination key reused by every recovery attempt."""
        return (str(self.journal.run_id), _effect_id(effect_id))

    def _key(self, effect_id: object) -> tuple[str, str]:
        return self.idempotency_key(effect_id)

    def _records(self) -> list[dict]:
        records = getattr(self.journal, "records", None)
        if callable(records):
            records = records()
        if records is None or isinstance(records, (str, bytes, Mapping)):
            return []
        try:
            values = list(records)
        except TypeError:
            return []
        return [event for event in values if isinstance(event, Mapping)]

    def _state(self, effect_id: object) -> dict | None:
        effect = _effect_id(effect_id)
        state: dict | None = None
        for event in self._records():
            if event.get("kind") not in {
                    INTENT_EVENT, DISPATCH_EVENT, RECEIPT_EVENT, ACCEPTANCE_EVENT}:
                continue
            if event.get("effect_id") != effect:
                continue
            state = state or {
                "effect_id": effect, "intent": None, "dispatch": None,
                "receipt": None, "acceptance": None,
            }
            state_kind = event["kind"]
            if state_kind == INTENT_EVENT:
                state["intent"] = event
            elif state_kind == DISPATCH_EVENT:
                state["dispatch"] = event
            elif state_kind == RECEIPT_EVENT:
                state["receipt"] = event
            elif state_kind == ACCEPTANCE_EVENT:
                state["acceptance"] = event
        return state

    @staticmethod
    def _local_status(state: dict | None) -> str:
        if state is None:
            return "ABSENT"
        if state.get("receipt") is not None:
            outcome = str(state["receipt"].get("outcome", "PRESENT")).upper()
            return outcome if outcome in _outcome_vocabulary() else "UNKNOWN"
        if state.get("acceptance") is not None:
            return "PRESENT"
        if state.get("intent") is not None:
            return "PENDING"
        return "ABSENT"

    def _snapshot(self, effect_id: object, *, status: str | None = None,
                  retry: bool = False, reason: str | None = None) -> dict:
        state = self._state(effect_id)
        effect = _effect_id(effect_id)
        intent = state.get("intent") if state else None
        receipt = state.get("receipt") if state else None
        result = {
            "schema": SCHEMA,
            "effect_id": effect,
            "idempotency_key": list(self._key(effect)),
            "destination": intent.get("destination") if intent else None,
            "status": status or self._local_status(state),
            "retry": bool(retry),
            "has_intent": bool(state and state.get("intent")),
            "has_dispatch": bool(state and state.get("dispatch")),
            "has_receipt": bool(receipt),
            "accepted": bool(state and state.get("acceptance")),
        }
        if receipt is not None:
            result["receipt"] = receipt.get("receipt")
            result["outcome"] = receipt.get("outcome", "PRESENT")
            result["receipt_ref"] = receipt.get("receipt_ref") or receipt.get("event_id")
        if reason is not None:
            result["reason"] = reason
        return result

    def _resolve_destination(self, expected_name: str | None = None):
        candidate = self.destination
        if candidate is None and callable(self.destination_provider):
            try:
                candidate = self.destination_provider(expected_name)
            except Exception as error:
                raise GuardedEffectError(
                    f"destination lookup failed: {type(error).__name__}: {error}") from error
        if candidate is None:
            return None
        actual_name = _destination_name(candidate)
        if expected_name is not None and actual_name != expected_name:
            raise GuardedEffectError(
                f"destination identity mismatch: expected {expected_name!r}, got {actual_name!r}")
        return candidate

    def _emit(self, kind: str, **fields: object) -> dict:
        # Keep ``schema`` for the journal's own envelope; event payloads use a
        # distinct field so reserved-field sweeps cannot be bypassed.
        return self.journal.emit(kind, event_schema=SCHEMA, producer=PRODUCER, **fields)

    def begin(self, effect_id: object, payload: object, *, destination: object | None = None,
              origin_ref: str | None = None) -> dict | None:
        if not self.enabled:
            return None
        effect = _effect_id(effect_id)
        frozen_payload = _json_value(payload, label="effect payload")
        payload_hash = digest(frozen_payload)
        candidate = destination if destination is not None else self.destination
        if candidate is None and callable(self.destination_provider):
            candidate = self._resolve_destination(None)
        name = _destination_name(candidate) if candidate is not None else None
        if destination is not None and self.destination is None:
            # An explicitly supplied destination is the caller's binding for
            # this sequence; preserve it for dispatch after the intent write.
            self.destination = candidate
        existing = self._state(effect)
        if existing is not None:
            intent = existing.get("intent")
            if intent is None:
                raise GuardedEffectError("effect journal is missing its durable intent")
            if (intent.get("payload_digest") != payload_hash
                    or intent.get("destination") != name
                    or intent.get("idempotency_key") != list(self._key(effect))):
                raise GuardedEffectError(
                    f"effect {effect!r} was already bound to a different payload or destination")
            return intent
        fields: dict[str, object] = {
            "effect_id": effect,
            "idempotency_key": list(self._key(effect)),
            "destination": name,
            "payload_digest": payload_hash,
        }
        if origin_ref is not None:
            fields["origin_ref"] = str(origin_ref)
        return self._emit(INTENT_EVENT, **fields)

    def mark_dispatched(self, effect_id: object, *, dispatch_ref: str | None = None) -> dict | None:
        if not self.enabled:
            return None
        effect = _effect_id(effect_id)
        state = self._state(effect)
        if state is None or state.get("intent") is None:
            raise GuardedEffectError(f"effect {effect!r} has no durable intent")
        if state.get("dispatch") is not None:
            return state["dispatch"]
        intent = state["intent"]
        fields: dict[str, object] = {
            "effect_id": effect,
            "idempotency_key": list(self._key(effect)),
            "destination": intent.get("destination"),
            "payload_digest": intent.get("payload_digest"),
            "intent_ref": intent.get("event_id"),
            "dispatch_status": "started",
        }
        if dispatch_ref is not None:
            fields["dispatch_ref"] = str(dispatch_ref)
        return self._emit(DISPATCH_EVENT, **fields)

    def _outcome(self, effect_id: object, status: str, *, retry: bool,
                 reason: str | None = None) -> dict:
        return self._snapshot(effect_id, status=status, retry=retry, reason=reason)

    def record_receipt(self, effect_id: object, receipt: object, *, outcome: str = "PRESENT",
                       receipt_ref: str | None = None) -> dict | None:
        if not self.enabled:
            return None
        effect = _effect_id(effect_id)
        if receipt is None:
            raise GuardedEffectError("a null destination result is not a receipt")
        frozen = _json_value(receipt, label="effect receipt")
        state = self._state(effect)
        if state is None or state.get("intent") is None:
            raise GuardedEffectError(f"effect {effect!r} has no durable intent")
        if state.get("dispatch") is None:
            raise GuardedEffectError(f"effect {effect!r} has no durable dispatch")
        existing = state.get("receipt")
        if existing is not None:
            if (existing.get("receipt") != frozen
                    or str(existing.get("outcome", "PRESENT")).upper()
                    != str(outcome).strip().upper()):
                raise GuardedEffectError(f"effect {effect!r} has conflicting receipts")
            return existing
        normalized_outcome = str(outcome).strip().upper()
        if normalized_outcome not in _outcome_vocabulary():
            raise GuardedEffectError("effect receipt outcome is invalid")
        intent = state["intent"]
        dispatch = state.get("dispatch")
        fields: dict[str, object] = {
            "effect_id": effect,
            "idempotency_key": list(self._key(effect)),
            "destination": intent.get("destination"),
            "payload_digest": intent.get("payload_digest"),
            "intent_ref": intent.get("event_id"),
            "dispatch_ref": dispatch.get("event_id") if dispatch else None,
            "receipt": frozen,
            "receipt_digest": digest(frozen),
            "outcome": normalized_outcome,
        }
        if receipt_ref is not None:
            fields["receipt_ref"] = str(receipt_ref)
        return self._emit(RECEIPT_EVENT, **fields)

    def _lookup(self, effect_id: str, state: dict, destination: object | None = None) -> dict:
        intent = state.get("intent")
        expected_name = intent.get("destination") if intent else None
        try:
            target = destination if destination is not None else self._resolve_destination(expected_name)
        except GuardedEffectError as error:
            return self._outcome(
                effect_id, "UNKNOWN", retry=False,
                reason=f"destination lookup failed: {type(error).__name__}")
        if target is None:
            return self._outcome(effect_id, "UNKNOWN", retry=False,
                                 reason="destination unavailable")
        try:
            result = target.result_for(self._key(effect_id))
        except Exception as error:
            return self._outcome(
                effect_id, "UNKNOWN", retry=False,
                reason=f"destination lookup failed: {type(error).__name__}")
        if result is None:
            return self._outcome(effect_id, "ABSENT", retry=True)
        if result is False:
            return self._outcome(effect_id, "UNKNOWN", retry=False,
                                 reason="destination returned an invalid confirmation")
        self.record_receipt(effect_id, result, outcome="PRESENT")
        return self._outcome(effect_id, "PRESENT", retry=False)

    def reconcile(self, effect_id: object) -> dict | None:
        if not self.enabled:
            return None
        effect = _effect_id(effect_id)
        state = self._state(effect)
        if state is None:
            return self._outcome(effect, "ABSENT", retry=False)
        local = self._local_status(state)
        if state.get("receipt") is not None:
            return self._outcome(effect, local, retry=False)
        return self._lookup(effect, state)

    def dispatch(self, effect_id: object, *, dispatch_ref: str | None = None) -> dict | None:
        if not self.enabled:
            return None
        effect = _effect_id(effect_id)
        state = self._state(effect)
        if state is None or state.get("intent") is None:
            raise GuardedEffectError(f"effect {effect!r} has no durable intent")
        if state.get("receipt") is not None:
            return self._outcome(effect, self._local_status(state), retry=False)
        # A repeated dispatch after a crash first asks the destination.  This
        # is the local half of the no-duplicate guarantee; UNKNOWN stops here.
        if state.get("dispatch") is not None:
            checked = self._lookup(effect, state)
            if checked["status"] != "ABSENT":
                return checked
        self.mark_dispatched(effect, dispatch_ref=dispatch_ref)
        state = self._state(effect)
        intent = state["intent"]
        try:
            target = self._resolve_destination(intent.get("destination"))
        except GuardedEffectError as error:
            return self._outcome(
                effect, "UNKNOWN", retry=False,
                reason=f"destination lookup failed: {type(error).__name__}")
        if target is None:
            return self._outcome(effect, "UNKNOWN", retry=False,
                                 reason="destination unavailable")
        try:
            result = target.submit(self._key(effect))
        except Exception as error:
            # A normal transport failure is still ambiguous: the destination
            # may have committed before reporting the failure.  Do not let the
            # caller treat it as a safe retry without a later lookup.
            return self._outcome(
                effect, "UNKNOWN", retry=False,
                reason=f"destination submit failed: {type(error).__name__}")
        except BaseException:
            # Process-death simulations and real interrupts leave the durable
            # dispatch marker in place for the next worker to reconcile.
            raise
        if result is False:
            return self._outcome(effect, "ABSENT", retry=True, reason="destination refused")
        if result is None:
            return self._outcome(effect, "UNKNOWN", retry=False,
                                 reason="destination returned no confirmation")
        self.record_receipt(effect, result, outcome="PRESENT")
        return self._outcome(effect, "PRESENT", retry=False)

    def recover(self, effect_id: object) -> dict | None:
        if not self.enabled:
            return None
        effect = _effect_id(effect_id)
        state = self._state(effect)
        if state is None:
            return self._outcome(effect, "ABSENT", retry=False)
        if state.get("receipt") is not None:
            return self._outcome(effect, self._local_status(state), retry=False)
        checked = self.reconcile(effect)
        if checked["status"] != "ABSENT":
            return checked
        # ABSENT is the only safe retry result.  UNKNOWN is returned above and
        # never reaches this call, so a lost lookup cannot become a duplicate.
        return self.dispatch(effect)

    def accept(self, effect_id: object) -> dict | None:
        if not self.enabled:
            return None
        effect = _effect_id(effect_id)
        state = self._state(effect)
        if state is None or state.get("intent") is None:
            raise GuardedEffectError(f"effect {effect!r} has no durable intent")
        if state.get("acceptance") is not None:
            return self._outcome(effect, "PRESENT", retry=False)
        if state.get("receipt") is None:
            raise GuardedEffectError(f"effect {effect!r} cannot be accepted without a receipt")
        receipt = state["receipt"]
        if self._local_status(state) != "PRESENT":
            raise GuardedEffectError(f"effect {effect!r} does not have a positive receipt")
        self._emit(
            ACCEPTANCE_EVENT,
            effect_id=effect,
            idempotency_key=list(self._key(effect)),
            destination=state["intent"].get("destination"),
            receipt_ref=receipt.get("event_id"),
            accepted=True,
        )
        return self._outcome(effect, "PRESENT", retry=False)

    def context_dict(self) -> dict:
        if not self.enabled:
            return {}
        effects: dict[str, dict] = {}
        for event in self._records():
            if event.get("kind") == INTENT_EVENT and isinstance(event.get("effect_id"), str):
                effects[event["effect_id"]] = self._snapshot(event["effect_id"])
        return {
            "schema": SCHEMA,
            "event_schema": SCHEMA,
            "effects": list(effects.values()),
        }
