"""Durable reconciliation for external effects whose result is uncertain.

The adapter records an intent and a dispatch marker before asking a host-owned
destination to do anything.  A lookup can then say ``UNKNOWN``, ``PRESENT``
or ``ABSENT``.  UNKNOWN is never treated as absence: once a dispatch marker is
durable, a later ABSENT lookup cannot authorize another dispatch for the same
idempotency key.

The continuation runtime loads this module only when
``KERNEL_UNKNOWN_OUTCOMES=on``.  The adapter itself is deliberately small and
host-facing so a destination can be rebound after a process restart without
putting a second effect ledger beside the continuation journal.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Mapping, Sequence


MODE_ENV = "KERNEL_UNKNOWN_OUTCOMES"
SCHEMA = "external-outcome-reconciliation/1"

UNKNOWN = "UNKNOWN"
PRESENT = "PRESENT"
ABSENT = "ABSENT"
OUTCOMES = frozenset({UNKNOWN, PRESENT, ABSENT})

INTENT_EVENT = "external_effect_intent"
DISPATCH_EVENT = "external_effect_dispatch"
OUTCOME_EVENT = "external_effect_outcome"
RECONCILIATION_EVENT = "external_effect_reconciliation"


def mode() -> str:
    """Return the strict opt-in mode for this reconciliation mechanism."""
    value = os.environ.get(MODE_ENV, "off")
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ValueError("external effect data must be JSON-serializable") from error


def _copy_json(value: object) -> object:
    return json.loads(_canonical(value).decode("utf-8"))


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _key_parts(key: object) -> tuple[str, ...]:
    if isinstance(key, str):
        values: Sequence[object] = (key,)
    elif isinstance(key, Sequence) and not isinstance(key, (bytes, bytearray, str)):
        values = key
    else:
        raise TypeError("idempotency key must be a string or sequence of strings")
    if not values or len(values) > 8:
        raise ValueError("idempotency key must contain one to eight parts")
    return tuple(_text(value, "idempotency key part") for value in values)


def _key_id(key: tuple[str, ...]) -> str:
    return _digest(list(key))


def _destination_name(destination: object) -> str:
    name = getattr(destination, "name", None)
    if name is None:
        name = getattr(destination, "destination", None)
    return _text(name, "destination name")


def _validate_destination(destination: object) -> None:
    if not callable(getattr(destination, "lookup", None)):
        raise TypeError("destination must provide lookup(key, payload)")
    if not callable(getattr(destination, "dispatch", None)):
        raise TypeError("destination must provide dispatch(key, payload)")


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _normalize_destination_result(value: object, source: str) -> tuple[str, str | None, str | None]:
    """Normalize an untrusted destination result without making it positive by default."""
    status_value = value
    receipt_value = None
    if isinstance(value, Mapping):
        status_value = value.get("status")
        receipt_value = value.get("receipt_ref")
    if not isinstance(status_value, str) or status_value.strip().upper() not in OUTCOMES:
        return UNKNOWN, None, f"{source} returned an invalid outcome"
    try:
        receipt_ref = _optional_text(receipt_value, "receipt reference")
    except (TypeError, ValueError):
        return UNKNOWN, None, f"{source} returned an invalid receipt reference"
    return status_value.strip().upper(), receipt_ref, None


def _recorded_status(record: Mapping[str, object], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or value.strip().upper() not in OUTCOMES:
        raise ValueError(f"external effect journal field {field!r} is invalid")
    return value.strip().upper()


def _recorded_receipt(record: Mapping[str, object]) -> str | None:
    try:
        return _optional_text(record.get("receipt_ref"), "receipt reference")
    except (TypeError, ValueError) as error:
        raise ValueError("external effect journal receipt reference is invalid") from error


def _recorded_reason(record: Mapping[str, object]) -> str | None:
    reason = record.get("reason")
    if reason is not None and not isinstance(reason, str):
        raise ValueError("external effect journal reason is invalid")
    return reason


class ExternalEffectAdapter:
    """Journal-backed intent/lookup/dispatch reconciliation.

    A destination is a host-owned object with ``name``, ``lookup(key,
    payload)`` and ``dispatch(key, payload)``.  The adapter never calls
    ``dispatch`` unless a fresh lookup explicitly returned ABSENT and no
    dispatch marker exists for that durable idempotency key.
    """

    def __init__(self, journal: object, *, destination: object | None = None,
                 destinations: Mapping[str, object] | None = None):
        if not callable(getattr(journal, "emit", None)):
            raise TypeError("journal must provide emit(kind, **fields)")
        if not callable(getattr(journal, "records", None)):
            raise TypeError("journal must provide records()")
        self.journal = journal
        self._lock = threading.RLock()
        self._destinations: dict[str, object] = {}
        self._intents: dict[str, dict[str, object]] = {}

        if destinations is not None:
            if not isinstance(destinations, Mapping):
                raise TypeError("destinations must be a mapping")
            for name, bound in destinations.items():
                _validate_destination(bound)
                normalized_name = _text(name, "destination name")
                if _destination_name(bound) != normalized_name:
                    raise ValueError("destination binding name does not match destination.name")
                self._destinations[normalized_name] = bound
        if destination is not None:
            self.bind(destination)
        self._replay()

    def _replay(self) -> None:
        for record in self.journal.records():
            if not isinstance(record, Mapping):
                continue
            kind = record.get("kind")
            if kind == INTENT_EVENT:
                self._replay_intent(record)
            elif kind == DISPATCH_EVENT:
                intent = self._intent_for_record(record)
                self._validate_event_binding(record, intent, payload_hash=True)
                if record.get("dispatch_marker") != "DURABLE_BEFORE_CALL":
                    raise ValueError("external effect dispatch marker is invalid")
                attempt = record.get("attempt")
                expected_attempt = int(intent.get("dispatch_count", 0)) + 1
                if type(attempt) is not int or attempt != expected_attempt:
                    raise ValueError("external effect dispatch attempt is invalid")
                intent["dispatch_count"] = int(intent.get("dispatch_count", 0)) + 1
                intent["dispatch_started"] = True
            elif kind == OUTCOME_EVENT:
                intent = self._intent_for_record(record)
                self._validate_event_binding(record, intent)
                outcome = _recorded_status(record, "status")
                source = record.get("outcome_source", "journal")
                if (not intent.get("dispatch_started") and source != "lookup"):
                    raise ValueError("external effect outcome has no preceding dispatch")
                intent["status"] = outcome
                if not isinstance(source, str) or not source.strip():
                    raise ValueError("external effect outcome source is invalid")
                intent["outcome_source"] = source.strip()
                intent["reason"] = _recorded_reason(record)
                receipt_ref = _recorded_receipt(record)
                if receipt_ref is not None:
                    intent["receipt_ref"] = receipt_ref
                intent["outcome_event_ref"] = record.get("event_id")
            elif kind == RECONCILIATION_EVENT:
                intent = self._intent_for_record(record)
                self._validate_event_binding(record, intent)
                intent["last_observation"] = _recorded_status(record, "observed")
                current_status = _recorded_status(record, "status")
                if current_status != intent["status"]:
                    raise ValueError("external effect reconciliation status is out of order")
                if type(record.get("safe_to_dispatch")) is not bool:
                    raise ValueError("external effect reconciliation safety field is invalid")
                if record["safe_to_dispatch"]:
                    raise ValueError("external effect reconciliation cannot authorize dispatch")
                intent["reason"] = _recorded_reason(record)
                receipt_ref = _recorded_receipt(record)
                if receipt_ref is not None:
                    intent["receipt_ref"] = receipt_ref
        records = list(self.journal.records())
        for intent in self._intents.values():
            if intent["dispatch_started"] or intent["status"] not in {PRESENT, ABSENT}:
                continue
            outcome_ref = intent.get("outcome_event_ref")
            outcome_event = next((record for record in records
                                  if record.get("event_id") == outcome_ref), None)
            outcome_sequence = outcome_event.get("sequence") if isinstance(outcome_event, Mapping) else None
            reconciled = type(outcome_sequence) is int and any(
                record.get("kind") == RECONCILIATION_EVENT
                and record.get("external_effect_id") == intent["external_effect_id"]
                and type(record.get("sequence")) is int
                and record["sequence"] > outcome_sequence
                and record.get("observed") == intent["status"]
                and record.get("status") == intent["status"]
                and record.get("safe_to_dispatch") is False
                for record in records
            )
            if not reconciled:
                raise ValueError("external effect outcome has no preceding dispatch")

    def _replay_intent(self, record: Mapping[str, object]) -> None:
        if record.get("external_outcome_schema") != SCHEMA:
            raise ValueError("external effect intent schema is invalid")
        raw_key = record.get("idempotency_key")
        key = _key_parts(raw_key)
        effect_id = _text(record.get("external_effect_id"), "external effect id")
        if effect_id != _key_id(key):
            raise ValueError("external effect intent key digest mismatch")
        destination = _text(record.get("destination"), "destination")
        payload = _copy_json(record.get("payload"))
        payload_hash = _text(record.get("payload_hash"), "payload hash")
        if payload_hash != _digest(payload):
            raise ValueError("external effect payload digest mismatch")
        prior = self._intents.get(effect_id)
        if prior is not None:
            if (prior["idempotency_key"] != list(key)
                    or prior["destination"] != destination
                    or prior["payload_hash"] != payload_hash):
                raise ValueError("conflicting external effect intent")
            return
        self._intents[effect_id] = {
            "external_effect_id": effect_id,
            "idempotency_key": list(key),
            "destination": destination,
            "payload": payload,
            "payload_hash": payload_hash,
            "status": UNKNOWN,
            "dispatch_started": False,
            "dispatch_count": 0,
            "outcome_source": None,
            "reason": None,
            "last_observation": UNKNOWN,
            "receipt_ref": None,
            "outcome_event_ref": None,
        }

    @staticmethod
    def _validate_event_binding(record: Mapping[str, object], intent: Mapping[str, object], *,
                                payload_hash: bool = False) -> None:
        if record.get("external_outcome_schema") != SCHEMA:
            raise ValueError("external effect event schema is invalid")
        try:
            key = list(_key_parts(record.get("idempotency_key")))
        except (TypeError, ValueError) as error:
            raise ValueError("external effect event idempotency key is invalid") from error
        if key != intent["idempotency_key"]:
            raise ValueError("external effect event key does not match its intent")
        try:
            destination = _text(record.get("destination"), "destination")
        except (TypeError, ValueError) as error:
            raise ValueError("external effect event destination is invalid") from error
        if destination != intent["destination"]:
            raise ValueError("external effect event destination does not match its intent")
        if payload_hash:
            supplied = record.get("payload_hash")
            if supplied != intent["payload_hash"]:
                raise ValueError("external effect dispatch payload does not match its intent")

    def _intent_for_record(self, record: Mapping[str, object]) -> dict[str, object]:
        effect_id = _text(record.get("external_effect_id"), "external effect id")
        try:
            return self._intents[effect_id]
        except KeyError as error:
            raise ValueError("external effect event has no preceding intent") from error

    def bind(self, destination: object) -> str:
        """Bind a host destination by its durable name after construction/resume."""
        with self._lock:
            _validate_destination(destination)
            name = _destination_name(destination)
            self._destinations[name] = destination
            return name

    def _bound_destination(self, name: str) -> object | None:
        return self._destinations.get(name)

    def _intent(self, key: object) -> tuple[str, dict[str, object]]:
        normalized = _key_parts(key)
        effect_id = _key_id(normalized)
        try:
            return effect_id, self._intents[effect_id]
        except KeyError as error:
            raise KeyError(f"unknown external effect key: {list(normalized)!r}") from error

    def begin(self, key: object, payload: object, *, destination: object | None = None,
              destination_name: str | None = None) -> dict[str, object]:
        """Durably register one external intent before any lookup or dispatch."""
        normalized_key = _key_parts(key)
        effect_id = _key_id(normalized_key)
        frozen_payload = _copy_json(payload)
        payload_hash = _digest(frozen_payload)

        with self._lock:
            bound_name = None
            if destination is not None:
                bound_name = self.bind(destination)
            if destination_name is not None:
                requested_name = _text(destination_name, "destination name")
                if bound_name is not None and bound_name != requested_name:
                    raise ValueError("destination and destination_name disagree")
                bound_name = requested_name

            prior = self._intents.get(effect_id)
            if prior is not None:
                if (prior["idempotency_key"] != list(normalized_key)
                        or prior["payload_hash"] != payload_hash):
                    raise ValueError("external effect key was reused with different intent")
                if bound_name is not None and prior["destination"] != bound_name:
                    raise ValueError("external effect key was rebound to a different destination")
                return self.snapshot(normalized_key)

            if bound_name is None:
                names = [name for name in self._destinations]
                if len(names) != 1:
                    raise ValueError("destination or destination_name is required")
                bound_name = names[0]

            self.journal.emit(
                INTENT_EVENT,
                external_outcome_schema=SCHEMA,
                external_effect_id=effect_id,
                idempotency_key=list(normalized_key),
                destination=bound_name,
                payload=frozen_payload,
                payload_hash=payload_hash,
            )
            self._intents[effect_id] = {
                "external_effect_id": effect_id,
                "idempotency_key": list(normalized_key),
                "destination": bound_name,
                "payload": frozen_payload,
                "payload_hash": payload_hash,
                "status": UNKNOWN,
                "dispatch_started": False,
                "dispatch_count": 0,
                "outcome_source": None,
                "reason": None,
            "last_observation": UNKNOWN,
            "receipt_ref": None,
            "outcome_event_ref": None,
            }
            return self.snapshot(normalized_key)

    def _record_reconciliation(self, intent: dict[str, object], *, observed: str,
                               reason: str, safe_to_dispatch: bool) -> None:
        intent["last_observation"] = observed
        intent["reason"] = reason
        self.journal.emit(
            RECONCILIATION_EVENT,
            external_outcome_schema=SCHEMA,
            external_effect_id=intent["external_effect_id"],
            idempotency_key=intent["idempotency_key"],
            destination=intent["destination"],
            observed=observed,
            status=intent["status"],
            safe_to_dispatch=safe_to_dispatch,
            reason=reason,
            receipt_ref=intent.get("receipt_ref"),
        )

    def _record_outcome(self, intent: dict[str, object], *, outcome: str,
                        source: str, reason: str | None = None,
                        receipt_ref: str | None = None) -> None:
        intent["status"] = outcome
        intent["outcome_source"] = source
        intent["reason"] = reason
        if receipt_ref is not None:
            intent["receipt_ref"] = receipt_ref
        event = self.journal.emit(
            OUTCOME_EVENT,
            external_outcome_schema=SCHEMA,
            external_effect_id=intent["external_effect_id"],
            idempotency_key=intent["idempotency_key"],
            destination=intent["destination"],
            status=outcome,
            outcome_source=source,
            reason=reason,
            receipt_ref=intent.get("receipt_ref"),
        )
        intent["outcome_event_ref"] = event["event_id"]

    def _record_dispatch(self, intent: dict[str, object]) -> None:
        attempt = int(intent.get("dispatch_count", 0)) + 1
        self.journal.emit(
            DISPATCH_EVENT,
            external_outcome_schema=SCHEMA,
            external_effect_id=intent["external_effect_id"],
            idempotency_key=intent["idempotency_key"],
            destination=intent["destination"],
            payload_hash=intent["payload_hash"],
            attempt=attempt,
            dispatch_marker="DURABLE_BEFORE_CALL",
        )
        intent["dispatch_started"] = True
        intent["dispatch_count"] = attempt

    def reconcile(self, key: object) -> dict[str, object]:
        """Reconcile one intent, refusing every unsafe retry."""
        with self._lock:
            _, intent = self._intent(key)
            if intent["status"] in {PRESENT, ABSENT}:
                return self.snapshot(key)

            destination = self._bound_destination(str(intent["destination"]))
            if destination is None:
                self._record_reconciliation(
                    intent,
                    observed=UNKNOWN,
                    reason="destination is not bound",
                    safe_to_dispatch=False,
                )
                return self.snapshot(key)

            try:
                observed_raw = destination.lookup(tuple(intent["idempotency_key"]),
                                                  _copy_json(intent["payload"]))
            except Exception as error:  # a failed lookup is itself UNKNOWN
                observed, receipt_ref = UNKNOWN, None
                lookup_reason = f"lookup raised {type(error).__name__}"
            else:
                observed, receipt_ref, lookup_reason = _normalize_destination_result(
                    observed_raw, "lookup")
                if lookup_reason is None:
                    lookup_reason = "lookup reconciled"

            if observed == PRESENT:
                self._record_outcome(intent, outcome=PRESENT, source="lookup",
                                     receipt_ref=receipt_ref)
                return self.snapshot(key)

            if observed == ABSENT and not intent["dispatch_started"]:
                # This append is the crash boundary: no destination call can occur
                # before the durable marker is visible in the journal.
                self._record_dispatch(intent)
                try:
                    dispatched_raw = destination.dispatch(
                        tuple(intent["idempotency_key"]), _copy_json(intent["payload"]))
                except Exception as error:  # marker makes this ambiguous, never retry
                    dispatched, dispatch_receipt = UNKNOWN, None
                    dispatch_reason = f"dispatch raised {type(error).__name__}"
                else:
                    dispatched, dispatch_receipt, dispatch_reason = _normalize_destination_result(
                        dispatched_raw, "dispatch")
                self._record_outcome(intent, outcome=dispatched, source="dispatch",
                                     reason=dispatch_reason, receipt_ref=dispatch_receipt)
                return self.snapshot(key)

            if observed == ABSENT and intent["dispatch_started"]:
                self._record_reconciliation(
                    intent,
                    observed=ABSENT,
                    reason="dispatch marker exists; ABSENT cannot authorize a retry",
                    safe_to_dispatch=False,
                )
                return self.snapshot(key)

            self._record_outcome(intent, outcome=UNKNOWN, source="lookup",
                                 reason=lookup_reason, receipt_ref=receipt_ref)
            self._record_reconciliation(
                intent,
                observed=UNKNOWN,
                reason=("lookup is unresolved; no dispatch authorized"
                        if lookup_reason == "lookup reconciled" else lookup_reason),
                safe_to_dispatch=False,
            )
            return self.snapshot(key)

    def reconcile_all(self) -> list[dict[str, object]]:
        with self._lock:
            keys = [list(intent["idempotency_key"]) for intent in self._intents.values()]
        return [self.reconcile(key) for key in keys]

    def reconcile_pending(self) -> list[dict[str, object]]:
        return self.reconcile_all()

    def snapshot(self, key: object) -> dict[str, object]:
        with self._lock:
            _, intent = self._intent(key)
            return {
                "schema": SCHEMA,
                "external_effect_id": intent["external_effect_id"],
                "idempotency_key": list(intent["idempotency_key"]),
                "destination": intent["destination"],
                "status": intent["status"],
                "dispatch_started": bool(intent["dispatch_started"]),
                "dispatch_count": int(intent["dispatch_count"]),
                "outcome_source": intent["outcome_source"],
                "last_observation": intent["last_observation"],
                "receipt_ref": intent.get("receipt_ref"),
                "outcome_event_ref": intent.get("outcome_event_ref"),
                "reason": intent["reason"],
                "safe_to_dispatch": False,
            }

    def status(self, key: object) -> str:
        return str(self.snapshot(key)["status"])

    def context(self) -> dict[str, object]:
        with self._lock:
            snapshots = [self.snapshot(intent["idempotency_key"])
                         for intent in self._intents.values()]
            pending = [item for item in snapshots if item["status"] == UNKNOWN]
            resolved = [item for item in snapshots if item["status"] != UNKNOWN]
            return {"schema": SCHEMA, "pending": pending, "resolved": resolved}
