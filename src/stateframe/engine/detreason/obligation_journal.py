"""Versioned, replayable journal adapter for the single ObligationLedger owner."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import fields, is_dataclass
from typing import Mapping, Sequence

from . import obligations as ob


SCHEMA = "obligation-ledger/1"


class ObligationJournalError(RuntimeError):
    """Missing, corrupt, or uncommitted obligation state cannot authorize work."""


class ObligationJournalAdapter:
    """Persists complete ledger projections before publishing ledger events.

    ``journal`` is the existing ContinuationJournal-compatible emit boundary.
    The adapter is opt-in and does not attach itself to a live request path.
    """

    def __init__(self, journal: object | None = None) -> None:
        self.journal = journal
        self._ledger: ob.ObligationLedger | None = None
        self._tail: list[dict[str, object]] = []
        self._cursor = 0
        self._head = ""
        self._projection: dict[str, object] | None = None
        self._fault = ""

    def attach(self, ledger: ob.ObligationLedger) -> None:
        if self._ledger is not None and self._ledger is not ledger:
            raise ObligationJournalError("adapter already owns a different obligation ledger")
        if getattr(ledger, "_journal_adapter", None) not in (None, self):
            raise ObligationJournalError("obligation ledger already has a journal adapter")
        self._ledger = ledger
        self._projection = _capture(ledger)
        ledger._journal_adapter = self
        if self.journal is not None:
            try:
                _emit(self.journal, "obligation_snapshot", obligation=self.checkpoint())
            except BaseException as error:
                self._fault = "initial obligation snapshot persistence failed"
                raise ObligationJournalError(self._fault) from error

    def detach(self) -> None:
        if self._ledger is not None and getattr(self._ledger, "_journal_adapter", None) is self:
            self._ledger._journal_adapter = None
        self._ledger = None

    def _require_ledger(self) -> ob.ObligationLedger:
        if self._ledger is None or self._fault:
            raise ObligationJournalError("fork handoff requires a healthy attached obligation owner")
        return self._ledger

    def persist_ledger_event(self, ledger: ob.ObligationLedger, event: Mapping[str, object]) -> None:
        if ledger is not self._ledger:
            raise ObligationJournalError("event came from an unattached obligation ledger")
        if self._fault:
            raise ObligationJournalError(self._fault)
        projected = _capture(ledger, pending_event=event)
        record = self._make_record("obligation_transition", event, projected)
        if self.journal is not None:
            try:
                _emit(self.journal, "obligation_transition", obligation=copy.deepcopy(record))
            except BaseException as error:
                self._fault = "obligation event write is uncertain; replay is required"
                raise ObligationJournalError(self._fault) from error
        self._commit_record(record, projected)

    def persist_ledger_transaction(
            self, ledger: ob.ObligationLedger,
            events: Sequence[Mapping[str, object]]) -> None:
        """Write all events and the final projection through one journal append."""
        if ledger is not self._ledger:
            raise ObligationJournalError("transaction came from an unattached obligation ledger")
        if self._fault:
            raise ObligationJournalError(self._fault)
        if not events:
            raise ObligationJournalError("empty obligation transaction")
        payload = {"events": tuple(copy.deepcopy(dict(event)) for event in events)}
        projected = _capture(ledger, pending_events=events)
        record = self._make_record("obligation_transaction", payload, projected)
        if self.journal is not None:
            try:
                _emit(self.journal, "obligation_transaction", obligation=copy.deepcopy(record))
            except BaseException as error:
                self._fault = "atomic obligation transaction write is uncertain; replay is required"
                raise ObligationJournalError(self._fault) from error
        self._commit_record(record, projected)

    def stage_fork_handoff(
        self, obligation_id: str, fork_id: str, *, expected_revision: int, event_id: str,
    ) -> dict[str, object]:
        """Ask the obligation owner to stage one assignment CAS without writing it."""
        ledger = self._require_ledger()
        stage = ledger.stage_fork_handoff(
            obligation_id, fork_id, expected_revision=expected_revision, event_id=event_id)
        return stage.as_dict()

    def apply_committed_fork_handoff(
        self, stage_value: Mapping[str, object], committed_entry: object,
    ) -> ob.Obligation:
        """Apply assignment only after the shared root event is durably committed."""
        ledger = self._require_ledger()
        stage = _fork_handoff_stage(stage_value)
        entry_id = getattr(committed_entry, "entry_id", None)
        entry_kind = getattr(committed_entry, "kind", None)
        entry_payload = getattr(committed_entry, "payload", None)
        if (entry_id != stage.event_id or entry_kind != "OBLIGATION_HANDOFF"
                or not isinstance(entry_payload, Mapping)
                or entry_payload.get("handoff") != stage.as_dict()
                or entry_payload.get("commit_notice") != "root_committed"):
            raise ObligationJournalError("handoff owner apply requires its matching committed root entry")
        for record in self._tail:
            payload = _decode(record.get("payload")) if record.get("event_kind") == "fork_assignment_transition" else None
            if isinstance(payload, Mapping) and payload.get("root_entry_id") == entry_id:
                if payload.get("handoff") != stage.as_dict():
                    raise ObligationJournalError("handoff event id was reused with changed owner state")
                return ledger.get(stage.obligation_id)
        try:
            updated = ledger.apply_committed_fork_handoff(stage, committed_entry_id=entry_id)
            projection = _capture(ledger)
            record = self._make_record("fork_assignment_transition", {
                "root_entry_id": entry_id, "handoff": stage.as_dict(),
            }, projection)
            self._commit_record(record, projection)
            return updated
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise ObligationJournalError("committed fork handoff owner apply failed") from error

    def append_planned_artifact_event(self, event: Mapping[str, object]) -> None:
        if self._ledger is None:
            raise ObligationJournalError("planned-artifact routing requires an attached ledger")
        if self._fault:
            raise ObligationJournalError(self._fault)
        projected = _capture(self._ledger)
        record = self._make_record("planned_artifact", event, projected)
        if self.journal is not None:
            _emit(self.journal, "planned_artifact_event", planned_artifact=copy.deepcopy(record))
        self._commit_record(record, projected)

    def rollback_ledger(self, ledger: ob.ObligationLedger) -> None:
        if ledger is not self._ledger or self._projection is None:
            raise ObligationJournalError("cannot roll back an unattached obligation ledger")
        _restore(ledger, self._projection)

    def checkpoint(self) -> dict[str, object]:
        if self._fault:
            raise ObligationJournalError(self._fault)
        if self._projection is None:
            raise ObligationJournalError("missing obligation checkpoint")
        body = {
            "schema": SCHEMA, "version": 1, "cursor": self._cursor,
            "head": self._head, "projection": copy.deepcopy(self._projection),
        }
        return {**body, "snapshot_hash": _digest(body)}

    def tail(self) -> tuple[Mapping[str, object], ...]:
        return tuple(copy.deepcopy(item) for item in self._tail)

    @classmethod
    def replay(cls, snapshot: object | None, tail: Sequence[Mapping[str, object]],
               *, journal: object | None = None
               ) -> tuple[ob.ObligationLedger, "ObligationJournalAdapter"]:
        if snapshot is None:
            raise ObligationJournalError("missing obligation snapshot; fresh empty state is forbidden")
        raw = _mapping(snapshot, "obligation snapshot")
        expected = {"schema", "version", "cursor", "head", "projection", "snapshot_hash"}
        _exact_fields(raw, expected, "obligation snapshot")
        body = {key: raw[key] for key in expected if key != "snapshot_hash"}
        if raw.get("schema") != SCHEMA or raw.get("version") != 1 \
                or raw.get("snapshot_hash") != _digest(body):
            raise ObligationJournalError("obligation snapshot schema/hash mismatch")
        cursor = raw.get("cursor")
        head = raw.get("head")
        if type(cursor) is not int or cursor < 0 or not isinstance(head, str):
            raise ObligationJournalError("invalid obligation snapshot cursor")
        projection = _mapping(raw.get("projection"), "obligation projection")
        ledger = _new_ledger(projection)
        adapter = cls(journal)
        adapter._ledger = ledger
        adapter._cursor = cursor
        adapter._head = head
        adapter._projection = copy.deepcopy(dict(projection))
        saw_uncommitted = False
        for index, raw_event in enumerate(tail):
            record, event_kind = _normalize_record(raw_event)
            if record.get("committed") is not True:
                saw_uncommitted = True
                if any(_is_committed(item) for item in tail[index + 1:]):
                    raise ObligationJournalError("uncommitted event precedes a committed event")
                break
            if saw_uncommitted:
                raise ObligationJournalError("committed event follows an uncommitted tail")
            if event_kind not in {
                    "obligation_transition", "obligation_transaction",
                    "planned_artifact", "fork_assignment_transition"}:
                raise ObligationJournalError(f"unknown committed obligation event {event_kind!r}")
            sequence = record.get("sequence")
            if type(sequence) is not int or sequence != adapter._cursor + 1:
                raise ObligationJournalError("obligation event sequence mismatch")
            if record.get("previous_hash") != adapter._head:
                raise ObligationJournalError("obligation event parent hash mismatch")
            supplied = record.get("record_hash")
            unhashed = {key: value for key, value in record.items() if key != "record_hash"}
            if supplied != _digest(unhashed):
                raise ObligationJournalError("obligation event hash mismatch")
            state_value = _mapping(record.get("state"), "obligation event state")
            next_ledger = _new_ledger(state_value)
            if event_kind == "obligation_transaction":
                _validate_transaction(record.get("payload"), ledger, next_ledger)
            adapter._projection = copy.deepcopy(dict(state_value))
            ledger = next_ledger
            adapter._ledger = ledger
            adapter._cursor = sequence
            adapter._head = str(supplied)
            adapter._tail.append(copy.deepcopy(dict(record)))
        ledger._journal_adapter = adapter
        adapter._ledger = ledger
        adapter._projection = _capture(ledger)
        return ledger, adapter

    def _make_record(self, kind: str, payload: Mapping[str, object],
                     projection: Mapping[str, object]) -> dict[str, object]:
        body: dict[str, object] = {
            "schema": SCHEMA, "sequence": self._cursor + 1,
            "previous_hash": self._head, "event_kind": kind,
            "payload": _encode(dict(payload)), "state": dict(projection),
            "host_receipt_ids": list(_receipt_ids(payload, projection)), "committed": True,
        }
        return {**body, "record_hash": _digest(body)}

    def _commit_record(self, record: dict[str, object],
                       projection: Mapping[str, object]) -> None:
        self._tail.append(copy.deepcopy(record))
        self._cursor = int(record["sequence"])
        self._head = str(record["record_hash"])
        self._projection = copy.deepcopy(dict(projection))
        audit_refs = projection.get("audit_refs")
        if self._ledger is not None and isinstance(audit_refs, list):
            self._ledger._audit_refs.update(
                ref for ref in audit_refs if isinstance(ref, str) and ref)


def _emit(journal: object, kind: str, **fields: object) -> object:
    emit = getattr(journal, "emit", None)
    if not callable(emit):
        raise ObligationJournalError("journal adapter requires an emit(kind, **fields) worker")
    return emit(kind, **fields)


def _capture(ledger: ob.ObligationLedger,
             pending_event: Mapping[str, object] | None = None,
             pending_events: Sequence[Mapping[str, object]] = ()) -> dict[str, object]:
    events = list(ledger.events)
    if pending_event is not None:
        events.append(dict(pending_event))
    events.extend(dict(event) for event in pending_events)
    projection = {
        "workspace_id": ledger.workspace_id, "session_id": ledger.session_id,
        "obligations": _encode(ledger._obligations),
        "by_registration_key": _encode(ledger._by_registration_key),
        "by_canonical_key": _encode(ledger._by_canonical_key),
        "authorization_receipts": _encode(ledger._authorization_receipts),
        "rules": _encode(ledger._rules),
        "rule_occurrences": _encode(ledger._rule_occurrences),
        "parks": _encode(ledger._parks), "acts": _encode(ledger._acts),
        "events": _encode(events), "false_close_attempts": ledger.false_close_attempts,
        "park_seq": ledger._park_seq,
    }
    retained_refs = _receipt_ids(
        {}, {key: value for key, value in projection.items() if key != "audit_refs"})
    projection["audit_refs"] = sorted(set(ledger._audit_refs).union(retained_refs))
    return projection


def _new_ledger(projection: Mapping[str, object]) -> ob.ObligationLedger:
    required = {
        "workspace_id", "session_id", "obligations", "by_registration_key",
        "by_canonical_key", "rules", "rule_occurrences", "parks", "acts",
        "events", "false_close_attempts", "park_seq",
    }
    optional = {"authorization_receipts", "audit_refs"}
    if not required <= set(projection) or not (set(projection) - required) <= optional:
        _exact_fields(projection, required, "obligation projection")
    workspace = projection.get("workspace_id")
    session = projection.get("session_id")
    if not isinstance(workspace, str) or not workspace or not isinstance(session, str) or not session:
        raise ObligationJournalError("obligation projection has invalid identity")
    ledger = ob.ObligationLedger(workspace_id=workspace, session_id=session)
    _restore(ledger, projection)
    return ledger


def _restore(ledger: ob.ObligationLedger, projection: Mapping[str, object]) -> None:
    try:
        obligations = _decode(projection["obligations"])
        by_registration = _decode(projection["by_registration_key"])
        by_canonical = _decode(projection["by_canonical_key"])
        authorization_receipts = _decode(projection.get("authorization_receipts", {}))
        rules = _decode(projection["rules"])
        rule_occurrences = _decode(projection["rule_occurrences"])
        parks = _decode(projection["parks"])
        acts = _decode(projection["acts"])
        events = _decode(projection["events"])
    except (KeyError, TypeError, ValueError) as error:
        raise ObligationJournalError("invalid obligation state projection") from error
    if not all(isinstance(item, dict) for item in
               (obligations, by_registration, by_canonical, rules,
                rule_occurrences, parks, acts, authorization_receipts)):
        raise ObligationJournalError("obligation projection maps are invalid")
    if any(not isinstance(key, str) or not isinstance(value, str)
           for key, value in authorization_receipts.items()):
        raise ObligationJournalError("obligation authorization receipt index is invalid")
    if (any(not isinstance(key, str) or not isinstance(value, ob.ObligationRule)
            or key != value.rule_id for key, value in rules.items())
            or any(not isinstance(key, str) or not isinstance(value, str)
                   or value not in obligations for key, value in rule_occurrences.items())):
        raise ObligationJournalError("obligation recurring-rule indexes are invalid")
    counts = {rule_id: 0 for rule_id in rules}
    for occurrence_key, obligation_id in rule_occurrences.items():
        occurrence = obligations[obligation_id]
        if not isinstance(occurrence, ob.Obligation) or not occurrence.rule_id:
            raise ObligationJournalError("rule occurrence index points to a non-occurrence")
        if occurrence.rule_id not in rules:
            raise ObligationJournalError("rule occurrence references a missing rule")
        if bool(occurrence.occurrence_event_id) == bool(occurrence.dependency_revision):
            raise ObligationJournalError("rule occurrence has an invalid trigger identity")
        expected_key = ob.cse1_digest({
            "rule_id": occurrence.rule_id,
            "event_id": occurrence.occurrence_event_id,
            "dependency_revision": occurrence.dependency_revision,
        })
        if occurrence_key != expected_key:
            raise ObligationJournalError("rule occurrence key does not match its evidence")
        counts[occurrence.rule_id] += 1
    if any(rule.occurrence_count != counts[rule_id] for rule_id, rule in rules.items()):
        raise ObligationJournalError("rule occurrence counts do not match their index")
    if any(isinstance(item, ob.Obligation) and item.rule_id
           and item.id not in rule_occurrences.values() for item in obligations.values()):
        raise ObligationJournalError("admitted rule occurrence is missing its identity index")
    if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
        raise ObligationJournalError("obligation projection events are invalid")
    expected_refs = _receipt_ids(
        {}, {key: value for key, value in projection.items() if key != "audit_refs"})
    audit_refs = _decode(projection.get("audit_refs", sorted(set(expected_refs))))
    if (not isinstance(audit_refs, list)
            or any(not isinstance(ref, str) or not ref for ref in audit_refs)
            or len(audit_refs) != len(set(audit_refs))
            or not set(expected_refs).issubset(audit_refs)):
        raise ObligationJournalError("obligation audit references are invalid or incomplete")
    ledger._obligations = obligations
    ledger._by_registration_key = by_registration
    ledger._by_canonical_key = by_canonical
    ledger._authorization_receipts = authorization_receipts
    ledger._audit_refs = set(audit_refs)
    ledger._rules = rules
    ledger._rule_occurrences = rule_occurrences
    ledger._parks = parks
    ledger._acts = acts
    ledger.events = events
    ledger.false_close_attempts = _nonnegative_int(projection.get("false_close_attempts"),
                                                   "false_close_attempts")
    ledger._park_seq = _nonnegative_int(projection.get("park_seq"), "park_seq")


def _encode(value: object) -> object:
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ObligationJournalError("non-finite number in obligation state")
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "$type": type(value).__name__,
            "fields": {field.name: _encode(getattr(value, field.name)) for field in fields(value)},
        }
    if isinstance(value, Mapping):
        pairs = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise ObligationJournalError("obligation map keys must be strings")
            pairs.append([key, _encode(item)])
        return {"$map": pairs}
    if isinstance(value, tuple):
        return {"$tuple": [_encode(item) for item in value]}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, (set, frozenset)):
        packed = [_encode(item) for item in value]
        return {"$set": sorted(packed, key=_canonical)}
    raise ObligationJournalError(f"unsupported obligation state value {type(value).__name__}")


def _decode(value: object) -> object:
    if value is None or type(value) in {str, bool, int, float}:
        return value
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, Mapping):
        raise ObligationJournalError("obligation state contains a non-data value")
    if set(value) == {"$tuple"}:
        return tuple(_decode(item) for item in _sequence(value["$tuple"], "tuple"))
    if set(value) == {"$set"}:
        return frozenset(_decode(item) for item in _sequence(value["$set"], "set"))
    if set(value) == {"$map"}:
        pairs = _sequence(value["$map"], "map")
        result: dict[str, object] = {}
        for pair_value in pairs:
            pair = _sequence(pair_value, "map pair")
            if len(pair) != 2 or not isinstance(pair[0], str) or pair[0] in result:
                raise ObligationJournalError("invalid obligation map entry")
            result[pair[0]] = _decode(pair[1])
        return result
    if set(value) == {"$type", "fields"}:
        name, raw_fields = value["$type"], value["fields"]
        if not isinstance(name, str):
            raise ObligationJournalError("invalid obligation record type")
        cls = _DATACLASS_TYPES.get(name)
        if cls is None:
            raise ObligationJournalError(f"unknown obligation record type {name!r}")
        fields_raw = _mapping(raw_fields, "typed fields")
        class_fields = fields(cls)
        expected = {field.name for field in class_fields}
        if name == "Obligation" and "fork_assignment" in expected and "fork_assignment" not in fields_raw:
            # Legacy obligations had no fork assignment and are explicitly unassigned.
            fields_raw = {**dict(fields_raw), "fork_assignment": None}
        _exact_fields(fields_raw, expected, f"{name} fields")
        decoded = {key: _decode(item) for key, item in fields_raw.items()}
        instance = cls(**{field.name: decoded[field.name]
                          for field in class_fields if field.init})
        if any(getattr(instance, field.name) != decoded[field.name]
               for field in class_fields if not field.init):
            raise ObligationJournalError(f"invalid derived fields for {name}")
        return instance
    raise ObligationJournalError("unknown obligation state encoding")


_DATACLASS_TYPES = {
    name: value for name, value in vars(ob).items()
    if isinstance(value, type) and is_dataclass(value) and value.__module__ == ob.__name__
}


def _fork_handoff_stage(value: Mapping[str, object]) -> ob.ForkHandoffStage:
    expected = {"schema", "obligation_id", "fork_id", "expected_revision",
                "assignment_revision", "event_id", "previous_digest"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ObligationJournalError("invalid staged fork handoff fields")
    try:
        return ob.ForkHandoffStage(**dict(value))
    except (TypeError, ValueError) as error:
        raise ObligationJournalError("invalid staged fork handoff") from error




def _normalize_record(value: object) -> tuple[Mapping[str, object], str]:
    raw = _mapping(value, "obligation journal entry")
    if "event_kind" in raw:
        if raw.get("schema") != SCHEMA:
            raise ObligationJournalError("unknown obligation event schema")
        return raw, str(raw["event_kind"])
    kind = raw.get("kind")
    field = "obligation" if kind in {"obligation_transition", "obligation_transaction"} else (
        "planned_artifact" if kind == "planned_artifact_event" else "")
    nested = raw.get(field) if field else None
    if not isinstance(nested, Mapping):
        raise ObligationJournalError("journal envelope has no typed obligation payload")
    return nested, str(nested.get("event_kind", ""))


def _validate_transaction(payload: object, previous: ob.ObligationLedger,
                           candidate: ob.ObligationLedger) -> None:
    decoded = _mapping(_decode(payload), "obligation transaction payload")
    batch = _sequence(decoded.get("events"), "obligation transaction events")
    if not batch:
        raise ObligationJournalError("obligation transaction has no events")
    actual = candidate.events[len(previous.events):]
    expected = [_mapping(event, "obligation transaction event") for event in batch]
    if _canonical(actual) != _canonical(expected):
        raise ObligationJournalError("transaction payload does not match its committed projection")
    for offset, event in enumerate(expected):
        if event.get("seq") != len(previous.events) + offset:
            raise ObligationJournalError("obligation transaction event sequence is not contiguous")

    required_by_parent = {
        str(event.get("obligation_id")): event
        for event in expected if event.get("event") == "FOLLOW_UP_REQUIRED"
    }
    positions = {id(event): index for index, event in enumerate(expected)}
    registration_positions = {
        str(event.get("obligation_id")): index
        for index, event in enumerate(expected)
        if event.get("event") == "OBLIGATION_REGISTERED"
    }
    discharge_positions = {
        str(event.get("obligation_id")): index
        for index, event in enumerate(expected)
        if event.get("event") == "OBLIGATION_DISCHARGED"
    }
    followups = [event for event in expected if event.get("event") == "FOLLOW_UP_REGISTERED"]
    for followup in followups:
        child_id = str(followup.get("obligation_id"))
        parent_id = str(followup.get("parent_id"))
        registered_at = registration_positions.get(child_id)
        discharged_at = discharge_positions.get(parent_id)
        if registered_at is None or registered_at >= positions[id(followup)]:
            raise ObligationJournalError("follow-up receipt precedes durable child registration")
        if discharged_at is None or registered_at >= discharged_at:
            raise ObligationJournalError("parent settlement precedes durable child registration")
    for parent_id, required in required_by_parent.items():
        requirement_count = len(_sequence(required.get("requirements"), "follow-up requirements"))
        count = sum(1 for event in followups if str(event.get("parent_id")) == parent_id)
        if count < requirement_count:
            raise ObligationJournalError("settled parent is missing a required child registration")
        if parent_id not in discharge_positions:
            raise ObligationJournalError("follow-up requirements have no parent settlement")


def _is_committed(value: object) -> bool:
    try:
        record, _kind = _normalize_record(value)
    except ObligationJournalError:
        return False
    return record.get("committed") is True


def _receipt_ids(payload: Mapping[str, object],
                 projection: Mapping[str, object]) -> tuple[str, ...]:
    found: list[str] = []

    def walk(value: object) -> None:
        if is_dataclass(value) and not isinstance(value, type):
            walk({field.name: getattr(value, field.name) for field in fields(value)})
        elif isinstance(value, Mapping):
            for key, item in value.items():
                if key in {"receipt_id", "receipt_ids", "event_id", "origin_event_id",
                           "act_id", "authorization_receipt_id", "procedure_receipt_id"}:
                    candidates = item if isinstance(item, (list, tuple)) else (item,)
                    found.extend(candidate.strip() for candidate in candidates
                                 if isinstance(candidate, str) and candidate.strip())
                elif key == "authorization_receipts" and isinstance(item, Mapping):
                    found.extend(receipt_id.strip() for receipt_id in item
                                 if isinstance(receipt_id, str) and receipt_id.strip())
                    walk(item)
                elif key == "evidence" and isinstance(item, (list, tuple)):
                    for evidence in item:
                        reference = (getattr(evidence, "ref", None)
                                     if is_dataclass(evidence) else
                                     evidence.get("ref") if isinstance(evidence, Mapping) else None)
                        if isinstance(reference, str) and reference.strip():
                            found.append(reference.strip())
                else:
                    walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(payload)
    walk({key: _decode(value) for key, value in projection.items()})
    return tuple(dict.fromkeys(found))


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ObligationJournalError(f"{label} must be a string-keyed object")
    return value


def _exact_fields(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        missing, extra = sorted(expected - set(value)), sorted(set(value) - expected)
        raise ObligationJournalError(f"{label} fields mismatch: missing={missing}, extra={extra}")


def _sequence(value: object, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ObligationJournalError(f"{label} must be a sequence")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ObligationJournalError(f"{label} must be a non-negative integer")
    return value


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()
