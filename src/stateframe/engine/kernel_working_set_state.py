"""Default-off durable exposure debt and certified source residency.

Captures live in the append-only journal/blob store; this index holds only live
exposure debt and residency. Only a certified submission followed by an adopted
response changes residency; delivery counters and substrings grant no authority.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Literal

if __package__:
    from .kernel_read_receipts import (CaptureLedger, DecodedSpan, ReadCapture,
                                       ReceiptIdentity, enabled)
else:
    from kernel_read_receipts import (CaptureLedger, DecodedSpan, ReadCapture,
                                      ReceiptIdentity, enabled)

# 256 live identities, with 16 (one maximum native batch) reserved between
# accepted boundaries. Retirement still requires a certified, accepted O carrier.
MAX_ACTIVE_OBLIGATIONS = 256
NATIVE_BATCH_RESERVE = 16
MAX_RETAINED_RESIDENTS = MAX_ACTIVE_OBLIGATIONS - NATIVE_BATCH_RESERVE
SCHEMA = "stateframe-working-set/1"
State = Literal["DUE", "RESIDENT", "DEGRADED", "RETIRING", "DISCHARGED", "SUPERSEDED"]
Mode = Literal["F", "D", "O"]
EVENT_KINDS = frozenset({"working_set_capture", "working_set_submission",
                        "working_set_response", "working_set_blocked"})
# Item-1 audit rows. They are SAFE_TAIL for the resume gate, but the reducer still
# reads them so a read interrupted by a restart is never silently dropped.
READ_AUDIT_KINDS = frozenset({"read_attempt", "read_result"})
CHECKPOINT_COUNTERS = ("event_ordinal", "logical_ordinal", "capture_ordinal", "admission_ordinal",
                       "use_ordinal", "overflow_count", "read_sequence")


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def identity_from_document(value: Mapping[str, object]) -> ReceiptIdentity:
    fields = dict(value)
    if fields.get("lossy_span") is not None:
        fields["lossy_span"] = DecodedSpan(**fields["lossy_span"])
    return ReceiptIdentity(**fields)


def capture_document(capture: ReadCapture) -> dict[str, object]:
    return {**asdict(capture), "recovery_arguments": capture.recovery_arguments.decode("utf-8")}


def capture_from_document(row: Mapping[str, object]) -> ReadCapture:
    fields = dict(row)
    fields["identity"] = identity_from_document(fields["identity"])
    fields["recovery_arguments"] = fields["recovery_arguments"].encode("utf-8")
    for name in ("line_digests", "line_numbers"):
        fields[name] = tuple(fields[name])
    return ReadCapture(**fields)


def rebuild_ledger(runtime: object, records: Iterable[Mapping[str, object]]) -> CaptureLedger:
    """Rebuild the process-memory capture ledger from the journal's capture rows.

    The ledger is never checkpointed: the journal is its only durable form, so a
    restarted worker reads every working_set_capture row, dropping duplicates by
    event ordinal.
    """
    ledger = CaptureLedger()
    seen: set[int] = set()
    for record in records:
        ordinal = record.get("working_set_event_ordinal")
        if record.get("kind") != "working_set_capture" or ordinal in seen:
            continue
        seen.add(ordinal)
        ledger.add(capture_from_document(record["capture"]))
    runtime.read_capture_ledger = ledger
    return ledger


@dataclass
class ReadObligation:
    obligation_id: str
    capture_ids: tuple[str, ...]
    identity: ReceiptIdentity
    due_ordinal: int
    state: State
    admission_ordinal: int
    last_use_ordinal: int
    recovery_pin: bool = False
    superseded_by: str | None = None
    previous_mode: Mode | None = None

    def validate(self) -> None:
        if (not self.obligation_id or not self.capture_ids
                or len(set(self.capture_ids)) != len(self.capture_ids)
                or not all(isinstance(c, str) and c for c in self.capture_ids)
                or self.state not in {"DUE", "RESIDENT", "DEGRADED", "RETIRING"}
                or any(type(n) is not int or n < 1 for n in (
                    self.due_ordinal, self.admission_ordinal, self.last_use_ordinal))
                or self.previous_mode not in {None, "F", "D", "O"}
                or type(self.recovery_pin) is not bool
                or (self.recovery_pin and self.state != "DUE")
                or (self.superseded_by is not None and not isinstance(self.superseded_by, str))):
            raise ValueError("invalid active read obligation")


class WorkingSetState:
    def __init__(self) -> None:
        self._active: dict[str, ReadObligation] = {}
        self.event_ordinal = 0
        self.logical_ordinal = 0
        self.capture_ordinal = 0
        self.admission_ordinal = 0
        self.use_ordinal = 0
        self.capture_index_root: str | None = None
        self.pending: dict[str, dict[str, object]] = {}
        self.accepted_exposure_id: str | None = None
        self.overflow_count = 0
        self.suspension = None
        # Journal position of the last reduced read audit row; makes replay of
        # read_attempt/read_result rows idempotent without a seen-set.
        self.read_sequence = 0
        self.open_read_attempt: str | None = None
        self.interrupted_reads: tuple[str, ...] = ()

    @property
    def shadow_complete(self) -> bool:
        """True while the active index is within the live identity limit.

        overflow_count is the live overshoot past MAX_ACTIVE_OBLIGATIONS. It
        never hides a capture: every capture is indexed, and the overshoot is
        cleared by the accepted O carriers that reserve retirement queues.
        """
        return self.overflow_count == 0

    @property
    def next_due_ordinal(self) -> int:
        return self.logical_ordinal + 1

    def active_obligations(self) -> tuple[ReadObligation, ...]:
        return tuple(sorted(self._active.values(), key=lambda o: o.admission_ordinal))

    def checkpoint(self) -> dict[str, object]:
        return {"schema": SCHEMA,
                "active": [json.loads(json.dumps(asdict(o))) for o in self.active_obligations()],
                **{key: copy.deepcopy(getattr(self, key)) for key in (
                    *CHECKPOINT_COUNTERS, "capture_index_root", "pending",
                    "accepted_exposure_id")},
                "open_read_attempt": self.open_read_attempt,
                "interrupted_reads": list(self.interrupted_reads),
                "suspension": self.suspension.document() if self.suspension else None}

    @classmethod
    def from_checkpoint(cls, document: Mapping[str, object]) -> WorkingSetState:
        state = cls()
        if not isinstance(document, Mapping) or document.get("schema") != SCHEMA:
            raise ValueError("invalid working-set checkpoint schema")
        if set(document) != set(state.checkpoint()):
            raise ValueError("invalid working-set checkpoint fields")
        for name in CHECKPOINT_COUNTERS:
            value = document[name]
            if type(value) is not int or value < 0:
                raise ValueError("invalid working-set counter")
            setattr(state, name, value)
        active = document["active"]
        # Refuse legacy incomplete indexes: their dropped captures can only be
        # recovered by journal replay, never by clearing the overflow counter.
        if (not isinstance(active, list)
                or state.overflow_count != max(0, len(active) - MAX_ACTIVE_OBLIGATIONS)):
            raise ValueError("invalid active working-set index")
        admissions = set()
        identities = set()
        for row in active:
            fields = dict(row)
            fields.setdefault("previous_mode", {"RESIDENT": "F", "DEGRADED": "D",
                                                 "RETIRING": "O"}.get(fields.get("state")))
            fields["identity"] = identity_from_document(fields["identity"])
            fields["capture_ids"] = tuple(fields["capture_ids"])
            entry = ReadObligation(**fields)
            entry.validate()
            if (entry.obligation_id in state._active or entry.identity in identities
                    or entry.admission_ordinal in admissions
                    or entry.admission_ordinal > state.admission_ordinal
                    or entry.last_use_ordinal > state.use_ordinal
                    or entry.due_ordinal > state.next_due_ordinal):
                raise ValueError("inconsistent active working-set index")
            identities.add(entry.identity)
            admissions.add(entry.admission_ordinal)
            state._active[entry.obligation_id] = entry
        for name in ("capture_index_root", "accepted_exposure_id", "open_read_attempt"):
            value = document[name]
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError("invalid working-set journal reference")
            setattr(state, name, value)
        interrupted = document["interrupted_reads"]
        if (not isinstance(interrupted, list) or len(set(interrupted)) != len(interrupted)
                or not all(isinstance(a, str) and a for a in interrupted)
                or state.open_read_attempt in interrupted):
            raise ValueError("invalid interrupted read attempts")
        state.interrupted_reads = tuple(interrupted)
        pending = document["pending"]
        if not isinstance(pending, dict) or len(pending) > 1:
            raise ValueError("invalid working-set pending submission")
        for request_id, row in pending.items():
            if (not isinstance(request_id, str) or not request_id
                    or set(row) != {"body_sha256", "ordinal", "assignments"}
                    or type(row["ordinal"]) is not int or row["ordinal"] != state.logical_ordinal
                    or not isinstance(row["body_sha256"], str) or len(row["body_sha256"]) != 64
                    or not isinstance(row["assignments"], dict)):
                raise ValueError("invalid working-set submission certificate")
            for oid, assignment in row["assignments"].items():
                if (oid not in state._active or assignment.get("mode") not in {"F", "D", "O"}
                        or not isinstance(assignment.get("capture_ids"), (tuple, list))
                        or not set(assignment["capture_ids"]) <= set(state._active[oid].capture_ids)):
                    raise ValueError("invalid working-set assignment")
        state.pending = copy.deepcopy(pending)
        if document["suspension"] is not None:
            if __package__:
                from .kernel_working_set_stop import Suspension
            else:
                from kernel_working_set_stop import Suspension
            state.suspension = Suspension.from_document(document["suspension"])
        return state

    def apply(self, event: Mapping[str, object]) -> None:
        """Reduce ordered typed events. A checkpoint watermark makes replay idempotent.

        No growing seen-event set: the authenticated journal supplies ordering.
        A gap is refused rather than silently accepting incomplete reconstruction.
        """
        if event.get("kind") not in EVENT_KINDS:
            return
        ordinal = event.get("working_set_event_ordinal")
        if type(ordinal) is not int or ordinal < 1:
            raise ValueError("missing working-set event ordinal")
        if ordinal <= self.event_ordinal:
            return
        if ordinal != self.event_ordinal + 1:
            raise ValueError("working-set replay gap")
        kind = event["kind"]
        if kind == "working_set_capture":
            self._capture(event)
        elif kind == "working_set_submission":
            self._submission(event)
        elif kind == "working_set_response":
            self._response(event)
        elif kind == "working_set_blocked":
            if __package__:
                from .kernel_working_set_stop import Suspension
            else:
                from kernel_working_set_stop import Suspension
            self.suspension = Suspension.from_document(event["suspension"])
        self.event_ordinal = ordinal

    def apply_record(self, record: Mapping[str, object]) -> None:
        """Reduce one journal record: typed working-set events or read audit rows."""
        kind = record.get("kind")
        if kind in EVENT_KINDS:
            self.apply(record)
        elif kind in READ_AUDIT_KINDS:
            self._read_audit(record)

    def _read_audit(self, record: Mapping[str, object]) -> None:
        sequence = record.get("sequence")
        if type(sequence) is not int or sequence < 1:
            raise ValueError("read audit row without journal sequence")
        if sequence <= self.read_sequence:
            return
        attempt_id = record.get("attempt_id")
        if not isinstance(attempt_id, str) or not attempt_id:
            raise ValueError("read audit row without attempt id")
        if record["kind"] == "read_attempt":
            if self.open_read_attempt is not None:
                self._interrupt_open_read()
            self.open_read_attempt = attempt_id
        elif attempt_id == self.open_read_attempt:
            self.open_read_attempt = None
        self.read_sequence = sequence

    def _interrupt_open_read(self) -> None:
        if self.open_read_attempt is not None and self.open_read_attempt not in self.interrupted_reads:
            self.interrupted_reads += (self.open_read_attempt,)
        self.open_read_attempt = None

    def close_interrupted_reads(self) -> tuple[str, ...]:
        """A trailing read_attempt with no read_result reduces to INTERRUPTED.

        Called once the journal has been read to its end: no later row can
        complete the attempt, so the capture it may have produced is never
        advertised as delivered.
        """
        self._interrupt_open_read()
        return self.interrupted_reads

    def _capture(self, event: Mapping[str, object]) -> None:
        row = event["capture"]
        identity = identity_from_document(row["identity"])
        capture_id = row["capture_id"]
        self.capture_ordinal += 1
        self.capture_index_root = event["capture_index_root"]
        self.use_ordinal += 1
        existing = next((o for o in self._active.values() if o.identity == identity), None)
        if existing is not None:
            if capture_id not in existing.capture_ids:
                existing.capture_ids += (capture_id,)
                existing.state = "DUE"
                existing.due_ordinal = self.next_due_ordinal
                existing.last_use_ordinal = self.use_ordinal
                existing.recovery_pin |= event["recovery_pin"]
            self._reserve_retirements()
            return
        self.admission_ordinal += 1
        oid = f"read-obligation-{self.admission_ordinal}"
        # A validated successful capture proves the raw revision changed. Old
        # bytes of that path are obsolete whether or not they were exposed: an
        # unexposed one is never owed at the wrong revision (the capture row
        # stays in the journal), an exposed one is never carried again.
        for old in tuple(self._active.values()):
            if old.identity.path == identity.path and old.identity.revision != identity.revision:
                old.superseded_by = oid
                old.state = "SUPERSEDED"
                del self._active[old.obligation_id]
        # The capture is always indexed: dropping it would advertise bytes the
        # model saw as never delivered. Past the limit the index overshoots
        # (more captures than the reserve between two accepted boundaries) and
        # reserve retirement queues enough O carriers to clear it.
        self._active[oid] = ReadObligation(oid, (capture_id,), identity,
            self.next_due_ordinal, "DUE", self.admission_ordinal, self.use_ordinal,
            event["recovery_pin"])
        self._reserve_retirements()

    def _reserve_retirements(self) -> None:
        """Queue oldest residency for O without discharging any exposure debt.

        Captures may consume the native-batch reserve, but the next allocator
        assignment must replenish it. RETIRING stays in the active index through
        preparation, failed sends and discarded responses; only an accepted O
        assignment removes it. A reread makes that lifetime DUE again and queues
        another eligible resident instead.
        """
        self.overflow_count = max(0, len(self._active) - MAX_ACTIVE_OBLIGATIONS)
        needed = (len(self._active) - MAX_RETAINED_RESIDENTS
                  - sum(o.state == "RETIRING" for o in self._active.values()))
        if needed <= 0:
            return
        residents = sorted((o for o in self._active.values()
                            if o.state in {"RESIDENT", "DEGRADED"} and not o.recovery_pin),
                           key=lambda o: (o.last_use_ordinal, o.admission_ordinal, o.obligation_id))
        for entry in residents[:needed]:
            entry.state = "RETIRING"

    def _submission(self, event: Mapping[str, object]) -> None:
        request_id = event["request_id"]
        if request_id in self.pending:
            if self.pending[request_id]["body_sha256"] != event["body_sha256"]:
                raise ValueError("working-set submitted wire changed")
            return
        assignments = {}
        for oid, mode in event["modes"].items():
            entry = self._active[oid]
            if mode not in {"F", "D", "O"}:
                raise ValueError("invalid working-set mode")
            if (entry.state == "DUE" or entry.recovery_pin) and mode != "F":
                raise ValueError("fresh debt and recovery pins are F-only")
            if entry.state == "RETIRING" and mode == "D":
                # Retirement is O; a resident without a valid retirement
                # representation (protected, or no recovery route) stays F and
                # the next accepted response queues another resident.
                raise ValueError("reserve retirement is an O or kept F assignment")
            ranks = {"F": 3, "D": 2, "O": 1}
            if ((entry.state == "DEGRADED" and mode == "F")
                    or (entry.state != "DUE" and entry.previous_mode is not None
                        and ranks[mode] > ranks[entry.previous_mode])):
                raise ValueError("promotion requires a reread or recovery")
            assignments[oid] = {"mode": mode, "capture_ids": list(entry.capture_ids)}
        self.logical_ordinal += 1
        # A new primary call closes unmatched failed/discarded predecessors.
        self.pending = {request_id: {"body_sha256": event["body_sha256"],
            "ordinal": self.logical_ordinal, "assignments": assignments}}

    def _response(self, event: Mapping[str, object]) -> None:
        request_id = event["request_id"]
        submitted = self.pending.get(request_id)
        if submitted is None or submitted["body_sha256"] != event["body_sha256"]:
            return
        disposition = event["disposition"]
        if disposition not in {"accepted", "failed", "discarded"}:
            raise ValueError("invalid response disposition")
        if disposition == "accepted":
            self.accepted_exposure_id = event["response_id"]
            kept = []
            for oid, assignment in submitted["assignments"].items():
                entry = self._active.get(oid)
                if entry is None or tuple(assignment["capture_ids"]) != entry.capture_ids:
                    continue  # a newer read cannot be paid by an older wire
                mode = assignment["mode"]
                if entry.state == "RETIRING" and mode == "F":
                    kept.append(entry)
                entry.previous_mode = mode
                self.use_ordinal += 1
                entry.last_use_ordinal = self.use_ordinal
                entry.recovery_pin = False
                entry.state = "RESIDENT" if mode == "F" else "DEGRADED" if mode == "D" else "RETIRING"
                if entry.superseded_by is not None or mode == "O":
                    entry.state = "SUPERSEDED" if entry.superseded_by is not None else "DISCHARGED"
                    del self._active[oid]
            # A resident kept FULL instead of retiring had no valid retirement
            # representation on this wire; make it the most recently used so
            # the reserve queues the next oldest resident rather than it again.
            for entry in kept:
                self.use_ordinal += 1
                entry.last_use_ordinal = self.use_ordinal
            self._reserve_retirements()
        del self.pending[request_id]


def state_for(runtime: object) -> WorkingSetState:
    state = getattr(runtime, "working_set_state", None)
    if state is None:
        state = WorkingSetState()
        if enabled():
            runtime.working_set_state = state
    return state


def emit_event(runtime: object, kind: str, **fields: object) -> None:
    if not enabled():
        return
    state = state_for(runtime)
    event = {"kind": kind, "working_set_event_ordinal": state.event_ordinal + 1, **fields}
    # Validate before I/O; publish only after the established journal fsync.
    preview = copy.deepcopy(state)
    preview.apply(event)
    runtime.journal.emit(kind, **{key: value for key, value in event.items() if key != "kind"})
    state.apply(event)


def record_capture(runtime: object, capture: ReadCapture, *, recovery_pin: bool = False) -> None:
    if enabled():
        row = capture_document(capture)
        emit_event(runtime, "working_set_capture", capture=row, recovery_pin=recovery_pin,
                   capture_index_root=_digest(row))


def record_submission(runtime: object, *, request_id: str, body_sha256: str,
                      modes: Mapping[str, Mode]) -> None:
    emit_event(runtime, "working_set_submission", request_id=request_id,
               body_sha256=body_sha256, modes=dict(modes))


def record_response(runtime: object, *, request_id: str, response_id: str,
                    body_sha256: str, disposition: str) -> None:
    emit_event(runtime, "working_set_response", request_id=request_id,
               response_id=response_id, body_sha256=body_sha256, disposition=disposition)


def observe_submission(runtime: object, capture: object, wire: bytes) -> None:
    if not enabled():
        return
    if __package__:
        from .kernel_request_kind import is_auxiliary
    else:
        from kernel_request_kind import is_auxiliary
    if not is_auxiliary(capture):
        certificate = getattr(capture, "_working_set_certificate", None)
        modes = {}
        if certificate is not None:
            from kernel_working_set_proof import verify_certificate
            if certificate.snapshot.request_id != capture.request_id:
                raise ValueError("certificate request identity differs")
            verify_certificate(wire, certificate)
            names = {"FULL": "F", "DEGRADED": "D", "DEFERRED": "O"}
            selected = dict(certificate.assignment)
            active = {o.obligation_id for o in state_for(runtime).active_obligations()}
            for slot in certificate.snapshot.sources:
                for oid in slot.obligation_ids:
                    if oid in active:
                        modes[oid] = names[selected[slot.key]]
            if set(modes) != active:
                raise ValueError("certificate does not cover active obligations")
        record_submission(runtime, request_id=capture.request_id,
                          body_sha256=hashlib.sha256(wire).hexdigest(), modes=modes)


def observe_response(runtime: object, capture: object, *, disposition: str,
                     response_id: str | None = None) -> None:
    if not enabled():
        return
    state = state_for(runtime)
    submitted = state.pending.get(capture.request_id)
    if response_id is None:
        response_id = getattr(capture, "selected_response_event_id", None)
    if submitted is not None and response_id:
        record_response(runtime, request_id=capture.request_id, response_id=response_id,
                        body_sha256=submitted["body_sha256"], disposition=disposition)


def reconstruct(events: Iterable[Mapping[str, object]]) -> WorkingSetState:
    state = WorkingSetState()
    for event in events:
        state.apply_record(event)
    state.close_interrupted_reads()
    return state


def replay_tail(state: WorkingSetState, records: Iterable[Mapping[str, object]],
                after_sequence: int) -> WorkingSetState:
    """Fold the journal rows written after a checkpoint into a restored state.

    Due ordinals come from the journal's own request boundaries
    (working_set_submission rows), never from the provider request counter that
    restore_provider_request_count and the bookmark replay overwrite.
    """
    for record in records:
        sequence = record.get("sequence")
        if type(sequence) is int and sequence > after_sequence:
            state.apply_record(record)
    state.close_interrupted_reads()
    return state


def restore_tail(runtime: object, records: Iterable[Mapping[str, object]],
                 checkpoint_sequence: int) -> None:
    """Continuation restart: rebuild captures/obligations the checkpoint predates.

    The checkpointed index is the floor; every later capture, submission,
    response and blocked row is replayed so a read issued after the last
    checkpoint still owes exposure on the next wire. The stop control is then
    cleared: the restarted worker retries at the unsent boundary.
    """
    if not enabled():
        return
    records = list(records)
    rebuild_ledger(runtime, records)
    state = replay_tail(state_for(runtime), records, checkpoint_sequence)
    retry_unsent_boundary(state)


def retry_unsent_boundary(state: WorkingSetState) -> None:
    """Only clear the stop control; no submission, exposure, recency or debt change."""
    state.suspension = None
