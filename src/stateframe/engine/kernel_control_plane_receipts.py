"""Language-neutral, host-owned control-plane evidence for blindspot closure.

This module is an adapter at the StateFrame boundary, not a second completion
authority.  It stores immutable host observations and joins them only when the
full run/candidate/revision identity matches.  Presentation is opt-in; absent
or incomplete evidence remains ``UNKNOWN``/``UNAVAILABLE``.

The records deliberately use strings and JSON-compatible values so Go, Rust,
TypeScript, Java, PHP, Vue, and Python hosts can emit the same contract without
importing Python implementation details.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final


SCHEMA: Final = "stateframe-control-plane/1"
MODE_ENV: Final = "STATEFRAME_CONTROL_PLANE"
UNKNOWN: Final = "UNKNOWN"
UNAVAILABLE: Final = "UNAVAILABLE"

EFFECT_STATES: Final = frozenset({
    "APPLIED", "REJECTED", "NO_OP", "SUPERSEDED", "STALE", "UNRESOLVED",
    "OBSERVED", "DELIVERED", "NOT_EXECUTED",
})
EVIDENCE_STATES: Final = frozenset({
    "FRESH_EXECUTED", "HOST_AUTO", "CACHED", "STAGED", "DESCRIBED",
    "UNEXECUTED", "UNKNOWN",
})
FAILURE_CLASSES: Final = frozenset({
    "BEHAVIORAL", "SETUP", "REFUSAL", "RESOURCE", "TIMEOUT", "NEVER_STARTED",
    "UNKNOWN",
})
ATTEMPT_CLASSES: Final = frozenset({
    "LIVE_MODEL", "ORACLE", "FILLER", "ZERO_REQUEST", "REFUSED", "INTERRUPTED",
    "SALVAGE",
})
WORKFLOW_STATES: Final = (
    "PUBLIC_GREEN", "OBLIGATION_COMPLETE", "READY_TO_SUBMIT", "ACCEPTED",
    "CERTIFIED", "EXTERNALLY_CORRECT",
)


class ControlPlaneError(ValueError):
    """Base error for malformed or conflicting host evidence."""


class IdentityError(ControlPlaneError):
    """A record is missing an identity field or uses a batch-only identity."""


class IdentityMismatch(ControlPlaneError):
    """Records from different runs, releases, candidates, or revisions were joined."""


class ReceiptConflict(ControlPlaneError):
    """A supposedly immutable receipt was replaced under the same receipt id."""


class SettlementConflict(ControlPlaneError):
    """An effect was given two different terminal outcomes."""


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def digest(value: object) -> str:
    """Return a stable, namespaced digest for JSON-compatible evidence."""
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def bytes_digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _text(value: object, name: str, *, allow_unknown: bool = True) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ControlPlaneError(f"{name} must be a non-empty string")
    result = value.strip()
    if not allow_unknown and result == UNKNOWN:
        raise ControlPlaneError(f"{name} cannot be UNKNOWN")
    return result


@dataclass(frozen=True)
class RunIdentity:
    """Complete join key; a batch number is metadata, never identity."""

    run_uuid: str
    task_id: str
    task_version: str
    release_digest: str
    image_digest: str
    candidate_digest: str
    export_digest: str
    archive_id: str
    revision: str
    environment_digest: str = UNKNOWN
    tool_policy_digest: str = UNKNOWN
    batch_number: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "run_uuid", "task_id", "task_version", "release_digest", "image_digest",
            "candidate_digest", "export_digest", "archive_id", "revision",
            "environment_digest", "tool_policy_digest",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.batch_number is not None:
            object.__setattr__(self, "batch_number", _text(self.batch_number, "batch_number"))
            if self.batch_number == self.run_uuid:
                raise IdentityError("batch_number cannot stand in for run_uuid")

    @property
    def complete(self) -> bool:
        return all(
            getattr(self, name) not in {UNKNOWN, UNAVAILABLE}
            for name in (
                "run_uuid", "task_id", "task_version", "release_digest", "image_digest",
                "candidate_digest", "export_digest", "archive_id", "revision",
                "environment_digest", "tool_policy_digest",
            )
        )

    def as_dict(self) -> dict[str, str | None]:
        return {
            "run_uuid": self.run_uuid,
            "task_id": self.task_id,
            "task_version": self.task_version,
            "release_digest": self.release_digest,
            "image_digest": self.image_digest,
            "candidate_digest": self.candidate_digest,
            "export_digest": self.export_digest,
            "archive_id": self.archive_id,
            "revision": self.revision,
            "environment_digest": self.environment_digest,
            "tool_policy_digest": self.tool_policy_digest,
            "batch_number": self.batch_number,
        }

    @property
    def key(self) -> tuple[str, ...]:
        return tuple(str(self.as_dict()[name]) for name in self.as_dict())

    @property
    def identity_digest(self) -> str:
        return digest(self.as_dict())

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "RunIdentity":
        if not isinstance(value, Mapping):
            raise IdentityError("identity must be a mapping")
        required = (
            "run_uuid", "task_id", "task_version", "release_digest", "image_digest",
            "candidate_digest", "export_digest", "archive_id", "revision",
        )
        missing = [name for name in required if not str(value.get(name, "")).strip()]
        if missing:
            raise IdentityError("identity missing " + ", ".join(missing))
        return cls(
            run_uuid=str(value["run_uuid"]),
            task_id=str(value["task_id"]),
            task_version=str(value["task_version"]),
            release_digest=str(value["release_digest"]),
            image_digest=str(value["image_digest"]),
            candidate_digest=str(value["candidate_digest"]),
            export_digest=str(value["export_digest"]),
            archive_id=str(value["archive_id"]),
            revision=str(value["revision"]),
            environment_digest=str(value.get("environment_digest", UNKNOWN)),
            tool_policy_digest=str(value.get("tool_policy_digest", UNKNOWN)),
            batch_number=(None if value.get("batch_number") is None
                          else str(value["batch_number"])),
        )


def require_same_identity(records: Iterable[Mapping[str, object] | "HostReceipt"],
                          expected: RunIdentity | None = None) -> RunIdentity:
    """Validate every record against the full identity, not only batch number."""
    identities: list[RunIdentity] = []
    for record in records:
        identity = record.identity if isinstance(record, HostReceipt) else RunIdentity.from_mapping(
            record.get("identity", record) if isinstance(record, Mapping) else {}
        )
        identities.append(identity)
    if expected is not None:
        identities.insert(0, expected)
    if not identities:
        raise IdentityError("at least one identity is required")
    first = identities[0]
    if any(identity.key != first.key for identity in identities[1:]):
        raise IdentityMismatch("records do not share the full run identity")
    if not first.complete:
        raise IdentityError("incomplete run identity cannot support a promoted claim")
    return first


@dataclass(frozen=True)
class HostReceipt:
    """Immutable host observation with effect and evidence truth separated."""

    receipt_id: str
    kind: str
    sequence: int
    identity: RunIdentity
    effect_state: str
    evidence_state: str
    actual_effect: bool
    payload_json: str
    payload_digest: str
    source: str = "host"

    def __post_init__(self) -> None:
        if not self.receipt_id.strip() or not self.kind.strip():
            raise ControlPlaneError("receipt_id and kind are required")
        if type(self.sequence) is not int or self.sequence < 0:
            raise ControlPlaneError("receipt sequence must be a non-negative integer")
        if self.effect_state not in EFFECT_STATES:
            raise ControlPlaneError(f"unknown effect state: {self.effect_state}")
        if self.evidence_state not in EVIDENCE_STATES:
            raise ControlPlaneError(f"unknown evidence state: {self.evidence_state}")
        if not isinstance(self.actual_effect, bool):
            raise ControlPlaneError("actual_effect must be bool")
        if digest(json.loads(self.payload_json)) != self.payload_digest:
            raise ReceiptConflict("receipt payload digest does not match payload")

    @classmethod
    def create(cls, *, kind: str, sequence: int, identity: RunIdentity,
               payload: Mapping[str, object] | Sequence[object] | str | int | bool | None,
               effect_state: str = "OBSERVED", evidence_state: str = "FRESH_EXECUTED",
               actual_effect: bool = True, source: str = "host",
               receipt_id: str | None = None) -> "HostReceipt":
        payload_json = _canonical(payload)
        payload_digest = digest(payload)
        if receipt_id is None:
            receipt_id = "receipt:" + hashlib.sha256(
                f"{identity.identity_digest}|{kind}|{sequence}|{payload_digest}".encode()
            ).hexdigest()[:24]
        return cls(
            receipt_id=receipt_id,
            kind=_text(kind, "kind"),
            sequence=sequence,
            identity=identity,
            effect_state=effect_state,
            evidence_state=evidence_state,
            actual_effect=actual_effect,
            payload_json=payload_json,
            payload_digest=payload_digest,
            source=_text(source, "source"),
        )

    @property
    def payload(self) -> object:
        return json.loads(self.payload_json)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "receipt_id": self.receipt_id,
            "kind": self.kind,
            "sequence": self.sequence,
            "identity": self.identity.as_dict(),
            "effect_state": self.effect_state,
            "evidence_state": self.evidence_state,
            "actual_effect": self.actual_effect,
            "payload": self.payload,
            "payload_digest": self.payload_digest,
            "source": self.source,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "HostReceipt":
        if value.get("schema") not in {None, SCHEMA}:
            raise ControlPlaneError("unsupported control-plane receipt schema")
        payload = value.get("payload")
        payload_json = _canonical(payload)
        payload_digest = str(value.get("payload_digest", digest(payload)))
        return cls(
            receipt_id=_text(value.get("receipt_id"), "receipt_id"),
            kind=_text(value.get("kind"), "kind"),
            sequence=int(value.get("sequence", -1)),
            identity=RunIdentity.from_mapping(value.get("identity", {})),
            effect_state=str(value.get("effect_state", "UNKNOWN")),
            evidence_state=str(value.get("evidence_state", UNKNOWN)),
            actual_effect=bool(value.get("actual_effect", False)),
            payload_json=payload_json,
            payload_digest=payload_digest,
            source=str(value.get("source", "host")),
        )


class ReceiptJournal:
    """Append-only receipt store with loss-preserving JSONL recovery."""

    def __init__(self, *, source_id: str = "live") -> None:
        self.source_id = _text(source_id, "source_id")
        self._records: dict[str, HostReceipt] = {}
        self.recovery_issues: list[str] = []

    @property
    def records(self) -> tuple[HostReceipt, ...]:
        return tuple(sorted(self._records.values(), key=lambda item: (item.sequence, item.receipt_id)))

    def append(self, receipt: HostReceipt) -> bool:
        current = self._records.get(receipt.receipt_id)
        if current is None:
            self._records[receipt.receipt_id] = receipt
            return True
        if current == receipt:
            return False
        raise ReceiptConflict(f"receipt {receipt.receipt_id} was mutated")

    def snapshot(self) -> list[dict[str, object]]:
        return [record.as_dict() for record in self.records]

    def recover_jsonl(self, text: str, *, source_id: str | None = None) -> dict[str, object]:
        """Read valid rows and preserve prior authority across suffix damage/replacement."""
        prior = dict(self._records)
        changed_source = source_id is not None and source_id != self.source_id
        if source_id is not None:
            self.source_id = _text(source_id, "source_id")
        recovered = 0
        ignored = 0
        issues: list[str] = []
        lines = str(text).splitlines(keepends=True)
        for index, raw in enumerate(lines, 1):
            complete_line = raw.endswith(("\n", "\r")) or index < len(lines)
            try:
                value = json.loads(raw)
                if not isinstance(value, Mapping):
                    raise ControlPlaneError("row is not an object")
                record = HostReceipt.from_mapping(value)
                self.append(record)
                recovered += 1
            except (ControlPlaneError, TypeError, ValueError, json.JSONDecodeError) as exc:
                ignored += 1
                if not complete_line:
                    issues.append(f"PARTIAL_SUFFIX:{index}")
                else:
                    issues.append(f"MALFORMED_SUFFIX:{index}:{type(exc).__name__}")
        # A replaced journal must not erase earlier authority rows; a conflicting
        # replacement row remains an explicit issue instead of silently winning.
        self.recovery_issues = list(dict.fromkeys(self.recovery_issues + issues))
        status = "REPLACED" if changed_source else ("RECOVERED" if issues else "CLEAN")
        return {
            "schema": SCHEMA,
            "status": status,
            "source_id": self.source_id,
            "recovered": recovered,
            "ignored": ignored,
            "preserved_prior": len(prior),
            "issues": tuple(self.recovery_issues),
            "authority_count": len(self._records),
        }


@dataclass(frozen=True)
class Settlement:
    effect_id: str
    outcome: str
    origin: str
    signal: str
    initiator: str
    evidence_id: str
    identity_digest: str


class PendingEffectLedger:
    """Exactly-once effect settlement, including candidate-switch invalidation."""

    def __init__(self) -> None:
        self._pending: dict[str, RunIdentity] = {}
        self._settled: dict[str, Settlement] = {}

    def register(self, effect_id: str, identity: RunIdentity) -> None:
        effect_id = _text(effect_id, "effect_id")
        prior = self._pending.get(effect_id)
        if prior is not None and prior.key != identity.key:
            raise IdentityMismatch(f"pending effect {effect_id} changed identity")
        if effect_id in self._settled:
            return
        self._pending[effect_id] = identity

    def settle(self, settlement: Settlement) -> bool:
        _text(settlement.effect_id, "effect_id")
        current = self._settled.get(settlement.effect_id)
        if current is not None:
            if current != settlement:
                raise SettlementConflict(f"effect {settlement.effect_id} settled twice differently")
            return False
        expected = self._pending.get(settlement.effect_id)
        if expected is not None and expected.identity_digest != settlement.identity_digest:
            raise IdentityMismatch(f"effect {settlement.effect_id} settled for another identity")
        self._settled[settlement.effect_id] = settlement
        self._pending.pop(settlement.effect_id, None)
        return True

    def reconcile_candidate(self, active: RunIdentity, *, origin: str = "candidate-switch") -> tuple[Settlement, ...]:
        settled: list[Settlement] = []
        for effect_id, identity in tuple(self._pending.items()):
            if identity.key != active.key:
                item = Settlement(
                    effect_id=effect_id,
                    outcome="STALE_CANDIDATE",
                    origin=origin,
                    signal="candidate_changed",
                    initiator="host",
                    evidence_id="identity:" + identity.identity_digest,
                    identity_digest=identity.identity_digest,
                )
                self.settle(item)
                settled.append(item)
        return tuple(settled)

    @property
    def pending(self) -> tuple[str, ...]:
        return tuple(sorted(self._pending))

    @property
    def settled(self) -> tuple[Settlement, ...]:
        return tuple(self._settled.values())


class UnknownLedger:
    """Unknowns remain visible until a matching host receipt resolves scope."""

    def __init__(self) -> None:
        self._rows: dict[str, dict[str, object]] = {}

    def add(self, unknown_id: str, identity: RunIdentity, *, reason: str) -> None:
        unknown_id = _text(unknown_id, "unknown_id")
        if unknown_id in self._rows:
            return
        self._rows[unknown_id] = {
            "unknown_id": unknown_id, "state": "OPEN", "reason": _text(reason, "reason"),
            "identity_digest": identity.identity_digest,
        }

    def resolve(self, unknown_id: str, receipt: HostReceipt, *, invalidates: bool = False) -> dict[str, object]:
        row = self._rows.get(_text(unknown_id, "unknown_id"))
        if row is None:
            return {"unknown_id": unknown_id, "state": "UNAVAILABLE"}
        if row["identity_digest"] != receipt.identity.identity_digest:
            row["state"] = "STALE_SCOPE"
            row["resolution_receipt"] = receipt.receipt_id
        else:
            row["state"] = "IRRELEVANT" if invalidates else "RESOLVED"
            row["resolution_receipt"] = receipt.receipt_id
        return dict(row)

    def rows(self) -> tuple[dict[str, object], ...]:
        return tuple(dict(row) for row in self._rows.values())


@dataclass(frozen=True)
class Freshness:
    admitted: bool
    status: str
    changed: tuple[str, ...]
    observed: Mapping[str, str]
    current: Mapping[str, str]


def check_freshness(observed: Mapping[str, object], current: Mapping[str, object]) -> Freshness:
    keys = tuple(sorted(set(observed) | set(current)))
    changed = tuple(key for key in keys if str(observed.get(key, UNKNOWN)) != str(current.get(key, UNKNOWN)))
    return Freshness(
        admitted=not changed and bool(keys),
        status="FRESH" if not changed and keys else ("STALE" if changed else UNKNOWN),
        changed=changed,
        observed={key: str(observed.get(key, UNKNOWN)) for key in keys},
        current={key: str(current.get(key, UNKNOWN)) for key in keys},
    )


def classify_failure(*, status: object, phase: object = None, signal: object = None,
                     started: object = True, refused: object = False) -> str:
    """Classify execution without turning setup/refusal/resource faults into red behavior."""
    if not bool(started):
        return "NEVER_STARTED"
    if bool(refused):
        return "REFUSAL"
    raw = " ".join(str(item or "").upper() for item in (status, phase, signal))
    if any(word in raw for word in ("TIMEOUT", "TIMED_OUT", "DEADLINE")):
        return "TIMEOUT"
    if any(word in raw for word in ("CPU", "MEMORY", "OOM", "RESOURCE", "KILL", "SIGNAL")):
        return "RESOURCE"
    if any(word in raw for word in ("IMPORT", "SETUP", "DATABASE", "ENVIRONMENT", "COLLECTION")):
        return "SETUP"
    if any(word in raw for word in ("FAIL", "ASSERT", "BEHAVIOR", "RED")):
        return "BEHAVIORAL"
    return "UNKNOWN"


@dataclass(frozen=True)
class BudgetSnapshot:
    responses: int
    tools: int
    tokens: int
    wall_clock_seconds: float
    terminal_reserve: int = 0
    host_elapsed_seconds: float = 0.0

    @property
    def binding_limit(self) -> str:
        values = {
            "responses": self.responses,
            "tools": self.tools,
            "tokens": self.tokens,
            "wall_clock_seconds": int(self.wall_clock_seconds),
        }
        exhausted = [name for name, value in values.items() if value <= 0]
        return exhausted[0] if exhausted else "NONE"

    def as_dict(self) -> dict[str, object]:
        return {
            "responses": self.responses, "tools": self.tools, "tokens": self.tokens,
            "wall_clock_seconds": self.wall_clock_seconds,
            "terminal_reserve": self.terminal_reserve,
            "host_elapsed_seconds": self.host_elapsed_seconds,
            "binding_limit": self.binding_limit,
        }


class ControlPlaneLedger:
    """Convenience host adapter used by StateFrame and offline replay fixtures."""

    def __init__(self, identity: RunIdentity, *, journal: ReceiptJournal | None = None) -> None:
        self.identity = identity
        self.journal = journal or ReceiptJournal()
        self.effects = PendingEffectLedger()
        self.unknowns = UnknownLedger()
        self._sequence = 0
        self._hypotheses: dict[str, dict[str, object]] = {}
        self._obligations: dict[str, dict[str, object]] = {}
        self._budgets: BudgetSnapshot | None = None

    def record(self, kind: str, payload: Mapping[str, object] | Sequence[object] | str | int | bool | None,
               *, effect_state: str = "OBSERVED", evidence_state: str = "FRESH_EXECUTED",
               actual_effect: bool = True, source: str = "host") -> HostReceipt:
        receipt = HostReceipt.create(
            kind=kind, sequence=self._sequence, identity=self.identity, payload=payload,
            effect_state=effect_state, evidence_state=evidence_state,
            actual_effect=actual_effect, source=source,
        )
        self.journal.append(receipt)
        self._sequence += 1
        return receipt

    def record_read(self, path: str, revision: str, *, start_line: int = 1,
                    end_line: int | None = None, content_digest: str | None = None) -> HostReceipt:
        return self.record("read", {
            "path": path, "revision": revision, "start_line": start_line,
            "end_line": end_line, "content_digest": content_digest or UNKNOWN,
        })

    def record_edit(self, path: str, before_revision: str, after_revision: str | None,
                    result: str, *, diff: str = "", supersedes: str | None = None) -> HostReceipt:
        result = _text(result, "result")
        return self.record("edit", {
            "path": path, "before_revision": before_revision, "after_revision": after_revision or UNKNOWN,
            "result": result, "diff": diff, "supersedes": supersedes,
        }, effect_state=result if result in EFFECT_STATES else "OBSERVED",
           actual_effect=result in {"APPLIED", "NO_OP"})

    def record_check(self, selector: str, status: str, *, evidence_state: str = "FRESH_EXECUTED",
                     failure_phase: str = "NONE", started: bool = True, signal: str = "") -> HostReceipt:
        failure = classify_failure(status=status, phase=failure_phase, signal=signal, started=started)
        return self.record("check", {
            "selector": selector, "status": status, "failure_class": failure,
            "failure_phase": failure_phase, "signal": signal,
        }, effect_state="OBSERVED" if started else "NOT_EXECUTED",
           evidence_state=evidence_state, actual_effect=started)

    def record_delivery(self, wire_bytes: bytes, *, message_id: str) -> HostReceipt:
        return self.record("delivery", {
            "message_id": message_id, "wire_digest": bytes_digest(wire_bytes),
            "wire_bytes": len(wire_bytes),
        }, effect_state="DELIVERED", evidence_state="FRESH_EXECUTED", actual_effect=True)

    def cumulative_diff(self) -> tuple[dict[str, object], ...]:
        rows = []
        for receipt in self.journal.records:
            if receipt.kind == "edit" and isinstance(receipt.payload, Mapping):
                row = dict(receipt.payload)
                row.update({"receipt_id": receipt.receipt_id, "sequence": receipt.sequence,
                            "candidate_digest": receipt.identity.candidate_digest,
                            "revision": receipt.identity.revision})
                rows.append(row)
        return tuple(rows)

    def read_coverage(self) -> tuple[dict[str, object], ...]:
        seen: dict[tuple[str, str, object, object], int] = {}
        rows: list[dict[str, object]] = []
        for receipt in self.journal.records:
            if receipt.kind != "read" or not isinstance(receipt.payload, Mapping):
                continue
            payload = receipt.payload
            key = (str(payload.get("path")), str(payload.get("revision")),
                   payload.get("start_line"), payload.get("end_line"))
            repeat = seen.get(key, 0)
            seen[key] = repeat + 1
            rows.append({"receipt_id": receipt.receipt_id, "path": key[0], "revision": key[1],
                         "start_line": key[2], "end_line": key[3], "repeat_count": repeat,
                         "new_information": repeat == 0})
        return tuple(rows)

    def set_hypothesis(self, hypothesis_id: str, claim: str, *, evidence: Sequence[str] = (),
                       competing: Sequence[str] = (), switch_on: str = UNKNOWN) -> None:
        self._hypotheses[_text(hypothesis_id, "hypothesis_id")] = {
            "id": hypothesis_id, "claim": _text(claim, "claim"), "evidence": tuple(evidence),
            "competing": tuple(competing), "switch_on": _text(switch_on, "switch_on"),
            "state": "CURRENT",
        }

    def switch_hypothesis(self, hypothesis_id: str, *, reason: str, receipt_id: str) -> dict[str, object]:
        row = self._hypotheses.get(_text(hypothesis_id, "hypothesis_id"))
        if row is None:
            raise ControlPlaneError("unknown hypothesis")
        row.update({"state": "SWITCHED", "switch_reason": _text(reason, "reason"),
                    "switch_receipt": _text(receipt_id, "receipt_id")})
        return dict(row)

    def set_obligation(self, obligation_id: str, *, status: str, evidence_ids: Sequence[str] = ()) -> None:
        self._obligations[_text(obligation_id, "obligation_id")] = {
            "id": obligation_id, "status": _text(status, "status"),
            "candidate_digest": self.identity.candidate_digest,
            "revision": self.identity.revision, "evidence_ids": tuple(evidence_ids),
        }

    def set_budget(self, snapshot: BudgetSnapshot) -> None:
        self._budgets = snapshot

    def context(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "identity": self.identity.as_dict(),
            "identity_complete": self.identity.complete,
            "receipts": [record.as_dict() for record in self.journal.records],
            "cumulative_diff": list(self.cumulative_diff()),
            "read_coverage": list(self.read_coverage()),
            "hypotheses": list(self._hypotheses.values()),
            "obligations": list(self._obligations.values()),
            "budgets": None if self._budgets is None else self._budgets.as_dict(),
            "unknowns": list(self.unknowns.rows()),
            "pending_effects": self.effects.pending,
        }


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    source = os.environ if environ is None else environ
    value = str(source.get(MODE_ENV, "off")).strip().lower()
    if value not in {"off", "on"}:
        raise ControlPlaneError(f"{MODE_ENV} must be off or on")
    return value == "on"


def render_control_plane_message(frame: Mapping[str, object]) -> str | None:
    """Render only host receipts supplied by the caller; never infer from prose."""
    if not enabled() or not isinstance(frame.get("control_plane"), Mapping):
        return None
    control = frame["control_plane"]
    identity = control.get("identity")
    if not isinstance(identity, Mapping):
        return "[CONTROL PLANE | UNKNOWN] identity unavailable; no claim promoted"
    try:
        parsed = RunIdentity.from_mapping(identity)
    except (ControlPlaneError, TypeError, ValueError):
        return "[CONTROL PLANE | UNKNOWN] identity invalid; no claim promoted"
    raw_receipts = control.get("receipts")
    receipt_count = control.get("receipt_count", 0)
    receipt_status = "UNKNOWN"
    if isinstance(raw_receipts, Sequence) and not isinstance(raw_receipts, (str, bytes)):
        try:
            receipts = tuple(
                HostReceipt.from_mapping(row) for row in raw_receipts
                if isinstance(row, Mapping)
            )
            if receipts:
                require_same_identity(receipts, parsed)
                receipt_count = len(receipts)
                receipt_status = "READY"
        except (ControlPlaneError, TypeError, ValueError):
            receipt_status = "UNKNOWN"
    status = "READY" if parsed.complete and receipt_status == "READY" else "UNKNOWN"
    return (f"[CONTROL PLANE | {status}] run={parsed.run_uuid} task={parsed.task_id}@{parsed.task_version} "
            f"candidate={parsed.candidate_digest} revision={parsed.revision} receipts={receipt_count}")


__all__ = [
    "ATTEMPT_CLASSES", "BudgetSnapshot", "ControlPlaneError", "ControlPlaneLedger",
    "EFFECT_STATES", "EVIDENCE_STATES", "FAILURE_CLASSES", "Freshness", "HostReceipt",
    "IdentityError", "IdentityMismatch", "MODE_ENV", "PendingEffectLedger", "ReceiptConflict",
    "ReceiptJournal", "RunIdentity", "SCHEMA", "Settlement", "SettlementConflict",
    "UNAVAILABLE", "UNKNOWN", "UnknownLedger", "bytes_digest", "check_freshness",
    "classify_failure", "digest", "enabled", "render_control_plane_message",
    "require_same_identity",
]
