"""Durable, typed working-set suspension. No production caller invokes suspend.

Only a measured mandatory envelope over its byte cap is a BLOCKED condition.
Search exhaustion and the other diagnostic codes are not impossibility proofs.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
import hashlib

if __package__:
    from .kernel_read_receipts import enabled
    from .kernel_working_set_state import emit_event, state_for
else:
    from kernel_read_receipts import enabled
    from kernel_working_set_state import emit_event, state_for


class ReasonCode(str, Enum):
    MANDATORY_ENVELOPE_OVER_CAP = "MANDATORY_ENVELOPE_OVER_CAP"
    # The mandatory floor itself (fixed + protected + pins) cannot fit the cap:
    # the same measured proof, reported when no candidate below the floor exists.
    FLOOR_INFEASIBLE = "FLOOR_INFEASIBLE"
    BYTE_POLICY_DOMAIN_EXHAUSTED = "BYTE_POLICY_DOMAIN_EXHAUSTED"
    TOKEN_ADMISSION_SEARCH_EXHAUSTED = "TOKEN_ADMISSION_SEARCH_EXHAUSTED"
    ACTIVE_OBLIGATION_LIMIT = "ACTIVE_OBLIGATION_LIMIT"
    RECOVERY_ROUTE_UNAVAILABLE = "RECOVERY_ROUTE_UNAVAILABLE"
    SOURCE_SNAPSHOT_UNSTABLE = "SOURCE_SNAPSHOT_UNSTABLE"
    CAPTURE_BLOB_UNAVAILABLE = "CAPTURE_BLOB_UNAVAILABLE"
    PREPARATION_CONTRACT_FAILURE = "PREPARATION_CONTRACT_FAILURE"


# Only measured impossibility proofs may stop a run; everything else is diagnostic.
BLOCKING_REASONS = frozenset({ReasonCode.MANDATORY_ENVELOPE_OVER_CAP, ReasonCode.FLOOR_INFEASIBLE})


@dataclass(frozen=True)
class MandatoryEnvelopeEvidence:
    """Measured evidence restored from the authenticated journal, never a new proof."""
    mandatory_bytes: int
    byte_cap: int
    body_sha256: str
    breakdown: tuple[tuple[str, int], ...]
    measurement_method: str = "serialized_bytes"

    def validate(self) -> None:
        if (type(self.mandatory_bytes) is not int or type(self.byte_cap) is not int
                or self.byte_cap < 0 or self.mandatory_bytes <= self.byte_cap):
            raise ValueError("mandatory envelope must exceed the cap")
        if (len(self.body_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.body_sha256)
                or self.measurement_method != "serialized_bytes"):
            raise ValueError("invalid mandatory envelope measurement")
        names = [name for name, _ in self.breakdown]
        if (not names or len(names) != len(set(names))
                or not set(names) <= {"fixed", "protected", "pins"}
                or any(type(n) is not int or n < 0 for _, n in self.breakdown)
                or sum(n for _, n in self.breakdown) != self.mandatory_bytes):
            raise ValueError("mandatory envelope breakdown must cover exact bytes")

    def document(self) -> dict[str, object]:
        return {"mandatory_bytes": self.mandatory_bytes, "byte_cap": self.byte_cap,
                "body_sha256": self.body_sha256, "breakdown": dict(self.breakdown),
                "measurement_method": self.measurement_method,
                "verified": True, "mandatory_only": True}


@dataclass(frozen=True)
class MandatoryEnvelopeProof:
    mandatory_wire: bytes = field(repr=False)
    byte_cap: int
    breakdown: tuple[tuple[str, int], ...]

    def evidence(self) -> MandatoryEnvelopeEvidence:
        if not isinstance(self.mandatory_wire, bytes):
            raise ValueError("mandatory envelope requires exact serialized bytes")
        return MandatoryEnvelopeEvidence(len(self.mandatory_wire), self.byte_cap,
            hashlib.sha256(self.mandatory_wire).hexdigest(), self.breakdown)

    def validate(self) -> None:
        self.evidence().validate()

    def document(self) -> dict[str, object]:
        return self.evidence().document()


@dataclass(frozen=True)
class Suspension:
    reason_code: ReasonCode
    logical_request_id: str
    due_ordinal: int
    snapshot_digest: str
    active_obligation_digest: str
    violated_limits: tuple[str, ...]
    proof: MandatoryEnvelopeProof | MandatoryEnvelopeEvidence
    minimum_policy_candidate_ref: str
    trials_ref: str
    recovery_condition: str

    def validate(self) -> None:
        if (not isinstance(self.reason_code, ReasonCode)
                or self.reason_code not in BLOCKING_REASONS):
            raise ValueError("BLOCKED requires typed mandatory-envelope-over-cap")
        if (not self.logical_request_id or type(self.due_ordinal) is not int
                or self.due_ordinal < 1 or not self.minimum_policy_candidate_ref
                or not self.trials_ref or not self.recovery_condition
                or self.violated_limits != ("serialized_bytes",)):
            raise ValueError("invalid suspension boundary or evidence references")
        for digest in (self.snapshot_digest, self.active_obligation_digest):
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)):
                raise ValueError("invalid suspension digest")
        if not isinstance(self.proof, (MandatoryEnvelopeProof, MandatoryEnvelopeEvidence)):
            raise ValueError("suspension requires typed mandatory-envelope evidence")
        self.proof.validate()

    def document(self) -> dict[str, object]:
        self.validate()
        return {"reason_code": self.reason_code.value,
                "logical_request_id": self.logical_request_id, "due_ordinal": self.due_ordinal,
                "snapshot_digest": self.snapshot_digest,
                "active_obligation_digest": self.active_obligation_digest,
                "violated_limits": list(self.violated_limits), "proof": self.proof.document(),
                "minimum_policy_candidate_ref": self.minimum_policy_candidate_ref,
                "trials_ref": self.trials_ref, "recovery_condition": self.recovery_condition}

    @classmethod
    def from_document(cls, value: Mapping[str, object]) -> Suspension:
        fields = dict(value)
        fields["reason_code"] = ReasonCode(fields["reason_code"])
        fields["violated_limits"] = tuple(fields["violated_limits"])
        proof = dict(fields["proof"])
        if proof.pop("verified", None) is not True or proof.pop("mandatory_only", None) is not True:
            raise ValueError("unverified mandatory envelope evidence")
        proof["breakdown"] = tuple(proof["breakdown"].items())
        fields["proof"] = MandatoryEnvelopeEvidence(**proof)
        suspension = cls(**fields)
        suspension.validate()
        return suspension


class WorkingSetSuspended(Exception):
    """Internal orderly stop signal, caught before all generic request failures."""
    def __init__(self, suspension: Suspension, *, counts: Mapping[str, int] | None = None):
        suspension.validate()
        self.suspension = suspension
        self.counts = dict(counts or {})
        super().__init__(suspension.reason_code.value)

    def outcome(self) -> dict[str, object]:
        document = self.suspension.document()
        problem = {"reason_code": "WORKING_SET_ALLOCATION_BLOCKED", **self.counts,
                   "suspension": document}
        return {"result": "BLOCKED", "reason_code": "WORKING_SET_ALLOCATION_BLOCKED",
                "accepted": False, "certified": False, "resumable": True,
                "suspension": document, "problems": [problem]}


def is_suspended_result(result: Mapping[str, object]) -> bool:
    if (not enabled() or result.get("result") != "BLOCKED"
            or result.get("reason_code") != "WORKING_SET_ALLOCATION_BLOCKED"
            or result.get("resumable") is not True):
        return False
    try:
        Suspension.from_document(result["suspension"])
    except (KeyError, TypeError, ValueError, AttributeError):
        return False
    return True


def record_problem(runtime: object, problem: Mapping[str, object]) -> None:
    """Bounded local diagnostics; the outcome always carries its own problem too."""
    problems = getattr(runtime, "_working_set_problems", None)
    if problems is None:
        problems = runtime._working_set_problems = []
    if len(problems) < 32:
        problems.append(dict(problem))


def suspend(runtime: object, suspension: Suspension) -> None:
    if not enabled():
        return
    suspension.validate()
    if not isinstance(suspension.proof, MandatoryEnvelopeProof):
        raise ValueError("a new suspension requires freshly verified mandatory bytes")
    if __package__:
        from .kernel_read_receipts import _blob
    else:
        from kernel_read_receipts import _blob
    state = state_for(runtime)
    # capture_read writes blobs and its lifecycle event synchronously before
    # returning a successful page. There are no deferred capture writes to flush.
    # Persist the exact measured candidate too, before the durable blocked event.
    candidate_blob = _blob(runtime, suspension.proof.mandatory_wire.decode("utf-8"))
    counts = {"active_count": len(state.active_obligations()),
              "due_count": sum(o.state == "DUE" for o in state.active_obligations()),
              "pinned_count": sum(o.recovery_pin for o in state.active_obligations())}
    signal = WorkingSetSuspended(suspension, counts=counts)
    emit_event(runtime, "working_set_blocked", suspension=suspension.document(),
               active_index=state.checkpoint(), mandatory_envelope_blob_ref=candidate_blob,
               **counts)
    # emit_event sets state.suspension only AFTER the event has been fsynced.
    record_problem(runtime, signal.outcome()["problems"][0])
    raise signal


def inject_suspension_for_test(runtime: object, suspension: Suspension) -> None:
    """In-process injection only. No environment/configuration activation path."""
    runtime._working_set_test_injection = suspension


def check_test_injection(runtime: object) -> None:
    if not enabled():
        return
    suspension = getattr(runtime, "_working_set_test_injection", None)
    if suspension is not None:
        del runtime._working_set_test_injection
        suspend(runtime, suspension)
