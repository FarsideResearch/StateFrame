"""Carrier-bound proof of a complete provider wire, with exact recomputation."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields
from functools import cache
import json

if __package__:
    from .kernel_working_set_cache import independent
    from .kernel_read_receipts import DecodedSpan, ReceiptIdentity, line_digests_with_numbers
    from .kernel_working_set_allocator import FULL, DEFERRED
    from .kernel_working_set_snapshot import PreparationSnapshot, SourceSlot
    from .kernel_working_set_render import (AssignmentCertificate, CarrierWitness,
                                           DeferredPlanWitness, _recovery_arguments, _unit_active,
                                           deferred_plan_slot, digest, render_assignment)
else:
    from kernel_working_set_cache import independent
    from kernel_read_receipts import DecodedSpan, ReceiptIdentity, line_digests_with_numbers
    from kernel_working_set_allocator import FULL, DEFERRED
    from kernel_working_set_snapshot import PreparationSnapshot, SourceSlot
    from kernel_working_set_render import (AssignmentCertificate, CarrierWitness,
                                          DeferredPlanWitness, _recovery_arguments, _unit_active,
                                          deferred_plan_slot, digest, render_assignment)


class CertificateError(ValueError):
    """The trusted assignment, reconstructed body or physical witness differs."""


@cache
def _certificate_fields():
    return {
        cls: tuple(field.name for field in fields(cls))
        for cls in (AssignmentCertificate, PreparationSnapshot, SourceSlot, ReceiptIdentity,
                    DecodedSpan, CarrierWitness, DeferredPlanWitness)
    }


_IMMUTABLE_LEAVES = frozenset((str, bytes, bool, int, float, type(None)))
_PROOF_VERSION = 1
_MAX_PROOF_NODES = 100_000


@dataclass
class _QualificationProof:
    request_id: str
    lineage: str
    generation: int
    verified: tuple[tuple[object, ...], bytes] | None = None
    closed: bool = False

    def close(self) -> None:
        self.closed = True
        self.verified = None


_QUALIFICATION_PROOF: ContextVar[_QualificationProof | None] = ContextVar(
    "qualification_proof", default=None,
)


@contextmanager
def qualification_proof(request_id: str, lineage: str, generation: int):
    """Retain one independently verified proof within a fresh qualification."""
    scope = _QualificationProof(request_id, lineage, generation)
    token = _QUALIFICATION_PROOF.set(scope)
    try:
        yield
    finally:
        scope.close()
        _QUALIFICATION_PROOF.reset(token)


def discard_qualification_proof(request_id: str) -> None:
    scope = _QUALIFICATION_PROOF.get()
    if scope is not None and scope.request_id == request_id:
        scope.close()


def _proof_key(scope: _QualificationProof, certificate: AssignmentCertificate) -> tuple | None:
    # Unit mode participates in the renderer's assignment domains. Keep those
    # checks eager. The plain renderer otherwise uses detached authority only.
    if scope.closed or type(certificate) is not AssignmentCertificate or _unit_active():
        return None
    snapshot = getattr(certificate, "snapshot", None)
    layout = getattr(certificate, "layout", None)
    request_id = getattr(snapshot, "request_id", None)
    if (type(snapshot) is not PreparationSnapshot
            or type(request_id) is not str or request_id != scope.request_id
            or type(layout) is not str or layout != "plain"):
        return None
    # Walk the actual fields, including witnesses, rather than trusting claimed
    # digests. Exact leaf identity avoids hashing source text or treating True
    # and 1 as equal. Keep strong references until the qualifier exits.
    order = []
    stack = [certificate]
    certificate_fields = _certificate_fields()
    while stack:
        value = stack.pop()
        kind = type(value)
        order.append(value)
        if kind is tuple:
            if len(order) + len(stack) + len(value) > _MAX_PROOF_NODES:
                return None
            stack.extend(reversed(value))
        elif kind in certificate_fields:
            try:
                stack.extend(getattr(value, name) for name in reversed(certificate_fields[kind]))
            except AttributeError:
                return None
        elif kind not in _IMMUTABLE_LEAVES:
            return None
        if len(order) + len(stack) > _MAX_PROOF_NODES:
            return None
    return (_PROOF_VERSION, scope.request_id, scope.lineage, scope.generation,
            certificate.body_sha256, certificate.snapshot_digest, certificate.assignment_digest,
            tuple(order))


def _verified_wire(scope: _QualificationProof, key: tuple) -> bytes | None:
    entry = scope.verified
    if entry is not None and not scope.closed:
        previous, wire = entry
        if (previous[:-1] == key[:-1] and len(previous[-1]) == len(key[-1])
                and all(old is new for old, new in zip(previous[-1], key[-1]))):
            return wire
    return None


def verify_messages(messages, certificate: AssignmentCertificate) -> bool:
    """Product-boundary source proof without legacy path-based reconciliation."""
    scope = _QUALIFICATION_PROOF.get()
    key = None if scope is None else _proof_key(scope, certificate)
    verified_wire = None if key is None else _verified_wire(scope, key)
    wire = verified_wire
    if wire is None:
        modes = {FULL.name: FULL, DEFERRED.name: DEFERRED}
        wire = render_assignment(certificate.snapshot,
            {key: modes[mode] for key, mode in certificate.assignment}, certificate.layout).wire
    physical = json.loads(wire)["messages"]
    if physical != messages:
        # The product validator sees semantic messages for a final-cache wire.
        from kernel_cache_layout_view import semantic_messages
        if semantic_messages(json.loads(wire)["messages"]) != messages:
            raise CertificateError("product source carriers differ from certified wire")
    return verified_wire is not None or verify_certificate(wire, certificate)


def verify_certificate(wire: bytes, certificate: AssignmentCertificate) -> bool:
    scope = _QUALIFICATION_PROOF.get()
    key = None if scope is None or type(wire) is not bytes else _proof_key(scope, certificate)
    if key is not None and wire == _verified_wire(scope, key):
        return True
    verified = _verify_certificate(wire, certificate)
    if key is not None and not scope.closed:
        scope.verified = (key, wire)
    return verified


@independent
def _verify_certificate(wire: bytes, certificate: AssignmentCertificate) -> bool:
    """Validate exact bytes at host-owned coordinates, then recompute the wire.

    The caller supplies a trusted renderer certificate. Model-authored paths,
    range strings and notices are never inputs to assignment authorization.
    """
    try:
        if not isinstance(wire, bytes) or digest(wire) != certificate.body_sha256:
            raise CertificateError("certificate body digest differs")
        modes = {FULL.name: FULL, DEFERRED.name: DEFERRED}
        assignment = {key: modes[mode] for key, mode in certificate.assignment}
        if len(assignment) != len(certificate.assignment):
            raise CertificateError("duplicate certified assignment")
        snapshot = certificate.snapshot
        slots = {s.key: s for s in snapshot.slots}
        ordered = sorted(snapshot.slots, key=lambda s: (s.admission_ordinal, s.key))
        aggregate_ids = tuple(s.key for s in ordered
                              if assignment[s.key] == DEFERRED and deferred_plan_slot(s))
        expected_order = tuple(s.key for s in ordered if s.key not in aggregate_ids)
        if tuple(w.slot_id for w in certificate.witnesses) != expected_order:
            raise CertificateError("witnesses do not cover frozen slots exactly once")
        body = json.loads(wire)
        aggregate = certificate.deferred_plan_witness
        if aggregate_ids:
            if aggregate is None or aggregate.slot_ids != aggregate_ids:
                raise CertificateError("deferred plan witness does not cover frozen units exactly once")
            if type(aggregate.message_index) is not int or aggregate.message_index < 0:
                raise CertificateError("invalid deferred plan carrier index")
            raw = body["messages"][aggregate.message_index]["content"].encode()
            if (aggregate.content_start != 0 or aggregate.content_end != len(raw)
                    or aggregate.carrier_sha256 != digest(raw)):
                raise CertificateError("deferred plan carrier differs")
            # Path-level prose never authorizes omission by itself. Each unit
            # needs a route or explicit unavailability under frozen authority.
            for key in aggregate_ids:
                _recovery_arguments(snapshot, slots[key])
        elif aggregate is not None:
            raise CertificateError("unexpected deferred plan witness")
        for witness in certificate.witnesses:
            slot = slots[witness.slot_id]
            recovery = (slot.recovery_arguments if assignment[slot.key] == FULL
                        else _recovery_arguments(snapshot, slot))
            if (witness.receipt_identity != slot.identity
                    or witness.obligation_ids != slot.obligation_ids
                    or witness.mode != assignment[slot.key].name
                    or witness.recovery_arguments != recovery):
                raise CertificateError("witness differs from trusted assignment")
            if type(witness.message_index) is not int or witness.message_index < 0:
                raise CertificateError("invalid carrier index")
            raw = body["messages"][witness.message_index]["content"].encode()
            if (witness.content_start != 0 or witness.content_end != len(raw)
                    or digest(raw) != witness.carrier_sha256):
                raise CertificateError("owned carrier differs")
            if assignment[slot.key] == FULL:
                start, end = witness.payload_start, witness.payload_end
                if type(start) is not int or type(end) is not int or not 0 <= start <= end <= len(raw):
                    raise CertificateError("invalid payload coordinates")
                payload = raw[start:end]
                if payload != slot.payload.encode() or digest(payload) != slot.identity.payload_sha256:
                    raise CertificateError("FULL payload differs from immutable capture")
                if line_digests_with_numbers(payload.decode(), slot.identity.first_line) != (
                        slot.line_digests, slot.line_numbers):
                    raise CertificateError("FULL receipt line evidence differs")
            elif witness.payload_start is not None or witness.payload_end is not None:
                raise CertificateError("omission cannot claim a payload span")
        rebuilt = render_assignment(snapshot, assignment, certificate.layout)
        if rebuilt.wire != wire or rebuilt.certificate != certificate:
            raise CertificateError("complete candidate recomputation differs")
    except CertificateError:
        raise
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, UnicodeError) as error:
        raise CertificateError("malformed assignment certificate") from error
    return True
