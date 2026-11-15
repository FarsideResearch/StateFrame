"""Carrier-bound proof of a complete provider wire, with exact recomputation."""
from __future__ import annotations

import json

if __package__:
    from .kernel_working_set_cache import independent
    from .kernel_read_receipts import line_digests_with_numbers
    from .kernel_working_set_allocator import FULL, DEFERRED
    from .kernel_working_set_render import (AssignmentCertificate, _recovery_arguments,
                                           deferred_plan_slot, digest, render_assignment)
else:
    from kernel_working_set_cache import independent
    from kernel_read_receipts import line_digests_with_numbers
    from kernel_working_set_allocator import FULL, DEFERRED
    from kernel_working_set_render import (AssignmentCertificate, _recovery_arguments,
                                          deferred_plan_slot, digest, render_assignment)


class CertificateError(ValueError):
    """The trusted assignment, reconstructed body or physical witness differs."""


def verify_messages(messages, certificate: AssignmentCertificate) -> bool:
    """Product-boundary source proof without legacy path-based reconciliation."""
    modes = {FULL.name: FULL, DEFERRED.name: DEFERRED}
    rendered = render_assignment(certificate.snapshot,
        {key: modes[mode] for key, mode in certificate.assignment}, certificate.layout)
    if json.loads(rendered.wire)["messages"] != messages:
        # The product validator sees semantic messages for a final-cache wire.
        from kernel_cache_layout_view import semantic_messages
        if semantic_messages(json.loads(rendered.wire)["messages"]) != messages:
            raise CertificateError("product source carriers differ from certified wire")
    return verify_certificate(rendered.wire, certificate)


@independent
def verify_certificate(wire: bytes, certificate: AssignmentCertificate) -> bool:
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
