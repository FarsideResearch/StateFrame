"""Carrier-bound proof of a complete shadow wire, with exact recomputation."""
from __future__ import annotations

import json

if __package__:
    from .kernel_read_receipts import line_digests_with_numbers
    from .kernel_working_set_allocator import FULL, DEFERRED
    from .kernel_working_set_render import (AssignmentCertificate, _recovery_arguments,
                                           digest, render_assignment)
else:
    from kernel_read_receipts import line_digests_with_numbers
    from kernel_working_set_allocator import FULL, DEFERRED
    from kernel_working_set_render import (AssignmentCertificate, _recovery_arguments,
                                          digest, render_assignment)


class CertificateError(ValueError):
    """The trusted assignment, reconstructed body or physical witness differs."""


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
        expected_order = tuple(s.key for s in sorted(snapshot.slots,
                               key=lambda s: (s.admission_ordinal, s.key)))
        if tuple(w.slot_id for w in certificate.witnesses) != expected_order:
            raise CertificateError("witnesses do not cover frozen slots exactly once")
        body = json.loads(wire)
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
