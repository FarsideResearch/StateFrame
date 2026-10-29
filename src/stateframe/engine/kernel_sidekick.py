"""Receipt-based attempt observer. Success remains exclusively the grader's decision."""
from __future__ import annotations

import os
from collections.abc import Mapping

from kernel_verdict import SATISFIED, evaluate_required_tests


def review(bundle: Mapping) -> dict:
    def finding(name: str, reason: str, evidence: list, **extra) -> dict:
        return {"finding": name, "reason": reason, "evidence": evidence,
                "system_denominator_included": True, **extra}

    # Only the kernel's validated rule event can invalidate/isolate an attempt.
    for receipt in bundle.get("integrity_receipts", ()):
        if (receipt.get("producer") == "kernel_integrity"
                and receipt.get("validated") is True and receipt.get("receipt_id")
                and receipt.get("rule") == "protected_oracle_modified_by_worker"
                and receipt.get("actor") == "worker" and receipt.get("before_sha")
                and receipt.get("after_sha") != receipt["before_sha"]):
            return finding("INVALIDATE_ATTEMPT", receipt["rule"], [receipt])
    for receipt in bundle.get("fault_receipts", ()):
        if (receipt.get("producer") == "harness_transport" and receipt.get("receipt_id")
                and receipt.get("validated") is True and receipt.get("scope")
                and receipt.get("fault") in {"transport_exception", "transport_timeout", "render_overflow"}):
            return finding("QUARANTINE_HARNESS", "isolate the named failed request", [receipt],
                           scope=receipt["scope"])
    required = tuple(bundle.get("required_selectors", ()))
    binding = bundle.get("current_binding", {})
    if not required or not binding or any(v in (None, "MISSING", "UNKNOWN") for v in binding.values()):
        return finding("UNVERIFIED", "required coverage or revision identity unavailable", [])
    authority = evaluate_required_tests(binding, required, bundle.get("test_receipts", ()))
    evidence = list(authority.supporting_evidence)
    if authority.state != SATISFIED and authority.state == "UNRESOLVED" \
            and bundle.get("event") == "completion_request" and not bundle.get("challenged"):
        challenge = finding("CHALLENGE_ONCE", "missing current verification", evidence,
                            reason_codes=list(authority.reason_codes), selectors=list(required))
        if os.environ.get("STATEFRAME_CHALLENGE_SCHEMA", "off").strip().lower() == "on":
            from kernel_challenge_schema import repair_finding
            return repair_finding(challenge, required)
        return challenge
    if authority.state != SATISFIED:
        reason = ("current checks are not green" if authority.state == "VIOLATED"
                  else "current verification is unresolved" if authority.state == "UNRESOLVED"
                  else "current verification conflicts")
        return finding("UNVERIFIED", reason, evidence,
                       reason_codes=list(authority.reason_codes))
    return finding("READY_FOR_GRADING", "current receipt coverage; authoritative grading still required",
                   evidence, next_authority="existing_grader",
                   reason_codes=list(authority.reason_codes))


class Sidekick:
    EVENTS = {"completion_request", "attempt_termination", "harness_fault", "integrity_violation"}

    def __init__(self):
        self.challenged = False

    def observe(self, event: str, bundle: Mapping) -> dict:
        if event not in self.EVENTS:
            raise ValueError("Sidekick only observes attempt lifecycle events")
        return review({**bundle, "event": event, "challenged": self.challenged})
