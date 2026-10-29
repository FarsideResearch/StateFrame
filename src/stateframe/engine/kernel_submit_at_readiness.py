"""Typed owner for submission at the first certified current-revision boundary.

The module feeds ``kernel_completion_owner.before_provider_request`` through
``kernel_green_then_exhaust.submit_if_ready``.  It never renders a menu or
terminal line and retries a refusal only when the certification key changes.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from kernel_done_acceptance import done_accepted


MODE_ENV = "STATEFRAME_SUBMIT_AT_READINESS"
MANIFEST_KEY = "stateframe_submit_at_readiness"
PRODUCER = "kernel_submit_at_readiness"
SCHEMA = "stateframe-submit-at-readiness/1"
COMMITTED_EVENT = "readiness_submit_committed"
REFUSED_EVENT = "readiness_submit_refused"
DEFERRED_EVENT = "readiness_submit_deferred"
BOUNDARY = "readiness_submit"

_KEY = "_readiness_submit_certified_key"
_ATTEMPTED = "_readiness_submit_attempted_keys"
_DEFERRED_KEY = "_readiness_submit_uncertified_deferred_key"


def mode(environ: Mapping[str, str] | None = None) -> str:
    value = str((os.environ if environ is None else environ).get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(runtime: object | None = None) -> bool:
    configured = getattr(runtime, "stateframe_submit_at_readiness_mode", None) if runtime is not None else None
    return str(configured).strip().lower() == "on" if configured is not None else mode() == "on"


def _emit(runtime: object, kind: str, **fields: object) -> None:
    emit = getattr(getattr(runtime, "journal", None), "emit", None)
    if not callable(emit):
        return
    try:
        emit(kind, producer=PRODUCER, payload_schema=SCHEMA, dispatch=False, **fields)
    except Exception:  # noqa: BLE001 - telemetry never changes the boundary
        return


def _verdict(runtime: object) -> str:
    from kernel_greenlock_terminal_fixes import public_verdict
    return public_verdict(runtime)


def certified_key(runtime: object) -> tuple[str, str] | None:
    """Return ``(revision, candidate_digest)`` for trusted current GREEN readiness."""
    from kernel_green_then_exhaust import _authority_enabled, _continuation_ready, _host_submittable
    if not _authority_enabled(runtime) or done_accepted(runtime):
        return None
    if _verdict(runtime) != "GREEN":
        return None
    method = getattr(runtime, "readiness_decision", None)
    if not callable(method):
        return None
    try:
        decision = method()
    except (AttributeError, TypeError, ValueError):
        return None
    if decision is None or not _host_submittable(decision) or not _continuation_ready(runtime):
        return None
    revision = getattr(decision, "revision", None)
    if revision in (None, ""):
        binding = getattr(runtime, "binding", None)
        if not callable(binding):
            return None
        try:
            revision = binding()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return None
    if revision in (None, ""):
        return None
    return str(revision), str(getattr(decision, "candidate_digest", "") or "")


def _codes(completion: object) -> list[str]:
    codes = getattr(completion, "reason_codes", None)
    if isinstance(codes, (list, tuple)):
        return [str(code) for code in codes][:16]
    return []


def at_request_boundary(runtime: object) -> bool:
    """Submit once at the readiness boundary through the shared done seal."""
    if not enabled(runtime) or done_accepted(runtime):
        return False
    from kernel_done_is_submit_authority import (
        enabled as done_submit_enabled,
        request_submission,
        wait_for_pending_mutations,
    )
    if done_submit_enabled(runtime):
        # DESIGN RULE: done comes from the MODEL; DONE_IS_SUBMIT only means the host never
        # refuses a model done. This branch submitted as the HOST with no readiness check, at step 0 before
        # request 0 (r2EY DONE-env smoke: readiness_submit_committed seq 134, requests=0, host_stopped). Under
        # DONE_IS_SUBMIT the host never submits on its own.
        del request_submission, wait_for_pending_mutations
        return False
    from kernel_done_readiness_authority import cached_decision, decision_for_runtime
    shared_decision = cached_decision(runtime)
    if shared_decision is None:
        shared_decision = decision_for_runtime(
            runtime,
            full_tools=(getattr(runtime, "_completion_owner_full_tools", ())
                        or getattr(runtime, "_readiness_full_tools", ())),
        )
    if (shared_decision is not None
            and getattr(shared_decision, "decision", None) == "ACCEPTED_UNCERTIFIED"):
        # This boundary is the certified fast path.  Keep the shared host-debt
        # decision intact and leave the first model done call to the completion
        # owner; emitting a readiness refusal here creates a second authority.
        setattr(runtime, _KEY, None)
        decision_key = getattr(shared_decision, "key", None)
        if getattr(runtime, _DEFERRED_KEY, None) != decision_key:
            setattr(runtime, _DEFERRED_KEY, decision_key)
            open_ids = tuple(dict.fromkeys((
                *tuple(getattr(shared_decision, "blocker_ids", ()) or ()),
                *tuple(getattr(shared_decision, "open_obligation_ids", ()) or ()),
            )))
            _emit(
                runtime, DEFERRED_EVENT,
                reason="shared_uncertified_submit_requires_done_dispatch",
                decision=getattr(shared_decision, "decision", "UNKNOWN"),
                reason_code=getattr(shared_decision, "reason_code", ""),
                open_ids=list(open_ids),
            )
        return False
    setattr(runtime, _DEFERRED_KEY, None)
    key = certified_key(runtime)
    previous = getattr(runtime, _KEY, None)
    if previous is not None and key != previous:
        setattr(runtime, _KEY, None)
        _emit(runtime, REFUSED_EVENT, reason="eligibility_revoked", terminal=False,
              revision=previous[0], candidate_digest=previous[1],
              new_revision=None if key is None else key[0])
    if key is None:
        return False
    setattr(runtime, _KEY, key)
    attempted = getattr(runtime, _ATTEMPTED, None)
    if not isinstance(attempted, set):
        attempted = set()
        setattr(runtime, _ATTEMPTED, attempted)
    if key in attempted:
        return False
    attempted.add(key)
    from kernel_green_then_exhaust import submit_if_ready
    completion = submit_if_ready(runtime, boundary=BOUNDARY)
    fields = {"revision": key[0], "candidate_digest": key[1]}
    if bool(getattr(completion, "accepted", False)):
        _emit(runtime, COMMITTED_EVENT, submission_initiator="host", rendered=True, **fields)
        return True
    _emit(runtime, REFUSED_EVENT, terminal=False,
          reason="not_submittable" if completion is None else "completion_refused",
          reason_codes=_codes(completion), **fields)
    return False


__all__ = [
    "BOUNDARY", "COMMITTED_EVENT", "DEFERRED_EVENT", "MANIFEST_KEY", "MODE_ENV", "REFUSED_EVENT",
    "at_request_boundary", "certified_key", "enabled", "mode",
]
