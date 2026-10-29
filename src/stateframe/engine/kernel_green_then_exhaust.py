"""Host-owned completion at a public-green response-budget boundary.

The response budget is charged before a provider call.  A final prepared frame
can therefore contain enough host evidence for completion while the next model
response is no longer affordable.  This module gives that boundary one
switch-gated, fail-closed path through the existing completion authority.
"""
from __future__ import annotations

import copy
import os
from collections.abc import Callable, Mapping, Sequence
from kernel_done_acceptance import done_accepted


MODE_ENV = "STATEFRAME_GREEN_THEN_EXHAUST"
# r46 f3 #6: a stall-submit at a stale (reused, never re-run) verification or with an open
# advisory counterexample can accept a hidden-red candidate as done.  Gated, default off,
# scoped to the post_green_stall boundary only; the response_budget boundary is unchanged.
STALL_STRICT_ENV = "STATEFRAME_STALL_SUBMIT_STRICT"
SCHEMA = "stateframe-green-then-exhaust/1"
PRODUCER = "kernel_green_then_exhaust"
SUBMISSION_REASON = "RESPONSE_BUDGET_EXHAUSTED_READY"
STALL_SUBMISSION_REASON = "POST_GREEN_STALL_READY"   # r45 in-flight fix 3 (STATEFRAME_INFLIGHT_FIXES)
# These two r47 boundaries feed this same submitter; their own modules own only
# eligibility/counting and telemetry, never terminal rendering or commitment.
POSTGREEN_BUDGET_SUBMISSION_REASON = "POSTGREEN_BUDGET_READY"
READINESS_SUBMISSION_REASON = "READINESS_CERTIFIED_READY"
BOUNDARIES = ("response_budget", "post_green_stall", "postgreen_budget", "readiness_submit")
BOUNDARY_NOTICE = (
    "[StateFrame] Response budget boundary: the current public-green candidate "
    "is host-ready; completion authority will submit the preserved artifact if "
    "its final gates remain admissible."
)


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = str(source.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(runtime: object | None = None) -> bool:
    configured = getattr(runtime, "stateframe_green_then_exhaust_mode", None)
    if configured is not None:
        return str(configured).strip().lower() == "on"
    return mode() == "on"


def stall_strict_mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = str(source.get(STALL_STRICT_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{STALL_STRICT_ENV} must be off or on")
    return value


def stall_strict_enabled(runtime: object | None = None) -> bool:
    configured = getattr(runtime, "stateframe_stall_submit_strict_mode", None)
    if configured is not None:
        return str(configured).strip().lower() == "on"
    return stall_strict_mode() == "on"


def _stall_evidence_stale(runtime: object) -> bool:
    """True when the current binding's public-check evidence was reused (REUSED_VERIFICATION),
    never freshly re-run: the stall's "green" candidate is unverified at this exact binding."""
    import kernel_inflight_fixes as inflight
    if not inflight.enabled():
        # No reuse tracking exists without the fix that records it; fail closed by treating the
        # evidence as unverifiable rather than silently skipping the strict gate.
        return True
    key = inflight.verification_key(runtime)
    return inflight.reused_repeatedly(runtime, key)


def _stall_open_counterexample(decision: object) -> bool:
    """True when any counterexample row remains open at this decision -- blocking or advisory.
    ``repairable_advisories`` alone would miss a non-repairable advisory that is still open."""
    advisory = getattr(decision, "advisory", ())
    return bool(advisory)


def _authority_enabled(runtime: object) -> bool:
    return getattr(runtime, "stateframe_readiness_authority_mode", "off") == "on"


def _readiness(runtime: object, *, boundary: str = "response_budget"):
    if boundary == "postgreen_budget":
        from kernel_postgreen_budget import enabled as postgreen_budget_enabled
        switched_on = postgreen_budget_enabled(runtime)
    elif boundary == "readiness_submit":
        from kernel_submit_at_readiness import enabled as submit_at_readiness_enabled
        switched_on = submit_at_readiness_enabled(runtime)
    else:
        switched_on = enabled(runtime)
    if not switched_on or not _authority_enabled(runtime):
        return None
    if done_accepted(runtime):
        return None
    if getattr(runtime, "_green_then_exhaust_attempted", False):
        return None
    method = getattr(runtime, "readiness_decision", None)
    if not callable(method):
        return None
    try:
        return method()
    except (AttributeError, TypeError, ValueError):
        # A malformed or unavailable readiness snapshot is never permission to
        # submit a candidate at a hard budget boundary.
        return None


def _decision_payload(decision: object) -> dict[str, object]:
    converter = getattr(decision, "to_dict", None)
    if callable(converter):
        try:
            value = converter()
        except (AttributeError, TypeError, ValueError):
            value = None
        if isinstance(value, Mapping):
            return copy.deepcopy(dict(value))
    return {
        "state": str(getattr(decision, "state", "UNKNOWN")),
        "ready": bool(getattr(decision, "ready", False)),
        "ready_to_submit": bool(getattr(decision, "ready_to_submit", False)),
        "host_submittable": _host_submittable(decision),
    }


def _host_submittable(decision: object) -> bool:
    """Use the readiness authority's actual commit predicate, fail-closed."""
    if not bool(getattr(decision, "ready", False)):
        return False
    value = getattr(decision, "host_submittable", None)
    if value is not None:
        if not bool(value):
            return False
        done_allowed = getattr(decision, "done_allowed", None)
        return done_allowed is True
    # Compatibility with pre-split decision doubles; the real authority always
    # supplies host_submittable and done_allowed.
    return bool(getattr(decision, "ready_to_submit", False))


def prepare_notice(runtime: object) -> str | None:
    """Return the final-frame notice only for a current host-submittable snapshot."""
    decision = _readiness(runtime)
    if not _continuation_ready(runtime):
        return None
    if decision is None or not _host_submittable(decision):
        return None
    return BOUNDARY_NOTICE


def _record(runtime: object, kind: str, **fields: object) -> None:
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if not callable(emit):
        return
    payload = {"producer": PRODUCER, "payload_schema": SCHEMA, "dispatch": False, **fields}
    capture = getattr(runtime, "_active_capture", None)
    request_id = getattr(capture, "request_id", None)
    if request_id:
        payload["request_id"] = request_id
    try:
        emit(kind, **payload)
    except Exception:  # noqa: BLE001 - telemetry cannot change boundary behavior
        return


def append_prepare_notice(
        runtime: object,
        messages: Sequence[dict],
        notice: str,
        *,
        fit: Callable[[Sequence[dict]], int] | None = None,
        ceiling: int | None = None,
) -> tuple[list[dict], bool]:
    """Append the boundary notice without changing an already fitting request."""
    prepared = [copy.deepcopy(dict(message)) for message in messages]
    setattr(runtime, "_green_then_exhaust_rendered", False)
    if not enabled(runtime) or not str(notice).strip():
        return prepared, False
    if any(str(message.get("content", "")) == notice
           for message in prepared if isinstance(message, Mapping)):
        setattr(runtime, "_green_then_exhaust_rendered", True)
        return prepared, True
    block = {"role": "user", "content": str(notice)}
    state_index = next(
        (index for index, message in enumerate(prepared)
         if isinstance(message, Mapping)
         and str(message.get("content", "")).startswith("[StateFrame]\n")),
        len(prepared),
    )
    candidate = [*prepared[:state_index], block, *prepared[state_index:]]
    if fit is not None and ceiling is not None:
        try:
            estimate = int(fit(candidate))
        except Exception:  # noqa: BLE001 - a fit failure must fail closed
            _record(runtime, "green_then_exhaust_boundary_omitted", rendered=False,
                    reason="fit_error")
            return prepared, False
        if estimate > int(ceiling):
            _record(runtime, "green_then_exhaust_boundary_omitted", rendered=False,
                    reason="ceiling", estimated_tokens=estimate,
                    available_tokens=int(ceiling))
            return prepared, False
    _record(runtime, "green_then_exhaust_boundary_rendered", rendered=True)
    setattr(runtime, "_green_then_exhaust_rendered", True)
    return candidate, True


def _selectors(runtime: object) -> tuple[str, ...]:
    provider = getattr(runtime, "required_public_selectors", None)
    if not callable(provider):
        return ()
    try:
        values = provider()
    except (AttributeError, TypeError, ValueError):
        return ()
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return ()
    return tuple(str(value) for value in values if str(value).strip())


def _claims(runtime: object) -> object:
    provider = getattr(runtime, "done_facts_claims", None)
    if not callable(provider):
        return None
    try:
        return provider()
    except (AttributeError, TypeError, ValueError):
        return None


def _completion_commit(runtime: object) -> Callable[[], object] | None:
    continuation = getattr(runtime, "continuation", None)
    if continuation is None:
        return None
    owner = getattr(runtime, "owner", None)
    verdict_provider = getattr(owner, "completion_verdict", None)
    if not callable(verdict_provider):
        raise AttributeError("continuation completion verdict unavailable")
    verdict = verdict_provider()
    host_dispatch = getattr(continuation, "host_done_dispatch", None)
    if callable(host_dispatch):
        host_dispatch()
        setattr(runtime, "_green_then_exhaust_host_dispatch_staged", True)
    elif not (isinstance(getattr(continuation, "dispatch", None), Mapping)
              and continuation.dispatch.get("tool") == "done"):
        raise AttributeError("continuation done fence unavailable")
    if getattr(runtime, "stateframe_submit_idempotent_mode", "off") == "on":
        destination = getattr(continuation, "keyed_destination", None)
        if not callable(destination):
            raise AttributeError("continuation keyed destination unavailable")
        return destination(verdict)
    accept_done = getattr(continuation, "accept_done", None)
    if not callable(accept_done):
        raise AttributeError("continuation completion commit unavailable")
    return lambda: accept_done(verdict)


def _continuation_ready(runtime: object) -> bool:
    continuation = getattr(runtime, "continuation", None)
    if continuation is None:
        return True
    finding = getattr(continuation, "completion_finding", None)
    if callable(finding):
        try:
            return finding() is None
        except (AttributeError, TypeError, ValueError):
            return False
    blocker = getattr(continuation, "completion_blocker", None)
    if callable(blocker):
        try:
            return blocker() is None
        except (AttributeError, TypeError, ValueError):
            return False
    return False


def _path(runtime: object) -> str | None:
    target = getattr(runtime, "target", None)
    if target is None:
        return None
    try:
        root = getattr(getattr(runtime, "owner", None), "root", None)
        if root is not None:
            return str(target.resolve().relative_to(root.resolve()))
        return str(target)
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def submit_if_ready(runtime: object, *, boundary: str = "response_budget"):
    """Submit the preserved candidate through the normal completion reducer.

    ``None`` means the boundary was not eligible or the final authority was
    unavailable.  A rejected decision is returned and is intentionally not
    converted into a system outcome: the caller must retain the ordinary
    budget-exhausted failure surface.
    """
    if boundary not in BOUNDARIES:
        raise ValueError(f"unknown green-then-exhaust boundary {boundary!r}")
    stall = boundary == "post_green_stall"
    budget = boundary == "postgreen_budget"
    readiness = boundary == "readiness_submit"
    own_switch = budget or readiness
    from kernel_done_is_submit_authority import (
        enabled as done_submit_enabled,
        request_submission,
        wait_for_pending_mutations,
    )
    if done_submit_enabled(runtime):
        # DESIGN RULE: done comes from the MODEL. DONE_IS_SUBMIT only means the host never
        # REFUSES a model done; it must never let the host submit or stop on its own. This branch used to call
        # request_submission(source="host") at every boundary with no readiness check, so the readiness_submit
        # boundary submitted at step 0 BEFORE request 0 (r2EY DONE-env strict smoke: requests=0, host_stopped,
        # SUBMITTED, product_files_changed=0). Under DONE_IS_SUBMIT host-initiated submission is off entirely.
        _record(runtime, "green_then_exhaust_host_submit_suppressed", rendered=False,
                boundary=boundary, done_is_submit=True)
        return None
    del request_submission, wait_for_pending_mutations
    if enabled(runtime) and not own_switch:
        # r46: own event for every call while the switch is on, regardless of outcome
        # (kit/fired_check_r45.py SITUATIONAL note: this OWN_EVENT had no emitter).
        _record(runtime, "green_then_exhaust_evaluated", rendered=False, boundary=boundary)
    # The response-budget boundary requires its rendered notice; a post-green stall is the runner's own terminal
    # (no notice exists), so only readiness + continuation gate it.
    if not stall and not own_switch and not bool(getattr(runtime, "_green_then_exhaust_rendered", True)):
        return None
    decision = _readiness(runtime, boundary=boundary)
    if (decision is None
            or not _host_submittable(decision)
            or not _continuation_ready(runtime)):
        return None
    if stall and stall_strict_enabled(runtime):
        _record(runtime, "stall_submit_strict_evaluated", rendered=False, stage="evaluated")
        if _stall_evidence_stale(runtime):
            _record(runtime, "green_then_exhaust_not_submitted", rendered=False,
                    reason="stall_submit_strict_reused_evidence")
            return None
        if _stall_open_counterexample(decision):
            _record(runtime, "green_then_exhaust_not_submitted", rendered=False,
                    reason="stall_submit_strict_open_counterexample")
            return None
    setattr(runtime, "_green_then_exhaust_attempted", True)
    setattr(runtime, "_green_then_exhaust_host_dispatch_staged", False)
    selectors = _selectors(runtime)
    readiness_payload = _decision_payload(decision)
    if readiness:
        reason_code = READINESS_SUBMISSION_REASON
        detail = ("public checks were green and readiness was certified submit-ready at the "
                  "current revision before the next provider request")
        evidence = {"readiness_submit": True, "readiness": readiness_payload}
    elif budget:
        reason_code = POSTGREEN_BUDGET_SUBMISSION_REASON
        detail = ("public checks were green and readiness was submit-ready when "
                  "the post-green response budget was spent")
        evidence = {"postgreen_budget": True, "readiness": readiness_payload}
    elif stall:
        reason_code = STALL_SUBMISSION_REASON
        detail = ("public checks were green and readiness was submit-ready when "
                  "the post-green verification loop stalled")
        evidence = {"post_green_stall": True, "readiness": readiness_payload}
    else:
        reason_code = SUBMISSION_REASON
        detail = ("public checks were green and readiness was submit-ready when "
                  "the response budget became exhausted")
        evidence = {"response_budget_exhausted": True, "readiness": readiness_payload}
    try:
        completion = runtime.completion_decision(
            admissible=True,
            detail=detail,
            reason_codes=(reason_code,),
            selectors=selectors,
            path=_path(runtime),
            evidence=evidence,
            claims=_claims(runtime),
            artifact_preserved=True,
            commit=_completion_commit(runtime),
        )
    except Exception as error:  # noqa: BLE001 - fail closed at the hard boundary
        if getattr(runtime, "_green_then_exhaust_host_dispatch_staged", False):
            reset_action = getattr(getattr(runtime, "continuation", None), "reset_action", None)
            if callable(reset_action):
                reset_action()
        setattr(runtime, "_green_then_exhaust_attempted", False)
        setattr(runtime, "_green_then_exhaust_host_dispatch_staged", False)
        setattr(runtime, "_green_then_exhaust_boundary_active", False)
        _record(runtime, "green_then_exhaust_not_submitted", rendered=False,
                reason="completion_authority_error", error_type=type(error).__name__)
        return None
    if not bool(getattr(completion, "accepted", False)):
        if getattr(runtime, "_green_then_exhaust_host_dispatch_staged", False):
            reset_action = getattr(getattr(runtime, "continuation", None), "reset_action", None)
            if callable(reset_action):
                reset_action()
        setattr(runtime, "_green_then_exhaust_attempted", False)
        setattr(runtime, "_green_then_exhaust_host_dispatch_staged", False)
        setattr(runtime, "_green_then_exhaust_boundary_active", False)
        _record(runtime, "green_then_exhaust_not_submitted", rendered=False,
                reason="completion_refused", decision=_decision_payload(completion))
        return completion
    runtime.completion_accepted = True
    runtime.system_outcome = None
    setattr(runtime, "_tool_call_loop_breaker_ready", False)
    setattr(runtime, "_green_then_exhaust_boundary_active", False)
    _record(runtime, "green_then_exhaust_submission", rendered=True,
            submission_initiator="host", reason=reason_code,
            boundary=boundary,
            decision=_decision_payload(completion))
    return completion


__all__ = [
    "BOUNDARIES", "BOUNDARY_NOTICE", "MODE_ENV", "POSTGREEN_BUDGET_SUBMISSION_REASON",
    "READINESS_SUBMISSION_REASON", "STALL_STRICT_ENV", "STALL_SUBMISSION_REASON", "PRODUCER",
    "SCHEMA", "SUBMISSION_REASON",
    "append_prepare_notice", "enabled", "mode", "prepare_notice", "stall_strict_enabled", "stall_strict_mode",
    "submit_if_ready",
]
