"""Classify provider calls before fitting, independently of draft wire text.

Primary requests require the complete-envelope seal. Auxiliary calls have a
bounded, non-workspace response contract; a missing capture is never an
auxiliary classification. Legacy callers remain available outside FrameFit5.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass

from kernel_request_reprepare import RequestPreparationContractError

PRIMARY = "framefit5_primary"
AUXILIARY = frozenset({"self_plan", "turtle", "summary", "public_review", "bind_quantity", "decide2"})


@dataclass(frozen=True, slots=True)
class RequestProtocol:
    kind: str


class RequestKindError(RequestPreparationContractError):
    def __init__(self, code: str):
        self.stop_code = code
        super().__init__(code)


def framefit_expected(runtime=None) -> bool:
    return (getattr(runtime, "stateframe_framefit_mode", "off") == "on"
            or os.environ.get("STATEFRAME_FRAMEFIT", "off").strip().lower() == "on")


def bind_primary(capture, runtime) -> None:
    previous = getattr(capture, "request_kind", None)
    protocol = getattr(capture, "_request_protocol", None)
    if previous not in (None, PRIMARY) or (protocol is not None and protocol.kind != PRIMARY):
        raise RequestKindError("primary_request_kind_conflict")
    capture.request_kind = PRIMARY
    capture._request_protocol = RequestProtocol(PRIMARY)
    capture.completion_owner_runtime = runtime


def bind_auxiliary(capture, purpose: str) -> None:
    protocol = getattr(capture, "_request_protocol", None)
    if (purpose not in AUXILIARY or getattr(capture, "request_kind", None) not in (None, purpose)
            or (protocol is not None and protocol.kind != purpose)
            or getattr(capture, "request_action_contract", None) is not None
            or getattr(capture, "qualified_request", None) is not None):
        raise RequestKindError("auxiliary_request_kind_conflict")
    capture.request_kind = purpose
    capture._request_protocol = RequestProtocol(purpose)


def is_auxiliary(capture) -> bool:
    return (getattr(capture, "request_kind", None) in AUXILIARY
            or getattr(getattr(capture, "_request_protocol", None), "kind", None) in AUXILIARY)


@contextmanager
def isolated_auxiliary_runtime(capture):
    """Retain generic budget checks without borrowing the parent's owner/source state."""
    if not is_auxiliary(capture):
        yield
        return
    from kernel_pilot import ACTIVE_RUNTIME
    from kernel_measurement import ACTIVE_REQUEST
    runtime_token = ACTIVE_RUNTIME.set(None)
    request_token = ACTIVE_REQUEST.set(capture)
    try:
        yield
    finally:
        ACTIVE_REQUEST.reset(request_token)
        ACTIVE_RUNTIME.reset(runtime_token)


def validate_auxiliary_response(capture, choice) -> None:
    if not is_auxiliary(capture):
        return
    message = choice.get("message", {}) if isinstance(choice, Mapping) else {}
    if isinstance(message, Mapping) and message.get("function_call"):
        raise RequestKindError("auxiliary_workspace_response_forbidden")
    calls = message.get("tool_calls", ()) if isinstance(message, Mapping) else ()
    if calls and (getattr(capture, "request_kind", None) != "decide2" or any(
            not isinstance(call, Mapping)
            or not isinstance(call.get("function"), Mapping)
            or call["function"].get("name") != "decide2_answer" for call in calls)):
        raise RequestKindError("auxiliary_workspace_response_forbidden")


def require_capture(capture, runtime=None) -> None:
    if capture is None and framefit_expected(runtime):
        raise RequestKindError("primary_capture_missing")


def classify(capture, runtime, body: Mapping[str, object], *, side_call_contract=None) -> bool:
    """Return primary status, refusing missing prerequisites before any writers."""
    from kernel_stateframe_framefit5 import has_framefit5_state

    require_capture(capture, runtime)
    kind = getattr(capture, "request_kind", None)
    protocol = getattr(capture, "_request_protocol", None)
    if protocol is not None and protocol.kind != kind:
        raise RequestKindError("request_kind_binding_changed")
    if kind in AUXILIARY:
        if not isinstance(protocol, RequestProtocol) or protocol.kind != kind:
            raise RequestKindError("auxiliary_protocol_missing")
        if getattr(capture, "qualified_request", None) is not None:
            raise RequestKindError("auxiliary_cannot_reuse_primary_seal")
        if kind == "decide2":
            from kernel_v46_mechanism_wiring import Decide2SideCallContract
            if not isinstance(side_call_contract, Decide2SideCallContract):
                raise RequestKindError("auxiliary_decide2_contract_missing")
            side_call_contract.validate(body)
        elif side_call_contract is not None or (body.get("tools") and not (
                kind == "self_plan" and body.get("tool_choice") == "none")):
            raise RequestKindError("auxiliary_workspace_tools_forbidden")
        return False
    frame = has_framefit5_state(body.get("messages", []))
    primary = (kind == PRIMARY or framefit_expected(runtime) or frame
               or getattr(capture, "qualified_request", None) is not None)
    if not primary:
        return False
    if capture is None:
        raise RequestKindError("primary_capture_missing")
    # Sticky across lost draft flags and worker threads with empty ContextVars.
    capture.request_kind = PRIMARY
    capture._request_protocol = RequestProtocol(PRIMARY)
    if runtime is None:
        raise RequestKindError("primary_runtime_missing")
    owner = getattr(capture, "completion_owner_runtime", None)
    if owner is not None and owner is not runtime:
        raise RequestKindError("primary_runtime_binding_changed")
    if side_call_contract is not None:
        raise RequestKindError("primary_cannot_use_auxiliary_contract")
    if not frame:
        raise RequestKindError("primary_frame_missing")
    return True


def require_adapter(capture, runtime, messages, *, gate, host_preparer, adapter: str) -> bool:
    if capture is None:
        raise RequestKindError("primary_capture_missing")
    primary = classify(capture, runtime, {"messages": messages})
    if primary:
        capture._qualified_adapter = adapter
        capture._qualified_request_gate = gate
        capture._qualified_host_text_preparer = host_preparer
        validate_adapter(capture)
    return primary


def validate_adapter(capture) -> None:
    if getattr(capture, "_qualified_adapter", None) is not None:
        if not callable(getattr(capture, "_qualified_request_gate", None)):
            raise RequestKindError("required_host_gate_missing")
        if not callable(getattr(capture, "_qualified_host_text_preparer", None)):
            raise RequestKindError("required_host_preparer_missing")


def gate_required(capture) -> bool:
    return (getattr(capture, "_qualified_adapter", None) is not None
            or getattr(capture, "_qualified_request_gate", None) is not None)


def guard_physical_send(capture, runtime, body, wire) -> None:
    if not classify(capture, runtime, body,
                    side_call_contract=getattr(capture, "_wire_truth_side_call_contract", None)):
        return
    validate_adapter(capture)
    record = getattr(capture, "qualified_request", None)
    if record is None:
        raise RequestKindError("primary_seal_missing")
    if not getattr(capture, "_qualified_adapter", None):
        raise RequestKindError("primary_adapter_missing")
    record.validate(capture, runtime, body)
    if record.wire != wire or not getattr(capture, "_qualified_request_gate_passed", False):
        raise RequestKindError("primary_gate_binding_missing")


def reject_changed_body_retry(capture) -> None:
    if getattr(capture, "request_kind", None) == PRIMARY or getattr(capture, "qualified_request", None) is not None:
        capture.journal.emit("model_retry", request_id=capture.request_id,
                             reason="primary_changed_body_requires_successor",
                             retry_scheduled=False, tool_dispatched=False)
        raise RequestKindError("primary_changed_body_requires_successor")


def reject_ambiguous_timeout(future, *, capture=None) -> None:
    # The unclassified legacy transport has a separate timeout contract.
    # Every FrameFit primary and explicitly bound auxiliary uses this guard.
    if (getattr(capture, "_request_protocol", None) is None
            and not framefit_expected(getattr(capture, "completion_owner_runtime", None))
            and getattr(capture, "qualified_request", None) is None):
        return
    if future.done() and not future.cancelled() and future.exception() is not None:
        # The worker itself raised a settled transport timeout. Existing retry
        # policy may handle it; no thread still owns this attempt.
        return
    code = ("provider_attempt_still_in_flight" if not future.done()
            else "provider_attempt_completed_after_timeout")
    journal = getattr(capture, "journal", None)
    if journal is not None:
        journal.emit("provider_attempt_outcome_unknown", request_id=capture.request_id,
                     retry_scheduled=False, failure_origin="harness", reason=code, dispatch=False)
    raise RequestKindError(code)


def reject_unprepared_branch(runtime) -> None:
    if framefit_expected(runtime):
        raise RequestKindError("branch_bound_preparation_unsupported")
