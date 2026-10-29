"""Bounded, owner-authorized recovery for provider-bound request contracts."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import threading
from collections.abc import Mapping, Sequence


ENV = "STATEFRAME_BOUNDARY_RECOVERY"
# live r2EF benchmark (batch 1322) recovered 8/8 then died on this cap alone; recovery is already
# limited to one per request, so the run cap is only a runaway guard. Env-tunable, default 64.
RUN_CAP_ENV = "STATEFRAME_BOUNDARY_RECOVERY_RUN_CAP"
DEFAULT_RECOVERIES_PER_RUN = 64


def max_recoveries_per_run() -> int:
    raw = os.environ.get(RUN_CAP_ENV, str(DEFAULT_RECOVERIES_PER_RUN)).strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{RUN_CAP_ENV} must be a positive integer") from error
    if value < 1:
        raise ValueError(f"{RUN_CAP_ENV} must be a positive integer")
    return value


MAX_RECOVERIES_PER_RUN = DEFAULT_RECOVERIES_PER_RUN  # back-compat name; live cap is max_recoveries_per_run()
CAPABILITY_FLOOR = frozenset({
    "read", "grep", "edit", "str_replace", "run_tests", "authored_check",
    "read_check_receipt", "done",
})
_LOCK = threading.RLock()

_SAFE_ERROR_MESSAGES = frozenset({
    # Complete-envelope preparation/allocation failures are fixed host text.
    # Keep source paths and arbitrary exception metadata outside this authority.
    "final FrameFit5 recovery lost its shared frame state",
    "replacement did not publish a qualified request",
    "request preparation has an invalid frame request allocation",
    "replacement deadline must be finite",
    "immutable request preparation has no message list",
    "request preparation frame allocation failed integrity validation",
    "request frame allocation changed during recovery",
    "replacement qualification requires the provider endpoint",
    "cannot persist unverified complete source plan",
    "immutable request preparation body is unavailable",
    "qualified request changed during recovery",
    "captured frame allocation failed integrity validation",
    "final source delivery failed after frame-only wire ordering",
    # Added at r2EV merge: SUBMIT's fixed deferred-source boundary messages (no paths/secrets).
    "final boundary body retained an unrouted deferred-source notice",
    "final boundary menu cannot dispatch deferred source reads",
    "hard host completion violation has no callable current recovery",
    "refused done has no callable run_tests/edit recovery",
    "green readiness lacks a receipt and callable run_tests recovery",
    "verified source delivery omitted its deferred path set",
    "verified source delivery changed the request's deferred path set",
    "final request source delivery is not wire-verified",
    "final request source delivery remained unqualified after bounded recovery",
    "final request surface failed authoritative validation",
    "request pre-wire preparation failed before provider dispatch",
    "wire-truth validation remained blocked without an eligible replacement",
    "bounded wire-truth recovery could not prepare a qualified replacement",
    "replacement preparation did not commit the active capture",
    "final request preparation failed after bounded recovery",
    "final authority changed qualified serialized request",
    "final request wire remained blocked after bounded recovery",
    "request preparation identity changed before send",
    "request preparation binding changed before send",
    "request preparation binding changed during recovery",
    "original source closure changed during recovery",
    "boundary recovery could not render owner evidence pin",
    "replacement requires the original active unsent preparation",
    "replacement owner identity changed",
    "boundary recovery has no frozen owner contract",
    "boundary recovery owner contract identity changed",
    "replacement input lacks prepared messages/tools",
    "boundary source deferral produced invalid messages",
    "boundary source deferral did not retain its verified paths",
    "boundary source deferral lost its dispatchable read route",
    "boundary capability floor is not a tool list",
    "fresh owner contract removed the read route for deferred sources",
    "wire truth retry did not pass strict qualification",
    "wire truth validation rejected the prepared request",
    "serialized request failed wire string hygiene",
    "transport body differs from qualified serialized request",
    "retry authority changed qualified serialized request",
})
_SAFE_FAILURE_CODES = frozenset({
    "source_deferred_path_set_omitted",
    "source_deferred_path_set_mismatch",
    "source_delivery_unqualified",
    "request_pre_wire_preparation_failed",
    "wire_truth_recovery_unavailable",
    "wire_truth_recovery_failed",
    "wire_truth_recovery_uncommitted",
    "final_request_preparation_failed",
    "final_request_bytes_changed",
    "final_request_surface_unqualified",
    "final_wire_truth_unqualified",
    "final_request_source_unqualified",
    "request_bytes_exceed_max_request_bytes",
    "deferred_source_notice_unverified",
    "deferred_source_path_unselected",
    "workspace_authorization_unverified",
    "final_wire_proof_missing",
    "final_wire_proof_incomplete",
    "delivery_proof_bytes",
    "delivery_proof_handle",
    "delivery_range_invalid",
    "final_message_location_unavailable",
    "source_explicit_absence",
    "source_revision_mismatch",
    "source_projection_bytes_mismatch",
    "source_bytes_unavailable",
    "source_window_bytes_mismatch",
    "source_body_blank",
    "candidate_proof_unqualified",
    "request_receipt_reentry",
    "replacement_capture_reentry",
    "current_body_digest_mismatch",
    "transport_body_wire_mismatch",
    "request_contract_violation",
    "request_boundary_contract_violation",
    "boundary_unrecoverable",
    "wire_string_hygiene_violation",
    "retry_authority_wire_mismatch",
    "unclassified_failure",
    "RequestPreparationContractError",
    "RequestActionContractError",
    "ProviderBoundaryContractError",
    "WireTruthBlocked",
    "BoundaryRecoveryMenuConflict",
    "greenlock_menu_excludes_required_read",
    "owner_menu_excludes_required_read",
    "dropped_source_claim_still_visible",
})
_SAFE_ERROR_CLASSES = frozenset({
    "RequestPreparationContractError",
    "RequestActionContractError",
    "ProviderBoundaryContractError",
    "RuntimeError",
    "WireTruthBlocked",
    "BoundaryRecoveryMenuConflict",
    "BoundaryRecoveryStop",
})
_SAFE_OBLIGATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}$")


def _safe_failure_code(value: object) -> str:
    candidate = value.get("kind", value.get("code", "")) if isinstance(value, Mapping) else value
    if not isinstance(candidate, str):
        return "unclassified_failure"
    candidate = candidate.strip().split(":", 1)[0]
    return candidate if candidate in _SAFE_FAILURE_CODES else "unclassified_failure"


def safe_failure_codes(values: object, *, fallback: object = ()) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)) or not values:
        values = fallback
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        values = ()
    safe = list(dict.fromkeys(_safe_failure_code(item) for item in values[:16]))
    return safe or ["unclassified_failure"]


def safe_proof_failure_kinds(value: object) -> list[str]:
    if isinstance(value, Mapping):
        value = value.get("proof_failures", ())
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    kinds = []
    for item in value[:32]:
        kind = _safe_failure_code(item)
        if kind != "unclassified_failure" and kind not in kinds:
            kinds.append(kind)
    return kinds


def safe_obligation_ids(value: object, *, limit: int = 16) -> list[str]:
    """Keep bounded opaque IDs while excluding paths and free-form source text."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    bounded = max(0, min(int(limit), 32))
    result: list[str] = []
    for item in value[:bounded]:
        if not isinstance(item, str):
            continue
        candidate = item.strip()
        if candidate and _SAFE_OBLIGATION_ID.fullmatch(candidate) and candidate not in result:
            result.append(candidate)
    return result


def safe_error_class(error: BaseException) -> str:
    error_class = type(error).__name__
    return error_class if error_class in _SAFE_ERROR_CLASSES else "ContractError"


def safe_error_detail(error: BaseException) -> str:
    message = " ".join(str(error).split())
    if message in _SAFE_ERROR_MESSAGES:
        return message[:320]
    error_class = safe_error_class(error)
    return f"{error_class} (message redacted)"


class BoundaryRecoveryStop(RuntimeError):
    """A request contract could not be recovered; the worker must stop normally."""

    boundary_recovery_stop = True
    stop_code = "boundary_unrecoverable"

    def __init__(self, detail: str = "request boundary recovery was unavailable") -> None:
        super().__init__(detail)


class BoundaryRecoveryMenuConflict(RuntimeError):
    """The active request menu cannot safely carry a required read deferral."""

    harness_contract = True
    outcome = "HARNESS_CONTRACT"
    failure_origin = "harness"

    def __init__(self, detail: str, *, code: str) -> None:
        super().__init__(detail)
        self.code = code
        self.stop_code = code


def enabled() -> bool:
    value = os.environ.get(ENV, "on").strip().lower()
    if value not in {"on", "off"}:
        raise ValueError(f"{ENV} must be on or off")
    return value == "on"


def _state(journal: object) -> dict[str, object]:
    current = getattr(journal, "_stateframe_boundary_recovery", None)
    if current is not None:
        if (isinstance(current, dict)
                and isinstance(current.get("request_ids"), set)
                and type(current.get("count")) is int):
            return current
        return {}
    current = {"count": 0, "request_ids": set()}
    try:
        setattr(journal, "_stateframe_boundary_recovery", current)
    except (AttributeError, TypeError):
        # Journal implementations used by the runtime are mutable. An immutable
        # journal cannot safely provide the run-wide cap, so fail closed below.
        return {}
    return current


def reserve(journal: object, request_id: str) -> tuple[bool, str]:
    """Atomically claim at most one recovery per request and max_recoveries_per_run() per run."""
    with _LOCK:
        state = _state(journal)
        request_ids = state.get("request_ids")
        count = state.get("count")
        if not isinstance(request_ids, set) or type(count) is not int:
            return False, "recovery_cap_state_unavailable"
        if request_id in request_ids:
            return False, "request_recovery_already_used"
        if count >= max_recoveries_per_run():
            return False, "run_recovery_limit_reached"
        request_ids.add(request_id)
        state["count"] = count + 1
        return True, "reserved"


def digest_bytes(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def error_fields(error: BaseException, *, phase: str) -> dict[str, object]:
    failures = getattr(error, "failures", getattr(error, "failed", ()))
    failure_codes = safe_failure_codes(failures, fallback=(type(error).__name__,))
    code = safe_failure_codes((
        getattr(error, "stop_code", getattr(error, "code", "request_contract_violation")),
    ))[0]
    detail = safe_error_detail(error)
    proof_kinds = safe_proof_failure_kinds(
        getattr(error, "proof_failures", getattr(error, "proof_failure_kinds", ())),
    )
    return {
        "phase": str(phase)[:80],
        "invariant": detail,
        "error_detail": detail,
        "error_class": safe_error_class(error),
        "code": code,
        "failures": failure_codes,
        "proof_failure_kinds": proof_kinds,
    }


def _frozen_greenlock_menu(runtime: object) -> tuple[list[dict[str, object]], list[str]]:
    frozen = getattr(runtime, "_greenlock_frozen_menu", None)
    if not isinstance(frozen, Mapping):
        raise BoundaryRecoveryMenuConflict(
            "active Greenlock authority has no frozen admitted request menu",
            code="greenlock_frozen_menu_unavailable",
        )
    tools = frozen.get("tools")
    allow_set = frozen.get("allow_set")
    if (frozen.get("schema") != "stateframe-greenlock-final-menu/1"
            or not isinstance(tools, list) or not isinstance(allow_set, list)
            or any(not isinstance(name, str) or not name for name in allow_set)
            or len(allow_set) != len(set(allow_set))):
        raise BoundaryRecoveryMenuConflict(
            "active Greenlock frozen request menu is malformed",
            code="greenlock_frozen_menu_malformed",
        )
    from kernel_completion_owner import _tool_name

    names: list[str] = []
    for tool in tools:
        if not isinstance(tool, Mapping) or not _tool_name(tool):
            raise BoundaryRecoveryMenuConflict(
                "active Greenlock frozen request menu contains a malformed tool schema",
                code="greenlock_frozen_menu_malformed",
            )
        names.append(_tool_name(tool))
    if names != allow_set or len(names) != len(set(names)):
        raise BoundaryRecoveryMenuConflict(
            "active Greenlock frozen allow-set does not match its admitted tool schemas",
            code="greenlock_frozen_menu_schema_mismatch",
        )
    payload = {
        "schema": frozen["schema"],
        "allow_set": allow_set,
        "tools": tools,
    }
    try:
        encoded = json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise BoundaryRecoveryMenuConflict(
            "active Greenlock frozen request menu is not serializable",
            code="greenlock_frozen_menu_malformed",
        ) from error
    expected_digest = "sha256:" + hashlib.sha256(encoded).hexdigest()
    if frozen.get("digest") != expected_digest:
        raise BoundaryRecoveryMenuConflict(
            "active Greenlock frozen request menu failed its digest check",
            code="greenlock_frozen_menu_digest_mismatch",
        )
    return copy.deepcopy([dict(tool) for tool in tools]), names


def recovery_request_menu(
        runtime: object, *, require_read: bool = False,
) -> tuple[list[dict[str, object]], str]:
    """Select the exact active menu authority used by request wire validation."""
    from kernel_greenlock_menu_authority import applies

    if applies(getattr(runtime, "green_lock", None)):
        tools, names = _frozen_greenlock_menu(runtime)
        if require_read and "read" not in names:
            raise BoundaryRecoveryMenuConflict(
                "Greenlock frozen menu excludes the read route required by deferred sources",
                code="greenlock_menu_excludes_required_read",
            )
        return tools, "greenlock_frozen_menu"
    try:
        return recovery_capability_floor(runtime, require_read=require_read), "owner_capability_floor"
    except ValueError as error:
        if require_read:
            raise BoundaryRecoveryMenuConflict(
                "owner recovery menu excludes the read route required by deferred sources",
                code="owner_menu_excludes_required_read",
            ) from error
        raise


def recovery_capability_floor(runtime: object, *, require_read: bool = False) -> list[dict[str, object]]:
    """Return registered, currently dispatchable schemas in the recovery floor."""
    from kernel_completion_owner import _action_dispatch_admissible, _tool_name

    full = getattr(runtime, "_completion_owner_full_tools", ())
    if not isinstance(full, Sequence) or isinstance(full, (str, bytes, bytearray)):
        raise ValueError("owner registered tool surface is unavailable")
    floor: list[dict[str, object]] = []
    seen: set[str] = set()
    for tool in full:
        if not isinstance(tool, Mapping):
            continue
        name = _tool_name(tool)
        if name not in CAPABILITY_FLOOR or name in seen:
            continue
        if not _action_dispatch_admissible(runtime, name, {}):
            continue
        floor.append(copy.deepcopy(dict(tool)))
        seen.add(name)
    if not floor:
        raise ValueError("owner capability floor has no dispatchable tools")
    if require_read and "read" not in seen:
        raise ValueError("deferred source recovery requires a dispatchable read tool")
    return floor
