"""Revalidate a frozen owner contract against the actual transport surface."""
from collections.abc import Mapping, Sequence
import copy
import hashlib
import json


def _is_nonstring_sequence(value):
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def budget_verdict_signature(receipt: object) -> tuple[object, ...] | None:
    """Fields that determine whether a frozen request passed final validation."""
    if not isinstance(receipt, Mapping):
        return None
    failures = receipt.get("final_validation_failures", ())
    if (not _is_nonstring_sequence(failures)
            or any(not isinstance(item, str) for item in failures)):
        return None
    source = receipt.get("source_delivery")
    if source is None:
        source_signature = None
    elif isinstance(source, Mapping):
        deferred_paths = source.get("deferred_paths", ())
        if (not _is_nonstring_sequence(deferred_paths)
                or any(not isinstance(path, str) for path in deferred_paths)):
            return None
        # The notice/read-tool flags only decide a verdict for paths actually deferred
        # (see deferred receipt acceptance). With nothing deferred they are vacuous and
        # can flip on byte-identical wire when the live plan discharges a path (b1421).
        has_deferred = bool(deferred_paths)
        source_signature = (
            source.get("status"),
            source.get("proof_status"),
            tuple(deferred_paths),
            source.get("deferred_notice_verified") if has_deferred else None,
            source.get("read_tool_available") if has_deferred else None,
            source.get("transport_identity"),
        )
    else:
        return None
    return receipt.get("status"), tuple(failures), source_signature


def _catalog_gate_blocker_ids(pass_gates):
    if not isinstance(pass_gates, Mapping) or pass_gates.get("status") != "HOST_UNSUPPORTED":
        return None
    for field in ("missing_tools", "invalid_arguments", "dispatch_refused"):
        entries = pass_gates.get(field)
        if not _is_nonstring_sequence(entries) or entries:
            return None
    raw_ids = pass_gates.get("blocker_ids")
    if not _is_nonstring_sequence(raw_ids) or not raw_ids:
        return None
    if any(not isinstance(item, str) or not item.strip() for item in raw_ids):
        return None
    return tuple(dict.fromkeys(item.strip() for item in raw_ids))


def _catalog_resolver_binding(blocker, recovery):
    blocker_id = getattr(blocker, "blocker_id", None)
    resolver = getattr(blocker, "resolver", None)
    tool = resolver.get("tool") if isinstance(resolver, Mapping) else None
    recovery_tool = recovery.get("tool") if isinstance(recovery, Mapping) else None
    if (not isinstance(blocker_id, str) or not blocker_id.strip()
            or getattr(blocker, "code", None) != "CATALOG_UNKNOWN"
            or not isinstance(tool, str) or not tool.strip()
            or recovery_tool != tool):
        return None
    return blocker_id, tool.strip()


def _raw_catalog_resolver_tools(factory, blocker_ids):
    if not callable(factory):
        return None
    try:
        # An empty surface keeps the original GateBlocker resolver metadata.
        vector = factory()
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return None
    blockers = getattr(vector, "blockers", None)
    recoveries = getattr(vector, "recoveries", None)
    unsupported = getattr(vector, "unsupported", None)
    if any(
        not _is_nonstring_sequence(value)
        for value in (blockers, recoveries, unsupported)
    ):
        return None
    if unsupported or not blockers or len(recoveries) != len(blockers):
        return None
    by_id = {}
    resolver_tools = []
    for blocker, recovery in zip(blockers, recoveries):
        binding = _catalog_resolver_binding(blocker, recovery)
        if binding is None:
            return None
        blocker_id, tool = binding
        if blocker_id in by_id:
            return None
        by_id[blocker_id] = blocker
        resolver_tools.append(tool)
    if set(by_id) != set(blocker_ids):
        return None
    return resolver_tools


def _catalog_resolver_off_surface(pass_gates, factory, submitted_names):
    blocker_ids = _catalog_gate_blocker_ids(pass_gates)
    if blocker_ids is None:
        return None
    resolver_tools = _raw_catalog_resolver_tools(factory, blocker_ids)
    if resolver_tools is None:
        return None
    surface = {
        name.strip() for name in submitted_names
        if isinstance(name, str) and name.strip()
    }
    if surface.intersection(resolver_tools):
        return None
    return {
        "reason": "catalog_resolver_off_surface",
        "blocker_ids": list(blocker_ids),
        "resolver_tools": list(dict.fromkeys(resolver_tools)),
    }


def _source_delivery_sources(capture, runtime):
    """Use the request snapshot because the runtime source map is a refreshable cache."""
    preparation = getattr(capture, "_request_preparation_input", None)
    if preparation is None:
        rows = getattr(runtime, "_empirical_frame_sources", {})
        if not isinstance(rows, Mapping):
            return {}
        sources = {
            str(path): row["content"]
            for path, row in rows.items()
            if isinstance(row, Mapping) and isinstance(row.get("content"), str)
        }
    else:
        if (
            getattr(preparation, "request_id", None)
            != getattr(capture, "request_id", None)
        ):
            return {}
        snapshot = getattr(preparation, "sources", None)
        if not isinstance(snapshot, (bytes, bytearray)):
            return {}
        try:
            sources = json.loads(snapshot)
        except (TypeError, UnicodeDecodeError, ValueError):
            return {}
        if not isinstance(sources, Mapping):
            return {}
    from kernel_public_localization import public_path
    return {
        str(path): content
        for path, content in sources.items()
        if isinstance(content, str) and public_path(str(path)) == str(path)
    }


_MISSING = object()


def snapshot_revalidation_inputs(capture, runtime, body):
    """Freeze the non-body BOOKKEEPING inputs of reconcile_frozen_budget at the wire-truth PASS.

    B1403/B1421 class: revalidation of byte-identical wire must not change verdict because state that the
    pre-wire extensions / visibility pass legitimately move afterwards (source plan override or runtime
    closure, runtime source cache, the receipt byte cap) moved. Those are read once here.
    Deliberately NOT frozen: the admitted owner contract (immutable by its own integrity check) and the
    runtime pass-gates vector, which is live safety state; a gate that stops qualifying after wire truth
    must still block (test_b1403_revalidation_unittest.test_state_mutation_still_drifts...).
    Returns None for an unusable body; callers then keep the live behaviour.
    """
    from kernel_wire_truth_gate import schema_digest

    tools = body.get("tools") if isinstance(body, Mapping) else None
    if not isinstance(tools, list):
        return None
    plan = getattr(capture, "_request_source_closure_override", None)
    if not isinstance(plan, Mapping):
        plan = getattr(runtime, "_request_source_closure", None)
    receipt = getattr(capture, "request_budget_final_wire", None)
    return {
        "tools_digest": schema_digest(tools),
        "plan": copy.deepcopy(dict(plan)) if isinstance(plan, Mapping) else None,
        "sources": copy.deepcopy(_source_delivery_sources(capture, runtime)),
        "max_request_bytes": (receipt.get("max_request_bytes", _MISSING)
                              if isinstance(receipt, Mapping) else _MISSING),
    }


def _stateframe_json_failures(body: Mapping[str, object]) -> tuple[str, ...]:
    messages = body.get("messages") if isinstance(body, Mapping) else None
    if not isinstance(messages, list):
        return ()
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.startswith("[StateFrame]\n"):
            continue
        try:
            payload = json.loads(content.split("\n", 1)[1])
        except (TypeError, ValueError, RecursionError):
            return ("stateframe_json_invalid",)
        if not isinstance(payload, dict):
            return ("stateframe_json_invalid",)
    return ()


def reconcile_frozen_budget(capture, runtime, body, *, frozen_inputs=None):
    from kernel_completion_owner import dispatch_tool_call
    from kernel_continuation_journal import journal_safe_fields
    from kernel_pass_gates_contract import validate_final_surface
    from kernel_wire_truth_gate import schema_digest

    wire_tools = body.get("tools", [])
    raw_surface = getattr(capture, "final_tool_surface", None)
    surface = (
        raw_surface
        if isinstance(raw_surface, Mapping)
        and raw_surface.get("schema") == "stateframe-final-tool-surface/1"
        and raw_surface.get("request_id") == str(getattr(capture, "request_id", ""))
        and isinstance(raw_surface.get("tools"), list)
        and isinstance(raw_surface.get("names"), list)
        and isinstance(raw_surface.get("tool_schema_digest"), str)
        else None
    )
    tools = surface.get("tools") if isinstance(surface, Mapping) else wire_tools
    if frozen_inputs is not None and (
            not isinstance(wire_tools, list)
            or schema_digest(wire_tools) != frozen_inputs.get("tools_digest")):
        frozen_inputs = None  # snapshot belongs to another surface: never apply it
    contract = getattr(capture, "request_action_contract", {})
    admitted = contract.get("admitted_tool_schemas") if isinstance(contract, Mapping) else None
    failures = list(_stateframe_json_failures(body))
    if isinstance(surface, Mapping):
        surface_digest = str(surface.get("tool_schema_digest", ""))
        if (not isinstance(wire_tools, list)
                or schema_digest(wire_tools) != surface_digest
                or schema_digest(tools) != surface_digest):
            failures.append("final_tool_surface_body_mismatch")
        if (not isinstance(admitted, list)
                or schema_digest(admitted) != surface_digest):
            failures.append("frozen_owner_schema_mismatch")
    elif (not isinstance(tools, list) or not isinstance(admitted, list)
            or schema_digest(tools) != schema_digest(admitted)):
        failures.append("frozen_owner_schema_mismatch")
    names = (
        [str(name) for name in surface.get("names", ())]
        if isinstance(surface, Mapping) else []
    )
    if not names:
        for tool in tools if isinstance(tools, list) else ():
            if not isinstance(tool, Mapping):
                continue
            function = tool.get("function")
            names.append(str(function.get("name", "")) if isinstance(function, Mapping)
                         else str(tool.get("name", "")))
    receipt = dict(getattr(capture, "request_budget_final_wire", {}) or {})
    receipt.pop("pass_gates_degraded", None)
    if frozen_inputs is not None:
        if frozen_inputs["max_request_bytes"] is _MISSING:
            receipt.pop("max_request_bytes", None)
        else:
            receipt["max_request_bytes"] = frozen_inputs["max_request_bytes"]
    wire = json.dumps(body).encode()
    receipt.update({
        "schema": "stateframe-request-wire/1", "phase": "frozen_transport_revalidation",
        "request_bytes": len(wire),
        "body_sha256": "sha256:" + hashlib.sha256(wire).hexdigest(),
        "schema_bytes": len(json.dumps(tools, ensure_ascii=False, sort_keys=True).encode()),
        "tool_names": tuple(names),
        "tool_schema_digest": (
            str(surface.get("tool_schema_digest"))
            if isinstance(surface, Mapping) else schema_digest(tools)
        ),
        "messages": isinstance(body.get("messages"), list), "tools": isinstance(tools, list),
    })
    # Sole final capacity authority: only reuse a fitted count with immutable
    # evidence for this exact body, provider contract and request generation.
    from kernel_request_budget import (
        ENDPOINT_PROMPT_TOKEN_BUDGET, FREE_STAGE, HARD_MAX_REQUEST_BYTES,
        _lean_request_limits, _request_token_meter,
    )
    from kernel_request_measurement import RequestMeasurement
    meter = _request_token_meter(runtime, capture)
    carried = getattr(capture, "_frame_candidate_measurement", None)
    request_id = str(getattr(capture, "request_id", ""))
    if (isinstance(carried, RequestMeasurement)
            and carried.matches(body, tokenizer_identity=meter.tokenizer_identity(body),
                                request_id=request_id, logical_request_id=meter._request_id or request_id)
            and carried.method == meter.measure_method
            and carried.prompt_tokens is not None):
        tokens, method = carried.prompt_tokens, carried.method
    else:
        tokens, method = meter.measure(body)
    if (method == "calibrated_bytes"
            and getattr(capture, "_frame_candidate_fallback_fit", False) is True):
        from kernel_frame_allocation_capture import measured_request_tokens

        rejection_bound = meter.measure(body)[0]
        tokens = measured_request_tokens(body, meter)
        receipt["fallback_estimate_fit"] = True
        receipt["fallback_rejection_bound"] = rejection_bound
    offline_limit, token_limit = _lean_request_limits(body, FREE_STAGE)
    verified_limit = _lean_request_limits(body, FREE_STAGE, exact_tokenizer=True)[0]
    byte_limit = verified_limit if method in {"tokenize", "verified_byte_bound"} else offline_limit
    previous_limit = receipt.get("max_request_bytes")
    generated_limits = {offline_limit, HARD_MAX_REQUEST_BYTES, verified_limit}
    previous_measurement = receipt.get("measurement")
    if isinstance(previous_measurement, Mapping):
        previous_body = {**body, "max_tokens": previous_measurement.get("output_tokens")}
        generated_limits.update((_lean_request_limits(previous_body, FREE_STAGE)[0],
                                 _lean_request_limits(previous_body, FREE_STAGE, exact_tokenizer=True)[0]))
    if (type(previous_limit) is int and previous_limit >= 0
            and previous_limit not in generated_limits):
        byte_limit = min(byte_limit, previous_limit)
    if not (isinstance(contract, Mapping) and contract.get("contract_kind") == "lean_stable_surface"):
        token_limit = min(token_limit, ENDPOINT_PROMPT_TOKEN_BUDGET)
    fitted_inputs = getattr(capture, "_frame_candidate_inputs", None)
    fitted_ceiling = fitted_inputs.get("token_ceiling") if isinstance(fitted_inputs, Mapping) else None
    if type(fitted_ceiling) is int and fitted_ceiling >= 0:
        token_limit = min(token_limit, fitted_ceiling)
    measurement = meter.bind_measurement(
        body, tokens=tokens, method=method, request_id=request_id,
        max_request_bytes=byte_limit, max_prompt_tokens=token_limit,
    )
    capture._request_measurement = measurement
    receipt.update({**meter.journal_fields(method), **measurement.receipt_fields()})
    if tokens > token_limit:
        failures.append("request_tokens_exceed_max_prompt_tokens")
    max_request_bytes = receipt.get("max_request_bytes")
    if (type(max_request_bytes) is int and max_request_bytes >= 0
            and receipt["request_bytes"] > max_request_bytes):
        failures.append("request_bytes_exceed_max_request_bytes")
    if not receipt["messages"] or not receipt["tools"]:
        failures.append("invalid_provider_surface")

    def dispatchable(name, arguments):
        try:
            dispatch_tool_call(tools, name, arguments)
            return True
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return False

    factory = getattr(runtime, "pass_gates_vector", None)  # live on purpose: safety state
    if callable(factory):
        try:
            gate_vector = factory(tools)
            receipt["pass_gates"] = validate_final_surface(tools, gate_vector, dispatchable=dispatchable)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as error:
            receipt["pass_gates"] = {"status": "HOST_UNSUPPORTED", "error_type": type(error).__name__}
        if receipt["pass_gates"].get("status") != "VALID":
            degraded = _catalog_resolver_off_surface(receipt["pass_gates"], factory, names)
            if degraded is None:
                failures.append("gate_surface_unqualified")
            else:
                receipt["pass_gates_degraded"] = degraded
    if frozen_inputs is not None:
        plan = frozen_inputs["plan"]
    else:
        plan = getattr(capture, "_request_source_closure_override", None)
        if not isinstance(plan, Mapping):
            plan = getattr(runtime, "_request_source_closure", None)
    if isinstance(plan, Mapping) and receipt["messages"]:
        from kernel_request_source_window import (
            source_delivery_satisfied, verify_final_source_delivery,
        )
        sources = (copy.deepcopy(frozen_inputs["sources"]) if frozen_inputs is not None
                   else _source_delivery_sources(capture, runtime))
        # Receipt recomputation must not rewrite the snapshotted source authority.
        receipt["source_delivery"] = verify_final_source_delivery(
            copy.deepcopy(plan), body["messages"], sources=sources, payload=json.dumps(body).encode())
        if not source_delivery_satisfied(receipt["source_delivery"]):
            failures.append("source_delivery_unqualified")
    receipt["status"] = "HOST_BLOCKED" if failures else "READY"
    receipt["final_validation_failures"] = failures
    capture.request_budget_final_wire = receipt
    emit = getattr(getattr(capture, "journal", None), "emit", None)
    if callable(emit):
        emit("request_budget_final_wire", producer="kernel_request_budget",
             request_id=getattr(capture, "request_id", None),
             step=getattr(capture, "step", None), dispatch=False, **journal_safe_fields(receipt))
        degraded = receipt.get("pass_gates_degraded")
        if isinstance(degraded, Mapping):
            emit("frozen_gate_surface_degraded", producer="kernel_final_wire_budget",
                 request_id=getattr(capture, "request_id", None),
                 step=getattr(capture, "step", None), dispatch=False,
                 **journal_safe_fields(degraded))
    return receipt


def _signature_deferred_paths(signature: tuple[object, ...]) -> Sequence[object]:
    if len(signature) <= 2 or not isinstance(signature[2], tuple):
        return ()
    source_signature = signature[2]
    if len(source_signature) <= 2 or not _is_nonstring_sequence(source_signature[2]):
        return ()
    return source_signature[2]


def _emit_revalidation_consistency(
        capture: object, *, previous_signature: tuple[object, ...],
        current_digest: str, receipt: Mapping[str, object],
        stable: bool, failures: Sequence[object],
) -> None:
    emit = getattr(getattr(capture, "journal", None), "emit", None)
    if not callable(emit):
        return
    previous_failures = (
        previous_signature[1]
        if len(previous_signature) > 1
        and _is_nonstring_sequence(previous_signature[1]) else ()
    )
    current_source = receipt.get("source_delivery")
    current_paths = (
        current_source.get("deferred_paths", ())
        if isinstance(current_source, Mapping) else ()
    )
    emit(
        "request_budget_revalidation_consistency",
        producer="kernel_final_wire_budget",
        request_id=getattr(capture, "request_id", None),
        step=getattr(capture, "step", None),
        body_sha256=current_digest,
        status="STABLE" if stable else "DRIFTED",
        previous_status=previous_signature[0] if previous_signature else None,
        current_status=receipt.get("status"),
        previous_failure_count=len(previous_failures),
        current_failure_count=len(failures),
        previous_deferred_path_count=len(_signature_deferred_paths(previous_signature)),
        current_deferred_path_count=(
            len(current_paths) if _is_nonstring_sequence(current_paths) else 0
        ),
        # b1403 (diagnostic only): counts hid the cause of a DRIFTED verdict.
        # Name the failures and the signature fields that differ; values are safe constants/bools.
        previous_failures=[str(item)[:80] for item in previous_failures][:16],
        current_failures=[str(item)[:80] for item in failures][:16],
        changed_fields=_signature_changed_fields(
            previous_signature, budget_verdict_signature(receipt)),
        dispatch=False,
    )


_SOURCE_SIGNATURE_FIELDS = (
    "status", "proof_status", "deferred_paths", "deferred_notice_verified",
    "read_tool_available", "transport_identity",
)


def _signature_changed_fields(previous: object, current: object) -> list[dict[str, object]]:
    """Diagnostic diff of two budget verdict signatures (names + bounded safe values)."""
    def _safe(value: object) -> object:
        if isinstance(value, (bool, int)) or value is None:
            return value
        if isinstance(value, (list, tuple)):
            return len(value)
        return str(value)[:80]

    if not isinstance(previous, tuple) or not isinstance(current, tuple):
        return [{"field": "signature", "previous": previous is None, "current": current is None}]
    changed: list[dict[str, object]] = []
    for name, index in (("status", 0), ("failures", 1)):
        a = previous[index] if len(previous) > index else None
        b = current[index] if len(current) > index else None
        if a != b:
            changed.append({"field": name, "previous": _safe(a), "current": _safe(b)})
    a_src = previous[2] if len(previous) > 2 else None
    b_src = current[2] if len(current) > 2 else None
    if a_src != b_src:
        if not isinstance(a_src, tuple) or not isinstance(b_src, tuple):
            changed.append({"field": "source_delivery", "previous": _safe(a_src), "current": _safe(b_src)})
        else:
            for i, name in enumerate(_SOURCE_SIGNATURE_FIELDS):
                a = a_src[i] if len(a_src) > i else None
                b = b_src[i] if len(b_src) > i else None
                if a != b:
                    changed.append({"field": "source_delivery." + name,
                                    "previous": _safe(a), "current": _safe(b)})
    return changed[:12]


def record_wire_truth_revalidation_consistency(
        capture: object, *, previous_digest: str,
        previous_signature: tuple[object, ...],
        current_digest: str, receipt: Mapping[str, object],
) -> dict[str, object]:
    """Fail closed if the same final wire changes budget verdict after wire truth."""
    if previous_digest != current_digest:
        return dict(receipt)
    stable = budget_verdict_signature(receipt) == previous_signature
    updated = dict(receipt)
    failures = updated.get("final_validation_failures", ())
    failures = list(failures) if _is_nonstring_sequence(failures) else []
    if not stable:
        if "frozen_revalidation_verdict_changed_after_wire_truth" not in failures:
            failures.append("frozen_revalidation_verdict_changed_after_wire_truth")
        updated["status"] = "HOST_BLOCKED"
    updated["wire_truth_revalidation"] = {
        "status": "STABLE" if stable else "DRIFTED",
        "body_sha256": current_digest,
    }
    updated["final_validation_failures"] = failures
    setattr(capture, "request_budget_final_wire", updated)
    _emit_revalidation_consistency(
        capture,
        previous_signature=previous_signature,
        current_digest=current_digest,
        receipt=receipt,
        stable=stable,
        failures=failures,
    )
    return updated
