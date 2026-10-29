"""Snapshot and render host-known facts at the RequestCapture cutoff."""
from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping, Sequence


FACT_SCHEMA = "stateframe-wire-truth-facts/1"
FACT_FIELDS = (
    "candidate_revision",
    "check_execution_ref",
    "result_scope",
    "open_requirement_ids",
    "green_state",
    "turns_left",
    "unsettled_effect_ids",
    "last_refused_action_reason",
)
FACT_AUX_FIELDS = ("applied_edit_status",)


def _tool_name(tool: object) -> str:
    if not isinstance(tool, Mapping):
        return ""
    function = tool.get("function")
    if isinstance(function, Mapping):
        return str(function.get("name", ""))
    return str(tool.get("name", ""))


def fact_value(value: object) -> str:
    if isinstance(value, Mapping):
        return str(value.get("value", "UNKNOWN"))
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _fact_handle(value: object) -> str | None:
    if isinstance(value, Mapping):
        handle = value.get("handle")
        return str(handle) if handle not in (None, "") else None
    return None


def render_fact_messages(facts: Mapping[str, object]) -> list[dict[str, str]]:
    """Render each cutoff fact as a small, independently verifiable message."""
    messages: list[dict[str, str]] = []
    fields = (*FACT_FIELDS, *FACT_AUX_FIELDS)
    if "host_debt_disposition" in facts:
        fields = (*fields, "host_debt_disposition")
    for field in fields:
        if field not in facts:
            continue
        value = facts[field]
        value_text = fact_value(value)
        handle = _fact_handle(value)
        handle_text = f" handle={handle}" if handle else ""
        messages.append({
            "role": "user",
            "content": f"[WIRE_TRUTH_FACT {FACT_SCHEMA}] {field}: {value_text}{handle_text}",
        })
    dispatch = facts.get("done_dispatch_outcome")
    if dispatch not in (None, ""):
        messages.append({
            "role": "user",
            "content": f"[WIRE_TRUTH_FACT {FACT_SCHEMA}] done_dispatch_outcome: {dispatch}",
        })
    return messages


def _fact_line_values(messages: object) -> dict[str, list[tuple[str, str | None]]]:
    found: dict[str, list[tuple[str, str | None]]] = {}
    if not isinstance(messages, list):
        return found
    pattern = re.compile(
        r"(?im)^\s*(?:\[WIRE_TRUTH_FACT\s+[^\]]+\]\s*)?"
        r"([a-z][a-z0-9_]*)\s*[:=]\s*(.*?)\s*$"
    )
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        for match in pattern.finditer(content):
            field = match.group(1).lower()
            line = match.group(2).strip()
            handle_match = re.search(r"\bhandle=([^\s]+)", line)
            handle = handle_match.group(1) if handle_match else None
            value = re.sub(r"\s+handle=[^\s]+", "", line).strip()
            found.setdefault(field, []).append((value, handle))
    return found


def fact_failures(messages: object, expected: object,
                  retrievable_handles: object) -> list[str]:
    if not isinstance(expected, Mapping) or not expected:
        return []
    observed = _fact_line_values(messages)
    known_handles = {
        str(item) for item in (retrievable_handles or ()) if str(item).strip()
    }
    failures: list[str] = []
    for field, expected_value in expected.items():
        name = str(field).strip().lower()
        rows = observed.get(name, ())
        wanted = fact_value(expected_value)
        match = next((row for row in rows if row[0] == wanted), None)
        if match is None:
            failures.append(f"fact_missing_or_mismatched:{name}")
            continue
        if wanted.upper() in {"UNKNOWN", "OMITTED"}:
            handle = _fact_handle(expected_value)
            if handle is None or match[1] != handle or handle not in known_handles:
                failures.append(f"fact_omission_unretrievable:{name}")
    return failures


def _revision(value: object) -> str | None:
    if not isinstance(value, Mapping):
        return None
    for key in ("product_revision", "candidate_revision", "artifact", "revision"):
        text = str(value.get(key, "") or "").strip()
        if text:
            return text
    return None


def _result(receipt: Mapping[str, object]) -> str:
    for key in ("result_class", "result", "outcome", "status", "verdict"):
        value = str(receipt.get(key, "") or "").strip().upper()
        if value:
            return value
    return "UNKNOWN"


def _action_name(receipt: Mapping[str, object]) -> str:
    fallback = "UNKNOWN"
    for key in ("action", "tool", "tool_name", "kind"):
        value = str(receipt.get(key, "") or "").strip().lower()
        if not value:
            continue
        if fallback == "UNKNOWN":
            fallback = value
        if value in {"edit", "str_replace", "write", "apply_patch", "done"}:
            return value
    return fallback


def _public_evidence_refs(evidence: object, journal: object) -> str | None:
    """Render stable references while preserving host journal evidence internally."""
    host_run_id = getattr(journal, "run_id", None)
    from kernel_wire_identity import wire_run_id
    public_run_id = wire_run_id(journal) if journal is not None else None
    return ",".join(
        public_run_id + str(item)[len(host_run_id):]
        if isinstance(host_run_id, str) and public_run_id is not None
        and str(item).startswith(host_run_id + ":") else str(item)
        for item in evidence if str(item).strip()
    ) or None


def _current_public_check(runtime: object, binding: Mapping[str, object] | None
                          ) -> tuple[bool, object | None, object | None, object | None]:
    """Read the shared public-selector verdict and its exact supporting receipts."""
    if runtime is None or binding is None:
        return False, None, None, None
    try:
        from kernel_green_lock import _declared_selectors
        from kernel_verdict import evaluate_required_tests

        selectors = _declared_selectors(runtime)
        if not selectors:
            return False, None, None, None
        verdict = evaluate_required_tests(
            binding, selectors, getattr(runtime, "test_receipts", ()) or (),
        )
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return False, None, None, None
    state = str(getattr(verdict, "state", "UNKNOWN")).upper()
    if state not in {"SATISFIED", "VIOLATED"}:
        return False, None, None, None
    evidence = getattr(verdict, "supporting_evidence", ())
    check_ref = _public_evidence_refs(evidence, getattr(runtime, "journal", None))
    result = "GREEN" if state == "SATISFIED" else "RED"
    result_scope = f"{result}; scope={','.join(str(item) for item in selectors)}"
    green_state = "GREEN" if state == "SATISFIED" else "NOT_GREEN"
    return state == "SATISFIED", check_ref, result_scope, green_state


def _omission_fact(field: str, request_id: str,
                   store: dict[str, object]) -> dict[str, str]:
    handle = f"wire-truth:{request_id}:{field}"
    store[handle] = {"field": field, "status": "UNKNOWN", "cutoff_request_id": request_id}
    return {"value": "UNKNOWN", "handle": handle}


def _context_facts(
        runtime: object,
        request_id: str,
        candidate_revision: str | None,
        binding: Mapping[str, object] | None,
) -> tuple[dict[str, object], bool, bool, int | None, dict[str, object]]:
    store: dict[str, object] = {}
    owner = getattr(runtime, "owner", None)
    receipts = getattr(owner, "_receipts", ()) if owner is not None else ()
    receipts = list(receipts) if isinstance(receipts, Sequence) else []
    applied_edit = any(
        isinstance(row, Mapping)
        and _result(row) == "APPLIED"
        and _action_name(row) in {"edit", "str_replace", "write", "apply_patch"}
        for row in receipts
    )
    refused_edit = any(
        isinstance(row, Mapping)
        and _result(row) in {"REFUSED", "REJECTED", "DENIED", "BLOCKED"}
        and _action_name(row) in {"edit", "str_replace", "write", "apply_patch"}
        for row in receipts
    )
    current_green, check_ref, result_scope, green_state = _current_public_check(
        runtime, binding,
    )
    try:
        from kernel_done_readiness_authority import cached_decision
        shared_done_decision = cached_decision(runtime)
    except (ImportError, AttributeError, TypeError, ValueError):
        shared_done_decision = None
    if shared_done_decision is not None:
        current_green = bool(shared_done_decision.current_green)
        green_state = shared_done_decision.public_check_status
    owner = getattr(runtime, "owner", None)
    parent_revision = str(
        getattr(owner, "base_revision", "")
        or getattr(getattr(runtime, "green_lock", None), "_starting_revision", "")
        or ""
    ).strip() or None
    try:
        from kernel_green_lock import product_revision_change_from_parent
        changed, _current_product_revision, _comparison = product_revision_change_from_parent(runtime)
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        changed = False
    limits = getattr(runtime, "continuation_limits", None)
    turns_left = getattr(limits, "model_decisions_remaining", None)
    unbounded = limits is not None and getattr(limits, "limit", 0) is None
    if type(turns_left) is not int and not unbounded:
        current_step = getattr(owner, "current_step", None)
        max_steps = getattr(runtime, "max_steps", None)
        turns_left = (max(0, int(max_steps) - int(current_step))
                      if type(current_step) is int and type(max_steps) is int else None)
    turns_available = unbounded or (type(turns_left) is int and turns_left > 0)
    if shared_done_decision is not None:
        raw_open = (
            shared_done_decision.blocker_ids
            if shared_done_decision.decision != "ACCEPTED_CERTIFIED" else ()
        )
        open_ids = list(raw_open)
        require_done = bool(
            shared_done_decision.decision in {"ACCEPTED_CERTIFIED", "ACCEPTED_UNCERTIFIED"}
            and changed and turns_available
        )
        decision: object | None = None
    else:
        require_done = bool(changed and current_green and turns_available)
        decision = getattr(runtime, "_last_completion_decision", None)
        raw_open = getattr(decision, "needs_work_ids", None)
        if raw_open is None and isinstance(decision, Mapping):
            raw_open = decision.get("needs_work_ids", decision.get("open_requirement_ids"))
        open_ids = list(raw_open) if isinstance(raw_open, Sequence) and not isinstance(
            raw_open, (str, bytes)) else None
    if shared_done_decision is None and open_ids is None and decision is not None and getattr(decision, "accepted", None) is True:
        open_ids = []
    refused = next((row for row in reversed(receipts)
                    if isinstance(row, Mapping)
                    and _result(row) in {"REFUSED", "REJECTED", "DENIED", "BLOCKED"}), None)
    if refused is not None:
        refused_action = _action_name(refused)
        refused_reason = str(refused.get("reason", refused.get("refusal_reason", "UNKNOWN")))
        last_refused: object = f"{refused_action}: {refused_reason}"
    else:
        last_refused = None
    unsettled = getattr(runtime, "unsettled_effect_ids", None)
    if unsettled is None:
        unsettled = getattr(runtime, "pending_effect_ids", None)
    if unsettled is None:
        unsettled = getattr(runtime, "unsettled_effects", None)
    values: dict[str, object] = {
        "candidate_revision": candidate_revision,
        "check_execution_ref": check_ref,
        "result_scope": result_scope,
        "open_requirement_ids": open_ids,
        "green_state": green_state,
        "turns_left": turns_left,
        "unsettled_effect_ids": unsettled,
        "last_refused_action_reason": last_refused,
        "applied_edit_status": (
            "APPLIED" if applied_edit else "REFUSED_ONLY" if refused_edit else "NONE"
        ),
    }
    if (shared_done_decision is not None
            and shared_done_decision.decision == "ACCEPTED_UNCERTIFIED"):
        from kernel_done_readiness_authority import host_debt_disposition_text
        values["host_debt_disposition"] = host_debt_disposition_text(shared_done_decision)
    facts: dict[str, object] = {}
    for field in (*FACT_FIELDS, *FACT_AUX_FIELDS):
        value = values.get(field)
        facts[field] = _omission_fact(field, request_id, store) if value is None else value
    if "host_debt_disposition" in values:
        facts["host_debt_disposition"] = values["host_debt_disposition"]
    certification_complete = (
        shared_done_decision.decision == "ACCEPTED_CERTIFIED"
        if shared_done_decision is not None
        else bool(
            getattr(decision, "accepted", False)
            or (isinstance(decision, Mapping) and decision.get("accepted") is True)
        )
    )
    route = getattr(runtime, "_completion_owner_completion_decision", None)
    if shared_done_decision is None and route is not None:
        certification_complete = bool(getattr(route, "certified_acceptance", False))
    return facts, require_done, applied_edit, turns_left, {
        "candidate_changed": changed,
        "current_green": current_green,
        "certification_complete": certification_complete,
        "done_dispatch_outcome": (
            "UNCERTIFIED_SUBMIT" if require_done and not certification_complete
            else "CERTIFIED_SUBMIT" if require_done else None
        ),
        "fact_store": store,
        "parent_revision": parent_revision,
        "done_authority_decision": (
            shared_done_decision.to_dict() if shared_done_decision is not None else None
        ),
    }


def capture_context(capture: object, runtime: object, body: Mapping[str, object], *,
                    side_call: bool = False) -> dict[str, object]:
    """Snapshot cutoff facts and menu authorities for a RequestCapture attempt."""
    override = getattr(capture, "wire_truth_context", None)
    request_id = str(getattr(capture, "request_id", "UNKNOWN"))
    binding: Mapping[str, object] | None = None
    if runtime is not None:
        try:
            raw_binding = runtime.binding()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            raw_binding = None
        if isinstance(raw_binding, Mapping):
            binding = raw_binding
    candidate_revision = _revision(binding)
    decision_request = bool(
        not side_call and runtime is not None and isinstance(body.get("tools"), list)
        and body.get("tools")
    )
    context: dict[str, object] = {
        "request_id": request_id,
        "candidate_revision": candidate_revision or "UNKNOWN",
        "decision_request": decision_request,
    }
    if isinstance(override, Mapping):
        context.update(override)
    elif decision_request:
        facts, require_done, applied_edit, turns_left, state = _context_facts(
            runtime, request_id, candidate_revision, binding,
        )
        context.update(state)
        context.update({
            "required_facts": facts,
            "require_done": require_done,
            "has_applied_edit": applied_edit,
            "turns_left": turns_left,
            "certification_complete": state["certification_complete"],
            "done_dispatch_outcome": state["done_dispatch_outcome"],
            "retrievable_fact_handles": tuple(state["fact_store"]),
        })
        setattr(capture, "wire_truth_fact_store", dict(state["fact_store"]))
        if runtime is not None:
            stores = getattr(runtime, "_wire_truth_omission_store", None)
            if not isinstance(stores, dict):
                stores = {}
                try:
                    setattr(runtime, "_wire_truth_omission_store", stores)
                except (AttributeError, TypeError):
                    stores = None
            if isinstance(stores, dict):
                stores.update(state["fact_store"])
    context.pop("lean_required_source_paths", None)
    raw_lean_paths = getattr(capture, "_lean_required_source_paths", ())
    if isinstance(raw_lean_paths, Sequence) and not isinstance(
            raw_lean_paths, (str, bytes, bytearray)):
        lean_paths = tuple(dict.fromkeys(
            str(path).replace("\\", "/").strip()
            for path in raw_lean_paths if str(path).strip()
        ))
        if lean_paths:
            context["lean_required_source_paths"] = lean_paths
    authorities: list[dict[str, object]] = []
    contract = getattr(capture, "request_action_contract", None)
    raw_surface = getattr(capture, "final_tool_surface", None)
    surface = (
        raw_surface
        if isinstance(raw_surface, Mapping)
        and raw_surface.get("schema") == "stateframe-final-tool-surface/1"
        and raw_surface.get("request_id") == request_id
        and isinstance(raw_surface.get("tools"), list)
        and isinstance(raw_surface.get("names"), list)
        and isinstance(raw_surface.get("tool_schema_digest"), str)
        else None
    )
    shared_tools = surface.get("tools") if isinstance(surface, Mapping) else None
    shared_names = surface.get("names") if isinstance(surface, Mapping) else None
    shared_digest = surface.get("tool_schema_digest") if isinstance(surface, Mapping) else None
    if (isinstance(contract, Mapping)
            and (isinstance(shared_tools, list)
                 or isinstance(contract.get("admitted_tool_schemas"), list))):
        authorities.append({
            "name": "completion_owner_admitted_route",
            "tools": shared_tools if isinstance(shared_tools, list)
            else contract["admitted_tool_schemas"],
        })
    frozen = getattr(runtime, "_greenlock_frozen_menu", None) if runtime is not None else None
    if isinstance(frozen, Mapping) and (
            isinstance(shared_tools, list) or isinstance(frozen.get("tools"), list)):
        authorities.append({
            "name": "green_lock_frozen_menu",
            "tools": shared_tools if isinstance(shared_tools, list) else frozen["tools"],
        })
    budget = getattr(capture, "request_budget_final_wire", None)
    if isinstance(budget, Mapping) or isinstance(surface, Mapping):
        authority: dict[str, object] = {"name": "request_budget_final_wire"}
        if isinstance(surface, Mapping):
            authority["tool_schema_digest"] = shared_digest
            authority["tool_names"] = shared_names
        elif isinstance(budget, Mapping):
            for key in ("tool_schema_digest", "schema_digest", "tool_names", "tools"):
                if key in budget:
                    authority["tool_schema_digest" if key == "schema_digest" else key] = budget[key]
        if any(key in authority for key in ("tool_schema_digest", "schema_digest", "tool_names", "tools")):
            authorities.append(authority)
    if isinstance(budget, Mapping):
        context["budget_verdict"] = budget
    supplied_authorities = context.get("schema_authorities")
    if isinstance(supplied_authorities, Sequence) and not isinstance(
            supplied_authorities, (str, bytes)):
        authorities = [
            dict(item) for item in supplied_authorities if isinstance(item, Mapping)
        ] + authorities
    context["schema_authorities"] = authorities
    if runtime is not None:
        full_tools = getattr(runtime, "_completion_owner_full_tools", None)
        if isinstance(full_tools, list):
            done_schema = next((tool for tool in full_tools if _tool_name(tool) == "done"), None)
            if isinstance(done_schema, Mapping):
                from kernel_cache_layout_candidate import registered_done_for_wire
                context["done_tool_schema"] = dict(registered_done_for_wire(
                    capture, done_schema, shared_tools))
        source_plan = getattr(capture, "_request_source_closure_override", None)
        if not isinstance(source_plan, Mapping):
            source_plan = getattr(runtime, "_request_source_closure", None)
        if isinstance(source_plan, Mapping):
            # Verification enriches its plan; cutoff facts cannot alias authority.
            context["source_plan"] = copy.deepcopy(source_plan)
            sources = {
                str(path): str(row.get("content"))
                for path, row in getattr(runtime, "_empirical_frame_sources", {}).items()
                if isinstance(row, Mapping) and isinstance(row.get("content"), str)
            }
            context["source_contents"] = sources
            bounded = source_plan.get("bounded_delivery")
            if isinstance(bounded, Mapping):
                context["retrievable_omission_handles"] = tuple(
                    str(item) for item in bounded.get("omitted_handles", ()) if str(item)
                )
    if context.get("require_done") is True:
        disposition = str(context.get("done_dispatch_outcome", "UNCERTIFIED_SUBMIT"))
        try:
            setattr(capture, "wire_truth_done_disposition", disposition)
            if runtime is not None:
                setattr(runtime, "wire_truth_done_disposition", disposition)
        except (AttributeError, TypeError):
            pass
    return context


def retrieve_fact_omission(capture: object, runtime: object, handle: str) -> object | None:
    """Resolve an UNKNOWN fact handle recorded for this exact request cutoff."""
    stores = (
        getattr(capture, "wire_truth_fact_store", None),
        getattr(runtime, "_wire_truth_omission_store", None) if runtime is not None else None,
    )
    for store in stores:
        if isinstance(store, Mapping) and handle in store:
            return store[handle]
    return None


def add_fact_messages(
        body: dict[str, object], context: Mapping[str, object], *,
        append_only: bool = False,
) -> dict[str, object]:
    """Carry the cutoff tuple before final budgeting; the post-wire gate verifies it."""
    facts = context.get("required_facts")
    if context.get("decision_request") is not True or not isinstance(facts, Mapping):
        return body
    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list):
        return body
    rendered = render_fact_messages(facts | {
        "done_dispatch_outcome": context.get("done_dispatch_outcome"),
    })
    if append_only:
        messages = list(raw_messages)
        messages.extend(message for message in rendered if message not in messages)
    else:
        messages = [message for message in raw_messages
                    if not (isinstance(message, Mapping)
                            and str(message.get("content", "")).startswith(
                                f"[WIRE_TRUTH_FACT {FACT_SCHEMA}]"))]
        messages.extend(rendered)
    updated = dict(body)
    updated["messages"] = messages
    return updated


__all__ = [
    "FACT_AUX_FIELDS", "FACT_FIELDS", "FACT_SCHEMA", "add_fact_messages",
    "capture_context", "fact_failures", "fact_value", "render_fact_messages",
    "retrieve_fact_omission",
]
