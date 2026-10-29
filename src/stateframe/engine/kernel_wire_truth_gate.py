"""Post-serialization checks for the exact request bytes sent to a provider."""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from kernel_wire_truth_facts import (
    FACT_AUX_FIELDS,
    FACT_FIELDS,
    FACT_SCHEMA,
    add_fact_messages,
    capture_context,
    fact_failures,
    render_fact_messages,
    retrieve_fact_omission,
)

MODE_ENV = "STATEFRAME_WIRE_TRUTH_GATE"


class WireTruthBlocked(RuntimeError):
    """A serialized provider body violated a required wire invariant."""

    code = "WIRE_TRUTH_BLOCKED"
    outcome = "WIRE_TRUTH_BLOCKED"

    def __init__(self, failed: Sequence[str], *, body_sha256: str = "") -> None:
        self.failed = tuple(str(item) for item in failed)
        self.body_sha256 = body_sha256
        # An unsent candidate may attach its source layout for bounded repair.
        # This is never a sealed request or wire authorization.
        self.source_layout_proposal: bytes | None = None
        # Failure codes are journaled through the bounded allowlist separately;
        # keep exception text fixed so it cannot smuggle paths or request data.
        super().__init__("wire truth validation rejected the prepared request")


@dataclass(frozen=True)
class WireTruthResult:
    request_id: str
    revision: str
    body_sha256: str
    schema_digest: str
    verdict: str
    failed: tuple[str, ...]
    body: Mapping[str, object]
    wire: bytes
    repaired: bool = False
    stripped_by: tuple[str, ...] = ()
    attempt: int = 1

    @property
    def key(self) -> tuple[str, str, str]:
        """Content identity for a request, candidate revision, and exact bytes."""
        return self.request_id, self.revision, self.body_sha256


def enabled() -> bool:
    """The truth gate is enabled unless the explicit kill switch says off."""
    return os.environ.get(MODE_ENV, "on").strip().lower() not in {
        "off", "0", "false", "no",
    }


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def schema_digest(tools: object) -> str:
    """Return the canonical digest for one parsed provider tool list."""
    return "sha256:" + hashlib.sha256(_canonical_json(tools)).hexdigest()


def _sha256(wire: bytes) -> str:
    return "sha256:" + hashlib.sha256(wire).hexdigest()


def _tool_name(tool: object) -> str:
    if not isinstance(tool, Mapping):
        return ""
    function = tool.get("function")
    if isinstance(function, Mapping):
        return str(function.get("name", ""))
    return str(tool.get("name", ""))


def _tool_names(tools: object) -> list[str]:
    if not isinstance(tools, list):
        return []
    return [_tool_name(tool) for tool in tools]


def _lean_source_plan_view(
        plan: Mapping[str, object], required_paths: Sequence[str],
) -> dict[str, object]:
    """Restrict LEAN's final source proof to task-declared/editable paths."""
    allowed = {
        str(path).replace("\\", "/").strip()
        for path in required_paths if str(path).strip()
    }
    raw_deferred = plan.get("deferred_paths", ())
    deferred_paths = (
        {
            str(path).replace("\\", "/").strip()
            for path in raw_deferred if str(path).strip()
        }
        if isinstance(raw_deferred, Sequence)
        and not isinstance(raw_deferred, (str, bytes, bytearray)) else set()
    )
    # The wire notice names the complete host-approved deferred set. Keep its
    # source rows in the proof view so the exact notice and every read route are
    # validated together, even when a dependency is outside the editable scope.
    allowed.update(deferred_paths)
    selected = dict(plan)

    raw_units = plan.get("units", ())
    units = list(raw_units) if isinstance(raw_units, Sequence) and not isinstance(
        raw_units, (str, bytes, bytearray)
    ) else []
    selected_units: list[object] = []
    for unit in units:
        if not isinstance(unit, Mapping):
            selected_units.append(unit)
            continue
        path = str(unit.get("path", "")).replace("\\", "/").strip()
        if not path or path in allowed:
            unit_view = dict(unit)
            if path:
                unit_view["path"] = path
            selected_units.append(unit_view)
    selected["units"] = selected_units
    unit_paths = [
        str(unit.get("path", "")).replace("\\", "/").strip()
        for unit in selected_units
        if isinstance(unit, Mapping) and str(unit.get("path", "")).strip()
    ]

    raw_paths = plan.get("paths", ())
    paths = [
        str(path).replace("\\", "/").strip()
        for path in raw_paths
        if str(path).strip()
        and str(path).replace("\\", "/").strip() in allowed
    ] if isinstance(raw_paths, Sequence) and not isinstance(
        raw_paths, (str, bytes, bytearray)
    ) else []
    for path in unit_paths:
        if path not in paths:
            paths.append(path)
    selected["paths"] = paths

    for key in ("revisions", "source_contents", "delivery", "final_delivery"):
        value = plan.get(key)
        if isinstance(value, Mapping):
            selected[key] = {}
            for path, row in value.items():
                normalized = str(path).replace("\\", "/").strip()
                if normalized in allowed:
                    selected[key][normalized] = row
    if isinstance(raw_deferred, Sequence) and not isinstance(
            raw_deferred, (str, bytes, bytearray)):
        selected["deferred_paths"] = [
            path for path in raw_deferred
            if not str(path).strip()
            or str(path).replace("\\", "/").strip() in allowed
        ]
    for key in ("omissions", "skipped_files"):
        rows = plan.get(key)
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)):
            selected_rows: list[object] = []
            for row in rows:
                if not isinstance(row, Mapping):
                    selected_rows.append(row)
                    continue
                path = str(row.get("path", "")).replace("\\", "/").strip()
                if not path or path in allowed:
                    row_view = dict(row)
                    if path:
                        row_view["path"] = path
                    selected_rows.append(row_view)
            selected[key] = selected_rows

    bounded = plan.get("bounded_delivery")
    if isinstance(bounded, Mapping):
        selected_handles = {
            str(unit.get("unit_handle", unit.get("unit_id", unit.get("handle", ""))))
            for unit in selected_units if isinstance(unit, Mapping)
        }
        bounded_view = dict(bounded)
        for key in ("omitted_handles", "retained_handles"):
            handles = bounded.get(key)
            if isinstance(handles, Sequence) and not isinstance(
                    handles, (str, bytes, bytearray)):
                bounded_view[key] = [str(handle) for handle in handles
                                     if str(handle) in selected_handles]
        source_paths = bounded.get("source_paths")
        if isinstance(source_paths, Sequence) and not isinstance(
                source_paths, (str, bytes, bytearray)):
            bounded_view["source_paths"] = [
                path for path in source_paths
                if str(path).replace("\\", "/").strip() in allowed
            ]
        selected["bounded_delivery"] = bounded_view
    return selected


def _bound_source_delivery_paths(
        messages: object, context: Mapping[str, object], wire: bytes,
        plan: Mapping[str, object],
) -> set[str] | None:
    """Use a host receipt only when it proves this exact wire and this plan's revisions."""
    if not isinstance(messages, list):
        return None
    budget = context.get("budget_verdict")
    receipt = budget.get("source_delivery") if isinstance(budget, Mapping) else None
    if not isinstance(receipt, Mapping):
        return None
    try:
        from kernel_request_source_window import source_delivery_satisfied
    except ImportError:
        return None

    context_digest = _sha256(_canonical_json({"messages": messages}))
    if (
        receipt.get("transport_identity") != "VERIFIED"
        or receipt.get("submitted_http_bytes_digest") != _sha256(wire)
        or receipt.get("context_digest") != context_digest
        or receipt.get("submitted_payload_hash") != context_digest
        or receipt.get("proof_failures")
        or not source_delivery_satisfied(receipt)
    ):
        return None

    raw_results = receipt.get("paths")
    if not isinstance(raw_results, Mapping):
        return None
    raw_revisions = plan.get("revisions")
    revisions = raw_revisions if isinstance(raw_revisions, Mapping) else {}
    units_by_path: dict[str, list[Mapping[str, object]]] = {}
    planned_paths = {
        str(path).replace("\\", "/").strip()
        for path in plan.get("paths", ())
        if str(path).strip()
    } if isinstance(plan.get("paths", ()), Sequence) and not isinstance(
        plan.get("paths", ()), (str, bytes, bytearray)
    ) else set()
    raw_units = plan.get("units", ())
    if isinstance(raw_units, Sequence) and not isinstance(
            raw_units, (str, bytes, bytearray)):
        for unit in raw_units:
            if not isinstance(unit, Mapping):
                return None
            path = str(unit.get("path", "")).replace("\\", "/").strip()
            if path:
                planned_paths.add(path)
                units_by_path.setdefault(path, []).append(unit)
    if not planned_paths:
        return None

    deferred = {
        str(path).replace("\\", "/").strip()
        for path in receipt.get("deferred_paths", ()) if str(path).strip()
    } if isinstance(receipt.get("deferred_paths", ()), Sequence) and not isinstance(
        receipt.get("deferred_paths", ()), (str, bytes, bytearray)
    ) else set()
    plan_deferred = {
        str(path).replace("\\", "/").strip()
        for path in plan.get("deferred_paths", ()) if str(path).strip()
    } if isinstance(plan.get("deferred_paths", ()), Sequence) and not isinstance(
        plan.get("deferred_paths", ()), (str, bytes, bytearray)
    ) else set()
    if not plan_deferred.issubset(deferred):
        return None

    verified: set[str] = set()
    for path in planned_paths:
        row = raw_results.get(path)
        if not isinstance(row, Mapping):
            return None
        expected_revision = str(revisions.get(path, "") or "").strip()
        for unit in units_by_path.get(path, ()):
            unit_revision = str(unit.get("revision", "") or "").strip()
            if expected_revision and unit_revision and unit_revision != expected_revision:
                return None
            expected_revision = expected_revision or unit_revision
        if expected_revision and str(row.get("revision", "") or "").strip() != expected_revision:
            return None
        status = str(row.get("status", "")).upper()
        proof_status = str(row.get("proof_status", "")).upper()
        if status in {"COMPLETE", "DEFERRED"} and proof_status == "VERIFIED":
            verified.add(path)
        elif status == "NOT_SELECTED" and proof_status == "VERIFIED" and path not in plan_deferred:
            continue
        else:
            return None
    return verified


def _source_failures(messages: object, context: Mapping[str, object],
                     wire: bytes) -> list[str]:
    failures: list[str] = []
    content = "\n".join(
        str(message.get("content") or "")
        for message in messages if isinstance(message, Mapping)
    ) if isinstance(messages, list) else ""
    known_handles = {
        str(item) for item in context.get("retrievable_omission_handles", ())
        if str(item).strip()
    }
    plan = context.get("source_plan")
    raw_lean_paths = context.get("lean_required_source_paths", ())
    lean_required_paths = tuple(dict.fromkeys(
        str(path).replace("\\", "/").strip()
        for path in raw_lean_paths if str(path).strip()
    )) if isinstance(raw_lean_paths, Sequence) and not isinstance(
        raw_lean_paths, (str, bytes, bytearray)
    ) else ()
    if isinstance(plan, Mapping) and lean_required_paths:
        plan = _lean_source_plan_view(plan, lean_required_paths)
        planned_paths = {
            str(unit.get("path", "")).replace("\\", "/").strip()
            for unit in plan.get("units", ()) if isinstance(unit, Mapping)
            and str(unit.get("path", "")).strip()
        }
        failures.extend(
            f"source_plan_missing_required_path:{path}"
            for path in lean_required_paths if path not in planned_paths
        )
    elif lean_required_paths:
        failures.extend(
            f"source_plan_missing_required_path:{path}"
            for path in lean_required_paths
        )
    requested_deferred_paths: set[str] = set()
    if isinstance(plan, Mapping):
        raw_deferred = plan.get("deferred_paths", ())
        if isinstance(raw_deferred, Sequence) and not isinstance(raw_deferred, (str, bytes)):
            requested_deferred_paths = {
                str(path).replace("\\", "/").strip()
                for path in raw_deferred if str(path).strip()
            }
    verified_source_paths: set[str] = set()
    if isinstance(plan, Mapping) and plan.get("units"):
        try:
            from kernel_request_source_window import (
                source_delivery_satisfied, verify_final_source_delivery,
            )
            exact_source_paths = _bound_source_delivery_paths(
                messages, context, wire, plan,
            )
            bounded = plan.get("bounded_delivery")
            omitted = {
                str(item) for item in bounded.get("omitted_handles", ()) if str(item)
            } if isinstance(bounded, Mapping) else set()
            omissions = plan.get("omissions", ())
            if isinstance(omissions, Sequence):
                omitted.update(
                    str(item.get("unit_handle", item.get("handle", "")))
                    for item in omissions if isinstance(item, Mapping)
                    and item.get("unit_handle", item.get("handle"))
                )
            known_handles.update(omitted)
            plan_units = plan.get("units", ())
            for index, unit in enumerate(plan_units):
                if not isinstance(unit, Mapping):
                    failures.append(f"source_unit_invalid:{index}")
            if exact_source_paths is not None:
                verified_source_paths = exact_source_paths
            elif plan_units or requested_deferred_paths:
                # First prove complete/deferred paths from the exact payload.
                # A unit handle may be absent from bounded context while its
                # full source is still present and verified on this wire.
                visible_plan = dict(plan)
                sources = context.get("source_contents")
                receipt = verify_final_source_delivery(
                    visible_plan,
                    messages if isinstance(messages, list) else [],
                    payload=wire,
                    sources=sources if isinstance(sources, Mapping) else None,
                )
                if not source_delivery_satisfied(receipt):
                    failures.append("source_delivery_incomplete")
                else:
                    path_results = receipt.get("paths")
                    if isinstance(path_results, Mapping):
                        verified_source_paths = {
                            str(path).replace("\\", "/").strip()
                            for path, row in path_results.items()
                            if isinstance(row, Mapping)
                            and str(row.get("status", "")).upper()
                            in {"COMPLETE", "DEFERRED"}
                            and str(row.get("proof_status", "")).upper() == "VERIFIED"
                        }
            for index, unit in enumerate(plan_units):
                if not isinstance(unit, Mapping):
                    continue
                handle = str(unit.get("unit_handle", unit.get("unit_id", "")) or "")
                path = str(unit.get("path", "")).replace("\\", "/").strip()
                if (handle in omitted and path not in verified_source_paths
                        and handle not in content):
                    failures.append(
                        f"source_omission_unretrievable:{index}:{path}:plan_unit"
                    )
        except (AttributeError, ImportError, OSError, TypeError, ValueError):
            failures.append("source_delivery_unverifiable")

    units = context.get("required_source_units", ())
    if isinstance(units, Sequence) and not isinstance(units, (str, bytes)):
        for index, unit in enumerate(units):
            if not isinstance(unit, Mapping):
                failures.append(f"source_unit_invalid:{index}")
                continue
            path = str(unit.get("path", "")).replace("\\", "/").strip()
            if path and path in verified_source_paths:
                continue
            handle = str(unit.get("omission_handle", "") or "").strip()
            if handle:
                if handle not in known_handles or handle not in content:
                    failures.append(
                        f"source_omission_unretrievable:{index}:{path}:required_unit"
                    )
                continue
            body = unit.get("body", unit.get("text"))
            if not isinstance(body, str) or not body.strip() or body not in content:
                failures.append(f"source_body_missing_or_blank:{index}")
                continue
            revision = str(unit.get("revision", "") or "").strip()
            body_hash = str(unit.get("sha256", unit.get("hash", "")) or "").strip()
            if not ((revision and revision in content) or (body_hash and body_hash in content)):
                failures.append(f"source_identity_missing:{index}")
    return failures


def _authority_failures(tools: object, authorities: object,
                        *, allow_done_expansion: bool) -> list[str]:
    if not isinstance(authorities, Sequence) or isinstance(authorities, (str, bytes)):
        return []
    failures: list[str] = []
    parsed_names = _tool_names(tools)
    for authority in authorities:
        if not isinstance(authority, Mapping):
            continue
        name = str(authority.get("name", "menu_authority"))
        schemas = authority.get("tools", authority.get("schemas"))
        digest = authority.get("schema_digest", authority.get("tool_schema_digest"))
        names = authority.get("tool_names", authority.get("names"))
        matches = False
        if isinstance(schemas, list):
            matches = schema_digest(schemas) == schema_digest(tools)
            if (not matches and allow_done_expansion and "done" in parsed_names
                    and "done" not in _tool_names(schemas)):
                without_done = [tool for tool in tools
                                if _tool_name(tool) != "done"] if isinstance(tools, list) else []
                matches = schema_digest(schemas) == schema_digest(without_done)
        elif isinstance(digest, str):
            matches = digest == schema_digest(tools)
            if (not matches and allow_done_expansion and "done" in parsed_names):
                without_done = [tool for tool in tools
                                if _tool_name(tool) != "done"] if isinstance(tools, list) else []
                matches = digest == schema_digest(without_done)
        elif isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
            expected_names = [str(item) for item in names]
            matches = expected_names == parsed_names
            if (not matches and allow_done_expansion and "done" in parsed_names):
                matches = expected_names == [item for item in parsed_names if item != "done"]
        if not matches:
            failures.append(f"schema_authority_mismatch:{name}")
    return failures


def evaluate_wire(wire: bytes, context: Mapping[str, object], *,
                  allow_done_expansion: bool = False,
                  attempt: int = 1) -> WireTruthResult:
    """Parse and validate the serialized body; never consult a pre-dump dict."""
    request_id = str(context.get("request_id", "UNKNOWN"))
    revision = str(context.get("candidate_revision", "UNKNOWN"))
    body_sha = _sha256(wire)
    failed: list[str] = []
    try:
        parsed = json.loads(wire)
    except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError):
        parsed = None
        failed.append("wire_invalid_json")
    body = parsed if isinstance(parsed, Mapping) else {}
    if parsed is not None and not isinstance(parsed, Mapping):
        failed.append("wire_body_not_object")
    tools = body.get("tools") if isinstance(body, Mapping) else None
    messages = body.get("messages") if isinstance(body, Mapping) else None
    digest = schema_digest(tools)
    if not isinstance(tools, list):
        failed.append("wire_tools_missing")
    if not isinstance(messages, list):
        failed.append("wire_messages_missing")
    budget = context.get("budget_verdict", context.get("request_budget_final_wire"))
    if isinstance(budget, Mapping):
        budget_status = str(budget.get("status", budget.get("reason", ""))).upper()
        if budget_status in {"HOST_BLOCKED", "HOST_CONTRACT_LIMIT"}:
            failed.append(f"budget_blocked:{budget_status}")
    if context.get("require_done") is True:
        if "done" not in _tool_names(tools):
            failed.append("required_done_missing")
    if (context.get("require_done") is True
            and context.get("certification_complete") is False
            and context.get("done_dispatch_outcome") != "UNCERTIFIED_SUBMIT"):
        failed.append("uncertified_done_dispatch_not_typed")
    names = _tool_names(tools)
    if len(names) != len(set(names)):
        failed.append("duplicate_tool_name")
    done_schemas = [tool for tool in tools if _tool_name(tool) == "done"] \
        if isinstance(tools, list) else []
    registered_done = context.get("done_tool_schema")
    if (done_schemas and isinstance(registered_done, Mapping)
            and schema_digest(done_schemas[0]) != schema_digest(registered_done)):
        failed.append("done_schema_not_registered")
    if names == ["done"] and context.get("has_applied_edit") is not True:
        failed.append("done_only_without_applied_edit")
    failed.extend(_authority_failures(
        tools, context.get("schema_authorities"),
        allow_done_expansion=allow_done_expansion,
    ))
    failed.extend(fact_failures(
        messages, context.get("required_facts"),
        context.get("retrievable_fact_handles"),
    ))
    failed.extend(_source_failures(messages, context, wire))
    return WireTruthResult(
        request_id=request_id,
        revision=revision,
        body_sha256=body_sha,
        schema_digest=digest,
        verdict="PASS" if not failed else "BLOCKED",
        failed=tuple(dict.fromkeys(failed)),
        body=body,
        wire=wire,
        attempt=attempt,
    )


def _stripped_by(context: Mapping[str, object], tools: object) -> tuple[str, ...]:
    current = set(_tool_names(tools))
    names: list[str] = []
    authorities = context.get("schema_authorities", ())
    if isinstance(authorities, Sequence) and not isinstance(authorities, (str, bytes)):
        for authority in authorities:
            if not isinstance(authority, Mapping):
                continue
            authority_tools = authority.get("tools", authority.get("schemas"))
            authority_names = _tool_names(authority_tools)
            declared_names = authority.get("tool_names", authority.get("names"))
            if not authority_names and isinstance(declared_names, Sequence):
                authority_names = [str(item) for item in declared_names]
            if "done" not in current and "done" not in authority_names:
                names.append(str(authority.get("name", "unknown_menu")))
    explicit = context.get("done_stripped_by")
    if explicit:
        if isinstance(explicit, str):
            names.append(explicit)
        elif isinstance(explicit, Sequence):
            names.extend(str(item) for item in explicit)
    return tuple(dict.fromkeys(names or ["late_menu_projection"]))


def check_wire(
        wire: bytes,
        context: Mapping[str, object],
        *,
        emit: Callable[..., object] | None = None,
        attempt: int = 1,
        allow_done_repair: bool = True,
) -> WireTruthResult:
    """Check one attempt and repair only a missing required ``done`` once."""
    if not enabled():
        if emit is not None:
            emit("wire_truth_gate_disabled", producer="kernel_wire_truth_gate",
                 request_id=context.get("request_id"), reason="STATEFRAME_WIRE_TRUTH_GATE=off",
                 attempt=attempt, dispatch=False)
        try:
            parsed = json.loads(wire)
        except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError):
            parsed = {}
        body = parsed if isinstance(parsed, Mapping) else {}
        tools = body.get("tools") if isinstance(body, Mapping) else None
        return WireTruthResult(
            request_id=str(context.get("request_id", "UNKNOWN")),
            revision=str(context.get("candidate_revision", "UNKNOWN")),
            body_sha256=_sha256(wire), schema_digest=schema_digest(tools),
            verdict="DISABLED", failed=(), body=body, attempt=attempt,
            wire=wire,
        )

    initial = evaluate_wire(wire, context, attempt=attempt)
    stripped_by: tuple[str, ...] = ()
    repairable_failures = set(initial.failed)
    repairable_failures.discard("required_done_missing")
    repairable = (
        "required_done_missing" in initial.failed
        and all(item.startswith("schema_authority_mismatch:")
                for item in repairable_failures)
    )
    if (repairable and allow_done_repair
            and context.get("require_done") is True):
        done_schema = context.get("done_tool_schema")
        tools = initial.body.get("tools")
        if isinstance(done_schema, Mapping) and isinstance(tools, list):
            stripped_by = _stripped_by(context, tools)
            rebuilt = dict(initial.body)
            rebuilt["tools"] = [*tools, dict(done_schema)]
            wire = json.dumps(rebuilt).encode()
            final = evaluate_wire(
                wire, context, allow_done_expansion=True, attempt=attempt,
            )
            final = WireTruthResult(
                **{**final.__dict__, "wire": wire, "repaired": True,
                   "stripped_by": stripped_by}
            )
            if emit is not None:
                emit(
                    "wire_truth_checked", producer="kernel_wire_truth_gate",
                    request_id=initial.request_id, revision=initial.revision,
                    body_sha256=initial.body_sha256,
                    schema_digest=initial.schema_digest, verdict=initial.verdict,
                    failed=list(initial.failed), attempt=attempt, final=False,
                    dispatch=False,
                )
        else:
            final = initial
    else:
        final = initial
    if emit is not None:
        emit(
            "wire_truth_checked", producer="kernel_wire_truth_gate",
            request_id=final.request_id, revision=final.revision,
            body_sha256=final.body_sha256, schema_digest=final.schema_digest,
            verdict=final.verdict, failed=list(final.failed), attempt=attempt,
            current_green=context.get("current_green") is True,
            candidate_changed=context.get("candidate_changed") is True,
            final=True, dispatch=False,
        )
    if final.failed:
        if emit is not None:
            emit(
                "WIRE_TRUTH_BLOCKED", producer="kernel_wire_truth_gate",
                request_id=final.request_id, revision=final.revision,
                body_sha256=final.body_sha256, schema_digest=final.schema_digest,
                failed=list(final.failed), attempt=attempt,
                current_green=context.get("current_green") is True,
                candidate_changed=context.get("candidate_changed") is True,
                outcome="WIRE_TRUTH_BLOCKED", dispatch=False,
            )
        raise WireTruthBlocked(final.failed, body_sha256=final.body_sha256)
    if final.repaired and emit is not None:
        emit(
            "done_restored_on_wire", producer="kernel_wire_truth_gate",
            request_id=final.request_id, revision=final.revision,
            body_sha256=final.body_sha256, schema_digest=final.schema_digest,
            stripped_by=list(final.stripped_by), attempt=attempt, dispatch=False,
        )
    if context.get("require_done") is True and "done" in _tool_names(
            final.body.get("tools")) and emit is not None:
        emit(
            "done_offered_after_green", producer="kernel_wire_truth_gate",
            request_id=final.request_id, revision=final.revision,
            body_sha256=final.body_sha256, schema_digest=final.schema_digest,
            turns_left=context.get("turns_left"), attempt=attempt, dispatch=False,
        )
    return final


def check_wire_no_stall(
        wire: bytes, context: Mapping[str, object], *,
        emit: Callable[..., object], counters: object,
        rebuild: Callable[[], tuple[bytes, Mapping[str, object]]],
        attempt: int = 1,
        allow_done_repair: bool = False,
) -> WireTruthResult:
    """Reprepare once after a block, then explicitly degrade instead of stalling.

    Low-level check_wire remains strict for contract callers. Only this transport
    boundary opts into the design review's bounded recovery policy.
    """
    try:
        return check_wire(
            wire, context, emit=emit, attempt=attempt,
            allow_done_repair=allow_done_repair,
        )
    except WireTruthBlocked:
        counters.wire_truth_blocks = getattr(counters, "wire_truth_blocks", 0) + 1
    try:
        wire, context = rebuild()
        return check_wire(
            wire, context, emit=emit, attempt=attempt,
            allow_done_repair=allow_done_repair,
        )
    except WireTruthBlocked:
        counters.wire_truth_blocks += 1
    except Exception as error:
        # Recovery itself is advisory; never turn a recoverable block into a
        # host exception. Log type only, as request/error strings can be secret.
        emit("wire_truth_rebuild_failed", producer="kernel_wire_truth_gate",
             request_id=context.get("request_id"), error_type=type(error).__name__,
             dispatch=False)
    parsed = json.loads(wire)
    if not isinstance(parsed, dict):
        raise ValueError("wire recovery requires a serialized request object")
    tools = parsed.get("tools")
    done = context.get("done_tool_schema")
    if (context.get("require_done") is True and isinstance(done, Mapping)
            and isinstance(tools, list) and "done" not in _tool_names(tools)):
        parsed["tools"] = [*tools, dict(done)]
        wire = json.dumps(parsed).encode()
    final = evaluate_wire(
        wire, context, allow_done_expansion=allow_done_repair, attempt=attempt,
    )
    counters.wire_truth_degrades = getattr(counters, "wire_truth_degrades", 0) + 1
    emit("WIRE_TRUTH_DEGRADED", producer="kernel_wire_truth_gate",
         request_id=final.request_id, revision=final.revision,
         body_sha256=final.body_sha256, schema_digest=final.schema_digest,
         reason="bounded_rebuild_did_not_qualify", failed=list(final.failed),
         current_green=context.get("current_green") is True,
         candidate_changed=context.get("candidate_changed") is True,
         wire_truth_blocks=counters.wire_truth_blocks,
         wire_truth_degrades=counters.wire_truth_degrades, attempt=attempt,
         dispatch=False)
    return WireTruthResult(**{**final.__dict__, "verdict": "DEGRADED"})


__all__ = [
    "FACT_AUX_FIELDS", "FACT_FIELDS", "FACT_SCHEMA", "MODE_ENV", "WireTruthBlocked",
    "WireTruthResult",
    "add_fact_messages", "capture_context", "check_wire", "check_wire_no_stall",
    "enabled", "evaluate_wire",
    "render_fact_messages", "retrieve_fact_omission", "schema_digest",
]
