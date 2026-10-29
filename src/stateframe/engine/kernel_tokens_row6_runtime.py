"""Narrow runtime adapter for the pure v29 TOKENS basis/state/projection modules."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

from kernel_tokens_row6_basis import Basis, build_basis
from kernel_tokens_row6_projection import (
    continuity_snapshot,
    guidance_enabled,
    render_continuity,
    render_guidance_block,
    render_repeat_advice,
    replace_handoff_extension,
)
from kernel_tokens_row6_state import RepeatEvent, RepeatState

_EDIT_TOOLS = frozenset({"edit", "str_replace", "file_create", "write", "replace",
                         "replace_module"})
_MAX_UNEXECUTED_BATCH_ACTIONS = 16


def _text(value: object) -> str:
    return str(value).strip() if isinstance(value, (str, int, float, bool)) else ""


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                         default=str)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _binding_row(runtime: object) -> dict[str, str]:
    binding = runtime.binding()
    if not isinstance(binding, Mapping):
        return {}
    return {str(key): _text(value) for key, value in binding.items() if _text(value)}


def _decision_parts(decision: object) -> tuple[str, tuple[str, ...], str, str, str]:
    outcome = _text(getattr(decision, "outcome", "UNKNOWN")).upper()
    reason_codes = tuple(_text(item) for item in (getattr(decision, "reason_codes", ()) or ())
                         if _text(item))
    needs_work = tuple(_text(item) for item in (getattr(decision, "needs_work_ids", ()) or ())
                       if _text(item))
    basis = getattr(decision, "basis", None)
    evidence_digest = _text(getattr(basis, "evidence_digest", ""))
    snapshot_id = _text(getattr(basis, "snapshot_id", ""))
    return outcome, reason_codes, needs_work, evidence_digest, snapshot_id


def basis_for(runtime: object, decision: object = None, record: Mapping[str, object] | None = None
              ) -> Basis:
    """Build row-6 semantic evidence from current host bindings and the decision contract."""
    state = getattr(runtime, "tokens_row6_state")
    binding = _binding_row(runtime)
    artifact = binding.get("artifact") or binding.get("product_revision")
    environment_revision = binding.get("environment") or binding.get("environment_revision")
    selectors_value = runtime.required_public_selectors()
    selectors = tuple(sorted(_text(item) for item in (selectors_value or ()) if _text(item)))
    menu = tuple(sorted(_text(item) for item in (getattr(runtime, "completion_menu", ()) or ())
                        if _text(item)))
    outcome, reason_codes, needs_work, evidence_digest, snapshot_id = _decision_parts(decision)
    row = record if isinstance(record, Mapping) else {}
    if not evidence_digest:
        evidence_digest = _text(row.get("observation"))
    clause_ids = reason_codes or needs_work or ("completion.current",)
    clauses = [{
        "id": clause_id,
        "status": "SATISFIED" if outcome == "ACCEPTED" else "OPEN",
        "reason": ";".join(reason_codes),
        "applicability": "current-request",
        "freshness": environment_revision,
        "evidence_refs": tuple(_text(item) for item in (row.get("receipt_ids") or ())
                                if _text(item)),
    } for clause_id in clause_ids]
    effect_status = "SETTLED" if outcome == "ACCEPTED" else "PENDING"
    effect = {
        "id": "completion.authority",
        "status": effect_status,
        "tool": "done",
        "before_digest": artifact,
        "after_digest": artifact,
        "artifact_refs": (artifact,) if artifact else (),
        "unique_observation": False,
    }
    source_binding = artifact or ""
    harness_binding = _digest({"selectors": selectors, "menu": menu})
    authority_revision = f"{getattr(runtime, 'completion_contract_version', '')}|{','.join(menu)}"
    return build_basis({
        "request": {"identity": state.request_id},
        "contract": {"revision": _text(getattr(runtime, "completion_contract_version", ""))},
        "product": {"identity": "workspace-artifact", "revision": artifact},
        "harness": {"identity": "public-selectors", "revision": harness_binding,
                     "digest": harness_binding},
        "environment": [{"name": "host-environment", "version": environment_revision}],
        "tool_authority": {"identity": "completion-menu", "revision": authority_revision},
        "selectors": selectors,
        "clauses": clauses,
        "evidence": [{
            "check_id": "completion-observation",
            "check_version": _text(getattr(runtime, "completion_contract_version", "")),
            "inputs": selectors,
            "fixture": "runtime-receipt",
            "source_binding": source_binding,
            "harness_binding": harness_binding,
            "environment_binding": environment_revision,
            "verdict": outcome or "UNKNOWN",
            "witness_digest": evidence_digest,
            "admission": "HOST",
            "freshness": "CURRENT",
            "provenance_refs": ((snapshot_id,) if snapshot_id else ()) +
                               tuple(_text(item) for item in (row.get("receipt_ids") or ())
                                     if _text(item)),
        }],
        "effects": [effect],
        "unexecuted_batch_suffix": tuple(needs_work),
        "observation_coverage": {
            "spans": ("completion-final",),
            "revisions": (environment_revision,),
            "complete": outcome == "ACCEPTED",
            "final_delivery_proven": False,
        },
        "repair_witness": {
            "reachable": bool(getattr(decision, "repair_action", None) or needs_work),
            "clause_ids": needs_work,
            "witness_refs": (evidence_digest,),
        },
        "provenance_refs": tuple(_text(item) for item in (row.get("receipt_ids") or ())
                                  if _text(item)),
    })


def _record_action_id(record: Mapping[str, object]) -> str:
    for key in ("step_record_id", "receipt_id"):
        value = _text(record.get(key))
        if value:
            return value
    receipts = record.get("receipt_ids")
    if isinstance(receipts, Sequence) and not isinstance(receipts, (str, bytes, bytearray)):
        for item in receipts:
            value = _text(item)
            if value:
                return value
    return _digest(record)


def _proven_no_op(record: Mapping[str, object]) -> bool:
    tool = _text(record.get("tool")).casefold()
    result = _text(record.get("result", record.get("result_class", ""))).upper()
    before = record.get("before_binding")
    after = record.get("after_binding")
    if isinstance(before, Mapping):
        before = {str(key): value for key, value in before.items()
                  if key in {"artifact", "product_revision", "environment", "environment_revision",
                             "content_digest", "product_files", "harness_files"}}
    if isinstance(after, Mapping):
        after = {str(key): value for key, value in after.items()
                 if key in {"artifact", "product_revision", "environment", "environment_revision",
                            "content_digest", "product_files", "harness_files"}}
    unchanged = bool(before and after and _digest(before) == _digest(after))
    effect = _text(record.get("effect")).upper()
    return (tool in _EDIT_TOOLS and unchanged and bool(record.get("closed", True))
            and (result in {"NO_OP", "UNCHANGED"} or effect in {"NO_OP", "UNCHANGED"}))


def _emit_state(
        runtime: object, event: RepeatEvent, basis: Basis,
        authority_fields: Mapping[str, object] | None = None,
) -> None:
    journal = runtime.journal
    payload = {"event": event.to_dict(), "basis": basis.to_dict()}
    if authority_fields:
        payload.update(authority_fields)
    journal.emit("tokens_row6_observed", **payload, dispatch=False)
    if event.detected:
        journal.emit("tokens_row6_repeat_detected", **payload, dispatch=False)
    journal.emit("tokens_row6_state", state=runtime.tokens_row6_state.snapshot(),
                 event=event.to_dict(), dispatch=False)


def observe_done(runtime: object, decision: object) -> None:
    """Record a completed decision without modifying its authority or return value."""
    try:
        from kernel_done_readiness_authority import cached_decision

        state: RepeatState = runtime.tokens_row6_state
        shared = cached_decision(runtime)
        shared_fields: dict[str, object] = {}
        shared_open_ids: tuple[str, ...] = ()
        if shared is not None:
            shared_open_ids = tuple(dict.fromkeys((
                *tuple(getattr(shared, "blocker_ids", ()) or ()),
                *tuple(getattr(shared, "open_obligation_ids", ()) or ()),
            )))
            shared_result = (
                "UNCERTIFIED_SUBMIT"
                if getattr(shared, "decision", None) == "ACCEPTED_UNCERTIFIED"
                else str(getattr(shared, "decision", "UNKNOWN"))
            )
            shared_fields = {
                "shared_done_decision": shared_result,
                "shared_done_reason_code": str(getattr(shared, "reason_code", "")),
                "shared_done_open_ids": list(shared_open_ids),
                "shared_done_accepted": shared_result == "ACCEPTED_CERTIFIED",
                "shared_done_certified": shared_result == "ACCEPTED_CERTIFIED",
            }
        response_id = _text(getattr(runtime, "current_request_id", "")) or state.request_id
        action_id = f"done:{response_id}"
        completion_open_ids = tuple(getattr(decision, "needs_work_ids", ()) or ())
        completion_outcome = _text(getattr(decision, "outcome", "UNKNOWN"))
        completion_accepted = bool(getattr(decision, "accepted", False))
        if (shared is not None
                and getattr(shared, "decision", None) == "ACCEPTED_UNCERTIFIED"):
            completion_open_ids = tuple(dict.fromkeys((*completion_open_ids, *shared_open_ids)))
            completion_outcome = "UNCERTIFIED_SUBMIT"
            completion_accepted = False
        record = {
            "accepted": completion_accepted,
            "disposition": completion_outcome,
            "result": completion_outcome,
            "open_clause_ids": completion_open_ids,
            "pending_effects": completion_open_ids,
            "repair_reachable": bool(getattr(decision, "repair_action", None)),
        }
        basis = basis_for(runtime, decision, record)
        event = state.observe_done(action_id=action_id, response_id=response_id, basis=basis,
                                   accepted=completion_accepted,
                                   record=record)
        _emit_state(runtime, event, basis, shared_fields)
        if getattr(runtime, "stateframe_provability_advisory_mode", "off") == "on":
            from kernel_provability import completion_record, runtime_report
            report = getattr(runtime, "_provability_last_report", None)
            if report is None:
                report = runtime_report(
                    runtime,
                    decision,
                    frame=getattr(runtime, "_provability_last_frame", None),
                    tools=getattr(runtime, "_provability_last_tools", None),
                )
            record = completion_record(report, accepted=bool(getattr(decision, "accepted", False)))
            signature = (record["basis_digest"], record["accepted"], record["unprovable_count"])
            if signature != getattr(runtime, "_provability_last_record_signature", None):
                runtime.journal.emit(
                    "provability_completion_record",
                    record=record,
                    dispatch=False,
                )
                runtime._provability_last_record_signature = signature
    except (AttributeError, ImportError, OSError, TypeError, ValueError) as error:
        runtime.journal.emit("tokens_row6_observer_error", error_type=type(error).__name__,
                             phase="done", dispatch=False)


def observe_closed(runtime: object, record: Mapping[str, object]) -> None:
    """Record every closed action while keeping proven unchanged edits separate."""
    try:
        state: RepeatState = runtime.tokens_row6_state
        action_id = _record_action_id(record)
        response_id = _text(getattr(runtime, "current_request_id", "")) or state.request_id
        basis = basis_for(runtime, getattr(runtime, "_last_completion_decision", None), record)
        if _proven_no_op(record):
            event = state.observe_no_op(action_id=action_id, response_id=response_id,
                                        basis=basis, record=record)
        else:
            event = state.observe_closed(action_id, record, basis=basis)
        _emit_state(runtime, event, basis)
    except (AttributeError, ImportError, OSError, TypeError, ValueError) as error:
        runtime.journal.emit("tokens_row6_observer_error", error_type=type(error).__name__,
                             phase="closed", dispatch=False)


def _batch_argument(raw: object) -> object:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return raw[:1024]
    return _bounded_batch_value(raw)


def _bounded_batch_value(value: object, depth: int = 0) -> object:
    if depth >= 3:
        return _text(value)[:256]
    if isinstance(value, Mapping):
        return {str(key)[:64]: _bounded_batch_value(item, depth + 1)
                for key, item in list(value.items())[:32]}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_bounded_batch_value(item, depth + 1) for item in list(value)[:32]]
    if isinstance(value, str):
        return value[:1024]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _text(value)[:256]


def _batch_index(call: Mapping[str, object], fallback: int) -> int:
    value = call.get("id")
    try:
        return int(str(value).rsplit("_", 1)[1])
    except (IndexError, TypeError, ValueError):
        return fallback


def _journal_unexecuted_rows(runtime: object) -> list[dict[str, object]]:
    journal = getattr(runtime, "journal", None)
    records = getattr(journal, "records", None)
    if callable(records):
        try:
            events = list(records())
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            events = []
    else:
        events = getattr(journal, "events", ())
    rows: list[dict[str, object]] = []
    for event in events if isinstance(events, Sequence) else ():
        if isinstance(event, tuple) and len(event) == 2:
            kind, fields = event
            event = dict(fields) if isinstance(fields, Mapping) else {}
        elif isinstance(event, Mapping):
            kind = event.get("kind")
        else:
            continue
        if kind not in {"native_batch_discarded", "native_batch_unexecuted"}:
            continue
        calls = event.get("calls")
        if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes, bytearray)):
            continue
        origin = _text(event.get("origin_request_id"))
        start = event.get("next_index")
        start = start if type(start) is int and start >= 0 else 0
        for offset, call in enumerate(calls[:_MAX_UNEXECUTED_BATCH_ACTIONS]):
            if not isinstance(call, Mapping):
                continue
            function = call.get("function")
            function = function if isinstance(function, Mapping) else {}
            arguments = _batch_argument(function.get("arguments"))
            row = {
                "tool": _text(function.get("name")) or "unknown",
                "batch_index": _batch_index(call, start + offset),
                "arguments": arguments,
                "status": "STAGED_NOT_APPLIED",
                "reason": _text(event.get("reason")) or "batch stopped before execution",
                "origin_request_id": origin,
            }
            if isinstance(arguments, Mapping):
                for key in ("path", "target", "selector"):
                    value = _text(arguments.get(key))
                    if value:
                        row[key] = value[:512]
            rows.append(row)
    return rows[-_MAX_UNEXECUTED_BATCH_ACTIONS:]


def _unexecuted_batch_rows(runtime: object) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    retained = getattr(runtime, "_native_unexecuted_actions", ())
    if isinstance(retained, Sequence) and not isinstance(retained, (str, bytes, bytearray)):
        rows.extend(row for row in retained if isinstance(row, Mapping))
    rows.extend(_journal_unexecuted_rows(runtime))
    retired = set(getattr(runtime, "_native_unexecuted_reconciled", ()))
    unique: dict[tuple[str, int], Mapping[str, object]] = {}
    for row in rows[-_MAX_UNEXECUTED_BATCH_ACTIONS:]:
        try:
            index = int(row.get("batch_index", 0))
        except (TypeError, ValueError):
            index = 0
        key = (_text(row.get("origin_request_id")), index)
        if key not in retired:
            unique[key] = row
    return list(unique.values())[-_MAX_UNEXECUTED_BATCH_ACTIONS:]


def reconcile_unexecuted_batch_actions(runtime: object, *, origin_request_id: str | None = None,
                                       batch_indexes: Sequence[int] = ()) -> None:
    """Retire journaled intent only after an explicit host reconciliation."""
    retired = set(getattr(runtime, "_native_unexecuted_reconciled", ()))
    indexes = {int(index) for index in batch_indexes}
    for row in _unexecuted_batch_rows(runtime):
        origin = _text(row.get("origin_request_id"))
        index = row.get("batch_index")
        if origin_request_id is not None and origin != origin_request_id:
            continue
        if indexes and index not in indexes:
            continue
        retired.add((origin, index))
    setattr(runtime, "_native_unexecuted_reconciled", tuple(retired))
    setattr(runtime, "_native_unexecuted_actions", tuple(_unexecuted_batch_rows(runtime)))


def _unexecuted_batch_panel(runtime: object) -> str:
    rows = _unexecuted_batch_rows(runtime)
    if not rows:
        return ""
    lines = [
        "[TOKENS row6 unexecuted batch actions]",
        "The following intended actions were staged but not applied; do not execute them implicitly.",
    ]
    for row in list(rows)[-16:]:
        if not isinstance(row, Mapping):
            continue
        target = next((_text(row.get(key)) for key in ("path", "target", "selector")
                       if _text(row.get(key))), "unspecified target")
        lines.append(f"- {row.get('tool', 'unknown')} {target}: STAGED_NOT_APPLIED; "
                     f"reissue only after a fresh decision ({_text(row.get('reason'))})")
    lines.append("[/TOKENS row6 unexecuted batch actions]")
    return "\n".join(lines) if len(lines) > 2 else ""


def project_handoff(runtime: object, handoff: Mapping[str, object], *, current_check: str,
                    current_disposition: str, open_clause_ids: Sequence[str],
                    pending_intention: str, dependency_revision: str,
                    provability: object = None, tools: object = None) -> dict[str, object]:
    """Add only bounded receipt-backed panels to the existing F4 handoff."""
    updated = dict(handoff)
    from kernel_route_budget import render as render_route_budget
    route_panel = render_route_budget(
        runtime, getattr(runtime, "_last_completion_decision", None), tools)
    if route_panel:
        updated = replace_handoff_extension(
            updated, route_panel, marker=("[STATEFRAME recovery view]",
                                          "[/STATEFRAME recovery view]"))
    # Row 6 is an advisory observer.  The measured guidance-off profile must
    # not receive its continuity projection, even when state has been recorded.
    # Keep the original handoff byte-for-byte so observation and completion
    # authority remain untouched.
    if not guidance_enabled(getattr(runtime, "tokens_row6_policy", None)):
        return updated
    state: RepeatState = runtime.tokens_row6_state
    unprovable = tuple(getattr(provability, "unprovable_items", ()) or ())
    capability_unknown = (provability is not None
                          and not bool(getattr(provability, "host_known", False)))
    unprovable_ids = {
        _text(getattr(item, "item_id", "")) for item in unprovable
        if _text(getattr(item, "item_id", ""))
    }
    seen_before = set(getattr(runtime, "_provability_seen_keys", ()))
    fresh_unprovable = tuple(item for item in unprovable
                             if _text(getattr(item, "key", "")) not in seen_before)
    display_check = current_check
    if unprovable:
        if fresh_unprovable:
            display_check = "Provability status: unprovable items disclosed below"
        else:
            display_check = "Provability notice already delivered"
    elif capability_unknown:
        display_check = "Provability status: host capability matrix unknown"
    display_open_clause_ids = tuple(
        item for item in open_clause_ids if _text(item) not in unprovable_ids)
    continuity = render_continuity(
        continuity_snapshot(
            state,
            current_check=display_check,
            current_disposition=current_disposition,
            open_clause_ids=display_open_clause_ids,
            pending_intention=pending_intention,
            dependency_revision=dependency_revision,
        )
    )
    if continuity:
        updated = replace_handoff_extension(updated, continuity)
    abandoned = _unexecuted_batch_panel(runtime)
    if abandoned:
        updated = replace_handoff_extension(
            updated, abandoned,
            marker=("[TOKENS row6 unexecuted batch actions]",
                    "[/TOKENS row6 unexecuted batch actions]"),
        )
    guidance = ("" if unprovable or capability_unknown else render_guidance_block(
        getattr(runtime, "tokens_row6_policy", None),
        final=current_disposition.upper() == "ACCEPTED",
    ))
    if guidance:
        updated = replace_handoff_extension(
            updated, guidance,
            marker=("[TOKENS row6 concise guidance]", "[/TOKENS row6 concise guidance]"),
        )
    event = state.last_event
    repeat = (render_repeat_advice(event, open_clause_ids=open_clause_ids)
              if event and not unprovable and not capability_unknown else "")
    if repeat and guidance_enabled(getattr(runtime, "tokens_row6_policy", None)):
        updated = replace_handoff_extension(
            updated, repeat,
            marker=("[TOKENS row6 repeat advice]", "[/TOKENS row6 repeat advice]"),
        )
    if provability is not None:
        from kernel_provability import ADVISORY_END, ADVISORY_START, advisory_for
        advisory, consumed = advisory_for(
            provability,
            accepted=current_disposition.upper() == "ACCEPTED",
            authority_known=True,
            seen=seen_before,
        )
        if advisory:
            updated = replace_handoff_extension(
                updated, advisory, marker=(ADVISORY_START, ADVISORY_END),
            )
        if consumed:
            seen = set(getattr(runtime, "_provability_seen_keys", ()))
            seen.update(consumed)
            runtime._provability_seen_keys = seen
    return updated


__all__ = ["basis_for", "observe_closed", "observe_done", "project_handoff"]
