"""Publish an immutable FRAME_ONLY allocation from a qualified candidate."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence

from kernel_frame_optional_allocation import identity, selection_from_frame
from kernel_frame_request_allocation import (
    FrameRequestAllocation,
    FrameRequestAllocationError,
)
from kernel_evictfix_mode import enabled as evictfix_enabled


_TOKEN_FIT_RESERVE = 64
# The existing 922 capture measured 2.46 serialized bytes per prompt token.
# For ASCII JSON this is the initial character ratio; exact tokenizer pairs
# from this run replace it as soon as they exist.
_DEFAULT_CHARS_PER_TOKEN = 2.46


def _evictfix_enabled() -> bool:
    return evictfix_enabled()


def calibrated_estimate(body: Mapping[str, object], meter: object) -> int:
    """Estimate frame tokens from this run's exact pairs during an outage.

    Endpoint admission still uses the meter's conservative fallback count and
    independent byte envelope.
    """
    chars = len(json.dumps(body, sort_keys=True, ensure_ascii=False))
    overhead = getattr(meter, "template_overhead_tokens", 0)
    overhead = overhead if type(overhead) is int and overhead >= 0 else 0
    pair = getattr(meter, "_evictfix_token_pair", None)
    if (isinstance(pair, tuple) and len(pair) == 2
            and all(type(value) is int and value > 0 for value in pair)):
        tokens, measured_chars = pair
        return math.ceil(chars * tokens / measured_chars) + overhead
    return math.ceil(chars / _DEFAULT_CHARS_PER_TOKEN) + overhead


def measured_request_tokens(body: Mapping[str, object], meter: object) -> int:
    """Use the existing request meter for frame pressure and remember exact pairs."""
    tokens, method = meter.measure(body)
    namespace = getattr(meter, "_namespace", None)
    if getattr(meter, "_evictfix_pair_namespace", namespace) != namespace:
        meter._evictfix_token_pair = None
    meter._evictfix_pair_namespace = namespace
    if method == "verified_byte_bound":
        return tokens
    if method != "tokenize":
        return calibrated_estimate(body, meter)
    overhead = getattr(meter, "template_overhead_tokens", 0)
    overhead = overhead if type(overhead) is int and overhead >= 0 else 0
    chars = len(json.dumps(body, sort_keys=True, ensure_ascii=False))
    core = max(1, tokens - overhead)
    if chars:
        prior = getattr(meter, "_evictfix_token_pair", None)
        # Keep the most conservative exact ratio observed in this run.
        if (not isinstance(prior, tuple) or len(prior) != 2
                or core * prior[1] > prior[0] * chars):
            meter._evictfix_token_pair = (core, chars)
    return tokens


def _capture_carrier_roles(
    messages: Sequence[object],
    *,
    state_index: int,
    suffix: Sequence[Mapping[str, object]],
    suffix_start: int,
    exchange_start: int,
    exchange_end: int,
) -> dict[int, str]:
    """Bind message positions to captured semantic roles before using proof locations."""
    suffix_messages = [dict(message) for message in suffix]
    roles: dict[int, str] = {}
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            continue
        if suffix_start <= index < exchange_start:
            if message.get("role") == "user":
                roles[index] = "newest_user_source"
            continue
        if exchange_start <= index < exchange_end:
            if message.get("role") == "tool":
                roles[index] = "newest_tool_result"
            continue
        if state_index < index < suffix_start:
            if message.get("role") == "user":
                roles[index] = "prepared_source"
            continue
        if index >= exchange_end:
            roles[index] = (
                "duplicate_suffix"
                if dict(message) in suffix_messages
                else "trailing_generated"
            )
    return roles


def _unit_carriers(receipt: Mapping[str, object], path: str, byte_start: int, byte_end: int) -> set[int]:
    """Message indexes of exact final-wire proofs whose ONE window contains the unit's bytes."""
    indexes = set()
    all_proofs = receipt.get("delivery_proofs", {})
    path_proofs = all_proofs.get(path, ()) if isinstance(all_proofs, Mapping) else ()
    if (not isinstance(path_proofs, Sequence)
            or isinstance(path_proofs, (str, bytes, bytearray))):
        path_proofs = ()
    for proof in path_proofs:
        if not isinstance(proof, Mapping):
            continue
        raw_key = proof.get("key", {})
        key = raw_key if isinstance(raw_key, Mapping) else {}
        proof_start, proof_end = key.get("byte_start"), key.get("byte_end")
        location_value = proof.get("final_message_location", "")
        if (
            proof.get("wire_contains_exact") is True
            and type(proof_start) is int and proof_start <= byte_start
            and type(proof_end) is int and proof_end >= byte_end
            and isinstance(location_value, str)
        ):
            match = re.fullmatch(
                r"\$\.messages\[(\d+)\]\.content",
                location_value,
            )
            if match is not None:
                indexes.add(int(match.group(1)))
    return indexes


def _obligations_and_locations(
    plan: Mapping[str, object],
    receipt: Mapping[str, object],
    sources: Mapping[str, str],
    carrier_roles: Mapping[int, str],
) -> tuple[list[dict[str, object]], dict[int, list[dict[str, str]]]]:
    """Consume the existing verifier's exact-byte proofs, without a new parser."""
    obligations = []
    locations: dict[int, list[dict[str, str]]] = {}
    deferred = set(receipt.get("deferred_paths", ()))
    plan_deferred = set(plan.get("deferred_paths", ()))
    for unit in plan.get("units", ()):
        path = unit["path"]
        handle = str(
            unit.get("unit_handle") or unit.get("unit_id") or unit.get("handle") or ""
        )
        source = sources.get(path)
        if not handle or not isinstance(source, str):
            raise FrameRequestAllocationError(
                "allocation unit lacks bound source bytes"
            )
        start, end = unit["start_line"], unit["end_line"]
        lines = source.splitlines(keepends=True)
        if (type(start) is not int or type(end) is not int
                or not 1 <= start <= end <= len(lines)):
            raise FrameRequestAllocationError(
                "allocation unit range is outside bound source bytes"
            )
        byte_start = len("".join(lines[: start - 1]).encode("utf-8"))
        byte_end = len("".join(lines[:end]).encode("utf-8"))
        row = copy.deepcopy(dict(unit))
        row.update(
            {
                "handle": handle,
                "revision": plan["revisions"][path],
                "content_digest": "sha256:"
                + hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "ranges": [{"start_line": start, "end_line": end}],
            }
        )
        indexes = set() if path in deferred else _unit_carriers(receipt, path, byte_start, byte_end)
        # ALLOC_DIAG / two-views-1 (CHECKPOINT 10-25, benchmark died9 req 9): a closure unit of a path the PLAN deferred
        # that no single delivered carrier contains stays deferred (read windows of the path are carried as their own
        # unit:visibility:* units); one that a carrier does contain is retained there, so obligations always agree
        # with a COMPLETE receipt (_validate_document: outcome DEFERRED <=> some obligation DEFERRED).
        unit_deferred = path in deferred or (
            not indexes and path in plan_deferred and not handle.startswith("unit:visibility:"))
        row["disposition"] = "DEFERRED" if unit_deferred else "RETAINED"
        obligations.append(row)
        if unit_deferred:
            continue
        if not indexes:
            raise FrameRequestAllocationError(
                "source allocation requires one qualified carrier per unit"
            )
        if any(carrier_roles.get(index) == "trailing_generated" for index in indexes):
            raise FrameRequestAllocationError(
                "unexpected generated source carrier outside the captured layout"
            )
        selected: tuple[int, str] | None = None
        for role in ("newest_user_source", "newest_tool_result", "prepared_source"):
            eligible = sorted(
                index for index in indexes if carrier_roles.get(index) == role
            )
            if eligible:
                selected = (eligible[0], role)
                break
        if selected is None:
            raise FrameRequestAllocationError(
                "source allocation requires one qualified carrier per unit"
            )
        location, role = selected
        locations.setdefault(location, []).append({"handle": handle, "kind": role})
    return obligations, locations


def capture_candidate(
    *,
    runtime: object,
    capture: object,
    body: Mapping[str, object],
    source_plan: Mapping[str, object],
    receipt: Mapping[str, object],
    sources: Mapping[str, str],
    optional_inputs: tuple[dict[str, object], dict[str, object]],
    byte_ceiling: int,
    token_ceiling: int,
    measurement_identity: str,
    previous_allocation: FrameRequestAllocation | None = None,
    request_lineage: str | None = None,
) -> FrameRequestAllocation:
    from kernel_request_source_window import source_delivery_satisfied
    from kernel_stateframe_framefit5 import _state_payload
    from kernel_stateframe_envelope import STATE_HEADER
    from kernel_stateframe_f4 import latest_complete_tool_exchange
    from kernel_frame_allocation_validation import template_environment
    from kernel_request_budget import HARD_MAX_REQUEST_BYTES

    if not source_delivery_satisfied(receipt):
        raise FrameRequestAllocationError(
            "cannot allocate an unqualified source candidate"
        )
    messages = body["messages"]
    state_index = next(
        i
        for i, msg in enumerate(messages)
        if str(msg.get("content") or "").startswith(STATE_HEADER + "\n")
    )
    from kernel_cache_layout_view import semantic_messages
    _payload, state = _state_payload(semantic_messages(messages)[state_index])
    frame = state["framefit5"]
    exchange = latest_complete_tool_exchange(messages)
    if exchange is None:
        raise FrameRequestAllocationError(
            "allocation has an incomplete newest exchange"
        )
    if exchange:
        # Send-time contract messages can trail the volatile tail, so locate
        # the exchange itself instead of assuming a fixed offset.
        exchange_start = next(
            (
                start
                for start in range(len(messages) - len(exchange), -1, -1)
                if all(
                    messages[start + offset] is message
                    for offset, message in enumerate(exchange)
                )
            ),
            None,
        )
        if exchange_start is None:
            raise FrameRequestAllocationError(
                "newest tool exchange moved during allocation"
            )
    else:
        exchange_start = len(messages) - 1
    original_suffix = getattr(capture, "_framefit5_newest_user_suffix", ())
    # Original ownership is captured before the first projection, not rediscovered
    # from a body whose user facts have already moved around the StateFrame.
    suffix = [copy.deepcopy(dict(msg)) for msg in original_suffix]
    suffix_start = exchange_start - len(suffix)
    if (suffix_start < 0 or suffix_start > exchange_start
            or list(messages[suffix_start:exchange_start]) != suffix):
        raise FrameRequestAllocationError(
            "newest user suffix changed during allocation"
        )
    exchange_end = exchange_start + len(exchange)
    carrier_roles = _capture_carrier_roles(
        messages,
        state_index=state_index,
        suffix=suffix,
        suffix_start=suffix_start,
        exchange_start=exchange_start,
        exchange_end=exchange_end,
    )
    obligations, locations = _obligations_and_locations(
        source_plan, receipt, sources, carrier_roles,
    )
    carriers, facts = [], []
    for index in range(state_index + 1, len(messages)):
        msg = messages[index]
        carrier_rows = locations.get(index, [])
        handles = [row["handle"] for row in carrier_rows]
        kind = carrier_roles.get(index)
        if carrier_rows:
            kinds = {row["kind"] for row in carrier_rows}
            if len(kinds) != 1:
                raise FrameRequestAllocationError(
                    "source allocation assigned conflicting semantic carrier roles"
                )
            kind = next(iter(kinds))
        if handles or kind == "newest_tool_result":
            carriers.append(
                {
                    "carrier_id": f"carrier:{index}:{identity(msg)}",
                    "kind": kind,
                    "location": index,
                    "message": copy.deepcopy(dict(msg)),
                    "unit_handles": handles,
                }
            )
        elif index < suffix_start:
            facts.append(
                {
                    "fact_id": f"fact:{index}:{identity(msg)}",
                    "location": index,
                    "message": copy.deepcopy(dict(msg)),
                }
            )
    optional, inventory = selection_from_frame(*optional_inputs, frame)
    binding = runtime.binding()
    old = previous_allocation or getattr(capture, "frame_request_allocation", None)
    tools_digest = identity(body.get("tools", []))
    read_route = None
    if receipt["status"] == "DEFERRED":
        from kernel_completion_owner import _current_tool_admission

        admitted, reason = _current_tool_admission(runtime, "read")
        names = [tool.get("function", {}).get("name") for tool in body.get("tools", [])]
        if not admitted or "read" not in names:
            raise FrameRequestAllocationError(
                f"deferred read admission failed: {reason}"
            )
        read_route = {
            "executable": True,
            "runner_identity": "kernel_tool_surface.runner_dispatchable:read",
            "tool_schema_digest": tools_digest,
            "paths": {
                path: copy.deepcopy(row.get("retrieval_route"))
                for path, row in receipt.get("paths", {}).items()
                if row.get("status") == "DEFERRED"
            },
        }
    return FrameRequestAllocation.create(
        identity={
            "request_lineage": old.identity["request_lineage"]
            if old
            else str(request_lineage or capture.request_id),
            "generation": old.identity["generation"] + 1 if old else 0,
            "workspace_authorization": copy.deepcopy(
                receipt["workspace_authorization"]
            ),
            "product_revision": str(
                binding.get("product_revision") or binding.get("artifact")
            ),
            "task_digest": identity(messages[state_index - 1]),
            "journal_semantic_cutoff": identity(frame.get("action_ledger", {})),
            "ledger_digest": identity(frame.get("action_ledger", {})),
            "newest_exchange_digest": identity(exchange),
        },
        capacity={
            "byte_ceiling": byte_ceiling,
            "token_ceiling": token_ceiling,
            "frame_ceiling": int(runtime.frame_ceiling),
            "measurement_identity": measurement_identity,
            "template_identity": identity(
                {
                    "systems": messages[: state_index - 1],
                    "body_parameters": {
                        k: v for k, v in body.items() if k not in {"messages", "tools"}
                    },
                }
            ),
            "tool_schema_digest": tools_digest,
            "host_byte_ceiling": HARD_MAX_REQUEST_BYTES,
            "system_messages_digest": identity(
                [dict(msg) for msg in messages if msg.get("role") == "system"]
            ),
            "template_environment": template_environment(),
        },
        source_obligations=obligations,
        source_layout={
            "task_message": copy.deepcopy(messages[state_index - 1]),
            "current_user_facts": facts,
            "newest_user_suffix": suffix,
            "newest_user_suffix_start": suffix_start,
            "newest_tool_exchange": copy.deepcopy(list(exchange)),
            "newest_exchange_state": "COMPLETE" if exchange else "NO_TOOL_CALLS",
            "newest_exchange_start": exchange_start,
            "carriers": carriers,
            "source_plan": copy.deepcopy(dict(source_plan)),
        },
        optional_heads=optional,
        site_inventory=inventory,
        outcome={
            "status": receipt["status"],
            "read_route": read_route,
            "final_verification_digest": identity(receipt),
            "body_digest": identity(body),
        },
    )


class FrameFit5ImpossibleMinimum(RuntimeError):
    """A fully assembled, source-qualified minimum cannot fit its ceilings."""

    def __init__(self, detail: Mapping[str, object]):
        super().__init__("FRAMEFIT5_IMPOSSIBLE_MINIMUM")
        self.detail = {"kind": "FRAMEFIT5_IMPOSSIBLE_MINIMUM", **dict(detail)}


def source_refusal(plan, receipt, sources):
    """Keep exact omitted-unit diagnostics off the sanitized exception message."""
    failures = sorted({str(row.get("kind", "UNKNOWN"))
                       for row in receipt.get("proof_failures", ())
                       if isinstance(row, Mapping)})
    missing = []
    for path, row in receipt.get("paths", {}).items():
        if row.get("status") in {"COMPLETE", "DEFERRED"}:
            continue
        omitted = set(row.get("omitted_unit_handles", ()))
        units = [unit for unit in plan.get("units", ()) if unit.get("path") == path]
        for index, unit in enumerate(units):
            handle = str(unit.get("handle", unit.get("unit_id", index)))
            if handle not in omitted:
                continue
            unit_id = str(unit.get("unit_handle") or unit.get("unit_id") or handle)
            bounded = plan.get("bounded_delivery")
            bounded = bounded if isinstance(bounded, Mapping) else {}
            bounded_omitted = unit_id in bounded.get("omitted_handles", ())
            start, end = unit.get("start_line"), unit.get("end_line")
            source = sources.get(path)
            size = None
            if (isinstance(source, str) and type(start) is int and type(end) is int
                    and 1 <= start <= end <= len(source.splitlines(keepends=True))):
                size = len("".join(source.splitlines(keepends=True)[start - 1:end]).encode())
            missing.append({
                "unit_id": unit_id, "receipt_handle": handle, "path": path, "range": [start, end],
                "bytes": size, "tokens_estimate": None if size is None else (size + 3) // 4,
                "token_method": "diagnostic_utf8_bytes_div4_not_admission",
                "delivered_ranges": row.get("ranges", []),
                "reason": ("bounded_source_selection_omitted_unit" if bounded_omitted
                           else "closure_unit_not_covered_by_final_ranges"),
                "declared_deferred": path in plan.get("deferred_paths", ()),
                "verified_deferred": path in receipt.get("deferred_paths", ()),
                "retrieval_route": row.get("retrieval_route"),
            })
    if missing:
        failures.append("closure_units_undelivered")
    receipt_paths = receipt.get("paths", {})
    selected_paths = {
        str(path) for path, row in receipt_paths.items()
        if isinstance(row, Mapping)
        and ((type(row.get("selected_units")) is int and row["selected_units"] > 0)
             or str(row.get("status", "")).upper() not in {"NOT_SELECTED", "UNKNOWN"})
    } if isinstance(receipt_paths, Mapping) else set()

    def omission_is_relevant(row: Mapping[str, object]) -> bool:
        path = str(row.get("path", "")).strip()
        if not path:
            return True
        if path in selected_paths:
            return True
        return str(row.get("status", "")).upper() != "NOT_SELECTED"

    omission_rows = [
        {key: row[key] for key in ("path", "status", "reason", "start_line", "end_line", "unit_handle")
         if key in row}
        for name in ("omissions", "skipped_files") for row in plan.get(name, ())
        if isinstance(row, Mapping) and omission_is_relevant(row)
    ]
    if omission_rows:
        failures.append("closure_omissions_unreconciled")
    if not failures:
        failures.append("source_delivery_" + str(receipt.get("status", "UNKNOWN")).lower())
    error = FrameRequestAllocationError(
        "final allocation source proof is not satisfied: "
        f"status={receipt.get('status')}; proof={receipt.get('proof_status')}; "
        f"failures={','.join(failures)}")
    error.source_diagnostic = {
        "missing_units": missing, "receipt": copy.deepcopy(receipt),
        "declared_deferred_paths": list(plan.get("deferred_paths", ())),
        "failures": failures,
        "omissions": omission_rows,
    }
    return error


def finalize_frame_candidate(
    runtime: object,
    capture: object,
    body: Mapping[str, object],
    *,
    frame_counter: Callable[[Mapping[str, object]], int],
) -> dict[str, object]:
    """Fit a detached candidate; return a proposal without wire authorization."""
    from kernel_request_budget import (
        ENDPOINT_PROMPT_TOKEN_BUDGET,
        FREE_STAGE,
        HARD_MAX_REQUEST_BYTES,
        _graceful_trim_optional_messages,
        _lean_request_limits,
        _request_token_meter,
        _wire_bytes,
    )
    from kernel_request_source_window import (
        verify_final_source_delivery,
        source_delivery_satisfied,
        partial_source_paths_for_deferred_delivery,
    )
    from kernel_stateframe_framefit5 import _state_payload, _workspace_root
    from kernel_stateframe_envelope import STATE_HEADER
    from kernel_frame_optional_allocation import snapshot_optional_inputs

    candidate = copy.deepcopy(body)
    meter = _request_token_meter(runtime, capture)
    from kernel_context import FRAME_CEILING_TOKENS, token_estimate
    # The unconfigured 8K default is the historical local selection estimate,
    # not a provider prompt budget. Explicit frame ceilings opt into measured
    # fitting; endpoint admission always uses the meter independently.
    measured_frame_budget = (
        int(runtime.frame_ceiling) != FRAME_CEILING_TOKENS
        or "STATEFRAME_CEILING" in os.environ
    )
    use_meter_for_frame = (
        _evictfix_enabled() and frame_counter is token_estimate and measured_frame_budget
    )

    def frame_tokens(value: Mapping[str, object]) -> int:
        return (measured_request_tokens(value, meter) if use_meter_for_frame
                else frame_counter(value))

    def fitted_request_tokens(value: Mapping[str, object]) -> int:
        count, method = meter.measure(value)
        # The outage count is a rejection bound for unknown tokenizers. Fit the
        # whole request with the same calibrated estimate used by frame_tokens;
        # the byte envelope remains an independent hard admission limit.
        return measured_request_tokens(value, meter) if method == "calibrated_bytes" else count

    def verified_envelope(method: str) -> bool:
        # A caller-supplied frame counter cannot change the meter's wire proof.
        return _evictfix_enabled() and method in {"tokenize", "verified_byte_bound"}

    active = getattr(capture, "frame_request_allocation", None)
    if active is None:
        active = getattr(capture, "_frame_request_allocation_candidate", None)
    frame = next(
        _state_payload(msg)[1]["framefit5"]
        for msg in candidate["messages"]
        if str(msg.get("content") or "").startswith(STATE_HEADER + "\n")
    )
    optional = (
        (active.optional_heads, active.site_inventory)
        if active is not None
        else snapshot_optional_inputs(frame, _workspace_root(runtime))
    )
    if active is not None:
        from kernel_frame_optional_allocation import (
            render_optional_selection,
            FrameAllocationReplanNeeded,
        )

        if isinstance(active, FrameRequestAllocation):
            active.validate()
            binding = runtime.binding()
            revision = str(binding.get("product_revision") or binding.get("artifact"))
            if active.identity["product_revision"] != revision or active.identity[
                "ledger_digest"
            ] != identity(frame.get("action_ledger", {})):
                raise FrameAllocationReplanNeeded("allocation semantic authority changed")
        from kernel_stateframe_framefit5 import _serialize_state_message
        for message in candidate["messages"]:
            if str(message.get("content") or "").startswith(STATE_HEADER + "\n"):
                payload, state = _state_payload(message)
                state["framefit5"] = render_optional_selection(
                    state["framefit5"], *optional, root=_workspace_root(runtime))
                message["content"] = _serialize_state_message(payload)
    estimated_byte_limit = (
        active.capacity["byte_ceiling"]
        if active is not None
        and active.capacity.get("measurement_identity") != "tokenize"
        else HARD_MAX_REQUEST_BYTES
    )
    token_limit = (
        active.capacity["token_ceiling"]
        if active is not None
        else ENDPOINT_PROMPT_TOKEN_BUDGET
    )

    def request_byte_limit(method: str) -> int:
        # Exact admission already has a finite endpoint envelope. A saved
        # estimate is not a second token ceiling, nor is a previous exact
        # measurement authority to relax a later estimate.
        if method in {"tokenize", "verified_byte_bound"}:
            return _lean_request_limits(candidate, FREE_STAGE, exact_tokenizer=True)[0]
        return estimated_byte_limit

    def effective_tokens() -> int:
        # Preserve the independent frame ceiling as well as endpoint capacity.
        fit_limit = max(1, token_limit - _TOKEN_FIT_RESERVE) if use_meter_for_frame else token_limit
        frame_limit = (max(1, int(runtime.frame_ceiling) - _TOKEN_FIT_RESERVE)
                       if use_meter_for_frame else int(runtime.frame_ceiling))
        return max(
            fitted_request_tokens(candidate),
            frame_tokens(candidate) + fit_limit - frame_limit,
        )

    from kernel_stateframe_f4 import latest_complete_tool_exchange
    semantic = getattr(capture, "_framefit5_semantic_messages", body["messages"])
    from kernel_stateframe_framefit5 import _task_and_latest_turn
    protected = [
        getattr(capture, "_framefit5_task_message", None) or _task_and_latest_turn(semantic)[0],
        *getattr(capture, "_framefit5_newest_user_suffix", ()),
        *(latest_complete_tool_exchange(semantic) or ()),
    ]
    plan = getattr(capture, "_request_source_closure_override", None)
    if not isinstance(plan, Mapping):
        plan = getattr(runtime, "_request_source_closure", None)
    if not isinstance(plan, Mapping):
        raise FrameRequestAllocationError("final allocation lacks source obligations")
    required_paths = tuple(unit["path"] for unit in plan.get("units", ())
                           if isinstance(unit, Mapping) and isinstance(unit.get("path"), str))
    required_paths = tuple(dict.fromkeys((*required_paths,
        *getattr(capture, "_visibility_required_paths", ()))))
    def trim_optional_candidate() -> int:
        nonlocal semantic, protected
        from kernel_stateframe_capacity import fit_protected_frame
        from kernel_execution_reply_budget import fit_execution_replies, project_execution_delivery

        fit_execution_replies(
            candidate, max_request_bytes=request_byte_limit(meter.measure(candidate)[1]),
            measure_bytes=lambda: _wire_bytes(candidate),
            max_request_tokens=(max(1, token_limit - _TOKEN_FIT_RESERVE)
                                if use_meter_for_frame else token_limit),
            measure_tokens=effective_tokens,
        )
        semantic = project_execution_delivery(semantic, candidate["messages"])
        protected = project_execution_delivery(protected, candidate["messages"])

        def compact_history(*, allow_read_body_eviction: bool = False) -> None:
            method = meter.measure(candidate)[1]
            fit_limit = max(1, token_limit - _TOKEN_FIT_RESERVE) if use_meter_for_frame else token_limit
            frame_limit = (max(1, int(runtime.frame_ceiling) - _TOKEN_FIT_RESERVE)
                           if use_meter_for_frame else int(runtime.frame_ceiling))
            audit = fit_protected_frame(
                candidate, semantic_messages=semantic,
                action_context=getattr(runtime, "_framefit_action_context", None),
                protected_messages=protected,
                required_deferred_paths=(
                    plan.get("deferred_paths", ())
                    if isinstance(plan.get("deferred_paths", ()), (list, tuple)) else ()
                ),
                max_wire_bytes=request_byte_limit(method),
                max_request_tokens=fit_limit,
                local_request_tokens=lambda value: (
                    frame_tokens(value) + fit_limit - frame_limit),
                measure_request_tokens=fitted_request_tokens,
                exact_tokenizer=verified_envelope(method),
                allow_read_body_eviction=allow_read_body_eviction,
            )
            journal = getattr(capture, "journal", None)
            emit = getattr(journal, "emit", None)
            if audit.get("changed") and callable(emit):
                emit("framefit5_capacity_compacted", request_id=capture.request_id,
                     before_bytes=audit["before"], after_bytes=audit["after"],
                     protected_byte_limit=audit["limit"],
                     capacity_measurement_calls=audit["measurement_calls"],
                     tokenizer_calls=meter.journal_fields(meter.measure_method)["tokenizer_calls"],
                     dispatch=False)

        # A meter may become unavailable while a changed candidate is fitted.
        # It falls back for this request, so at most one repeat under the offline
        # byte ceiling is needed before source qualification.
        for _ in range(2):
            compact_history(allow_read_body_eviction=False)
            limit = request_byte_limit(meter.measure(candidate)[1])
            _graceful_trim_optional_messages(
                candidate,
                protected_contents=(),
                protected_messages=protected,
                required_source_paths=required_paths,
                max_request_bytes=limit,
                max_request_tokens=(max(1, token_limit - _TOKEN_FIT_RESERVE)
                                    if use_meter_for_frame else token_limit),
                measure_request_tokens=effective_tokens,
            )
            # Head/site overhead and read-result truth share this same protected
            # envelope. First compact losslessly, then spend optional source
            # overhead, then allow an owed read body to retire under pressure.
            from kernel_stateframe_capacity import fit_protected_source_overhead
            fit_protected_source_overhead(
                candidate, protected_messages=protected,
                action_context=getattr(runtime, "_framefit_action_context", None),
                exact_tokenizer=verified_envelope(meter.measure(candidate)[1]),
            )
            compact_history(allow_read_body_eviction=True)
            current_limit = request_byte_limit(meter.measure(candidate)[1])
            if current_limit == limit:
                break
        return current_limit

    byte_limit = trim_optional_candidate()
    state_message = next(
        message
        for message in candidate["messages"]
        if str(message.get("content") or "").startswith(STATE_HEADER + "\n")
    )
    payload, state = _state_payload(state_message)
    frame = state["framefit5"]
    from kernel_reread_result_truth import read_is_owed
    needs_bookmarks = bool(getattr(runtime, "_code_bookmark_carry", ())) or any(
        isinstance(row, Mapping) and row.get("body_delivery") == "not_shown"
        and read_is_owed(row)
        for row in frame.get("read_facts", ())
    )
    bookmarks = []
    if needs_bookmarks:
        from kernel_code_bookmarks import collect as collect_code_bookmarks
        bookmarks = collect_code_bookmarks(runtime, frame)
        if bookmarks:
            from kernel_stateframe_framefit5 import _serialize_state_message
            frame["code_bookmarks"] = bookmarks
            state_message["content"] = _serialize_state_message(payload)
    from kernel_frame_source_snapshot import (
        fit_recent_read_windows, public_check_source_windows,
        recent_editable_read_windows,
    )
    recent_windows = recent_editable_read_windows(runtime, frame)
    ordinary_recent_windows = recent_windows
    public_windows = public_check_source_windows(runtime, frame)
    bookmarks = frame.get("code_bookmarks")
    bookmark_reconciliation = isinstance(bookmarks, list) and bool(bookmarks)
    if isinstance(bookmarks, list) and bookmarks:
        from kernel_code_bookmarks import resolve as resolve_bookmark, windows as bookmark_windows
        from kernel_frame_optional_allocation import FrameAllocationReplanNeeded
        root = _workspace_root(runtime)
        for row in bookmarks:
            if not isinstance(row, Mapping):
                continue
            current = resolve_bookmark(root, row)
            if current is not None and current.get("revision") != row.get("revision"):
                raise FrameAllocationReplanNeeded("bookmarked source changed before allocation")
        selected = bookmark_windows(root, bookmarks)
        if selected:
            # Keep ordinary ranges even where they overlap: if fitting demotes
            # a bookmark, its earlier read is still a fallback candidate.
            # The fitter deduplicates exact carriers and delivery truth merges
            # coverage by path, revision and range on the final wire.
            recent_windows = [*selected, *recent_windows]
    plan = getattr(capture, "_request_source_closure_override", None)
    if not isinstance(plan, Mapping):
        plan = getattr(runtime, "_request_source_closure", None)
    if not isinstance(plan, Mapping):
        raise FrameRequestAllocationError("final allocation lacks source obligations")
    plan = copy.deepcopy(dict(plan))
    from kernel_frame_source_snapshot import frame_source_snapshot
    sources = frame_source_snapshot(runtime, plan)
    from kernel_completion_owner import _current_tool_admission
    from kernel_request_source_window import _executable_deferred_read_route
    from kernel_stateframe_framefit5 import assemble_framefit5_request
    from kernel_request_budget import _remove_deferred_source_excerpts

    def assemble_selection():
        candidate["messages"] = assemble_framefit5_request(
            candidate["messages"], semantic,
            task_message=getattr(capture, "_framefit5_task_message", None),
            source_plan=plan, sources=sources, runtime=runtime,
            owner_contract=getattr(capture, "request_action_contract", None),
            owner_source_messages=body["messages"],
            owner_presentation=getattr(capture, "owner_presentation", None),
            host_text_messages=candidate["messages"],
        )

    # Closure deferral does not discard verified editable read pages before the
    # complete-envelope fit. Their other units remain deferred; actual pressure
    # below can remove these host copies without touching newest input.
    visibility_paths = set(getattr(capture, "_visibility_required_paths", ()))
    visibility_paths.update(row["path"] for row in recent_windows)
    if plan.get("deferred_paths"):
        _remove_deferred_source_excerpts(
            candidate["messages"], [path for path in plan["deferred_paths"]
                                    if path not in visibility_paths],
            preserve_stateframe=True, protected_messages=protected,
        )
        assemble_selection()
        # Reassembly can add source notices and durable coverage proofs while
        # repairing evicted read carriers. Fit those exact bytes before deciding
        # that the remaining source-qualified candidate is irreducible.
        byte_limit = trim_optional_candidate()

    read_available = _current_tool_admission(runtime, "read")[0] and any(
        tool.get("function", {}).get("name") == "read"
        for tool in candidate.get("tools", [])
    )
    from kernel_stateframe_capacity import within_protected_capacity

    def source_windows_fit(trial: Mapping[str, object]) -> bool:
        if bookmark_reconciliation and isinstance(trial.get("messages"), list):
            # The selected source windows and the typed visibility claim are
            # one measured proposal. Reconcile before admission, never after.
            from kernel_reread_result_truth import refresh_messages
            from kernel_stateframe_framefit5 import FrameFit5Error
            try:
                refresh_messages(
                    trial["messages"],
                    required_deferred_paths=tuple(plan.get("deferred_paths", ())),
                )
            except (FrameFit5Error, TypeError, ValueError):
                return False  # Optional source advice cannot reject the base.
        _count, trial_method = meter.measure(trial)
        byte_limit_for_windows = request_byte_limit(trial_method)
        # A bound release prefix must stay within its standard wire cap;
        # standalone exact-tokenized proposals may use the verified bound.
        if getattr(capture, "_release_prefix_boundary", None) is not None:
            byte_limit_for_windows = min(byte_limit_for_windows, HARD_MAX_REQUEST_BYTES)
        return (
            _wire_bytes(trial) <= byte_limit_for_windows
            and fitted_request_tokens(trial) <= token_limit
            and frame_tokens(trial) <= runtime.frame_ceiling
            and within_protected_capacity(
                trial, protected, exact_tokenizer=verified_envelope(trial_method),
            )
        )

    recent_paths = {row["path"] for row in recent_windows}

    def retained_reads_fit() -> bool:
        if not recent_windows:
            return True
        # Check the full current read frontier before accepting a base request.
        # Its closure carriers can be deferred through the verified read route;
        # spending their bytes first makes retained pages disappear on the wire.
        full_reads = fit_recent_read_windows(
            candidate, recent_windows,
            protected_messages=protected,
            newest_user_suffix=getattr(capture, "_framefit5_newest_user_suffix", ()),
            deferred_source_paths=tuple(plan.get("deferred_paths", ())),
            fits=source_windows_fit,
        )
        contents = {message.get("content") for message in full_reads["messages"]
                    if isinstance(message, Mapping) and isinstance(message.get("content"), str)}
        return all(window["content"] in contents
                   or window.get("compact_content") in contents
                   for window in recent_windows)

    seen: set[tuple[tuple[str, ...], str]] = set()
    # Each selected path may first defer its omitted closure units and later
    # lose its retained read pages under pressure. The initial candidate may
    # also contain retained pages before any path is deferred.
    for _attempt in range(2 * len(plan.get("units", ())) + 3):
        receipt = verify_final_source_delivery(
            plan,
            candidate["messages"],
            # Existing omission reconciliation requires proof over exact wire
            # bytes, including COMPLETE paths alongside deferred closures.
            payload=json.dumps(candidate).encode(),
            sources=sources,
        )
        if not source_delivery_satisfied(receipt):
            # The bounded source selector can omit units before an allocation
            # proposal exists. Reuse its existing verified-partial authority;
            # byte/revision failures are still terminal, never repair triggers.
            partial_paths = tuple(
                path for path in partial_source_paths_for_deferred_delivery(plan, receipt)
                if path not in plan.get("deferred_paths", ())
            ) if read_available else ()
            if not partial_paths:
                raise source_refusal(plan, receipt, sources)
            plan["deferred_paths"] = list(dict.fromkeys((
                *plan.get("deferred_paths", ()), *partial_paths)))
            _remove_deferred_source_excerpts(
                candidate["messages"], [path for path in partial_paths if path not in visibility_paths],
                preserve_stateframe=True, protected_messages=protected,
            )
            assemble_selection()
            byte_limit = trim_optional_candidate()
            # Reprove the rebuilt notice and exact bytes before any capacity
            # acceptance. CURRENT read pages remain eligible until real pressure.
            continue
        from kernel_step_contract import reconcile_final_source_access
        if reconcile_final_source_access(candidate["messages"], receipt):
            # The StateFrame is part of the measured and source-proven body.
            # Rebind both receipts after the truthful source-access rewrite.
            receipt = verify_final_source_delivery(
                plan, candidate["messages"],
                payload=json.dumps(candidate).encode(), sources=sources,
            )
            if not source_delivery_satisfied(receipt):
                raise source_refusal(plan, receipt, sources)
        rejection_bound, method = meter.measure(candidate)
        tokens = fitted_request_tokens(candidate) if method == "calibrated_bytes" else rejection_bound
        byte_limit = request_byte_limit(method)
        request_bytes = _wire_bytes(candidate)
        frame_token_count = frame_tokens(candidate)
        from kernel_stateframe_capacity import (
            protected_bytes, protected_limit,
        )
        base_fits = (
            request_bytes <= byte_limit
            and tokens <= token_limit
            and frame_token_count <= runtime.frame_ceiling
            and within_protected_capacity(candidate, protected,
                                          exact_tokenizer=verified_envelope(method))
        )
        if base_fits and retained_reads_fit():
            break
        if not base_fits:
            # Retained pages are optional, even when they arrived in the base
            # candidate. Fit newest first, then reprove the exact wire before
            # considering a mandatory-minimum refusal or source deferral.
            without_excess_reads = fit_recent_read_windows(
                candidate, recent_windows,
                protected_messages=protected,
                newest_user_suffix=getattr(capture, "_framefit5_newest_user_suffix", ()),
                deferred_source_paths=tuple(plan.get("deferred_paths", ())),
                fits=source_windows_fit,
            )
            if without_excess_reads != candidate:
                candidate = without_excess_reads
                continue
            if bookmarks:
                # No optional source selection may create a new minimum. Retry
                # the measured base with ordinary reads and no bookmark state.
                state_message = next(message for message in candidate["messages"]
                                     if str(message.get("content") or "").startswith(STATE_HEADER + "\n"))
                payload, state = _state_payload(state_message)
                state["framefit5"].pop("code_bookmarks", None)
                state_message["content"] = _serialize_state_message(payload)
                bookmarks = []
                recent_windows = ordinary_recent_windows
                from kernel_reread_result_truth import refresh_messages
                refresh_messages(candidate["messages"],
                                 required_deferred_paths=tuple(plan.get("deferred_paths", ())))
                candidate = fit_recent_read_windows(
                    candidate, recent_windows, protected_messages=protected,
                    newest_user_suffix=getattr(capture, "_framefit5_newest_user_suffix", ()),
                    deferred_source_paths=tuple(plan.get("deferred_paths", ())),
                    fits=source_windows_fit,
                )
                continue
        current = tuple(plan.get("deferred_paths", ()))
        if not base_fits and current:
            # An editable path can be deferred while its earlier visibility
            # copies remain on the draft. Spend those host copies only after
            # the complete wire fails to fit; the newest exchange stays pinned
            # and the rebuilt candidate is source-proved on the next pass.
            without_deferred_copies = copy.deepcopy(candidate)
            removed = _remove_deferred_source_excerpts(
                without_deferred_copies["messages"], current,
                preserve_stateframe=True, protected_messages=protected,
            )
            if removed:
                candidate = without_deferred_copies
                assemble_selection()
                byte_limit = trim_optional_candidate()
                continue
        selection = (current, identity(candidate["messages"]))
        if selection in seen:
            raise FrameRequestAllocationError("repeated final allocation candidate")
        seen.add(selection)
        eligible = []
        if read_available:
            for path, row in receipt.get("paths", {}).items():
                if path in current:
                    continue
                # COMPLETE delivery has no retrieval route because nothing was
                # omitted. Ask the existing closure authority for this candidate's
                # route after omission, on a detached plan; never interpret a
                # missing COMPLETE route as proof that the source is mandatory.
                from kernel_request_evidence_closure import record_source_delivery
                route_plan = copy.deepcopy(plan)
                route = record_source_delivery(route_plan, path, ()).get("retrieval_route")
                if not _executable_deferred_read_route(
                    path,
                    row.get("revision"),
                    route,
                ):
                    continue
                indexes = []
                for location in row.get("final_message_locations", ()):
                    match = re.fullmatch(r"\$\.messages\[(\d+)\]\.content", location)
                    if match is not None:
                        indexes.append(int(match.group(1)))
                if indexes and any(
                    candidate["messages"][idx] not in protected for idx in indexes
                ):
                    eligible.append(path)
        if not eligible:
            # No verified retrieval route can free more source bytes. Preserve
            # the fitting base; the read fitter below keeps the largest prefix
            # of recent pages that the actual remaining capacity permits.
            if base_fits:
                break
            raise FrameFit5ImpossibleMinimum(
                {
                    "request_bytes": request_bytes,
                    "prompt_tokens": None if method == "verified_byte_bound" else tokens,
                    "prompt_tokens_rejection_bound": tokens if method == "verified_byte_bound" else None,
                    "measure_method": method,
                    "frame_tokens": frame_token_count,
                    "max_request_bytes": byte_limit,
                    "max_prompt_tokens": token_limit,
                    "frame_ceiling": runtime.frame_ceiling,
                    "read_unavailable": not read_available,
                    "protected_request_bytes": protected_bytes(candidate, protected),
                    "protected_byte_limit": protected_limit(
                        candidate, exact_tokenizer=verified_envelope(method)),
                    "protected_component_bytes": [
                        _wire_bytes(msg) for msg in candidate["messages"]
                    ],
                }
            )
        eligible.sort(key=lambda path: path in recent_paths)
        plan["deferred_paths"] = list(dict.fromkeys((*current, eligible[0])))
        _remove_deferred_source_excerpts(
            candidate["messages"],
            plan["deferred_paths"],
            preserve_stateframe=True,
            protected_messages=protected,
        )
        assemble_selection()
        byte_limit = trim_optional_candidate()
    else:
        raise FrameRequestAllocationError(
            "final allocation search exceeded finite candidates"
        )
    if recent_windows or public_windows:
        candidate = fit_recent_read_windows(
            candidate, recent_windows, public_checks=public_windows,
            protected_messages=protected,
            newest_user_suffix=getattr(capture, "_framefit5_newest_user_suffix", ()),
            deferred_source_paths=tuple(plan.get("deferred_paths", ())),
            fits=source_windows_fit,
        )
        receipt = verify_final_source_delivery(
            plan, candidate["messages"], payload=json.dumps(candidate).encode(),
            sources=sources,
        )
        if not source_delivery_satisfied(receipt):
            raise source_refusal(plan, receipt, sources)
        tokens, method = meter.measure(candidate)
        byte_limit = request_byte_limit(method)
    # Carry the same source input through allocation and replacement snapshots.
    plan["source_contents"] = copy.deepcopy(sources)
    capture._frame_candidate_measurement = meter.bind_measurement(
        candidate, tokens=tokens, method=method, request_id=str(capture.request_id),
        max_request_bytes=byte_limit, max_prompt_tokens=token_limit,
    )
    capture._frame_candidate_inputs = {
        "source_plan": copy.deepcopy(plan), "receipt": copy.deepcopy(receipt),
        "sources": copy.deepcopy(sources), "optional_inputs": copy.deepcopy(optional),
        "byte_ceiling": byte_limit, "token_ceiling": token_limit,
        "measurement_identity": method,
    }
    capture._frame_candidate_fallback_fit = method == "calibrated_bytes"
    capture._frame_candidate_rejection_bound = (
        rejection_bound if method == "calibrated_bytes" else None
    )
    capture._request_source_closure_override = copy.deepcopy(plan)
    capture.request_source_final_delivery = receipt
    capture.request_budget_exceeded = False
    final_state = next(
        _state_payload(message)[1]["framefit5"]
        for message in candidate["messages"]
        if str(message.get("content") or "").startswith(STATE_HEADER + "\n")
    )
    final_bookmarks = final_state.get("code_bookmarks")
    bookmarked_paths = (
        {row["path"] for row in final_bookmarks
         if isinstance(row, Mapping) and isinstance(row.get("path"), str)}
        if isinstance(final_bookmarks, list) else set()
    )
    from kernel_request_source_window import parse_model_source_window
    carried_source_paths = {
        parsed["path"]
        for message in candidate["messages"]
        if isinstance(message, Mapping)
        and isinstance(message.get("content"), str)
        and "[code bookmark]" in message["content"]
        if (parsed := parse_model_source_window(message["content"])) is not None
        and isinstance(parsed.get("path"), str)
    }
    active = bool(bookmarked_paths & carried_source_paths) or any(
        isinstance(fact, Mapping) and fact.get("path") in bookmarked_paths
        and fact.get("body_delivery") in {"present", "bookmarked"}
        for fact in final_state.get("read_facts", ())
    )
    capture._code_bookmark_transition = (
        active, copy.deepcopy(final_bookmarks) if active
        and isinstance(final_bookmarks, list) else []
    )
    return candidate
