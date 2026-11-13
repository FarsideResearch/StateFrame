"""Publish an immutable FRAME_ONLY allocation from a qualified candidate."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

from kernel_frame_optional_allocation import identity, selection_from_frame
from kernel_body_serialization_memo import canonical_digest, plain_wire, sorted_chars
from kernel_frame_request_allocation import (
    FrameRequestAllocation,
    FrameRequestAllocationError,
)
from kernel_evictfix_mode import enabled as evictfix_enabled


_TOKEN_FIT_RESERVE = 64
_FINAL_LAYOUT_MAX_EXTRA_ALLOCATION_PASSES = 2
# Mirrors kernel_cache_layout_candidate.MAX_STABLE_TRIMS (not imported: that module loads after this one).
_MAX_FINAL_LAYOUT_TRIMS = 3
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
    from kernel_request_diet import _density_floor_tokens
    return max(math.ceil(chars / _DEFAULT_CHARS_PER_TOKEN) + overhead,
               _density_floor_tokens(body, overhead))


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
    _remember_exact_pair(body, meter, tokens)
    return tokens


def _measure_source_candidate(
        meter: object, candidate: Mapping[str, object], serialized_bytes: int,
) -> tuple[int, str]:
    """Use the prepared wire size without bypassing meter state transitions."""
    from kernel_request_diet import RequestTokenMeter

    measure_with_wire = getattr(meter, "_measure_with_wire_bytes", None)
    instance_methods = getattr(meter, "__dict__", {})
    if (callable(measure_with_wire) and type(meter) is RequestTokenMeter
            and "measure" not in instance_methods
            and "_measure_with_wire_bytes" not in instance_methods):
        return measure_with_wire(candidate, serialized_bytes)
    return meter.measure(candidate)


def _source_candidate_nonfit(
        meter: object, candidate: Mapping[str, object], serialized_bytes: int,
        request_byte_limit: Callable[[str], int], *,
        hard_max_request_bytes: int, clip_to_hard_limit: bool,
) -> tuple[bool, tuple[int, str] | None]:
    """Skip proven nonfits only after their normal meter transition has run."""
    from kernel_request_diet import RequestTokenMeter

    instance_methods = getattr(meter, "__dict__", {})
    if (type(meter) is not RequestTokenMeter
            or "measure" in instance_methods
            or "_measure_with_wire_bytes" in instance_methods):
        return False, None

    # Every proposal follows the ordinary meter path before byte-cap rejection;
    # this preserves deadline, calibration, outage, and re-probe updates.
    measurement = _measure_source_candidate(meter, candidate, serialized_bytes)
    byte_limit = request_byte_limit(measurement[1])
    if clip_to_hard_limit:
        byte_limit = min(byte_limit, hard_max_request_bytes)
    return serialized_bytes > byte_limit, measurement


def _remember_exact_pair(body: Mapping[str, object], meter: object, tokens: int) -> None:
    overhead = getattr(meter, "template_overhead_tokens", 0)
    overhead = overhead if type(overhead) is int and overhead >= 0 else 0
    chars = sorted_chars(body)
    core = max(1, tokens - overhead)
    if chars:
        prior = getattr(meter, "_evictfix_token_pair", None)
        # Keep the most conservative exact ratio observed in this run.
        if (not isinstance(prior, tuple) or len(prior) != 2
                or core * prior[1] > prior[0] * chars):
            meter._evictfix_token_pair = (core, chars)


def _calibrated_ratio(meter: object) -> float:
    pair = getattr(meter, "_evictfix_token_pair", None)
    if (isinstance(pair, tuple) and len(pair) == 2
            and all(type(value) is int and value > 0 for value in pair)):
        return pair[1] / pair[0]
    return _DEFAULT_CHARS_PER_TOKEN


def _serialized_chars(body: Mapping[str, object]) -> int:
    return sorted_chars(body)


def _body_identity(body: Mapping[str, object]) -> str:
    return canonical_digest(body, identity)


def _copy_source_plan_snapshot(plan: Mapping[str, object]) -> dict[str, object]:
    """Detach mutable source-plan metadata while sharing its read-only text map."""
    metadata = {key: value for key, value in plan.items() if key != "source_contents"}
    copied = copy.deepcopy(metadata)
    source_contents = plan.get("source_contents")
    if "source_contents" in plan:
        # Proof and layout consumers only read path→text snapshots. Sharing this
        # map avoids recursively copying large file bodies at each finalize seam.
        if (isinstance(source_contents, dict)
                and all(isinstance(path, str) and isinstance(text, str)
                        for path, text in source_contents.items())):
            copied["source_contents"] = source_contents
        else:
            copied["source_contents"] = copy.deepcopy(source_contents)
    return copied


def _prepared_measurement_wire(meter: object, wire: bytes | None) -> dict[str, bytes]:
    if wire is None:
        return {}
    from kernel_request_diet import RequestTokenMeter

    if (type(meter) is RequestTokenMeter
            and "bind_measurement" not in getattr(meter, "__dict__", {})):
        return {"wire": wire}
    return {}


def _copy_source_delivery_route_plan(plan: Mapping[str, object]) -> dict[str, object]:
    """Isolate the maps mutated by route probing without cloning source text."""
    route_plan = dict(plan)
    delivery = plan.get("delivery")
    if isinstance(delivery, dict):
        route_plan["delivery"] = dict(delivery)
    coverage = plan.get("coverage")
    if isinstance(coverage, dict):
        route_coverage = dict(coverage)
        for span_name in ("delivered_span", "acquired_span"):
            span = coverage.get(span_name)
            if isinstance(span, dict):
                route_coverage[span_name] = dict(span)
        route_plan["coverage"] = route_coverage
    return route_plan


@dataclass(frozen=True, slots=True)
class FrozenRequestCounter:
    """The allocation's count and ratio remain authoritative for this request."""

    method: str
    tokens: int
    chars: int
    ratio: float
    body_digest: str
    tokenizer_identity: str
    request_id: str
    logical_request_id: str

    @classmethod
    def capture(cls, body: Mapping[str, object], meter: object,
                tokens: int, method: str, request_id: str) -> FrozenRequestCounter:
        from kernel_frame_optional_allocation import identity
        return cls(method, tokens, _serialized_chars(body),
                   _calibrated_ratio(meter), _body_identity(body),
                   meter.tokenizer_identity(body), request_id,
                   meter._request_id or request_id)


def final_request_tokens(body: Mapping[str, object], meter: object,
                         capture: object, *, wire_bytes: int | None = None) -> tuple[int, str]:
    """Count the final body with the allocation's frozen measurement method.

    A changed exact-counted body is tokenized when possible. If tokenize has
    failed, only its size delta is estimated on top of the fitted exact count.
    The independent byte envelope bounds serialized growth. If tokenize
    recovers, the proof also checks its exact count for a real undercount; the
    conservative fallback rejection bound remains diagnostic during an outage.
    ``wire_bytes`` carries the already-serialized transport body's byte length
    through the meter without changing its token or frozen-counter authority.
    """
    if wire_bytes is not None and (type(wire_bytes) is not int or wire_bytes < 0):
        raise ValueError("prepared request byte count must be a non-negative integer")
    request_meter_type = None
    if wire_bytes is not None:
        from kernel_request_diet import RequestTokenMeter

        request_meter_type = RequestTokenMeter

    def measure_body() -> tuple[int, str]:
        instance_methods = getattr(meter, "__dict__", {})
        if (request_meter_type is not None and type(meter) is request_meter_type
                and "measure" not in instance_methods
                and "_measure_with_wire_bytes" not in instance_methods):
            return meter._measure_with_wire_bytes(body, wire_bytes)
        return meter.measure(body)

    frozen = getattr(capture, "_frame_candidate_counter", None)
    if not isinstance(frozen, FrozenRequestCounter):
        from kernel_request_measurement import RequestMeasurement
        carried = getattr(capture, "_frame_candidate_measurement", None)
        request_id = str(getattr(capture, "request_id", ""))
        if (isinstance(carried, RequestMeasurement)
                and carried.matches(body, tokenizer_identity=meter.tokenizer_identity(body),
                                    request_id=request_id,
                                    logical_request_id=meter._request_id or request_id)
                and carried.method == meter.measure_method
                and carried.prompt_tokens is not None):
            return carried.prompt_tokens, carried.method
        return measure_body()
    request_id = str(getattr(capture, "request_id", ""))
    if (frozen.tokenizer_identity != meter.tokenizer_identity(body)
            or frozen.request_id != request_id
            or frozen.logical_request_id != (meter._request_id or request_id)):
        raise FrameRequestAllocationError("frozen request counter identity changed")
    if frozen.method == "verified_byte_bound":
        # This is an oversize rejection sentinel, not an allocation count.
        # Freeze the real method only after trimming makes the body measurable.
        tokens, method = measure_body()
        if method == "calibrated_bytes":
            tokens = calibrated_estimate(body, meter)
        elif method == "tokenize":
            _remember_exact_pair(body, meter, tokens)
        if method != "verified_byte_bound":
            capture._frame_candidate_counter = FrozenRequestCounter.capture(
                body, meter, tokens, method, request_id)
        return tokens, method
    if _body_identity(body) == frozen.body_digest:
        return frozen.tokens, frozen.method
    if frozen.method == "tokenize":
        tokens, method = measure_body()
        if method in {"tokenize", "verified_byte_bound"}:
            if method == "tokenize":
                _remember_exact_pair(body, meter, tokens)
            return tokens, frozen.method
    delta = _serialized_chars(body) - frozen.chars
    return max(0, frozen.tokens + math.ceil(delta / frozen.ratio)), frozen.method


def carry_final_request_tokens(body: Mapping[str, object], capture: object,
                               tokens: int) -> None:
    """Bind an admitted projection to the original method and ratio."""
    frozen = getattr(capture, "_frame_candidate_counter", None)
    if isinstance(frozen, FrozenRequestCounter):
        capture._frame_candidate_counter = replace(
            frozen, tokens=tokens, chars=_serialized_chars(body),
            body_digest=_body_identity(body))


def rebind_final_layout_trim(body: Mapping[str, object], meter: object,
                             capture: object, *, byte_limit: int,
                             token_limit: int, wire: bytes | None = None,
                             replay_selection: bool = False) -> tuple[int, str]:
    """Rebind a trim's proof; recorded selections never ask the meter to refit."""
    if replay_selection:
        frozen = capture._frame_candidate_counter
        tokens, method = frozen.tokens, frozen.method
    else:
        tokens, method = final_request_tokens(
            body, meter, capture, wire_bytes=(len(wire) if wire is not None else None),
        )
    carry_final_request_tokens(body, capture, tokens)
    capture._frame_candidate_measurement = meter.bind_measurement(
        body, tokens=tokens, method=method, request_id=str(capture.request_id),
        max_request_bytes=byte_limit, max_prompt_tokens=token_limit,
        **_prepared_measurement_wire(meter, wire))
    capture._frame_candidate_inputs["measurement_identity"] = method
    capture._frame_candidate_fallback_fit = method == "calibrated_bytes"
    if method == "calibrated_bytes":
        rejection_measurement = ((tokens, method) if replay_selection else (
            _measure_source_candidate(meter, body, len(wire))
            if wire is not None else meter.measure(body)
        ))
        capture._frame_candidate_rejection_bound = rejection_measurement[0]
    else:
        capture._frame_candidate_rejection_bound = None
    return tokens, method


def _final_layout_trim_reached_target(
        body: Mapping[str, object], meter: object, capture: object, *,
        token_limit: int, byte_limit: int) -> bool:
    """Check the plain body against both limits requested by a layout trim."""
    wire = json.dumps(body).encode()
    tokens, _method = final_request_tokens(
        body, meter, capture, wire_bytes=len(wire),
    )
    return tokens <= token_limit and len(wire) <= byte_limit


def _final_layout_reallocation_enabled() -> bool:
    """Keep fitted-growth retries out of stable, page-log, and replay requests."""
    if os.environ.get("STATEFRAME_FINAL_CACHE_LAYOUT", "on").strip().lower() != "on":
        return False
    from kernel_cache_layout_candidate import final_layout_mode
    from kernel_stateframe_page_log import mode as page_log_mode
    from kernel_stable_wire import mode as stable_wire_mode

    candidate_module = sys.modules["kernel_cache_layout_candidate"]
    return (
        final_layout_mode() == "on"
        and page_log_mode() == "off"
        and stable_wire_mode() == "off"
        and getattr(candidate_module, "_REPLAY_LAYOUT_DECISION_PROVIDER", None) is None
        and getattr(candidate_module, "_REPLAY_STABLE_WIRE_DECISION_PROVIDER", None) is None
    )


def _next_final_layout_reserve(
        *, retry_pass: int, current_reserve: int, measured_growth: int,
        projected_tokens: int, token_ceiling: int) -> int | None:
    """Return a strictly larger reserve for one of the two bounded retries."""
    if retry_pass >= _FINAL_LAYOUT_MAX_EXTRA_ALLOCATION_PASSES:
        return None
    if projected_tokens <= token_ceiling:
        return None
    next_reserve = max(current_reserve, measured_growth, 0)
    return next_reserve if next_reserve > current_reserve else None


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
    location_map: Mapping[int, int] | None = None,
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
        physical_indexes = set() if path in deferred else _unit_carriers(receipt, path, byte_start, byte_end)
        indexes = {location_map[index] for index in physical_indexes} if location_map is not None else physical_indexes
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
    physical_messages = body["messages"]
    from kernel_cache_layout_view import FINAL_LAYOUT_NAME, semantic_index_map, semantic_messages
    final_layout = bool(physical_messages and physical_messages[-1].get("name") == FINAL_LAYOUT_NAME)
    location_map = semantic_index_map(physical_messages) if final_layout else None
    messages = semantic_messages(physical_messages) if final_layout else physical_messages
    state_index = next(
        i
        for i, msg in enumerate(messages)
        if str(msg.get("content") or "").startswith(STATE_HEADER + "\n")
    )
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
        # Something between capture and allocation inserted, moved or edited a
        # row of the newest user turn. The request is unsent: re-prepare it once
        # from the workspace (logged) instead of ending the run.
        from kernel_frame_optional_allocation import FrameAllocationReplanNeeded
        raise FrameAllocationReplanNeeded("newest user suffix changed during allocation")
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
        source_plan, receipt, sources, carrier_roles, location_map,
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

    from kernel_request_evidence_closure import omission_blocks_delivery

    def omission_is_relevant(row: Mapping[str, object]) -> bool:
        return omission_blocks_delivery(row, selected_paths)

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
    token_ceiling: int | None = None,
    byte_ceiling: int | None = None,
    _layout_retry_pass: int = 0,
    _layout_reserve_floor: int = 0,
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
    from kernel_request_diet import RequestTokenMeter
    from kernel_request_source_window import (
        verify_final_source_delivery,
        source_delivery_satisfied,
        partial_source_paths_for_deferred_delivery,
    )
    from kernel_stateframe_framefit5 import _state_payload, _workspace_root
    from kernel_stateframe_envelope import STATE_HEADER
    from kernel_frame_optional_allocation import snapshot_optional_inputs

    layout_reallocation_enabled = _final_layout_reallocation_enabled()
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

    def frame_tokens(value: Mapping[str, object], *, wire_bytes: int | None = None) -> int:
        return (fitted_request_tokens(value, wire_bytes=wire_bytes) if use_meter_for_frame
                else frame_counter(value))

    def fitted_request_tokens(value: Mapping[str, object], *,
                              wire_bytes: int | None = None) -> int:
        return final_request_tokens(value, meter, capture, wire_bytes=wire_bytes)[0]

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
    capacity_retry_limits = getattr(capture, "_capacity_retry_limits", None)
    if capacity_retry_limits is not None:
        estimated_byte_limit = min(
            estimated_byte_limit, capacity_retry_limits.max_request_bytes)
    layout_byte_ceiling = None
    token_limit = (
        active.capacity["token_ceiling"]
        if active is not None
        else ENDPOINT_PROMPT_TOKEN_BUDGET
    )
    if token_ceiling is not None:
        if type(token_ceiling) is not int or token_ceiling <= 0:
            raise FrameRequestAllocationError("final token ceiling must be a positive integer")
        token_limit = min(token_limit, token_ceiling)
    if capacity_retry_limits is not None:
        token_limit = min(token_limit, capacity_retry_limits.max_prompt_tokens)
    token_ceiling = token_limit
    existing_counter = getattr(capture, "_frame_candidate_counter", None)
    if isinstance(existing_counter, FrozenRequestCounter):
        # A later body rewrite can trigger a second fit. Keep the first fit's
        # method and ratio, even if tokenizer availability has since changed.
        initial_count, initial_method = final_request_tokens(candidate, meter, capture)
    else:
        initial_count, initial_method = meter.measure(candidate)
        if initial_method == "calibrated_bytes":
            initial_count = calibrated_estimate(candidate, meter)
        elif initial_method == "tokenize":
            _remember_exact_pair(candidate, meter, initial_count)
        capture._frame_candidate_counter = FrozenRequestCounter.capture(
            candidate, meter, initial_count, initial_method, str(capture.request_id))

    def frozen_method() -> str:
        return capture._frame_candidate_counter.method

    def request_byte_limit(method: str) -> int:
        # Exact admission already has a finite endpoint envelope. A saved
        # estimate is not a second token ceiling, nor is a previous exact
        # measurement authority to relax a later estimate.
        if method in {"tokenize", "verified_byte_bound"}:
            limit = _lean_request_limits(candidate, FREE_STAGE, exact_tokenizer=True)[0]
        else:
            limit = estimated_byte_limit
        if byte_ceiling is not None:
            if type(byte_ceiling) is not int or byte_ceiling < 0:
                raise FrameRequestAllocationError("final byte ceiling must be a nonnegative integer")
            limit = min(limit, byte_ceiling)
        if capacity_retry_limits is not None:
            limit = min(limit, capacity_retry_limits.max_request_bytes)
        return min(limit, layout_byte_ceiling) if layout_byte_ceiling is not None else limit

    layout_token_reserve = 0
    if os.environ.get("STATEFRAME_FINAL_CACHE_LAYOUT", "on").strip().lower() == "on":
        from kernel_cache_layout_candidate import measured_final_cache_layout_token_overhead

        layout_token_reserve = measured_final_cache_layout_token_overhead(
            candidate, meter, capture,
            plain_tokens=initial_count, plain_method=initial_method,
            byte_ceiling=request_byte_limit(initial_method),
        )
    if layout_reallocation_enabled:
        layout_token_reserve = max(layout_token_reserve, _layout_reserve_floor)
    token_limit = max(1, token_ceiling - layout_token_reserve)

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
            candidate, max_request_bytes=request_byte_limit(frozen_method()),
            measure_bytes=lambda: _wire_bytes(candidate),
            max_request_tokens=(max(1, token_limit - _TOKEN_FIT_RESERVE)
                                if use_meter_for_frame else token_limit),
            measure_tokens=effective_tokens,
        )
        semantic = project_execution_delivery(semantic, candidate["messages"])
        protected = project_execution_delivery(protected, candidate["messages"])

        def compact_history(*, allow_read_body_eviction: bool = False) -> None:
            method = frozen_method()
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
            limit = request_byte_limit(frozen_method())
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
                exact_tokenizer=verified_envelope(frozen_method()),
            )
            compact_history(allow_read_body_eviction=True)
            current_limit = request_byte_limit(frozen_method())
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
    bookmark_decision: dict[str, object] = {"reason": "no_owed_reads"}
    collected_bookmark_count = 0
    if needs_bookmarks:
        from kernel_code_bookmarks import collect as collect_code_bookmarks
        bookmarks = collect_code_bookmarks(runtime, frame, decision=bookmark_decision)
        collected_bookmark_count = len(bookmarks)
        if bookmarks:
            from kernel_stateframe_framefit5 import _serialize_state_message
            frame["code_bookmarks"] = bookmarks
            state_message["content"] = _serialize_state_message(payload)
    # The collected rows came from this parsed state or the collector and are
    # only read below; fitter trials rewrite serialized messages, not these rows.
    collected_bookmarks = bookmarks
    from kernel_frame_source_snapshot import (
        editable_product_windows, fit_recent_read_windows, public_check_source_windows,
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
    product_windows, editable_product_paths = editable_product_windows(
        runtime, frame, bookmarks if isinstance(bookmarks, list) else (),
    )
    plan = getattr(capture, "_request_source_closure_override", None)
    if not isinstance(plan, Mapping):
        plan = getattr(runtime, "_request_source_closure", None)
    if not isinstance(plan, Mapping):
        raise FrameRequestAllocationError("final allocation lacks source obligations")
    plan = _copy_source_plan_snapshot(plan)
    layout_retry_source_plan = (
        _copy_source_plan_snapshot(plan) if layout_reallocation_enabled else None
    )
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
    visibility_paths.update(row["path"] for row in [*recent_windows, *product_windows])
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

    prepared_source_candidates: dict[
        int, tuple[Mapping[str, object], int | None, tuple[int, str] | None]
    ] = {}

    def prepare_source_candidate(trial: Mapping[str, object]) -> bool:
        if not ((bookmark_reconciliation or product_windows)
                and isinstance(trial.get("messages"), list)):
            return True
        # The selected source windows and the typed visibility claim are one
        # measured proposal. Reconcile before admission, never after.
        from kernel_reread_result_truth import refresh_messages
        from kernel_stateframe_framefit5 import FrameFit5Error
        try:
            refresh_messages(
                trial["messages"],
                required_deferred_paths=tuple(plan.get("deferred_paths", ())),
            )
            from kernel_frame_source_snapshot import _prune_redundant_source_metadata
            _prune_redundant_source_metadata(trial["messages"])
        except (FrameFit5Error, TypeError, ValueError):
            return False
        return True

    def source_trial_byte_limit(method: str) -> int:
        limit = request_byte_limit(method)
        if getattr(capture, "_release_prefix_boundary", None) is not None:
            limit = min(limit, HARD_MAX_REQUEST_BYTES)
        return limit

    def source_candidate_nonfit(trial: Mapping[str, object]) -> bool:
        if (type(meter) is not RequestTokenMeter
                or "measure" in getattr(meter, "__dict__", {})
                or "_measure_with_wire_bytes" in getattr(meter, "__dict__", {})):
            # Custom meter hooks keep the exhaustive measurement path.
            return False
        if not prepare_source_candidate(trial):
            prepared_source_candidates[id(trial)] = (trial, None, None)
            return False  # Let the normal callback keep its established failure path.
        serialized_bytes = _wire_bytes(trial)
        nonfit, measurement = _source_candidate_nonfit(
            meter, trial, serialized_bytes, request_byte_limit,
            hard_max_request_bytes=HARD_MAX_REQUEST_BYTES,
            clip_to_hard_limit=(getattr(capture, "_release_prefix_boundary", None) is not None),
        )
        if nonfit:
            return True
        prepared_source_candidates[id(trial)] = (trial, serialized_bytes, measurement)
        return False

    def source_windows_fit(trial: Mapping[str, object]) -> bool:
        prepared = prepared_source_candidates.pop(id(trial), None)
        if prepared is not None and prepared[0] is trial:
            if prepared[1] is None:
                return False
            serialized_bytes = prepared[1]
            measured = prepared[2]
        else:
            if not prepare_source_candidate(trial):
                return False  # Optional source advice cannot reject the base.
            serialized_bytes = _wire_bytes(trial)
            measured = None
        _count, trial_method = (
            measured if measured is not None
            else _measure_source_candidate(meter, trial, serialized_bytes)
        )
        byte_limit_for_windows = source_trial_byte_limit(trial_method)
        return (
            serialized_bytes <= byte_limit_for_windows
            and fitted_request_tokens(trial, wire_bytes=serialized_bytes) <= token_limit
            and frame_tokens(trial, wire_bytes=serialized_bytes) <= runtime.frame_ceiling
            and within_protected_capacity(
                trial, protected, exact_tokenizer=verified_envelope(trial_method),
            )
        )

    def defer_partial_paths(
            receipt: object, *, absent_pages_deferrable: bool = False,
    ) -> tuple[str, ...]:
        raw_deferred = plan.get("deferred_paths", ())
        deferred_values = (
            raw_deferred
            if isinstance(raw_deferred, Sequence)
            and not isinstance(raw_deferred, (str, bytes, bytearray))
            else ()
        )
        deferred = tuple(dict.fromkeys(
            path for path in deferred_values
            if isinstance(path, str)
            and path
        ))
        deferred_set = set(deferred)
        partial_paths = tuple(
            path for path in partial_source_paths_for_deferred_delivery(
                plan, receipt, absent_pages_deferrable=absent_pages_deferrable,
            )
            if path not in deferred_set
        ) if read_available else ()
        if not partial_paths:
            return ()
        plan["deferred_paths"] = list(dict.fromkeys((*deferred, *partial_paths)))
        _remove_deferred_source_excerpts(
            candidate["messages"],
            [path for path in partial_paths if path not in visibility_paths],
            preserve_stateframe=True,
            protected_messages=protected,
        )
        assemble_selection()
        return partial_paths

    recent_paths = {row["path"] for row in [*recent_windows, *product_windows]}

    def retained_reads_fit() -> bool:
        if not recent_windows:
            return True
        # Check the full current read frontier before accepting a base request.
        # Its closure carriers can be deferred through the verified read route;
        # spending their bytes first makes retained pages disappear on the wire.
        full_reads = fit_recent_read_windows(
            candidate, recent_windows,
            product_windows=product_windows,
            protected_messages=protected,
            newest_user_suffix=getattr(capture, "_framefit5_newest_user_suffix", ()),
            deferred_source_paths=tuple(plan.get("deferred_paths", ())),
            editable_paths=editable_product_paths,
            fits=source_windows_fit,
            candidate_nonfit=source_candidate_nonfit,
            runtime=runtime,
            persist_source_order=False,
        )
        contents = {message.get("content") for message in full_reads["messages"]
                    if isinstance(message, Mapping) and isinstance(message.get("content"), str)}
        return all(window["content"] in contents
                   or window.get("compact_content") in contents
                   for window in recent_windows)

    seen: set[tuple[tuple[str, ...], str]] = set()
    candidate_wire = b""
    # Each selected path may first defer its omitted closure units and later
    # lose its retained read pages under pressure. The initial candidate may
    # also contain retained pages before any path is deferred.
    for _attempt in range(2 * len(plan.get("units", ())) + 3):
        candidate_wire = plain_wire(candidate)
        receipt = verify_final_source_delivery(
            plan,
            candidate["messages"],
            # Existing omission reconciliation requires proof over exact wire
            # bytes, including COMPLETE paths alongside deferred closures.
            payload=candidate_wire,
            sources=sources,
        )
        if not source_delivery_satisfied(receipt):
            # The bounded source selector can omit units before an allocation
            # proposal exists. Reuse its existing verified-partial authority;
            # byte/revision failures are still terminal, never repair triggers.
            partial_paths = defer_partial_paths(receipt)
            if not partial_paths:
                raise source_refusal(plan, receipt, sources)
            byte_limit = trim_optional_candidate()
            # Reprove the rebuilt notice and exact bytes before any capacity
            # acceptance. CURRENT read pages remain eligible until real pressure.
            continue
        from kernel_step_contract import reconcile_final_source_access
        if reconcile_final_source_access(candidate["messages"], receipt):
            # The StateFrame is part of the measured and source-proven body.
            # Rebind both receipts after the truthful source-access rewrite.
            candidate_wire = json.dumps(candidate).encode()
            receipt = verify_final_source_delivery(
                plan, candidate["messages"], payload=candidate_wire, sources=sources,
            )
            if not source_delivery_satisfied(receipt):
                raise source_refusal(plan, receipt, sources)
        request_bytes = len(candidate_wire)
        tokens, method = final_request_tokens(
            candidate, meter, capture, wire_bytes=request_bytes,
        )
        byte_limit = request_byte_limit(method)
        frame_token_count = frame_tokens(candidate, wire_bytes=request_bytes)
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
                product_windows=product_windows,
                protected_messages=protected,
                newest_user_suffix=getattr(capture, "_framefit5_newest_user_suffix", ()),
                deferred_source_paths=tuple(plan.get("deferred_paths", ())),
                editable_paths=editable_product_paths,
                fits=source_windows_fit,
                candidate_nonfit=source_candidate_nonfit,
                runtime=runtime,
                persist_source_order=False,
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
                    product_windows=(),
                    newest_user_suffix=getattr(capture, "_framefit5_newest_user_suffix", ()),
                    deferred_source_paths=tuple(plan.get("deferred_paths", ())),
                    editable_paths=editable_product_paths,
                    fits=source_windows_fit,
                    candidate_nonfit=source_candidate_nonfit,
                    runtime=runtime,
                    persist_source_order=False,
                )
                continue
        current = tuple(plan.get("deferred_paths", ()))
        if not base_fits and current:
            # An editable path can be deferred while its earlier visibility
            # copies remain on the draft. Spend those host copies only after
            # the complete wire fails to fit; the newest exchange stays pinned
            # and the rebuilt candidate is source-proved on the next pass.
            # The candidate is our owned body clone. This helper only replaces
            # its messages list; with no removals the resulting list is identical.
            removed = _remove_deferred_source_excerpts(
                candidate["messages"], current,
                preserve_stateframe=True, protected_messages=protected,
            )
            if removed:
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
                # record_source_delivery writes delivery rows and two coverage
                # summaries. Isolate only those mutable maps; units, revisions,
                # authority and source text are read-only in this route probe.
                route_plan = _copy_source_delivery_route_plan(plan)
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
            # Only the terminal minimum path may relax the legacy protected
            # byte estimate. A frozen "tokenize" label can survive a later
            # tokenizer outage, so require fresh exact proof for this body.
            if (method == "tokenize"
                    and request_bytes <= byte_limit
                    and frame_token_count <= runtime.frame_ceiling):
                exact_tokens, exact_method = meter.measure(candidate)
                if (exact_method == "tokenize"
                        and exact_tokens <= token_limit
                        and within_protected_capacity(
                            candidate, protected, exact_tokenizer=True,
                        )):
                    tokens = exact_tokens
                    _remember_exact_pair(candidate, meter, tokens)
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
    from kernel_code_bookmarks import enabled as bookmarks_enabled
    from kernel_request_source_window import parse_model_source_window
    has_source_windows = any(
        isinstance(message, Mapping)
        and parse_model_source_window(message.get("content")) is not None
        for message in candidate["messages"]
    )
    if recent_windows or product_windows or public_windows or bookmarks_enabled() and has_source_windows:
        selected_paths = {
            unit["path"]
            for unit in plan.get("units", ())
            if isinstance(unit, Mapping)
            and isinstance(unit.get("path"), str)
            and unit["path"].strip()
        }
        late_deferrals = 0
        # The initial fit plus at most one repair per selected path is finite.
        for _attempt in range(len(selected_paths) + 1):
            candidate = fit_recent_read_windows(
                candidate, recent_windows, public_checks=public_windows,
                product_windows=product_windows,
                protected_messages=protected,
                newest_user_suffix=getattr(capture, "_framefit5_newest_user_suffix", ()),
                deferred_source_paths=tuple(plan.get("deferred_paths", ())),
                editable_paths=editable_product_paths,
                fits=source_windows_fit,
                candidate_nonfit=source_candidate_nonfit,
                runtime=runtime,
                persist_source_order=False,
            )
            candidate_wire = json.dumps(candidate).encode()
            receipt = verify_final_source_delivery(
                plan, candidate["messages"], payload=candidate_wire,
                sources=sources,
            )
            if source_delivery_satisfied(receipt):
                break
            proof_status = (
                str(receipt.get("proof_status", "")).upper()
                if isinstance(receipt, Mapping) else ""
            )
            if (not isinstance(receipt, Mapping)
                    or str(receipt.get("status", "")).upper() != "PARTIAL"
                    or proof_status not in {"VERIFIED", "FAILED"}
                    or late_deferrals >= len(selected_paths)):
                raise source_refusal(plan, receipt, sources)
            partial_paths = defer_partial_paths(
                receipt, absent_pages_deferrable=True,
            )
            if (not partial_paths
                    or late_deferrals + len(partial_paths) > len(selected_paths)):
                raise source_refusal(plan, receipt, sources)
            late_deferrals += len(partial_paths)
            byte_limit = trim_optional_candidate()
        else:
            raise source_refusal(plan, receipt, sources)
        tokens, method = final_request_tokens(
            candidate, meter, capture, wire_bytes=len(candidate_wire),
        )
        byte_limit = request_byte_limit(method)
    # This per-finalize source snapshot is shared by the plan and later readers;
    # those consumers inspect its path→text entries without mutating them.
    plan["source_contents"] = sources
    # A late read-window pass refreshes tokens and candidate_wire itself. Without
    # that pass, reuse the count from the successful main-loop candidate.
    byte_limit = request_byte_limit(method)
    request_bytes = len(candidate_wire)
    if (tokens > token_limit or request_bytes > byte_limit):
        byte_limit = trim_optional_candidate()
        candidate_wire = json.dumps(candidate).encode()
        receipt = verify_final_source_delivery(
            plan, candidate["messages"], payload=candidate_wire, sources=sources)
        if not source_delivery_satisfied(receipt):
            raise source_refusal(plan, receipt, sources)
        request_bytes = len(candidate_wire)
        tokens, method = final_request_tokens(
            candidate, meter, capture, wire_bytes=request_bytes,
        )
        byte_limit = request_byte_limit(method)
    else:
        request_bytes = len(candidate_wire)
    if tokens > token_limit or request_bytes > byte_limit:
        raise FrameFit5ImpossibleMinimum({
            "request_bytes": request_bytes, "prompt_tokens": tokens,
            "measure_method": method, "max_request_bytes": byte_limit,
            "max_prompt_tokens": token_limit,
        })
    # Bind the fitted body while retaining the method and ratio captured before
    # trimming. Later tokenizer availability cannot change this allocation.
    carry_final_request_tokens(candidate, capture, tokens)
    capture._frame_candidate_measurement = meter.bind_measurement(
        candidate, tokens=tokens, method=method, request_id=str(capture.request_id),
        max_request_bytes=byte_limit, max_prompt_tokens=token_ceiling,
        **_prepared_measurement_wire(meter, candidate_wire),
    )
    # `plan` is already detached from its caller and remains the allocator's
    # private mutable snapshot; the layout trim callback updates this same plan.
    capture._frame_candidate_inputs = {
        "source_plan": plan,
        "receipt": copy.deepcopy(receipt),
        "sources": sources,
        # The snapshot helper returns fresh inputs when no allocation is active;
        # an active allocation's inputs remain separately detached from it.
        "optional_inputs": copy.deepcopy(optional) if active is not None else optional,
        "byte_ceiling": byte_limit, "token_ceiling": token_ceiling,
        "measurement_identity": method,
    }
    layout_renderer_inputs = capture._frame_candidate_inputs
    # The final renderer can request a bounded full allocation retry before
    # falling back to this candidate's optional-source trim callback.
    layout_enabled = os.environ.get("STATEFRAME_FINAL_CACHE_LAYOUT", "on").strip().lower() == "on"
    candidate_module = sys.modules.get("kernel_cache_layout_candidate")
    replay_decision_fn = getattr(candidate_module, "replay_final_layout_decision", None)
    replay_decision = replay_decision_fn() if callable(replay_decision_fn) else None
    if layout_enabled and replay_decision != "plain":
        base_token_limit = token_ceiling
        base_byte_limit = byte_limit
        trim_calls = 0

        from kernel_stable_wire import mode as stable_wire_mode
        stable_replay = (getattr(candidate_module, "replay_stable_wire_decision", lambda: None)()
                         == "applied")
        selection_enabled = stable_wire_mode() == "on" or stable_replay

        def trim_for_final_layout(token_reserve: int, byte_reserve: int, *,
                                  recorded_selection: list[dict[str, object]] | None = None) -> bool:
            nonlocal token_limit, layout_byte_ceiling, trim_calls
            # Stable wire re-asks with a larger running reserve when its first trim lands short.
            # The cap equals its own bound, so a runaway caller still fails loudly.
            if trim_calls >= _MAX_FINAL_LAYOUT_TRIMS:
                raise FrameRequestAllocationError("final layout reserve requested too many times")
            trim_calls += 1

            before_trim = (copy.deepcopy(candidate)
                           if selection_enabled and (not stable_replay or recorded_selection is not None)
                           else None)
            if recorded_selection is not None:
                from kernel_stable_wire_selection import apply_selection
                if not apply_selection(candidate, recorded_selection):
                    return False

            if layout_reallocation_enabled:
                from kernel_cache_layout_candidate import (
                    FinalCacheLayoutProjectionError,
                    _checked_final_cache_projection,
                    _final_layout_reserve,
                )

                def measured_projection(value):
                    try:
                        projection = _checked_final_cache_projection(value)
                    except FinalCacheLayoutProjectionError:
                        return None
                    plain_tokens, _plain_method = final_request_tokens(value, meter, capture)
                    projected_tokens, _projected_method = final_request_tokens(
                        projection, meter, capture,
                    )
                    return (
                        plain_tokens, projected_tokens,
                        len(json.dumps(projection).encode()),
                    )

                measurement = measured_projection(candidate)
                if (measurement is not None
                        and measurement[1] > base_token_limit):
                    missing = object()
                    capture_fields = (
                        "_frame_candidate_inputs", "_frame_candidate_measurement",
                        "_frame_candidate_counter", "_frame_candidate_fallback_fit",
                        "_frame_candidate_rejection_bound", "_request_source_closure_override",
                        "request_source_final_delivery", "request_budget_exceeded",
                        "_code_bookmark_transition",
                    )
                    saved_capture = {
                        name: getattr(capture, name, missing) for name in capture_fields
                    }

                    def restore_capture():
                        for name, value in saved_capture.items():
                            if value is missing:
                                if hasattr(capture, name):
                                    delattr(capture, name)
                            else:
                                setattr(capture, name, value)

                    current_body = candidate
                    current_inputs = layout_renderer_inputs
                    current_reserve = max(layout_token_reserve, _layout_reserve_floor)
                    retry_pass = _layout_retry_pass
                    did_retry = False
                    while measurement is not None:
                        next_reserve = _next_final_layout_reserve(
                            retry_pass=retry_pass,
                            current_reserve=current_reserve,
                            measured_growth=max(0, measurement[1] - measurement[0]),
                            projected_tokens=measurement[1],
                            token_ceiling=base_token_limit,
                        )
                        if next_reserve is None:
                            break
                        did_retry = True
                        capture._request_source_closure_override = (
                            _copy_source_plan_snapshot(layout_retry_source_plan)
                        )
                        try:
                            current_body = finalize_frame_candidate(
                                runtime, capture, body, frame_counter=frame_counter,
                                _layout_retry_pass=retry_pass + 1,
                                _layout_reserve_floor=next_reserve,
                            )
                        except FrameFit5ImpossibleMinimum:
                            restore_capture()
                            return False
                        current_inputs = capture._frame_candidate_inputs
                        current_reserve = next_reserve
                        retry_pass += 1
                        measurement = measured_projection(current_body)
                        if measurement is None:
                            restore_capture()
                            return False
                        if (measurement[1] <= base_token_limit
                                and measurement[2] <= base_byte_limit):
                            if not isinstance(candidate, dict):
                                restore_capture()
                                return False
                            candidate.clear()
                            candidate.update(current_body)
                            current_inputs.pop("_final_layout_trim", None)
                            layout_renderer_inputs.clear()
                            layout_renderer_inputs.update(current_inputs)
                            capture._frame_candidate_inputs = layout_renderer_inputs
                            return True

                    if did_retry:
                        latest_trim = current_inputs.get("_final_layout_trim")
                        if callable(latest_trim) and measurement is not None:
                            _overflow_reserve, trim_bytes = _final_layout_reserve(
                                measurement[1], base_token_limit,
                                measurement[2], base_byte_limit,
                            )
                            trim_tokens = max(0, measurement[1] - measurement[0])
                            if latest_trim(trim_tokens, trim_bytes):
                                measurement = measured_projection(current_body)
                                if (measurement is not None
                                        and measurement[1] <= base_token_limit
                                        and measurement[2] <= base_byte_limit
                                        and isinstance(candidate, dict)):
                                    candidate.clear()
                                    candidate.update(current_body)
                                    current_inputs.pop("_final_layout_trim", None)
                                    layout_renderer_inputs.clear()
                                    layout_renderer_inputs.update(current_inputs)
                                    capture._frame_candidate_inputs = layout_renderer_inputs
                                    return True
                        restore_capture()
                        return False

            target_token_limit = max(1, base_token_limit - max(0, token_reserve))
            target_byte_limit = max(1, base_byte_limit - max(0, byte_reserve))
            token_limit = target_token_limit
            layout_byte_ceiling = target_byte_limit
            try:
                try:
                    if recorded_selection is None:
                        trim_optional_candidate()
                    if (layout_reallocation_enabled
                            and not _final_layout_trim_reached_target(
                                candidate, meter, capture,
                                token_limit=target_token_limit,
                                byte_limit=request_byte_limit(frozen_method()),
                            )):
                        return False
                except FrameFit5ImpossibleMinimum:
                    return False
            finally:
                token_limit = base_token_limit
                layout_byte_ceiling = None
            selection = None
            if before_trim is not None:
                from kernel_stable_wire_selection import removed_candidates
                selection = removed_candidates(before_trim, candidate)
                if selection is None:
                    # Capacity compaction can rewrite protected history. Do not publish
                    # row-removal authority that cannot reproduce those bytes.
                    return False
            from kernel_request_source_window import verify_final_source_delivery, source_delivery_satisfied
            trimmed_wire = json.dumps(candidate).encode()
            trimmed_receipt = verify_final_source_delivery(
                plan, candidate["messages"], payload=trimmed_wire, sources=sources)
            if not source_delivery_satisfied(trimmed_receipt):
                return False
            rebind_final_layout_trim(
                candidate, meter, capture, byte_limit=request_byte_limit(frozen_method()),
                token_limit=base_token_limit, wire=trimmed_wire,
                replay_selection=recorded_selection is not None)
            trim_for_final_layout.selection = selection
            # source_plan already points at this owned plan and reflects the trim.
            capture._frame_candidate_inputs["receipt"] = copy.deepcopy(trimmed_receipt)
            capture.request_source_final_delivery = trimmed_receipt
            return True

        trim_for_final_layout.replays_selections = selection_enabled
        capture._frame_candidate_inputs["_final_layout_trim"] = trim_for_final_layout
    capture._frame_candidate_fallback_fit = method == "calibrated_bytes"
    capture._frame_candidate_rejection_bound = (
        _measure_source_candidate(meter, candidate, request_bytes)[0]
        if method == "calibrated_bytes" else None
    )
    capture._request_source_closure_override = _copy_source_plan_snapshot(plan)
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
    whole_source_revisions = {
        parsed["path"]: parsed.get("revision") for message in candidate["messages"]
        if isinstance(message, Mapping)
        if (parsed := parse_model_source_window(message.get("content"))) is not None
        and parsed.get("status") == "COMPLETE" and isinstance(parsed.get("path"), str)
    }
    current_revisions = {fact["path"]: fact.get("current_revision")
                         for fact in final_state.get("read_facts", ())
                         if isinstance(fact, Mapping) and isinstance(fact.get("path"), str)}
    whole_source_paths = {path for path, revision in whole_source_revisions.items()
                          if path not in current_revisions or current_revisions[path] == revision}
    carried_whole_paths = {row["path"] for row in collected_bookmarks
                           if isinstance(row, Mapping) and row.get("path") in whole_source_paths
                           and row.get("revision") == whole_source_revisions[row["path"]]}
    visible_bookmarked_paths = (bookmarked_paths & carried_source_paths) | (
        carried_whole_paths
    ) | {
        fact["path"] for fact in final_state.get("read_facts", ())
        if isinstance(fact, Mapping) and fact.get("path") in bookmarked_paths
        and fact.get("body_delivery") in {"present", "bookmarked"}
    }
    active = bool(visible_bookmarked_paths)
    reason = ("activated" if active else
              "capacity_dropped" if collected_bookmark_count and not final_bookmarks else
              "no_delivery" if collected_bookmark_count else bookmark_decision["reason"])
    from kernel_read_lockout import final_source_visible, release as release_read_lockout
    visible_bookmarks = ([row for row in final_bookmarks if isinstance(row, Mapping)
                          and row.get("path") in visible_bookmarked_paths]
                         if isinstance(final_bookmarks, list) else [])
    visible_bookmarks.extend(row for row in collected_bookmarks
                             if row.get("path") in carried_whole_paths
                             and row not in visible_bookmarks)
    if (getattr(runtime, "_read_lockout_notice", None)
            and not final_source_visible(runtime, visible_bookmarks, candidate["tools"])):
        # The prior request's visible bookmark was only provisional. Rebuild an
        # unrestricted request before a provider sees a completion-only trap.
        from kernel_frame_optional_allocation import FrameAllocationReplanNeeded
        release_read_lockout(runtime, "SOURCE_NOT_VISIBLE")
        raise FrameAllocationReplanNeeded("locked repair source absent from final request")
    journal = getattr(capture, "journal", getattr(runtime, "journal", None))
    emit = getattr(journal, "emit", None)
    from kernel_code_bookmarks import enabled as bookmarks_enabled
    if bookmarks_enabled() and callable(emit):
        for drop in bookmark_decision.get("drops", ()):
            emit("code_bookmark_dropped", request_id=getattr(capture, "request_id", None),
                 path=drop["path"], symbol=drop["symbol"], reason=drop["reason"],
                 dispatch=False)
        emit("code_bookmark_activation_decision",
             request_id=getattr(capture, "request_id", None),
             reason=reason, activated=active,
             collected_count=collected_bookmark_count,
             final_count=len(final_bookmarks) if isinstance(final_bookmarks, list) else 0,
             dispatch=False)
    # Each row is a parsed snapshot from serialized candidate state or the
    # immutable collected bookmark list; neither source is mutated afterward.
    capture._code_bookmark_transition = (active, visible_bookmarks if active else [])
    return candidate
