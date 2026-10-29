"""One byte budget for the complete protected FrameFit5 request.

The journal remains the historical authority. This projection keeps exact call
identities and recent detail, never a transcript or an assertion of absent text.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence

from kernel_stateframe_call_coverage import (
    CONTRACT, FRONTIER_BYTES, FRONTIER_COUNT, FRONTIER_ID_BYTES, CoverageSession, context_index,
)
from kernel_evictfix_mode import enabled as evictfix_enabled


def _bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _sha(value: object) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


def protected_limit(body: Mapping[str, object], *, exact_tokenizer: bool = False) -> int:
    from kernel_request_budget import FREE_STAGE, HARD_MAX_REQUEST_BYTES, _lean_request_limits

    if exact_tokenizer:
        return _lean_request_limits(body, FREE_STAGE, exact_tokenizer=True)[0]
    return min(HARD_MAX_REQUEST_BYTES, _lean_request_limits(body, FREE_STAGE)[0])


def protected_bytes(body: Mapping[str, object],
                    protected_messages: Sequence[Mapping[str, object]] = ()) -> int:
    from kernel_request_diet import _paired_tool_exchange_messages, _protected_request_sizes

    messages = body.get("messages", [])
    paired = _paired_tool_exchange_messages(messages) if isinstance(messages, list) else ()
    return _protected_request_sizes(
        body, protected_contents=(), protected_messages=(*protected_messages, *paired),
        required_source_paths=frozenset(),
    )[0]


def within_protected_capacity(body: Mapping[str, object],
                              protected_messages: Sequence[Mapping[str, object]] = (),
                              *, exact_tokenizer: bool = False) -> bool:
    return protected_bytes(body, protected_messages) <= protected_limit(
        body, exact_tokenizer=exact_tokenizer)


def row_ids(row: Mapping[str, object]) -> set[str]:
    ids = row.get("call_ids", ())
    result = {item for item in ids if isinstance(item, str) and item} if isinstance(ids, (list, tuple)) else set()
    if isinstance(row.get("call_id"), str) and row["call_id"]:
        result.add(row["call_id"])
    return result


def ledger_ids(ledger: Mapping[str, object], action_context: object = None) -> set[str]:
    """Literal frame identities, plus proven historical IDs when host-owned."""
    rows = [row for name in ("reads", "edits", "unclassified_actions")
            for row in ledger.get(name, ()) if isinstance(row, Mapping)]
    rows += [row for name in ("last_test", "history_digest")
             if isinstance((row := ledger.get(name)), Mapping)]
    result = set().union(*(row_ids(row) for row in rows))
    digest = ledger.get("history_digest")
    index = context_index(action_context)
    if isinstance(digest, Mapping) and "coverage_contract" in digest and index is not None:
        result.update(index.covered_ids(digest))
    return result


def _fold(ledger: dict[str, object], row: Mapping[str, object], *,
          kind: str, exact_ids: set[str] | None = None,
          coverage: CoverageSession | None = None) -> None:
    """Append a row hash; bounded ID coverage requires a durable host proof."""
    from kernel_stateframe_action_ledger import _receipt_line

    digest = ledger.setdefault("history_digest", {
        "kind": "action_history_digest", "call_ids": [], "row_count": 0,
        "kind_counts": {}, "tool_counts": {}, "rows_sha256": _sha([]), "source_text": "not_carried",
    })
    ids = row_ids(row) if exact_ids is None else exact_ids
    if "coverage_contract" in digest and coverage is None:
        from kernel_stateframe_framefit5 import FrameFit5Error

        raise FrameFit5Error("call coverage folding requires its host index")
    digest["call_ids"] = sorted(set(digest["call_ids"]) | ids)
    digest["row_count"] += 1
    counts = digest["kind_counts"]
    counts[kind] = counts.get(kind, 0) + 1
    tool = row.get("tool") or (kind if kind in {"read", "edit", "test"} else None)
    if isinstance(tool, str) and tool:
        tool_counts = digest["tool_counts"]
        tool_counts[tool] = tool_counts.get(tool, 0) + 1
    digest["rows_sha256"] = _sha([digest["rows_sha256"], kind, row])
    line = row.get("result_receipt", row.get("result_first_line", row.get("text", "")))
    if isinstance(line, str) and line:
        digest["latest_receipt_line"] = _receipt_line(line)
    if coverage is not None:
        coverage.fold(digest, ids)


def _source_order(messages: Sequence[Mapping[str, object]]) -> dict[str, int]:
    return {str(call["id"]): index for index, message in enumerate(messages)
            if message.get("role") == "assistant"
            for call in message.get("tool_calls", ()) or ()
            if isinstance(call, Mapping) and call.get("id")}


def _standalone_note_order(messages: Sequence[Mapping[str, object]]) -> dict[str, list[int]]:
    positions: dict[str, list[int]] = {}
    for index, message in enumerate(messages):
        if message.get("role") == "assistant" and not message.get("tool_calls"):
            text = str(message.get("content") or "")
            positions.setdefault(hashlib.sha256(text.encode()).hexdigest(), []).append(index)
    return positions


def _segments(messages: Sequence[Mapping[str, object]],
              frame: Mapping[str, object], reads: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    from kernel_request_diet import _current_source_segments, _read_page_source_segment

    segments = _current_source_segments(messages)
    for message in messages:
        segment = _read_page_source_segment(message)
        if segment is not None:
            segments.append({**segment, "text": segment["bytes"].decode("utf-8")})
    sites = frame.get("open_sites")
    if not isinstance(sites, list):
        sites = [frame["next_site"]] if isinstance(frame.get("next_site"), Mapping) else []
    elif isinstance(frame.get("next_site"), Mapping) and frame["next_site"] not in sites:
        sites = [frame["next_site"], *sites]
    for row in sites:
        text = row.get("current_bytes")
        lines = row.get("lines")
        if isinstance(text, str) and isinstance(lines, Mapping):
            segments.append({"path": row.get("path"), "revision": row.get("revision"),
                             "start_line": lines.get("start"), "end_line": lines.get("end"),
                             "text": text})
    headcarry = frame.get("headcarry")
    heads = headcarry.get("heads") if isinstance(headcarry, Mapping) else None
    if isinstance(heads, list):
        for head in heads:
            if not isinstance(head, Mapping):
                continue
            text = head.get("content")
            lines = head.get("line_range")
            revision = head.get("revision")
            if (not isinstance(text, str) or not isinstance(lines, Mapping)
                    or not isinstance(revision, str) or not revision.startswith("sha256:")
                    or head.get("content_sha256") != hashlib.sha256(text.encode()).hexdigest()
                    or head.get("source_sha256") != revision.removeprefix("sha256:")):
                continue
            segments.append({"path": head.get("path"), "revision": revision,
                             "start_line": lines.get("start"), "end_line": lines.get("end"),
                             "text": text})
    # A receipt can carry bytes only while its explicit pin remains present.
    # Other receipt fields describe prior work and cannot prove current delivery.
    for row in reads:
        text = row.get("pinned_range")
        lines = row.get("lines")
        if isinstance(text, str) and isinstance(lines, Mapping):
            segments.append({"path": row.get("path"), "revision": row.get("revision"),
                             "start_line": lines.get("start"), "end_line": lines.get("end"),
                             "text": text})
    return segments


def _read_source_segments(messages: Sequence[Mapping[str, object]],
                          frame: Mapping[str, object],
                          reads: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Read delivery may use selected source, never navigation inventory."""
    selected = frame.get("next_site")
    carriers = {"next_site": selected} if isinstance(selected, Mapping) else {}
    if isinstance(frame.get("headcarry"), Mapping):
        carriers["headcarry"] = frame["headcarry"]
    return _segments(messages, carriers, reads)


def _delivered(row: Mapping[str, object], segments: Sequence[Mapping[str, object]]) -> bool:
    lines = row.get("lines")
    if not isinstance(lines, Mapping):
        return False
    start, end = lines.get("start"), lines.get("end")
    if type(start) is not int or type(end) is not int or not 1 <= start <= end:
        return False
    covered: dict[int, str] = {}
    for segment in segments:
        if (segment.get("path") != row.get("path")
                or segment.get("revision") != row.get("revision")):
            continue
        left, right, text = segment.get("start_line"), segment.get("end_line"), segment.get("text")
        if (type(left) is not int or type(right) is not int or not isinstance(text, str)
                or right < start or left > end):
            continue
        parts = text.splitlines(keepends=True)
        if len(parts) != right - left + 1:
            continue
        for line in range(max(start, left), min(end, right) + 1):
            value = parts[line - left]
            if line in covered and covered[line] != value:
                return False
            covered[line] = value
    return (len(covered) == end - start + 1 and
            hashlib.sha256("".join(covered[line] for line in range(start, end + 1)).encode()).hexdigest()
            == row.get("content_sha256"))


def validate_read_ranges(frame: Mapping[str, object],
                         messages: Sequence[Mapping[str, object]]) -> None:
    from kernel_stateframe_framefit5 import FrameFit5Error

    ledger = frame["action_ledger"]
    reads = ledger["reads"]
    # Selected next-site bytes survive independently of navigation inventory.
    segments = _read_source_segments(messages, frame, reads)
    for row in reads:
        if row.get("range_delivery") == "same_request" and (
                "pinned_range" in row or not _delivered(row, segments)):
            raise FrameFit5Error("read receipt lacks its exact same-request source bytes")
    digest = ledger.get("history_digest")
    if digest is not None:
        if (not isinstance(digest, Mapping) or digest.get("kind") != "action_history_digest"
                or digest.get("source_text") != "not_carried"
                or not isinstance(digest.get("call_ids"), list)
                or any(not isinstance(item, str) or not item for item in digest["call_ids"])
                or digest["call_ids"] != sorted(set(digest["call_ids"]))
                or type(digest.get("row_count")) is not int or digest["row_count"] < 1
                or not isinstance(digest.get("kind_counts"), dict)
                or any(type(count) is not int or count < 1 for count in digest["kind_counts"].values())
                or sum(digest["kind_counts"].values()) != digest["row_count"]
                or not isinstance(digest.get("tool_counts"), dict)
                or any(not isinstance(tool, str) or not tool or type(count) is not int or count < 1
                       for tool, count in digest["tool_counts"].items())
                or sum(digest["tool_counts"].values()) > digest["row_count"]
                or not isinstance(digest.get("rows_sha256"), str)
                or len(digest["rows_sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in digest["rows_sha256"])):
            raise FrameFit5Error("typed historical action digest is malformed")
        if "coverage_contract" in digest and (
                digest["coverage_contract"] != CONTRACT
                or not isinstance(digest.get("coverage_run_id"), str)
                or not digest["coverage_run_id"]
                or type(digest.get("call_count")) is not int
                or digest["call_count"] < len(digest["call_ids"])
                or len(digest["call_ids"]) > FRONTIER_COUNT
                or len(_bytes(digest["call_ids"])) > FRONTIER_BYTES
                or any(len(item.encode("utf-8")) > FRONTIER_ID_BYTES for item in digest["call_ids"])
                or not isinstance(digest.get("call_ids_sha256"), str)
                or len(digest["call_ids_sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in digest["call_ids_sha256"])):
            raise FrameFit5Error("typed historical call coverage is malformed")
    from kernel_reread_result_truth import validate_delivery

    validate_delivery(frame, messages)


def _expected_ids(action_context: Mapping[str, object]) -> set[str]:
    return {item for name in ("covered_call_ids", "uncovered_call_ids", "superseded_test_call_ids")
            for item in action_context.get(name, ()) if isinstance(item, str) and item}


def _fold_superseded(ledger: dict[str, object], context: Mapping[str, object],
                     coverage: CoverageSession | None) -> None:
    covered = ledger_ids(ledger) | (coverage.covered_ids if coverage else set())
    superseded = {item for item in context.get("superseded_test_call_ids", ())
                  if isinstance(item, str) and item} - covered
    if superseded:
        _fold(ledger, {"call_ids": sorted(superseded)}, kind="test", coverage=coverage)


def validate_coverage(frame: Mapping[str, object], action_context: object) -> None:
    """Prove producer coverage using the explicitly selected coverage contract.

    Legacy digests enumerate every ID. journal-call-coverage/1 carries only a
    bounded recent frontier: its complete digest (including the original row
    chain, unique-ID count and ID chain) MUST match a durable host proof. Every
    producer ID must exist in that proof or in a retained typed row. Missing
    host authority never falls back to treating the frontier as full coverage.
    """
    from kernel_stateframe_framefit5 import FrameFit5Error

    ledger = frame["action_ledger"]
    if "history_digest" not in ledger:
        index = context_index(action_context)
        if index is None:
            return
        index.sync()
        if not index.proofs:
            return
    digest = ledger.get("history_digest", {})
    if "coverage_contract" in digest:
        if digest["coverage_contract"] != CONTRACT or context_index(action_context) is None:
            raise FrameFit5Error("call coverage validation requires its host index")
    if not isinstance(action_context, Mapping):
        return
    missing = _expected_ids(action_context) - ledger_ids(ledger, action_context)
    if missing:
        raise FrameFit5Error("compacted frame lacks prior typed action coverage")


def _reads_in_action_order(rows: Sequence[Mapping[str, object]],
                           action_context: Mapping[str, object],
                           semantic_messages: Sequence[Mapping[str, object]] = ()
                           ) -> list[Mapping[str, object]]:
    """Merge compact and full read rows before the newest-first E2 cap."""
    raw = action_context.get("ledger")
    originals = raw.get("rows", ()) if isinstance(raw, Mapping) else ()
    order: dict[str, int] = {}
    for index, original in enumerate(originals):
        if isinstance(original, Mapping):
            for call_id in row_ids(original):
                order.setdefault(call_id, index)
    if not order and semantic_messages:
        order = _source_order(semantic_messages)
    indexed = list(enumerate(rows))
    if not order:
        return [row for _, row in indexed]
    indexed.sort(key=lambda pair: (
        min((order[call_id] for call_id in row_ids(pair[1]) if call_id in order),
            default=len(order) + pair[0]), pair[0]))
    return [row for _, row in indexed]


def _repair_receipts(frame: dict[str, object], messages: Sequence[Mapping[str, object]],
                     action_context: Mapping[str, object],
                     coverage: CoverageSession | None = None) -> None:
    """A later source eviction must restore the pin or make absence explicit."""
    ledger = frame["action_ledger"]
    reads = ledger["reads"]
    raw = action_context.get("ledger")
    originals = raw.get("rows", ()) if isinstance(raw, Mapping) else ()
    originals = {tuple(row.get("call_ids", ())): row for row in originals if isinstance(row, Mapping)}
    for row in list(reads):
        if row.get("range_delivery") != "same_request":
            continue
        if _delivered(row, _read_source_segments(messages, frame, reads)):
            continue
        original = originals.get(tuple(row.get("call_ids", ())), {})
        pin = original.get("pinned_range")
        if isinstance(pin, str) and hashlib.sha256(pin.encode()).hexdigest() == row.get("content_sha256"):
            row["pinned_range"] = pin
            row.pop("range_delivery")
        else:
            # No source assertion survives eviction; immutable action identity does.
            _fold(ledger, row, kind="read", coverage=coverage)
            reads.remove(row)
            if evictfix_enabled():
                from kernel_read_working_set import compact_read_ledger

                prior = ledger.get("read_ledger")
                prior = prior if isinstance(prior, list) else []
                revisions = {fact.get("path"): fact["current_revision"]
                             for fact in frame.get("read_facts", [])
                             if isinstance(fact, Mapping) and "current_revision" in fact}
                ledger["read_ledger"] = compact_read_ledger(
                    _reads_in_action_order([*reversed(prior), row], action_context),
                    revisions)


def refresh_read_ranges(messages: list[dict[str, object]],
                        action_context: Mapping[str, object] | None = None) -> None:
    """Reprove capacity receipts after canonical assembly changes source carriers."""
    from kernel_stateframe_envelope import STATE_HEADER
    from kernel_stateframe_framefit5 import (
        _state_payload, _serialize_state_message, _is_framefit5_source_notice,
    )

    context = action_context if isinstance(action_context, Mapping) else {}
    for message in messages:
        if not str(message.get("content", "")).startswith(STATE_HEADER + "\n"):
            continue
        payload, state = _state_payload(message)
        frame = state["framefit5"]
        ledger = frame.get("action_ledger")
        reads = ledger.get("reads") if isinstance(ledger, Mapping) else None
        original = copy.deepcopy(frame)
        has_same_request_receipt = isinstance(reads, list) and any(
            isinstance(row, Mapping) and row.get("range_delivery") == "same_request"
            for row in reads
        )
        if has_same_request_receipt:
            index = context_index(context)
            coverage = CoverageSession(index, ledger.get("history_digest")) if index else None
            _repair_receipts(frame, messages, context, coverage)
            if original != frame:
                if coverage is not None:
                    if coverage.digest is not None:
                        _fold_superseded(ledger, context, coverage)
                    coverage.commit()
                validate_coverage(frame, context)

        from kernel_reread_result_truth import reconcile_source_truth

        reconcile_source_truth(
            messages, frame,
            include_host_fact=any(
                _is_framefit5_source_notice(row) for row in messages
            ),
        )
        if original != frame:
            message["content"] = _serialize_state_message(payload)


def fit_protected_frame(body: dict[str, object], *,
                        semantic_messages: Sequence[Mapping[str, object]] = (),
                        action_context: Mapping[str, object] | None = None,
                        protected_messages: Sequence[Mapping[str, object]] = (),
                        max_request_bytes: int | None = None,
                        max_wire_bytes: int | None = None,
                        max_request_tokens: int | None = None,
                        measure_request_tokens: Callable[[Mapping[str, object]], int] | None = None,
                        local_request_tokens: Callable[[Mapping[str, object]], int] | None = None,
                        required_deferred_paths: Sequence[object] = (),
                        allow_read_body_eviction: bool = True,
                        exact_tokenizer: bool = False,
                        ) -> dict[str, object]:
    """Compact only under pressure, measuring rows, facts and pins in one envelope.

    Non-historical authority and the newest exchange cannot be compacted. If even
    exact identity coverage cannot fit, the finalizer must refuse that minimum.
    """
    from kernel_stateframe_envelope import STATE_HEADER
    from kernel_request_budget import _wire_bytes
    from kernel_stateframe_framefit5 import (
        FrameFit5Error, _state_payload,
        _is_framefit5_source_notice, _framefit5_deferred_paths,
    )
    from kernel_reread_result_truth import (
        enabled as result_truth_enabled, refresh_messages,
    )
    from kernel_capacity_memo import JsonFragmentMemo, applied_edit_reductions

    evictfix = evictfix_enabled()
    target = body
    original_bytes = protected_bytes(body, protected_messages)
    # Reconciliation and fitting mutate nested message rows. Work on a detached
    # message list so rejected measurements cannot publish partial truth changes.
    body = dict(body)
    body["messages"] = copy.deepcopy(target.get("messages", []))
    explicit_deferred_paths = (
        tuple(path for path in required_deferred_paths if isinstance(path, str) and path)
        if not isinstance(required_deferred_paths, (str, bytes, bytearray)) else ()
    )
    notice_paths, _notice_indexes = _framefit5_deferred_paths(body["messages"])
    known_deferred_paths = tuple(dict.fromkeys((*notice_paths, *explicit_deferred_paths)))
    if result_truth_enabled():
        refresh_messages(
            body.get("messages", []), required_deferred_paths=known_deferred_paths,
        )
    limit = protected_limit(body, exact_tokenizer=exact_tokenizer)
    if (max_request_tokens is None) != (measure_request_tokens is None):
        raise ValueError("capacity token limit and measurement must be supplied together")
    if local_request_tokens is not None and max_request_tokens is None:
        raise ValueError("local capacity measurement requires a token limit")
    if max_request_bytes is not None:
        limit = min(limit, max_request_bytes)
    before = protected_bytes(body, protected_messages)
    tokenizer_calls = 0
    measurements: dict[str, int] = {}

    def measured_tokens(value: Mapping[str, object]) -> int:
        nonlocal tokenizer_calls
        # Only exact identical bodies can share evidence within this pass.
        # This cache cannot manufacture additive tokenizer counts across joins.
        key = _sha(value)
        if key not in measurements:
            tokenizer_calls += 1
            measurements[key] = measure_request_tokens(value)
        return measurements[key]

    if (body != target and original_bytes <= limit
            and measure_request_tokens is not None):
        # Preserve the original candidate's measurement failure semantics; the
        # count is never reused for the reconciled replacement.
        measured_tokens(target)
    def under_pressure(size: int) -> bool:
        # Always measure this detached candidate, never a count from its parent.
        # The protected subset can fit while source notices/carriers make the
        # complete wire exceed its independently selected byte envelope.
        return (
            size > limit
            or (max_wire_bytes is not None and _wire_bytes(body) > max_wire_bytes)
            or (local_request_tokens is not None
                and local_request_tokens(body) > max_request_tokens)
            or (measure_request_tokens is not None
                and measured_tokens(body) > max_request_tokens)
        )

    pressure = under_pressure(before)
    audit: dict[str, object] = {
        "before": original_bytes, "after": before, "limit": limit,
        "changed": body != target,
        "measurement_calls": tokenizer_calls,
    }
    messages = body.get("messages", [])
    states = [row for row in messages if isinstance(row, Mapping)
              and str(row.get("content", "")).startswith(STATE_HEADER + "\n")
              and '"framefit5"' in str(row.get("content", ""))]
    if len(states) != 1:
        if body != target:
            target["messages"] = body["messages"]
        return audit
    payload, state = _state_payload(states[0])
    frame = state["framefit5"]
    ledger = frame["action_ledger"]
    # Row identity fields stay stable while the fitter removes or repairs rows.
    # Retain references, not another copy of potentially large read bodies.
    original_read_rows = list(ledger["reads"])
    initial_compact_rows = ledger.get("read_ledger")
    initial_compact_rows = list(initial_compact_rows) if isinstance(initial_compact_rows, list) else []
    context = action_context if isinstance(action_context, Mapping) else {}
    index = context_index(context)
    if index is not None:
        index.sync()
    # A fresh rebuilt frame can fit without folding but still needs a proof for
    # identities represented only by aggregate counts. A receipt-history digest
    # is useful only when this journal index validates its exact coverage proof.
    validated_coverage: set[str] = set()
    digest = ledger.get("history_digest")
    if (index is not None and isinstance(digest, Mapping)
            and "coverage_contract" in digest):
        if digest["coverage_contract"] != CONTRACT:
            raise FrameFit5Error("unknown call coverage contract")
        # A malformed proof must fail even when explicit rows would otherwise
        # let this candidate take the no-pressure fast path.
        validated_coverage = index.covered_ids(digest)
    needs_coverage = bool(
        index and index.proofs
        and _expected_ids(context) - (ledger_ids(ledger) | validated_coverage)
    )
    has_receipts = any(row.get("range_delivery") == "same_request" for row in ledger["reads"])
    if not pressure and not has_receipts and not needs_coverage:
        if body != target:
            target["messages"] = body["messages"]
        return audit
    state_index = messages.index(states[0])
    messages = body["messages"]
    states = [messages[state_index]]
    original_ids = ledger_ids(ledger, context)
    coverage = CoverageSession(index, ledger.get("history_digest")) if index else None
    previous_frame = copy.deepcopy(frame)
    _repair_receipts(frame, messages, context, coverage)
    if coverage is not None and coverage.digest is not None:
        _fold_superseded(ledger, context, coverage)
    if not pressure and frame == previous_frame and not needs_coverage:
        validate_read_ranges(frame, messages)
        if body != target:
            target["messages"] = messages
        return audit
    order = _source_order(semantic_messages)
    note_order = _standalone_note_order(semantic_messages)
    full_ids_by_hash = {hashlib.sha256(call_id.encode()).hexdigest(): call_id for call_id in order}

    def exact_ids(row: Mapping[str, object]) -> set[str]:
        full = full_ids_by_hash.get(row.get("call_id_sha256"))
        return {full} if full else row_ids(row)

    notice_paths, _notice_indexes = _framefit5_deferred_paths(messages)
    known_deferred_paths = tuple(dict.fromkeys((*known_deferred_paths, *notice_paths)))
    reconcile_notices = bool(known_deferred_paths) or any(
        _is_framefit5_source_notice(message) for message in messages
    )
    fragments = JsonFragmentMemo()

    def publish() -> int:
        from kernel_reread_result_truth import reconcile_source_truth

        reconcile_source_truth(
            messages, frame, required_deferred_paths=known_deferred_paths,
            include_host_fact=reconcile_notices,
        )
        # _state_payload has already normalized this envelope. Reuse unchanged
        # JSON fragments rather than deepcopy/serialize the entire frame.
        states[0]["content"] = STATE_HEADER + "\n" + fragments.dumps(payload)
        return protected_bytes(body, protected_messages)

    size = publish()
    if under_pressure(size):
        from kernel_open_site_capacity import location_inventory_candidate

        compact_sites = location_inventory_candidate(frame)
        if compact_sites is not None:
            frame["open_sites"] = compact_sites["open_sites"]
            frame["site_inventory_encoding"] = compact_sites["site_inventory_encoding"]
            _repair_receipts(frame, messages, context, coverage)
            size = publish()
    if under_pressure(size) or needs_coverage:
        # Superseded checks were previously represented only by a count. When
        # folding history, retain their exact IDs in the same typed digest too.
        _fold_superseded(ledger, context, coverage)
        if needs_coverage and coverage is not None:
            missing_ids = _expected_ids(context) - (ledger_ids(ledger) | coverage.covered_ids)
            if missing_ids:
                # RW may attach a valid receipt digest from an older coverage
                # prefix. Repair only its missing identities via CI's row fold,
                # preserving a nonempty history and its hash/count invariants.
                _fold(ledger, {"call_ids": sorted(missing_ids)},
                      kind="call_identity", coverage=coverage)
        # Existing bounded receipts retain only an ID prefix plus its hash.
        # Preserve the full identity even when its recent detail survives fitting.
        covered_ids = ledger_ids(ledger) | (coverage.covered_ids if coverage else set())
        for field in ("reads", "edits", "unclassified_actions"):
            for row in ledger.get(field, []):
                full_id = full_ids_by_hash.get(row.get("call_id_sha256"))
                if full_id and full_id not in covered_ids:
                    _fold(ledger, {"call_ids": [full_id],
                                   "call_id_sha256": row["call_id_sha256"]}, kind="call_identity",
                          coverage=coverage)
                    covered_ids.add(full_id)
        # Replace only byte-identical range copies. The newest retained ledger
        # pin or a current source carrier owns each same-request receipt.
        retained = []
        for row in reversed(ledger["reads"]):
            pin = row.get("pinned_range")
            if (isinstance(pin, str)
                    and hashlib.sha256(pin.encode()).hexdigest() == row.get("content_sha256")
                    and _delivered(row, _read_source_segments(messages, frame, retained))):
                row.pop("pinned_range")
                row["range_delivery"] = "same_request"
            else:
                retained.append(row)
        size = publish()

    # Repeated typed reads and journal receipts are two views of the same
    # historical work. Reduce both before sacrificing the active read frontier.
    # One aggregate request budget owns them; there is no per-list allowance.
    current_revisions = {fact.get("path"): fact["current_revision"]
                         for fact in frame.get("read_facts", []) if "current_revision" in fact}
    def compact_evicted_reads() -> bool:
        """Keep E2's bounded identity projection when full read rows are folded."""
        if not evictfix_enabled():
            return False
        live_ids = {call_id for row in ledger["reads"] for call_id in row_ids(row)}
        prior_ids = {call_id for row in initial_compact_rows for call_id in row_ids(row)}
        newly_evicted = [row for row in original_read_rows
                         if row_ids(row) - live_ids - prior_ids]
        import kernel_read_working_set

        compact_read_ledger = getattr(kernel_read_working_set, "compact_read_ledger", None)
        if not callable(compact_read_ledger):
            return False
        # The rows are in original action order, independent of which
        # reduction happened to fold them first.
        compact = compact_read_ledger(
            _reads_in_action_order(
                [*reversed(initial_compact_rows), *newly_evicted],
                context, semantic_messages), current_revisions)
        if compact == ledger.get("read_ledger", []):
            return False
        ledger["read_ledger"] = compact
        return True
    stale_pin = any("pinned_range" in row and row.get("path") in current_revisions
                    and row.get("revision") != current_revisions[row.get("path")]
                    for row in ledger["reads"])
    # Deduplicating identical old pins may itself fit. Initial pressure still
    # requires removing the remaining pin when current authority says it is old.
    if under_pressure(size) or (pressure and stale_pin):
        def read_group(row: Mapping[str, object]) -> tuple[object, ...]:
            identity: tuple[object, ...] = (row.get("path"), row.get("revision"))
            if evictfix:
                identity += (json.dumps(row.get("lines", row.get("ranges")), sort_keys=True),
                             row.get("content_sha256"))
            return identity

        newest = {read_group(row): row for row in ledger["reads"]}
        current = {row.get("path"): row.get("revision") for row in ledger["reads"]}
        # Explicitly unavailable current authority must not promote the latest
        # historical revision (for example, after a file was removed).
        current.update(current_revisions)
        for row in list(ledger["reads"]):
            if (evictfix and not under_pressure(size)
                    and not ("pinned_range" in row and
                             row.get("revision") != current.get(row.get("path")))):
                continue
            if (row is newest[read_group(row)]
                    and row.get("revision") == current.get(row.get("path"))):
                continue
            ledger["reads"].remove(row)
            _fold(ledger, row, kind="read", exact_ids=exact_ids(row), coverage=coverage)
            if evictfix:
                compact_evicted_reads()
                size = publish()
        # Receipts are journal identities, not assertions that historical source
        # bytes are resident. Spend only as many as the measured fit requires.
        for fact in frame.get("read_facts", []):
            for field, summary_name in (("receipts", "historical_receipts"),
                                        ("journal_workspace_revisions", "historical_workspace_revisions")):
                entries = fact.get(field, [])
                while len(entries) > 1 and (not evictfix or under_pressure(size)):
                    old = entries.pop(0)
                    summary = fact.setdefault(summary_name, {"count": 0, "sha256": _sha([])})
                    summary["count"] += 1
                    summary["sha256"] = _sha([summary["sha256"], old])
                    if evictfix:
                        size = publish()
        _repair_receipts(frame, messages, context, coverage)
        compact_evicted_reads()
        size = publish()

    # One chronological action stream, including ordinary rows, opaque spill
    # and notes. A note inherits its adjacent call's position in the source.
    candidates = []
    for field in ("reads", "edits", "unclassified_actions"):
        rows = ledger.get(field, [])
        for index, row in enumerate(rows):
            ids = exact_ids(row)
            rank = min((order[item] for item in ids if item in order), default=None)
            if rank is None and field == "unclassified_actions":
                turn = row.get("turn_index")
                related = [order[item] for other in rows if other.get("turn_index") == turn
                           for item in exact_ids(other) if item in order]
                rank = min(related) if related else None
                if rank is None:
                    positions = note_order.get(row.get("text_sha256", row.get("sha256")), [])
                    if positions:
                        rank = positions.pop(0)
            # Frozen reduced fixtures may have no semantic transcript. Their
            # stable per-stream row order still gives deterministic oldest-first.
            fallback = row.get("turn_index", index)
            fallback = fallback if type(fallback) is int else index
            candidates.append((rank if rank is not None else fallback, field, index, row))
    # Earlier lossless reductions can fold reads too. Account for their compact
    # identities before deciding that chronological compaction has finished.
    if compact_evicted_reads():
        size = publish()
    newest_pin = next((row for row in reversed(ledger["reads"]) if "pinned_range" in row), None)
    recent_read_frontier = ({id(row) for row in ledger["reads"][-3:]}
                            if evictfix else set())
    for _rank, field, _index, row in sorted(candidates, key=lambda item: item[:3]):
        if not under_pressure(size):
            break
        if row not in ledger[field]:
            continue  # Repairing a removed carrier may already have folded this receipt.
        if row is newest_pin:
            continue
        # The recent read frontier outlives notes and ordinary action history,
        # regardless of their relative chronology. Legacy has no frontier.
        if field == "reads" and id(row) in recent_read_frontier:
            continue
        # A pin that backs a retained receipt cannot disappear without turning
        # those receipt rows into explicit historical evidence as well.
        ledger[field].remove(row)
        _fold(ledger, row, kind=str(row.get("kind", field)), exact_ids=exact_ids(row), coverage=coverage)
        _repair_receipts(frame, messages, context, coverage)
        compact_evicted_reads()
        size = publish()

    # Read-fact receipts have no separate budget. They compete with the same
    # protected envelope, keeping each path's newest freshness receipt first.
    facts = frame.get("read_facts", [])
    for fact in facts:
        receipts = fact.get("receipts", [])
        while under_pressure(size) and len(receipts) > 1:
            old = receipts.pop(0)
            summary = fact.setdefault("historical_receipts", {"count": 0, "sha256": _sha([])})
            summary["count"] += 1
            summary["sha256"] = _sha([summary["sha256"], old])
            size = publish()
        revisions = fact.get("journal_workspace_revisions", [])
        while under_pressure(size) and len(revisions) > 1:
            old = revisions.pop(0)
            summary = fact.setdefault("historical_workspace_revisions", {"count": 0, "sha256": _sha([])})
            summary["count"] += 1
            summary["sha256"] = _sha([summary["sha256"], old])
            size = publish()
    while under_pressure(size) and len(facts) > 1:
        old = facts.pop(0)
        summary = frame.setdefault("historical_read_facts", {"count": 0, "sha256": _sha([])})
        summary["count"] += 1
        summary["sha256"] = _sha([summary["sha256"], old])
        size = publish()

    # Journal-applied edits are duplicated in two views. Drop old repetitions
    # per path together, preserving path order for the existing head inventory.
    edits = frame.get("applied_edits", [])
    reductions = applied_edit_reductions(edits)
    while under_pressure(size):
        row = next(reductions, None)
        if row is None:
            break
        summary = frame.setdefault("historical_applied_edits", {"count": 0, "sha256": _sha([])})
        summary["count"] += 1
        summary["sha256"] = _sha([summary["sha256"], row])
        # Both views are owned by this detached candidate; surviving edit rows
        # are never mutated here, so per-row deep copies provide no isolation.
        ledger["applied_edits"] = list(edits)
        size = publish()

    # The finalizer first spends optional messages and source overhead. Its
    # early history pass must not consume the frontier before those stages.
    for row in list(ledger["reads"]) if evictfix and allow_read_body_eviction else []:
        if not under_pressure(size):
            break
        if row is newest_pin:
            continue
        if row not in ledger["reads"]:
            continue  # Repairing a carrier may fold a later snapshot row.
        ledger["reads"].remove(row)
        _fold(ledger, row, kind="read", exact_ids=exact_ids(row), coverage=coverage)
        _repair_receipts(frame, messages, context, coverage)
        compact_evicted_reads()
        size = publish()

    # Keep the current read body until smaller historical reductions are spent.
    # If that final carrier must leave, the shared reconciler emits its exact
    # absence notice while the receipt identity is folded into CoverageSession.
    newest_pin = next((row for row in reversed(ledger["reads"])
                       if isinstance(row, Mapping) and "pinned_range" in row), None)
    if (allow_read_body_eviction and result_truth_enabled() and under_pressure(size)
            and newest_pin is not None and newest_pin in ledger["reads"]
            and any(fact.get("path") == newest_pin.get("path")
                    and fact.get("current_revision") == newest_pin.get("revision")
                    for fact in frame.get("read_facts", []))):
        ledger["reads"].remove(newest_pin)
        _fold(ledger, newest_pin, kind="read", exact_ids=exact_ids(newest_pin),
              coverage=coverage)
        _repair_receipts(frame, messages, context, coverage)
        size = publish()
        under_pressure(size)

    if compact_evicted_reads():
        size = publish()

    # The frontier is a recent-ID display, not the coverage authority. Only
    # after ordinary history reductions are exhausted, omit it in one step.
    # The exact smaller digest is durably committed before candidate adoption.
    digest = ledger.get("history_digest")
    if (under_pressure(size) and coverage is not None and isinstance(digest, dict)
            and coverage.minimize_frontier(digest)):
        size = publish()

    covered = ledger_ids(ledger) | (coverage.covered_ids if coverage else set())
    missing = original_ids - covered
    # An oversized ID may have been expanded from its bounded row's exact hash.
    missing -= {call_id[:160] for call_id in covered if len(call_id) > 160}
    if missing:
        raise FrameFit5Error("capacity projection lost prior action identities")
    if _expected_ids(context) - covered:
        raise FrameFit5Error("compacted frame lacks prior typed action coverage")
    validate_read_ranges(frame, messages)
    if coverage is not None:
        coverage.commit()
    validate_coverage(frame, context)
    audit.update(after=protected_bytes(body, protected_messages),
                 changed=before != protected_bytes(body, protected_messages),
                 measurement_calls=tokenizer_calls)
    target["messages"] = messages
    return audit


def fit_protected_source_overhead(
        body: dict[str, object], *,
        protected_messages: Sequence[Mapping[str, object]] = (),
        action_context: Mapping[str, object] | None = None,
        exact_tokenizer: bool = False,
) -> None:
    """Fit optional heads and lossless site aliases to the protected ceiling."""
    from kernel_frame_optional_allocation import singleton_alias_candidate
    from kernel_stateframe_framefit5 import (
        fit_headcarry_to_request_budget, _state_payload, _serialize_state_message,
    )

    candidate = copy.deepcopy(body)
    messages = candidate.get("messages", [])
    refresh_read_ranges(messages, action_context)
    if within_protected_capacity(candidate, protected_messages,
                                 exact_tokenizer=exact_tokenizer):
        body["messages"] = messages
        return

    def measure_selected_heads() -> int:
        measured = copy.deepcopy(candidate)
        refresh_read_ranges(measured["messages"], action_context)
        return protected_bytes(measured, protected_messages)

    fit_headcarry_to_request_budget(
        messages, max_request_bytes=protected_limit(candidate,
                                                    exact_tokenizer=exact_tokenizer),
        measure_request_bytes=measure_selected_heads,
    )
    if not within_protected_capacity(candidate, protected_messages,
                                     exact_tokenizer=exact_tokenizer):
        for message in messages:
            if not str(message.get("content", "")).startswith("[StateFrame]\n"):
                continue
            payload, state = _state_payload(message)
            alias = singleton_alias_candidate(state["framefit5"])
            if alias is not None:
                state["framefit5"] = alias
                message["content"] = _serialize_state_message(payload)
            break
    refresh_read_ranges(messages, action_context)
    body["messages"] = messages


def bound_typed_state(frame: dict[str, object], *,
                      semantic_messages: Sequence[Mapping[str, object]],
                      action_context: Mapping[str, object]) -> dict[str, object]:
    """Keep construction below its 2 MiB safety guard without admission trimming.

    Final fitting charges the actual task, schemas, newest exchange and outer
    state against the measured request budget. Legacy keeps its offline cap.
    """
    from kernel_stateframe_framefit5 import (
        MAX_PRODUCT_BYTES, _serialize_state_message, _state_payload,
    )

    body = {"messages": [{"role": "user", "content": _serialize_state_message(
        {"state": {"framefit5": frame}})}], "tools": []}
    legacy = not evictfix_enabled()
    audit = fit_protected_frame(
        body, semantic_messages=semantic_messages, action_context=action_context,
        exact_tokenizer=not legacy,
        max_request_bytes=None if legacy else MAX_PRODUCT_BYTES - 4096,
    )
    if audit["changed"]:
        bounded = _state_payload(body["messages"][0])[1]["framefit5"]
        validate_coverage(bounded, action_context)
        return bounded
    return frame
