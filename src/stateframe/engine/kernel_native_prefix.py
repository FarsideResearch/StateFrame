"""Effect-only authority for a model-planned, ordered native action batch."""
from __future__ import annotations

import hashlib
import json
import os

from kernel_continuation_state import ContinuationError
from kernel_measurement import digest

KINDS = frozenset({"native_batch_received", "native_batch_action_selected",
                  "native_batch_action_closed", "native_batch_discarded",
                  "native_batch_unexecuted", "native_batch_action_unresolved",
                  "request_source_closure"})
VALIDATION_KINDS = KINDS | frozenset({
    "action_outcome_closed", "request_prepared", "request_final_wire_prepared",
})


def enabled() -> bool:
    value = os.environ.get("KERNEL_NATIVE_PREFIX_AUTHORITY", "0")
    if value not in {"0", "1"}:
        raise ValueError("KERNEL_NATIVE_PREFIX_AUTHORITY must be 0 or 1")
    return value == "1"


def require(condition, message):
    if not condition:
        raise ContinuationError("native prefix: " + message)


def one(rows, message):
    require(len(rows) == 1, message)
    return rows[0]


def arguments(call):
    try:
        value = json.loads(call["function"]["arguments"])
    except (KeyError, TypeError, ValueError) as error:
        raise ContinuationError("native prefix: invalid arguments") from error
    require(isinstance(value, dict), "arguments must be an object")
    return value


def argument_hash(value):
    return digest({k: v for k, v in value.items() if k not in {"why", "reason"}})


def delivered_source(messages, path, revision):
    # Only complete host source blocks count. References and prose never count.
    for message in messages:
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, str):
            continue
        from kernel_source_presentation import strip_callsite_census
        content = strip_callsite_census(content)
        prefix = f"[Working source: {path}]\n"
        candidates = [content[len(prefix):]] if content.startswith(prefix) else []
        if content.startswith(prefix + "Coverage: COMPLETE FILE;"):
            from kernel_source_presentation import working_source_block
            raw = content[len(prefix):].partition("\n")[2]
            if working_source_block(path, raw, exists=True) == content:
                candidates.append(raw)
        header = f"[Retained source: {path}]\n[Source metadata]\n"
        marker, end = f"\n[Source content: {path}]\n", f"\n[End source content: {path}]"
        if content.startswith(header) and marker in content and content.endswith(end):
            metadata, raw = content[len(header):].split(marker, 1)
            try:
                meta = json.loads(metadata)
            except ValueError:
                continue
            if (meta.get("path") == path and meta.get("revision") == revision
                    and meta.get("freshness") == "current" and meta.get("start_line") == 1
                    and meta.get("start_column", 0) == 0
                    and meta.get("omitted_chars_in_selection") == 0
                    and meta.get("more_content_available") is False):
                candidates.append(raw[:-len(end)])
        # STATEFRAME_REREAD_FIX: plain "[Current source: path]" blocks are the same
        # complete source under another label; still hash-verified below.
        from kernel_reread_fix import enabled as _reread_fix_enabled
        if _reread_fix_enabled():
            from kernel_reread_fix import current_source_candidates
            candidates.extend(current_source_candidates(content, path))
        for raw in candidates:
            if "sha256:" + hashlib.sha256(raw.encode()).hexdigest() == revision:
                return raw
    raise ContinuationError("native prefix: complete target source was not delivered")


def validate(evidence, state, data):
    from kernel_continuation_evidence import binding_versions
    require(evidence.native_prefix, "authority disabled")
    require(data.get("native_selected_ref") not in evidence.native_consumed, "selection already consumed")
    selected = evidence.event(data.get("native_selected_ref"), "native_batch_action_selected")
    fence = evidence.event(data.get("dispatch_ref"), "step_dispatch_binding")
    index = selected.get("batch_index")
    require(type(index) is int and 0 < index < 16, "not a queued action")
    require(selected["sequence"] < fence["sequence"], "selection follows dispatch")
    require(selected.get("request_id") == data.get("request_id")
            and selected.get("dispatch_snapshot_id") == data.get("snapshot_id"),
            "selection belongs to another request")
    origin = selected.get("origin_request_id")
    events = evidence.native_validation_events(origin)
    batch = one([e for e in events if e["kind"] == "native_batch_received"
                 and e.get("request_id") == origin], "missing or duplicate batch")
    response = evidence.event(batch.get("origin_response_event_id"), "response_received")
    submitted = evidence.event(response.get("transport_attempt_id"), "request_submitted")
    require(submitted["sequence"] < response["sequence"] < batch["sequence"]
            < selected["sequence"], "invalid proposal order")
    require(response.get("request_id") == submitted.get("request_id") == origin
            and submitted.get("snapshot_id") == batch.get("origin_snapshot_id"),
            "foreign proposal")
    from kernel_native_prefix_delivery import submitted_payload, validate_prepared_delivery
    payload = submitted_payload(evidence, submitted)
    prepared = one([e for e in events if e["kind"] == "request_prepared"
                    and e.get("request_id") == origin], "missing or duplicate prepared request")
    final_delivery = validate_prepared_delivery(evidence.final_delivery_events(submitted), prepared, submitted, payload)
    choices = response.get("response", {}).get("choices", [])
    require(len(choices) == 1 and choices[0].get("finish_reason") != "length",
            "ambiguous or truncated provider choice")
    calls = batch.get("calls")
    require(isinstance(calls, list) and 1 < len(calls) <= 16
            and calls == choices[0].get("message", {}).get("tool_calls")
            and index < len(calls), "proposal calls differ")
    require(all(isinstance(c, dict) and isinstance(c.get("function"), dict)
                and isinstance(c["function"].get("name"), str)
                and isinstance(c["function"].get("arguments"), str) for c in calls),
            "malformed proposal")
    done = [i for i, call in enumerate(calls) if call["function"]["name"] == "done"]
    require(not done or done == [len(calls)-1], "done must be unique and final")
    # Future siblings cannot poison an earlier valid effect. The queue owns
    # whole-list syntax and done ordering; this proof owns only the selected prefix.
    args = [arguments(c) for c in calls[:index+1]]
    names = [c["function"]["name"] for c in calls[:index+1]]
    from kernel_continuation_evidence import READ_ONLY, result_class_disposition
    from kernel_continuation_menu_authority import EDIT_TOOLS
    require(all(name in EDIT_TOOLS | READ_ONLY | {"run_tests"} for name in names),
            "prefix action has no replay contract")
    menu = payload.get("tools")
    require(isinstance(menu, list) and all(name in {
        item.get("function", {}).get("name") for item in menu if isinstance(item, dict)
    } for name in names), "prefix tool was not offered")
    expected = batch.get("origin_binding")
    require(isinstance(expected, dict), "missing origin binding")
    from kernel_native_effect_chain import OrderedSources, revision
    sources = OrderedSources(events, prepared, payload, expected, final_delivery=final_delivery,
                             source_evidence=evidence)
    prior_sequence = batch["sequence"]
    for number in range(index + 1):
        selection = one([e for e in events if e["kind"] == "native_batch_action_selected"
                         and e.get("origin_request_id") == origin and e.get("batch_index") == number],
                        "missing or duplicate selection")
        call_id = f"native_{origin}_{number}"
        require(selection["sequence"] > prior_sequence
                and selection.get("origin_response_event_id") == response["event_id"]
                and selection.get("origin_snapshot_id") == batch.get("origin_snapshot_id")
                and selection.get("tool_call", {}).get("id") == call_id
                and selection["tool_call"].get("provider_call_id") == calls[number].get("id")
                and selection["tool_call"].get("function") == calls[number]["function"],
                "selection does not match original proposal")
        path = sources.apply(names[number], args[number], number, pending=number == index)
        if number == index:
            require(selection["event_id"] == selected["event_id"], "wrong selected index")
            break
        closed = one([e for e in events if e["kind"] == "native_batch_action_closed"
                      and e.get("origin_request_id") == origin and e.get("batch_index") == number],
                     "missing or duplicate prefix closure")
        outcome = one([e for e in events if e["kind"] == "action_outcome_closed"
                       and e.get("record", {}).get("step_record_id") == closed.get("step_record_id")],
                      "missing prefix step")
        _, record = evidence._closed(outcome["event_id"])
        effect = state.effects.get(record.get("continuation_effect_id"))
        require(selection["sequence"] < outcome["sequence"] < closed["sequence"]
                and closed.get("tool_call_id") == call_id
                and closed.get("origin_response_event_id") == response["event_id"]
                and closed.get("result") == record.get("result")
                and (record.get("result") == "APPLIED" if names[number] in EDIT_TOOLS else
                     result_class_disposition(names[number], record.get("result")) == "settled_applied")
                and record.get("tool") == names[number]
                and record.get("arguments_hash") == argument_hash(args[number])
                and record.get("continuation_id") == data.get("intention_id")
                and effect is not None and effect.status == "applied"
                and effect.tool == names[number]
                and effect.arguments_hash == record.get("arguments_hash")
                and effect.intention_id == data.get("intention_id")
                and effect.receipt_ref == outcome["event_id"]
                and effect.dependencies == binding_versions(expected)
                and record.get("before_binding") == expected, "prefix lacks successful bound effect")
        after = record.get("after_binding")
        require(isinstance(after, dict), "missing resulting binding")
        before_files, after_files = expected.get("product_files", {}), after.get("product_files", {})
        if path is None:
            require(after == expected, "non-mutating prefix changed workspace binding")
        else:
            planned_files = dict(before_files)
            planned_files[path] = revision(sources.texts[path])
            require(after_files == planned_files,
                    "prefix result differs from exact declared replacement or changed another product")
        ignored = {"artifact", "product_revision", "product_files"}
        require({k: v for k, v in expected.items() if k not in ignored}
                == {k: v for k, v in after.items() if k not in ignored},
                "prefix changed harness or environment")
        expected, prior_sequence = after, closed["sequence"]
    require(not any(e["kind"] in {"native_batch_discarded", "native_batch_unexecuted",
                                 "native_batch_action_unresolved"}
                    and e.get("origin_request_id") == origin
                    and batch["sequence"] < e["sequence"] < selected["sequence"]
                    for e in events), "batch was abandoned")
    require(expected == data.get("current_binding"), "unexpected workspace change")
    require(data.get("tool") == names[index]
            and data.get("arguments_hash") == argument_hash(args[index]), "wrong dispatched edit")
    require(not any(effect.dispatch_ref == fence["event_id"]
                    or effect.request_id == selected["request_id"]
                    for effect in state.effects.values()), "selection already consumed")
