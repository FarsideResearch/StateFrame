"""Exact native proposal-queue replay for explicit continuation restart.

A selected but unclosed call is never replayed. Retained proposals are not
permission to execute against changed source; that case requests a new decision.
"""
from __future__ import annotations

import copy
import json

from kernel_continuation_restart import WorkerResumeError
from kernel_native_tools import MAX_BATCH_CALLS
from kernel_tool_reply import Outcome, tool_reply

FIELDS = ("pending", "origin", "origin_snapshot", "index", "size",
          "inflight", "finalized", "origin_response_event_id", "expected_binding")


def empty() -> dict:
    return dict(pending=[], origin=None, origin_snapshot=None, index=0, size=0,
                inflight=None, finalized=False, origin_response_event_id=None, expected_binding=None)


def snapshot(batch) -> dict | None:
    if batch is None:
        return None
    if batch.inflight is not None or batch.finalized:
        raise WorkerResumeError("native checkpoint requires a closed, active boundary")
    if (getattr(batch, "_pending_tool_result_messages", None)
            or getattr(batch, "_frame_integrity_receipts", None)
            or (batch.pending and getattr(batch, "frozen_tools", None) is not None)):
        raise WorkerResumeError("native menu/result checkpoint requires reconciliation")
    return {key: copy.deepcopy(getattr(batch, key)) for key in FIELDS}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise WorkerResumeError("invalid native replay: " + message)


def replay(records: list[dict]) -> dict:
    state = empty()
    closed_records = {}
    responses = {}
    for event in records:
        kind = event["kind"]
        if kind == "response_received":
            responses[event["event_id"]] = event
        elif kind == "action_outcome_closed":
            record = event["record"]
            closed_records[record["step_record_id"]] = record
        elif kind == "native_batch_received":
            require(not state["pending"] and state["inflight"] is None, "overlapping batches")
            calls = event.get("calls")
            require(isinstance(calls, list) and 0 < len(calls) <= MAX_BATCH_CALLS, "batch size")
            require(event.get("batch_size") == len(calls), "declared batch size")
            require(all(isinstance(call, dict) and isinstance(call.get("function"), dict)
                        and isinstance(call["function"].get("name"), str)
                        and isinstance(call["function"].get("arguments"), str)
                        for call in calls), "call structure")
            done = [i for i, call in enumerate(calls) if call["function"]["name"] == "done"]
            require(not done or done == [len(calls) - 1], "DONE ordering")
            origin, origin_snapshot = event.get("request_id"), event.get("origin_snapshot_id")
            require(isinstance(origin, str) and bool(origin)
                    and isinstance(origin_snapshot, str) and bool(origin_snapshot), "origin")
            response = responses.get(event.get("origin_response_event_id"))
            require(response is not None and response.get("request_id") == origin,
                    "batch lacks its selected response")
            try:
                choice = response["response"]["choices"][0]
                message = choice["message"]
                provider_calls = message.get("tool_calls") or []
                if not provider_calls:
                    from probes.react_repair import parse_dsml_tool_calls
                    provider_calls = parse_dsml_tool_calls(
                        (message.get("content") or "").strip() or message.get("reasoning_content") or "")
            except (KeyError, IndexError, TypeError, ValueError) as error:
                raise WorkerResumeError("invalid native response payload") from error
            require(choice.get("finish_reason") != "length" and provider_calls == calls,
                    "queue differs from selected provider response")
            state = empty()
            state.update(pending=copy.deepcopy(calls), origin=origin,
                         origin_snapshot=origin_snapshot, size=len(calls),
                         origin_response_event_id=event.get("origin_response_event_id"),
                         expected_binding=copy.deepcopy(event.get("origin_binding")))
            for index, call in enumerate(state["pending"]):
                call["provider_call_id"] = call.get("id")
                call["id"] = f"native_{origin}_{index}"
        elif kind == "native_batch_action_selected":
            require(bool(state["pending"]) and state["inflight"] is None, "selection without queue")
            require(event.get("origin_request_id") == state["origin"]
                    and event.get("origin_snapshot_id") == state["origin_snapshot"]
                    and event.get("batch_index") == state["index"]
                    and event.get("batch_size") == state["size"]
                    and event.get("remaining") == len(state["pending"]) - 1
                    and event.get("tool_call") == state["pending"][0], "selection differs from proposal")
            state["inflight"] = state["pending"].pop(0)
            state["index"] += 1
        elif kind == "native_batch_action_closed":
            call = state["inflight"]
            record = closed_records.get(event.get("step_record_id"))
            require(call is not None and event.get("origin_request_id") == state["origin"]
                    and event.get("batch_index") == state["index"] - 1
                    and event.get("tool_call_id") == call["id"], "closure identity")
            require(record is not None and record.get("result") == event.get("result")
                    and (record.get("tool") == call["function"]["name"]
                         or (record.get("tool") == "invalid_proposal"
                             and record.get("result") == "NO_EXECUTION_RECEIPT")), "missing action closure")
            require(isinstance(record.get("after_binding"), dict), "closure lacks resulting binding")
            state["expected_binding"] = copy.deepcopy(record["after_binding"])
            state["inflight"] = None
        elif kind == "native_batch_discarded":
            require(event.get("origin_request_id") == state["origin"]
                    and event.get("calls") == state["pending"] and bool(state["pending"]),
                    "discard differs from queue")
            raise WorkerResumeError("native discarded-result delivery requires reconciliation")
        elif kind in {"native_batch_action_unresolved", "native_batch_unexecuted"}:
            raise WorkerResumeError("native terminal/unresolved batch requires reconciliation")
        elif kind in {"native_batch_call_receipt", "native_batch_receipts_delivered"}:
            # Schema 4 has no receipt-delivery state. A crash after a checkpoint
            # must not replay its tail while silently dropping model feedback.
            raise WorkerResumeError("native receipt-delivery replay requires reconciliation")
    return state


def prepare_restore(runtime, saved: dict, records: list[dict], boundary: dict) -> bool:
    """Validate checkpoint and safe post-checkpoint receive; return retained receive."""
    active = runtime.journal.manifest["native_tool_batches"]
    value = saved.get("native")
    if not active:
        require(value is None, "native state in single-action profile")
        runtime.resume_native_state = None
        return False
    require(isinstance(value, dict) and set(value) == set(FIELDS), "checkpoint fields")
    prefix = [event for event in records if event["sequence"] <= boundary["sequence"]]
    require(value == replay(prefix), "checkpoint differs from journal")
    result = replay(records)
    require(result["inflight"] is None and not result["finalized"], "selected call remains unclosed")
    received = [event for event in records if event["sequence"] > boundary["sequence"]
                and event["kind"] == "native_batch_received"]
    require(len(received) <= 1, "multiple post-checkpoint batches")
    tail_responses = [event for event in records if event["sequence"] > boundary["sequence"]
                      and event["kind"] == "response_received"]
    if tail_responses and not received:
        raise WorkerResumeError("unconsumed native response requires reconciliation before resampling")
    if received:
        requests = {event.get("request_id") for event in records
                    if event["sequence"] > boundary["sequence"] and event["kind"] == "request_submitted"}
        require(received[0]["request_id"] in requests, "received batch lacks submitted request")
    require(not result["pending"] or isinstance(result["expected_binding"], dict),
            "pending queue lacks its expected binding")
    if result["pending"] and runtime.journal.manifest.get("stateframe_empirical_fixes") == "on":
        raise WorkerResumeError("native frozen menu replay requires reconciliation")
    runtime.resume_native_state = result
    runtime.resume_native_binding = copy.deepcopy(saved["binding"])
    return bool(received)


def restore_batch(runtime) -> None:
    batch = runtime.native_batch
    value = runtime.resume_native_state
    require(batch is not None and value is not None, "missing attached holder")
    require(snapshot(batch) == empty(), "holder already used before restoration")
    for key, item in value.items():
        setattr(batch, key, copy.deepcopy(item))
    batch.journal = runtime.journal
    basis = batch.expected_binding if batch.pending else runtime.resume_native_binding
    changed = runtime.binding() != basis
    if changed:
        batch.discard_pending("restart_binding_changed")
    runtime.journal.emit("continuation_native_restored",
                        origin_request_id=batch.origin, next_index=batch.index,
                        remaining=len(batch.pending), binding_changed=changed,
                        new_model_request_required=not bool(batch.pending))


def latest_closed_exchange(records: list[dict], ledger: list[dict]) -> tuple[str | None, list[dict]]:
    """Reconstruct a public pair after native replay has validated this prefix.

    No dialogue or model reasoning survives. The normal exchange builder still
    checks receipt presence, public projection, arguments, and current binding.
    """
    closed = [event["record"] for event in records if event["kind"] == "action_outcome_closed"]
    require(ledger == closed, "exchange ledger differs from closed journal actions")
    if not closed:
        return None, []
    record = closed[-1]
    if record["tool"] == "invalid_proposal":
        require(record["result"] == "NO_EXECUTION_RECEIPT", "invalid proposal claims execution")
        # No selected call exists for prose/empty responses. Preserve the
        # failure in the ledger; there is no tool exchange to reconstruct.
        return record["step_record_id"], []
    endings = [event for event in records if event["kind"] == "native_batch_action_closed"
               and event.get("step_record_id") == record["step_record_id"]]
    require(len(endings) == 1, "exchange lacks unique native closure")
    ending = endings[0]
    selections = [event for event in records if event["kind"] == "native_batch_action_selected"
                  and event["sequence"] < ending["sequence"]
                  and event.get("tool_call", {}).get("id") == ending["tool_call_id"]]
    require(len(selections) == 1, "exchange lacks unique selected call")
    selected = selections[0]["tool_call"]
    try:
        arguments = json.loads(selected["function"]["arguments"])
    except (KeyError, TypeError, ValueError) as error:
        raise WorkerResumeError("invalid native exchange arguments") from error
    require(isinstance(arguments, dict), "exchange arguments are not an object")
    arguments = {key: value for key, value in arguments.items() if key not in {"why", "reason"}}
    call = {"id": selected["id"], "type": selected.get("type", "function"),
            "function": {"name": selected["function"]["name"],
                         "arguments": json.dumps(arguments)}}
    require(call["function"]["name"] == record["tool"]
            or record["tool"] == "invalid_proposal", "exchange tool differs from closure")
    return record["step_record_id"], [
        {"role": "assistant", "content": "", "tool_calls": [call]},
        tool_reply(call["id"], call["function"]["name"], Outcome.ERROR, "",
                   receipt={"cause": "missing_reply"}),
    ]
