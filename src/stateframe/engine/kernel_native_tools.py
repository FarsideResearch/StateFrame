"""Opt-in native tool proposals with ordered, audited batch execution.

Each queued tool still consumes one worker decision and uses the existing dispatch
boundary. History normalizes a batch to assistant/tool pairs; it is not a byte-for-
byte replay of the provider's original assistant message.
"""
from __future__ import annotations

import copy
import os

from kernel_tool_reply import Outcome, tool_reply

MAX_BATCH_CALLS = 16
RESTORE_CANDIDATE = "restore_candidate"


def restore_candidate_schema() -> dict[str, object]:
    """Return the native factual restore contract owned by completion surface."""
    return {
        "type": "object",
        "properties": {
            "checkpoint_id": {"type": "string"},
            "expected_current_revision": {"type": "string"},
            "why": {"type": "string"},
        },
        "required": ["checkpoint_id", "expected_current_revision"],
        "additionalProperties": True,
    }


def restore_candidate_call(checkpoint_id: object, expected_current_revision: object) -> dict[str, object]:
    """Build a typed native call; execution remains at the shared dispatcher."""
    return {
        "type": "function",
        "function": {
            "name": RESTORE_CANDIDATE,
            "arguments": {
                "checkpoint_id": str(checkpoint_id or ""),
                "expected_current_revision": str(expected_current_revision or ""),
            },
        },
    }


def enabled() -> bool:
    value = os.environ.get('REACT_NATIVE_TOOL_BATCHES', '0')
    if value not in {'0', '1'}:
        raise ValueError('REACT_NATIVE_TOOL_BATCHES must be 0 or 1')
    return value == '1'


def validate_configuration() -> bool:
    active = enabled()
    from kernel_step_contract import action_schema_enabled
    if active and (os.environ.get('REACT_STEP_CONTRACT', '0') != '0'
                   or action_schema_enabled()
                   or os.environ.get('REACT_TOOLCALL', '1') != '1'):
        raise ValueError('native tool batches require tool-call mode without a visible action schema')
    return active


class NativeToolBatch:
    """Never ask the model again while a returned batch still has pending calls."""

    def __init__(self):
        self.pending: list[dict] = []
        self.origin: str | None = None
        self.origin_snapshot: str | None = None
        self.index = 0
        self.size = 0
        self.journal = None
        self.inflight: dict | None = None
        self.finalized = False
        self.origin_response_event_id = None
        self.expected_binding = None
        self.selected_ref = None
        self.frozen_tools: list[dict] | None = None
        self._frame_integrity_receipts: list[dict] = []
        self._pending_tool_result_messages: list[dict] = []

    def _lineage(self) -> dict:
        return ({"origin_response_event_id": self.origin_response_event_id}
                if self.origin_response_event_id is not None else {})

    def next(self, ask, messages, **kwargs):
        from kernel_measurement import ACTIVE_REQUEST
        capture = ACTIVE_REQUEST.get()
        if capture is None:
            raise ValueError('native batches require captured requests')
        if self.inflight is not None and not self.finalized:
            # r46 B2 (A3B r45 442 ldev_01 seq 2458-2505 and 5 more runs): a queued call rejected before the
            # step ledger opened a step (e.g. a garbled str_replace path, REJECTED at claim grounding) never reaches
            # close_step's action_closed(), so the next selection crashed the run.  Close it as not executed and
            # drop the rest of the batch (it was built on a decision whose action never ran), then continue.
            journal = self.journal if self.journal is not None else capture.journal
            journal.emit('native_batch_action_force_closed', origin_request_id=self.origin,
                         tool_call_id=self.inflight.get('id'), batch_index=self.index - 1,
                         result='NO_EXECUTION_RECEIPT', reason='unclosed_before_next_selection',
                         **self._lineage())
            call = copy.deepcopy(self.inflight)
            if not any(message.get("role") == "tool" and message.get("tool_call_id") == call.get("id")
                       for message in messages):
                self._pending_tool_result_messages.extend([
                    {"role": "assistant", "content": None, "tool_calls": [{
                        "id": call["id"], "type": call.get("type", "function"),
                        "function": call["function"],
                    }]},
                    tool_reply(call["id"], call["function"]["name"], Outcome.ERROR,
                        "OUTCOME UNKNOWN: no execution receipt; inspect current state before re-issuing this call.",
                        receipt={"code": "host_no_result"}),
                ])
            if self._integrity_enabled():
                self._record_receipt(call, "UNKNOWN", "ACTION_UNRESOLVED: no execution receipt")
            self.inflight = None
            self.journal = journal
            self.discard_pending(
                'previous action closed without an execution receipt',
                unclosed_call_index=self.index,
            )
            # This capture was prepared before the missing-result discovery.
            # The worker must render these facts at a fresh request boundary.
            return "Previous execution outcome is unknown; prepare a fresh decision with its receipts.", []
        if self.inflight is not None or self.finalized:
            raise RuntimeError('previous native action must close before another selection')
        self.journal = capture.journal
        from kernel_pilot import ACTIVE_RUNTIME
        runtime = ACTIVE_RUNTIME.get()
        content = ''
        if not self.pending:
            content, calls = ask(messages, **kwargs)
            # A completed fresh provider decision has consumed the previous
            # batch's feedback. Local queued selections never clear it.
            from kernel_native_prefix import enabled as prefix_enabled
            if prefix_enabled():
                self._consume_model_receipts()
            response_ref = getattr(capture, "selected_response_event_id", None)
            response_lineage = ({"origin_response_event_id": response_ref}
                                if response_ref is not None else {})
            if not calls:
                return content, []
            if not isinstance(calls, list) or len(calls) > MAX_BATCH_CALLS or any(
                    not isinstance(call, dict) or not isinstance(call.get('function'), dict)
                    or not isinstance(call['function'].get('name'), str)
                    or not isinstance(call['function'].get('arguments'), str)
                    for call in calls):
                if self._integrity_enabled() and isinstance(calls, list):
                    for call in calls:
                        self._record_receipt(
                            call if isinstance(call, dict) else {},
                            "NOT_RUN",
                            "BATCH_REJECTED: malformed or oversized tool batch",
                        )
                self.journal.emit('native_batch_rejected', request_id=capture.request_id,
                                  reason='batch exceeds 16 calls or lacks function name and serialized arguments', dispatch=False, **response_lineage)
                return 'Invalid tool-call batch. Return named tools with JSON arguments.', []
            done_positions = [index for index, call in enumerate(calls)
                              if call["function"]["name"] == "done"]
            if done_positions and done_positions != [len(calls) - 1]:
                if self._integrity_enabled():
                    for call in calls:
                        self._record_receipt(
                            call,
                            "NOT_RUN",
                            "BATCH_REJECTED: done must be the unique final call",
                        )
                self.journal.emit("native_batch_rejected", request_id=capture.request_id,
                                  reason="done_must_be_unique_final", calls=calls,
                                  dispatch=False, **response_lineage)
                return ("Invalid tool-call batch: done must appear at most once, as the final call. "
                        "No calls from this batch were executed; submit a corrected batch."), []
            self.origin_response_event_id = response_ref
            self.pending = copy.deepcopy(calls)
            self.origin, self.index, self.size = capture.request_id, 0, len(calls)
            self.origin_snapshot = capture.snapshot_id
            self.expected_binding = copy.deepcopy(getattr(runtime, "prepared_binding", None))
            self.frozen_tools = (copy.deepcopy(kwargs.get("tools"))
                                 if os.environ.get("STATEFRAME_EMPIRICAL_FIXES", "off") == "on"
                                 else None)
            for index, item in enumerate(self.pending):
                item['provider_call_id'] = item.get('id')
                item['id'] = f'native_{self.origin}_{index}'
            self.journal.emit('native_batch_received', request_id=self.origin,
                              origin_snapshot_id=self.origin_snapshot,
                              origin_binding=self.expected_binding,
                              calls=calls, batch_size=self.size,
                              execution='ordered', history='normalized_assistant_tool_pairs', **self._lineage())
        if self.expected_binding is not None:
            if runtime is None:
                raise RuntimeError("bound native queue lost its runtime")
            if runtime.binding() != self.expected_binding:
                self.discard_pending("binding_changed_before_selection")
                return ("Queued proposals discarded because the workspace changed outside "
                        "their recorded action chain. Request a fresh decision.", [])
        if self.index > 0 and self.pending and runtime is not None:
            # r45 one owner (r44c 368 ldev_01 seq 1647): a QUEUED sibling proposed under an older menu is not
            # dispatched into a refusal once the greenlock authority no longer permits its tool -- the rest of the
            # batch is discarded and the model gets a fresh decision under the current menu.
            lock = getattr(runtime, "green_lock", None)
            name = (self.pending[0].get("function") or {}).get("name")
            if lock is not None and isinstance(name, str):
                from kernel_greenlock_menu_authority import _permitted_on_surface, applies, refresh
                if applies(lock):
                    refresh(lock)
                    if not _permitted_on_surface(lock, name):
                        self.discard_pending("authority_menu_changed")
                        return ("Queued proposals discarded because the allowed actions changed after this batch "
                                "was proposed. Request a fresh decision.", [])
        call = self.pending.pop(0)
        self.inflight = call
        selected = self.journal.emit('native_batch_action_selected', request_id=capture.request_id,
                          origin_request_id=self.origin, batch_index=self.index,
                          origin_snapshot_id=self.origin_snapshot,
                          dispatch_snapshot_id=capture.snapshot_id,
                          batch_size=self.size, remaining=len(self.pending),
                          tool_call=call, new_model_request=self.index == 0, **self._lineage())
        self.selected_ref = selected.get("event_id") if isinstance(selected, dict) else None
        self.index += 1
        return content, [call]

    def action_closed(self, record: dict) -> None:
        if self.finalized:
            return
        if self.inflight is None:
            return
        if self.expected_binding is not None and not isinstance(record.get("after_binding"), dict):
            raise ValueError("bound native action closure requires its resulting binding")
        if self._integrity_enabled():
            result = record.get("result", record.get("result_class", "UNKNOWN"))
        else:
            result = record["result"]
        closed = self.journal.emit('native_batch_action_closed', origin_request_id=self.origin,
                          tool_call_id=self.inflight['id'], batch_index=self.index - 1,
                          step_record_id=record['step_record_id'], result=result, **self._lineage())
        self._note_same_file_edit(record, result, closed)
        if self._integrity_enabled():
            self._record_closed_receipt(record, result)
        from kernel_native_prefix import enabled as prefix_enabled
        from kernel_continuation_menu_authority import EDIT_TOOLS
        if prefix_enabled() and self.inflight["function"]["name"] in EDIT_TOOLS and result != "APPLIED":
            self.discard_pending(
                "mutation failed; request a fresh model decision",
                refused_call_index=self.index,
                refusal_reason=self._failure_reason(record, result),
            )
        self.inflight = None
        if (self.expected_binding is not None
                and result in {"NO_EXECUTION_RECEIPT", "REJECTED_CONTRACT",
                                         "DENIED_PATH", "REFUSED_PREDISPATCH"}
                and self.expected_binding != record.get("after_binding")):
            # A refused call cannot make an external mutation part of this batch.
            self.discard_pending(
                "binding_changed_without_execution",
                refused_call_index=self.index,
                refusal_reason=self._failure_reason(record, result),
            )

        if isinstance(record.get("after_binding"), dict):
            self.expected_binding = copy.deepcopy(record["after_binding"])

    def _note_same_file_edit(self, record: dict, result: object, closed: object) -> None:
        """Report repeated-path execution only from changed, bound APPLIED closures."""
        if result != "APPLIED" or not isinstance(closed, dict) or not closed.get("event_id"):
            return
        function = self.inflight.get("function", {}) if isinstance(self.inflight, dict) else {}
        if function.get("name") != "str_replace":
            return
        import json
        try:
            arguments = json.loads(function.get("arguments", "{}"))
        except (TypeError, ValueError):
            return
        path = arguments.get("path") if isinstance(arguments, dict) else None
        before_binding = record.get("before_binding")
        after_binding = record.get("after_binding")
        before = before_binding.get("product_files", {}) if isinstance(before_binding, dict) else {}
        after = after_binding.get("product_files", {}) if isinstance(after_binding, dict) else {}
        if (not isinstance(self.origin, str) or not isinstance(path, str)
                or not isinstance(before, dict) or not isinstance(after, dict)
                or not isinstance(before.get(path), str) or not isinstance(after.get(path), str)
                or before[path] == after[path]):
            return
        if getattr(self, "_same_file_edit_origin", None) != self.origin:
            self._same_file_edit_origin = self.origin
            self._same_file_edit_proofs = {}
        prior = self._same_file_edit_proofs.get(path)
        if prior is not None and prior["after"] == before[path]:
            self.journal.emit("same_file_multi_edit_applied", producer="kernel_native_tools",
                              dispatch=False, origin_request_id=self.origin, path=path,
                              previous_closure_ref=prior["event_id"], closure_ref=closed["event_id"],
                              before_revision=before[path], after_revision=after[path],
                              proof_boundary="completed_native_effect", on_wire=False)
        self._same_file_edit_proofs[path] = {"event_id": closed["event_id"], "after": after[path]}

    def force_close(self, reason: str, *, result: str = "REFUSED_PREDISPATCH") -> dict | None:
        """Close the selected call before a host-owned menu transition.

        A transition can invalidate the menu that selected ``inflight``.  The
        normal worker close path cannot be used because no tool execution took
        place, so emit an explicit refusal receipt and replan all siblings.
        """
        if self.inflight is None:
            return None
        if self.journal is None:
            raise RuntimeError("in-flight native call lacks its journal")
        call = copy.deepcopy(self.inflight)
        self.journal.emit(
            "native_batch_action_force_closed",
            origin_request_id=self.origin,
            tool_call_id=call.get("id"),
            batch_index=self.index - 1,
            result=result,
            reason=reason,
            **self._lineage(),
        )
        if self._integrity_enabled():
            self._record_receipt(call, "REJECTED", reason)
        self.inflight = None
        self.discard_pending(
            reason,
            receipt_status="REJECTED",
            receipt_prefix="STAGE_TRANSITION_REPLAN",
            refused_call_index=self.index,
            refusal_reason=reason,
        )
        return call

    def skip_inflight(self, reason: str) -> dict | None:
        """Close a selected call that the host skips before dispatch."""
        if self.inflight is None:
            return None
        if self.journal is None:
            raise RuntimeError("in-flight native call lacks its journal")
        call = copy.deepcopy(self.inflight)
        self.journal.emit(
            "native_batch_action_force_closed",
            origin_request_id=self.origin,
            tool_call_id=call.get("id"),
            batch_index=self.index - 1,
            result="NO_EXECUTION_RECEIPT",
            reason=reason,
            **self._lineage(),
        )
        if self._integrity_enabled():
            self._record_receipt(call, "NOT_RUN", reason)
        self.inflight = None
        self._queue_unexecuted_tool_results([call], reason=reason)
        self.discard_pending(reason)
        return call

    def discard_pending(
            self,
            reason: str,
            *,
            receipt_status: str = "NOT_RUN",
            receipt_prefix: str = "BATCH_STOPPED_AFTER_REJECTION",
            refused_call_index: int | None = None,
            refusal_reason: str | None = None,
            unclosed_call_index: int | None = None,
    ) -> None:
        if self.pending:
            if self.journal is None:
                raise RuntimeError("queued native calls lack their journal")
            calls = copy.deepcopy(self.pending)
            self.journal.emit("native_batch_discarded", origin_request_id=self.origin,
                              calls=calls, reason=reason, **self._lineage())
            if self._integrity_enabled():
                for call in calls:
                    self._record_receipt(
                        call, receipt_status,
                        f"NOT EXECUTED: {receipt_prefix}: {reason}",
                    )
            self._queue_unexecuted_tool_results(
                calls,
                reason=reason,
                refused_call_index=refused_call_index,
                refusal_reason=refusal_reason,
                unclosed_call_index=unclosed_call_index,
            )
            self.pending = []
            self.frozen_tools = None

    def _queue_unexecuted_tool_results(
            self,
            calls: list[dict],
            *,
            reason: str,
            refused_call_index: int | None = None,
            refusal_reason: str | None = None,
            unclosed_call_index: int | None = None,
    ) -> None:
        if not calls:
            return
        assistant_calls = []
        tool_results = []
        safe_reason = " ".join(str(reason or "unspecified").split())[:240] or "unspecified"
        safe_refusal = " ".join(str(refusal_reason or "").split())[:240]
        for call in calls:
            if not isinstance(call, dict):
                continue
            call_id = call.get("id")
            if not isinstance(call_id, str) or not call_id:
                continue
            function = call.get("function")
            function = function if isinstance(function, dict) else {}
            name = function.get("name")
            arguments = function.get("arguments")
            if not isinstance(name, str) or not name:
                name = "unknown"
            if not isinstance(arguments, str):
                arguments = "{}"
            assistant_calls.append({
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            })
            if refused_call_index is not None and safe_refusal:
                content = (
                    f"NOT EXECUTED: discarded after call {refused_call_index} was refused "
                    f"({safe_refusal}); re-issue it."
                )
            elif unclosed_call_index is not None:
                content = (
                    f"NOT EXECUTED: skipped after call {unclosed_call_index} had no execution "
                    f"receipt ({safe_reason}); re-issue it."
                )
            else:
                content = f"NOT EXECUTED: skipped because {safe_reason}; re-issue it."
            tool_results.append(tool_reply(
                call_id, name, Outcome.ERROR, content,
                receipt={"code": "tool_not_executed"},
            ))
        if assistant_calls:
            self._pending_tool_result_messages.append({
                "role": "assistant", "content": None, "tool_calls": assistant_calls,
            })
            self._pending_tool_result_messages.extend(tool_results)

    def append_unexecuted_tool_results(self, messages: list[dict]) -> int:
        """Append queued tool results to conversation history exactly once."""
        if not self._pending_tool_result_messages:
            return 0
        # A host exit can discover the missing closure after the worker already
        # appended a truthful refusal. Do not manufacture a second tool result
        # when draining the queue into the full conversation history.
        existing = {message.get("tool_call_id") for message in messages if message.get("role") == "tool"}
        proposed = {call.get("id") for message in messages if message.get("role") == "assistant"
                    for call in (message.get("tool_calls") or []) if isinstance(call, dict)}
        queued = []
        for message in self._pending_tool_result_messages:
            if message.get("role") == "tool":
                if message.get("tool_call_id") in existing:
                    continue
                existing.add(message.get("tool_call_id"))
            elif message.get("role") == "assistant" and message.get("tool_calls"):
                calls = [call for call in message["tool_calls"]
                         if call.get("id") not in existing and call.get("id") not in proposed]
                if not calls:
                    continue
                proposed.update(call.get("id") for call in calls)
                message = {**message, "tool_calls": calls}
            queued.append(message)
        messages.extend(queued)
        self._pending_tool_result_messages = []
        return sum(message.get("role") == "tool" for message in queued)

    def finish(self):
        if self.finalized:
            return
        self.finalized = True
        if self.inflight is not None and self.journal is not None:
            self.journal.emit('native_batch_action_unresolved', origin_request_id=self.origin,
                              origin_snapshot_id=self.origin_snapshot, batch_index=self.index - 1,
                              call=self.inflight, effect_status='UNKNOWN',
                              reason='worker terminated without closing selected action', **self._lineage())
            if self._integrity_enabled():
                self._record_receipt(
                    self.inflight, "UNKNOWN", "ACTION_UNRESOLVED: worker terminated",
                )
            call_id = self.inflight.get("id")
            if isinstance(call_id, str) and call_id:
                function = self.inflight.get("function")
                name = function.get("name") if isinstance(function, dict) else "unknown"
                if not isinstance(name, str) or not name:
                    name = "unknown"
                self._pending_tool_result_messages.append(tool_reply(
                    call_id, name, Outcome.ERROR,
                    (
                        "OUTCOME UNKNOWN: worker terminated without an execution receipt; "
                        "inspect current state before re-issuing this call."
                    ),
                    receipt={"code": "host_no_result"},
                ))
        self.inflight = None
        if self.pending and self.journal is not None:
            calls = copy.deepcopy(self.pending)
            self.journal.emit('native_batch_unexecuted', origin_request_id=self.origin,
                              origin_snapshot_id=self.origin_snapshot,
                              next_index=self.index, calls=calls,
                              reason='worker ended before pending batch exhausted', **self._lineage())
            if self._integrity_enabled():
                for call in calls:
                    self._record_receipt(
                        call, "NOT_RUN", "BATCH_UNEXECUTED: worker ended before selection",
                    )
            self._queue_unexecuted_tool_results(
                calls, reason="worker ended before pending batch exhausted",
            )
            self.pending = []
        self.frozen_tools = None

    def _integrity_enabled(self) -> bool:
        from kernel_frame_edit_integrity import enabled as integrity_enabled
        from kernel_native_prefix import enabled as prefix_enabled
        return integrity_enabled() or prefix_enabled()

    @staticmethod
    def _call_index(call: dict, fallback: int = 0) -> int:
        value = call.get("id") if isinstance(call, dict) else None
        try:
            return int(str(value).rsplit("_", 1)[1])
        except (IndexError, TypeError, ValueError):
            return int(fallback)

    @staticmethod
    def _failure_reason(record: dict, result: object) -> str:
        detail = str(record.get("observation") or record.get("detail") or result or "unknown")
        if "complete target source was not delivered" in detail:
            return "TARGET_SOURCE_NOT_DELIVERED: " + detail
        normalized = "_".join(str(result).upper().split()) or "UNKNOWN_RESULT"
        return f"RESULT_{normalized}: {detail}"

    def _record_closed_receipt(self, record: dict, result: object) -> None:
        from kernel_step_contract import successful_lookup_status
        tool = str(self.inflight.get("function", {}).get("name", "")) if isinstance(
            self.inflight, dict) else ""
        lookup_applied = successful_lookup_status(
            tool, str(result), str(record.get("observation") or record.get("detail") or result),
        ) == "APPLIED"
        unresolved = str(result).upper() in {"UNKNOWN", "NO_EXECUTION_RECEIPT"}
        successful = lookup_applied or str(result).upper() in {
            "APPLIED", "OK", "ACCEPTED", "PASS", "PASSED", "GREEN", "SUCCESS",
            "CERTIFIED", "REUSED_VERIFICATION", "NO_OP",
        }
        self._record_receipt(
            self.inflight,
            "UNKNOWN" if unresolved else "APPLIED" if successful else "REJECTED",
            "" if successful else self._failure_reason(record, result),
        )

    def _record_receipt(self, call: dict, status: str, reason: str) -> None:
        if not self._integrity_enabled():
            return
        call = call if isinstance(call, dict) else {}
        function = call.get("function", {}) if isinstance(call.get("function"), dict) else {}
        receipt = {
            "call_id": str(call.get("id", "unknown")),
            "batch_index": self._call_index(call),
            "tool": str(function.get("name", "unknown")),
            "status": str(status),
            "reason": str(reason),
        }
        self._frame_integrity_receipts.append(receipt)
        if self.journal is not None:
            self.journal.emit(
                "native_batch_call_receipt",
                origin_request_id=self.origin,
                tool_call_id=receipt["call_id"],
                batch_index=receipt["batch_index"],
                tool=receipt["tool"],
                receipt_status=receipt["status"],
                receipt_reason=receipt["reason"],
                **self._lineage(),
            )

    def take_model_receipts(self) -> list[dict]:
        """Render pending receipts; prefix mode acknowledges only a completed send."""
        if not self._integrity_enabled() or not self._frame_integrity_receipts:
            return []
        from kernel_native_prefix import enabled as prefix_enabled
        if prefix_enabled():
            return copy.deepcopy(self._frame_integrity_receipts)
        return self._consume_model_receipts()

    def _consume_model_receipts(self) -> list[dict]:
        if not self._frame_integrity_receipts:
            return []
        rows = copy.deepcopy(self._frame_integrity_receipts)
        self._frame_integrity_receipts.clear()
        if self.journal is not None:
            self.journal.emit(
                "native_batch_receipts_delivered",
                origin_request_id=self.origin,
                receipt_count=len(rows),
                receipts=copy.deepcopy(rows),
                **self._lineage(),
            )
        return rows

    @staticmethod
    def render_model_receipts(receipts: list[dict]) -> str:
        lines = ["[NATIVE BATCH RECEIPTS]", "Every queued call has an explicit host receipt."]
        for receipt in receipts:
            status = str(receipt.get("status", "UNKNOWN"))
            reason = str(receipt.get("reason", ""))
            suffix = f"(reason={reason})" if reason else ""
            lines.append(
                f"call {receipt.get('batch_index', '?')} {receipt.get('tool', 'unknown')}: "
                f"{status}{suffix}"
            )
        return "\n".join(lines)

    def model_receipt_message(self) -> dict | None:
        receipts = self.take_model_receipts()
        if not receipts:
            return None
        return {"role": "user", "content": self.render_model_receipts(receipts)}

    def menu_for_pending(self, tools):
        """Keep the provider's menu stable until an ordered batch is exhausted.

        Green-lock/applicability changes still guard each dispatched action, but
        a sibling proposed under the old menu is parsed as that old tool rather
        than misreported as an unknown tool on the next request.
        """
        if os.environ.get("STATEFRAME_EMPIRICAL_FIXES", "off") != "on":
            return tools
        if not self.pending or self.frozen_tools is None:
            return tools
        current_names = [item.get("function", {}).get("name") for item in tools]
        frozen_names = [item.get("function", {}).get("name") for item in self.frozen_tools]
        if current_names != frozen_names and self.journal is not None:
            self.journal.emit(
                "native_batch_menu_frozen",
                origin_request_id=self.origin,
                current_tools=current_names,
                frozen_tools=frozen_names,
                reason="menu frozen for batch duration",
                dispatch=False,
                **self._lineage(),
            )
        return copy.deepcopy(self.frozen_tools)


def instructions(base: str) -> str:
    return base.replace('Choose one tool per decision.',
        'You may return up to 16 tool calls in one response. They execute in the order '
        'returned, without another model turn between them. Each tool consumes one decision. '
        'Wait for results before choosing actions that depend on unknown results. '
        'Use done only as the final call when your work is complete.')


def frame_instructions(base: str) -> str:
    return base.replace('each request is a fresh decision',
                        'each model request receives a fresh view of state').replace(
        'select one useful action.', 'select useful tool calls. A returned batch executes in order; '
        'the next model request receives the resulting state, without conversation history.')
