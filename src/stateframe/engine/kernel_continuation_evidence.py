"""Host evidence joins for continuation transitions; provider prose has no authority."""
from __future__ import annotations

from collections.abc import Mapping
from heapq import merge
import os

from kernel_continuation_state import ContinuationError, ContinuationState, text
from kernel_measurement import digest
from kernel_verdict import SATISFIED, _selectors, evaluate_required_tests

EVIDENCE_KINDS = frozenset({
    "action_outcome_closed", "action_receipt", "step_dispatch_binding",
    "test_subprocess_completed", "completion_certification",
    "host_obligation_check", "host_repair_check",
    "continuation_public_certification", "continuation_signal", "continuation_control",
    "observation_probe_completed",
    "request_prepared", "request_submitted", "request_body", "response_received",
    "request_final_wire_prepared", "provider_request_not_dispatched",
})
# Host obligation receipts are a separate evidence producer from read/action
# effects, but their result classes are registered here so continuation-aware
# consumers cannot silently classify REUSED or COUNTEREXAMPLE as unknown.
HOST_OBLIGATION_RESULT_CLASSES = frozenset({
    "PASS", "COUNTEREXAMPLE", "NOT_EXECUTED", "HARNESS_ERROR", "REUSED",
})
HOST_OBLIGATION_RESULT_DISPOSITIONS = {
    result: "host_observed" for result in HOST_OBLIGATION_RESULT_CLASSES
}

# ``run_tests`` has a different result vocabulary from host obligation checks.
# Keep the exit-code boundary in one registry so an executor cannot emit an
# exit-2/3 collection or internal error that continuation later treats as an
# unresolved effect.
RUN_TEST_RESULT_CLASSES_BY_EXIT = {
    0: "GREEN",
    1: "RED",
    2: "HARNESS_ERROR",
    3: "HARNESS_ERROR",
    4: "SELECTOR_ERROR",
    5: "NO_TESTS",
}
RUN_TEST_RESULT_CLASSES = {
    "GREEN": "settled_applied",
    "RED": "settled_applied",
    "FAIL_WITH_WITNESS": "settled_applied",
    "HARNESS_ERROR": "settled_not_applied",
    "SELECTOR_ERROR": "settled_not_applied",
    "NO_TESTS": "settled_not_applied",
    # An unstructured runner result is deliberately not guessed as a completed
    # attempt.  LiveContinuation exposes its exact typed run_tests retry and
    # reconciles that effect once the retry produces structured evidence.
    "NO_VERDICT": "retryable_unknown",
    "REUSED_VERIFICATION": "settled_not_applied",
    "UNSUPPORTED_ACTION": "settled_not_applied",
}
RUN_TEST_OUTCOMES = frozenset(RUN_TEST_RESULT_CLASSES)

# Observation probes are settled read-only effects. They never satisfy the
# public-check completion contract, but every host status still settles the
# continuation effect so a timeout or policy denial cannot deadlock the frame.
PROBE_RESULT_CLASSES = {
    "COMPLETED": "settled_applied",
    "RAISED": "settled_applied",
    "TIMED_OUT": "settled_applied",
    "RESOURCE_LIMIT": "settled_applied",
    "POLICY_DENIED": "settled_applied",
    "RUNNER_ERROR": "settled_applied",
    "NOT_RUN": "settled_applied",
    "STALE": "settled_applied",
    "HOST_OBSERVED": "settled_applied",
}
PROBE_OUTCOMES = frozenset(PROBE_RESULT_CLASSES)
PROBE_REFUSAL_RESULTS = frozenset({"REJECTED", "REFUSED", "REFUSED_PREDISPATCH"})


def probe_refusal_settle_enabled() -> bool:
    value = os.environ.get("STATEFRAME_PROBE_REFUSAL_SETTLE", "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError("STATEFRAME_PROBE_REFUSAL_SETTLE must be off or on")
    return value == "on"


def result_class_disposition(producer: str, result: object) -> str:
    """Return the host disposition for every typed producer result.

    ``unknown`` is reserved for a class the producer registry forgot.  A
    ``retryable_unknown`` run-test result is still a registered, actionable
    outcome: the continuation exposes its exact retry instead of silently
    settling or inventing a verdict.
    """
    producer = str(producer or "")
    result = str(result or "")
    from kernel_agent_tools import is_registered
    if is_registered(producer):
        return {"NO_OP": "settled_applied", "APPLIED": "settled_applied",
                "REFUSED": "settled_not_applied"}.get(result, "unknown")
    if producer in READ_RESULT_CLASSES:
        if result in READ_OUTCOMES.get(producer, frozenset()):
            return "settled_applied"
        if result in READ_NOT_APPLIED_RESULTS_BY_TOOL.get(producer, READ_NOT_APPLIED_RESULTS):
            return "settled_not_applied"
        return "unknown"
    if producer == "run_tests":
        return RUN_TEST_RESULT_CLASSES.get(result, "unknown")
    if producer in {"host_obligation_check", "host_repair_check"}:
        return HOST_OBLIGATION_RESULT_DISPOSITIONS.get(result, "unknown")
    if producer == "probe":
        if result in PROBE_REFUSAL_RESULTS:
            return "settled_not_applied" if probe_refusal_settle_enabled() else "unknown"
        return PROBE_RESULT_CLASSES.get(result, "unknown")
    return "unknown"


def classify_run_test_exit(exit_code: object) -> str:
    """Classify a host subprocess exit without turning unknown text into a verdict.

    A child killed by a signal (negative code or 128+N, e.g. 137 = SIGKILL from the sandbox
    RLIMIT_CPU hard limit) or by a wall timeout (124) is a HOST outcome: it settles the effect as
    HARNESS_ERROR (settled, not applied) so the run can continue.  10-06 batch 843: an unmapped 137
    left the continuation effect "unknown" and every later gate REFUSED_PREDISPATCH.
    """
    if type(exit_code) is int:
        if exit_code < 0 or exit_code >= 128 or exit_code == 124:
            return "HARNESS_ERROR"
        return RUN_TEST_RESULT_CLASSES_BY_EXIT.get(exit_code, "NO_VERDICT")
    return "NO_VERDICT"
READ_ONLY = frozenset({
    "job_wait", "job_output", "git", "web_fetch", "skill_load",
    "read", "text", "grep", "ls", "list_files", "where_defined",
    "who_calls", "references", "recall", "recall_evidence",
    "recall_generated_artifact", "read_check_receipt", "probe",
})


# An observed empty search or missing file finishes an acquisition attempt.
# It does not prove that a source-read contract or root task was satisfied.
READ_OUTCOMES = {
    **{name: frozenset({"NO_OP"}) for name in ("job_wait", "job_output", "git", "web_fetch", "skill_load")},
    # ALREADY_PRESENT (empirical read dedup): the exact source is already in the frame at the
    # current revision -- a settled, successful read-only outcome. The registry guard below
    # keeps any future producer class from falling through to an unknown effect.
    "read": frozenset({"OK", "MISSING", "DIRECTORY", "ALREADY_PRESENT"}),
    "text": frozenset({"OK", "MISSING", "DIRECTORY", "ALREADY_PRESENT"}),
    "grep": frozenset({"HITS", "NO_HITS"}),
    "ls": frozenset({"OK"}),
    "list_files": frozenset({"HOST_OBSERVED"}),
    "where_defined": frozenset({"HOST_OBSERVED"}),
    "who_calls": frozenset({"HOST_OBSERVED"}),
    "references": frozenset({"HOST_OBSERVED"}),
    "recall": frozenset({"RECALLED"}),
    "read_check_receipt": frozenset({"HOST_OBSERVED"}),
    "recall_evidence": frozenset({"RECALLED"}),
    "recall_generated_artifact": frozenset({"RECALLED"}),
    "probe": PROBE_OUTCOMES,
}
# These are failed, non-mutating acquisition attempts.  They deliberately do
# not grant source credit, but they must settle the current effect.
READ_NOT_APPLIED_RESULTS = frozenset({"ERROR", "OUT_OF_RANGE"})
READ_NOT_APPLIED_RESULTS_BY_TOOL = {
    **{name: frozenset({"REFUSED"}) for name in ("job_wait", "job_output", "git", "web_fetch", "skill_load")},
    # These host producers have distinct refusal classes; keep each vocabulary
    # scoped to its producer instead of broadening shared read/search results.
    "list_files": frozenset({"REJECTED"}),
    "recall": frozenset({"RECALL_REJECTED"}),
    "read_check_receipt": READ_NOT_APPLIED_RESULTS | frozenset({"NOT_FOUND", "REJECTED"}),
    "recall_evidence": frozenset({"RECALL_REJECTED"}),
    "recall_generated_artifact": frozenset({"RECALL_REJECTED"}),
}
READ_RESULT_CLASSES = {
    **{name: frozenset({"NO_OP", "REFUSED"}) for name in ("job_wait", "job_output", "git", "web_fetch", "skill_load")},
    "read": frozenset(READ_OUTCOMES["read"]) | READ_NOT_APPLIED_RESULTS,
    "text": frozenset(READ_OUTCOMES["text"]) | READ_NOT_APPLIED_RESULTS,
    "grep": frozenset(READ_OUTCOMES["grep"]) | READ_NOT_APPLIED_RESULTS,
    "ls": frozenset(READ_OUTCOMES["ls"]),
    "list_files": frozenset(READ_OUTCOMES["list_files"])
                  | READ_NOT_APPLIED_RESULTS_BY_TOOL["list_files"],
    "where_defined": frozenset(READ_OUTCOMES["where_defined"]) | READ_NOT_APPLIED_RESULTS,
    "who_calls": frozenset(READ_OUTCOMES["who_calls"]) | READ_NOT_APPLIED_RESULTS,
    "references": frozenset(READ_OUTCOMES["references"]) | READ_NOT_APPLIED_RESULTS,
    "recall": frozenset(READ_OUTCOMES["recall"]) | READ_NOT_APPLIED_RESULTS_BY_TOOL["recall"],
    "read_check_receipt": READ_OUTCOMES["read_check_receipt"] | READ_NOT_APPLIED_RESULTS_BY_TOOL["read_check_receipt"],
    "recall_evidence": READ_OUTCOMES["recall_evidence"] | READ_NOT_APPLIED_RESULTS_BY_TOOL["recall_evidence"],
    "recall_generated_artifact": READ_OUTCOMES["recall_generated_artifact"] | READ_NOT_APPLIED_RESULTS_BY_TOOL["recall_generated_artifact"],
}
READ_RECEIPT_KINDS = {tool: ("read" if tool in {
    "text", "recall", "read_check_receipt", "recall_evidence", "recall_generated_artifact",
} else tool) for tool in READ_OUTCOMES}


# Non-read execution routes have typed host receipts or control contracts.
# Read routes are checked against all three registries below.
EXECUTION_RECEIPT_CONTRACTS = {
    "done": "completion_control",
    "continuation_detour": "continuation_control",
    "work_update": "work_record_control",
    "note_update": "work_record_control",
    "work": "work_record_control",
    "edit": "edit_action_receipt",
    "str_replace": "edit_action_receipt",
    "file_create": "edit_action_receipt",
    "file_delete": "edit_action_receipt",
    "file_move": "edit_action_receipt",
    "shell": "shell_effect_receipt",
    "job_stop": "job_lifecycle_receipt",
    "insert_before": "edit_action_receipt",
    "run_tests": "test_action_receipt",
    "authored_check": "authored_check_receipt",
}


def has_execution_receipt_contract(tool: object) -> bool:
    """Return whether a tool has a complete host execution-receipt contract."""
    name = str(tool or "").strip().lower()
    if not name:
        return False
    from kernel_agent_tools import is_registered
    if is_registered(name):
        return True
    if name == "probe":
        outcomes = READ_OUTCOMES.get(name)
        return (name in READ_ONLY and bool(outcomes and outcomes.issubset(PROBE_RESULT_CLASSES))
                and bool(READ_RECEIPT_KINDS.get(name)))
    if name in READ_ONLY:
        outcomes = READ_OUTCOMES.get(name)
        result_classes = READ_RESULT_CLASSES.get(name)
        return bool(outcomes and result_classes and outcomes.issubset(result_classes)
                    and READ_RECEIPT_KINDS.get(name))
    return name in EXECUTION_RECEIPT_CONTRACTS


def binding_versions(binding: object) -> dict[str, str]:
    if not isinstance(binding, dict) or not binding:
        raise ContinuationError("nonempty host binding required")
    from kernel_test_artifact_policy import filter_binding, policy_from_binding
    policy = policy_from_binding(binding)
    identity = filter_binding(binding, policy) if policy is not None else binding
    versions = {"workspace": digest(identity)}
    for group in ("product_files", "harness_files"):
        files = identity.get(group, {})
        if not isinstance(files, dict):
            raise ContinuationError("invalid host file binding")
        if len(files) > _PER_FILE_VERSION_LIMIT:
            # Real repositories (e.g. a 1,787-file backend) exceed the bounded
            # dependency map.  Collapse the group to one digest: any file change
            # still changes it, so staleness detection is preserved; only the
            # per-file granularity of the stale report is lost.
            for path, revision in files.items():
                text(path, "path")
                text(revision, "revision", 256)
            versions[f"{group}:*"] = digest(files)
            continue
        for path, revision in files.items():
            versions[f"{group}:{text(path, 'path')}"] = text(revision, "revision", 256)
    return versions


# Leaves room for "workspace" plus both groups inside the 256-entry dependency bound.
_PER_FILE_VERSION_LIMIT = 120


class HostEvidence:
    def __init__(self, run_id: str, *, source_refresh: bool = False,
                 native_prefix: bool = False, directory=None):
        self.native_prefix = native_prefix
        self.native_consumed: set[str] = set()
        self.source_refresh = source_refresh
        self.run_id = run_id
        from kernel_memfix_history import ReceiptAliases, SpoolEventMap, legacy
        self.events = {} if legacy() else SpoolEventMap(directory=directory)
        self.receipts = {} if legacy() else ReceiptAliases(self.events)
        self._scan_ids: list[str] = []
        self._by_receipt: dict[str, str] = {}
        self._kind_ids: dict[str, list[str]] = {}
        from kernel_evidence_index import EvidenceIndex
        self._joins = EvidenceIndex()

    def _index_event(self, event: dict) -> None:
        self._joins.add(event)
        event_id = event["event_id"]
        self._kind_ids.setdefault(event["kind"], []).append(event_id)
        row = event.get("receipt")
        row = row if isinstance(row, Mapping) else event
        for receipt_id in (event.get("receipt_id"), row.get("receipt_id")):
            if isinstance(receipt_id, str) and receipt_id:
                # The first matching event wins if a receipt ID was reused.
                self._by_receipt.setdefault(receipt_id, event_id)
        if (event.get("producer") == "test_subprocess"
                or event.get("kind") == "test_subprocess_completed"
                or row.get("producer") == "test_subprocess"
                or _selectors(event) or _selectors(row)):
            self._scan_ids.append(event_id)

    def _ensure_indexes(self) -> None:
        # Older ordinary pickles predate these derived indexes. The checkpoint
        # reducer rebuilds them eagerly; this covers direct pickle restores.
        if not all(hasattr(self, name) for name in ("_kind_ids", "_by_receipt", "_scan_ids", "_joins")):
            self.rebuild_indexes()

    def rebuild_indexes(self) -> None:
        """Rebuild derived indexes in one ordered pass after checkpoint restore."""
        self._scan_ids = []
        self._by_receipt = {}
        self._kind_ids = {}
        from kernel_evidence_index import EvidenceIndex
        self._joins = EvidenceIndex()
        for event in self.events.values():
            self._index_event(event)

    def scan_rows(self) -> list[dict]:
        """Events relevant to the test verdict, in original insertion order."""
        self._ensure_indexes()
        return [self.events[event_id] for event_id in self._scan_ids]

    def event_for_receipt(self, ref: str) -> dict | None:
        """Resolve an event ID first, then the first matching receipt ID."""
        self._ensure_indexes()
        event = self.events.get(ref)
        if event is not None:
            return event
        event_id = self._by_receipt.get(ref)
        return self.events.get(event_id) if event_id is not None else None

    def events_of_kinds(self, kinds: frozenset[str]) -> list[dict]:
        """Hydrate only matching kinds while preserving global event order."""
        self._ensure_indexes()
        if not kinds:
            return []
        from kernel_memfix_history import SpoolEventMap
        if isinstance(self.events, SpoolEventMap):
            positions = self.events._positions
            ids = merge(*(self._kind_ids.get(kind, ()) for kind in kinds),
                        key=positions.__getitem__)
            return [self.events[event_id] for event_id in ids]
        return [event for event in self.events.values() if event["kind"] in kinds]

    def native_validation_events(self, origin: str) -> list[dict]:
        from kernel_memfix_history import legacy
        from kernel_native_prefix import VALIDATION_KINDS
        if legacy():
            return self.events_of_kinds(VALIDATION_KINDS)
        self._ensure_indexes()
        return self._joins.native_events(self.events, origin)

    def final_delivery_events(self, submitted: dict) -> list[dict]:
        from kernel_memfix_history import legacy
        if legacy():
            return self.events_of_kinds(frozenset({'request_final_wire_prepared'}))
        self._ensure_indexes()
        return self._joins.final_events(self.events, submitted.get('request_id'),
                                       submitted.get('transport_attempt'))

    def acquired_source_events(self, path: str, revision: str, before: int) -> list[dict]:
        from kernel_memfix_history import legacy
        if legacy():
            return self.events_of_kinds(frozenset({'request_source_closure'}))
        self._ensure_indexes()
        return self._joins.acquired_events(self.events, path, revision, before)

    def __setstate__(self, state: dict) -> None:
        """Resume either checkpoint format under the selected retention mode."""
        self.__dict__.update(state)
        # Older W1 pickles carried full verdict candidates; the ID index is
        # rebuilt lazily for direct loads or by the checkpoint restore hook.
        self.__dict__.pop("_scan_rows", None)
        from kernel_memfix_history import ReceiptAliases, SpoolEventMap, legacy
        if legacy() and isinstance(self.events, SpoolEventMap):
            events = dict(self.events.items())
            receipts = {key: events[event["event_id"]]
                        for key, event in self.receipts.items()}
            self.events, self.receipts = events, receipts
        elif not legacy() and type(self.events) is dict:
            events = SpoolEventMap()
            for key, event in self.events.items():
                events.append(key, event)
            receipts = ReceiptAliases(events)
            for key, event in self.receipts.items():
                event_id = event.get("event_id")
                if self.events.get(event_id) != event:
                    raise ContinuationError("restored receipt has no matching evidence event")
                receipts.add(key, event_id)
            self.events, self.receipts = events, receipts

    def observe(self, event: dict) -> None:
        if (self.native_prefix and event["kind"] == "continuation_transition"
                and event.get("operation") == "begin_effect"
                and "native_selected_ref" in event.get("data", {})):
            self.native_consumed.add(event["data"]["native_selected_ref"])
        from kernel_native_prefix import KINDS
        if event["kind"] not in EVIDENCE_KINDS and not (self.native_prefix and event["kind"] in KINDS):
            return
        self._ensure_indexes()
        from kernel_memfix_history import SpoolEventMap
        if (isinstance(self.events, SpoolEventMap)
                and not (self.native_prefix or self.source_refresh)
                and event["kind"] == "request_final_wire_prepared"):
            return
        if not (self.source_refresh or self.native_prefix) and event["kind"] in {
                "request_prepared", "request_submitted", "request_body", "response_received"}:
            return
        if event["run_id"] != self.run_id:
            raise ContinuationError("foreign evidence run")
        receipt_key = None
        if event["kind"] == "action_receipt":
            receipt = event.get("receipt")
            if not isinstance(receipt, dict) or not isinstance(receipt.get("receipt_id"), str):
                raise ContinuationError("invalid host action receipt")
            receipt_key = receipt["receipt_id"]
            if receipt_key in self.receipts:
                raise ContinuationError("duplicate host receipt identity")
        if isinstance(self.events, SpoolEventMap):
            self.events.append(event["event_id"], event)
            if receipt_key is not None:
                self.receipts.add(receipt_key, event["event_id"])
        else:
            self.events[event["event_id"]] = event
            if receipt_key is not None:
                self.receipts[receipt_key] = event
        self._index_event(event)

    def event(self, reference: object, kind: str) -> dict:
        key = text(reference, "host evidence reference", 256)
        event = self.events.get(key)
        if event is None or event["kind"] != kind:
            raise ContinuationError(f"missing host {kind} evidence")
        return event

    def _closed(self, reference: object) -> tuple[dict, dict]:
        event = self.event(reference, "action_outcome_closed")
        record = event.get("record")
        if not isinstance(record, dict) or record.get("step_record_id") != digest({
                k: v for k, v in record.items() if k != "step_record_id"}):
            raise ContinuationError("invalid closed-step digest")
        return event, record

    def _read(self, reference: object, current: dict) -> dict:
        event, record = self._closed(reference)
        evidence = record.get("read_evidence")
        if (record.get("tool") != "read" or record.get("result") != "OK"
                or record.get("after_binding") != current
                or not isinstance(evidence, dict)
                or evidence.get("authority") != "HOST_OBSERVED_READ"
                or evidence.get("producer") != "workspace_read"
                or not record.get("receipt_ids")):
            raise ContinuationError("closed step does not prove a current source read")
        path = evidence.get("path")
        revisions = {**current.get("product_files", {}), **current.get("harness_files", {})}
        if not path or not revisions.get(path) or evidence.get("file_revision") != revisions[path]:
            raise ContinuationError("source read revision is stale or unknown")
        content = evidence.get("content")
        start_line = evidence.get("start_line")
        end_line = evidence.get("end_line")
        if (os.environ.get("STATEFRAME_READ_NEWLINE_FIX", "off").strip().lower() == "on"
                and isinstance(content, str) and type(start_line) is int
                and type(end_line) is int):
            from kernel_read_newline_fix import complete_line_count_matches
            line_count_valid = complete_line_count_matches(
                content, start_line, end_line)
        else:
            line_count_valid = (
                isinstance(content, str)
                and type(start_line) is int and type(end_line) is int
                and len(content.split("\n"))
                == end_line - start_line + 1)
        if (type(evidence.get("start_line")) is not int or type(evidence.get("end_line")) is not int
                or evidence["start_line"] < 1 or evidence["end_line"] < evidence["start_line"]
                or type(evidence.get("total_lines")) is not int
                or evidence["total_lines"] < evidence["end_line"]
                or not isinstance(evidence.get("content"), str)
                or evidence.get("content_hash") != digest(evidence["content"])
                or not line_count_valid
                or evidence.get("omitted_chars", 0) != 0):
            raise ContinuationError("source read range is empty, invalid or truncated")
        matching = [self.receipts.get(ref) for ref in record["receipt_ids"]]
        if not any(
                r is not None and r["sequence"] < event["sequence"]
                and r["receipt"].get("run_id") == self.run_id
                and r["receipt"].get("kind") == "read"
                and r["receipt"].get("result_class") == "OK"
                and r["receipt"].get("turn") == record.get("step") for r in matching):
            raise ContinuationError("missing matching host read receipt")
        return evidence

    def _fence(self, data: dict) -> dict:
        fence = self.event(data.get("dispatch_ref"), "step_dispatch_binding")
        current = data.get("current_binding")
        if (fence.get("accepted") is not True or fence.get("current") != current
                or fence.get("expected") != current or not fence.get("snapshot_id")
                or binding_versions(current) != data.get("current_dependencies")
                or fence.get("producer") != "stateframe_runtime"
                or fence.get("continuation_id") != data.get("intention_id")
                or fence.get("continuation_effect_id") != data.get("effect_id")
                or any(fence.get(key) != data.get(key) for key in
                       ("request_id", "step", "tool", "arguments_hash", "snapshot_id"))):
            raise ContinuationError("missing current pre-dispatch fence")
        return fence

    def validate(self, state: ContinuationState, operation: str, data: dict) -> None:
        if operation in {"create", "enqueue", "detour"}:
            new = data if operation == "create" else data.get("child")
            if not isinstance(new, dict):
                raise ContinuationError("scoped host intention required")
            deps = new.get("dependencies")
            workspace = deps.get("workspace") if isinstance(deps, dict) else None
            if not isinstance(workspace, str) or not workspace.startswith("sha256:") or len(workspace) != 71:
                raise ContinuationError("new intentions require a workspace binding")
        elif operation == "begin_effect":
            fence = self._fence(data)
            if "native_selected_ref" in data:
                from kernel_native_prefix import validate
                validate(self, state, data)
            tool = text(data.get("tool"), "tool", 128)
            arguments_hash = text(data.get("arguments_hash"), "arguments_hash", 256)
            text(data.get("request_id"), "request_id", 256)
            if (data.get("read_only") is not (tool in READ_ONLY)
                    or data.get("action_hash") != digest({"tool": tool, "arguments_hash": arguments_hash})
                    or type(data.get("step")) is not int or data["step"] < 0
                    or type(data.get("intent_sequence")) is not int
                    or fence["sequence"] >= data["intent_sequence"]):
                raise ContinuationError("effect intent lacks host action identity")
        elif operation == "settle_effect":
            status = self.effect_outcome(state, data)
            _, record = self._closed(data.get("receipt_ref"))
            if data.get("binding_changed") is not (record.get("before_binding") != record.get("after_binding")):
                raise ContinuationError("effect change classification must come from host bindings")
            if data.get("status") != status:
                raise ContinuationError(f"host evidence proves {status}, not the proposed outcome")
        elif operation == "repair_effect":
            effect = state.effects.get(data.get("effect_id"))
            event = self.event(data.get("receipt_ref"), "continuation_control")
            if (effect is None or effect.status not in {"pending", "unknown"}
                    or effect.intention_id != data.get("intention_id")
                    or not (effect.read_only or effect.tool == "run_tests")
                    or event.get("producer") != "stateframe_runtime"
                    or event.get("command") != "repair_effect"
                    or event.get("effect_id") != effect.effect_id
                    or event.get("intention_id") != effect.intention_id
                    or event.get("tool") != effect.tool
                    or event.get("current_binding") != data.get("current_binding")):
                raise ContinuationError("repair lacks scoped host authorization")
        elif operation in {"revalidate", "complete"}:
            item = state.intentions.get(data.get("intention_id"))
            current = data.get("current_binding")
            if item is None or binding_versions(current) != data.get("current_dependencies"):
                raise ContinuationError("current dependency binding required")
            if operation == "revalidate":
                if "source_prepared_ref" in data:
                    if not self.source_refresh:
                        raise ContinuationError("source delivery refresh is disabled")
                    from kernel_continuation_source import validate_delivery
                    validate_delivery(self, data)
                    return
                receipt_ref = data.get("receipt_ref")
                test_event = (self.events.get(receipt_ref)
                              if isinstance(receipt_ref, str) else None)
                if (isinstance(test_event, Mapping)
                        and test_event.get("kind") == "test_subprocess_completed"):
                    required = item.acceptance.get("selectors")
                    authority = evaluate_required_tests(current, required, self.scan_rows())
                    typed_receipt = next(
                        (receipt for receipt in authority.trusted_receipts
                         if receipt.receipt_id == receipt_ref), None)
                    if (authority.state != SATISFIED
                            or receipt_ref not in authority.supporting_evidence
                            or authority.current_binding != current
                            or typed_receipt is None
                            or typed_receipt.binding != current
                            or typed_receipt.after_binding != current):
                        raise ContinuationError(
                            "verification refresh requires a trusted current passing receipt")
                    return
                _, record = self._closed(receipt_ref)
                if record.get("tool") == "run_tests":
                    if (record.get("before_binding") != current or record.get("after_binding") != current
                            or not any(effect.tool == "run_tests" and effect.status == "applied"
                                       and effect.receipt_ref == receipt_ref
                                       for effect in state.effects.values())):
                        raise ContinuationError("verification refresh requires a stable, closed host execution")
                else:
                    self._read(data.get("receipt_ref"), current)
                return
            contract = item.acceptance
            if contract.get("kind") == "source_read":
                evidence = self._read(data.get("receipt_ref"), current)
                if (evidence.get("path") != contract.get("path")
                        or type(contract.get("start_line")) is not int
                        or type(contract.get("end_line")) is not int
                        or contract["start_line"] < 1 or contract["end_line"] < contract["start_line"]
                        or evidence["start_line"] > contract["start_line"]
                        or evidence["end_line"] < contract["end_line"]):
                    raise ContinuationError("read did not satisfy the declared source range")
                # Completion must be tied to an effect issued by this continuation,
                # not a convenient source read belonging to a sibling.
                if not any(e.intention_id == item.intention_id and e.status == "applied"
                           and e.receipt_ref == data.get("receipt_ref") for e in state.effects.values()):
                    raise ContinuationError("source receipt belongs to no closed effect in this continuation")
            elif contract.get("kind") == "public_checks":
                self._public_completion(state, data, contract, current)
            else:
                raise ContinuationError("no implemented host completion contract")
        elif operation == "wake":
            event = self.event(data.get("receipt_ref"), "continuation_signal")
            if (event.get("producer") != "host_event_router"
                    or event.get("intention_id") != data.get("intention_id")
                    or event.get("wake_token") != data.get("wake_token")
                    or data.get("event_id") != event["event_id"]
                    or data.get("event_hash") != digest(event)):
                raise ContinuationError("signal lacks correlated host provenance")
        elif operation == "cancel":
            event = self.event(data.get("receipt_ref"), "continuation_control")
            if (event.get("producer") != "host_event_router" or event.get("command") != "cancel"
                    or event.get("intention_id") != data.get("intention_id")
                    or event.get("reason") != data.get("reason")):
                raise ContinuationError("cancellation lacks scoped host authorization")

    def effect_outcome(self, state: ContinuationState, data: dict) -> str:
        """Classify execution from host lineage; no model-proposed success is used."""
        effect = state.effects.get(data.get("effect_id"))
        if effect is None:
            raise ContinuationError("unknown effect")
        event, record = self._closed(data.get("receipt_ref"))
        if (event["sequence"] <= effect.intent_sequence
                or record.get("tool") != effect.tool or record.get("arguments_hash") != effect.arguments_hash
                or record.get("step") != effect.step
                or record.get("continuation_effect_id") != effect.effect_id
                or record.get("continuation_dispatch_ref") != effect.dispatch_ref
                or record.get("continuation_id") != effect.intention_id):
            raise ContinuationError("closure does not match the durable effect intent")
        receipts = [self.receipts.get(ref) for ref in record.get("receipt_ids", [])]
        matching = [r["receipt"] for r in receipts if r is not None
                    and effect.intent_sequence < r["sequence"] < event["sequence"]
                    and r["receipt"].get("run_id") == self.run_id
                    and r["receipt"].get("request_id") == effect.request_id
                    and r["receipt"].get("turn") == effect.step
                    and r["receipt"].get("continuation_effect_id") == effect.effect_id
                    and r["receipt"].get("continuation_dispatch_ref") == effect.dispatch_ref
                    and r["receipt"].get("continuation_id") == effect.intention_id]
        status = "unknown"
        current = data.get("current_binding")
        if (record.get("after_binding") == current
                and binding_versions(record.get("before_binding")) == effect.dependencies):
            result = record.get("result")
            if any(r.get("result_class") == result for r in matching):
                disposition = result_class_disposition(effect.tool, result)
                if result in {"REJECTED", "REFUSED", "REFUSED_PREDISPATCH", "RECALL_REJECTED",
                                "NO_OP", "FAILED", "NO_TESTS", "SELECTOR_ERROR",
                                "HARNESS_ERROR"}:
                    refusal_allowed = not (
                        effect.tool == "probe" and result in PROBE_REFUSAL_RESULTS
                        and not probe_refusal_settle_enabled())
                    read_receipt_contract = not (
                        effect.tool in READ_NOT_APPLIED_RESULTS_BY_TOOL
                        and result in READ_NOT_APPLIED_RESULTS_BY_TOOL[effect.tool]
                    ) or any(
                        r.get("kind") == READ_RECEIPT_KINDS[effect.tool]
                        and r.get("result_class") == result for r in matching)
                    if (refusal_allowed and read_receipt_contract
                            and record.get("before_binding") == current
                            and not record.get("changes")):
                        status = "not_applied"
                elif (effect.tool in {"read", "text", "grep"}
                      # ERROR / OUT_OF_RANGE are typed host outcomes of a failed read/search attempt:
                      # no acquired-source credit, but they must settle or the next frame strands.
                      # (Restored 09-24: a later lane's rewrite of this block dropped the gate-r35 fix;
                      # guarded by probes/test_result_class_registry.py.)
                      and result in {"ERROR", "OUT_OF_RANGE"}
                      and record.get("before_binding") == current and not record.get("changes")
                      and any(r.get("kind") == READ_RECEIPT_KINDS[effect.tool]
                              and r.get("result_class") == result for r in matching)):
                    status = "not_applied"
                elif (result in {"REUSED_VERIFICATION", "UNSUPPORTED_ACTION"} and effect.tool == "run_tests"
                      and record.get("before_binding") == current and not record.get("changes")
                      and any(r.get("kind") == "test" and r.get("result_class") == result for r in matching)):
                    status = "not_applied"
                elif result == "APPLIED" and record.get("changes"):
                    status = "applied"
                elif (effect.read_only and record.get("before_binding") == current
                      and not record.get("changes")
                      and result in READ_OUTCOMES.get(effect.tool, frozenset())
                      and any(r.get("result_class") == result
                              and r.get("kind") == READ_RECEIPT_KINDS[effect.tool] for r in matching)):
                    status = "applied"
                elif (disposition == "settled_not_applied"
                      and effect.tool in READ_RESULT_CLASSES
                      and record.get("before_binding") == current
                      and not record.get("changes")
                      and any(r.get("result_class") == result
                              and r.get("kind") == READ_RECEIPT_KINDS[effect.tool] for r in matching)):
                    # Keep this fallback tied to the producer registry.  The
                    # explicit ERROR/OUT_OF_RANGE clause above remains the
                    # auditable read/search path and must not be removed.
                    status = "not_applied"
                elif effect.tool == "run_tests" and result in {"GREEN", "RED", "FAIL_WITH_WITNESS"}:
                    tests = [test for test in self.scan_rows()
                             if test["kind"] == "test_subprocess_completed"
                             and test.get("producer") == "test_subprocess"
                             and test.get("continuation_effect_id") == effect.effect_id
                             and test.get("continuation_dispatch_ref") == effect.dispatch_ref
                             and test.get("continuation_id") == effect.intention_id
                             and test.get("request_id") == effect.request_id
                             and effect.intent_sequence < test["sequence"] < event["sequence"]]
                    receipt_confirms_result = any(
                        receipt.get("kind") == "test"
                        and receipt.get("result_class") == result
                        for receipt in matching
                    )
                    if tests and all(type(test.get("exit_code")) is int for test in tests):
                        green = all(test["exit_code"] == 0 for test in tests)
                        # JobQueue completion may add FAIL_WITH_WITNESS after the test
                        # subprocess has completed.  That is a failed audit
                        # observation, not an unclosed execution: the test
                        # effect still settled and may be revalidated.
                        if result == "FAIL_WITH_WITNESS" or green == (result == "GREEN"):
                            status = "applied"
                            try:
                                from kernel_green_evidence_owner import (
                                    enabled as _owner_enabled, record_fresh_test,
                                )
                            except ImportError:
                                _owner_enabled = None
                            if _owner_enabled is not None and _owner_enabled():
                                record_fresh_test()
                    elif (
                        result in {"GREEN", "RED"}
                        and receipt_confirms_result
                        and record.get("before_binding") == current
                        and record.get("after_binding") == current
                        and not record.get("changes")
                    ):
                        # Some deterministic host runners publish only their
                        # bound test action receipt. That receipt settles the
                        # execution effect; certification still requires the
                        # separate subprocess evidence above.
                        status = "applied"
        if status == "unknown":
            # Q23: host evidence can prove a refused/malformed edit never ran;
            # keep that effect out of the red-terminal dead end.
            from kernel_greenlock_terminal_fixes import settle_unexecuted_edit
            settled = settle_unexecuted_edit(effect, record, current)
            if settled is not None:
                self._r47_unexecuted_settled = getattr(effect, "effect_id", None)
                status = settled
        return status

    def _certification_follows_dispatch(self, receipt: Mapping, original: Mapping, event: Mapping) -> bool:
        """The certifying verdict is bound to the done dispatch it certifies (b1420).

        Same request/turn is the common case. The host may instead finish the certifying run on a
        LATER turn, stamping that turn's request id; it still certifies this dispatch when the
        receipt carries the dispatch ref (checked by the caller), its turn is not before the
        dispatch step, and it was journaled after the dispatch binding. Anything earlier, or not
        anchored to a journaled dispatch, stays refused.
        """
        if (receipt.get("request_id") == event.get("request_id")
                and receipt.get("turn") == event.get("step")):
            return True
        turn, step = receipt.get("turn"), event.get("step")
        if type(turn) is not int or type(step) is not int or turn <= step:
            return False
        dispatch = self.events.get(str(event.get("dispatch_ref") or ""))
        return (isinstance(dispatch, Mapping)
                and dispatch.get("kind") == "step_dispatch_binding"
                and dispatch.get("tool") == "done"
                and dispatch.get("request_id") == event.get("request_id")
                and dispatch.get("step") == step
                and type(dispatch.get("sequence")) is int
                and dispatch["sequence"] < original["sequence"])

    def _public_completion(self, state: ContinuationState, data: dict, contract: dict, current: dict) -> None:
        event = self.event(data.get("receipt_ref"), "continuation_public_certification")
        required = contract.get("selectors")
        verdict = event.get("verdict", {})
        if (not isinstance(required, list) or not required or any(not isinstance(s, str) or not s for s in required)
                or event.get("producer") != "stateframe_runtime"
                or event.get("intention_id") != data.get("intention_id")
                or event.get("binding") != current or event.get("required_selectors") != required
                or not isinstance(verdict, Mapping)):
            raise ContinuationError("no structured current host certification")
        authority = evaluate_required_tests(current, required, self.scan_rows())
        if authority.state != SATISFIED:
            raise ContinuationError(
                "required test verdict is " + authority.state.lower().replace("_", " "))
        marker = self.event(event.get("certification_event_ref"), "completion_certification")
        receipt = marker.get("receipt", {})
        original = self.receipts.get(receipt.get("receipt_id"))
        if (marker["sequence"] >= event["sequence"] or receipt.get("result_class") != "CERTIFIED"
                or receipt.get("kind") != "verdict" or receipt.get("run_id") != self.run_id
                or original is None or original["sequence"] >= marker["sequence"]
                or original.get("receipt") != receipt
                or receipt.get("continuation_id") != data.get("intention_id")
                or receipt.get("continuation_dispatch_ref") != event.get("dispatch_ref")
                or not self._certification_follows_dispatch(receipt, original, event)):
            raise ContinuationError("host certification marker missing")
        self._fence({**data, "effect_id": None, "dispatch_ref": event.get("dispatch_ref"),
                     "request_id": event.get("request_id"), "step": event.get("step"), "tool": "done",
                     "arguments_hash": event.get("arguments_hash"), "snapshot_id": event.get("snapshot_id")})
        winning_refs = set(authority.supporting_evidence)
        try:
            import kernel_green_evidence_owner as _owner
        except ImportError:
            _owner = None
        if _owner is not None and _owner.enabled():
            # R4': one shared, typed completion-evidence predicate (preflight uses
            # the same function via ContinuationRuntime.completion_finding).
            for reference in sorted(winning_refs):
                code = _owner.completion_evidence_failure(
                    self, state, reference, data.get("intention_id"), current,
                    certification_sequence=event["sequence"])
                if code is not None:
                    _owner.record_refusal(_owner.STALE_CERTIFICATION_REASON)
                    if _owner.is_exhausted():
                        raise ContinuationError(_owner.EXHAUSTED_REASON + " [failed_predicate=" + code + "]")
                    raise ContinuationError(_owner.typed_refusal(code))
            return
        for reference in winning_refs:
            test = self.event_for_receipt(reference)
            test = test or {}
            effect = state.effects.get(test.get("continuation_effect_id"))
            ancestor = test.get("continuation_id")
            while ancestor in state.intentions and ancestor != data.get("intention_id"):
                ancestor = state.intentions[ancestor].parent_id
            if (ancestor != data.get("intention_id") or effect is None or effect.status != "applied"
                    or effect.intention_id != test.get("continuation_id")
                    or effect.dispatch_ref != test.get("continuation_dispatch_ref")
                    or test.get("producer") != "test_subprocess" or test.get("exit_code") != 0
                    or type(test.get("exit_code")) is not int or test.get("binding_stable") is not True
                    or test.get("binding") != current or test.get("after_binding") != current
                    or test.get("sequence", event["sequence"]) >= event["sequence"]):
                reason = "required test is missing, stale, red or newer than certification"
                try:
                    from kernel_green_evidence_owner import (
                        EXHAUSTED_REASON, enabled as _owner_enabled, is_exhausted, record_refusal,
                    )
                except ImportError:
                    _owner_enabled = None
                if _owner_enabled is not None and _owner_enabled():
                    record_refusal(reason)
                    if is_exhausted():
                        raise ContinuationError(EXHAUSTED_REASON)
                raise ContinuationError(reason)
