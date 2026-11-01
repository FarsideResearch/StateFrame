"""Explicit host restart at durable, public decision boundaries.

This profile is deliberately fail-closed. A checkpoint is not permission to retry
an ambiguous effect or reset an activated holder whose replay is unsupported.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from kernel_history_spool import HistorySpool, history
from kernel_history_receipts import receipts as receipt_history

from kernel_continuation_workspace import ContinuationWorkspace
from kernel_measurement import canonical_bytes, digest
from kernel_evictfix_mode import enabled as evictfix_enabled
from kernel_stateframe_resume import (
    ResumeError,
    append_follow_up,
    capture_owner_restore,
    checkpoint_conversation,
    latest_checkpoint,
    require_snapshot,
    restore_conversation,
    restore_owner_state,
    restore_tail_follow_up,
    user_request_event,
    validate_owner_restore,
    workspace_revision_matches,
)

SCHEMA = "continuation-worker-checkpoint/4"
SLIM_LEDGER_SCHEMA = "continuation-worker-slim-ledger/2"
_SLIM_LEDGER_SCHEMAS = {SLIM_LEDGER_SCHEMA, "continuation-worker-slim-ledger/1"}
FIRST_GREEN_CHECKPOINT_KEY = "first_green_disclosure"
FIRST_GREEN_CHECKPOINT_SCHEMA = "stateframe-first-green-disclosure-checkpoint/1"
_FGD_CHECKPOINT_FIELDS = (
    "_fgd_challenge_revision", "_fgd_progress", "_fgd_read_paths",
    "_fgd_probe_fingerprints", "_fgd_identical_reruns", "_fgd_disclosed_revisions",
)
_DONE_CHALLENGE_CHECKPOINT_FIELDS = (
    "_done_challenge_revision", "_done_challenge_count",
    "_done_challenge_overrides", "_done_submission_record",
)
WORKER_FIELDS = (
    "edits", "tests", "unparsed_streak", "harness_suspect", "last_line",
    "same_streak", "run_temp", "edits_since_test", "baseline_tests",
    "created_this_run", "last_surgery_step", "tool_calls_seen", "decoy_call_names",
    "schema_violation_details", "unknown_tool_names", "first_edit_step",
    "wrong_tool_before_first_edit",
)
_HOST_PROGRESS_WORKER_FIELD = "host_progress_facts"
_HOST_PROGRESS_ADVISORY_FIELD = "zero_progress_advised_checkpoints"


def _valid_host_progress_facts(facts: object) -> bool:
    if not isinstance(facts, Mapping):
        return False
    if facts.get("schema") != "kernel-host-progress-facts/1" or facts.get("source") != "host":
        return False
    for name in ("edits", "checks", "product_edits", "edits_since_check"):
        if type(facts.get(name)) is not int or facts[name] < 0:
            return False
    for name in ("changed_files", "edit_receipt_ids", "check_receipt_ids"):
        rows = facts.get(name)
        if not isinstance(rows, (tuple, list)) or any(not isinstance(row, str) for row in rows):
            return False
    first_edit_step = facts.get("first_edit_step")
    return (first_edit_step is None
            or type(first_edit_step) is int and first_edit_step >= 0)


def _valid_advised_checkpoints(value: object) -> bool:
    return (
        isinstance(value, (tuple, list))
        and len(value) <= 3
        and all(type(step) is int and step in {50, 100, 150} for step in value)
        and len(set(value)) == len(value)
    )


def _valid_worker_state_fields(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    if "unbounded_progress_watch" in value:
        from kernel_tool_call_loop_breaker import UnboundedProgressWatch
        try:
            watch = UnboundedProgressWatch.from_snapshot(value["unbounded_progress_watch"])
        except ValueError:
            return False
        if watch.threshold != 32:
            return False
        value = {key: child for key, child in value.items() if key != "unbounded_progress_watch"}
    base = set(WORKER_FIELDS)
    extra = set(value) - base
    if not extra:
        return set(value) == base
    allowed_extra = {
        frozenset({_HOST_PROGRESS_WORKER_FIELD}),
        frozenset({_HOST_PROGRESS_WORKER_FIELD, _HOST_PROGRESS_ADVISORY_FIELD}),
    }
    if frozenset(extra) not in allowed_extra or set(value) != base | extra:
        return False
    from kernel_progress_facts import host_truth_progress_enabled
    if not host_truth_progress_enabled():
        return False
    return (
        _valid_host_progress_facts(value[_HOST_PROGRESS_WORKER_FIELD])
        and (_HOST_PROGRESS_ADVISORY_FIELD not in extra
             or _valid_advised_checkpoints(value[_HOST_PROGRESS_ADVISORY_FIELD]))
    )


OWNER_FIELDS = (
    "_receipts", "trace", "edits", "last_test_exit", "product_edits",
    "workspace_revision", "_last_relevant_match", "_self_oracle_tainted",
    "f2p_ids", "f2p_names", "f2p_inworld", "oracle_file_ids",
    "oracle_selector_counts", "_selector_error_fallback_files", "_futile_answered",
)
RUNTIME_FIELDS = (
    "observations", "test_receipts", "fault_receipts", "integrity_receipts",
    "last_completed_receipt", "artifact_hashes", "snapshot_id",
)
_EVICTFIX_NOTICE_FIELDS = (
    "_evictfix_environment_notice_pending", "_evictfix_environment_notice_announced",
)


def runtime_fields(runtime=None) -> tuple[str, ...]:
    """Return checkpoint fields without changing the legacy OFF shape."""
    fields = list(RUNTIME_FIELDS)
    if getattr(runtime, "stateframe_challenge_budget", None) is not None:
        fields.append("_challenge_budget_state")
    if getattr(runtime, "stateframe_preedit_coverage_arm", None) in {"treatment", "placebo"}:
        fields.append("_preedit_coverage_state")
    if getattr(runtime, "micrograph_wire", None) is not None:
        fields.append("micrograph_wire")
    if (evictfix_enabled()
            and all(hasattr(runtime, field) for field in _EVICTFIX_NOTICE_FIELDS)):
        fields.extend(_EVICTFIX_NOTICE_FIELDS)
    return tuple(fields)


def checkpoint_runtime_fields_match(runtime, saved_runtime) -> bool:
    """An older checkpoint can omit COV_A; its advisory then stays NOOP."""
    if not isinstance(saved_runtime, dict):
        return False
    actual, expected = set(saved_runtime), set(runtime_fields(runtime))
    notice_fields = set(_EVICTFIX_NOTICE_FIELDS)
    present_notice_fields = actual & notice_fields
    if present_notice_fields:
        if present_notice_fields != notice_fields:
            return False
        pending = saved_runtime["_evictfix_environment_notice_pending"]
        announced = saved_runtime["_evictfix_environment_notice_announced"]
        if (pending is not None and (not isinstance(pending, str)
                                    or not 0 < len(pending) <= 160)):
            return False
        if type(announced) is not bool or (announced and pending is not None):
            return False
    if actual == expected:
        return True
    optional = []
    if getattr(runtime, "stateframe_preedit_coverage_arm", None) in {"treatment", "placebo"}:
        optional.append({"_preedit_coverage_state"})
    if set(_EVICTFIX_NOTICE_FIELDS) <= expected:
        optional.append(set(_EVICTFIX_NOTICE_FIELDS))
    return any(actual == expected - missing for missing in (
        *optional,
        (set().union(*optional) if len(optional) > 1 else set()),
    ))


def restore_runtime_field_values(runtime, saved_runtime: Mapping[str, object]) -> None:
    """Restore only checkpoint-owned fields, including pending notice delivery."""
    if not checkpoint_runtime_fields_match(runtime, saved_runtime):
        raise WorkerResumeError("invalid worker checkpoint")
    for key, value in saved_runtime.items():
        if key == "micrograph_wire":
            from kernel_micrograph_wire import MicroGraphWire
            try:
                runtime.micrograph_wire = MicroGraphWire.from_checkpoint(
                    value, str(getattr(runtime.owner, "task_id", "task")))
            except (KeyError, TypeError, ValueError) as error:
                raise WorkerResumeError("invalid MicroGraph checkpoint") from error
        else:
            setattr(runtime, key, copy.deepcopy(value))


def reconcile_environment_notice_tail(
        runtime: object, tail: list[dict], checkpoint_sequence: int,
) -> bool:
    """Acknowledge a notice only from a received response to its durable wire."""
    if not getattr(runtime, "_evictfix_environment_notice_pending", None):
        return False
    from kernel_step_contract import commit_environment_notice_delivery

    prepared_by_request: dict[str, dict] = {}
    submitted: dict[str, tuple[str, dict, dict]] = {}
    bodies: dict[str, dict] = {}
    responses: dict[str, tuple[str, str]] = {}
    for event in tail:
        sequence = event.get("sequence")
        if type(sequence) is not int or sequence <= checkpoint_sequence:
            continue
        kind = event.get("kind")
        request_id = event.get("request_id")
        if not isinstance(request_id, str):
            continue
        if kind == "request_final_wire_prepared":
            prepared_by_request[request_id] = event
        elif kind == "request_submitted":
            sent_id = event.get("event_id")
            prepared = prepared_by_request.get(request_id)
            if isinstance(sent_id, str) and prepared is not None:
                submitted[sent_id] = (request_id, prepared, event)
        elif kind == "request_body":
            sent_id = event.get("submitted_event_id")
            if isinstance(sent_id, str):
                bodies[sent_id] = event
        elif kind == "response_received":
            response_id = event.get("event_id")
            sent_id = event.get("transport_attempt_id")
            if (isinstance(response_id, str) and isinstance(sent_id, str)
                    and sent_id in submitted and submitted[sent_id][0] == request_id):
                responses[response_id] = (request_id, sent_id)
        elif kind == "native_batch_received":
            response_id = event.get("origin_response_event_id")
            matched = responses.get(response_id) if isinstance(response_id, str) else None
            if matched is None or matched[0] != request_id:
                continue
            sent_id = matched[1]
            body = bodies.get(sent_id)
            _, prepared, sent = submitted[sent_id]
            wire_text = sent.get("payload_utf8")
            if not isinstance(wire_text, str):
                delta = sent.get("payload_delta")
                base = submitted.get(sent.get("payload_ref"))
                if (not isinstance(delta, Mapping)
                        or delta.get("identical_wire") is not True
                        or delta.get("base_attempt") != 1
                        or delta.get("changed_keys") != []
                        or base is None or base[0] != request_id
                        or sent.get("payload_hash") != base[2].get("payload_hash")):
                    continue
                wire_text = base[2].get("payload_utf8")
            if not isinstance(wire_text, str):
                continue
            wire = wire_text.encode("utf-8")
            sha = "sha256:" + hashlib.sha256(wire).hexdigest()
            if (sent.get("payload_hash") != sha
                    or prepared.get("payload_sha256") != sha
                    or (body is not None and (
                        body.get("request_id") != request_id
                        or body.get("payload_sha256") != sha
                        or body.get("payload_utf8") != wire_text))):
                continue
            try:
                payload = json.loads(wire)
            except (TypeError, ValueError, RecursionError):
                continue
            messages = payload.get("messages") if isinstance(payload, Mapping) else None
            if (not isinstance(messages, list)
                    or messages != prepared.get("messages")):
                continue
            return commit_environment_notice_delivery(runtime, messages)
    return False
PROJECTION_FIELDS = (
    "frame", "last_step", "seen_trace_events", "model_claims",
    "initial_harness_revision", "initial_harness_files",
)
SAFE_TAIL = frozenset({
    "continuation_writer_resumed", "rich_state_committed", "snapshot_committed",
    "history_decision", "history_exchange_removed", "selection_decision",
    "exposure_composition", "latest_exchange_decision", "request_prepared", "request_submitted",
    "request_body",
    "request_final_wire_prepared", "provider_request_not_dispatched",
    "provider_boundary_host_stop",
    "response_received", "transport_error", "parent_budget",
    "parent_attached", "continuation_worker_restored",
    "continuation_resume_authorized", "continuation_owner_restore", "user_request",
    "perpetual_soak_prepare", "post_green_menu_restored",
    # Audit-only events emitted while PREPARING a request (no state change; replay re-emits them).
    # Missing from this set made every resume refuse once prompt v2 / router / cards were on (09-25 gate).
    # State-changing ops (e.g. decision_cache_op_accepted) must NOT be listed here.
    "system_prompt_variant", "thinking_router_classified", "pattern_cards_rendered",
    "decision_cache_render", "tokens_row6_delivery_audit", "tool_sidecar_stripped",
    # STATEFRAME_PERPETUAL_SOAK=on prepare audit event (no state change).  Found by the
    # real-child soak: without it every soak resume refused (fix round, perpetual_soak).
    "perpetual_soak_prepare",
    # Restore-time audit of the post-green checkpoint section (r44 fix DONE, item 9).
    "post_green_menu_restored",
})
CHALLENGE_BUDGET_SAFE_TAIL = frozenset({
    "challenge_budget_state", "done_challenge_once", "done_challenge_overridden",
    "done_submission_record", "hstate_plus_done_challenge", "hstate_plus_done_override",
    "hstate_plus_submission_record",
})
PREPARE_TRANSITIONS = frozenset({"create", "select", "expire", "exhaust"})


def safe_tail_for(runtime=None) -> frozenset[str]:
    """Return audit events admissible after a checkpoint for this runtime.

    Router events stay out of the baseline tail.  They are admitted only when
    the runtime explicitly enabled and constructed the durable replay gate.
    """
    allowed = SAFE_TAIL
    if getattr(runtime, "stateframe_model_router_replay_mode", "off") == "on":
        from kernel_stateframe_model_router_replay import MODEL_ROUTER_SAFE_TAIL
        allowed |= MODEL_ROUTER_SAFE_TAIL
    if getattr(runtime, "stateframe_challenge_budget", None) is not None:
        allowed |= CHALLENGE_BUDGET_SAFE_TAIL
    if getattr(runtime, "stateframe_preedit_coverage_arm", None) in {"treatment", "placebo"}:
        # Derived audit only; durable state is joined from request_submitted.
        allowed |= frozenset({"preedit_coverage_fired"})
    if getattr(runtime, "micrograph_wire", None) is not None:
        allowed |= frozenset({"micrograph_fact_ingested", "micrograph_input_skipped"})
    return allowed


def restore_router_replay_tail(runtime: object, records: list[Mapping[str, object]],
                               checkpoint_sequence: int) -> None:
    """Restore one committed post-checkpoint router state before the next route."""
    if getattr(runtime, "stateframe_model_router_replay_mode", "off") != "on":
        return
    from kernel_stateframe_model_router_replay import latest_state_record, validate_tail
    tail = [record for record in records
            if type(record.get("sequence")) is int
            and record["sequence"] > checkpoint_sequence]
    validate_tail(tail)
    router = getattr(runtime, "model_router", None)
    latest_router_state = latest_state_record(runtime.journal)
    if (router is None or latest_router_state is None
            or type(latest_router_state.get("sequence")) is not int
            or latest_router_state["sequence"] <= checkpoint_sequence):
        return
    state_checkpoint = latest_router_state.get("checkpoint_sequence")
    if state_checkpoint is not None and state_checkpoint != checkpoint_sequence:
        raise WorkerResumeError("post-checkpoint router state has the wrong checkpoint identity")
    router.restore_replay_state(latest_router_state["state"])


class WorkerResumeError(RuntimeError):
    pass


def _register_checkpoint(runtime, event: dict) -> None:
    """Preserve the retention engine's checkpoint index at the journal seam."""
    continuation = getattr(runtime, "continuation", None)
    engine = getattr(continuation, "engine", None)
    register = getattr(engine, "register_checkpoint", None)
    if callable(register):
        register(event)


def _reconcile_external_effects(runtime) -> set[str]:
    """Reconcile host-owned outcomes before applying the unresolved-effect gate."""
    continuation = getattr(runtime, "continuation", None)
    reconcile = getattr(continuation, "reconcile_external_effects", None)
    if not callable(reconcile):
        return set()
    events = reconcile()
    if events is None:
        return set()
    if not isinstance(events, (set, list, tuple)) or any(
            not isinstance(event_id, str) or not event_id for event_id in events):
        raise WorkerResumeError("external reconciliation returned invalid event references")
    return set(events)


def enabled() -> bool:
    value = os.environ.get("KERNEL_CONTINUATION_RESTART", "0")
    if value not in {"0", "1"}:
        raise ValueError("KERNEL_CONTINUATION_RESTART must be 0 or 1")
    return value == "1"


def validate_profile(information) -> None:
    from kernel_continuation_runtime import enabled as executive_enabled
    from kernel_intervention_policy import interventions_enabled
    if not enabled() or not executive_enabled():
        raise WorkerResumeError("worker restart requires both explicit continuation switches")
    if not information.strict or not information.off or information.feedback != "FINAL_ONLY":
        raise WorkerResumeError("worker restart requires strict unassisted FINAL_ONLY information policy")
    try:
        from kernel_stateframe_model_router import (
            model_router_mode, validate_restart_compatibility,
        )
        validate_restart_compatibility(True, mode=model_router_mode())
    except ValueError as error:
        raise WorkerResumeError(str(error)) from error
    if interventions_enabled():
        raise WorkerResumeError("legacy organ replay is not implemented for worker restart")
    from kernel_decision_state import notes_enabled
    if notes_enabled() and os.environ.get("KERNEL_DECISION_STATE", "0") != "0":
        raise WorkerResumeError("working notes cannot restore full decision state")
    from kernel_native_tools import validate_configuration
    native = validate_configuration()
    if notes_enabled() and not native:
        raise WorkerResumeError("working-note restart requires native provenance")
    from kernel_frame_series_runtime import configuration
    series, _ = configuration()
    if series and not native:
        raise WorkerResumeError("frame-series restart requires native dispatch")
    required = {
        "REACT_TOOLCALL": "1", "REACT_STEP_CONTRACT": "0" if native else "1",
        "STATEFRAME_PROJECTOR": "rich",
        "STATEFRAME_VIEW": "operational",
        "KERNEL_DECISION_STATE": "0",
        "KERNEL_REJECTED_CANDIDATE": "0",
        "FACT_COMPILER": "0", "CROSSRUN_MEMORY": "0",
        # A capsule bank is an external immutable input, but this r43 restart
        # checkpoint has no bank binding/replay section.  Refuse it rather than
        # silently loading a different bank after worker restart.
        "STATEFRAME_MEMORY_CAPSULE": "off",
        "FAIR_BRIEF": "0", "GROUNDED_BRIEF": "0",
    }
    defaults = {"REACT_TOOLCALL": "1", "STATEFRAME_PROJECTOR": "legacy",
                "STATEFRAME_VIEW": "existing", "STATEFRAME_LATEST_EXCHANGE": "off",
                "FACT_COMPILER": "1", "CROSSRUN_MEMORY": "1",
                "STATEFRAME_MEMORY_CAPSULE": "off",
                "FAIR_BRIEF": "1", "GROUNDED_BRIEF": "1"}
    bad = [key for key, value in required.items()
           if os.environ.get(key, defaults.get(key, "0")) != value]
    exchange = os.environ.get("STATEFRAME_LATEST_EXCHANGE", "off")
    if exchange not in {"off", "quoted", "roles"} or (exchange != "off" and not native):
        bad.append("STATEFRAME_LATEST_EXCHANGE")
    if bad:
        raise WorkerResumeError("unsupported restart state holders/profile: " + ", ".join(bad))


@dataclass(frozen=True)
class WorkerResume:
    workspace: Path
    expected_journal_head: str
    follow_up: str | None = None
    request_id: str | None = None

    def open_workspace(self, parent: Path, *, task_digest: str) -> ContinuationWorkspace:
        path = Path(self.workspace).absolute()
        if (path.is_symlink() or path.parent.resolve() != parent.resolve()
                or len(path.name) != 32 or any(c not in "0123456789abcdef" for c in path.name)):
            raise WorkerResumeError("resume workspace must be an original direct session")
        if (not isinstance(self.expected_journal_head, str)
                or len(self.expected_journal_head) != 71
                or not self.expected_journal_head.startswith("sha256:")
                or any(c not in "0123456789abcdef" for c in self.expected_journal_head[7:])):
            raise WorkerResumeError("explicit trusted journal head required")
        if self.follow_up is not None and (not isinstance(self.follow_up, str) or not self.follow_up):
            raise WorkerResumeError("follow-up user message must be nonempty text")
        session = ContinuationWorkspace.open_existing(path, task_digest=task_digest)
        try:
            if (path / "grading.json").exists():
                raise WorkerResumeError("terminal grading session cannot resume execution")
            execution_paths(session)
            return session
        except BaseException:
            session.close()
            raise


def execution_paths(workspace):
    initial = workspace.read("initialized.json")
    paths = []
    for parent, key in ((workspace.directory, "root"), (None, "target")):
        parent = paths[0] if parent is None else parent
        raw = initial.get(key)
        if (not isinstance(raw, str) or not raw or Path(raw).is_absolute()
                or any(part in {".", ".."} for part in raw.split("/"))):
            raise WorkerResumeError("invalid retained execution path")
        path = parent / raw
        if path.is_symlink() or path.resolve() != path or not path.is_relative_to(parent):
            raise WorkerResumeError("retained execution path escaped or was aliased")
        paths.append(path)
    if not paths[0].is_dir() or not paths[1].is_file():
        raise WorkerResumeError("retained execution files are missing")
    return tuple(paths)


def resume_journal(runtime, output, manifest, descriptor):
    from kernel_continuation_journal import ContinuationJournal
    initial = runtime.workspace.read("initialized.json")
    root, target = execution_paths(runtime.workspace)
    if root != Path(runtime.owner.root) or target != runtime.target:
        raise WorkerResumeError("worker paths differ from original execution binding")
    journal_path = Path(initial["journal_path"])
    if (journal_path.name != "events.jsonl"
            or journal_path.parent.parent.resolve() != Path(output).resolve()
            or journal_path.parent.name != initial["run_id"]):
        raise WorkerResumeError("journal is outside the original measurement allocation")
    journal = ContinuationJournal.resume(journal_path, task_id=runtime.owner.task_id,
        manifest=manifest, expected_head=descriptor.expected_journal_head)
    try:
        bindings = [e for e in journal.records() if e["kind"] == "continuation_workspace_bound"]
        if (len(bindings) != 1 or bindings[0].get("initialization_hash") != digest(initial)
                or bindings[0].get("allocation_hash") != digest(runtime.workspace.read("allocation.json"))
                or bindings[0].get("directory") != str(runtime.workspace.directory)
                or bindings[0].get("root") != str(root)
                or initial["config_hash"] != journal.config_hash
                or initial["task_id"] != journal.task_id):
            raise WorkerResumeError("workspace metadata is not bound to this journal")
    except BaseException:
        journal.close()
        raise
    runtime.resume_follow_up = getattr(descriptor, "follow_up", None)
    runtime.resume_request_id = getattr(descriptor, "request_id", None)
    return journal


def _plain(value):
    from kernel_memfix_history import SpoolSequence
    if isinstance(value, (tuple, set, HistorySpool, SpoolSequence)):
        return [_plain(v) for v in (sorted(value) if isinstance(value, set) else value)]
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return copy.deepcopy(value)


def _checkpoint_ledger(ledger):
    """Journal full actions once; repeat only compact identities in each checkpoint."""
    from kernel_memfix_history import legacy
    if legacy():
        return _plain(ledger.records)
    return {
        "schema": SLIM_LEDGER_SCHEMA,
        "record_ids": [record["step_record_id"] for record in ledger.records],
        "slim": _plain(ledger.slim_state()),
    }


def _journal_ledger(saved, records, boundary):
    """Authenticate compact checkpoint identities against the closed journal prefix."""
    closed = [event["record"] for event in records
              if event["kind"] == "action_outcome_closed"
              and event["sequence"] <= boundary["sequence"]]
    stored = saved.get("ledger")
    if type(stored) is list:
        if canonical_bytes(stored) != canonical_bytes(closed):
            raise WorkerResumeError("checkpoint ledger differs from closed journal actions")
        return closed, saved, None
    if (not isinstance(stored, dict) or stored.get("schema") not in _SLIM_LEDGER_SCHEMAS
            or type(stored.get("slim")) is not dict
            or type(stored.get("record_ids")) is not list
            or len(stored["record_ids"]) != len(closed)
            or any(type(ref) is not str or ref != record.get("step_record_id")
                   for ref, record in zip(stored["record_ids"], closed))):
        raise WorkerResumeError("checkpoint ledger differs from closed journal actions")
    # Existing replay validators need the full records. The authenticated
    # journal already owns them, so this transient view adds only references.
    replay_saved = dict(saved)
    replay_saved["ledger"] = closed
    return closed, replay_saved, stored.get("slim")


def _restore_ledger(ledger, closed, slim, directory):
    try:
        if slim is None:  # Only the explicitly authenticated legacy full-list envelope.
            ledger.restore_records(closed, directory=directory)
        else:
            ledger.restore_slim(slim, full_records=closed, directory=directory)
    except (KeyError, TypeError, ValueError) as error:
        raise WorkerResumeError("checkpoint slim ledger differs from closed journal actions") from error
    if len(ledger.records) != len(closed):
        raise WorkerResumeError("restored slim ledger lost closed actions")


def _string_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, (set, frozenset, tuple, list)):
        return []
    return sorted({str(item) for item in value if str(item).strip()})


def _rerun_rows(value) -> list[dict[str, object]]:
    if not isinstance(value, Mapping):
        return []
    rows = []
    for key, count in value.items():
        if not isinstance(key, (tuple, list)) or len(key) != 2:
            continue
        try:
            number = int(count)
        except (TypeError, ValueError):
            continue
        rows.append({"selector": str(key[0]), "revision": str(key[1]), "count": number})
    return sorted(rows, key=lambda row: (row["selector"], row["revision"]))


def first_green_checkpoint_section(runtime) -> dict[str, object] | None:
    """Encode the opt-in first-green/done policy state without JSON key tuples."""
    if getattr(runtime, "stateframe_first_green_disclosure_mode", "off") != "on":
        return None
    state = {
        "_fgd_challenge_revision": _plain(
            getattr(runtime, "_fgd_challenge_revision", None)),
        "_fgd_progress": bool(getattr(runtime, "_fgd_progress", False)),
        "_fgd_read_paths": _string_list(getattr(runtime, "_fgd_read_paths", None)),
        "_fgd_probe_fingerprints": _string_list(
            getattr(runtime, "_fgd_probe_fingerprints", None)),
        "_fgd_identical_reruns": _rerun_rows(
            getattr(runtime, "_fgd_identical_reruns", None)),
        "_fgd_disclosed_revisions": _string_list(
            getattr(runtime, "_fgd_disclosed_revisions", None)),
        "_done_challenge_revision": _plain(
            getattr(runtime, "_done_challenge_revision", None)),
        "_done_challenge_count": int(
            getattr(runtime, "_done_challenge_count", 0) or 0),
        "_done_challenge_overrides": int(
            getattr(runtime, "_done_challenge_overrides", 0) or 0),
        "_done_submission_record": _plain(
            getattr(runtime, "_done_submission_record", None)),
    }
    return {"schema": FIRST_GREEN_CHECKPOINT_SCHEMA, "state": state}


def restore_first_green_checkpoint(runtime, section) -> None:
    """Restore the opt-in policy section into a fresh worker runtime."""
    if section is None:
        return
    if getattr(runtime, "stateframe_first_green_disclosure_mode", "off") != "on":
        raise WorkerResumeError("first-green checkpoint in a disabled profile")
    if (not isinstance(section, Mapping)
            or section.get("schema") != FIRST_GREEN_CHECKPOINT_SCHEMA
            or not isinstance(section.get("state"), Mapping)):
        raise WorkerResumeError("invalid first-green policy checkpoint")
    state = section["state"]
    expected = set(_FGD_CHECKPOINT_FIELDS) | set(_DONE_CHALLENGE_CHECKPOINT_FIELDS)
    if set(state) != expected:
        raise WorkerResumeError("incomplete first-green policy checkpoint")
    rows = state["_fgd_identical_reruns"]
    if not isinstance(rows, list):
        raise WorkerResumeError("invalid first-green rerun checkpoint")
    reruns = {}
    for row in rows:
        if (not isinstance(row, Mapping)
                or set(row) != {"selector", "revision", "count"}
                or not isinstance(row["selector"], str)
                or not isinstance(row["revision"], str)
                or type(row["count"]) is not int
                or row["count"] < 0):
            raise WorkerResumeError("invalid first-green rerun row")
        reruns[(row["selector"], row["revision"])] = row["count"]
    for name in ("_fgd_read_paths", "_fgd_probe_fingerprints", "_fgd_disclosed_revisions"):
        if not isinstance(state[name], list) or any(not isinstance(item, str) for item in state[name]):
            raise WorkerResumeError("invalid first-green set checkpoint")
    if type(state["_fgd_progress"]) is not bool:
        raise WorkerResumeError("invalid first-green progress checkpoint")
    for name in ("_done_challenge_count", "_done_challenge_overrides"):
        if type(state[name]) is not int or state[name] < 0:
            raise WorkerResumeError("invalid done challenge checkpoint")
    submission = state["_done_submission_record"]
    if submission is not None and not isinstance(submission, Mapping):
        raise WorkerResumeError("invalid done submission checkpoint")

    setattr(runtime, "_fgd_challenge_revision",
            copy.deepcopy(state["_fgd_challenge_revision"]))
    setattr(runtime, "_fgd_progress", state["_fgd_progress"])
    setattr(runtime, "_fgd_read_paths", set(state["_fgd_read_paths"]))
    setattr(runtime, "_fgd_probe_fingerprints", set(state["_fgd_probe_fingerprints"]))
    setattr(runtime, "_fgd_identical_reruns", reruns)
    setattr(runtime, "_fgd_disclosed_revisions", set(state["_fgd_disclosed_revisions"]))
    setattr(runtime, "_done_challenge_revision",
            copy.deepcopy(state["_done_challenge_revision"]))
    setattr(runtime, "_done_challenge_count", state["_done_challenge_count"])
    setattr(runtime, "_done_challenge_overrides", state["_done_challenge_overrides"])
    setattr(runtime, "_done_submission_record", copy.deepcopy(submission))


# r44 fix DONE (consult item 9): STATEFRAME_POST_GREEN_WORK_MENU state (edit cycles, pending
# verification, reopen sets) survives a worker restart; without it a restart replenished the
# post-green edit budget and dropped the pending re-verification.
POST_GREEN_CHECKPOINT_KEY = "post_green_work_menu"
POST_GREEN_CHECKPOINT_SCHEMA = "post-green-work-menu-checkpoint/1"
RESUME_WORK_CHECKPOINT_KEY = "resume_work_state"
RESUME_WORK_CHECKPOINT_SCHEMA = "stateframe-resume-work-state/2"
LEGACY_RESUME_WORK_CHECKPOINT_SCHEMA = "stateframe-resume-work-state/1"
_RESUME_WORK_HOST_FIELDS = (
    "assessments", "payload", "snapshot", "check_file", "repair_episode",
    "repair_packet_audit", "check_bundle_source", "retired_check_ids",
)
_RESUME_WORK_AUTO_VERIFY_FIELDS = (
    "pending_edit", "deferred_edit", "delivery", "pending_edit_id",
)


def post_green_checkpoint_section(runtime) -> dict[str, object] | None:
    if getattr(runtime, "stateframe_post_green_work_menu_mode", "off") != "on":
        return None
    from kernel_post_green_menu import state_snapshot
    return {"schema": POST_GREEN_CHECKPOINT_SCHEMA, "state": state_snapshot(runtime)}


def restore_post_green_checkpoint(runtime, section) -> None:
    if getattr(runtime, "stateframe_post_green_work_menu_mode", "off") != "on":
        raise WorkerResumeError("post-green checkpoint in a disabled profile")
    if (not isinstance(section, Mapping)
            or section.get("schema") != POST_GREEN_CHECKPOINT_SCHEMA
            or not isinstance(section.get("state"), Mapping)
            or not isinstance(section["state"].get("cycles", {}), Mapping)):
        raise WorkerResumeError("invalid post-green policy checkpoint")
    runtime._post_green_menu_state = copy.deepcopy(dict(section["state"]))
    runtime.journal.emit("post_green_menu_restored", producer="kernel_continuation_restart",
                         dispatch=False, state=copy.deepcopy(dict(section["state"])))


def _resume_work_assessment(value: object):
    if not isinstance(value, Mapping):
        raise WorkerResumeError("invalid resume-work obligation assessment")
    from kernel_obligation_assess import CheckReceipt, ObligationAssessment
    raw_receipts = value.get("receipts", ())
    if not isinstance(raw_receipts, (list, tuple)):
        raise WorkerResumeError("invalid resume-work obligation receipts")
    try:
        receipts = tuple(
            CheckReceipt.from_mapping(item) for item in raw_receipts
            if isinstance(item, Mapping))
        return ObligationAssessment(
            obligation_id=str(value.get("obligation_id", "")),
            verdict=str(value.get("verdict", "")),
            reason=str(value.get("reason", "")),
            check_ids=tuple(value.get("check_ids", ())), receipts=receipts,
            observed=copy.deepcopy(value.get("observed")),
            expected=copy.deepcopy(value.get("expected")),
            evidence_refs=tuple(value.get("evidence_refs", ())),
            clause=str(value.get("clause", "")), location_hint=value.get("location_hint"),
            coupled_obligations=tuple(value.get("coupled_obligations", ())),
            localization=copy.deepcopy(value.get("localization")),
            coverage_required=bool(value.get("coverage_required", False)),
            coverage_disposition=str(value.get("coverage_disposition", "REQUIRED")),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise WorkerResumeError("invalid resume-work obligation assessment") from error


def resume_work_checkpoint_section(runtime) -> dict[str, object] | None:
    """Persist host adjudication, witness and deferred verification state."""
    host_state = {
        "assessments": [
            _plain(item.to_dict() if hasattr(item, "to_dict") else item)
            for item in getattr(runtime, "host_closure_assessments", ())
        ],
        "payload": _plain(getattr(runtime, "host_closure_payload", None)),
        "snapshot": _plain(getattr(runtime, "host_closure_snapshot", None)),
        "check_file": _plain(getattr(runtime, "host_closure_check_file", None)),
        "repair_episode": _plain(getattr(runtime, "host_repair_episode", None)),
        "repair_packet_audit": _plain(getattr(runtime, "host_repair_packet_audit", None)),
        "check_bundle_source": _plain(getattr(runtime, "host_check_bundle_source", None)),
        "retired_check_ids": sorted(
            str(item) for item in getattr(runtime, "host_retired_check_ids", ())
        ),
    }
    auto_verify = {
        "pending_edit": _plain(getattr(runtime, "_auto_verify_pending_edit", None)),
        "deferred_edit": _plain(getattr(runtime, "_auto_verify_deferred_edit", None)),
        "delivery": _plain(getattr(runtime, "_auto_verify_delivery", None)),
        "pending_edit_id": _plain(getattr(runtime, "_auto_verify_pending_edit_id", None)),
    }
    behavior_records = _plain(getattr(runtime, "behavior_witness_records", []))
    from kernel_completion_owner import snapshot as completion_owner_snapshot
    completion_owner = completion_owner_snapshot(runtime)
    host_values = tuple(host_state[key] for key in _RESUME_WORK_HOST_FIELDS)
    if not (any(value not in (None, [], {}, ()) for value in auto_verify.values())
            or any(value not in (None, [], {}, ()) for value in host_values)
            or behavior_records or completion_owner is not None):
        return None
    return {
        "schema": RESUME_WORK_CHECKPOINT_SCHEMA,
        "host_closure": host_state,
        "behavior_witness_records": behavior_records,
        "auto_verify": auto_verify,
        "completion_owner": completion_owner,
    }


def restore_resume_work_checkpoint(runtime, section) -> None:
    if section is None:
        return
    if not isinstance(section, Mapping):
        raise WorkerResumeError("invalid resume-work checkpoint")
    schema = section.get("schema")
    if schema not in {LEGACY_RESUME_WORK_CHECKPOINT_SCHEMA, RESUME_WORK_CHECKPOINT_SCHEMA}:
        raise WorkerResumeError("invalid resume-work checkpoint")
    expected_keys = {"schema", "host_closure", "behavior_witness_records", "auto_verify"}
    if schema == RESUME_WORK_CHECKPOINT_SCHEMA:
        expected_keys.add("completion_owner")
    if set(section) != expected_keys:
        raise WorkerResumeError("invalid resume-work checkpoint")
    host = section["host_closure"]
    auto_verify = section["auto_verify"]
    records = section["behavior_witness_records"]
    if (not isinstance(host, Mapping) or set(host) != set(_RESUME_WORK_HOST_FIELDS)
            or not isinstance(auto_verify, Mapping)
            or set(auto_verify) != set(_RESUME_WORK_AUTO_VERIFY_FIELDS)
            or not isinstance(records, list)
            or not isinstance(host["assessments"], list)
            or not isinstance(host["retired_check_ids"], list)):
        raise WorkerResumeError("incomplete resume-work checkpoint")
    assessments = tuple(_resume_work_assessment(item) for item in host["assessments"])
    runtime.host_closure_assessments = assessments
    runtime.host_closure_payload = copy.deepcopy(host["payload"])
    runtime.host_closure_snapshot = copy.deepcopy(host["snapshot"])
    runtime.host_closure_check_file = copy.deepcopy(host["check_file"])
    runtime.host_repair_episode = copy.deepcopy(host["repair_episode"])
    runtime.host_repair_packet_audit = copy.deepcopy(host["repair_packet_audit"])
    runtime.host_check_bundle_source = copy.deepcopy(host["check_bundle_source"])
    runtime.host_retired_check_ids = {str(item) for item in host["retired_check_ids"]}
    runtime.behavior_witness_records = copy.deepcopy(records)
    runtime._auto_verify_pending_edit = copy.deepcopy(auto_verify["pending_edit"])
    deferred_edit = copy.deepcopy(auto_verify["deferred_edit"])
    if isinstance(deferred_edit, Mapping):
        deferred_edit["selectors"] = tuple(deferred_edit.get("selectors", ()))
    runtime._auto_verify_deferred_edit = deferred_edit
    runtime._auto_verify_delivery = copy.deepcopy(auto_verify["delivery"])
    runtime._auto_verify_pending_edit_id = copy.deepcopy(auto_verify["pending_edit_id"])
    if schema == RESUME_WORK_CHECKPOINT_SCHEMA:
        try:
            from kernel_completion_owner import restore as restore_completion_owner
            restore_completion_owner(runtime, section["completion_owner"])
        except (TypeError, ValueError) as error:
            raise WorkerResumeError("invalid completion-owner checkpoint") from error
    runtime.journal.emit(
        "resume_work_state_restored", producer="kernel_continuation_restart",
        dispatch=False, assessment_count=len(assessments),
        behavior_witness_count=len(records),
        deferred_revalidation=auto_verify["deferred_edit"] is not None,
        completion_owner_restored=schema == RESUME_WORK_CHECKPOINT_SCHEMA,
    )


def checkpoint(runtime, messages, worker_state):
    if not enabled():
        return
    if not _valid_worker_state_fields(worker_state):
        raise WorkerResumeError("worker checkpoint fields are incomplete")
    if runtime.step_ledger.pending is not None:
        raise WorkerResumeError("cannot checkpoint an open action")
    projection = runtime.live_projection
    if projection is not None:
        projection.update(list(messages), runtime.owner.trace, runtime.owner.current_step,
                          runtime.binding(), runtime.test_receipts)
    from kernel_native_restart import snapshot
    from kernel_frame_series_replay import snapshot_series
    from kernel_working_notes_restart import snapshot as note_snapshot
    state = {
        "frame_series": snapshot_series(runtime.frame_series),
        "working_notes": note_snapshot(runtime.decision_state),
        "native": snapshot(getattr(runtime, "native_batch", None)),
        "generated": runtime.generated.checkpoint_snapshot() if runtime.generated is not None else None,
        "retained": runtime.retained.checkpoint_snapshot() if runtime.retained is not None else None,
        "schema": SCHEMA, "step": runtime.owner.current_step,
        "binding": runtime.binding(), "worker": _plain(worker_state),
        "owner": {key: _plain(getattr(runtime.owner, key))
                  for key in OWNER_FIELDS if hasattr(runtime.owner, key)},
        "runtime": {key: (runtime.micrograph_wire.checkpoint() if key == "micrograph_wire"
                          else _plain(getattr(runtime, key))) for key in runtime_fields(runtime)},
        "ledger": _checkpoint_ledger(runtime.step_ledger),
        "actions": _plain(list(runtime.actions.actions)),
        "projection": None if projection is None else {
            key: _plain(getattr(projection, key)) for key in PROJECTION_FIELDS},
        "trimmed": sorted(runtime._trimmed),
        "conversation": checkpoint_conversation(messages),
        "owner_restore": capture_owner_restore(runtime),
    }
    from kernel_code_bookmarks import enabled as bookmarks_enabled
    if bookmarks_enabled() and (
            getattr(runtime, "_code_bookmarks_active", False) is True
            or getattr(runtime, "_reread_result_truth", None)
            or getattr(runtime, "_read_lockout_state", None)):
        from kernel_reread_result_truth import checkpoint_replay
        state["code_bookmark_replay"] = checkpoint_replay(runtime)
    policy = first_green_checkpoint_section(runtime)
    if policy is not None:
        state[FIRST_GREEN_CHECKPOINT_KEY] = policy
    post_green = post_green_checkpoint_section(runtime)
    if post_green is not None:
        state[POST_GREEN_CHECKPOINT_KEY] = post_green
    resume_work = resume_work_checkpoint_section(runtime)
    if resume_work is not None:
        state[RESUME_WORK_CHECKPOINT_KEY] = resume_work
    checkpoint_event = runtime.journal.emit(
        "continuation_worker_checkpoint", checkpoint=state, checkpoint_hash=digest(state))
    _register_checkpoint(runtime, checkpoint_event)


KEYED_DONE_TAIL = frozenset({
    "step_dispatch_binding", "action_receipt", "completion_certification",
    "continuation_public_certification", "submission_idempotency", "challenge_budget_state",
    "completion_reducer_decision",
})


def _keyed_done_tail(runtime, tail):
    """STATEFRAME_SUBMIT_IDEMPOTENT only: a crash anywhere inside a done step
    before its terminal record is safe to retry.  Before a durable intent no
    effect can have happened; after it, the retried done reconciles against the
    keyed destination.  Authorization rests on durable state, never on an audit
    event: the only effect a done tail may contain is a ``complete`` that carries
    a submission key.  Returns the ids of the covered entries, or ``None``."""
    if getattr(runtime, "stateframe_submit_idempotent_mode", "off") != "on":
        return None
    covered = set()
    started = False
    for entry in tail:
        kind = entry["kind"]
        if kind == "step_dispatch_binding":
            tool = entry.get("tool")
            if tool not in (None, "done"):
                return None
            started = started or tool == "done"
            covered.add(id(entry))
            continue
        if not started:
            continue
        if kind in KEYED_DONE_TAIL:
            covered.add(id(entry))
        elif kind == "parent_budget" and entry.get("category") == "tool_work":
            covered.add(id(entry))
        elif (kind == "continuation_transition" and entry.get("operation") == "complete"
              and isinstance(entry.get("data"), dict) and entry["data"].get("submission_key")):
            covered.add(id(entry))
        elif kind in SAFE_TAIL:
            continue        # ordinary safe entries keep their normal check
        else:
            return None
    return covered if started else None


def restore_runtime(runtime):
    from kernel_true_resume import require_enabled
    try:
        require_enabled()
    except RuntimeError as error:
        raise WorkerResumeError(str(error)) from error
    from kernel_continuation_state import unresolved
    from kernel_live_projection import LiveProjection
    reconciled_events = _reconcile_external_effects(runtime)
    from kernel_candidate_restart_reconcile import enabled as candidate_reconcile_enabled
    if candidate_reconcile_enabled():
        from kernel_candidate_restart_reconcile import reconcile
        reconciled_events.update(reconcile(runtime))
    state = runtime.continuation.engine.state
    if any(unresolved(state, identity) for identity in state.intentions):
        raise WorkerResumeError("unresolved effect requires host reconciliation before worker resume")
    records = list(runtime.journal.records())
    try:
        saved = latest_checkpoint(records)
        event = next(record for record in reversed(records)
                     if record.get("kind") == "continuation_worker_checkpoint"
                     and record.get("checkpoint") == saved)
        require_snapshot(records, saved)
        conversation = restore_conversation(saved.get("conversation"))
        owner_restore = validate_owner_restore(saved.get("owner_restore"))
        restore_owner_state(runtime, owner_restore)
    except (ResumeError, StopIteration) as error:
        raise WorkerResumeError(str(error)) from error
    try:
        actual_binding = runtime.binding()
    except (AttributeError, OSError, TypeError, ValueError) as error:
        raise WorkerResumeError("cannot verify retained workspace revision") from error
    if not workspace_revision_matches(saved.get("binding"), actual_binding):
        raise WorkerResumeError("retained workspace revision differs from checkpoint")
    if (type(saved.get("step")) is not int or saved["step"] < 0
            or (runtime.step_limit is not None and saved["step"] >= runtime.step_limit)
            or not _valid_worker_state_fields(saved.get("worker"))
            or not checkpoint_runtime_fields_match(runtime, saved.get("runtime", {}))
            or set(saved.get("owner", {})) - set(OWNER_FIELDS)):
        raise WorkerResumeError("invalid worker checkpoint")
    closed_ledger, replay_saved, slim_ledger = _journal_ledger(saved, records, event)
    runtime.resume_messages = conversation
    runtime.resume_owner_restore = owner_restore
    follow_up = getattr(runtime, "resume_follow_up", None)
    try:
        runtime.resume_messages, tail_follow_up_event = restore_tail_follow_up(
            runtime.resume_messages, records, event["sequence"], follow_up)
    except ResumeError as error:
        raise WorkerResumeError(str(error)) from error
    if tail_follow_up_event is not None:
        follow_up = None
    elif follow_up is not None:
        try:
            runtime.resume_messages = append_follow_up(runtime.resume_messages, follow_up)
        except ResumeError as error:
            raise WorkerResumeError(str(error)) from error
    tail = [e for e in records if e["sequence"] > event["sequence"]]
    if getattr(runtime, "stateframe_model_router_replay_mode", "off") == "on":
        try:
            restore_router_replay_tail(runtime, records, event["sequence"])
        except ValueError as error:
            raise WorkerResumeError(str(error)) from error
    if getattr(runtime, "stateframe_challenge_budget", None) is not None:
        from kernel_challenge_budget import durable_state
        durable_budget = durable_state(
            runtime, records=records,
            checkpoint_state=saved.get("runtime", {}).get("_challenge_budget_state"),
        )
        if durable_budget is not None and durable_budget.get("terminal"):
            raise WorkerResumeError("durable terminal submission cannot resume worker")
    allowed_tail = safe_tail_for(runtime)
    keyed_done = _keyed_done_tail(runtime, tail)
    for entry in tail:
        if entry.get("event_id") in reconciled_events:
            continue
        if keyed_done is not None and id(entry) in keyed_done:
            continue
        if entry["kind"] == "continuation_transition":
            if entry["operation"] not in PREPARE_TRANSITIONS:
                raise WorkerResumeError("post-checkpoint transition requires action reconciliation")
        elif entry["kind"] == "native_batch_received" and runtime.journal.manifest["native_tool_batches"]:
            pass  # Durable proposal only; selection/effect events still refuse.
        elif entry["kind"] not in allowed_tail:
            raise WorkerResumeError("post-checkpoint state requires reconciliation: " + entry["kind"])
        if entry["kind"] == "parent_budget" and entry["category"] in {"tool_work", "attempts", "challenges"}:
            raise WorkerResumeError("post-checkpoint dispatch cannot be retried")
    # A begun request consumes its decision even if no response was received.
    begun = any(e["kind"] == "request_submitted" or (
        e["kind"] == "parent_budget" and e["category"] == "model_calls" and not e["blocked"]) for e in tail)
    from kernel_native_restart import prepare_restore
    retained_receive = prepare_restore(runtime, saved, records, event)
    begun = begun and not retained_receive
    runtime.resume_step = saved["step"] + int(begun)
    if runtime.step_limit is not None and runtime.resume_step >= runtime.step_limit:
        raise WorkerResumeError("original worker step allocation is exhausted")
    if runtime.generated is not None:
        try:
            runtime.generated.restore_snapshot(saved.get("generated"), records, event["sequence"])
        except (KeyError, TypeError, ValueError) as error:
            raise WorkerResumeError("invalid generated proposal restoration") from error
    elif saved.get("generated") is not None:
        raise WorkerResumeError("generated state in a disabled profile")
    if runtime.retained is not None:
        try:
            runtime.retained.restore_snapshot(saved.get("retained"), records, event["sequence"])
            if runtime.retained.checkpoint_snapshot() != saved.get("retained"):
                raise ValueError("retained restoration postcondition mismatch")
        except (KeyError, TypeError, ValueError) as error:
            raise WorkerResumeError("invalid retained source restoration") from error
        runtime._retained_binding = copy.deepcopy(saved["binding"])
    elif saved.get("retained") is not None:
        raise WorkerResumeError("retained state in a disabled profile")
    from kernel_frame_series_replay import restore_series
    restore_series(runtime, replay_saved, records, event)
    from kernel_working_notes_restart import restore as restore_notes
    restore_notes(runtime, replay_saved, records, event)
    runtime.resume_worker_state = copy.deepcopy(saved["worker"])
    restore_runtime_field_values(runtime, saved["runtime"])
    if "code_bookmark_replay" in saved:
        from kernel_code_bookmarks import enabled as bookmarks_enabled
        from kernel_reread_result_truth import restore_replay
        if bookmarks_enabled():
            try:
                restore_replay(runtime, saved["code_bookmark_replay"])
            except ValueError as error:
                raise WorkerResumeError("invalid bookmark read replay") from error
    reconcile_environment_notice_tail(runtime, tail, event["sequence"])
    if getattr(runtime, "stateframe_challenge_budget", None) is not None:
        from kernel_challenge_budget import restore_state
        restore_state(
            runtime,
            records=records,
            checkpoint_state=saved["runtime"].get("_challenge_budget_state"),
        )
    for key, value in saved["owner"].items():
        if key in {"_selector_error_fallback_files", "_futile_answered"}:
            value = set(value)
        elif key in {"f2p_ids", "f2p_names", "f2p_inworld", "oracle_file_ids"}:
            value = tuple(value)
        if key in {"trace", "_receipts"}:
            restored_history = (history(directory=runtime.journal.path.parent) if key == "trace"
                                else receipt_history(directory=runtime.journal.path.parent))
            restored_history.extend(copy.deepcopy(value))
            setattr(runtime.owner, key, restored_history)
        else:
            setattr(runtime.owner, key, copy.deepcopy(value))
    if getattr(runtime, "stateframe_preedit_coverage_arm", None) in {"treatment", "placebo"}:
        from kernel_preedit_coverage_runtime import restore as restore_preedit_coverage
        restore_preedit_coverage(runtime, records, saved["runtime"].get("_preedit_coverage_state"))
    if FIRST_GREEN_CHECKPOINT_KEY in saved:
        restore_first_green_checkpoint(runtime, saved[FIRST_GREEN_CHECKPOINT_KEY])
    if POST_GREEN_CHECKPOINT_KEY in saved:
        restore_post_green_checkpoint(runtime, saved[POST_GREEN_CHECKPOINT_KEY])
    if RESUME_WORK_CHECKPOINT_KEY in saved:
        restore_resume_work_checkpoint(runtime, saved[RESUME_WORK_CHECKPOINT_KEY])
    _restore_ledger(runtime.step_ledger, closed_ledger, slim_ledger,
                    runtime.journal.path.parent)
    if runtime.latest_exchange_mode != "off":
        from kernel_native_restart import latest_closed_exchange
        prefix = [entry for entry in records if entry["sequence"] <= event["sequence"]]
        identity, pair = latest_closed_exchange(prefix, closed_ledger)
        runtime.resume_latest_exchange = (identity, pair)
    runtime.actions.actions.extend(copy.deepcopy(saved["actions"]))
    runtime._trimmed = set(saved["trimmed"])
    initial = runtime.workspace.read("initialized.json")
    root = Path(runtime.owner.root)
    runtime.protected_tests = {root / name: sha for name, sha in initial["protected_tests"].items()}
    runtime.live_projection = LiveProjection(task_id=runtime.owner.task_id,
        symbol=runtime.owner.fn, path=runtime.target.relative_to(root).as_posix(),
        brief=runtime.information.ordinary_brief, selectors=list(runtime.owner.f2p_ids),
        binding=initial["binding"])
    projection = saved["projection"]
    if projection is not None:
        if (set(projection) != set(PROJECTION_FIELDS)
                or projection["initial_harness_revision"] != initial["binding"]["test_harness_revision"]
                or projection["initial_harness_files"] != initial["binding"]["harness_files"]):
            raise WorkerResumeError("checkpoint replaced original public acceptance identity")
        for key, value in projection.items():
            setattr(runtime.live_projection, key,
                    set(value) if key == "seen_trace_events" else copy.deepcopy(value))
    follow_up_event = tail_follow_up_event
    if follow_up is not None:
        try:
            follow_up_event = user_request_event(
                runtime.journal, follow_up,
                request_id=getattr(runtime, "resume_request_id", None))
        except ResumeError as error:
            raise WorkerResumeError(str(error)) from error
    if follow_up_event is not None:
        runtime.resume_follow_up_event = copy.deepcopy(follow_up_event)
    runtime.journal.emit(
        "continuation_resume_authorized",
        checkpoint_ref=event["event_id"],
        execution_authorized=True,
        authorization="explicit_resume_command",
        follow_up_event_id=(follow_up_event["event_id"] if follow_up_event is not None else None),
    )
    runtime.journal.emit(
        "continuation_owner_restore",
        checkpoint_ref=event["event_id"],
        owner_restore=owner_restore,
    )
    runtime.journal.emit("continuation_worker_restored", checkpoint_ref=event["event_id"],
        original_step=saved["step"], next_step=runtime.resume_step,
        discarded_unexecuted_request=begun, parent_budget=runtime.parent.snapshot(),
        previous_binding=saved["binding"], current_binding=runtime.binding(),
        restored_without_chat_history=False)
