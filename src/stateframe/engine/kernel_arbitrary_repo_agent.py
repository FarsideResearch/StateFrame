"""Host-driven arbitrary-repository session for the canonical v29 runtime.

This module is deliberately small: repository checkout, file operations, and
patch export stay in the Lane V29 adapter.  The row below carries those host
capabilities into the same ``StateFrameRuntime`` prepare/observation/finish
boundaries used by ``probes.react_repair.work``.
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from kernel_measurement import digest
from kernel_tool_reply import Outcome, outcome_from_receipt, tool_reply


class Provider(Protocol):
    def __call__(
        self,
        messages: Sequence[Mapping[str, object]],
        tools: Sequence[Mapping[str, object]],
    ) -> Mapping[str, object]: ...


class ToolDispatcher(Protocol):
    def __call__(self, name: str, arguments: Mapping[str, object]) -> object: ...


class RevisionProvider(Protocol):
    def __call__(self) -> str: ...


@dataclass(frozen=True)
class ArbitraryRepoRow:
    """Explicit real-repository row; never coerced into a synthetic build row."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    workspace: Path
    output: Path
    model: str
    max_steps: int
    frame_enabled: bool
    tools: tuple[Mapping[str, object], ...]
    provider: Provider
    dispatch: ToolDispatcher
    revision: RevisionProvider
    check_command: tuple[str, ...]
    source_sha256: str
    target_path: str | None = None
    venv_path: Path | None = None
    environment_revision: str = "swebench-v29-arbitrary-repo/1"
    continuation_parent: Path | None = None
    continuation_workspace: object | None = None

    def public_dict(self) -> dict[str, object]:
        return {
            "schema": "arbitrary-repo-row/1",
            "instance_id": self.instance_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "problem_statement": self.problem_statement,
            "workspace": str(self.workspace),
            "output": str(self.output),
            "model": self.model,
            "max_steps": self.max_steps,
            "frame_enabled": self.frame_enabled,
            "check_command": list(self.check_command),
            "source_sha256": self.source_sha256,
            "target_path": self.target_path,
            "venv_path": str(self.venv_path) if self.venv_path is not None else None,
            "environment_revision": self.environment_revision,
        }


class _Owner:
    """Small owner contract required by StateFrameRuntime's host callbacks."""

    def __init__(self, row: ArbitraryRepoRow) -> None:
        self.root = row.workspace
        self.task_id = row.instance_id
        self.fn = row.target_path or ""
        self.current_step = 0
        self.current_request_id = None
        self.current_prompt_hash = None
        # The canonical continuation root is admitted only with materialized
        # public checks.  Keep the typed row's declared check target as the
        # owner-owned public-check projection instead of inventing a fixture
        # loop or allowing an empty acceptance surface.
        check_target = next(
            (str(token) for token in reversed(row.check_command)
             if str(token) and not str(token).startswith("-")),
            "",
        )
        self.f2p_ids: list[str] = [check_target] if check_target else []
        self.f2p_names: list[str] = [check_target] if check_target else []
        self.f2p_inworld: list[str] = [check_target] if check_target else []
        self.trace: list[dict[str, object]] = []
        self.task_row = row.public_dict()
        self.task_row["information_policy"] = {
            "information_policy_version": "information-policy/1",
            "initial_oracle_symptom": "OFF",
            "live_grader_feedback": "FINAL_ONLY",
            "target_disclosure": "FILE_AND_SYMBOL",
        }
        from kernel_information_policy import InformationPolicy
        self.information_policy = InformationPolicy(self.task_row)
        self._receipts: list[dict[str, object]] = []
        self._effect_audit_log: list[object] = []
        self.predispatch_gate = None

    def receipt(self, kind: str, name: str, target: str, result_class: str,
                detail: str = "", **fields: object) -> dict[str, object]:
        receipt = {
            "kind": kind,
            "name": name,
            "target": target,
            "result_class": result_class,
            "detail": detail,
            "turn": self.current_step,
            **fields,
        }
        self._receipts.append(receipt)
        self.trace.append(receipt)
        return receipt


@dataclass
class _SessionState:
    verification_status: str = "unverified"
    verification_revision: str | None = None
    verification_command: tuple[str, ...] = ()
    verification_scope: str = "declared_repository_check"
    stop_requested: bool = False
    completion_reason: str | None = None
    admission_status: str | None = None
    admission: dict[str, object] | None = None
    done_refusals: int = 0
    created_paths: set[str] = field(default_factory=set)
    tool_log: list[dict[str, object]] = field(default_factory=list)

    def invalidate(self) -> None:
        self.verification_status = "unverified"
        self.verification_revision = None
        self.verification_command = ()


def _json_text(value: object, limit: int = 2_000) -> str:
    try:
        rendered = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        rendered = repr(value)
    return rendered[:limit]


def _decode_call(raw_call: Mapping[str, object]) -> tuple[str, dict[str, object], str]:
    function = raw_call.get("function")
    if not isinstance(function, Mapping) or not isinstance(function.get("name"), str):
        raise ValueError("model returned a tool call without a function name")
    raw_arguments = function.get("arguments", "{}")
    if not isinstance(raw_arguments, str):
        raise ValueError("tool arguments must be JSON text")
    arguments = json.loads(raw_arguments)
    if not isinstance(arguments, Mapping):
        raise ValueError("tool arguments must decode to an object")
    return str(function["name"]), dict(arguments), str(raw_call.get("id") or "call")


def _tool_mutates(name: str) -> bool:
    return name in {"edit_file", "create_file", "delete_file", "rename_file"}


def _runtime_manifest(runtime: object) -> dict[str, object]:
    journal = getattr(runtime, "journal", None)
    manifest = getattr(journal, "manifest", {})
    if not isinstance(manifest, Mapping):
        manifest = {}
    records = getattr(journal, "records", ())
    records = records() if callable(records) else records
    return {
        "entrypoint": "probes.react_repair.work",
        "runtime": "kernel_stateframe_runtime.StateFrameRuntime",
        "source_sha256": getattr(runtime, "source_sha256", None),
        "journal_manifest": copy.deepcopy(dict(manifest)),
        "receipt_count": len(records) if isinstance(records, Sequence) else None,
    }


def _provider_call(row: ArbitraryRepoRow, messages: Sequence[Mapping[str, object]],
                   tools: Sequence[Mapping[str, object]], *, routed: object | None = None,
                   runtime: object | None = None) -> Mapping[str, object]:
    if routed is None:
        return row.provider(messages, tools)
    request = getattr(row.provider, "request", None)
    if not callable(request):
        raise TypeError("model routing requires a provider request seam")
    body = getattr(routed, "body", {})
    model = body.get("model") if isinstance(body, Mapping) else None
    endpoint = getattr(routed, "endpoint", None)
    journal = getattr(runtime, "journal", None)
    receipt = {
        "model_id": model,
        "endpoint": endpoint,
        "assignment_id": getattr(getattr(routed, "decision", None), "lease", None).lease_id
        if getattr(getattr(routed, "decision", None), "lease", None) is not None else None,
        "request_id": getattr(runtime, "request_id", None),
        "request_digest": digest(body),
        "transport": "provider.request",
    }
    if callable(getattr(journal, "emit", None)):
        journal.emit("model_router_provider_request_started", **receipt)
    try:
        response = request(messages, tools, model=model, endpoint=endpoint, routing=routed)
    except Exception:
        if callable(getattr(journal, "emit", None)):
            journal.emit("model_router_provider_request_failed", **receipt)
        raise
    if callable(getattr(journal, "emit", None)):
        journal.emit("model_router_provider_request_completed", **receipt)
    return response


def _completion_record(runtime: object, state: _SessionState, row: ArbitraryRepoRow,
                       summary: str) -> tuple[bool, dict[str, object]]:
    current_revision = row.revision()
    verified = (
        state.verification_status == "passed"
        and state.verification_revision == current_revision
        and state.verification_command == row.check_command
    )
    decision_payload: dict[str, object] = {
        "requested": True,
        "verification_current": verified,
        "verification_status": state.verification_status,
        "verification_revision": state.verification_revision,
        "current_revision": current_revision,
        "verification_command": list(state.verification_command),
    }
    accepted = verified
    if runtime is not None:
        try:
            decision = runtime.completion_decision(
                admissible=verified,
                detail=summary[:2_000],
                reason_codes=() if verified else ("UNVERIFIED",),
                selectors=(),
                path=row.target_path,
                evidence={"verification_revision": current_revision},
                artifact_preserved=True,
            )
            accepted = bool(getattr(decision, "accepted", False))
            decision_payload["runtime_decision"] = (
                decision.to_dict() if hasattr(decision, "to_dict") else str(decision)
            )
        except Exception as exc:  # runtime authority remains fail-closed
            accepted = False
            decision_payload["runtime_error"] = type(exc).__name__
    if not verified:
        state.completion_reason = "completion requested without current verification"
        state.stop_requested = True
        if state.verification_status == "passed":
            state.verification_status = "stale"
        return False, decision_payload
    if not accepted:
        state.completion_reason = "canonical runtime did not accept completion"
        state.stop_requested = True
        return False, decision_payload
    state.completion_reason = "verified against current artifact"
    state.stop_requested = True
    return True, decision_payload


def _host_progress_facts(owner: _Owner, state: _SessionState,
                         runtime: object | None, *, enabled: bool | None = None):
    """Read arbitrary-repository progress from host receipts when enabled."""
    from kernel_progress_facts import (
        derive_host_progress_facts,
        host_truth_progress_enabled,
    )

    selected = host_truth_progress_enabled() if enabled is None else enabled
    if not selected:
        return None
    resumed = getattr(runtime, "resume_worker_state", None)
    prior_facts = (resumed.get("host_progress_facts")
                   if isinstance(resumed, Mapping) else None)
    return derive_host_progress_facts(
        edit_receipts=owner._receipts,
        test_receipts=getattr(runtime, "test_receipts", ()),
        host_check_receipts=getattr(runtime, "host_check_receipts", ()),
        owner_receipts=owner._receipts,
        file_create_results=state.created_paths,
        prior_facts=prior_facts,
        enabled=True,
    )


def _checkpoint_state(owner: _Owner, state: _SessionState,
                      runtime: object | None = None,
                      progress_enabled: bool | None = None) -> dict[str, object]:
    """Adapt the typed row state to the restart checkpoint contract."""
    saved: dict[str, object] = {
        "edits": len(state.created_paths),
        "tests": int(state.verification_status == "passed"),
        "unparsed_streak": 0,
        "harness_suspect": False,
        "last_line": "",
        "same_streak": 0,
        "run_temp": 0.0,
        "edits_since_test": 0 if state.verification_status == "passed" else len(state.created_paths),
        "baseline_tests": 0,
        "created_this_run": sorted(state.created_paths),
        "last_surgery_step": owner.current_step,
        "tool_calls_seen": len(state.tool_log),
        "decoy_call_names": [],
        "schema_violation_details": [],
        "unknown_tool_names": [],
        "first_edit_step": 1 if state.created_paths else None,
        "wrong_tool_before_first_edit": False,
    }
    facts = _host_progress_facts(owner, state, runtime, enabled=progress_enabled)
    if facts is not None:
        saved.update({
            "edits": facts.edits,
            "tests": facts.checks,
            "edits_since_test": facts.edits_since_check,
            "first_edit_step": facts.first_edit_step,
            "host_progress_facts": facts.to_record(),
        })
    return saved


def run(row: ArbitraryRepoRow, *, resume: object | None = None,
        continuation_workspace: object | None = None) -> dict[str, object]:
    """Run one typed arbitrary-repository session through canonical runtime hooks."""
    if row.max_steps < 1:
        raise ValueError("max_steps must be positive")
    from kernel_progress_facts import host_truth_progress_enabled

    progress_enabled = host_truth_progress_enabled()
    from kernel_wire_string_hygiene import (
        record_replacements,
        sanitize_message,
        sanitize_messages,
        sanitize_text,
        sanitize_value,
    )

    owner = _Owner(row)
    runtime = None
    session = continuation_workspace or row.continuation_workspace
    restart_enabled = False
    if row.frame_enabled:
        from kernel_continuation_restart import enabled as restart_switch
        restart_enabled = restart_switch()
        if restart_enabled and session is None:
            raise ValueError("true resume requires an adapter-owned continuation workspace")
    if row.frame_enabled:
        from kernel_stateframe_runtime import StateFrameRuntime

        target = (row.workspace / row.target_path) if row.target_path else None
        runtime = StateFrameRuntime(
            owner,
            output=row.output,
            target=target if target is not None and target.is_file() else None,
            model=row.model,
            tools=[dict(tool) for tool in row.tools],
            temperature=0.0,
            source_resolver=lambda source, symbol: source,
            step_limit=row.max_steps,
            multi_file=True,
            workspace=session,
            resume=resume,
        )

    messages: list[dict[str, object]] = [
        {
            "role": "system",
            "content": (
                "Repair one public repository issue. Use repository tools to inspect the current "
                "checkout, make the smallest correct change, run the declared check, and call done "
                "only after that check passes on the unchanged current artifact."
            ),
        },
        {
            "role": "user",
            "content": f"Issue {row.instance_id} ({row.repo}):\n{row.problem_statement}",
        },
    ]
    _, replacements = sanitize_messages(messages)
    record_replacements(getattr(runtime, "journal", None), replacements, step=0)
    state = _SessionState()
    completed = False
    error: str | None = None
    step = 0
    active_token: object | None = None
    try:
        start_step = int(getattr(runtime, "resume_step", 0) or 0) + 1
        if runtime is not None and getattr(runtime, "resume_messages", None) is not None:
            messages = [copy.deepcopy(dict(message))
                        for message in runtime.resume_messages]
            _, replacements = sanitize_messages(messages)
            record_replacements(getattr(runtime, "journal", None), replacements,
                                step=start_step)
        for step in range(start_step, row.max_steps + 1):
            owner.current_step = step
            _, replacements = sanitize_messages(messages)
            record_replacements(getattr(runtime, "journal", None), replacements,
                                step=step)
            offered_tools = [dict(tool) for tool in row.tools]
            if runtime is not None and runtime.target is None:
                offered_tools = runtime.targetless_tool_surface(offered_tools)
            offered_tools, replacements = sanitize_value(offered_tools)
            record_replacements(getattr(runtime, "journal", None), replacements,
                                step=step)
            request_messages: Sequence[Mapping[str, object]] = messages
            if runtime is not None:
                prepared = runtime.prepare(
                    messages,
                    uncapped=messages,
                    tools=offered_tools,
                )
                if prepared.overflow is not None:
                    error = "canonical runtime could not prepare a provider request"
                    state.completion_reason = error
                    break
                request_messages = prepared.messages
                active_token = runtime.activate(prepared)

            routed_call = None
            if runtime is not None and getattr(runtime, "model_router", None) is not None:
                router = runtime.model_router
                body = {"model": row.model, "messages": list(request_messages), "tools": offered_tools}
                routed_call = router.prepare_call(
                    prepared.capture, body, os.environ.get("DS_URL", ""),
                    os.environ.get("OPENAI_API_KEY", ""),
                )
            try:
                assistant = _provider_call(row, request_messages, offered_tools,
                                           routed=routed_call, runtime=runtime)
            finally:
                if active_token is not None and runtime is not None:
                    from kernel_measurement import ACTIVE_REQUEST

                    ACTIVE_REQUEST.reset(active_token)
                    active_token = None
            if not isinstance(assistant, Mapping):
                error = "provider response must be an object"
                break
            assistant_message = {
                key: value for key, value in assistant.items()
                if key in {"role", "content", "tool_calls"}
            }
            assistant_message.setdefault("role", "assistant")
            assistant_message, replacements = sanitize_message(
                assistant_message, message_index=len(messages))
            record_replacements(getattr(runtime, "journal", None), replacements,
                                step=step)
            if isinstance(assistant_message, Mapping):
                assistant_message = dict(assistant_message)
            messages.append(assistant_message)
            calls = assistant.get("tool_calls")
            if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)) or not calls:
                state.tool_log.append({
                    "step": step,
                    "tool": None,
                    "status": "no_action",
                    "detail": "text-only provider response is recoverable, not completion",
                })
                messages.append({
                    "role": "user",
                    "content": "No terminal action was recorded. Continue with a repository read, edit, check, or explicit done call.",
                })
                if runtime is not None:
                    runtime.close_step(messages)
                if runtime is not None and restart_enabled:
                    from kernel_continuation_restart import checkpoint
                    checkpoint(runtime, messages, _checkpoint_state(
                        owner, state, runtime, progress_enabled))
                continue

            for raw_call in calls:
                if not isinstance(raw_call, Mapping):
                    error = "provider returned a malformed tool call"
                    break
                try:
                    name, arguments, call_id = _decode_call(raw_call)
                    feedback = None
                    handled_result: object | None = None
                    if runtime is not None:
                        feedback = runtime.idealpath_admission(name, arguments)
                        if feedback is None:
                            handled = runtime.idealpath_operation(name, arguments)
                            if handled is not None and name == "localize":
                                handled_result = json.loads(handled)
                            else:
                                feedback = handled
                        if feedback is None:
                            feedback = runtime.start_action(name, arguments, charge_dispatch=False)
                    result: object
                    if handled_result is not None:
                        result = handled_result
                    elif feedback is not None:
                        result = {"status": "REJECTED", "error": str(feedback)}
                    else:
                        result = row.dispatch(name, arguments)
                    result, replacements = sanitize_message(
                        result, message_index=len(messages))
                    record_replacements(getattr(runtime, "journal", None), replacements,
                                        step=step)
                    result_mapping = result if isinstance(result, Mapping) else {}
                    result_text = _json_text(result)
                    result_text, replacements = sanitize_text(
                        result_text, message_index=len(messages))
                    record_replacements(getattr(runtime, "journal", None), replacements,
                                        step=step)
                    result_status = str(result_mapping.get("status") or "").strip().upper()
                    mutation_succeeded = (
                        result_status in {"APPLIED", "CHANGED", "CREATED", "SUCCESS", "OK",
                                           "DELETED", "RENAMED"}
                        and result_mapping.get("changed") is not False
                        and result_mapping.get("effect_applied") is not False
                    )
                    legacy_mutation_succeeded = (
                        result_mapping.get("status") not in {"ERROR", "REJECTED"}
                    )
                    if _tool_mutates(name) and (
                            mutation_succeeded if progress_enabled else legacy_mutation_succeeded):
                        state.invalidate()
                    create_succeeded = (
                        mutation_succeeded if progress_enabled else legacy_mutation_succeeded
                    )
                    if (name == "create_file" and create_succeeded
                            and isinstance(result_mapping.get("path"), str)):
                        state.created_paths.add(str(result_mapping["path"]))
                    progress_receipt: dict[str, object] = {
                        "kind": "tool",
                        "result_class": "REJECTED",
                        "receipt_id": f"{row.instance_id}:{step}:{call_id}",
                    }
                    if progress_enabled and name in {
                            "edit_file", "create_file", "delete_file", "rename_file"}:
                        progress_receipt.update({
                            "kind": name,
                            "result_class": result_status if mutation_succeeded else "REJECTED",
                            "changed": mutation_succeeded,
                            "path": str(result_mapping.get("path")
                                        or arguments.get("path") or ""),
                        })
                    if name == "run_check":
                        code = result_mapping.get("returncode")
                        executed = (
                            type(code) is int
                            and result_status not in {"SKIPPED", "ERROR", "REJECTED"}
                        )
                        check_receipt_id = f"{row.instance_id}:{step}:{call_id}"
                        if progress_enabled:
                            progress_receipt.update({
                                "kind": "run_check",
                                "result_class": (
                                    "PASSED" if executed and code == 0 else
                                    "FAILED" if executed else "NOT_EXECUTED"
                                ),
                                "execution": "EXECUTED" if executed else "NOT_EXECUTED",
                                "execution_id": check_receipt_id,
                                "receipt_id": check_receipt_id,
                                "path": str(result_mapping.get("path") or "run_check"),
                            })
                        command_tokens = tuple(str(item) for item in row.check_command)
                        created_in_check = tuple(
                            path for path in state.created_paths
                            if any(path in token for token in command_tokens)
                        )
                        if created_in_check:
                            state.verification_scope = (
                                "agent_authored_test_only"
                                if set(command_tokens[3:]) and set(command_tokens[3:]).issubset(state.created_paths)
                                else "declared_check_includes_agent_authored_test"
                            )
                        if code == 0 and result_mapping.get("status") not in {"SKIPPED", "ERROR"}:
                            state.verification_status = "passed"
                            state.verification_revision = row.revision()
                            state.verification_command = row.check_command
                        elif result_mapping.get("status") == "SKIPPED":
                            state.verification_status = "unverified"
                        else:
                            state.verification_status = "failed"
                        should_observe_check = (
                            type(code) is int and executed if progress_enabled
                            else isinstance(code, int)
                        )
                        if runtime is not None and should_observe_check:
                            if progress_enabled:
                                typed_check = {
                                    "status": "PASS" if code == 0 else "UNKNOWN",
                                    "outcome": "PASS" if code == 0 else "UNKNOWN",
                                    "execution_complete": True,
                                    "reached_observation": True if code == 0 else None,
                                }
                                runtime.test_observation(
                                    [], code, runtime.binding(), observation=result_text,
                                    typed_observation=typed_check,
                                )
                            else:
                                runtime.test_observation(
                                    [], code, runtime.binding(), observation=result_text)
                            if progress_enabled:
                                rows = getattr(runtime, "test_receipts", ())
                                latest = (rows[-1] if isinstance(rows, Sequence) and rows
                                          and isinstance(rows[-1], Mapping) else None)
                                if latest is not None:
                                    for identity_field in (
                                            "receipt_id", "execution_id", "execution_ids",
                                            "host_execution_id", "check_receipt_id",
                                            "test_receipt_id", "event_id", "check_id",
                                            "instance_key", "check_receipt_ids"):
                                        if latest.get(identity_field) not in (None, ""):
                                            progress_receipt[identity_field] = latest[identity_field]
                    if runtime is not None:
                        runtime.observe(name, result_text)
                    if name == "done":
                        summary = str(arguments.get("summary") or "")
                        completed, decision = _completion_record(runtime, state, row, summary)
                        result = {**dict(result_mapping), "accepted": completed, "decision": decision}
                        result, replacements = sanitize_message(
                            result, message_index=len(messages))
                        record_replacements(getattr(runtime, "journal", None), replacements,
                                            step=step)
                        result_text = _json_text(result)
                        result_text, replacements = sanitize_text(
                            result_text, message_index=len(messages))
                        record_replacements(getattr(runtime, "journal", None), replacements,
                                            step=step)
                    state.tool_log.append({
                        "step": step,
                        "tool": name,
                        "status": "ok" if result_mapping.get("status") not in {"ERROR", "REJECTED"} else "error",
                        "result": result_text[:1_000],
                    })
                    if progress_enabled:
                        progress_fields = {
                            key: value for key, value in progress_receipt.items()
                            if key not in {"kind", "result_class", "path"}
                        }
                        if progress_receipt["kind"] in {
                                "edit_file", "create_file", "delete_file", "rename_file",
                                "run_check"}:
                            progress_fields["progress_kind"] = progress_receipt["kind"]
                            progress_fields["progress_result_class"] = progress_receipt["result_class"]
                        owner.receipt(
                            "tool", name,
                            str(progress_receipt.get("path") or arguments.get("path") or name),
                            "APPLIED" if state.tool_log[-1]["status"] == "ok" else "REJECTED",
                            result_text[:1_000], **progress_fields,
                        )
                    else:
                        owner.receipt(
                            "tool", name, str(arguments.get("path") or name),
                            "APPLIED" if state.tool_log[-1]["status"] == "ok" else "REJECTED",
                            result_text[:1_000],
                        )
                    reply_outcome, reply_receipt = outcome_from_receipt(result_mapping)
                    messages.append(tool_reply(
                        call_id, name, reply_outcome, result_text, receipt=reply_receipt,
                    ))
                    if state.stop_requested:
                        break
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    error_result, replacements = sanitize_message(
                        {"status": "ERROR", "error": str(exc)[:2_000]},
                        message_index=len(messages))
                    record_replacements(getattr(runtime, "journal", None), replacements,
                                        step=step)
                    result_text = _json_text(error_result)
                    result_text, replacements = sanitize_text(
                        result_text, message_index=len(messages))
                    record_replacements(getattr(runtime, "journal", None), replacements,
                                        step=step)
                    state.tool_log.append({"step": step, "tool": "unknown", "status": "error", "result": result_text})
                    messages.append(tool_reply(
                        "call", "unknown", Outcome.ERROR, result_text,
                        receipt={"code": "tool_error"},
                    ))
            if runtime is not None and not error:
                runtime.close_step(messages)
            if error or state.stop_requested:
                break
            if runtime is not None and restart_enabled and not state.stop_requested:
                from kernel_continuation_restart import checkpoint
                checkpoint(runtime, messages, _checkpoint_state(
                    owner, state, runtime, progress_enabled))
    except Exception as exc:  # host/runtime failures are terminal and explicit
        from kernel_optional_engine_module import optional_module
        endpoint_extension = optional_module("STATEFRAME_ENDPOINT_PROFILE_MODULE")
        admission_error_type = getattr(endpoint_extension, "EndpointAdmissionError", None)
        if not isinstance(admission_error_type, type) or not issubclass(admission_error_type, BaseException):
            error = f"canonical session failed: {type(exc).__name__}: {str(exc)[:500]}"
            state.completion_reason = error
        else:
            if isinstance(exc, admission_error_type):
                state.admission_status = exc.decision.status.value
                state.admission = exc.decision.mapping()
                state.completion_reason = exc.decision.reason
                error = None
            else:
                error = f"canonical session failed: {type(exc).__name__}: {str(exc)[:500]}"
                state.completion_reason = error
    finally:
        if active_token is not None and runtime is not None:
            from kernel_measurement import ACTIVE_REQUEST

            ACTIVE_REQUEST.reset(active_token)
        if not completed and state.completion_reason is None:
            state.completion_reason = (
                "step limit exhausted without explicit terminal intent"
                if step >= row.max_steps else "session stopped before verified completion"
            )
        final = {
            "schema": "arbitrary-repo-result/1",
            "completed": completed,
            "steps": step,
            "verification": {
                "status": state.verification_status,
                "revision": state.verification_revision,
                "current_revision": row.revision(),
                "command": list(state.verification_command),
                "scope": state.verification_scope,
            },
            "stop": {
                "requested": state.stop_requested,
                "status": "verified" if completed else "unverified",
                "reason": state.completion_reason,
            },
            "admission": state.admission,
            "admission_status": state.admission_status,
            "tool_log": state.tool_log,
            "runtime": (
                {
                    **_runtime_manifest(runtime),
                    "source_sha256": row.source_sha256,
                }
                if runtime is not None
                else {
                    "entrypoint": "probes.react_repair.work",
                    "runtime": "canonical-arbitrary-repo-agent",
                    "frame_enabled": False,
                    "source_sha256": row.source_sha256,
                }
            ),
            "error": error,
        }
        facts = _host_progress_facts(
            owner, state, runtime, enabled=progress_enabled)
        if facts is not None:
            final["progress"] = facts.to_record()
        if runtime is not None:
            try:
                runtime.finish(final, messages=messages)
            except Exception as exc:  # preserve the patch/result if telemetry finalization fails
                final["runtime_finish_error"] = type(exc).__name__
        if session is not None:
            session.close()
    return final
