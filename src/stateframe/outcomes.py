"""Product outcomes project engine evidence; they never certify semantic correctness."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .test_runners import parse_command, verification_evidence_for_tests


# Explicit statuses from the engine stop taxonomy; unknown statuses fail closed.
BUDGET_STOPS = {"RESPONSE_BUDGET_EXHAUSTED", "ATTEMPT_BUDGET_EXHAUSTED", "BUDGET_EXHAUSTED"}
INCOMPLETE = {"RUN_INCOMPLETE", "STOPPED_INCOMPLETE", "TEST_REPORTED_WRONG"}
# The engine could not build/seal its next request (a harness contract or finalization failure): an engine
# error with a named reason, never a silently "missing" result.
SEAL_FAILURES = {"HARNESS_CONTRACT", "FINALIZATION_ERROR"}
# Non-seal engine failures whose child terminal may carry error_type/error_message (see _child.py).
ENGINE_FAILURES_WITH_MESSAGE = {"ENGINE_ERROR", "MUTATION_CONTAINMENT"}
# Refusals that would only repeat on resume (runner copies their guidance from engine.log).
REPEATING_ENGINE_ERRORS = {"ReplayUnfaithful", "CheckpointUnfaithful"}


RECOVERED_FROM_JOURNAL = "recovered_from_journal"


def terminal_receipt(run: Path) -> dict[str, object]:
    try:
        envelope = json.loads((run / "engine-result.json").read_text())
    except (OSError, ValueError):
        return journal_submission(run)
    result = envelope.get("result") if isinstance(envelope, dict) else None
    if not isinstance(result, dict) or not isinstance(result.get("result"), str):
        return journal_submission(run)
    recovery_fields = ("check_receipt", "source_map_path", "final_source_paths")
    if (result.get("result") == "SUBMITTED"
            and any(result.get(key) is None for key in recovery_fields)):
        recovered = journal_submission(run)
        submission = result.get("submission")
        recovered_submission = recovered.get("submission")
        if (isinstance(submission, dict) and isinstance(recovered_submission, dict)
                and submission.get("seal_id") == recovered_submission.get("seal_id")
                and submission.get("artifact_digest") == recovered_submission.get("artifact_digest")):
            recovered_fields = {
                key: recovered[key] for key in recovery_fields
                if result.get(key) is None and recovered.get(key) is not None
            }
            if recovered_fields:
                return {**result, **recovered_fields}
    return result


def journal_submission(run: Path) -> dict[str, object]:
    return _journal_submission_details(run)[0]


def _journal_submission_details(
        run: Path) -> tuple[dict[str, object], dict[str, object] | None]:
    """A stored DoneSeal from the engine journal, for an engine stopped before its terminal receipt.

    In one long run the engine sealed and stored its submission inside its wall, then was killed while
    finishing up, so engine-result.json never appeared. Recovery requires a stored done_sealed and a
    matching done_submission_stored; the export still re-verifies the artifact bytes against the digest.
    Auto-finished subframes also retain their candidate-bound check receipt in the journal.
    """
    from .engine.kernel_journal_store import journal_paths, open_journal

    journals = journal_paths(run / "worker" / "measurement-runs")
    if len(journals) != 1:
        return {}, None
    sealed: dict[str, object] | None = None
    stored: dict[str, object] | None = None
    latest_auto_finish_receipt: dict[str, object] | None = None
    sealed_check_receipt: dict[str, object] | None = None
    latest_final_sources: dict[str, object] | None = None
    try:
        with open_journal(journals[0]) as source:
            lines = source.read().splitlines()
    except (OSError, UnicodeError):
        return {}, None
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        kind = event.get("kind")
        # These evidence records are durable journal events but do not carry the
        # `stored` marker used by the submission pair below. Their association
        # with a submission is established by event order and the sealed digest.
        metadata_is_durable = event.get("stored") is None or event.get("stored") is True
        if kind == "subframe_auto_finished" and metadata_is_durable:
            receipt = event.get("check_receipt")
            latest_auto_finish_receipt = receipt if isinstance(receipt, dict) else None
        elif kind == "subframe_final_sources" and metadata_is_durable:
            artifact_digest = event.get("artifact_digest")
            source_map_path = event.get("source_map_path")
            final_source_paths = event.get("final_source_paths")
            if (isinstance(artifact_digest, str) and isinstance(source_map_path, str)
                    and isinstance(final_source_paths, list)
                    and all(isinstance(path, str) for path in final_source_paths)):
                latest_final_sources = {
                    "artifact_digest": artifact_digest,
                    "source_map_path": source_map_path,
                    "final_source_paths": final_source_paths,
                }
            else:
                latest_final_sources = None
        elif kind == "done_sealed" and event.get("stored") is True:
            sealed, stored = event, None
            sealed_check_receipt = latest_auto_finish_receipt
        elif (kind == "done_submission_stored" and event.get("stored") is True
              and sealed is not None):
            stored = event
    if sealed is None or stored is None:
        return {}, None
    digest, seal_id = sealed.get("artifact_digest"), sealed.get("seal_id")
    if stored.get("artifact_digest") != digest:
        return {}, None
    submission = {"stored": True, "seal_id": seal_id, "artifact_digest": digest}
    terminal = {"result": "SUBMITTED", "reason_code": RECOVERED_FROM_JOURNAL,
                "submission": submission}
    if sealed.get("schema") == "stateframe-done-seal/2":
        if (stored.get("schema") != sealed.get("schema")
                or stored.get("file_digests") != sealed.get("file_digests")
                or not isinstance(sealed.get("file_digests"), dict)
                or not isinstance(sealed.get("labels"), dict)):
            return {}, None
        submission.update({
            "schema": sealed["schema"], "source": sealed.get("source"),
            "label": sealed.get("label"), "file_digests": sealed["file_digests"],
        })
        terminal["labels"] = sealed["labels"]
    if not sealed_artifact_digest(terminal):
        return {}, None
    check_receipt = sealed_check_receipt if sealed.get("source") == "host" else None
    if check_receipt is not None:
        terminal["check_receipt"] = check_receipt
    if (latest_final_sources is not None
            and latest_final_sources.get("artifact_digest") == digest):
        terminal["source_map_path"] = latest_final_sources["source_map_path"]
        terminal["final_source_paths"] = latest_final_sources["final_source_paths"]
    return terminal, check_receipt


def sealed_artifact_digest(terminal: dict[str, object]) -> str | None:
    """Read the engine's submission identity, never infer one from a capture index."""
    submission = terminal.get("submission")
    if not isinstance(submission, dict) or submission.get("stored") is not True:
        return None
    for key in ("seal_id", "artifact_digest"):
        value = submission.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
            return None
    return submission["artifact_digest"]


def check_label(result: dict[str, object]) -> str:
    if result.get("execution_complete") is False:
        return "NO PROGRESS" if result.get("termination_reason") == "no_progress" else "INCOMPLETE"
    passed = result["passed"]
    return "PASS" if passed else "FAIL" if passed is False else "NOT RUN"


def project(summary: dict[str, object], *, cancelled: bool = False,
            timed_out: bool = False, check_output: str = "",
            check_repo: Path | None = None, declared_test_command: bool = False) -> dict[str, object]:
    terminal = summary.get("engine_terminal") or {}
    status = terminal.get("result")
    child = summary.get("child_exit_code")
    recovered = (status == "SUBMITTED" and terminal.get("reason_code") == RECOVERED_FROM_JOURNAL
                 and not summary.get("candidate_error"))
    if summary.get("runtime_outcome") == "hook_containment" or summary.get("hook_containment"):
        runtime = "hook_containment"
    elif status == "HOOK_CONTAINMENT":
        runtime = "hook_containment"
    elif status == "HOOK_BLOCKED":
        runtime = "hook_blocked"
    elif status == "CONTINUATION_BLOCKED":
        runtime = "continuation_blocked"
    elif status == "BLOCKED" and terminal.get("resumable") is True:
        # Working-set suspension: a typed, resumable stop at an unsent request boundary.
        runtime = "suspended"
    elif cancelled or child in (130, -2) or status in ("CANCELLED", "CANCELED"):
        runtime = "cancelled"
    elif recovered:
        # The engine sealed and stored its submission before it was stopped: that is a submission.
        runtime = "completed"
    elif timed_out:
        runtime = "budget_stop"
    elif status == "RECOVERY_REQUIRED":
        runtime = "recovery_required"
    elif (child != 0 or summary.get("candidate_error") or status == "ENGINE_ERROR" or status in SEAL_FAILURES
          or (isinstance(status, str) and status.startswith("ERR:"))):
        runtime = "engine_error"
    elif status in BUDGET_STOPS:
        runtime = "budget_stop"
    elif status == "RUN_INCOMPLETE" and terminal.get("reason_code") == "CONTEXT_TOO_SMALL":
        runtime = "context_too_small"
    elif status == "SUBMITTED":
        # completed = the engine ended normally with a submission, not proof the task is correct.
        runtime = "completed"
    elif status in INCOMPLETE:
        runtime = "incomplete"
    else:
        runtime = "missing_result"

    after = summary.get("tests_after") or {}
    digest = summary.get("candidate_sha256")
    bound = bool(digest) and after.get("candidate_sha256") == digest
    check = "not_run"
    if bound:
        from .engine.kernel_check_watchdog import validate_execution_stop

        stop = validate_execution_stop(after.get("execution_stop"))
        if stop is not None or after.get("execution_complete") is False:
            check = "no_progress" if stop is not None and stop["cause"] == "no_progress" else "incomplete"
        else:
            check = "pass" if after.get("passed") is True and after.get("exit_code") == 0 else "fail"
    sealed_digest = sealed_artifact_digest(terminal)
    stored = sealed_digest is not None and summary.get("candidate_artifact_digest") == sealed_digest
    captured = summary.get("candidate_captured") is True
    if summary.get("candidate_error"):
        artifact = "partial"
    elif not captured:
        artifact = "none"
    elif runtime == "completed" and status == "SUBMITTED" and stored:
        artifact = "saved"
    else:
        artifact = "partial"
    projected = {"runtime_outcome": runtime, "check_outcome": check, "artifact": artifact}
    source = summary.get("check_source")
    after_command = after.get("command")
    kind = str(summary.get("runner_kind") or "")
    try:
        command = parse_command(after_command, kind)[1] if isinstance(after_command, str) else ()
    except ValueError:
        command = ()
    final_source, test_label, evidence = verification_evidence_for_tests(
        source if isinstance(source, str) else None, kind, command, check_output,
        passed=check == "pass", declared=declared_test_command, repo=check_repo)
    projected["check_source"] = final_source
    projected["test_execution_evidence"] = evidence
    if summary.get("runner_kind") == "generic" and runtime == "completed" and artifact == "saved":
        if check == "pass":
            projected["verification"] = test_label or {"build": "verified by build only",
                                                        "command": "verified by check command"}.get(
                                                            final_source, "verified by check command")
        elif final_source == "none":
            projected["verification"] = ("verified by an agent-written check"
                                         if summary.get("authored_check_receipt_id") else
                                         "unverified: review the diff")
    elif (summary.get("runner_kind") == "pytest" and summary.get("protect_tests") != "all"
          and runtime == "completed" and check == "pass" and artifact == "saved"):
        projected["verification"] = test_label or "verified by check command"
    if projected.get("verification") and summary.get("test_weakening_findings"):
        findings = summary["test_weakening_findings"]
        detail = "; ".join(f"{item['test']} ({item['reason']}; heuristic)" for item in findings)
        count = len({item["test"].split("::", 1)[0] for item in findings})
        projected["verification"] += f", but {count} test file weakened: {detail}"
    if check in {"no_progress", "incomplete"}:
        projected["verification"] = "verification incomplete"
        projected["verification_detail"] = str((stop or {}).get("cause", "check interrupted")).replace("_", " ")
    elif status == "TEST_REPORTED_WRONG" and summary.get("wrong_test_evidence"):
        projected["verification"] = "stopped: test reported wrong, see evidence"
    elif projected.get("verification"):
        projected["verification_detail"] = (
            f"exit {after.get('exit_code')}: {after_command}"
            if after_command and after.get("exit_code") is not None else
            f"check receipt {summary['authored_check_receipt_id']}"
            if summary.get("authored_check_receipt_id") else "no check command ran")
    reason = engine_error_reason(summary)
    if runtime == "engine_error" and reason:
        projected["engine_error_reason"] = reason
    return projected


def resume_hint_applies(summary: dict[str, object]) -> bool:
    """True when a stopped run should be offered `stateframe resume`.

    Terminal submissions and seal/finalization failures are already finished or would only repeat on
    resume, so they get no hint; an ERR:* crash keeps it.
    """
    delivery_refusal = summary.get("delivery_refusal")
    if (summary.get("engine_result") == "DELIVERY_REFUSED"
            and isinstance(delivery_refusal, dict)
            and delivery_refusal.get("engine_result") == "SUBMITTED"
            and delivery_refusal.get("reason_code") in (
                "CHECKS_STALE", "CANDIDATE_NOT_DELIVERED", "CANDIDATE_OUT_OF_SCOPE")):
        return False
    if summary.get("runtime_outcome") not in ("cancelled", "engine_error", "missing_result", "budget_stop",
                                              "recovery_required"):
        return False
    terminal = summary.get("engine_terminal")
    if not isinstance(terminal, dict):
        terminal = {}
    if terminal.get("result") == "SUBMITTED":
        return False
    result = terminal.get("result")
    if isinstance(result, str) and result.startswith("ERR:"):
        return True
    # A crash whose reason is only its own exception text may succeed on resume; replay/checkpoint
    # refusals (their reason is copied from engine.log) would repeat, so they keep the old rule below.
    if (result in ENGINE_FAILURES_WITH_MESSAGE and terminal.get("error_message")
            and terminal.get("error") not in REPEATING_ENGINE_ERRORS):
        return True
    # Everything else keeps the pre-existing rule. A reason on a stopped run (seal failure, or a
    # ReplayUnfaithful/CheckpointUnfaithful copied from engine.log) means the resume would repeat it.
    return not summary.get("engine_error_reason")


def engine_error_reason(summary: dict[str, object]) -> str | None:
    """One readable line for an engine failure, from the engine's own terminal receipt.

    ERR:* outcomes name the exception class and message. Seal/finalization failures keep their own wording.
    ENGINE_ERROR and MUTATION_CONTAINMENT name the class and message only when the child recorded a message.
    """
    terminal = summary.get("engine_terminal") or {}
    result = terminal.get("result")
    if isinstance(result, str) and result.startswith("ERR:"):
        message = str(terminal.get("error_message") or "").strip()
        if not message:
            return f"engine error: {result}"
        error_type = str(terminal.get("error_type") or result[len("ERR:"):])
        return f"engine error: {error_type}: {message}"
    if result in ENGINE_FAILURES_WITH_MESSAGE:
        # A reason already copied from engine.log (ReplayUnfaithful guidance) is the actionable one: keep it.
        existing = summary.get("engine_error_reason")
        if isinstance(existing, str) and existing:
            return existing
        message = str(terminal.get("error_message") or "").strip()
        if not message:
            return None
        error_type = str(terminal.get("error_type") or terminal.get("error") or result)
        return f"engine error: {error_type}: {message}"
    if result == "HARNESS_CONTRACT" and terminal.get("error_type") == "ProviderHTTPError":
        message = str(terminal.get("error_message") or "provider rejected the request").strip()
        return f"provider error: {message}"
    if result not in SEAL_FAILURES:
        return None
    detail = str(terminal.get("error_message") or terminal.get("error") or terminal.get("error_type") or "no detail")
    requests = summary.get("requests")
    target = f"request {int(requests) + 1}" if isinstance(requests, int) else "its next request"
    return f"engine could not seal {target}: {detail} ({terminal['result']})"


def exit_code(summary: dict[str, object]) -> int:
    runtime = summary.get("runtime_outcome")
    if runtime == "cancelled":
        return 130
    if runtime in {"recovery_required", "hook_containment"}:
        return 2
    if runtime == "hook_blocked":
        return 1
    if runtime == "continuation_blocked":
        return 1
    if runtime == "suspended":
        return 1
    if summary.get("apply_error"):
        return 2
    if runtime in ("engine_error", "missing_result", "context_too_small") or runtime is None:
        return 2
    if (runtime == "completed" and summary.get("engine_result") == "SUBMITTED"
            and summary.get("artifact") == "saved"
            and (summary.get("check_outcome") == "pass"
                 or (summary.get("runner_kind") == "generic"
                     and summary.get("check_source") == "none"
                     and summary.get("check_outcome") == "not_run"))
            and (summary.get("child_exit_code") == 0
                 or (summary.get("engine_terminal") or {}).get("reason_code") == RECOVERED_FROM_JOURNAL)):
        return 0
    return 1
