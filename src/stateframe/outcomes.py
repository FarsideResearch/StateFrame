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


RECOVERED_FROM_JOURNAL = "recovered_from_journal"


def terminal_receipt(run: Path) -> dict[str, object]:
    try:
        envelope = json.loads((run / "engine-result.json").read_text())
    except (OSError, ValueError):
        return journal_submission(run)
    result = envelope.get("result") if isinstance(envelope, dict) else None
    if not isinstance(result, dict) or not isinstance(result.get("result"), str):
        return journal_submission(run)
    return result


def journal_submission(run: Path) -> dict[str, object]:
    """A stored DoneSeal from the engine journal, for an engine stopped before its terminal receipt.

    In one long run the engine sealed and stored its submission inside its wall, then was killed while
    finishing up, so engine-result.json never appeared. Recovery requires a stored done_sealed and a
    matching done_submission_stored; the export still re-verifies the artifact bytes against the digest.
    """
    journals = sorted((run / "worker" / "measurement-runs").glob("*/events.jsonl"))
    if len(journals) != 1:
        return {}
    sealed: dict[str, object] | None = None
    stored: dict[str, object] | None = None
    try:
        lines = journals[0].read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return {}
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict) or event.get("stored") is not True:
            continue
        if event.get("kind") == "done_sealed":
            sealed, stored = event, None
        elif event.get("kind") == "done_submission_stored" and sealed is not None:
            stored = event
    if sealed is None or stored is None:
        return {}
    digest, seal_id = sealed.get("artifact_digest"), sealed.get("seal_id")
    if stored.get("artifact_digest") != digest:
        return {}
    submission = {"stored": True, "seal_id": seal_id, "artifact_digest": digest}
    terminal = {"result": "SUBMITTED", "reason_code": RECOVERED_FROM_JOURNAL, "submission": submission}
    return terminal if sealed_artifact_digest(terminal) else {}


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
    elif cancelled or child in (130, -2) or status in ("CANCELLED", "CANCELED"):
        runtime = "cancelled"
    elif recovered:
        # The engine sealed and stored its submission before it was stopped: that is a submission.
        runtime = "completed"
    elif timed_out:
        runtime = "budget_stop"
    elif (child != 0 or summary.get("candidate_error") or status == "ENGINE_ERROR" or status in SEAL_FAILURES
          or (isinstance(status, str) and status.startswith("ERR:"))):
        runtime = "engine_error"
    elif status in BUDGET_STOPS:
        runtime = "budget_stop"
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
    check = ("pass" if after.get("passed") is True and after.get("exit_code") == 0 else "fail") \
        if bound else "not_run"
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
    if status == "TEST_REPORTED_WRONG" and summary.get("wrong_test_evidence"):
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


def engine_error_reason(summary: dict[str, object]) -> str | None:
    """One readable line for a seal/finalization failure, from the engine's own terminal receipt."""
    terminal = summary.get("engine_terminal") or {}
    if terminal.get("result") not in SEAL_FAILURES:
        return None
    detail = str(terminal.get("error_message") or terminal.get("error") or terminal.get("error_type") or "no detail")
    requests = summary.get("requests")
    target = f"request {int(requests) + 1}" if isinstance(requests, int) else "its next request"
    return f"engine could not seal {target}: {detail} ({terminal['result']})"


def exit_code(summary: dict[str, object]) -> int:
    runtime = summary.get("runtime_outcome")
    if runtime == "cancelled":
        return 130
    if runtime == "hook_containment":
        return 2
    if runtime == "hook_blocked":
        return 1
    if summary.get("apply_error"):
        return 2
    if runtime in ("engine_error", "missing_result") or runtime is None:
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
