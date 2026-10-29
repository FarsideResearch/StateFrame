"""Runtime rail for model-authored reproduce-before-fix checks.

The domain gate owns revision identity and red-to-green semantics.  This module
owns the coding adapter concerns: the menu action, scratch lifecycle, bounded
attempts, and the shared evidence-pin representation.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from kernel_clause_discriminator import SCHEMA as REQWITNESS_SCHEMA
from kernel_clause_discriminator import evidence_pin
from kernel_repro_first import GateDecision, ReproFirstGate, code_hash
from kernel_repro_first_coding import observe_pytest


SCRATCH_DIR = ".repro-first-scratch"
MAX_SOURCE_CHARS = 16_000
SCHEMA = "stateframe-repro-first-runtime/1"
_FORBIDDEN_SOURCE = re.compile(
    r"(?:grader|hidden|gold|private|backup|backups|a3b|production\s+database)",
    re.IGNORECASE,
)
_EXIT_CODE = re.compile(r"(?:pytest\s+exit|exit(?:_code)?)\s*[:=]?\s*(-?\d+)", re.IGNORECASE)


def active(runtime: object) -> bool:
    gate = getattr(runtime, "repro_first_gate", None)
    from kernel_repro_first import repro_first_gate_mode
    return (repro_first_gate_mode() == "on"
            and isinstance(gate, ReproFirstGate) and gate.active)


def tool_schema() -> dict[str, object]:
    """Return the one registered action used for all reproduction attempts."""
    return {
        "type": "function",
        "function": {
            "name": "reproduce",
            "description": (
                "Write and run one small ticket-derived check in a temporary "
                "repro-first scratch path. It must fail on the current code before "
                "the first product edit, then pass unchanged after the edit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Complete Python check source; never a public-test replacement.",
                    },
                    "test_name": {
                        "type": "string",
                        "description": "Stable logical check name used for the witness identity.",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Short ticket-grounded reason this check exercises the report.",
                    },
                    "why": {"type": "string"},
                },
                "required": ["source"],
                "additionalProperties": False,
            },
        },
    }


def add_menu_tool(tools: object, runtime: object) -> list[dict[str, object]]:
    """Add reproduce before completion/menu projections can hide recoverable work."""
    if not isinstance(tools, list):
        return list(tools) if isinstance(tools, Sequence) else []
    result = list(tools)
    if active(runtime) and not any(
            isinstance(tool, Mapping)
            and isinstance(tool.get("function"), Mapping)
            and tool["function"].get("name") == "reproduce"
            for tool in result):
        result.append(tool_schema())
    return result


def is_scratch_path(path: object) -> bool:
    """Identify transient paths so candidate exports can exclude them defensively."""
    normalized = str(path or "").replace("\\", "/").strip("/")
    parts = tuple(part for part in normalized.split("/") if part)
    return SCRATCH_DIR in parts


def exclude_scratch_paths(paths: object) -> tuple[object, ...]:
    if not isinstance(paths, Sequence) or isinstance(paths, (str, bytes, bytearray)):
        return ()
    return tuple(path for path in paths if not is_scratch_path(path))


def _validate_source(source: object) -> tuple[str | None, str | None]:
    if not isinstance(source, str) or not source.strip():
        return None, "source is required"
    source = source.strip()
    if len(source) > MAX_SOURCE_CHARS:
        return None, f"source exceeds the {MAX_SOURCE_CHARS}-character scratch limit"
    if _FORBIDDEN_SOURCE.search(source):
        return None, "source references forbidden grader, hidden, gold, backup, or production material"
    try:
        tree = ast.parse(source, filename="repro_first_check.py")
    except SyntaxError as error:
        return None, f"source does not parse: {error.msg}"
    has_check = any(
        isinstance(node, ast.Assert)
        or (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_"))
        for node in ast.walk(tree)
    )
    if not has_check:
        return None, "source must contain an assert or a test_* function"
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"write_text", "write_bytes", "unlink", "remove", "rmtree"}:
                return None, "scratch checks cannot mutate files"
        if isinstance(node, ast.ImportFrom) and str(node.module or "").split(".", 1)[0] in {
            "subprocess", "socket", "requests", "boto3",
        }:
            return None, "scratch checks cannot perform external I/O"
    return source + "\n", None


def _attempt_result(runtime: object, *, status: str, code: str, detail: str,
                    revision: object = None, observed: object = None,
                    scratch_path: str | None = None) -> dict[str, object]:
    gate = getattr(runtime, "repro_first_gate", None)
    attempts = int(getattr(gate, "attempts", 0) or 0)
    receipt = getattr(gate, "receipt", None)
    receipt_id = getattr(receipt, "receipt_id", None)
    observed_payload = observed if isinstance(observed, Mapping) else {"value": str(observed or "")}
    latest_probe = getattr(gate, "_latest_probe", None)
    observed_outcome = (
        observed_payload.get("outcome")
        if isinstance(observed_payload, Mapping)
        else None
    ) or getattr(latest_probe, "outcome", None)
    reached_observation = (
        observed_payload.get("reached_observation")
        if isinstance(observed_payload, Mapping)
        else None
    )
    if not isinstance(reached_observation, bool):
        reached_observation = getattr(latest_probe, "reached_observation", None)
    failure_phase = (
        observed_payload.get("failure_phase")
        if isinstance(observed_payload, Mapping)
        else None
    ) or getattr(latest_probe, "failure_phase", None)
    pin, pin_text = evidence_pin(
        receipt_id=str(receipt_id or f"repro-attempt-{attempts}"),
        revision=revision,
        observed=observed_payload,
        expectation_origin="ticket_reproduction",
        scope={"kind": "repro_first", "attempt": attempts},
    )
    result = {
        "schema": SCHEMA,
        "status": status,
        "code": code,
        "detail": detail,
        "attempt": attempts,
        "revision": revision,
        "receipt_id": receipt_id,
        "scratch_path": scratch_path,
        "evidence_pin": pin,
        "evidence_pin_text": pin_text,
        "outcome": observed_outcome or "UNKNOWN",
        "reached_observation": reached_observation,
        "failure_phase": failure_phase or "UNKNOWN",
        "reproduced": (
            code == "RED_CAPTURED"
            and observed_outcome == "BEHAVIOR_FAIL"
            and reached_observation is True
        ),
    }
    records = getattr(runtime, "_repro_first_receipts", None)
    if not isinstance(records, list):
        records = []
        setattr(runtime, "_repro_first_receipts", records)
    records.append(dict(result))
    emitter = getattr(getattr(runtime, "journal", None), "emit", None)
    if callable(emitter):
        # Deliver through the existing REQWITNESS receipt rail; REPROFIRST is
        # a new witness producer, not a second evidence transport.
        emitter("clause_discriminator_receipt", producer="kernel_clause_discriminator",
                event_schema=REQWITNESS_SCHEMA, clause_ref="repro-first",
                code=code, result=status, certifies_clause=status == "GREEN",
                receipt_id=receipt_id, evidence_pin=pin, dispatch=False)
    return result


def _exit_code(value: object) -> int | None:
    if isinstance(value, Mapping):
        raw = value.get("exit_code", value.get("exit"))
        if type(raw) is int:
            return raw
        value = value.get("observation", value.get("output", ""))
    match = _EXIT_CODE.search(str(value or ""))
    return int(match.group(1)) if match else None


def execute_reproduction(
        runtime: object, *, root: Path, arguments: Mapping[str, object],
        runner: Callable[[str, str], object]) -> dict[str, object]:
    """Run one model-authored scratch check through the caller's sandbox runner."""
    gate = getattr(runtime, "repro_first_gate", None)
    if not active(runtime) or not isinstance(gate, ReproFirstGate):
        return {"schema": SCHEMA, "status": "REJECTED", "code": "NOT_APPLICABLE",
                "detail": "reproduce is only available for an active bug-fix ticket"}
    action_attempt = int(getattr(runtime, "_repro_first_action_attempts", 0) or 0) + 1
    setattr(runtime, "_repro_first_action_attempts", action_attempt)
    if action_attempt > gate.max_attempts or gate.attempts >= gate.max_attempts:
        return _attempt_result(
            runtime, status="REJECTED", code="UNREPRODUCED_REQUIRED",
            detail="bounded reproduction attempts are exhausted; file UNREPRODUCED(risk)",
            revision=code_hash(root))
    source, error = _validate_source(arguments.get("source"))
    if error is not None or source is None:
        return _attempt_result(runtime, status="REJECTED", code="REPRO_SOURCE_INVALID",
                               detail=error or "invalid source", revision=code_hash(root))
    source_digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    prior_digest = getattr(runtime, "_repro_first_source_digest", None)
    if gate.receipt is not None and prior_digest != source_digest:
        return _attempt_result(
            runtime, status="REJECTED", code="REPRO_PROBE_MISMATCH",
            detail="rerun the exact scratch source that produced the red receipt",
            revision=code_hash(root))
    test_name = str(arguments.get("test_name") or "ticket_reproduction").strip()
    probe_id = "repro-first:" + source_digest
    command = "pytest:" + SCRATCH_DIR + "/reproduction.py"
    scratch_path = f"{SCRATCH_DIR}/check_{uuid.uuid4().hex}.py"
    before_revision = code_hash(root)
    try:
        outcome = runner(source, scratch_path)
    except Exception as error:  # runner errors are environment observations, never repros
        outcome = {
            "observation": f"{type(error).__name__}: {error}",
            "status": "ENVIRONMENT",
            "outcome": "ENVIRONMENT",
            "failure_phase": "EXECUTION",
            "reached_observation": False,
        }
    # Preserve typed runner failures through the coding adapter.  Reducing a
    # mapping to its display string would erase ENVIRONMENT/phase/reached facts
    # and could make an execution failure look like a reproducing probe.
    observation = outcome if isinstance(outcome, Mapping) else outcome
    exit_code = _exit_code(outcome)
    setattr(runtime, "_repro_first_source_digest", source_digest)
    setattr(runtime, "_repro_first_probe_id", probe_id)
    decision = observe_pytest(
        gate, observation, command=command, probe_id=probe_id,
        root=root, exit_code=exit_code)
    result = _attempt_result(
        runtime, status=decision.status, code=decision.code, detail=decision.detail,
        revision=code_hash(root), observed={
            "status": decision.status,
            "code": decision.code,
            "exit_code": exit_code,
            "before_revision": before_revision,
            "first_error_line": getattr(gate.receipt, "first_error_line", None),
        }, scratch_path=scratch_path)
    result["test_name"] = test_name
    result["probe_id"] = probe_id
    result["command"] = command
    return result


def complete(runtime: object, *, root: Path, public_suite_green: bool) -> GateDecision:
    """Apply strict readiness: witness flip, then a current green public suite."""
    gate = getattr(runtime, "repro_first_gate", None)
    if not active(runtime) or not isinstance(gate, ReproFirstGate):
        return GateDecision(True, "INACTIVE", "NOT_APPLICABLE")
    if gate.receipt is None and getattr(gate, "_unreproduced_risk", None) is None:
        return GateDecision(
            False, "REJECTED", "REPRO_REQUIRED",
            "completion requires a scratch reproduction that failed before editing",
        )
    if gate.receipt is None and getattr(gate, "_unreproduced_risk", None) is not None:
        return GateDecision(
            True, "PARTIAL", "UNREPRODUCED",
            "bounded reproduction attempts were exhausted; completion is disclosed as partial",
        )
    decision = gate.complete(code_hash=code_hash(root))
    if not decision.accepted:
        return decision
    if decision.status == "PARTIAL":
        return decision
    if not public_suite_green:
        return GateDecision(
            False, "REJECTED", "REPRO_PUBLIC_SUITE_REQUIRED",
            "the reproduction passed, but the public suite is not green at the current revision",
            decision.receipt,
        )
    return decision


def select_product_fix(
        runtime: object, *, root: Path, public_checks: Sequence[object],
        candidate_provider: Callable[[object], object],
        check_runner: Callable[[Path, object], object],
        initial_candidate: object | None = None, budget: object | None = None,
        context: object = "", seed: int = 0, max_candidates: int = 3,
        allowance: float = 0.20, apply_winner: Callable[[Path, object], object] | None = None,
) -> object:
    """Select a reproduced ticket fix through BESTOFN's disposable-check rail.

    Keeping this adapter beside REPROFIRST makes the product-edit boundary
    explicit: callers cannot accidentally enable candidate selection without a
    typed ticket and the gate's current reproduction receipt.
    """
    gate = getattr(runtime, "repro_first_gate", None)
    from kernel_bestofn import select_best_candidate
    return select_best_candidate(
        root,
        ticket=getattr(gate, "ticket", getattr(runtime, "ticket", None)),
        repro=getattr(gate, "receipt", None),
        public_checks=public_checks,
        candidate_provider=candidate_provider,
        check_runner=check_runner,
        initial_candidate=initial_candidate,
        budget=budget,
        journal=getattr(runtime, "journal", None),
        context=context,
        seed=seed,
        max_candidates=max_candidates,
        allowance=allowance,
        apply_winner=apply_winner,
    )


def render_result(result: Mapping[str, object]) -> str:
    return json.dumps(dict(result), ensure_ascii=False, sort_keys=True)


__all__ = [
    "MAX_SOURCE_CHARS", "SCHEMA", "SCRATCH_DIR", "active", "add_menu_tool",
    "complete", "exclude_scratch_paths", "execute_reproduction", "is_scratch_path",
    "render_result", "select_product_fix", "tool_schema",
]
