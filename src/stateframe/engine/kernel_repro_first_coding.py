"""Thin coding-pack adapter for the domain-general repro-first gate."""
from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path

from kernel_repro_first import (
    GateDecision,
    ProbeResult,
    ReproFirstGate,
    code_hash,
    requires_repro,
    repro_first_gate_mode,
)


_UNREPRODUCED = re.compile(r"\bUNREPRODUCED\s*\(([^)]*)\)", re.IGNORECASE)


def pytest_probe_result(observation: object, *, command: object = None,
                        probe_id: object = None, code_hash: object = None,
                        exit_code: object = None) -> ProbeResult:
    """Convert the repair runner's legacy pytest text into host evidence."""
    return ProbeResult.from_value(
        observation, command=command, probe_id=probe_id,
        code_hash=code_hash, exit_code=exit_code)


def _ticket_value(ticket: object, *keys: str) -> object:
    if isinstance(ticket, Mapping):
        for key in keys:
            if key in ticket:
                return ticket[key]
    for key in keys:
        value = getattr(ticket, key, None)
        if value is not None:
            return value
    return None


def attach_repro_first_gate(runtime: object, *, ticket: object,
                            root: str | os.PathLike[str] | Path | None = None,
                            initial_probe: object = None,
                            max_attempts: int | None = None,
                            discipline: bool = False) -> ReproFirstGate | None:
    """Attach the gate only in its explicit arm and seed a host preflight probe."""
    if repro_first_gate_mode() != "on" or not requires_repro(ticket):
        return None
    revision = code_hash(Path(root)) if root is not None else None
    raw_attempts = max_attempts
    if raw_attempts is None:
        raw_attempts = _ticket_value(ticket, "repro_max_attempts", "max_repro_attempts")
    if raw_attempts is None:
        raw_attempts = int(os.environ.get("STATEFRAME_REPRO_FIRST_MAX_ATTEMPTS", "3"))
    gate = ReproFirstGate(
        ticket,
        revision=revision,
        max_attempts=int(raw_attempts),
        mode="on",
        journal=getattr(runtime, "journal", None),
        task_id=getattr(runtime, "task_id", None),
    )
    if discipline:
        # The live REPROFIRST adapter is an enforced workflow.  Existing
        # domain-gate callers retain their explicit advisory compatibility.
        gate.edit_permit = "enforce"
        runtime._repro_first_require_witness = True
    runtime.repro_first_gate = gate
    if initial_probe is not None:
        _probe_id = _ticket_value(
            ticket, "repro_probe_id", "repro_selector", "rel", "path", "fn")
        _probe_id = str(_probe_id or "pytest")
        _command = _ticket_value(ticket, "repro_command", "test_command", "test_cmd")
        probe = pytest_probe_result(
            initial_probe,
            command=_command or "pytest:" + _probe_id,
            probe_id=_probe_id,
            code_hash=revision,
        )
        gate.observe_probe(probe)
    return gate


def observe_pytest(gate: ReproFirstGate, observation: object, *,
                   command: object = None, probe_id: object = None,
                   root: str | os.PathLike[str] | Path | None = None,
                   exit_code: object = None) -> GateDecision:
    """Record one coding-pack test action against the active gate."""
    revision = code_hash(Path(root)) if root is not None else None
    return gate.observe_probe(
        pytest_probe_result(observation, command=command, probe_id=probe_id,
                            code_hash=revision, exit_code=exit_code))


def unreproduced_risk(text: object) -> str | None:
    """Extract the explicit ``UNREPRODUCED(risk)`` completion declaration."""
    match = _UNREPRODUCED.search(str(text or ""))
    if not match:
        return None
    risk = match.group(1).strip()
    return risk or None


def file_unreproduced(gate: ReproFirstGate, text: object) -> GateDecision | None:
    """Route a model's explicit partial-outcome declaration to the kernel."""
    risk = unreproduced_risk(text)
    return gate.mark_unreproduced(risk) if risk is not None else None


def render_edit_refusal(decision: GateDecision, *, path: object = None) -> str:
    """Render a typed refusal suitable for the model-facing coding tool."""
    payload = {
        "status": "REJECTED" if not decision.allowed else "ADMITTED",
        "authority": "HOST",
        "code": decision.code,
        "reason": decision.detail,
        "path": str(path) if path is not None else None,
        "alternatives": ([{
            "tool": "run_tests",
            "hint": "Run the exact reproduction command until a red first-error line is captured.",
        }] if not decision.allowed and decision.code in {
            "REPRO_REQUIRED", "REPRO_STALE", "REPRO_PROBE_MISMATCH",
        } else []),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


__all__ = [
    "attach_repro_first_gate", "observe_pytest", "pytest_probe_result",
    "file_unreproduced", "render_edit_refusal", "unreproduced_risk",
]
