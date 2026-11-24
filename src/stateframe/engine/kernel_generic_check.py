"""Whole-command public checks for repositories without pytest selectors."""
from __future__ import annotations

import hashlib
import sys
from collections.abc import Mapping
from pathlib import Path

from kernel_execution_policy import (
    check_cpu_backstop_seconds,
    policy_memory_limit_mechanism,
    policy_read_grants,
    run_sandboxed,
)
from kernel_check_watchdog import (IncompleteCheck, check_watchdog_config, execution_stop_outcome,
                                   validate_execution_stop)
from kernel_pilot import ACTIVE_RUNTIME


CHECK_ID = "@check"
OUTPUT_TAIL = 4000
OUTPUT_LIMIT = 256 * 1024


def result_class(code: int | None) -> str:
    """Classify a whole-command result without pytest's selector exit meanings."""
    if code == 0:
        return "GREEN"
    if code is None or code < 0 or code in {124, 125, 126, 127} or code >= 128:
        return "HARNESS_ERROR"
    return "RED"


def _protected_state(root: Path, row: Mapping[str, object]) -> dict[str, str | None]:
    state = {}
    for relative in row.get("protected_test_paths", ()):
        if not isinstance(relative, str):
            continue
        path = root / relative
        state[relative] = (hashlib.sha256(path.read_bytes()).hexdigest()
                           if path.is_file() and not path.is_symlink() else None)
    return state


def _check_tool_option() -> dict[str, str]:
    """The host-recorded install root, held outside the row so no rendering can reach it."""
    tools = sys.modules.get("kernel_check_tool_root")
    root = tools.recorded_root() if tools is not None else None
    return {"check_tool_root": root} if root is not None else {}


def check_result(root: Path, row: Mapping[str, object]) -> tuple[str, int]:
    """Execute the host-declared argv in the existing sandbox and publish its receipt."""
    raw = row.get("test_command")
    if not isinstance(raw, list) or not raw or any(not isinstance(x, str) or not x for x in raw):
        return "check refused: missing host-declared command", 127
    command = tuple(raw)
    runtime = ACTIVE_RUNTIME.get()
    binding = runtime.binding() if runtime is not None else {}
    try:
        before = _protected_state(root, row)
    except OSError as error:
        return f"check refused: protected resource could not be read: {error}", 125
    if any(value is None for value in before.values()):
        return "check refused: a protected test resource is absent or unsafe", 125
    try:
        result = run_sandboxed(
            command, root, timeout=None, check_watchdog=check_watchdog_config(),
            cpu_seconds=check_cpu_backstop_seconds(),
            output_limit=OUTPUT_LIMIT, generic_check=True,
            **_check_tool_option())
    except (RuntimeError, OSError, ValueError) as error:
        detail = f"check refused: {type(error).__name__}: {error}"
        _record(runtime, binding, command, 127, detail, outcome="ENVIRONMENT")
        return detail, 127
    memory_limit_mechanism = policy_memory_limit_mechanism(result)
    memory_limit_killed = bool(getattr(result, "memory_limit_killed", False))
    output = (result.completed.stdout + result.completed.stderr)[-OUTPUT_LIMIT:]
    from kernel_check_receipts import sandbox_exec_denial, sandbox_open_denial
    denial = sandbox_open_denial(result.completed.stderr + "\n" + result.completed.stdout,
                                 policy_read_grants(result))
    # The sandbox's own launch failure ("bwrap: execvp <path>: ...") arrives on stderr
    # with an ordinary exit (bwrap exits 1): no check ran, so it is never a RED check.
    exec_denial = (sandbox_exec_denial(result.completed.stderr)
                   if result.completed.returncode != 0 else None)
    execution_stop = validate_execution_stop(getattr(result, "execution_stop", None))
    if execution_stop is not None:
        code = result.completed.returncode
        outcome = execution_stop_outcome(execution_stop)
        label = "check incomplete: " + str(execution_stop["cause"]).replace("_", " ")
    elif memory_limit_killed:
        code, outcome, label = result.completed.returncode, "RESOURCE_LIMIT", "check stopped: memory limit"
    elif result.timed_out:
        code, outcome, label = 124, "RESOURCE_LIMIT", "check timed out"
    elif denial is not None:
        code, outcome, label = 126, "ENVIRONMENT", denial
    elif exec_denial is not None:
        code, outcome, label = 126, "ENVIRONMENT", f"check command could not run: {exec_denial}"
    else:
        code = result.completed.returncode
        if code == 0:
            outcome, label = "PASS", "check passed"
        elif code in {126, 127}:
            outcome, label = "ENVIRONMENT", "check command could not run"
        else:
            outcome, label = "BEHAVIOR_FAIL", "check failed"
    try:
        after = _protected_state(root, row)
    except OSError:
        after = {}
    if before != after and execution_stop is None:
        code, outcome, label = 125, "INVALID_CHECK", "protected test resource changed"
    tail = output[-OUTPUT_TAIL:]
    _record(runtime, binding, command, code, output, outcome=outcome, tail=tail,
            output_truncated=result.output_truncated,
            memory_limit_mechanism=memory_limit_mechanism,
            cgroup_leaf_pattern=getattr(result, "cgroup_leaf_pattern", None),
            memory_limit_killed=memory_limit_killed, execution_stop=execution_stop)
    if execution_stop is not None:
        return IncompleteCheck(label, execution_stop, exit_code=code), code if code else 124
    return f"check exit {code}: {label}\n{tail}", code


def _record(runtime: object | None, binding: object, command: tuple[str, ...],
            code: int, output: str, *, outcome: str,
            tail: str | None = None, output_truncated: bool | None = None,
            memory_limit_mechanism: str | None = None,
            cgroup_leaf_pattern: str | None = None,
            memory_limit_killed: bool = False,
            execution_stop: Mapping[str, object] | None = None) -> None:
    if runtime is None:
        return
    if not isinstance(binding, dict):
        binding = {}
    typed = {
        "outcome": outcome,
        "status": "PASS" if outcome == "PASS" else "FAIL",
        "reached_observation": outcome in {"PASS", "BEHAVIOR_FAIL"},
        "failure_phase": ("NONE" if outcome == "PASS" else "TIMEOUT" if code == 124
                          else "HOST" if code in {126, 127} else "COLLECTION"
                          if code == 125 else "EXECUTION"),
        "assertion_context": tail if tail is not None else output[-OUTPUT_TAIL:],
        "execution_complete": outcome in {"PASS", "BEHAVIOR_FAIL"},
        "counts": {"collected": 0 if code in {124, 125, 126, 127} else 1,
                   "passed": 1 if code == 0 else 0,
                   "failed": 1 if outcome == "BEHAVIOR_FAIL" else 0},
    }
    if execution_stop is not None:
        typed.update(execution_stop=execution_stop, termination_reason=execution_stop["cause"],
                     status="UNKNOWN", execution_complete=False, reached_observation=False,
                     counts={})
    runtime.test_observation(
        [CHECK_ID], code, binding, observation=output,
        command=command,
        counts=typed["counts"], typed_observation=typed,
        output_truncated=output_truncated,
        memory_limit_mechanism=memory_limit_mechanism,
        cgroup_leaf_pattern=cgroup_leaf_pattern,
        memory_limit_killed=memory_limit_killed,
        **({"termination_evidence": {"execution_stop": execution_stop}}
           if execution_stop is not None else {}))
