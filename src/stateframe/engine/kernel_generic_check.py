"""Whole-command public checks for repositories without pytest selectors."""
from __future__ import annotations

import hashlib
import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
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
_CAUSE_DELIMITER = "The above exception was the direct cause of the following exception:"
_TRACEBACK_HEADER = "Traceback (most recent call last):"
_VENDOR_DIRS = {"vendor", "_vendor", "vendors", "vendored", "third_party", "third-party"}
_LOG_LEVELS = {"DEBUG", "INFO", "NOTICE", "WARNING", "WARN", "ERROR", "CRITICAL", "FATAL"}
_TERMINAL_START = re.compile(r"(?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*(?=:|[ \t]*$)")


@dataclass(frozen=True)
class CauseFrame:
    exception: str
    path: str
    function: str
    printed_path: str
    line: int

    @property
    def location(self) -> str:
        return f"{self.path}:{self.line}"

    def visible_in(self, text: str) -> bool:
        terminal = rf"(?m)^{re.escape(self.exception)}(?=[:\s]|$)"
        paths = rf'{re.escape(self.printed_path)}|(?:[^"\r\n]*/)?{re.escape(self.path)}'
        frame = rf'(?m)^[ \t]*File "(?:{paths})", line {self.line}(?=[,\s]|$)'
        return bool(re.search(terminal, text) and (
            re.search(frame, text) or re.search(rf"{re.escape(self.location)}(?=[:\s]|$)", text)))


@dataclass(frozen=True)
class CheckCauseExcerpt:
    observation: str
    frames: tuple[CauseFrame, ...]
    source_line_ends: frozenset[int]
    execution_id: str
    revision: str
    diagnostic_starts_midline: bool


def cause_excerpt_enabled() -> bool:
    return os.environ.get("STATEFRAME_EXCEPTION_CAUSE_EXCERPT", "off").strip().lower() == "on"


def _cause_chain(root: Path, stdout: str, stderr: str,
                 binding: Mapping[str, object]) -> tuple[tuple[CauseFrame, ...], frozenset[int]] | None:
    streams = (stdout, stderr)
    candidates = [stream for stream in streams if _CAUSE_DELIMITER in stream]
    if len(candidates) != 1 or sum(_TRACEBACK_HEADER in stream for stream in streams) != 1:
        return None
    stream = candidates[0]
    count = stream.count(_TRACEBACK_HEADER)
    if count not in (2, 3) or stream.count(_CAUSE_DELIMITER) != count - 1:
        return None

    import tracebacks

    paths = {path for group in (binding.get("product_files"), binding.get("harness_files"))
             if isinstance(group, Mapping) for path in group}
    vendored = [path for path in paths if _VENDOR_DIRS.intersection(path.split("/")[:-1])]
    repo_index = tracebacks.repo_index_from_paths(paths, repo_roots=(str(root),), vendored=vendored)
    try:
        evidence = tracebacks.ingest(tracebacks.TracebackInput(
            text=stream, repo_index=repo_index, max_input_bytes=OUTPUT_LIMIT, max_tracebacks=3))
    except ValueError:  # An unrepresentable printed line number is unusable evidence.
        return None
    records = evidence.tracebacks
    expected = tuple(tracebacks.ChainLink(first.traceback_id, second.traceback_id,
                                          "direct_cause", True)
                     for first, second in zip(records, records[1:]))
    if (evidence.parse_status != "complete" or evidence.diagnostics or len(records) != count
            or len(evidence.chains) != 1 or evidence.links != expected):
        return None
    lines = stream.splitlines()
    newlines = stream.count("\n")
    # The traceback parser uses splitlines(); only map its line numbers when
    # they agree with the LF boundaries used by the reply fitter.
    can_map_lines = newlines == len(lines) - int(not stream.endswith("\n"))
    following_newlines = stderr.count("\n") if stream == stdout else 0
    source_line_ends = set()
    frames = []
    for index, record in enumerate(records):
        exception = record.exception
        frame = record.deepest_actionable_in_repo_frame
        terminal = lines[exception.terminal_input_line - 1]
        if (record.group_path or exception.class_leaf in {"ExceptionGroup", "BaseExceptionGroup"}
                or exception.qualified_class in _LOG_LEVELS
                or terminal.partition(":")[0].rstrip() != exception.qualified_class
                or frame is None or not frame.repo_path.isprintable()
                or record.frames[0].input_line != record.input_start_line + 1
                or any(f.resolution == "ambiguous" for f in record.frames)):
            return None
        if index + 1 < count:
            # A link inferred across intervening log text may join two incidents.
            between = lines[exception.terminal_input_line:records[index + 1].input_start_line - 1]
            if [line.strip() for line in between if line.strip()] != [_CAUSE_DELIMITER]:
                return None
        elif any(_TERMINAL_START.match(line) for line in lines[exception.terminal_input_line:]):
            # The parser can stop at the first class-shaped line, including a log entry.
            return None
        if can_map_lines:
            for entry in record.frames:
                # The next line's zero-based index is the frame's one-based line number.
                source_index = entry.input_line
                source = entry.source_text
                if (source and source_index < exception.terminal_input_line - 1
                        and lines[source_index].strip() == source):
                    source_line_ends.add(newlines + following_newlines - source_index - 1)
        function = frame.function or ""
        if not re.fullmatch(r"(?:[\w.]+|<[\w ]+>)", function):
            function = ""
        frames.append(CauseFrame(exception.qualified_class, frame.repo_path,
                                 function, frame.file_raw, frame.line))
    return tuple(frames), frozenset(source_line_ends)


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


def check_result(root: Path, row: Mapping[str, object], *,
                 cause_excerpt: list[CheckCauseExcerpt] | None = None) -> tuple[str, int]:
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
    observation = f"check exit {code}: {label}\n{tail}"
    project_cause = (cause_excerpt is not None and runtime is not None
                     and outcome == "BEHAVIOR_FAIL" and result_class(code) == "RED"
                     and not result.output_truncated)
    receipt_count = len(runtime.test_receipts) if project_cause else 0
    _record(runtime, binding, command, code, output, outcome=outcome, tail=tail,
            output_truncated=result.output_truncated,
            memory_limit_mechanism=memory_limit_mechanism,
            cgroup_leaf_pattern=getattr(result, "cgroup_leaf_pattern", None),
            memory_limit_killed=memory_limit_killed, execution_stop=execution_stop)
    if execution_stop is not None:
        return IncompleteCheck(label, execution_stop, exit_code=code), code if code else 124
    if project_cause and len(runtime.test_receipts) == receipt_count + 1:
        receipt = runtime.test_receipts[-1]
        revision = binding.get("product_revision")
        execution_ids = receipt.get("canonical_check_execution_ids", ())
        if (receipt.get("binding_stable") and revision
                and receipt["after_binding"].get("product_revision") == revision
                and len(execution_ids) == 1 and execution_ids[0]):
            chain = _cause_chain(root, result.completed.stdout, result.completed.stderr, binding)
            if chain is not None:
                frames, source_line_ends = chain
                tail_start = len(output) - len(tail)
                starts_midline = tail_start > 0 and output[tail_start - 1] != "\n"
                cause_excerpt.append(CheckCauseExcerpt(
                    observation, frames, source_line_ends, execution_ids[0], revision, starts_midline))
    return observation, code


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
