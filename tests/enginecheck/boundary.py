"""Boundary-value file contents for model-free engine runs, and the survival check every boundary run must pass.

The engine must neither crash, nor end outside a clean completion or refusal, nor alter a file it was only asked to
read. Incidents this guards: an empty file crashed the engine for several releases, CRLF/BOM content was corrupted,
binary was read lossily, and an exact-fit or limit+1 size broke the fitter.
"""
from __future__ import annotations

import json
from pathlib import Path

from enginecheck import Need, Scenario, Trace

# Exit codes a model-free run may end with. 0: completed (src/stateframe/outcomes.py:284-292). 2: reported error
# (outcomes.py:280-282; cli.py:561-566 _report_error). 1 is a clean refusal only when the summary names one of
# _REFUSAL_OUTCOMES (outcomes.py:274-279); the fallthrough at outcomes.py:293 also returns 1 and is flagged.
# Any other code, including signal-style codes, is a crash for this check.
CLEAN_EXIT_CODES = frozenset({0, 2})
_REFUSAL_EXIT_CODE = 1
_REFUSAL_OUTCOMES = frozenset({"hook_blocked", "continuation_blocked", "suspended"})

_TRACEBACK = "Traceback (most recent call last)"
# The engine refuses a read-only task without a named output file ("could not pick a starting file; name an output
# path in --task"), so every boundary repo carries one. The script never writes to it, so its bytes are still checked.
_OUTPUT_NAME = "notes.md"
_OUTPUT_CONTENT = b"# notes\n"

BOUNDARY_CONTENTS: dict[str, bytes] = {
    "empty": b"",
    "one_byte": b"x",
    "newline_only": b"\n",
    "no_trailing_newline": b"a = 1",
    "crlf": b"a = 1\r\nb = 2\r\n",
    "utf8_bom": b"\xef\xbb\xbfa = 1\n",
    "non_ascii": "s = \"é漢\"\n".encode("utf-8"),
    "invalid_utf8": b"\xff\xfe\x00a\n",
    "binary_nul": b"\x00\x01\x02" * 64,
    "long_line": b"y" * 20000 + b"\n",
    "reserved_name": (
        b"def description():\n    return \"d\"\n\n\n"
        b"class Schema:\n    type = \"object\"\n\n\n"
        b"DATA = {\"type\": \"string\", \"description\": \"d\"}\n"
    ),
}

_SENT_A_REQUEST = Need("the engine sent at least one provider request", lambda trace: len(trace.requests) > 0)


def limit_values(limit: int) -> tuple[int, ...]:
    """Sizes around a limit: 0, 1, limit-1, limit, limit+1. Duplicates and negatives drop out, the rest sorted."""
    candidates = {0, 1, limit - 1, limit, limit + 1}
    return tuple(sorted(value for value in candidates if value >= 0))


def read_scenario(name: str, content: bytes, *, path: str = "target.py",
                  extra_files: dict[str, bytes] | None = None) -> Scenario:
    """A repo holding `path`, the named output file, and extra_files; a script that reads `path` once, then finishes."""
    extras = extra_files or {}
    for reserved in (path, _OUTPUT_NAME):
        if reserved in extras:
            raise ValueError(f"extra_files must not repeat the fixture path {reserved!r}")
    steps = [
        {"tool": "read", "required": True, "arguments": {"path": path}},
        {"tool": "done", "required": True, "arguments": {"summary": f"Read {path}."}},
    ]
    return Scenario(
        files={path: content, _OUTPUT_NAME: _OUTPUT_CONTENT, **extras},
        task=f"Read {path}, then update {_OUTPUT_NAME}. Boundary case: {name}.",
        steps=steps,
        needs=[_SENT_A_REQUEST],
    )


def _has_traceback(run_dir: Path) -> bool:
    """True if any run artifact outside wire/ carries a Python traceback. Wire is excluded because tool results
    sent to the model may legitimately quote tracebacks."""
    for artifact in run_dir.rglob("*"):
        if not artifact.is_file() or "wire" in artifact.relative_to(run_dir).parts:
            continue
        if _TRACEBACK.encode() in artifact.read_bytes():
            return True
    return False


def _refusal_problems(run_dir: Path) -> list[str]:
    """Exit 1 is clean only when summary.json names a refusal outcome and records no error."""
    try:
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ["engine crashed: exit code 1 with no readable summary.json"]
    if not isinstance(summary, dict):
        return ["engine crashed: exit code 1 with a non-object summary.json"]
    problems: list[str] = []
    outcome = summary.get("runtime_outcome")
    if outcome not in _REFUSAL_OUTCOMES:
        problems.append(f"engine crashed: exit code 1 with runtime_outcome {outcome!r}, not a refusal")
    if summary.get("error"):
        problems.append("engine crashed: exit code 1 with a summary error")
    return problems


def check_survives(trace: Trace, scenario: Scenario) -> list[str]:
    """Problems found in a read-only session. An empty list means the engine survived."""
    problems: list[str] = []
    if trace.exit_code == _REFUSAL_EXIT_CODE:
        problems.extend(_refusal_problems(trace.run_dir))
    elif trace.exit_code not in CLEAN_EXIT_CODES:
        problems.append(f"engine crashed: exit code {trace.exit_code} is not a completion or clean refusal")
    if _TRACEBACK in trace.output or _has_traceback(trace.run_dir):
        problems.append("engine crashed: Python traceback in engine output or run artifacts")
    for name, expected in sorted(scenario.files.items()):
        target = trace.repo / name
        try:
            actual = target.read_bytes()
        except OSError:
            problems.append(f"{name}: missing after read-only session")
            continue
        if actual != expected:
            problems.append(f"{name}: bytes changed by a read-only session")
    if not trace.requests:
        problems.append("no provider requests were sent")
    return problems


def assert_survives(trace: Trace, scenario: Scenario) -> None:
    problems = check_survives(trace, scenario)
    if problems:
        raise AssertionError("engine did not survive boundary run: " + "; ".join(problems))
