"""Resume equivalence for model-free engine runs.

A run interrupted after turn k and resumed in a fresh process must send, as its first provider request, the request
an uninterrupted run sent at turn k+1. Differences fail unless their JSON pointer is declared volatile by the caller.

Prior art: tests/test_checkpoint_b_resume_fidelity.py:83 (holds a scripted reply open to interrupt the run) and :428
(resume in a new process). The engine's clean stop is SIGTERM: src/stateframe/runner.py:3907 Terminated ("stop like
Ctrl-C ... so the run can be resumed"). An HTTP refusal instead stops the engine with HARNESS_CONTRACT, which resume
refuses. Replay serves turns 1..k from the tape on resume (src/stateframe/replay_tape.py:403 Replayer), so the
resumed endpoint only answers turns after k.
"""
from __future__ import annotations

import json
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Sequence

from enginecheck import (
    HarnessError,
    Scenario,
    _engine_environment,
    _post_run_failure,
    _request_paths,
    _write_repo,
    run,
)
from engine_run import run_monitored
from stateframe.scripted_endpoint import Script, ScriptStepUnavailable, serve

_ENGINE_TIMEOUT_SECONDS = 600
_HOLD_SECONDS = 300
_PENDING_WAIT_SECONDS = 120
_VALUE_CHARS = 200
_MASK = "<masked>"
_ABSENT = object()
_MISSING = object()


class ResumeDivergence(AssertionError):
    """A resumed run did not send the request the uninterrupted run sent at the same turn."""


class _HoldAfterTurns(Script):
    """Answers the first `turns` requests, then holds turn turns+1 open until the driver has stopped the engine."""

    def __init__(self, document: dict, turns: int) -> None:
        super().__init__(document)
        self._turns = turns
        self._answered = 0
        self.held = threading.Event()
        self.release = threading.Event()

    def next_call(self, offered: list[str]) -> dict:
        if self._answered >= self._turns:
            self.held.set()
            if not self.release.wait(_HOLD_SECONDS):
                raise TimeoutError("driver never released the held turn")
            raise ScriptStepUnavailable(f"scripted interrupt before turn {self._turns + 1}")
        step = super().next_call(offered)
        self._answered += 1
        return step


def _pointer_parts(pointer: str) -> list[str]:
    if not pointer.startswith("/"):
        raise ValueError(f"volatile path must be a JSON pointer starting with '/': {pointer!r}")
    return [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]


def _step(node: object, part: str) -> object:
    if isinstance(node, dict) and part in node:
        return node[part]
    if isinstance(node, list) and part.isdigit() and int(part) < len(node):
        return node[int(part)]
    return _MISSING


def _leaf(document: object, pointer: str) -> object:
    node: object = document
    for part in _pointer_parts(pointer):
        node = _step(node, part)
    return node


def _mask(document: object, pointers: Sequence[str]) -> None:
    """Replace each present pointer's value with a sentinel. Absent pointers are skipped here, not hidden."""
    for pointer in pointers:
        parts = _pointer_parts(pointer)
        parent: object = document
        for part in parts[:-1]:
            parent = _step(parent, part)
        if parent is _MISSING or _step(parent, parts[-1]) is _MISSING:
            continue
        if isinstance(parent, dict):
            parent[parts[-1]] = _MASK
        else:
            parent[int(parts[-1])] = _MASK  # type: ignore[index]


def _show(value: object) -> str:
    text = "<absent>" if value is _ABSENT else json.dumps(value, ensure_ascii=False, sort_keys=True)
    return text if len(text) <= _VALUE_CHARS else text[:_VALUE_CHARS] + "...(truncated)"


def _first_difference(left: object, right: object, path: str = "") -> tuple[str, object, object] | None:
    """First differing JSON path (pointer form, sorted keys) with both values, or None when equal."""
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            child = f"{path}/{key.replace('~', '~0').replace('/', '~1')}"
            if key not in left or key not in right:
                return child, left.get(key, _ABSENT), right.get(key, _ABSENT)
            found = _first_difference(left[key], right[key], child)
            if found is not None:
                return found
        return None
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return path or "/", f"length {len(left)}", f"length {len(right)}"
        for index, (a, b) in enumerate(zip(left, right)):
            found = _first_difference(a, b, f"{path}/{index}")
            if found is not None:
                return found
        return None
    if type(left) is not type(right) or left != right:
        return path or "/", left, right
    return None


def _byte_difference(expected: bytes, actual: bytes) -> str:
    """First differing byte offset with a short window of both sides."""
    limit = min(len(expected), len(actual))
    offset = next((i for i in range(limit) if expected[i] != actual[i]), limit)
    lo, hi = max(0, offset - 40), offset + 40
    return (f"first differing byte offset {offset} (expected {len(expected)} bytes, actual {len(actual)}): "
            f"expected {expected[lo:hi]!r} vs actual {actual[lo:hi]!r}")


def check_resume(uninterrupted: Sequence[bytes], resumed: Sequence[bytes], *, stop_after: int,
                 allowed_volatile: Sequence[str] = ()) -> str:
    """Compare the first post-resume request bytes with the uninterrupted turn stop_after+1 request bytes.

    Returns "byte-equal", or "json-equal-after-mask" when the bodies differ in bytes but are JSON-equal once the
    declared allowed_volatile pointers are masked. Raises ResumeDivergence otherwise: a JSON path diff when the
    parsed bodies differ, or a byte-level diff when they are JSON-equal but no volatile path was declared.
    """
    if stop_after < 1:
        raise ValueError("stop_after must be at least 1")
    for pointer in allowed_volatile:
        _pointer_parts(pointer)
    if len(uninterrupted) <= stop_after:
        raise HarnessError(f"uninterrupted run sent {len(uninterrupted)} requests; need more than {stop_after}")
    turn = stop_after + 1
    if not resumed:
        raise ResumeDivergence(f"resume sent no provider request; expected the turn {turn} request")
    expected, actual = uninterrupted[stop_after], resumed[0]
    if expected == actual:
        return "byte-equal"
    try:
        left = json.loads(expected.decode("utf-8"))
        right = json.loads(actual.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise ResumeDivergence(f"turn {turn} request bytes differ and are not JSON; "
                               f"{_byte_difference(expected, actual)}") from None
    for pointer in allowed_volatile:
        if _leaf(left, pointer) is _MISSING:
            raise ValueError(f"volatile path {pointer!r} is not present in the expected request")
    _mask(left, allowed_volatile)
    _mask(right, allowed_volatile)
    found = _first_difference(left, right)
    if found is not None:
        path, a, b = found
        raise ResumeDivergence(f"turn {turn} request differs after resume at {path}: "
                               f"expected {_show(a)} vs actual {_show(b)}; {_byte_difference(expected, actual)}")
    if allowed_volatile:
        return "json-equal-after-mask"
    raise ResumeDivergence(f"turn {turn} request is JSON-equal but not byte-equal with no volatile path declared; "
                           f"{_byte_difference(expected, actual)}")


def _engine_arguments(scenario: Scenario, repo: Path, run_dir: Path, url: str) -> list[str]:
    # Mirrors the run() argument list in tests/enginecheck/__init__.py; keep the two in step.
    return [
        sys.executable, "-m", "stateframe", "run",
        "--repo", str(repo), "--task", scenario.task,
        "--endpoint", url,
        "--model", "scripted", "--run-dir", str(run_dir), "--quiet", "--json",
        "--max-turns", str(scenario.max_turns), "--test-python", sys.executable,
        *scenario.cli_args,
    ]


def _document(steps: list[dict], **extra: object) -> dict:
    return {"model": "scripted", "steps": steps, "tokenize": {"mode": "ok"}, **extra}


def _await_pending(run_dir: Path, turn: int) -> None:
    """Wait for the engine to record turn's request on disk, which happens before the call is sent."""
    pending = run_dir / "wire" / f"{turn:04d}.request.json"
    deadline = time.monotonic() + _PENDING_WAIT_SECONDS
    while not pending.is_file():
        if time.monotonic() > deadline:
            raise HarnessError(f"engine did not record the turn {turn} request")
        time.sleep(0.05)


def assert_resume_equivalent(scenario: Scenario, tmp_path: Path, *, stop_after: int,
                             allowed_volatile: Sequence[str] = ()) -> str:
    """Run uninterrupted, SIGTERM after turn stop_after is in flight, resume in a fresh process, compare turn k+1."""
    if stop_after < 1:
        raise ValueError("stop_after must be at least 1")
    full = run(scenario, tmp_path / "full")
    if full.exit_code != 0:
        raise HarnessError(f"uninterrupted baseline exited {full.exit_code}", exit_code=full.exit_code,
                           output=full.output)

    case = tmp_path / "cut"
    repo, run_dir = case / "repo", case / "run"
    _write_repo(repo, scenario.files)
    environment = _engine_environment(scenario.switches)

    # Phase 1: stop the engine while turn stop_after+1 is in flight; its request stays recorded and unanswered.
    held = _HoldAfterTurns(_document(scenario.steps), stop_after)
    endpoint = serve(held)
    url = f"http://127.0.0.1:{endpoint.server_port}/v1"
    engine = subprocess.Popen(_engine_arguments(scenario, repo, run_dir, url), env=environment,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        if not held.held.wait(_PENDING_WAIT_SECONDS):
            raise HarnessError(f"engine never sent turn {stop_after + 1}")
        _await_pending(run_dir, stop_after + 1)
        engine.send_signal(signal.SIGTERM)
    finally:
        held.release.set()
        endpoint.shutdown()
        endpoint.server_close()
    try:
        out, err = engine.communicate(timeout=_ENGINE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as error:
        engine.kill()
        raise HarnessError("interrupted engine did not stop after SIGTERM") from error
    if len(_request_paths(run_dir)) < stop_after + 1:
        raise _post_run_failure(f"interrupted run recorded fewer than {stop_after + 1} requests", run_dir,
                                subprocess.CompletedProcess(engine.args, engine.returncode, out, err))

    # Phase 2: a fresh process resumes the run; its endpoint only answers the turns after stop_after.
    # resume moves the interrupted wire/ to attempts/<n>/wire (resume.py:38, archive_attempt :971); run_dir/wire
    # then holds the resumed attempt's own raw request bytes.
    endpoint = serve(Script(_document(scenario.steps[stop_after:])))
    url = f"http://127.0.0.1:{endpoint.server_port}/v1"
    try:
        resumed = run_monitored(
            [sys.executable, "-m", "stateframe", "resume", str(run_dir), "--endpoint", url, "--quiet"],
            env=environment, expected_seconds=_ENGINE_TIMEOUT_SECONDS, run_dir=run_dir,
            step="enginecheck resume: resumed run", check=False)
    finally:
        endpoint.shutdown()
        endpoint.server_close()
    if resumed.returncode != 0:
        tail = (resumed.stdout + resumed.stderr)[-_VALUE_CHARS * 5:]
        raise ResumeDivergence(f"resumed engine exited {resumed.returncode}: {tail}")

    # The resumed attempt re-records the replayed turns 1..k into its own wire/ (same bytes as the tape), so the
    # first request sent live after resume is file stop_after+1 (0-based index stop_after), not file 0001.
    paths = _request_paths(run_dir)
    live = paths[stop_after:]
    try:
        return check_resume(full.requests, [path.read_bytes() for path in live], stop_after=stop_after,
                            allowed_volatile=allowed_volatile)
    except ResumeDivergence as error:
        shape = [(path.name, len(json.loads(path.read_bytes()).get("messages", []))) for path in paths]
        raise ResumeDivergence(f"{error} [resumed wire (name, messages): {shape}; "
                               f"uninterrupted messages: {[len(json.loads(r).get('messages', [])) for r in full.requests]}]") from error
