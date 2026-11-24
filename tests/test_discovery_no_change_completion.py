from __future__ import annotations

import sys
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "tests"),
    str(ROOT / "src"),
    str(ROOT / "src" / "stateframe" / "engine"),
    str(ROOT / "src" / "stateframe" / "engine" / "probes"),
]

from test_discovery_scope import (  # noqa: E402
    DISCOVERY_TASK, _repo, _repository_snapshot, _script, _tool_results,
)
from acceptance._support import PYTHON, environment, run_case  # noqa: E402


def test_discovery_no_change_candidate_answer_is_accepted_once(tmp_path: Path, capsys) -> None:
    repo = _repo(tmp_path, 300)
    before = _repository_snapshot(repo)
    first = "pkg/source_00000.py"
    last = "pkg/source_00299.py"
    script = _script(tmp_path / "discovery.script.json", [
        {"tool": "read", "arguments": {"path": first}},
        {"tool": "read", "arguments": {"path": last}},
        {"tool": "done", "arguments": {
            "summary": f"I read {first} and {last}. Which file should I change?"}},
    ])
    case = run_case(
        tmp_path, str(script), repo=repo, task=DISCOVERY_TASK,
        flags=("--in-place", "--json"), default_test_cmd=False, timeout=360,
    )

    assert _repository_snapshot(repo) == before
    assert case.summary is not None
    assert case.process.returncode == case.summary.get("exit_code") == 1
    assert case.summary.get("engine_result") == "RUN_INCOMPLETE"
    assert case.summary.get("runtime_outcome") == "incomplete"
    assert case.summary.get("succeeded") is False
    assert not case.summary.get("verification")
    assert "verified by tests" not in case.process.stdout.lower()
    assert case.summary.get("files_changed") == []
    assert case.summary.get("product_files_changed") == []
    terminal = case.summary.get("engine_terminal") or {}
    assert terminal.get("reason_code") == "DISCOVERY_SCOPE_NEEDS_TARGET"
    assert terminal.get("candidate_paths") == [first, last]
    message = str(terminal.get("message", ""))
    assert "No files changed" in message
    assert first in message and last in message
    assert "--target PATH" in message and "--edit PATH" in message
    assert json.loads(case.process.stdout) == case.summary

    from stateframe.cli import _print_summary
    _print_summary(case.summary)
    printed = capsys.readouterr().out
    assert message in printed
    assert "verified by tests" not in printed.lower()

    assert len(case.requests) == 3  # two reads, then one done; no refused retry
    assert not any("done refused" in result for result in _tool_results(case))

    resumed = subprocess.run(
        [PYTHON, "-m", "stateframe", "resume", str(case.run_dir), "--quiet"],
        capture_output=True, text=True, env=environment(), timeout=60, check=False,
    )
    assert resumed.returncode == 2
    assert "empty discovery answer needs --target or --edit" in resumed.stderr


def test_empty_discovery_done_without_a_named_read_path_refuses_once(tmp_path: Path) -> None:
    repo = _repo(tmp_path, 201)
    before = _repository_snapshot(repo)
    candidate = "pkg/source_00000.py"
    script = _script(tmp_path / "discovery.script.json", [
        {"tool": "read", "arguments": {"path": candidate}},
        {"tool": "done", "arguments": {
            "summary": "I inspected the repository. Which source file should I change?"}},
    ])
    case = run_case(
        tmp_path, str(script), repo=repo, task=DISCOVERY_TASK,
        flags=("--json",), default_test_cmd=False, timeout=360,
    )

    assert _repository_snapshot(repo) == before
    assert case.summary is not None
    assert case.process.returncode == case.summary.get("exit_code") == 1
    assert case.summary.get("engine_result") == "RUN_INCOMPLETE"
    assert case.summary.get("succeeded") is False
    assert not case.summary.get("verification")
    terminal = case.summary.get("engine_terminal") or {}
    assert terminal.get("candidate_paths") == []
    message = str(terminal.get("message", ""))
    assert "No files changed" in message
    assert message.count("done refused") == 1
    assert "no product file has been changed yet" in message
    assert json.loads(case.process.stdout) == case.summary

    assert len(case.requests) == 2  # one read and one done/refusal, no second done
