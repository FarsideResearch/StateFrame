from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from engine_run import run_monitored
from stateframe import chat, profiles, runner
from stateframe.endpoint import Endpoint as StateFrameEndpoint
from tools.immediate_baseline import baseline_checkout  # noqa: F401 (pytest fixture)

from ._support import Endpoint, environment, read_requests, run_case, scaled
from .test_resume_completion import _interrupted_run


pytestmark = pytest.mark.host_sandbox
TASK = 'Create answer.json with the JSON object {"answer":42}.'
PARITY_TASK = 'Create the JSON output with the object {"answer":42}.'


def _script(path: Path, steps: list[dict]) -> Path:
    path.write_text(json.dumps({"model": "scripted", "tokenize": {"mode": "ok"},
                                "steps": steps}))
    return path


def _repo(path: Path) -> Path:
    path.mkdir()
    (path / "app.py").write_text("value = 1\n")
    return path


def _profile(work: Path, **switches: str) -> str:
    path = work / "profile.json"
    path.write_text(json.dumps({"schema": "stateframe-product-profile/1",
                                "switches": {**profiles.load(profiles.DEFAULT), **switches}}))
    return str(path)


def _sent_hashes(run_dir: Path) -> tuple[str, ...]:
    rows = []
    for line in (run_dir / "engine.log").read_text().splitlines():
        if line.startswith("[sf] "):
            event = json.loads(line[5:])
            if event.get("kind") == "send":
                rows.append(event["request_sha256"])
    return tuple(rows)


def _create() -> dict:
    return {"tool": "file_create", "arguments": {
        "path": "answer.json", "content": "{\"answer\":42}\n"}}


def _done() -> dict:
    return {"tool": "done", "arguments": {"summary": "Created the requested file."}}


def _assert_submitted(run_dir: Path, *, check_outcome="not_run") -> dict:
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["engine_result"] == "SUBMITTED"
    if check_outcome is not None:
        assert summary["check_outcome"] == check_outcome
    assert summary["engine_terminal"]["submission"]["stored"] is True
    assert summary["candidate_files_written"] == ["answer.json"]
    return summary


def test_file_create_submits_and_preserves_bytes_until_first_done(tmp_path):
    repo = _repo(tmp_path / "repo")
    script = _script(tmp_path / "create_done.json", [_create(), _done()])
    case = run_case(tmp_path, str(script), repo=repo, task=TASK,
                    flags=("--shell", "off"), default_test_cmd=False)

    summary = _assert_submitted(case.run_dir)
    assert case.process.returncode == 0, case.process.stdout + case.process.stderr
    assert summary["engine_terminal"]["done_facts"]["no_host_checks"] is True
    assert len(_sent_hashes(case.run_dir)) == 2


@pytest.mark.parametrize("delivery_mode", ("on", "off"))
def test_nested_markdown_create_submits_on_first_done(tmp_path, delivery_mode):
    repo = _repo(tmp_path / "repo")
    path = "notes/answer.md"
    create = {"tool": "file_create", "arguments": {"path": path, "content": "# Answer\n"}}
    script = _script(tmp_path / "nested.json", [create, _done()])
    profile = _profile(tmp_path, STATEFRAME_DELIVERY_CONTRACT=delivery_mode)
    case = run_case(tmp_path, str(script), repo=repo, task=f"Create {path} with '# Answer'.",
                    flags=("--shell", "off", "--profile", profile), default_test_cmd=False)

    spec = json.loads((case.run_dir / "spec.json").read_text())
    assert spec["environment"]["STATEFRAME_DELIVERY_CONTRACT"] == delivery_mode
    assert f"created {path}" in json.dumps(case.requests[1]["messages"])
    assert len(case.requests) == 2, [
        message["content"] for request in case.requests[2:] for message in request["messages"]
        if message["role"] == "tool" and "done refused" in message["content"]
    ]
    assert case.summary["engine_result"] == "SUBMITTED", case.process.stderr
    assert case.summary["candidate_files_written"] == [path]


def test_done_submit_off_requests_match_base(tmp_path, baseline_checkout):
    profile = _profile(tmp_path, STATEFRAME_DONE_IS_SUBMIT="off")
    script = _script(tmp_path / "create_done.json", [_create(), _done()])
    baseline_src = baseline_checkout / "src"
    baseline_env = {"PYTHONPATH": os.pathsep.join(map(str, (
        baseline_src, baseline_src / "stateframe" / "engine")))}
    hashes = []
    for name, env in (("base", baseline_env), ("current", {})):
        work = tmp_path / name
        work.mkdir()
        case = run_case(work, str(script), repo=_repo(work / "repo"), task=PARITY_TASK,
                        flags=("--shell", "off", "--profile", profile, "--max-turns", "3"),
                        default_test_cmd=False,
                        env_overrides=env)
        spec = json.loads((case.run_dir / "spec.json").read_text())
        assert spec["environment"]["STATEFRAME_DONE_IS_SUBMIT"] == "off"
        assert "created answer.json" in json.dumps(case.requests[1]["messages"])
        hashes.append(_sent_hashes(case.run_dir))

    assert len(hashes[0]) == 3
    assert hashes[0] == hashes[1]


def test_done_refuses_after_tool_deletes_created_file(tmp_path):
    delete = {"tool": "file_delete", "arguments": {"path": "answer.json"}}
    script = _script(tmp_path / "create_delete.json", [_create(), delete, _done()])
    case = run_case(tmp_path, str(script), repo=_repo(tmp_path / "repo"),
                    task="Create answer.json and remove it again.",
                    flags=("--shell", "off", "--max-turns", "4"), default_test_cmd=False)

    case.assert_reply("file_delete applied. answer.json")
    case.assert_reply("done refused: no product file has been changed yet; no check has passed")


def test_done_refuses_after_failing_host_check_writes_file(tmp_path):
    repo = _repo(tmp_path / "repo")
    (repo / "verify.sh").write_text(
        "set -e\nprintf '{}\\n' > check-output.json\necho 'check wrote check-output.json'\nexit 1\n")
    check = {"tool": "run_tests", "arguments": {"target": ""}}
    script = _script(tmp_path / "check_done.json", [check, _done()])
    case = run_case(tmp_path, str(script), repo=repo, task="Run the repository check.",
                    flags=("--shell", "off", "--check-cmd", "sh verify.sh", "--max-turns", "3"),
                    default_test_cmd=False)

    case.assert_reply("check exit 1")
    case.assert_reply("check wrote check-output.json")
    case.assert_reply("done refused: no product file has been changed yet; no check has passed")


def test_resume_replays_applied_file_create_before_done(tmp_path):
    _repo_path, run_dir, expected = _interrupted_run(
        tmp_path, script_name="file_create_resume.script.json", task=TASK, pause_at=2)
    second = _script(tmp_path / "done_only.json", [_done()])

    with Endpoint(second) as endpoint:
        resumed = run_monitored(
            [sys.executable, "-m", "stateframe", "resume", str(run_dir),
             "--endpoint", endpoint.url, "--model", "scripted", "--quiet"],
            capture_output=True, text=True, env=environment(),
            expected_seconds=scaled(240), check=False)

    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    _assert_submitted(run_dir, check_outcome=None)
    actual = read_requests(run_dir)[1]
    assert actual["messages"] == expected["messages"]
    assert actual["tools"] == expected["tools"]


def test_chat_task_submits_applied_file_create(tmp_path):
    repo = _repo(tmp_path / "repo")
    script = _script(tmp_path / "chat.json", [_create(), _done()])

    with Endpoint(script) as endpoint:
        base = runner.RunOptions(
            repo=repo, task="", endpoint=StateFrameEndpoint.from_args(endpoint.url, "scripted", None),
            max_turns=4, quiet=True, no_project_hooks=True,
        )
        session = chat.Session(repo=repo, home=tmp_path / "session", base=base)
        session.start(printer=lambda *_: None)
        record = session.run_task(TASK, printer=lambda *_: None)

    assert record.result == "SUBMITTED"
    assert record.kept is True
    run_dir = Path(record.run_dir)
    _assert_submitted(run_dir)
    assert len(_sent_hashes(run_dir)) == 2
