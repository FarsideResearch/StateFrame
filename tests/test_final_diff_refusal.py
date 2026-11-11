"""A refused diff export keeps the completed run summary visible."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from stateframe import chat, cli, runner
from stateframe.endpoint import Endpoint


_REFUSAL_REASON = "STATEFRAME_HOME_IDENTITY_CHANGED"
_STALE_DIFF = "STALE_FINAL_DIFF_MUST_NOT_BE_SHOWN"


def _summary(run_dir: Path) -> dict:
    candidate_digest = "sha256:" + "c" * 64
    artifact_digest = "sha256:" + "a" * 64
    return {
        "run_dir": str(run_dir),
        "final_diff_error": _REFUSAL_REASON,
        "runtime_outcome": "completed",
        "child_exit_code": 0,
        "check_outcome": "pass",
        "artifact": "saved",
        "engine_result": "SUBMITTED",
        "engine_terminal": {"result": "SUBMITTED", "submission": {
            "stored": True, "seal_id": artifact_digest, "artifact_digest": artifact_digest}},
        "candidate_captured": True,
        "candidate_sha256": candidate_digest,
        "candidate_artifact_digest": artifact_digest,
        "tests_before": {"passed": True},
        "tests_after": {"passed": True, "exit_code": 0, "command": "pytest",
                        "candidate_sha256": candidate_digest},
        "files_changed": [],
        "verification": "verified by tests",
        "succeeded": True,
        "exit_code": 0,
    }


@pytest.mark.parametrize("stale_diff", (False, True), ids=("absent-diff", "stale-diff"))
def test_cli_shows_completed_summary_when_final_diff_is_refused(
        tmp_path, monkeypatch, capsys, stale_diff):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    if stale_diff:
        (run_dir / "final.diff").write_text(_STALE_DIFF, encoding="utf-8")
    summary = _summary(run_dir)
    monkeypatch.setattr(cli, "_run_options", lambda _args, _endpoint: object())
    monkeypatch.setattr(cli, "run", lambda _options, printer: summary)

    result = cli.main([
        "run", "--repo", str(tmp_path), "--task", "verify completed summary",
        "--endpoint", "http://127.0.0.1:8080/v1", "--model", "test",
    ])
    output = capsys.readouterr().out

    assert result == 0
    assert f"final.diff export refused: {_REFUSAL_REASON}" in output
    assert "runtime=completed checks=pass artifact=saved" in output
    assert any(line.lstrip().startswith("engine result:") and line.rstrip().endswith("SUBMITTED")
               for line in output.splitlines())
    assert _STALE_DIFF not in output


def _session(tmp_path: Path) -> chat.Session:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("value = 1\n", encoding="utf-8")
    base = runner.RunOptions(
        repo=repo,
        task="",
        endpoint=Endpoint.from_args("http://127.0.0.1:8080/v1", "test", None),
        max_turns=1,
        test_cmd="true",
        checks=["tests/test_sample.py"],
        target="sample.py",
    )
    return chat.Session(repo=repo, home=tmp_path / "session", base=base)


@pytest.mark.parametrize("stale_diff", (False, True), ids=("absent-diff", "stale-diff"))
def test_chat_shows_completed_summary_when_final_diff_is_refused(
        tmp_path, monkeypatch, capsys, stale_diff):
    session = _session(tmp_path)

    def refuse_diff_export(options, printer):
        run_dir = options.run_dir
        run_dir.mkdir(parents=True)
        if stale_diff:
            (run_dir / "final.diff").write_text(_STALE_DIFF, encoding="utf-8")
        summary = _summary(run_dir)
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return summary

    monkeypatch.setattr(chat, "run", refuse_diff_export)
    commands = iter(("verify completed summary", "/quit"))
    messages = []
    result = chat.repl(
        session,
        read=lambda _prompt: next(commands),
        printer=messages.append,
        print_summary=lambda path: cli._print_summary(json.loads(Path(path).read_text())),
    )
    output = capsys.readouterr().out + "\n".join(messages)

    assert result == 0
    assert f"final.diff export refused: {_REFUSAL_REASON}" in output
    assert "runtime=completed checks=pass artifact=saved" in output
    assert session.history[0].error is None
    assert session.history[0].runtime_outcome == "completed"
    assert session.history[0].check_outcome == "pass"
    assert session.history[0].artifact == "saved"
    assert session.history[0].result == "SUBMITTED"
    assert session.history[0].tests_passed
    assert _STALE_DIFF not in output
