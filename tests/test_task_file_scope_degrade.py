"""Regression coverage for task file-scope parsing and safe fail-open behavior."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

TASK = (
    "<question>\n\nCandidates: a, b\n\nAnswer only by creating the file ANSWER.json "
    'at the repository root containing JSON {"answer": [...]} with the selected candidates, '
    'spelled exactly as listed (["none"] if none apply). Do not change any other file.'
)


def test_normal_answer_only_task_parses_under_strict_mode():
    from stateframe.task_file_scope import parse_task_file_scope

    assert parse_task_file_scope(TASK, strict=True) == ("ANSWER.json",)


def test_unparseable_restriction_degrades_without_scope_and_emits_typed_warning(caplog):
    from stateframe.task_row import DECLARE_ALL_LIMIT, resolve_edit_scope

    caplog.set_level(logging.WARNING, logger="stateframe.task_row")
    product = {f"src/module_{index}.py": "VALUE = 1\n"
               for index in range(DECLARE_ALL_LIMIT + 1)}
    assert resolve_edit_scope(
        "Do not change any other file.", product, target=None, edits=[], steering={}
    ) == (product, False, "whole_repo")
    events = [record for record in caplog.records
              if getattr(record, "event_type", None) == "task_file_scope_degraded"]
    assert len(events) == 1
    assert "without a task file scope limit" in events[0].getMessage()


def test_running_guide_documents_unparseable_scope_fallback():
    guide = Path(__file__).resolve().parents[1] / "docs/guide/running.md"
    text = guide.read_text(encoding="utf-8")

    assert (
        "An ambiguous task file-scope rule (for example, `Do not change any other file.` "
        "without a named path) emits a warning and leaves all product files in edit scope, "
        "including in repositories over 200 files. Use `--edit` to enforce a file set."
    ) in text


def test_degraded_task_scope_preserves_explicit_edit_precedence(caplog):
    from stateframe.task_row import resolve_edit_scope

    caplog.set_level(logging.WARNING, logger="stateframe.task_row")
    product = {"a.py": "A = 1\n", "b.py": "B = 2\n"}
    assert resolve_edit_scope(
        "Do not change any other file.", product, target=None, edits=["b.py"], steering={}
    ) == ({"b.py": "B = 2\n"}, True, "explicit")


def test_cli_run_starts_for_answer_only_task(tmp_path, monkeypatch, capsys):
    from stateframe import cli
    from stateframe.scripted_endpoint import Script, serve

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("Candidates are listed in the task.\n", encoding="utf-8")
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "stateframe-home"))
    monkeypatch.delenv("STATEFRAME_ALLOW_REMOTE_ENDPOINT", raising=False)

    script = Script({"model": "scripted", "steps": [
        {"tool": "read", "arguments": {"path": "README.md"}},
        {"tool": "done", "arguments": {"summary": "The answer task was received."}},
    ]})
    server = serve(script)
    run_dir = tmp_path / "run"
    try:
        cli.main([
            "run", "--repo", str(repo), "--task", TASK,
            "--endpoint", f"http://127.0.0.1:{server.server_port}/v1", "--model", "scripted",
            "--run-dir", str(run_dir), "--test-python", sys.executable,
            "--check-cmd", f"{sys.executable} -c pass", "--max-turns", "4",
        ])
    finally:
        server.shutdown()
        server.server_close()

    output = capsys.readouterr()
    assert script.requests, "the CLI must reach the first model request"
    assert "task file-scope rule" not in output.err
    spec = json.loads((run_dir / "spec.json").read_text(encoding="utf-8"))
    assert spec["public_row"]["task_file_scope"] == ["ANSWER.json"]
