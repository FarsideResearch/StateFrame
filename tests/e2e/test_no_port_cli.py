"""Unavailable database evidence survives a real CLI and child-engine run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from stateframe import cli, runner
from stateframe.scripted_endpoint import Script, serve


pytestmark = pytest.mark.host_sandbox
LIMITATION = "runtime_namespace_acceptance_not_observed"


def test_unavailable_namespace_limitation_reaches_run_summary(tmp_path: Path, monkeypatch, capsys):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "db.py").write_text(
        "def require_isolated_f2_database(name, backend, host, port):\n"
        "    return name.startswith('test_')\n", encoding="utf-8")
    (repo / "test_db.py").write_text(
        "from db import require_isolated_f2_database\n"
        "def test_namespace():\n"
        "    assert require_isolated_f2_database('test_example', 'postgresql', 'localhost', 5432)\n",
        encoding="utf-8")

    # Public CLI rows do not choose a legacy semantic function. Supply that
    # namespace case at the row seam; CLI, child, worker, and summary remain real.
    original_build_row = runner.build_row

    def namespace_row(*args, **kwargs):
        row = original_build_row(*args, **kwargs)
        row["fn"] = "require_isolated_f2_database"
        row["entry"]["fn"] = row["fn"]
        return row

    monkeypatch.setattr(runner, "build_row", namespace_row)
    script = Script({"model": "scripted", "steps": [
        {"tool": "read", "arguments": {"path": "db.py"}},
        {"tool": "done", "arguments": {"summary": "No source change required."}},
    ]})
    server = serve(script)
    run_dir = tmp_path / "run"
    try:
        status = cli.main([
            "run", "--repo", str(repo), "--task", "Inspect the database namespace function.",
            "--target", "db.py", "--check", "test_db.py",
            "--test-python", sys.executable,
            "--endpoint", f"http://127.0.0.1:{server.server_port}/v1",
            "--model", "scripted", "--run-dir", str(run_dir), "--max-turns", "4", "--quiet",
        ])
    finally:
        server.shutdown()
        server.server_close()

    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert (run_dir / "summary.json").is_file(), f"CLI status {status}: {output}"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["limitations"] == [LIMITATION]
    assert LIMITATION in output and "limitation" in output
    assert LIMITATION in (run_dir / "engine-result.json").read_text(encoding="utf-8")


def test_normal_summary_omits_limitation_field(tmp_path: Path, capsys):
    (tmp_path / "engine-result.json").write_text(
        json.dumps({"result": {"result": "SUBMITTED"}}), encoding="utf-8")
    summary = runner.summarize(tmp_path, runner.time.monotonic())
    assert "limitations" not in summary
    (tmp_path / "final.diff").write_text("", encoding="utf-8")
    summary.update({"run_dir": str(tmp_path), "tests_before": {"passed": True},
                    "tests_after": {"command": "pytest"}, "runtime_outcome": "completed",
                    "check_outcome": "pass", "artifact": "captured", "succeeded": False})
    cli._print_summary(summary)
    assert LIMITATION not in capsys.readouterr().out
    summary["limitations"] = None
    cli._print_summary(summary)
    assert LIMITATION not in capsys.readouterr().out
