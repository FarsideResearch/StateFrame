"""Runner logging settings compose with the chain's declared wall allowance."""
import json
import sys
import time

import pytest

from stateframe import runner


@pytest.mark.parametrize("timeout", [None, 600.0])
@pytest.mark.parametrize("explicit_level", [None, "debug"])
def test_drive_preserves_chain_environment_and_adds_logging(tmp_path, monkeypatch, timeout, explicit_level):
    monkeypatch.setenv("STATEFRAME_LOG_LEVEL", "error")
    runner._reason_log.reset_for_tests()
    child = tmp_path / "logging_child.py"
    line = "[stateframe] ERROR child: refused (test reason)"
    child.write_text(f"print({line!r}, flush=True)\n")
    monkeypatch.setattr(runner, "CHILD", child)
    switches = {"STATEFRAME_UNIT_COALESCE": "on", "STATEFRAME_RUN_TIME_ALLOWANCE_S": "old"}
    (tmp_path / "spec.json").write_text(json.dumps({
        "environment": switches,
        "environment_sources": {"STATEFRAME_UNIT_COALESCE": "profile"},
    }))
    options = runner.RunOptions(
        repo=tmp_path, task="run", endpoint=None, quiet=True,
        timeout_s=timeout, log_level=explicit_level,
    )
    options._test_python = sys.executable
    if timeout is not None:
        options._deadline = time.monotonic() + timeout
    lines = []
    try:
        assert runner.drive(tmp_path, options, lines.append) == 0
        spec = json.loads((tmp_path / "spec.json").read_text())
        env, sources = spec["environment"], spec["environment_sources"]
        assert env["STATEFRAME_UNIT_COALESCE"] == "on"
        assert sources["STATEFRAME_UNIT_COALESCE"] == "profile"
        assert env["STATEFRAME_LOG_LEVEL"] == (explicit_level or "error")
        assert sources["STATEFRAME_LOG_LEVEL"] == ("cli:--log-level" if explicit_level else "wrapper:log-level")
        assert env["STATEFRAME_PROBLEMS_FILE"] == str(tmp_path / "problems.jsonl")
        assert sources["STATEFRAME_PROBLEMS_FILE"] == "wrapper:problems-file"
        if timeout is None:
            assert "STATEFRAME_RUN_TIME_ALLOWANCE_S" not in env
            assert "STATEFRAME_RUN_WALL_SECONDS" not in env
        else:
            assert env["STATEFRAME_RUN_TIME_ALLOWANCE_S"] == "600.0"
            assert sources["STATEFRAME_RUN_TIME_ALLOWANCE_S"] == "wrapper:run-deadline"
            assert 0 < float(env["STATEFRAME_RUN_WALL_SECONDS"]) < timeout
        assert lines == [line]  # quiet mode still reports the child's reason.
        assert (tmp_path / "engine.log").read_text() == line + "\n"
    finally:
        runner._reason_log.reset_for_tests()
