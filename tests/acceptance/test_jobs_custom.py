"""Background job lifecycle and explicitly declared custom command tools."""
from __future__ import annotations

import json
import socket

import pytest

from ._support import FIXTURES, run_case


pytestmark = pytest.mark.host_sandbox


def _free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def test_u12_dev_server_wait_client_and_stop(tmp_path):
    port = _free_port()
    case = run_case(tmp_path, "jobs.script.json", substitutions={"PORT": str(port)},
                    flags=("--shell-network", "loopback"))
    case.assert_reply("started job j1:")
    case.assert_reply("endpoint reachable")
    case.assert_reply("200")
    case.assert_reply("job j1 stopped")
    assert len(case.requests) >= 2
    assert "jobs" in str(case.requests[1]).lower()
    case.assert_offered("job_wait", request=1)


def test_u12_done_stops_running_job(tmp_path):
    port = _free_port()
    case = run_case(tmp_path, "jobs_done.script.json",
                    substitutions={"PORT": str(port)}, flags=("--shell-network", "loopback"))
    case.assert_reply("started job j1:")
    assert case.summary is not None
    with pytest.raises(OSError):
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            pass


def test_u14_custom_tool_schema_argv_and_bad_arguments(tmp_path):
    case = run_case(tmp_path, "custom.script.json",
                    flags=("--tools", str(FIXTURES / "tools.json")))
    case.assert_offered("sample_tool")
    case.assert_reply("accepted")
    case.assert_reply("REFUSED BAD_ARGUMENTS:")


def test_u14_custom_tool_workspace_effect_is_captured(tmp_path):
    case = run_case(tmp_path, "custom_effect.script.json",
                    flags=("--tools", str(FIXTURES / "tools.json"), "--in-place"))
    case.assert_offered("generate_tool")
    case.assert_reply("files changed by this command: created: generated.py")
    assert "generated.py" in case.diff
    assert case.summary is not None and case.summary["succeeded"] is True, case.summary
    assert (case.repo / "generated.py").exists()


@pytest.mark.parametrize("invalid,fragment", (
    ({"name": "shell"}, "duplicates a built-in"),
    ({"argv": ["/bin/echo", "pre{value}"]}, "placeholders must occupy a whole item"),
))
def test_e29_invalid_custom_definitions_fail_before_first_request(tmp_path, invalid, fragment):
    document = json.loads((FIXTURES / "tools.json").read_text())
    document["tools"] = [{**document["tools"][0], **invalid}]
    definition = tmp_path / "invalid-tools.json"
    definition.write_text(json.dumps(document))
    case = run_case(tmp_path, "custom_invalid.script.json", flags=("--tools", str(definition)))
    assert case.process.returncode != 0
    assert fragment in case.process.stderr + case.process.stdout
    assert not case.requests


def test_u15_shell_ask_without_tty_declines_unlisted_command(tmp_path):
    case = run_case(tmp_path, "one_shell.script.json",
                    substitutions={"COMMAND": "touch asked.txt"}, flags=("--shell-ask",))
    case.assert_reply("REFUSED ASK_DECLINED:")
    assert not (case.run_dir / "workspace" / "asked.txt").exists()
