"""Additional offline catalog rows for command grammar and data boundaries."""
from __future__ import annotations

import json
import shutil

import pytest

from ._support import FIXTURES, run_case


pytestmark = pytest.mark.host_sandbox


@pytest.mark.skipif(shutil.which("jq") is None, reason="jq is not installed")
def test_u13_data_work_with_jq(tmp_path):
    case = run_case(tmp_path, "one_shell.script.json",
                    substitutions={"COMMAND": "jq . data.json"})
    case.assert_reply("ready")
    case.assert_reply("exit 0")


@pytest.mark.parametrize("command", (
    "find . -delete", "sed -i.bak 's/before/after/' app.py",
    "PATH=. ls", "LD_PRELOAD=/tmp/no-such-library ls",
))
def test_e3_e5_read_only_rejects_dangerous_flags_and_env_assignments(tmp_path, command):
    case = run_case(tmp_path, "one_shell.script.json",
                    substitutions={"COMMAND": command}, flags=("--shell", "read-only"))
    case.assert_reply("REFUSED READ_ONLY_POLICY:")
    assert not (case.run_dir / "workspace" / "app.py.bak").exists()
    assert (case.run_dir / "workspace" / "app.py").read_bytes() == (case.repo / "app.py").read_bytes()


@pytest.mark.parametrize("command", ("", "  ", "ls\n touch x", "ls \\; touch x"))
def test_e4_empty_or_quoted_commands_never_mutate_in_read_only(tmp_path, command):
    case = run_case(tmp_path, "one_shell.script.json",
                    substitutions={"COMMAND": command}, flags=("--shell", "read-only"))
    assert not (case.run_dir / "workspace" / "x").exists()
    assert not (case.repo / "x").exists()
    assert len(case.requests) >= 2


def test_e6_outside_read_root_does_not_reveal_content(tmp_path):
    secret = tmp_path / "outside-secret"
    secret.write_text("SENTINEL_NOT_FOR_MODEL_71A9\n")
    case = run_case(tmp_path, "one_shell.script.json",
                    substitutions={"COMMAND": f"cat {secret}"})
    assert "SENTINEL_NOT_FOR_MODEL_71A9" not in case.transcript


def test_e15_binary_output_is_typed_and_frame_survives(tmp_path):
    case = run_case(tmp_path, "one_shell.script.json",
                    substitutions={"COMMAND": "printf '\\377\\376'"})
    case.assert_reply("binary")
    assert len(case.requests) >= 2


@pytest.mark.parametrize("url", (
    "http://169.254.169.254/latest/meta-data/",
    "http://10.1.2.3/",
    "http://192.168.1.1/",
    "http://[fc00::1]/",
))
def test_e26_private_addresses_refused_before_network(tmp_path, url):
    case = run_case(tmp_path, "one_web.script.json",
                    substitutions={"URL": url}, flags=("--web",))
    case.assert_reply("REFUSED WEB_REFUSED_ADDRESS:")


def test_e30_custom_network_exceeding_host_setting_fails_preflight(tmp_path):
    document = json.loads((FIXTURES / "tools.json").read_text())
    document["tools"] = [{**document["tools"][0], "network": "full", "replay": "external"}]
    tools_file = tmp_path / "network-tools.json"
    tools_file.write_text(json.dumps(document))
    case = run_case(tmp_path, "one_custom.script.json",
                    flags=("--tools", str(tools_file), "--shell-network", "none"))
    assert case.process.returncode != 0
    assert "network: exceeds --shell-network" in case.process.stderr + case.process.stdout
    assert not case.requests
