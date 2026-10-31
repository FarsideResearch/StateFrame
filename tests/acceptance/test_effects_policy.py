"""Policy refusal, workspace reconciliation, and bounded observation cases."""
from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from ._support import run_case


pytestmark = pytest.mark.host_sandbox


@pytest.mark.parametrize("command", (
    "ls; touch x", "ls && touch x", "$(touch x)", "`touch x`", "ls | sh",
))
def test_e1_e2_read_only_smuggling_is_refused(tmp_path, command):
    case = run_case(tmp_path, "shell_policy.script.json",
                    substitutions={"COMMAND": command}, flags=("--shell", "read-only"))
    case.assert_reply("REFUSED READ_ONLY_POLICY:")
    assert not (case.run_dir / "workspace" / "x").exists()
    assert not (case.repo / "x").exists()


@pytest.mark.parametrize("command", (
    "sed -i.bak 's/assertIn/assertEqual/' tests/test_app.py",
    "echo 'tampered' > tests/test_app.py",
    "rm tests/test_app.py",
    "mv tests/test_app.py tests/moved.py",
))
def test_e8_protected_tests_restored_after_shell(tmp_path, command):
    case = run_case(tmp_path, "shell_policy.script.json",
                    substitutions={"COMMAND": command})
    case.assert_reply("restored (not allowed):")
    case.assert_reply("tests/test_app.py (protected test or inode alias)")
    original = (case.repo / "tests" / "test_app.py").read_bytes()
    assert (case.run_dir / "workspace" / "tests" / "test_app.py").read_bytes() == original
    assert not (case.run_dir / "workspace" / "tests" / "moved.py").exists()


@pytest.mark.parametrize("destination", ("home", "tmp"))
def test_e7_host_path_cannot_be_written(tmp_path, destination):
    name = "sf-escape-" + uuid.uuid4().hex
    outside = (Path.home() if destination == "home" else Path("/tmp")) / name
    assert not outside.exists()
    case = run_case(tmp_path, "shell_policy.script.json",
                    substitutions={"COMMAND": f"echo escape > {outside}"})
    assert not outside.exists()
    case.assert_reply("exit")


def test_e9_out_of_edit_scope_write_is_restored(tmp_path):
    case = run_case(tmp_path, "shell_policy.script.json",
                    substitutions={"COMMAND": "echo changed > scripts/build.py"},
                    flags=("--edit", "app.py"))
    case.assert_reply("restored (not allowed): scripts/build.py")
    assert (case.run_dir / "workspace" / "scripts" / "build.py").read_bytes() == (
        case.repo / "scripts" / "build.py").read_bytes()


def test_e10_created_symlink_removed_and_reported(tmp_path):
    case = run_case(tmp_path, "shell_policy.script.json",
                    substitutions={"COMMAND": "ln -s app.py linked.py"})
    case.assert_reply("restored (not allowed): linked.py")
    assert not (case.run_dir / "workspace" / "linked.py").is_symlink()


def test_e11_e20_output_flood_is_bounded(tmp_path):
    case = run_case(tmp_path, "one_shell.script.json",
                    substitutions={"COMMAND": "yes | head -c 50000000"})
    case.assert_reply("bytes omitted")
    assert len(case.transcript) < 300_000


def test_e12_small_bounded_fork_tree_exits(tmp_path):
    command = ("python3 -c 'import subprocess; children = "
               "[subprocess.Popen([\"sleep\", \"0.2\"]) for _ in range(4)]; "
               "[child.wait() for child in children]; print(\"CHILDREN_DONE\")'")
    case = run_case(tmp_path, "one_shell.script.json",
                    substitutions={"COMMAND": command})
    case.assert_reply("CHILDREN_DONE")


def test_e13_stdin_wait_receives_eof(tmp_path):
    case = run_case(tmp_path, "one_shell.script.json",
                    substitutions={"COMMAND": "cat"}, timeout=60)
    case.assert_reply("exit 0")


def test_e14_ansi_is_stripped(tmp_path):
    case = run_case(tmp_path, "one_shell.script.json",
                    substitutions={"COMMAND": "printf '\\033[31mCOLOR\\033[0m\\n'"})
    case.assert_reply("COLOR")
    assert "\\u001b" not in case.transcript and "\\x1b" not in case.transcript
