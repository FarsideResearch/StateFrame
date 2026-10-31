from dataclasses import replace

import pytest

from kernel_shell_policy import ShellPolicyRefused, admit_shell, normalize_output
from kernel_tool_settings import ToolSettings


@pytest.mark.parametrize("command", ["ls -la", "find . -name '*.py'", "cat x | wc -l",
    "pwd && uname -a; date", "cat x > /dev/null", "/bin/ls", "sed -n '1,20p' x",
    "sed 's/old/new/g' x", "env", "ls\nwc -l x"])
def test_readonly_inspections(command):
    admitted = admit_shell(command, ToolSettings(shell="read-only", shell_network="full"))
    assert admitted.network == "none"
    assert not admitted.workspace_writable


@pytest.mark.parametrize("command", ["ls | sh", "ls; python -c 'print(1)'", "ls\nsh",
    "cat $(id)", "cat `id`", "cat <(id)", "PATH=. ls", "./ls", "/tmp/ls",
    "find . -exec sh -c x ';'", "find . -delete", "find . -fprint /tmp/x", "sed -i x y",
    "sed 's/x/y/e' x", "sed -n '1p' -e '1e id' x", "sed -f script x",
    "awk -f script", "awk 'BEGIN {system(\"id\")}'", "env sh", "env -S sh",
    "rg --pre=sh x", "cat x > /tmp/output", "cat x >> /dev/null", "cat x &",
    "ls || cat x", "cat x 2>&1", "sort -o /tmp/x y", "sort --compress-program=sh x", "xxd -r x",
    "ls; ｓｈ", "", " ", "cat\0x", "ls '"])
def test_readonly_refusals(command):
    with pytest.raises(ShellPolicyRefused) as caught:
        admit_shell(command, ToolSettings(shell="read-only"))
    assert caught.value.code == "READ_ONLY_POLICY"
    assert caught.value.reply().startswith("REFUSED READ_ONLY_POLICY:")


@pytest.mark.parametrize("command", ["git status", "ls && /usr/bin/ssh host", "ls; sudo id",
                                      "ls\nscp file host:", "ls | /bin/su"])
def test_standard_filter_across_commands(command):
    with pytest.raises(ShellPolicyRefused, match="denied") as caught:
        admit_shell(command, ToolSettings())
    assert caught.value.code == "DENIED_PROGRAM"


def test_allow_deny_and_confirmation_are_separate():
    settings = ToolSettings(shell_ask=True, shell_allow=("python",), shell_deny=("node",))
    assert not admit_shell("python --version", settings).requires_confirmation
    assert admit_shell("make build", settings).requires_confirmation
    assert not admit_shell("ls", settings).requires_confirmation
    with pytest.raises(ShellPolicyRefused):
        admit_shell("node --version", settings)
    assert admit_shell("node --version", replace(settings, shell="unrestricted"))
    with pytest.raises(ShellPolicyRefused) as caught:
        admit_shell("ls", replace(settings, shell="off"))
    assert caught.value.code == "SHELL_OFF"


@pytest.mark.parametrize("command", ["pip install requests", "python -m pip install x",
    "npm install", "npm ci", "pnpm add x", "uv sync", "uv pip install x"])
def test_network_off_install_names_required_setting(command):
    with pytest.raises(ShellPolicyRefused, match="--shell-network full") as caught:
        admit_shell(command, ToolSettings(shell_network="none"))
    assert caught.value.code == "NETWORK_OFF"
    assert admit_shell(command, ToolSettings(shell_network="full"))


def test_standard_interpreters_are_not_claimed_to_be_a_security_filter():
    assert admit_shell("python -c 'import os; os.system(\"git status\")'", ToolSettings())


def test_normalization_preserves_meaning_and_bounds_utf8():
    assert normalize_output(b"\x1b[31mred\x1b[0m\n10%\r20%\r100%\n") == "red\n100%\n"
    assert normalize_output(b"/run/work/x /run/y pid=123 duration=3s", replacements={
        "/run": "<run>", "/run/work": "<workspace>"}) == "<workspace>/x <run>/y pid=123 duration=3s"
    output = normalize_output(("é" * 10000).encode())
    head, _, tail = output.split("\n")
    assert len(head.encode()) <= 6144 and len(tail.encode()) <= 6144
    assert "7712 bytes omitted" in output
    assert normalize_output(b"\0", total_bytes=1000) == "[binary output, 1000 bytes]"
    assert normalize_output(b"invalid \xff") == "[binary output, 9 bytes]"
    assert normalize_output(b"\xff\xfe") == "[binary output, 2 bytes]"
    assert normalize_output(b"abcdef", total_bytes=100) == "abc\n[... 94 bytes omitted ...]\ndef"


@pytest.mark.host_sandbox
def test_readonly_admission_and_capability_execute_together(tmp_path):
    from kernel_exec_capability import shell_capability
    from kernel_exec_session import ExecSession

    for name in ("workspace", "env", "home", "run"):
        (tmp_path / name).mkdir()
    workspace = tmp_path / "workspace"
    (workspace / "code.txt").write_text("observed\n")
    admission = admit_shell("cat code.txt", ToolSettings(shell="read-only"))
    capability = shell_capability(workspace=workspace, run_env_dir=tmp_path / "env",
        scratch_home=tmp_path / "home", network=admission.network,
        writable=admission.workspace_writable)
    with ExecSession(tmp_path / "run") as session:
        observed = session.run(["/bin/sh", "-c", "cat code.txt"], capability)
        assert observed.cleanup.ok and observed.exit_code == 0
        assert normalize_output(observed.output) == "observed\n"
        # Exercise the capability independently of the grammar: a missed filter
        # must still not be able to modify the workspace.
        denied = session.run(["/bin/sh", "-c", "echo changed > code.txt"], capability)
        assert denied.cleanup.ok and denied.exit_code != 0
    assert (workspace / "code.txt").read_text() == "observed\n"
