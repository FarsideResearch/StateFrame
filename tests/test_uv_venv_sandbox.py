"""A uv-created .venv must run inside the check sandbox (11-02).

uv links .venv/bin/python into ~/.local/share/uv/python/cpython-3.12-<platform>/bin/python3.12,
where cpython-3.12-<platform> is itself a symlinked directory. The macOS check profile admitted
only the fully resolved file, so sandbox-exec refused the exec ("execvp() ... Operation not
permitted"), every check came back RED, and the engine refused reads with check_not_settled
until the run deadline.
"""
import os
import platform
import pwd
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import kernel_check_receipts as receipts
import kernel_test_isolation as isolation

DARWIN_DENIAL = ("sandbox-exec: execvp() of '.venv/bin/python' failed: "
                 "Operation not permitted")

# The command builders only need a real bwrap path; skip where the host has none.
BWRAP = isolation.sandbox_executable("bwrap")
needs_bwrap = pytest.mark.skipif(BWRAP is None, reason="bwrap is not installed")


def _uv_layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    """project/.venv/bin/python -> store/cpython-3.12-x -> store/cpython-3.12.14-x/bin/python3.12."""
    store = tmp_path / "uv" / "python"
    real = store / "cpython-3.12.14-x"
    (real / "bin").mkdir(parents=True)
    (real / "lib" / "python3.12" / "site-packages").mkdir(parents=True)
    binary = real / "bin" / "python3.12"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    link_dir = store / "cpython-3.12-x"
    link_dir.symlink_to(real)
    project = tmp_path / "project"
    (project / ".venv" / "bin").mkdir(parents=True)
    (project / ".venv" / "pyvenv.cfg").write_text("home = x\n")
    (project / ".venv" / "bin" / "python").symlink_to(link_dir / "bin" / "python3.12")
    return project, link_dir, real


def test_link_chain_keeps_every_symlink_hop(tmp_path):
    project, link_dir, real = _uv_layout(tmp_path)
    chain = isolation._link_chain(project / ".venv" / "bin" / "python")
    assert project / ".venv" / "bin" / "python" in chain
    assert link_dir in chain  # the hop a resolved path skips
    assert chain[-1] == real / "bin" / "python3.12"


def test_program_roots_admit_the_link_hop_and_the_install(tmp_path):
    project, link_dir, real = _uv_layout(tmp_path)
    roots = isolation._program_read_roots(project / ".venv" / "bin" / "python")
    assert str(link_dir) in roots
    assert str(real) in roots  # the interpreter's stdlib lives under its install prefix


def test_darwin_profile_admits_a_uv_interpreter(monkeypatch, tmp_path):
    project, link_dir, real = _uv_layout(tmp_path)
    monkeypatch.setattr(isolation, "backend", lambda: "/usr/bin/sandbox-exec")
    monkeypatch.setattr(isolation.platform, "system", lambda: "Darwin")
    command, _ = isolation.isolated_command([".venv/bin/python", "-c", "pass"], project, {},
                                            generic_check=True)
    profile = command[2]
    assert f'(subpath "{link_dir}")' in profile
    assert f'(subpath "{real}")' in profile


def test_darwin_profile_never_admits_a_link_to_a_data_file(monkeypatch, tmp_path):
    secret = tmp_path / "secret" / "key"
    secret.parent.mkdir()
    secret.write_text("x")
    project = tmp_path / "project"
    project.mkdir()
    (project / "run").symlink_to(secret)
    monkeypatch.setattr(isolation, "backend", lambda: "/usr/bin/sandbox-exec")
    monkeypatch.setattr(isolation.platform, "system", lambda: "Darwin")
    command, _ = isolation.isolated_command(["./run"], project, {})
    assert str(secret.parent) not in command[2].replace(f'(literal "{secret.parent}")', "")
    assert f'(subpath "{secret}")' not in command[2]


@pytest.mark.host_sandbox
@needs_bwrap
@pytest.mark.skipif(platform.system() != "Linux", reason="requires Linux bwrap behavior")
def test_linux_binds_the_uv_install_and_its_link_hop(monkeypatch, tmp_path):
    project, link_dir, real = _uv_layout(tmp_path)
    monkeypatch.setattr(isolation, "backend", lambda: BWRAP)
    monkeypatch.setattr(isolation.platform, "system", lambda: "Linux")
    command, _ = isolation.isolated_command([".venv/bin/python", "-c", "pass"], project, {})
    head = command[:command.index("--")]
    binds = [head[i:i + 3] for i, part in enumerate(head) if part == "--ro-bind"]
    assert ["--ro-bind", str(real), str(real)] in binds
    assert ["--ro-bind", str(real), str(link_dir)] in binds


def test_interpreter_roots_survive_missing_home_and_unknown_uid(monkeypatch, tmp_path):
    monkeypatch.delenv("HOME", raising=False)

    def missing_passwd_entry(_uid):
        raise KeyError("no passwd entry")

    monkeypatch.setattr(pwd, "getpwuid", missing_passwd_entry)
    prefix = tmp_path / "python-install"
    (prefix / "bin").mkdir(parents=True, exist_ok=True)
    (prefix / "pyvenv.cfg").write_text("home = python\n")
    selected = prefix / "bin" / "python"
    assert isolation._interpreter_roots(selected) == {str(prefix)}


def test_program_read_roots_survive_missing_home_and_unknown_uid(monkeypatch, tmp_path):
    monkeypatch.delenv("HOME", raising=False)

    def missing_passwd_entry(_uid):
        raise KeyError("no passwd entry")

    monkeypatch.setattr(pwd, "getpwuid", missing_passwd_entry)
    program = tmp_path / "python-install" / "bin" / "python"
    program.parent.mkdir(parents=True)
    (program.parents[1] / "pyvenv.cfg").write_text("home = python\n")
    program.write_text("#!/bin/sh\n")
    program.chmod(0o755)
    assert str(program.parents[1]) in isolation._program_read_roots(program)


@pytest.mark.host_sandbox
@needs_bwrap
@pytest.mark.skipif(platform.system() != "Linux", reason="requires Linux bwrap behavior")
def test_isolated_command_builds_with_missing_home_and_unknown_uid(monkeypatch, tmp_path):
    monkeypatch.delenv("HOME", raising=False)

    def missing_passwd_entry(_uid):
        raise KeyError("no passwd entry")

    monkeypatch.setattr(pwd, "getpwuid", missing_passwd_entry)
    monkeypatch.setattr(isolation, "backend", lambda: BWRAP)
    monkeypatch.setattr(isolation.platform, "system", lambda: "Linux")

    command, child_env = isolation.isolated_command([sys.executable, "-c", "pass"], tmp_path, {})

    assert command[0:4] == [BWRAP, "--die-with-parent", "--unshare-all", "--new-session"]
    assert command[command.index("--") + 1:command.index("--") + 4] == [sys.executable, "-c", "pass"]
    assert child_env["HOME"] == str(tmp_path / ".test-tmp")


@pytest.mark.host_sandbox
@needs_bwrap
@pytest.mark.skipif(platform.system() != "Linux", reason="requires Linux uid/passwd behavior")
def test_linux_check_command_with_home_removed_under_unknown_uid(tmp_path):
    if os.getuid() == 0:
        pytest.skip("creating an unknown uid would require root; test only uses the current non-root uid")
    try:
        pwd.getpwuid(os.getuid())
    except KeyError:
        pass
    else:
        pytest.skip("current non-root uid has a passwd entry; creating an unknown uid would require root")

    source_root = Path(__file__).resolve().parents[1] / "src"
    engine_root = source_root / "stateframe" / "engine"
    script = (
        "import pathlib, sys\n"
        f"sys.path[:0] = [{str(source_root)!r}, {str(engine_root)!r}]\n"
        "import kernel_test_isolation as isolation\n"
        f"isolation.backend = lambda: {BWRAP!r}\n"
        "isolation.platform.system = lambda: 'Linux'\n"
        f"command, child_env = isolation.isolated_command([sys.executable, '-c', 'pass'], "
        f"pathlib.Path({str(tmp_path)!r}), {{}})\n"
        f"assert command[0] == {BWRAP!r}\n"
        "assert command[command.index('--') + 1:command.index('--') + 4] == [sys.executable, '-c', 'pass']\n"
        "assert child_env['HOME'].endswith('/.test-tmp')\n"
    )
    child_env = dict(os.environ)
    child_env.pop("HOME", None)
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=child_env,
                            capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_exec_denial_is_an_environment_error_never_red():
    outcome, phase, reached = receipts.classify_check_outcome(
        "FAIL", stderr=DARWIN_DENIAL, exit_code=71)
    assert (outcome, phase, reached) == ("ENVIRONMENT", "HOST", False)
    message = receipts.sandbox_exec_denial(DARWIN_DENIAL)
    assert message == ("the sandbox blocked the project interpreter at .venv/bin/python: "
                       "this is an environment error, not a test result (no test ran)")


@pytest.mark.parametrize("reason", ["Permission denied", "Operation not permitted"])
def test_linux_exec_permission_denial_has_blocked_message(reason):
    message = receipts.sandbox_exec_denial(f"bwrap: execvp /repo/.venv/bin/python: {reason}")
    assert message == ("the sandbox blocked the project interpreter at /repo/.venv/bin/python: "
                       "this is an environment error, not a test result (no test ran)")


def test_missing_interpreter_inside_check_sandbox_is_an_environment_error():
    missing = "bwrap: execvp /repo/.venv/bin/python: No such file or directory"
    assert receipts.sandbox_exec_denial(missing) == (
        "test interpreter /repo/.venv/bin/python not found inside the check sandbox: "
        "this is an environment error, not a test result (no test ran)")
    assert receipts.classify_check_outcome("FAIL", stderr=missing, exit_code=127) == (
        "ENVIRONMENT", "HOST", False)


def test_exec_denial_text_cannot_override_a_reached_assertion():
    outcome, _, _ = receipts.classify_check_outcome(
        "FAIL", stderr=DARWIN_DENIAL, assertion_reached=True)
    assert outcome == "BEHAVIOR_FAIL"


def test_ordinary_red_check_is_unchanged():
    outcome, _, _ = receipts.classify_check_outcome(
        "FAIL", stdout="1 failed, 3 passed", exit_code=1)
    assert outcome == "BEHAVIOR_FAIL"
    assert receipts.sandbox_exec_denial("1 failed, 3 passed") is None


def _uv_python() -> Path | None:
    store = Path.home() / ".local" / "share" / "uv" / "python"
    if platform.system() != "Darwin" or shutil.which("uv") is None or not store.is_dir():
        return None
    return store


@pytest.mark.host_sandbox
@pytest.mark.skipif(_uv_python() is None, reason="needs macOS with uv and a uv-managed Python")
def test_live_uv_venv_runs_under_the_check_sandbox(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    made = subprocess.run(["uv", "venv", "-q", "--python-preference", "only-managed", ".venv"],
                          cwd=project, capture_output=True, text=True, check=False)
    if made.returncode != 0:
        pytest.skip(f"uv venv failed: {made.stderr.strip()[:200]}")
    python = project / ".venv" / "bin" / "python"
    assert python.resolve().is_relative_to(_uv_python().resolve())
    for program in (".venv/bin/python", str(python)):
        argv, env = isolation.isolated_command([program, "-c", "import json; print('ok')"],
                                               project, dict(os.environ))
        result = subprocess.run(argv, cwd=project, env=env, capture_output=True, text=True,
                                timeout=60, check=False)
        assert (result.returncode, result.stdout.strip()) == (0, "ok"), result.stderr
    assert isolation.check_sandbox_exec_denial(str(python)) is None
