"""A package manager named by --test-cmd: only its own install root is mounted, read-only."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

import kernel_check_tool_root as tools
from stateframe import outer_checks, preflight, runner

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux check mounts")


def _script(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n" + text)
    path.chmod(0o755)


def _install(root: Path, name="npm", text='echo "tool-ok $*"\n', linked=True) -> Path:
    _script(root / "bin" / "node", "echo node\n")
    target = root / "lib" / "node_modules" / name / "bin" / f"{name}-cli.js"
    _script(target, text)
    if linked:
        (root / "bin" / name).symlink_to(os.path.relpath(target, root / "bin"))
    for directory in (root, root / "bin", root / "lib", root / "lib" / "node_modules",
                      target.parent, target.parent.parent):
        directory.chmod(0o755)
    return root


@pytest.fixture
def home(tmp_path, monkeypatch):
    value = tmp_path / "home"
    value.mkdir(mode=0o755)
    monkeypatch.setenv("HOME", str(value))
    return value


@pytest.fixture
def repo(tmp_path):
    value = tmp_path / "repo"
    value.mkdir()
    (value / "README").write_text("x\n")
    return value


def _path(*roots):
    return os.pathsep.join([*(str(root / "bin") for root in roots), "/usr/bin", "/bin"])


def _find(root, repo, program="npm"):
    return tools.find_tool_root(program, _path(root), repo)


@pytest.mark.parametrize("program", ["npm", "npx", "pnpm", "yarn"])
def test_plain_install_is_selected(home, repo, program):
    root = _install(home / "opt" / "node-v1", program)
    found, refusal = _find(root, repo, program)
    assert refusal is None
    assert found.record() == {"program": program, "root": str(root)}


def test_symlinked_install_and_symlinked_bin_select_the_real_root(home, repo):
    root = _install(home / ".local" / "opt" / "node-v1")
    whole, bindir = home / "node", home / "shim"
    whole.symlink_to(root)
    bindir.mkdir()
    (bindir / "bin").symlink_to(root / "bin")
    for link in (whole, bindir):
        found, refusal = _find(link, repo)
        assert refusal is None and found.root == str(root)
        assert tools.sandbox_path(_path(link), found).split(":")[0] == str(root / "bin")


def test_other_programs_and_system_tools_do_not_apply(home, repo):
    root = _install(home / "opt" / "node-v1", "cargo")
    assert _find(root, repo, "cargo") == (None, None)
    assert tools.find_tool_root("npm", "/usr/bin-none", repo) == (None, None)


def _refused(root, repo, words, program="npm"):
    found, refusal = _find(root, repo, program)
    assert found is None and refusal and words in refusal, refusal


def test_home_itself_is_refused(home, repo):
    _refused(_install(home), repo, "home directory")


def test_parent_of_home_is_refused(tmp_path, monkeypatch, repo):
    root = _install(tmp_path / "people")
    monkeypatch.setenv("HOME", str(root / "me"))
    _refused(root, repo, "home directory")


def test_account_home_is_refused_when_home_points_elsewhere(home, tmp_path, monkeypatch, repo):
    account = tmp_path / "account"
    root = _install(account)
    real = tools.safe_home
    monkeypatch.setattr(tools, "safe_home", lambda env=None: account if env == {} else real(env))
    _refused(root, repo, "home directory")


@pytest.mark.parametrize("root", ["/", "/opt"])
def test_top_level_directories_are_refused(root):
    assert "top-level" in tools._root_refusal(Path(root), set())


def test_private_directory_in_home_is_refused(home, repo):
    _refused(_install(home / ".local"), repo, "private directory")


@pytest.mark.parametrize("name", [".config", ".cache", ".ssh", ".aws", ".npm", ".secrets"])
def test_configuration_directories_are_refused(home, repo, name):
    _refused(_install(home / name / "nvm" / "node-v1"), repo, "configuration or credential")


@pytest.mark.parametrize("name", [".npmrc", "etc/npmrc", ".netrc", "client_secret.json"])
def test_root_that_holds_a_credential_file_is_refused(home, repo, name):
    root = _install(home / "opt" / "node-v1")
    (root / name).parent.mkdir(exist_ok=True)
    (root / name).write_text("token\n")
    _refused(root, repo, "credential file")


@pytest.mark.parametrize("part", [
    ".", "bin", "bin/node", "lib", "lib/node_modules", "lib/node_modules/npm/bin/npm-cli.js"])
@pytest.mark.parametrize("bits", [0o020, 0o002], ids=["group", "everyone"])
def test_writable_by_others_is_refused(home, repo, part, bits):
    root = _install(home / "opt" / "node-v1")
    target = root / part
    target.chmod(target.stat().st_mode | bits)
    _refused(root, repo, "writable")


def test_parent_writable_by_everyone_is_refused(home, repo):
    root = _install(home / "opt" / "node-v1")
    root.parent.chmod(0o777)
    _refused(root, repo, "writable by everyone")


def test_other_owner_is_refused(home, repo, monkeypatch):
    root = _install(home / "opt" / "node-v1")
    monkeypatch.setattr(tools.os, "getuid", lambda: os.geteuid() + 1)
    _refused(root, repo, "not owned")


def test_tool_or_node_outside_the_root_is_refused(home, repo):
    root = _install(home / "opt" / "node-v1", linked=False)
    outside = home / "elsewhere" / "npm"
    _script(outside, "echo no\n")
    (root / "bin" / "npm").symlink_to(outside)
    _refused(root, repo, "outside the install root")
    second = _install(home / "opt" / "node-v2")
    (second / "bin" / "node").unlink()
    (second / "bin" / "node").symlink_to(outside)
    _refused(second, repo, "outside the install root")


def test_directory_that_is_not_a_node_installation_is_refused(home, repo):
    _script(home / "tools" / "bin" / "npm", "echo no\n")
    _refused(home / "tools", repo, "not a Node installation")


def test_relative_path_entry_is_refused(home, repo):
    assert tools.find_tool_root("npm", "bin:/usr/bin", repo) == (None, "PATH has a relative entry")


def test_repository_supplied_tool_is_refused(home, repo):
    _refused(_install(repo / "tools"), repo, "overlaps the checked repository")
    _refused(_install(repo), repo, "overlaps the checked repository")
    inner = home / "project" / "node_modules" / "x"
    _refused(_install(inner), repo, "node_modules")


def _check(command, repo, root, **options):
    return outer_checks.run(command, repo, python=sys.executable, runner_kind="generic",
                            test_env={"PATH": _path(root)}, **options)


@pytest.mark.host_sandbox
@pytest.mark.parametrize("program", ["npm", "pnpm", "yarn"])
@pytest.mark.parametrize("linked", [False, True], ids=["plain", "symlinked"])
def test_recorded_root_runs_the_tool_in_a_real_check(home, repo, program, linked):
    root = _install(home / "opt" / "node-v1", program)
    entry = root
    if linked:
        entry = home / "current"
        entry.symlink_to(root)
    result = _check(f"{program} test", repo, entry, check_tool_root=str(root))
    assert result["exit_code"] == 0, result["tail"]
    assert "tool-ok test" in result["tail"]


@pytest.mark.host_sandbox
def test_without_the_recorded_root_nothing_is_mounted(home, repo):
    root = _install(home / "opt" / "node-v1")
    for options in ({}, {"check_tool_root": str(home / "opt")}):
        with pytest.raises(outer_checks.OuterCheckError, match="check command not found"):
            _check("npm test", repo, root, **options)


@pytest.mark.host_sandbox
def test_other_programs_cannot_use_the_recorded_root(home, repo):
    root = _install(home / "opt" / "node-v1")
    result = _check(f"sh -c 'test ! -e {root} && echo absent'", repo, root, check_tool_root=str(root))
    assert result["exit_code"] == 0 and "absent" in result["tail"]


@pytest.mark.host_sandbox
def test_mount_is_read_only_and_secrets_beside_it_stay_unreadable(home, repo):
    (home / ".npmrc").write_text("planted-secret\n")
    (home / ".config").mkdir()
    (home / ".config" / "token").write_text("planted-secret\n")
    root = _install(home / "opt" / "node-v1", text=f"""
test ! -e {home}/.npmrc || echo LEAK-npmrc
test ! -e {home}/.config || echo LEAK-config
test ! -e {home}/opt/other || echo LEAK-sibling
if touch {home}/opt/node-v1/bin/new 2>/dev/null; then echo WROTE; fi
echo done
""")
    (home / "opt" / "other").write_text("planted-secret\n")
    result = _check("npm test", repo, root, check_tool_root=str(root))
    assert result["exit_code"] == 0, result["tail"]
    assert "done" in result["tail"]
    assert "LEAK" not in result["tail"] and "WROTE" not in result["tail"]
    assert "planted-secret" not in result["tail"]
    assert not (root / "bin" / "new").exists()


@pytest.mark.host_sandbox
def test_root_changed_after_recording_is_refused(home, repo):
    root = _install(home / "opt" / "node-v1")
    (root / "bin").chmod(0o777)
    with pytest.raises(outer_checks.OuterCheckError, match="check command not found"):
        _check("npm test", repo, root, check_tool_root=str(root))


@pytest.fixture
def recorded():
    yield tools.record_root
    tools.record_root(None)


@pytest.mark.host_sandbox
def test_engine_check_uses_only_the_process_record(home, repo, monkeypatch, recorded):
    import kernel_generic_check
    seen = []

    def fake(_command, _root, **options):
        seen.append(options.get("check_tool_root"))
        raise RuntimeError("stop")

    monkeypatch.setattr(kernel_generic_check, "run_sandboxed", fake)
    row = {"test_command": ["npm", "test"], "check_tool_root": "/from/the/row"}
    kernel_generic_check.check_result(repo, row)
    recorded({"program": "npm", "root": str(home)})
    kernel_generic_check.check_result(repo, row)
    for bad in (None, "x", {}, {"root": 1}, {"root": "relative"}):
        recorded(bad)
        kernel_generic_check.check_result(repo, row)
    assert seen == [None, str(home), None, None, None, None, None]


def test_task_field_list_has_no_tool_root():
    from kernel_information_policy import PUBLIC_FIELDS
    assert not any("tool_root" in name for name in PUBLIC_FIELDS)


@pytest.mark.host_sandbox
def test_full_run_mounts_the_root_and_no_request_names_it(tmp_path, home, repo):
    """The real engine runs the check from the root while every received request omits it."""
    import json
    import subprocess
    from stateframe.scripted_endpoint import Script, serve

    root = _install(home / "opt" / "node-private-marker")
    (repo / "app.py").write_text("a = 1\n")
    received = []
    script = Script({"model": "scripted", "tokenize": {"mode": "ok"}, "steps": [
        {"tool": "str_replace", "required": True,
         "arguments": {"path": "app.py", "old_str": "a = 1", "new_str": "a = 2"}},
        {"tool": "run_tests", "required": True, "arguments": {}},
        {"tool": "done", "required": True, "arguments": {"summary": "Changed a."}}]})
    with serve(script) as server:
        handler = server.RequestHandlerClass
        original = handler.do_POST

        def capture(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            received.append(body)
            import io
            self.rfile = io.BytesIO(body)
            return original(self)

        handler.do_POST = capture
        try:
            env = {"HOME": str(home), "PATH": _path(root), "TMPDIR": str(tmp_path),
                   "PYTHONPATH": str(Path(runner.__file__).resolve().parents[1]),
                   "STATEFRAME_HOME": str(tmp_path / "sfhome"), "PYTHONDONTWRITEBYTECODE": "1"}
            result = subprocess.run(
                [sys.executable, "-m", "stateframe", "run", "--repo", str(repo),
                 "--run-dir", str(tmp_path / "run"), "--task", "Set a to 2 in app.py",
                 "--endpoint", f"http://127.0.0.1:{server.server_port}/v1", "--model", "scripted",
                 "--test-cmd", "npm test", "--target", "app.py", "--max-turns", "6",
                 "--test-python", sys.executable],
                env=env, cwd=tmp_path, capture_output=True, text=True)
        finally:
            handler.do_POST = original
            server.shutdown()
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "check exit 0" in output, output
    spec = json.loads((tmp_path / "run" / "spec.json").read_text())
    assert spec["check_tool_root"] == {"program": "npm", "root": str(root)}
    assert "node-private-marker" not in json.dumps(spec["public_row"])
    assert any(b"Set a to 2" in body for body in received)
    assert not any(b"node-private-marker" in body for body in received)


@pytest.mark.parametrize("change, expected", [
    ({}, True), ({"check_cmd": "npm test"}, False), ({"check_source": "package.json"}, False),
    ({"runner_kind": "pytest"}, False), ({"test_cmd": None}, False)])
def test_only_the_users_test_command_counts(repo, change, expected):
    values = {"test_cmd": "npm test", "check_source": "tests", "runner_kind": "generic", **change}
    options = runner.RunOptions(repo=repo, task="t", endpoint=None, **values)
    assert runner._is_user_test_command(options) is expected


def test_refusal_reason_reaches_the_user(home, repo):
    root = _install(home / ".local")
    message = preflight.missing_program_message(
        "npm test", test_env={"PATH": _path(root)}, repo=repo, user_test_command=True,
        run_probe=lambda *_a: {"exit_code": preflight.MISSING_PROGRAM_EXIT})
    assert "not mounted" in message and "private directory" in message


def test_discovered_command_never_selects_a_root(home, repo):
    root = _install(home / "opt" / "node-v1")
    assert preflight.check_tool_root("npm test", test_env={"PATH": _path(root)}, repo=repo,
                                     user_test_command=False) == (None, None)


@pytest.mark.host_sandbox
def test_real_npm_from_this_host(repo):
    npm = shutil.which("npm")
    if npm is None or Path(npm).resolve().is_relative_to("/usr"):
        pytest.skip("no npm outside the system directories")
    found, refusal = tools.find_tool_root("npm", os.environ["PATH"], repo)
    if found is None:
        pytest.skip(f"host installation refused: {refusal}")
    (repo / "package.json").write_text('{"name":"x","version":"1.0.0","scripts":{"test":"node -e 0"}}')
    result = outer_checks.run("npm test", repo, python=sys.executable, runner_kind="generic",
                              check_tool_root=found.root)
    assert "Fatal process out of memory" in result["tail"] or result["exit_code"] == 0, result["tail"]
