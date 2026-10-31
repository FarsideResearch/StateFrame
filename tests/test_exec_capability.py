from dataclasses import replace
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernel_exec_capability import ExecLimits, shell_capability, capability_from_dict
from kernel_test_isolation import capability_command
from kernel_exec_capability import executable_read_roots, symlink_hops


@pytest.fixture
def cap(tmp_path):
    for name in ("workspace", "env", "home"):
        (tmp_path / name).mkdir()
    return shell_capability(workspace=tmp_path / "workspace", run_env_dir=tmp_path / "env",
                            scratch_home=tmp_path / "home", network="none", writable=True)


def test_digest_stable_and_immutable(cap):
    environment = {"B": "2", "A": "1"}
    first = replace(cap, env=environment)
    second = replace(cap, env={"A": "1", "B": "2"}, read_roots=tuple(reversed(cap.read_roots)))
    environment["A"] = "changed"
    assert first.digest() == second.digest()
    assert capability_from_dict(first.as_dict()).digest() == first.digest()
    with pytest.raises(TypeError):
        first.env["A"] = "changed"
    assert replace(first, network="full").digest() != first.digest()


@pytest.mark.parametrize("limits", [{"wall_seconds": 601}, {"wall_seconds": float("nan")},
                                    {"nproc": 0}, {"file_bytes": (1 << 30) + 1},
                                    {"nofile": 4}, {"output_bytes": True}])
def test_invalid_limits(limits):
    with pytest.raises(ValueError):
        ExecLimits(**limits)


def test_readonly_overlap_refused(cap):
    with pytest.raises(ValueError, match="overlaps"):
        replace(cap, workspace_writable=False, write_roots=(cap.workspace,))


def test_host_test_environment_never_inherited(cap, monkeypatch):
    monkeypatch.setenv("STATEFRAME_TASK_NETWORK", "container")
    monkeypatch.setenv("STATEFRAME_TASK_ENV_KEYS", "SECRET")
    monkeypatch.setenv("SECRET", "private")
    host = replace(cap, sandboxed=False)
    argv, env = capability_command(["/usr/bin/env"], host)
    assert argv == ["/usr/bin/env"]
    assert env == dict(cap.env)
    assert "SECRET" not in env
    assert "STATEFRAME_TASK_NETWORK" not in env


def test_missing_backend_fails_closed(cap, monkeypatch):
    # Truly absent: not on PATH and not in any standard location.
    monkeypatch.setattr("kernel_test_isolation.shutil.which", lambda name: None)
    monkeypatch.setattr("kernel_test_isolation._SANDBOX_DIRS", ())
    with pytest.raises(RuntimeError, match="no unsandboxed fallback"):
        capability_command(["/bin/true"], cap)


def test_launcher_file_is_explicit_readable(cap):
    # On this host profile generation is a pure operation, no sandbox is started.
    if sys.platform == "darwin":
        command, env = capability_command(["/bin/true"], cap)
        assert "sf_launch.py" in command[2]
        assert env == dict(cap.env)


@pytest.mark.parametrize("network", ["none", "loopback", "full"])
def test_linux_builder_network_and_mount_contract(cap, monkeypatch, network):
    monkeypatch.setattr("kernel_test_isolation.shutil.which", lambda name: "/usr/bin/bwrap")
    cap = replace(cap, network=network, workspace_writable=False)
    command, env = capability_command(["/bin/true"], cap, platform_backend="bwrap")
    assert "--unshare-all" in command and "--die-with-parent" in command
    assert "--new-session" in command
    assert ("--share-net" in command) is (network == "full")
    index = command.index(str(cap.workspace))
    assert command[index - 1] == "--ro-bind"
    assert env == dict(cap.env)
    nested, _ = capability_command(["/bin/true"], cap, platform_backend="bwrap", shared_namespace=True)
    assert "--unshare-pid" not in nested and "--unshare-all" not in nested
    assert ("--unshare-net" in nested) is (network == "none")
    assert "--cap-drop" in nested
    assert ("--network-none" in nested) is (network == "none")


def test_equal_numeric_limits_hash_identically(cap):
    assert replace(cap, limits=ExecLimits(wall_seconds=120)).digest() == cap.digest()


def test_executable_directory_and_shebang_hops(cap, tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "python"
    target.symlink_to(sys.executable)
    alias = tmp_path / "alias"
    alias.symlink_to(outside, target_is_directory=True)
    first = tmp_path / "interpreter"
    first.symlink_to("alias/python")
    script = tmp_path / "command"
    script.write_text(f"#!{first}\n")
    roots = executable_read_roots(script)
    assert {script, first, alias / "python", target} <= set(roots)
    assert outside not in roots and alias not in roots
    assert tmp_path not in roots
    for purpose in ("shell", "job", "custom_tool"):
        authority = replace(cap, purpose=purpose, read_roots=(*cap.read_roots, *roots))
        assert {script, first, alias / "python", target} <= set(authority.read_roots)
        assert outside not in authority.read_roots
        restored = capability_from_dict(authority.as_dict())
        assert restored.digest() == authority.digest()
        monkeypatch.setattr("kernel_test_isolation.shutil.which", lambda name: "/usr/bin/" + name)
        command, _ = capability_command([str(script)], authority, platform_backend="sandbox-exec")
        assert str(first) in command[2] and str(alias) in command[2]
        command, _ = capability_command([str(script)], authority, platform_backend="bwrap")
        assert any(command[i:i + 3] == ["--symlink", "alias/python", str(first)]
                   for i in range(len(command)))


def test_symlink_cycle_and_missing_target_fail_closed(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.symlink_to(second)
    second.symlink_to(first)
    with pytest.raises(ValueError, match="40 hops"):
        symlink_hops(first)
    with pytest.raises(FileNotFoundError):
        symlink_hops(tmp_path / "missing")


def test_deserialization_never_discovers_new_script_authority(cap, tmp_path):
    script, secret = tmp_path / "script", tmp_path / "secret"
    script.write_text("#!/bin/sh\n")
    secret.write_text("private")
    authority = replace(cap, read_roots=(*cap.read_roots, *executable_read_roots(script)))
    script.write_text(f"#!{secret}\n")
    assert capability_from_dict(authority.as_dict()).digest() == authority.digest()
    assert secret not in capability_from_dict(authority.as_dict()).read_roots


def test_changed_symlink_target_cannot_expand_authority(cap, tmp_path):
    script, secret, link = tmp_path / "script", tmp_path / "secret", tmp_path / "link"
    script.write_text("#!/bin/sh\n")
    secret.write_text("private")
    link.symlink_to(script)
    authority = replace(cap, read_roots=(*cap.read_roots, *executable_read_roots(link)))
    link.unlink()
    link.symlink_to(secret)
    with pytest.raises(ValueError, match="lacks explicit authority"):
        capability_from_dict(authority.as_dict())


@pytest.mark.skipif(sys.platform != 'darwin', reason='macOS runtime authority')
def test_macos_developer_aliases_and_homebrew_runtime_roots(cap):
    from pathlib import Path
    from kernel_exec_capability import _macos_developer_roots
    developer = Path('/var/select/developer_dir')
    if developer.is_symlink():
        assert developer in _macos_developer_roots()
        assert developer.resolve() in _macos_developer_roots()
        assert developer in cap.read_roots
    prefix = Path('/opt/homebrew')
    if (prefix / 'Cellar').is_dir():
        assert prefix / 'Cellar' in cap.read_roots
        assert prefix / 'var' not in cap.read_roots
        assert prefix not in cap.read_roots


def test_darwin_cache_filter_only_matches_cache_basename(cap, tmp_path, monkeypatch):
    import kernel_test_isolation as isolation
    directory = tmp_path / 'temp.[literal]'
    directory.mkdir()
    monkeypatch.setattr(isolation.sys, 'platform', 'darwin')
    monkeypatch.setattr(isolation.os, 'confstr', lambda key: str(directory) + '/')
    monkeypatch.setattr(isolation.shutil, 'which', lambda name: '/usr/bin/' + name)
    cache_filter = isolation._darwin_xcrun_filter()
    encoded = re.findall(r'\(regex #("(?:[^"\\]|\\.)*")\)', cache_filter)
    patterns = [json.loads(value) for value in encoded]
    assert patterns
    assert any(re.fullmatch(pattern, str(directory / 'xcrun_db-Ab123')) for pattern in patterns)
    for path in (directory, directory / 'unrelated', directory / 'xcrun_db-',
                 directory / 'xcrun_db-dir/child', directory / 'prefix-xcrun_db-a',
                 directory.parent / 'other/xcrun_db-a'):
        assert not any(re.fullmatch(pattern, str(path)) for pattern in patterns)
    command, _ = capability_command(['/bin/true'], cap, platform_backend='sandbox-exec')
    assert command[2].count(cache_filter) == 2
    assert f'(subpath {json.dumps(str(directory))})' not in command[2]


@pytest.mark.parametrize('directory', [None, '', 'relative/path'])
def test_darwin_cache_unknown_directory_fails_closed(monkeypatch, directory):
    import kernel_test_isolation as isolation
    monkeypatch.setattr(isolation.sys, 'platform', 'darwin')
    monkeypatch.setattr(isolation.os, 'confstr', lambda key: directory)
    with pytest.raises(RuntimeError, match='Darwin user temporary directory'):
        isolation._darwin_xcrun_filter()


@pytest.mark.parametrize('failure', [False, True])
def test_darwin_confstr_name_falls_back_to_bounded_getconf(tmp_path, monkeypatch, failure):
    import kernel_test_isolation as isolation
    def unsupported(key):
        raise ValueError('unrecognized configuration name')
    def getconf(argv, **kwargs):
        assert argv == ['/usr/bin/getconf', 'DARWIN_USER_TEMP_DIR']
        assert kwargs['timeout'] == 5 and kwargs['check']
        assert kwargs['cwd'] == '/'
        assert set(kwargs['env']) == {'PATH', 'LANG', 'LC_ALL'}
        if failure:
            raise subprocess.CalledProcessError(1, argv)
        return SimpleNamespace(stdout=str(tmp_path) + '/\n')
    monkeypatch.setattr(isolation.os, 'confstr', unsupported)
    monkeypatch.setattr(isolation.subprocess, 'run', getconf)
    if failure:
        with pytest.raises(RuntimeError, match='via getconf'):
            isolation._darwin_user_temp_dir()
    else:
        assert isolation._darwin_user_temp_dir() == tmp_path


@pytest.mark.skipif(sys.platform != 'darwin', reason='macOS system Git config')
def test_system_git_config_readonly_or_disabled_when_absent(cap):
    config = Path('/etc/gitconfig')
    if config.is_file():
        assert config.resolve() in cap.read_roots
        assert 'GIT_CONFIG_NOSYSTEM' not in cap.env
    else:
        assert cap.env['GIT_CONFIG_NOSYSTEM'] == '1'
    assert config.parent not in cap.read_roots
    assert not any(config.resolve().is_relative_to(root) for root in cap.write_roots)


def test_sandbox_binary_resolves_without_path(monkeypatch):
    # Regression: a minimal PATH (IDE, cron, a test planting a fake PATH) must not lose the sandbox.
    import kernel_test_isolation
    name = "sandbox-exec" if sys.platform == "darwin" else "bwrap"
    expected = kernel_test_isolation.sandbox_executable(name)
    if expected is None:
        pytest.skip(f"{name} not installed on this host")
    monkeypatch.setenv("PATH", "/nonexistent")
    assert kernel_test_isolation.sandbox_executable(name) == expected or \
        os.path.realpath(kernel_test_isolation.sandbox_executable(name)) == os.path.realpath(expected)
