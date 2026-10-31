from dataclasses import replace
import sys

import pytest

from kernel_exec_capability import ExecLimits, shell_capability, capability_from_dict
from kernel_test_isolation import capability_command


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
    monkeypatch.setattr("kernel_test_isolation.shutil.which", lambda name: None)
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
