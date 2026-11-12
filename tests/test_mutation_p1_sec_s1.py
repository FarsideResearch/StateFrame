"""Regression tests for mutation payload keyring isolation."""
from __future__ import annotations

import ctypes
import errno
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from _portable import TRUE

import sf_launch


@pytest.mark.parametrize("machine,key_calls", [
    ("x86_64", (248, 249, 250)),
    ("aarch64", (217, 218, 219)),
])
@pytest.mark.parametrize("post_init", [False, True])
@pytest.mark.parametrize("deny_keyring", [False, True])
def test_mutation_keyring_syscalls_denied_in_generated_filter(
        monkeypatch, machine, key_calls, post_init, deny_keyring):
    class Filter(ctypes.Structure):
        _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte),
                    ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint32)]

    class Program(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ushort), ("filters", ctypes.POINTER(Filter))]

    class Prctl:
        restype = None

        def __call__(self, option, *args):
            if option == 22:
                program = ctypes.cast(args[1], ctypes.POINTER(Program)).contents
                self.instructions = [
                    (row.code, row.jt, row.jf, row.k)
                    for row in program.filters[:program.length]
                ]
            return 0

    prctl = Prctl()
    monkeypatch.setattr(sf_launch.platform, "machine", lambda: machine)
    monkeypatch.setattr(sf_launch.ctypes, "CDLL", lambda *_args, **_kwargs: SimpleNamespace(prctl=prctl))
    sf_launch.install_syscall_filter(network_none=True, post_init=post_init,
                                     deny_keyring=deny_keyring)

    deny = (0x06, 0, 0, 0x00050000 | errno.EPERM)
    pairs = list(zip(prctl.instructions, prctl.instructions[1:]))
    for number in key_calls:
        assert (((0x15, 0, 1, number), deny) in pairs) is deny_keyring


@pytest.mark.parametrize("machine,keyctl_number", [("x86_64", 250), ("aarch64", 219)])
def test_mutation_keyring_join_uses_anonymous_session(monkeypatch, machine, keyctl_number):
    class Syscall:
        restype = None

        def __call__(self, number, operation, name):
            assert (number, operation, name.value) == (keyctl_number, 1, None)
            return 123

    syscall = Syscall()
    monkeypatch.setattr(sf_launch.platform, "machine", lambda: machine)
    monkeypatch.setattr(sf_launch.ctypes, "CDLL", lambda *_args, **_kwargs: SimpleNamespace(syscall=syscall))
    sf_launch._join_empty_session_keyring()
    assert syscall.restype is ctypes.c_long


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux mutation launch")
@pytest.mark.parametrize("purpose,network,mode", [
    ("mutation", "none", "--mutation-network-none"),
    ("mutation_broker", "full", "--mutation-process-filter"),
])
def test_mutation_command_selects_keyring_isolating_wrapper(
        monkeypatch, tmp_path, purpose, network, mode):
    import kernel_test_isolation as isolation
    from kernel_exec_capability import ExecCapability, ExecLimits

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    mounts = {}
    for name in ("boot", "unit", "ctl", "in", "resp"):
        root = workspace / name
        root.mkdir()
        mounts[name] = str(root)
    capability = ExecCapability(
        purpose=purpose, workspace=workspace, workspace_writable=False,
        read_roots=(Path(sys.prefix).resolve(strict=True), Path(sys.executable).resolve(strict=True)),
        write_roots=(), network=network, env={}, limits=ExecLimits())
    monkeypatch.setattr(isolation, "sandbox_executable", lambda _name: "/usr/bin/bwrap")
    monkeypatch.setattr(isolation, "_bwrap_userns_guard_supported", lambda _name: False)

    command, _environment = isolation.capability_command(
        [TRUE], capability, platform_backend="bwrap", mutation_mounts=mounts)
    inner = command[command.index("--") + 1:]
    assert inner[:4] == [str(Path(sys.executable).resolve()), "-I", "-S",
                         "/sf/boot/sf_launch.py"]
    assert inner[4] == mode


@pytest.mark.parametrize("mode,network_none", [
    ("--mutation-network-none", True),
    ("--mutation-process-filter", False),
])
def test_mutation_launch_joins_keyring_before_filter_and_exec(
        monkeypatch, mode, network_none):
    events = []
    monkeypatch.setattr(sys, "argv", ["sf_launch.py", mode, "{}", TRUE])
    monkeypatch.setattr(sf_launch, "_join_empty_session_keyring",
                        lambda: events.append("join"), raising=False)
    monkeypatch.setattr(sf_launch, "install_syscall_filter",
                        lambda **kwargs: events.append(("filter", kwargs)))
    monkeypatch.setattr(sf_launch, "_exec_wrapped",
                        lambda *_args: events.append("exec"))

    assert sf_launch.main() == 125
    assert events == ["join", ("filter", {
        "network_none": network_none, "deny_keyring": True}), "exec"]


def test_mutation_launch_refuses_when_keyring_join_fails(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["sf_launch.py", "--mutation-network-none", "{}", TRUE])

    def refuse():
        raise OSError(errno.EPERM, "cannot isolate session keyring")

    monkeypatch.setattr(sf_launch, "_join_empty_session_keyring", refuse, raising=False)
    monkeypatch.setattr(sf_launch, "install_syscall_filter",
                        lambda **_kwargs: pytest.fail("filter ran after failed keyring isolation"))
    monkeypatch.setattr(sf_launch, "_exec_wrapped",
                        lambda *_args: pytest.fail("payload ran after failed keyring isolation"))

    assert sf_launch.main() == 125
    assert "cannot isolate session keyring" in capsys.readouterr().err
