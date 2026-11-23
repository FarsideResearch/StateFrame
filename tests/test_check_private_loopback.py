"""Checks get private loopback on Linux and no network access on macOS."""
from __future__ import annotations

import os
import shlex
import shutil
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from kernel_check_watchdog import CheckWatchdogConfig
import kernel_execution_policy as policy
import kernel_test_isolation as isolation
from stateframe import outer_checks


pytestmark = pytest.mark.host_sandbox
requires_linux = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="requires Linux check isolation")
requires_bwrap = pytest.mark.skipif(
    shutil.which("bwrap") is None, reason="needs the bwrap sandbox")


def _open_host_listener() -> tuple[socket.socket, int]:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    return listener, listener.getsockname()[1]


def _loopback_probe(host_port: int) -> str:
    return textwrap.dedent("""
        import socket

        interfaces = sorted(name for _, name in socket.if_nameindex())
        if interfaces != ["lo"]:
            raise SystemExit("unexpected interface set")

        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            port = server.getsockname()[1]
            with socket.create_connection(("127.0.0.1", port)) as client:
                connection, _ = server.accept()
                with connection:
                    client.sendall(b"request")
                    if connection.recv(32) != b"request":
                        raise SystemExit("local request failed")
                    connection.sendall(b"reply")
                    if client.recv(32) != b"reply":
                        raise SystemExit("local reply failed")

        with socket.socket() as host:
            result = host.connect_ex(("127.0.0.1", {host_port}))
        if result == 0:
            raise SystemExit("host loopback was reachable")

        print("loopback=ok")
        print("host_loopback=blocked")
        print("interfaces=lo")
    """).format(host_port=host_port)


def _run_outer_check(root: Path, source: str) -> dict[str, object]:
    command = shlex.join([sys.executable, "-I", "-B", "-c", source])
    return outer_checks.run(
        command,
        root,
        python=sys.executable,
        test_python=sys.executable,
        runner_kind="generic",
        capture_output=True,
    )


def _run_inner_check(root: Path, source: str) -> subprocess.CompletedProcess[str]:
    return policy.run_sandboxed(
        [sys.executable, "-I", "-B", "-c", source],
        root,
        env=os.environ,
        generic_check=True,
        check_watchdog=CheckWatchdogConfig(),
    ).completed


@requires_linux
@requires_bwrap
def test_outer_check_has_private_loopback_and_only_lo(tmp_path, monkeypatch):
    monkeypatch.delenv("STATEFRAME_TASK_NETWORK", raising=False)
    host_listener, host_port = _open_host_listener()
    try:
        result = _run_outer_check(tmp_path, _loopback_probe(host_port))
    finally:
        host_listener.close()

    assert result["passed"] is True
    assert result["output"] == (
        "loopback=ok\nhost_loopback=blocked\ninterfaces=lo\n")


@requires_linux
@requires_bwrap
def test_inner_check_has_private_loopback_and_only_lo(tmp_path, monkeypatch):
    monkeypatch.delenv("STATEFRAME_TASK_NETWORK", raising=False)
    host_listener, host_port = _open_host_listener()
    try:
        result = _run_inner_check(tmp_path, _loopback_probe(host_port))
    finally:
        host_listener.close()

    assert result.returncode == 0
    assert result.stdout == "loopback=ok\nhost_loopback=blocked\ninterfaces=lo\n"


@requires_linux
@requires_bwrap
def test_second_check_cannot_reach_first_checks_live_loopback(tmp_path, monkeypatch):
    monkeypatch.delenv("STATEFRAME_TASK_NETWORK", raising=False)
    server_source = textwrap.dedent("""
        import select
        import socket
        import sys

        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            print(server.getsockname()[1], flush=True)
            while True:
                readable, _, _ = select.select([server, sys.stdin], [], [])
                if sys.stdin in readable:
                    sys.stdin.readline()
                    break
                connection, _ = server.accept()
                connection.close()
    """)
    first_check, _ = policy.sandboxed_popen(
        [sys.executable, "-I", "-B", "-c", server_source],
        tmp_path,
        env=os.environ,
        generic_check=True,
        cpu_seconds=None,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert first_check.stdout is not None
    assert first_check.stdin is not None
    try:
        port_text = first_check.stdout.readline().decode("ascii").strip()
        assert port_text.isdecimal()
        client_source = textwrap.dedent("""
            import socket

            with socket.socket() as client:
                result = client.connect_ex(("127.0.0.1", {port}))
            print("cross-check=reachable" if result == 0 else "cross-check=blocked")
        """).format(port=int(port_text))
        result = _run_outer_check(tmp_path, client_source)
        assert result["passed"] is True
        assert result["output"] == "cross-check=blocked\n"
    finally:
        if first_check.poll() is None and first_check.stdin is not None:
            try:
                first_check.stdin.write(b"stop\n")
                first_check.stdin.flush()
            except BrokenPipeError:
                pass
        first_check.wait()
        assert first_check.returncode == 0
        if first_check.stdout is not None:
            first_check.stdout.close()
        if first_check.stdin is not None:
            first_check.stdin.close()


def test_macos_check_profile_still_denies_network(tmp_path, monkeypatch):
    monkeypatch.setattr(isolation.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(isolation, "backend", lambda: "/usr/bin/sandbox-exec")

    command, _ = isolation.isolated_command(
        [sys.executable, "-I", "-B", "-c", "pass"],
        tmp_path,
        dict(os.environ),
        selected_python=sys.executable,
    )

    profile = command[2]
    assert "(deny network*)" in profile
    assert "(allow network" not in profile
