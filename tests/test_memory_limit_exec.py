"""Exec-session launcher memory policy and mechanism reporting regressions."""
from __future__ import annotations

import resource
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import sf_launch
from kernel_exec_capability import ExecLimits
from kernel_exec_capability import shell_capability
from kernel_exec_session import CleanupReport, ExecResult, ExecSession


LIMIT = 64 << 20


class _StopExec(Exception):
    pass


def _run_launch(monkeypatch, tmp_path, mechanism):
    calls = []
    monkeypatch.setattr(sf_launch.sys, "platform", "linux")
    monkeypatch.setattr(sf_launch.os, "closerange", lambda *_args: None)
    monkeypatch.setattr(sf_launch.os, "setsid", lambda: None)
    monkeypatch.setattr(sf_launch.os, "umask", lambda _mask: 0)
    monkeypatch.setattr(sf_launch.os, "chdir", lambda _path: None)
    monkeypatch.setattr(sf_launch, "_set_relative_nproc_limit", lambda _limit: None)
    monkeypatch.setattr(sf_launch.resource, "getrlimit", lambda _kind: (
        resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    monkeypatch.setattr(sf_launch.resource, "setrlimit", lambda kind, bounds: calls.append((kind, bounds)))
    monkeypatch.setattr(sf_launch.os, "execvpe", lambda *_args: (_ for _ in ()).throw(_StopExec()))
    spec = {
        "argv": ["/bin/true"], "cwd": str(tmp_path), "env": {},
        "limits": {**asdict(ExecLimits()), "memory_bytes": LIMIT},
        "memory_supervised": mechanism == "rlimit_data",
        "memory_limit_mechanism": mechanism,
    }
    with pytest.raises(_StopExec):
        sf_launch.launch(spec)
    return calls


def test_linux_exec_launcher_uses_rlimit_data_without_rlimit_as(monkeypatch, tmp_path):
    calls = _run_launch(monkeypatch, tmp_path, "rlimit_data")
    kinds = [kind for kind, _bounds in calls]
    assert resource.RLIMIT_AS not in kinds
    assert (resource.RLIMIT_DATA, (LIMIT, LIMIT)) in calls


def test_linux_exec_launcher_leaves_memory_to_configured_cgroup(monkeypatch, tmp_path):
    calls = _run_launch(monkeypatch, tmp_path, "cgroup_memory_max")
    kinds = [kind for kind, _bounds in calls]
    assert resource.RLIMIT_AS not in kinds
    assert resource.RLIMIT_DATA not in kinds


def test_exec_cgroup_entry_records_successful_memory_join(tmp_path):
    group = tmp_path / "group"
    group.mkdir()
    (group / "memory.max").write_text(str(LIMIT))
    (group / "memory.swap.max").write_text("0")
    (group / "cgroup.procs").touch()

    assert sf_launch._enter_command_cgroup(group, LIMIT) == "cgroup_memory_max"
    assert (group / "cgroup.procs").read_text() == str(sf_launch.os.getpid())


@pytest.mark.parametrize(
    ("swap_control", "swap_device", "expected"),
    [("0", True, True), ("16", False, False), (None, False, True),
     (None, True, False)],
    ids=["swap-max-zero", "swap-max-nonzero", "missing-swap-no-device",
         "missing-swap-device-present"],
)
def test_cgroup_memory_acceptance_requires_swap_to_be_disabled(
        tmp_path, swap_control, swap_device, expected):
    group = tmp_path / "cgroup"
    group.mkdir()
    (group / "memory.max").write_text(str(LIMIT))
    # OOM grouping is best-effort and must not gate acceptance of the memory cap.
    (group / "memory.oom.group").write_text("0")
    if swap_control is not None:
        (group / "memory.swap.max").write_text(swap_control)
    swaps = tmp_path / "proc-swaps"
    swaps.write_text(
        "Filename Type Size Used Priority\n"
        + ("/dev/test partition 1024 0 -2\n" if swap_device else ""),
    )

    assert sf_launch._cgroup_memory_limit_configured(
        group, LIMIT, proc_swaps_path=swaps) is expected


def test_exec_cgroup_join_failure_uses_rlimit_data(monkeypatch, tmp_path):
    group = tmp_path / "group"
    group.mkdir()
    (group / "memory.max").write_text(str(LIMIT))
    (group / "cgroup.procs").touch()
    calls = []
    monkeypatch.setattr(sf_launch.resource, "getrlimit", lambda _kind: (
        resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    monkeypatch.setattr(sf_launch.resource, "setrlimit",
                        lambda kind, bounds: calls.append((kind, bounds)))
    write_text = Path.write_text

    def fail_join(path, *args, **kwargs):
        if path.name == "cgroup.procs":
            raise PermissionError("join denied")
        return write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_join)
    assert sf_launch._enter_command_cgroup(group, LIMIT) == "rlimit_data"
    assert (resource.RLIMIT_DATA, (LIMIT, LIMIT)) in calls
    assert resource.RLIMIT_AS not in [kind for kind, _bounds in calls]


def test_darwin_exec_launcher_keeps_supervised_rss_path(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(sf_launch.sys, "platform", "darwin")
    monkeypatch.setattr(sf_launch.os, "closerange", lambda *_args: None)
    monkeypatch.setattr(sf_launch.os, "setsid", lambda: None)
    monkeypatch.setattr(sf_launch.os, "umask", lambda _mask: 0)
    monkeypatch.setattr(sf_launch.os, "chdir", lambda _path: None)
    monkeypatch.setattr(sf_launch.resource, "getrlimit", lambda _kind: (
        resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    monkeypatch.setattr(sf_launch.resource, "setrlimit", lambda kind, bounds: calls.append((kind, bounds)))
    monkeypatch.setattr(sf_launch.os, "execvpe", lambda *_args: (_ for _ in ()).throw(_StopExec()))
    spec = {"argv": ["/bin/true"], "cwd": str(tmp_path), "env": {},
            "limits": {**asdict(ExecLimits()), "memory_bytes": LIMIT},
            "memory_supervised": True, "memory_limit_mechanism": "supervised_rss"}
    with pytest.raises(_StopExec):
        sf_launch.launch(spec)
    assert resource.RLIMIT_AS not in [kind for kind, _bounds in calls]
    assert resource.RLIMIT_DATA not in [kind for kind, _bounds in calls]


def test_exec_result_exposes_memory_limit_mechanism():
    result = ExecResult(0, False, b"", False, 0, 0.0,
                        CleanupReport(True, 0, (), ""),
                        memory_limit_mechanism="cgroup_memory_max",
                        cgroup_leaf_pattern="applied")
    assert result.memory_limit_mechanism == "cgroup_memory_max"
    assert result.cgroup_leaf_pattern == "applied"


def test_shell_memory_mechanism_is_journaled_without_entering_model_receipt():
    from kernel_shell_tool import ShellReply, publish_effects

    events = []
    receipts = []
    runtime = SimpleNamespace(
        journal=SimpleNamespace(emit=lambda kind, **fields: events.append((kind, fields))),
        target=None,
    )
    bolt = SimpleNamespace(receipt=lambda *args, **fields: receipts.append((args, fields)))
    reply = ShellReply(
        "exit -9", exit_code=-9,
        memory_limit_mechanism="rlimit_data+rss_supervisor",
        memory_limit_killed=True,
        cgroup_leaf_pattern="applied",
    )

    publish_effects(reply, runtime, bolt, set())

    assert events == [("shell_memory_enforcement", {
        "producer": "kernel_shell_tool",
        "memory_limit_mechanism": "rlimit_data+rss_supervisor",
        "cgroup_leaf_pattern": "applied",
        "memory_limit_killed": True,
        "memory_limit_reason": "aggregate resident-memory limit exceeded",
        "dispatch": False,
    })]
    assert receipts and "memory_limit_mechanism" not in receipts[0][1]


def test_exec_session_result_reports_applied_fallback(tmp_path):
    for name in ("workspace", "env", "home", "run"):
        (tmp_path / name).mkdir()
    cap = shell_capability(workspace=tmp_path / "workspace", run_env_dir=tmp_path / "env",
                           scratch_home=tmp_path / "home", network="none", writable=True)
    cap = replace(cap, sandboxed=False)
    with ExecSession(tmp_path / "run") as session:
        result = session.run([sys.executable, "-c", "print('exec-memory-ok')"], cap)
    assert result.exit_code == 0, result.output
    assert b"exec-memory-ok" in result.output
    assert result.memory_limit_mechanism == "rlimit_data+rss_supervisor"
