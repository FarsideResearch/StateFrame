"""Linux child processes prove memory limits and large virtual reservations."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid

import pytest

import kernel_execution_policy as policy
import sf_launch


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux memory policy")
LIMIT = 64 << 20
SHARED_MAPPING_LIMIT = 32 << 20


def _memory_cgroup(memory_bytes: int = LIMIT):
    group = sf_launch._command_cgroup(uuid.uuid4().hex, 32, memory_bytes=memory_bytes)
    if (group is None or sf_launch._cgroup_leaf_pattern_status() != "applied"
            or not sf_launch._cgroup_memory_limit_configured(group, memory_bytes)):
        if group is not None:
            try:
                group.rmdir()
            except OSError:
                pass
        pytest.skip("writable delegated cgroup v2 memory.max is unavailable")
    return group


def _run(script: str, cgroup: Path | None, *,
         memory_bytes: int = LIMIT, file_bytes: int = 1 << 20):
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, timeout=15,
        preexec_fn=policy._resource_preexec(
            10, memory_bytes, file_bytes, 128, 32, cgroup),
    )


def _identity_isolation(command, _root, environment, **_options):
    return list(command), dict(environment)


def _run_supervised(script: str, root: Path, monkeypatch, *,
                    memory_bytes: int = LIMIT, file_bytes: int = 1 << 20):
    report = policy.CapabilityReport("Linux", "test", True, True, True, True, True, True)
    monkeypatch.setattr(policy, "require_capabilities", lambda: report)
    monkeypatch.setattr(policy, "_command_cgroup", lambda *_args, **_kwargs: None)
    popen = policy.sandboxed_popen

    def without_namespace(command, cwd, **options):
        return popen(command, cwd, isolation_builder=_identity_isolation, **options)

    monkeypatch.setattr(policy, "sandboxed_popen", without_namespace)
    return policy.run_sandboxed(
        [sys.executable, "-c", script], root,
        timeout=10, cpu_seconds=10, memory_bytes=memory_bytes,
        file_bytes=file_bytes, nproc=32,
    )


def _completed(result):
    return result.completed if isinstance(result, policy.PolicyResult) else result


def _text(value: bytes | str | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value or ""


def test_delegated_leaf_pattern_enforces_memory_in_a_real_sandbox(tmp_path):
    report = policy.capability_report()
    if not report.available or report.backend != "bwrap":
        pytest.skip("Linux bwrap execution boundary is unavailable")
    origin = sf_launch._current_cgroup()
    if origin is not None and sf_launch._read_cgroup_pids(origin) != [os.getpid()]:
        pytest.skip("run pytest directly as the delegated scope's only process")
    initial = policy.run_sandboxed(
        [sys.executable, "-c", "print('leaf-pattern-ready')"], tmp_path,
        timeout=10, cpu_seconds=10, memory_bytes=LIMIT,
        file_bytes=1 << 20, nproc=32,
    )
    if (sf_launch._cgroup_leaf_pattern_status() != "applied"
            or initial.memory_limit_mechanism != "cgroup_memory_max"):
        pytest.skip("current cgroup is not delegated and writable for the leaf pattern")
    assert initial.completed.returncode == 0, initial.completed.stderr
    supervisor = sf_launch._current_cgroup()
    assert supervisor is not None and supervisor.name == "supervisor"
    parent = supervisor.parent
    group = sf_launch._command_cgroup("real-leaf-probe", 32, memory_bytes=LIMIT)
    assert group is not None
    supervisor = parent / "supervisor"
    try:
        assert sf_launch._current_cgroup() == supervisor
        assert str(os.getpid()) in (supervisor / "cgroup.procs").read_text().split()
        assert set((parent / "cgroup.subtree_control").read_text().split()) >= {
            "memory", "pids"}
        assert group.parent == parent
        assert (group / "memory.max").read_text().strip() == str(LIMIT)
        assert (group / "pids.max").read_text().strip() == "40"

        script = f"""block = bytearray(2 * ({LIMIT}))
for offset in range(0, len(block), 4096):
    block[offset] = 1
import time; time.sleep(0.2)
"""
        result = policy.run_sandboxed(
            [sys.executable, "-c", script], tmp_path,
            timeout=10, cpu_seconds=10, memory_bytes=LIMIT,
            file_bytes=2 * LIMIT, nproc=32,
        )
        assert result.memory_limit_mechanism == "cgroup_memory_max"
        assert result.completed.returncode == -signal.SIGKILL
    finally:
        try:
            group.rmdir()
        except OSError:
            pass


@pytest.mark.parametrize("mechanism", ["rlimit_data", "cgroup_memory_max"])
def test_process_cannot_touch_twice_its_memory_budget(mechanism, tmp_path, monkeypatch):
    group = _memory_cgroup() if mechanism == "cgroup_memory_max" else None
    script = """blocks = []
try:
    for _ in range(4):
        block = bytearray(32 << 20)
        for offset in range(0, len(block), 4096):
            block[offset] = 1
        blocks.append(block)
except MemoryError:
    raise SystemExit(7)
import time; time.sleep(0.4)
raise SystemExit(0)
"""
    try:
        result = (_run_supervised(script, tmp_path, monkeypatch)
                  if mechanism == "rlimit_data" else _run(script, group))
    finally:
        if group is not None:
            try:
                group.rmdir()
            except OSError:
                pass
    completed = _completed(result)
    assert completed.returncode != 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("mechanism", ["rlimit_data", "cgroup_memory_max"])
def test_process_can_reserve_large_prot_none_address_space(mechanism, tmp_path, monkeypatch):
    group = _memory_cgroup() if mechanism == "cgroup_memory_max" else None
    script = """import mmap
size = 8 * (64 << 20)
mapping = mmap.mmap(-1, size,
                    flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS |
                          getattr(mmap, 'MAP_NORESERVE', 0),
                    prot=0)
mapping.close()
print('reservation-ok')
"""
    try:
        result = (_run_supervised(script, tmp_path, monkeypatch)
                  if mechanism == "rlimit_data" else _run(script, group))
    finally:
        if group is not None:
            try:
                group.rmdir()
            except OSError:
                pass
    completed = _completed(result)
    assert completed.returncode == 0, completed.stderr
    assert _text(completed.stdout).strip() == "reservation-ok"
    if mechanism == "rlimit_data":
        assert result.memory_limit_killed is False


@pytest.mark.parametrize("mechanism", ["rlimit_data", "cgroup_memory_max"])
def test_process_cannot_touch_twice_budget_in_shared_anonymous_mapping(
        mechanism, tmp_path, monkeypatch):
    group = (_memory_cgroup(SHARED_MAPPING_LIMIT)
             if mechanism == "cgroup_memory_max" else None)
    script = """import mmap
size = 2 * (32 << 20)
mapping = mmap.mmap(-1, size,
                    flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE)
for offset in range(0, size, 4096):
    mapping[offset:offset + 1] = b'x'
import time; time.sleep(2.2)
raise SystemExit(0)
"""
    try:
        result = (_run_supervised(
            script, tmp_path, monkeypatch, memory_bytes=SHARED_MAPPING_LIMIT,
            file_bytes=2 * SHARED_MAPPING_LIMIT)
            if mechanism == "rlimit_data" else
            _run(script, group, memory_bytes=SHARED_MAPPING_LIMIT,
                 file_bytes=2 * SHARED_MAPPING_LIMIT))
    finally:
        if group is not None:
            try:
                group.rmdir()
            except OSError:
                pass
    completed = _completed(result)
    assert completed.returncode != 0, completed.stdout + completed.stderr
    if mechanism == "rlimit_data":
        assert completed.returncode == -signal.SIGKILL
        assert result.memory_limit_killed is True


@pytest.mark.parametrize("mechanism", ["rlimit_data", "cgroup_memory_max"])
def test_process_cannot_touch_twice_budget_in_memfd_shared_mapping(
        mechanism, tmp_path, monkeypatch):
    if not hasattr(os, "memfd_create"):
        pytest.skip("os.memfd_create is unavailable")
    group = (_memory_cgroup(SHARED_MAPPING_LIMIT)
             if mechanism == "cgroup_memory_max" else None)
    script = """import mmap
import os
size = 2 * (32 << 20)
fd = os.memfd_create('stateframe-memory-limit-probe')
os.ftruncate(fd, size)
mapping = mmap.mmap(fd, size, flags=mmap.MAP_SHARED,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE)
for offset in range(0, size, 4096):
    mapping[offset:offset + 1] = b'x'
import time; time.sleep(2.2)
raise SystemExit(0)
"""
    try:
        result = (_run_supervised(
            script, tmp_path, monkeypatch, memory_bytes=SHARED_MAPPING_LIMIT,
            file_bytes=2 * SHARED_MAPPING_LIMIT)
            if mechanism == "rlimit_data" else
            _run(script, group, memory_bytes=SHARED_MAPPING_LIMIT,
                 file_bytes=2 * SHARED_MAPPING_LIMIT))
    finally:
        if group is not None:
            try:
                group.rmdir()
            except OSError:
                pass
    completed = _completed(result)
    assert completed.returncode != 0, completed.stdout + completed.stderr
    if mechanism == "rlimit_data":
        assert completed.returncode == -signal.SIGKILL
        assert result.memory_limit_killed is True


def test_fallback_rss_supervisor_does_not_kill_under_budget_process(tmp_path, monkeypatch):
    script = """import mmap, time
size = 16 << 20
mapping = mmap.mmap(-1, size,
                    flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE)
for offset in range(0, size, 4096):
    mapping[offset:offset + 1] = b'x'
time.sleep(0.4)
print('under-budget-ok')
"""
    result = _run_supervised(script, tmp_path, monkeypatch)
    assert result.completed.returncode == 0, result.completed.stderr
    assert _text(result.completed.stdout).strip() == "under-budget-ok"
    assert result.memory_limit_killed is False
    assert result.memory_limit_mechanism == "rlimit_data+rss_supervisor"


def test_sandboxed_popen_supervises_rlimit_data_fallback_directly(tmp_path, monkeypatch):
    report = policy.CapabilityReport("Linux", "test", True, True, True, True, True, True)
    monkeypatch.setattr(policy, "require_capabilities", lambda: report)
    monkeypatch.setattr(policy, "_command_cgroup", lambda *_args, **_kwargs: None)
    script = """import mmap, time
size = 2 * (32 << 20)
mapping = mmap.mmap(-1, size,
                    flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE)
for offset in range(0, size, 4096):
    mapping[offset:offset + 1] = b'x'
time.sleep(2.2)
"""
    process, _report = policy.sandboxed_popen(
        [sys.executable, "-c", script], tmp_path,
        memory_bytes=SHARED_MAPPING_LIMIT,
        file_bytes=2 * SHARED_MAPPING_LIMIT,
        isolation_builder=_identity_isolation,
    )
    try:
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == -signal.SIGKILL, stdout + stderr
        assert process.memory_limit_killed is True
        assert process.memory_limit_reason == "aggregate resident-memory limit exceeded"
        assert process.memory_limit_mechanism == "rlimit_data+rss_supervisor"
    finally:
        if process.poll() is None:
            policy._kill_group(process)
            process.wait(timeout=2)


def test_pid_namespace_double_fork_setsids_shared_memory_escape_is_killed(
        tmp_path, monkeypatch):
    report = policy.capability_report()
    if not report.available or report.backend != "bwrap":
        pytest.skip("Linux bwrap execution boundary is unavailable")
    if not hasattr(os, "memfd_create"):
        pytest.skip("os.memfd_create is unavailable")
    monkeypatch.setattr(policy, "_command_cgroup", lambda *_args, **_kwargs: None)
    script = f'''import mmap, os, time
first = os.fork()
if first == 0:
    second = os.fork()
    if second:
        os._exit(0)
    os.setsid()
    size = 2 * ({SHARED_MAPPING_LIMIT})
    fd = os.memfd_create('stateframe-double-fork-memory-probe')
    os.ftruncate(fd, size)
    mapping = mmap.mmap(fd, size, flags=mmap.MAP_SHARED,
                       prot=mmap.PROT_READ | mmap.PROT_WRITE)
    for offset in range(0, size, 4096):
        mapping[offset:offset + 1] = b'x'
    time.sleep(2.2)
    os._exit(0)
os.waitpid(first, 0)
time.sleep(3)
'''
    result = policy.run_sandboxed(
        [sys.executable, "-c", script], tmp_path,
        timeout=8, cpu_seconds=8, memory_bytes=SHARED_MAPPING_LIMIT,
        file_bytes=2 * SHARED_MAPPING_LIMIT, nproc=32,
    )

    assert result.completed.returncode == -signal.SIGKILL
    assert result.memory_limit_killed is True
    assert result.memory_limit_reason == "aggregate resident-memory limit exceeded"
