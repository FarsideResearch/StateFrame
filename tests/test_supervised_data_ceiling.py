"""The RLIMIT_DATA ceiling rises only where the resident-memory supervisor enforces the budget."""
from __future__ import annotations

import os
import resource
import shutil
import subprocess
import sys
import time

import pytest

import kernel_execution_policy as policy

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux memory policy")
BUDGET = 64 << 20
CEILING = policy.SUPERVISED_DATA_CEILING_BYTES
PRINT_LIMIT = "import resource; print(resource.getrlimit(resource.RLIMIT_DATA)[0])"


def _identity(command, _root, environment, **_options):
    return list(command), dict(environment)


def _run(argv, root, monkeypatch, *, memory_bytes=BUDGET, file_bytes=1 << 20):
    report = policy.CapabilityReport("Linux", "test", True, True, True, True, True, True)
    monkeypatch.setattr(policy, "require_capabilities", lambda: report)
    monkeypatch.setattr(policy, "_command_cgroup", lambda *_a, **_k: None)
    popen = policy.sandboxed_popen
    monkeypatch.setattr(policy, "sandboxed_popen", lambda command, cwd, **options: popen(
        command, cwd, isolation_builder=_identity, **options))
    return policy.run_sandboxed(argv, root, timeout=30, cpu_seconds=20,
                                memory_bytes=memory_bytes, file_bytes=file_bytes, nproc=64)


def _inherited(value):
    hard = resource.getrlimit(resource.RLIMIT_DATA)[1]
    return value if hard == resource.RLIM_INFINITY else min(value, hard)


def test_supervised_check_gets_the_ceiling_and_keeps_its_budget(tmp_path, monkeypatch):
    result = _run([sys.executable, "-c", PRINT_LIMIT], tmp_path, monkeypatch)
    assert result.completed.returncode == 0, result.completed.stderr
    assert int(result.completed.stdout) == _inherited(CEILING)
    assert result.memory_limit_mechanism == "rlimit_data+rss_supervisor"


def test_budget_above_the_ceiling_is_not_reduced(tmp_path, monkeypatch):
    result = _run([sys.executable, "-c", PRINT_LIMIT], tmp_path, monkeypatch,
                  memory_bytes=CEILING + (1 << 30))
    assert int(result.completed.stdout) == _inherited(CEILING + (1 << 30))


def test_unsupervised_preexec_keeps_the_budget_as_the_limit():
    output = subprocess.run([sys.executable, "-c", PRINT_LIMIT], capture_output=True, check=True,
                            preexec_fn=policy._resource_preexec(10, BUDGET, 1 << 20, 128, 64))
    assert int(output.stdout) == BUDGET


def test_custom_spawn_without_a_supervisor_keeps_the_budget(tmp_path, monkeypatch):
    report = policy.CapabilityReport("Linux", "test", True, True, True, True, True, True)
    monkeypatch.setattr(policy, "require_capabilities", lambda: report)

    def spawn(*args, **kwargs):
        return subprocess.Popen(*args, **kwargs)

    process, _ = policy.sandboxed_popen([sys.executable, "-c", PRINT_LIMIT], tmp_path,
                                        memory_bytes=BUDGET, isolation_builder=_identity,
                                        popen_factory=spawn)
    out, _err = process.communicate()
    assert not hasattr(process, "_stateframe_memory_supervisor")
    assert int(out) == BUDGET


def test_supervisor_start_failure_stops_the_child(tmp_path, monkeypatch):
    report = policy.CapabilityReport("Linux", "test", True, True, True, True, True, True)
    monkeypatch.setattr(policy, "require_capabilities", lambda: report)
    seen = []

    class Broken:
        def __init__(self, process, _budget):
            seen.append(process)
            raise RuntimeError("no supervisor")

    monkeypatch.setattr(policy, "_ResidentMemorySupervisor", Broken)
    with pytest.raises(policy.ExecutionPolicyError):
        policy.sandboxed_popen([sys.executable, "-c", "import time; time.sleep(60)"], tmp_path,
                               memory_bytes=BUDGET, isolation_builder=_identity)
    assert seen[0].wait(timeout=5) != 0


def test_supervisor_that_dies_on_an_unexpected_error_stops_the_child(tmp_path, monkeypatch):
    monkeypatch.setattr(policy, "_note_peak_process_cpu",
                        lambda *_a: (_ for _ in ()).throw(KeyError("unexpected")))
    started = time.monotonic()
    with pytest.raises(Exception):
        result = _run([sys.executable, "-c", "import time; time.sleep(25)"], tmp_path, monkeypatch)
        if result.completed.returncode != 0:
            raise RuntimeError("stopped")
    assert time.monotonic() - started < 20


PRIVATE = """b = bytearray({n})
for i in range(0, len(b), 4096): b[i] = 1
import time; time.sleep(20)
"""
SHARED = """import mmap
m = mmap.mmap(-1, {n}, flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS)
for i in range(0, {n}, 4096): m[i] = 1
import time; time.sleep(20)
"""
MEMFD = """import mmap, os
fd = os.memfd_create('x'); os.ftruncate(fd, {n})
m = mmap.mmap(fd, {n})
for i in range(0, {n}, 4096): m[i] = 1
import time; time.sleep(20)
"""
CHILDREN = """import os, time
for _ in range(4):
    if not os.fork():
        b = bytearray({n} // 3)
        for i in range(0, len(b), 4096): b[i] = 1
        time.sleep(20); os._exit(0)
time.sleep(20)
"""


@pytest.mark.parametrize("script", [PRIVATE, SHARED, MEMFD, CHILDREN],
                         ids=["private", "shared", "memfd", "children"])
def test_program_over_its_resident_budget_is_stopped(script, tmp_path, monkeypatch):
    size = 256 << 20
    result = _run([sys.executable, "-c", script.format(n=size)], tmp_path, monkeypatch,
                  file_bytes=64 << 20) if script is not MEMFD else _run(
        [sys.executable, "-c", script.format(n=60 << 20)], tmp_path, monkeypatch,
        memory_bytes=48 << 20, file_bytes=64 << 20)
    assert result.completed.returncode != 0
    assert result.memory_limit_killed is True
    assert result.memory_limit_reason == "aggregate resident-memory limit exceeded"


def test_node_starts_under_the_default_check_budget(tmp_path, monkeypatch):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    result = _run([os.path.realpath(node), "-e", "console.log('node-ok')"], tmp_path, monkeypatch,
                  memory_bytes=policy.DEFAULT_MEMORY_BYTES)
    assert result.completed.returncode == 0, result.completed.stderr
    assert "node-ok" in str(result.completed.stdout)
    assert result.memory_limit_killed is False


@pytest.mark.host_sandbox
def test_node_starts_in_a_real_outer_check(tmp_path):
    from stateframe import outer_checks
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README").write_text("x\n")
    result = outer_checks.run(f"{node} -e 0", repo, python=sys.executable, runner_kind="generic")
    assert result["exit_code"] == 0, result["tail"]
