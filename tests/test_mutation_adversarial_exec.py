"""Adversarial acceptance tests for mutation containment and launch cleanup."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import time
import uuid

import pytest

from mutate_leftovers import make_launch_spec, mutation_root, stage_test_unit, sweep_and_assert


pytestmark = pytest.mark.host_sandbox


def _run(executor_api, wire, staged, runtime_root, *, input_value=None, limits=None,
         network=False, writes=False, call="case-1"):
    executor = executor_api.MutationExecutor(
        runtime_root, network_envelope="full" if network else "none")
    spec = make_launch_spec(executor_api, wire, staged, call=call, input_value=input_value,
                            network=network, writes=writes, limits=limits)
    try:
        result = executor.run(spec)
    finally:
        cleanup = executor.close()
    assert cleanup.ok, cleanup
    sweep_and_assert(runtime_root)
    return result


@pytest.mark.parametrize("level,network,writes", [
    ("off", "none", ()),
    pytest.param("prebuilt", "none", (), marks=pytest.mark.skipif(
        sys.platform != "linux", reason="Linux bubblewrap and seccomp are required for mutation containment")),
    pytest.param("authored", "none", (), marks=pytest.mark.skipif(
        sys.platform != "linux", reason="Linux bubblewrap and seccomp are required for mutation containment")),
    pytest.param("prebuilt", "public-web", (), marks=pytest.mark.skipif(
        sys.platform != "linux", reason="Linux bubblewrap and seccomp are required for mutation containment")),
    pytest.param("prebuilt", "none", ("src/**",), marks=pytest.mark.skipif(
        sys.platform != "linux", reason="Linux bubblewrap and seccomp are required for mutation containment")),
    pytest.param("prebuilt", "public-web", ("src/**",), marks=pytest.mark.skipif(
        sys.platform != "linux", reason="Linux bubblewrap and seccomp are required for mutation containment")),
])
@pytest.mark.parametrize("ordinary_shell_sandbox", [False, True])
def test_mutation_level_matrix_and_host_mode_never_escapes_sandbox(
        mutation_root, tmp_path, level, network, writes, ordinary_shell_sandbox):
    from kernel_mutate_policy import policy_from

    policy = policy_from({"level": level, "network": network, "domains": (),
                          "write_globs": writes, "read_globs": (), "retain": "run",
                          "audit": "source",
                          "shell_network": "full" if network == "public-web" else "none"})
    assert policy.enabled is (level != "off")
    assert policy.network == network
    assert policy.write_globs == writes

    from kernel_exec_capability import shell_capability
    from kernel_exec_session import ExecSession

    shell_root = tmp_path / "shell-run"
    workspace, env_dir, home = (tmp_path / name for name in ("workspace", "env", "home"))
    for path in (shell_root, workspace, env_dir, home):
        path.mkdir()
    shell_cap = shell_capability(workspace=workspace, run_env_dir=env_dir,
                                 scratch_home=home, network="none", writable=False)
    with ExecSession(shell_root) as shell_session:
        shell_result = shell_session.run([sys.executable, "-I", "-c", "pass"],
                                         replace(shell_cap, sandboxed=ordinary_shell_sandbox))
    assert shell_result.cleanup.ok
    # Ordinary shell sessions retain their launch pidfd until process GC.
    import gc
    del shell_session
    gc.collect()

    if level == "off":
        assert not mutation_root.joinpath("launches.jsonl").exists()
        return

    import kernel_mutate_build as build
    import kernel_mutate_exec as execution
    import kernel_mutate_wire as wire

    source = (
        "import os\n"
        "def handle(input, ctx):\n"
        "    try: pid = os.fork()\n"
        "    except OSError: return {'fork': 'denied'}\n"
        "    if pid == 0: os._exit(0)\n"
        "    os.waitpid(pid, 0)\n"
        "    return {'fork': 'allowed'}\n"
    )
    staged = stage_test_unit(build, mutation_root, source)
    outcome = _run(execution, wire, staged, mutation_root,
                   network=(network == "public-web"),
                   limits=execution.MutationLimits(wall_s=10, cpu_s=2, mem_mb=128))
    assert outcome.exit_code == 0
    assert outcome.cleanup.ok
    assert not outcome.stdout_overflow
    assert wire.parse_stdout(outcome.stdout, call="case-1").result == {"fork": "denied"}


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="Linux bubblewrap mount isolation tests host path denial",
)
def test_mutation_cannot_read_engine_home_secrets_or_other_run(mutation_root, tmp_path):
    import kernel_mutate_build as build
    import kernel_mutate_exec as execution
    import kernel_mutate_wire as wire

    private = tmp_path / "private"
    other_run = tmp_path / "other-run"
    private.mkdir()
    other_run.mkdir()
    home_secret = private / "credentials"
    run_secret = other_run / "run-secret"
    engine_secret = tmp_path / "engine-source.py"
    for path in (home_secret, run_secret, engine_secret):
        path.write_text("MUTATION_SECRET_CANARY", encoding="utf-8")
    source = (
        "def handle(input, ctx):\n"
        "    found = []\n"
        "    for path in input['forbidden']:\n"
        "        try:\n"
        "            found.append('MUTATION_SECRET_CANARY' in open(path).read())\n"
        "        except OSError:\n"
        "            found.append(False)\n"
        "    return {'read': found}\n"
    )
    staged = stage_test_unit(build, mutation_root, source)
    outcome = _run(execution, wire,
                   staged, mutation_root,
                   input_value={"forbidden": [str(home_secret), str(run_secret), str(engine_secret)]})
    frame = wire.parse_stdout(outcome.stdout, call="case-1")
    assert frame.kind == "result"
    assert frame.result == {"read": [False, False, False]}
    assert "MUTATION_SECRET_CANARY" not in outcome.stdout.decode("utf-8", "replace")


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="Linux seccomp/bubblewrap test blocked network and process access",
)
def test_mutation_network_none_blocks_dns_ip_unix_abstract_and_subprocess(mutation_root):
    import kernel_mutate_build as build
    import kernel_mutate_exec as execution
    import kernel_mutate_wire as wire

    source = (
        "import socket, subprocess, sys\n"
        "def denied(action):\n"
        "    try:\n"
        "        action()\n"
        "    except (OSError, RuntimeError):\n"
        "        return True\n"
        "    return False\n"
        "def handle(input, ctx):\n"
        "    checks = {\n"
        "      'dns': denied(lambda: socket.getaddrinfo('stateframe.invalid', 80)),\n"
        "      'ip': denied(lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM)),\n"
        "      'unix': denied(lambda: socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)),\n"
        "      'abstract': denied(lambda: socket.socket(socket.AF_UNIX, socket.SOCK_STREAM).bind('\\0sf-mutate-test')),\n"
        "      'subprocess': denied(lambda: subprocess.run([sys.executable, '-c', 'pass'], timeout=1)),\n"
        "    }\n"
        "    return checks\n"
    )
    staged = stage_test_unit(build, mutation_root, source)
    outcome = _run(execution, wire, staged, mutation_root)
    frame = wire.parse_stdout(outcome.stdout, call="case-1")
    assert frame.kind == "result"
    assert set(frame.result) == {"dns", "ip", "unix", "abstract", "subprocess"}
    assert all(frame.result.values()), frame.result
    assert outcome.cleanup.ok


@pytest.mark.parametrize("files", [
    {"main.py": b"def handle(input, ctx): return {}\n",
     "../escape.py": b"def helper(): return None\n"},
    {"main.py": b"def handle(input, ctx): return {}\n",
     "json.py": b"def helper(): return None\n"},
])
def test_mutation_path_symlink_hardlink_and_swap_races_fail_closed(mutation_root, tmp_path, files):
    import kernel_mutate_build as build

    manifest = {
        "schema": "stateframe-mutation-unit/1", "name": "path_probe", "revision": 1,
        "parent": None, "entry": "main.py", "summary": "Path probe", "kind": "reader",
        "input_schema": {"type": "object"}, "output_schema": {"type": "object"},
        "requires": {"network": False, "writes": False}, "smoke": [],
        "limits": {"wall_s": 1, "cpu_s": 1, "mem_mb": 128},
    }
    with pytest.raises((ValueError, RuntimeError)):
        build.stage_unit(files, manifest, mutation_root)
    assert not (mutation_root.parent / "escape.py").exists()

    from kernel_mutate_policy import policy_from
    from kernel_mutate_runtime import admit_inputs

    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    source = repo / "src" / "input.txt"
    source.write_text("ordinary input", encoding="utf-8")
    hardlink = repo / "src" / "hardlink.txt"
    outside_secret = tmp_path / "outside-secret.txt"
    outside_secret.write_text("outside input root", encoding="utf-8")
    hardlink.hardlink_to(outside_secret)
    symlink = repo / "src" / "symlink.txt"
    symlink.symlink_to(source.name)
    policy = policy_from({"level": "authored", "network": "none", "domains": (),
                          "write_globs": (), "read_globs": ("**/*",), "retain": "run",
                          "audit": "source"})
    with pytest.raises((PermissionError, ValueError, RuntimeError, OSError)):
        admit_inputs(repo, ["src/hardlink.txt"], policy,
                     snapshot_root=tmp_path / "hardlink-snapshot")
    with pytest.raises((PermissionError, ValueError, RuntimeError, OSError)):
        admit_inputs(repo, ["src/symlink.txt"], policy,
                     snapshot_root=tmp_path / "symlink-snapshot")

    moved = repo / "moved"
    moved.mkdir()
    original_bytes = source.read_bytes()

    def swap_parent():
        source.rename(moved / source.name)
        hardlink.unlink()
        symlink.unlink()
        (repo / "src").rmdir()
        (repo / "src").symlink_to(moved, target_is_directory=True)

    with pytest.raises((PermissionError, ValueError, RuntimeError, OSError)):
        admit_inputs(repo, ["src/input.txt"], policy, before_open=swap_parent,
                     snapshot_root=tmp_path / "swap-snapshot")
    assert (moved / source.name).read_bytes() == original_bytes
    assert (repo / "src").is_symlink()
    assert (repo / "src" / "input.txt").read_bytes() == original_bytes


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="Linux bubblewrap/seccomp isolate payload execution",
)
def test_mutation_validated_bytes_equal_executed_bytes(mutation_root):
    import kernel_mutate_build as build
    import kernel_mutate_exec as execution
    import kernel_mutate_wire as wire

    original = "def handle(input, ctx):\n    return {'value': 'validated'}\n"
    staged = stage_test_unit(build, mutation_root, original,
                             limits={"wall_s": 10, "cpu_s": 5, "mem_mb": 256})
    executor = execution.MutationExecutor(mutation_root, network_envelope="none")
    try:
        receipt = build.validate(staged, executor)
        assert receipt.ok
        first = executor.run(make_launch_spec(execution, wire, staged, call="validated"))
        assert wire.parse_stdout(first.stdout, call="validated").result == {"value": "validated"}
        target = staged.dir / "main.py"
        target.chmod(0o600)
        target.write_text("def handle(input, ctx):\n    return {'value': 'changed'}\n", encoding="utf-8")
        with pytest.raises((ValueError, RuntimeError, execution.MutationContainmentError)):
            executor.run(make_launch_spec(execution, wire, staged, call="tampered"))
    finally:
        cleanup = executor.close()
        assert cleanup.ok
        sweep_and_assert(mutation_root)


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="Linux PID namespace/pidfd cleanup check process escapes",
)
def test_mutation_fork_setsid_daemon_zombie_and_inherited_fd_cleanup(mutation_root):
    import kernel_mutate_build as build
    import kernel_mutate_exec as execution
    import kernel_mutate_wire as wire

    source = (
        "import os\n"
        "def handle(input, ctx):\n"
        "    results = {}\n"
        "    for name, action in [('fork', os.fork), ('setsid', os.setsid), ('double_fork', os.fork), ('pipe_holder', os.fork)]:\n"
        "        try:\n"
        "            value = action()\n"
        "            if name in ('fork', 'double_fork', 'pipe_holder') and value == 0:\n"
        "                os.setsid()\n"
        "                second = os.fork()\n"
        "                if second == 0:\n"
        "                    import time; time.sleep(5); os._exit(0)\n"
        "                os._exit(0)\n"
        "            if name in ('double_fork', 'pipe_holder'): os.waitpid(value, 0)\n"
        "            results[name] = 'allowed'\n"
        "        except OSError:\n"
        "            results[name] = 'denied'\n"
        "    fds = {}\n"
        "    if os.path.isdir('/proc/self/fd'):\n"
        "        for fd in os.listdir('/proc/self/fd'):\n"
        "            try: os.fstat(int(fd)); fds[fd] = os.readlink('/proc/self/fd/' + fd)\n"
        "            except OSError: pass\n"
        "    results['fds'] = fds\n"
        "    return results\n"
    )
    staged = stage_test_unit(build, mutation_root, source)
    outcome = _run(execution, wire, staged, mutation_root)
    frame = wire.parse_stdout(outcome.stdout, call="case-1")
    assert frame.result["fork"] == "denied"
    assert frame.result["setsid"] == "denied"
    assert frame.result["double_fork"] == "denied"
    assert frame.result["pipe_holder"] == "denied"
    assert set(frame.result["fds"]) <= {"0", "1", "2"}, frame.result["fds"]


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="Linux cgroups/PID namespace/seccomp enforce process burst limits",
)
def test_mutation_fork_bomb_memory_cpu_and_process_burst_limits(mutation_root):
    import kernel_mutate_build as build
    import kernel_mutate_exec as execution
    import kernel_mutate_wire as wire

    source = (
        "import os, time\n"
        "def handle(input, ctx):\n"
        "    children = []\n"
        "    for _ in range(8):\n"
        "        try:\n"
        "            pid = os.fork()\n"
        "        except OSError:\n"
        "            break\n"
        "        if pid == 0: time.sleep(0.2); os._exit(0)\n"
        "        children.append(pid)\n"
        "    for pid in children: os.waitpid(pid, 0)\n"
        "    return {'forks': len(children)}\n"
    )
    staged = stage_test_unit(build, mutation_root, source)
    limits = execution.MutationLimits(wall_s=1.5, cpu_s=0.5, mem_mb=64, output_bytes=4096)
    outcome = _run(execution, wire, staged, mutation_root, limits=limits)
    assert outcome.cleanup.ok
    assert outcome.exit_code == 0
    assert outcome.wall_s < 1.5
    assert wire.parse_stdout(outcome.stdout, call="case-1").result == {"forks": 0}


@pytest.mark.parametrize("payload,limits", [
    ("import time\\ndef handle(input, ctx):\\n    block = bytearray(96 << 20)\\n    time.sleep(5)\\n    return {'size': len(block)}\\n",
     (1.5, 0.5, 32)),
    ("def handle(input, ctx):\\n    value = 0\\n    while True: value += 1\\n",
     (2, 0.25, 128)),
])
@pytest.mark.skipif(sys.platform != "linux", reason="Linux cgroups and seccomp are required for resource burst limits")
def test_mutation_resource_bursts_hit_memory_and_cpu_caps(mutation_root, payload, limits):
    import kernel_mutate_build as build
    import kernel_mutate_exec as execution
    import kernel_mutate_wire as wire

    source = payload.replace("\\n", "\n")
    staged = stage_test_unit(build, mutation_root, source)
    wall_s, cpu_s, mem_mb = limits
    outcome = _run(execution, wire, staged, mutation_root,
                   limits=execution.MutationLimits(wall_s=wall_s, cpu_s=cpu_s,
                                                   mem_mb=mem_mb, output_bytes=4096))
    assert outcome.cleanup.ok
    assert outcome.exit_code != 0 or outcome.timed_out


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="Linux bubblewrap/seccomp enforce the owner envelope",
)
def test_mutation_owner_envelope_never_widens(mutation_root):
    from dataclasses import replace
    from kernel_exec_session import ExecSessionError
    import kernel_mutate_build as build
    import kernel_mutate_exec as execution
    import kernel_mutate_wire as wire

    staged = stage_test_unit(build, mutation_root,
                             "def handle(input, ctx):\n    return {'ok': True}\n")
    executor = execution.MutationExecutor(mutation_root, network_envelope="none")
    try:
        offline = make_launch_spec(execution, wire, staged, call="owner-offline")
        assert executor.run(offline).cleanup.ok
        widened = make_launch_spec(execution, wire, staged, call="owner-network", network=True)
        with pytest.raises((ValueError, RuntimeError, execution.MutationPlatformError)):
            executor.run(widened)
        cap = executor._session._initial_capability
        outside = mutation_root.parents[1] / "read-widening"
        outside.mkdir()
        with pytest.raises(ExecSessionError, match="read authority"):
            executor._session._start(replace(cap, read_roots=(*cap.read_roots, outside)))
    finally:
        cleanup = executor.close()
    assert cleanup.ok
    sweep_and_assert(mutation_root)


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="Linux PID namespace/cgroups contain fork bombs",
)
def test_mutation_mike_fork_bomb(mutation_root):
    import kernel_mutate_build as build
    import kernel_mutate_exec as execution
    import kernel_mutate_wire as wire

    source = (
        "import os, time\n"
        "def handle(input, ctx):\n"
        "    children = []\n"
        "    for _ in range(16):\n"
        "        try: pid = os.fork()\n"
        "        except OSError: break\n"
        "        if pid == 0: time.sleep(0.05); os._exit(0)\n"
        "        children.append(pid)\n"
        "    for pid in children: os.waitpid(pid, 0)\n"
        "    return {'children': len(children)}\n"
    )
    staged = stage_test_unit(build, mutation_root, source)
    outcome = _run(execution, wire, staged, mutation_root,
                   limits=execution.MutationLimits(wall_s=1.5, cpu_s=1, mem_mb=128,
                                                   output_bytes=4096))
    assert outcome.cleanup.ok
    assert outcome.exit_code == 0
    assert wire.parse_stdout(outcome.stdout, call="case-1").result["children"] == 0


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="Linux tmpfs mounts/cgroups enforce workspace quotas",
)
def test_mutation_disk_fill_tmp_workspace_and_shared_memory_limits(mutation_root, tmp_path):
    import kernel_mutate_build as build
    import kernel_mutate_exec as execution
    import kernel_mutate_wire as wire

    source = (
        "def handle(input, ctx):\n"
        "    results = {}\n"
        "    for name, path in input['paths'].items():\n"
        "        try:\n"
        "            with open(path, 'wb') as output: output.write(b'x' * (2 << 20))\n"
        "            results[name] = 'wrote'\n"
        "        except OSError:\n"
        "            results[name] = 'denied'\n"
        "    return results\n"
    )
    host_write_dir = tmp_path / "host-visible"
    host_write_dir.mkdir()
    paths = {"tmp": f"/tmp/stateframe-mutate-disk-fill-{uuid.uuid4().hex}",
             "repo": str(host_write_dir / "write"),
             "shm": f"/dev/shm/stateframe-mutate-disk-fill-{uuid.uuid4().hex}"}
    staged = stage_test_unit(build, mutation_root, source)
    outcome = _run(execution, wire, staged, mutation_root, input_value={"paths": paths})
    frame = wire.parse_stdout(outcome.stdout, call="case-1")
    assert frame.result == {"tmp": "denied", "repo": "denied", "shm": "denied"}
    assert not Path(paths["tmp"]).exists()
    assert not Path(paths["shm"]).exists()


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="Linux seccomp/cgroups bound payload output",
)
def test_mutation_gigabyte_output_and_truncation_are_bounded(mutation_root):
    import kernel_mutate_build as build
    import kernel_mutate_exec as execution
    import kernel_mutate_wire as wire

    source = (
        "import os\n"
        "def handle(input, ctx):\n"
        "    block = b'x' * (1 << 20)\n"
        "    for _ in range(1024): os.write(1, block)\n"
        "    return {'finished': True}\n"
    )
    staged = stage_test_unit(build, mutation_root, source)
    limits = execution.MutationLimits(wall_s=3, cpu_s=2, mem_mb=128, output_bytes=1 << 20)
    outcome = _run(execution, wire, staged, mutation_root, limits=limits)
    assert outcome.stdout_overflow
    assert len(outcome.stdout) <= limits.output_bytes
    assert outcome.exit_code != 0
    assert outcome.wall_s < limits.wall_s
    assert outcome.cleanup.ok


def test_payload_wall_starts_after_slow_registered_launch(mutation_root):
    import kernel_mutate_build as build
    import kernel_mutate_exec as execution
    import kernel_mutate_wire as wire

    staged = stage_test_unit(build, mutation_root,
                             "import time\ndef handle(input, ctx):\n"
                             "    time.sleep(0.1)\n    return {'finished': True}\n")
    def delay_release(state):
        if state == "REGISTERED":
            time.sleep(2.5)

    executor = execution.MutationExecutor(mutation_root, network_envelope="none",
                                         launch_stage_hook=delay_release)
    spec = make_launch_spec(execution, wire, staged,
                            limits=execution.MutationLimits(wall_s=2, cpu_s=2, mem_mb=128))
    try:
        outcome = executor.run(spec)
    finally:
        assert executor.close().ok
    sweep_and_assert(mutation_root)
    assert outcome.exit_code == 0
    assert not outcome.timed_out
    assert wire.parse_stdout(outcome.stdout, call="case-1").result == {"finished": True}


@pytest.mark.skipif(sys.platform != "linux", reason="Linux bubblewrap is required for mutation startup cleanup")
def test_stalled_startup_refuses_before_release_and_reaps_everything(mutation_root, monkeypatch):
    import json
    import kernel_mutate_build as build
    import kernel_mutate_exec as execution
    import kernel_mutate_wire as wire

    sentinel = mutation_root.parent / "payload-ran"
    source = ("from pathlib import Path\n"
              f"Path({str(sentinel)!r}).write_text('ran')\n"
              "def handle(input, ctx):\n    return {'finished': True}\n")
    staged = stage_test_unit(build, mutation_root, source)
    monkeypatch.setattr(execution, "rpc_timeout_seconds", lambda: 1.5)

    def stall_before_release(state):
        if state == "REGISTERED":
            time.sleep(1.25)

    executor = execution.MutationExecutor(mutation_root, network_envelope="none",
                                         launch_stage_hook=stall_before_release)
    spec = make_launch_spec(execution, wire, staged)
    with pytest.raises(execution.MutationStartupTimeout, match="startup exceeded bound"):
        executor.run(spec)
    assert not sentinel.exists()
    states = [json.loads(line)["state"] for line in
              (mutation_root / "launches.jsonl").read_text().splitlines()]
    assert "REGISTERED" in states
    assert "RELEASED" not in states
    assert states[-1] == "REAPED"
    assert executor._closed and executor._ledger.closed
    assert executor._session._closed
    assert executor._session._control.fileno() == -1
    assert executor._session._process.poll() is not None
    assert not executor._stage.exists()
    assert not executor._boot.exists()
    assert not executor._session_dir.exists()
    sweep_and_assert(mutation_root)


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="Linux pidfd signaling/PID namespace cleanup stop runaway payloads",
)
def test_mutation_infinite_loop_is_killed_with_verified_cleanup(mutation_root):
    import kernel_mutate_build as build
    import kernel_mutate_exec as execution
    import kernel_mutate_wire as wire

    source = "import time\ndef handle(input, ctx):\n    while True:\n        time.sleep(0.01)\n"
    staged = stage_test_unit(build, mutation_root, source)
    limits = execution.MutationLimits(wall_s=0.5, cpu_s=0.25, mem_mb=128, output_bytes=4096)
    outcome = _run(execution, wire, staged, mutation_root, limits=limits)
    assert outcome.timed_out
    assert outcome.cleanup.ok
    assert outcome.wall_s < 2
