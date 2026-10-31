from dataclasses import replace
import os
import sys
import time

import pytest

from kernel_exec_capability import shell_capability
from kernel_exec_session import ExecSession

pytestmark = pytest.mark.host_sandbox

# Sandboxed targets run the real interpreter: the venv symlink lies outside the declared read roots.
PYTHON = os.path.realpath(sys.executable)


@pytest.fixture
def setup(tmp_path):
    for name in ("workspace", "env", "home", "run"):
        (tmp_path / name).mkdir()
    cap = shell_capability(workspace=tmp_path / "workspace", run_env_dir=tmp_path / "env",
                           scratch_home=tmp_path / "home", network="none", writable=True)
    return tmp_path / "run", cap


def script(cap, text):
    path = cap.workspace / "command.py"
    path.write_text(text)
    return [PYTHON, "-I", str(path)]


@pytest.mark.parametrize("sandboxed", [False, True])
def test_environment_stdin_and_exit(setup, sandboxed):
    run, cap = setup
    cap = replace(cap, sandboxed=sandboxed)
    with ExecSession(run) as session:
        result = session.run(["/usr/bin/env"], cap)
        assert result.exit_code == 0, result.output
        assert dict(line.split("=", 1) for line in result.output.decode().splitlines()) == dict(cap.env)
        assert result.cleanup.ok
        result = session.run(["/bin/cat"], cap)
        assert result.exit_code == 0 and result.output == b"" and result.cleanup.ok
        result = session.run(["/bin/cat"], cap, stdin=b"explicit input")
        assert result.output == b"explicit input"


@pytest.mark.parametrize("sandboxed", [False, True])
def test_output_flood_is_bounded(setup, sandboxed):
    run, cap = setup
    cap = replace(cap, sandboxed=sandboxed)
    command = script(cap, 'import os\nfor _ in range(800): os.write(1, b"x" * 65536)\n')
    with ExecSession(run) as session:
        result = session.run(command, cap)
    assert result.exit_code == 0, result.output
    assert result.output_total_bytes == 50 * 1024 * 1024
    assert len(result.output) <= 256 * 1024 and result.output_truncated
    assert result.cleanup.ok
    assert max(p.stat().st_size for p in run.glob("exec-*/*.output")) <= 1 << 20


@pytest.mark.parametrize("sandboxed", [False, True])
@pytest.mark.parametrize("leader_exits", [False, True])
def test_escaped_child_is_killed_after_leader_exit_or_timeout(setup, leader_exits, sandboxed):
    run, cap = setup
    cap = replace(cap, sandboxed=sandboxed, limits=replace(cap.limits, nproc=4096))
    command = script(cap, '''import os, signal, time
pid = os.fork()
if pid == 0:
    os.setsid()
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    print(os.getpid(), flush=True)
    time.sleep(30)
else:
    time.sleep(0.5)
''' + ("    os._exit(0)\n" if leader_exits else "    time.sleep(30)\n"))
    cap = replace(cap, limits=replace(cap.limits, wall_seconds=1.5))
    with ExecSession(run) as session:
        result = session.run(command, cap)
    assert result.cleanup.ok, result.cleanup
    assert result.cleanup.killed >= 1
    assert result.timed_out is (not leader_exits)
    pid = int(result.output.strip())
    # Linux PID is namespace-local and cannot be probed from the host.
    if sys.platform == "darwin":
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


@pytest.mark.parametrize("sandboxed", [False, True])
def test_spawn_read_wait_stop_and_leader_independent_cleanup(setup, sandboxed):
    run, cap = setup
    command = script(cap, 'import time\nprint("READY", flush=True)\ntime.sleep(30)\n')
    cap = replace(cap, purpose="job", workspace_writable=False, sandboxed=sandboxed)
    with ExecSession(run) as session:
        job = session.spawn(command, cap)
        output = b""
        deadline = time.monotonic() + 5
        while b"READY" not in output and time.monotonic() < deadline:
            output += job.read_available()
            time.sleep(0.02)
        assert b"READY" in output
        assert job.poll() is None
        assert job.wait(0.01) is None
        report = job.stop(0.1)
        assert report.ok and report.killed >= 1
        assert job.poll() is not None
        assert job.stop().ok


@pytest.mark.parametrize("sandboxed", [False, True])
def test_small_bounded_fork_limit(setup, sandboxed):
    run, cap = setup
    command = script(cap, '''import os, time
children = []
try:
    for _ in range(12):
        try: pid = os.fork()
        except OSError:
            print("bounded", flush=True)
            break
        if pid == 0:
            time.sleep(5)
            os._exit(0)
        children.append(pid)
    else:
        time.sleep(5)
        print("limit-not-enforced", flush=True)
finally:
    for pid in children:
        try: os.kill(pid, 9)
        except ProcessLookupError: pass
    for pid in children: os.waitpid(pid, 0)
''')
    cap = replace(cap, sandboxed=sandboxed, limits=replace(cap.limits, nproc=8, wall_seconds=10))
    with ExecSession(run) as session:
        result = session.run(command, cap)
    assert result.cleanup.ok
    assert b"limit-not-enforced" not in result.output
    if sys.platform == "darwin":
        assert "process limit" in result.cleanup.detail
        assert result.cleanup.killed >= 1
    # Linux counts processes under the UID and can refuse the interpreter itself.
    assert b"bounded" in result.output or result.exit_code != 0


@pytest.mark.parametrize("sandboxed", [False, True])
def test_resident_memory_limit_stops_process(setup, sandboxed):
    run, cap = setup
    command = script(cap, '''import time
blocks = []
for _ in range(8):
    blocks.append(bytearray(16 * 1024 * 1024))
    time.sleep(0.1)
time.sleep(20)
''')
    cap = replace(cap, sandboxed=sandboxed, limits=replace(cap.limits, memory_bytes=64 << 20))
    with ExecSession(run) as session:
        result = session.run(command, cap)
    assert result.cleanup.ok
    assert result.exit_code != 0
    assert result.wall_seconds < 5
    if sys.platform == "darwin":
        assert "resident-memory" in result.cleanup.detail


@pytest.mark.parametrize("sandboxed", [False, True])
def test_engine_death_stops_supervised_job(setup, sandboxed):
    import json
    import subprocess
    from kernel_exec_tracker import ProcessTable

    run, cap = setup
    cap = replace(cap, sandboxed=sandboxed, workspace_writable=False,
                  limits=replace(cap.limits, nproc=4096))
    child = script(cap, '''import os, signal, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
print(os.getpid(), flush=True)
time.sleep(30)
''')
    owner = run / "owner.py"
    engine_path = str(__import__("kernel_exec_session").__file__.rsplit("/", 1)[0])
    owner.write_text(f'''import json, sys, time
from pathlib import Path
sys.path.insert(0, {engine_path!r})
from kernel_exec_capability import capability_from_dict
from kernel_exec_session import ExecSession
session = ExecSession(Path({str(run)!r}))
cap = capability_from_dict(json.loads({json.dumps(cap.as_dict())!r}))
job = session.spawn({child!r}, cap)
data = b""
while b"\\n" not in data:
    data += job.read_available()
    time.sleep(0.01)
print(json.dumps({{"job": int(data.strip()), "supervisor": session._process.pid}}), flush=True)
time.sleep(30)
''')
    process = subprocess.Popen([sys.executable, "-I", str(owner)], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    try:
        import selectors
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            assert selector.select(8), "owner did not publish job pid"
        line = process.stdout.readline()
        assert line.startswith(b"{"), process.stderr.read() if process.poll() is not None else line
        pids = json.loads(line)
        table = ProcessTable()
        watched = [pids["supervisor"]]
        if sys.platform == "darwin" or not sandboxed:
            watched.append(pids["job"])
        identities = [table.identity(pid) for pid in watched]
        assert all(identity is not None for identity in identities)
        process.kill()
        process.wait(timeout=3)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            alive = []
            for identity in identities:
                current = table.identity(identity.pid)
                if current is not None and not current.zombie and current.start == identity.start:
                    alive.append(current.pid)
            if not alive:
                break
            time.sleep(0.02)
        else:
            raise AssertionError(f"execution processes survived engine death: {alive}")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
        process.stdout.close()
        process.stderr.close()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux namespace narrowing")
def test_custom_loopback_narrows_full_session_network(setup):
    import socket
    run, cap = setup
    full = replace(cap, network="full")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        code = ("import socket\n"
                "s=socket.socket(); s.bind(('127.0.0.1',0)); s.close()\n"
                "try:\n"
                f" s=socket.create_connection(('127.0.0.1',{port}),.5)\n"
                "except OSError:\n print('isolated loopback')\n"
                "else:\n s.close(); raise RuntimeError('host network leaked')\n")
        argv = script(cap, code)
        with ExecSession(run) as session:
            session.configure(full)
            narrowed = replace(full, purpose="custom_tool", network="loopback", workspace_writable=False)
            result = session.run(argv, narrowed)
            assert result.exit_code == 0 and result.cleanup.ok, result.output
            assert b"isolated loopback" in result.output
