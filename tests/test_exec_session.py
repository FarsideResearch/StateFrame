from dataclasses import replace
import errno
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from kernel_exec_capability import shell_capability
from kernel_exec_session import ExecSession
from kernel_exec_capability import executable_read_roots
from kernel_test_isolation import _bwrap_userns_guard_supported, sandbox_executable
from fixtures.process_signals import (
    close_signal_binding,
    record_identity,
    retain_signal_binding,
    signal_bound,
    signal_owned,
)

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


def _wait_for_stable_file(path, *, stable_seconds=0.4, timeout=8.0):
    deadline = time.monotonic() + timeout
    previous_size = path.stat().st_size
    unchanged_since = time.monotonic()
    while time.monotonic() < deadline:
        time.sleep(0.03)
        current_size = path.stat().st_size
        if current_size != previous_size:
            previous_size = current_size
            unchanged_since = time.monotonic()
        elif time.monotonic() - unchanged_since >= stable_seconds:
            return True
    return False


def _wait_until_gone(identity, timeout=5.0):
    from kernel_exec_tracker import ProcessTable

    table = ProcessTable()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = table.identity(identity.pid)
        if current is None or current.start != identity.start or current.zombie:
            return True
        time.sleep(0.02)
    current = table.identity(identity.pid)
    return current is None or current.start != identity.start or current.zombie


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
    beat = cap.workspace / "heartbeat"
    command = script(cap, f'''import os, signal, time
pid = os.fork()
if pid == 0:
    os.setsid()
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    print(os.getpid(), flush=True)
    for _ in range(300):
        with open({str(beat)!r}, "a") as handle:
            handle.write(".")
        time.sleep(0.1)
else:
    time.sleep(0.5)
''' + ("    os._exit(0)\n" if leader_exits else "    time.sleep(30)\n"))
    cap = replace(cap, limits=replace(cap.limits, wall_seconds=1.5))
    with ExecSession(run) as session:
        result = session.run(command, cap)
    assert result.cleanup.ok, result.cleanup
    assert result.timed_out is (not leader_exits)
    # The escaped child must be dead, whoever killed it: its heartbeat stops growing.
    size = beat.stat().st_size
    time.sleep(0.6)
    assert beat.stat().st_size == size, "escaped child is still running after cleanup"
    if sandboxed and sys.platform.startswith("linux") and leader_exits:
        # Each Linux command has its own PID namespace: when the leader exits, the namespace
        # ends and the kernel kills the escaped child, so the supervisor may count no kill.
        pass
    else:
        assert result.cleanup.killed >= 1
    pid = int(result.output.strip())
    # Linux PID is namespace-local and cannot be probed from the host.
    if sys.platform == "darwin":
        probe = subprocess.run(["ps", "-p", str(pid), "-o", "pid="],
                               capture_output=True, text=True, check=False)
        assert probe.returncode != 0 or not probe.stdout.strip()


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


@pytest.mark.skipif(sys.platform != "linux", reason="nested PID namespace requires Linux")
def test_nested_namespace_children_count_toward_shell_aggregate_limit(setup):
    from pathlib import Path

    run, cap = setup
    fixture = Path(__file__).parent / "fixtures/mcp_server.py"
    cap = replace(cap, read_roots=(*cap.read_roots, *executable_read_roots(fixture)),
                  limits=replace(cap.limits, memory_bytes=96 << 20,
                                 nproc=4096, wall_seconds=4))
    marker = cap.workspace / "nested.lock"
    with ExecSession(run) as session:
        result = session.run([PYTHON, "-I", str(fixture), "nested_worker", str(marker)], cap)
    status = Path(str(marker) + ".status").read_text().strip()
    executable = sandbox_executable("bwrap")
    assert executable is not None
    if _bwrap_userns_guard_supported(executable):
        assert status.startswith("refused:"), status
    else:
        assert status == "nested" or status in {
            f"refused:{errno.EPERM}", f"refused:{errno.ENOSPC}"}, status
    assert result.cleanup.ok, result.cleanup
    assert result.cleanup.detail == "aggregate resident-memory limit exceeded"
    assert result.wall_seconds < 4
    with marker.open("r+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(handle, fcntl.LOCK_UN)


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
    unrelated_ready = run / "unrelated-ready"
    unrelated_stop = run / "unrelated-stop"
    engine_path = str(__import__("kernel_exec_session").__file__.rsplit("/", 1)[0])
    owner.write_text(f'''import json, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, {engine_path!r})
from kernel_exec_capability import capability_from_dict
from kernel_exec_session import ExecSession
import kernel_exec_session
unrelated_ready = Path({str(unrelated_ready)!r})
unrelated_stop = Path({str(unrelated_stop)!r})
real_popen = subprocess.Popen
wrapper_pid_file = Path({str(run / "supervisor.pid")!r})
if not {sandboxed!r}:
    def launch_through_short_lived_wrapper(command, **kwargs):
        source = """import json, os, sys, time
command = json.loads(sys.argv[1])
pid_file = sys.argv[2]
child = os.fork()
if child == 0:
    time.sleep(0.35)
    os.execvpe(command[0], command, os.environ)
    os._exit(127)
with open(pid_file, "w") as stream: stream.write(str(child))
os._exit(0)
"""
        return real_popen([sys.executable, "-c", source, json.dumps(command),
                           str(wrapper_pid_file)], stdin=kwargs["stdin"], stdout=kwargs["stdout"],
                          stderr=kwargs["stderr"], env=kwargs["env"], close_fds=True,
                          # Older launchers omit pass_fds; let the lifetime check catch the leak.
                          pass_fds=kwargs.get("pass_fds", ()),
                          start_new_session=kwargs["start_new_session"])
    kernel_exec_session.subprocess.Popen = launch_through_short_lived_wrapper
session = ExecSession(Path({str(run)!r}))
cap = capability_from_dict(json.loads({json.dumps(cap.as_dict())!r}))
job = session.spawn({child!r}, cap)
data = b""
while b"\\n" not in data:
    data += job.read_available()
    time.sleep(0.01)
unrelated = os.fork()
if unrelated == 0:
    os.setsid()
    unrelated_ready.write_text("ready")
    while not unrelated_stop.exists():
        time.sleep(0.01)
    os._exit(0)
deadline = time.monotonic() + 5
while not unrelated_ready.exists() and time.monotonic() < deadline:
    time.sleep(0.01)
if not unrelated_ready.exists():
    raise RuntimeError("unrelated child did not leave the owner's process group")
supervisor_pid = int(wrapper_pid_file.read_text()) if not {sandboxed!r} else session._process.pid
print(json.dumps({{"job": int(data.strip()), "supervisor": supervisor_pid,
                   "unrelated": unrelated}}), flush=True)
time.sleep(30)
''')
    owner_started_at = time.time()
    process = subprocess.Popen([sys.executable, "-I", str(owner)], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, start_new_session=True)
    unrelated_identity = None
    unrelated_binding = None
    unrelated = None
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
        unrelated = table.identity(pids["unrelated"])
        assert unrelated is not None
        unrelated_identity = record_identity(
            pids["unrelated"], launched_after=owner_started_at, command_fragment=owner.name)
        unrelated_binding = retain_signal_binding(unrelated_identity)
        signal_owned(process, signal.SIGKILL)
        process.wait(timeout=3)
        current_unrelated = table.identity(unrelated.pid)
        assert current_unrelated is not None and current_unrelated.start == unrelated.start
        assert not current_unrelated.zombie, "unrelated fork child should remain alive through owner exit"
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
            signal_owned(process, signal.SIGKILL)
            process.wait(timeout=3)
        unrelated_stop.touch(exist_ok=True)
        if unrelated_binding is not None:
            try:
                signal_bound(unrelated_binding, signal.SIGKILL)
            except (ProcessLookupError, RuntimeError):
                pass
            finally:
                close_signal_binding(unrelated_binding)
        if unrelated is not None:
            assert _wait_until_gone(unrelated), "unrelated child survived test cleanup"
        process.stdout.close()
        process.stderr.close()


@pytest.mark.skipif(sys.platform not in {"darwin", "linux"},
                    reason="owner-pipe lifetime control requires macOS or Linux")
def test_owner_pipe_eof_stops_job_while_forked_child_holds_control_socket(setup, tmp_path):
    run, cap = setup
    linux_bwrap = sys.platform.startswith("linux")
    if linux_bwrap:
        if os.getuid() == 0:
            pytest.skip("Linux execution sessions require a non-root host UID")
        bwrap = sandbox_executable("bwrap")
        if bwrap is None:
            pytest.skip("Linux owner-pipe control requires bwrap")
        try:
            help_result = subprocess.run([bwrap, "--help"], capture_output=True, timeout=5, check=False)
            help_text = (help_result.stdout + help_result.stderr).decode("utf-8", errors="replace")
            if "--as-pid-1" not in help_text or "--sync-fd" not in help_text:
                pytest.skip("Linux owner-pipe control requires bwrap --as-pid-1 and --sync-fd")
            probe = subprocess.run(
                [bwrap, "--unshare-all", "--ro-bind", "/", "/", "--proc", "/proc",
                 "--dev", "/dev", "--", sys.executable, "-I", "-c", "pass"],
                capture_output=True, timeout=5, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            pytest.skip(f"Linux bwrap namespace probe could not run: {type(error).__name__}")
        if probe.returncode != 0:
            reason = probe.stderr.decode("utf-8", errors="replace").strip()
            pytest.skip(f"Linux bwrap namespace is unavailable: {reason or probe.returncode}")

    workspace = cap.workspace
    home = cap.write_roots[-1]
    heartbeat = home / "job-heartbeat"
    owner_ready = run / "owner-ready.json"
    pipe_closed = run / "owner-pipe-closed"
    holder_confirm = run / "confirm-socket-held"
    socket_held = run / "socket-held"
    release_holder = run / "release-holder"
    holder_ready = run / "holder-ready"
    owner = run / "owner.py"
    engine_path = str(Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine")
    job_source = '''import signal, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
while True:
    with open(sys.argv[1], "ab") as stream:
        stream.write(b"x")
        stream.flush()
    time.sleep(0.02)
'''
    child_cap = replace(cap, purpose="job", workspace_writable=False,
                        sandboxed=linux_bwrap, limits=replace(cap.limits, nproc=4096))
    owner.write_text(f'''import json, os, sys, time
from pathlib import Path
sys.path.insert(0, {engine_path!r})
from kernel_exec_capability import capability_from_dict
from kernel_exec_session import ExecSession

run_dir = Path({str(run)!r})
workspace = Path({str(workspace)!r})
heartbeat = Path({str(heartbeat)!r})
holder_confirm = Path({str(holder_confirm)!r})
socket_held = Path({str(socket_held)!r})
release_holder = Path({str(release_holder)!r})
holder_ready = Path({str(holder_ready)!r})
pipe_closed = Path({str(pipe_closed)!r})
cap = capability_from_dict(json.loads({json.dumps(child_cap.as_dict())!r}))
session = ExecSession(run_dir)
job = session.spawn([{sys.executable!r}, "-I", "-c", {job_source!r}, str(heartbeat)], cap,
                    cwd=workspace)
deadline = time.monotonic() + 8
while not heartbeat.exists() and time.monotonic() < deadline:
    time.sleep(.01)
if not heartbeat.exists():
    raise RuntimeError("job did not start its heartbeat")
holder = os.fork()
if holder == 0:
    os.setsid()
    holder_ready.write_text("ready", encoding="ascii")
    descriptor = session._control.fileno()
    while not holder_confirm.exists():
        time.sleep(.02)
    os.fstat(descriptor)
    socket_held.write_text(str(descriptor), encoding="ascii")
    while not release_holder.exists():
        time.sleep(.02)
    os._exit(0)
while not holder_ready.exists():
    time.sleep(.01)
Path({str(owner_ready)!r}).write_text(json.dumps({{
    "holder": holder,
    "supervisor": session._process.pid,
}}), encoding="utf-8")
if {linux_bwrap!r}:
    session._close_owner_write_end()
    pipe_closed.write_text("closed", encoding="ascii")
time.sleep(60)
''', encoding="utf-8")
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join((
        str(Path(__file__).resolve().parents[1] / "src"), engine_path))}
    owner_started_at = time.time()
    process = subprocess.Popen([sys.executable, str(owner)], stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               env=environment, start_new_session=True)
    from stateframe.engine.kernel_exec_tracker import ProcessTable

    table = ProcessTable()
    holder_identity = None
    supervisor_identity = None
    holder_binding = None
    supervisor_binding = None
    try:
        deadline = time.monotonic() + 10
        while not owner_ready.is_file() and time.monotonic() < deadline:
            if process.poll() is not None:
                raise AssertionError(f"owner exited before creating {owner_ready.name}: {process.returncode}")
            time.sleep(.02)
        assert owner_ready.is_file(), "owner did not publish its forked socket holder"
        ready = json.loads(owner_ready.read_text(encoding="utf-8"))
        holder_identity = table.identity(ready["holder"])
        supervisor_identity = table.identity(ready["supervisor"])
        assert holder_identity is not None, "forked socket holder identity was unavailable"
        assert supervisor_identity is not None, "supervisor launch identity was unavailable"
        holder_signal_identity = record_identity(
            ready["holder"], launched_after=owner_started_at, command_fragment=owner.name)
        holder_binding = retain_signal_binding(holder_signal_identity)
        supervisor_signal_identity = record_identity(
            ready["supervisor"], launched_after=owner_started_at,
            command_fragment="kernel_exec_supervisor.py")
        supervisor_binding = retain_signal_binding(supervisor_signal_identity)

        if linux_bwrap:
            assert pipe_closed.is_file(), "owner did not close its lifetime-pipe writer"
            assert process.poll() is None, "owner must remain alive to isolate the pipe signal"
        else:
            signal_owned(process, signal.SIGKILL)
            process.wait(timeout=5)

        assert _wait_for_stable_file(heartbeat), "owner-pipe EOF did not stop the supervised job"
        holder_confirm.write_text("verify", encoding="ascii")
        deadline = time.monotonic() + 5
        while not socket_held.is_file() and time.monotonic() < deadline:
            time.sleep(.02)
        assert socket_held.is_file(), "forked child did not retain its control-socket descriptor"
        current_holder = table.identity(holder_identity.pid)
        assert (current_holder is not None and current_holder.start == holder_identity.start
                and not current_holder.zombie), "socket holder must still be alive after job cleanup"

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            current_supervisor = table.identity(supervisor_identity.pid)
            if (current_supervisor is None or current_supervisor.start != supervisor_identity.start
                    or current_supervisor.zombie):
                break
            time.sleep(.02)
        else:
            raise AssertionError("supervisor survived owner-pipe EOF while its client socket was held")
        if linux_bwrap:
            assert process.poll() is None, "owner-pipe EOF, not owner death, must stop the Linux session"
    finally:
        release_holder.touch(exist_ok=True)
        if process.poll() is None:
            signal_owned(process, signal.SIGKILL)
            process.wait(timeout=5)
        for binding in (holder_binding, supervisor_binding):
            if binding is None:
                continue
            try:
                signal_bound(binding, signal.SIGKILL)
            except (ProcessLookupError, RuntimeError):
                pass
            finally:
                close_signal_binding(binding)
        if holder_identity is not None:
            assert _wait_until_gone(holder_identity), "forked socket holder survived test cleanup"
        if supervisor_identity is not None:
            assert _wait_until_gone(supervisor_identity), "execution supervisor survived test cleanup"


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


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux procfs inside the bwrap session")
def test_procfs_tools_work_without_exposing_the_supervisor(setup):
    # Regression: the persistent session used to mount an empty tmpfs on /proc, so df, ps and
    # free failed ("cannot read table of mounted file systems"). Each command now gets its own
    # PID namespace and procfs: the tools work and the supervisor process stays invisible.
    run, cap = setup
    cap = replace(cap, sandboxed=True)
    with ExecSession(run) as session:
        result = session.run(["/bin/sh", "-c", "df -h /"], cap)
        assert result.exit_code == 0, result.output
        assert b"Filesystem" in result.output
        listing = script(cap, '''import os
for pid in sorted(p for p in os.listdir("/proc") if p.isdigit()):
    try:
        print(pid, open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\\0", b" ").decode())
    except OSError:
        pass
''')
        result = session.run(listing, cap)
        assert result.exit_code == 0, result.output
        assert b"kernel_exec_supervisor" not in result.output, result.output
        assert b"command.py" in result.output
        assert result.cleanup.ok
