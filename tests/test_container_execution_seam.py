"""Container execution seam keeps platform refusal and process supervision."""
import os
from pathlib import Path
import shlex
import subprocess
import sys

from _timeouts import scaled


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "src" / "stateframe" / "engine"
SRC = ROOT / "src"
TESTS = ROOT / "tests"


def _fake_bwrap(tmp_path):
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    marker = tmp_path / "bwrap-called"
    executable = binary_dir / "bwrap"
    executable.write_text(
        "#!/bin/sh\n"
        f"echo called >> {shlex.quote(str(marker))}\n"
        "echo 'bwrap: No permissions to create new namespace' >&2\n"
        "echo 'bwrap: execvp /usr/bin/python: Operation not permitted' >&2\n"
        "exit 1\n"
    )
    executable.chmod(0o755)
    return binary_dir, marker


def _scenario(tmp_path, mode, body):
    binary_dir, marker = _fake_bwrap(tmp_path)
    env = dict(os.environ, PATH=str(binary_dir) + os.pathsep + os.environ.get("PATH", ""))
    code = f"""import sys
sys.path[:0] = [{str(SRC)!r}, {str(TESTS)!r}, {str(ENGINE)!r}]
try:
    from stateframe.engine import kernel_execution_boundary as boundary
except ImportError:
    import _container_boundary_stub as boundary
sys.modules['kernel_execution_boundary'] = boundary
if {mode!r} == 'container':
    boundary.activate(boundary.Boundary(boundary.BACKEND_CONTAINER,
        boundary.Detection('container', ('mount:docker',), {{'pid': '4026533000'}}, 1)))
{body}
"""
    result = subprocess.run([sys.executable, "-I", "-c", code], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=scaled(30), check=False)
    assert result.returncode == 0, result.stderr or result.stdout
    return marker


def test_platform_probe_check_shell_and_hook_keep_bwrap_refusal(tmp_path):
    body = f"""
from pathlib import Path
from kernel_test_isolation import check_sandbox_exec_denial
assert check_sandbox_exec_denial(sys.executable, timeout=5)
import kernel_execution_policy as policy
root = Path({str(tmp_path)!r})
root.mkdir(exist_ok=True)
from kernel_exec_capability import ExecCapability, ExecLimits
import kernel_test_isolation as isolation
capability = ExecCapability('shell', root, False,
    (Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()), (), 'none',
    {{'PATH': '/usr/bin:/bin'}}, ExecLimits())
automatic = isolation.capability_command(['/bin/true'], capability)
base_argv = isolation.capability_command(['/bin/true'], capability, platform_backend='bwrap')
assert automatic == base_argv
from kernel_exec_session import ExecSession, ExecSessionError
from kernel_safe_probe import ProbeCleanupError, run_probe
try:
    run_probe([sys.executable, '-I', '-B', '-c', 'pass'], root, timeout=5)
except (ExecSessionError, ProbeCleanupError):
    pass
else:
    raise AssertionError('platform startup probe unexpectedly ran')
check = policy.run_sandboxed([sys.executable, '-c', 'print(1)'], root,
                             timeout=5, env={{'PATH': '/usr/bin:/bin'}})
assert check.completed.returncode != 0
from stateframe import outer_checks
from stateframe.outer_checks import OuterCheckError
try:
    outer = outer_checks.run(f"{{sys.executable}} -c 'print(2)'", root,
        python=sys.executable, test_python=sys.executable, runner_kind='generic',
        capture_output=True, timeout_s=20)
except OuterCheckError:
    pass
else:
    assert not outer['passed']
from kernel_exec_capability import shell_capability
from kernel_exec_session import ExecSession, ExecSessionError
for name in ('work', 'env', 'home', 'run'):
    (root / name).mkdir(exist_ok=True)
cap = shell_capability(workspace=root / 'work', run_env_dir=root / 'env',
                       scratch_home=root / 'home', network='none', writable=False)
session = ExecSession(root / 'run')
try:
    try:
        session.run([sys.executable, '-c', 'print(1)'], cap)
    except ExecSessionError:
        pass
    else:
        raise AssertionError('platform shell unexpectedly ran')
finally:
    try: session.close()
    except ExecSessionError: pass
from kernel_hook_exec import HookContainmentError
from test_hook_exec import setup_runner
(root / 'hook').mkdir(exist_ok=True)
runner, _ = setup_runner(root / 'hook')
try:
    event = runner.run_event('pre_tool', boundary='pre_tool:test', data={{}},
                             match={{'tool': 'shell'}})
except HookContainmentError:
    pass
else:
    assert event.decision != 'continue'
"""
    marker = _scenario(tmp_path, "platform", body)
    assert marker.is_file()


def test_container_probe_check_shell_and_hook_run_without_bwrap(tmp_path):
    body = f"""
from pathlib import Path
from kernel_test_isolation import check_sandbox_exec_denial
assert check_sandbox_exec_denial(sys.executable, timeout=5) is None
import kernel_execution_policy as policy
root = Path({str(tmp_path)!r})
root.mkdir(exist_ok=True)
import kernel_test_isolation as isolation
assert isolation.backend() == 'no sandbox binary: container'
env = {{'PATH': '/usr/bin:/bin', 'PROVIDER_KEY': 'must-not-pass'}}
check = policy.run_sandboxed([sys.executable, '-c',
    \"import os; print(os.getenv('PROVIDER_KEY', 'scrubbed'))\"], root,
    timeout=5, env=env)
assert check.completed.returncode == 0, check.completed.stderr
assert check.completed.stdout.strip() in (b'scrubbed', 'scrubbed')
report = policy.capability_report()
assert report.backend == 'no sandbox binary: container' and not report.network_isolated and not report.path_contained
from kernel_safe_probe import run_probe
startup = run_probe([sys.executable, '-I', '-B', '-c', \"print('probe')\"], root, timeout=5)
assert startup.exit_code == 0 and startup.output.strip() == b'probe'
from stateframe import outer_checks
from stateframe.outer_checks import OuterCheckError
try:
    outer_checks.run(f"{{sys.executable}} -c 'print(2)'", root,
        python=sys.executable, test_python=sys.executable, runner_kind='generic',
        capture_output=True, timeout_s=20)
except OuterCheckError as error:
    assert 'CONTAINER_NOT_DETECTED' in str(error), error
else:
    raise AssertionError('unverified container check ran')
import stateframe.send1_probe_cache as probe_cache
assert probe_cache._sandbox_backend() is None
from kernel_plain_host import sandboxed_check_runner
(root / 'test_canary.py').write_text('def test_canary(): pass\\n')
receipts = []
test_argv = [sys.executable, '-m', 'pytest', '-q', 'test_canary.py']
check_runner = sandboxed_check_runner(root / 'study.lock', root / 'audit.jsonl',
    preflight_argv=test_argv, on_preflight=receipts.append)
outcome = check_runner(test_argv, root, {{}})
assert outcome.exit_code == 0 and receipts[0]['exit_code'] == 0
assert receipts[0]['passed'] is False and receipts[0]['backend'] == 'container'
assert receipts[0]['network_isolation'] == receipts[0]['path_isolation'] == 'unknown'
assert receipts[0]['secret_isolation'] == 'unknown'
from kernel_exec_capability import shell_capability
from kernel_exec_session import ExecSession
import os
os.environ['PROVIDER_KEY'] = 'must-not-pass'
for name in ('work', 'env', 'home', 'run'):
    (root / name).mkdir(exist_ok=True)
cap = shell_capability(workspace=root / 'work', run_env_dir=root / 'env',
                       scratch_home=root / 'home', network='full', writable=False)
with ExecSession(root / 'run') as session:
    shell = session.run([sys.executable, '-c',
        \"import os; print(os.getenv('PROVIDER_KEY', 'scrubbed'))\"], cap)
assert shell.exit_code == 0 and shell.output.strip() == b'scrubbed'
from test_hook_exec import setup_runner
(root / 'hook').mkdir(exist_ok=True)
runner, _ = setup_runner(root / 'hook')
from kernel_execution_boundary import BoundaryRefused
try:
    runner.run_event('pre_tool', boundary='pre_tool:test', data={{}},
                     match={{'tool': 'shell'}})
except BoundaryRefused as error:
    assert error.code == 'CONTAINER_UNSUPPORTED:hooks'
else:
    raise AssertionError('container mode ran a hook')
"""
    marker = _scenario(tmp_path, "container", body)
    assert not marker.exists()


def test_container_hook_refusal_is_typed_and_precedes_supervisor(tmp_path):
    body = f"""
from pathlib import Path
from kernel_execution_boundary import BoundaryRefused
import kernel_hook_exec
from test_hook_exec import setup_runner
root = Path({str(tmp_path)!r})
(root / 'hook').mkdir(exist_ok=True)
runner, _ = setup_runner(root / 'hook')
starts = []
def start_supervisor(*args, **kwargs):
    starts.append(True)
    raise AssertionError('hook supervisor started')
kernel_hook_exec.ExecSession = start_supervisor
try:
    try:
        runner.run_event('pre_tool', boundary='pre_tool:test', data={{}},
                         match={{'tool': 'shell'}})
    except BoundaryRefused as error:
        assert error.code == 'CONTAINER_UNSUPPORTED:hooks'
    else:
        raise AssertionError('container mode ran a hook')
    assert not starts
    assert not runner.sessions
    assert not runner.tape.path.exists()
    assert not tuple(runner.scratch_root.iterdir())
finally:
    runner.close()
"""
    marker = _scenario(tmp_path, "container", body)
    assert not marker.exists()


def test_container_supervisor_cleans_setsid_double_fork_on_timeout(tmp_path):
    source = f"""import os, signal, time
from pathlib import Path
pidfile = Path({str(tmp_path / 'escaped-pid')!r})
heartbeat = Path({str(tmp_path / 'heartbeat')!r})
child = os.fork()
if child == 0:
    grandchild = os.fork()
    if grandchild == 0:
        os.setsid()
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        pidfile.write_text(str(os.getpid()))
        while True:
            with heartbeat.open('a') as stream: stream.write('.')
            time.sleep(0.05)
    while not pidfile.exists(): time.sleep(0.01)
    time.sleep(0.3)
    os._exit(0)
os.waitpid(child, 0)
time.sleep(30)
"""
    body = f"""
from dataclasses import replace
from pathlib import Path
from kernel_exec_capability import shell_capability
from kernel_exec_session import ExecSession
root = Path({str(tmp_path)!r})
for name in ('work', 'env', 'home', 'run'):
    (root / name).mkdir(exist_ok=True)
cap = shell_capability(workspace=root / 'work', run_env_dir=root / 'env',
                       scratch_home=root / 'home', network='full', writable=True)
script = root / 'work' / 'double_fork.py'
script.write_text({source!r})
cap = replace(cap, limits=replace(cap.limits, wall_seconds=1.5))
with ExecSession(root / 'run') as session:
    result = session.run([sys.executable, '-I', str(script)], cap)
assert result.timed_out and result.cleanup.ok, result.cleanup
assert result.cleanup.killed >= 1
pid = int((root / 'escaped-pid').read_text())
from kernel_exec_tracker import ProcessTable
row = ProcessTable().identity(pid)
assert row is None or row.zombie
size = (root / 'heartbeat').stat().st_size
import time
time.sleep(0.2)
assert (root / 'heartbeat').stat().st_size == size
"""
    marker = _scenario(tmp_path, "container", body)
    assert not marker.exists()


def test_container_mutation_builder_refuses_without_bwrap(tmp_path):
    body = f"""
from pathlib import Path
import subprocess
from kernel_exec_capability import ExecCapability, ExecLimits
from kernel_test_isolation import capability_command
from kernel_execution_boundary import BoundaryRefused
root = Path({str(tmp_path)!r})
mounts = {{}}
for name in ('boot', 'unit', 'ctl', 'in', 'resp'):
    mounts[name] = str(root / name)
    (root / name).mkdir()
cap = ExecCapability('mutation', root, False,
    (root, Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()), (),
    'none', {{'PATH': '/usr/bin:/bin'}}, ExecLimits())
try:
    capability_command(['/bin/true'], cap, mutation_mounts=mounts)
except BoundaryRefused as error:
    assert error.code == 'CONTAINER_UNSUPPORTED:mutations'
else:
    raise AssertionError('container mode admitted a mutation launch')
"""
    marker = _scenario(tmp_path, "container", body)
    assert not marker.exists()

