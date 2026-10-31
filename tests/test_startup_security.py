"""Startup must establish containment before a project interpreter runs code."""
import json
from pathlib import Path
import socket
import subprocess
import sys

import pytest

from stateframe import runner
from stateframe.engine.kernel_exec_session import CleanupReport, ExecResult, ExecSessionError
from stateframe.engine.kernel_safe_probe import ProbeCleanupError


@pytest.mark.parametrize('check', ['engine', 'project'])
@pytest.mark.parametrize('outcome', ['ok', 'timeout', 'flood', 'exit', 'launch', 'cleanup', 'version'])
def test_startup_capability_and_fail_closed_errors(tmp_path, monkeypatch, check, outcome):
    from stateframe.engine import kernel_safe_probe
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('HOST_SECRET', 'credential')
    monkeypatch.setenv('PYTHONPATH', str(tmp_path))
    seen = []

    class Boundary:
        def __init__(self, directory):
            pass

        def run(self, argv, cap, **kwargs):
            seen.append(cap)
            assert argv[1:4] == ['-I', '-B', '-c']
            assert cap.sandboxed and not cap.workspace_writable and cap.network == 'none'
            assert cap.workspace != tmp_path.resolve() and kwargs['cwd'].resolve() == cap.workspace
            assert 'HOST_SECRET' not in cap.env and 'PYTHONPATH' not in cap.env
            assert cap.limits.wall_seconds == 10 and cap.limits.output_bytes == 65536
            assert tmp_path not in cap.write_roots
            if outcome == 'launch':
                raise ExecSessionError('sandbox unavailable')
            data = b'not a version' if outcome == 'version' else json.dumps(list(sys.version_info[:3])).encode()
            return ExecResult(1 if outcome == 'exit' else 0, outcome == 'timeout', data,
                              outcome == 'flood', len(data), .01,
                              CleanupReport(outcome != 'cleanup', 0, (), 'survivor'))

        def close(self):
            pass

    monkeypatch.setattr(kernel_safe_probe, 'ExecSession', Boundary)
    options = runner.RunOptions(repo=tmp_path, task='test', endpoint=None, test_python=sys.executable)

    def invoke():
        if check == 'engine':
            return runner._check_python(sys.executable)
        return runner._select_test_python(options, printer=lambda _: None)

    if outcome == 'cleanup':
        with pytest.raises(ProbeCleanupError, match='cleanup failed'):
            invoke()
    elif outcome in {'timeout', 'flood', 'exit', 'launch'} or (outcome == 'version' and check == 'engine'):
        expected = {'timeout': 'wall limit', 'flood': 'output limit', 'exit': 'cannot import pytest',
                    'launch': 'could not run safely', 'version': 'invalid Python version'}[outcome]
        with pytest.raises(runner.RunError, match=expected):
            invoke()
    else:
        assert invoke() == sys.executable
    assert len(seen) == 1


@pytest.mark.host_sandbox
@pytest.mark.parametrize('check', ['engine', 'project'])
def test_startup_hooks_cannot_write_or_connect_and_repo_pytest_is_not_imported(tmp_path, monkeypatch, check):
    repo = tmp_path / 'repo'
    repo.mkdir()
    environment = repo / '.venv'
    subprocess.run([sys.executable, '-I', '-m', 'venv', '--without-pip', str(environment)],
                   check=True, timeout=60)
    site = next(environment.glob('lib/python*/site-packages'))
    outside = tmp_path / 'outside'
    outside.write_text('untouched')
    local_marker = tmp_path / 'local-pytest-ran'
    monkeypatch.chdir(repo)
    monkeypatch.setenv('PYTHONPATH', str(repo))
    monkeypatch.setenv('HOST_SECRET', 'credential')
    with socket.socket() as server:
        server.bind(('127.0.0.1', 0))
        server.listen()
        with socket.create_connection(server.getsockname(), timeout=1):
            connection, _ = server.accept()
            connection.close()
        attack = f'''import builtins, os, socket
from pathlib import Path
assert 'HOST_SECRET' not in os.environ
blocked = 0
for path in [Path({str(outside)!r}), Path({str(repo / 'repo-write')!r})]:
    try:
        path.write_text('ESCAPE')
    except PermissionError:
        blocked += 1
try:
    socket.create_connection(('127.0.0.1', {server.getsockname()[1]}), timeout=1).close()
except OSError:
    blocked += 1
Path(os.environ['HOME'], 'scratch-proof').write_text('allowed')
builtins.STARTUP_BLOCKED = blocked
'''
        (site / 'startup_attack.py').write_text(attack)
        (site / 'attack.pth').write_text('import startup_attack\n')
        (site / 'sitecustomize.py').write_text(
            'import builtins\nassert builtins.STARTUP_BLOCKED == 3\nbuiltins.SITE_RAN = True\n')
        # A successful import proves startup actually ran and all three attacks
        # failed; Python swallowing sitecustomize exceptions cannot pass this.
        (site / 'pytest.py').write_text(
            'import builtins\nassert builtins.STARTUP_BLOCKED == 3\nassert builtins.SITE_RAN\n')
        (repo / 'pytest.py').write_text(attack + f'Path({str(local_marker)!r}).write_text("ran")\n')
        python = str(environment / 'bin/python')
        if check == 'engine':
            assert runner._check_python(python) == python
        else:
            options = runner.RunOptions(repo=repo, task='test', endpoint=None)
            selected = runner._select_test_python(options, printer=lambda _: None)
            assert Path(selected).parent == environment / 'bin'
    assert outside.read_text() == 'untouched'
    assert not (repo / 'repo-write').exists() and not local_marker.exists()
