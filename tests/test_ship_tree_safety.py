"""Automatic import diagnostics must never become product mutations."""
from dataclasses import replace
import sys

import pytest

from kernel_exec_session import CleanupReport, ExecResult, ExecSessionError
from kernel_ship_tree_smoke import ImportSmokeCleanupError, _execute_smoke, run_smoke


@pytest.mark.parametrize('cleanup_error', [None, ExecSessionError('cleanup failed'),
                                          OSError('cleanup denied')])
def test_smoke_execution_authority_is_readonly_offline(tmp_path, monkeypatch, cleanup_error):
    root = tmp_path / 'repo'
    root.mkdir()
    (root / 'app.py').write_text('VALUE = 1\n')
    observed = {}

    class SessionBoundary:
        def __init__(self, directory):
            observed['session'] = directory

        def close(self):
            if cleanup_error is not None:
                raise cleanup_error

        def run(self, argv, cap):
            observed.update(argv=argv, cap=cap)
            return ExecResult(0, False, b'\nSTATEFRAME_SMOKE_RESULT=[]\n', False, 28,
                              0, CleanupReport(True, 0, (), ''))

    monkeypatch.setattr('kernel_exec_session.ExecSession', SessionBoundary)
    monkeypatch.setenv('STATEFRAME_TEST_PYTHON', sys.executable)
    if cleanup_error is None:
        _execute_smoke(root, ('app.py',), (str(root),), 3)
    else:
        with pytest.raises(ImportSmokeCleanupError, match='cleanup'):
            _execute_smoke(root, ('app.py',), (str(root),), 3)
    cap = observed['cap']
    assert cap.workspace == root.resolve()
    assert cap.workspace_writable is False and cap.sandboxed is True
    assert cap.network == 'none' and cap.purpose == 'probe'
    assert cap.limits.wall_seconds == 3
    assert '-B' in observed['argv'] and '-I' in observed['argv']
    assert all(not root.is_relative_to(p) and not p.is_relative_to(root)
               for p in cap.write_roots)
    assert not observed['session'].parent.exists()


def test_smoke_noisy_output_is_not_a_json_failure(tmp_path, monkeypatch):
    (tmp_path / 'app.py').write_text('print("noise")\n')
    result = ExecResult(0, False, b'noise\nSTATEFRAME_SMOKE_RESULT=[{"path":"app.py","status":"PASS"}]\n',
                        False, 72, 0, CleanupReport(True, 0, (), ''))
    monkeypatch.setattr('kernel_ship_tree_smoke._execute_smoke', lambda *args: result)
    assert run_smoke(tmp_path, ['app.py']).status == 'PASS'
    monkeypatch.setattr('kernel_ship_tree_smoke._execute_smoke',
                        lambda *args: replace(result, output=b'noise only'))
    assert run_smoke(tmp_path, ['app.py']).status == 'ENVIRONMENT'


def test_smoke_cleanup_failure_stops_instead_of_becoming_environment_fact(tmp_path, monkeypatch):
    (tmp_path / 'app.py').write_text('VALUE = 1\n')
    result = ExecResult(0, False, b'', False, 0, 0,
                        CleanupReport(False, 0, (123,), 'descendant still alive'))
    monkeypatch.setattr('kernel_ship_tree_smoke._execute_smoke', lambda *args: result)
    with pytest.raises(ImportSmokeCleanupError, match='descendant still alive'):
        run_smoke(tmp_path, ['app.py'])


@pytest.mark.host_sandbox
def test_smoke_cannot_mutate_workspace_or_absolute_host_path(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    root.mkdir()
    target = root / 'app.py'
    target.write_text('VALUE = "before"\n')
    outside = tmp_path / 'outside'
    outside.write_text('untouched')
    (root / 'format.py').write_text(
        'from pathlib import Path\n'
        'Path("app.py").write_text(\'VALUE = "after"\\n\')\n')
    (root / 'escape.py').write_text(
        f'from pathlib import Path\nPath({str(outside)!r}).write_text("bad")\n')
    monkeypatch.setenv('STATEFRAME_TEST_PYTHON', sys.executable)
    result = run_smoke(root, ['format.py', 'escape.py'])
    assert result.status == 'FAIL', result
    assert all('PermissionError' in error or 'OSError' in error for _, error in result.failures)
    assert target.read_text() == 'VALUE = "before"\n'
    assert outside.read_text() == 'untouched'
    assert {p.name for p in root.iterdir()} == {'app.py', 'format.py', 'escape.py'}


@pytest.mark.host_sandbox
def test_smoke_real_noisy_import_and_syntax_error(tmp_path, monkeypatch):
    (tmp_path / 'noisy.py').write_text('print("import noise")\n')
    (tmp_path / 'broken.py').write_text('def broken(:\n')
    monkeypatch.setenv('STATEFRAME_TEST_PYTHON', sys.executable)
    result = run_smoke(tmp_path, ['noisy.py', 'broken.py'])
    assert result.status == 'FAIL'
    assert len(result.failures) == 1
    assert result.failures[0][0] == 'broken.py'
    assert 'SyntaxError' in result.failures[0][1]
