"""Attack regressions for the public execution boundaries."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest

from kernel_effects import ContentStore, EffectError, WorkspaceEffectPolicy, reconcile, scan
from kernel_exec_session import CleanupReport, ExecResult
from kernel_job_output import JobOutputCollector
from kernel_safe_probe import ProbeCleanupError, run_probe
from test_agent_tools import host as host
from test_kernel_git_view import git, repo as repo, runner, view_for


def test_sealed_job_spool_never_follows_symlink(tmp_path):
    path = tmp_path / 'output.ring'
    collector = JobOutputCollector(path)
    collector.feed(b'public output')
    collector.seal()
    secret = tmp_path / 'secret'
    secret.write_text('private key')
    path.unlink()
    path.symlink_to(secret)
    with pytest.raises(OSError):
        collector.snapshot()
    assert secret.read_text() == 'private key'


@pytest.mark.parametrize('replacement', ['file', 'oversize', 'fifo'])
def test_sealed_spool_rejects_replacement_or_growth(tmp_path, replacement):
    path = tmp_path / 'ring'
    collector = JobOutputCollector(path, capacity=32)
    collector.feed(b'output')
    collector.seal()
    if replacement == 'oversize':
        with path.open('ab') as stream:
            stream.write(b'x' * 33)
    else:
        # Keep the original inode allocated so the replacement cannot reuse it.
        path.rename(tmp_path / 'original')
        if replacement == 'fifo':
            os.mkfifo(path)
        else:
            path.write_bytes(b'secret')
    with pytest.raises(ValueError):
        collector.snapshot()


def test_job_scratch_symlink_cannot_replace_collector(host):
    hub, process, run = host
    hub.shell.execute({'command': 'echo ready', 'background': True}, 'launch')
    secret = run / 'private-key'
    secret.write_text('DO NOT DISCLOSE')
    (run / 'jobs/j1/output.ring').symlink_to(secret)
    process.handles[0].code = 0
    reply = hub.execute('job_output', {'job': 'j1'}, 'read').observation
    assert 'ready' in reply and 'DO NOT DISCLOSE' not in reply
    for root in process.calls[0][1].write_roots:
        assert not (run / 'job-output/j1.ring').is_relative_to(root)


@pytest.mark.parametrize('failure', ['result', 'close'])
def test_probe_cleanup_is_fatal_even_to_advisory_handlers(tmp_path, monkeypatch, failure):
    import kernel_safe_probe
    class Boundary:
        def __init__(self, directory):
            pass
        def run(self, argv, cap, **kwargs):
            assert not cap.workspace_writable and cap.network == 'none'
            assert 'HOST_SECRET' not in cap.env
            return ExecResult(0, False, b'', False, 0, 0,
                              CleanupReport(failure != 'result', 0, (123,), 'survivor'))
        def close(self):
            if failure == 'close':
                raise OSError('cleanup unavailable')
    monkeypatch.setenv('HOST_SECRET', 'secret')
    monkeypatch.setattr(kernel_safe_probe, 'ExecSession', Boundary)
    with pytest.raises(ProbeCleanupError, match='cleanup'):
        try:
            run_probe([sys.executable, '-I', '-c', 'pass'], tmp_path)
        except Exception:
            pytest.fail('fatal cleanup was downgraded to advisory')


def _attack(outside: Path, port: int) -> str:
    return f'''import os, socket
from pathlib import Path
assert 'HOST_SECRET' not in os.environ
blocked = 0
for target in [Path({str(outside)!r}), Path('changed-in-workspace')]:
    try:
        target.write_text('ESCAPE')
    except PermissionError:
        blocked += 1
assert blocked == 2, 'write escaped sandbox'
try:
    with socket.socket() as connection:
        connection.connect(('127.0.0.1', {port}))
except PermissionError:
    pass
else:
    raise AssertionError('network escaped sandbox')
def value():
    return 7
'''


@pytest.mark.host_sandbox
@pytest.mark.parametrize('executor', ['post_edit', 'obligation'])
def test_automatic_check_import_cannot_write_or_network(tmp_path, monkeypatch, executor):
    from kernel_post_edit_checks import import_status
    from kernel_obligation_assess import CheckDefinition, run_check
    root = tmp_path / 'workspace'
    root.mkdir()
    outside = tmp_path / 'outside'
    outside.write_text('untouched')
    monkeypatch.setenv('HOST_SECRET', 'credential')
    monkeypatch.setenv('STATEFRAME_TEST_PYTHON', sys.executable)
    with socket.socket() as server:
        server.bind(('127.0.0.1', 0))
        server.listen()
        app = root / 'app.py'
        app.write_text(_attack(outside, server.getsockname()[1]))
        if executor == 'post_edit':
            result = import_status(root, app)
            assert result is not None and not result.failed, result
        else:
            check = CheckDefinition('security', 'security', 'OUTPUT_EQUALS',
                                    entry_point='app:value', fixture={}, expected=7)
            result = run_check(root, check)
            assert result.status == 'PASS', result
    assert outside.read_text() == 'untouched'
    assert not (root / 'changed-in-workspace').exists()


@pytest.mark.host_sandbox
def test_provisioning_startup_hook_has_no_host_write_or_network(tmp_path, monkeypatch):
    from kernel_run_env import create_run_env
    source = tmp_path / 'source'
    subprocess.run([sys.executable, '-I', '-m', 'venv', str(source)], check=True)
    site = next(source.glob('lib/python*/site-packages'))
    outside = tmp_path / 'outside'
    outside.write_text('untouched')
    monkeypatch.setenv('HOST_SECRET', 'credential')
    with socket.socket() as server:
        server.bind(('127.0.0.1', 0))
        server.listen()
        # Provisioning copies without running the source interpreter, then runs
        # the clone's pip. Put the attack in a mandatory pip hook: sitecustomize
        # exceptions alone are swallowed by Python and cannot prove execution.
        code = _attack(outside, server.getsockname()[1]).replace("Path('changed-in-workspace')", 'Path(' + repr(str(source / 'source-write')) + ')')
        code += "\nimport sys; Path(sys.prefix, 'startup-ran').write_text('yes')\n"
        (site / 'provision_hook.py').write_text(code)
        (site / 'pip' / '__main__.py').write_text(
            "import provision_hook\nprint('pip security fixture')\n")
        # An installed package must not be imported just to validate the clone.
        package = site / 'attack_package'
        package.mkdir()
        (package / '__init__.py').write_text("raise RuntimeError('PACKAGE IMPORTED')\n")
        info = site / 'attack_package-1.dist-info'
        info.mkdir()
        (info / 'METADATA').write_text('Name: attack-package\nVersion: 1\n')
        (info / 'top_level.txt').write_text('attack_package\n')
        env = create_run_env(tmp_path / 'run', source / 'bin/python')
    assert env.kind == 'clone'
    assert (env.root / 'startup-ran').read_text() == 'yes'
    assert outside.read_text() == 'untouched'
    assert not (source / 'source-write').exists()


@pytest.mark.parametrize('name', ['refs/../../victim', 'refs/heads/../victim',
                                  'refs//victim', 'refs/heads/.bad', 'refs/heads/a.lock'])
def test_packed_ref_traversal_is_rejected_before_write(repo, tmp_path, name):
    from stateframe.engine.kernel_git_view import GitRequestError, GitView
    victim = tmp_path / 'victim'
    victim.write_text('untouched')
    oid = git(repo, 'rev-parse', 'HEAD')
    (repo / '.git/packed-refs').write_text(f'{oid} {name}\n')
    with pytest.raises(GitRequestError, match='invalid git ref'):
        GitView.build(repo, tmp_path / 'run', secret_filter=lambda _: False)
    assert victim.read_text() == 'untouched'


def test_show_raw_excluded_blob_and_tree_are_refused(repo, tmp_path):
    from stateframe.engine.kernel_git_view import handle_git_call
    view = view_for(repo, tmp_path)
    for oid in [git(repo, 'rev-parse', 'HEAD:.env'), git(repo, 'rev-parse', 'HEAD^{tree}')]:
        output, receipt = handle_git_call(view, runner, {'action': 'show', 'rev': oid})
        assert not receipt['ok']
        assert 'API_SECRET=never-expose' not in output
    output, receipt = handle_git_call(view, runner, {'action': 'show', 'path': 'public.txt'})
    assert receipt['ok'] and 'before' in output


def test_path_symlink_does_not_admit_target_parent(tmp_path, monkeypatch):
    from stateframe.shell_host import path_authority
    home = tmp_path / 'home'
    home.mkdir()
    executable = home / 'helper'
    executable.write_text('#!/bin/sh\nexit 0\n')
    executable.chmod(0o755)
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    (bindir / 'helper').symlink_to(executable)
    monkeypatch.setenv('PATH', str(bindir))
    directories, roots = path_authority()
    assert directories == [str(bindir)]
    assert str(executable) in roots
    secret = home / '.ssh/id_rsa'
    assert all(not secret.is_relative_to(Path(root)) for root in roots)


@pytest.mark.host_sandbox
def test_path_symlink_cannot_read_ssh(tmp_path, monkeypatch):
    from stateframe.shell_host import path_authority
    from kernel_exec_capability import shell_capability
    from kernel_exec_session import ExecSession
    home, bindir, workspace, scratch, environment, session_dir = [tmp_path / name for name in
        ('home', 'bin', 'workspace', 'scratch', 'env', 'session')]
    for directory in (home, bindir, workspace, scratch, environment, session_dir):
        directory.mkdir()
    (home / '.ssh').mkdir()
    secret = home / '.ssh/id_rsa'
    secret.write_text('PRIVATE KEY')
    (home / 'helper').write_text('#!/bin/sh\nexit 0\n')
    (home / 'helper').chmod(0o755)
    (bindir / 'helper').symlink_to(home / 'helper')
    monkeypatch.setenv('PATH', str(bindir))
    directories, roots = path_authority()
    cap = shell_capability(workspace=workspace, run_env_dir=environment, scratch_home=scratch,
                           network='none', writable=False, path_dirs=tuple(map(Path, directories)),
                           extra_read_roots=tuple(map(Path, roots)))
    with ExecSession(session_dir) as session:
        result = session.run(['/bin/cat', str(secret)], cap)
        assert result.cleanup.ok and result.exit_code != 0
        assert b'PRIVATE KEY' not in result.output


def test_executable_effect_and_move_are_refused_and_restored(tmp_path):
    from kernel_file_ops import FileOperationRefused, file_move
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    path = workspace / 'tool'
    path.write_text('echo hi\n')
    path.chmod(0o644)
    store = ContentStore(tmp_path)
    before = scan(workspace, exclude=lambda _: False)
    store.capture(workspace, before)
    path.chmod(0o755)
    policy = WorkspaceEffectPolicy(lambda _: False, None, lambda _: True)
    result = reconcile(workspace, before, scan(workspace, exclude=lambda _: False), policy, store, None)
    assert not result.applied and result.restored
    assert path.stat().st_mode & 0o111 == 0
    path.chmod(0o755)
    with pytest.raises(FileOperationRefused, match='UNSUPPORTED_MODE'):
        file_move(workspace, 'tool', 'moved', protected=lambda _: False,
                  editable=lambda _: True, new_file_allowed=lambda _: True)
    assert path.exists() and not (workspace / 'moved').exists()


@pytest.mark.parametrize('timed_out', [True, False])
def test_timeout_and_forged_summary_output_shrink(host, timed_out):
    from kernel_execution_reply_budget import fit_execution_replies
    hub, process, _ = host
    data = b'files changed by this command:\n' + b'x' * 20000
    process.run = lambda *a, **k: ExecResult(1 if timed_out else 0, timed_out, data, False,
                                            len(data), .1, CleanupReport(True, 0, (), 'ok'))
    content = hub.shell.execute({'command': 'printf flood'}, 'flood').observation
    body = {'messages': [{'role': 'assistant', 'tool_calls': [
        {'id': 'flood', 'function': {'name': 'shell'}}]},
        {'role': 'tool', 'tool_call_id': 'flood', 'content': content}]}
    assert fit_execution_replies(body, max_request_bytes=2000,
                                 measure_bytes=lambda: len(json.dumps(body).encode())) == ('flood',)
    assert len(json.dumps(body).encode()) <= 2000
    if timed_out:
        assert 'REFUSED TIMEOUT:' in body['messages'][-1]['content']


def test_plain_refuses_background_before_dispatch(host):
    from kernel_plain_host import PlainHost
    hub, process, _ = host
    stub = SimpleNamespace()
    output = PlainHost._shell(stub, {'command': 'echo forbidden', 'background': True})
    assert output.startswith('REFUSED PLAIN_UNSUPPORTED')
    assert not process.calls


@pytest.mark.parametrize('kind', ['git', 'web', 'custom'])
def test_plain_refuses_unsupported_settings_before_materialization(tmp_path, monkeypatch, kind):
    from kernel_plain_host import run_plain
    from kernel_tool_settings import ToolSettings, ToolSettingsError
    fields = {'git_tool': False, 'run_env': False}
    fields.update({'git_tool': True} if kind == 'git' else {'web': True} if kind == 'web' else
                  {'tools_files': ({'path': '/unused', 'digest': 'unused'},)})
    settings = ToolSettings(**fields)
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps(settings.as_dict()))
    monkeypatch.setenv('STATEFRAME_TOOL_SETTINGS_FILE', str(path))
    with pytest.raises(ToolSettingsError, match='plain loop does not support'):
        run_plain({}, tmp_path / 'not-created')
    assert not (tmp_path / 'not-created').exists()


def test_huge_generated_file_stops_before_capture_or_promotion(tmp_path):
    from kernel_effects import MAX_FILE_BYTES
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    store = ContentStore(tmp_path)
    with (workspace / 'huge').open('wb') as stream:
        stream.truncate(MAX_FILE_BYTES + 1)
    with pytest.raises(EffectError, match='byte limit'):
        store.capture(workspace, scan(workspace, exclude=lambda _: False))
    assert list(store.root.iterdir()) == []


def test_aggregate_and_entry_limits_are_enforced(tmp_path, monkeypatch):
    import kernel_effects
    for name in ('a', 'b'):
        (tmp_path / name).write_bytes(b'12345678')
    monkeypatch.setattr(kernel_effects, 'MAX_WORKSPACE_BYTES', 10)
    with pytest.raises(EffectError, match='aggregate'):
        scan(tmp_path, exclude=lambda _: False)
    monkeypatch.setattr(kernel_effects, 'MAX_WORKSPACE_BYTES', 100)
    monkeypatch.setattr(kernel_effects, 'MAX_WORKSPACE_ENTRIES', 1)
    with pytest.raises(EffectError, match='entry-count'):
        scan(tmp_path, exclude=lambda _: False)


@pytest.mark.host_sandbox
def test_repo_intel_never_executes_repository_config(repo, tmp_path):
    from kernel_git_view import legacy_git_query
    marker = tmp_path / 'fsmonitor-ran'
    hook = tmp_path / 'monitor.sh'
    hook.write_text(f'#!/bin/sh\ntouch {marker}\n')
    hook.chmod(0o755)
    git(repo, 'config', 'core.fsmonitor', str(hook))
    assert legacy_git_query(repo, ['status', '--porcelain']) is not None
    assert not marker.exists()


def test_provisioning_boundary_never_imports_package_or_inherits_host(tmp_path, monkeypatch):
    import kernel_safe_probe
    from kernel_run_env import create_run_env
    source = tmp_path / 'source'
    (source / 'bin').mkdir(parents=True)
    (source / 'bin/python').symlink_to(sys.executable)
    (source / 'pyvenv.cfg').write_text(f'home = {Path(sys.executable).parent}\n')
    (source / 'lib/python3.11/site-packages').mkdir(parents=True)
    seen = []
    class Boundary:
        def __init__(self, directory):
            pass
        def run(self, argv, cap, **kwargs):
            seen.append((argv, cap))
            output = b'[]' if '-c' in argv else b'pip available'
            return ExecResult(0, False, output, False, len(output), 0,
                              CleanupReport(True, 0, (), 'ok'))
        def close(self):
            pass
    monkeypatch.setattr(kernel_safe_probe, 'ExecSession', Boundary)
    monkeypatch.setenv('HOST_SECRET', 'credential')
    environment = create_run_env(tmp_path / 'run', source / 'bin/python')
    assert len(seen) == 1
    assert all('import_module' not in ' '.join(argv) for argv, _ in seen)
    for _, cap in seen:
        assert cap.network == 'none' and not cap.workspace_writable
        assert 'HOST_SECRET' not in cap.env
        assert environment.root in cap.write_roots
        assert source not in cap.write_roots
        assert cap.limits.output_bytes == 65536


def test_probe_package_import_without_engine_on_sys_path(tmp_path):
    script = tmp_path / 'imports.py'
    script.write_text(
        'import sys\nsys.path.insert(0, ' + repr(str(Path(__file__).resolve().parents[1] / 'src')) + ')\n'
        'from stateframe.engine.kernel_safe_probe import run_probe\n'
        'assert callable(run_probe)\n')
    result = subprocess.run([sys.executable, '-I', '-S', str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_executable_shell_rename_restores_both_sides(tmp_path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    original = workspace / 'tool'
    original.write_text('echo hi\n')
    original.chmod(0o755)
    store = ContentStore(tmp_path)
    before = scan(workspace, exclude=lambda _: False)
    store.capture(workspace, before)
    original.rename(workspace / 'moved')
    policy = WorkspaceEffectPolicy(lambda _: False, None, lambda _: True)
    result = reconcile(workspace, before, scan(workspace, exclude=lambda _: False), policy, store, None)
    assert not result.applied
    assert original.read_text() == 'echo hi\n' and original.stat().st_mode & 0o111
    assert not (workspace / 'moved').exists()


@pytest.mark.host_sandbox
def test_plain_grep_cannot_follow_repository_symlink_to_host_secret(tmp_path):
    from kernel_plain_host import PlainHost
    root = tmp_path / 'workspace'
    root.mkdir()
    secret = tmp_path / 'secret'
    secret.write_text('TOKEN=private\n')
    (root / 'link').symlink_to(secret)
    # Only the path-authority adapter is supplied; the real grep executor and
    # capability boundary must refuse access to the linked host file.
    stub = SimpleNamespace(root=root, authority=SimpleNamespace(resolve=lambda _: root), alias=lambda s: s)
    output = PlainHost._grep(stub, {'pattern': 'TOKEN', 'path': '.'})
    assert 'TOKEN=private' not in output


def test_checkpoint_clone_does_not_dereference_repository_symlink(tmp_path):
    from kernel_checkpoint import _clone
    secret = tmp_path / 'secret'
    secret.write_text('private')
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'link').symlink_to(secret)
    destination = tmp_path / 'copy'
    _clone(source, destination)
    assert (destination / 'link').is_symlink()
    assert os.readlink(destination / 'link') == str(secret)


def test_import_metadata_does_not_grant_home(tmp_path, monkeypatch):
    from kernel_run_env import RunEnvError, _external_root
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HOME', str(home))
    with pytest.raises(RunEnvError, match='home directory'):
        _external_root(str(home), None, tmp_path / 'source')
