"""Import metadata requests paths but cannot issue host read capabilities."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from stateframe.engine.kernel_run_env import RunEnvError, _relocate, create_run_env
from stateframe.engine.kernel_tool_settings import ToolSettings
from stateframe.shell_host import provision


def environment(tmp_path):
    repo = tmp_path / 'repo'
    source = repo / '.venv'
    (source / 'bin').mkdir(parents=True)
    (source / 'bin/python').symlink_to(sys.executable)
    (source / 'pyvenv.cfg').write_text(f'home = {Path(sys.executable).parent}\n')
    site = source / 'lib/python3.11/site-packages'
    site.mkdir(parents=True)
    return repo, source, site


def metadata(site, target, kind):
    if kind == 'finder':
        name = '__editable___fixture_finder'
        (site / (name + '.py')).write_text(f'MAPPING = {{"fixture": {str(target)!r}}}\nNAMESPACES = {{}}\n')
        (site / 'fixture.pth').write_text(f'import {name}; {name}.install()\n')
    else:
        import os
        value = os.path.relpath(target, site) if kind == 'relative' else str(target)
        (site / 'fixture.pth').write_text(value + '\n')


@pytest.mark.parametrize('kind', ['absolute', 'relative', 'finder', 'symlink', 'missing'])
def test_metadata_cannot_authorize_home_descendants(tmp_path, monkeypatch, kind):
    repo, source, site = environment(tmp_path)
    home = tmp_path / 'home'
    secret = home / '.ssh'
    secret.mkdir(parents=True)
    monkeypatch.setenv('HOME', str(home))
    target = secret
    if kind == 'symlink':
        target = repo / 'linked'
        target.symlink_to(secret, target_is_directory=True)
    elif kind == 'missing':
        target = home / '.aws'
    metadata(site, target, kind)
    with pytest.raises(RunEnvError, match='external import path lacks host read authorization'):
        create_run_env(tmp_path / 'run', source / 'bin/python', source_repo=repo)
    assert not (tmp_path / 'run/env-sites.json').exists()


@pytest.mark.parametrize('kind', ['absolute', 'relative', 'finder'])
def test_authorized_external_paths_and_workspace_remapping(tmp_path, kind):
    import shutil
    repo, source, site = environment(tmp_path)
    external = tmp_path / 'authorized/package'
    external.mkdir(parents=True)
    package = repo / 'src/package'
    package.mkdir(parents=True)
    metadata(site, external, kind)
    (site / 'project.pth').write_text(str(package) + '\n')
    destination = tmp_path / 'run/env'
    shutil.copytree(source, destination, symlinks=True)
    workspace = tmp_path / 'run/workspace'
    roots = _relocate(source, destination, repo, workspace, (external.parent,))
    assert roots == (external,)
    copied_site = destination / site.relative_to(source)
    assert (copied_site / 'project.pth').read_text().strip() == str(workspace / 'src/package')


def denied_provision(tmp_path, monkeypatch, capsys):
    repo, source, site = environment(tmp_path)
    secret = tmp_path / 'home/.ssh/id_rsa'
    secret.parent.mkdir(parents=True)
    secret.write_text('PRIVATE KEY')
    monkeypatch.setenv('HOME', str(secret.parent.parent))
    monkeypatch.setenv('PATH', '/usr/bin:/bin')
    metadata(site, secret.parent, 'absolute')
    run = tmp_path / 'run'
    (run / 'workspace').mkdir(parents=True)
    options = SimpleNamespace(tool_settings=ToolSettings(), _test_python=str(source / 'bin/python'),
                              _capabilities={})
    spec = {'repo': str(repo), 'capabilities': options._capabilities}
    provision(options, run, spec)
    assert not options.tool_settings.run_env
    assert 'run environment off: external import path lacks host read authorization' in capsys.readouterr().out
    saved = json.loads((run / 'spec.json').read_text())
    assert 'run_env' not in saved
    assert all(not secret.is_relative_to(Path(root)) for root in saved['shell_read_roots'])
    return run, secret, saved


def test_host_provision_refuses_metadata_grant(tmp_path, monkeypatch, capsys):
    denied_provision(tmp_path, monkeypatch, capsys)


@pytest.mark.host_sandbox
def test_refused_pth_cannot_make_credentials_readable_in_shell(tmp_path, monkeypatch, capsys):
    from kernel_exec_capability import shell_capability
    from kernel_exec_session import ExecSession
    run, secret, saved = denied_provision(tmp_path, monkeypatch, capsys)
    scratch = run / 'scratch'
    scratch.mkdir()
    session_dir = run / 'session'
    session_dir.mkdir()
    cap = shell_capability(workspace=run / 'workspace', run_env_dir=run / 'env', scratch_home=scratch,
                           network='none', writable=False,
                           extra_read_roots=tuple(map(Path, saved['shell_read_roots'])))
    with ExecSession(session_dir) as session:
        result = session.run(['/bin/cat', str(secret)], cap)
    assert result.cleanup.ok and result.exit_code != 0
    assert b'PRIVATE KEY' not in result.output


@pytest.mark.parametrize('grant', ['explicit', 'runtime'])
def test_provision_uses_independent_host_roots(tmp_path, monkeypatch, grant):
    from stateframe.engine import kernel_safe_probe
    from stateframe.engine.kernel_exec_session import CleanupReport, ExecResult
    repo, source, site = environment(tmp_path)
    external = tmp_path / 'authorized/package' if grant == 'explicit' else Path(sys.base_prefix) / 'lib'
    if grant == 'explicit':
        external.mkdir(parents=True)
    metadata(site, external, 'absolute')
    seen = []

    class Boundary:
        def __init__(self, directory):
            pass

        def run(self, argv, cap, **kwargs):
            seen.append(cap)
            return ExecResult(0, False, b'pip fixture', False, 11, .01, CleanupReport(True, 0, (), 'ok'))

        def close(self):
            pass

    monkeypatch.setattr(kernel_safe_probe, 'ExecSession', Boundary)
    monkeypatch.setenv('PATH', '/usr/bin:/bin')
    run = tmp_path / 'run'
    (run / 'workspace').mkdir(parents=True)
    settings = ToolSettings(shell_read=(str(external.parent),) if grant == 'explicit' else ())
    options = SimpleNamespace(tool_settings=settings, _test_python=str(source / 'bin/python'), _capabilities={})
    spec = {'repo': str(repo), 'capabilities': options._capabilities}
    provision(options, run, spec)
    assert options.tool_settings.run_env and len(seen) == 1
    assert spec['run_env']['external_import_roots'] == [str(external.resolve())]
    assert any(external.resolve().is_relative_to(root) for root in seen[0].read_roots)
    assert options._test_python == str(run / 'env/bin/python')
