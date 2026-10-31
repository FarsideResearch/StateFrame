"""Typical developer shell workflows cross the real request and effect pipeline."""
from __future__ import annotations

import shutil
import platform

import pytest

from ._support import run_case


pytestmark = pytest.mark.host_sandbox


@pytest.mark.parametrize("mode", ("standard", "read-only"))
def test_u1_environment_inspection(tmp_path, mode):
    case = run_case(tmp_path, "inspection.script.json", flags=("--shell", mode))
    case.assert_offered("shell")
    # Interpreters are deliberately outside the enforced inspection grammar.
    version_result = "REFUSED READ_ONLY_POLICY:" if mode == "read-only" else "Python"
    for fragment in (version_result, "Filesystem", platform.system()):
        case.assert_reply(fragment)
    assert case.summary is not None
    assert case.summary["files_changed"] == []


def test_u1_read_only_refuses_write(tmp_path):
    case = run_case(tmp_path, "one_shell.script.json",
                    substitutions={"COMMAND": "touch x"}, flags=("--shell", "read-only"))
    case.assert_reply("REFUSED READ_ONLY_POLICY:")
    assert not (case.repo / "x").exists()
    assert not (case.run_dir / "workspace" / "x").exists()


def test_u2_exploration_is_visible_without_source_read_credit(tmp_path):
    case = run_case(tmp_path, "explore.script.json")
    for fragment in ("app.py", "test_app.py", "TODO"):
        case.assert_reply(fragment)
    assert case.summary is not None
    assert case.summary["files_changed"] == []


def test_u3_u4_make_build_and_non_pytest_check(tmp_path):
    case = run_case(tmp_path, "build.script.json", flags=("--test-cmd", "make check"))
    case.assert_reply("BUILD_OK")
    case.assert_reply("make check")
    assert (case.run_dir / "workspace" / "build.ok").read_text() == "built\n"
    assert "build.ok" in case.diff


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_u3_node_runner_when_installed(tmp_path):
    case = run_case(tmp_path, "one_shell.script.json",
                    substitutions={"COMMAND": "node --version"})
    case.assert_reply("exit 0")
    case.assert_reply("v")


def test_u5_formatter_effect_invalidates_prior_read(tmp_path):
    case = run_case(tmp_path, "format.script.json", flags=("--in-place",))
    case.assert_reply("FORMATTED")
    case.assert_reply("files changed by this command: modified: app.py")
    assert 'VALUE = "after"' in case.diff
    assert 'VALUE = "after"' in (case.run_dir / "workspace" / "app.py").read_text()
    assert case.summary is not None and case.summary["succeeded"] is True, case.summary
    assert 'VALUE = "after"' in (case.repo / "app.py").read_text()
    # The request after the shell step must carry a stale-read indication before native re-read.
    assert len(case.requests) >= 3
    assert "stale" in str(case.requests[2]).lower()


def test_u6_local_install_is_used_by_shell_and_engine_checks(tmp_path, clean_test_environment):
    python = clean_test_environment(tmp_path / "test-env")
    case = run_case(tmp_path, "install.script.json", repo_name="install_repo",
                    flags=("--shell-network", "full", "--test-python", str(python)))
    case.assert_reply("environment changed:")
    case.assert_reply("installed")
    assert case.summary is not None and case.summary["succeeded"] is True, case.summary
    assert case.summary["tests_after"]["passed"] is True
    assert str(case.run_dir / "env/bin/python") in case.summary["tests_after"]["command"]
    assert (case.run_dir / "env").is_dir()
    assert not (case.repo / "localpkg.egg-info").exists()


def test_u6_network_install_refused_when_network_none(tmp_path):
    case = run_case(tmp_path, "one_shell.script.json",
                    substitutions={"COMMAND": "pip install requests"},
                    flags=("--shell-network", "none"))
    case.assert_reply("REFUSED NETWORK_OFF:")
    case.assert_reply("--shell-network full")


def test_u7_script_execution_and_u8_codegen_is_captured(tmp_path):
    case = run_case(tmp_path, "codegen.script.json", flags=("--in-place",))
    case.assert_reply("GENERATED_OK")
    case.assert_reply("files changed by this command: created: generated.py")
    assert "generated.py" in case.diff
    assert case.summary is not None and case.summary["succeeded"] is True, case.summary
    assert (case.repo / "generated.py").read_text() == 'GENERATED = "ready"\n'


def test_u10_file_housekeeping_stays_coherent_with_shell_effect(tmp_path):
    case = run_case(tmp_path, "housekeeping.script.json", flags=("--in-place",))
    case.assert_reply("files changed by this command: modified: app.py")
    assert case.summary is not None and case.summary["succeeded"] is True, case.summary
    assert not (case.repo / "obsolete.py").exists()
    assert not (case.repo / "renamed.py").exists()
    assert 'VALUE = "after"' in (case.repo / "app.py").read_text()


@pytest.mark.parametrize('mode', ['off', 'read-only', 'standard'])
def test_zero_product_actions_preserve_workspace_and_empty_diff(tmp_path, mode):
    import json
    from ._support import copy_repo
    repo = copy_repo(tmp_path)
    baseline = {p.relative_to(repo): p.read_bytes() for p in repo.rglob('*') if p.is_file()}
    script = tmp_path / 'no-product-actions.json'
    script.write_text(json.dumps({'model': 'scripted', 'tokenize': {'mode': 'ok'}, 'steps': [
        {'tool': 'done', 'arguments': {'summary': 'No product change requested.'}}]}))
    case = run_case(tmp_path, str(script), repo=repo, flags=('--shell', mode),
                    task='Inspect the repository without changing any product file.')
    assert case.summary is not None, case.process.stdout + case.process.stderr
    assert case.summary['provider_sends'] >= 1
    assert case.summary['engine_result'] != 'ENGINE_ERROR', case.summary
    assert case.summary['files_changed'] == []
    assert case.diff.strip() in {'', '(no changes)'}
    workspace = case.run_dir / 'workspace'
    assert {p: (workspace / p).read_bytes() for p in baseline} == baseline
    assert not (workspace / 'generated.py').exists()
    assert not (workspace / 'build.ok').exists()
    workers = tuple((case.run_dir / 'worker/continuation-workspaces').glob('*/repo'))
    assert workers
    for worker in workers:
        assert {p: (worker / p).read_bytes() for p in baseline} == baseline
        assert not (worker / 'generated.py').exists()
        assert not (worker / 'build.ok').exists()


def test_file_operation_directory_refusal_uses_frozen_contract(tmp_path):
    import json
    script = tmp_path / 'file-refusals.json'
    script.write_text(json.dumps({'model': 'scripted', 'tokenize': {'mode': 'ok'}, 'steps': [
        {'tool': 'file_delete', 'required': True, 'arguments': {'path': 'scripts'}},
        {'tool': 'file_move', 'required': True,
         'arguments': {'source': 'scripts', 'destination': 'moved'}},
        {'tool': 'run_tests', 'arguments': {'target': 'tests/test_app.py'}},
        {'tool': 'done', 'arguments': {'summary': 'File refusals inspected.'}}]}))
    case = run_case(tmp_path, str(script))
    replies = [str(m.get('content', '')) for request in case.requests
               for m in request['messages'] if m.get('role') == 'tool']
    assert sum(reply.startswith('REFUSED IS_DIRECTORY:') for reply in replies) >= 2
    assert case.summary is not None and case.summary['files_changed'] == []
