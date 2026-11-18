from pathlib import Path
from dataclasses import replace
import shlex
import subprocess
import sys
import threading
import time
import os
import re
import secrets
import signal
from types import SimpleNamespace

import pytest
from fixtures.process_signals import record_identity, signal_owned, signal_verified

import kernel_unit_policy
import kernel_check_permits
import kernel_check_truth_adapter
import kernel_pilot
import kernel_execution_policy
import kernel_test_isolation
from kernel_check_receipts import record_check_output
from kernel_stateframe_action_ledger import build_action_context
from kernel_stateframe_framefit5 import (
    _serialize_state_message, assemble_framefit5_request, build_typed_state,
)
from kernel_pilot import ACTIVE_RUNTIME
from probes import react_repair

from kernel_parallel_records import CheckRecipe, ParallelLimits, UnitContract, UnitSpec
from kernel_stateframe_runtime import StateFrameRuntime
from kernel_unit_policy import UnitPolicy, ready_to_auto_finish, run_frozen_check


@pytest.fixture(autouse=True)
def installed_check_permit(tmp_path, monkeypatch):
    monkeypatch.setattr(kernel_check_permits, '_installed', (tmp_path / 'permits', 1))


def policy():
    recipe = CheckRecipe(('tests/test_a.py',), ('tests/test_a.py::test_value',), 'pytest')
    contract = UnitContract('a', 'fix a', ('pkg/a.py',), (), (), (), (), (), recipe)
    spec = UnitSpec('call', 'a', 'execution', 'seed', contract, (), (),
                    ('pkg/a.py', 'tests/test_a.py'), (), '', 0, 0, 'manifest', 'policy',
                    ParallelLimits())
    return UnitPolicy(spec)


def candidate(tmp_path):
    root = tmp_path / 'candidate'
    (root / 'pkg').mkdir(parents=True)
    (root / 'tests').mkdir()
    (root / 'pkg/a.py').write_text('value = 1\n')
    (root / 'tests/test_a.py').write_text('def test_value(): pass\n')
    return root


def test_frozen_selector_and_check_writes_invalidate_without_candidate_effect(tmp_path):
    root = candidate(tmp_path)
    called = []

    def execute(clone: Path, selector: str) -> str:
        called.append(selector)
        (clone / 'pkg/a.py').write_text('value = 2\n')
        (clone / 'other.py').write_text('created by test\n')
        return 'pytest exit 0 — 1 passed'

    receipt = run_frozen_check(policy(), root, execute=execute,
                               evidence_dir=tmp_path / 'evidence')
    assert called == ['tests/test_a.py::test_value']
    assert receipt.status == 'INVALID'
    assert receipt.invalid_reason == 'CHECK_WROTE_FILES'
    assert receipt.wrote_paths == ('other.py', 'pkg/a.py')
    assert (root / 'pkg/a.py').read_text() == 'value = 1\n'
    assert not (root / 'other.py').exists()
    assert 'pytest exit 0' in Path(receipt.raw_evidence_ref).read_text()


def test_clean_check_pass_and_planted_failure_control(tmp_path):
    root = candidate(tmp_path)
    passing = run_frozen_check(policy(), root, execute=lambda clone, selector:
                               'pytest exit 0 — 1 passed', evidence_dir=tmp_path / 'evidence')
    failing = run_frozen_check(policy(), root, execute=lambda clone, selector:
                               'pytest exit 1 — 1 failed\nE   expected pytest exit 0',
                               evidence_dir=tmp_path / 'evidence')
    assert passing.status == 'PASS'
    assert failing.status == 'FAIL'
    assert passing.recipe_digest == failing.recipe_digest
    assert passing.product_manifest_digest == failing.product_manifest_digest
    from kernel_effects import scan
    assert passing.product_manifest_modes == {
        path: entry.mode for path, entry in scan(root, exclude=kernel_unit_policy._check_ignored).items()
    }


def test_frozen_check_reuses_only_an_identical_passing_receipt(tmp_path):
    root = candidate(tmp_path)
    calls = []

    def execute(_clone, selector):
        calls.append(selector)
        return 'pytest exit 0 — 1 passed'

    original = run_frozen_check(
        policy(), root, execute=execute, evidence_dir=tmp_path / 'evidence')
    reused = run_frozen_check(
        policy(), root, execute=lambda *_: pytest.fail('identical check was re-executed'),
        evidence_dir=tmp_path / 'evidence', previous_receipt=original)

    assert reused is original
    assert calls == ['tests/test_a.py::test_value']

    (root / 'tests/test_a.py').write_text('def test_value(): assert True\n# changed input\n')
    rerun = run_frozen_check(
        policy(), root, execute=execute, evidence_dir=tmp_path / 'evidence',
        previous_receipt=original)
    assert rerun is not original
    assert rerun.status == 'PASS'
    assert rerun.harness_manifest_digest != original.harness_manifest_digest
    assert calls == ['tests/test_a.py::test_value', 'tests/test_a.py::test_value']


def test_frozen_check_reuse_invalidates_on_dependency_environment_or_selector_change(
        tmp_path, monkeypatch):
    root = candidate(tmp_path)
    selected = policy()
    calls = []

    def execute(_clone, selector):
        calls.append(selector)
        return 'pytest exit 0 — 1 passed'

    receipt = run_frozen_check(
        selected, root, execute=execute, evidence_dir=tmp_path / 'evidence')
    (root / 'requirements.txt').write_text('sample-dependency==1.0\n')
    dependency_changed = run_frozen_check(
        selected, root, execute=execute, evidence_dir=tmp_path / 'evidence',
        previous_receipt=receipt)
    assert dependency_changed.product_manifest_digest != receipt.product_manifest_digest

    old_path = os.environ.get('PATH', '')
    monkeypatch.setenv('PATH', old_path + ':different-check-path')
    environment_changed = run_frozen_check(
        selected, root, execute=execute, evidence_dir=tmp_path / 'evidence',
        previous_receipt=dependency_changed)
    assert environment_changed.environment_digest != dependency_changed.environment_digest

    if os.name == 'posix':
        monkeypatch.setenv('SF_UNIT_CHECK_SURROGATE', '\udcff')
        surrogate_environment = run_frozen_check(
            selected, root, execute=execute, evidence_dir=tmp_path / 'evidence',
            previous_receipt=environment_changed)
        assert surrogate_environment.environment_digest != environment_changed.environment_digest
    else:
        surrogate_environment = environment_changed

    changed_recipe = replace(
        selected.spec.contract.check,
        selectors=('tests/test_a.py::test_value_again',))
    changed_policy = UnitPolicy(replace(
        selected.spec,
        contract=replace(selected.spec.contract, check=changed_recipe)))
    selector_changed = run_frozen_check(
        changed_policy, root, execute=execute, evidence_dir=tmp_path / 'evidence',
        previous_receipt=surrogate_environment)
    assert selector_changed.recipe_digest != surrogate_environment.recipe_digest
    assert calls == [
        'tests/test_a.py::test_value', 'tests/test_a.py::test_value',
        'tests/test_a.py::test_value',
        *(['tests/test_a.py::test_value'] if os.name == 'posix' else []),
        'tests/test_a.py::test_value_again',
    ]


def test_live_detached_descendant_never_returns_a_check_receipt(tmp_path):
    root = candidate(tmp_path)

    def uncontained(_clone: Path, _selector: str) -> str:
        raise kernel_unit_policy.UnitCheckContainmentError(
            'check containment unverified: detached descendant remained live')

    with pytest.raises(kernel_unit_policy.UnitCheckContainmentError,
                       match='descendant remained live'):
        run_frozen_check(policy(), root, execute=uncontained,
                         evidence_dir=tmp_path / 'evidence')
    assert (root / 'pkg/a.py').read_text() == 'value = 1\n'
    assert not (tmp_path / 'evidence').exists()


def test_auto_finish_requires_current_green_check_complete_paths_and_no_pending_work(tmp_path):
    root = candidate(tmp_path)
    selected = policy()
    original = {'pkg/a.py': 'value = 1\n'}
    (root / 'pkg/a.py').write_text('value = 2\n')
    green = run_frozen_check(selected, root, execute=lambda _clone, _selector:
                             'pytest exit 0 — 1 passed', evidence_dir=tmp_path / 'evidence')
    red = run_frozen_check(selected, root, execute=lambda _clone, _selector:
                           'pytest exit 1 — 1 failed', evidence_dir=tmp_path / 'evidence')
    assert ready_to_auto_finish(selected, root, green, original)
    assert not ready_to_auto_finish(selected, root, red, original)
    assert not ready_to_auto_finish(selected, root, green, original, pending_calls=True)
    assert not ready_to_auto_finish(selected, root, green, original, pending_mutations=True)
    assert not ready_to_auto_finish(selected, root, green, original, open_refusal=True)
    (root / 'pkg/a.py').write_text('value = 3\n')
    assert not ready_to_auto_finish(selected, root, green, original)
    (root / 'pkg/a.py').unlink()
    assert not ready_to_auto_finish(selected, root, green, original)
    no_check = CheckRecipe((), (), 'none')
    no_check_spec = replace(selected.spec, contract=replace(selected.spec.contract, check=no_check))
    assert not ready_to_auto_finish(UnitPolicy(no_check_spec), root, green, original)


def test_auto_finish_reports_exact_decline_for_pass_receipt(tmp_path):
    root = candidate(tmp_path)
    selected = policy()
    original = {'pkg/a.py': 'value = 1\n'}
    (root / 'pkg/a.py').write_text('value = 2\n')
    green = run_frozen_check(selected, root, execute=lambda _clone, _selector:
                             'pytest exit 0 — 1 passed', evidence_dir=tmp_path / 'evidence')
    reasons = []
    assert not ready_to_auto_finish(selected, root, green, original,
                                    pending_calls=True, decline=reasons.append)
    assert reasons == ['pending_calls']
    (root / 'pkg/a.py').write_text('value = 3\n')
    assert not ready_to_auto_finish(selected, root, green, original, decline=reasons.append)
    assert reasons == ['pending_calls', 'manifest_mismatch']


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux orphan reaping behavior')
def test_check_group_accepts_only_zombie_members():
    ready_read, ready_write = os.pipe()
    release_read, release_write = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(ready_read)
        os.close(release_write)
        os.setsid()
        os.write(ready_write, b'1')
        os.close(ready_write)
        os.read(release_read, 1)
        os.close(release_read)
        os._exit(0)
    os.close(ready_write)
    os.close(release_read)
    try:
        assert os.read(ready_read, 1) == b'1'
        assert not kernel_unit_policy._group_is_only_zombies(pid)
        os.write(release_write, b'1')
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if kernel_unit_policy._group_is_only_zombies(pid):
                break
            time.sleep(0.01)
        assert kernel_unit_policy._group_is_only_zombies(pid)
        kernel_unit_policy.reap_check_group(pid)
    finally:
        os.close(ready_read)
        os.close(release_write)
        os.waitpid(pid, 0)


def test_projected_status_has_one_bounded_row_per_frozen_selector(tmp_path):
    root = candidate(tmp_path)
    recipe = CheckRecipe(('tests/test_a.py',),
                         ('tests/test_a.py::test_value', 'tests/test_a.py::test_other'), 'pytest')
    contract = UnitContract('a', 'fix a', ('pkg/a.py',), (), (), (), (), (), recipe)
    spec = UnitSpec('call', 'a', 'execution', 'seed', contract, (), (),
                    ('pkg/a.py', 'tests/test_a.py'), ('pkg/hidden.py',), '', 0, 0,
                    'manifest', 'policy', ParallelLimits())
    outputs = iter(('pytest exit 0\n1 passed in 0.01s',
                    'pytest exit 1\nFile "pkg/hidden.py", line 7\n'
                    'E   HIDDEN_SENTINEL\n1 failed in 0.01s'))
    receipt = run_frozen_check(UnitPolicy(spec), root,
                               execute=lambda _clone, _selector: next(outputs),
                               evidence_dir=tmp_path / 'evidence')
    assert receipt.status == 'FAIL'
    assert receipt.projected_observation.count('selector=') == 2
    assert 'selector=tests/test_a.py::test_value exit=0 passed=1 failed=0 errors=0' in receipt.projected_observation
    assert 'selector=tests/test_a.py::test_other exit=1 passed=0 failed=1 errors=0' in receipt.projected_observation
    assert '[frame in withheld file pkg/hidden.py:7]' in receipt.projected_observation
    assert 'HIDDEN_SENTINEL' not in receipt.projected_observation
    assert UnitPolicy(spec).project_selector_status('tests/test_a.py::test_value', '') == (
        'selector=tests/test_a.py::test_value exit=unknown passed=? failed=? errors=?')


@pytest.mark.host_sandbox
def test_command_recipe_cannot_write_outside_disposable_clone(tmp_path):
    root = candidate(tmp_path)
    outside = tmp_path / 'outside.txt'
    script = root / 'tests/write_outside.py'
    script.write_text('from pathlib import Path\n'
                      'print("CHECK_STARTED", flush=True)\n'
                      f'Path({str(outside)!r}).write_text("escaped")\n')
    selector = f'{shlex.quote(sys.executable)} tests/write_outside.py'
    recipe = CheckRecipe(('tests/write_outside.py',), (selector,), 'run_check_command')
    contract = UnitContract('a', 'check a', ('pkg/a.py',), (), (), (), (), (), recipe)
    spec = UnitSpec('call', 'a', 'execution', 'seed', contract, (), (),
                    ('pkg/a.py', 'tests/write_outside.py'), (), '', 0, 0,
                    'manifest', 'policy', ParallelLimits())
    receipt = run_frozen_check(UnitPolicy(spec), root,
                               execute=kernel_unit_policy.run_check_command,
                               evidence_dir=tmp_path / 'evidence')
    assert receipt.status in {'FAIL', 'INVALID'}
    assert 'CHECK_STARTED' in Path(receipt.raw_evidence_ref).read_text()
    assert not outside.exists()


def test_command_recipe_uses_check_sandbox_boundary(tmp_path, monkeypatch):
    from kernel_check_watchdog import CheckWatchdogConfig
    monkeypatch.delenv("STATEFRAME_CHECK_STALL_SECONDS", raising=False)
    monkeypatch.delenv("STATEFRAME_PUBLIC_CHECK_CPU_SECONDS", raising=False)
    root = candidate(tmp_path)
    script = root / 'tests/check.py'
    script.write_text('assert True\n')
    selector = f'{shlex.quote(sys.executable)} tests/check.py'
    recipe = CheckRecipe(('tests/check.py',), (selector,), 'run_check_command')
    contract = UnitContract('a', 'check a', ('pkg/a.py',), (), (), (), (), (), recipe)
    spec = UnitSpec('call', 'a', 'execution', 'seed', contract, (), (),
                    ('pkg/a.py', 'tests/check.py'), (), '', 0, 0,
                    'manifest', 'policy', ParallelLimits())
    calls = []

    def sandbox(command, clone, **options):
        calls.append((command, clone, options))
        assert (clone / 'tests/check.py').read_text() == 'assert True\n'
        return SimpleNamespace(completed=subprocess.CompletedProcess(command, 0, 'ok', ''),
                               timed_out=False, pid=9_999_999, cleanup_finding=None)

    monkeypatch.setattr(kernel_execution_policy, 'run_sandboxed', sandbox)
    monkeypatch.setattr(kernel_unit_policy, '_matching_check_processes', lambda _token: ())
    receipt = run_frozen_check(UnitPolicy(spec), root,
                               execute=kernel_unit_policy.run_check_command,
                               evidence_dir=tmp_path / 'evidence')
    assert receipt.status == 'PASS'
    assert len(calls) == 1
    assert calls[0][0] == [sys.executable, 'tests/check.py']
    assert calls[0][1] != root
    assert calls[0][2]['timeout'] is None
    assert calls[0][2]['cpu_seconds'] is None
    assert isinstance(calls[0][2]['check_watchdog'], CheckWatchdogConfig)
    assert calls[0][2]['check_watchdog'].stall_seconds > 0
    assert calls[0][2]['output_limit'] == 256 * 1024
    assert re.fullmatch(r'[0-9a-f]{32}',
                        calls[0][2]['internal_env']['STATEFRAME_CHECK_TOKEN'])


def test_only_linux_subframe_checks_request_namespace_init(tmp_path, monkeypatch):
    root = candidate(tmp_path)
    monkeypatch.setattr(kernel_execution_policy.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(kernel_execution_policy, 'require_capabilities', lambda: object())
    calls = []

    def builder(command, _root, env, **options):
        calls.append(options)
        return command, env

    kernel_execution_policy._prepared([sys.executable], root, None, None, False, 64,
                                      isolation_builder=builder)
    kernel_execution_policy._prepared(
        [sys.executable], root, None, {'STATEFRAME_CHECK_TOKEN': secrets.token_hex(16)},
        False, 64, isolation_builder=builder)
    assert calls == [
        {'readonly_workspace': False, 'nproc_budget': 64},
        {'readonly_workspace': False, 'nproc_budget': 64, 'private_pid_check': True},
    ]
    monkeypatch.setattr(kernel_test_isolation, 'backend', lambda: 'bwrap')
    normal, _ = kernel_test_isolation.isolated_command([sys.executable], root, {},
                                                        nproc_budget=64)
    check, _ = kernel_test_isolation.isolated_command([sys.executable], root, {},
                                                       nproc_budget=64,
                                                       private_pid_check=True)
    assert '--unshare-all' in normal
    assert '--as-pid-1' not in normal
    assert '--as-pid-1' in check


@pytest.mark.host_sandbox
def test_check_command_refuses_reaped_leader_with_live_child(tmp_path, monkeypatch):
    root = candidate(tmp_path)
    (root / 'tests/spawn.py').write_text(
        'import subprocess, sys\n'
        'child = subprocess.Popen([sys.executable, "tests/delayed.py"], '
        'stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, '
        'stderr=subprocess.DEVNULL)\n'
        'print(f"child_pid={child.pid}", flush=True)\n')
    (root / 'tests/delayed.py').write_text('import time\ntime.sleep(30)\n')
    child_pid = []
    child_identity = []

    def process_boundary(command, clone, **options):
        launched_after = time.time()
        leader = subprocess.Popen(command, cwd=clone, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, start_new_session=True, text=True,
                                  env={**os.environ, **options['internal_env']})
        try:
            stdout, stderr = leader.communicate(timeout=5)
            child_pid.append(int(re.search(r'child_pid=(\d+)', stdout).group(1)))
            child_identity.append(record_identity(
                child_pid[0], launched_after=launched_after,
                command_fragment='tests/delayed.py'))
            return SimpleNamespace(
                completed=subprocess.CompletedProcess(command, leader.returncode, stdout, stderr),
                timed_out=False, pid=leader.pid, cleanup_finding=None)
        finally:
            if leader.poll() is None:
                signal_owned(leader, 9)
                leader.wait(timeout=5)

    monkeypatch.setattr(kernel_execution_policy, 'run_sandboxed', process_boundary)
    try:
        with pytest.raises(kernel_unit_policy.UnitCheckCleanupError,
                           match='containment unverified'):
            kernel_unit_policy.run_check_command(root, f'{sys.executable} tests/spawn.py')
        assert len(child_pid) == 1
        signal_verified(child_pid[0], child_identity[0], 0)
    finally:
        if child_pid:
            try:
                signal_verified(child_pid[0], child_identity[0], 9)
            except ProcessLookupError:
                pass


def test_check_process_scan_is_scoped_to_own_user_and_token(monkeypatch):
    token = secrets.token_hex(16)
    other = secrets.token_hex(16)
    calls = []

    def process_table(command, **kwargs):
        calls.append(command)
        kwargs['stdout'].write((
            f'101 {os.getuid()} python STATEFRAME_CHECK_TOKEN={token}\n'
            f'102 {os.getuid()} python STATEFRAME_CHECK_TOKEN={other}\n'
            f'103 {os.getuid() + 1} python STATEFRAME_CHECK_TOKEN={token}\n').encode())
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(kernel_unit_policy, 'sys', SimpleNamespace(platform='darwin'))
    monkeypatch.setattr(kernel_unit_policy.subprocess, 'run', process_table)
    assert kernel_unit_policy._matching_check_processes(token) == (101,)
    assert calls == [['ps', '-Eww', '-axo', 'pid=,uid=,command=']]


def test_unverified_check_cleanup_invalidates_command_pass(tmp_path, monkeypatch):
    root = candidate(tmp_path)
    script = root / 'tests/check.py'
    script.write_text('assert True\n')
    selector = f'{shlex.quote(sys.executable)} tests/check.py'
    recipe = CheckRecipe(('tests/check.py',), (selector,), 'run_check_command')
    unit = UnitPolicy(replace(policy().spec,
                              contract=replace(policy().spec.contract, check=recipe)))
    monkeypatch.setattr(kernel_execution_policy, 'run_sandboxed',
                        lambda command, _root, **_kwargs: SimpleNamespace(
                            completed=subprocess.CompletedProcess(command, 0, 'ok', ''),
                            timed_out=False, pid=None if sys.platform == 'linux' else 9_999_999,
                            cleanup_finding=None))

    def unavailable(_token):
        raise kernel_unit_policy.UnitCheckCleanupError('process scan unavailable')

    if sys.platform == 'darwin':
        monkeypatch.setattr(kernel_unit_policy, '_matching_check_processes', unavailable)
    receipt = run_frozen_check(unit, root, execute=kernel_unit_policy.run_check_command,
                               evidence_dir=tmp_path / 'evidence')
    assert receipt.status == 'INVALID'
    assert receipt.invalid_reason == 'CHECK_CLEANUP_FAILED'
    assert ('check PID namespace cleanup cannot be verified' if sys.platform == 'linux'
            else 'process scan unavailable') in receipt.projected_observation


@pytest.mark.parametrize('cleanup_result, expected_status', [
    (None, 'PASS'),
    ('check result invalid: detached descendants (742,) terminated after the check',
     'INVALID'),
])
def test_pre_reap_cleanup_finding_survives_group_sweep_and_reap(
        tmp_path, monkeypatch, cleanup_result, expected_status):
    root = candidate(tmp_path)
    selector = f'{shlex.quote(sys.executable)} tests/check.py'
    recipe = CheckRecipe(('tests/check.py',), (selector,), 'run_check_command')
    unit = UnitPolicy(replace(policy().spec,
                              contract=replace(policy().spec.contract, check=recipe)))
    events = []

    class FakeProcess:
        pid = 741
        stdin = stdout = stderr = None
        returncode = None

        def wait(self, timeout):
            assert timeout == 1
            events.append('leader reaped')
            self.returncode = 0
            return 0

    process = FakeProcess()
    monkeypatch.setattr(kernel_execution_policy, 'sandboxed_popen',
                        lambda *_args, **_kwargs: (process, object()))
    monkeypatch.setattr(kernel_execution_policy, 'child_exited_unreaped',
                        lambda child: child is process and child.returncode is None)
    monkeypatch.setattr(kernel_execution_policy, 'recorded_launch_identity',
                        lambda child: SimpleNamespace(pid=child.pid))
    monkeypatch.setattr(kernel_execution_policy, 'signal_launch_group',
                        lambda child, _identity, _signal: events.append(
                            ('group swept', child.returncode)) or True)
    monkeypatch.setattr(kernel_execution_policy, 'ProcessTable',
                        lambda: SimpleNamespace(snapshot=lambda: {}))
    monkeypatch.setattr(kernel_execution_policy, '_kill_group',
                        lambda _child: pytest.fail('second launch-unit kill after group sweep'))
    monkeypatch.setattr(kernel_unit_policy, '_terminate_check_token',
                        lambda _token: events.append('token checked') or cleanup_result)
    monkeypatch.setattr(kernel_unit_policy, 'reap_check_group',
                        lambda _pid, **_kwargs: events.append('group verified'))

    receipt = run_frozen_check(unit, root, execute=kernel_unit_policy.run_check_command,
                               evidence_dir=tmp_path / 'evidence')
    assert receipt.status == expected_status
    assert receipt.invalid_reason == ('CHECK_CLEANUP_FAILED' if cleanup_result else None)
    assert receipt.wrote_paths == ()
    assert events[:3] == [('group swept', None), 'token checked', 'leader reaped']
    if cleanup_result:
        assert events == [('group swept', None), 'token checked', 'leader reaped']
        assert cleanup_result in Path(receipt.raw_evidence_ref).read_text()
    else:
        assert events[-1] == 'group verified'


def test_pre_reap_cleanup_exception_is_not_masked_by_second_kill(tmp_path, monkeypatch):
    process = SimpleNamespace(pid=741, stdin=None, stdout=None, stderr=None, returncode=None)
    events = []

    def wait(timeout):
        assert timeout == 1
        events.append('leader reaped')
        process.returncode = 0

    process.wait = wait
    monkeypatch.setattr(kernel_execution_policy, 'sandboxed_popen',
                        lambda *_args, **_kwargs: (process, object()))
    monkeypatch.setattr(kernel_execution_policy, 'child_exited_unreaped',
                        lambda child: child is process and child.returncode is None)
    monkeypatch.setattr(kernel_execution_policy, 'recorded_launch_identity',
                        lambda child: SimpleNamespace(pid=child.pid))
    monkeypatch.setattr(kernel_execution_policy, 'signal_launch_group',
                        lambda child, _identity, _signal: events.append('group swept') or True)
    monkeypatch.setattr(kernel_execution_policy, 'ProcessTable',
                        lambda: SimpleNamespace(snapshot=lambda: {}))
    monkeypatch.setattr(kernel_execution_policy, '_kill_group',
                        lambda _child: pytest.fail('second launch-unit kill masked cleanup error'))

    def failed_cleanup():
        events.append('token checked')
        raise kernel_unit_policy.UnitCheckCleanupError('detached scan unavailable')

    with pytest.raises(kernel_unit_policy.UnitCheckCleanupError,
                       match='detached scan unavailable'):
        kernel_execution_policy.run_sandboxed(
            [sys.executable, 'tests/check.py'], tmp_path,
            pre_reap_cleanup=failed_cleanup)
    assert events == ['group swept', 'token checked', 'leader reaped']


@pytest.mark.host_sandbox
@pytest.mark.skipif(sys.platform != 'linux', reason='Linux procfs and bubblewrap only')
def test_linux_check_ignores_unreadable_unrelated_environ(tmp_path, monkeypatch):
    blocker = tmp_path / 'unreadable_environ.py'
    blocker.write_text('import time\nprint("ready", flush=True)\ntime.sleep(120)\n')
    unrelated = subprocess.Popen([sys.executable, str(blocker)], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True)
    try:
        assert unrelated.stdout is not None
        assert unrelated.stdout.readline().strip() == 'ready'
        environment = Path('/proc') / str(unrelated.pid) / 'environ'
        assert environment.parent.stat().st_uid == os.getuid()
        original_open = Path.open

        def unreadable_environ(path, *args, **kwargs):
            if path == environment:
                raise PermissionError(13, 'Permission denied', str(path))
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(Path, 'open', unreadable_environ)
        with pytest.raises(PermissionError):
            environment.open('rb')
        root = candidate(tmp_path)
        (root / 'tests/pass.py').write_text('from pathlib import Path\n'
                                            'assert Path("pkg/a.py").read_text() == "value = 1\\n"\n')
        (root / 'tests/fail.py').write_text('raise AssertionError("planted failure")\n')
        for script, expected in (('pass.py', 'PASS'), ('fail.py', 'FAIL')):
            selector = f'{shlex.quote(sys.executable)} tests/{script}'
            recipe = CheckRecipe((f'tests/{script}',), (selector,), 'run_check_command')
            selected = UnitPolicy(replace(policy().spec,
                                          contract=replace(policy().spec.contract, check=recipe)))
            receipt = run_frozen_check(selected, root,
                                       execute=kernel_unit_policy.run_check_command,
                                       evidence_dir=tmp_path / 'evidence')
            assert receipt.status == expected, receipt.projected_observation
            assert receipt.invalid_reason is None
            assert receipt.wrote_paths == ()
    finally:
        unrelated.terminate()
        unrelated.communicate(timeout=5)


@pytest.mark.host_sandbox
def test_command_recipe_executes_frozen_command_in_clone(tmp_path):
    root = candidate(tmp_path)
    script = root / 'tests/check.py'
    script.write_text('from pathlib import Path\n'
                      'assert Path("pkg/a.py").read_text() == "value = 1\\n"\n')
    selector = f'{shlex.quote(sys.executable)} tests/check.py'
    recipe = CheckRecipe(('tests/check.py',), (selector,), 'run_check_command')
    contract = UnitContract('a', 'check a', ('pkg/a.py',), (), (), (), (), (), recipe)
    spec = UnitSpec('call', 'a', 'execution', 'seed', contract, (), (),
                    ('pkg/a.py', 'tests/check.py'), (), '', 0, 0, 'manifest', 'policy',
                    ParallelLimits())
    from kernel_unit_policy import run_check_command
    receipt = run_frozen_check(UnitPolicy(spec), root, execute=run_check_command,
                               evidence_dir=tmp_path / 'evidence')
    assert receipt.status == 'PASS', receipt.projected_observation
    assert receipt.wrote_paths == ()
    assert '[tests/check.py' not in receipt.projected_observation
    assert 'check exit 0' in receipt.projected_observation


def test_pytest_raw_output_reaches_only_private_evidence(tmp_path, monkeypatch):
    root = candidate(tmp_path)
    hidden = root / 'pkg/hidden.py'
    hidden.write_text('HIDDEN_TRACEBACK_SENTINEL = 7\n')
    unit = policy()
    unit = UnitPolicy(replace(unit.spec, hidden_paths=('pkg/hidden.py',)))
    monkeypatch.setattr(kernel_unit_policy, 'active', lambda: unit)
    monkeypatch.setattr(react_repair, '_effect_test_subprocess', lambda *_args, **_kwargs: (
        subprocess.CompletedProcess([], 1,
                                    'File "pkg/hidden.py", line 7\n'
                                    'E   HIDDEN_TRACEBACK_SENTINEL\n1 failed\n', ''), None))
    monkeypatch.setattr(kernel_check_truth_adapter, 'begin_check_execution',
                        lambda *_args, **_kwargs: 'execution-1')
    monkeypatch.setattr(kernel_pilot, 'charge', lambda *_args, **_kwargs: None)
    events = []
    journal = SimpleNamespace(path=tmp_path / 'journal.jsonl',
                              emit=lambda name, **fields: events.append((name, fields)))
    runtime = SimpleNamespace(journal=journal, owner=SimpleNamespace(root=root, current_step=1),
                              binding=lambda **_kwargs: {'product_revision': 'sha256:product'})

    def observe(selectors, exit_code, binding, observation, **_kwargs):
        record_check_output(runtime, selector=selectors[0], status='FAIL',
                            stdout=observation, exit_code=exit_code,
                            revision=binding['product_revision'])

    runtime.test_observation = observe
    token = kernel_pilot.ACTIVE_RUNTIME.set(runtime)
    try:
        result = react_repair.run_tests(root, 'a', 'tests/test_a.py::test_value',
                                        row={'_skip_db': True, 'python': sys.executable})
    finally:
        kernel_pilot.ACTIVE_RUNTIME.reset(token)
    evidence = list((tmp_path / 'unit-check-evidence').glob('*.txt'))
    assert len(evidence) == 1
    assert 'HIDDEN_TRACEBACK_SENTINEL' in evidence[0].read_text()
    public = str(runtime.host_check_receipts) + result
    assert '[frame in withheld file pkg/hidden.py:7]' in public
    assert 'HIDDEN_TRACEBACK_SENTINEL' not in public
    assert 'HIDDEN_TRACEBACK_SENTINEL' not in str(runtime._check_output_receipts)
    task = {'role': 'user', 'content': 'Task: Repair this subframe.'}
    messages = [task,
                {'role': 'assistant', 'tool_calls': [{
                    'id': 'check-1', 'type': 'function', 'function': {
                        'name': 'run_tests',
                        'arguments': '{"selector":"tests/test_a.py::test_value"}'}}]},
                {'role': 'tool', 'tool_call_id': 'check-1', 'content': result}]
    context = build_action_context(messages, workspace_root=root)
    context['journal_receipt_records'] = []
    runtime._framefit_action_context = context
    runtime.edit_contract = {'edit_authority': {'editable_paths': ['pkg/a.py']}}
    runtime.protected_tests = {}
    runtime.frame_ceiling = 10_000
    runtime.owner.task_row = {}
    frame = build_typed_state(messages, context, runtime)
    prepared = [{'role': 'system', 'content': 'rules'}, task,
                {'role': 'user', 'content': _serialize_state_message(
                    {'state': {'framefit5': frame}})}]
    next_request = assemble_framefit5_request(prepared, messages,
                                              live_frame=frame, runtime=runtime)
    assert '[frame in withheld file pkg/hidden.py:7]' in str(next_request)
    assert 'HIDDEN_TRACEBACK_SENTINEL' not in str(next_request)


def test_worker_run_tests_pins_selector_even_when_other_requested(tmp_path, monkeypatch):
    root = candidate(tmp_path)
    selected = []
    events = []
    journal = SimpleNamespace(path=tmp_path / 'journal.jsonl',
                              emit=lambda name, **fields: events.append((name, fields)))
    runtime = SimpleNamespace(journal=journal)
    monkeypatch.setattr(kernel_unit_policy, 'active', policy)

    def execute(clone, fn, explicit, rel_hint=None, row=None):
        selected.append((explicit, (clone / 'pkg/a.py').read_text()))
        return 'pytest exit 0 — 1 passed'

    monkeypatch.setattr(react_repair, '_run_tests_impl', execute)
    token = ACTIVE_RUNTIME.set(runtime)
    try:
        observation = react_repair.run_tests(root, 'a', 'tests/other.py::test_wrong')
    finally:
        ACTIVE_RUNTIME.reset(token)
    assert selected == [('tests/test_a.py::test_value', 'value = 1\n')]
    assert 'pytest exit 0' in observation
    assert runtime._unit_check_receipt.status == 'PASS'
    assert events == [('unit_check', {
        'recipe_digest': policy().spec.contract.check.digest(),
        'status': 'PASS', 'invalid_reason': None, 'wrote_paths': [], 'reused': False,
    })]


def test_invalid_unit_check_receipt_is_a_settled_check_verdict():
    from kernel_catalog import _check_settled

    # An INVALID frozen check carries an UNKNOWN outcome; it is still the host's final verdict.
    assert _check_settled({'status': 'INVALID', 'outcome': 'UNKNOWN'})
    assert not _check_settled({'status': 'UNKNOWN', 'outcome': 'UNKNOWN'})


def test_invalid_frozen_check_test_receipt_is_settled_not_stranded(tmp_path, monkeypatch):
    """An INVALID frozen check must reach the catalog as a settled host verdict.

    The canonical row types an INVALID check UNKNOWN.  Unsettled, the catalog demands a
    by-identity re-run resolver a subframe cannot dispatch, so the next request fails the
    final-surface proof and the unit stops after one check.
    """
    from kernel_catalog import _check_settled

    root = candidate(tmp_path)
    monkeypatch.setattr(kernel_unit_policy, 'active', policy)

    def execute(clone, fn, explicit, rel_hint=None, row=None):
        (clone / 'created_by_check.txt').write_text('not allowed\n')
        return 'pytest exit 0 \u2014 1 passed'

    monkeypatch.setattr(react_repair, '_run_tests_impl', execute)
    receipt = run_frozen_check(policy(), root, execute=lambda clone, selector: execute(clone, 'a', selector),
                               evidence_dir=tmp_path / 'evidence')
    assert receipt.status == 'INVALID'
    fields = kernel_unit_policy.host_final_check_fields(receipt)
    assert fields == {'status': 'INVALID'}
    # What the runtime stores for this run: canonical UNKNOWN outcome, no status of its own.
    stored = {'kind': 'test', 'canonical_outcome': 'UNKNOWN', 'outcome': 'UNKNOWN'}
    assert not _check_settled(stored)
    assert _check_settled({**stored, **fields})


def test_host_final_check_fields_only_mark_invalid_and_not_run():
    fields = kernel_unit_policy.host_final_check_fields
    assert fields(SimpleNamespace(status='NOT_RUN')) == {'status': 'NOT_RUN'}
    assert fields(SimpleNamespace(status='PASS')) == {}
    assert fields(SimpleNamespace(status='FAIL')) == {}
    assert fields(None) == {}


def test_unit_check_binding_follows_the_files_on_disk_not_a_stale_view(tmp_path):
    """The repair rail checks an edit before the view epoch advances; the binding must not lag."""
    from kernel_live_projection import workspace_binding

    root = candidate(tmp_path)
    stale = workspace_binding(root, 'env')
    (root / 'pkg/a.py').write_text('value = 2\n')
    fresh = workspace_binding(root, 'env')
    assert fresh['product_revision'] != stale['product_revision']
    calls = []

    class Runtime:
        epoch = 0

        def binding(self, command=()):
            current = fresh if self.epoch else stale
            return {'product_revision': current['product_revision'],
                    'test_harness_revision': current['test_harness_revision'],
                    'environment_revision': 'env'}

        def note_mutation(self, changed_paths=(), outcome=None):
            calls.append(('note', outcome))

        def _invalidate_binding_cache(self):
            self.epoch += 1

    runtime = Runtime()
    stale_binding = runtime.binding()
    refreshed = react_repair._fresh_unit_check_binding(runtime, policy(), root, stale_binding)
    assert refreshed['product_revision'] == fresh['product_revision']
    assert calls == [('note', 'check_binding_refresh')]
    # An already-current binding is returned untouched and does not advance any epoch.
    assert react_repair._fresh_unit_check_binding(runtime, policy(), root, refreshed) is refreshed
    assert calls == [('note', 'check_binding_refresh')]


def test_repeated_check_in_one_step_repeats_a_fail_and_journals_one_event(tmp_path, monkeypatch):
    root = candidate(tmp_path)
    runs = []
    events = []
    journal = SimpleNamespace(path=tmp_path / 'journal.jsonl',
                              emit=lambda name, **fields: events.append((name, fields)))
    runtime = SimpleNamespace(journal=journal, owner=SimpleNamespace(current_step=3))
    monkeypatch.setattr(kernel_unit_policy, 'active', policy)

    def execute(clone, fn, explicit, rel_hint=None, row=None):
        runs.append(explicit)
        return 'pytest exit 1 — 1 failed'

    monkeypatch.setattr(react_repair, '_run_tests_impl', execute)
    token = ACTIVE_RUNTIME.set(runtime)
    try:
        react_repair.run_tests(root, 'a', 'tests/test_a.py::test_value')
        first = runtime._unit_check_receipt
        react_repair.run_tests(root, 'a', 'tests/test_a.py::test_value')
        assert runtime._unit_check_receipt is first
        assert len(runs) == 1
        assert [fields['status'] for name, fields in events if name == 'unit_check'] == ['FAIL']
        # A later step is an explicit retry and must execute the check again.
        runtime.owner.current_step = 4
        react_repair.run_tests(root, 'a', 'tests/test_a.py::test_value')
    finally:
        ACTIVE_RUNTIME.reset(token)
    assert len(runs) == 2
    assert [fields['status'] for name, fields in events if name == 'unit_check'] == ['FAIL', 'FAIL']


def test_current_receipt_requires_identical_inputs_and_a_completed_status(tmp_path):
    root = candidate(tmp_path)
    unit = policy()
    failing = run_frozen_check(unit, root, execute=lambda *_: 'pytest exit 1 — 1 failed',
                               evidence_dir=tmp_path / 'evidence')
    assert kernel_unit_policy.current_receipt(unit, root, failing) is failing
    assert kernel_unit_policy.current_receipt(unit, root, None) is None
    (root / 'pkg/a.py').write_text('value = 2\n')
    assert kernel_unit_policy.current_receipt(unit, root, failing) is None
    def writing_execute(clone, selector):
        (clone / 'x.py').write_text('x')
        return 'pytest exit 0 — 1 passed'

    wrote = run_frozen_check(unit, root, evidence_dir=tmp_path / 'evidence',
                             execute=writing_execute)
    assert wrote.status == 'INVALID'
    assert kernel_unit_policy.current_receipt(unit, root, wrote) is None


def test_passing_pytest_write_never_publishes_green_runtime_receipt(tmp_path, monkeypatch):
    root = candidate(tmp_path)
    unit = policy()
    monkeypatch.setattr(kernel_unit_policy, 'active', lambda: unit)
    monkeypatch.setattr(kernel_pilot, 'charge', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(kernel_check_truth_adapter, 'begin_check_execution',
                        lambda *_args, **_kwargs: 'execution-1')

    def subprocess_boundary(_command, *, root, **_kwargs):
        (root / 'pkg/a.py').write_text('value = 2\n')
        return subprocess.CompletedProcess([], 0, '1 passed in 0.01s\n', ''), None

    monkeypatch.setattr(react_repair, '_effect_test_subprocess', subprocess_boundary)
    owner = SimpleNamespace(root=root, task_id='unit-check', current_step=1,
                            task_row={'files': ['pkg/a.py'], 'test_file': 'tests/test_a.py'},
                            authorize=lambda *_args, **_kwargs: None)
    runtime = StateFrameRuntime(
        owner, output=tmp_path / 'journal.jsonl', target=root / 'pkg/a.py',
        model='scripted', tools=[], temperature=0,
        source_resolver=lambda _path, _symbol: None)
    runtime.stateframe_readiness_authority_mode = 'on'
    token = ACTIVE_RUNTIME.set(runtime)
    try:
        result = react_repair.run_tests(root, 'a', 'tests/test_a.py::test_value',
                                        row={'_skip_db': True, 'python': sys.executable})
    finally:
        ACTIVE_RUNTIME.reset(token)
    assert runtime._unit_check_receipt.status == 'INVALID'
    assert runtime._unit_check_receipt.wrote_paths == ('pkg/a.py',)
    assert len(runtime.test_receipts) == 1
    assert runtime.test_receipts[0]['exit_code'] == 1
    assert runtime.host_check_receipts[-1]['status'] == 'INVALID'
    assert runtime._readiness_request_decision is not None
    assert runtime._readiness_request_decision.ready is False
    assert runtime._readiness_request_decision.green is False
    assert 'INVALID' in result
    assert (root / 'pkg/a.py').read_text() == 'value = 1\n'


def test_pytest_cleanup_failure_has_one_final_invalid_receipt(tmp_path, monkeypatch):
    root = candidate(tmp_path)
    monkeypatch.setattr(kernel_unit_policy, 'active', policy)
    monkeypatch.setattr(kernel_pilot, 'charge', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(kernel_check_truth_adapter, 'begin_check_execution',
                        lambda *_args, **_kwargs: 'execution-1')
    monkeypatch.setattr(react_repair, '_effect_boundary_enabled', lambda: False)
    sandbox_calls = []

    def sandbox(command, _root, **options):
        sandbox_calls.append(options)
        return SimpleNamespace(
            completed=subprocess.CompletedProcess(command, 0, '1 passed\n', ''),
            timed_out=False, pid=123456, output_truncated=False)

    monkeypatch.setattr(kernel_execution_policy, 'run_sandboxed',
                        sandbox)

    def cleanup_failure(_pid, **_kwargs):
        raise kernel_unit_policy.UnitCheckCleanupError('descendant identity unavailable')

    monkeypatch.setattr(kernel_unit_policy, 'reap_check_group', cleanup_failure)
    premature = []
    monkeypatch.setattr(react_repair, '_host_refusal_observation',
                        lambda *_args, **_kwargs: premature.append('HOST_REFUSED') or 'HOST_REFUSED')
    events = []
    observations = []
    runtime = SimpleNamespace(
        journal=SimpleNamespace(path=tmp_path / 'journal.jsonl',
                                emit=lambda name, **fields: events.append((name, fields))),
        owner=SimpleNamespace(root=root, current_step=1),
        binding=lambda **_kwargs: {'product_revision': 'sha256:product'},
        test_observation=lambda selectors, code, binding, **kwargs:
            observations.append((tuple(selectors), code, kwargs['typed_observation']['status'])),
    )
    token = ACTIVE_RUNTIME.set(runtime)
    try:
        result = react_repair.run_tests(root, 'a', 'tests/test_a.py::test_value',
                                        row={'_skip_db': True, 'python': sys.executable})
    finally:
        ACTIVE_RUNTIME.reset(token)
    assert premature == []
    assert len(sandbox_calls) == 1
    assert re.fullmatch(r'[0-9a-f]{32}',
                        sandbox_calls[0]['internal_env']['STATEFRAME_CHECK_TOKEN'])
    assert runtime._unit_check_receipt.status == 'INVALID'
    assert runtime._unit_check_receipt.invalid_reason == 'CHECK_CLEANUP_FAILED'
    assert observations == [(('tests/test_a.py::test_value',), 1, 'INVALID')]
    assert [row['status'] for name, row in events if name == 'unit_check'] == ['INVALID']
    assert 'INVALID CHECK_CLEANUP_FAILED' in result


def test_check_without_installed_permit_refuses_before_execute(tmp_path, monkeypatch):
    monkeypatch.setattr(kernel_check_permits, '_installed', None)
    called = []
    receipt = run_frozen_check(policy(), candidate(tmp_path),
                               execute=lambda *_args: called.append(True) or 'pytest exit 0',
                               evidence_dir=tmp_path / 'evidence')
    assert receipt.status == 'INVALID'
    assert receipt.invalid_reason == 'RESOURCES_UNAVAILABLE'
    assert called == []


def test_run_tests_refusal_publishes_invalid_readiness_once(tmp_path, monkeypatch):
    root = candidate(tmp_path)
    monkeypatch.setattr(kernel_check_permits, '_installed', None)
    monkeypatch.setattr(kernel_unit_policy, 'active', policy)
    called = []
    monkeypatch.setattr(react_repair, '_run_tests_impl',
                        lambda *_args, **_kwargs: called.append('executed'))
    runtime = SimpleNamespace(
        journal=SimpleNamespace(path=tmp_path / 'journal.jsonl',
                                emit=lambda *_args, **_kwargs: None),
        binding=lambda: {'product_revision': 'sha256:product'},
        test_observation=lambda selectors, exit_code, binding, **kwargs:
            called.append((selectors, exit_code, kwargs['typed_observation']['status'])),
    )
    token = ACTIVE_RUNTIME.set(runtime)
    try:
        observation = react_repair.run_tests(root, 'a', 'tests/test_a.py::test_value')
    finally:
        ACTIVE_RUNTIME.reset(token)
    assert called == [(['tests/test_a.py::test_value'], 1, 'INVALID')]
    assert runtime._unit_check_receipt.invalid_reason == 'RESOURCES_UNAVAILABLE'
    assert 'INVALID RESOURCES_UNAVAILABLE' in observation


def test_one_check_permit_serializes_two_concurrent_checks(tmp_path):
    root = candidate(tmp_path)
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()
    receipts = []

    def first_execute(_clone, _selector):
        first_entered.set()
        assert release_first.wait(5)
        return 'pytest exit 0 — 1 passed'

    def second_execute(_clone, _selector):
        second_entered.set()
        return 'pytest exit 1 — 1 failed'

    first = threading.Thread(target=lambda: receipts.append(run_frozen_check(
        policy(), root, execute=first_execute, evidence_dir=tmp_path / 'evidence')))
    second = threading.Thread(target=lambda: receipts.append(run_frozen_check(
        policy(), root, execute=second_execute, evidence_dir=tmp_path / 'evidence')))
    first.start()
    assert first_entered.wait(5)
    second.start()
    assert not second_entered.wait(0.15)
    release_first.set()
    first.join(5)
    second.join(5)
    assert not first.is_alive() and not second.is_alive()
    assert second_entered.is_set()
    assert sorted(receipt.status for receipt in receipts) == ['FAIL', 'PASS']


def test_clone_tracebacks_project_identically_across_runs(tmp_path):
    root = candidate(tmp_path)
    (root / 'pkg/hidden.py').write_text('secret = 1\n')
    unit = UnitPolicy(replace(policy().spec, hidden_paths=('pkg/hidden.py',)))
    projections = []
    for index in range(2):
        def execute(clone, _selector):
            return ('pytest exit 1\n'
                    f'  File "{clone / "pkg/a.py"}", line 7, in test_value\n'
                    'E   expected 4\n'
                    f'  File "{clone / "pkg/hidden.py"}", line 9, in helper\n'
                    'E   secret\n1 failed in 0.01s')
        receipt = run_frozen_check(unit, root, execute=execute,
                                   evidence_dir=tmp_path / f'evidence-{index}')
        assert receipt.status == 'FAIL'
        projections.append(receipt.projected_observation)
    assert projections[0] == projections[1]
    assert 'File "pkg/a.py", line 7' in projections[0]
    assert '[frame in withheld file pkg/hidden.py:9]' in projections[0]
    assert 'stateframe-unit-check-' not in projections[0]


@pytest.mark.host_sandbox
def test_check_reaps_delayed_child_before_clone_scan(tmp_path):
    root = candidate(tmp_path)
    delayed_name = f'delayed_{tmp_path.name}.py'
    script = root / 'tests/spawn.py'
    script.write_text('import subprocess, sys\n'
                      f'child = subprocess.Popen([sys.executable, "tests/{delayed_name}"], '
                      'stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, '
                      'stderr=subprocess.DEVNULL)\n'
                      'print(f"child_pid={child.pid}", flush=True)\n')
    (root / 'tests' / delayed_name).write_text(
        'import time\nfrom pathlib import Path\n'
        'time.sleep(2)\nPath("pkg/a.py").write_text("late\\n")\n')
    selector = f'{shlex.quote(sys.executable)} tests/spawn.py'
    recipe = CheckRecipe(('tests/spawn.py',), (selector,), 'run_check_command')
    contract = replace(policy().spec.contract, check=recipe)
    unit = UnitPolicy(replace(policy().spec, contract=contract))
    receipt = run_frozen_check(unit, root, execute=kernel_unit_policy.run_check_command,
                               evidence_dir=tmp_path / 'evidence')
    assert receipt.status in {'PASS', 'INVALID'}
    if receipt.status == 'INVALID':
        assert receipt.invalid_reason == 'CHECK_CLEANUP_FAILED'
    match = re.search(r'child_pid=(\d+)', Path(receipt.raw_evidence_ref).read_text())
    if receipt.status == 'PASS':
        assert match is not None
    if match is not None and sys.platform != 'linux':
        # A PID may be reused after the check reaps its child; inspect identity.
        child = subprocess.run(['ps', '-p', match.group(1), '-o', 'command='],
                               capture_output=True, text=True, check=False)
        assert delayed_name not in child.stdout
    processes = subprocess.run(['ps', '-axo', 'command='], capture_output=True,
                               text=True, check=True)
    assert delayed_name not in processes.stdout
    assert receipt.wrote_paths == ()
    assert (root / 'pkg/a.py').read_text() == 'value = 1\n'


@pytest.mark.host_sandbox
def test_detached_check_child_cannot_publish_pass_or_write_later(tmp_path, monkeypatch):
    root = candidate(tmp_path)
    delayed_name = f'detached_{tmp_path.name}.py'
    script = root / 'tests/spawn_detached.py'
    script.write_text(
        'import subprocess, sys\n'
        f'child = subprocess.Popen([sys.executable, "tests/{delayed_name}"], '
        'stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, '
        'stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True)\n'
        'print(f"child_pid={child.pid}", flush=True)\n')
    (root / 'tests' / delayed_name).write_text(
        'import time\nfrom pathlib import Path\n'
        'time.sleep(5)\nPath("pkg/a.py").write_text("late\\n")\n')
    check_token = secrets.token_hex(16)
    monkeypatch.setattr(kernel_unit_policy, 'new_check_token', lambda: check_token)
    selector = f'{shlex.quote(sys.executable)} tests/spawn_detached.py'
    recipe = CheckRecipe(('tests/spawn_detached.py',), (selector,), 'run_check_command')
    unit = UnitPolicy(replace(policy().spec,
                              contract=replace(policy().spec.contract, check=recipe)))
    launched_after = time.time()
    receipt = run_frozen_check(unit, root, execute=kernel_unit_policy.run_check_command,
                               evidence_dir=tmp_path / 'evidence')
    try:
        assert receipt.status == ('PASS' if sys.platform == 'linux' else 'INVALID')
        assert receipt.invalid_reason == (None if sys.platform == 'linux' else 'CHECK_CLEANUP_FAILED')
        assert receipt.wrote_paths == ()
        if sys.platform == 'darwin':
            assert 'detached descendants' in Path(receipt.raw_evidence_ref).read_text()
            assert kernel_unit_policy._matching_check_processes(check_token) == ()
        else:
            assert 'child_pid=' in Path(receipt.raw_evidence_ref).read_text()
            processes = subprocess.run(['ps', '-axo', 'command='], capture_output=True,
                                       text=True, check=True)
            assert delayed_name not in processes.stdout
        assert (root / 'pkg/a.py').read_text() == 'value = 1\n'
    finally:
        if sys.platform == 'darwin':
            for pid in kernel_unit_policy._matching_check_processes(check_token):
                try:
                    identity = record_identity(pid, launched_after=launched_after,
                                               command_fragment=delayed_name)
                    signal_verified(pid, identity, signal.SIGKILL)
                except ProcessLookupError:
                    pass
