"""Explicit subframe blockage remains a terminal under the shipped profile."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from stateframe import profiles
from stateframe.resume import ResumeRefused, assess

from kernel_stateframe_runtime import StateFrameRuntime
from kernel_terminal_policy import authorize_subframe_block, terminal_category


PROFILE = Path(__file__).resolve().parents[1] / "src/stateframe/profiles/r2hc3.json"


def _runtime(*, subframe: bool, reason: str | None, run_dir: Path) -> StateFrameRuntime:
    switches = json.loads(PROFILE.read_text())['switches']
    assert switches['STATEFRAME_HARD_BUDGET_ONLY_TERMINALS'] == 'on'
    runtime = object.__new__(StateFrameRuntime)
    object.__setattr__(runtime, 'stateframe_hard_budget_only_terminals_mode', switches[
        'STATEFRAME_HARD_BUDGET_ONLY_TERMINALS'])
    object.__setattr__(runtime, '_unit_policy', SimpleNamespace() if subframe else None)
    object.__setattr__(runtime, 'current_tool_call_id', 'blocked-dispatch')
    object.__setattr__(runtime, 'journal', SimpleNamespace(path=run_dir / 'events.jsonl'))
    if reason is not None:
        object.__setattr__(runtime, 'unit_blocked_reason', reason)
        object.__setattr__(runtime, 'unit_blocked_detail', 'contract cannot be preserved')
    return runtime


def test_explicit_subframe_blocked_survives_runtime_terminal_guard(tmp_path: Path):
    runtime = _runtime(subframe=True, reason='INTERFACE_WRONG', run_dir=tmp_path)
    authorize_subframe_block(runtime, 'INTERFACE_WRONG', 'contract cannot be preserved')
    assert terminal_category('unit_blocked', runtime) == 'subframe'
    assert not (tmp_path / 'subframe-blocked.json').exists()
    runtime.system_outcome = 'unit_blocked'
    assert runtime.system_outcome == 'unit_blocked'
    assert runtime.unit_blocked_reason == 'INTERFACE_WRONG'
    assert runtime.unit_blocked_detail == 'contract cannot be preserved'


def test_unqualified_blocked_outcomes_remain_advisory(tmp_path: Path):
    for subframe, reason in ((False, 'INTERFACE_WRONG'), (True, None),
                             (True, 'INTERFACE_WRONG')):
        runtime = _runtime(subframe=subframe, reason=reason, run_dir=tmp_path)
        runtime.system_outcome = 'unit_blocked'
        assert 'system_outcome' not in runtime.__dict__
        assert terminal_category('blocked', runtime) == 'heuristic'
    runtime = _runtime(subframe=True, reason=None, run_dir=tmp_path)
    object.__setattr__(runtime, 'unit_blocked_reason', ['OTHER'])
    object.__setattr__(runtime, 'unit_blocked_detail', 'invalid reason shape')
    runtime.system_outcome = 'unit_blocked'
    assert 'system_outcome' not in runtime.__dict__


def test_off_mode_keeps_pre_subframe_unit_blocked_outcome(tmp_path: Path):
    runtime = _runtime(subframe=False, reason=None, run_dir=tmp_path)
    runtime.stateframe_hard_budget_only_terminals_mode = 'off'
    runtime.system_outcome = 'unit_blocked'
    assert runtime.system_outcome == 'unit_blocked'
    assert not (tmp_path / 'subframe-blocked.json').exists()


def test_damaged_block_marker_array_reason_refuses_cleanly(tmp_path: Path):
    run = tmp_path / 'run'
    (run / 'parent').mkdir(parents=True)
    (run / 'spec.json').write_text(json.dumps({
        'schema': 'stateframe-run/1', 'environment': {}, 'public_row': {},
        'parent_snapshot': str(run / 'parent'),
        'chat_url': 'http://127.0.0.1:8080/v1/chat/completions',
        'model': 'scripted', 'max_turns': 3, 'repo': str(run), 'profile': 'r2hc3',
        'engine_source': profiles.engine_source_summary(),
    }))
    marker = run / 'worker/measurement-runs/attempt/subframe-blocked.json'
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({'dispatch_id': 'blocked-dispatch',
                                  'reason_code': ['OTHER'], 'detail': 'bad'}))
    with pytest.raises(ResumeRefused, match='invalid fields'):
        assess(run)


def test_dispatch_authorization_is_consumed_once(tmp_path: Path):
    runtime = _runtime(subframe=True, reason='INTERFACE_WRONG', run_dir=tmp_path)
    authorize_subframe_block(runtime, 'INTERFACE_WRONG', 'contract cannot be preserved')
    runtime.system_outcome = 'unit_blocked'
    assert terminal_category('unit_blocked', runtime) == 'heuristic'
    marker = tmp_path / 'subframe-blocked.json'
    assert json.loads(marker.read_text()) == {
        'dispatch_id': 'blocked-dispatch', 'reason_code': 'INTERFACE_WRONG',
        'detail': 'contract cannot be preserved'}
    runtime.system_outcome = None
    runtime.system_outcome = 'unit_blocked'
    assert runtime.system_outcome is None
    replayed = _runtime(subframe=True, reason='INTERFACE_WRONG', run_dir=tmp_path)
    with pytest.raises(ValueError, match='invalid subframe_blocked terminal authorization'):
        authorize_subframe_block(replayed, 'INTERFACE_WRONG', 'contract cannot be preserved')
    replayed.system_outcome = 'unit_blocked'
    assert 'system_outcome' not in replayed.__dict__
    assert json.loads(marker.read_text())['dispatch_id'] == 'blocked-dispatch'


def test_dispatch_authorization_rejects_mismatched_call_id(tmp_path: Path):
    runtime = _runtime(subframe=True, reason='INTERFACE_WRONG', run_dir=tmp_path)
    authorize_subframe_block(runtime, 'INTERFACE_WRONG', 'contract cannot be preserved')
    runtime.current_tool_call_id = 'another-dispatch'
    runtime.system_outcome = 'unit_blocked'
    assert 'system_outcome' not in runtime.__dict__
    assert not (tmp_path / 'subframe-blocked.json').exists()


def test_finished_blocked_subframe_cannot_resume_or_redispatch(tmp_path: Path):
    run = tmp_path / 'run'
    (run / 'parent').mkdir(parents=True)
    spec = {
        'schema': 'stateframe-run/1', 'environment': {}, 'public_row': {},
        'parent_snapshot': str(run / 'parent'), 'chat_url': 'http://127.0.0.1:8080/v1/chat/completions',
        'model': 'scripted', 'max_turns': 3, 'repo': str(run), 'profile': 'r2hc3',
        'engine_source': profiles.engine_source_summary(),
    }
    (run / 'spec.json').write_text(json.dumps(spec))
    terminal = {'result': 'SUBFRAME_BLOCKED', 'reason_code': 'INTERFACE_WRONG',
                'detail': 'contract cannot be preserved'}
    (run / 'engine-result.json').write_text(json.dumps({'result': terminal}))
    before = sorted((path.relative_to(run).as_posix(), path.read_bytes())
                    for path in run.rglob('*') if path.is_file())
    with pytest.raises(ResumeRefused, match='subframe run already finished blocked: INTERFACE_WRONG'):
        assess(run)
    assert sorted((path.relative_to(run).as_posix(), path.read_bytes())
                  for path in run.rglob('*') if path.is_file()) == before


def test_consumed_block_dispatch_without_terminal_receipt_cannot_replay(tmp_path: Path):
    run = tmp_path / 'run'
    (run / 'parent').mkdir(parents=True)
    spec = {
        'schema': 'stateframe-run/1', 'environment': {}, 'public_row': {},
        'parent_snapshot': str(run / 'parent'), 'chat_url': 'http://127.0.0.1:8080/v1/chat/completions',
        'model': 'scripted', 'max_turns': 3, 'repo': str(run), 'profile': 'r2hc3',
        'engine_source': profiles.engine_source_summary(),
    }
    (run / 'spec.json').write_text(json.dumps(spec))
    marker = run / 'worker/measurement-runs/attempt/subframe-blocked.json'
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({
        'dispatch_id': 'blocked-dispatch', 'reason_code': 'INTERFACE_WRONG',
        'detail': 'contract cannot be preserved'}))
    before = sorted((path.relative_to(run).as_posix(), path.read_bytes())
                    for path in run.rglob('*') if path.is_file())
    with pytest.raises(ResumeRefused, match='subframe run already finished blocked: INTERFACE_WRONG'):
        assess(run)
    assert sorted((path.relative_to(run).as_posix(), path.read_bytes())
                  for path in run.rglob('*') if path.is_file()) == before


@pytest.mark.host_sandbox
def test_parent_wire_refuses_unoffered_subframe_blocked(tmp_path: Path):
    from fixtures.barrier_endpoint import BarrierEndpoint
    from fixtures.parallel.e2e_support import (
        assert_shipped_wire, copy_example, create_engine_venv, parent_wire_files,
        read_json, run_cli,
    )

    python = create_engine_venv(tmp_path / 'engine-venv')
    repo = copy_example(tmp_path)
    run_dir = tmp_path / 'run'
    parent = [
        {'tool': 'subframe_blocked', 'force': True, 'arguments': {
            'reason_code': 'OTHER', 'detail': 'parent must not block as a subframe'}},
        {'tool': 'done', 'arguments': {'summary': 'The parent refused the unavailable tool.'}},
    ]
    with BarrierEndpoint(parent, {}) as endpoint:
        completed = run_cli(python, endpoint.url, repo, run_dir,
                            'Refuse a subframe terminal on the parent surface.', max_turns=2)
    assert (run_dir / 'wire').is_dir(), completed.stdout + completed.stderr
    wires = parent_wire_files(run_dir)
    assert len(wires) >= 2
    assert_shipped_wire(wires[:2])
    first = read_json(wires[0])
    assert 'subframe_blocked' not in {
        tool['function']['name'] for tool in first['tools']}
    second = read_json(wires[1])
    assert any('[kernel-menu] This action is not available' in str(message.get('content'))
               for message in second['messages'] if message.get('role') == 'tool')
    assert not (run_dir / 'parallel').exists()
