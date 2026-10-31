"""Quiet CLI mode still exposes actionable engine refusals."""
import json

from stateframe import runner


def test_quiet_drive_prints_replay_refusal(tmp_path, monkeypatch):
    child = tmp_path / 'child_boundary.py'
    events = [
        {'kind': 'engine_start', 'turns': 1},
        {'kind': 'engine_error', 'error': 'ToolReplayRefused',
         'detail': 'external operation shell requires --replay-external'},
    ]
    child.write_text('\n'.join('print(' + repr(runner.LOG_PREFIX + json.dumps(e)) + ')'
                               for e in events) + '\nraise SystemExit(1)\n')
    monkeypatch.setattr(runner, 'CHILD', child)
    options = runner.RunOptions(repo=tmp_path, task='Resume', endpoint=None, quiet=True)
    lines = []
    assert runner.drive(tmp_path, options, lines.append) == 1
    assert lines == ['engine error: ToolReplayRefused: external operation shell requires --replay-external']
    assert 'engine_error' in (tmp_path / 'engine.log').read_text()
