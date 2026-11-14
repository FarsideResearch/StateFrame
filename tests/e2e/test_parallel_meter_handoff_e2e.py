"""Hosted usage calibration reaches shipped-profile subframes."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from _journal import measurement_events, measurement_journal_paths
from barrier_endpoint import BarrierEndpoint
from e2e_support import (assert_shipped_wire, copy_example, read_json, result_records,
                         run_cli, subframe_directories, subframe_wire_files, summary_for)


pytestmark = pytest.mark.host_sandbox


class HostedUsageEndpoint(BarrierEndpoint):
    """Serve generation usage while providing neither optional introspection API."""

    def _handler(self):
        base = super()._handler()

        class Handler(base):
            def do_GET(self) -> None:  # noqa: N802
                if self.path.rstrip('/') == '/slots':
                    self._send(404, {'error': {'message': 'not found'}})
                    return
                super().do_GET()

            def do_POST(self) -> None:  # noqa: N802
                if self.path.rstrip('/').endswith('/tokenize'):
                    self.rfile.read(int(self.headers.get('Content-Length') or 0))
                    self._send(404, {'error': {'message': 'not found'}})
                    return
                super().do_POST()

        return Handler


@pytest.mark.xfail(strict=True, reason="needs sf-alloc RequestTokenMeter.calibration_snapshot/from_calibration and the meter_calibration_seed seam (not on main yet)")
def test_two_calibrated_subframes_run_on_hosted_endpoint(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    repo = copy_example(tmp_path)
    for name in ('discount', 'quantity'):
        path = repo / 'pricing' / f'{name}.py'
        path.write_text(path.read_text() + ''.join(f'# context {index}: {name}\n'
                                                  for index in range(600)))
    script = json.loads((Path(__file__).resolve().parents[2] /
                         'examples/parallel_fixes.script.json').read_text())
    parent = [
        {'tool': 'read', 'arguments': {'path': 'pricing/discount.py'}},
        {'tool': 'read', 'arguments': {'path': 'pricing/quantity.py'}},
        script['parent'][0],
        {'tool': 'done', 'arguments': {'summary': 'Both subframes completed.'}},
    ]
    parent[2]['arguments']['subframes'] = parent[2]['arguments']['subframes'][:2]
    workers = {name: script['subframes'][name] for name in ('discount', 'quantity')}
    run_dir = tmp_path / 'run'
    with HostedUsageEndpoint(parent, workers, slots=2) as endpoint:
        completed = run_cli(isolated_engine_python, endpoint.url, repo, run_dir,
                            'Fix discount and quantity using subframes.', timeout=900)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        endpoint.assert_consumed()
        assert [record['marker'] for record in endpoint.requests[:3]] == [
            'parent', 'parent', 'parent']
    result, = result_records(run_dir).values()
    assert [(outcome.unit_id, outcome.status) for outcome in result.outcomes] == [
        ('discount', 'COMMITTED'), ('quantity', 'COMMITTED')]
    children = subframe_directories(run_dir)
    assert set(children) == {'discount', 'quantity'}
    reported_sources = {item['subframe_id']: item['meter_source']
                        for item in summary_for(run_dir)['parallel']['subframes']}
    assert reported_sources == {'discount': 'calibrated_usage',
                                'quantity': 'calibrated_usage'}
    for name, child in children.items():
        spec = read_json(child / 'unit-spec.json')
        assert spec['meter_calibration']['source'] == 'calibrated_usage'
        assert spec['meter_calibration']['samples'] >= 2
        requests = subframe_wire_files(child)
        assert len(requests) == 1
        assert_shipped_wire(requests, task_marker=f'Subframe {name}:')
        journals = measurement_journal_paths(child)
        assert len(journals) == 1
        events = measurement_events(child)
        measured = [event for event in events if event.get('kind') == 'unit_frame_tokens']
        assert len(measured) == 1
        assert measured[0]['meter_source'] == 'calibrated_usage'
        assert 0 < measured[0]['tokens'] <= 12000
