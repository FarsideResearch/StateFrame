"""Subframe byte-profile, cap, and request-meter handoff proofs."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernel_parallel_admission import DirectorySnapshot, admit
from kernel_parallel_meter import (_meter_calibration_snapshot, _meter_for_unit_request,
                                   _meter_from_calibration, _meter_source, _pinned_calibration,
                                   _parent_meter_calibration, estimate_subframe_request_bytes,
                                   measure_subframe_cap, select_subframe_cap_calibration)
from kernel_parallel_projection import project_unit
from kernel_parallel_records import (CheckRecipe, ParallelLimits, ParallelRecordError,
                                     ParallelRequest, UnitContract, UnitSpec,
                                     default_subframe_token_cap)
from kernel_request_budget import ENDPOINT_PROMPT_TOKEN_BUDGET
from kernel_request_diet import RequestTokenMeter, _wire_bytes
from kernel_unit_policy import UnitPolicy


_MODEL = 'deepseek-ai/DeepSeek-V4-Flash-0731'
_DEEPI = {'provider': 'deepinfra', 'model': _MODEL}
_CAP = default_subframe_token_cap(ENDPOINT_PROMPT_TOKEN_BUDGET)
_FIXTURE = Path(__file__).parent / 'fixtures' / 'subframe_meter_armA_20261104.json'


def _plan(tmp_path):
    contents = {'owned_a.txt': 'a' * 6000, 'read_a.txt': 'r' * 7000,
                'check_a.txt': 'c' * 3000, 'owned_b.txt': 'b' * 64}
    for path, content in contents.items():
        (tmp_path / path).write_text(content)
    snapshot = DirectorySnapshot(tmp_path, 0)
    a = UnitContract('a', 'update a', ('owned_a.txt',), (), (), (), ('read_a.txt',), (),
                     CheckRecipe(('check_a.txt',), (), 'none'))
    b = UnitContract('b', 'update b', ('owned_b.txt',), (), (), (), (), (),
                     CheckRecipe((), (), 'none'))
    return snapshot, ParallelRequest('call', (a, b), '2')


def _admit(snapshot, request):
    return admit(request, snapshot, (), ParallelLimits(),
                 protected=lambda _: False, case_insensitive=False)


def _calibration(*, source='calibrated_usage', samples=2, ratio=1.0):
    return {'schema': 'stateframe-request-meter-calibration/1', 'source': source,
            'bytes_per_token': ratio, 'samples': samples, 'residual_tokens': 0,
            'key': {'route': None, 'model': None, 'template': None}}


def test_cold_estimate_never_refuses_a_source_bound(tmp_path):
    snapshot, request = _plan(tmp_path)
    snapshot.meter_calibration = _calibration(source='calibrated_bytes', samples=0, ratio=0.5)
    result = _admit(snapshot, request)
    assert result.status == 'ADMITTED'
    assert result.refusal is None
    assert result.order == ('a', 'b')


def test_calibrated_upper_estimate_cannot_refuse_before_any_start_artifact(tmp_path):
    snapshot, request = _plan(tmp_path)
    snapshot.meter_calibration = _calibration()
    result = _admit(snapshot, request)
    assert result.status == 'ADMITTED'
    assert result.refusal is None
    assert result.order == ('a', 'b')


def test_run_level_usage_samples_never_create_a_prequeue_lower_bound_refusal(tmp_path):
    snapshot, request = _plan(tmp_path)
    snapshot.meter_calibration = _calibration(samples=1)
    assert _admit(snapshot, request).status == 'ADMITTED'
    snapshot.meter_calibration = _calibration(samples=2)
    assert _admit(snapshot, request).status == 'ADMITTED'


def test_routed_parent_calibration_cannot_refuse_a_different_worker_route(tmp_path):
    snapshot, request = _plan(tmp_path)
    snapshot.meter_calibration = _calibration()
    snapshot.meter_calibration['key']['route'] = 'parent-edit'
    assert _admit(snapshot, request).status == 'ADMITTED'


def test_old_unit_spec_loads_cold_and_rejects_new_field_under_old_schema(tmp_path):
    snapshot, request = _plan(tmp_path)
    admission = _admit(snapshot, request)
    spec = project_unit(admission, request, 'a', snapshot, parent_seed='seed',
                        policy_digest='policy', failure_projection='', requirements=())
    new = replace(spec, meter_calibration=_calibration())
    assert UnitSpec.from_json(new.to_json()) == new
    old = spec.to_json()
    old.pop('meter_calibration')
    old.pop('subframe_cap_calibration')
    old['schema'] = 'stateframe-unit-spec/1'
    loaded = UnitSpec.from_json(old)
    assert loaded.meter_calibration is None
    assert loaded.subframe_cap_calibration is None
    assert loaded.schema == 'stateframe-unit-spec/3'
    with pytest.raises(ParallelRecordError, match='legacy schema'):
        UnitSpec.from_json({**old, 'meter_calibration': _calibration()})


def test_snapshot_round_trip_uses_request_meter_api():
    parent = RequestTokenMeter()
    parent.observe_response(12000, 2000, 0)
    parent.observe_response(12000, 2000, 0)
    snapshot = _meter_calibration_snapshot(parent)
    assert snapshot is not None
    assert snapshot['source'] == 'calibrated_usage'
    assert snapshot['samples'] == 2
    assert snapshot['bytes_per_token'] == 6.0
    runtime = SimpleNamespace(_last_capture=SimpleNamespace(_request_token_meter=parent))
    assert _parent_meter_calibration(runtime) == snapshot
    worker = SimpleNamespace()
    capture = SimpleNamespace(request_id='first', journal=None)
    restored = _meter_from_calibration(snapshot, runtime=worker, capture=capture)
    assert worker._request_token_meter is restored
    assert restored.calibration_snapshot()['bytes_per_token'] == snapshot['bytes_per_token']
    assert restored.calibration_snapshot()['samples'] == snapshot['samples']
    assert _meter_source(restored) == 'calibrated_usage'


def test_factory_hook_installs_once_and_keeps_the_capture_meter():
    snapshot = _calibration(source='calibrated_bytes', samples=0, ratio=2.46)
    runtime = SimpleNamespace(_unit_policy=SimpleNamespace(
        spec=SimpleNamespace(meter_calibration=snapshot)))
    capture = SimpleNamespace(request_id='first', journal=None)
    first = _meter_for_unit_request(runtime, capture, None)
    assert isinstance(first, RequestTokenMeter)
    assert runtime._request_token_meter is first
    assert _meter_for_unit_request(runtime, capture, first) is first
    assert _meter_source(first) == 'calibrated_bytes'


def test_hosted_usage_calibration_fits_two_realistic_first_frames():
    parent = RequestTokenMeter()
    parent.observe_response(55000, 10000, 0)
    parent.observe_response(55000, 10000, 0)
    snapshot = _meter_calibration_snapshot(parent)
    assert snapshot is not None
    assert snapshot['samples'] == 2
    assert snapshot['source'] == 'calibrated_usage'
    profile = select_subframe_cap_calibration(_DEEPI)
    assert profile['tokens_per_byte'] == 0.3193
    # The supplied Arm A TSV has 80 parent requests and a measured maximum
    # of 0.290301 prompt tokens/byte. The versioned profile adds 10% headroom.
    # That profile estimates 12,792 and 14,388 tokens for these exact bodies;
    # the handoff meter stays more conservative at 14,502 and 16,168. The old
    # 10,600 ceiling was below the measured-profile estimate and rejected valid
    # frames; both fit the 26,675 default cap.
    for size in (40000, 45000):
        runtime = SimpleNamespace()
        capture = SimpleNamespace(request_id=f'first-{size}', journal=None)
        meter = _meter_from_calibration(snapshot, runtime=runtime, capture=capture)
        body = {'messages': [{'role': 'user', 'content': 'x' * size}], 'tools': []}
        calibrated_tokens, method = meter.measure(body)
        cold_tokens, cold_method = RequestTokenMeter().measure(body)
        profile_tokens = estimate_subframe_request_bytes(_wire_bytes(body), profile)
        assert method == cold_method == 'calibrated_bytes'
        assert 10600 < profile_tokens <= calibrated_tokens <= _CAP
        assert cold_tokens <= _CAP


def test_hosted_no_tokenize_35kb_first_frame_uses_default_cap_and_journals_profile(
        tmp_path, monkeypatch):
    import kernel_request_budget
    from probes.react_repair import _enforce_unit_first_request

    monkeypatch.delenv('STATEFRAME_PARALLEL_BYTE_TOKEN_CALIBRATION', raising=False)
    monkeypatch.delenv('STATEFRAME_PARALLEL_BYTE_TOKEN_HEADROOM_PCT', raising=False)
    source = 'VALUE = "' + 'x' * 34_900 + '"\n'
    source_path = 'pkg/a.py'
    (tmp_path / 'pkg').mkdir()
    (tmp_path / source_path).write_text(source, encoding='utf-8')
    messages = [{'role': 'user',
                 'content': f'[Current source: {source_path}]\n{source}'}]
    tools = [{'type': 'function', 'function': {'name': 'read'}}]
    calibration = select_subframe_cap_calibration(_DEEPI)
    limits = ParallelLimits.from_settings({}, {}, prompt_budget=33344)
    assert limits.unit_request_tokens == 26675
    contract = UnitContract('a', 'update the value', (source_path,), (), (), (), (), (),
                            CheckRecipe((), (), 'none'))
    spec = UnitSpec('call', 'a', 'execution', 'seed', contract, (), (), (source_path,), (),
                    '', 0, 0, 'manifest', 'policy', limits,
                    subframe_cap_calibration=calibration)
    unit = UnitPolicy(spec)
    meter = RequestTokenMeter(disabled=True)
    meter.provider_identity = json.dumps(_DEEPI, sort_keys=True)
    meter.subframe_cap_calibration = calibration
    monkeypatch.setattr(kernel_request_budget, '_request_token_meter', lambda *_args: meter)
    events = []
    runtime = SimpleNamespace(journal=SimpleNamespace(
        emit=lambda kind, **fields: events.append((kind, fields))))
    prepared = SimpleNamespace(messages=messages, capture=SimpleNamespace(request_id='hosted-35kb'))

    assert _enforce_unit_first_request(runtime, prepared, tools, unit, tmp_path,
                                       tmp_path / 'run') is None

    event, = events
    fields = event[1]
    assert event[0] == 'unit_frame_tokens'
    assert fields['request'] == 'hosted-35kb'
    assert 34000 <= fields['request_bytes'] <= 36000
    assert fields['meter_kind'] == 'calibrated_bytes'
    assert fields['confidence'] == 'default'
    assert fields['calibration_id'] == calibration['calibration_id']
    assert fields['effective_tokens_per_byte'] == 0.3193
    assert fields['estimated_tokens'] == fields['tokens']
    assert fields['tokens'] <= limits.unit_request_tokens
    assert fields['cap'] == limits.unit_request_tokens
    assert fields['counter'] == 'calibrated_bytes'
    assert meter.endpoint is None and meter._disabled is True


def test_versioned_profile_replays_all_recorded_unit_sends_without_underestimates(monkeypatch):
    monkeypatch.delenv('STATEFRAME_PARALLEL_BYTE_TOKEN_CALIBRATION', raising=False)
    monkeypatch.delenv('STATEFRAME_PARALLEL_BYTE_TOKEN_HEADROOM_PCT', raising=False)
    calibration = select_subframe_cap_calibration(_DEEPI)
    assert calibration['calibration_id'] == 'deepinfra/deepseek-ai/deepseek-v4-flash-0731@v1'
    assert calibration['tokens_per_byte'] == 0.3193
    assert calibration['residual_tokens'] == 0
    rows = json.loads(_FIXTURE.read_text(encoding='utf-8'))
    assert len(rows) == 15
    assert all(estimate_subframe_request_bytes(row['body_bytes'], calibration)
               >= row['prompt_tokens'] for row in rows)
    cap = ParallelLimits.from_settings({}, {}, prompt_budget=33344).unit_request_tokens
    assert all(estimate_subframe_request_bytes(row['body_bytes'], calibration)
               <= cap
               for row in rows)


def test_unknown_provider_uses_generic_conservative_ratio(monkeypatch):
    monkeypatch.delenv('STATEFRAME_PARALLEL_BYTE_TOKEN_CALIBRATION', raising=False)
    calibration = select_subframe_cap_calibration({'provider': 'other', 'model': 'other/model'})

    assert calibration['calibration_id'] == 'generic-conservative@v1'
    assert calibration['tokens_per_byte'] == 1 / 3
    assert estimate_subframe_request_bytes(30000, calibration) == 10000


def test_model_specific_profile_requires_exact_provider_and_model(monkeypatch):
    monkeypatch.delenv('STATEFRAME_PARALLEL_BYTE_TOKEN_CALIBRATION', raising=False)
    calibration = select_subframe_cap_calibration({'provider': 'deepinfra', 'model': _MODEL.lower()})
    assert calibration['calibration_id'] == 'generic-conservative@v1'
    assert calibration['tokens_per_byte'] == 1 / 3


def test_headroom_override_cannot_reduce_the_published_model_profile(monkeypatch):
    monkeypatch.setenv('STATEFRAME_PARALLEL_BYTE_TOKEN_HEADROOM_PCT', '0')

    calibration = select_subframe_cap_calibration(_DEEPI)

    assert calibration['calibration_id'] == 'deepinfra/deepseek-ai/deepseek-v4-flash-0731@v1'
    assert calibration['tokens_per_byte'] == 0.3193
    assert calibration['headroom_pct'] == 10


def test_replay_uses_the_pinned_calibration_when_headroom_changes(monkeypatch):
    monkeypatch.delenv('STATEFRAME_PARALLEL_BYTE_TOKEN_CALIBRATION', raising=False)
    monkeypatch.setenv('STATEFRAME_PARALLEL_BYTE_TOKEN_HEADROOM_PCT', '20')
    pinned = select_subframe_cap_calibration(_DEEPI)
    monkeypatch.setenv('STATEFRAME_PARALLEL_BYTE_TOKEN_HEADROOM_PCT', '50')
    meter = SimpleNamespace(subframe_cap_calibration=pinned)

    restored = _pinned_calibration(meter, _DEEPI, None, _MODEL)

    assert restored == pinned
    assert restored['tokens_per_byte'] == 0.34836


def test_unknown_or_understated_pinned_profiles_fall_back_to_generic(monkeypatch):
    monkeypatch.delenv('STATEFRAME_PARALLEL_BYTE_TOKEN_CALIBRATION', raising=False)
    pinned = select_subframe_cap_calibration(_DEEPI)
    pinned['calibration_id'] = 'unregistered-profile@v99'
    pinned['tokens_per_byte'] = 0.0001
    meter = SimpleNamespace(subframe_cap_calibration=pinned)

    restored = _pinned_calibration(meter, _DEEPI, None, _MODEL)

    assert restored['calibration_id'] == 'generic-conservative@v1'
    assert restored['tokens_per_byte'] == 1 / 3


def test_run_usage_feedback_does_not_change_subframe_cap_estimate(monkeypatch):
    monkeypatch.delenv('STATEFRAME_PARALLEL_BYTE_TOKEN_CALIBRATION', raising=False)
    monkeypatch.delenv('STATEFRAME_PARALLEL_BYTE_TOKEN_HEADROOM_PCT', raising=False)
    monkeypatch.setenv('STATEFRAME_PARALLEL_USAGE_FEEDBACK', 'observe')
    body = {'messages': [{'role': 'user', 'content': 'x' * 48000}], 'tools': [], 'model': _MODEL}
    meter = RequestTokenMeter(disabled=True)
    meter.endpoint = 'https://api.deepinfra.com/v1/tokenize'
    meter.provider_identity = json.dumps(_DEEPI, sort_keys=True)
    calibration = select_subframe_cap_calibration(_DEEPI)
    meter.subframe_cap_calibration = calibration
    meter.observe_response(48020, 33005, 0)
    assert meter._minimum_bytes_per_token is not None
    retained_ratio = meter._minimum_bytes_per_token

    measured = measure_subframe_cap(body, meter, cap=_CAP)

    assert measured.meter_kind == 'calibrated_bytes'
    assert measured.effective_tokens_per_byte == 0.3193
    assert measured.tokens == estimate_subframe_request_bytes(measured.request_bytes, calibration)
    assert meter._minimum_bytes_per_token == retained_ratio
    assert meter._fallback_pending is False
    assert meter._last_fallback_rejection_bound is None


def test_cap_refuses_a_conservative_overestimate(tmp_path, monkeypatch):
    monkeypatch.delenv('STATEFRAME_PARALLEL_BYTE_TOKEN_CALIBRATION', raising=False)
    monkeypatch.delenv('STATEFRAME_PARALLEL_BYTE_TOKEN_HEADROOM_PCT', raising=False)
    sample = json.loads(_FIXTURE.read_text(encoding='utf-8'))[0]
    calibration = select_subframe_cap_calibration(_DEEPI)
    estimate = estimate_subframe_request_bytes(sample['body_bytes'], calibration)
    assert estimate > sample['prompt_tokens']

    snapshot, request = _plan(tmp_path)
    admission = _admit(snapshot, request)
    spec = project_unit(admission, request, 'a', snapshot, parent_seed='seed',
                        policy_digest='policy', failure_projection='', requirements=())
    spec = replace(spec, limits=replace(spec.limits, unit_request_tokens=16000))
    policy = UnitPolicy(spec)
    assert not policy.first_request_fits(estimate, owned_source_complete=True)


def test_exact_tokenizer_count_overrides_byte_profile(monkeypatch):
    monkeypatch.delenv('STATEFRAME_PARALLEL_BYTE_TOKEN_CALIBRATION', raising=False)
    body = {'messages': [{'role': 'user', 'content': 'short prompt'}], 'tools': [], 'model': _MODEL}
    meter = RequestTokenMeter(tokenizer=lambda _content: [1, 2, 3],
                              template_overhead_tokens=7)
    meter.provider_identity = json.dumps(_DEEPI, sort_keys=True)
    meter.endpoint = 'https://api.deepinfra.com/v1/tokenize'

    measured = measure_subframe_cap(body, meter, cap=_CAP)

    assert (measured.tokens, measured.meter_kind, measured.confidence) == (10, 'tokenize', 'exact')
    assert measured.calibration_id == 'tokenize-exact'
    assert meter._availability == 'unknown'
    assert meter._fallback_pending is False


def test_exact_tokenizer_is_tried_even_above_the_byte_budget(monkeypatch):
    import kernel_request_diet as request_diet

    calls = []

    def tokenize(content, endpoint, **_kwargs):
        calls.append((content, endpoint))
        return [1, 2, 3]

    monkeypatch.setattr(request_diet, '_http_tokenize', tokenize)
    body = {'messages': [{'role': 'user', 'content': 'x' * 100_000}],
            'tools': [], 'model': _MODEL}
    meter = RequestTokenMeter(endpoint='https://api.deepinfra.com/v1/tokenize',
                              template_overhead_tokens=0)
    meter.provider_identity = json.dumps(_DEEPI, sort_keys=True)

    measured = measure_subframe_cap(body, meter, cap=_CAP)

    assert (measured.tokens, measured.meter_kind, measured.confidence) == (3, 'tokenize', 'exact')
    assert len(calls) == 1
    assert calls[0][1] == 'https://api.deepinfra.com/v1/tokenize'


def test_image_input_is_unknown_without_an_exact_provider_rule():
    body = {'messages': [{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': 'https://example.test/image.png'}}]}], 'tools': []}
    meter = RequestTokenMeter(disabled=True)

    measured = measure_subframe_cap(body, meter, cap=_CAP)

    assert measured.tokens == _CAP + 1
    assert measured.meter_kind == 'unknown_content'
    assert measured.confidence == 'unknown'
