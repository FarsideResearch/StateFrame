import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints
import types

import pytest

from kernel_parallel_records import (Record, ParallelConfigError, ParallelRecordError, ConsumeSpec,
                                     Param, CheckRecipe, UnitContract, ExportSpec, canonical_json)
import kernel_parallel_records as records


def sample(cls):
    if cls is ConsumeSpec:
        return ConsumeSpec('export', None, None)
    values = {}
    for name, typ in get_type_hints(cls).items():
        if name == 'schema':
            continue
        values[name] = sample_type(typ)
    return cls(**values)


def sample_type(typ):
    origin = get_origin(typ)
    args = get_args(typ)
    if origin in (Union, types.UnionType):
        return None if type(None) in args else sample_type(args[0])
    if str(origin) == 'typing.Literal':
        return args[0]
    if origin is tuple:
        return ()
    if origin is dict:
        return {}
    if isinstance(typ, type) and issubclass(typ, Record):
        return sample(typ)
    return {str: 'value', int: 1, float: 1.5, bool: True, Any: None}[typ]


@pytest.mark.parametrize('cls', [getattr(records, name) for name in vars(records)
                                  if isinstance(getattr(records, name), type)
                                  and issubclass(getattr(records, name), Record)
                                  and getattr(records, name) is not Record])
def test_every_record_round_trips_strictly(cls):
    value = sample(cls)
    encoded = value.to_json()
    assert cls.from_json(json.loads(json.dumps(encoded))) == value
    assert cls.from_json(encoded).digest() == value.digest()
    with pytest.raises(ParallelRecordError, match='unknown'):
        cls.from_json({**encoded, 'extra': 1})
    with pytest.raises(ParallelRecordError, match='missing'):
        cls.from_json({key: item for key, item in encoded.items() if key != next(iter(encoded))})


def test_nested_types_and_canonical_digest():
    param = Param('x', 'pos', False, None)
    spec = ExportSpec('e', 'a.py', 'f', records.CanonicalSignature('function', (param,), None, ()))
    unit = UnitContract('a', 'done', ('a.py',), (), (spec,), (ConsumeSpec('other', None, None),), (), (),
                        CheckRecipe((), (), 'none'))
    parsed = UnitContract.from_json(unit.to_json())
    assert parsed.exports[0].signature.params[0] == param
    assert parsed.digest() == unit.digest()
    payload = unit.to_json()
    payload['exports'][0]['signature']['params'][0]['has_default'] = 1
    with pytest.raises(ParallelRecordError, match='has_default'):
        UnitContract.from_json(payload)
    with pytest.raises(ParallelRecordError, match='export_id or path and symbol'):
        ConsumeSpec(None, None, None)


def test_recorded_v1_sealed_proposal_keeps_canonical_bytes_and_digest():
    path = (Path(__file__).parent / 'fixtures' / 'parallel'
            / 'recorded_sealed_proposal_v1.json')
    recorded_bytes = path.read_bytes()
    assert recorded_bytes.endswith(b'\n')
    canonical_recorded = recorded_bytes[:-1]
    value = json.loads(recorded_bytes)
    assert canonical_json(value) == canonical_recorded

    expected_digest = hashlib.sha256(canonical_recorded).hexdigest()
    assert expected_digest == 'cfe8c4cd6c58f8d06c07cb9be0feb57e4d4bfeb40127cd509c3d288896f3afe1'
    proposal = records.Proposal.from_json(value)

    assert canonical_json(proposal.to_json()) == canonical_recorded
    assert proposal.digest() == expected_digest


def test_imports_both_contexts_in_fresh_process(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = root / 'src'
    engine = source / 'stateframe' / 'engine'
    modules = ('kernel_check_permits', 'kernel_parallel_records', 'kernel_parallel_admission',
               'kernel_parallel_contracts', 'kernel_parallel_projection', 'kernel_parallel_txn')
    script = tmp_path / 'imports.py'
    script.write_text('import importlib, sys\n'
                      f'sys.path[:0] = {[str(source), str(engine)]!r}\n'
                      f'for name in {modules!r}:\n'
                      '    package = importlib.import_module("stateframe.engine." + name)\n'
                      '    flat = importlib.import_module(name)\n'
                      '    assert package.__name__ != flat.__name__\n')
    result = subprocess.run([sys.executable, '-I', str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_non_json_payload_and_non_string_key_fail_closed():
    value = records.HostRequest(1, 'c', 1, 'launch', 'unit', {}, 'digest').to_json()
    value['payload'] = {'bad': object()}
    with pytest.raises(ParallelRecordError, match='expected JSON value'):
        records.HostRequest.from_json(value)
    value = records.HostRequest(1, 'c', 1, 'launch', 'unit', {}, 'digest').to_json()
    value[3] = 'extra'
    with pytest.raises(ParallelRecordError, match='unknown'):
        records.HostRequest.from_json(value)


def test_l1_modules_are_release_declared_with_current_hashes():
    import hashlib
    root = Path(__file__).resolve().parents[1] / 'src/stateframe/engine'
    manifest = json.loads((root / 'release-manifest.json').read_text())['files']
    for name in ('kernel_check_permits.py', 'kernel_parallel_records.py',
                 'kernel_parallel_admission.py', 'kernel_parallel_contracts.py',
                 'kernel_parallel_projection.py', 'kernel_parallel_txn.py'):
        assert manifest[name] == hashlib.sha256((root / name).read_bytes()).hexdigest()


def test_schema_literal_can_be_passed_explicitly_but_cannot_decode_wrong_version():
    recipe = CheckRecipe((), (), 'none', schema='stateframe-check-recipe/1')
    assert recipe.schema == 'stateframe-check-recipe/1'
    with pytest.raises(ParallelRecordError, match='schema'):
        CheckRecipe((), (), 'none', schema='stateframe-check-recipe/2')
    encoded = recipe.to_json()
    encoded['schema'] = 'stateframe-check-recipe/2'
    with pytest.raises(ParallelRecordError, match='schema'):
        CheckRecipe.from_json(encoded)


def test_check_receipt_records_manifest_modes_and_migrates_v1():
    receipt = records.CheckReceipt(
        'recipe', 'product', 'harness', 'environment', 'policy', 'PASS', None,
        (), 'evidence.txt', '1 passed', product_manifest_modes={'pkg/a.py': 0o755})
    encoded = receipt.to_json()
    assert encoded['schema'] == 'stateframe-parallel-check/2'
    assert records.CheckReceipt.from_json(encoded) == receipt

    legacy = dict(encoded)
    legacy.pop('product_manifest_modes')
    legacy['schema'] = 'stateframe-parallel-check/1'
    migrated = records.CheckReceipt.from_json(legacy)
    assert migrated.schema == 'stateframe-parallel-check/2'
    assert migrated.product_manifest_modes == {}

    encoded['product_manifest_modes'] = {'pkg/a.py': True}
    with pytest.raises(ParallelRecordError, match='expected int'):
        records.CheckReceipt.from_json(encoded)


def test_nonsemantic_contract_fields_have_stable_digest_and_requirement_refs_round_trip():
    recipe = CheckRecipe(('a.py', 'b.py'), ('test_b', 'test_a'), 'pytest')
    left = UnitContract('a', 'outcome', ('b.py', 'a.py'), (),
                        (ExportSpec('z', 'b.py', 'z', None), ExportSpec('a', 'a.py', 'a', None)),
                        (), ('z.py', 'c.py'), ('R2', 'R1'), recipe)
    right = UnitContract('a', 'outcome', ('a.py', 'b.py'), (),
                         (ExportSpec('a', 'a.py', 'a', None), ExportSpec('z', 'b.py', 'z', None)),
                         (), ('c.py', 'z.py'), ('R1', 'R2'), recipe)
    assert left.digest() == right.digest()
    other = UnitContract('b', 'outcome', ('c.py',), (), (), (), (), (), recipe)
    assert records.ParallelRequest('call', (left, other), '2').digest() == records.ParallelRequest(
        'call', (other, right), '2').digest()
    assert left.check.selectors == ('test_b', 'test_a')
    clause = records.RequirementClause('R1', 'neutral', 'intersect', ('b.py::f', 'a.py'))
    assert records.RequirementClause.from_json(clause.to_json()) == clause
    with pytest.raises(ParallelRecordError, match='structured reference'):
        records.RequirementClause('R1', 'neutral', 'intersect', ('../escape.py',))


def test_measured_parallel_records_round_trip_exact_values():
    measured = records.ParallelMeasured(18.75, 31.5, 1.68, 3, 4.25)
    encoded = json.loads(json.dumps(measured.to_json()))
    assert records.ParallelMeasured.from_json(encoded) == measured
    assert encoded == {
        'parallel_wall_seconds': 18.75,
        'estimated_serial_seconds': 31.5,
        'speedup': 1.68,
        'peak_active_workers': 3,
        'queue_wait_seconds': 4.25,
    }
    with pytest.raises(ParallelRecordError, match='missing'):
        records.ParallelMeasured.from_json({key: value for key, value in encoded.items()
                                            if key != 'queue_wait_seconds'})
    with pytest.raises(ParallelRecordError, match='unknown'):
        records.ParallelMeasured.from_json({**encoded, 'extra': 1})


def test_parallel_result_round_trip_keeps_unit_and_parallel_timings():
    usage = records.UnitUsage(2, 3, 150, 40, 100)
    outcome = records.UnitOutcome('a', 'COMMITTED', None, None, 1,
                                  (records.FileDelta('a.py', 2, 1, 'modify'),), True, 'PASS', usage,
                                  8.125, 0.25)
    admission = records.AdmissionResult('ADMITTED', None, (), (), ('a',), 4, 'manifest',
                                        records.ParallelLimits())
    measured = records.ParallelMeasured(10.0, 8.125, 0.8125, 1, 0.25)
    result = records.ParallelResult('call', admission, (outcome,), (1,), 'PASS', (), 'VERIFIED', 'done', measured)
    decoded = records.ParallelResult.from_json(json.loads(json.dumps(result.to_json())))
    assert decoded == result
    assert decoded.outcomes[0].wall_seconds == 8.125
    assert decoded.outcomes[0].queue_wait_seconds == 0.25
    assert decoded.measured.to_json() == {
        'parallel_wall_seconds': 10.0,
        'estimated_serial_seconds': 8.125,
        'speedup': 0.8125,
        'peak_active_workers': 1,
        'queue_wait_seconds': 0.25,
    }


def test_parallel_limit_options_are_the_cli_and_profile_source_of_truth():
    expected = {
        'max_units': ('--parallel-max-subframes', 'STATEFRAME_PARALLEL_MAX_SUBFRAMES'),
        'min_units': ('--parallel-min-subframes', 'STATEFRAME_PARALLEL_MIN_SUBFRAMES'),
        'max_paths_per_unit': ('--parallel-max-paths-per-subframe',
                               'STATEFRAME_PARALLEL_MAX_PATHS_PER_SUBFRAME'),
        'max_paths_total': ('--parallel-max-paths-total', 'STATEFRAME_PARALLEL_MAX_PATHS_TOTAL'),
        'unit_request_tokens': ('--parallel-subframe-tokens',
                                'STATEFRAME_PARALLEL_SUBFRAME_REQUEST_TOKENS'),
        'max_owned_source_bytes': ('--parallel-max-owned-bytes', 'STATEFRAME_PARALLEL_MAX_OWNED_SOURCE_BYTES'),
        'unit_max_requests': ('--parallel-subframe-max-requests',
                              'STATEFRAME_PARALLEL_SUBFRAME_MAX_REQUESTS'),
        'parent_reserve_requests': ('--parallel-parent-reserve', 'STATEFRAME_PARALLEL_PARENT_RESERVE_REQUESTS'),
        'check_permits': ('--parallel-check-permits', 'STATEFRAME_PARALLEL_CHECK_PERMITS'),
        'max_concurrency': ('--parallel-max', 'STATEFRAME_PARALLEL_MAX_CONCURRENCY'),
    }
    assert {name: (option['flag'], option['switch'])
            for name, option in records.PARALLEL_LIMIT_OPTIONS.items()} == expected
    with pytest.raises(TypeError):
        records.PARALLEL_LIMIT_OPTIONS['max_units']['flag'] = '--wrong'


def test_parallel_limits_load_defaults_and_cli_overrides_profile():
    defaults = records.ParallelLimits.from_settings({}, {}, prompt_budget=33344)
    assert defaults == records.ParallelLimits()
    assert defaults.unit_request_tokens == 26675
    assert defaults.check_permits == max(1, min(4, (os.cpu_count() or 1) // 4))
    profile = {'STATEFRAME_PARALLEL_MAX_SUBFRAMES': '5',
               'STATEFRAME_PARALLEL_MAX_CONCURRENCY': '9'}
    from_profile = records.ParallelLimits.from_settings(profile, {}, prompt_budget=20000)
    assert (from_profile.max_units, from_profile.max_concurrency) == (5, 9)
    overridden = records.ParallelLimits.from_settings(
        profile, {'max_units': 8, 'max_concurrency': 12}, prompt_budget=20000)
    assert (overridden.max_units, overridden.max_concurrency) == (8, 12)
    assert overridden.unit_request_tokens == 16000
    flag_overridden = records.ParallelLimits.from_settings(
        profile, {'--parallel-max-subframes': 7}, prompt_budget=20000)
    assert flag_overridden.max_units == 7
    raised_concurrency = records.ParallelLimits.from_settings(
        {}, {'max_concurrency': 100000}, prompt_budget=20000)
    assert raised_concurrency.max_concurrency == 100000


def test_default_subframe_token_cap_scales_from_prompt_budget_and_cli_still_overrides():
    small = records.ParallelLimits.from_settings({}, {}, prompt_budget=20000)
    explicit = records.ParallelLimits.from_settings(
        {}, {'unit_request_tokens': 15000}, prompt_budget=20000)

    assert small.unit_request_tokens == 16000
    assert explicit.unit_request_tokens == 15000


@pytest.mark.parametrize(('setting', 'value'), [
    ('max_units', 8),
    ('min_units', 3),
    ('max_paths_per_unit', 5),
    ('max_paths_total', 30),
    ('unit_request_tokens', 1000),
    ('max_owned_source_bytes', 30000),
    ('unit_max_requests', 20),
    ('parent_reserve_requests', 6),
    ('check_permits', 3),
    ('max_concurrency', 9),
])
def test_parallel_limits_accept_each_profile_setting(setting, value):
    switch = records.PARALLEL_LIMIT_OPTIONS[setting]['switch']
    limits = records.ParallelLimits.from_settings({switch: str(value)}, {}, prompt_budget=12000)
    assert getattr(limits, setting) == value


@pytest.mark.parametrize('source_kind', ['profile', 'cli'])
@pytest.mark.parametrize('setting', tuple(records.PARALLEL_LIMIT_OPTIONS))
def test_parallel_limits_require_positive_integer_values(setting, source_kind):
    option = records.PARALLEL_LIMIT_OPTIONS[setting]
    profile = {option['switch']: '0'} if source_kind == 'profile' else {}
    cli = {setting: 0} if source_kind == 'cli' else {}
    expected_source = option['switch'] if source_kind == 'profile' else option['flag']
    with pytest.raises(ParallelConfigError) as raised:
        records.ParallelLimits.from_settings(profile, cli, prompt_budget=20000)
    assert (raised.value.setting, raised.value.source) == (setting, expected_source)
    assert records.display_limit_name(setting) in str(raised.value)
    assert expected_source in str(raised.value)
    assert 'unit' not in str(raised.value).casefold()


@pytest.mark.parametrize(('profile', 'cli', 'prompt_budget', 'setting', 'source'), [
    ({}, {'max_units': 1025}, 20000, 'max_units', '--parallel-max-subframes'),
    ({}, {'min_units': 7}, 20000, 'min_units', '--parallel-min-subframes'),
    ({}, {'unit_request_tokens': 12001}, 12000, 'unit_request_tokens', '--parallel-subframe-tokens'),
    ({}, {'max_paths_per_unit': 25}, 20000, 'max_paths_per_unit',
     '--parallel-max-paths-per-subframe'),
    ({'STATEFRAME_PARALLEL_SUBFRAME_REQUEST_TOKENS': '13000'}, {}, 12000,
     'unit_request_tokens', 'STATEFRAME_PARALLEL_SUBFRAME_REQUEST_TOKENS'),
])
def test_parallel_limit_cross_field_validation_names_setting_and_source(
        profile, cli, prompt_budget, setting, source):
    with pytest.raises(ParallelConfigError) as raised:
        records.ParallelLimits.from_settings(profile, cli, prompt_budget=prompt_budget)
    assert (raised.value.setting, raised.value.source) == (setting, source)
    assert records.display_limit_name(setting) in str(raised.value) and source in str(raised.value)
    assert 'unit' not in str(raised.value).casefold()


def test_cli_override_does_not_parse_shadowed_invalid_profile_value():
    limits = records.ParallelLimits.from_settings(
        {'STATEFRAME_PARALLEL_MAX_SUBFRAMES': 'not-an-integer'}, {'max_units': 7}, prompt_budget=20000)
    assert limits.max_units == 7


@pytest.mark.parametrize(('profile', 'cli', 'setting', 'source'), [
    ({'STATEFRAME_PARALLEL_MAX_SUBFRAMES': '6.5'}, {}, 'max_units',
     'STATEFRAME_PARALLEL_MAX_SUBFRAMES'),
    ({'STATEFRAME_PARALLEL_MAX_UNIT': '8'}, {}, 'max_unit', 'STATEFRAME_PARALLEL_MAX_UNIT'),
    ({}, {'max_unit': 8}, 'max_unit', 'cli'),
])
def test_invalid_parallel_setting_names_and_values_fail_with_source(profile, cli, setting, source):
    with pytest.raises(ParallelConfigError) as raised:
        records.ParallelLimits.from_settings(profile, cli, prompt_budget=20000)
    assert (raised.value.setting, raised.value.source) == (setting, source)
    assert records.display_limit_name(setting) in str(raised.value) and source in str(raised.value)


@pytest.mark.parametrize(('profile', 'cli'), [
    ({'STATEFRAME_PARALLEL_MAX_UNITS': '7'}, {}),
    ({'STATEFRAME_PARALLEL_MIN_UNITS': '3'}, {}),
    ({'STATEFRAME_PARALLEL_MAX_PATHS_PER_UNIT': '8'}, {}),
    ({'STATEFRAME_PARALLEL_UNIT_REQUEST_TOKENS': '9000'}, {}),
    ({'STATEFRAME_PARALLEL_UNIT_MAX_REQUESTS': '9'}, {}),
    ({}, {'--parallel-max-units': 7}),
    ({}, {'--parallel-min-units': 3}),
    ({}, {'--parallel-max-paths-per-unit': 8}),
    ({}, {'--parallel-unit-tokens': 9000}),
    ({}, {'--parallel-unit-max-requests': 9}),
])
def test_previous_parallel_limit_spellings_are_not_aliases(profile, cli):
    with pytest.raises(ParallelConfigError):
        records.ParallelLimits.from_settings(profile, cli, prompt_budget=20000)


def test_prompt_budget_must_be_a_positive_integer():
    with pytest.raises(ParallelConfigError) as raised:
        records.ParallelLimits.from_settings({}, {}, prompt_budget=0)
    assert (raised.value.setting, raised.value.source) == ('prompt_budget', 'prompt_budget')


def test_duplicate_cli_aliases_for_one_setting_are_rejected():
    with pytest.raises(ParallelConfigError) as raised:
        records.ParallelLimits.from_settings(
            {}, {'max_units': 7, '--parallel-max-subframes': 8}, prompt_budget=20000)
    assert (raised.value.setting, raised.value.source) == ('max_units', 'cli')
