import json
import hashlib
import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import kernel_unit_policy
from kernel_stateframe_action_ledger import build_action_context
from kernel_stateframe_framefit5 import (
    _serialize_state_message, assemble_framefit5_request, build_typed_state,
    validate_final_product_boundary,
)
from kernel_nav_basics import render_file_tree
from kernel_parallel_records import CheckRecipe, ParallelLimits, UnitContract, UnitSpec
from kernel_read_exec import execute_read
from kernel_frame_source_snapshot import frame_source_snapshot
from kernel_request_source_window import bound_runtime_sources, verify_current_source_message
from kernel_unit_policy import UnitPolicy


def policy(owns=('pkg/a.py',), hidden=('pkg/hidden.py',)):
    contract = UnitContract('a', 'fix a', owns, (), (), (), (), (),
                            CheckRecipe((), (), 'none'))
    spec = UnitSpec('call', 'a', 'execution', 'seed', contract, (), (),
                    owns, tuple(hidden), '', 0, 0, 'manifest', 'policy',
                    ParallelLimits())
    return UnitPolicy(spec)


def test_unstructured_diagnostic_projection_is_bounded_and_labels_its_limits():
    raw = "\n".join(["old line"] * 5 + [f"visible line {index}" for index in range(39)]
                    + ["pkg/hidden.py: private diagnostic", "visible final line"])
    # With withheld files present the command output is withheld entirely (fail closed).
    closed = policy().project_check_output(raw, "other")
    assert "visible line" not in closed and "private diagnostic" not in closed
    assert closed == "check exit unknown\n[command output withheld]"
    assert "pkg/hidden.py: private diagnostic" in raw
    # Without withheld files the bounded best-effort projection applies.
    projected = policy(hidden=()).project_check_output(
        raw.replace("pkg/hidden.py: private diagnostic\n", ""), "other")
    assert projected.startswith("best-effort projection\n")
    assert "old line" not in projected
    assert "visible line 38" in projected
    assert "visible final line" in projected


def test_hidden_read_and_tree_listing_exclude_sibling_source(tmp_path, monkeypatch):
    (tmp_path / 'pkg').mkdir()
    (tmp_path / 'pkg/a.py').write_text('visible = 1\n')
    (tmp_path / 'pkg/hidden.py').write_text('HIDDEN_SENTINEL = 42\n')
    monkeypatch.setattr(kernel_unit_policy, 'active', policy)
    events = []
    runtime = SimpleNamespace(journal=SimpleNamespace(emit=lambda *args, **kwargs:
                                                      events.append((args, kwargs))))
    result, observation, source = execute_read(
        tmp_path, tmp_path / 'pkg/hidden.py', 'pkg/hidden.py', 'pkg/hidden.py', 0,
        args={'path': 'pkg/hidden.py'}, measurement=runtime, step=1)
    assert (result, source) == ('HIDDEN_PATH', None)
    assert observation == 'REFUSED HIDDEN_PATH: pkg/hidden.py is withheld'
    assert events[0][0] == ('unit_hidden_path',)
    assert events[0][1] == {'producer': 'read', 'path': 'pkg/hidden.py'}
    tree = json.loads(render_file_tree(tmp_path))
    assert [entry['path'] for entry in tree['entries']] == ['pkg', 'pkg/a.py']
    assert 'HIDDEN_SENTINEL' not in json.dumps(tree)
    result, observation, _ = execute_read(
        tmp_path, tmp_path / 'pkg', 'pkg', 'pkg', 0,
        args={'path': 'pkg', 'page_size': 1}, measurement=runtime, step=2)
    assert result == 'DIRECTORY'
    assert 'a.py' in observation
    assert 'hidden.py' not in observation
    assert 'total=1' in observation


def test_owned_source_delivery_is_complete_before_allocation(tmp_path):
    (tmp_path / 'pkg').mkdir()
    source = 'head = 0\n' + ''.join(f'line_{number} = {number}\n' for number in range(400))
    (tmp_path / 'pkg/a.py').write_text(source)
    second = 'start = 1\n' + ''.join(f'other_{number} = {number}\n' for number in range(350))
    (tmp_path / 'pkg/b.py').write_text(second)
    runtime = SimpleNamespace(_unit_policy=policy(('pkg/a.py', 'pkg/b.py')),
                              owner=SimpleNamespace(root=tmp_path),
                              target=None)
    messages = []
    result = bound_runtime_sources(messages, runtime, max_request_bytes=100000)
    assert result['paths'] == ('pkg/a.py', 'pkg/b.py')
    assert len(messages) == 2
    assert messages[0]['content'] == '[Current source: pkg/a.py]\n' + source
    assert 'line_399 = 399' in messages[0]['content']
    assert messages[1]['content'] == '[Current source: pkg/b.py]\n' + second
    assert 'other_349 = 349' in messages[1]['content']
    assert result['unit_full_source_revisions'] == {
        'pkg/a.py': 'sha256:' + hashlib.sha256(source.encode()).hexdigest(),
        'pkg/b.py': 'sha256:' + hashlib.sha256(second.encode()).hexdigest(),
    }
    assert [unit['path'] for unit in result['closure']['units']] == ['pkg/a.py', 'pkg/b.py']
    assert [unit['end_line'] for unit in result['closure']['units']] == [401, 351]
    assert [unit['role'] for unit in result['closure']['units']] == [
        'subframe_owned_source', 'subframe_owned_source',
    ]
    assert [unit['reason'] for unit in result['closure']['units']] == [
        'subframe_contract', 'subframe_contract',
    ]
    assert result['closure']['bounded_delivery']['retained_handles'] == [
        'subframe:owned:pkg/a.py', 'subframe:owned:pkg/b.py',
    ]
    assert result['closure']['bounded_delivery']['omitted_handles'] == []
    assert frame_source_snapshot(runtime, runtime._request_source_closure) == {
        'pkg/a.py': source, 'pkg/b.py': second,
    }


def test_shipped_framefit_allocation_keeps_complete_discount_source(
        tmp_path, monkeypatch, shipped_runtime_profile):
    fixture = Path(__file__).parents[1] / 'examples/discount_bug'
    (tmp_path / 'shop').mkdir()
    source = (fixture / 'shop/pricing.py').read_text()
    (tmp_path / 'shop/pricing.py').write_text(source)
    runtime = SimpleNamespace(
        owner=SimpleNamespace(root=tmp_path, task_row={}),
        target=tmp_path / 'shop/pricing.py',
        _unit_policy=policy(('shop/pricing.py',)),
        edit_contract={'edit_authority': {'editable_paths': ['shop/pricing.py']}},
        protected_tests={}, frame_ceiling=33_344,
        _empirical_frame_sources={},
    )
    task = {'role': 'user', 'content': 'Task: Fix the subframe discount.'}
    context = build_action_context([task], workspace_root=tmp_path)
    context['journal_receipt_records'] = []
    runtime._framefit_action_context = context
    state = build_typed_state([task], context, runtime)
    prepared = [{'role': 'system', 'content': 'rules'}, task,
                {'role': 'user', 'content': _serialize_state_message(
                    {'state': {'framefit5': state}})}]
    bound_runtime_sources(prepared, runtime, max_request_bytes=100_000)
    plan = runtime._request_source_closure
    assembled = assemble_framefit5_request(
        prepared, prepared[:2], source_plan=plan,
        sources=frame_source_snapshot(runtime, plan), runtime=runtime,
    )
    assert any(verify_current_source_message(
        message.get('content'), path='shop/pricing.py', source=source,
    ) for message in assembled)
    without_required_delivery = copy.deepcopy(plan)
    without_required_delivery.pop('bounded_delivery')
    omitted = assemble_framefit5_request(
        prepared, prepared[:2], source_plan=without_required_delivery,
        sources=frame_source_snapshot(runtime, without_required_delivery), runtime=runtime,
    )
    assert not any(verify_current_source_message(
        message.get('content'), path='shop/pricing.py', source=source,
    ) for message in omitted)


def test_shipped_runtime_prepares_first_discount_request_with_owned_source(
        tmp_path, monkeypatch, shipped_runtime_profile):
    from kernel_bolt import KernelBolt
    from kernel_pilot import ACTIVE_PARENT, ACTIVE_RUNTIME
    from kernel_stateframe_runtime import StateFrameRuntime
    from kernel_unit_task import build_public_row
    from detreason.world_knowledge import WorldKnowledgeStore
    from probes.react_repair import TOOLS

    fixture = Path(__file__).parents[1] / 'examples/discount_bug'
    source = (fixture / 'shop/pricing.py').read_bytes()
    check = (fixture / 'tests/test_pricing.py').read_bytes()
    (tmp_path / 'shop').mkdir()
    (tmp_path / 'tests').mkdir()
    (tmp_path / 'shop/pricing.py').write_bytes(source)
    (tmp_path / 'tests/test_pricing.py').write_bytes(check)
    recipe = CheckRecipe(('tests/test_pricing.py',),
                         ('tests/test_pricing.py',), 'pytest')
    contract = UnitContract('discount', 'fix discount', ('shop/pricing.py',),
                            (), (), (), (), (), recipe)
    spec = UnitSpec('call', 'discount', 'execution', 'seed', contract, (), (),
                    ('shop/pricing.py', 'tests/test_pricing.py'), (), '', 0, 0,
                    'manifest', 'policy', ParallelLimits())
    unit = UnitPolicy(spec)
    monkeypatch.setattr(kernel_unit_policy, 'active', lambda: unit)
    row = build_public_row(spec, {'shop/pricing.py': source,
                                  'tests/test_pricing.py': check})
    owner = KernelBolt(tmp_path, '', 'execution')
    owner.world = WorldKnowledgeStore(sandbox_id='execution')
    owner.task_row = row
    owner.f2p_ids = ('tests/test_pricing.py',)
    owner.f2p_names = ('tests/test_pricing.py',)
    owner.current_step = 1
    tools = [tool for tool in TOOLS if tool['function']['name'] in unit.allowed_tools]
    tools.append(kernel_unit_policy.UNIT_BLOCKED_TOOL)
    parent_token = ACTIVE_PARENT.set(None)
    runtime_token = ACTIVE_RUNTIME.set(None)
    try:
        runtime = StateFrameRuntime(
            owner, output=tmp_path / 'measurement', target=tmp_path / 'shop/pricing.py',
            model='scripted', tools=tools, temperature=0,
            source_resolver=lambda _path, _symbol: None,
            multi_file=True, edit_contract={'edit_authority': {
                'editable_paths': ['shop/pricing.py']}},
        )
        messages = [{'role': 'system', 'content': 'rules'},
                    {'role': 'user', 'content': 'Task: ' + row['msg']}]
        prepared = runtime.prepare(messages, uncapped=messages, tools=tools)
    finally:
        ACTIVE_RUNTIME.reset(runtime_token)
        ACTIVE_PARENT.reset(parent_token)
    assert prepared.overflow is None
    f4_messages = [message['content'] for message in prepared.messages
                   if isinstance(message.get('content'), str)
                   and message['content'].startswith('[StateFrame]\n')]
    assert len(f4_messages) == 1
    f4 = json.loads(f4_messages[0].removeprefix('[StateFrame]\n').split('\n[', 1)[0])
    assert f4['schema'] == 'stateframe-f4/1'
    assert f4['state']['framefit5']['mode'] == 'FRAME_ONLY'
    assert any(verify_current_source_message(
        message.get('content'), path='shop/pricing.py', source=source.decode(),
    ) for message in prepared.messages)
    assert any(verify_current_source_message(
        message.get('content'), path='tests/test_pricing.py', source=check.decode(),
    ) for message in prepared.messages)


def test_shipped_subframe_post_green_request_seals_frozen_check(
        tmp_path, monkeypatch, request, shipped_runtime_profile):
    from kernel_bolt import KernelBolt
    import kernel_check_permits
    from kernel_catalog import normalize_receipt
    from kernel_completion_owner import dispatch_tool_call
    from kernel_pass_gates_contract import build_gate_vector, validate_final_surface
    from kernel_pilot import ACTIVE_PARENT, ACTIVE_RUNTIME
    import kernel_pilot
    from kernel_stateframe_runtime import StateFrameRuntime
    from kernel_unit_task import build_public_row
    from detreason.world_knowledge import WorldKnowledgeStore
    from probes import react_repair

    fixture = Path(__file__).parents[1] / 'examples/discount_bug'
    for relative in ('shop/pricing.py', 'tests/test_pricing.py'):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((fixture / relative).read_bytes())
    recipe = CheckRecipe(('tests/test_pricing.py',), ('tests/test_pricing.py',), 'pytest')
    contract = UnitContract('discount', 'fix discount', ('shop/pricing.py',),
                            (), (), (), (), (), recipe)
    spec = UnitSpec('call', 'discount', 'execution', 'seed', contract, (), (),
                    ('shop/pricing.py', 'tests/test_pricing.py'), (), '', 0, 0,
                    'manifest', 'policy', ParallelLimits())
    unit = UnitPolicy(spec)
    monkeypatch.setattr(kernel_unit_policy, 'active', lambda: unit)
    monkeypatch.setattr(kernel_check_permits, '_installed', (tmp_path / 'permits', 1))
    monkeypatch.setattr(kernel_pilot, 'charge', lambda *_args, **_kwargs: None)
    row = build_public_row(spec, {
        relative: (tmp_path / relative).read_bytes()
        for relative in ('shop/pricing.py', 'tests/test_pricing.py')})
    owner = KernelBolt(tmp_path, '', 'execution')
    owner.world = WorldKnowledgeStore(sandbox_id='execution')
    owner.task_row = row
    owner.f2p_ids = recipe.selectors
    owner.f2p_names = recipe.selectors
    owner.current_step = 2
    tools = [tool for tool in react_repair.TOOLS
             if tool['function']['name'] in unit.allowed_tools]
    tools.append(kernel_unit_policy.UNIT_BLOCKED_TOOL)
    parent_token = ACTIVE_PARENT.set(None)
    request.addfinalizer(lambda: ACTIVE_PARENT.reset(parent_token))
    runtime = StateFrameRuntime(
        owner, output=tmp_path / 'measurement', target=tmp_path / 'shop/pricing.py',
        model='scripted', tools=tools, temperature=0,
        source_resolver=lambda _path, _symbol: None, multi_file=True,
        edit_contract={'edit_authority': {'editable_paths': ['shop/pricing.py']}},
    )

    def check_boundary(_command, *, root, targets, env, runner):
        del env, runner
        assert targets == ['tests/test_pricing.py']
        fixed = 'price * (100 - percent) / 100' in (root / 'shop/pricing.py').read_text()
        return (SimpleNamespace(returncode=0 if fixed else 1,
                                stdout='5 passed in 0.01s\n' if fixed else '5 failed in 0.01s\n',
                                stderr=''), None)

    monkeypatch.setattr(react_repair, '_effect_test_subprocess', check_boundary)
    token = ACTIVE_RUNTIME.set(runtime)
    try:
        task = {'role': 'user', 'content': 'Task: ' + row['msg']}
        messages = [{'role': 'system', 'content': 'rules'}, task]
        first = runtime.prepare(messages, uncapped=messages, tools=tools)
        assert first.overflow is None
        assert 'subframe_blocked' in {tool['function']['name'] for tool in first.tools}
        # The frozen recipe is bound before any check runs, so a refused first check keeps
        # the same binding before and after the step.
        assert runtime.binding()['check_recipe_digest'] == recipe.digest()
        red = react_repair.run_tests(tmp_path, '', 'tests/test_pricing.py', row=row)
        assert 'exit=1' in red
        assert runtime._unit_check_receipt.status == 'FAIL'
        pricing = tmp_path / 'shop/pricing.py'
        pricing.write_text(pricing.read_text().replace(
            'price * percent / 100', 'price * (100 - percent) / 100'))
        runtime.note_mutation(('shop/pricing.py',), {'result_class': 'APPLIED'})
        owner.current_step = 3
        green = react_repair.run_tests(tmp_path, '', 'tests/test_pricing.py', row=row)
        assert 'exit=0 passed=5' in green
        assert runtime._unit_check_receipt.status == 'PASS'
        assert runtime.test_receipts[-1]['binding_stable'] is True
        assert runtime.test_receipts[-1]['binding']['check_recipe_digest'] == recipe.digest()
        assert runtime.test_receipts[-1]['after_binding']['check_recipe_digest'] == recipe.digest()
        assert (runtime.test_receipts[-1]['binding']['product_revision']
                == runtime.test_receipts[-1]['after_binding']['product_revision'])
        exchange = [
            {'role': 'assistant', 'tool_calls': [{
                'id': 'check-1', 'type': 'function', 'function': {
                    'name': 'run_tests',
                    'arguments': json.dumps({'target': 'tests/test_pricing.py'})}}]},
            {'role': 'tool', 'tool_call_id': 'check-1', 'content': green},
        ]
        post_green = [*messages, *exchange]
        prepared = runtime.prepare(post_green, uncapped=post_green, tools=tools)
    finally:
        ACTIVE_RUNTIME.reset(token)
    assert prepared.overflow is None
    assert prepared.capture.request_budget_final_wire['pass_gates']['status'] == 'VALID'
    wire_tools = {tool['function']['name'] for tool in prepared.tools}
    assert wire_tools <= unit.allowed_tools
    assert 'run_tests' in wire_tools
    assert 'subframe_blocked' in wire_tools
    wire = json.dumps({'messages': prepared.messages, 'tools': prepared.tools})
    assert '[StateFrame]\\n' in wire
    assert 'exit=0 passed=5' in wire
    assert 'Subframe' in wire
    f4_messages = [message['content'] for message in prepared.messages
                   if isinstance(message.get('content'), str)
                   and message['content'].startswith('[StateFrame]\n')]
    assert len(f4_messages) == 1
    assert json.loads(f4_messages[0].removeprefix('[StateFrame]\n').split('\n[', 1)[0])[
        'schema'] == 'stateframe-f4/1'

    identity = normalize_receipt(runtime.test_receipts[-1])['check_identity']
    assert identity.startswith('check:')
    good = {'unknown_id': 'unknown:binding', 'check_identity': identity,
            'check_id': identity, 'selector': identity, 'target': identity,
            'revision': runtime.binding()['product_revision'], 'cursor': '0',
            'required_by': 'execution', 'scope': 'task'}

    def surface(arguments):
        vector = build_gate_vector(
            artifact_identity='candidate', evidence_identity='receipt',
            blockers=[{'code': 'CATALOG_UNKNOWN', 'detail': 'check.binding',
                       'resolver': {'tool': 'run_tests', 'arguments': arguments}}],
            available_actions=('run_tests',), available_tools=prepared.tools)

        def dispatchable(name, values):
            if not unit.matches_check_resolver(values, runtime.test_receipts):
                return False
            dispatch_tool_call(prepared.tools, name, values)
            return True

        return validate_final_surface(prepared.tools, vector,
                                      dispatchable=dispatchable)

    assert surface(good)['status'] == 'VALID'
    wrong = {**good, 'check_identity': 'check:other', 'check_id': 'check:other',
             'selector': 'check:other', 'target': 'check:other'}
    refused = surface(wrong)
    assert refused['status'] == 'HOST_UNSUPPORTED'
    assert refused['dispatch_refused'] == ({'tool': 'run_tests', 'arguments': wrong},)
    missing_identity = {key: value for key, value in good.items()
                        if key != 'check_identity'}
    assert surface(missing_identity)['dispatch_refused'] == (
        {'tool': 'run_tests', 'arguments': missing_identity},)


def test_first_request_refusal_records_typed_reason_and_exact_unsent_body(
        tmp_path, monkeypatch):
    import kernel_request_budget
    from kernel_context import token_estimate
    from kernel_terminal_policy import terminal_category
    from probes.react_repair import _UnitContextTooLarge, _enforce_unit_first_request

    (tmp_path / 'pkg').mkdir()
    source = 'value = "' + 'x' * 4000 + '"\n'
    (tmp_path / 'pkg/a.py').write_text(source)
    original = policy(('pkg/a.py',))
    unit = UnitPolicy(replace(original.spec,
                              limits=replace(original.spec.limits,
                                             unit_request_tokens=100)))
    messages = [{'role': 'user', 'content': '[Current source: pkg/a.py]\n' + source}]
    tools = [{'type': 'function', 'function': {'name': 'read'}}]
    measured = token_estimate({'messages': messages, 'tools': tools})
    assert measured > 100
    monkeypatch.setattr(kernel_request_budget, '_request_token_meter',
                        lambda *_args: SimpleNamespace(
                            measure=lambda body: (token_estimate(body), 'fixture_counter'),
                            measure_subframe_cap=lambda body, **_kwargs:
                                (token_estimate(body), 'tokenize')))
    events = []
    runtime = SimpleNamespace(journal=SimpleNamespace(
        emit=lambda kind, **fields: events.append((kind, fields))))
    prepared = SimpleNamespace(messages=messages, capture=SimpleNamespace(request_id='r1'))
    run_dir = tmp_path / 'run'
    with pytest.raises(_UnitContextTooLarge, match='ceiling 100'):
        _enforce_unit_first_request(runtime, prepared, tools, unit, tmp_path, run_dir)
    assert runtime.system_outcome == 'unit_context_too_large'
    assert terminal_category(runtime.system_outcome, runtime) == 'fatal'
    assert events[0][0] == 'unit_frame_tokens'
    assert {key: events[0][1][key] for key in
            ('request', 'tokens', 'counter', 'meter_kind', 'confidence', 'calibration_id', 'cap')} == {
        'request': 'r1', 'tokens': measured, 'counter': 'tokenize', 'meter_kind': 'tokenize',
        'confidence': 'exact', 'calibration_id': 'tokenize-exact', 'cap': 100,
    }
    assert events[1][0] == 'provider_request_not_dispatched'
    assert events[1][1]['reason_code'] == 'unit_context_too_large'
    assert json.loads((run_dir / 'unit-refused-request.json').read_text()) == {
        'messages': messages, 'tools': tools,
    }
    assert not (run_dir / 'wire').exists()

    # A cheap meter cannot certify a request that lost its mandatory source.
    within_ceiling = UnitPolicy(replace(original.spec,
                                        limits=replace(original.spec.limits,
                                                       unit_request_tokens=measured)))
    missing = SimpleNamespace(messages=[{'role': 'user', 'content': 'Task: fix a'}],
                              capture=SimpleNamespace(request_id='r2'))
    with pytest.raises(_UnitContextTooLarge, match='missing full source: pkg/a.py'):
        _enforce_unit_first_request(runtime, missing, tools, within_ceiling,
                                    tmp_path, tmp_path / 'missing-run')
    assert json.loads((tmp_path / 'missing-run/unit-refused-request.json').read_text()) == {
        'messages': missing.messages, 'tools': tools,
    }
    (tmp_path / 'tests').mkdir()
    (tmp_path / 'tests/test_a.py').write_text('def test_a():\n    assert False\n')
    checked_spec = replace(original.spec, contract=replace(
        original.spec.contract,
        check=CheckRecipe(('tests/test_a.py',), ('tests/test_a.py',), 'pytest')))
    checked = UnitPolicy(replace(checked_spec, limits=replace(
        checked_spec.limits, unit_request_tokens=100_000)))
    owned_only = SimpleNamespace(messages=messages,
                                 capture=SimpleNamespace(request_id='r3'))
    with pytest.raises(_UnitContextTooLarge, match='missing full source: tests/test_a.py'):
        _enforce_unit_first_request(runtime, owned_only, tools, checked,
                                    tmp_path, tmp_path / 'missing-check-run')
    assert not (tmp_path / 'missing-check-run/wire').exists()


@pytest.mark.host_sandbox
def test_real_oversized_subframe_has_typed_terminal_and_zero_sends(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    import os
    import select
    import shutil
    import subprocess
    import sys
    import time

    from stateframe.endpoint import Endpoint
    from stateframe.parallel_host import ParallelSupervisor
    from stateframe.runner import RunOptions, prepare
    from test_parallel_host import _request, _snapshot, _worker_diagnostics

    root = Path(__file__).parents[1]
    repo = tmp_path / 'repo'
    shutil.copytree(root / 'examples/discount_bug', repo)
    pricing = repo / 'shop/pricing.py'
    pricing.write_text(pricing.read_text() + '\n# ' + 'x' * 8000 + '\n')
    environment = dict(os.environ)
    environment.pop('PYTHONPATH', None)
    endpoint_process = subprocess.Popen([
        sys.executable, '-I', '-m', 'stateframe.scripted_endpoint', '--script',
        str(root / 'examples/discount_bug.script.json'), '--port', '0',
    ], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=environment)
    supervisor = None
    worker_dir = tmp_path / 'run/parallel/call/units/a'
    try:
        assert endpoint_process.stdout is not None
        readable, _, _ = select.select([endpoint_process.stdout], [], [], 10)
        assert readable
        endpoint = Endpoint.from_args(endpoint_process.stdout.readline().strip(),
                                      'scripted', None)
        options = RunOptions(repo, 'Fix the pricing code so the tests pass.', endpoint,
                             run_dir=tmp_path / 'run', max_turns=16, profile='r2hc3',
                             parallel_mode='off', python=sys.executable,
                             test_python=sys.executable)
        options._capabilities['python'] = options.python
        parent_run, _, _ = prepare(options)
        call = parent_run / 'parallel/call'
        frozen = _snapshot(parent_run / 'parent', call, 0)
        contract = UnitContract(
            'a', 'fix the pricing calculation', ('shop/pricing.py',),
            (), (), (), (), (), CheckRecipe(('tests/test_pricing.py',),
                                            ('tests/test_pricing.py',), 'pytest'))
        spec = UnitSpec('call', 'a', 'execution-a', '0123456789abcdef0123456789abcdef',
                        contract, (), (), ('shop/pricing.py', 'tests/test_pricing.py'),
                        (), '', 0, 0, frozen.manifest_digest, 'policy-a',
                        ParallelLimits(unit_request_tokens=100))
        worker_dir.mkdir(parents=True)
        spec_path = worker_dir / 'unit-spec.json'
        spec_path.write_text(json.dumps(spec.to_json(), sort_keys=True))
        supervisor = ParallelSupervisor(parent_run)
        payload = {'unit_id': 'a', 'execution_id': 'execution-a',
                   'spec_path': str(spec_path), 'spec_digest': spec.digest(),
                   'run_dir': str(worker_dir), 'incarnation': 1,
                   'start_snapshot_path': str(frozen.root),
                   'start_manifest_digest': frozen.manifest_digest,
                   'snapshot_manifest_digest': frozen.snapshot_manifest_digest}
        supervisor._launch(_request(1, 'call', 1, 'launch', payload), call)
        deadline = time.monotonic() + 180
        state = None
        while time.monotonic() < deadline:
            supervisor.tick()
            state = supervisor._latest_worker('call', 'a', call)
            if state is not None and state.state == 'EXITED':
                break
            time.sleep(0.1)
        assert state is not None and state.terminal == 'SUBFRAME_CONTEXT_TOO_LARGE'
        result = json.loads((worker_dir / 'engine-result.json').read_text())['result']
        assert result['result'] == 'SUBFRAME_CONTEXT_TOO_LARGE'
        assert result['reason_code'] == 'SUBFRAME_CONTEXT_TOO_LARGE'
        assert not list((worker_dir / 'wire').glob('*.request.json'))
        ledger = worker_dir / 'sends-ledger.jsonl'
        assert not ledger.exists() or not ledger.read_text().strip()
        refused = json.loads((worker_dir / 'unit-refused-request.json').read_text())
        assert refused['tools']
        assert any(verify_current_source_message(
            message.get('content'), path='shop/pricing.py', source=pricing.read_text(),
        ) for message in refused['messages'])
    except Exception as error:
        pytest.fail(f'{error}\n{_worker_diagnostics(worker_dir)}', pytrace=False)
    finally:
        if supervisor is not None:
            supervisor.cancel_all('test cleanup')
            supervisor.reap()
        endpoint_process.terminate()
        endpoint_process.wait(timeout=5)


def test_visible_grep_counts_and_pages_exclude_hidden_files(tmp_path):
    (tmp_path / 'pkg').mkdir()
    (tmp_path / 'pkg/a.py').write_text('needle one\nneedle two\n')
    (tmp_path / 'pkg/hidden.py').write_text('needle HIDDEN_SENTINEL\n')
    first = policy().search_visible(tmp_path, 'needle', page_size=1)
    assert first.metadata['total'] == 2
    assert first.metadata['shown'] == 1
    assert first.text.startswith('pkg/a.py:1:needle one')
    assert 'HIDDEN_SENTINEL' not in first.text
    second = policy().search_visible(tmp_path, 'needle', cursor=first.metadata['next_cursor'],
                                     page_size=1)
    assert second.metadata['total'] == 2
    assert second.text == 'pkg/a.py:2:needle two'
    assert second.metadata['next_cursor'] is None


def test_framefit_composition_headcarry_withholds_imported_sibling(
        tmp_path, monkeypatch):
    profile = json.loads((Path(__file__).parents[1]
                          / 'src/stateframe/profiles/r2hc3.json').read_text())
    switches = profile['switches']
    monkeypatch.setenv('STATEFRAME_HEADCARRY', switches['STATEFRAME_HEADCARRY'])
    monkeypatch.setenv('STATEFRAME_FRAMEFIT', switches['STATEFRAME_FRAMEFIT'])
    assert switches['STATEFRAME_HEADCARRY'] == 'on'
    assert switches['STATEFRAME_FRAMEFIT'] == 'on'
    (tmp_path / 'pkg').mkdir()
    source = 'from pkg import hidden\nvisible = 1\n'
    (tmp_path / 'pkg/a.py').write_text(source)
    (tmp_path / 'pkg/hidden.py').write_text('HIDDEN_HEAD_SENTINEL = 42\n')
    monkeypatch.setattr(kernel_unit_policy, 'active', policy)
    task = {'role': 'user', 'content': 'Task: Fix the subframe source.'}
    messages = [task, {'role': 'assistant', 'tool_calls': [{
        'id': 'read-1', 'type': 'function', 'function': {
            'name': 'read', 'arguments': json.dumps({'path': 'pkg/a.py'})}}]},
        {'role': 'tool', 'tool_call_id': 'read-1',
         'content': '[read page] path=pkg/a.py revision=sha256:'
                    + hashlib.sha256(source.encode()).hexdigest()
                    + f' lines=1-2/2 chars={len(source)} clip=none\ncurrent source\n{source}'}]
    context = build_action_context(messages, workspace_root=tmp_path)
    context['journal_receipt_records'] = []
    runtime = SimpleNamespace(
        owner=SimpleNamespace(root=tmp_path, task_row={}),
        edit_contract={'edit_authority': {'editable_paths': ['pkg/a.py']}},
        protected_tests={}, frame_ceiling=10_000,
        _framefit_action_context=context, _unit_policy=policy())
    state = build_typed_state(messages, context, runtime)
    prepared = [{'role': 'system', 'content': 'rules'}, task,
                {'role': 'user', 'content': _serialize_state_message(
                    {'state': {'framefit5': state}})}]
    assembled = assemble_framefit5_request(prepared, prepared[:2],
                                           live_frame=state, runtime=runtime)
    validate_final_product_boundary(runtime, assembled, [],
                                    refresh_sources=True, refresh_headcarry=False)
    wire = json.dumps(assembled, sort_keys=True)
    assert 'HIDDEN_HEAD_SENTINEL' not in wire
    assert 'pkg/hidden.py' not in wire
    monkeypatch.setattr(kernel_unit_policy, 'active', lambda: None)
    unfiltered_state = build_typed_state(messages, context, runtime)
    unfiltered = [{'role': 'system', 'content': 'rules'}, task,
                  {'role': 'user', 'content': _serialize_state_message(
                      {'state': {'framefit5': unfiltered_state}})}]
    unfiltered_wire = assemble_framefit5_request(
        unfiltered, unfiltered[:2], live_frame=unfiltered_state, runtime=runtime)
    assert 'HIDDEN_HEAD_SENTINEL' in json.dumps(unfiltered_wire, sort_keys=True)
