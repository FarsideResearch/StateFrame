from types import SimpleNamespace
from dataclasses import replace

import pytest

import kernel_unit_policy
from kernel_capability_floor import admissible_floor_names
from kernel_edit_tool_names import EDIT_TOOL_NAMES
from kernel_parallel_records import CheckRecipe, ParallelLimits, UnitContract, UnitSpec
from kernel_stateframe_runtime import StateFrameRuntime
from kernel_unit_policy import UNIT_BLOCKED_TOOL, UnitPolicy


def make_spec(*, check=True, hidden=('pkg/hidden.py',)):
    recipe = (CheckRecipe(('tests/test_a.py',), ('tests/test_a.py::test_value',), 'pytest')
              if check else CheckRecipe((), (), 'none'))
    contract = UnitContract('a', 'fix a', ('pkg/a.py',), ('pkg/new.py',), (), (),
                            ('pkg/support.py',), (), recipe)
    return UnitSpec('call', 'a', 'execution', 'seed', contract, (), (),
                    ('pkg/a.py', 'pkg/new.py', 'pkg/support.py', 'tests/test_a.py'),
                    tuple(hidden), '', 0, 0, 'manifest', 'policy',
                    ParallelLimits())


def test_tool_and_mutation_capabilities_are_exact():
    policy = UnitPolicy(make_spec())
    assert policy.allowed_tools == frozenset({
        'read', 'grep', 'str_replace', 'insert_before', 'file_create',
    'run_tests', 'done', 'subframe_blocked',
    })
    assert policy.mutating_tools & policy.allowed_tools == frozenset({
        'str_replace', 'insert_before', 'file_create',
    })
    assert policy.mutating_tools == EDIT_TOOL_NAMES
    assert policy.authorize_tool('shell').reason == 'SCOPE_VIOLATION'
    assert policy.authorize_tool('run_tests') is None
    assert policy.authorize_mutation('edit', 'pkg/a.py', created_by_unit=False) is None
    assert policy.authorize_mutation('edit', 'pkg/new.py', created_by_unit=False).reason == 'SCOPE_VIOLATION'
    assert policy.authorize_mutation('create', 'pkg/new.py', created_by_unit=False) is None
    assert policy.authorize_mutation('edit', 'pkg/new.py', created_by_unit=True) is None
    assert policy.authorize_mutation('create', 'pkg/new.py', created_by_unit=True).reason == 'SCOPE_VIOLATION'
    assert policy.authorize_mutation('create', 'pkg/other.py', created_by_unit=False).paths == ('pkg/other.py',)
    assert policy.authorize_mutation('edit', '../pkg/a.py', created_by_unit=False).reason == 'SCOPE_VIOLATION'
    assert policy.visible('pkg/support.py')
    assert not policy.visible('pkg/hidden.py')
    assert UNIT_BLOCKED_TOOL['function']['name'] == 'subframe_blocked'
    assert 'unit' not in str(UNIT_BLOCKED_TOOL).casefold()
    assert UNIT_BLOCKED_TOOL['function']['parameters']['properties']['detail']['maxLength'] == 300


@pytest.mark.parametrize("editor", ["edit_source", "rewrite_source"])
def test_unit_classifies_protocol_mutations_without_offering_them(editor):
    policy = UnitPolicy(make_spec())

    assert "str_replace" in policy.mutating_tools
    assert editor in policy.mutating_tools
    assert policy.authorize_tool(editor).reason == "SCOPE_VIOLATION"


def test_first_request_uses_admitted_token_limit():
    original = make_spec()
    raised = UnitPolicy(replace(original, limits=replace(original.limits,
                                                        unit_request_tokens=20_000)))
    lowered = UnitPolicy(replace(original, limits=replace(original.limits,
                                                         unit_request_tokens=8_000)))
    assert raised.first_request_fits(15_000, owned_source_complete=True)
    assert not lowered.first_request_fits(9_000, owned_source_complete=True)
    assert not raised.first_request_fits(15_000, owned_source_complete=False)
    assert not lowered.first_request_fits(8_001, owned_source_complete=True)


def test_no_tests_unit_has_no_check_tool():
    assert 'run_tests' not in UnitPolicy(make_spec(check=False)).allowed_tools


def test_runtime_menu_and_capability_floor_stay_within_policy(monkeypatch):
    unit = UnitPolicy(make_spec(check=False))
    monkeypatch.setattr(kernel_unit_policy, 'active', lambda: unit)
    tools = [{'type': 'function', 'function': {'name': name}}
             for name in ('read', 'grep', 'str_replace', 'file_create', 'shell', 'work', 'done')]
    tools.append(UNIT_BLOCKED_TOOL)
    runtime = SimpleNamespace(
        _unit_policy=unit, stateframe_completion_owner_mode='off',
        _pass_gates_menu=lambda selected: selected,
        _idealpath_project_menu=lambda selected: selected,
    )
    menu = StateFrameRuntime._base_tool_menu(runtime, tools, [])
    names = [tool['function']['name'] for tool in menu]
    assert names == ['read', 'grep', 'str_replace', 'file_create', 'done', 'subframe_blocked']
    assert set(names) <= unit.allowed_tools
    assert set(admissible_floor_names(tools)) <= unit.allowed_tools


def test_diagnostic_projection_replaces_hidden_frames_and_drops_captured_section():
    policy = UnitPolicy(make_spec())
    raw = ('pytest exit 1\n'
           'pkg/a.py:12: in test_value\n'
           'E   expected 4\n'
           'pkg/hidden.py:39: in helper\n'
           'E   hidden body\n'
           '----------------------------- Captured stdout call -----------------------------\n'
           'pkg/hidden.py emitted secret\n'
           '=========================== 1 failed in 0.02s ===========================\n')
    shown = policy.project_check_output(raw, 'pytest')
    assert 'pkg/a.py:12' in shown
    assert '[frame in withheld file pkg/hidden.py:39]' in shown
    assert 'hidden body' not in shown
    assert 'emitted secret' not in shown
    assert '1 failed' in shown
    classic = policy.project_check_output(
        'pytest exit 1\n  File "pkg/hidden.py", line 41, in helper\n'
        'E   pkg/hidden.py secret\n  File "pkg/a.py", line 12, in main\n'
        'E   expected 4\n', 'pytest')
    assert '[frame in withheld file pkg/hidden.py:41]' in classic
    assert 'pkg/hidden.py secret' not in classic
    # Message bodies are free text and never survive projection, even from visible frames.
    assert 'expected 4' not in classic
    assert 'File "pkg/a.py", line 12, in main' in classic
    assert 'E   [exception detail withheld]' in classic
    other = UnitPolicy(make_spec(hidden=())).project_check_output('ok\nlast', 'other')
    assert other == 'best-effort projection\nok\nlast'
    assert policy.project_check_output('ok\npkg/hidden.py secret\nlast', 'other') == (
        'check exit unknown\n[command output withheld]')


def test_pytest_short_summary_message_from_withheld_source_is_scrubbed():
    policy = UnitPolicy(make_spec())
    raw = ('pytest exit 1\n'
           'tests/test_a.py:5: in test_value\n'
           'pkg/hidden.py:2: in explode\n'
           'E   RuntimeError: HIDDEN_EXCEPTION_SENTINEL\n'
           '=========================== short test summary info ===========================\n'
           'FAILED tests/test_a.py::test_value - RuntimeError: HIDDEN_EXCEPTION_SENTINEL\n'
           'ERROR tests/test_a.py::test_other - ValueError: HIDDEN_EXCEPTION_SENTINEL\n'
           '=========================== 1 failed, 1 error in 0.02s ===========================\n')
    shown = policy.project_check_output(raw, 'pytest')
    assert 'HIDDEN_EXCEPTION_SENTINEL' not in shown
    assert 'FAILED tests/test_a.py::test_value' in shown
    assert 'ERROR tests/test_a.py::test_other' in shown
    # Without a withheld frame the message is still free text the test itself controls.
    plain = policy.project_check_output(
        'pytest exit 1\nFAILED tests/test_a.py::test_value - AssertionError: expected 4\n', 'pytest')
    assert 'expected 4' not in plain
    assert 'FAILED tests/test_a.py::test_value - [message withheld]' in plain


def test_subframe_can_never_delete_or_move_any_file():
    policy = UnitPolicy(make_spec())
    for name in ('file_delete', 'file_move', 'shell', 'git'):
        assert policy.authorize_tool(name).reason == 'SCOPE_VIOLATION'
    for path in ('pkg/a.py', 'pkg/new.py', 'pkg/other.py', 'pkg/protected.py'):
        for created in (False, True):
            assert policy.authorize_mutation('delete', path, created_by_unit=created).reason == 'SCOPE_VIOLATION'


def test_run_tests_target_must_name_the_frozen_check():
    policy = UnitPolicy(make_spec())
    # The frozen selector, its file, its bare node name, and no target are the frozen check.
    for arguments in ({}, {'target': ''}, {'target': 'tests/test_a.py::test_value'},
                      {'target': 'tests/test_a.py'}, {'target': 'test_value'}):
        assert policy.authorize_check_target(arguments) is None
    refusal = policy.authorize_check_target({'target': 'tests/other.py'})
    assert refusal.reason == 'SCOPE_VIOLATION'
    assert 'tests/test_a.py::test_value' in refusal.detail
    assert policy.authorize_check_target({'target': 'tests/test_a.py::test_other'}).reason == 'SCOPE_VIOLATION'
    # Catalog resolver calls carry an identity and are validated by matches_check_resolver instead.
    assert policy.authorize_check_target(
        {'target': 'check:1', 'check_identity': 'check:1'}) is None
    # No frozen pytest check means nothing to compare a target against.
    assert UnitPolicy(make_spec(check=False)).authorize_check_target({'target': 'tests/other.py'}) is None


def test_pytest_error_lines_precede_their_frame_and_follow_its_visibility():
    """pytest prints `E` lines before the `path:line: ExcType` frame that raised them."""
    policy = UnitPolicy(make_spec())
    raw = ('pytest exit 1\n'
           '    def test_value():\n'
           '>       explode()\n'
           '\n'
           'tests/test_a.py:4: \n'
           '_ _ _ _ _ _ _ _ _ _ _ _ _ _\n'
           '\n'
           '    def explode():\n'
           ">       raise RuntimeError('HIDDEN_EXCEPTION_SENTINEL')\n"
           'E       RuntimeError: HIDDEN_EXCEPTION_SENTINEL\n'
           '\n'
           'pkg/hidden.py:2: RuntimeError\n'
           '=========================== 1 failed in 0.02s ===========================\n')
    shown = policy.project_check_output(raw, 'pytest')
    assert 'HIDDEN_EXCEPTION_SENTINEL' not in shown
    assert '[frame in withheld file pkg/hidden.py:2]' in shown
    # The same shape raised from visible source keeps the exception type, not the message.
    visible = policy.project_check_output(
        raw.replace('pkg/hidden.py:2', 'pkg/a.py:2'), 'pytest')
    assert 'HIDDEN_EXCEPTION_SENTINEL' not in visible
    assert 'E   RuntimeError: [message withheld]' in visible
    assert 'pkg/a.py:2: RuntimeError' in visible


# Fail-closed projection: a subframe's own check runs on a full clone, so any `E` message body,
# captured stream, parametrize id or short-summary message can carry hidden file contents.
_SECRET = 'DOTENV_SECRET=abc123'


def test_child_check_output_cannot_carry_hidden_file_contents_through_its_own_test():
    policy = UnitPolicy(make_spec())
    raw = ('pytest exit 1\n'
           '    def test_value():\n'
           f'>       assert False, open(".env").read()\n'
           f'E       AssertionError: {_SECRET}\n'
           f'E       assert False\n'
           '\n'
           'tests/test_a.py:3: AssertionError\n'
           '----------------------------- Captured stdout call -----------------------------\n'
           f'{_SECRET}\n'
           '----------------------------- Captured stderr call -----------------------------\n'
           f'warn {_SECRET}\n'
           '------------------------------ Captured log call -------------------------------\n'
           f'INFO root:test_a.py:2 {_SECRET}\n'
           '=========================== short test summary info ===========================\n'
           f'FAILED tests/test_a.py::test_value - AssertionError: {_SECRET}\n'
           '=========================== 1 failed in 0.02s ===========================\n')
    shown = policy.project_check_output(raw, 'pytest')
    assert _SECRET not in shown
    assert 'abc123' not in shown
    # The owned-file failure stays useful: location, exception type, node id, counts, exit.
    assert 'pytest exit 1' in shown
    assert 'tests/test_a.py:3: AssertionError' in shown
    assert 'E   AssertionError: [message withheld]' in shown
    assert '[captured stdout call withheld]' in shown
    assert 'FAILED tests/test_a.py::test_value - [message withheld]' in shown
    assert '1 failed' in shown


def test_child_check_output_scrubs_parametrize_ids_and_inline_stdout():
    policy = UnitPolicy(make_spec())
    raw = ('pytest exit 1\n'
           f'{_SECRET}\n'
           f'pytest exit 0 {_SECRET}\n'
           f'tests/test_a.py:3: {_SECRET}\n'
           f'FAILED tests/test_a.py::test_value[{_SECRET}] - assert 1 == 2\n'
           f'ERROR tests/test_a.py::test_other[{_SECRET}]\n'
           f'=========================== 1 failed, 1 error, {_SECRET} in 0.02s ===========================\n')
    shown = policy.project_check_output(raw, 'pytest')
    assert _SECRET not in shown
    assert 'FAILED tests/test_a.py::test_value[...] - [message withheld]' in shown
    assert 'ERROR tests/test_a.py::test_other[...]' in shown
    assert 'tests/test_a.py:3:' in shown
    assert '1 failed, 1 error' in shown
    assert shown.count('pytest exit') == 2 and 'pytest exit 0' in shown


def test_collection_error_source_text_from_withheld_module_is_dropped():
    """B6(a): a SyntaxError in a withheld module prints its source text as an `E` line."""
    policy = UnitPolicy(make_spec())
    raw = ('pytest exit 2\n'
           '_____________________ ERROR collecting tests/test_a.py _____________________\n'
           'E     File "/work/pkg/hidden.py", line 3\n'
           f'E       SECRET = "{_SECRET}\n'
           'E                ^\n'
           'E   SyntaxError: unterminated string literal\n'
           '=========================== short test summary info ===========================\n'
           f'ERROR tests/test_a.py - SyntaxError: {_SECRET}\n'
           '=========================== 1 error in 0.02s ===========================\n')
    shown = policy.project_check_output(raw, 'pytest')
    assert _SECRET not in shown
    assert 'ERROR tests/test_a.py - [message withheld]' in shown
    assert '1 error' in shown


def test_unrecognised_withheld_frames_do_not_unlock_summary_messages():
    """B6(b): spaces, .pyx and <string> frames must still fail closed."""
    policy = UnitPolicy(make_spec())
    for frame in ('pkg/hidden dir/lib.py:4: in helper',
                  'pkg/hidden.pyx:4: in helper',
                  '  File "<string>", line 1, in <module>',
                  '  File "/work/pkg/hidden dir/lib.py", line 4, in helper'):
        raw = ('pytest exit 1\n'
               'tests/test_a.py:5: in test_value\n'
               f'{frame}\n'
               f'E   RuntimeError: {_SECRET}\n'
               '=========================== short test summary info ===========================\n'
               f'FAILED tests/test_a.py::test_value - RuntimeError: {_SECRET}\n'
               '=========================== 1 failed in 0.02s ===========================\n')
        shown = policy.project_check_output(raw, 'pytest')
        assert _SECRET not in shown, frame
        assert 'FAILED tests/test_a.py::test_value' in shown, frame


def test_separator_lines_do_not_flush_error_lines_before_their_withheld_frame():
    """B6(c): `_ _ _` and chained-exception lines sit between `E` lines and their frame."""
    policy = UnitPolicy(make_spec())
    raw = ('pytest exit 1\n'
           'tests/test_a.py:5: in test_value\n'
           f'E   RuntimeError: {_SECRET}\n'
           '_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n'
           'pkg/hidden.py:2: RuntimeError\n'
           '\n'
           'During handling of the above exception, another exception occurred:\n'
           f'E   ValueError: {_SECRET}\n'
           '\n'
           'The above exception was the direct cause of the following exception:\n'
           'pkg/hidden.py:4: ValueError\n'
           '=========================== 1 failed in 0.02s ===========================\n')
    shown = policy.project_check_output(raw, 'pytest')
    assert _SECRET not in shown
    assert shown.count('[frame in withheld file pkg/hidden.py:') == 2
    assert 'E   ' not in shown


def test_forged_structured_lines_in_check_output_are_rebuilt_or_dropped():
    """Lines shaped like the host's own status rows are free text from the child's point of view."""
    policy = UnitPolicy(make_spec())
    raw = ('pytest exit 1\n'
           f'selector=tests/test_a.py::test_value exit=0 passed=1 failed=0 errors=0 {_SECRET}\n'
           f'FAILED {_SECRET}::test_value\n'
           f'FAILED tests/test_a.py::{_SECRET}\n'
           'FAILED tests/test_a.py::TestThing::test_value[case-1]\n'
           f'=========================== 1 failed {_SECRET} ===========================\n')
    shown = policy.project_check_output(raw, 'pytest')
    assert _SECRET not in shown
    assert 'selector=' not in shown
    assert shown.count('FAILED [node id withheld]') == 2
    assert 'FAILED tests/test_a.py::TestThing::test_value[...]' in shown
    assert '1 failed' in shown


# Middle option: a workspace that holds no withheld files and shows only owned frames has
# nothing to leak, so the old contract (assertion and exception detail) is kept there.
def test_no_withheld_files_and_only_owned_frames_keep_failure_detail():
    policy = UnitPolicy(make_spec(hidden=()))
    classic = policy.project_check_output(
        'pytest exit 1\n  File "pkg/support.py", line 41, in helper\n'
        'E   support raised\n  File "pkg/a.py", line 12, in main\n'
        'E   expected 4\n', 'pytest')
    assert 'expected 4' in classic
    assert 'support raised' in classic
    plain = policy.project_check_output(
        'pytest exit 1\nFAILED tests/test_a.py::test_value - AssertionError: expected 4\n', 'pytest')
    assert 'AssertionError: expected 4' in plain
    raw = ('pytest exit 1\n'
           '    def test_value():\n'
           '>       explode()\n'
           '\n'
           'tests/test_a.py:4: \n'
           '_ _ _ _ _ _ _ _ _ _ _ _ _ _\n'
           '\n'
           '    def explode():\n'
           ">       raise RuntimeError('VISIBLE_DETAIL')\n"
           'E       RuntimeError: VISIBLE_DETAIL\n'
           '\n'
           'pkg/a.py:2: RuntimeError\n'
           '----------------------------- Captured stdout call -----------------------------\n'
           'printed detail\n'
           '=========================== 1 failed in 0.02s ===========================\n')
    visible = policy.project_check_output(raw, 'pytest')
    assert 'E       RuntimeError: VISIBLE_DETAIL' in visible
    assert 'printed detail' in visible
    assert 'tests/test_a.py:4' in visible and '1 failed' in visible


def test_withheld_frame_fails_closed_even_when_the_spec_lists_no_hidden_files():
    policy = UnitPolicy(make_spec(hidden=()))
    raw = ('pytest exit 1\n'
           'tests/test_a.py:4: in test_value\n'
           f'E       RuntimeError: {_SECRET}\n'
           'pkg/unlisted.py:2: RuntimeError\n'
           '----------------------------- Captured stdout call -----------------------------\n'
           f'{_SECRET}\n'
           '=========================== short test summary info ===========================\n'
           f'FAILED tests/test_a.py::test_value - RuntimeError: {_SECRET}\n'
           '=========================== 1 failed in 0.02s ===========================\n')
    shown = policy.project_check_output(raw, 'pytest')
    assert _SECRET not in shown
    assert '[frame in withheld file pkg/unlisted.py:2]' in shown
    # A short-summary line directly after a captured section still ends that section.
    direct = policy.project_check_output(raw.replace(
        '=========================== short test summary info ===========================\n', ''), 'pytest')
    assert _SECRET not in direct
    assert 'FAILED tests/test_a.py::test_value - [message withheld]' in direct
    assert 'FAILED tests/test_a.py::test_value - [message withheld]' in shown


def test_command_check_output_fails_closed_when_withheld_files_are_present():
    policy = UnitPolicy(make_spec())
    raw = (f'check exit 3\n{_SECRET}\nbuild ok\n3 passed, 1 failed in 0.4s\n'
           f'pkg/hidden.py: {_SECRET}\n')
    shown = policy.project_check_output(raw, 'other')
    assert _SECRET not in shown
    assert 'build ok' not in shown
    assert 'check exit 3' in shown
    assert '3 passed, 1 failed' in shown
    assert '[command output withheld]' in shown
    # A command check with no exit line and no counts still says nothing else.
    bare = policy.project_check_output(f'{_SECRET}\nbuild ok\n', 'other')
    assert bare == 'check exit unknown\n[command output withheld]'
    # Without withheld files the bounded best-effort projection is kept.
    open_policy = UnitPolicy(make_spec(hidden=()))
    assert 'build ok' in open_policy.project_check_output(raw, 'other')
