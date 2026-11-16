"""A subframe's entry must be an owned source file, never its own test file.

When a unit owns both a test file and the source it tests, the test file can
sort first (``tests/...`` < ``textutil.py``). The repair harness moves test
files out of ``files`` into ``public_tests``, so an entry pointing at the test
file is rejected with "row.entry.path is not declared in files" and the unit
dies before its first request.
"""
from kernel_parallel_records import CheckRecipe, ParallelLimits, UnitContract, UnitSpec
from kernel_unit_task import build_public_row


def make_spec(owns, check):
    contract = UnitContract('a', 'make result', owns, (), (), (), (), (), check)
    visible = tuple(sorted((*owns, *check.source_paths)))
    return UnitSpec('call', 'a', 'execution', 'seed', contract, (), (), visible,
                    (), '', 0, 0, 'manifest', 'policy', ParallelLimits())


def _row(owns, check_paths):
    check = CheckRecipe(check_paths, tuple(f'{p}::test_x' for p in check_paths), 'pytest')
    files = {path: b'X = 1\n' for path in (*owns, *check_paths)}
    return build_public_row(make_spec(owns, check), files)


def test_entry_skips_owned_test_file_that_sorts_first():
    row = _row(('tests/test_textutil.py', 'textutil.py'), ('tests/test_textutil.py',))
    assert row['entry'] == {'path': 'textutil.py', 'fn': ''}
    assert row['path'] == row['rel'] == 'textutil.py'


def test_entry_skips_owned_test_file_outside_tests_dir():
    row = _row(('pkg/test_a.py', 'pkg/zeta.py'), ())
    assert row['entry']['path'] == 'pkg/zeta.py'


def test_entry_keeps_first_source_when_it_sorts_first():
    row = _row(('pricing.py', 'tests/test_pricing.py'), ('tests/test_pricing.py',))
    assert row['entry']['path'] == 'pricing.py'


def test_entry_falls_back_to_first_owned_when_unit_owns_only_tests():
    row = _row(('tests/test_a.py',), ())
    assert row['entry']['path'] == 'tests/test_a.py'
