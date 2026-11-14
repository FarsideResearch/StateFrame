import hashlib
import json
import subprocess
import sys
from pathlib import Path

from kernel_parallel_admission import DirectorySnapshot, admit
from kernel_parallel_contracts import canonical_signature
from kernel_parallel_projection import project_unit, render_unit_task
from kernel_parallel_records import (CheckRecipe, ConsumeSpec, ExportSpec, ParallelLimits,
                                     ParallelRequest, RequirementClause, UnitContract, UnitSpec)


def fixture(tmp_path):
    (tmp_path / 'a.py').write_text('def f(x):\n    """An interface."""\n    return x\n')
    (tmp_path / 'b.py').write_text('pass\n')
    (tmp_path / 'tests').mkdir()
    (tmp_path / 'tests/test_b.py').write_text('def test_value(): pass\n')
    snapshot = DirectorySnapshot(tmp_path, 5)
    signature = canonical_signature((tmp_path / 'a.py').read_text(), 'f')
    a = UnitContract('a', 'provide f', ('a.py',), (), (ExportSpec('f', 'a.py', 'f', signature),),
                     (), (), (), CheckRecipe((), (), 'none'))
    b = UnitContract('b', 'use f', ('b.py',), (), (), (ConsumeSpec('f', None, None),),
                     (), ('R2',), CheckRecipe(('tests/test_b.py',), ('tests/test_b.py::test_value',), 'pytest'))
    req = ParallelRequest('call', (a, b), '2')
    admission = admit(req, snapshot, (RequirementClause('R2', 'planner obligation', 'planner', ()),),
                      ParallelLimits(), protected=lambda _: False, case_insensitive=False)
    assert admission.status == 'ADMITTED'
    return snapshot, req, admission


def test_projection_is_stable_and_keeps_only_needed_clauses(tmp_path):
    snapshot, req, admission = fixture(tmp_path)
    clauses = (RequirementClause('R2', 'The result stays stable', 'planner', ()),
               RequirementClause('R1', 'preserve behavior', 'intersect', ('b.py',)),
               RequirementClause('R3', 'unrelated obligation', 'intersect', ('other.py',)))
    first = project_unit(admission, req, 'b', snapshot, parent_seed='seed', policy_digest='policy',
                         failure_projection='one failed assertion', requirements=clauses)
    second = project_unit(admission, req, 'b', snapshot, parent_seed='seed', policy_digest='policy',
                          failure_projection='one failed assertion', requirements=clauses)
    assert first == second
    assert first.limits == admission.limits
    assert first.to_json()['limits'] == admission.limits.to_json()
    assert UnitSpec.from_json(first.to_json()) == first
    assert first.digest() == second.digest()
    expected_execution = hashlib.sha256(b'call\0b').hexdigest()[:32]
    expected_seed = hashlib.sha256(('seed\0' + expected_execution).encode()).hexdigest()[:32]
    assert (first.execution_id, first.run_seed) == (expected_execution, expected_seed)
    assert [clause.id for clause in first.requirement_clauses] == ['R1', 'R2']
    assert first.visible_paths == ('b.py', 'tests/test_b.py')
    assert first.hidden_paths == ('a.py',)
    assert first.interfaces[0].path == 'a.py'
    task = render_unit_task(first)
    assert 'subframe' in task.lower()
    assert 'unit' not in task.lower()
    assert 'use f' in task and 'one failed assertion' in task
    assert 'tests/test_b.py::test_value' in task
    assert 'read-only or withheld' in task
    assert len(task) <= 2500
    assert 'incarnation' not in json.dumps(first.to_json())


def test_requirement_refs_intersect_without_prose_and_unknown_planner_id_refuses(tmp_path):
    snapshot, req, admission = fixture(tmp_path)
    clauses = (RequirementClause('R2', 'planner obligation', 'planner', ()),
               RequirementClause('R1', 'neutral wording', 'intersect', ('b.py',)),
               RequirementClause('R3', 'b.py appears in prose', 'intersect', ('other.py',)))
    spec = project_unit(admission, req, 'b', snapshot, parent_seed='seed', policy_digest='policy',
                        failure_projection='', requirements=clauses)
    assert [(c.id, c.refs) for c in spec.requirement_clauses] == [('R1', ('b.py',)), ('R2', ())]
    assert admit(req, snapshot, clauses, ParallelLimits(), protected=lambda _: False,
                 case_insensitive=False).status == 'ADMITTED'
    refused = admit(req, snapshot, (), ParallelLimits(), protected=lambda _: False, case_insensitive=False)
    assert (refused.status, refused.refusal.reason) == ('REFUSED', 'UNKNOWN_REQUIREMENT')


def test_symbol_refs_match_declared_consumed_export_and_owned_path(tmp_path):
    snapshot, req, admission = fixture(tmp_path)
    clauses = (RequirementClause('R2', 'planner', 'planner', ()),
               RequirementClause('R3', 'symbol', 'intersect', ('a.py::f',)),
               RequirementClause('R4', 'owned symbol', 'intersect', ('b.py::helper',)))
    spec = project_unit(admission, req, 'b', snapshot, parent_seed='seed', policy_digest='policy',
                        failure_projection='', requirements=clauses)
    assert [clause.id for clause in spec.requirement_clauses] == ['R2', 'R3', 'R4']


def test_digest_matches_fresh_process(tmp_path):
    snapshot, req, admission = fixture(tmp_path)
    clauses = (RequirementClause('R2', 'planner obligation', 'planner', ()),)
    spec = project_unit(admission, req, 'b', snapshot, parent_seed='seed', policy_digest='policy',
                        failure_projection='', requirements=clauses)
    script = tmp_path.parent / (tmp_path.name + '_project_again.py')
    root = Path(__file__).resolve().parents[1]
    script.write_text('import sys\n'
                      f'sys.path[:0] = {[str(root / "src"), str(root / "src/stateframe/engine")]!r}\n'
                      'from kernel_parallel_admission import DirectorySnapshot\n'
                      'from kernel_parallel_records import AdmissionResult, ParallelRequest\n'
                      'from kernel_parallel_records import RequirementClause\n'
                      'from kernel_parallel_projection import project_unit\n'
                      f's = DirectorySnapshot(__import__("pathlib").Path({str(tmp_path)!r}), 5)\n'
                      f'a = AdmissionResult.from_json({admission.to_json()!r})\n'
                      f'r = ParallelRequest.from_json({req.to_json()!r})\n'
                      'print(project_unit(a, r, "b", s, parent_seed="seed", policy_digest="policy", '
                      'failure_projection="", requirements=(RequirementClause("R2", "planner obligation", "planner", ()),)).digest())\n')
    result = subprocess.run([sys.executable, '-I', str(script)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == spec.digest()
