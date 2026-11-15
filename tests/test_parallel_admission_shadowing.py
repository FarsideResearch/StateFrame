"""Scope-aware name coupling: a bare name only couples to a producer when it is a free global use."""
import textwrap

import pytest

from kernel_parallel_admission import DirectorySnapshot, admit, scan_coupling
from kernel_parallel_records import CheckRecipe, ParallelLimits, ParallelRequest, UnitContract

PRODUCER = 'def name(): pass\nid = 1\nitems = []\nclass Config: pass\n'

SHADOWED = {
    'param': 'def f(name):\n    return name\n',
    'kwonly_param': 'def f(*, name):\n    return name\n',
    'posonly_param': 'def f(name, /):\n    return name\n',
    'vararg': 'def f(*name):\n    return name\n',
    'kwarg': 'def f(**name):\n    return name\n',
    'lambda_param': 'g = lambda name: name\n',
    'local_assign': 'def f():\n    name = 1\n    return name\n',
    'augassign': 'def f():\n    name = 0\n    name += 1\n    return name\n',
    'annassign': 'def f():\n    name: int = 1\n    return name\n',
    'for_target': 'def f(xs):\n    for name in xs:\n        print(name)\n',
    'with_as': 'def f(c):\n    with c as name:\n        return name\n',
    'except_as': 'def f():\n    try:\n        pass\n    except Exception as name:\n        return name\n',
    'list_comp': 'def f(xs):\n    return [name for name in xs]\n',
    'set_comp': 'def f(xs):\n    return {name for name in xs}\n',
    'dict_comp': 'def f(xs):\n    return {name: 1 for name in xs}\n',
    'gen_exp': 'def f(xs):\n    return sum(1 for name in xs if name)\n',
    'walrus': 'def f(xs):\n    if (name := xs):\n        return name\n',
    'nested_def': 'def f():\n    def name(): pass\n    return name\n',
    'nested_class': 'def f():\n    class name: pass\n    return name\n',
    'module_def': 'def name(): pass\nx = name\n',
    'module_assign': 'name = 3\nx = name\n',
    'import_alias': 'import os as name\nx = name\n',
    'from_import_alias': 'from os import path as name\nx = name\n',
    'match_capture': 'def f(v):\n    match v:\n        case [name]:\n            return name\n',
    'match_star': 'def f(v):\n    match v:\n        case [*name]:\n            return name\n',
    'enclosing_function': 'def f(name):\n    def g():\n        return name\n    return g\n',
}

COUPLED = {
    'from_import': ('from producer import name\nx = name\n', 'static_import'),
    'import_module': ('import producer\n', 'static_import'),
    'attribute_access': ('import producer\nx = producer.name\n', 'static_import'),
    'free_global': ('def f():\n    return name()\n', 'static_name'),
    'free_global_module_level': ('x = id\n', 'static_name'),
    'free_in_comprehension_body': ('def f(xs):\n    return [name for _ in xs]\n', 'static_name'),
    'free_in_default': ('def f(a=Config):\n    return a\n', 'static_name'),
    'free_in_nested_function': ('def f():\n    def g():\n        return items\n    return g\n', 'static_name'),
    'class_scope_not_visible_to_method': ('class K:\n    name = 1\n    def m(self):\n        return name\n', 'static_name'),
    'shadow_in_other_function_only': ('def f(name):\n    return name\ndef g():\n    return name\n', 'static_name'),
}


def contract(uid, path):
    return UnitContract(uid, 'result', (path,), (), (), (), (), (), CheckRecipe((), (), 'none'))


def edges_for(tmp_path, consumer, producer=PRODUCER):
    (tmp_path / 'producer.py').write_text(producer)
    (tmp_path / 'consumer.py').write_text(consumer)
    edges, _ = scan_coupling([contract('p', 'producer.py'), contract('c', 'consumer.py')],
                             DirectorySnapshot(tmp_path, 1), ParallelLimits())
    return edges


@pytest.mark.parametrize('form', sorted(SHADOWED))
def test_shadowing_binding_form_creates_no_edge(tmp_path, form):
    assert edges_for(tmp_path, SHADOWED[form]) == []


@pytest.mark.parametrize('form', sorted(COUPLED))
def test_real_coupling_still_found(tmp_path, form):
    source, kind = COUPLED[form]
    edges = edges_for(tmp_path, source)
    assert [(e.src_unit, e.dst_unit, e.source) for e in edges] == [('p', 'c', kind)]


def test_free_use_reports_the_free_line(tmp_path):
    edges = edges_for(tmp_path, 'def f(name):\n    return name\n\nx = id\n')
    assert edges[0].evidence[0].line == 4 and edges[0].evidence[0].text == 'id'


def test_syntax_error_consumer_is_unknown_not_refused(tmp_path):
    (tmp_path / 'producer.py').write_text(PRODUCER)
    (tmp_path / 'consumer.py').write_text('def (:\n')
    edges, pairs = scan_coupling([contract('p', 'producer.py'), contract('c', 'consumer.py')],
                                 DirectorySnapshot(tmp_path, 1), ParallelLimits())
    assert edges == [] and pairs[0].status == 'UNKNOWN'


def test_admit_refusal_lists_all_undeclared_edges(tmp_path):
    files = {'a.py': 'def alpha(): pass\n', 'b.py': 'def beta(): pass\nx = alpha\n', 'c.py': 'y = alpha\nz = beta\n'}
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    req = ParallelRequest('call', tuple(contract(u, f'{u}.py') for u in 'abc'), '2')
    result = admit(req, DirectorySnapshot(tmp_path, 1), (), ParallelLimits(), protected=lambda _: False,
                   case_insensitive=False)
    assert result.status == 'REFUSED' and result.refusal.reason == 'UNDECLARED_COUPLING'
    parts = result.refusal.detail.split(', ')
    assert sorted(parts) == ['a->b', 'a->c', 'b->c']
    assert parts[0] == 'a->b'
    assert list(result.refusal.paths) == ['b.py', 'c.py']
