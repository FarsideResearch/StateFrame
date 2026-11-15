
import pytest

from kernel_parallel_admission import DirectorySnapshot
from kernel_parallel_contracts import canonical_signature, project_interface, resolve_imports, verify_contracts
from kernel_parallel_records import CheckRecipe, ExportSpec, UnitContract


def snapshot(tmp_path, files):
    for name, text in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    return DirectorySnapshot(tmp_path, 0)


def unit(export):
    return UnitContract('a', 'outcome', (export.path,), (), (export,), (), (), (), CheckRecipe((), (), 'none'))


def test_signature_matrix():
    source = '''
def f(a: int, /, b=1, *args: str, c: list[int], d=2, **kw: object) -> bool: pass
async def af(x): pass
class Widget:
    def __init__(self, x: int, *, active=False): pass
    @property
    def value(self) -> int: pass
ANSWER: int = 42
'''
    sig = canonical_signature(source, 'f')
    assert sig.kind == 'function'
    assert [(p.name, p.kind, p.has_default, p.annotation) for p in sig.params] == [
        ('a', 'posonly', False, 'int'), ('b', 'pos', True, None),
        ('args', 'varpos', False, 'str'), ('c', 'kwonly', False, 'list[int]'),
        ('d', 'kwonly', True, None), ('kw', 'varkw', False, 'object')]
    assert sig.returns == 'bool'
    assert canonical_signature(source, 'af').kind == 'async_function'
    assert [(p.name, p.kind) for p in canonical_signature(source, 'Widget').params] == [
        ('self', 'pos'), ('x', 'pos'), ('active', 'kwonly')]
    assert canonical_signature(source, 'Widget.value').kind == 'method'
    assert canonical_signature(source, 'ANSWER').returns == 'int'


@pytest.mark.parametrize('source,name', [
    ('if flag:\n    def f(): pass\n', 'f'),
    ('def f(): pass\ndef f(x): pass\n', 'f'),
    ('@replace\ndef f(): pass\n', 'f'),
    ('f = factory()\n', 'f'),
    ('from other import *\n', 'f'),
    ('@overload\ndef f(x): pass\n', 'f'),
])
def test_unsupported_forms_are_unknown(source, name):
    assert canonical_signature(source, name) == 'UNKNOWN'


def test_reexport_projection_and_verdict_without_execution(tmp_path):
    snap = snapshot(tmp_path, {'src/pkg/__init__.py': 'from .impl import f\n__all__ = ["f"]\n',
                               'src/pkg/impl.py': 'def f(x: int) -> str:\n    """First line.\n\nMore."""\n    return str(x)\n'})
    expected = canonical_signature(snap.read_bytes('src/pkg/impl.py').decode(), 'f')
    export = ExportSpec('f', 'src/pkg/__init__.py', 'f', expected)
    projection = project_interface(snap, export.path, export.symbol, export.id)
    assert (projection.path, projection.line, projection.doc_first_line, projection.signature) == (
        'src/pkg/impl.py', 1, 'First line.', expected)
    report = verify_contracts([unit(export)], snap, unit_id='a')
    assert [(item.status, item.reason) for item in report.verdicts] == [('PASS', 'MATCH')]
    assert report.ok


def test_import_resolver_root_src_namespace_relative_and_external(tmp_path):
    snap = snapshot(tmp_path, {'src/pkg/__init__.py': '', 'src/pkg/a.py': 'from . import b\nimport json\nimport missing_dependency\n',
                               'src/pkg/b.py': '', 'ns/mod.py': 'from ns import sibling\n',
                               'ns/sibling.py': ''})
    verdicts = resolve_imports(snap, ['src/pkg/a.py', 'ns/mod.py'])
    assert {(item.module, item.status) for item in verdicts} == {
        ('pkg.b', 'RESOLVED'), ('json', 'EXTERNAL'), ('missing_dependency', 'EXTERNAL'),
        ('ns.sibling', 'RESOLVED')}


def test_signature_mismatch_and_missing_export(tmp_path):
    snap = snapshot(tmp_path, {'a.py': 'def f(x, *, y): pass\n'})
    expected = canonical_signature('def f(x): pass\n', 'f')
    wrong = ExportSpec('e', 'a.py', 'f', expected)
    report = verify_contracts([unit(wrong)], snap, unit_id='a')
    assert [(v.status, v.reason) for v in report.verdicts] == [('FAIL', 'SIGNATURE_MISMATCH')]
    missing = ExportSpec('m', 'a.py', 'missing', expected)
    report = verify_contracts([unit(missing)], snap, unit_id='a')
    assert [(v.status, v.reason) for v in report.verdicts] == [('FAIL', 'MISSING_EXPORT')]


def test_undeclared_new_python_export_requires_check_and_presence(tmp_path):
    snap = snapshot(tmp_path, {'a.py': 'def new(value):\n    return value\n'})
    export = ExportSpec('new', 'a.py', 'new', None)
    verdict = verify_contracts([unit(export)], snap, unit_id='a').verdicts[0]
    assert (verdict.status, verdict.reason, verdict.expected, verdict.actual) == (
        'UNKNOWN', 'SIGNATURE_UNDECLARED', None,
        canonical_signature('def new(value):\n    return value\n', 'new'))
    absent = verify_contracts([unit(ExportSpec('missing', 'a.py', 'missing', None))],
                              snap, unit_id='a').verdicts[0]
    assert (absent.status, absent.reason) == ('FAIL', 'MISSING_EXPORT')


def test_unknown_decorator_does_not_pass_vacuously(tmp_path):
    snap = snapshot(tmp_path, {'a.py': '@replace\ndef f(x): pass\n'})
    expected = canonical_signature('def f(x): pass\n', 'f')
    report = verify_contracts([unit(ExportSpec('e', 'a.py', 'f', expected))], snap, unit_id='a')
    assert [(v.status, v.reason) for v in report.verdicts] == [('UNKNOWN', 'UNSUPPORTED_STATIC_FORM')]
    assert all(v.status != 'PASS' for v in report.verdicts)


def test_non_python_definition_presence_is_reported_without_pass(tmp_path):
    snap = snapshot(tmp_path, {'a.js': 'export function f(x) { return x }\n',
                               'b.js': 'const unrelated = true;\n'})
    first = ExportSpec('f', 'a.js', 'f', None)
    second = ExportSpec('missing', 'a.js', 'missing', None)
    contract = UnitContract('a', 'outcome', ('a.js',), (), (first, second), (), (), (), CheckRecipe((), (), 'none'))
    report = verify_contracts([contract], snap, unit_id='a')
    assert [(v.status, v.reason) for v in report.verdicts] == [
        ('UNKNOWN', 'DEFINITION_PRESENT'), ('UNKNOWN', 'DEFINITION_MISSING')]
    assert project_interface(snap, 'a.js', 'f', 'f').line == 1


def test_dataclass_generated_init_and_conditional_assignment_are_unknown(tmp_path):
    assert canonical_signature('@dataclass\nclass Item:\n    x: int\n', 'Item') == 'UNKNOWN'
    snap = snapshot(tmp_path, {'a.py': 'if flag:\n    VALUE = 1\n'})
    expected = canonical_signature('VALUE = 1\n', 'VALUE')
    report = verify_contracts([unit(ExportSpec('value', 'a.py', 'VALUE', expected))], snap, unit_id='a')
    assert [(v.status, v.reason) for v in report.verdicts] == [('UNKNOWN', 'UNSUPPORTED_STATIC_FORM')]


@pytest.mark.parametrize('source,name', [
    ('def f(): pass\nif flag:\n    f = factory()\n', 'f'),
    ('def f(): pass\ntry:\n    from other import f\nexcept ImportError:\n    pass\n', 'f'),
    ('class C:\n    def method(self): pass\n    if flag:\n        method = factory()\n', 'C.method'),
    ('def f(): pass\nfor f in values:\n    pass\n', 'f'),
])
def test_conditional_rebinding_never_has_pass_signature(source, name):
    assert canonical_signature(source, name) == 'UNKNOWN'


@pytest.mark.parametrize('source,name', [
    ('class C:\n    def __init__(self, x): pass\n    __init__ = factory()\n', 'C'),
    ('class C:\n    def __init__(self, x): pass\n    if flag:\n        __init__ = factory()\n', 'C'),
    ('class C:\n    def method(self): pass\n    method += factory()\n', 'C.method'),
    ('def f(): pass\nf += factory()\n', 'f'),
    ('def f(): pass\ndel f\n', 'f'),
    ('def f(): pass\nimport other as f\n', 'f'),
    ('def f(): pass\nf: object\n', 'f'),
    ('def f(): pass\nwith context() as f:\n    pass\n', 'f'),
    ('def f(): pass\nif (f := factory()):\n    pass\n', 'f'),
    ('def f(): pass\nmatch value:\n    case f:\n        pass\n', 'f'),
])
def test_every_export_binding_makes_signature_unknown(source, name):
    assert canonical_signature(source, name) == 'UNKNOWN'


def test_missing_internal_import_is_unresolved_and_dynamic_package_is_unknown(tmp_path):
    snap = snapshot(tmp_path, {'pkg/__init__.py': '', 'consumer.py': 'from pkg import missing\n'})
    assert [(v.module, v.status) for v in resolve_imports(snap, ['consumer.py'])] == [
        ('pkg.missing', 'UNRESOLVED')]
    contract = UnitContract('a', 'outcome', ('consumer.py',), (), (), (), (), (), CheckRecipe((), (), 'none'))
    assert verify_contracts([contract], snap, unit_id='a').ok is False
    (tmp_path / 'pkg/__init__.py').write_text('def __getattr__(name): return 1\n')
    snap = DirectorySnapshot(tmp_path, 0)
    assert resolve_imports(snap, ['consumer.py'])[0].status == 'UNKNOWN'
    ns = snapshot(tmp_path / 'namespace', {'ns/module.py': '', 'use.py': 'from ns import missing\n'})
    assert resolve_imports(ns, ['use.py'])[0].status == 'UNRESOLVED'


def test_non_python_comment_and_nested_declaration_are_not_top_level_exports(tmp_path):
    snap = snapshot(tmp_path, {'a.js': '// function fake() {}\nfunction outer() {\n'
                               'function nested() {}\n}\n'})
    contract = UnitContract('a', 'outcome', ('a.js',), (),
                            (ExportSpec('fake', 'a.js', 'fake', None), ExportSpec('nested', 'a.js', 'nested', None)),
                            (), (), (), CheckRecipe((), (), 'none'))
    verdicts = verify_contracts([contract], snap, unit_id='a').verdicts
    assert all(item.status == 'UNKNOWN' for item in verdicts)
    assert all(item.reason != 'DEFINITION_PRESENT' for item in verdicts)
