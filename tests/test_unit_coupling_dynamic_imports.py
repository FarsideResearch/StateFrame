"""Dynamic imports must not make coupled units look independent."""
import pytest

from kernel_parallel_admission import DirectorySnapshot, scan_coupling
from kernel_parallel_records import CheckRecipe, ParallelLimits, UnitContract


def _unit(unit_id, path):
    return UnitContract(unit_id, 'result', (path,), (), (), (), (), (),
                        CheckRecipe((), (), 'none'))


def _scan(tmp_path, consumer_source):
    package = tmp_path / 'pkg'
    package.mkdir(exist_ok=True)
    (package / '__init__.py').write_text('')
    (package / 'producer.py').write_text('def produced(): pass\n')
    (package / 'consumer.py').write_text(consumer_source)
    units = (_unit('producer', 'pkg/producer.py'),
             _unit('consumer', 'pkg/consumer.py'))
    return scan_coupling(units, DirectorySnapshot(tmp_path, 1), ParallelLimits())


@pytest.mark.parametrize('source', [
    'import importlib\nimportlib.import_module("pkg.producer")\n',
    'import importlib as loader\nloader.import_module("pkg.producer")\n',
    'from importlib import import_module as load\nload("pkg.producer")\n',
    '__import__("pkg.producer")\n',
])
def test_literal_dynamic_imports_create_coupling_edges(tmp_path, source):
    edges, pairs = _scan(tmp_path, source)

    assert [(edge.src_unit, edge.dst_unit, edge.source) for edge in edges] == [
        ('producer', 'consumer', 'dynamic_import')]
    assert pairs[0].status == 'COUPLED'
    assert pairs[0].evidence[0].path == 'pkg/consumer.py'


@pytest.mark.parametrize('source', [
    'import importlib\nimportlib.import_module(module_name)\n',
    '__import__(module_name)\n',
])
def test_computed_dynamic_imports_conservatively_couple_units(tmp_path, source):
    edges, pairs = _scan(tmp_path, 'module_name = resolve_name()\n' + source)

    assert [(edge.src_unit, edge.dst_unit, edge.source) for edge in edges] == [
        ('producer', 'consumer', 'dynamic_import')]
    assert pairs[0].status == 'COUPLED'
    assert pairs[0].evidence[0].path == 'pkg/consumer.py'


def test_dynamic_import_of_unrelated_module_keeps_units_independent(tmp_path):
    edges, pairs = _scan(
        tmp_path,
        'import importlib\nimportlib.import_module("other.module")\n',
    )

    assert edges == []
    assert [(pair.status, pair.reason) for pair in pairs] == [
        ('INDEPENDENT_BY_SCAN', '')]
