"""A mutation unit cannot raise host-owned resource ceilings."""

import pytest

from kernel_exec_capability import ExecLimits
from kernel_mutate_build import BuildError, source_digest
from kernel_mutate_exec import MutationLimits


def _manifest(mem_mb):
    return {
        "schema": "stateframe-mutation-unit/1", "name": "reader", "revision": 1,
        "parent": None, "entry": "main.py", "summary": "reader", "kind": "reader",
        "input_schema": {"type": "object"}, "output_schema": {"type": "object"},
        "requires": {"network": False, "writes": False},
        "smoke": [{"input": {}, "expect": {}}],
        "limits": {"wall_s": 10, "cpu_s": 5, "mem_mb": mem_mb},
    }


@pytest.mark.parametrize("mem_mb", [257, 1048576])
def test_authored_manifest_cannot_raise_memory_ceiling(mem_mb):
    with pytest.raises(BuildError) as error:
        source_digest({"main.py": b"def handle(input, ctx): return {}\n"}, _manifest(mem_mb))
    assert error.value.code == "LIMITS"


@pytest.mark.parametrize("mem_mb", [257, 1048576])
def test_direct_launch_limits_cannot_raise_memory_ceiling(mem_mb):
    with pytest.raises(ValueError, match="memory"):
        MutationLimits(mem_mb=mem_mb)


def test_mutation_memory_below_and_at_ceiling_remains_available():
    files = {"main.py": b"def handle(input, ctx): return {}\n"}
    for mem_mb in (64, 256):
        assert source_digest(files, _manifest(mem_mb)).startswith("sha256:")
        assert MutationLimits(mem_mb=mem_mb).mem_mb == mem_mb


@pytest.mark.parametrize(("name", "value"), [
    ("wall_s", 121), ("wall_s", 1048576),
    ("cpu_s", 301), ("cpu_s", 1048576),
])
def test_authored_manifest_cannot_raise_time_ceiling(name, value):
    manifest = _manifest(256)
    manifest["limits"][name] = value
    with pytest.raises(BuildError) as error:
        source_digest({"main.py": b"def handle(input, ctx): return {}\n"}, manifest)
    assert error.value.code == "LIMITS"


@pytest.mark.parametrize(("name", "value"), [
    ("wall_s", 121), ("wall_s", 1048576),
    ("cpu_s", 301), ("cpu_s", 1048576),
])
def test_direct_launch_limits_cannot_raise_time_ceiling(name, value):
    with pytest.raises(ValueError, match=name.split("_")[0]):
        MutationLimits(**{name: value})


def test_mutation_time_at_ceiling_remains_available():
    manifest = _manifest(256)
    manifest["limits"].update(wall_s=120, cpu_s=300)
    assert source_digest({"main.py": b"def handle(input, ctx): return {}\n"}, manifest).startswith("sha256:")
    limits = MutationLimits(wall_s=120, cpu_s=300)
    assert (limits.wall_s, limits.cpu_s) == (120, 300)
    host_defaults = ExecLimits()
    assert (limits.wall_s, limits.cpu_s) == (host_defaults.wall_seconds, host_defaults.cpu_seconds)
