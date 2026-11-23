"""The engine switch registry ships precomputed; a run hashes the sources instead of AST-parsing them.

AST-parsing every declared engine file took 22-24 s per scan on a loaded host and ran twice per run
(host and engine child), which pushed first-request waits past their limits.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from collections import Counter
from hashlib import sha256
from pathlib import Path

import pytest

from stateframe import profiles, routes
from stateframe.engine import kernel_route_budget as route_budget

ENGINE = Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"
sys.path.insert(0, str(ENGINE))
import kernel_release_switch_registry as registry  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    registry._release_registry.cache_clear()
    yield
    registry._release_registry.cache_clear()


def _toy_root(tmp_path: Path) -> Path:
    root = tmp_path / "engine"
    root.mkdir()
    (root / "a.py").write_text('import os\nX = os.environ.get("STATEFRAME_ALPHA")\n')
    (root / "b.py").write_text('import os\nif os.getenv("STATEFRAME_BETA") == "on":\n    pass\n')
    (root / "release-manifest.json").write_text(json.dumps({"files": {"a.py": "", "b.py": ""}}))
    return root


def _forbid_scan(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("source scan ran although the shipped registry matches")
    monkeypatch.setattr(registry, "_scan", refuse)


def test_shipped_registry_matches_a_fresh_scan_of_this_release():
    shipped = json.loads((ENGINE / registry.REGISTRY_FILE).read_text())
    expected = registry.build_registry_document(ENGINE)
    assert shipped == expected, (
        f"{registry.REGISTRY_FILE} is stale; regenerate it with "
        "python -I src/stateframe/engine/kernel_release_switch_registry.py --write")


def test_matching_shipped_registry_is_used_without_parsing(tmp_path, monkeypatch):
    root = _toy_root(tmp_path)
    registry.write_registry(root)
    scanned = json.loads((root / registry.REGISTRY_FILE).read_text())
    assert scanned["switches"] == ["STATEFRAME_ALPHA", "STATEFRAME_BETA"]
    registry._release_registry.cache_clear()
    _forbid_scan(monkeypatch)
    switches, digest = registry.release_stateframe_switches(root)
    assert switches == frozenset({"STATEFRAME_ALPHA", "STATEFRAME_BETA"}) and digest == scanned["digest"]


def _launch_registry(root: Path) -> tuple[int, dict]:
    probe = """
import json
from pathlib import Path
import sys

sys.path.insert(0, sys.argv[1])
import kernel_release_switch_registry as registry

scans = 0
scan = registry._scan

def counted_scan(files):
    global scans
    scans += 1
    return scan(files)

registry._scan = counted_scan
root = Path(sys.argv[2])
registry.release_stateframe_switches(root)
detail = registry.unconsumed_stateframe_switches(
    {"STATEFRAME_ALPHA": "on", "STATEFRAME_BETA": "on", "STATEFRAME_ZETA": "on"},
    root=root,
)
print(json.dumps([scans, detail]))
"""
    scans, detail = json.loads(subprocess.check_output(
        [sys.executable, "-I", "-c", probe, str(ENGINE), str(root)]))
    return scans, detail


def test_launches_reuse_scan_until_source_bytes_change(tmp_path):
    root = _toy_root(tmp_path)
    registry.write_registry(root)
    with (root / "a.py").open("a") as source:
        source.write("# Source edit after registry generation\n")
    first_scan_count, first = _launch_registry(root)
    assert first_scan_count == 1
    assert first["unknown_switches"] == ["STATEFRAME_ZETA"]
    assert first["registry_digest"] == registry.build_registry_document(root)["digest"]
    assert _launch_registry(root) == (0, first)

    source = root / "b.py"
    before = source.stat()
    source.write_text(source.read_text().replace("STATEFRAME_BETA", "STATEFRAME_ZETA"))
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert source.stat().st_size == before.st_size

    changed_scan_count, changed = _launch_registry(root)
    assert changed_scan_count == 1
    assert changed["unknown_switches"] == ["STATEFRAME_BETA"]
    assert changed["registry_digest"] == registry.build_registry_document(root)["digest"]
    assert _launch_registry(root) == (0, changed)


def test_preflight_recovers_from_a_damaged_disk_cache(tmp_path):
    root = _toy_root(tmp_path)
    scans, expected = _launch_registry(root)
    assert scans == 1
    cache = registry._registry_cache_path(root)
    document = json.loads(cache.read_text())
    document["switches"].append("STATEFRAME_ZETA")
    cache.write_text(json.dumps(document))
    assert _launch_registry(root) == (1, expected)
    assert _launch_registry(root) == (0, expected)


def test_shipped_registry_fingerprint_matches_the_real_engine_sources():
    shipped = json.loads((ENGINE / registry.REGISTRY_FILE).read_text())
    files = registry._declared_python_files(ENGINE)
    assert shipped["sources_sha256"] == registry._sources_fingerprint(ENGINE, files)


def test_engine_source_seals_the_current_switch_registry():
    source = json.loads((ENGINE / "ENGINE_SOURCE.json").read_bytes())
    registry_bytes = (ENGINE / registry.REGISTRY_FILE).read_bytes()
    expected = json.loads(registry_bytes)
    files = registry._declared_python_files(ENGINE)
    expected["sources_sha256"] = registry._sources_fingerprint(ENGINE, files)
    expected_bytes = (json.dumps(expected, indent=1) + "\n").encode("utf-8")

    assert source["files"][registry.REGISTRY_FILE] == sha256(expected_bytes).hexdigest()


def test_the_real_engine_starts_without_parsing_its_sources(monkeypatch):
    _forbid_scan(monkeypatch)
    switches, digest = registry.release_stateframe_switches(ENGINE)
    assert len(switches) > 100 and digest.startswith("sha256:")


def test_host_only_env_seed_setting_is_not_in_engine_switch_registry():
    switches, _ = registry.release_stateframe_switches(ENGINE)
    assert "STATEFRAME_ENV_SEED" not in switches


def _parsed_environment_prefixes(package: Path) -> dict[str, set[str]]:
    """Find namespaces consumed by parsers, excluding mere env filters.

    Prefix removal identifies parsing ownership: unlike startswith-only filters
    (child env isolation, privileged settings), it interprets the remaining name.
    Read literals and named constants from code, never a copied prefix list.
    """
    # This one removal formats an already-rejected setting for an error. It
    # does not consume valid settings or reserve a namespace. Review both its
    # exact expression/count here and its prefix-literal count below.
    error_label_strips = {
        ("engine/kernel_parallel_records.py", "ParallelLimits.from_settings",
         "switch.removeprefix('STATEFRAME_PARALLEL_')"): 1,
    }
    prefixes: dict[str, set[str]] = {}
    for path in package.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if "STATEFRAME_" not in source or "removeprefix" not in source:
            continue
        tree = ast.parse(source, filename=str(path))
        owner = path.relative_to(package).as_posix()
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

        def scope_name(node):
            scopes = []
            parent = parents.get(node)
            while parent is not None:
                if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    scopes.append(parent.name)
                parent = parents.get(parent)
            return ".".join(reversed(scopes)) or "<module>"

        strips = [node for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and node.func.attr == "removeprefix" and node.args]
        strip_counts = Counter((owner, scope_name(node), ast.unparse(node)) for node in strips)
        constants = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        constants[target.id] = node.value.value
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "removeprefix" and node.args):
                continue
            argument = node.args[0]
            prefix = (argument.value if isinstance(argument, ast.Constant)
                      else constants.get(argument.id) if isinstance(argument, ast.Name) else None)
            if isinstance(prefix, str) and prefix.startswith("STATEFRAME_"):
                site = (owner, scope_name(node), ast.unparse(node))
                if site in error_label_strips and strip_counts[site] == error_label_strips[site]:
                    continue
                prefixes.setdefault(prefix, set()).add(owner)
    return prefixes


def _assert_namespace_discovery_is_complete(package: Path, parsed: dict[str, set[str]]) -> None:
    """Fail closed when parser syntax escapes discovery.

    Inventory namespace literals independently of string operations. A new
    namespace must be discovered above or reviewed here as a non-parser use.
    Exemptions bind to a file, scope and occurrence count, so a new parser cannot
    silently inherit a whole-file or whole-prefix exemption. STATEFRAME_ itself
    is the configuration root, not a subnamespace.
    """
    reviewed_filters_and_builders = {
        # These sites filter the opt-in parallel settings, never parse role keys.
        # One startswith filter and one rejected-name error-label removal.
        ("engine/kernel_parallel_records.py", "ParallelLimits.from_settings", "STATEFRAME_PARALLEL_"): 2,
        ("runner.py", "resolve_opt_in_switches", "STATEFRAME_PARALLEL_"): 1,
        ("runner.py", "prepare", "STATEFRAME_PARALLEL_"): 2,
        ("runner.py", "<module>", "STATEFRAME_PARALLEL_"): 1,
        ("runner.py", "_provider_environment", "STATEFRAME_PARALLEL_"): 1,
        ("runner.py", "_compare_resume_routes", "STATEFRAME_ROUTE_"): 1,
        ("runner.py", "ignored_shell_settings", "STATEFRAME_ROUTE_"): 1,
        ("_child.py", "install_environment", "STATEFRAME_TASK_"): 1,
        ("engine/capability_release_runner.py", "<module>", "STATEFRAME_TASK_"): 1,
        ("engine/kernel_protocol_spec.py", "_reader_default", "STATEFRAME_LEAN_STACK_"): 1,
        ("engine/kernel_model_profile.py", "<module>", "STATEFRAME_LARGE_"): 1,
        ("engine/kernel_model_profile.py", "<module>", "STATEFRAME_PROFILE_RELAX_"): 1,
        ("engine/kernel_host_cache.py", "cache_enabled", "STATEFRAME_HOST_CACHE_"): 1,
    }
    undiscovered: Counter[tuple[str, str, str]] = Counter()
    for path in package.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if "STATEFRAME_" not in source:
            continue
        tree = ast.parse(source, filename=str(path))
        owner = path.relative_to(package).as_posix()
        literals = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and node.value.startswith("STATEFRAME_") and node.value.endswith("_")
                    and node.value.isidentifier() and node.value != "STATEFRAME_"):
                continue
            if owner in parsed.get(node.value, set()):
                continue
            literals.append(node)
        if not literals:
            continue
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in literals:
            scope = []
            parent = parents.get(node)
            while parent is not None:
                if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    scope.append(parent.name)
                parent = parents.get(parent)
            undiscovered[(owner, ".".join(reversed(scope)) or "<module>", node.value)] += 1
    unreviewed = {site: count for site, count in undiscovered.items()
                  if count > reviewed_filters_and_builders.get(site, 0)}
    assert unreviewed == {}, f"unreviewed environment namespaces (extend parser discovery): {unreviewed}"


def test_registered_switches_do_not_collide_with_parser_namespaces():
    shipped = json.loads((ENGINE / registry.REGISTRY_FILE).read_text())
    prefixes = _parsed_environment_prefixes(ENGINE.parent)
    assert any("routes.py" in owners for owners in prefixes.values())
    _assert_namespace_discovery_is_complete(ENGINE.parent, prefixes)
    # These exact engine metadata names are explicitly outside parser ownership.
    candidates = set(shipped["switches"]) - routes._SPEC_ONLY
    collisions = [f"{name}: {prefix} belongs to {', '.join(sorted(owners))}"
                  for name in sorted(candidates)
                  for prefix, owners in prefixes.items() if name.startswith(prefix)]
    assert collisions == []


@pytest.mark.parametrize("replacement", [None, "off", "on"])
def test_deprecated_route_budget_alias_is_normalized_before_engine_preflight(
        tmp_path, monkeypatch, capsys, replacement):
    """An alias must survive both public parsers without becoming a registered switch."""
    route_budget._warn_legacy_once.cache_clear()
    switches = {"STATEFRAME_ROUTE_BUDGET": "on"}
    if replacement is not None:
        switches["STATEFRAME_ROUTEBUDGET_MODE"] = replacement
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"schema": "stateframe-product-profile/1", "switches": switches}))
    normalized = profiles.load(str(path))
    assert normalized == {"STATEFRAME_ROUTEBUDGET_MODE": replacement or "on"}
    assert registry.unconsumed_stateframe_switches(normalized)["unknown_switches"] == []
    assert routes._env_blocks(switches) == {}
    monkeypatch.setattr(route_budget.os, "environ", switches)
    assert route_budget.mode() == (replacement or "on")
    assert "deprecated" in capsys.readouterr().err
    for registered in (json.loads((ENGINE / registry.REGISTRY_FILE).read_text())["switches"],
                       registry.build_registry_document(ENGINE)["switches"]):
        assert "STATEFRAME_ROUTEBUDGET_MODE" in registered
        assert "STATEFRAME_ROUTE_BUDGET" not in registered


def test_deprecated_switch_alias_exemption_is_exact():
    # Only this compatibility spelling is excluded from registry discovery.
    # It is normalized at the public boundary, never admitted as an engine switch.
    assert registry.DEPRECATED_SWITCH_ALIASES == frozenset({"STATEFRAME_ROUTE_BUDGET"})


def test_namespace_guard_still_rejects_a_registered_deprecated_alias(tmp_path, monkeypatch):
    package = tmp_path / "stateframe"
    engine = package / "engine"
    engine.mkdir(parents=True)
    (package / "routes.py").write_text((ENGINE.parent / "routes.py").read_text())
    (engine / registry.REGISTRY_FILE).write_text(json.dumps({
        "switches": ["STATEFRAME_ROUTE_BUDGET"],
    }))
    monkeypatch.setitem(globals(), "ENGINE", engine)
    with pytest.raises(AssertionError, match="STATEFRAME_ROUTE_BUDGET"):
        test_registered_switches_do_not_collide_with_parser_namespaces()


def test_parser_namespace_discovery_reads_literals_and_constants(tmp_path):
    (tmp_path / "parser.py").write_text(
        'PREFIX = "STATEFRAME_ANOTHER_"\n'
        'tail = name.removeprefix(PREFIX)\n'
        'other = name.removeprefix("STATEFRAME_LITERAL_")\n'
        'filtered = name.startswith("STATEFRAME_FILTER_")\n')
    assert _parsed_environment_prefixes(tmp_path) == {
        "STATEFRAME_ANOTHER_": {"parser.py"}, "STATEFRAME_LITERAL_": {"parser.py"}}


@pytest.mark.parametrize("tail", [
    'name[len(PREFIX):]',
    'name.split(PREFIX, 1)[1]',
])
def test_namespace_collision_check_rejects_unseen_parser_syntax(tmp_path, monkeypatch, tail):
    package = tmp_path / "stateframe"
    engine = package / "engine"
    engine.mkdir(parents=True)
    (package / "routes.py").write_text((ENGINE.parent / "routes.py").read_text())
    (engine / registry.REGISTRY_FILE).write_text(json.dumps({
        "switches": ["STATEFRAME_NEW_ENABLED"],
    }))
    monkeypatch.setitem(globals(), "ENGINE", engine)
    # The real routes.py discovery assertion passes, and this setup starts green.
    test_registered_switches_do_not_collide_with_parser_namespaces()
    (package / "new_parser.py").write_text(
        'PREFIX = "STATEFRAME_NEW_"\n'
        'def parse(environ):\n'
        '    result = {}\n'
        '    for name, value in environ.items():\n'
        '        if name.startswith(PREFIX):\n'
        f'            result[{tail}] = value\n'
        '    return result\n')
    with pytest.raises(AssertionError, match="STATEFRAME_NEW_"):
        test_registered_switches_do_not_collide_with_parser_namespaces()


@pytest.mark.parametrize("scope", ["new_parser", "install_environment"])
def test_namespace_review_exemptions_are_scoped_and_counted(tmp_path, scope):
    path = tmp_path / "_child.py"
    filter_source = (
        'def install_environment(environ):\n'
        '    filtered = [name for name in environ if name.startswith("STATEFRAME_TASK_")]\n')
    path.write_text(filter_source)
    _assert_namespace_discovery_is_complete(tmp_path, {})
    parser_header = '\ndef new_parser(environ):\n' if scope == "new_parser" else ''
    path.write_text(filter_source + parser_header +
                    '    for name in environ:\n'
                    '        if name.startswith("STATEFRAME_TASK_"):\n'
                    '            tail = name[16:]\n')
    with pytest.raises(AssertionError, match="STATEFRAME_TASK_"):
        _assert_namespace_discovery_is_complete(tmp_path, {})


@pytest.mark.parametrize("damage", ["source_changed", "switches_edited", "digest_edited", "schema", "garbage",
                                    "missing"])
def test_a_stale_or_damaged_registry_falls_back_to_the_scan(tmp_path, damage):
    root = _toy_root(tmp_path)
    registry.write_registry(root)
    path = root / registry.REGISTRY_FILE
    document = json.loads(path.read_text())
    if damage == "source_changed":
        (root / "b.py").write_text('import os\nos.environ.get("STATEFRAME_GAMMA")\n')
    elif damage == "switches_edited":
        document["switches"] = ["STATEFRAME_ALPHA", "STATEFRAME_BETA", "STATEFRAME_FORGED"]
        path.write_text(json.dumps(document))
    elif damage == "digest_edited":
        document["digest"] = "sha256:" + "0" * 64
        path.write_text(json.dumps(document))
    elif damage == "schema":
        document["schema"] = "something-else/1"
        path.write_text(json.dumps(document))
    elif damage == "garbage":
        path.write_text("{not json")
    else:
        path.unlink()
    registry._release_registry.cache_clear()
    switches, digest = registry.release_stateframe_switches(root)
    expected = {"STATEFRAME_ALPHA", "STATEFRAME_GAMMA"} if damage == "source_changed" else {
        "STATEFRAME_ALPHA", "STATEFRAME_BETA"}
    assert switches == frozenset(expected)
    assert "STATEFRAME_FORGED" not in switches
    assert digest == registry.build_registry_document(root)["digest"]


@pytest.mark.parametrize("scope,count", [("unreviewed", 1), ("resolve_opt_in_switches", 2)])
def test_parallel_filter_review_rejects_new_scope_or_extra_occurrence(tmp_path, scope, count):
    literals = ", ".join(repr("STATEFRAME_PARALLEL_") for _ in range(count))
    (tmp_path / "runner.py").write_text(f"def {scope}():\n    return ({literals},)\n")
    with pytest.raises(AssertionError, match="unreviewed environment namespaces"):
        _assert_namespace_discovery_is_complete(tmp_path, {})


@pytest.mark.parametrize("extra_scope", [False, True])
def test_parallel_error_label_review_does_not_hide_parser_use(tmp_path, extra_scope):
    engine = tmp_path / "engine"
    engine.mkdir()
    path = engine / "kernel_parallel_records.py"
    source = ("class ParallelLimits:\n"
              "    def from_settings(self, switch):\n"
              "        label = switch.removeprefix('STATEFRAME_PARALLEL_')\n"
              "        raise ValueError(label)\n")
    path.write_text(source)
    assert _parsed_environment_prefixes(tmp_path) == {}
    addition = ("\ndef parse_setting(switch):\n"
                "    return switch.removeprefix('STATEFRAME_PARALLEL_')\n"
                if extra_scope else
                "        tail = switch.removeprefix('STATEFRAME_PARALLEL_')\n")
    path.write_text(source + addition)
    assert _parsed_environment_prefixes(tmp_path) == {
        "STATEFRAME_PARALLEL_": {"engine/kernel_parallel_records.py"}}
