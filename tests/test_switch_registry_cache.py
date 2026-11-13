"""The engine switch registry ships precomputed; a run hashes the sources instead of AST-parsing them.

AST-parsing every declared engine file took 22-24 s per scan on a loaded host and ran twice per run
(host and engine child), which pushed first-request waits past their limits.
"""
from __future__ import annotations

import ast
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

from stateframe import routes

ENGINE = Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"
sys.path.insert(0, str(ENGINE))
import kernel_release_switch_registry as registry  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_cache():
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


def test_shipped_registry_fingerprint_matches_the_real_engine_sources():
    shipped = json.loads((ENGINE / registry.REGISTRY_FILE).read_text())
    files = registry._declared_python_files(ENGINE)
    assert shipped["sources_sha256"] == registry._sources_fingerprint(ENGINE, files)


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
    prefixes: dict[str, set[str]] = {}
    for path in package.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if "STATEFRAME_" not in source or "removeprefix" not in source:
            continue
        tree = ast.parse(source, filename=str(path))
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
                prefixes.setdefault(prefix, set()).add(path.relative_to(package).as_posix())
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
