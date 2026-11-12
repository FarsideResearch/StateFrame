"""Safe home discovery and static guard for home-sensitive path calls."""
from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import subprocess
import sys

import pytest

from _guard_scan_cache import source_tree_signature
from _timeouts import scaled

import kernel_safe_paths as safe_paths

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "stateframe"

# No reviewed exceptions: build_parser runs for every command, so a home lookup there crashed every invocation
# without HOME (the cli config-init default, removed 11-04). Add an entry only for code no normal run reaches.
ALLOWLIST: dict[tuple[str, str, str], str] = {}


@dataclass(frozen=True)
class _SourceTreeScan:
    scanned_files: int
    home_calls: tuple[tuple[str, str, str, int], ...]
    passwd_uid_calls: tuple[tuple[str, str, str], ...]


def _function_name(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current.name
        current = parents.get(current)
    return "<module>"


def _path_home_calls(tree: ast.AST, relative_path: str) -> list[tuple[str, str, str, int]]:
    nodes = list(ast.walk(tree))
    parents = {child: parent for parent in nodes for child in ast.iter_child_nodes(parent)}
    home_calls, _ = _source_call_sites(nodes, parents, relative_path)
    return home_calls


def _source_call_sites(
    nodes: list[ast.AST], parents: dict[ast.AST, ast.AST], relative_path: str,
) -> tuple[list[tuple[str, str, str, int]], list[tuple[str, str, str]]]:
    home_calls = []
    passwd_uid_calls = []
    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"home", "expanduser"}:
            home_calls.append((relative_path, _function_name(node, parents), ast.unparse(node.func), node.lineno))
        if (isinstance(node.func, ast.Attribute) and node.func.attr == "getpwuid"
                or isinstance(node.func, ast.Name) and node.func.id == "getpwuid"):
            passwd_uid_calls.append((relative_path, _function_name(node, parents), ast.unparse(node.func)))
    return home_calls, passwd_uid_calls


@lru_cache(maxsize=2)
def _cached_source_tree_scan(
    _signature: tuple[tuple[str, int, int], ...],
) -> _SourceTreeScan:
    home_calls = []
    passwd_uid_calls = []
    paths = sorted(SOURCE.rglob("*.py"))
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        nodes = list(ast.walk(tree))
        parents = {child: parent for parent in nodes for child in ast.iter_child_nodes(parent)}
        file_home_calls, file_passwd_uid_calls = _source_call_sites(nodes, parents, relative)
        home_calls.extend(file_home_calls)
        passwd_uid_calls.extend(file_passwd_uid_calls)
    return _SourceTreeScan(len(paths), tuple(home_calls), tuple(passwd_uid_calls))


def _source_tree_scan() -> _SourceTreeScan:
    return _cached_source_tree_scan(source_tree_signature(SOURCE))


def _site_key(site: tuple[str, str, str, int]) -> tuple[str, str, str]:
    return site[:3]


def _violations(
    sites: list[tuple[str, str, str, int]],
    allowed: Mapping[tuple[str, str, str], str],
) -> tuple[list[tuple[str, str, str, int]], list[tuple[str, str, str]]]:
    expected = Counter({key: 1 for key in allowed})
    found = Counter(_site_key(site) for site in sites)
    violations = [site for site in sites
                  if _site_key(site) not in expected or found[_site_key(site)] > expected[_site_key(site)]]
    missing = list((expected - found).elements())
    return violations, missing


def test_safe_home_uses_supplied_home_and_empty_home_means_none(monkeypatch):
    monkeypatch.setenv("HOME", "/ambient-home")
    if safe_paths.pwd is not None:
        monkeypatch.setattr(safe_paths.pwd, "getpwuid", lambda _uid: pytest.fail("unexpected passwd fallback"))

    assert safe_paths.safe_home() == Path("/ambient-home")
    assert safe_paths.safe_home({"HOME": "/configured-home"}) == Path("/configured-home")
    assert safe_paths.safe_home({"HOME": ""}) is None


@pytest.mark.parametrize("error", [RuntimeError("runtime"), KeyError("missing"), OSError("unavailable")])
def test_safe_home_never_raises_when_passwd_lookup_fails(monkeypatch, error):
    monkeypatch.delenv("HOME", raising=False)
    if safe_paths.pwd is None:
        pytest.skip("platform has no passwd database")

    def fail(_uid):
        raise error

    monkeypatch.setattr(safe_paths.pwd, "getpwuid", fail)
    assert safe_paths.safe_home() is None


def test_safe_expanduser_uses_the_supplied_environment():
    assert safe_paths.safe_expanduser(Path("~/config.toml"), {"HOME": "/configured-home"}) == (
        Path("/configured-home/config.toml"))


def test_flat_engine_child_and_skills_import_shared_home_helper(tmp_path):
    package_root = SOURCE
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(package_root)!r})\n"
        "import _child, skills\n"
        "assert _child.safe_home is skills.safe_home\n"
    )
    child_env = dict(os.environ)
    child_env.pop("PYTHONPATH", None)
    child_env.pop("PYTHONHOME", None)
    result = subprocess.run([sys.executable, "-I", "-c", script], cwd=tmp_path, env=child_env,
                            capture_output=True, text=True, timeout=scaled(30), check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_home_call_site_scan_has_only_reviewed_calls(capsys):
    scan = _source_tree_scan()
    sites = list(scan.home_calls)
    scanned_files = scan.scanned_files
    violations, missing = _violations(sites, ALLOWLIST)
    violated_count = len(violations) + len(missing)
    with capsys.disabled():
        print(f"expected={len(ALLOWLIST)} files={scanned_files} sites={len(sites)} violated={violated_count}")
    # Zero sites is the goal now; non-vacuity is the file count here plus the planted-violation test below.
    assert scanned_files > 100, f"home-call scan read only {scanned_files} source files; is SOURCE right?"
    assert not any("\n" in reason for reason in ALLOWLIST.values()), "allowlist reasons must stay one line"
    assert violated_count == 0, f"unreviewed home calls: {violations}; missing allowlisted calls: {missing}"


def test_home_call_scan_detects_a_planted_violation():
    planted = ast.parse(
        "from pathlib import Path as P\n"
        "import pathlib as p\n"
        "def candidate():\n"
        "    return P.home(), p.Path.home(), Path('/tmp').expanduser()\n"
    )
    sites = _path_home_calls(planted, "src/stateframe/planted.py")
    violations, missing = _violations(sites, {})
    assert len(sites) == 3
    assert len(violations) == 3
    assert not missing


def test_passwd_uid_lookup_is_confined_to_safe_home():
    sites = list(_source_tree_scan().passwd_uid_calls)
    assert sites == [("src/stateframe/engine/kernel_safe_paths.py", "safe_home", "pwd.getpwuid")]
