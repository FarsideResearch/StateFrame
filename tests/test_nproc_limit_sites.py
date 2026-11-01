"""Prevent another uid-wide process limit from bypassing the relative policy."""
import ast
from pathlib import Path

import pytest

import kernel_execution_policy as policy


SOURCE = Path(__file__).resolve().parents[1] / "src"
ENGINE = SOURCE / "stateframe" / "engine"
ALLOWED = (ENGINE / "kernel_execution_policy.py", "_set_relative_nproc_limit")


def _nproc_sites(path: Path, source: str):
    if "RLIMIT_NPROC" not in source:
        return
    tree = ast.parse(source, filename=str(path))
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or node.attr != "RLIMIT_NPROC":
            continue
        owner = node
        while owner in parents and not isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef)):
            owner = parents[owner]
        yield path, getattr(owner, "name", None)


def _assert_single_site(sources):
    sites = [site for path, source in sources for site in _nproc_sites(path, source)]
    assert sites == [ALLOWED, ALLOWED], sites


def test_only_relative_helper_sets_nproc():
    _assert_single_site((path, path.read_text()) for path in sorted(SOURCE.rglob("*.py")))


def test_planted_absolute_limit_is_caught():
    sources = [(path, path.read_text()) for path in sorted(SOURCE.rglob("*.py"))]
    sources.append((ENGINE / "planted_violation.py",
                    "def launch():\n    resource.setrlimit(resource.RLIMIT_NPROC, (256, 256))\n"))
    with pytest.raises(AssertionError, match="planted_violation"):
        _assert_single_site(sources)


def test_planted_launcher_list_limit_is_caught():
    sources = [(path, path.read_text()) for path in sorted(SOURCE.rglob("*.py"))]
    sources.append((ENGINE / "planted_violation.py",
                    "def launch():\n    limits.append((resource.RLIMIT_NPROC, 256))\n"))
    with pytest.raises(AssertionError, match="planted_violation"):
        _assert_single_site(sources)


def test_outer_limit_uses_task_count_and_headroom(monkeypatch):
    calls = []
    monkeypatch.setattr(policy.platform, "system", lambda: "Linux")
    monkeypatch.setattr(policy, "_user_task_count", lambda: 400)
    monkeypatch.setattr(policy.resource, "getrlimit",
                        lambda kind: (policy.resource.RLIM_INFINITY, policy.resource.RLIM_INFINITY))
    monkeypatch.setattr(policy.resource, "setrlimit", lambda kind, limits: calls.append((kind, limits)))
    policy._resource_preexec(30, 512 * 1024 * 1024, 1024 * 1024, 128, 64)()
    assert (policy.resource.RLIMIT_NPROC, (592, 592)) in calls
