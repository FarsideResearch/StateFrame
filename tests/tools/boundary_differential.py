"""Compare every compiler-test invocation with the pinned release, including None.

The frozen SHA-256 projections keep the same assertions running without Git.
They cover all checks, including advisory checks, rather than filtering gates.
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

BASE = "56af0dc6d6"
ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests/fixtures/boundary_base_projections.json"
LIMITS = ("MAX_FILES", "MAX_SOURCE_BYTES", "MAX_TEST_BYTES", "MAX_CELLS", "MAX_CHECKS")


def load_base():
    if os.environ.get("SF_BOUNDARY_NO_GIT") == "1":
        return None
    try:
        result = subprocess.run(
            ["git", "show", f"{BASE}:src/stateframe/engine/kernel_boundary_checks.py"],
            cwd=ROOT, capture_output=True, text=True, timeout=30, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    module = ModuleType("_boundary_base_differential")
    sys.modules[module.__name__] = module
    exec(compile(result.stdout, f"{BASE}:kernel_boundary_checks.py", "exec"), module.__dict__)  # noqa: S102 - pinned source
    return module


def base_runtime_methods():
    if os.environ.get("SF_BOUNDARY_NO_GIT") == "1":
        return None
    try:
        source = subprocess.run(
            ["git", "show", f"{BASE}:src/stateframe/engine/kernel_stateframe_runtime.py"],
            cwd=ROOT, capture_output=True, text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    import kernel_stateframe_runtime as runtime
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "StateFrameRuntime")
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
               and node.name in {"_host_check_bundle", "_starter_conflict_bundle"}]
    namespace = dict(vars(runtime))
    exec(compile(ast.Module(body=methods, type_ignores=[]), "base-runtime", "exec"), namespace)  # noqa: S102 - pinned source
    return {node.name: namespace[node.name] for node in methods}


def projection(bundle):
    # Include bundle-level admission certificates as well as every definition,
    # cell, spec, provenance field and the distinction between None/empty bundles.
    return dataclasses.asdict(bundle) if bundle is not None else None


def fingerprint(value):
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def canonical(value):
    """Preserve Python literal types, including tuple/list and integer/string keys."""
    if isinstance(value, dict):
        items = [[canonical(key), canonical(item)] for key, item in value.items()]
        return ["dict", sorted(items, key=lambda item: json.dumps(item[0]))]
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [canonical(item) for item in value]]
    if value is None or type(value) in (str, int, bool, float):
        return [type(value).__name__, value]
    raise TypeError(f"unrepresented differential input: {type(value).__name__}")


def normalize_paths(value, root):
    if isinstance(value, Path):
        return {"Path": str(value).replace(str(root), "<root>") if root else str(value)}
    if isinstance(value, dict):
        return {key: normalize_paths(item, root) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        values = [normalize_paths(item, root) for item in value]
        return tuple(values) if isinstance(value, tuple) else values
    return value


@pytest.fixture(scope="module")
def boundary_base():
    return load_base()


@pytest.fixture(autouse=True)
def boundary_differential(request, monkeypatch, boundary_base):
    """Apply the differential oracle to every shape in the compiler test file."""
    from stateframe.engine import kernel_boundary_checks as current

    original = current.compile_boundary_bundle
    golden = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
    if boundary_base is None:
        assert golden["_base"] == BASE
    capture_path = os.environ.get("SF_BOUNDARY_CAPTURE")
    captured = {}

    def compare(*args, **kwargs):
        # Root directories differ between runs; their presence and brief-file
        # contents are inputs, while their temporary absolute spelling is not.
        normalized = dict(kwargs)
        root = normalized.pop("root", None)
        brief_files = {}
        if root is not None:
            brief_files = {str(path.relative_to(root)): path.read_text()
                           for path in sorted(Path(root).rglob("*.md"))}
        switches = {key: value for key, value in os.environ.items()
                    if key.startswith("STATEFRAME_") and key != "STATEFRAME_HOME"}
        limits = {key: getattr(current, key) for key in LIMITS}
        key = fingerprint([normalize_paths(args, root), normalize_paths(normalized, root),
                           root is not None, brief_files, switches, limits])
        expected = None
        raised = False
        if boundary_base is not None:
            for name, value in limits.items():
                setattr(boundary_base, name, value)
            try:
                expected = projection(boundary_base.compile_boundary_bundle(*args, **kwargs))
            except (OSError, TypeError, ValueError, SyntaxError):
                # Crash handling is tested separately through the runtime's
                # static fallback. It cannot supply a new dynamic bundle.
                expected = None
                raised = True
            captured[key] = {"projection": fingerprint(expected), "raises": raised}
        reference = captured[key] if boundary_base is not None else golden[key]
        try:
            actual = original(*args, **kwargs)
        except (OSError, TypeError, ValueError, SyntaxError):
            assert reference["raises"], "HEAD raised on base-accepted material"
            raise
        actual_projection = projection(actual)
        assert fingerprint(actual_projection) == reference["projection"], (request.node.nodeid, key)
        return actual

    monkeypatch.setattr(current, "compile_boundary_bundle", compare)
    yield
    if capture_path:
        path = Path(capture_path)
        previous = json.loads(path.read_text()) if path.exists() else {}
        previous.update(captured)
        path.write_text(json.dumps(previous, sort_keys=True, indent=2) + "\n")
