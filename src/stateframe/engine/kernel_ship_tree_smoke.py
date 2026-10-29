"""Ship-tree import/compile smoke for touched Python modules (rulebook N14).

Inert until ``STATEFRAME_SHIP_TREE_SMOKE=on``.  Scope of this lane: Python.

* Every module the run TOUCHED (host-applied edit receipts) plus every declared
  Python entrypoint is compiled and imported in a FRESH isolated interpreter
  from the candidate tree (never the long-lived runner process).
* An AST pass finds function-local imports that shadow a module-level name
  used earlier in the same function (the #167 UnboundLocalError class, which
  compiles and imports cleanly).
* A failure is a completion-blocking OBLIGATION: it is rendered in the frame
  and refuses ``done``; it never removes tools or stops the request, so the
  model can always read, edit and re-check.
* Baseline-relative: an import-time failure that is identical (path +
  normalized message) at the untouched parent revision is an ENVIRONMENT fact
  -- rendered as such, never an edit request and never a ``done`` blocker.
  Only NEW failures block.  Compile errors (SyntaxError family) and the #167
  local-import shadow findings are code defects and stay FAIL even when they
  are present at the baseline.  (Import-resolution errors are the motivating
  case; the class is not narrowed further because once the resolver finds the
  package, an unconfigured parent fails the same way one step later -- e.g. a
  settings ValidationError for a missing DATABASE_URL -- which is equally not
  the model's obligation.)
* The child resolves imports from ``source_roots``: the ancestor directories
  that actually contain each checked file's top-level import names, plus any
  pytest ``pythonpath`` declared beside them (the ``backend/app`` layout).
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from kernel_non_product_paths import is_non_product
from kernel_test_python import test_python


MODE_ENV: Final = "STATEFRAME_SHIP_TREE_SMOKE"
SCHEMA: Final = "rulebook-ship-tree-smoke/2"
BLOCK_HEADER: Final = "[SHIP-TREE SMOKE]"
DETAIL_ENV: Final = "STATEFRAME_TOKENLAB_SHIP_TREE_DETAIL"
DETAIL_SCHEMA: Final = "stateframe-v19-ship-tree-detail/1"
DETAIL_EVENT: Final = "tokenlab_v19_ship_tree_target"


def mode(environ: Mapping[str, str] | None = None) -> str:
    values = os.environ if environ is None else environ
    value = str(values.get(MODE_ENV, "off")).strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


@dataclass(frozen=True)
class ShipTreeSmokeResult:
    status: str                      # PASS | FAIL | ENVIRONMENT | UNKNOWN (nothing to check)
    checked: tuple[str, ...]
    failures: tuple[tuple[str, str], ...]   # (path, error)
    tree_digest: str = ""
    # Failures identical to the recorded baseline (never blocking, never an obligation).
    environment: tuple[tuple[str, str], ...] = ()
    roots: tuple[str, ...] = ()
    dependency_identity: str = ""
    environment_identity: str = ""
    declared_scope: tuple[str, ...] = ()
    resolver_identity: str = ""

    @property
    def blocking(self) -> bool:
        return self.status == "FAIL"

    def as_dict(self) -> dict[str, object]:
        return {"payload_schema": SCHEMA, "status": self.status,
                "checked": list(self.checked),
                "failures": [{"path": path, "error": error} for path, error in self.failures],
                "environment": [{"path": path, "error": error}
                                for path, error in self.environment],
                "baseline_matched": len(self.environment),
                "roots": list(self.roots),
                "tree_digest": self.tree_digest,
                "dependency_identity": self.dependency_identity,
                "environment_identity": self.environment_identity,
                "declared_scope": list(self.declared_scope),
                "resolver_identity": self.resolver_identity}


_CHILD = r'''
import importlib, importlib.util, json, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
items = json.loads(sys.argv[2])
roots = [pathlib.Path(value).resolve() for value in
         (json.loads(sys.argv[3]) if len(sys.argv) > 3 else [str(root)])]
sys.path[:0] = [str(value) for value in roots]
rows = []
def module_name(path):
    # Name the module from the deepest source root that contains it, so a
    # file under backend/app is imported once, as app.*, not as backend.app.*.
    base = root
    for candidate in roots:
        try:
            path.relative_to(candidate)
        except ValueError:
            continue
        if len(candidate.parts) > len(base.parts):
            base = candidate
    parts = list(path.relative_to(base).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) if parts and all(p.isidentifier() for p in parts) else None
for raw in items:
    path = root / raw
    try:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
        name = module_name(path)
        by_file = not name
        if name:
            try:
                importlib.import_module(name)
            except ModuleNotFoundError as missing:
                # The dotted name itself is unresolvable (no package marker, or a
                # same-named site package): that is a harness fact, not a defect
                # of the file.  Load the FILE instead; a genuinely missing
                # dependency still fails there.
                if not missing.name or not (name == missing.name
                                            or name.startswith(missing.name + ".")):
                    raise
                by_file = True
        if by_file:
            spec = importlib.util.spec_from_file_location("_ship_tree_smoke", str(path))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        rows.append({"path": raw, "status": "PASS"})
    except BaseException as error:
        rows.append({"path": raw, "status": "FAIL",
                     "error": type(error).__name__ + ": " + str(error)[:240]})
print(json.dumps(rows, sort_keys=True))
'''


def python_paths(root: Path, values: Sequence[object]) -> tuple[str, ...]:
    """Existing workspace-relative ``.py`` files among ``values`` (deduplicated)."""
    root = Path(root).resolve()
    result: list[str] = []
    for value in values:
        raw = str(value or "").strip()
        if not raw.endswith(".py"):
            continue
        candidate = Path(raw) if Path(raw).is_absolute() else root / raw
        try:
            relative = candidate.resolve().relative_to(root).as_posix()
        except (OSError, ValueError):
            continue
        if is_non_product(relative):
            continue
        if (root / relative).is_file():
            result.append(relative)
    return tuple(dict.fromkeys(result))


_INI_SECTIONS: Final = {"pytest.ini": ("pytest",), "tox.ini": ("pytest",),
                         "setup.cfg": ("tool:pytest",)}


def _top_level_import_names(source: str) -> tuple[str, ...]:
    """Top package names of the file's absolute module-level imports."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return ()
    names: list[str] = []
    stack: list[ast.AST] = list(tree.body)
    while stack:
        node = stack.pop(0)
        if isinstance(node, ast.Import):
            names.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            names.append(node.module.split(".")[0])
        elif isinstance(node, (ast.If, ast.Try, ast.With)):
            # Module-level conditional/guarded imports run at import time too.
            for field in ("body", "orelse", "finalbody", "handlers"):
                for child in getattr(node, field, ()) or ():
                    if isinstance(child, ast.ExceptHandler):
                        stack.extend(child.body)
                    else:
                        stack.append(child)
    stdlib = getattr(sys, "stdlib_module_names", frozenset())
    return tuple(dict.fromkeys(name for name in names
                               if name and name not in stdlib and name != "__future__"))


def _provides(directory: Path, name: str) -> bool:
    """``directory`` makes ``name`` importable (module file or package dir)."""
    if (directory / f"{name}.py").is_file():
        return True
    package = directory / name
    return package.is_dir() and ((package / "__init__.py").is_file()
                                 or any(package.glob("*.py")))


def _ini_pythonpath(directory: Path) -> tuple[Path, ...]:
    """``pythonpath`` entries declared by pytest config files in ``directory``."""
    import configparser
    found: list[Path] = []
    for name, sections in _INI_SECTIONS.items():
        ini = directory / name
        if not ini.is_file():
            continue
        parser = configparser.ConfigParser(interpolation=None)
        try:
            parser.read(ini, encoding="utf-8")
        except (configparser.Error, OSError, UnicodeError):
            continue
        for section in sections:
            if parser.has_option(section, "pythonpath"):
                found.extend(directory / item
                             for item in parser.get(section, "pythonpath").split())
    pyproject = directory / "pyproject.toml"
    if pyproject.is_file():
        try:
            import tomllib
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except (ImportError, OSError, UnicodeError, ValueError):
            data = {}
        tool = data.get("tool", {}) if isinstance(data, Mapping) else {}
        pytest_cfg = tool.get("pytest", {}) if isinstance(tool, Mapping) else {}
        options = pytest_cfg.get("ini_options", {}) if isinstance(pytest_cfg, Mapping) else {}
        value = options.get("pythonpath") if isinstance(options, Mapping) else None
        items = value.split() if isinstance(value, str) else value
        if isinstance(items, list):
            found.extend(directory / str(item) for item in items if str(item).strip())
    return tuple(found)


def source_roots(root: Path, paths: Sequence[str]) -> tuple[str, ...]:
    """Import roots for the checked files (host side, pure; repo root first).

    Python resolver: for every ancestor directory A between ``root`` and a
    checked ``.py`` file, A is a root when it provides one of the file's
    top-level import names (``A/<top>.py`` or package ``A/<top>/``), and every
    pytest ``pythonpath`` entry declared in A (relative to A) is a root.  A top
    name no ancestor provides is resolved from the checkout's first-level
    directories only when exactly ONE of them provides it (monorepo siblings:
    ``book-worker/`` importing ``app`` from ``backend/``); ``PYTHONPATH``
    entries inside the checkout (the run's own import path) come last.
    Stdlib names never select a root (no shadowing) and nothing outside the
    checkout is ever admitted.  Non-Python paths are ignored: the smoke checks
    only Python, and other languages never consult a module search path.
    """
    root = Path(root).resolve()
    result: list[Path] = [root]
    for raw in paths:
        value = str(raw or "").strip()
        if not value.endswith(".py"):
            continue
        file = (Path(value) if Path(value).is_absolute() else root / value).resolve()
        try:
            relative = file.relative_to(root)
        except ValueError:
            continue
        try:
            tops = _top_level_import_names(file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError):
            tops = ()
        ancestors = [root.joinpath(*relative.parts[:index])
                     for index in range(0, len(relative.parts))]
        resolved_tops: set[str] = set()
        for directory in ancestors:
            provided = {top for top in tops if _provides(directory, top)}
            if provided:
                result.append(directory)
                resolved_tops.update(provided)
            for entry in _ini_pythonpath(directory):
                try:
                    resolved = entry.resolve()
                    resolved.relative_to(root)
                except (OSError, ValueError):
                    continue        # never import from outside the checkout
                if resolved.is_dir():
                    result.append(resolved)
        for top in tops:
            if top in resolved_tops or any(_provides(Path(item), top) for item in result):
                continue
            providers = [child for child in _first_level_dirs(root) if _provides(child, top)]
            if len(providers) == 1:                 # unambiguous sibling project only
                result.append(providers[0])
    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if not entry.strip():
            continue
        candidate = Path(entry) if Path(entry).is_absolute() else root / entry
        try:
            resolved = candidate.resolve()
            resolved.relative_to(root)
        except (OSError, ValueError):
            continue
        if resolved.is_dir():
            result.append(resolved)
    return tuple(dict.fromkeys(str(item) for item in result))


def _first_level_dirs(root: Path) -> tuple[Path, ...]:
    try:
        return tuple(sorted(child for child in root.iterdir()
                            if child.is_dir() and not child.name.startswith(".")))
    except OSError:
        return ()


# Code defects: never reclassified, even when already present at the baseline.
_CODE_DEFECT_CLASSES: Final = frozenset({"SyntaxError", "IndentationError", "TabError"})
_SHADOW_MARKER: Final = "(UnboundLocalError at call time)"
_ADDRESS = re.compile(r"0x[0-9a-fA-F]+")
_ABS_PATH = re.compile(r"(?:[A-Za-z]:)?(?:/[^\s'\"():,]+)+")


def normalize_error(error: str) -> str:
    """Comparable form of one failure: no absolute paths, addresses or spacing."""
    text = _ADDRESS.sub("0x?", str(error))
    text = _ABS_PATH.sub("<path>", text)
    return " ".join(text.split())


def _environment_class(error: str) -> bool:
    """Eligible for ENVIRONMENT when baseline-identical (everything but code defects)."""
    text = str(error)
    return (text.split(":", 1)[0].strip() not in _CODE_DEFECT_CLASSES
            and _SHADOW_MARKER not in text)


def apply_baseline(result: "ShipTreeSmokeResult",
                   baseline: "ShipTreeSmokeResult | None") -> "ShipTreeSmokeResult":
    """Reclassify baseline-identical import failures as ENVIRONMENT.

    A failure moves only when (path, normalized error) also failed at the
    baseline AND it is not a code defect.  SyntaxError-family failures and the
    local-import shadow findings stay FAIL (N14), and a NEW failure (different
    path or message, e.g. an import an edit broke) stays FAIL.
    """
    if baseline is None or result.status == "UNKNOWN":
        return result
    if (baseline.dependency_identity != result.dependency_identity
            or baseline.environment_identity != result.environment_identity
            or baseline.resolver_identity != result.resolver_identity):
        # Dependency or execution-environment changes invalidate the baseline;
        # candidate source identity is intentionally independent from this key.
        return result
    known = {(path, normalize_error(error))
             for path, error in (*baseline.failures, *baseline.environment)}
    baseline_scope = set(baseline.declared_scope or baseline.checked)
    failures: list[tuple[str, str]] = []
    environment: list[tuple[str, str]] = list(result.environment)
    for path, error in result.failures:
        if (path in baseline_scope and _environment_class(error)
                and (path, normalize_error(error)) in known):
            environment.append((path, error))
        else:
            failures.append((path, error))
    status = "FAIL" if failures else ("ENVIRONMENT" if environment else "PASS")
    return ShipTreeSmokeResult(status, result.checked, tuple(failures), result.tree_digest,
                               tuple(environment), result.roots,
                               result.dependency_identity, result.environment_identity,
                               result.declared_scope, result.resolver_identity)


def tree_digest(root: Path, paths: Sequence[str], *, dependency_identity: object = "",
                environment_identity: object = "", resolver_identity: object = "") -> str:
    digest = hashlib.sha256()
    for label, value in (("dependencies", dependency_identity),
                         ("environment", environment_identity),
                         ("resolver", resolver_identity)):
        digest.update(label.encode("utf-8"))
        digest.update(json.dumps(value, sort_keys=True, default=str,
                                 separators=(",", ":")).encode("utf-8"))
    for path in sorted(str(item) for item in paths if not is_non_product(str(item))):
        digest.update(path.encode("utf-8"))
        try:
            digest.update((Path(root) / path).read_bytes())
        except OSError:
            digest.update(b"<missing>")
    return "sha256:" + digest.hexdigest()


def local_import_shadows(source: str) -> tuple[str, ...]:
    """#167: a function-local import of a module-level name, used before it.

    Python makes the name local for the WHOLE function body, so the earlier use
    raises UnboundLocalError at call time even though the file compiles and
    imports.  Returns human-readable findings (empty when clean/unparseable).
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return ()
    module_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module_names.update((alias.asname or alias.name).split(".")[0]
                                for alias in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            module_names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            module_names.update(t.id for t in targets if isinstance(t, ast.Name))
    findings: list[str] = []

    def own_nodes(function: ast.AST):
        # Walk the function body without descending into nested scopes.
        stack = list(ast.iter_child_nodes(function))
        while stack:
            node = stack.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                                 ast.Lambda)):
                continue
            yield node
            stack.extend(ast.iter_child_nodes(node))

    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        nodes = list(own_nodes(function))
        local_imports: dict[str, int] = {}
        for node in nodes:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    name = (alias.asname or alias.name).split(".")[0]
                    if name in module_names:
                        local_imports.setdefault(name, node.lineno)
        for node in nodes:
            if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                    and node.id in local_imports and node.lineno < local_imports[node.id]):
                findings.append(
                    f"{function.name}: `{node.id}` used at line {node.lineno} but imported "
                    f"locally at line {local_imports[node.id]} (UnboundLocalError at call time)")
                local_imports.pop(node.id)
    return tuple(findings)


def run_smoke(root: Path, paths: Sequence[str], *, timeout_s: float = 20.0,
              baseline: ShipTreeSmokeResult | None = None,
              roots: Sequence[str] | None = None,
              dependency_identity: object = "",
              environment_identity: object = "",
              declared_scope: Sequence[str] | None = None,
              resolver_identity: object = "") -> ShipTreeSmokeResult:
    """Fresh-interpreter compile+import plus the local-import shadow pass.

    ``baseline`` (the result recorded at the untouched parent) turns
    baseline-identical import failures into ENVIRONMENT; ``roots`` overrides
    the resolver (``source_roots``) -- tests pass ``(root,)`` to disable it.
    """
    root = Path(root).resolve()
    checked = python_paths(root, paths)
    scope = python_paths(root, declared_scope if declared_scope is not None else paths)
    if not checked:
        return ShipTreeSmokeResult("UNKNOWN", (), (), "", declared_scope=scope)
    if roots is not None:
        import_roots = tuple(str(Path(item).resolve()) for item in roots)
    else:
        import_roots = source_roots(root, checked)
    effective_dependencies = dependency_identity or {"roots": import_roots}
    effective_environment = environment_identity or {
        "executable": test_python(),
        "implementation": getattr(sys.implementation, "name", "python"),
        "version": tuple(sys.version_info[:3]),
    }
    effective_resolver = resolver_identity or {"roots": import_roots, "checked": checked}
    digest = tree_digest(root, checked, dependency_identity=effective_dependencies,
                         environment_identity=effective_environment,
                         resolver_identity=effective_resolver)
    dependency_key = json.dumps(effective_dependencies, sort_keys=True, default=str)
    environment_key = json.dumps(effective_environment, sort_keys=True, default=str)
    resolver_key = json.dumps(effective_resolver, sort_keys=True, default=str)
    failures: list[tuple[str, str]] = []
    host_failures: list[tuple[str, str]] = []
    runner_failed = False
    try:
        completed = subprocess.run(
            [test_python(), "-I", "-c", _CHILD, str(root), json.dumps(list(checked)),
             json.dumps(list(import_roots))],
            cwd=str(root), capture_output=True, text=True,
            timeout=max(0.1, float(timeout_s)), check=False)
        rows = json.loads(completed.stdout or "[]")
        if completed.returncode != 0 and not rows:
            runner_failed = True
            host_failures = [(path, f"fresh interpreter exited {completed.returncode}")
                             for path in checked]
    except subprocess.TimeoutExpired:
        rows = []
        runner_failed = True
        host_failures = [(path, "fresh interpreter timed out") for path in checked]
    except (OSError, json.JSONDecodeError) as error:
        rows = []
        runner_failed = True
        host_failures = [(path, f"fresh interpreter failed: {type(error).__name__}")
                         for path in checked]
    seen = set()
    for row in rows if isinstance(rows, list) else ():
        if isinstance(row, Mapping):
            seen.add(str(row.get("path")))
            if row.get("status") == "FAIL":
                failures.append((str(row.get("path")), str(row.get("error", "import failed"))))
    if not runner_failed:
        host_failures.extend((path, "fresh interpreter produced no result") for path in checked
                              if path not in seen)
    static_findings: list[tuple[str, str]] = []
    for path in checked:
        try:
            source = (root / path).read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        static_findings.extend((path, finding) for finding in local_import_shadows(source))
    if runner_failed:
        # A timeout, malformed JSON, or failed runner is a host outcome. Keep
        # static observations visible as deferred diagnostics, but never turn
        # them into an automatic product FAIL on the same broken execution.
        host_failures.extend(
            (path, f"runner unavailable; static check deferred: {finding}")
            for path, finding in static_findings
        )
    else:
        failures.extend(static_findings)
    status = "FAIL" if failures else ("ENVIRONMENT" if host_failures else "PASS")
    raw = ShipTreeSmokeResult(status, checked, tuple(failures), digest,
                              tuple(host_failures), import_roots,
                              dependency_key, environment_key, scope, resolver_key)
    return apply_baseline(raw, baseline)


def render_smoke(result: ShipTreeSmokeResult) -> str:
    if result.status == "PASS":
        return f"{BLOCK_HEADER} PASS: {len(result.checked)} touched module(s) import in a fresh interpreter."
    if result.status == "UNKNOWN":
        return f"{BLOCK_HEADER} nothing to check yet (no touched Python module)."
    if result.status == "ENVIRONMENT":
        # Never an edit request: the failure predates the run (baseline-identical).
        lines = [f"{BLOCK_HEADER} ENVIRONMENT — present at the untouched parent revision; "
                 "not a task obligation:"]
        lines.extend(f"- {path}: {' '.join(str(error).split())[:200]} "
                     "(host import fact; identical before any edit)"
                     for path, error in result.environment[:6])
        lines.append("done is not blocked by a baseline-identical environment failure; no edit is "
                     "requested for it. Only NEW import failures introduced by an edit are "
                     "reported as FAIL.")
        return "\n".join(lines)
    lines = [f"{BLOCK_HEADER} FAIL — completion is blocked until these import cleanly:"]
    lines.extend(f"- {path}: {error}" for path, error in result.failures[:6])
    if os.environ.get(DETAIL_ENV, "off").strip().lower() == "on":
        targets = failure_targets(result)
        for target in targets[:3]:
            lines.append(
                f"- {target['target_file']} must define {target['missing_symbol']}, "
                f"imported by {target['importer_file']}")
            lines.append(f"  Use one `str_replace` target: {target['target_file']}")
    lines.append("Read the module, fix it with an edit, and the smoke re-runs automatically.")
    if result.environment:
        lines.append(f"({len(result.environment)} other failure(s) are identical at the untouched "
                     "parent revision: environment, not a task obligation.)")
    return "\n".join(lines)


_MISSING_IMPORT = re.compile(
    r"cannot import name [\"'](?P<symbol>[A-Za-z_]\w*)[\"']\s+from\s+[\"'](?P<module>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)[\"']",
    re.IGNORECASE,
)


def failure_targets(result: ShipTreeSmokeResult) -> tuple[dict[str, str], ...]:
    """Return precise edit targets for missing-symbol import failures."""
    targets: list[dict[str, str]] = []
    for importer, error in result.failures:
        match = _MISSING_IMPORT.search(str(error))
        if match is None:
            continue
        module = str(match.group("module"))
        target_file = module.replace(".", "/") + ".py"
        row = {
            "target_file": target_file,
            "importer_file": str(importer),
            "missing_symbol": str(match.group("symbol")),
            "module": module,
            "schema": DETAIL_SCHEMA,
        }
        if row not in targets:
            targets.append(row)
    return tuple(targets)
