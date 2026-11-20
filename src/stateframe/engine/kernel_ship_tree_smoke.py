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
import json
import os
import re
import sys
import tempfile
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from functools import cache
from hashlib import sha256
from pathlib import Path
from threading import Thread
from typing import TYPE_CHECKING, Final

from kernel_non_product_paths import is_non_product
from kernel_test_python import selected_test_python
from kernel_safe_probe import ProbeCleanupError, probe_evidence, retry_probe, run_probe_session
from kernel_tree_digest import path_entry, tree_digest as canonical_tree_digest
if __package__:
    from .kernel_turn_work_counters import count_work
else:
    from kernel_turn_work_counters import count_work

if TYPE_CHECKING:
    from kernel_workspace_consumers import WorkspaceConsumers
    from kernel_workspace_state import WorkspaceSnapshot
    from kernel_workspace_tree import Directory


MODE_ENV: Final = "STATEFRAME_SHIP_TREE_SMOKE"
SCHEMA: Final = "rulebook-ship-tree-smoke/2"
TREE_DIGEST_FORMAT: Final = "stateframe.ship_tree_smoke/2"
BLOCK_HEADER: Final = "[SHIP-TREE SMOKE]"
DETAIL_ENV: Final = "STATEFRAME_TOKENLAB_SHIP_TREE_DETAIL"
DETAIL_SCHEMA: Final = "stateframe-v19-ship-tree-detail/1"
DETAIL_EVENT: Final = "tokenlab_v19_ship_tree_target"


class ImportSmokeCleanupError(ProbeCleanupError):
    """Automatic import execution did not establish process quiescence."""


def mode(environ: Mapping[str, str] | None = None) -> str:
    values = os.environ if environ is None else environ
    value = str(values.get(MODE_ENV, "off")).strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


@dataclass(frozen=True)
class ShipTreeSmokeResult:
    status: str
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
    digest_format: str = ""
    probe_evidence: dict[str, object] | None = None
    probe_complete: bool = field(default=False, repr=False, compare=False)

    @property
    def blocking(self) -> bool:
        return self.status in {"FAIL", "ENGINE_ERROR"}

    def as_dict(self) -> dict[str, object]:
        result = {"payload_schema": SCHEMA, "status": self.status,
                "checked": list(self.checked),
                "failures": [{"path": path, "error": error} for path, error in self.failures],
                "environment": [{"path": path, "error": error}
                                for path, error in self.environment],
                "baseline_matched": len(self.environment),
                "roots": list(self.roots),
                "tree_digest": self.tree_digest,
                "digest_format": self.digest_format,
                "dependency_identity": self.dependency_identity,
                "environment_identity": self.environment_identity,
                "declared_scope": list(self.declared_scope),
                "resolver_identity": self.resolver_identity}
        if self.probe_evidence is not None:
            result["probe_evidence"] = self.probe_evidence
        return result


_CHILD = r'''
import importlib, importlib.util, json, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
if sys.argv[2] == "--inputs":
    items, raw_roots = json.loads(pathlib.Path(sys.argv[3]).read_text(encoding="utf-8"))
else:
    items = json.loads(sys.argv[2])
    raw_roots = json.loads(sys.argv[3]) if len(sys.argv) > 3 else [str(root)]
roots = [pathlib.Path(value).resolve() for value in raw_roots]
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
print("\nSTATEFRAME_SMOKE_RESULT=" + json.dumps(rows, sort_keys=True))
'''


def _execute_smoke(root: Path, checked: Sequence[str], roots: Sequence[str], timeout: float):
    return retry_probe(lambda: _execute_smoke_once(root, checked, roots, timeout))


def _execute_smoke_once(root: Path, checked: Sequence[str], roots: Sequence[str], timeout: float):
    """Imports are arbitrary repository code, never host-side introspection.

    Keep both the candidate and selected environment read-only. Only disposable
    scratch is writable; the session owns descendants and denies networking.
    """
    from kernel_exec_capability import shell_capability
    from kernel_exec_session import ExecSession
    from kernel_test_isolation import _interpreter_roots

    selected = Path(selected_test_python())
    with tempfile.TemporaryDirectory(prefix="stateframe-smoke-") as temporary:
        scratch = Path(temporary)
        home, env = scratch / "home", scratch / "env"
        home.mkdir()
        env.mkdir()
        cap = shell_capability(
            workspace=root, run_env_dir=env, scratch_home=home,
            network="none", writable=False,
            extra_read_roots=(selected, *map(Path, _interpreter_roots(selected))))
        arguments = [json.dumps(list(checked)), json.dumps(list(roots))]
        limits = replace(cap.limits, wall_seconds=timeout)
        if projection_enabled() and sum(map(len, arguments)) > 32 << 10:
            # Large scopes can exceed exec's per-argument limit and the
            # usual output tail. Keep both bounds explicit.
            inputs = home / "smoke-inputs.json"
            inputs.write_text("[" + ",".join(arguments) + "]", encoding="utf-8")
            limits = replace(limits, output_bytes=min(
                4 << 20, max(limits.output_bytes, len(arguments[0]) + 320 * len(checked))))
            arguments = ["--inputs", str(inputs)]
        cap = replace(cap, purpose="probe", limits=limits)
        (scratch / "session").mkdir()
        session = ExecSession(scratch / "session")
        return run_probe_session(
            session, lambda: session.run(
                [str(selected), "-I", "-B", "-c", _CHILD, str(root), *arguments], cap),
            cleanup_error=ImportSmokeCleanupError)


def python_paths(root: Path, values: Sequence[object]) -> tuple[str, ...]:
    """Existing workspace-relative ``.py`` files among ``values`` (deduplicated)."""
    root = Path(root).resolve()
    consumers = _workspace_consumers(root)
    if consumers is None:
        return _python_paths(root, values)
    paths = tuple(str(value or "").strip() for value in values)
    return consumers.smoke_projection(
        "ship-tree-checked-paths", (paths,),
        lambda: (_python_paths(root, paths), _internal_paths(root, paths)),
    )


def _workspace_consumers(root: Path):
    from sys import modules
    module = modules.get("kernel_workspace_consumers")
    return module.consumers_for(root) if module is not None else None


def projection_enabled() -> bool:
    return (os.environ.get("STATEFRAME_WORKSPACE_OBJECTS") == "on"
            and os.environ.get("STATEFRAME_TURN_DELIVERY_PROOFS", "off").strip().lower() == "on")


def _internal_paths(root: Path, values: Sequence[str]) -> bool:
    return all((root / value).is_relative_to(root) and ".." not in Path(value).parts
               for value in values)


def _python_paths(root: Path, values: Sequence[object]) -> tuple[str, ...]:
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
        count_work("parses")
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


def _counted_is_file(path: Path) -> bool:
    count_work("files_statted")
    return path.is_file()


def _counted_is_dir(path: Path) -> bool:
    count_work("files_statted")
    return path.is_dir()


def _provides(directory: Path, name: str) -> bool:
    """``directory`` makes ``name`` importable (module file or package dir)."""
    if _counted_is_file(directory / f"{name}.py"):
        return True
    package = directory / name
    if not _counted_is_dir(package):
        return False
    return _counted_is_file(package / "__init__.py") or any(package.glob("*.py"))


def _ini_pythonpath(directory: Path) -> tuple[Path, ...]:
    """``pythonpath`` entries declared by pytest config files in ``directory``."""
    import configparser
    found: list[Path] = []
    for name, sections in _INI_SECTIONS.items():
        ini = directory / name
        if not _counted_is_file(ini):
            continue
        parser = configparser.ConfigParser(interpolation=None)
        try:
            loaded = parser.read(ini, encoding="utf-8")
            count_work("files_read", len(loaded))
        except (configparser.Error, OSError, UnicodeError):
            continue
        for section in sections:
            if parser.has_option(section, "pythonpath"):
                found.extend(directory / item
                             for item in parser.get(section, "pythonpath").split())
    pyproject = directory / "pyproject.toml"
    if _counted_is_file(pyproject):
        try:
            import tomllib
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            count_work("files_read")
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
    pythonpath = os.environ.get("PYTHONPATH", "")
    consumers = _workspace_consumers(root)
    if consumers is None:
        return _source_roots(root, paths, pythonpath)[0]
    return consumers.smoke_projection(
        "ship-tree-roots",
        (tuple(paths), pythonpath, selected_test_python(), tuple(sys.version_info),
         _defer_source_providers()),
        lambda: _source_roots(root, paths, pythonpath),
    )


def _defer_source_providers() -> bool:
    if "STATEFRAME_TURN_DELIVERY_PROOFS" not in os.environ:
        return False
    from kernel_turn_proofs import enabled

    return enabled()


def _source_roots(
    root: Path, paths: Sequence[str], pythonpath: str, *,
    imports_for: Callable[[str], tuple[str, ...]] | None = None,
) -> tuple[tuple[str, ...], bool]:
    defer_providers = _defer_source_providers()

    internal_only = True

    def resolve(candidate: Path) -> Path:
        nonlocal internal_only
        # External prefixes can change without any change below the workspace.
        if not candidate.is_relative_to(root) or ".." in candidate.parts:
            internal_only = False
        return candidate.resolve()

    def checked_imports() -> Iterable[tuple[Path, tuple[str, ...]]]:
        for raw in paths:
            if imports_for is not None:
                yield Path(raw), imports_for(raw)
                continue
            value = str(raw or "").strip()
            if not value.endswith(".py"):
                continue
            file = resolve(root / value)
            try:
                relative = file.relative_to(root)
            except ValueError:
                continue
            try:
                source = file.read_text(encoding="utf-8")
                count_work("files_read")
                tops = _top_level_import_names(source)
            except (OSError, UnicodeError):
                tops = ()
            yield relative, tops

    checked_files = checked_imports()
    import_names: set[str] = set()
    if not defer_providers:
        checked_files = tuple(checked_files)
        import_names = {top for _, tops in checked_files for top in tops}

    # These observations live for one traversal. Warm workspace reuse is
    # handled by the generation-fenced projection around this resolver.
    probe = cache(_provides)
    config_paths = cache(_ini_pythonpath)

    @cache
    def provider_names(directory: Path) -> frozenset[str] | None:
        if defer_providers:
            return None
        if not import_names:
            return frozenset()
        try:
            with os.scandir(directory) as entries:
                names = {entry.name for entry in entries}
        except OSError:
            return None
        candidates = (names | {name[:-3] for name in names if name.endswith(".py")})
        return frozenset(name for name in import_names.intersection(candidates)
                         if probe(directory, name))

    def provides(directory: Path, name: str) -> bool:
        names = provider_names(directory)
        return name in names if names is not None else probe(directory, name)

    @cache
    def sibling_index() -> tuple[dict[str, list[Path]], list[Path]]:
        count_work("index_builds")
        found: dict[str, list[Path]] = {}
        unindexed: list[Path] = []
        for child in _first_level_dirs(root):
            names = provider_names(child)
            if names is None:
                unindexed.append(child)
            else:
                for name in names:
                    found.setdefault(name, []).append(child)
        return found, unindexed

    @cache
    def sibling_providers(name: str) -> tuple[Path, ...]:
        found, unindexed = sibling_index()
        return (*found.get(name, ()), *(child for child in unindexed if probe(child, name)))

    result: dict[Path, None] = {}
    result_names: set[str] = set()
    unindexed_result: list[Path] = []
    observed_result: dict[str, tuple[int, bool]] = {}

    def add_root(directory: Path) -> None:
        if directory in result:
            return
        result[directory] = None
        names = provider_names(directory)
        if names is None:
            unindexed_result.append(directory)
        else:
            result_names.update(names)

    def selected_root_provides(top: str) -> bool:
        if top in result_names:
            return True
        # A negative observation only needs to check roots added since then.
        start, found = observed_result.get(top, (0, False))
        if not found:
            found = any(probe(unindexed_result[index], top)
                        for index in range(start, len(unindexed_result)))
            observed_result[top] = len(unindexed_result), found
        return found

    @cache
    def ancestor_roots(parent: tuple[str, ...], tops: tuple[str, ...]) -> tuple[Path, ...]:
        roots: list[Path] = []
        for index in range(len(parent) + 1):
            directory = root.joinpath(*parent[:index])
            if any(provides(directory, top) for top in tops):
                roots.append(directory)
            for entry in config_paths(directory):
                try:
                    resolved = resolve(entry)
                    resolved.relative_to(root)
                except (OSError, ValueError):
                    continue
                if _counted_is_dir(resolved):
                    roots.append(resolved)
        return tuple(roots)

    add_root(root)
    for relative, tops in checked_files:
        for directory in ancestor_roots(relative.parts[:-1], tops):
            add_root(directory)
        for top in tops:
            if selected_root_provides(top):
                continue
            providers = sibling_providers(top)
            if len(providers) == 1:
                add_root(providers[0])
    for entry in pythonpath.split(os.pathsep):
        if not entry.strip():
            continue
        try:
            resolved = resolve(root / entry)
            resolved.relative_to(root)
        except (OSError, ValueError):
            continue
        if _counted_is_dir(resolved):
            result[resolved] = None
    return tuple(str(item) for item in result), internal_only


def _first_level_dirs(root: Path) -> tuple[Path, ...]:
    try:
        return tuple(sorted(child for child in root.iterdir()
                            if _counted_is_dir(child) and not child.name.startswith(".")))
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
    if (baseline is None or result.status in {"UNKNOWN", "ENGINE_ERROR"}
            or baseline.digest_format != TREE_DIGEST_FORMAT):
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
    return replace(result, status=status, failures=tuple(failures), environment=tuple(environment),
                   digest_format=TREE_DIGEST_FORMAT)


def _digest_identities(dependencies: object, environment: object,
                       resolver: object) -> list[list[str]]:
    return [
        ["identity", label, json.dumps(value, sort_keys=True, default=str,
                                      separators=(",", ":"), ensure_ascii=True)]
        for label, value in (("dependencies", dependencies), ("environment", environment),
                             ("resolver", resolver))
    ]


def tree_digest(root: Path, paths: Sequence[str], *, dependency_identity: object = "",
                environment_identity: object = "", resolver_identity: object = "") -> str:
    entries = _digest_identities(dependency_identity, environment_identity, resolver_identity)
    def calculate() -> str:
        rows = [*entries]
        for path in sorted(str(item) for item in paths if not is_non_product(str(item))):
            rows.append(path_entry(Path(root) / path, path))
        return "sha256:" + canonical_tree_digest(TREE_DIGEST_FORMAT, rows)

    resolved = Path(root).absolute()
    consumers = _workspace_consumers(resolved)
    if consumers is None:
        return calculate()
    return consumers.smoke_projection(
        "ship-tree-wire-digest", (tuple(paths), tuple(tuple(row) for row in entries)),
        lambda: (calculate(), _internal_paths(resolved, paths)),
    )



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
    from contextlib import nullcontext
    consumers = _workspace_consumers(root)
    pin = consumers.state.pin("ship-tree-execution") if consumers is not None else nullcontext()
    with pin:
        return _run_smoke(
            root, paths, timeout_s=timeout_s, baseline=baseline, roots=roots,
            dependency_identity=dependency_identity, environment_identity=environment_identity,
            declared_scope=declared_scope, resolver_identity=resolver_identity,
        )


def _run_smoke(root: Path, paths: Sequence[str], *, timeout_s: float,
               baseline: ShipTreeSmokeResult | None, roots: Sequence[str] | None,
               dependency_identity: object, environment_identity: object,
               declared_scope: Sequence[str] | None, resolver_identity: object) -> ShipTreeSmokeResult:
    checked = python_paths(root, paths)
    scope = python_paths(root, declared_scope if declared_scope is not None else paths)
    if not checked:
        return ShipTreeSmokeResult("UNKNOWN", (), (), "", declared_scope=scope,
                                   digest_format=TREE_DIGEST_FORMAT, probe_complete=True)
    if roots is not None:
        import_roots = tuple(str(Path(item).resolve()) for item in roots)
    else:
        import_roots = source_roots(root, checked)
    effective_dependencies = dependency_identity or {"roots": import_roots}
    effective_environment = environment_identity or {
        "executable": selected_test_python(),
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
    return _probe_smoke(root, checked, import_roots, timeout_s, baseline, scope,
                        digest, dependency_key, environment_key, resolver_key,
                        lambda: _static_findings(root, checked))


def _static_findings(root: Path, checked: Sequence[str]) -> tuple[tuple[str, str], ...]:
    findings: list[tuple[str, str]] = []
    for path in checked:
        try:
            source = (root / path).read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        findings.extend((path, finding) for finding in local_import_shadows(source))
    return tuple(findings)


def _probe_smoke(root: Path, checked: tuple[str, ...], import_roots: tuple[str, ...],
                 timeout_s: float, baseline: ShipTreeSmokeResult | None,
                 scope: tuple[str, ...], digest: str, dependency_key: str,
                 environment_key: str, resolver_key: str,
                 static_findings: Callable[[], Sequence[tuple[str, str]]] |
                 Sequence[tuple[str, str]]) -> ShipTreeSmokeResult:
    failures: list[tuple[str, str]] = []
    host_failures: list[tuple[str, str]] = []
    runner_failed = False
    evidence = None
    engine_error = False
    from kernel_exec_session import ExecSessionError
    try:
        completed = _execute_smoke(root, checked, import_roots, max(0.1, float(timeout_s)))
        evidence = probe_evidence(completed)
        if not completed.cleanup.ok:
            raise ImportSmokeCleanupError("import smoke cleanup failed: " + completed.cleanup.detail)
        if completed.timed_out or completed.exit_code != 0:
            raise ExecSessionError("import smoke timed out" if completed.timed_out else
                                   f"import smoke exited {completed.exit_code}")
        # Repository prints (including setup.py usage) are not the result protocol.
        payload = completed.output.rsplit(b"\nSTATEFRAME_SMOKE_RESULT=", 1)
        rows = json.loads(payload[1] if len(payload) == 2 else b"")
    except (OSError, ValueError, ExecSessionError) as error:
        evidence = probe_evidence(error) or evidence
        engine_error = getattr(error, "outcome", "") == "ENGINE_ERROR"
        rows = []
        runner_failed = True
        host_failures = [(path, f"fresh interpreter failed: {type(error).__name__}: {error}")
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
    static_findings = static_findings() if callable(static_findings) else static_findings
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
    status = ("ENGINE_ERROR" if engine_error else
              "FAIL" if failures else ("ENVIRONMENT" if host_failures else "PASS"))
    raw = ShipTreeSmokeResult(status, checked, tuple(failures), digest,
                              tuple(host_failures), import_roots,
                              dependency_key, environment_key, scope, resolver_key,
                              TREE_DIGEST_FORMAT, probe_evidence=evidence,
                              probe_complete=not runner_failed and not host_failures)
    return apply_baseline(raw, baseline)


@dataclass(frozen=True)
class SmokeScope:
    paths: tuple[str, ...]
    dependencies: tuple[str, ...]
    parent: str
    untouched: bool


def _interpreter_stamp() -> tuple[str, tuple[object, ...]]:
    executable = selected_test_python()

    def signature(path: Path) -> tuple[object, ...]:
        try:
            info = path.stat()
        except FileNotFoundError:
            return (str(path),)
        return (str(path), info.st_dev, info.st_ino, info.st_mode, info.st_size,
                info.st_mtime_ns, info.st_ctime_ns)

    try:
        selected = Path(executable)
        resolved = selected.resolve(strict=True)
        prefixes = dict.fromkeys((selected.parent.parent, resolved.parent.parent))
        settings = (path for prefix in prefixes for path in
                    (prefix / "pyvenv.cfg", prefix / "lib"))
        return executable, (signature(resolved), *map(signature, settings))
    except (OSError, RuntimeError):
        return executable, ()


def _parser_policy() -> tuple[object, ...]:
    return (sys.implementation.name, tuple(sys.version_info),
            getattr(sys, "stdlib_module_names", frozenset()))


@dataclass(frozen=True)
class SmokeBasis:
    tree: bytes
    epoch: bytes
    scope: SmokeScope
    admitted_parent: str | None
    interpreter: tuple[str, tuple[object, ...]]
    pythonpath: str
    parser: tuple[object, ...]

    @property
    def baseline_context(self) -> tuple[object, ...]:
        return (self.scope.parent, self.admitted_parent, self.interpreter,
                self.pythonpath, self.parser)


@dataclass(frozen=True)
class _SmokePlan:
    checked: tuple[str, ...]
    roots: tuple[str, ...]
    digest: str
    dependencies: str
    environment: str
    resolver: str
    findings: tuple[tuple[str, str], ...]
    reusable: bool

    def run(self, root: Path) -> ShipTreeSmokeResult:
        if not self.checked:
            return ShipTreeSmokeResult("UNKNOWN", (), (), digest_format=TREE_DIGEST_FORMAT,
                                       probe_complete=True)
        return _probe_smoke(root, self.checked, self.roots, 20.0, None, self.checked,
                            self.digest, self.dependencies, self.environment,
                            self.resolver, self.findings)


def _tree_parts(root: Path, path: str) -> tuple[bytes, ...] | None:
    candidate = Path(path)
    if candidate.is_absolute():
        if not candidate.is_relative_to(root):
            return None
        candidate = candidate.relative_to(root)
    if ".." in candidate.parts:
        return None
    return tuple(map(os.fsencode, candidate.parts))


class SmokeProjection:
    """Attempt-local static facts and interpreter results for fenced smoke inputs."""

    def __init__(self, consumers: WorkspaceConsumers):
        self._consumers = consumers
        self._root = consumers.state.fence.root
        self._facts: OrderedDict[
            tuple[tuple[object, ...], bytes], tuple[tuple[str, ...], tuple[str, ...]]
        ] = OrderedDict()
        self._tree: Directory | None = None
        self._links = 0
        self._current: tuple[SmokeBasis, ShipTreeSmokeResult] | None = None
        self._launch: Future[tuple[SmokeBasis, ShipTreeSmokeResult | None] | None] | None = None
        self.launch_attempted = False

    def _basis(self, view: WorkspaceSnapshot, scope: SmokeScope) -> SmokeBasis:
        return SmokeBasis(view.tree.digest, view.epoch, scope, self._consumers.admitted_parent,
                          _interpreter_stamp(), os.environ.get("PYTHONPATH", ""),
                          _parser_policy())

    def _has_links(self, tree: Directory) -> bool:
        from kernel_workspace_tree import Directory, Symlink, children, diff

        if self._tree is None:
            pending = [tree]
            while pending:
                for _, entry in children(pending.pop()):
                    if isinstance(entry, Directory):
                        pending.append(entry)
                    elif isinstance(entry, Symlink):
                        self._links += 1
        elif self._tree.digest != tree.digest:
            for change in diff(self._tree, tree):
                self._links += isinstance(change.after, Symlink) - isinstance(
                    change.before, Symlink)
        self._tree = tree
        return bool(self._links)

    def _blob(self, path: str, digest: bytes, policy: tuple[object, ...]
              ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        from kernel_workspace_state import WorkspaceUnavailable

        key = policy, digest
        if key not in self._facts:
            try:
                payload = (self._root / path).read_bytes()
            except OSError as error:
                raise WorkspaceUnavailable("smoke source unavailable") from error
            if sha256(payload).digest() != digest:
                raise WorkspaceUnavailable("smoke source changed during projection")
            try:
                source = payload.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
            except UnicodeError:
                facts = (), ()
            else:
                facts = _top_level_import_names(source), local_import_shadows(source)
            self._facts[key] = facts
            if len(self._facts) > 16_384:
                self._facts.popitem(last=False)
        self._facts.move_to_end(key)
        return self._facts[key]

    def _plan(self, tree: Directory, basis: SmokeBasis) -> _SmokePlan:
        from kernel_workspace_tree import File, lookup

        root, scope = self._root, basis.scope
        paths = tuple(path for path in scope.paths if path.endswith(".py"))
        parts = tuple(_tree_parts(root, path) for path in paths)
        files: dict[str, File] = {}
        reusable = not self._has_links(tree) and None not in parts
        static: list[tuple[str, str]] = []
        if reusable:
            for item in parts:
                assert item is not None
                path = os.fsdecode(b"/".join(item))
                if not is_non_product(path) and isinstance(entry := lookup(tree, item), File):
                    files[path] = entry
            checked = tuple(files)
        else:
            checked = _python_paths(root, paths)
            for path in checked:
                entry = lookup(tree, tuple(map(os.fsencode, Path(path).parts)))
                if isinstance(entry, File):
                    files[path] = entry

        def imports_for(path: str) -> tuple[str, ...]:
            entry = files.get(path)
            if entry is not None:
                names, findings = self._blob(path, entry.sha256, basis.parser)
            else:
                try:
                    source = (root / path).read_text(encoding="utf-8")
                except (OSError, UnicodeError):
                    return ()
                names, findings = _top_level_import_names(source), local_import_shadows(source)
            static.extend((path, finding) for finding in findings)
            return names

        roots, internal = _source_roots(root, checked, basis.pythonpath, imports_for=imports_for)

        dependency_parts = tuple(_tree_parts(root, path) for path in scope.dependencies)
        reusable = reusable and internal and None not in dependency_parts
        dependency_rows = _digest_identities("", "", "")
        for path in sorted(path for path in scope.dependencies if not is_non_product(path)):
            item = _tree_parts(root, path)
            entry = lookup(tree, item) if reusable and item is not None else None
            if isinstance(entry, File):
                dependency_rows.append(["file", path, entry.sha256.hex()])
            else:
                dependency_rows.append(path_entry(root / path, path))
        dependencies = {
            "parent": json.loads(scope.parent),
            "checked_dependencies": scope.dependencies,
            "dependency_content_digest": (
                "sha256:" + canonical_tree_digest(TREE_DIGEST_FORMAT, dependency_rows)),
        }
        environment = {"python": tuple(sys.version_info[:3]),
                       "executable": basis.interpreter[0], "source_roots": roots}
        resolver = {"source_roots": roots, "pythonpath": basis.pythonpath, "checked": checked}
        entries = _digest_identities(dependencies, environment, resolver)
        for path in sorted(checked):
            if path in files:
                entries.append(["file", path, files[path].sha256.hex()])
            else:
                entries.append(path_entry(root / path, path))
        keys = (json.dumps(item, sort_keys=True, default=str)
                for item in (dependencies, environment, resolver))
        return _SmokePlan(checked, roots, "sha256:" + canonical_tree_digest(TREE_DIGEST_FORMAT, entries),
                          *keys, tuple(static), reusable and bool(basis.interpreter[1]))

    def offer(self, scope: SmokeScope) -> None:
        """Stage optional bootstrap work without retaining runtime or journal state."""
        from kernel_workspace_state import WorkspaceUnavailable

        if self.launch_attempted:
            return
        self.launch_attempted = True
        try:
            epoch = self._consumers.smoke_epoch()
        except WorkspaceUnavailable:
            return
        future: Future[tuple[SmokeBasis, ShipTreeSmokeResult | None] | None] = Future()
        self._launch = future

        def build() -> None:
            try:
                with self._consumers.smoke_read(background=True) as view:
                    basis = self._basis(view, scope)
                    plan = self._plan(view.tree, basis) if view.epoch == epoch else None
                result = None
                # The import sandbox owns cleanup. Do not hold a live writer
                # fence during execution, and qualify the outcome at take().
                if plan is not None and plan.reusable and self._basis(view, scope) == basis:
                    result = plan.run(self._root)
                    if self._basis(view, scope) != basis:
                        result = None
                future.set_result((basis, result) if plan is not None else None)
            except BaseException as error:
                # Fatal cleanup errors must reach the owner even when this
                # staged result no longer qualifies for the live scope.
                future.set_exception(error)

        Thread(target=build, name="ship-tree-smoke", daemon=True).start()

    def take(self, scope: SmokeScope,
             publish: Callable[[SmokeBasis, ShipTreeSmokeResult, bool, bool], ShipTreeSmokeResult]
             ) -> ShipTreeSmokeResult:
        """Qualify staged evidence or run the required check against the current pin."""
        from kernel_workspace_state import WorkspaceUnavailable

        staged = None
        if self._launch is not None:
            try:
                staged = self._launch.result()
            except Exception:
                pass
            finally:
                self._launch = None
        try:
            with self._consumers.smoke_read() as view:
                basis = self._basis(view, scope)
                same_launch_tree = (staged is not None
                                    and staged[0].tree == basis.tree
                                    and staged[0].epoch == basis.epoch
                                    and staged[0].scope.parent == scope.parent
                                    and staged[0].admitted_parent == basis.admitted_parent)
                may_record_parent = not self.launch_attempted or same_launch_tree
                if self._current is not None and self._current[0] == basis:
                    return publish(basis, self._current[1], True, may_record_parent)
                if staged is not None and staged[0] == basis and staged[1] is not None:
                    raw, reusable = staged[1], True
                else:
                    plan = self._plan(view.tree, basis)
                    raw, reusable = plan.run(self._root), plan.reusable
                if (self._basis(view, scope) != basis
                        or not view.validate_for_use("ship-tree-proof")):
                    raise WorkspaceUnavailable("smoke proof inputs changed during execution")
                self._current = (basis, raw) if reusable and raw.probe_complete else None
                return publish(basis, raw, False, may_record_parent)
        except WorkspaceUnavailable:
            self._current = None
            return ShipTreeSmokeResult("ENGINE_ERROR", (), ())


def render_smoke(result: ShipTreeSmokeResult) -> str:
    recovery = ""
    if result.probe_evidence is not None:
        recovery = "\n" + json.dumps(result.probe_evidence, sort_keys=True)
    if result.status == "ENGINE_ERROR":
        return (f"{BLOCK_HEADER} ENGINE_ERROR: fresh interpreter unavailable."
                + recovery)
    if result.status == "PENDING":
        return f"{BLOCK_HEADER} PENDING: fresh interpreter smoke is in progress."
    if result.status == "PASS":
        return (f"{BLOCK_HEADER} PASS: {len(result.checked)} touched module(s) import in a fresh interpreter."
                + recovery)
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
        return "\n".join(lines) + recovery
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
    return "\n".join(lines) + recovery


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
