"""Host-only pytest collection and node-identity evidence for opt-in unit gates.

Collection broadens selectors to their files and removes selection filters. The
execution itself retains the requested selectors and fail-fast behavior. Both
subprocesses use the host's sandbox/effect tape; neither collection text nor this
metadata is delivered in the compact model-facing receipt.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from unit_coalesce_contract import GateRun

_STATUS = re.compile(
    r"^(\S+?\.py::.+?)\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS|DESELECTED)\b",
    re.MULTILINE,
)
_FAILURE = re.compile(r"^(?:FAILED|ERROR)\s+(\S+?\.py::\S+)", re.MULTILINE)
# The terminal reporter uses the gateway ID, including explicitly named --tx
# workers; neither the default gwN name nor a progress percentage is required.
_DISTRIBUTED_STATUS = re.compile(
    r"^\[[^\]\r\n]+\](?:\s+\[\s*\d+%\])?\s+"
    r"(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS|DESELECTED)\s+"
    r"(\S+?\.py::.+?)\s*$", re.MULTILINE,
)
_NODE = re.compile(r"^\S+?\.py::.+$", re.MULTILINE)
MAX_TEST_FILES = 400
MAX_NODES = 20000
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_SOURCE_BYTES = 256 * 1024 * 1024
_IDENTITY_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache",
})


def _identity_walk_error(error: OSError) -> None:
    raise error


def _artifact_paths(root: Path, runtime: object) -> set[str]:
    """Exclude host-proven artifacts, not directories guessed from their names."""
    paths: set[str] = set()
    # Keep the logical prefix too: a worktree may link .venv to a shared host
    # environment outside the workspace. Both aliases are host-proven artifacts.
    for prefix in {Path(sys.prefix).absolute(), Path(sys.prefix).resolve()}:
        if prefix != root and prefix.is_relative_to(root) and (prefix / "pyvenv.cfg").is_file():
            paths.add(prefix.relative_to(root).as_posix())
    journal_path = getattr(getattr(runtime, "journal", None), "path", None)
    if journal_path is not None:
        journal = Path(journal_path).resolve()
        if journal.is_relative_to(root):
            # Same journal-directory exclusion as the host's stable product
            # revision. A journal directly in the workspace excludes only itself.
            artifact = journal if journal.parent == root else journal.parent
            paths.add(artifact.relative_to(root).as_posix())
    return paths


def source_identity(root: Path, *, exclude: Sequence[str] = (), runtime: object = None) -> str:
    """Bounded workspace input identity, excluding only owned products/artifacts.

    Constants-only modules, dotfiles and binary/data inputs all influence tests.
    Unknown input types must not fall outside the freshness boundary. Prune
    tooling directories before walking, and fail closed on unreadable inputs.
    """
    root = root.resolve()
    excluded = set(exclude)
    artifacts = _artifact_paths(root, runtime)
    sources: list[tuple[str, str]] = []
    total_bytes = 0
    entries = 0
    for directory, dirs, names in os.walk(root, onerror=_identity_walk_error):
        dirs[:] = sorted(name for name in dirs if name not in _IDENTITY_SKIP_DIRS
                         and (Path(directory) / name).relative_to(root).as_posix() not in artifacts)
        entries += len(dirs) + len(names)
        if entries > 20000:
            raise ValueError("unit gate source identity exceeds entry bound")
        # os.walk does not follow directory symlinks. Silently omitting one
        # would certify an input tree that the tests can still import/read.
        if any((Path(directory) / name).is_symlink() for name in dirs):
            raise ValueError("unit gate source identity cannot verify directory symlink")
        for name in sorted(names):
            path = Path(directory) / name
            rel = path.relative_to(root).as_posix()
            if rel in excluded or rel in artifacts:
                continue
            if not path.resolve().is_relative_to(root):
                raise ValueError("unit gate source escapes workspace")
            if not path.is_file():
                raise ValueError("unit gate source identity cannot verify special file")
            if path.stat().st_size > MAX_FILE_BYTES:
                raise ValueError("unit gate source identity exceeds byte bound")
            with path.open("rb") as stream:
                data = stream.read(MAX_FILE_BYTES + 1)
            total_bytes += len(data)
            if len(data) > MAX_FILE_BYTES or total_bytes > MAX_SOURCE_BYTES:
                raise ValueError("unit gate source identity exceeds byte bound")
            sources.append((rel, hashlib.sha256(data).hexdigest()))
    # Old checkpoints may have discarded a red before retaining these same
    # bytes. A new authority namespace expires both accumulated node evidence
    # and completed passes even when the workspace itself did not change.
    basis = {"schema": "unit-gate-inputs/2", "sources": sorted(sources)}
    return hashlib.sha256(json.dumps(basis, separators=(",", ":")).encode()).hexdigest()


def shared_source_identity(plan, root: Path, runtime: object = None) -> str:
    """Only unit-owned products have a separate per-pass identity."""
    return source_identity(root, exclude=tuple(path for unit in plan.units for path in unit.files), runtime=runtime)


def pass_snapshot(unit, root: Path, evidence, test_sources: str) -> tuple[str, str, tuple[str, ...] | None]:
    """Bind a pass to owned products, shared workspace inputs and full inventory.

    Inventory is compared only when a later capture has a valid collection; a
    failed inventory is never permission to fabricate a new set of test cases.
    """
    files = []
    for path in unit.files:
        source = root / path
        if not source.resolve().is_relative_to(root.resolve()):
            raise ValueError("unit gate product escapes workspace")
        if source.is_file():
            if source.stat().st_size > MAX_FILE_BYTES:
                raise ValueError("unit gate product exceeds byte bound")
            with source.open("rb") as stream:
                data = stream.read(MAX_FILE_BYTES + 1)
            if len(data) > MAX_FILE_BYTES:
                raise ValueError("unit gate product exceeds byte bound")
            files.append((path, hashlib.sha256(data).hexdigest()))
        else:
            files.append((path, None))
    product = hashlib.sha256(json.dumps(files, separators=(",", ":")).encode()).hexdigest()
    runs = getattr(evidence, "unit_gate_runs", ())
    _nodes, collected, valid, _identity = run_evidence(runs)
    inventory = (tuple(sorted(node for node in collected if test_name(node) in unit.gate_tests))
                 if valid and all(run.collected is not None for run in runs) else None)
    return product, test_sources, inventory


def test_name(node: str) -> str:
    return node.split("[", 1)[0].rsplit("::", 1)[-1]


def statuses(stdout: str) -> dict[str, bool]:
    """Retain parameter/class/file identity; any nonpass wins over a repeated pass."""
    found: dict[str, bool] = {}
    for node, status in _STATUS.findall(stdout):
        found[node] = found.get(node, True) and status == "PASSED"
    for status, node in _DISTRIBUTED_STATUS.findall(stdout):
        found[node] = found.get(node, True) and status == "PASSED"
    for node in _FAILURE.findall(stdout):
        found[node] = False
    return found


def collection_command(command: Sequence[str], targets: Sequence[str], root: Path,
                       plan: object = None, *, runtime: object = None) -> tuple[list[str], str]:
    """Collect every case of gate names, including duplicate names in other files.

    Keep interpreter isolation, config and artifact-policy options. Override
    addopts and selection only for collection, never for the requested execution.
    Exceeding bounded discovery fails closed rather than certifying partial scope.
    """
    names = {name for unit in getattr(plan, "units", ()) for name in unit.gate_tests}
    scopes = {str(target).split("::", 1)[0] for target in targets}
    scopes.update(str(path).split("::", 1)[0] for path in getattr(plan, "public_gate_paths", ()))
    if names:
        count = 0
        for path in sorted(root.rglob("*.py")):
            rel = path.relative_to(root)
            if any(part.startswith(".") or part in {"venv", "env", "node_modules", "__pycache__"}
                   for part in rel.parts[:-1]):
                continue
            if not (path.name.startswith("test_") or path.name.endswith("_test.py")
                    or any(part in {"test", "tests"} for part in rel.parts[:-1])):
                continue
            count += 1
            if count > MAX_TEST_FILES:
                raise ValueError("unit gate collection exceeds test-file bound")
            if not path.resolve().is_relative_to(root.resolve()) or path.stat().st_size > MAX_FILE_BYTES:
                raise ValueError("unit gate test source exceeds workspace bounds")
            source = path.read_text(encoding="utf-8")
            if not any(re.search(r"\bdef\s+" + re.escape(name) + r"\s*\(", source) for name in names):
                continue
            tree = ast.parse(source, filename=str(rel))
            if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
                   for node in ast.walk(tree)):
                scopes.add(rel.as_posix())
    if len(scopes) > MAX_TEST_FILES:
        raise ValueError("unit gate collection exceeds selector bound")
    args = list(command)
    module = next(i for i in range(1, len(args) - 1) if args[i:i + 2] == ["-m", "pytest"])
    result = args[:module + 2]
    options = iter(args[module + 2:])
    for option in options:
        if option in {"-k", "-m", "--maxfail", "--verbosity", "--deselect", "--ignore", "--ignore-glob",
                      "--last-failed-no-failures"}:
            next(options, None)
        elif (option in {"-x", "--exitfirst", "-q", "-v", "-vv", "--quiet", "--verbose"}
              or option in {"--lf", "--last-failed", "--ff", "--failed-first", "--sw", "--stepwise",
                            "--sw-skip", "--stepwise-skip", "--sw-reset", "--stepwise-reset"}
              or option.startswith(("--maxfail=", "--verbosity=", "-k", "-m", "--deselect=",
                                    "--ignore=", "--ignore-glob=", "--last-failed-no-failures="))):
            continue
        else:
            result.append(option)
    result += ["--collect-only", "-q", "-o", "addopts=", "-k", "", "-m", "", *sorted(scopes)]
    # Bind accumulated selector evidence to test/harness contents as well as the
    # indexed product tree. A test edit must never borrow an earlier case's pass.
    return result, source_identity(root, runtime=runtime)


def collected_nodes(stdout: str, exit_code: int, truncated: bool | None) -> tuple[str, ...] | None:
    if exit_code != 0 or truncated:
        return None
    nodes = tuple(dict.fromkeys(_NODE.findall(stdout)))
    return nodes if 0 < len(nodes) <= MAX_NODES else None


def run_evidence(runs: Sequence[GateRun]) -> tuple[dict[str, bool], set[str], bool, str]:
    observed: dict[str, bool] = {}
    collected: set[str] = set()
    valid = bool(runs) and all(run.complete and run.exit_code in (0, 1, 5) for run in runs)
    for run in runs:
        if run.collected is not None:
            collected.update(run.collected)
        for node, passed in statuses(run.stdout).items():
            observed[node] = observed.get(node, True) and passed
    identity = json.dumps(sorted({run.source_digest for run in runs}))
    return observed, collected, valid, identity
