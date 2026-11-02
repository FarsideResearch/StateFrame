"""Build a local SWE-bench-Lite row corpus without downloading or leaking gold data.

The adapter has three deliberately separate boundaries:

* JSONL ingestion decodes the upstream instance shape and validates selectors.
* WorkspaceMaterializer creates an immutable base worktree and a gold worktree from a
  local repository or the canonical GitHub URL.
* Row conversion emits the existing repair-row shape, including a grader-only test
  runner whose assertion carries the failing test tail.

The command line interface never downloads a dataset. Use
scripts/fetch_swebench_lite.py to fetch a pinned Hugging Face export first.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import cast

try:
    from build_tasks.multifile_row import (
        MultiFileRowError,
        normalize_multifile_row,
        normalize_repo_path,
    )
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution
    from multifile_row import (
        MultiFileRowError,
        normalize_multifile_row,
        normalize_repo_path,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kernel_test_python import selected_test_python  # noqa: E402


DATASET_ID = "SWE-bench/SWE-bench_Lite"
DEFAULT_OUTPUT = Path("results/swebench_lite_sample.jsonl")
DEFAULT_SKIPPED_OUTPUT = Path("results/swebench_lite_skipped.json")
DEFAULT_CACHE_DIR = Path(".cache/swebench_lite")
_HUNK_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@(?P<section>.*)$"
)
_FUNCTION_RE = re.compile(r"\b(?:async\s+)?def\s+(?P<name>[A-Za-z_]\w*)\b")
_INSTANCE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,64}$")


class AdapterError(ValueError):
    """The input or workspace cannot produce a trustworthy row."""


class MaterializationRequired(AdapterError):
    """Raised when conversion would need a checkout without the explicit opt-in."""


class SkipInstance(AdapterError):
    """A valid input is intentionally excluded with a named, reportable reason."""

    def __init__(self, reason: str, detail: str) -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}")


@dataclass(frozen=True)
class SWEInstance:
    """Validated subset of one SWE-bench JSONL record."""

    instance_id: str
    repo: str
    base_commit: str
    patch: str
    problem_statement: str
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]
    version: str | None = None
    environment_setup_commit: str | None = None
    repo_url: str | None = None
    test_patch: str | None = None
    raw: Mapping[str, object] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_mapping(
        cls,
        record: Mapping[str, object],
        *,
        line_no: int | None = None,
    ) -> "SWEInstance":
        prefix = f"line {line_no}: " if line_no is not None else ""
        instance_id = _required_string(record, "instance_id", prefix=prefix)
        repo = _required_string(record, "repo", prefix=prefix)
        if "/" not in repo or repo.startswith("/") or ".." in PurePosixPath(repo).parts:
            raise AdapterError(f"{prefix}repo must be an owner/name slug: {repo!r}")
        base_commit = _required_string(record, "base_commit", prefix=prefix)
        if not _SHA_RE.fullmatch(base_commit):
            raise AdapterError(
                f"{prefix}base_commit must be a git SHA: {base_commit!r}"
            )
        patch = _required_string(record, "patch", prefix=prefix)
        problem_statement = _required_string(record, "problem_statement", prefix=prefix)
        fail_to_pass = decode_selectors(
            _first_present(record, "FAIL_TO_PASS", "fail_to_pass", "f2p"),
            field_name="FAIL_TO_PASS",
            prefix=prefix,
        )
        pass_to_pass = decode_selectors(
            _first_present(record, "PASS_TO_PASS", "pass_to_pass", "p2p"),
            field_name="PASS_TO_PASS",
            prefix=prefix,
            allow_empty=True,
        )
        if not fail_to_pass:
            raise AdapterError(
                f"{prefix}FAIL_TO_PASS must contain at least one selector"
            )
        return cls(
            instance_id=instance_id,
            repo=repo,
            base_commit=base_commit,
            patch=patch,
            problem_statement=problem_statement,
            fail_to_pass=tuple(fail_to_pass),
            pass_to_pass=tuple(pass_to_pass),
            version=_optional_string(record.get("version")),
            environment_setup_commit=_optional_string(
                record.get("environment_setup_commit")
            ),
            repo_url=_optional_string(
                _first_present(record, "repo_url", "repository_url")
            ),
            test_patch=_optional_string(record.get("test_patch")),
            raw=dict(record),
        )

    @property
    def source_url(self) -> str:
        """Return the explicit local/remote source, or the canonical GitHub URL."""
        if self.repo_url:
            if self.repo_url.startswith("file://"):
                return self.repo_url[7:]
            return self.repo_url
        return f"https://github.com/{self.repo}.git"


@dataclass(frozen=True)
class PatchHunk:
    """One unified-diff hunk and the changed line numbers in each side."""

    path: str
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    section: str
    changed_old_lines: tuple[int, ...]
    changed_new_lines: tuple[int, ...]


@dataclass(frozen=True)
class PatchAnalysis:
    """Validated patch structure retaining every touched path and hunk."""

    path: str
    hunks: tuple[PatchHunk, ...]
    files: Mapping[str, tuple[PatchHunk, ...]] = field(default_factory=dict)
    added_paths: tuple[str, ...] = ()
    deleted_paths: tuple[str, ...] = ()
    renamed_paths: tuple[tuple[str, str], ...] = ()

    @property
    def paths(self) -> tuple[str, ...]:
        """Return touched paths in patch order."""
        return tuple(self.files)


@dataclass(frozen=True)
class MaterializedWorkspace:
    """The two checkout worlds used by conversion and oracle verification."""

    instance_id: str
    base_dir: Path
    gold_dir: Path
    target_rel: str
    cache_key: str


@dataclass(frozen=True)
class CommandResult:
    """Bounded subprocess evidence returned by the two-sided verifier."""

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def output(self) -> str:
        return (self.stdout + self.stderr).strip()

    @property
    def inner_returncode(self) -> int | None:
        match = re.search(r"exit=(\d+)", self.output)
        return int(match.group(1)) if match else None


@dataclass(frozen=True)
class VerificationResult:
    """Parent/red and gold/green evidence for one adapted row."""

    instance_id: str
    parent: CommandResult
    gold: CommandResult

    @property
    def passed(self) -> bool:
        """Require a real inner pytest failure on parent and a clean gold pass."""
        return (
            not self.parent.timed_out
            and not self.gold.timed_out
            and self.parent.returncode == 1
            and self.parent.inner_returncode == 1
            and self.gold.returncode == 0
            and self.gold.inner_returncode is None
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "instance_id": self.instance_id,
            "passed": self.passed,
            "parent": _command_dict(self.parent),
            "gold": _command_dict(self.gold),
        }


@dataclass(frozen=True)
class BuildReport:
    """Deterministic output summary emitted by build_corpus."""

    emitted: tuple[dict[str, object], ...]
    skipped: tuple[dict[str, object], ...]
    reason_counts: Mapping[str, int]


def _first_present(record: Mapping[str, object], *keys: str) -> object:
    for key in keys:
        if key in record:
            return record[key]
    return None


def _required_string(
    record: Mapping[str, object], key: str, *, prefix: str = ""
) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AdapterError(f"{prefix}{key} must be a non-empty string")
    return value


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _decode_jsonish(value: object, *, field_name: str, prefix: str) -> object:
    if value is None:
        return []
    if not isinstance(value, str):
        return value
    if not value.strip():
        return []
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            raise AdapterError(
                f"{prefix}{field_name} must be a JSON/Python list, got {value[:80]!r}"
            ) from exc


def decode_selectors(
    value: object,
    *,
    field_name: str,
    prefix: str = "",
    allow_empty: bool = False,
) -> list[str]:
    """Decode upstream list strings while preserving selector text and parameters."""
    decoded = _decode_jsonish(value, field_name=field_name, prefix=prefix)
    if isinstance(decoded, str):
        decoded = [decoded]
    if not isinstance(decoded, (list, tuple)):
        raise AdapterError(f"{prefix}{field_name} must be a list of selectors")
    selectors: list[str] = []
    for index, selector in enumerate(decoded):
        if not isinstance(selector, str) or not selector:
            raise AdapterError(
                f"{prefix}{field_name}[{index}] must be a non-empty string"
            )
        if "\x00" in selector or "\n" in selector or "\r" in selector:
            raise AdapterError(f"{prefix}{field_name}[{index}] contains a newline/NUL")
        # SWE-bench stores opaque runner arguments here; option-looking values are
        # valid data (notably prose-like PASS_TO_PASS entries beginning with "--").
        normalized = selector.replace("\\", "/")
        selectors.append(normalized)
    if not allow_empty and not selectors:
        raise AdapterError(f"{prefix}{field_name} must contain at least one selector")
    return selectors


def load_instances(source: Path | str) -> list[SWEInstance]:
    """Load and validate a local JSONL source; no network access occurs here."""
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"SWE-bench source does not exist: {path}")
    instances: list[SWEInstance] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AdapterError(f"line {line_no}: invalid JSON: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise AdapterError(f"line {line_no}: instance must be a JSON object")
            record = cast(dict[str, object], value)
            instance = SWEInstance.from_mapping(record, line_no=line_no)
            if instance.instance_id in seen:
                raise AdapterError(f"duplicate instance_id: {instance.instance_id}")
            seen.add(instance.instance_id)
            instances.append(instance)
    return instances


def _patch_path(raw: str) -> str | None:
    token = raw.split("\t", 1)[0].strip()
    if token.startswith('"'):
        try:
            parts = shlex.split(token)
            token = parts[0] if parts else token
        except ValueError:
            pass
    if token == "/dev/null":
        return None
    if token.startswith("a/") or token.startswith("b/"):
        token = token[2:]
    token = token.replace("\\", "/")
    try:
        return normalize_repo_path(token, field="gold patch path")
    except MultiFileRowError as exc:
        raise SkipInstance(
            "unsafe_patch_path", str(exc)
        )


def _patch_paths(patch: str) -> tuple[str, ...]:
    paths: list[str] = []
    old_path: str | None = None
    for line in patch.splitlines():
        if line.startswith("--- "):
            old_path = _patch_path(line[4:])
        elif line.startswith("+++ "):
            new_path = _patch_path(line[4:])
            path = new_path or old_path
            if path and path not in paths:
                paths.append(path)
            old_path = None
    return tuple(paths)


def analyze_gold_patch(patch: str) -> PatchAnalysis:
    """Parse every gold unified-diff file/hunk and validate its supported shape."""
    if not patch.strip():
        raise SkipInstance("missing_gold_patch", "patch is empty")
    if "GIT binary patch" in patch or "Binary files " in patch:
        raise SkipInstance("binary_patch", "binary patches have no Python source hunk")

    files: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    current_hunk: dict[str, object] | None = None

    for line in patch.splitlines():
        if line.startswith("diff --git "):
            current = {"old": None, "new": None, "hunks": []}
            files.append(current)
            current_hunk = None
            continue
        if current is None and (line.startswith("--- ") or line.startswith("+++ ")):
            current = {"old": None, "new": None, "hunks": []}
            files.append(current)
        if current is None:
            continue
        if current_hunk is None and line.startswith("--- "):
            if current["old"] is not None and current["new"] is not None:
                current = {"old": None, "new": None, "hunks": []}
                files.append(current)
            current["old"] = _patch_path(line[4:])
            continue
        if current_hunk is None and line.startswith("+++ "):
            current["new"] = _patch_path(line[4:])
            continue
        match = _HUNK_RE.match(line)
        if match:
            current_hunk = {
                "old_start": int(match.group("old_start")),
                "old_count": int(match.group("old_count") or "1"),
                "new_start": int(match.group("new_start")),
                "new_count": int(match.group("new_count") or "1"),
                "section": match.group("section").strip(),
                "old_line": int(match.group("old_start")),
                "new_line": int(match.group("new_start")),
                "changed_old": [],
                "changed_new": [],
            }
            cast(list[dict[str, object]], current["hunks"]).append(current_hunk)
            if current_hunk["old_count"] == 0 and current_hunk["new_count"] == 0:
                current_hunk = None
            continue
        if current_hunk is None or line.startswith("\\"):
            continue
        old_line = cast(int, current_hunk["old_line"])
        new_line = cast(int, current_hunk["new_line"])
        if line.startswith("+"):
            cast(list[int], current_hunk["changed_new"]).append(new_line)
            current_hunk["new_line"] = new_line + 1
        elif line.startswith("-"):
            cast(list[int], current_hunk["changed_old"]).append(old_line)
            current_hunk["old_line"] = old_line + 1
        elif line.startswith(" "):
            current_hunk["old_line"] = old_line + 1
            current_hunk["new_line"] = new_line + 1
        else:
            continue
        if cast(int, current_hunk["old_line"]) >= cast(
            int, current_hunk["old_start"]
        ) + cast(int, current_hunk["old_count"]) and cast(
            int, current_hunk["new_line"]
        ) >= cast(int, current_hunk["new_start"]) + cast(
            int, current_hunk["new_count"]
        ):
            current_hunk = None

    parsed_files: list[tuple[str | None, str | None, list[PatchHunk]]] = []
    path_hunks: dict[str, list[PatchHunk]] = {}
    added_paths: list[str] = []
    deleted_paths: list[str] = []
    renamed_paths: list[tuple[str, str]] = []
    for record in files:
        old_path = cast(str | None, record["old"])
        new_path = cast(str | None, record["new"])
        path = new_path or old_path
        if path is None:
            continue
        raw_hunks = cast(list[dict[str, object]], record["hunks"])
        hunks = tuple(
            PatchHunk(
                path=path,
                old_start=cast(int, hunk["old_start"]),
                old_count=cast(int, hunk["old_count"]),
                new_start=cast(int, hunk["new_start"]),
                new_count=cast(int, hunk["new_count"]),
                section=cast(str, hunk["section"]),
                changed_old_lines=tuple(cast(list[int], hunk["changed_old"])),
                changed_new_lines=tuple(cast(list[int], hunk["changed_new"])),
            )
            for hunk in raw_hunks
        )
        parsed_files.append((old_path, new_path, list(hunks)))
        path_hunks.setdefault(path, []).extend(hunks)
        if old_path is None and new_path is not None:
            added_paths.append(path)
        if new_path is None and old_path is not None:
            deleted_paths.append(path)
        if old_path is not None and new_path is not None and old_path != new_path:
            renamed_paths.append((old_path, new_path))

    if not path_hunks:
        raise SkipInstance("no_patch_files", "gold patch has no file headers")
    if any(not hunks for _old_path, _new_path, hunks in parsed_files):
        raise SkipInstance(
            "unsupported_patch_shape",
            "every patched file must contain at least one unified-diff hunk",
        )
    if deleted_paths or any(
        hunk.new_count == 0 and hunk.old_count > 0
        for hunks in path_hunks.values()
        for hunk in hunks
    ):
        raise SkipInstance("deleted_target", "the patched target file is deleted")
    if renamed_paths:
        old_path, new_path = renamed_paths[0]
        raise SkipInstance(
            "unsupported_patch_shape",
            f"renames are not representable as complete file rows: {old_path} -> {new_path}",
        )
    non_python = sorted(path for path in path_hunks if not path.endswith(".py"))
    if non_python:
        raise SkipInstance(
            "non_python_source",
            "patch touches non-Python files: " + ", ".join(non_python),
        )
    paths = tuple(path_hunks)
    all_hunks = tuple(hunk for path in paths for hunk in path_hunks[path])
    return PatchAnalysis(
        path=paths[0],
        hunks=all_hunks,
        files={path: tuple(path_hunks[path]) for path in paths},
        added_paths=tuple(added_paths),
        deleted_paths=tuple(deleted_paths),
        renamed_paths=tuple(renamed_paths),
    )


def _python_functions(source: str) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise SkipInstance(
            "invalid_python_source", f"source does not parse: {exc.msg}"
        ) from exc
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _function_start(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    return min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)])


def _innermost_function(
    functions: Sequence[ast.FunctionDef | ast.AsyncFunctionDef],
    line_no: int,
) -> str | None:
    owners = [
        node
        for node in functions
        if _function_start(node)
        <= line_no
        <= (getattr(node, "end_lineno", node.lineno) or node.lineno)
    ]
    if not owners:
        return None
    return min(
        owners,
        key=lambda node: (
            (getattr(node, "end_lineno", node.lineno) or node.lineno)
            - _function_start(node),
            -node.lineno,
        ),
    ).name


def derive_function(
    analysis: PatchAnalysis,
    *,
    parent_src: str,
    fixed_src: str,
) -> tuple[str, int]:
    """Derive the touched function from changed lines and primary-hunk context."""
    fixed_functions = _python_functions(fixed_src)
    parent_functions = _python_functions(parent_src)
    names: set[str] = set()
    changed_line = next(
        (line_no for hunk in analysis.hunks for line_no in hunk.changed_new_lines),
        analysis.hunks[0].new_start,
    )
    for hunk in analysis.hunks:
        if hunk.changed_new_lines:
            changed_line = min(changed_line, min(hunk.changed_new_lines))
        for line_no in hunk.changed_new_lines or (hunk.new_start,):
            name = _innermost_function(fixed_functions, line_no)
            if name:
                names.add(name)
        for line_no in hunk.changed_old_lines or (hunk.old_start,):
            name = _innermost_function(parent_functions, line_no)
            if name:
                names.add(name)
        names.update(
            match.group("name") for match in _FUNCTION_RE.finditer(hunk.section)
        )
    if len(names) > 1:
        raise SkipInstance(
            "ambiguous_source_mapping",
            f"gold patch touches multiple functions: {', '.join(sorted(names))}",
        )
    if not names:
        raise SkipInstance(
            "no_function_hunk",
            f"primary hunk does not identify a Python function in {analysis.path}",
        )
    return next(iter(names)), changed_line


@dataclass(frozen=True)
class _FunctionRef:
    """A function's qualified name and complete source span."""

    name: str
    start: int
    end: int


class _FunctionRefCollector(ast.NodeVisitor):
    """Collect top-level, class, and nested functions with stable names."""

    def __init__(self) -> None:
        self._scope: list[str] = []
        self.refs: list[_FunctionRef] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        name = ".".join([*self._scope, node.name])
        start = _function_start(node)
        end = getattr(node, "end_lineno", node.lineno) or node.lineno
        self.refs.append(_FunctionRef(name=name, start=start, end=end))
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()


def _function_refs(source: str, *, path: str) -> tuple[_FunctionRef, ...]:
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        raise SkipInstance(
            "invalid_python_source", f"{path} does not parse: {exc.msg}"
        ) from exc
    collector = _FunctionRefCollector()
    collector.visit(tree)
    return tuple(collector.refs)


def _function_ref_at_line(
    refs: Sequence[_FunctionRef],
    line_no: int,
) -> _FunctionRef | None:
    owners = [ref for ref in refs if ref.start <= line_no <= ref.end]
    if not owners:
        return None
    return min(owners, key=lambda ref: (ref.end - ref.start, -ref.start))


def derive_changed_functions(
    analysis: PatchAnalysis,
    *,
    parent_sources: Mapping[str, str],
    fixed_sources: Mapping[str, str],
) -> dict[str, list[str]]:
    """Map every changed hunk to all touched qualified Python functions.

    A hunk with no owner on either side is not safe to attribute to a repair
    entry point; it is named as an ambiguous source mapping instead of being
    guessed into a function.
    """
    changed: dict[str, list[str]] = {}
    for path, hunks in analysis.files.items():
        parent_refs = _function_refs(parent_sources[path], path=path)
        fixed_refs = _function_refs(fixed_sources[path], path=path)
        names: set[str] = set()
        for hunk in hunks:
            hunk_names: set[str] = set()
            for line_no in hunk.changed_old_lines:
                ref = _function_ref_at_line(parent_refs, line_no)
                if ref is not None:
                    hunk_names.add(ref.name)
            for line_no in hunk.changed_new_lines:
                ref = _function_ref_at_line(fixed_refs, line_no)
                if ref is not None:
                    hunk_names.add(ref.name)
            context_names = {
                match.group("name") for match in _FUNCTION_RE.finditer(hunk.section)
            }
            if context_names:
                for context_name in context_names:
                    qualified = [
                        ref.name
                        for ref in (*parent_refs, *fixed_refs)
                        if ref.name == context_name or ref.name.endswith(f".{context_name}")
                    ]
                    exact = [name for name in qualified if name == context_name]
                    hunk_names.update(exact or qualified or {context_name})
            if not hunk_names:
                raise SkipInstance(
                    "ambiguous_source_mapping",
                    f"hunk in {path} does not map to a Python function",
                )
            names.update(hunk_names)
        if not names:
            raise SkipInstance(
                "ambiguous_source_mapping",
                f"no changed function could be mapped in {path}",
            )
        changed[path] = sorted(names)
    return changed


def _single_entry(
    changed_functions: Mapping[str, Sequence[str]],
) -> dict[str, str] | None:
    entries = [
        {"path": path, "fn": function}
        for path, functions in changed_functions.items()
        for function in functions
    ]
    return entries[0] if len(entries) == 1 else None


def _safe_slug(value: str) -> str:
    slug = _INSTANCE_ID_RE.sub("_", value).strip("._-").lower()
    return slug[:64] or "instance"


def runner_path(instance_id: str) -> str:
    digest = hashlib.sha256(instance_id.encode("utf-8")).hexdigest()[:10]
    return f"tests/test_swebench_{_safe_slug(instance_id)}_{digest}.py"


def _django_test_labels(selectors: Sequence[str]) -> tuple[str, ...]:
    labels: list[str] = []
    for selector in selectors:
        match = re.fullmatch(
            r"(?P<method>[A-Za-z_]\w*) \((?P<class_label>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\)",
            selector,
        )
        labels.append(
            f"{match.group('class_label')}.{match.group('method')}"
            if match
            else selector
        )
    return tuple(labels)


def make_runner_test(
    selectors: Sequence[str], *, repo: str | None = None
) -> str:
    """Generate the overlay file expected by the local repair/convergence harness."""
    django = repo == "django/django"
    runner_selectors = _django_test_labels(selectors) if django else tuple(selectors)
    selector_literal = json.dumps(list(runner_selectors), ensure_ascii=False)
    command = (
        '[sys.executable, "tests/runtests.py", "--verbosity=1", "--noinput", *FAIL_TO_PASS]'
        if django
        else '[sys.executable, "-m", "pytest", "-q", "--no-header", "--", *FAIL_TO_PASS]'
    )
    return f'''"""Generated SWE-bench FAIL_TO_PASS bridge; grader-only overlay."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

FAIL_TO_PASS = {selector_literal}
REPO_ROOT = Path(__file__).resolve().parents[1]

def test_swebench_fail_to_pass() -> None:
    env = {{
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            filter(None, (str(REPO_ROOT), os.environ.get("PYTHONPATH")))
        ),
    }}
    completed = subprocess.run(
        {command},
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    combined = (completed.stdout + completed.stderr).strip()
    failing_tail = "\\n".join(combined.splitlines()[-40:])
    assert completed.returncode == 0, (
        "SWE-bench FAIL_TO_PASS selectors failed "
        f"(exit={{completed.returncode}}).\\n"
        f"selectors={{FAIL_TO_PASS!r}}\\n"
        f"failing tail:\\n{{failing_tail}}"
    )
'''


def _command_dict(result: CommandResult) -> dict[str, object]:
    return {
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _completed_process(
    args: Sequence[str],
    *,
    cwd: Path | None = None,
    input_text: str | None = None,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(args),
            cwd=str(cwd) if cwd else None,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise AdapterError(f"required executable is unavailable: {args[0]}") from exc


def _git_failure(
    label: str,
    process: subprocess.CompletedProcess[str],
    *,
    redactions: Sequence[str] = (),
) -> AdapterError:
    output = (process.stdout + process.stderr).strip().splitlines()
    detail = " | ".join(output[-4:])[:800] or "no git output"
    for value in redactions:
        if value:
            detail = detail.replace(value, "<redacted-repository>")
    return AdapterError(f"{label} failed with exit {process.returncode}: {detail}")


class WorkspaceMaterializer:
    """Materialize base/gold worlds in content-addressed cache directories.

    The cache is intentionally separate from the project checkout. A bare repository is
    cloned once, then each instance gets two git worktrees, so no existing project files
    are reset, checked out over, or rewritten.
    """

    def __init__(self, cache_dir: Path | str, *, git_timeout: int = 900) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.git_timeout = git_timeout

    def materialize(self, instance: SWEInstance) -> MaterializedWorkspace:
        analysis = analyze_gold_patch(instance.patch)
        if analysis.added_paths:
            raise SkipInstance(
                "target_missing",
                "new target files have no complete parent source: "
                + ", ".join(analysis.added_paths),
            )
        test_paths = set(_patch_paths(instance.test_patch or ""))
        overlap = sorted(set(analysis.paths) & test_paths)
        if overlap:
            raise SkipInstance(
                "test_patch_overlaps_target",
                "hidden test patch also changes code paths: " + ", ".join(overlap),
            )
        source_url = instance.source_url
        repo_key = hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:20]
        repo_dir = (
            self.cache_dir / "repos" / f"{_safe_slug(instance.repo)}-{repo_key}.git"
        )
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        if not repo_dir.exists():
            staging_parent = Path(
                tempfile.mkdtemp(prefix=".swebench-clone-", dir=repo_dir.parent)
            )
            staged_repo = staging_parent / "repo.git"
            try:
                process = _completed_process(
                    ["git", "clone", "--bare", source_url, str(staged_repo)],
                    timeout=self.git_timeout,
                )
                if process.returncode != 0:
                    raise _git_failure("git clone", process, redactions=(source_url,))
                if repo_dir.exists():
                    raise AdapterError(f"repository cache collision: {repo_dir}")
                os.replace(staged_repo, repo_dir)
            finally:
                shutil.rmtree(staging_parent, ignore_errors=True)

        cache_key = hashlib.sha256(
            json.dumps(
                {
                    "instance_id": instance.instance_id,
                    "repo": instance.repo,
                    "source_url": source_url,
                    "base_commit": instance.base_commit,
                    "environment_setup_commit": instance.environment_setup_commit,
                    "patch": instance.patch,
                    "test_patch": instance.test_patch,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:24]
        instance_dir = self.cache_dir / "instances" / cache_key
        base_dir = instance_dir / "base"
        gold_dir = instance_dir / "gold"
        test_patch_digest = hashlib.sha256(
            (instance.test_patch or "").encode("utf-8")
        ).hexdigest()
        base_marker = instance_dir / "base.ready"
        if not base_dir.exists():
            self._add_worktree(repo_dir, base_dir, instance.base_commit)
            self._apply_patch(base_dir, instance.test_patch, "test patch")
            base_marker.write_text(test_patch_digest, encoding="utf-8")
        elif (
            not base_marker.is_file()
            or base_marker.read_text(encoding="utf-8") != test_patch_digest
        ):
            raise AdapterError(f"incomplete base cache: {base_dir}")
        self._assert_head(base_dir, instance.base_commit, "base")
        marker = instance_dir / "gold.ready"
        gold_digest = hashlib.sha256(
            f"{test_patch_digest}:{instance.patch}".encode("utf-8")
        ).hexdigest()
        if not gold_dir.exists():
            self._add_worktree(repo_dir, gold_dir, instance.base_commit)
            self._apply_patch(gold_dir, instance.test_patch, "gold test patch")
            self._apply_patch(gold_dir, instance.patch, "gold code patch")
            marker.write_text(gold_digest, encoding="utf-8")
        elif not marker.is_file() or marker.read_text(encoding="utf-8") != gold_digest:
            raise AdapterError(f"incomplete gold cache: {gold_dir}")
        return MaterializedWorkspace(
            instance_id=instance.instance_id,
            base_dir=base_dir,
            gold_dir=gold_dir,
            target_rel=analysis.path,
            cache_key=cache_key,
        )

    def _apply_patch(self, worktree: Path, patch: str | None, label: str) -> None:
        if not patch:
            return
        check = _completed_process(
            ["git", "apply", "--check", "--whitespace=nowarn", "-"],
            cwd=worktree,
            input_text=patch,
            timeout=self.git_timeout,
        )
        if check.returncode != 0:
            raise _git_failure(f"{label} check", check)
        apply = _completed_process(
            ["git", "apply", "--whitespace=nowarn", "-"],
            cwd=worktree,
            input_text=patch,
            timeout=self.git_timeout,
        )
        if apply.returncode != 0:
            raise _git_failure(f"{label} apply", apply)

    def _add_worktree(self, repo_dir: Path, destination: Path, commit: str) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        process = _completed_process(
            [
                "git",
                "--git-dir",
                str(repo_dir),
                "worktree",
                "add",
                "--detach",
                str(destination),
                commit,
            ],
            timeout=self.git_timeout,
        )
        if process.returncode != 0:
            raise _git_failure("git worktree add", process)

    def _assert_head(self, worktree: Path, expected: str, label: str) -> None:
        process = _completed_process(
            ["git", "-C", str(worktree), "rev-parse", "HEAD"],
            timeout=60,
        )
        if process.returncode != 0:
            raise _git_failure(f"{label} checkout validation", process)
        actual = process.stdout.strip()
        if actual != expected and not actual.startswith(expected):
            raise AdapterError(f"{label} checkout is {actual}, expected {expected}")


def _read_source(root: Path, rel: str, *, label: str) -> str:
    path = root / PurePosixPath(rel)
    try:
        resolved = path.resolve()
        if root.resolve() not in resolved.parents:
            raise AdapterError(f"{label} target escapes repository root: {rel}")
        return resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise SkipInstance(
            "non_text_source", f"{label} target is not UTF-8 text: {rel}"
        ) from exc
    except FileNotFoundError as exc:
        raise SkipInstance(
            "target_missing", f"{label} target is absent at {rel}"
        ) from exc


def _selector_source_path(selector: str) -> str | None:
    """Extract a pytest file path when an F2P selector carries one."""
    candidate = selector.split("::", 1)[0]
    if not candidate.endswith(".py"):
        return None
    try:
        return normalize_repo_path(candidate)
    except MultiFileRowError:
        return None


def _read_oracle_test_files(
    instance: SWEInstance,
    *,
    workspace: MaterializedWorkspace,
    code_paths: Sequence[str],
) -> dict[str, str]:
    """Read grader-side visible test sources without exposing them to model input."""
    patched_test_paths = list(_patch_paths(instance.test_patch or ""))
    candidate_paths = [*patched_test_paths]
    if not candidate_paths:
        candidate_paths.extend(
            path
            for selector in instance.fail_to_pass
            if (path := _selector_source_path(selector)) is not None
        )
    code_path_set = set(code_paths)
    test_files: dict[str, str] = {}
    for path in candidate_paths:
        if path in code_path_set:
            raise SkipInstance(
                "test_patch_overlaps_target",
                f"oracle test path overlaps code path: {path}",
            )
        if not path.endswith(".py") or path in test_files:
            continue
        test_file = workspace.gold_dir / PurePosixPath(path)
        if not test_file.is_file() and path not in patched_test_paths:
            # A Django-style selector or an externally supplied runner may not
            # have a repository path that can be serialized as a visible file.
            continue
        test_files[path] = _read_source(
            workspace.gold_dir,
            path,
            label="oracle test",
        )
    return dict(sorted(test_files.items()))


def _changed_line(analysis: PatchAnalysis) -> int:
    return next(
        (
            line_no
            for hunk in analysis.hunks
            for line_no in hunk.changed_new_lines
        ),
        analysis.hunks[0].new_start,
    )


def convert_instance_with_workspace(
    instance: SWEInstance,
    *,
    materializer: WorkspaceMaterializer,
) -> tuple[dict[str, object], MaterializedWorkspace]:
    """Materialize one instance and return its row plus the worlds used to build it."""
    analysis = analyze_gold_patch(instance.patch)
    workspace = materializer.materialize(instance)
    parent_sources = {
        path: _read_source(workspace.base_dir, path, label="parent")
        for path in analysis.paths
    }
    fixed_sources = {
        path: _read_source(workspace.gold_dir, path, label="gold")
        for path in analysis.paths
    }
    changed_functions = derive_changed_functions(
        analysis,
        parent_sources=parent_sources,
        fixed_sources=fixed_sources,
    )
    function_count = sum(len(functions) for functions in changed_functions.values())
    if len(analysis.paths) > 1 or function_count > 1:
        test_path = runner_path(instance.instance_id)
        row: dict[str, object] = {
            "instance_id": instance.instance_id,
            "task_id": instance.instance_id,
            "repo": instance.repo,
            "sha": instance.base_commit,
            "provenance_sha": instance.base_commit,
            "base_commit": instance.base_commit,
            "version": instance.version,
            "environment_setup_commit": instance.environment_setup_commit,
            "msg": instance.problem_statement,
            "msg_source": "swebench.problem_statement",
            "symptom": instance.problem_statement,
            "symptom_source": "swebench.problem_statement",
            "family": "swebench_lite_multi_file",
            "shape": "multi_file",
            "tier": "external_benchmark",
            "strict_tier": False,
            "expressible_now": True,
            "changed_line": _changed_line(analysis),
            "changed_lines": sum(
                len(hunk.changed_new_lines) for hunk in analysis.hunks
            ),
            "hunks": len(analysis.hunks),
            "content_tokens": [],
            "content_token_count": 0,
            "leak_findings": [],
            "original_subject_len": len(instance.problem_statement),
            "files": {
                path: {
                    "parent_src": parent_sources[path],
                    "fixed_src": fixed_sources[path],
                }
                for path in analysis.paths
            },
            "oracle": {
                "id": f"oracle:{instance.instance_id}",
                "selectors": list(instance.fail_to_pass),
                "test_files": _read_oracle_test_files(
                    instance,
                    workspace=workspace,
                    code_paths=analysis.paths,
                ),
            },
            "changed_functions": changed_functions,
            "test_src": make_runner_test(instance.fail_to_pass, repo=instance.repo),
            "test_path": test_path,
            "fail_to_pass": list(instance.fail_to_pass),
            "pass_to_pass": list(instance.pass_to_pass),
            "python": None,
            "source_dataset": DATASET_ID,
        }
        entry = _single_entry(changed_functions)
        if entry is not None:
            row["entry"] = entry
        try:
            return normalize_multifile_row(row), workspace
        except MultiFileRowError as exc:
            raise AdapterError(f"multi_file row failed schema validation: {exc}") from exc

    parent_src = parent_sources[analysis.path]
    fixed_src = fixed_sources[analysis.path]
    fn, changed_line = derive_function(
        analysis,
        parent_src=parent_src,
        fixed_src=fixed_src,
    )
    test_path = runner_path(instance.instance_id)
    row: dict[str, object] = {
        "instance_id": instance.instance_id,
        "fn": fn,
        "rel": analysis.path,
        "path": analysis.path,
        "repo": instance.repo,
        "sha": instance.base_commit,
        "provenance_sha": instance.base_commit,
        "base_commit": instance.base_commit,
        "version": instance.version,
        "environment_setup_commit": instance.environment_setup_commit,
        "msg": instance.problem_statement,
        "msg_source": "swebench.problem_statement",
        "symptom": instance.problem_statement,
        "symptom_source": "swebench.problem_statement",
        "family": "swebench_lite_single_function",
        "shape": "single_function",
        "tier": "external_benchmark",
        "strict_tier": False,
        "expressible_now": True,
        "changed_line": changed_line,
        "changed_lines": sum(len(hunk.changed_new_lines) for hunk in analysis.hunks),
        "hunks": len(analysis.hunks),
        "content_tokens": [],
        "content_token_count": 0,
        "leak_findings": [],
        "original_subject_len": len(instance.problem_statement),
        "parent_src": parent_src,
        "fixed_src": fixed_src,
        "test_src": make_runner_test(instance.fail_to_pass, repo=instance.repo),
        "test_path": test_path,
        "fail_to_pass": list(instance.fail_to_pass),
        "pass_to_pass": list(instance.pass_to_pass),
        "python": None,
        "source_dataset": DATASET_ID,
    }
    return row, workspace


def convert_instance(
    instance: SWEInstance,
    *,
    materializer: WorkspaceMaterializer,
) -> dict[str, object]:
    """Convert one validated instance to the existing repair-row shape."""
    row, _workspace = convert_instance_with_workspace(
        instance, materializer=materializer
    )
    return row


def verify_instance(
    row: Mapping[str, object],
    workspace: MaterializedWorkspace | Path | str | None = None,
    *,
    python_executable: str | None = None,
    timeout: int = 360,
) -> VerificationResult:
    """Run the generated overlay on parent and gold worlds and enforce the oracle law."""
    instance_id = (
        _required_string(row, "instance_id")
        if "instance_id" in row
        else _required_string(row, "fn")
    )
    if workspace is None:
        candidate = row.get("_workspace")
        workspace = (
            candidate
            if isinstance(candidate, (MaterializedWorkspace, str, Path))
            else None
        )
    if workspace is None:
        raise MaterializationRequired(
            "verify_instance needs a MaterializedWorkspace or workspace path; "
            "conversion keeps grader worlds separate from serialized rows"
        )
    if isinstance(workspace, MaterializedWorkspace):
        base_dir, gold_dir = workspace.base_dir, workspace.gold_dir
    else:
        root = Path(workspace)
        base_dir = root / "base" if (root / "base").is_dir() else root
        gold_dir = root / "gold"
        if not gold_dir.is_dir():
            raise AdapterError(f"gold workspace does not exist: {gold_dir}")
    test_path_value = row.get("test_path")
    test_path = (
        test_path_value
        if isinstance(test_path_value, str)
        else runner_path(instance_id)
    )
    test_src_value = row.get("test_src")
    if not isinstance(test_src_value, str):
        raise AdapterError("row.test_src must contain the generated runner file")
    executable = python_executable or selected_test_python()
    parent = _run_overlay(base_dir, test_path, test_src_value, executable, timeout)
    gold = _run_overlay(gold_dir, test_path, test_src_value, executable, timeout)
    return VerificationResult(instance_id=instance_id, parent=parent, gold=gold)


@dataclass(frozen=True)
class ConversionSummary:
    """Patch-shape census produced without cloning or reading a repository."""

    considered: int
    would_emit_multi_file: int
    would_emit_single_file: int
    skipped: int
    reason_counts: Mapping[str, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "considered": self.considered,
            "would_emit_multi_file": self.would_emit_multi_file,
            "would_emit_single_file": self.would_emit_single_file,
            "skipped": self.skipped,
            "reason_counts": dict(self.reason_counts),
        }


def _conversion_function_names(analysis: PatchAnalysis) -> dict[str, list[str]]:
    """Use unified-diff function context for a no-checkout shape census."""
    changed: dict[str, list[str]] = {}
    for path, hunks in analysis.files.items():
        names = sorted(
            {
                match.group("name")
                for hunk in hunks
                for match in _FUNCTION_RE.finditer(hunk.section)
            }
        )
        if not names:
            raise SkipInstance(
                "ambiguous_source_mapping",
                f"conversion-only census cannot map a hunk in {path} without checkout source",
            )
        changed[path] = names
    return changed


def summarize_conversion(instances: Sequence[SWEInstance]) -> ConversionSummary:
    """Count row shapes and named skips without materializing repositories."""
    multi_file = 0
    single_file = 0
    skipped = 0
    counts: Counter[str] = Counter()
    for instance in instances:
        try:
            analysis = analyze_gold_patch(instance.patch)
            if analysis.added_paths:
                raise SkipInstance(
                    "target_missing",
                    "new target files have no complete parent source: "
                    + ", ".join(analysis.added_paths),
                )
            code_paths = set(analysis.paths)
            test_paths = set(_patch_paths(instance.test_patch or ""))
            overlap = sorted(code_paths & test_paths)
            if overlap:
                raise SkipInstance(
                    "test_patch_overlaps_target",
                    "hidden test patch also changes code paths: " + ", ".join(overlap),
                )
            changed_functions = _conversion_function_names(analysis)
            function_count = sum(
                len(functions) for functions in changed_functions.values()
            )
            if len(analysis.paths) > 1 or function_count > 1:
                multi_file += 1
            else:
                single_file += 1
        except SkipInstance as exc:
            skipped += 1
            counts[exc.reason] += 1
    return ConversionSummary(
        considered=len(instances),
        would_emit_multi_file=multi_file,
        would_emit_single_file=single_file,
        skipped=skipped,
        reason_counts=dict(sorted(counts.items())),
    )


def _run_overlay(
    root: Path,
    test_path: str,
    test_src: str,
    executable: str,
    timeout: int,
) -> CommandResult:
    relative = PurePosixPath(test_path)
    runner = root / relative
    if root.resolve() not in runner.resolve().parents:
        raise AdapterError(f"runner path escapes workspace: {test_path}")
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_text(test_src, encoding="utf-8")
    try:
        process = subprocess.run(
            [executable, "-m", "pytest", "-q", "--no-header", test_path],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env={
                **os.environ,
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": os.pathsep.join(
                    filter(None, (str(root), os.environ.get("PYTHONPATH")))
                ),
            },
        )
        return CommandResult(process.returncode, process.stdout, process.stderr)
    except subprocess.TimeoutExpired as exc:
        stdout = _decode_output(exc.stdout)
        stderr = _decode_output(exc.stderr)
        return CommandResult(1, stdout, stderr, timed_out=True)


def _decode_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def build_corpus(
    instances: Sequence[SWEInstance],
    *,
    output: Path | str,
    skipped_output: Path | str,
    cache_dir: Path | str,
    limit: int,
    materialize: bool,
) -> BuildReport:
    """Emit the first N convertible rows and a complete named skip report."""
    if limit < 1:
        raise AdapterError("limit must be at least 1")
    if not materialize:
        has_convertible_shape = False
        for instance in instances:
            try:
                analyze_gold_patch(instance.patch)
            except SkipInstance:
                continue
            has_convertible_shape = True
            break
        if has_convertible_shape:
            raise MaterializationRequired(
                "SWE-bench source conversion requires checkout access; pass --materialize"
            )

    materializer = WorkspaceMaterializer(cache_dir)
    emitted: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    for instance in instances:
        try:
            analyze_gold_patch(instance.patch)
        except SkipInstance as exc:
            record = {
                "instance_id": instance.instance_id,
                "repo": instance.repo,
                "status": "SKIPPED",
                "reason": exc.reason,
                "detail": exc.detail,
            }
            skipped.append(record)
            counts[exc.reason] += 1
            continue
        if len(emitted) >= limit:
            record = {
                "instance_id": instance.instance_id,
                "repo": instance.repo,
                "status": "SKIPPED",
                "reason": "limit_reached",
                "detail": f"first {limit} convertible instances already emitted",
            }
            skipped.append(record)
            counts["limit_reached"] += 1
            continue
        try:
            row = convert_instance(instance, materializer=materializer)
        except SkipInstance as exc:
            record = {
                "instance_id": instance.instance_id,
                "repo": instance.repo,
                "status": "SKIPPED",
                "reason": exc.reason,
                "detail": exc.detail,
            }
            skipped.append(record)
            counts[exc.reason] += 1
            continue
        except AdapterError as exc:
            record = {
                "instance_id": instance.instance_id,
                "repo": instance.repo,
                "status": "SKIPPED",
                "reason": "materialization_failed",
                "detail": str(exc)[:1000],
            }
            skipped.append(record)
            counts["materialization_failed"] += 1
            continue
        emitted.append(row)

    _write_jsonl(Path(output), emitted)
    skipped_doc = {
        "schema": "swebench-lite-skipped/1",
        "status": "SKIPPED",
        "dataset_id": DATASET_ID,
        "considered": len(instances),
        "emitted": len(emitted),
        "skipped": len(skipped),
        "reason_counts": dict(sorted(counts.items())),
        "items": skipped,
    }
    skipped_path = Path(skipped_output)
    skipped_path.parent.mkdir(parents=True, exist_ok=True)
    skipped_path.write_text(
        json.dumps(skipped_doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return BuildReport(tuple(emitted), tuple(skipped), dict(sorted(counts.items())))


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a local SWE-bench-Lite JSONL into repair rows."
    )
    parser.add_argument(
        "--source", required=True, type=Path, help="local JSONL; never fetched here"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skipped-output", type=Path, default=DEFAULT_SKIPPED_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--limit", type=int, default=10, help="number of convertible rows to emit"
    )
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="explicitly allow git clone/worktree creation in --cache-dir",
    )
    parser.add_argument(
        "--conversion-only",
        action="store_true",
        help="census patch shapes without cloning repositories or writing rows",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        instances = load_instances(args.source)
        if args.conversion_only:
            summary = summarize_conversion(instances)
            print(json.dumps(summary.as_dict(), sort_keys=True))
            return 0
        report = build_corpus(
            instances,
            output=args.output,
            skipped_output=args.skipped_output,
            cache_dir=args.cache_dir,
            limit=args.limit,
            materialize=args.materialize,
        )
    except (AdapterError, OSError) as exc:
        print(f"swebench-adapter: {exc}", file=sys.stderr)
        return 2
    print(
        f"emitted={len(report.emitted)} skipped={len(report.skipped)} "
        f"reasons={dict(report.reason_counts)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
