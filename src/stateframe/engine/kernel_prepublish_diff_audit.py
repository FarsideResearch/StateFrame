"""Host-owned pre-publish diff and scope audit.

The audit is a truth mechanism: it compares two host-provided workspace
snapshots and never asks the model to classify its own diff.  The normal arm
does not read either snapshot, emit a journal event, or alter the command
policy.  ``STATEFRAME_PREPUBLISH_DIFF_AUDIT=on`` is the explicit opt-in arm.

The module is intentionally independent of git.  A runner supplies the task
base (or the host captures it before the first edit), which keeps dirty
pre-existing state separate from the candidate and prevents an agent from
changing the reference with ``reset`` or ``checkout``.
"""
from __future__ import annotations

import ast
import fnmatch
import hashlib
import json
import os
import re
import shlex
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from difflib import SequenceMatcher


MODE_ENV = "STATEFRAME_PREPUBLISH_DIFF_AUDIT"
THRESHOLD_ENV = "STATEFRAME_PREPUBLISH_NET_DELETION_THRESHOLD"
SCHEMA = "stateframe-prepublish-diff-audit/1"
DEFAULT_NET_DELETION_THRESHOLD = 100

# Keep this vocabulary local rather than passing it through journal.emit().
# The continuation journal owns these fields and rejects callers that supply
# any of them.  The static reserved-field probe also scans this module.
JOURNAL_RESERVED_FIELDS = frozenset({
    "schema", "kind", "run_id", "task_id", "event_id", "sequence",
    "timestamp_ns", "journal_format", "previous_hash", "record_hash",
})

_PATH_KEYS = frozenset({
    "path", "paths", "file", "files", "file_path", "file_paths", "filename",
    "filenames", "target", "targets", "target_path", "target_paths", "module",
    "modules", "source_path", "source_paths", "changed_file", "changed_files",
    "affected_file", "affected_files", "editable_paths", "allowed_product_paths",
    "mentioned_files", "mentioned_paths", "where_used_path", "where_used_paths",
    "site_path", "site_paths", "locations", "location",
})
_WHERE_KEYS = frozenset({
    "where_used", "where_used_sites", "sites", "site", "references", "callers",
    "consumers", "counterparts", "locations", "location", "where_used_path",
    "where_used_paths", "site_path", "site_paths",
})
_REPRO_KEYS = frozenset({
    "repro", "repros", "reproduction", "reproductions", "test", "tests",
    "test_path", "test_paths", "test_file", "test_files", "selector", "selectors",
    "nodeid", "node_ids", "evidence", "evidence_ref", "evidence_refs", "repro_path",
    "repro_paths", "public_tests", "public_test_path", "public_test_paths",
    "fail_to_pass", "pass_to_pass", "fail_to_pass_tests", "pass_to_pass_tests",
})
_OBLIGATION_KEYS = frozenset({
    "obligation", "obligations", "obligation_path", "obligation_paths",
    "mentioned_files", "mentioned_paths", "requirements", "requirement",
})

_ARTIFACT_DIRS = frozenset({
    ".pytest_cache", ".ruff_cache", ".mypy_cache", "__pycache__", ".tox", ".nox",
    "htmlcov", "test-results", "measurement-runs", "pytest-of-user", "scratch",
})
_ARTIFACT_NAMES = frozenset({
    ".coverage", "coverage.xml", "junit.xml", "pytestdebug.log", "test-results.xml",
})
_ARTIFACT_PATTERNS = (
    "*.pyc", "*.pyo", "*.coverage", ".coverage.*", "coverage-*.xml",
)

_GIT_MUTATION_SUBCOMMANDS = frozenset({"reset", "checkout", "clean", "restore"})
_GIT_OPTION_ARGUMENTS = frozenset({
    "-C", "--git-dir", "--work-tree", "-c", "--config-env", "--exec-path",
    "--namespace", "--super-prefix",
})
_GIT_REASON = "git state mutation is host-denied; use the typed host restore path"


def mode() -> str:
    """Return the strict switch value; a typo must not silently enable a half-arm."""
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def deletion_threshold() -> int:
    """Return the host threshold for net line deletion."""
    raw = os.environ.get(THRESHOLD_ENV, str(DEFAULT_NET_DELETION_THRESHOLD)).strip()
    try:
        value = int(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{THRESHOLD_ENV} must be a non-negative integer") from error
    if value < 0:
        raise ValueError(f"{THRESHOLD_ENV} must be a non-negative integer")
    return value


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _path_text(value: object) -> str:
    text = str(value).strip().replace("\\", "/")
    if "::" in text:
        text = text.split("::", 1)[0]
    # pytest selectors occasionally arrive as ``file.py:line``.  A colon in a
    # normal POSIX path is legal, so only strip a numeric suffix.
    text = re.sub(r"(?<=\.py):\d+(?::\d+)?$", "", text)
    while text.startswith("./"):
        text = text[2:]
    return text


def _normal_path(value: object, *, root: Path | None = None) -> str | None:
    if not isinstance(value, (str, Path)):
        return None
    raw = str(value).strip().replace("\\", "/")
    if not raw:
        return None
    if root is not None and Path(raw).is_absolute():
        try:
            raw = Path(raw).resolve().relative_to(root.resolve()).as_posix()
        except (OSError, ValueError):
            return None
    raw = _path_text(raw)
    if not raw or raw == ".":
        return None
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts:
        return None
    return path.as_posix()


def _source_bytes(value: object) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, Path):
        return value.read_bytes()
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, Mapping):
        for key in ("source", "content", "text", "bytes"):
            if key in value:
                return _source_bytes(value[key])
    raise TypeError("workspace snapshots must map paths to text or bytes")


def _excluded(path: str, excludes: Sequence[str]) -> bool:
    return any(path == item or path.startswith(item.rstrip("/") + "/") for item in excludes)


def _snapshot(value: object, *, root: Path | None = None,
              exclude: Sequence[object] = ()) -> dict[str, bytes]:
    """Normalize a mapping or directory into a deterministic file snapshot."""
    if isinstance(value, Mapping):
        result: dict[str, bytes] = {}
        for raw_path, payload in value.items():
            path = _normal_path(raw_path, root=root)
            if path is None:
                continue
            if payload is None:
                continue
            exclusions = tuple(
                item for item in (_normal_path(item, root=root) for item in exclude)
                if item is not None
            )
            if not _excluded(path, exclusions):
                result[path] = _source_bytes(payload)
        return result

    if not isinstance(value, (str, Path)):
        raise TypeError("workspace must be a path or a path-to-source mapping")
    directory = Path(value).resolve()
    if not directory.is_dir():
        raise ValueError("workspace path must be an existing directory")
    exclusions = tuple(
        item for item in (_normal_path(item, root=directory) for item in exclude)
        if item is not None
    )
    result = {}
    for path in directory.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            relative = path.relative_to(directory).as_posix()
        except ValueError:
            continue
        if _excluded(relative, exclusions) or ".git" in PurePosixPath(relative).parts:
            continue
        result[relative] = path.read_bytes()
    return result


def snapshot_workspace(root: str | Path, *, exclude: Sequence[object] = ()) -> dict[str, bytes]:
    """Capture a workspace without following symlinks or entering ``.git``."""
    return _snapshot(Path(root), exclude=exclude)


def _tree_digest(tree: Mapping[str, bytes]) -> str:
    return _digest({path: hashlib.sha256(tree[path]).hexdigest() for path in sorted(tree)})


def _text_lines(value: bytes | None) -> list[str]:
    if value is None:
        return []
    return value.decode("utf-8", errors="replace").splitlines()


def _line_stats(before: bytes | None, after: bytes | None) -> tuple[int, int, int, bool]:
    old, new = _text_lines(before), _text_lines(after)
    binary = any(b"\x00" in value for value in (before or b"", after or b""))
    removed = added = 0
    for tag, old_start, old_end, new_start, new_end in SequenceMatcher(
            None, old, new, autojunk=False).get_opcodes():
        if tag in {"delete", "replace"}:
            removed += old_end - old_start
        if tag in {"insert", "replace"}:
            added += new_end - new_start
    return len(old), len(new), max(0, len(old) - len(new)), binary


def _bound_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            for target in targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
                elif isinstance(target, (ast.Tuple, ast.List)):
                    names.update(item.id for item in target.elts if isinstance(item, ast.Name))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".", 1)[0])
    return names


def top_level_symbols(source: bytes | str | None) -> tuple[str, ...]:
    """Return host-observed top-level bindings; syntax failure yields no claim."""
    if source is None:
        return ()
    raw = source if isinstance(source, bytes) else source.encode("utf-8")
    try:
        tree = ast.parse(raw.decode("utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return ()
    return tuple(sorted(_bound_names(tree)))


def _looks_like_path(value: object) -> bool:
    if not isinstance(value, (str, Path)):
        return False
    text = str(value).strip().replace("\\", "/")
    return bool(text and ("/" in text or "." in PurePosixPath(text).name
                          or text.endswith((".py", ".json", ".toml", ".yaml", ".yml"))))


def _path_values(value: object) -> Iterable[str]:
    if isinstance(value, (str, Path)):
        if _looks_like_path(value):
            yield str(value)
        return
    if isinstance(value, Mapping):
        # A task manifest commonly uses {"app.py": {"source": ...}}.
        for key, nested in value.items():
            lowered = str(key).lower().replace("-", "_")
            if lowered in _PATH_KEYS or lowered in _WHERE_KEYS \
                    or lowered in _REPRO_KEYS or lowered in _OBLIGATION_KEYS:
                yield from _path_values(nested)
            elif _looks_like_path(key):
                yield str(key)
            if isinstance(nested, (Mapping, Sequence)) and not isinstance(nested, (str, bytes)):
                yield from _path_values(nested)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            yield from _path_values(item)
    elif isinstance(value, (set, frozenset)):
        for item in value:
            yield from _path_values(item)


def _link_kind(key: str, fallback: str) -> str:
    lowered = key.lower().replace("-", "_")
    if lowered in _WHERE_KEYS or "where_used" in lowered or "counterpart" in lowered:
        return "where_used"
    if lowered in _REPRO_KEYS or "repro" in lowered or "test" in lowered or "evidence" in lowered:
        return "repro"
    if lowered in _OBLIGATION_KEYS or "obligation" in lowered or "requirement" in lowered:
        return "obligation"
    return fallback


def _collect_links(value: object, links: dict[str, set[str]], *, fallback: str) -> None:
    """Collect only path-bearing host fields, never arbitrary prose."""
    def visit(item: object, default: str, source: str = "plan") -> None:
        if isinstance(item, Mapping):
            local = default
            if any(str(key).lower() in _WHERE_KEYS for key in item):
                local = "where_used"
            elif any(str(key).lower() in _REPRO_KEYS for key in item):
                local = "repro"
            elif any(str(key).lower() in _OBLIGATION_KEYS for key in item):
                local = "obligation"
            ident = item.get("id", item.get("obligation_id", item.get("check_id", source)))
            for raw_key, nested in item.items():
                key = str(raw_key).lower().replace("-", "_")
                if key in _PATH_KEYS or key in _WHERE_KEYS or key in _REPRO_KEYS \
                        or key in _OBLIGATION_KEYS:
                    kind = _link_kind(key, local)
                    structured_rows = (
                        isinstance(nested, Mapping)
                        and ("id" in nested or "obligation_id" in nested
                             or any(str(child).lower().replace("-", "_") in _PATH_KEYS
                                    for child in nested))
                    ) or (
                        isinstance(nested, Sequence) and not isinstance(nested, (str, bytes))
                        and any(isinstance(child, Mapping) for child in nested)
                    )
                    if structured_rows:
                        visit(nested, kind, source=str(ident))
                    else:
                        for raw_path in _path_values(nested):
                            path = _normal_path(raw_path)
                            if path is not None:
                                links.setdefault(path, set()).add(f"{kind}:{ident}")
                elif isinstance(nested, Mapping):
                    visit(nested, local, source=str(ident))
                elif isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
                    # Lists of obligation/site rows are meaningful; lists of
                    # source text are not traversed unless a path key handled them.
                    if nested and all(isinstance(child, Mapping) for child in nested):
                        for child in nested:
                            visit(child, local, source=str(ident))
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for child in item:
                visit(child, default, source=source)

    visit(value, fallback)


def _artifact_match(path: str, *, patterns: Sequence[str] = (), harness_paths: Sequence[str] = ()) -> bool:
    normalized = path.replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    lowered = normalized.lower()
    if any(normalized == item or normalized.startswith(item.rstrip("/") + "/")
           for item in harness_paths):
        return True
    if any(part.lower() in _ARTIFACT_DIRS for part in parts):
        return True
    name = PurePosixPath(normalized).name.lower()
    if name in _ARTIFACT_NAMES or name.startswith(".coverage"):
        return True
    if any(fnmatch.fnmatch(lowered, pattern.lower())
           or fnmatch.fnmatch(name, pattern.lower())
           for pattern in (*_ARTIFACT_PATTERNS, *patterns)):
        return True
    if name.endswith(".jsonl") and any(part.lower() in {
            "artifact", "artifacts", "evidence", "run", "runs", "results", "scratch"
    } for part in parts[:-1]):
        return True
    return False


def is_harness_artifact(path: object, *, patterns: Sequence[str] = (),
                        harness_paths: Sequence[str] = ()) -> bool:
    normalized = _normal_path(path)
    return normalized is not None and _artifact_match(
        normalized, patterns=patterns, harness_paths=harness_paths)


@dataclass(frozen=True, slots=True)
class FileDiff:
    path: str
    change: str
    classification: str
    links: tuple[str, ...] = ()
    before_lines: int = 0
    after_lines: int = 0
    net_deletion: int = 0
    removed_lines: int = 0
    added_lines: int = 0
    removed_symbols: tuple[str, ...] = ()
    risk_tags: tuple[str, ...] = ()
    strip: bool = False

    @property
    def blocking(self) -> bool:
        return self.classification in {"unexplained", "destructive"}

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "change": self.change,
            "classification": self.classification,
            "links": list(self.links),
            "before_lines": self.before_lines,
            "after_lines": self.after_lines,
            "net_deletion": self.net_deletion,
            "removed_lines": self.removed_lines,
            "added_lines": self.added_lines,
            "removed_symbols": list(self.removed_symbols),
            "risk_tags": list(self.risk_tags),
            "strip": self.strip,
        }


@dataclass(frozen=True, slots=True)
class AuditResult(Mapping[str, object]):
    schema: str = SCHEMA
    status: str = "OFF"
    files: tuple[FileDiff, ...] = ()
    blocking_obligations: tuple[Mapping[str, object], ...] = ()
    stripped_paths: tuple[str, ...] = ()
    base_digest: str = ""
    current_digest: str = ""
    diff_digest: str = ""
    fired: bool = False

    @classmethod
    def off(cls) -> "AuditResult":
        return cls()

    @property
    def blocked(self) -> bool:
        return self.status == "BLOCKED"

    @property
    def can_publish(self) -> bool:
        return self.status in {"OFF", "PASS"}

    @property
    def publish_paths(self) -> tuple[str, ...]:
        return tuple(item.path for item in self.files if not item.strip)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "status": self.status,
            "fired": self.fired,
            "files": [item.to_dict() for item in self.files],
            "blocking_obligations": [dict(item) for item in self.blocking_obligations],
            "stripped_paths": list(self.stripped_paths),
            "base_digest": self.base_digest,
            "current_digest": self.current_digest,
            "diff_digest": self.diff_digest,
        }

    as_dict = to_dict

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]

    def __iter__(self):
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


def compute_diff(
    base: object,
    current: object,
    *,
    plan: object = None,
    obligations: object = (),
    where_used: object = (),
    repros: object = (),
    deletion_threshold: int = DEFAULT_NET_DELETION_THRESHOLD,
    artifact_patterns: Sequence[str] = (),
    harness_paths: Sequence[object] = (),
) -> AuditResult:
    """Compute a complete host diff without consulting the feature switch."""
    if type(deletion_threshold) is not int or deletion_threshold < 0:
        raise ValueError("deletion_threshold must be a non-negative integer")
    base_tree = _snapshot(base)
    current_tree = _snapshot(current)
    links: dict[str, set[str]] = {}
    _collect_links(plan, links, fallback="plan")
    _collect_links(obligations, links, fallback="obligation")
    _collect_links(where_used, links, fallback="where_used")
    _collect_links(repros, links, fallback="repro")
    normalized_harness = tuple(
        item for item in (_normal_path(value) for value in harness_paths) if item is not None
    )

    rows: list[FileDiff] = []
    for path in sorted(set(base_tree) | set(current_tree)):
        before, after = base_tree.get(path), current_tree.get(path)
        if before == after:
            continue
        if before is None:
            change = "added"
        elif after is None:
            change = "deleted"
        else:
            change = "modified"
        before_lines, after_lines, net_deletion, binary = _line_stats(before, after)
        removed_lines = added_lines = 0
        if not binary:
            old, new = _text_lines(before), _text_lines(after)
            for tag, old_start, old_end, new_start, new_end in SequenceMatcher(
                    None, old, new, autojunk=False).get_opcodes():
                if tag in {"delete", "replace"}:
                    removed_lines += old_end - old_start
                if tag in {"insert", "replace"}:
                    added_lines += new_end - new_start
        removed_symbols = tuple(sorted(
            set(top_level_symbols(before)) - set(top_level_symbols(after))))
        risks: list[str] = []
        if net_deletion > deletion_threshold:
            risks.append("NET_DELETION")
        if removed_symbols:
            risks.append("PRE_EXISTING_SYMBOL_REMOVED")
        if change == "deleted":
            risks.append("FILE_DELETED")
        linked = tuple(sorted(links.get(path, ())))
        artifact = _artifact_match(path, patterns=artifact_patterns,
                                   harness_paths=normalized_harness)
        if artifact:
            classification = "harness_artifact"
        elif linked:
            classification = "explained"
        elif risks:
            classification = "destructive"
        else:
            classification = "unexplained"
        rows.append(FileDiff(
            path=path, change=change, classification=classification, links=linked,
            before_lines=before_lines, after_lines=after_lines, net_deletion=net_deletion,
            removed_lines=removed_lines, added_lines=added_lines,
            removed_symbols=removed_symbols, risk_tags=tuple(risks), strip=artifact,
        ))

    blockers = []
    for item in rows:
        if not item.blocking:
            continue
        kind = "PREPUBLISH_DESTRUCTIVE" if item.classification == "destructive" \
            else "PREPUBLISH_UNEXPLAINED"
        detail = ("host restore or an explicit obligation/where-used/repro link is required"
                  if item.classification == "unexplained" else
                  "host restore or an explicit linked obligation is required for destructive risk")
        blockers.append({
            "id": f"{kind}:{item.path}", "kind": kind, "path": item.path,
            "detail": detail, "risk_tags": list(item.risk_tags),
            "required_action": "justify_or_host_restore",
        })
    status = "BLOCKED" if blockers else "PASS"
    serial = [item.to_dict() for item in rows]
    return AuditResult(
        status=status, files=tuple(rows), blocking_obligations=tuple(blockers),
        stripped_paths=tuple(item.path for item in rows if item.strip),
        base_digest=_tree_digest(base_tree), current_digest=_tree_digest(current_tree),
        diff_digest=_digest(serial), fired=False,
    )


def _emit(journal: object, kind: str, **fields: object) -> None:
    emitter = getattr(journal, "emit", None)
    if callable(emitter):
        emitter(kind, **fields)


def audit_workspace(
    base: object,
    current: object,
    *,
    plan: object = None,
    obligations: object = (),
    where_used: object = (),
    repros: object = (),
    deletion_threshold: int | None = None,
    artifact_patterns: Sequence[str] = (),
    harness_paths: Sequence[object] = (),
    journal: object = None,
) -> AuditResult:
    """Run the switch-gated audit and journal one fired event per invocation."""
    if not enabled():
        return AuditResult.off()
    threshold = (globals()["deletion_threshold"]()
                 if deletion_threshold is None else deletion_threshold)
    result = compute_diff(
        base, current, plan=plan, obligations=obligations, where_used=where_used,
        repros=repros,
        deletion_threshold=threshold,
        artifact_patterns=artifact_patterns, harness_paths=harness_paths,
    )
    _emit(
        journal, "prepublish_diff_audit_fired",
        producer="kernel_prepublish_diff_audit", status=result.status,
        file_count=len(result.files),
        explained_count=sum(item.classification == "explained" for item in result.files),
        unexplained_count=sum(item.classification == "unexplained" for item in result.files),
        destructive_count=sum(item.classification == "destructive" for item in result.files),
        harness_artifact_count=len(result.stripped_paths),
        blocking_obligation_ids=[item["id"] for item in result.blocking_obligations],
        diff_digest=result.diff_digest, dispatch=False,
    )
    return AuditResult(
        schema=result.schema, status=result.status, files=result.files,
        blocking_obligations=result.blocking_obligations, stripped_paths=result.stripped_paths,
        base_digest=result.base_digest, current_digest=result.current_digest,
        diff_digest=result.diff_digest, fired=True,
    )


def stripped_snapshot(result: AuditResult, current: object) -> dict[str, bytes]:
    """Return the candidate publish view with recognized harness artifacts removed."""
    tree = _snapshot(current)
    if result.status == "OFF":
        return tree
    return {path: payload for path, payload in tree.items() if path not in result.stripped_paths}


def strip_harness_artifacts(result: AuditResult, current: object) -> dict[str, bytes]:
    return stripped_snapshot(result, current)


def _safe_root_path(root: Path, relative: str) -> Path:
    root = root.resolve()
    candidate = root / relative
    cursor = root
    for part in PurePosixPath(relative).parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError("restore refuses symlink paths")
    resolved = candidate.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise ValueError("restore path escapes the workspace")
    if candidate == root:
        raise ValueError("restore path must name a file")
    return candidate


def restore_blocking_file(
    root: str | Path,
    result: AuditResult,
    path: str,
    baseline: object,
    *,
    journal: object = None,
) -> dict[str, object]:
    """Host-apply one exact restore for a blocking result.

    New files are removed; deleted or modified pre-existing files are restored
    from the supplied base snapshot.  Symlinks and directories are refused.
    """
    if not isinstance(result, AuditResult):
        raise TypeError("restore requires an AuditResult")
    relative = _normal_path(path)
    if relative is None or not any(item.get("path") == relative for item in result.blocking_obligations):
        raise ValueError("only a blocking audit path may be host-restored")
    base_tree = _snapshot(baseline)
    target = _safe_root_path(Path(root), relative)
    if target.is_symlink() or target.is_dir():
        raise ValueError("host restore only permits regular files")
    if relative not in base_tree:
        if target.exists():
            target.unlink()
        outcome = "REMOVED_NEW_FILE"
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.",
                                         delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(base_tree[relative])
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        outcome = "RESTORED_BASE_FILE"
    receipt = {
        "path": relative, "outcome": outcome, "source": "host_base_snapshot",
        "audit_digest": result.diff_digest,
    }
    _emit(journal, "prepublish_diff_restore_applied",
          producer="kernel_prepublish_diff_audit", path=relative,
          outcome=outcome, audit_digest=result.diff_digest, dispatch=False)
    return receipt


def command_rejection(command: object) -> str | None:
    """Return a typed refusal for git state mutation commands in the ON arm."""
    if not enabled():
        return None
    try:
        words = shlex.split(str(command))
    except ValueError:
        return "PREPUBLISH_COMMAND_REJECTED: malformed command"
    for index, word in enumerate(words):
        if Path(word).name != "git":
            continue
        cursor = index + 1
        while cursor < len(words):
            token = words[cursor]
            if token in _GIT_OPTION_ARGUMENTS:
                cursor += 2
                continue
            if any(token.startswith(option + "=") for option in _GIT_OPTION_ARGUMENTS
                   if option.startswith("--")):
                cursor += 1
                continue
            if token.startswith("-"):
                cursor += 1
                continue
            break
        if cursor < len(words) and words[cursor] in _GIT_MUTATION_SUBCOMMANDS:
            return f"PREPUBLISH_COMMAND_REJECTED: {words[cursor]} ({_GIT_REASON})"
    return None


def command_allowed(command: object) -> bool:
    return command_rejection(command) is None


# Friendly aliases for host adapters and offline probes.
audit_diff = audit_workspace
classify_diff = compute_diff
host_restore = restore_blocking_file
denied_command = command_rejection


__all__ = [
    "AuditResult", "DEFAULT_NET_DELETION_THRESHOLD", "FileDiff", "JOURNAL_RESERVED_FIELDS",
    "MODE_ENV", "SCHEMA", "THRESHOLD_ENV", "audit_diff", "audit_workspace", "classify_diff",
    "command_allowed", "command_rejection", "compute_diff", "deletion_threshold", "denied_command",
    "enabled", "host_restore", "is_harness_artifact", "mode", "restore_blocking_file",
    "snapshot_workspace", "strip_harness_artifacts", "stripped_snapshot", "top_level_symbols",
]
