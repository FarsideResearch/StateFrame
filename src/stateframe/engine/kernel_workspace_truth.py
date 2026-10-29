"""Current-workspace truth probes and centralized fact rendering.

All N7 host absence/read claims pass through :func:`probe_live_workspace`.
The module is standard-library-only and deliberately bypasses cached inventories.
"""
from __future__ import annotations

import ast
import hashlib
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

EXISTENCE_ORACLE_ENV = "STATEFRAME_EXISTENCE_ORACLE"
LEGACY_SYMBOL_ABSENCE_PREFIX = " DOES NOT EXIST anywhere"
READ_BEFORE_USE_LABEL = "Read each before use:"
READ_REUSE_MISSING_PREFIXES = (
    "[read reuse] source bytes missing from this request;",
    "[read reuse] delivered source was clipped by ",
    "[read reuse] delivered source span is not present in this request;",
)
_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", ".tox", ".venv", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".test-tmp", "node_modules", "dist", "build",
    "site-packages", "backup", "backups", ".backup", ".backups",
})
_SKIP_BACKUP_SUFFIXES = (".orig", ".bak", ".backup", ".old", ".save", ".swp", ".swo", "~")
_READ_PATH = re.compile(r"\bpath=([^\s;]+)")
_READ_RANGE = re.compile(r"\blines=(\d+)-(\d+)\b")
_READ_REVISION = re.compile(r"\brevision=(sha256:[0-9a-f]{64})\b")
_READ_TURN = re.compile(r"\bturn=(\d+)\b")
_READ_PAGE = re.compile(
    r"^\[read page\] path=(?P<path>.+?) "
    r"revision=(?P<revision>sha256:[0-9a-f]{64}) "
    r"lines=(?P<start>\d+)-(?:\d+)/(?:\d+) .*"
    r"effective=lines:(?P<effective_start>\d+)-(?P<effective_end>\d+)@.*"
    r"truncated=(?P<truncated>true|false) "
    r"decoded_with_replacement=(?P<replaced>true|false)$"
)
_MAX_LITERAL_FILES = 20000
_MAX_LITERAL_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class Fact:
    """A bounded observation made against a live workspace or request receipt."""

    probe: str
    status: str
    subject: str
    revision: str | None
    exists: bool | None = None
    coverage: Mapping[str, object] = field(default_factory=dict)
    sequence: int | None = None
    line_range: tuple[int, int] | None = None
    definitions: tuple[Mapping[str, object], ...] = ()
    matches: int | None = None
    closest: tuple[str, ...] = ()
    detail: str = ""


def enabled() -> bool:
    """Invalid values act as OFF so an opt-in cannot block a send."""
    return os.environ.get(EXISTENCE_ORACLE_ENV, "off").strip().lower() == "on"


def _revision_text(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _downgrade(journal: object, probe: str, status: str, subject: str) -> None:
    emit = getattr(journal, "emit", None)
    if not callable(emit):
        return
    try:
        emit(
            "existence_claim_downgraded",
            producer="kernel_workspace_truth",
            probe=probe,
            reason=status,
            subject_sha256=hashlib.sha256(subject.encode("utf-8", "replace")).hexdigest(),
            dispatch=False,
        )
    except Exception:
        # Journaling is observability only; a probe downgrade must remain fail-open.
        return


def _unknown(
    probe: str, subject: str, revision: str | None, status: str, *,
    coverage: Mapping[str, object] | None = None, journal: object = None,
    line_range: tuple[int, int] | None = None,
) -> Fact:
    _downgrade(journal, probe, status, subject)
    return Fact(
        probe=probe, status="unknown", subject=subject, revision=revision,
        exists=None, coverage=dict(coverage or {}), line_range=line_range,
        detail=status,
    )


def _root_path(root: str | Path | None) -> Path | None:
    if root is None:
        return None
    try:
        candidate = Path(root).resolve(strict=True)
        return candidate if candidate.is_dir() else None
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def _relative_file(root: Path, subject: str) -> tuple[Path, str] | None:
    raw = Path(subject)
    candidate = raw if raw.is_absolute() else root / raw
    try:
        resolved = candidate.resolve(strict=False)
        relative = resolved.relative_to(root).as_posix()
    except (OSError, RuntimeError, ValueError):
        return None
    if not relative or relative == ".":
        return None
    return resolved, relative


def _walk_files(
    root: Path, suffix: str | None = None, *,
    on_error: Callable[[OSError], None] | None = None,
):
    def report(error: OSError) -> None:
        if on_error is not None:
            on_error(error)

    for directory, names, files in os.walk(
            root, topdown=True, onerror=report, followlinks=False):
        current = Path(directory)
        allowed: list[str] = []
        for name in sorted(names):
            child = current / name
            if name.lower() in _SKIP_DIRS:
                continue
            if child.is_symlink():
                report(OSError("symlinked directory skipped"))
                continue
            allowed.append(name)
        names[:] = allowed
        for name in sorted(files):
            path = current / name
            if path.is_symlink():
                report(OSError("symlinked file skipped"))
                continue
            lowered = name.lower()
            if (lowered.endswith(_SKIP_BACKUP_SUFFIXES)
                    or ".r2f-prior-" in lowered):
                continue
            if suffix is None or path.suffix.lower() == suffix:
                yield path.relative_to(root).as_posix(), path


def _ast_names(source: str, relative: str) -> tuple[tuple[str, Mapping[str, object]], ...]:
    tree = ast.parse(source, filename=relative)
    definitions: list[tuple[str, Mapping[str, object]]] = []

    def walk(node: ast.AST, owner: str | None = None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = child.args
                names = [arg.arg for arg in args.posonlyargs]
                if args.posonlyargs:
                    names.append("/")
                names.extend(arg.arg for arg in args.args)
                if args.vararg:
                    names.append("*" + args.vararg.arg)
                elif args.kwonlyargs:
                    names.append("*")
                names.extend(arg.arg for arg in args.kwonlyargs)
                if args.kwarg:
                    names.append("**" + args.kwarg.arg)
                prefix = "async def" if isinstance(child, ast.AsyncFunctionDef) else "def"
                signature = f"{prefix} {child.name}({', '.join(names)})"
                definitions.append((child.name, {
                    "file": relative, "line": child.lineno,
                    "kind": "method" if owner else "function",
                    "owner": owner, "signature": signature,
                }))
                walk(child, owner)
            elif isinstance(child, ast.ClassDef):
                definitions.append((child.name, {
                    "file": relative, "line": child.lineno, "kind": "class",
                    "owner": owner, "signature": f"class {child.name}",
                }))
                walk(child, child.name)
            else:
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name) and target.id.isupper():
                            definitions.append((target.id, {
                                "file": relative, "line": child.lineno, "kind": "constant",
                                "owner": owner, "signature": target.id,
                            }))
                walk(child, owner)

    walk(tree)
    return tuple(definitions)


def _symbol_index(root: Path) -> tuple[
    dict[str, tuple[Mapping[str, object], ...]], dict[str, object], str
]:
    definitions: dict[str, list[Mapping[str, object]]] = {}
    digest = hashlib.sha256()
    scanned = 0
    failed = 0

    def mark_failed(_error: OSError) -> None:
        nonlocal failed
        failed += 1

    for relative, path in _walk_files(root, ".py", on_error=mark_failed):
        try:
            raw = path.read_bytes()
            source = raw.decode("utf-8")
            rows = _ast_names(source, relative)
        except (OSError, UnicodeError, SyntaxError, ValueError):
            failed += 1
            continue
        digest.update(relative.encode("utf-8", "replace"))
        digest.update(b"\\0")
        digest.update(hashlib.sha256(raw).digest())
        scanned += 1
        for name, definition in rows:
            definitions.setdefault(name, []).append(definition)
    coverage: dict[str, object] = {
        "complete": failed == 0,
        "indexed": scanned,
        "failed": failed,
    }
    return ({name: tuple(rows) for name, rows in definitions.items()},
            coverage, "sha256:" + digest.hexdigest())


def _current_path_fact(
    root: Path, subject: str, revision: str | None, journal: object,
) -> Fact:
    resolved_pair = _relative_file(root, subject)
    if resolved_pair is None:
        return _unknown("stat", subject, revision, "path_outside_or_unresolvable", journal=journal)
    candidate, relative = resolved_pair
    try:
        metadata = candidate.stat()
    except FileNotFoundError:
        try:
            parent = candidate.parent.stat()
            current_revision = revision or (
                f"live:{parent.st_dev}:{parent.st_ino}:{parent.st_mtime_ns}"
            )
        except OSError:
            current_revision = revision
        return Fact("stat", "missing", relative, current_revision, exists=False)
    except (OSError, RuntimeError, ValueError) as error:
        return _unknown("stat", relative, revision, type(error).__name__, journal=journal)
    current_revision = revision or (
        f"live:{metadata.st_dev}:{metadata.st_ino}:{metadata.st_mtime_ns}:{metadata.st_size}"
    )
    return Fact("stat", "exists", relative, current_revision, exists=True)


def _current_symbol_fact(
    root: Path, subject: str, revision: str | None, journal: object,
) -> Fact:
    if not subject:
        return _unknown("index", subject, revision, "symbol_query_empty", journal=journal)
    try:
        definitions, coverage, index_revision = _symbol_index(root)
    except (OSError, RuntimeError, ValueError) as error:
        return _unknown("index", subject, revision, type(error).__name__, journal=journal)
    rows = definitions.get(subject, ())
    current_revision = revision or index_revision
    if rows:
        return Fact(
            "index", "indexed", subject, current_revision, exists=True,
            coverage=coverage, definitions=rows,
        )
    if coverage["complete"] is not True:
        return _unknown(
            "index", subject, current_revision, "index_incomplete",
            coverage=coverage, journal=journal,
        )
    from difflib import get_close_matches
    closest = tuple(get_close_matches(subject, definitions.keys(), n=5, cutoff=0.6))
    return Fact(
        "index", "no_symbol_index_entry", subject, current_revision, exists=False,
        coverage=coverage, closest=closest,
    )


def _literal_scope_target(
    root: Path, scope: str | None,
) -> tuple[Path, str, tuple[str, Path] | None] | None:
    if not scope or scope in {".", "./"}:
        return root, "", None
    scoped = _relative_file(root, scope)
    if scoped is None:
        return None
    selected_path, relative = scoped
    if selected_path.is_dir():
        return selected_path, relative, None
    if selected_path.is_file():
        return root, "", (relative, selected_path)
    return None


def _current_literal_fact(
    root: Path, subject: str, revision: str | None, journal: object,
    scope: str | None,
) -> Fact:
    if not subject:
        return _unknown("literal", subject, revision, "literal_query_empty", journal=journal)
    scope_target = _literal_scope_target(root, scope)
    if scope_target is None:
        return _unknown("literal", subject, revision, "literal_scope_unavailable",
                        journal=journal)
    scope_root, scope_prefix, single_file = scope_target
    matches = 0
    files = 0
    scanned_bytes = 0
    incomplete = False
    digest = hashlib.sha256()

    def mark_incomplete(_error: OSError) -> None:
        nonlocal incomplete
        incomplete = True

    source_files = (single_file,) if single_file is not None else _walk_files(
        scope_root, on_error=mark_incomplete,
    )
    for relative, path in source_files:
        if scope_prefix:
            relative = (Path(scope_prefix) / relative).as_posix()
        files += 1
        if files > _MAX_LITERAL_FILES:
            incomplete = True
            break
        try:
            raw = path.read_bytes()
        except OSError:
            incomplete = True
            continue
        scanned_bytes += len(raw)
        if scanned_bytes > _MAX_LITERAL_BYTES:
            incomplete = True
            break
        digest.update(relative.encode("utf-8", "replace"))
        digest.update(b"\\0")
        digest.update(hashlib.sha256(raw).digest())
        try:
            text = raw.decode("utf-8")
        except UnicodeError:
            # A binary/non-UTF-8 file is outside the literal-text probe's scope.
            continue
        matches += text.count(subject)
    coverage: dict[str, object] = {
        "complete": not incomplete, "files": min(files, _MAX_LITERAL_FILES),
        "bytes": min(scanned_bytes, _MAX_LITERAL_BYTES),
    }
    current_revision = revision or "sha256:" + digest.hexdigest()
    if incomplete:
        return _unknown(
            "literal", subject, current_revision, "literal_scan_incomplete",
            coverage=coverage, journal=journal,
        )
    if matches:
        return Fact(
            "literal", "matches", subject, current_revision, exists=True,
            coverage=coverage, matches=matches,
        )
    return Fact(
        "literal", "no_matches", subject, current_revision, exists=False,
        coverage=coverage, matches=0,
    )


def _receipt_sequence(message: Mapping[str, object], line: str) -> int | None:
    sequence = message.get("turn", message.get("step"))
    if type(sequence) is int and sequence >= 1:
        return sequence
    turn_match = _READ_TURN.search(line)
    return int(turn_match.group(1)) if turn_match is not None else None


def _receipt_path_relative(root: Path, path: str, expected: str) -> str | None:
    normalized = _relative_file(root, path)
    if normalized is not None:
        return normalized[1]
    candidate = Path(path)
    if not candidate.is_absolute():
        return None
    path_parts = tuple(part for part in candidate.parts if part not in {"/", ""})
    expected_parts = tuple(Path(expected).parts)
    if len(path_parts) < len(expected_parts):
        return None
    return expected if path_parts[-len(expected_parts):] == expected_parts else None


def _receipt_line(
    root: Path, line: str, relative: str, line_range: tuple[int, int] | None,
) -> tuple[int, int, str | None, bool] | None:
    page_match = _READ_PAGE.fullmatch(line)
    if page_match is not None:
        path = page_match.group("path").replace("\\", "/").removeprefix("./")
        start = int(page_match.group("effective_start"))
        end = int(page_match.group("effective_end"))
        receipt = (
            start, end, page_match.group("revision"),
            page_match.group("truncated") == "true"
            or page_match.group("replaced") == "true",
        )
    else:
        if "[read reuse]" not in line or "present in this request" not in line:
            return None
        path_match = _READ_PATH.search(line)
        range_match = _READ_RANGE.search(line)
        revision_match = _READ_REVISION.search(line)
        if path_match is None or range_match is None:
            return None
        path = path_match.group(1).replace("\\", "/").removeprefix("./")
        start, end = int(range_match.group(1)), int(range_match.group(2))
        revision = revision_match.group(1) if revision_match else None
        receipt = (start, end, revision, "clipped" in line or "clip_reason=" in line)
    if _receipt_path_relative(root, path, relative) != relative:
        return None
    if line_range is not None:
        requested_start, requested_end = line_range
        if start > requested_start or end < requested_end:
            return None
    return receipt


def _receipt_candidates(
    root: Path, relative: str, messages: Sequence[object],
    line_range: tuple[int, int] | None,
) -> tuple[list[tuple[int, int, int, str | None, bool]], bool]:
    found: list[tuple[int, int, int, str | None, bool]] = []
    unsequenced = False
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        for line in content.splitlines():
            receipt = _receipt_line(root, line, relative, line_range)
            if receipt is None:
                continue
            sequence = _receipt_sequence(message, line)
            if sequence is None:
                unsequenced = True
                continue
            start, end, revision, partial = receipt
            found.append((sequence, start, end, revision, partial))
    return found, unsequenced


def _current_receipt_fact(
    root: Path, subject: str, revision: str | None,
    messages: Sequence[object], line_range: tuple[int, int] | None,
    journal: object,
) -> Fact:
    resolved = _relative_file(root, subject)
    if resolved is None:
        return _unknown("receipt", subject, revision, "path_outside_or_unresolvable",
                        journal=journal, line_range=line_range)
    current_path, relative = resolved
    found, unsequenced = _receipt_candidates(root, relative, messages, line_range)
    if not found:
        if unsequenced:
            return _unknown(
                "receipt", relative, revision, "receipt_sequence_unavailable",
                journal=journal, line_range=line_range,
            )
        return Fact(
            "receipt", "no_receipt", relative, revision, exists=None,
            line_range=line_range,
        )
    sequence, delivered_start, delivered_end, receipt_revision, partial = found[-1]
    if receipt_revision is None:
        return _unknown(
            "receipt", relative, revision, "receipt_revision_unavailable",
            journal=journal, line_range=line_range,
        )
    try:
        current_bytes = current_path.read_bytes()
        current_source = current_bytes.decode("utf-8")
    except (OSError, UnicodeError) as error:
        return _unknown(
            "receipt", relative, revision, type(error).__name__,
            journal=journal, line_range=line_range,
        )
    current_revision = "sha256:" + hashlib.sha256(current_source.encode("utf-8")).hexdigest()
    if current_revision != receipt_revision:
        return _unknown(
            "receipt", relative, current_revision, "receipt_revision_stale",
            journal=journal, line_range=line_range,
        )
    status = "partial" if partial else "delivered"
    return Fact(
        "receipt", status, relative, current_revision, exists=None,
        sequence=sequence, line_range=(delivered_start, delivered_end),
        detail="clipped" if partial else "",
    )


def probe_live_workspace(
    root: str | Path | None, probe: str, subject: str, *,
    revision: str | None = None, messages: Sequence[object] = (),
    line_range: tuple[int, int] | None = None, journal: object = None,
    scope: str | None = None,
) -> Fact:
    """Probe current filesystem/index/receipt state; never consult a cached inventory."""
    current_revision = _revision_text(revision)
    workspace = _root_path(root)
    if workspace is None:
        return _unknown(probe, subject, current_revision, "live_workspace_unavailable",
                        journal=journal, line_range=line_range)
    try:
        if probe == "stat":
            return _current_path_fact(workspace, subject, current_revision, journal)
        if probe == "index":
            return _current_symbol_fact(workspace, subject, current_revision, journal)
        if probe == "literal":
            return _current_literal_fact(
                workspace, subject, current_revision, journal, scope,
            )
        if probe == "receipt":
            return _current_receipt_fact(
                workspace, subject, current_revision, messages, line_range, journal,
            )
    except Exception as error:
        return _unknown(probe, subject, current_revision, type(error).__name__,
                        journal=journal, line_range=line_range)
    return _unknown(probe, subject, current_revision, "unsupported_probe",
                    journal=journal, line_range=line_range)


def exists(root: str | Path | None, path: str, *, revision: str | None = None,
           journal: object = None) -> Fact:
    return probe_live_workspace(root, "stat", path, revision=revision, journal=journal)


def symbol_defined(root: str | Path | None, name: str, *, revision: str | None = None,
                   journal: object = None) -> Fact:
    return probe_live_workspace(root, "index", name, revision=revision, journal=journal)


def literal_matches(
        root: str | Path | None, text: str, *, scope: str | None = None,
        revision: str | None = None, journal: object = None) -> Fact:
    return probe_live_workspace(
        root, "literal", text, revision=revision, journal=journal, scope=scope,
    )


def read_current(
    root: str | Path | None, path: str, line_range: tuple[int, int] | None, *,
    messages: Sequence[object], revision: str | None = None, journal: object = None,
) -> Fact:
    return probe_live_workspace(
        root, "receipt", path, revision=revision, messages=messages,
        line_range=line_range, journal=journal,
    )


def render_fact(fact: Fact) -> str:
    revision = f" revision={fact.revision}" if fact.revision else ""
    if fact.probe == "stat" and fact.status == "exists":
        revision_part = f";{revision.strip()}" if revision else ""
        return f"File {fact.subject} exists in the live workspace (probe=stat{revision_part})."
    if fact.probe == "stat" and fact.status == "missing":
        revision_part = f";{revision.strip()}" if revision else ""
        return f"File {fact.subject} does not exist in the live workspace (probe=stat{revision_part})."
    if fact.probe == "index" and fact.status == "indexed":
        locations = ", ".join(
            f"{row.get('file')}:{row.get('line')}" for row in fact.definitions[:6]
        )
        return (f"Symbol index has {len(fact.definitions)} current definition(s) for "
                f"`{fact.subject}`{revision}; {locations}.")
    if fact.probe == "index" and fact.status == "no_symbol_index_entry":
        coverage = fact.coverage
        suffix = (f"; closest indexed names: {', '.join(fact.closest)}" if fact.closest else "")
        return (f"No symbol index entry for `{fact.subject}` in the live workspace"
                f"{revision} (coverage: {coverage.get('indexed', 0)} indexed files, "
                f"complete={str(coverage.get('complete', False)).lower()}){suffix}.")
    if fact.probe == "literal" and fact.status in {"matches", "no_matches"}:
        count = fact.matches if fact.matches is not None else 0
        return (f"{count} matches for literal text `{fact.subject}` (literal mode) "
                f"in the live workspace{revision}.")
    if fact.probe == "receipt" and fact.status in {"delivered", "partial"}:
        line_range = fact.line_range or (0, 0)
        condition = "were delivered but clipped" if fact.status == "partial" else "were last delivered"
        return (f"Lines {line_range[0]}-{line_range[1]} of {fact.subject} {condition} "
                f"at turn {fact.sequence} (read receipt{revision}).")
    if fact.probe == "receipt" and fact.status == "no_receipt":
        requested = ""
        if fact.line_range is not None:
            requested = f" lines={fact.line_range[0]}-{fact.line_range[1]}"
        return (f"No current read receipt for {fact.subject}{requested}; fetch current source "
                "before relying on it.")
    return (f"Current workspace fact unavailable for {fact.subject} "
            f"(probe={fact.probe}, reason={fact.detail or fact.status}); refresh evidence.")


def render_legacy_claim_fact(cited: str, closest: Sequence[str]) -> str:
    return f"{cited} does not exist; closest: {', '.join(closest) or 'none'}"


def render_legacy_symbol_answer(
    name: str, hits: Sequence[Mapping[str, object]], near: Sequence[str],
    indexed: int, scanned: int, failed: int,
) -> str:
    if hits:
        out = [f"`{name}` EXISTS ({len(hits)} definition"
               f"{'s' if len(hits) > 1 else ''}):"]
        for row in hits[:6]:
            loc = f"{row['file']}:{row['line']}"
            owner = f" in class {row['cls']}" if row.get("cls") else ""
            out.append(f"  {loc:52s} {row['signature']}{owner}")
        if len(hits) > 6:
            out.append(f"  ... and {len(hits) - 6} MORE definition(s) not shown — this "
                       f"name is ambiguous, so confirm which one the caller means.")
        return "\n".join(out)
    msg = [f"`{name}`{LEGACY_SYMBOL_ABSENCE_PREFIX} "
           f"({indexed} symbols indexed, {scanned} files)."]
    if near:
        msg.append("Closest real names: " + ", ".join(near))
        msg.append("Do not search for it again — use one of the names above.")
    else:
        msg.append("No similar name either. The symptom may name a concept, not a symbol.")
    if failed:
        msg.append(f"(caveat: {failed} files could not be parsed and are not "
                   "covered by this answer)")
    return "\n".join(msg)


def render_legacy_deferred_notice(header: str, paths: Sequence[str]) -> str:
    return "\n".join((header, READ_BEFORE_USE_LABEL, *(f"- {path}" for path in paths)))


def render_legacy_read_missing(
    path: str, start: object, end: object, *, clip_reason: object = None,
    start_column: object = None, end_column: object = None,
    byte_length: object = None,
) -> str:
    if type(start) is int and type(end) is int and 1 <= start <= end:
        span = f"path={path} lines={start}-{end}"
    else:
        span = f"path={path} line range unavailable"
    if str(clip_reason or "") in {"max_chars", "max_bytes"}:
        columns = (f" columns={start_column}-{end_column}"
                   if type(start_column) is int and type(end_column) is int else "")
        size = f" delivered_bytes={byte_length}" if type(byte_length) is int else ""
        return (f"[read reuse] delivered source was clipped by {clip_reason}; "
                f"{span}{columns}{size}. The delivered span is not present in this request.")
    return f"[read reuse] delivered source span is not present in this request; {span}."


def render_legacy_target_gone_notice(path: str) -> str:
    """Keep the opt-out ReAct notice byte-identical while centralizing its claim text."""
    return ("\n[kernel] THE TARGET FILE IS GONE — your edits "
            f"removed {path}. If complete authorized content is "
            "available, use `file_create` now; otherwise report the "
            "capability blocker. Do not use `edit` to recreate a missing file.")


def render_legacy_entry_verification(
        result: Mapping[str, object] | None,
        header_prefix: str = "[ENTRY SYMBOL VERIFICATION") -> str | None:
    if not result or result.get("status") == "FOUND":
        return None
    symbol = result.get("symbol")
    paths = [str(item) for item in result.get("paths", ()) or ()]
    scope = (f"{paths[0]} or any declared product file" if paths
             else "the declared product files")
    header = f"{header_prefix} | host where_defined at revision {result.get('revision')}]"
    if result.get("status") == "FOUND_OUTSIDE_DECLARED":
        definition = result.get("definition") or {}
        return (f"{header}\nentry symbol `{symbol}`: not in {scope}; defined at "
                f"{definition.get('file')}:{definition.get('lineno')} (outside the declared "
                "product files).")
    if result.get("status") in {"UNAVAILABLE", "PARTIAL", "UNKNOWN", "STALE"}:
        return (f"{header}\nentry symbol `{symbol}`: {result.get('status')} at revision "
                f"{result.get('revision')}; repository resolution is not positive evidence. "
                "The source/dependency lookup must be refreshed before this fact can be used.")
    coverage = result.get("coverage") if isinstance(result.get("coverage"), Mapping) else {}
    indexed = coverage.get("indexed")
    repo = (f"; repository index: 0 definitions across {indexed} indexed files"
            if result.get("repo_status") == "NOT_FOUND" and indexed is not None
            else "; repository index unavailable" if result.get("repo_status") != "NOT_FOUND"
            else "")
    tests = [str(item) for item in result.get("test_paths", ()) or ()]
    marker = ("\nThe task row declares this entry as a NEW symbol (kind=new_symbol): "
              "it is expected to be absent; create it." if result.get("new_symbol") else "")
    if result.get("status") == "REFERENCED_ONLY":
        return (f"{header}\nentry symbol `{symbol}`: 0 definitions in {scope}{repo}; "
                f"{int(result.get('test_references', 0) or 0)} reference(s) in the public tests "
                f"({', '.join(tests[:4])}). The tests call it; it does not exist yet at this "
                "revision. Nothing further can be found by searching for it." + marker)
    roots = (f" Scanned: {', '.join([*paths[:4], *tests[:4]])}." if tests else "")
    return (f"{header}\nentry symbol `{symbol}`: NOT FOUND in {scope} "
            f"(0 definitions, {int(result.get('references', 0) or 0)} references{repo}"
            + (f"; 0 references in {len(tests)} public test file(s)" if tests else "") + ")."
            + roots + " The named entry point does not exist yet at this revision. Nothing "
            "further can be found by searching for it. Use only symbols that exist, or create "
            "what the task needs; do not call an undefined helper." + marker)


def render_entry_fact(
    result: Mapping[str, object] | None, root: str | Path | None, *,
    revision: str | None = None, journal: object = None,
    header_prefix: str = "[ENTRY SYMBOL VERIFICATION",
) -> str | None:
    if not result:
        return None
    if result.get("status") == "FOUND":
        return None
    symbol = str(result.get("symbol") or "")
    fact = symbol_defined(root, symbol, revision=revision, journal=journal)
    marker = (" The task row declares this entry as a NEW symbol; create it."
              if result.get("new_symbol") else "")
    return f"{header_prefix} | live-workspace index]\n{render_fact(fact)}{marker}"


def render_deferred_source_facts(
    header: str, paths: Sequence[str], root: str | Path | None, *,
    messages: Sequence[object], revision: str | None = None,
    journal: object = None,
) -> str:
    normalized = tuple(dict.fromkeys(str(path).strip() for path in paths if str(path).strip()))
    lines = [header, "Current live-workspace source status:"]
    for path in normalized:
        path_fact = exists(root, path, revision=revision, journal=journal)
        if path_fact.status == "exists":
            receipt = read_current(
                root, path, None, messages=messages,
                revision=path_fact.revision, journal=journal,
            )
            lines.append(render_fact(receipt))
        else:
            lines.append(render_fact(path_fact))
    return "\n".join(lines)


def render_read_visibility_fact(fact: Fact) -> str:
    range_part = (
        f" lines={fact.line_range[0]}-{fact.line_range[1]}"
        if fact.line_range is not None else ""
    )
    return f"[read reuse] path={fact.subject}{range_part}; {render_fact(fact)}"


__all__ = [
    "EXISTENCE_ORACLE_ENV", "Fact", "LEGACY_SYMBOL_ABSENCE_PREFIX",
    "READ_BEFORE_USE_LABEL", "READ_REUSE_MISSING_PREFIXES", "enabled", "exists",
    "literal_matches", "probe_live_workspace", "read_current", "render_entry_fact",
    "render_fact", "render_legacy_claim_fact", "render_legacy_deferred_notice",
    "render_legacy_entry_verification", "render_legacy_read_missing",
    "render_legacy_symbol_answer", "render_deferred_source_facts",
    "render_read_visibility_fact", "symbol_defined",
]
