"""Bounded, revision-current Python module heads for the FRAMEFIT5 StateFrame."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import re
import stat
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath

MODE_ENV = "STATEFRAME_HEADCARRY"
# T2FIX: carry the complete bytes of every read the frame still OWES
# (kernel_reread_result_truth.read_is_owed) as a head, not only open-site/edit heads.
# Live T2 on f63ff225 read 4 test files per batch; the frame then said
# "[not shown] ... re-read to see it" for 3 of them (only the newest exchange survives),
# and the model obeyed that line for 80 requests. Off = byte-identical to the parent tree.
OWED_READS_ENV = "STATEFRAME_HEADCARRY_OWED_READS"
BYTE_BUDGET_ENV = "STATEFRAME_HEADCARRY_MAX_BYTES"
DEFAULT_BYTE_BUDGET = 12 * 1024
MIN_BYTE_BUDGET = 256
MAX_BYTE_BUDGET = 64 * 1024
HEAD_BASE_LINES = 40
SCHEMA = "stateframe-framefit5-headcarry/1"
_SKIP_IMPORT_PARTS = frozenset({
    ".git", ".venv", "venv", "node_modules", "__pycache__", "backups",
})


def framefit5_headcarry_mode(environ: Mapping[str, str] | None = None) -> str:
    """Read the opt-in mode; invalid spellings retain the default-off behavior."""
    source = os.environ if environ is None else environ
    value = (source.get(MODE_ENV, "off") or "off").strip().lower()
    return value if value in {"off", "shadow", "on"} else "off"


def headcarry_owed_reads_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Owed-read heads default ON; only the exact spelling ``off`` disables them."""
    source = os.environ if environ is None else environ
    value = (source.get(OWED_READS_ENV, "on") or "on").strip().lower()
    if value not in {"on", "off"}:
        raise ValueError(f"{OWED_READS_ENV} must be on or off")
    return value == "on"


def owed_read_rows(read_facts: object) -> list[dict[str, object]]:
    """Reads the frame still owes: unchanged since read, so the current file IS the read bytes.

    Returns one row per path, newest fact wins, with the receipt count as its reference
    weight. A fact whose file changed since the read is not owed (the edit discharged it).
    """
    if not isinstance(read_facts, Sequence) or isinstance(read_facts, (str, bytes, bytearray)):
        return []
    from kernel_reread_result_truth import read_is_owed

    rows: dict[str, dict[str, object]] = {}
    for fact in read_facts:
        if not isinstance(fact, Mapping) or not read_is_owed(fact):
            continue
        path = _safe_relative_path(fact.get("path"))
        read_revision = fact.get("read_revision")
        if (path is None or not isinstance(read_revision, str)
                or not read_revision.startswith("sha256:")
                or fact.get("current_revision") != read_revision
                or fact.get("unchanged_since_read") is not True):
            continue
        receipts = fact.get("receipt_count")
        weight = receipts if type(receipts) is int and receipts > 0 else 1
        rows[path] = {"path": path, "revision": read_revision, "reference_count": weight}
    return list(rows.values())


def headcarry_byte_budget(
    environ: Mapping[str, str] | None = None,
    *,
    override: int | None = None,
) -> int:
    """Return a bounded total JSON byte budget, defaulting safely on invalid input."""
    if override is not None:
        value: object = override
    else:
        source = os.environ if environ is None else environ
        raw = source.get(BYTE_BUDGET_ENV)
        if raw is None:
            return DEFAULT_BYTE_BUDGET
        value = raw.strip() if isinstance(raw, str) else raw
        if isinstance(value, str) and re.fullmatch(r"\d{1,6}", value) is not None:
            value = int(value)
    if type(value) is not int or not MIN_BYTE_BUDGET <= value <= MAX_BYTE_BUDGET:
        return DEFAULT_BYTE_BUDGET
    return value


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def _seal_byte_count(panel: dict[str, object]) -> dict[str, object]:
    panel["byte_count"] = 0
    for _attempt in range(4):
        measured = len(_json_bytes(panel))
        if panel["byte_count"] == measured:
            return panel
        panel["byte_count"] = measured
    raise RuntimeError("head-carry byte count did not stabilize")


def fit_headcarry_panel_for_request(
        panel: Mapping[str, object],
        *,
        max_request_bytes: int | None = None,
        measure_request_bytes: Callable[[Mapping[str, object]], int] | None = None,
        max_request_tokens: int | None = None,
        measure_request_tokens: Callable[[Mapping[str, object]], int] | None = None,
) -> dict[str, object]:
    """Drop least-referenced heads until the complete request fits its active ceilings."""
    if (max_request_bytes is None) != (measure_request_bytes is None):
        raise ValueError("head-carry byte ceiling and measurement must be provided together")
    if (max_request_tokens is None) != (measure_request_tokens is None):
        raise ValueError("head-carry token ceiling and measurement must be provided together")
    if max_request_bytes is None and max_request_tokens is None:
        raise ValueError("head-carry fitting requires a byte or token ceiling")
    if max_request_bytes is not None and (
            type(max_request_bytes) is not int or max_request_bytes < 0):
        raise ValueError("head-carry request ceiling must be a non-negative integer")
    if max_request_tokens is not None and (
            type(max_request_tokens) is not int or max_request_tokens < 0):
        raise ValueError("head-carry token ceiling must be a non-negative integer")
    raw_heads = panel.get("heads")
    if not isinstance(raw_heads, list) or any(
            not isinstance(head, Mapping) for head in raw_heads):
        raise ValueError("head-carry heads must be a list of objects")
    omitted = panel.get("omitted_count")
    if type(omitted) is not int or omitted < 0:
        raise ValueError("head-carry omitted count must be a non-negative integer")

    fitted = copy.deepcopy(dict(panel))
    heads = copy.deepcopy(raw_heads)
    fitted["heads"] = heads
    fitted["omitted_count"] = omitted
    _seal_byte_count(fitted)

    def over_ceiling() -> bool:
        return (
            (max_request_bytes is not None
             and measure_request_bytes is not None
             and measure_request_bytes(fitted) > max_request_bytes)
            or (max_request_tokens is not None
                and measure_request_tokens is not None
                and measure_request_tokens(fitted) > max_request_tokens)
        )

    def drop_priority(head: Mapping[str, object]) -> tuple[int, int, str]:
        references = head.get("reference_count")
        if type(references) is not int or references < 0:
            references = 0
        # Preserve explicit open/edit heads when reference counts tie.
        source_priority = 0 if head.get("source_kind") == "import" else 1
        return references, source_priority, str(head.get("path", ""))

    while heads and over_ceiling():
        least_referenced = min(
            range(len(heads)),
            key=lambda index: drop_priority(heads[index]),
        )
        heads.pop(least_referenced)
        fitted["omitted_count"] += 1
        fitted["omitted_line"] = f"omitted {fitted['omitted_count']} heads"
        _seal_byte_count(fitted)
    return fitted


def _safe_relative_path(value: object) -> str | None:
    raw = str(value or "").replace("\\", "/")
    path = PurePosixPath(raw)
    if (not raw or path.is_absolute()
            or any(part in {"", ".", ".."} or part.casefold() == "backups"
                   for part in path.parts)):
        return None
    if any(part.casefold() in _SKIP_IMPORT_PARTS for part in path.parts):
        return None
    return path.as_posix()


def _read_source(root: Path, relative_path: str) -> bytes | None:
    from kernel_stateframe_action_ledger import _read_open_site_file_bytes

    if not _safe_regular_file(root, relative_path):
        return None
    try:
        return _read_open_site_file_bytes(root, relative_path)
    except (OSError, RuntimeError, ValueError):
        return None


def _safe_regular_file(root: Path, relative_path: str) -> bool:
    cursor = root
    parts = PurePosixPath(relative_path).parts
    for index, part in enumerate(parts):
        cursor = cursor / part
        try:
            mode = cursor.lstat().st_mode
        except OSError:
            return False
        if stat.S_ISLNK(mode):
            return False
        if index < len(parts) - 1 and not stat.S_ISDIR(mode):
            return False
        if index == len(parts) - 1:
            return stat.S_ISREG(mode)
    return False


def _module_files(root: Path, parts: Sequence[str]) -> list[str]:
    """Resolve a static Python import to in-repository module/package files."""
    if (not parts or any(not part.isidentifier() for part in parts)
            or any(part.casefold() in _SKIP_IMPORT_PARTS for part in parts)):
        return []
    found: list[str] = []
    for depth in range(1, len(parts)):
        package = root.joinpath(*parts[:depth], "__init__.py")
        relative = package.relative_to(root).as_posix()
        if _safe_regular_file(root, relative):
            if relative not in found:
                found.append(relative)
    module = root.joinpath(*parts).with_suffix(".py")
    package = root.joinpath(*parts, "__init__.py")
    for candidate in (module, package):
        relative = candidate.relative_to(root).as_posix()
        if _safe_regular_file(root, relative):
            if relative not in found:
                found.append(relative)
    return found


def _import_targets(root: Path, path: str, source: bytes) -> Counter[str]:
    if not path.endswith(".py"):
        return Counter()
    try:
        tree = ast.parse(source.decode("utf-8"))
    except (UnicodeDecodeError, SyntaxError, ValueError):
        return Counter()
    current = PurePosixPath(path).parts
    package_parts = current[:-1]
    targets: Counter[str] = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                for target in _module_files(root, parts):
                    targets[target] += 1
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            keep = len(package_parts) - node.level + 1
            if keep < 0:
                continue
            base = list(package_parts[:keep])
            if node.module:
                base.extend(node.module.split("."))
        else:
            base = node.module.split(".") if node.module else []
        if not base:
            continue
        for target in _module_files(root, base):
            targets[target] += 1
        for alias in node.names:
            if alias.name == "*":
                continue
            for target in _module_files(root, [*base, *alias.name.split(".")]):
                targets[target] += 1
    return targets


def _module_head(root: Path, path: str, raw: bytes, *,
                 minimum_end: int = 0) -> dict[str, object] | None:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    lines = raw.splitlines(keepends=True)
    if not lines:
        return None
    import_end = 0
    if path.endswith(".py"):
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            tree = None
        if tree is not None:
            parents: dict[ast.AST, ast.AST] = {}
            for parent in ast.walk(tree):
                for child in ast.iter_child_nodes(parent):
                    parents[child] = parent
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Import, ast.ImportFrom)):
                    continue
                ancestor = parents.get(node)
                in_module_scope = True
                while ancestor is not None:
                    if isinstance(
                        ancestor,
                        (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
                    ):
                        in_module_scope = False
                        break
                    ancestor = parents.get(ancestor)
                if in_module_scope and type(getattr(node, "end_lineno", None)) is int:
                    import_end = max(import_end, node.end_lineno)
    end = min(len(lines), max(HEAD_BASE_LINES, import_end, minimum_end))
    head = b"".join(lines[:end])
    full_sha = _sha256(raw)
    head_sha = _sha256(head)
    return {
        "header": (
            f"[MODULE HEAD | {path} lines 1-{end} "
            f"content_sha256={head_sha} revision=sha256:{full_sha}]"
        ),
        "path": path,
        "line_range": {"start": 1, "end": end},
        "revision": "sha256:" + full_sha,
        "source_sha256": full_sha,
        "content_sha256": head_sha,
        "content": head.decode("utf-8"),
    }


def _site_covers_head(site: Mapping[str, object], path: str, raw: bytes, end: int) -> bool:
    if site.get("path") != path or site.get("revision") != "sha256:" + _sha256(raw):
        return False
    bounds = site.get("lines")
    current = site.get("current_bytes")
    if not isinstance(bounds, Mapping) or not isinstance(current, str):
        return False
    start_line, end_line = bounds.get("start"), bounds.get("end")
    if (type(start_line) is not int or type(end_line) is not int
            or start_line < 1 or end_line < start_line):
        return False
    source_lines = raw.splitlines(keepends=True)
    expected = b"".join(source_lines[start_line - 1:end_line]).decode("utf-8")
    if (start_line > 1 or end_line < end or current != expected
            or site.get("content_sha256") != _sha256(expected.encode("utf-8"))):
        return False
    return True


def _message_covers_head(
    messages: Sequence[Mapping[str, object]],
    *,
    path: str,
    source: str,
    end: int,
) -> bool:
    try:
        from kernel_request_source_window import (
            _window_segments,
            parse_window,
            verify_window,
        )
    except ImportError:
        return False
    for message in messages:
        content = message.get("content")
        try:
            metadata = parse_window(content)
            if (metadata is None or metadata.get("path") != path
                    or not verify_window(content, path=path, source=source)):
                continue
            segments = _window_segments(str(content), metadata)
        except (KeyError, TypeError, ValueError, OSError):
            continue
        ranges = sorted(
            (int(row["start_line"]), int(row["end_line"]))
            for row, _body in segments
            if type(row.get("start_line")) is int and type(row.get("end_line")) is int
        )
        covered_through = 0
        for start, finish in ranges:
            if start > covered_through + 1:
                break
            covered_through = max(covered_through, finish)
            if covered_through >= end:
                return True
    return False


def _requested_paths(
    open_sites: Sequence[object],
    applied_edits: Sequence[object],
    owed_reads: Sequence[object] = (),
) -> tuple[list[str], dict[str, str]]:
    primary: list[str] = []
    role_by_path: dict[str, str] = {}
    for rows, role, key in (
        (open_sites, "open_site", "path"),
        (applied_edits, "applied_edit", "file"),
        (owed_reads, "owed_read", "path"),
    ):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            raw_path = row.get(key) or (row.get("path") if key == "file" else None)
            path = _safe_relative_path(raw_path)
            if path is None:
                continue
            if path not in role_by_path:
                primary.append(path)
                role_by_path[path] = role
    return primary, role_by_path


def _owed_read_requirements(
    owed_reads: Sequence[object],
) -> tuple[dict[str, str], dict[str, int]]:
    """Revision each owed read must still match, and its receipt weight."""
    revisions: dict[str, str] = {}
    weights: dict[str, int] = {}
    for row in owed_reads:
        if not isinstance(row, Mapping):
            continue
        path = _safe_relative_path(row.get("path"))
        revision = row.get("revision")
        if path is None or not isinstance(revision, str):
            continue
        revisions[path] = revision
        weight = row.get("reference_count")
        weights[path] = weight if type(weight) is int and weight > 0 else 1
    return revisions, weights


def _panel(
    heads: Sequence[Mapping[str, object]],
    *,
    omitted_count: int,
    deduplicated_count: int,
    byte_budget: int,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "byte_budget": byte_budget,
        "heads": [dict(head) for head in heads],
        "omitted_count": omitted_count,
        "deduplicated_count": deduplicated_count,
        "omitted_line": f"omitted {omitted_count} heads",
        "byte_count": 0,
    }
    return _seal_byte_count(payload)


def render_headcarry(
    workspace_root: str | Path,
    open_sites: Sequence[object],
    applied_edits: Sequence[object],
    *,
    messages: Sequence[Mapping[str, object]] = (),
    wire_sites: Sequence[object] = (),
    max_bytes: int | None = None,
    owed_reads: Sequence[object] = (),
) -> dict[str, object]:
    """Build current heads, one import hop deep, under a truthful total byte cap.

    ``owed_reads`` (see ``owed_read_rows``) are carried as COMPLETE files at their read
    revision: read-delivery truth (``kernel_reread_result_truth._complete_body``) only
    counts a whole revision-matching file, so a 40-line head could never discharge them.
    They rank after open sites and applied edits and before import dependencies.
    """
    budget = headcarry_byte_budget(override=max_bytes)
    try:
        root = Path(workspace_root).resolve(strict=True)
        if not root.is_dir():
            raise OSError("workspace root unavailable")
    except (OSError, RuntimeError, ValueError):
        root = None

    primary, roles = _requested_paths(open_sites, applied_edits, owed_reads)
    owed_revisions, owed_weights = _owed_read_requirements(owed_reads)
    unavailable = 0
    imports: Counter[str] = Counter()
    first_reference: dict[str, int] = {}
    reference_order = 0
    for path in primary:
        raw = _read_source(root, path) if root is not None else None
        if raw is None:
            continue
        for dependency, count in _import_targets(root, path, raw).items():
            imports[dependency] += count
            first_reference.setdefault(dependency, reference_order)
            reference_order += 1
    dependencies = [path for path in imports if path not in roles]
    dependencies.sort(key=lambda path: (-imports[path], first_reference[path], path))
    ordered_paths = [*primary, *dependencies]

    current_heads: list[dict[str, object]] = []
    deduplicated = 0
    for path in ordered_paths:
        if root is None:
            unavailable += 1
            continue
        raw = _read_source(root, path)
        if raw is not None and roles.get(path) == "owed_read":
            # The file must still be the read revision; otherwise the read is no
            # longer owed and the whole-file head would misstate what was read.
            if "sha256:" + _sha256(raw) != owed_revisions.get(path):
                unavailable += 1
                continue
            minimum_end = len(raw.splitlines(keepends=True))
        else:
            minimum_end = 0
        head = (_module_head(root, path, raw, minimum_end=minimum_end)
                if raw is not None and root is not None else None)
        if head is None:
            unavailable += 1
            continue
        covered = any(
            _site_covers_head(site, path, raw, int(head["line_range"]["end"]))
            for site in wire_sites
            if isinstance(site, Mapping)
        )
        if not covered:
            covered = _message_covers_head(
                messages, path=path, source=raw.decode("utf-8"),
                end=int(head["line_range"]["end"]),
            )
        if covered:
            deduplicated += 1
            continue
        head["source_kind"] = roles.get(path, "import")
        if path in roles:
            # Explicit heads weigh what the model asked for: an open site or edited file
            # that the model also re-read keeps its owed receipt count, so request fitting
            # (least referenced first) drops inferred import heads before owed reads.
            head["reference_count"] = max(1, owed_weights.get(path, 1))
        else:
            head["reference_count"] = imports[path]
        current_heads.append(head)

    total_omittable = unavailable + len(current_heads)
    selected: list[dict[str, object]] = []
    for head in current_heads:
        proposed = [*selected, head]
        proposed_omitted = total_omittable - len(proposed)
        if _panel(
            proposed,
            omitted_count=proposed_omitted,
            deduplicated_count=deduplicated,
            byte_budget=budget,
        )["byte_count"] <= budget:
            selected.append(head)
    return _panel(
        selected,
        omitted_count=total_omittable - len(selected),
        deduplicated_count=deduplicated,
        byte_budget=budget,
    )


def record_headcarry_shadow_mismatches(journal: object | None, panel: Mapping[str, object]) -> None:
    """Journal bounded field-level differences without changing the shadow result."""
    emit = getattr(journal, "emit", None)
    if not callable(emit):
        return
    heads = panel.get("heads")
    rows = heads if isinstance(heads, list) else []
    manifest = [
        {
            "path": row.get("path"),
            "line_range": row.get("line_range"),
            "revision": row.get("revision"),
            "content_sha256": row.get("content_sha256"),
        }
        for row in rows if isinstance(row, Mapping)
    ]
    summary = {
        "heads": "sha256:" + _sha256(_json_bytes({"heads": manifest})),
        "omitted_count": str(panel.get("omitted_count", 0)),
        "deduplicated_count": str(panel.get("deduplicated_count", 0)),
        "byte_count": str(panel.get("byte_count", 0)),
    }
    logged = getattr(journal, "_framefit5_headcarry_shadow_logged", None)
    logged = logged if isinstance(logged, set) else set()
    for field, candidate in summary.items():
        legacy = "absent" if field == "heads" else "0"
        signature = (field, candidate)
        if candidate == legacy or signature in logged:
            continue
        try:
            emit(
                f"framefit5_headcarry_shadow_mismatch_{field}",
                producer=__name__,
                field=field,
                legacy_outcome=legacy,
                candidate_outcome=candidate[:160],
                dispatch=False,
            )
        except Exception:
            continue
        logged.add(signature)
    try:
        setattr(journal, "_framefit5_headcarry_shadow_logged", logged)
    except Exception:
        pass


def record_headcarry_rendered(journal: object | None, panel: Mapping[str, object]) -> None:
    """Record that the opted-in renderer contributed to a FRAMEFIT5 frame."""
    emit = getattr(journal, "emit", None)
    if not callable(emit):
        return
    try:
        emit(
            "framefit5_headcarry_rendered",
            producer=__name__,
            head_count=len(panel.get("heads", ())) if isinstance(panel.get("heads"), list) else 0,
            omitted_count=int(panel.get("omitted_count", 0)),
            deduplicated_count=int(panel.get("deduplicated_count", 0)),
            byte_count=int(panel.get("byte_count", 0)),
            dispatch=False,
        )
    except Exception:
        pass


__all__ = [
    "BYTE_BUDGET_ENV",
    "DEFAULT_BYTE_BUDGET",
    "MODE_ENV",
    "OWED_READS_ENV",
    "SCHEMA",
    "fit_headcarry_panel_for_request",
    "framefit5_headcarry_mode",
    "headcarry_byte_budget",
    "headcarry_owed_reads_enabled",
    "owed_read_rows",
    "record_headcarry_rendered",
    "record_headcarry_shadow_mismatches",
    "render_headcarry",
]
