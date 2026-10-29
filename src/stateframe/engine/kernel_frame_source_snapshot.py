"""Detached, revision-bound source inputs for FRAME_ONLY candidate assembly."""

from __future__ import annotations

import copy
import ast
import hashlib
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

from kernel_frame_optional_allocation import FrameAllocationReplanNeeded


def _current_window(root: Path, path: str, start: int, end: int,
                    revision: str, *, owner: str = "recent_editable_read") -> tuple[str, str] | None:
    """Render only exact, confined current file lines as a typed source window."""
    from kernel_request_source_window import (
        WINDOW_CONTENT_MARKER, _safe_framefit_path, _render_ranges,
        project_source_window,
    )
    from kernel_stateframe_headcarry import _read_source

    if _safe_framefit_path(path) != path or not 1 <= start <= end:
        return None
    raw = _read_source(root, path)
    if raw is None or "sha256:" + hashlib.sha256(raw).hexdigest() != revision:
        return None
    try:
        source = raw.decode("utf-8")
    except UnicodeError:
        return None
    lines = source.splitlines(keepends=True)
    if end > len(lines):
        return None
    rendered = project_source_window(_render_ranges(
        lines, [{"start_line": start, "end_line": end}], path=path,
        revision=revision, total_lines=len(lines)), owner=owner)
    if not rendered:
        return None
    receipt = (f"[read page] path={path} revision={revision} "
               f"lines={start}-{end}/{len(lines)} complete=true")
    rendered = rendered.replace(
        "\n" + WINDOW_CONTENT_MARKER + "\n",
        "\n" + receipt + "\n" + WINDOW_CONTENT_MARKER + "\n", 1,
    )
    return rendered, "".join(lines[start - 1:end])


def _partial_read_window(root: Path, path: str, start: int, end: int,
                         revision: str, row: Mapping[str, object],
                         full_window: tuple[str, str], *,
                         owner: str = "recent_editable_read") -> tuple[str, str] | None:
    """Keep an exact clipped read without claiming its boundary lines as coverage."""
    from kernel_request_source_window import (
        MODEL_WINDOW_MARKER, MODEL_WINDOW_SCHEMA, WINDOW_CONTENT_MARKER,
        WINDOW_END_MARKER, parse_model_source_window,
    )

    full_rendered, full_lines = full_window
    full_metadata = parse_model_source_window(full_rendered)
    total_lines = full_metadata.get("total_lines") if isinstance(full_metadata, Mapping) else None
    if type(total_lines) is not int:
        return None
    observed = row.get("content")
    expected = row.get("content_sha256")
    start_column, end_column = row.get("start_column"), row.get("end_column")
    if (not isinstance(observed, str) or not isinstance(expected, str)
            or type(start_column) is not int or start_column < 0
            or (end_column is not None and (type(end_column) is not int or end_column < 0))):
        return None
    lines = full_lines.splitlines(keepends=True)
    if not lines or start_column > len(lines[0]) or (
            end_column is not None and end_column > len(lines[-1])):
        return None
    clipped = lines.copy()
    if start == end:
        clipped[0] = clipped[0][start_column:end_column]
    else:
        clipped[0] = clipped[0][start_column:]
        if end_column is not None:
            clipped[-1] = clipped[-1][:end_column]
    exact = "".join(clipped)
    if (exact != observed or hashlib.sha256(exact.encode()).hexdigest() != expected):
        return None
    first_full = start + int(start_column > 0)
    last_full = end - int(end_column is not None and end_column < len(lines[-1]))
    if first_full > last_full:
        projection = {
            "schema": MODEL_WINDOW_SCHEMA, "path": path,
            "owner": owner, "revision": revision,
            "ranges": [], "total_lines": total_lines,
            "status": "PARTIAL_READ",
            "continuation": {"status": "READ_MORE", "omitted_lines": None,
                             "expansion_handle": None},
            "absence": "NONE",
            "partial_read": {"start_line": start, "end_line": end,
                             "start_column": start_column, "end_column": end_column,
                             "chars": len(exact), "content_sha256": expected},
        }
        receipt = (f"[read page] path={path} revision={revision} "
                   f"lines={start}-{end}/{total_lines} complete=true")
        ending = "" if exact.endswith("\n") else "\n"
        rendered = (f"[Current source: {path}]\n{MODEL_WINDOW_MARKER}\n"
                    + json.dumps(projection, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":"))
                    + f"\n{receipt}\n{WINDOW_CONTENT_MARKER}\n" + exact
                    + ending + WINDOW_END_MARKER)
        return rendered, exact
    interior = _current_window(root, path, first_full, last_full, revision,
                               owner=owner)
    if interior is None:
        return None
    rendered, _interior_source = interior
    old_receipt = (f"[read page] path={path} revision={revision} "
                   f"lines={first_full}-{last_full}/")
    new_receipt = (f"[read page] path={path} revision={revision} "
                   f"lines={start}-{end}/")
    rendered = rendered.replace(old_receipt, new_receipt, 1)
    ending = "" if exact.endswith("\n") else "\n"
    return (rendered + f"\n[Exact clipped read: {path}]\n" + exact + ending
            + f"[End exact clipped read: {path}]", exact)


def _read_rows(frame: Mapping[str, object], runtime: object) -> Iterator[Mapping[str, object]]:
    # The display ledger can shed every read under frame pressure. The host's
    # visibility candidates and typed action context still have authenticated
    # read receipts; neither uses the ReadWorkingSet presentation budget.
    candidates = getattr(runtime, "_read_reuse_visibility_candidates", None)
    if isinstance(candidates, list):
        for candidate in reversed(candidates):
            if not isinstance(candidate, Mapping):
                continue
            content, expected = candidate.get("content"), candidate.get("content_sha256")
            if (not isinstance(content, str) or not isinstance(expected, str)
                    or hashlib.sha256(content.encode()).hexdigest() != expected):
                continue
            yield {"kind": "read", "path": candidate.get("path"),
                   "revision": candidate.get("revision"),
                   "lines": {"start": candidate.get("start_line"),
                             "end": candidate.get("end_line")},
                   "content_sha256": expected,
                   "start_column": candidate.get("start_column", 0),
                   "end_column": candidate.get("end_column"),
                   "content": content}
    context = getattr(runtime, "_framefit_action_context", None)
    raw_ledger = context.get("ledger") if isinstance(context, Mapping) else None
    raw_rows = raw_ledger.get("rows") if isinstance(raw_ledger, Mapping) else None
    if isinstance(raw_rows, list):
        yield from (row for row in reversed(raw_rows)
                    if isinstance(row, Mapping) and row.get("kind") == "read")
    ledger = frame.get("action_ledger")
    if not isinstance(ledger, Mapping):
        return
    reads, compact = ledger.get("reads"), ledger.get("read_ledger")
    if isinstance(reads, list):
        yield from (row for row in reversed(reads) if isinstance(row, Mapping))
    if isinstance(compact, list):
        yield from (row for row in compact if isinstance(row, Mapping))


def _read_spans(row: Mapping[str, object]) -> list[tuple[int, int]]:
    ranges = row.get("ranges")
    if isinstance(row.get("lines"), Mapping):
        span = row["lines"]
        ranges = [[span.get("start"), span.get("end")]]
    if not isinstance(ranges, list):
        return []
    return [(start, end) for span in ranges
            if isinstance(span, (list, tuple)) and len(span) == 2
            for start, end in [span]
            if type(start) is int and type(end) is int and 1 <= start <= end]


def _latest_applied_edits(frame: Mapping[str, object], runtime: object) -> dict[str, Mapping[str, object]]:
    """Find typed applied edits without relying on the display ledger's budget."""
    context = getattr(runtime, "_framefit_action_context", None)
    ledger = context.get("ledger") if isinstance(context, Mapping) else None
    rows = ledger.get("rows") if isinstance(ledger, Mapping) else None
    projected = frame.get("action_ledger")
    edits = projected.get("edits") if isinstance(projected, Mapping) else None
    result: dict[str, Mapping[str, object]] = {}
    for source in (edits, rows):
        if not isinstance(source, list):
            continue
        for row in source:
            if not isinstance(row, Mapping) or row.get("kind") != "edit":
                continue
            path = row.get("path")
            receipt = row.get("receipt")
            if (isinstance(path, str) and isinstance(receipt, str)
                    and receipt.startswith(("str_replace applied.", "edit applied."))):
                result[path] = row
    return result


def _refreshed_edit_window(root: Path, path: str, start: int, end: int,
                           row: Mapping[str, object],
                           edit: Mapping[str, object]) -> tuple[str, str] | None:
    """Refresh the previously read span only after a verified applied edit."""
    from kernel_stateframe_headcarry import _read_source

    raw = _read_source(root, path)
    if raw is None:
        return None
    try:
        source = raw.decode("utf-8")
    except UnicodeError:
        return None
    revision = "sha256:" + hashlib.sha256(raw).hexdigest()
    if row.get("revision") == revision:
        return None
    old_text, new_text = edit.get("old_text"), edit.get("new_text")
    post_revision = edit.get("post_edit_revision")
    if (not isinstance(post_revision, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", post_revision) is None
            or (isinstance(new_text, str) and new_text not in source)):
        return None
    lines = source.splitlines(keepends=True)
    edit_lines = edit.get("lines")
    if (isinstance(edit_lines, Mapping)
            and type(edit_lines.get("start")) is int
            and type(edit_lines.get("end")) is int
            and isinstance(old_text, str) and isinstance(new_text, str)):
        delta = new_text.count("\n") - old_text.count("\n")
        if edit_lines["end"] < start:
            start += delta
            end += delta
        elif edit_lines["start"] <= end:
            end += delta
    if start < 1 or start > len(lines):
        return None
    end = min(max(start, end), len(lines))
    full = _current_window(root, path, start, end, revision)
    if full is None:
        return None
    start_column, end_column = row.get("start_column", 0), row.get("end_column")
    if start_column in (None, 0) and end_column is None:
        return full
    if (type(start_column) is not int or start_column < 0
            or (end_column is not None and (type(end_column) is not int or end_column < 0))):
        return None
    clipped_lines = full[1].splitlines(keepends=True)
    if (start_column > len(clipped_lines[0])
            or (end_column is not None and end_column > len(clipped_lines[-1]))):
        return None
    if start == end:
        clipped_lines[0] = clipped_lines[0][start_column:end_column]
    else:
        clipped_lines[0] = clipped_lines[0][start_column:]
        if end_column is not None:
            clipped_lines[-1] = clipped_lines[-1][:end_column]
    exact = "".join(clipped_lines)
    return _partial_read_window(
        root, path, start, end, revision,
        {"start_column": start_column, "end_column": end_column,
         "content": exact, "content_sha256": hashlib.sha256(exact.encode()).hexdigest()},
        full,
    )


def recent_editable_read_windows(runtime: object, frame: Mapping[str, object]) -> list[dict[str, object]]:
    """Newest three distinct path/range reads, proved against current editable bytes."""
    from kernel_read_reuse_visibility import editable_read_paths
    from kernel_stateframe_framefit5 import _workspace_root

    editable = editable_read_paths(runtime)
    if not editable:
        owner = getattr(runtime, "owner", None)
        task = getattr(owner, "task_row", None)
        values = task.get("editable_paths", ()) if isinstance(task, Mapping) else ()
        if isinstance(values, (list, tuple)):
            from kernel_request_source_window import _safe_framefit_path
            editable = {path for value in values
                        if (path := _safe_framefit_path(value)) is not None}
    if not editable:
        return []
    root = _workspace_root(runtime)
    result: list[dict[str, object]] = []
    seen: set[tuple[str, int, int]] = set()
    applied_edits = _latest_applied_edits(frame, runtime)
    for row in _read_rows(frame, runtime):
        if not isinstance(row, Mapping):
            continue
        path, revision = row.get("path"), row.get("revision")
        if (not isinstance(path, str) or path not in editable
                or not isinstance(revision, str)):
            continue
        for start, end in _read_spans(row):
            key = (path, start, end)
            if key in seen:
                continue
            current = _current_window(root, path, start, end, revision)
            refreshed = False
            if current is None and path in applied_edits:
                current = _refreshed_edit_window(
                    root, path, start, end, row, applied_edits[path],
                )
                refreshed = current is not None
            if current is None:
                continue
            content, exact = current
            expected = row.get("content_sha256")
            if (not refreshed and isinstance(expected, str)
                    and hashlib.sha256(exact.encode()).hexdigest() != expected):
                partial = _partial_read_window(root, path, start, end, revision, row, current)
                if partial is None:
                    continue
                content, exact = partial
            seen.add(key)
            result.append({"path": path, "start_line": start, "end_line": end,
                           "content": content, "source_chars": len(exact)})
            if len(result) == 3:
                return result
    return result


def public_check_source_windows(runtime: object, state: Mapping[str, object]) -> list[dict[str, str]]:
    """Retain only authenticated ranges the model read from named public checks."""
    from kernel_request_source_window import (
        WINDOW_CONTENT_MARKER, _safe_framefit_path, _render_ranges,
        project_source_window,
    )
    from kernel_stateframe_framefit5 import _workspace_root
    from kernel_stateframe_headcarry import _read_source

    contract = state.get("completion_contract")
    if isinstance(contract, Mapping):
        selectors = contract.get("required_public_checks", ())
    else:
        owner = getattr(runtime, "owner", None)
        task = getattr(owner, "task_row", None)
        selectors = task.get("public_check_paths", ()) if isinstance(task, Mapping) else ()
    if isinstance(selectors, str):
        selectors = [selectors]
    if not isinstance(selectors, (list, tuple)):
        return []
    root = _workspace_root(runtime)
    result = []
    seen: set[str] = set()
    for selector in selectors:
        if not isinstance(selector, str):
            continue
        path, _, name = selector.partition("::")
        if _safe_framefit_path(path) != path or path in seen:
            continue
        matching = next(((row, start, end) for row in _read_rows(state, runtime)
                         if row.get("path") == path and isinstance(row.get("revision"), str)
                         for start, end in _read_spans(row)
                         if _current_window(root, path, start, end, row["revision"],
                                            owner="required_public_check") is not None), None)
        if matching is None:
            continue
        seen.add(path)
        read, read_start, read_end = matching
        raw = _read_source(root, path)
        if raw is None:
            continue
        try:
            source = raw.decode("utf-8")
        except UnicodeError:
            continue
        lines = source.splitlines(keepends=True)
        if not lines:
            continue
        revision = str(read["revision"])
        exact = "".join(lines[read_start - 1:read_end])
        expected = read.get("content_sha256")
        if isinstance(expected, str) and hashlib.sha256(exact.encode()).hexdigest() != expected:
            whole = _current_window(root, path, read_start, read_end, revision,
                                    owner="required_public_check")
            partial = (_partial_read_window(
                root, path, read_start, read_end, revision, read, whole,
                owner="required_public_check") if whole is not None else None)
            if partial is None:
                continue
            content, exact = partial
            result.append({"full": content, "excerpt": content,
                           "assertions": content, "full_chars": len(exact),
                           "excerpt_chars": len(exact),
                           "assertions_chars": len(exact)})
            continue
        def render(spans: list[tuple[int, int]]) -> str:
            window = _render_ranges(lines, [
                {"start_line": start, "end_line": end} for start, end in spans],
                path=path, revision=revision, total_lines=len(lines))
            projected = project_source_window(window, owner="required_public_check") or ""
            receipt = (f"[read page] path={path} revision={revision} "
                       f"lines={read_start}-{read_end}/{len(lines)} complete=true")
            return projected.replace("\n" + WINDOW_CONTENT_MARKER + "\n",
                                     "\n" + receipt + "\n" + WINDOW_CONTENT_MARKER + "\n", 1)
        full = render([(read_start, read_end)])
        bodies: list[tuple[int, int]] = []
        assertions: list[tuple[int, int]] = []
        if path.endswith(".py"):
            try:
                tree = ast.parse(source)
            except SyntaxError:
                tree = None
            if tree is not None:
                selected = [node for node in ast.walk(tree)
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and (node.name == name.split("::")[-1] if name
                                 else node.name.startswith("test_"))
                            and read_start <= node.lineno
                            and (node.end_lineno or node.lineno) <= read_end]
                selected.sort(key=lambda node: node.lineno)
                bodies = [(node.lineno, node.end_lineno or node.lineno)
                          for node in selected[:4]]
                assertions = [(item.lineno, item.end_lineno or item.lineno)
                              for node in selected for item in ast.walk(node)
                              if isinstance(item, ast.Assert)]
        excerpt = render(bodies) if bodies else full
        assert_window = render(sorted(assertions[:8])) if assertions else excerpt
        def span_chars(spans: Sequence[tuple[int, int]]) -> int:
            return sum(len("".join(lines[start - 1:end])) for start, end in spans)

        result.append({"full": full, "excerpt": excerpt, "assertions": assert_window,
                       "full_chars": len(exact),
                       "excerpt_chars": span_chars(bodies) if bodies else len(exact),
                       "assertions_chars": (span_chars(assertions[:8]) if assertions
                                            else span_chars(bodies) if bodies else len(exact))})
    return result


def fit_recent_read_windows(body: Mapping[str, object], windows: Sequence[Mapping[str, object]],
                            *, public_checks: Sequence[Mapping[str, object]] = (),
                            protected_messages: Sequence[Mapping[str, object]] = (),
                            newest_user_suffix: Sequence[Mapping[str, object]] = (),
                            deferred_source_paths: Sequence[str] = (),
                            fits: object) -> dict[str, object]:
    """Fit source messages against the caller's measured complete-wire budget."""
    from kernel_request_source_window import parse_model_source_window, parse_window

    base = copy.deepcopy(dict(body))
    messages = base.get("messages")
    if not isinstance(messages, list):
        return base
    messages[:] = [message for message in messages if not (
        isinstance(message, Mapping) and isinstance(message.get("content"), str)
        and (str(message["content"]).startswith("kept ")
             and " recent reads; about " in str(message["content"])
             or (meta := parse_model_source_window(message["content"])) is not None
             and meta.get("owner") in {"recent_editable_read", "required_public_check"}))]
    def window_key(content: object) -> tuple[object, ...] | None:
        metadata = parse_model_source_window(content) or parse_window(content)
        if not isinstance(metadata, Mapping):
            return None
        ranges = metadata.get("ranges")
        if not isinstance(ranges, list):
            return None
        return (metadata.get("path"), metadata.get("revision"),
                tuple((row.get("start_line"), row.get("end_line"))
                      for row in ranges if isinstance(row, Mapping)))

    deferred_paths = frozenset(deferred_source_paths)

    def candidate(reads: Sequence[Mapping[str, object]], checks: Sequence[str],
                  check_chars: int = 0, *, release_deferred: bool = False) -> dict[str, object]:
        trial = copy.deepcopy(base)
        rows = trial["messages"]
        selected = []
        for row in reads:
            metadata = parse_model_source_window(row.get("content"))
            if isinstance(metadata, Mapping):
                ranges = metadata.get("ranges", ())
                if not ranges:
                    selected.append((metadata.get("path"), metadata.get("revision"),
                                     row.get("start_line"), row.get("end_line")))
                for span in ranges:
                    if isinstance(span, Mapping):
                        selected.append((metadata.get("path"), metadata.get("revision"),
                                         span.get("start_line"), span.get("end_line")))
        def superseded(message: object) -> bool:
            if not isinstance(message, Mapping) or message in protected_messages:
                return False
            metadata = parse_model_source_window(message.get("content"))
            if not isinstance(metadata, Mapping) or metadata.get("owner") not in {
                    "UNKNOWN_OWNER", "read_reuse_visibility"}:
                return False
            if (release_deferred and metadata.get("path") in deferred_paths
                    and any(metadata.get("path") == path for path, *_rest in selected)):
                # The finalizer has an executable read route for this deferred
                # path. Keep the observed current page, not older closure copies.
                return True
            ranges = metadata.get("ranges")
            if any(metadata.get("path") == path and metadata.get("revision") != revision
                   for path, revision, _start, _end in selected):
                return True
            return bool(isinstance(ranges, list) and ranges and all(
                isinstance(span, Mapping)
                and type(span.get("start_line")) is int
                and type(span.get("end_line")) is int
                and any(
                    (metadata.get("path"), metadata.get("revision")) == (path, revision)
                    and type(start) is int and type(end) is int
                    and start <= span["start_line"] and end >= span["end_line"]
                    for path, revision, start, end in selected)
                for span in ranges))
        rows[:] = [message for message in rows if not superseded(message)]
        existing = {key for message in rows
                    if isinstance(message, Mapping)
                    if (key := window_key(message.get("content"))) is not None}
        position = next((index for index, message in enumerate(rows)
                         if message.get("role") in {"assistant", "tool"}
                         or str(message.get("content") or "").startswith("[FRAMEFIT5_VOLATILE_TAIL]")),
                        len(rows))
        if newest_user_suffix:
            suffix = [dict(message) for message in newest_user_suffix]
            suffix_start = position - len(suffix)
            if suffix_start < 0 or rows[suffix_start:position] != suffix:
                raise FrameAllocationReplanNeeded(
                    "newest user suffix changed before recent-read fitting"
                )
            position = suffix_start
        source_chars = sum(int(row.get("source_chars", 0)) for row in reads) + check_chars
        additions = [{"role": "user", "content": content}
                     for content in [*(str(row["content"]) for row in reads), *checks]
                     if content and window_key(content) not in existing]
        additions.append({"role": "user", "content":
                          f"kept {len(reads)} recent reads; about {source_chars} chars of source fit"})
        rows[position:position] = additions
        return trial
    check_levels = ("full", "excerpt", "assertions") if public_checks else ("full",)
    for count in range(min(3, len(windows)), -1, -1):
        for release_deferred in ((False, True) if deferred_paths and count else (False,)):
            for level in check_levels:
                checks = [str(row.get(level) or "") for row in public_checks]
                check_chars = sum(int(row.get(level + "_chars", 0)) for row in public_checks)
                trial = candidate(
                    windows[:count], checks, check_chars,
                    release_deferred=release_deferred,
                )
                if fits(trial):
                    return trial
    for count in range(min(3, len(windows)), -1, -1):
        for release_deferred in ((False, True) if deferred_paths and count else (False,)):
            trial = candidate(windows[:count], (), release_deferred=release_deferred)
            if fits(trial):
                return trial
    return base


def allocation_semantic_messages(
    messages: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Keep original source/user ownership separate from regenerated host panels.

    These panels remain in the prepared draft and are explicitly assembled as
    current host facts. They are not immutable user instructions in the newest
    suffix: owner admission and source deferral must be able to refresh them.
    """
    from kernel_stateframe_framefit5 import (
        _is_framefit5_host_context_panel, _is_framefit5_source_notice,
    )

    return [copy.deepcopy(dict(message)) for message in messages
            if not (_is_framefit5_host_context_panel(message)
                    or _is_framefit5_source_notice(message))]


def frame_source_snapshot(
    runtime: object, plan: Mapping[str, object] | None,
) -> dict[str, str]:
    """Fill absent cache entries from selected current workspace files only.

    The empirical cache is enrichment, not the closure's source authority.
    Never replace a stale cached/snapshotted value with a fresh file silently:
    that is authority drift and requires a new preparation.
    """
    from kernel_request_source_window import _safe_framefit_path, source_revision
    from kernel_public_localization import public_path
    from kernel_stateframe_framefit5 import _workspace_root
    from kernel_stateframe_headcarry import _read_source

    sources = {
        str(path): row["content"]
        for path, row in getattr(runtime, "_empirical_frame_sources", {}).items()
        if isinstance(row, Mapping) and isinstance(row.get("content"), str)
        and public_path(path) == str(path)
    }
    if not isinstance(plan, Mapping):
        return sources
    snapshots = plan.get("source_contents", {})
    snapshots = snapshots if isinstance(snapshots, Mapping) else {}
    revisions = plan.get("revisions", {})
    revisions = revisions if isinstance(revisions, Mapping) else {}
    root = _workspace_root(runtime)
    checked: set[str] = set()
    for unit in plan.get("units", ()):
        path = unit.get("path") if isinstance(unit, Mapping) else None
        if (not isinstance(path, str) or _safe_framefit_path(path) != path
                or public_path(path) != path):
            raise FrameAllocationReplanNeeded("source proof has an unsafe selected path")
        if path in checked:
            continue
        revision = revisions.get(path)
        cached = sources.get(path)
        if cached is not None and source_revision(cached) != revision:
            if getattr(runtime, "_source_replan_in_progress", False):
                # A fresh preparation can rebuild an empirical cache from a
                # historical read. That cache is optional; the new closure
                # snapshot and workspace must still agree below.
                sources.pop(path)
            else:
                raise FrameAllocationReplanNeeded("source proof snapshot revision changed")
        snapshot = snapshots.get(path)
        if snapshot is not None and (
                not isinstance(snapshot, str) or source_revision(snapshot) != revision):
            raise FrameAllocationReplanNeeded("source proof snapshot revision changed")
        if path not in sources and isinstance(snapshots.get(path), str):
            sources[path] = snapshots[path]
        # Existing source reader rejects symlinks and confines reads to root.
        # Even a complete cache is not evidence that workspace bytes stayed put.
        raw = _read_source(root, path)
        try:
            source = raw.decode("utf-8") if raw is not None else None
        except UnicodeError:
            source = None
        if source is None or source_revision(source) != revision:
            raise FrameAllocationReplanNeeded("source proof workspace revision unavailable")
        sources[path] = source
        checked.add(path)
    return sources


def retire_f4_deferred_carriers(
    prepared: Sequence[Mapping[str, object]],
    semantic_messages: Sequence[Mapping[str, object]],
    plan: Mapping[str, object],
    sources: Mapping[str, str] | None,
) -> list[Mapping[str, object]]:
    """Replace generated F4 paging pointers with the current allocation's policy.

    A paging pointer is not source text. The FRAME_ONLY assembler materializes
    retained units and authors canonical notices for deferred units; leaving the
    old Current-source wrapper makes the exact verifier treat the pointer as code.
    Original newest user facts and malformed/stale pointers remain untouched.
    """
    from kernel_request_source_window import source_revision
    from kernel_stateframe_f4_fit import parse_deferred_source_receipt
    from kernel_stateframe_framefit5 import _task_and_latest_turn

    _task, newest, _task_index, _action_index = _task_and_latest_turn(semantic_messages)
    revisions = plan.get("revisions", {})
    selected = {row.get("path") for row in plan.get("units", ()) if isinstance(row, Mapping)}
    result = []
    for message in prepared:
        content = message.get("content")
        if message.get("role") != "user" or message in newest or not isinstance(content, str):
            result.append(message)
            continue
        header, separator, body = content.partition("\n")
        route = parse_deferred_source_receipt(body) if separator else None
        path = route.get("path") if route is not None else None
        source = sources.get(path) if isinstance(sources, Mapping) else None
        if (
            route is not None and path in selected and isinstance(source, str)
            and isinstance(revisions, Mapping)
            and header in {f"[Current source: {path}]", f"[Working source: {path}]"}
            and route["revision"] == revisions.get(path) == source_revision(source)
            and route["start_line"] == 1
            and route["end_line"] == len(source.splitlines())
        ):
            continue
        result.append(message)
    return result
