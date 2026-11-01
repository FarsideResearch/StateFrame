"""Bounded, revision-bound code locations for pressured source selection.

The records identify symbols, not positions.  Positions and bytes are resolved
from the current admitted source each time a request is prepared.
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
import stat
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path


ENV = "STATEFRAME_CODE_BOOKMARKS"
MAX_BOOKMARKS = 8
MAX_SOURCE_BYTES = 512_000
MAX_FAILURE_BYTES = 64_000
MAX_EVIDENCE_FRAMES = 24
MAX_DISCOVERY_PATHS = 64
MAX_UNIT_LINES = 120
SMALL_FILE_MAX_LINES = 60
SMALL_FILE_MAX_BYTES = 2048
MAX_NEIGHBOR_EDGES = 64


def enabled() -> bool:
    value = os.environ.get(ENV, "on").strip().lower()
    if value not in {"on", "off"}:
        raise ValueError(f"{ENV} must be on or off")
    return value == "on"


def _source(root: Path, path: str) -> tuple[str, str] | None:
    from kernel_public_localization import public_path

    if (public_path(path) != path or not path.endswith(".py")
            or any(ord(char) < 32 or char in "\u202a\u202b\u202c\u202d\u202e"
                   for char in path)):
        return None
    raw = _confined_source_bytes(root, path)
    if raw is None:
        return None
    try:
        source = raw.decode("utf-8")
    except UnicodeError:
        return None
    return source, "sha256:" + hashlib.sha256(raw).hexdigest()


def _confined_source_bytes(root: Path, path: str) -> bytes | None:
    """Open each admitted component relative to its parent, without following links."""
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    file_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    opened: list[int] = []
    try:
        root = Path(root)
        if not root.is_absolute():
            return None
        parent = os.open(root.anchor, directory_flags)
        opened.append(parent)
        for part in root.parts[1:]:
            parent = os.open(part, directory_flags, dir_fd=parent)
            opened.append(parent)
        parts = path.split("/")
        for part in parts[:-1]:
            parent = os.open(part, directory_flags, dir_fd=parent)
            opened.append(parent)
        source_fd = os.open(parts[-1], file_flags, dir_fd=parent)
        opened.append(source_fd)
        if not stat.S_ISREG(os.fstat(source_fd).st_mode):
            return None
        chunks: list[bytes] = []
        remaining = MAX_SOURCE_BYTES + 1
        while remaining:
            chunk = os.read(source_fd, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        return raw if len(raw) <= MAX_SOURCE_BYTES else None
    except OSError:
        return None
    finally:
        for descriptor in reversed(opened):
            os.close(descriptor)


def _outline(source: str) -> list[dict[str, object]]:
    """Optional syntax discovery must never prevent ordinary source selection."""
    from kernel_paged_source import symbol_outline

    try:
        return symbol_outline(source)
    except Exception:
        # Parser and outline visitor implementations are advisory here.
        return []


def _symbol_at(source: str, line: int, name: str = "") -> dict[str, object] | None:
    rows = _outline(source)
    matching = [row for row in rows
                if int(row["start_line"]) <= line <= int(row["end_line"])]
    if name:
        matching = [row for row in matching
                    if row["name"] == name or row["qualified_name"] == name]
    if not matching:
        return None
    matching.sort(key=lambda row: (int(row["end_line"]) - int(row["start_line"]),
                                   str(row["qualified_name"])))
    selected = matching[0]
    if sum(row["qualified_name"] == selected["qualified_name"] for row in rows) != 1:
        return None
    return selected


def _decorated_start(source: str, definition_line: int, end_line: int) -> int:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError, OverflowError):
        return definition_line
    nodes = [node for node in ast.walk(tree)
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
             and node.lineno == definition_line and node.end_lineno == end_line]
    if len(nodes) != 1:
        return definition_line
    decorators = nodes[0].decorator_list
    return min((part.lineno for part in decorators), default=definition_line)


def _unit_span(start: int, end: int, rank: int,
               evidence_line: object) -> tuple[int, int, str]:
    if rank == 1 and end - start + 1 <= MAX_UNIT_LINES:
        return start, end, "full"
    if rank == 1:
        anchor = (evidence_line if type(evidence_line) is int
                  and start <= evidence_line <= end else start)
        return max(start, anchor - 30), min(end, anchor + 30), "slice"
    return start, min(end, start + 7), "signature"


def _compact_span(start: int, end: int, evidence_line: object) -> tuple[int, int]:
    anchor = (evidence_line if type(evidence_line) is int
              and start <= evidence_line <= end else start)
    return max(start, anchor - 10), min(end, anchor + 10)


def _failure_receipt(runtime: object, verdict: Mapping[str, object]) -> Mapping[str, object] | None:
    tested_revision = verdict.get("tested_revision")
    if (verdict.get("status") != "FAIL" or not isinstance(tested_revision, str)
            or not tested_revision or tested_revision != verdict.get("current_revision")):
        return None
    receipt_id = verdict.get("check_receipt_id")
    if not isinstance(receipt_id, str) or not receipt_id:
        return None
    # Journal receipts are the common evidence on uninterrupted and resumed
    # builds. Prefer them even if this process still has the private output.
    context = getattr(runtime, "_framefit_action_context", None)
    records = context.get("journal_receipt_records") if isinstance(context, Mapping) else None
    if isinstance(records, Sequence) and not isinstance(records, (str, bytes, bytearray)):
        for receipt in reversed(records):
            if (isinstance(receipt, Mapping) and receipt.get("receipt_id") == receipt_id
                    and receipt.get("revision") == verdict.get("tested_revision")
                    and str(receipt.get("status") or "").upper()
                    in {"FAIL", "FAILED", "BEHAVIOR_FAIL"}):
                return receipt
    private = getattr(runtime, "_check_output_receipts", None)
    if isinstance(private, Mapping) and isinstance(private.get(receipt_id), Mapping):
        receipt = private[receipt_id]
        if (receipt.get("revision") == verdict.get("tested_revision")
                and str(receipt.get("status") or "").upper()
                in {"FAIL", "FAILED", "BEHAVIOR_FAIL"}):
            return receipt
    latest = getattr(runtime, "_latest_check_receipt", None)
    if (isinstance(latest, Mapping) and latest.get("receipt_id") == receipt_id
            and latest.get("revision") == verdict.get("tested_revision")
            and str(latest.get("status") or "").upper() in {"FAIL", "FAILED", "BEHAVIOR_FAIL"}
            and isinstance(getattr(runtime, "host_check_receipts", None), list)
            and any(isinstance(row, Mapping) and row == latest
                    for row in runtime.host_check_receipts)):
        return latest
    return None


def _failure_receipts(runtime: object, verdict: Mapping[str, object]) -> list[Mapping[str, object]]:
    latest = _failure_receipt(runtime, verdict)
    if latest is None:
        return []
    context = getattr(runtime, "_framefit_action_context", None)
    records = context.get("journal_receipt_records") if isinstance(context, Mapping) else None
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes, bytearray)):
        return [latest]
    current = verdict["tested_revision"]
    by_selector: dict[str, Mapping[str, object]] = {}
    for row in records:
        if not isinstance(row, Mapping) or row.get("kind") != "check_receipt":
            continue
        if row.get("revision") != current or not isinstance(row.get("receipt_id"), str):
            continue
        selector = row.get("selector")
        key = selector if isinstance(selector, str) and selector else row["receipt_id"]
        by_selector[key] = row
    failures = [row for row in by_selector.values()
                if str(row.get("status") or "").upper() in {"FAIL", "FAILED", "BEHAVIOR_FAIL"}]
    if not any(row.get("receipt_id") == latest.get("receipt_id") for row in failures):
        failures.append(latest)
    return failures


def _failure_text(receipt: Mapping[str, object]) -> str:
    return "\n".join(str(receipt.get(key) or "")[:MAX_FAILURE_BYTES] for key in (
        "assertion_context", "stdout", "stderr", "assertion_context_preview", "summary",
    ))[:MAX_FAILURE_BYTES]


def _evidence_locations(text: str, paths: Sequence[str]) -> list[tuple[str, int, str]]:
    from tracebacks import repo_index_from_paths, traceback_signal

    try:
        signal = traceback_signal(text, repo_index_from_paths(paths))
    except (TypeError, ValueError, RecursionError):
        signal = {}
    result: list[tuple[str, int, str]] = []
    parsed = signal.get("evidence")
    for traceback in getattr(parsed, "tracebacks", ()):
        for frame in getattr(traceback, "frames", ()):
            path, line = getattr(frame, "repo_path", None), getattr(frame, "line", None)
            if path in paths and type(line) is int and line > 0:
                result.append((path, line, str(getattr(frame, "function", "") or "")))
                if len(result) > MAX_EVIDENCE_FRAMES:
                    result.pop(0)
    if not result:
        for anchor in signal.get("anchors", ())[:MAX_EVIDENCE_FRAMES]:
            path, line = anchor.get("repo_path"), anchor.get("line")
            if path in paths and type(line) is int and line > 0:
                result.append((path, line, str(anchor.get("function") or "")))
    # Pytest's ordinary assertion report is a location, not a traceback frame.
    admitted = set(paths)
    for match in re.finditer(r"(?m)(?:^|\\n)([A-Za-z0-9_./-]+\.py):(\d+):\s+", text):
        path, line = match.group(1), int(match.group(2))
        if path in admitted and line > 0:
            result.append((path, line, ""))
        if len(result) >= MAX_EVIDENCE_FRAMES:
            break
    return result


def _read_locations(frame: Mapping[str, object], runtime: object,
                    allowed: set[str]) -> list[tuple[str, int, int]]:
    from kernel_frame_source_snapshot import _read_rows, _read_spans

    result: list[tuple[str, int, int]] = []
    for row in _read_rows(frame, runtime):
        path = row.get("path")
        if not isinstance(path, str) or path not in allowed:
            continue
        result.extend((path, start, end) for start, end in _read_spans(row))
        if len(result) >= MAX_BOOKMARKS * 4:
            break
    return result


def _assertion_targets(root: Path, paths: Sequence[str],
                       locations: Sequence[tuple[str, int, str]]) -> list[tuple[str, int, str]]:
    """Resolve direct Python imports used by a failing public assertion."""
    admitted = set(paths)
    result: list[tuple[str, int, str]] = []
    for test_path, line, _name in locations[:MAX_EVIDENCE_FRAMES]:
        if not test_path.startswith("tests/"):
            continue
        current = _source(root, test_path)
        if current is None:
            continue
        try:
            tree = ast.parse(current[0])
        except (SyntaxError, ValueError, RecursionError, MemoryError, OverflowError):
            continue
        imports: dict[str, tuple[str, str]] = {}
        def module_path(module: str) -> str | None:
            stem = module.replace(".", "/")
            for path in (stem + ".py", stem + "/__init__.py"):
                if path in admitted:
                    return path
            return None

        for statement in ast.walk(tree):
            if getattr(statement, "lineno", line + 1) > line:
                continue
            if isinstance(statement, ast.ImportFrom) and statement.module:
                for alias in statement.names:
                    path = module_path(statement.module + "." + alias.name)
                    if path:
                        imports[alias.asname or alias.name] = (path, "")
                        continue
                    path = module_path(statement.module)
                    if path:
                        imports[alias.asname or alias.name] = (path, alias.name)
            elif isinstance(statement, ast.Import):
                for alias in statement.names:
                    path = module_path(alias.name)
                    if path:
                        imports[alias.asname or alias.name.split(".")[0]] = (path, "")
            elif isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target, value = statement.targets[0], statement.value
                if (isinstance(target, ast.Name) and isinstance(value, ast.Call)
                        and isinstance(value.func, ast.Attribute)
                        and value.func.attr == "import_module" and value.args
                        and isinstance(value.args[0], ast.Constant)
                        and isinstance(value.args[0].value, str)):
                    path = module_path(value.args[0].value)
                    if path:
                        imports[target.id] = (path, "")
        assertions = [node for node in ast.walk(tree)
                      if isinstance(node, ast.Assert)
                      and node.lineno <= line <= (node.end_lineno or node.lineno)]
        for assertion in assertions[:4]:
            names: list[tuple[str, str]] = []
            expressions: list[ast.AST] = [assertion]
            # Follow a bounded local value back to the call that produced the
            # asserted result. Pytest often asserts a variable, not the call.
            used = {node.id for node in ast.walk(assertion)
                    if isinstance(node, ast.Name)}
            parents = [node for node in ast.walk(tree)
                       if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                       and node.lineno <= assertion.lineno <= (node.end_lineno or node.lineno)]
            if parents:
                parent = min(parents, key=lambda node: node.end_lineno - node.lineno)
                assignments = [node for node in parent.body
                               if isinstance(node, (ast.Assign, ast.AnnAssign))
                               and node.lineno < assertion.lineno]
                for assignment in sorted(assignments, key=lambda node: node.lineno,
                                         reverse=True)[:32]:
                    targets = (assignment.targets if isinstance(assignment, ast.Assign)
                               else [assignment.target])
                    bound = {node.id for target in targets for node in ast.walk(target)
                             if isinstance(node, ast.Name)}
                    if used & bound and assignment.value is not None:
                        expressions.append(assignment.value)
                        used.update(node.id for node in ast.walk(assignment.value)
                                    if isinstance(node, ast.Name))
                    if len(expressions) >= 8:
                        break
                # A final assertion may verify calls recorded by earlier
                # assertions in the same test (for example three consumers
                # delegating to one package runner). Those call sites are
                # evidence from the failed test, bounded to this function.
                earlier = [node for node in parent.body
                           if isinstance(node, (ast.Assert, ast.Assign))
                           and node.lineno < assertion.lineno
                           and any(isinstance(child, ast.Name) and child.id in imports
                                   for child in ast.walk(node))]
                expressions.extend(earlier[-8:])
            for expression in (node for root_node in expressions
                                for node in ast.walk(root_node)):
                if (isinstance(expression, ast.Call)
                        and isinstance(expression.func, ast.Name)
                        and expression.func.id in imports):
                    names.append(imports[expression.func.id])
                elif (isinstance(expression, ast.Call)
                      and isinstance(expression.func, ast.Attribute)
                      and isinstance(expression.func.value, ast.Call)
                      and isinstance(expression.func.value.func, ast.Attribute)
                      and isinstance(expression.func.value.func.value, ast.Name)
                      and expression.func.value.func.value.id in imports):
                    owner = expression.func.value.func
                    path, imported = imports[owner.value.id]
                    symbol = imported + "." if imported else ""
                    names.append((path, symbol + owner.attr + "." + expression.func.attr))
                elif (isinstance(expression, ast.Attribute)
                      and isinstance(expression.value, ast.Name)
                      and expression.value.id in imports):
                    path, imported = imports[expression.value.id]
                    names.append((path, expression.attr if not imported else
                                  imported + "." + expression.attr))
            for path, name in dict.fromkeys(names[:12]):
                source = _source(root, path)
                if source is None:
                    continue
                matches = [row for row in _outline(source[0])
                           if row["qualified_name"] == name or row["name"] == name]
                if len(matches) == 1:
                    result.append((path, int(matches[0]["start_line"]), name))
                elif not matches:
                    result.append((path, 1, name or "<module>"))
    return result


def _direct_neighbors(
    root: Path, roots: Sequence[Mapping[str, object]],
) -> list[tuple[str, int, str]]:
    """Bound direct calls within the admitted file; uncertain calls add nothing."""
    result: list[tuple[str, int, str]] = []
    for bookmark in roots[:MAX_BOOKMARKS]:
        path = str(bookmark["path"])
        current = _source(root, path)
        if current is None:
            continue
        try:
            tree = ast.parse(current[0])
        except (SyntaxError, ValueError, RecursionError, MemoryError, OverflowError):
            continue
        rows = _outline(current[0])
        target = [row for row in rows if row["qualified_name"] == bookmark["symbol"]]
        if len(target) != 1:
            continue
        target_name = str(target[0]["name"])
        nodes = {node.lineno: node for node in ast.walk(tree)
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        target_node = nodes.get(target[0]["start_line"])
        target_calls = _calls_in(target_node) if target_node is not None else set()
        for row in rows[:MAX_NEIGHBOR_EDGES]:
            if row["qualified_name"] == bookmark["symbol"]:
                continue
            node = nodes.get(row["start_line"])
            if node is None:
                continue
            if target_name in _calls_in(node):
                result.append((path, int(row["start_line"]), str(row["qualified_name"])))
            if str(row["name"]) in target_calls:
                result.append((path, int(row["start_line"]), str(row["qualified_name"])))
            if len(result) >= MAX_NEIGHBOR_EDGES:
                return result
    return result


def _calls_in(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    result: set[str] = set()
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        if isinstance(call.func, ast.Name):
            result.add(call.func.id)
        elif (isinstance(call.func, ast.Attribute)
              and isinstance(call.func.value, ast.Name)
              and call.func.value.id in {"self", "cls"}):
            result.add(call.func.attr)
    return result


def collect(runtime: object, frame: Mapping[str, object],
            decision: MutableMapping[str, object] | None = None) -> list[dict[str, object]]:
    """Use only current failed checks and authenticated read/edit receipts."""
    def finish(rows: list[dict[str, object]], reason: str) -> list[dict[str, object]]:
        if decision is not None:
            decision["reason"] = reason
        return rows

    if not enabled():
        return finish([], "disabled")
    from kernel_public_localization import public_path
    from kernel_read_reuse_visibility import editable_read_paths
    from kernel_stateframe_framefit5 import _workspace_root

    read_facts = frame.get("read_facts")
    if not isinstance(read_facts, list):
        return finish([], "no_read_facts")
    editable = editable_read_paths(runtime)
    read_paths = {row["path"] for row in read_facts if isinstance(row, Mapping)
                  and isinstance(row.get("path"), str) and row["path"] in editable}
    # This boundary preserves the released request byte for byte on small work.
    if len(read_paths) <= 3:
        return finish([], "small_read_set")
    prior = getattr(runtime, "_code_bookmark_carry", ())
    prior = prior[:MAX_BOOKMARKS] if isinstance(prior, list) else ()
    verdict = frame.get("last_verdict")
    receipts = _failure_receipts(runtime, verdict) if isinstance(verdict, Mapping) else []
    selected_receipt = next((row for row in receipts
                             if row.get("receipt_id") == verdict.get("check_receipt_id")), None)
    if (selected_receipt is not None
            and (str(selected_receipt.get("outcome") or "").upper()
                 in {"INVALID_CHECK", "ENVIRONMENT", "NOT_RUN", "UNKNOWN"}
                 or str(selected_receipt.get("failure_phase") or "").upper()
                 in {"COLLECTION", "IMPORT", "SETUP", "HOST", "EXECUTION", "TIMEOUT"})):
        return finish([], "non_behavior_failure")
    if not receipts and not prior:
        reason = ("stale_revision" if isinstance(verdict, Mapping)
                  and verdict.get("status") == "FAIL"
                  and verdict.get("tested_revision") != verdict.get("current_revision")
                  else "no_failed_receipt")
        return finish([], reason)
    root = _workspace_root(runtime)
    protected_tests = getattr(runtime, "protected_tests", None)
    protected_tests = protected_tests if isinstance(protected_tests, Mapping) else {}
    public_paths = {row.get("path") for row in read_facts[-MAX_DISCOVERY_PATHS:]
                    if isinstance(row, Mapping) and isinstance(row.get("path"), str)}
    paths = sorted(path for path in public_paths if _source(root, path) is not None)
    if not paths:
        return finish([], "no_paths")
    locations = [location for receipt in receipts
                 for location in _evidence_locations(_failure_text(receipt), paths)]
    if receipts and not locations:
        # A failed check without a source location cannot authenticate a
        # repair bookmark. In particular, bounded journal records may omit
        # stdout/stderr from other runners or collection failures.
        return finish([], "no_failed_location")
    records: dict[tuple[str, str], dict[str, object]] = {}
    age = 0

    def add(path: str, line: int, name: str, rank: int, origin: str) -> None:
        nonlocal age
        if path not in public_paths or len(name) > 256:
            return
        current = _source(root, path)
        if current is None:
            return
        source, revision = current
        row = _symbol_at(source, line, name) if name != "<module>" else None
        if row is None and name != "<module>":
            if rank == 1 and origin == "failed_check" and name:
                if any(existing_path == path
                       and existing_symbol.rsplit(".", 1)[-1] == name
                       for existing_path, existing_symbol in records):
                    return
                age += 1
                records[(path, name)] = {
                    "path": path, "symbol": name, "rank": rank, "origin": origin,
                    "age": age, "evidence_line": line, "revision": revision,
                    "start_line": line, "end_line": line,
                    "tier": "outline",
                    "read_only": (path not in editable or path.startswith("tests/")
                                  or root / path in protected_tests or path in protected_tests),
                }
            return
        symbol = str(row["qualified_name"]) if row is not None else "<module>"
        if len(symbol) > 256 or public_path(path) != path:
            return
        key = (path, symbol)
        old = records.get(key)
        if old is not None and int(old["rank"]) <= rank:
            if int(old["rank"]) == rank and origin in {"targeted_read", "applied_edit"}:
                age += 1
                old["age"] = age
            return
        lines = source.splitlines(keepends=True)
        if not lines:
            return
        start, end = ((int(row["start_line"]), int(row["end_line"]))
                      if row is not None else (1, min(len(lines), MAX_UNIT_LINES)))
        if row is not None:
            start = _decorated_start(source, start, end)
        unit = "".join(lines[start - 1:end])
        shown_start, shown_end, tier = _unit_span(start, end, rank, line)
        shown = "".join(lines[shown_start - 1:shown_end])
        compact_start, compact_end = _compact_span(shown_start, shown_end, line)
        compact = "".join(lines[compact_start - 1:compact_end])
        age += 1
        records[key] = {
            "path": path, "symbol": symbol, "rank": rank, "origin": origin,
            "age": age, "evidence_line": line, "revision": revision,
            "start_line": start, "end_line": end,
            "unit_sha256": hashlib.sha256(unit.encode()).hexdigest(),
            "shown_start_line": shown_start, "shown_end_line": shown_end,
            "shown_sha256": hashlib.sha256(shown.encode()).hexdigest(),
            "compact_start_line": compact_start, "compact_end_line": compact_end,
            "compact_sha256": hashlib.sha256(compact.encode()).hexdigest(),
            "tier": tier,
            "read_only": (path not in editable or path.startswith("tests/")
                          or root / path in protected_tests or path in protected_tests),
        }

    for path, line, name in locations:
        add(path, line, name, 1, "failed_check")
    for path, line, name in _assertion_targets(root, paths, locations):
        add(path, line, name, 1, "failed_check")
    # Carry only identities already qualified into a prior request. A stale
    # check no longer grants rank one, but the symbol follows shifted lines.
    for old in prior if not receipts else ():
        if not isinstance(old, Mapping):
            continue
        path, symbol = old.get("path"), old.get("symbol")
        if not isinstance(path, str) or not isinstance(symbol, str) or path not in public_paths:
            continue
        current = resolve(root, old)
        if current is None:
            continue
        if current.get("status") == "current":
            add(path, int(current["start_line"]), symbol, 2, "carried")
        elif current.get("status") == "unresolved" and (path, symbol) not in records:
            age += 1
            records[(path, symbol)] = {
                "path": path, "symbol": symbol, "rank": 2, "origin": "carried",
                "age": age, "revision": current["revision"],
                "tier": "outline", "read_only": old.get("read_only") is True,
            }
    next_site = frame.get("next_site")
    if isinstance(next_site, Mapping):
        path, line = next_site.get("path"), next_site.get("line")
        if isinstance(path, str) and path in editable and type(line) is int:
            add(path, line, "", 1, "repair_target")
    roots = [row for row in records.values() if row["rank"] == 1]
    for path, line, name in _direct_neighbors(root, roots):
        add(path, line, name, 2, "direct_neighbor")
    for path, start, end in _read_locations(frame, runtime, editable):
        current = _source(root, path)
        if current is None:
            continue
        line_count = len(current[0].splitlines())
        if start == 1 and end >= line_count:
            continue
        add(path, start, "", 2, "targeted_read")
    edits = frame.get("applied_edits")
    if isinstance(edits, list):
        for edit in edits[-MAX_BOOKMARKS:]:
            if (not isinstance(edit, Mapping) or not isinstance(edit.get("path"), str)
                    or edit["path"] not in editable):
                continue
            lines = edit.get("lines")
            if isinstance(lines, Mapping) and type(lines.get("start")) is int:
                add(str(edit["path"]), lines["start"], "", 1, "applied_edit")
    eligible = []
    for row in records.values():
        current = resolve(root, row)
        if current is None or current.get("status") != "current":
            if decision is not None:
                decision.setdefault("drops", []).append({
                    "path": row["path"], "symbol": row["symbol"],
                    "reason": ("unresolved_symbol" if current is not None
                               else "source_unavailable"),
                })
            continue
        eligible.append(row)
    selected = sorted(eligible, key=lambda row: (
        int(row["rank"]), -int(row["age"]), row["path"], row["symbol"],
    ))[:MAX_BOOKMARKS]
    ordered = sorted(selected, key=lambda row: (
        int(row["rank"]), row["origin"] != "failed_check", row["path"], row["symbol"],
    ))
    # Age decides bounded selection, but a candidate's discovery sequence is
    # not part of its identity and must not change resumed request bytes.
    return finish([{key: value for key, value in row.items() if key != "age"}
                   for row in ordered], "activated" if ordered else "no_symbols")


def resolve(root: Path, bookmark: Mapping[str, object]) -> dict[str, object] | None:
    """Rebind by qualified name on current bytes; never transfer identity by similarity."""
    path, symbol = bookmark.get("path"), bookmark.get("symbol")
    if not isinstance(path, str) or not isinstance(symbol, str):
        return None
    current = _source(root, path)
    if current is None:
        return None
    source, revision = current
    if symbol == "<module>":
        lines = source.splitlines(keepends=True)
        if not lines:
            return None
        end = min(len(lines), MAX_UNIT_LINES)
        unit = "".join(lines[:end])
        return {"path": path, "symbol": symbol, "status": "current",
                "rank": bookmark.get("rank"), "revision": revision,
                "start_line": 1, "end_line": end, "total_lines": len(lines),
                "evidence_line": bookmark.get("evidence_line"),
                "unit_sha256": hashlib.sha256(unit.encode()).hexdigest(),
                "source": source}
    outline = _outline(source)
    rows = [row for row in outline if row["qualified_name"] == symbol]
    if len(rows) != 1:
        return {"path": path, "symbol": symbol, "status": "unresolved",
                "revision": revision, "outline": [row["qualified_name"]
                                                 for row in outline[:24]],
                "source": source}
    row = rows[0]
    start, end = int(row["start_line"]), int(row["end_line"])
    start = _decorated_start(source, start, end)
    lines = source.splitlines(keepends=True)
    unit = "".join(lines[start - 1:end])
    return {"path": path, "symbol": symbol, "status": "current",
            "rank": bookmark.get("rank"), "revision": revision,
            "start_line": start, "end_line": end, "total_lines": len(lines),
            "evidence_line": bookmark.get("evidence_line"),
            "unit_sha256": hashlib.sha256(unit.encode()).hexdigest(),
            "source": source}


def _merge(ranges: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for start, end in sorted(set(ranges)):
        if result and start <= result[-1][1] + 1:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def _context_ranges(source: str, symbol: str, start: int, end: int) -> list[tuple[int, int]]:
    """Bounded static context from the same Python syntax tree as the outline."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError, OverflowError):
        return []
    nodes = [node for node in ast.walk(tree)
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
             and node.end_lineno == end
             and _node_start(node) == start]
    if len(nodes) != 1:
        return []
    node = nodes[0]
    used = {part.id for part in ast.walk(node) if isinstance(part, ast.Name)}
    ranges: list[tuple[int, int]] = []
    # Keep import guards and order by taking their whole containing statement.
    for item in tree.body:
        if isinstance(item, (ast.Import, ast.ImportFrom)):
            bound = {alias.asname or alias.name.split(".")[0] for alias in item.names}
            if bound & used:
                ranges.append((item.lineno, item.end_lineno or item.lineno))
        elif isinstance(item, (ast.Assign, ast.AnnAssign)):
            targets = item.targets if isinstance(item, ast.Assign) else [item.target]
            bound = {part.id for target in targets for part in ast.walk(target)
                     if isinstance(part, ast.Name)}
            if bound & used and item.end_lineno - item.lineno < 12:
                ranges.append((item.lineno, item.end_lineno or item.lineno))
    if "." in symbol:
        class_name = symbol.rsplit(".", 1)[0]
        classes = [row for row in _outline(source)
                   if row["qualified_name"] == class_name and row["kind"] == "class"
                   and int(row["start_line"]) < start <= int(row["end_line"])]
        if len(classes) == 1:
            class_start = int(classes[0]["start_line"])
            ranges.append((class_start, min(start - 1, class_start + 12)))
    return _merge(ranges)[:12]


def _node_start(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> int:
    return min((part.lineno for part in node.decorator_list), default=node.lineno)


def windows(root: Path, bookmarks: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Render in the existing exact-source carrier, without a second panel."""
    from kernel_request_source_window import (
        WINDOW_CONTENT_MARKER, _render_ranges, project_source_window,
    )

    result: list[dict[str, object]] = []
    for bookmark in bookmarks[:MAX_BOOKMARKS]:
        resolved = resolve(root, bookmark)
        if resolved is None:
            continue
        if resolved.get("status") != "current":
            continue
        source = str(resolved["source"])
        lines = source.splitlines(keepends=True)
        start, end = int(resolved["start_line"]), int(resolved["end_line"])
        rank = int(bookmark.get("rank", 2))
        evidence_line = bookmark.get("evidence_line")
        original_start = bookmark.get("start_line")
        if type(evidence_line) is int and type(original_start) is int:
            evidence_line = start + evidence_line - original_start
        shown_start, shown_end, tier = _unit_span(
            start, end, rank, evidence_line)
        unit = (shown_start, shown_end)
        context = [span for span in _context_ranges(source, str(bookmark["symbol"]), start, end)
                   if span[1] - span[0] < 40]

        def render(span: tuple[int, int], label_tier: str, *, whole: bool = False) -> tuple[str, int] | None:
            spans = [(1, len(lines))] if whole else _merge([span, *context])
            displayed = (1, len(lines)) if whole else span
            raw = _render_ranges(lines, [{"start_line": left, "end_line": right}
                                         for left, right in spans],
                                 path=str(bookmark["path"]),
                                 revision=str(resolved["revision"]), total_lines=len(lines))
            projected = project_source_window(raw, owner="recent_editable_read")
            if not projected:
                return None
            slice_label = (f"slice of {end - start + 1} lines, showing "
                           f"{span[1] - span[0] + 1}; "
                           if label_tier == "slice" else "")
            label = (f"[code bookmark] {bookmark['path']}::{bookmark['symbol']} "
                     f"tier={label_tier} lines={displayed[0]}-{displayed[1]} "
                     f"of unit {start}-{end} ({end - start + 1} lines)"
                     + ("; whole file shown" if whole else
                        f"; {slice_label}read the symbol or range for more"))
            projected = projected.replace("\n" + WINDOW_CONTENT_MARKER + "\n",
                                          "\n" + label + "\n" + WINDOW_CONTENT_MARKER + "\n", 1)
            chars = sum(len("".join(lines[left - 1:right])) for left, right in spans)
            return projected, chars

        full = render(unit, tier, whole=unit == (1, len(lines)))
        if full is None:
            continue
        row = {"path": bookmark["path"], "start_line": unit[0],
               "end_line": unit[1], "content": full[0], "source_chars": full[1],
                "bookmark": True, "rank": rank, "tier": tier}
        if (len(lines) <= SMALL_FILE_MAX_LINES
                and len(source.encode("utf-8")) <= SMALL_FILE_MAX_BYTES
                and unit != (1, len(lines))):
            complete = render((1, len(lines)), "whole_file", whole=True)
            if complete is not None:
                row["compact_content"] = row["content"]
                row["compact_source_chars"] = row["source_chars"]
                row["content"], row["source_chars"] = complete
                row["start_line"], row["end_line"] = 1, len(lines)
        if rank == 1 and tier == "slice" and "compact_content" not in row:
            compact = _compact_span(*unit, evidence_line)
            if compact != unit:
                alternative = render(compact, "slice")
                if alternative is not None:
                    row["compact_content"] = alternative[0]
                    row["compact_source_chars"] = alternative[1]
        result.append(row)
    return result
