"""JobQueue completion JobQueue finish package.

This module is deliberately host-owned. It supplies the JobQueue-specific
write validator, obligation view, live-binding witness, and the small public
contract audit used by the shared repair loop. The switch is strict and
defaults to the historical no-op path.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import pathlib
import tempfile
import textwrap
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence
from kernel_test_python import test_python

import kernel_workspace_truth as _workspace_truth


MODE_ENV = "STATEFRAME_JQ_FINISH"
VALID_MODES = {"off", "on"}


def mode() -> str:
    value = os.environ.get(MODE_ENV, "off")
    if value not in VALID_MODES:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def _v21_nonpy_enabled(path: pathlib.Path) -> bool:
    from kernel_tokenlab_nonpy_edits import applies
    return applies(path)


def revision(source: str) -> str:
    return "sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest()


def current_revision(path: pathlib.Path) -> str:
    return revision(path.read_text(encoding="utf-8"))


_CANDIDATES: dict[str, dict[str, object]] = {}


def _candidate_id(source: str, *, path: pathlib.Path, reason: str) -> str:
    payload = f"{path.resolve()}\0{reason}\0{source}".encode("utf-8")
    return "candidate:" + hashlib.sha256(payload).hexdigest()[:24]


def retain_candidate(source: str, *, path: pathlib.Path, reason: str) -> str:
    candidate_id = _candidate_id(source, path=path, reason=reason)
    _CANDIDATES[candidate_id] = {
        "candidate_id": candidate_id,
        "path": str(path.resolve()),
        "source": source,
        "reason": reason,
        "revision": revision(source),
    }
    return candidate_id


def candidate_source(candidate_id: str, *, path: pathlib.Path | None = None) -> str | None:
    key = str(candidate_id)
    if key not in _CANDIDATES and not key.startswith("candidate:"):
        key = "candidate:" + key
    row = _CANDIDATES.get(key)
    if row is None:
        return None
    if path is not None and pathlib.Path(str(row.get("path"))).resolve() != path.resolve():
        return None
    source = row.get("source")
    return source if isinstance(source, str) else None


def retained_candidates() -> dict[str, dict[str, object]]:
    return {key: dict(value) for key, value in _CANDIDATES.items()}


def _relative(path: pathlib.Path, root: pathlib.Path | None) -> str:
    if root is None:
        return path.name
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return "<outside-workspace>"


def protected_path(path: pathlib.Path, *, root: pathlib.Path | None = None,
                   protected_paths: Iterable[object] = ()) -> bool:
    resolved = path.resolve()
    if any(resolved == pathlib.Path(item).resolve() for item in protected_paths):
        return True
    relative = _relative(path, root)
    parts = pathlib.PurePosixPath(relative).parts
    if relative == "<outside-workspace>" or ".git" in parts:
        return True
    if parts and parts[0] in {"results", "results.SNAPSHOT-20260920"}:
        return True
    return bool(parts and parts[0] == "tests") or path.name.startswith("test_") or path.name == "conftest.py"


def validate_source(source: str, *, filename: str = "candidate.py") -> ast.Module:
    if not isinstance(source, str):
        raise TypeError("candidate source must be text")
    try:
        return ast.parse(source, filename=filename)
    except SyntaxError as exc:
        raise ValueError(f"syntax error at line {exc.lineno}: {exc.msg}") from exc


def validate_edit_payload(current: str, target: str, replacement: str) -> str | None:
    """Validate an AST-scoped edit without mutating the file."""
    replacement = textwrap.dedent(replacement)
    try:
        source_tree = ast.parse(current)
        replacement_tree = ast.parse(replacement)
    except SyntaxError as exc:
        return f"edit REJECTED: syntax error at line {exc.lineno}: {exc.msg}"
    top_defs = [node for node in replacement_tree.body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
    boundary_multi = os.environ.get("STATEFRAME_BOUNDARY_CHECKS", "off").lower() == "on" \
        and top_defs and (
            len(top_defs) > 1
            or any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in replacement_tree.body)
            or (isinstance(replacement_tree.body[0], ast.Expr)
                and isinstance(replacement_tree.body[0].value, ast.Constant)
                and isinstance(replacement_tree.body[0].value.value, str))
        )
    if boundary_multi:
        invalid = next((node for node in replacement_tree.body
                        if not isinstance(node, (ast.Import, ast.ImportFrom,
                                                ast.ClassDef, ast.FunctionDef,
                                                ast.AsyncFunctionDef))
                        and not (
                            isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                            and isinstance(node.value.value, str)
                        )
                        and not (
                            isinstance(node, ast.Assign) and len(node.targets) == 1
                            and isinstance(node.targets[0], ast.Name)
                            and isinstance(node.value, (ast.Name, ast.Attribute))
                        )), None)
        if invalid is not None:
            return ("edit REJECTED: unsupported trailing top-level statement "
                    f"`{ast.unparse(invalid)}`; use str_replace instead")
        current_names = {name for name, _node in _qualified_definitions(source_tree)}
        missing = [node.name for node in top_defs if node.name not in current_names]
        if missing:
            return ("edit REJECTED: multi-definition edit cannot apply atomically; "
                    "unapplied definitions " + ", ".join(missing)
                    + "; use str_replace instead")
        return None
    if len(top_defs) != 1:
        names = ", ".join(getattr(node, "name", "<anonymous>") for node in top_defs)
        return (f"edit REJECTED: function_source defines {names or 'no top-level definition'}; "
                f"`edit` replaces only `{target}`. Use str_replace on the whole file for a multi-definition change.")
    first = replacement_tree.body[0] if replacement_tree.body else None
    if isinstance(first, (ast.Expr, ast.Import, ast.ImportFrom)):
        return (f"edit REJECTED: function_source starts with module material; `edit` replaces "
                f"only `{target}`. Use str_replace on the whole file for a multi-definition change.")
    if os.environ.get("STATEFRAME_BOUNDARY_CHECKS", "off").lower() == "on":
        for statement in replacement_tree.body[1:]:
            valid_alias = (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and isinstance(statement.value, (ast.Name, ast.Attribute))
            )
            if not valid_alias:
                try:
                    rendered = ast.unparse(statement)
                except (AttributeError, ValueError):
                    rendered = type(statement).__name__
                return ("edit REJECTED: unsupported trailing top-level statement "
                        f"`{rendered}`; use str_replace instead")
    if target not in {name for name, _node in _qualified_definitions(source_tree)}:
        return None
    candidate = _replace_qualified_source(current, source_tree, target, replacement)
    if candidate is None:
        return f"edit REJECTED: no AST target named `{target}`"
    try:
        ast.parse(candidate)
    except SyntaxError as exc:
        return f"edit REJECTED: resulting file does not parse at line {exc.lineno}: {exc.msg}"
    if os.environ.get("STATEFRAME_FRAME_EDIT_INTEGRITY", "off") == "on":
        from kernel_frame_edit_integrity import duplicate_decorator_rejection
        rejection = duplicate_decorator_rejection(current, candidate)
        if rejection is not None:
            return rejection
    candidate_tree = ast.parse(candidate)
    top_level_names = [node.name for node in candidate_tree.body
                       if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
    duplicates = sorted({name for name in top_level_names if top_level_names.count(name) > 1})
    if duplicates:
        return ("edit REJECTED: the resulting file has duplicate top-level definitions "
                + ", ".join(duplicates) + "; the class/method edit was not published.")
    return None


def _qualified_definitions(tree: ast.Module) -> list[tuple[str, ast.AST]]:
    found: list[tuple[str, ast.AST]] = []

    def visit(body: Sequence[ast.stmt], prefix: tuple[str, ...] = ()) -> None:
        for node in body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name = ".".join((*prefix, node.name))
                found.append((name, node))
                visit(node.body, (*prefix, node.name))

    visit(tree.body)
    return found


def _replace_qualified_source(current: str, tree: ast.Module, target: str,
                              replacement: str) -> str | None:
    node = next((node for name, node in _qualified_definitions(tree) if name == target), None)
    span = None
    if node is not None and os.environ.get("STATEFRAME_FRAME_EDIT_INTEGRITY", "off") == "on":
        from kernel_frame_edit_integrity import definition_span
        span = definition_span(current, node)
        old = current[span[0]:span[1]]
    else:
        old = ast.get_source_segment(current, node) if node is not None else None
    if not old:
        return None
    indent = " " * int(getattr(node, "col_offset", 0))
    lines = textwrap.dedent(replacement).splitlines()
    replacement = lines[0] + ("\n" + "\n".join(indent + line if line.strip() else line
                                                   for line in lines[1:]) if len(lines) > 1 else "")
    if span is not None:
        return current[:span[0]] + replacement + current[span[1]:]
    return current.replace(old, replacement, 1)


def _atomic_write(path: pathlib.Path, source: str, *, errors: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", errors=errors,
                                         dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".candidate", delete=False) as handle:
            temporary = pathlib.Path(handle.name)
            handle.write(source)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _reject(path: pathlib.Path, source: str, reason: str, prefix: str) -> str:
    candidate = retain_candidate(source, path=path, reason=reason)
    return f"{prefix} REJECTED: {reason} (candidate_id={candidate}; resubmittable)"


def validate_and_publish(path: pathlib.Path, source: str, *, root: pathlib.Path | None = None,
                         protected_paths: Iterable[object] = (), prefix: str = "write",
                         require_new: bool = False, public_brief: object = None) -> str:
    """Validate syntax/scope/protection and atomically publish source."""
    warning = None
    nonpy = _v21_nonpy_enabled(path)
    try:
        if protected_path(path, root=root, protected_paths=protected_paths):
            return _reject(path, source, "protected file", prefix)
        if root is not None and not path.resolve().is_relative_to(root.resolve()):
            return _reject(path, source, "path is outside the workspace", prefix)
        if require_new and path.exists():
            return _reject(path, source, "target already exists", prefix)
        if nonpy:
            from kernel_tokenlab_nonpy_edits import validate
            warning = validate(source, path)
        else:
            validate_source(source, filename=str(path))
    except (TypeError, ValueError) as exc:
        if nonpy:
            from kernel_tokenlab_nonpy_edits import emit
            emit(path, operation=prefix, outcome="REJECTED", reason=str(exc))
        return _reject(path, source, str(exc), prefix)
    existing = (path.read_bytes().decode("utf-8", errors="surrogateescape")
                if nonpy else path.read_text(encoding="utf-8")) if path.exists() else None
    if existing == source:
        return f"{prefix} NO-OP: replacement is identical"
    from kernel_destructive_edit_guard import guard_write
    shrink_refusal = guard_write(
        path, source, root=root, public_brief=public_brief,
    )
    if shrink_refusal is not None:
        return shrink_refusal
    from kernel_green_lock import guard_product_mutation, note_applied_product_edit
    refusal = guard_product_mutation(path)
    if refusal is not None:
        return f"{prefix} REJECTED: {refusal}"
    try:
        _atomic_write(path, source, errors="surrogateescape" if nonpy else None)
    except OSError as exc:
        if nonpy:
            from kernel_tokenlab_nonpy_edits import emit
            emit(path, operation=prefix, outcome="REJECTED", reason=f"atomic publish failed: {exc}")
        return _reject(path, source, f"atomic publish failed: {exc}", prefix)
    note_applied_product_edit(path)
    result = f"{prefix} applied."
    if warning:
        result += "\n" + warning
    if nonpy:
        from kernel_tokenlab_nonpy_edits import emit
        emit(path, operation=prefix, outcome="APPLIED", warning=warning)
    return result


def replace_module(path: pathlib.Path, expected_revision: str, source: str | None = None,
                   candidate_id: str | None = None, root: pathlib.Path | None = None,
                   protected_paths: Iterable[object] = (), public_brief: object = None) -> str:
    """Publish a complete module with an optimistic revision check."""
    if source is None and candidate_id is not None:
        source = candidate_source(candidate_id, path=path)
        if source is None:
            return f"replace_module REJECTED: unknown candidate_id={candidate_id}"
    if not isinstance(source, str):
        return "replace_module REJECTED: provide source or candidate_id"
    if path.exists():
        try:
            actual = current_revision(path)
        except (OSError, UnicodeError) as exc:
            return _reject(path, source, f"cannot read current revision: {exc}", "replace_module")
    else:
        actual = "ABSENT"
    expected = str(expected_revision or "")
    expected_values = {expected, expected.removeprefix("sha256:")}
    actual_values = {actual, actual.removeprefix("sha256:")}
    if not expected or not expected_values.intersection(actual_values):
        return _reject(path, source, f"stale revision (expected {expected or 'UNKNOWN'}, current {actual})",
                       "replace_module")
    return validate_and_publish(path, source, root=root, protected_paths=protected_paths,
                                prefix="replace_module", public_brief=public_brief)


def validated_str_replace(path: pathlib.Path, old_str: str, new_str: str, *,
                          root: pathlib.Path | None = None,
                          protected_paths: Iterable[object] = (), public_brief: object = None,
                          journal: object = None) -> str:
    if not path.exists():
        if _workspace_truth.enabled():
            fact = _workspace_truth.exists(root, str(path), journal=journal)
            return "str_replace REJECTED: " + _workspace_truth.render_fact(fact)
        return f"str_replace REJECTED: {path.name} does not exist."
    current = (path.read_bytes().decode("utf-8", errors="surrogateescape")
               if _v21_nonpy_enabled(path)
               else path.read_text(encoding="utf-8"))
    count = current.count(old_str)
    if count != 1:
        reason = ("old_str not found. Copy the exact current text (check whitespace)"
                  if count == 0 else f"old_str matches {count} places")
        return _reject(path, current.replace(old_str, new_str, 1) if count else new_str,
                       reason, "str_replace")
    candidate = current.replace(old_str, new_str, 1)
    return validate_and_publish(path, candidate, root=root, protected_paths=protected_paths,
                                prefix="str_replace", public_brief=public_brief)


@dataclass(frozen=True)
class ObligationSpec:
    key: str
    priority: int
    symbol: str
    marker: tuple[str, ...]


OBLIGATIONS: tuple[ObligationSpec, ...] = (
    ObligationSpec("priority", 1, "JobQueue.pop", ("priority",)),
    ObligationSpec("fifo_ties", 1, "JobQueue.pop", ("FIFO",)),
    ObligationSpec("now_availability", 1, "JobQueue.pop", ("now", "eligible")),
    ObligationSpec("backoff", 1, "JobQueue.fail", ("backoff", "eligible_at")),
    ObligationSpec("max_attempts", 1, "Job", ("max_attempts",)),
    ObligationSpec("public_dead_letter", 1, "JobQueue.dead_letter", ("dead_letter",)),
    ObligationSpec("lifecycle", 1, "JobQueue", ("push", "complete", "fail")),
    ObligationSpec("snapshot", 1, "QueueStats.snapshot", ("QueueStats", "snapshot")),
)


def is_jobqueue_brief(brief: str | None, path: str = "") -> bool:
    text = str(brief or "").lower()
    return ("dead_letter" in text and "queue" in text and
            ("queuestats" in text or "oldest_waiting" in text or "snapshot" in text)
            and (not path or "jobq" in path.lower() or "queue" in path.lower()))


def _nodes(source: str) -> tuple[dict[str, ast.AST], set[str], set[str]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}, set(), set()
    found: dict[str, list[ast.AST]] = {}

    def visit(body: Sequence[ast.stmt], prefix: str = "") -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                name = prefix + node.name
                found.setdefault(name, []).append(node)
                # ``dead_letter`` is a public instance attribute, so its
                # binding is an assignment rather than a class-level name.
                # Keep that assignment visible to the ledger without
                # treating every prose mention as evidence.
                for child in ast.walk(node):
                    if isinstance(child, (ast.Assign, ast.AnnAssign)):
                        targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                        if any(isinstance(target, ast.Attribute) and
                               isinstance(target.value, ast.Name) and
                               target.value.id == "self" and target.attr == "dead_letter"
                               for target in targets):
                            found.setdefault(name + ".dead_letter", []).append(child)
                visit(node.body, name + ".")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + node.name
                found.setdefault(name, []).append(node)
                visit(node.body, name + ".")

    visit(tree.body)
    duplicates = {name for name, values in found.items() if len(values) > 1}
    stubs = set()
    for name, values in found.items():
        node = values[-1]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
                isinstance(item, ast.Raise) and isinstance(item.exc, (ast.Name, ast.Call))
                and (item.exc.id if isinstance(item.exc, ast.Name) else
                     getattr(item.exc.func, "id", None)) == "NotImplementedError"
                for item in ast.walk(node)):
            stubs.add(name)
    return {name: values[-1] for name, values in found.items()}, duplicates, stubs


def _brief_span(brief: str, spec: ObligationSpec) -> dict[str, object] | None:
    lower = brief.lower()
    needles = [spec.symbol.lower().split(".")[0], *[marker.lower() for marker in spec.marker]]
    positions = [lower.find(needle) for needle in needles if lower.find(needle) >= 0]
    if not positions:
        return None
    start = min(positions)
    end = min(len(brief), start + 240)
    return {"start": start, "end": end, "text": brief[start:end]}


def _shape(spec: ObligationSpec, bindings: Mapping[str, ast.AST], duplicates: set[str],
           stubs: set[str]) -> str:
    if spec.symbol == "JobQueue.dead_letter":
        present = spec.symbol in bindings
        symbol = "JobQueue"
    else:
        symbol = spec.symbol
        present = symbol in bindings
    if not present:
        return "MISSING"
    if symbol in duplicates or symbol in stubs or any(name.startswith(symbol + ".") for name in stubs):
        return "LIVE_STUB"
    return "PRESENT"


def obligation_ledger(*, brief: str, source: str, path: str = "candidate.py",
                      receipts: Sequence[object] = ()) -> dict[str, object]:
    """Return the public-clause ledger; only supplied host receipts set evidence."""
    bindings, duplicates, stubs = _nodes(source)
    rev = revision(source)
    rows: list[dict[str, object]] = []
    for spec in OBLIGATIONS:
        span = _brief_span(brief, spec)
        if span is None:
            continue
        evidence = "UNEXERCISED"
        refs: list[str] = []
        for receipt in receipts:
            if not isinstance(receipt, Mapping):
                continue
            target = str(receipt.get("target_path", receipt.get("path", path))).lstrip("./")
            if target != path.lstrip("./"):
                continue
            refs.append(str(receipt.get("receipt_id", receipt.get("event_id", "receipt"))))
            checked = {str(receipt.get(key)) for key in
                       ("revision", "current_revision", "after_revision", "product_revision")
                       if receipt.get(key) is not None}
            stale = checked and rev not in checked and rev.removeprefix("sha256:") not in checked
            result = str(receipt.get("result_class", receipt.get("result", receipt.get("status", "")))).upper()
            detail = str(receipt.get("detail", receipt.get("observation", ""))).upper()
            if stale:
                evidence = "STALE"
            elif result in {"FAIL", "FAILED", "RED", "FAIL_WITH_WITNESS"} or "FAIL_WITH_WITNESS" in detail:
                evidence = "FAIL_WITH_WITNESS"
            elif result in {"GREEN", "PASS", "PASSED", "OK", "PASS_CASES"}:
                evidence = "PASS_CASES"
            elif result in {"OBSERVED", "APPLIED", "HOST_OBSERVED"}:
                evidence = "OBSERVED"
        rows.append({"id": spec.key, "priority": spec.priority, "authority": "EXPLICIT",
                     "brief_span": span, "symbol": spec.symbol,
                     "shape": _shape(spec, bindings, duplicates, stubs),
                     "evidence": evidence, "evidence_refs": list(dict.fromkeys(refs))[:4]})
    return {"schema": "stateframe-jobqueue-ledger/1", "authority": "HOST_CAPTURED",
            "path": path, "revision": rev, "rows": rows}


def render_ledger(ledger: Mapping[str, object]) -> str:
    lines = ["[OBLIGATION LEDGER | HOST_CAPTURED]"]
    for row in ledger.get("rows", ()) if isinstance(ledger, Mapping) else ():
        lines.append(f"{row.get('id')} p={row.get('priority')} {row.get('symbol')} "
                     f"shape={row.get('shape')} evidence={row.get('evidence')}")
    return "\n".join(lines)


def live_binding_witness(source: str, path: str = "candidate.py") -> list[str]:
    """Render short live-binding witnesses for duplicate/stub QueueStats shapes."""
    bindings, duplicates, stubs = _nodes(source)
    interesting = [name for name in sorted(stubs | duplicates)
                   if name == "QueueStats" or name.startswith("QueueStats.")]
    if not interesting:
        return []
    try:
        from kernel_stateframe_thinking import fresh_isolated_import
        live = fresh_isolated_import(source, path, simulate_missing_firstlineno=True)
    except Exception as exc:  # pragma: no cover - defensive host degradation
        return [f"LIVE-BINDING WITNESS unavailable: {type(exc).__name__}: {exc}"]
    locations = {str(item.get("name")): str(item.get("location"))
                 for item in live.get("bindings", ()) if isinstance(item, Mapping)}
    class_location = locations.get("QueueStats", "UNKNOWN")
    class_line = class_location.rsplit(":", 1)[-1] if ":" in class_location else "UNKNOWN"
    rows = []
    for name in interesting:
        if name == "QueueStats":
            rows.append(f"QueueStats@{class_line} imported")
            continue
        node = bindings.get(name)
        line = getattr(node, "lineno", "UNKNOWN")
        if name in stubs:
            rows.append(f"QueueStats@{class_line} imported; snapshot@{line} raises NotImplementedError")
        else:
            rows.append(f"QueueStats@{class_line} imported; snapshot@{line} has duplicate live binding")
    return rows


_AUDIT_WORKER = r'''
import importlib.util, json, pathlib, traceback
p = json.load(__import__('sys').stdin)
path = pathlib.Path(p['path'])
result = []
spec = importlib.util.spec_from_file_location('_astra22_candidate', path)
module = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(module)
    def check(name, expectation, fn):
        try:
            fn()
            result.append({'id': name, 'expectation': expectation, 'status': 'PASS_CASES'})
        except BaseException as exc:
            result.append({'id': name, 'expectation': expectation,
                           'status': 'FAIL_WITH_WITNESS',
                           'witness': f'{type(exc).__name__}: {exc}',
                           'traceback': traceback.format_exc(limit=3)})
    def interface():
        # Every assert carries the OBSERVED value: a bare assert delivered the useless
        # witness "AssertionError: " to the model (autopsy jq110), which then never repaired.
        def ident(job):
            return getattr(job, 'id', job)
        assert callable(getattr(module.QueueStats, 'snapshot', None)), 'QueueStats.snapshot is not callable'
        q = module.JobQueue()
        snapshot = module.QueueStats.snapshot(q, 0.0)
        assert isinstance(snapshot, dict), f'snapshot returned {type(snapshot).__name__}, expected dict'
        missing = {'queued','running','done','dead','oldest_waiting_s'} - set(snapshot)
        assert not missing, f'snapshot is missing keys {sorted(missing)}'
        first = module.Job('queued', 1); second = module.Job('running', 1)
        second.state = 'running'
        q.push(first); q.push(second); got = q.pop(0.0)
        assert ident(got) == 'queued', f'pop(0.0) returned {ident(got)!r}, expected the queued job'
        counted = module.QueueStats.snapshot(q, 0.0)
        assert counted['queued'] == 1 and counted['running'] == 1, (
            f"snapshot counted queued={counted['queued']!r} running={counted['running']!r}, expected 1 and 1")
        assert isinstance(counted['oldest_waiting_s'], float), (
            f"oldest_waiting_s is {type(counted['oldest_waiting_s']).__name__}, expected float")
    def ordering():
        def ident(job):
            return getattr(job, 'id', job)
        q = module.JobQueue()
        for job in (module.Job('a', 1), module.Job('b', 5), module.Job('c', 5)):
            q.push(job)
        order = [ident(q.pop(0.0)) for _ in range(3)]
        assert order == ['b','c','a'], f'pop order {order}, expected [b, c, a] (priority desc, FIFO ties)'
        delayed = module.Job('delay', 1, max_attempts=2)
        q.push(delayed); got = q.pop(0.0)
        assert ident(got) == 'delay', f'pop(0.0) returned {ident(got)!r}, expected delay'
        q.fail('delay', now=10.0)
        early = q.pop(11.0)
        assert early is None, f'pop(11.0) returned {ident(early)!r} during backoff, expected None'
        late = q.pop(12.0)
        assert ident(late) == 'delay', f'pop(12.0) returned {ident(late)!r} after backoff, expected delay'
    def lifecycle():
        q = module.JobQueue(); job = module.Job('dead', 1, max_attempts=1)
        q.push(job); got = getattr(q.pop(0.0), 'id', None)
        assert got == 'dead', f'pop(0.0) returned {got!r}, expected dead'
        q.fail('dead', now=10.0)
        assert job.state == 'dead', f'job.state is {job.state!r} after exceeding max_attempts, expected dead'
        assert isinstance(getattr(q, 'dead_letter', None), list), 'JobQueue has no public list attribute dead_letter'
        assert job in q.dead_letter, 'the dead job was not appended to q.dead_letter'
    def snapshot_all_states():
        # From the PUBLIC brief only: snapshot counts queued/running/done/dead "computed against
        # the queue's jobs", and dead jobs are moved to .dead_letter (they must still count as dead).
        # oldest_waiting_s semantics are not specified by the brief beyond "float", so not checked here.
        def ident(job):
            return getattr(job, 'id', job)
        q = module.JobQueue()
        done_job = module.Job('d', 5); dead_job = module.Job('x', 3, max_attempts=1)
        waiting = module.Job('w', 1)
        q.push(done_job); q.push(dead_job); q.push(waiting)
        got = q.pop(0.0)
        assert ident(got) == 'd', f'pop(0.0) returned {ident(got)!r}, expected d (highest priority)'
        q.complete('d')
        got = q.pop(0.0)
        assert ident(got) == 'x', f'pop(0.0) returned {ident(got)!r}, expected x'
        q.fail('x', now=1.0)
        counted = module.QueueStats.snapshot(q, 2.0)
        expected = {'queued': 1, 'running': 0, 'done': 1, 'dead': 1}
        observed = {key: counted.get(key) for key in expected}
        assert observed == expected, (f'snapshot counts {observed}, expected {expected} '
                                      '(done jobs and dead jobs in .dead_letter must be counted)')
        assert isinstance(counted.get('oldest_waiting_s'), float), 'oldest_waiting_s must be a float'
    check('interfaces_snapshot', 'interfaces+snapshot callable and returns required public keys', interface)
    check('snapshot_all_states', 'snapshot counts every state incl. done and dead (dead_letter) jobs', snapshot_all_states)
    check('ordering_fifo_now', 'priority desc, FIFO ties, explicit now availability and 2**attempts backoff', ordering)
    check('lifecycle_dead_letter', 'past max_attempts becomes dead and is in public .dead_letter', lifecycle)
except BaseException as exc:
    result.append({'id': 'module_import', 'expectation': 'candidate imports',
                   'status': 'FAIL_WITH_WITNESS', 'witness': f'{type(exc).__name__}: {exc}',
                   'traceback': traceback.format_exc(limit=3)})
print(json.dumps({'checks': result}, sort_keys=True, default=str))
'''


def bounded_contract_audit(source: str, *, path: str = "candidate.py",
                           brief: str = "") -> dict[str, object]:
    """Run the frozen public-brief checks in a disposable subprocess."""
    with tempfile.TemporaryDirectory(prefix="astra22-audit-") as root:
        target = pathlib.Path(root) / pathlib.Path(path).name
        target.write_text(source, encoding="utf-8")
        try:
            from kernel_execution_policy import ExecutionPolicyError, run_sandboxed
            execution = run_sandboxed(
                [test_python(), "-I", "-c", _AUDIT_WORKER],
                root,
                input_data=json.dumps({"path": str(target)}),
                timeout=3.0,
                output_limit=256 * 1024,
                readonly_workspace=True,
                cpu_seconds=3.0,
            )
        except ExecutionPolicyError as error:
            return {"schema": "stateframe-contract-audit/1", "status": "FAIL_WITH_WITNESS",
                    "checks": [{"id": "audit", "status": "FAIL_WITH_WITNESS",
                                "witness": f"sandbox unavailable: {error}"}],
                    "revision": revision(source)}
        if execution.timed_out:
            return {"schema": "stateframe-contract-audit/1", "status": "FAIL_WITH_WITNESS",
                    "checks": [{"id": "audit", "status": "FAIL_WITH_WITNESS",
                                "witness": "audit timeout"}], "revision": revision(source)}
        completed = execution.completed
        lines = completed.stdout.strip().splitlines()
        try:
            payload = json.loads(lines[-1])
        except (IndexError, json.JSONDecodeError):
            payload = {"checks": [{"id": "audit", "status": "FAIL_WITH_WITNESS",
                                    "witness": (completed.stderr or "audit produced no receipt")[-600:]}]}
    checks = payload.get("checks", []) if isinstance(payload, Mapping) else []
    failed = [row for row in checks if isinstance(row, Mapping) and row.get("status") == "FAIL_WITH_WITNESS"]
    return {"schema": "stateframe-contract-audit/1", "authority": "HOST_OBSERVED",
            "revision": revision(source), "brief_authority": "PUBLIC_BRIEF",
            "status": "FAIL_WITH_WITNESS" if failed else "PASS_CASES",
            "checks": checks}


def render_audit(audit: Mapping[str, object]) -> str:
    lines = [f"[CONTRACT AUDIT | {audit.get('authority', 'HOST_OBSERVED')}]"]
    for check in audit.get("checks", ()) if isinstance(audit, Mapping) else ():
        if not isinstance(check, Mapping):
            continue
        line = f"{check.get('id')}: {check.get('status')} — {check.get('expectation', '')}"
        if check.get("status") == "FAIL_WITH_WITNESS":
            line += f" witness={check.get('witness', '')}"
        lines.append(line)
    return "\n".join(lines)


__all__ = [
    "MODE_ENV", "OBLIGATIONS", "bounded_contract_audit", "candidate_source",
    "current_revision", "enabled", "is_jobqueue_brief", "live_binding_witness",
    "mode", "obligation_ledger", "protected_path", "render_audit", "render_ledger",
    "replace_module", "retain_candidate", "retained_candidates", "revision",
    "validate_and_publish", "validated_str_replace",
]
