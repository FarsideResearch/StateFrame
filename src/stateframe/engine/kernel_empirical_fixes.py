"""Small, opt-in repairs for empirically observed StateFrame failures.

The default path in this module is deliberately inert.  The live loop imports
these helpers only at the point where ``STATEFRAME_EMPIRICAL_FIXES=on`` has
already been selected, so the historical prompt and receipt bytes remain
unchanged when the switch is off.
"""
from __future__ import annotations

import ast
import builtins
import json
import os
import re
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path


MODE_ENV = "STATEFRAME_EMPIRICAL_FIXES"
STRUCTURE_FACTS_MODE_ENV = "STATEFRAME_STRUCTURE_FACTS"
HARNESS_PHRASE_RE = re.compile(
    r"\s*(?:your patch is wrong or incomplete|the patch is wrong or incomplete)\.?\s*",
    re.IGNORECASE,
)
TRACEBACK_RE = re.compile(
    r"(?P<path>[A-Za-z0-9_./\\-]+\.py)(?::|\s+line\s+)(?P<line>\d+)",
    re.IGNORECASE,
)
TRACEBACK_FUNCTION_RE = re.compile(
    r"(?P<path>[A-Za-z0-9_./\\-]+\.py)(?::|\s+line\s+)(?P<line>\d+)\s+"
    r"(?:in|inside)\s+(?P<symbol>[A-Za-z_]\w*)",
    re.IGNORECASE,
)
KEY_ERROR_RE = re.compile(r"KeyError\s*:\s*['\"](?P<key>[^'\"]+)['\"]")
_WITNESS_RE = re.compile(
    r"(?:AssertionError|AttributeError|KeyError|IndexError|TypeError|ValueError|"
    r"NameError|ImportError|ModuleNotFoundError|SyntaxError|StopIteration)"
    r"(?:\s*:\s*|\s+).{0,520}",
)


def empirical_mode() -> str:
    value = os.environ.get(MODE_ENV, "off")
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return empirical_mode() == "on"


def structure_facts_mode() -> str:
    """Return the opt-in mode for post-edit AST structure facts."""
    value = os.environ.get(STRUCTURE_FACTS_MODE_ENV, "off")
    if value not in {"off", "on"}:
        raise ValueError(f"{STRUCTURE_FACTS_MODE_ENV} must be off or on")
    return value


def structure_facts_enabled() -> bool:
    return structure_facts_mode() == "on"


def _clean_text(value: object, limit: int = 900) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]


def strip_harness_phrase(text: object) -> str:
    """Remove the known harness judgment from a stale result only."""
    value = str(text or "")
    value = HARNESS_PHRASE_RE.sub(" ", value)
    return re.sub(r"[ \t]{2,}", " ", value).strip()


def _row_revision(row: Mapping[str, object]) -> object:
    for key in ("revision", "current_revision", "product_revision", "artifact_revision"):
        if row.get(key) is not None:
            return row[key]
    for key in ("binding", "after_binding", "current_binding"):
        value = row.get(key)
        if isinstance(value, Mapping):
            for nested in ("product_revision", "artifact", "revision"):
                if value.get(nested) is not None:
                    return value[nested]
    return None


def _frame_revision(frame: Mapping[str, object]) -> object:
    for key in ("current_revision", "revision", "product_revision"):
        if frame.get(key) is not None:
            return frame[key]
    world = frame.get("world")
    if isinstance(world, Mapping):
        binding_state = world.get("revision_binding_state")
        if isinstance(binding_state, Mapping):
            product = binding_state.get("product")
            if isinstance(product, Mapping) and product.get("value") is not None:
                return product["value"]
    revisions = frame.get("artifact_revisions")
    if isinstance(revisions, Mapping) and len(revisions) == 1:
        return next(iter(revisions.values()))
    contract = frame.get("completion_contract")
    if isinstance(contract, Mapping):
        current = contract.get("current_public_checks")
        if isinstance(current, Mapping):
            for key in ("revision", "current_revision", "product_revision"):
                if current.get(key) is not None:
                    return current[key]
    verification = frame.get("verification")
    if isinstance(verification, Mapping):
        for key in ("revision", "current_revision", "product_revision"):
            if verification.get(key) is not None:
                return verification[key]
    return None


def _row_step(row: Mapping[str, object]) -> int | None:
    for key in ("step", "turn", "test_turn"):
        value = row.get(key)
        if type(value) is int:
            return value
    return None


def _receipt_kind(row: Mapping[str, object]) -> str:
    return str(row.get("kind", row.get("tool", ""))).lower()


def _receipt_result(row: Mapping[str, object]) -> str:
    value = row.get("result_class", row.get("result", row.get("status")))
    if value is None and row.get("exit_code") is not None:
        return "GREEN" if row.get("exit_code") == 0 else "RED"
    return str(value if value is not None else "UNKNOWN").upper()


def _selector(row: Mapping[str, object]) -> str:
    for key in ("selector", "target", "suite"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    parameters = row.get("parameters")
    if isinstance(parameters, Mapping):
        for key in ("target", "selector", "suite"):
            value = parameters.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    selectors = row.get("selectors")
    if isinstance(selectors, Sequence) and not isinstance(selectors, (str, bytes)):
        return ", ".join(str(item) for item in selectors if str(item).strip())
    return "UNKNOWN_SELECTOR"


_STRUCTURE_DEFINITION_TYPES = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
_STRUCTURE_DECORATORS = frozenset({"classmethod", "staticmethod", "property"})


def _structure_tree(source: str) -> ast.Module | None:
    try:
        return ast.parse(source)
    except (SyntaxError, TypeError, ValueError):
        return None


def _structure_decorator_name(value: ast.AST) -> str | None:
    if isinstance(value, ast.Call):
        value = value.func
    if isinstance(value, ast.Name):
        return value.id if value.id in _STRUCTURE_DECORATORS else None
    if isinstance(value, ast.Attribute):
        return value.attr if value.attr in _STRUCTURE_DECORATORS else None
    return None


def _structure_definitions(tree: ast.Module | None) -> dict[str, dict[str, object]]:
    """Index definitions while retaining their lexical container.

    This is deliberately narrower than a symbol resolver.  The facts are about
    source shape immediately before and after one edit, so a qualified AST path
    and the direct owning class are the only attribution needed.
    """
    if tree is None:
        return {}
    result: dict[str, dict[str, object]] = {}

    def visit(body: Sequence[ast.stmt], prefix: tuple[str, ...] = (),
              container: str = "module", class_path: tuple[str, ...] = ()) -> None:
        for node in body:
            if not isinstance(node, _STRUCTURE_DEFINITION_TYPES):
                continue
            qualified = ".".join((*prefix, node.name))
            result[qualified] = {
                "node": node,
                "name": node.name,
                "qualified_name": qualified,
                "container": container,
                "class_path": class_path,
                "parent_class": (class_path[-1]
                                  if container == "class" and class_path else None),
            }
            child_container = "class" if isinstance(node, ast.ClassDef) else "function"
            child_class_path = (*class_path, node.name) if isinstance(node, ast.ClassDef) else class_path
            visit(node.body, (*prefix, node.name), child_container, child_class_path)

    visit(tree.body)
    return result


def _structure_line(record: Mapping[str, object]) -> int:
    node = record.get("node")
    return int(getattr(node, "lineno", 1) or 1)


def _structure_fact(kind: str, *, path: str, line: int | None = None,
                    **fields: object) -> dict[str, object]:
    fact: dict[str, object] = {"kind": kind}
    if path:
        fact["path"] = path
    if line is not None:
        fact["line"] = int(line)
    fact.update(fields)
    return fact


def _structure_parameter_text(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Render only a function's parameters; return annotations/body are excluded."""
    args = node.args

    def expression(value: ast.AST) -> str:
        try:
            return ast.unparse(value)
        except (AttributeError, TypeError):
            return ast.dump(value, annotate_fields=True, include_attributes=False)

    positional = [*args.posonlyargs, *args.args]
    defaults_start = len(positional) - len(args.defaults)
    pieces: list[str] = []
    for index, argument in enumerate(positional):
        item = expression(argument)
        if index >= defaults_start:
            item += " = " + expression(args.defaults[index - defaults_start])
        pieces.append(item)
        if args.posonlyargs and index == len(args.posonlyargs) - 1:
            pieces.append("/")
    if args.vararg is not None:
        pieces.append("*" + expression(args.vararg))
    elif args.kwonlyargs:
        pieces.append("*")
    for argument, default in zip(args.kwonlyargs, args.kw_defaults):
        item = expression(argument)
        if default is not None:
            item += " = " + expression(default)
        pieces.append(item)
    if args.kwarg is not None:
        pieces.append("**" + expression(args.kwarg))
    return "(" + ", ".join(pieces) + ")"


def _structure_syntax_fact(source: str, *, path: str) -> dict[str, object] | None:
    try:
        ast.parse(source)
    except SyntaxError as error:
        return _structure_fact(
            "SYNTAX_ERROR", path=path, line=int(error.lineno or 1),
            msg=str(error.msg or error),
        )
    except (TypeError, ValueError) as error:
        return _structure_fact("SYNTAX_ERROR", path=path, line=1, msg=str(error))
    return None


def structure_facts_for_edit(before: str | None, after: str, *, path: str = "") -> list[dict[str, object]]:
    """Diff one edited Python file and return typed, host-observed structure facts.

    A syntax error in the post-edit bytes is reported on its own.  In that case
    no AST claim is made from the incomplete tree.  All other facts are derived
    solely from the before/after ASTs and are therefore deterministic and
    advisory.
    """
    if path and not path.replace("\\", "/").lower().endswith(".py"):
        return []
    before_source = before if isinstance(before, str) else ""
    after_source = after if isinstance(after, str) else str(after)
    syntax = _structure_syntax_fact(after_source, path=path)
    if syntax is not None:
        return [syntax]
    before_tree = _structure_tree(before_source)
    after_tree = _structure_tree(after_source)
    if after_tree is None:
        return []
    before_defs = _structure_definitions(before_tree)
    after_defs = _structure_definitions(after_tree)
    facts: list[dict[str, object]] = []

    # A method that left its class is the high-signal ldev_12 failure.  Require
    # the old qualified path to be absent so a legitimate helper with the same
    # leaf name does not create a false move fact.
    for qualified in sorted(before_defs):
        old = before_defs[qualified]
        old_class = old.get("parent_class")
        if not isinstance(old_class, str) or qualified in after_defs:
            continue
        name = str(old.get("name", ""))
        candidates = [row for row in after_defs.values()
                      if row.get("name") == name
                      and row.get("container") in {"module", "class"}]
        if len(candidates) != 1:
            continue
        current = candidates[0]
        current_class = current.get("parent_class")
        if current.get("container") == "module":
            now_at = "module"
        elif isinstance(current_class, str) and current_class != old_class:
            now_at = "other_class"
        else:
            continue
        fields: dict[str, object] = {
            "name": name,
            "class": old_class,
            "now_at": now_at,
            "qualified_name": qualified,
        }
        if now_at == "other_class":
            fields["now_class"] = current_class
        facts.append(_structure_fact(
            "DEFINITION_LEFT_CLASS", path=path, line=_structure_line(current), **fields))

    # Decorator placement and multiplicity are facts about the post-edit tree.
    for qualified in sorted(after_defs):
        record = after_defs[qualified]
        node = record.get("node")
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        names = [name for name in (_structure_decorator_name(item)
                                   for item in node.decorator_list) if name is not None]
        counts: dict[str, int] = {}
        for name in names:
            counts[name] = counts.get(name, 0) + 1
        if record.get("container") == "module":
            for decorator in sorted(counts):
                facts.append(_structure_fact(
                    "STRAY_DECORATOR", path=path, line=int(node.lineno),
                    name=node.name, qualified_name=qualified, decorator=decorator,
                    at="module-level function"))
        for decorator, count in sorted(counts.items()):
            if count > 1:
                facts.append(_structure_fact(
                    "DUPLICATE_DECORATOR", path=path, line=int(node.lineno),
                    name=node.name, qualified_name=qualified,
                    decorator=decorator, count=count))

    for qualified in sorted(set(before_defs) - set(after_defs)):
        facts.append(_structure_fact(
            "DEFINITION_REMOVED", path=path, line=_structure_line(before_defs[qualified]),
            qualified_name=qualified))
    for qualified in sorted(set(after_defs) - set(before_defs)):
        facts.append(_structure_fact(
            "DEFINITION_ADDED", path=path, line=_structure_line(after_defs[qualified]),
            qualified_name=qualified))

    for qualified in sorted(set(before_defs) & set(after_defs)):
        before_node = before_defs[qualified].get("node")
        after_node = after_defs[qualified].get("node")
        if not isinstance(before_node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                or not isinstance(after_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        before_params = _structure_parameter_text(before_node)
        after_params = _structure_parameter_text(after_node)
        if before_params != after_params:
            facts.append(_structure_fact(
                "SIGNATURE_CHANGED", path=path, line=int(after_node.lineno),
                qualified_name=qualified, before=before_params, after=after_params,
                params_before=before_params, params_after=after_params))
    return facts


def structure_facts_observation(facts: object) -> str:
    """Serialize facts for the history arm's post-edit tool result."""
    rows = [dict(row) for row in facts
            if isinstance(row, Mapping) and str(row.get("kind", ""))]
    if not rows:
        return ""
    return json.dumps({"schema": "stateframe-structure-facts/1",
                       "kind": "structure_facts", "facts": rows},
                      sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _attribute_error_parts(observation: object) -> tuple[str | None, str | None]:
    text = str(observation or "")
    patterns = (
        re.compile(r"(?:type object|module)\s+['\"](?P<object>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)['\"]\s+has no attribute\s+['\"](?P<name>[A-Za-z_]\w*)['\"]", re.I),
        re.compile(r"['\"](?P<object>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)['\"]\s+object\s+has no attribute\s+['\"](?P<name>[A-Za-z_]\w*)['\"]", re.I),
    )
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group("object"), match.group("name")
    generic = re.search(r"has no attribute\s+['\"]?(?P<name>[A-Za-z_]\w*)['\"]?", text, re.I)
    return (None, generic.group("name")) if generic else (None, None)


def _structure_candidate_paths(observation: object, checkout: Path,
                               preferred_paths: Sequence[str] = ()) -> list[Path]:
    root = checkout.resolve()
    raw_paths: list[str] = []
    for value in preferred_paths:
        if str(value).strip():
            raw_paths.append(str(value))
    for match in re.finditer(r"(?P<path>[A-Za-z0-9_./\\-]+\.py)(?::\d+|\s+line\s+\d+)",
                             str(observation or ""), re.I):
        raw_paths.append(match.group("path"))
    result: list[Path] = []
    for raw in raw_paths:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            candidate = candidate.resolve()
            candidate.relative_to(root)
        except (OSError, ValueError):
            continue
        if candidate.is_file() and candidate.suffix == ".py" and candidate not in result:
            result.append(candidate)
    if result:
        return result
    for candidate in sorted(root.rglob("*.py")):
        if any(part in {".git", "__pycache__", ".pytest_cache", ".ruff_cache"}
               for part in candidate.relative_to(root).parts):
            continue
        result.append(candidate)
    return result


def _structure_locations_in_source(source: str, object_name: str | None,
                                   attribute: str) -> list[dict[str, object]]:
    tree = _structure_tree(source)
    if tree is None:
        return []
    definitions = _structure_definitions(tree)
    rows: list[dict[str, object]] = []
    object_leaf = object_name.rsplit(".", 1)[-1] if object_name else None
    candidates = [record for record in definitions.values()
                  if record.get("name") == attribute]
    if object_leaf:
        class_matches = [record for record in candidates
                         if record.get("parent_class") == object_leaf]
        if class_matches:
            candidates = class_matches
        else:
            module_matches = [record for record in candidates
                              if record.get("container") == "module"]
            candidates = module_matches
    for record in sorted(candidates, key=lambda item: str(item.get("qualified_name", ""))):
        rows.append({
            "qualified_name": str(record.get("qualified_name", attribute)),
            "name": attribute,
            "class": record.get("parent_class"),
            "line": _structure_line(record),
        })
    # Class attributes can be data definitions rather than functions.
    if not rows:
        for qualified, record in definitions.items():
            node = record.get("node")
            if not isinstance(node, ast.ClassDef):
                continue
            if object_leaf and node.name != object_leaf:
                continue
            for child in node.body:
                targets: list[ast.AST] = []
                if isinstance(child, ast.Assign):
                    targets = list(child.targets)
                elif isinstance(child, ast.AnnAssign):
                    targets = [child.target]
                elif isinstance(child, ast.AugAssign):
                    targets = [child.target]
                if any(isinstance(target, ast.Name) and target.id == attribute for target in targets):
                    rows.append({"qualified_name": f"{qualified}.{attribute}", "name": attribute,
                                 "class": node.name, "line": int(child.lineno)})
    return rows


def location_facts_from_test_result(observation: object, checkout: Path,
                                    *, preferred_paths: Sequence[str] = ()) -> list[dict[str, object]]:
    """Locate an AttributeError attribute in current source, when it is defined."""
    object_name, attribute = _attribute_error_parts(observation)
    if not attribute:
        return []
    facts: list[dict[str, object]] = []
    root = checkout.resolve()
    for candidate in _structure_candidate_paths(observation, root, preferred_paths):
        try:
            source = candidate.read_text(encoding="utf-8")
            relative = candidate.relative_to(root).as_posix()
        except (OSError, UnicodeError, ValueError):
            continue
        for row in _structure_locations_in_source(source, object_name, attribute):
            line = int(row["line"])
            qualified_name = str(row["qualified_name"])
            location = f"{relative}:{line}"
            facts.append({
                "kind": "LOCATION",
                "path": relative,
                "line": line,
                "name": attribute,
                "qualified_name": qualified_name,
                "qualified_path": location,
                "location": location,
                "class": row.get("class"),
            })
    return facts


def _receipt_timeline(receipts: object) -> list[Mapping[str, object]]:
    if isinstance(receipts, Mapping):
        receipts = list(receipts.values())
    if not isinstance(receipts, Sequence) or isinstance(receipts, (str, bytes)):
        return []
    return [row for row in receipts if isinstance(row, Mapping)]


def change_after_green_from_receipts(receipts: object) -> str:
    """Return the empirical three-valued fact from the append-only receipt order."""
    rows = _receipt_timeline(receipts)
    green_positions = [index for index, row in enumerate(rows)
        if _receipt_kind(row) in {"test", "run_tests", "pytest", "verification",
                                  "test_subprocess_completed"}
                       and _receipt_result(row) in {
                           "GREEN", "PASS", "PASSED", "SUCCESS", "CERTIFIED",
                           "REUSED_VERIFICATION",
                       }]
    if not green_positions:
        return "no_green_yet"
    last_green = green_positions[-1]
    changed_after = any(
        index > last_green
        and _receipt_kind(row) in {"edit", "str_replace", "file_create", "replace_module", "write"}
        and _receipt_result(row) == "APPLIED"
        for index, row in enumerate(rows)
    )
    return "yes" if changed_after else "no"


def change_after_green_from_frame(frame: Mapping[str, object]) -> str:
    """Use the receipt timeline when present, with a bounded frame fallback."""
    for key in ("empirical_receipts", "receipts"):
        if key in frame:
            return change_after_green_from_receipts(frame[key])
    rows: list[dict] = []
    changes = frame.get("recent_changes")
    if isinstance(changes, Sequence) and not isinstance(changes, (str, bytes)):
        rows.extend({"kind": "edit", "result_class": "APPLIED", **dict(row)}
                    for row in changes if isinstance(row, Mapping))
    feedback = frame.get("recent_test_feedback")
    if isinstance(feedback, Sequence) and not isinstance(feedback, (str, bytes)):
        rows.extend({"kind": "test", "result_class": row.get("result"), **dict(row)}
                    for row in feedback if isinstance(row, Mapping))
    return change_after_green_from_receipts(rows)


def collapse_stale_test_results(frame: Mapping[str, object], receipts: object = ()) -> dict:
    """Replace all stale public test prose with one typed host-owned row."""
    public = dict(frame)
    test_keys = [key for key in ("recent_test_feedback", "test_receipts", "receipts")
                 if isinstance(frame.get(key), Sequence)
                 and not isinstance(frame.get(key), (str, bytes))]
    if not test_keys:
        return public
    current_revision = _frame_revision(frame)
    timeline = _receipt_timeline(receipts)
    latest_edit = max(
        (index for index, row in enumerate(timeline)
         if _receipt_kind(row) in {"edit", "str_replace", "file_create", "replace_module", "write"}
         and _receipt_result(row) == "APPLIED"),
        default=-1,
    )
    # A receipt's position is more robust than a missing/shifted turn number.
    receipt_position = {str(row.get("receipt_id")): index for index, row in enumerate(timeline)
                        if row.get("receipt_id") is not None}
    fresh_by_key: dict[str, list[dict]] = {key: [] for key in test_keys}
    stale: list[dict] = []
    for source_key in test_keys:
        rows = frame.get(source_key)
        for raw in rows:
            if not isinstance(raw, Mapping):
                continue
            row = dict(raw)
            kind = _receipt_kind(row)
            is_test = source_key != "receipts" or kind in {
                "test", "run_tests", "pytest", "verification",
            } or any(key in row for key in ("selector", "selectors", "suite", "exit_code"))
            if not is_test:
                fresh_by_key[source_key].append(row)
                continue
            revision = _row_revision(row)
            stale_revision = (revision is not None and current_revision is not None
                              and str(revision) != str(current_revision))
            step = _row_step(row)
            stale_step = False
            if latest_edit >= 0:
                identifier = row.get("receipt_id") or row.get("event_id")
                position = receipt_position.get(str(identifier), -1) if identifier else -1
                stale_step = (position >= 0 and position < latest_edit) or (
                    position < 0 and step is not None and any(
                        _row_step(edit) is not None and _row_step(edit) > step
                        for edit in timeline
                        if _receipt_kind(edit) in {"edit", "str_replace", "file_create", "replace_module", "write"}
                        and _receipt_result(edit) == "APPLIED"))
            if stale_revision or stale_step:
                stale.append(row)
            else:
                for field, value in list(row.items()):
                    if isinstance(value, str):
                        row[field] = strip_harness_phrase(value)
                fresh_by_key[source_key].append(row)
    if not stale:
        for key, rows in fresh_by_key.items():
            public[key] = rows
        return public
    selectors = list(dict.fromkeys(_selector(row) for row in stale))
    revisions = list(dict.fromkeys(
        "UNKNOWN" if _row_revision(row) is None else str(_row_revision(row))
        for row in stale))
    latest = stale[-1]
    stale_line = {
        "kind": "kernel_stale_test_result",
        "selector": ", ".join(selectors)[:320],
        "revision": revisions[0] if len(revisions) == 1 else "MULTIPLE",
        "result": _receipt_result(latest),
        "status": "STALE",
        "feedback": "[kernel] STALE test result superseded by a newer revision or patch.",
        "authority": "HOST_OBSERVED",
    }
    for key, rows in fresh_by_key.items():
        public[key] = rows
    primary_key = ("recent_test_feedback" if "recent_test_feedback" in fresh_by_key
                   else "test_receipts" if "test_receipts" in fresh_by_key
                   else "receipts")
    public[primary_key] = [*public[primary_key], stale_line]
    public["stale_test_results_collapsed"] = len(stale)
    return public


def _line_indent(value: str) -> str:
    first = value.splitlines()[0] if value.splitlines() else ""
    return first[:len(first) - len(first.lstrip(" \t"))]


def _normalized_lines(value: str) -> str:
    return textwrap.dedent(value).strip("\n")


def _reindent(value: str, delta: int) -> str:
    if delta == 0:
        return value
    result: list[str] = []
    for index, line in enumerate(value.splitlines(keepends=True)):
        if not line.strip():
            result.append(line)
            continue
        if delta > 0:
            result.append(" " * delta + line)
        else:
            remove = min(-delta, len(line) - len(line.lstrip(" \t")))
            result.append(line[remove:])
    return "".join(result)


def normalized_str_replace_candidate(source: str, old: str, new: str) -> tuple[str | None, str | None]:
    """Find a unique match after removing only common leading indentation."""
    old_lines = old.splitlines(keepends=True)
    if not old_lines or not _normalized_lines(old):
        return None, None
    target = _normalized_lines(old)
    source_lines = source.splitlines(keepends=True)
    matches: list[tuple[int, int, int]] = []
    offset = 0
    offsets = [0]
    for line in source_lines:
        offset += len(line)
        offsets.append(offset)
    wanted = len(old_lines)
    for start in range(0, len(source_lines) - wanted + 1):
        window = "".join(source_lines[start:start + wanted])
        if _normalized_lines(window) != target:
            continue
        source_indent = len(_line_indent(window).expandtabs(8))
        old_indent = len(_line_indent(old).expandtabs(8))
        matches.append((offsets[start], offsets[start + wanted], source_indent - old_indent))
    if len(matches) > 1:
        return None, (
            "str_replace REJECTED [typed] "
            + json.dumps({"schema": "stateframe-edit-rejection/1",
                          "kind": "str_replace",
                          "reason": "normalized_match_ambiguous",
                          "matches": len(matches)}, sort_keys=True,
                         separators=(",", ":"))
            + ": normalized old_str matches "
            f"{len(matches)} places; leading-indent normalization is ambiguous. "
            "Include more surrounding lines so it is unique."
        )
    if not matches:
        return None, None
    start, end, delta = matches[0]
    replacement = _reindent(new, delta)
    return source[:start] + replacement + source[end:], "normalized leading-indent match"


def unique_method_target(source: str, name: str) -> tuple[str | None, str | None]:
    """Resolve an unqualified method name only when exactly one method matches."""
    if "." in name:
        return name, None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None, None
    candidates: list[str] = []

    def visit(body: list[ast.stmt], prefix: tuple[str, ...] = ()) -> None:
        for node in body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = ".".join((*prefix, node.name))
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and "." in qualified \
                        and node.name == name:
                    candidates.append(qualified)
                visit(node.body, (*prefix, node.name))

    visit(tree.body)
    if len(candidates) == 1:
        return candidates[0], None
    if len(candidates) > 1:
        return None, (
            f"edit REJECTED: unqualified method `{name}` is ambiguous ({', '.join(candidates[:8])}); "
            "use the qualified class.method target."
        )
    return None, None


def identical_frame_fact(counter: int, last_receipt: Mapping[str, object] | None = None) -> str:
    last_receipt = last_receipt if isinstance(last_receipt, Mapping) else {}
    payload = {
        "schema": "stateframe-identical-request/1",
        "kind": "host_fact",
        "status": "unchanged",
        "stall_counter": int(counter),
        "last_action": str(last_receipt.get("kind", "UNKNOWN")),
        "last_result": str(last_receipt.get("result_class", last_receipt.get("result", "UNKNOWN"))),
    }
    return "[HOST_FACT typed] " + json.dumps(payload, sort_keys=True, separators=(",", ":"))


_VOLATILE_FRAME_KEYS = frozenset({"step", "receipt_count", "budget", "closure_budget",
                                  "last_observation"})


def stable_frame_identity(frame: Mapping[str, object]) -> dict:
    """Project the logical frame for the no-resend guard.

    Extraction and parent accounting advance while a request is being
    prepared.  Those counters are host bookkeeping, not a changed task frame;
    keeping them in the identity would miss the repeated-request failure that
    this opt-in guard is meant to catch.
    """
    def project(value: object, depth: int = 0) -> object:
        if isinstance(value, Mapping):
            return {
                str(key): project(item, depth + 1)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
                if not (depth == 0 and str(key) in _VOLATILE_FRAME_KEYS)
            }
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [project(item, depth + 1) for item in value]
        return value

    return project(frame)  # type: ignore[return-value]


def _failure_witness(receipts: object) -> str | None:
    rows = _receipt_timeline(receipts)
    for row in reversed(rows):
        if _receipt_kind(row) not in {"test", "run_tests", "pytest", "verification"}:
            continue
        detail = row.get("failure_tail") or row.get("detail") or row.get("observation") or row.get("why")
        text = str(detail or "")
        matches = _WITNESS_RE.findall(text)
        if matches:
            return _clean_text(matches[-1], 620)
        if _receipt_result(row) in {"RED", "FAILED", "ERROR"} and text.strip():
            return _clean_text(text.splitlines()[-1], 620)
    return None


def enrich_refusal(detail: object, receipts: object, *, selectors: Sequence[str] = (),
                   path: str | None = None, menu: Sequence[str] = ()) -> dict:
    """Attach a bounded failure witness and one schema-valid repair action."""
    text = str(detail or "completion is not admissible")
    witness = _failure_witness(receipts)
    selector = next((str(item) for item in selectors if str(item).strip()), "")
    action: dict[str, object] | None = None
    if selector and "run_tests" in menu:
        action = {"tool": "run_tests", "arguments": {"target": selector},
                  "reason": "rerun the failing assertion"}
    elif path and "read" in menu:
        action = {"tool": "read", "arguments": {"path": path},
                  "reason": "read the current source at the failing boundary"}
    payload = {
        "schema": "stateframe-refusal/1",
        "witness": witness or "NO_BOUNDED_ASSERTION_OR_EXCEPTION_RECEIPT",
        "repair_action": action,
    }
    marker = "[kernel-refusal] " + json.dumps(payload, sort_keys=True, separators=(",", ":"))
    if marker not in text:
        text = text.rstrip() + "\n" + marker
    return {"detail": text, "repair_action": action, "witness": witness}


def _module_name(path: Path, checkout: Path) -> str | None:
    try:
        relative = path.resolve().relative_to(checkout.resolve())
    except (OSError, ValueError):
        return None
    if relative.suffix != ".py":
        return None
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) if parts else None


def _function_line_map(source: str) -> dict[str, int]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    result: dict[str, int] = {}

    def visit(body: list[ast.stmt], prefix: tuple[str, ...] = ()) -> None:
        for node in body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = ".".join((*prefix, node.name))
                result[qualified] = int(getattr(node, "lineno", 1))
                visit(node.body, (*prefix, node.name))

    visit(tree.body)
    return result


def _brief_paths(brief: object, checkout: Path) -> list[tuple[Path, int, str | None]]:
    text = str(brief or "")
    found: list[tuple[Path, int, str | None]] = []
    for match in TRACEBACK_RE.finditer(text):
        raw = match.group("path").replace("\\", "/")
        candidate = (checkout / raw).resolve()
        if not candidate.is_relative_to(checkout.resolve()):
            continue
        symbol_match = TRACEBACK_FUNCTION_RE.match(text, match.start())
        symbol = symbol_match.group("symbol") if symbol_match else None
        found.append((candidate, int(match.group("line")), symbol))
    return found


def flag_brief_traceback_lines(brief: object, checkout: Path) -> list[dict]:
    """Flag only demonstrable brief/source mismatches; never rewrite the brief."""
    findings: list[dict] = []
    for path, line, symbol in _brief_paths(brief, checkout):
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            findings.append({"file": str(path), "line": line, "reason": "file_missing"})
            continue
        lines = source.splitlines()
        mismatch = line < 1 or line > len(lines) or not lines[line - 1].strip()
        if symbol:
            spans = []
            try:
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                            and node.name == symbol:
                        spans.append((node.lineno, getattr(node, "end_lineno", node.lineno)))
            except SyntaxError:
                mismatch = True
            if spans and not any(start <= line <= end for start, end in spans):
                mismatch = True
        key_match = KEY_ERROR_RE.search(str(brief or ""))
        if key_match:
            needle = key_match.group("key")
            anchors = [index for index, source_line in enumerate(lines, 1)
                       if re.search(rf"(?:[\"']{re.escape(needle)}[\"']|\.{re.escape(needle)}\b)",
                                    source_line)]
            if len(anchors) == 1 and anchors[0] != line:
                mismatch = True
        if mismatch:
            findings.append({"file": path.as_posix(), "line": line,
                             "reason": "brief_line_does_not_match_current_source"})
    return findings


def read_dedup_observation(runtime: object, path: str, start_line: int,
                           *, range_requested: bool = False) -> dict | None:
    """Return a zero-I/O result only when source bytes were in the last request.

    ``_empirical_frame_sources`` is a logical-frame index.  It is useful for
    diagnostics, but it is not delivery evidence: terminal selection may have
    omitted that source.  The runtime therefore records a second index only
    after final request composition, and this function trusts that serialized
    request ledger instead.
    """
    from kernel_read_fixes import (
        read_receipt_enabled, read_unchanged_receipt, source_header,
    )
    from kernel_host_autopilot import enabled as host_autopilot_enabled
    from kernel_serve_reads import enabled as _serve_reads_enabled, note_served as _serve_note
    receipt_enabled = read_receipt_enabled() or host_autopilot_enabled()
    empirical_enabled = getattr(runtime, "empirical_fixes_mode", "off") == "on"
    serve_reads_enabled = _serve_reads_enabled()
    stall_enabled = getattr(runtime, "stateframe_tokenlab_stall_breaker_mode", "off") == "on"
    from kernel_tokenlab_stall_breaker import stallwire_enabled
    stallwire_owner_enabled = stallwire_enabled(runtime)
    if not empirical_enabled and not receipt_enabled and not serve_reads_enabled and not stall_enabled:
        return None
    # Only a frame re-renders current source every request (runtime.frame_series is set only for
    # KERNEL_FRAME_SERIES=1). In a transcript (history arm) "was in the last request" does not survive
    # later trimming: the stub kept
    # saying "already present" after the bytes were trimmed away, blinding history to the file
    # (ldev_12-924105 dissection, 09-25; 22/36 history runs received stubs, 0/36 frame runs).
    if getattr(runtime, "frame_series", None) is None:
        return None
    relative = str(path).replace("\\", "/").lstrip("./")
    serialized = getattr(runtime, "_empirical_serialized_sources", {})
    row = serialized.get(relative) if isinstance(serialized, Mapping) else None
    current_revisions = getattr(runtime, "_empirical_current_revisions", {})
    current_revision = (current_revisions.get(relative)
                        if isinstance(current_revisions, Mapping) else None)
    if current_revision is None:
        artifact_hashes = getattr(runtime, "artifact_hashes", {})
        current_revision = (artifact_hashes.get(relative)
                            if isinstance(artifact_hashes, Mapping) else None)
    exact_full_revision = (
        isinstance(row, Mapping) and start_line == 0 and not range_requested
        and row.get("full") is True and current_revision is not None
        and str(row.get("revision")) == str(current_revision)
        and isinstance(row.get("request_hash"), str))
    if serve_reads_enabled and exact_full_revision:
        # even an exact full-frame hit must fall through to the
        # real read; only the matching counterfactual is journaled here.
        _serve_note(runtime, relative,
                    would_have="read_unchanged_receipt" if receipt_enabled else "stub",
                    predicate="read_dedup_exact_full")
        return None
    if serve_reads_enabled:
        dedup_candidate = (
            isinstance(row, Mapping) and start_line == 0 and not range_requested
            and row.get("full") is True and current_revision is not None
            and str(row.get("revision")) == str(current_revision)
            and isinstance(row.get("request_hash"), str))
        if dedup_candidate:
            _serve_note(runtime, relative, would_have="stub", predicate="read_dedup")
        return None
    # A range is a new read-receipt guard.  When that switch is OFF, ignore the
    # extra keyword so the pre-existing empirical-dedup arm remains byte-identical.
    effective_range_requested = range_requested if receipt_enabled else False
    if (isinstance(row, Mapping) and start_line == 0 and not effective_range_requested
            and row.get("full") is True
            and current_revision is not None
            and str(row.get("revision")) == str(current_revision)
            and isinstance(row.get("request_hash"), str)):
        if receipt_enabled:
            header = row.get("header")
            if not isinstance(header, str) or not header:
                header = source_header(relative, role="working")
            receipt = read_unchanged_receipt(
                path=relative, revision=row.get("revision", "UNKNOWN"), header=header)
            from kernel_reread_fix import enabled as _reread_fix_enabled
            if _reread_fix_enabled():
                from kernel_reread_fix import enrich_read_receipt
                enrich_read_receipt(receipt, runtime)
            if stall_enabled and not stallwire_owner_enabled:
                from kernel_tokenlab_stall_breaker import observe_read_unchanged
                observe_read_unchanged(runtime, relative, row.get("revision", "UNKNOWN"), receipt)
            return receipt
        receipt = {
            "schema": "stateframe-read-dedup/1",
            "status": "ALREADY_PRESENT",
            "path": relative,
            "revision": row.get("revision", "UNKNOWN"),
            "reference": row.get("reference", f"frame:{relative}"),
            "message": f"already present in frame (ref) {row.get('reference', f'frame:{relative}')}",
        }
        if stall_enabled and not stallwire_owner_enabled:
            from kernel_tokenlab_stall_breaker import observe_read_unchanged
            observe_read_unchanged(runtime, relative, row.get("revision", "UNKNOWN"), receipt)
        return receipt
    # A closed host read is not enough here.  The next request may be a
    # green-lock terminal projection that omitted the read bytes; only the
    # serialized-request ledger above proves model delivery.
    return None


def replay_change_after_green(events: Sequence[Mapping[str, object]]) -> str:
    """Replay evidence-journal events without trusting rendered prose."""
    states = replay_change_after_green_states(events)
    return states[-1] if states else "no_green_yet"


def replay_change_after_green_states(events: Sequence[Mapping[str, object]]) -> list[str]:
    """Return the three-valued fact after each relevant journal receipt."""
    receipts: list[Mapping[str, object]] = []
    states: list[str] = []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        row: Mapping[str, object] | None = None
        if event.get("kind") == "action_receipt":
            candidate = event.get("receipt")
            row = candidate if isinstance(candidate, Mapping) else None
        elif event.get("kind") == "action_outcome_closed":
            candidate = event.get("record")
            row = candidate if isinstance(candidate, Mapping) else None
        elif event.get("kind") == "test_subprocess_completed":
            row = dict(event)
            row["kind"] = "test"
            row["result_class"] = "GREEN" if row.get("exit_code") == 0 else "RED"
        if row is None:
            continue
        kind = _receipt_kind(row)
        if kind not in {"test", "run_tests", "pytest", "verification", "edit",
                        "str_replace", "file_create", "replace_module", "write"}:
            continue
        receipts.append(row)
        states.append(change_after_green_from_receipts(receipts))
    return states


__all__ = [
    "MODE_ENV", "STRUCTURE_FACTS_MODE_ENV", "change_after_green_from_frame", "change_after_green_from_receipts",
    "collapse_stale_test_results", "empirical_mode", "enabled", "enrich_refusal",
    "flag_brief_traceback_lines", "identical_frame_fact", "normalized_str_replace_candidate",
    "location_facts_from_test_result", "read_dedup_observation", "replay_change_after_green",
    "structure_facts_enabled", "structure_facts_for_edit", "structure_facts_mode",
    "structure_facts_observation", "strip_harness_phrase",
    "replay_change_after_green_states", "stable_frame_identity", "unique_method_target",
]
