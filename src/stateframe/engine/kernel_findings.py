"""Kernel-derived, bounded findings for the live StateFrame working view."""
from __future__ import annotations

import ast
import os
import re
from collections.abc import Mapping, Sequence


FINDINGS_TOKEN_CAP = 400
_UNKNOWN = "UNKNOWN"
_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_UNIT_PATTERN = re.compile(
    r"(?i)(?<!\w)(?:per\s+(?:second|minute|hour|request)|seconds?|minutes?|hours?|"
    r"milliseconds?|ms|tokens?|bytes?|requests?|items?|records?|kib|mib|gib|kb|mb|gb|percent)(?!\w)|%"
)
_VERDICT_NAMES = {
    "PASS": "GREEN", "PASSED": "GREEN", "PASSING": "GREEN", "GREEN": "GREEN",
    "OK": "GREEN", "SUCCESS": "GREEN", "REUSED_VERIFICATION": "GREEN",
    "FAIL": "RED", "FAILED": "RED", "FAILURE": "RED", "RED": "RED",
    "ERROR": "RED", "TESTS_FAILED": "RED",
}


def lexical_token_count(value: object) -> int:
    """Return the provider-independent lexical estimate used by the cap."""
    return len(_TOKEN_PATTERN.findall(value if isinstance(value, str) else str(value)))


def _mapping_values(value: object) -> list[object]:
    if isinstance(value, Mapping):
        if any(key in value for key in (
                "source", "test_src", "public_test_source", "content")):
            return [value]
        return list(value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


def _source_texts(frame: Mapping[str, object]) -> list[str]:
    values: list[object] = []
    for key in ("public_test_source", "public_test_src", "test_source", "test_src"):
        if key in frame:
            values.append(frame[key])
    for key in ("kernel_findings", "findings_input", "public_tests", "public_test",
                "row", "task_row"):
        container = frame.get(key)
        if isinstance(container, Mapping):
            for name in ("public_test_source", "public_test_src", "test_source", "test_src", "source",
                         "public_tests", "public_test"):
                if name in container:
                    values.extend(_mapping_values(container[name]) or [container[name]])
        elif key in {"public_tests", "public_test"}:
            values.extend(_mapping_values(container))
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, Mapping):
            nested = next((value[key] for key in ("source", "public_test_src", "test_src", "content")
                           if isinstance(value.get(key), str)), None)
            value = nested
        if isinstance(value, str) and value.strip() and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _brief_text(frame: Mapping[str, object]) -> str | None:
    values: list[object] = []
    for key in ("task_brief", "brief", "msg"):
        if key in frame:
            values.append(frame[key])
    for key in ("kernel_findings", "findings_input", "row", "task_row"):
        container = frame.get(key)
        if isinstance(container, Mapping):
            for name in ("brief", "task_brief", "msg"):
                if name in container:
                    values.append(container[name])
    return next((value for value in values if isinstance(value, str) and value.strip()), None)


def _node_text(node: ast.AST) -> str | None:
    try:
        return ast.unparse(node)
    except (AttributeError, TypeError):
        return None


def _literal(node: ast.AST, *, references: bool = False) -> bool:
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Name):
        return references
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(_literal(item, references=True) for item in node.elts)
    if isinstance(node, ast.Dict):
        return all(key is not None and _literal(key, references=True)
                   and _literal(value, references=True)
                   for key, value in zip(node.keys, node.values))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _literal(node.operand)
    return False


def _raises_call(node: ast.AST) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "raises" and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "pytest")


def _call_text(node: ast.AST) -> str | None:
    return _node_text(node) if isinstance(node, ast.Call) else None


def _mark_operation_names(node: ast.Call, names: set[str]) -> None:
    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
        names.add(node.func.value.id)
    for argument in (*node.args, *(keyword.value for keyword in node.keywords)):
        if isinstance(argument, ast.Name):
            names.add(argument.id)


def _comparison(assertion: ast.Assert, calls: Mapping[str, str]) -> tuple[str, str, str] | None:
    test = assertion.test
    if not isinstance(test, ast.Compare) or len(test.ops) != 1 or len(test.comparators) != 1:
        return None
    operator = test.ops[0]
    op = "==" if isinstance(operator, ast.Eq) else (
        "is" if isinstance(operator, ast.Is) else "in" if isinstance(operator, ast.In) else None)
    if op is None:
        return None
    left = _node_text(test.left)
    right = _node_text(test.comparators[0])
    if left is None or right is None:
        return None
    if isinstance(test.left, ast.Name) and test.left.id in calls:
        left = calls[test.left.id]
    return left, op, right


def _unsupported_test(name: str) -> str:
    try:
        from kernel_frame_clarity import enabled as frame_clarity_enabled
        if frame_clarity_enabled():
            return f"{name}: oracle extraction unavailable"
    except ImportError:
        pass
    return f"{name}: (not normalized; see test source)"


def _normalize_test_function(function: ast.FunctionDef | ast.AsyncFunctionDef,
                             source: str) -> str:
    arguments = function.args
    if (arguments.posonlyargs or arguments.args or arguments.kwonlyargs
            or arguments.vararg is not None or arguments.kwarg is not None):
        return _unsupported_test(function.name)

    fixtures: list[str] = []
    calls: list[str] = []
    call_bindings: dict[str, str] = {}
    operation_names: set[str] = set()
    assertions: list[tuple[ast.Assert, tuple[str, str, str] | None]] = []
    preserves: list[str] = []
    preserved_statements: set[int] = set()
    raises: list[tuple[str, str]] = []
    unsupported = False
    literal_names: set[str] = set()

    def assignment(statement: ast.Assign | ast.AnnAssign) -> bool:
        nonlocal unsupported
        if isinstance(statement, ast.Assign):
            if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
                unsupported = True
                return True
            target = statement.targets[0]
            value = statement.value
        else:
            if not isinstance(statement.target, ast.Name):
                unsupported = True
                return True
            target = statement.target
            value = statement.value
            if value is None:
                unsupported = True
                return True
        if _literal(value):
            rendered = _node_text(value)
            if rendered is None:
                unsupported = True
            else:
                literal_names.add(target.id)
                fixtures.append(f"{target.id} = {rendered}")
            return True
        rendered_call = _call_text(value)
        if rendered_call is not None and not _raises_call(value):
            call_bindings[target.id] = rendered_call
            calls.append(rendered_call)
            _mark_operation_names(value, operation_names)
            return True
        unsupported = True
        return True

    for statement in function.body:
        if (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)) or isinstance(statement, ast.Pass):
            continue
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            assignment(statement)
        elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            rendered = _call_text(statement.value)
            if rendered is None:
                unsupported = True
            else:
                calls.append(rendered)
                _mark_operation_names(statement.value, operation_names)
        elif isinstance(statement, ast.Assert):
            parsed = _comparison(statement, call_bindings)
            assertions.append((statement, parsed))
            if parsed is None:
                unsupported = True
            elif (isinstance(statement.test, ast.Compare)
                  and isinstance(statement.test.left, ast.Name)
                  and statement.test.left.id in literal_names
                  and statement.test.left.id in operation_names):
                text = _node_text(statement.test)
                if text is not None:
                    text = ast.get_source_segment(source, statement.test) or text
                    preserves.append(f"preserve: assert {text.strip()}")
                    preserved_statements.add(id(statement))
        elif isinstance(statement, ast.With) and len(statement.items) == 1:
            item = statement.items[0]
            if not _raises_call(item.context_expr) or len(statement.body) != 1:
                unsupported = True
                continue
            if not (isinstance(statement.body[0], ast.Expr)
                    and isinstance(statement.body[0].value, ast.Call)):
                unsupported = True
                continue
            exception = _node_text(item.context_expr)
            call = _call_text(statement.body[0].value)
            if exception is None or call is None:
                unsupported = True
            else:
                raises.append((exception, call))
                _mark_operation_names(statement.body[0].value, operation_names)
        else:
            unsupported = True

    if unsupported or (not assertions and not raises):
        return _unsupported_test(function.name)

    descriptions: list[str] = []
    for statement, parsed in assertions:
        if parsed is None:
            return _unsupported_test(function.name)
        left, operator, right = parsed
        if id(statement) in preserved_statements:
            continue
        if isinstance(statement.test, ast.Compare) and isinstance(statement.test.left, ast.Call):
            descriptions.append(f"{left} {operator} {right}")
        elif (isinstance(statement.test, ast.Compare)
              and isinstance(statement.test.left, ast.Name)
              and statement.test.left.id in call_bindings):
            descriptions.append(f"{left} {operator} {right}")
        else:
            descriptions.append(f"assert {left} {operator} {right}")
    descriptions.extend(f"{exception} → {call}" for exception, call in raises)

    represented_calls = {
        description.split(operator, 1)[0]
        for description in descriptions
        for operator in (" == ", " is ", " in ")
        if operator in description
    }
    unrepresented = [call for call in calls if call not in represented_calls]
    if unrepresented:
        descriptions.insert(0, "; ".join(unrepresented))
    descriptions.extend(preserves)
    body = "; ".join(descriptions)
    if fixtures:
        body = "; ".join(fixtures) + " → " + body
    return f"{function.name}: {body}"


def normalize_test_oracle(source: str | None) -> list[str]:
    """Normalize only literal/direct public-test shapes; never infer unsupported ones."""
    if not isinstance(source, str) or not source.strip():
        return [_UNKNOWN]
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [_UNKNOWN]
    functions = [node for node in ast.walk(tree)
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name.startswith("test_")]
    functions.sort(key=lambda node: (getattr(node, "lineno", 0), getattr(node, "col_offset", 0)))
    if not functions:
        return [_UNKNOWN]
    return [_normalize_test_function(function, source) for function in functions]


def _recorded_outcomes(frame: Mapping[str, object]) -> list[Mapping[str, object]]:
    values: list[object] = []
    for key in ("recent_test_feedback", "test_outcomes", "public_test_outcomes",
                "test_receipts"):
        value = frame.get(key)
        if isinstance(value, Mapping) and any(name in value for name in (
                "result", "result_class", "verdict", "exit_code", "feedback")):
            values.append(value)
        elif isinstance(value, Mapping):
            values.extend(value.values())
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            values.extend(value)
    nested = frame.get("kernel_findings")
    if isinstance(nested, Mapping):
        value = nested.get("test_outcomes")
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            values.extend(value)
    return [value for value in values if isinstance(value, Mapping)]


def _revision_from(value: object) -> str | None:
    if not isinstance(value, Mapping):
        return None
    for key in ("current_revision", "revision", "candidate_revision", "product_revision",
                "artifact_revision", "artifact"):
        item = value.get(key)
        if item is not None:
            return str(item)
    for key in ("binding", "after_binding", "current_binding", "resources"):
        item = _revision_from(value.get(key))
        if item is not None:
            return item
    for key in ("artifact_revisions", "product_files"):
        item = value.get(key)
        if isinstance(item, Mapping) and len(item) == 1:
            revision = next(iter(item.values()))
            if revision is not None:
                return str(revision)
    return None


def _current_revision(frame: Mapping[str, object]) -> str | None:
    revision = _revision_from(frame)
    if revision is not None:
        return revision
    contract = frame.get("completion_contract")
    if isinstance(contract, Mapping):
        revision = _revision_from(contract.get("current_public_checks"))
        if revision is not None:
            return revision
    return _revision_from(frame.get("verification"))


def _short_revision(value: str | None) -> str:
    if value is None:
        return _UNKNOWN
    text = str(value)
    if text.startswith("sha256:"):
        return text[7:15]
    return text if len(text) <= 16 else text[:12]


def _verdict(record: Mapping[str, object]) -> str:
    value: object = None
    for key in ("verdict", "result", "result_class", "status"):
        if record.get(key) is not None:
            value = record[key]
            break
    if value is None and type(record.get("exit_code")) is int:
        value = "GREEN" if record["exit_code"] == 0 else "RED" if record["exit_code"] == 1 else None
    if isinstance(value, str):
        normalized = value.upper().replace(" ", "_")
        if normalized in _VERDICT_NAMES:
            return _VERDICT_NAMES[normalized]
        match = re.search(r"pytest\s+exit\s+(\d+)", value, re.I)
        if match:
            return "GREEN" if match.group(1) == "0" else "RED" if match.group(1) == "1" else _UNKNOWN
    return _UNKNOWN


def _failure_detail(value: object) -> str | None:
    if isinstance(value, Mapping):
        if "expected" in value or "observed" in value:
            return (f"expected={value.get('expected', _UNKNOWN)} "
                    f"observed={value.get('observed', _UNKNOWN)}")
        return None
    if not isinstance(value, str):
        return None
    pair = re.search(
        r"\bexpected\s*[:=]\s*(.*?)\s*(?:(?:[,;]\s*)?observed\s*[:=])\s*(.*?)(?:\n|$)",
        value, re.I)
    if pair:
        return f"expected={pair.group(1).strip()} observed={pair.group(2).strip()}"
    expected = re.search(r"\bexpected\s*[:=]\s*(.*?)(?=\s+\bobserved\b|$)", value, re.I)
    observed = re.search(r"\bobserved\s*[:=]\s*(.*?)(?=\s+\bexpected\b|$)", value, re.I)
    if expected and observed:
        return f"expected={expected.group(1).strip()} observed={observed.group(1).strip()}"
    for line in value.splitlines():
        # Pytest prints the source context as an indented ``assert`` line and
        # the failing assertion again with an ``E`` prefix.  Only the latter
        # is an observed failure; accepting the former can point the ledger at
        # a passing assertion from the traceback context.
        failure_line = line.strip()
        if not re.match(r"^E(?:\s+|$)", failure_line):
            continue
        failure_line = re.sub(r"^E\s*", "", failure_line)
        match = re.search(r"\bassert\s+(.+?)\s+(==|is|in)\s+(.+?)\s*$",
                          failure_line, re.I)
        if match:
            return f"expected={match.group(3).strip()} observed={match.group(1).strip()}"
    return None


def failure_ledger_entries(frame: Mapping[str, object]) -> list[dict[str, str | None]]:
    """Return the latest recorded outcome for each of the last distinct revisions."""
    by_revision: dict[str, dict[str, str | None]] = {}
    for record in _recorded_outcomes(frame):
        revision = _revision_from(record)
        key = revision if revision is not None else _UNKNOWN
        verdict = _verdict(record)
        observation = (record.get("feedback") or record.get("observation")
                       or record.get("detail"))
        entry = {"revision": revision, "verdict": verdict,
                 "detail": _failure_detail(observation) if verdict == "RED" else None}
        by_revision.pop(key, None)
        by_revision[key] = entry
    current = _current_revision(frame)
    entries = list(by_revision.values())[-3:]
    for entry in entries:
        if current is not None and entry["revision"] is not None and entry["revision"] != current:
            entry["historical"] = "historical"
        else:
            entry["historical"] = None
    return entries


def render_failure_ledger(frame: Mapping[str, object]) -> list[str]:
    try:
        from kernel_frame_clarity import enabled as frame_clarity_enabled, project_model_frame
        if frame_clarity_enabled():
            frame = project_model_frame(frame)
    except ImportError:
        pass
    entries = failure_ledger_entries(frame)
    if not entries:
        return [_UNKNOWN]
    lines = []
    for entry in entries:
        line = f"{_short_revision(entry['revision'])}: {entry['verdict']}"
        if entry.get("detail"):
            line += f"; {entry['detail']}"
        if entry.get("historical"):
            line += " historical"
        lines.append(line)
    return lines


def extract_unit_clauses(brief: str | None) -> list[str]:
    """Return at most four verbatim brief sentences containing a unit word."""
    if not isinstance(brief, str) or not brief.strip():
        return []
    if os.environ.get("STATEFRAME_ORACLE_GROUNDING", "off").strip().lower() == "on":
        from kernel_oracle_grounding import extract_ticket_clauses
        clauses = extract_ticket_clauses(brief)
        return [clause for clause in clauses if _UNIT_PATTERN.search(clause)][:4]
    sentences = re.split(r"(?<=[.!?])(?=\s|$)", brief)
    if len(sentences) == 1 and "\n" in brief:
        sentences = brief.splitlines()
    return [sentence.strip() for sentence in sentences
            if sentence.strip() and _UNIT_PATTERN.search(sentence)][:4]


_NUMBERED_CLAUSE = re.compile(r"(?m)(?:^|\s)((?:\(\d+\)|\d+[.)])\s+)")
_NAMED_PIECE = re.compile(r"\b(?:class|def|function|method)\s+[A-Za-z_]\w*")


def extract_brief_clauses(brief: str | None, *, include_named: bool = False) -> list[str]:
    """Return numbered or explicitly named brief pieces when the guard is enabled."""
    if os.environ.get("STATEFRAME_ORACLE_GROUNDING", "off").strip().lower() == "on":
        from kernel_oracle_grounding import extract_ticket_clauses
        return list(extract_ticket_clauses(brief))
    if not include_named:
        return extract_unit_clauses(brief)
    if not isinstance(brief, str) or not brief.strip():
        return []
    numbered = list(_NUMBERED_CLAUSE.finditer(brief))
    if numbered:
        result = []
        for index, match in enumerate(numbered):
            start = match.start(1)
            end = numbered[index + 1].start(1) if index + 1 < len(numbered) else len(brief)
            clause = brief[start:end].strip()
            if clause:
                result.append(clause)
        return result

    pieces = re.split(r"(?m)^\s*(?=[-*•]\s+)", brief)
    named = [piece.strip() for piece in pieces if _NAMED_PIECE.search(piece)]
    if named:
        return named
    named_matches = list(_NAMED_PIECE.finditer(brief))
    if len(named_matches) > 1:
        return [brief[match.start():named_matches[index + 1].start()
                       if index + 1 < len(named_matches) else len(brief)].strip()
                for index, match in enumerate(named_matches)]
    return extract_unit_clauses(brief)


def _clause_symbols(clause: str) -> list[str]:
    return list(dict.fromkeys(match.group(1) for match in re.finditer(
        r"\b(?:class|def|function|method)\s+([A-Za-z_]\w*)", clause)))


def _public_test_coverage(clause: str, oracle: Sequence[str], sources: Sequence[str]) -> str:
    """Mark named brief pieces from normalized public-test evidence only."""
    if not sources or not oracle or all(str(item) == _UNKNOWN for item in oracle):
        return "unknown"
    symbols = _clause_symbols(clause)
    if not symbols:
        return "unknown"
    return "yes" if any(re.search(r"\b" + re.escape(symbol) + r"\b", str(line))
                        for symbol in symbols for line in oracle) else "no"


def brief_clause_coverage(frame: Mapping[str, object]) -> list[dict[str, str]]:
    """Return the same numbered/named clause coverage used by guarded findings."""
    sources = _source_texts(frame)
    oracle = normalize_test_oracle("\n\n".join(sources) if sources else None)
    clauses = extract_brief_clauses(_brief_text(frame), include_named=True)
    return [
        {"id": f"C{index}", "coverage": _public_test_coverage(clause, oracle, sources),
         "clause": clause}
        for index, clause in enumerate(clauses, 1)
    ]


def _findings_lines(frame: Mapping[str, object]) -> list[tuple[str, int, bool]]:
    frame_clarity = False
    try:
        from kernel_frame_clarity import enabled as frame_clarity_enabled, project_model_frame
        frame_clarity = frame_clarity_enabled()
        if frame_clarity:
            frame = project_model_frame(frame)
    except ImportError:
        pass
    sources = _source_texts(frame)
    oracle = normalize_test_oracle("\n\n".join(sources) if sources else None)
    from kernel_edit_guard import enabled as edit_guard_enabled
    guarded = edit_guard_enabled()
    clauses = extract_brief_clauses(_brief_text(frame), include_named=guarded)
    lines: list[tuple[str, int, bool]] = [
        ("OBSERVED / SPECIFIED (kernel facts only):", -1, True),
        ("TEST ORACLE:", -1, True),
    ]
    lines.extend((line, 30 if not line.startswith(_UNKNOWN) else 0, line == _UNKNOWN)
                 for line in oracle)
    if frame_clarity and any("oracle extraction unavailable" in line for line in oracle):
        lines.append(("oracle_extraction: UNAVAILABLE", -1, True))
    lines.append(("test_results:" if frame_clarity else "FAILURE LEDGER:", -1, True))
    ledger = render_failure_ledger(frame)
    lines.extend((line, 20 if line != _UNKNOWN else 0, line == _UNKNOWN) for line in ledger)
    host_findings = frame.get("host_observed_findings") if guarded else None
    if (host_findings is None and guarded
            and isinstance(frame.get("kernel_findings"), Mapping)):
        host_findings = frame["kernel_findings"].get("host_observed_findings")
    if isinstance(host_findings, Sequence) and not isinstance(host_findings, (str, bytes)):
        lines.append(("HOST_OBSERVED FINDINGS:", 100, True))
        lines.extend((str(line), 100, False) for line in host_findings if str(line).strip())
    lines.append(("BRIEF CLAUSES WITH UNITS:" if not guarded
                  else "BRIEF CLAUSES (numbered/named; public-test coverage):", -1, True))
    if clauses:
        if guarded:
            lines.extend((
                f'C{index}: covered_by_public_tests: '
                f'{_public_test_coverage(clause, oracle, sources)}; "{clause}"',
                10, False) for index, clause in enumerate(clauses, 1))
        else:
            lines.extend((f'C{index}: "{clause}"', 10, False)
                         for index, clause in enumerate(clauses, 1))
    else:
        lines.append((_UNKNOWN, 0, True))
    return lines


def render_findings(frame: Mapping[str, object], cap: int = FINDINGS_TOKEN_CAP) -> list[str]:
    """Render the bounded host-only block, dropping lower-priority items first."""
    entries = _findings_lines(frame)
    selected = list(range(len(entries)))
    omitted = 0

    def render() -> str:
        values = [entries[index][0] for index in selected]
        if omitted:
            values.append(f"{omitted} more omitted")
        return "\n".join(values)

    while lexical_token_count(render()) > cap:
        candidates = [index for index in selected if not entries[index][2]]
        if not candidates:
            break
        drop = min(candidates, key=lambda index: (entries[index][1], -index))
        selected.remove(drop)
        omitted += 1
    while omitted and lexical_token_count(render()) > cap:
        candidates = [index for index in selected if not entries[index][2]]
        if not candidates:
            break
        drop = min(candidates, key=lambda index: (entries[index][1], -index))
        selected.remove(drop)
        omitted += 1
    return render().splitlines()


__all__ = [
    "FINDINGS_TOKEN_CAP", "extract_brief_clauses", "extract_unit_clauses", "failure_ledger_entries",
    "brief_clause_coverage", "lexical_token_count", "normalize_test_oracle", "render_failure_ledger",
    "render_findings",
]
