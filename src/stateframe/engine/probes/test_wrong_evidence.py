"""Validate a wrong-test claim against the original judging test."""
from __future__ import annotations

import ast
import hashlib
import pathlib
import re
from collections.abc import Mapping


def _python_body(source: str, name: str) -> str | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    parts = name.split("::")
    if len(parts) == 1:
        candidates = tree.body
    elif len(parts) == 2:
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == parts[0]]
        candidates = classes[0].body if classes else []
    else:
        return None
    function_name = re.sub(r"\[[^\]]*\]$", "", parts[-1])
    function = next((node for node in candidates
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and node.name == function_name and node.name.startswith("test")), None)
    if function is None:
        return None
    assertions = (node for statement in function.body for node in ast.walk(statement)
                  if isinstance(node, ast.Assert) or
                  (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr.startswith("assert")))
    return "\n".join(ast.get_source_segment(source, node) or "" for node in assertions)


def _named_body(source: str, name: str, suffix: str) -> str | None:
    if suffix in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
        pattern = rf"\b(?:test|it)\s*\(\s*['\"]{re.escape(name)}['\"]"
    elif suffix == ".go":
        pattern = rf"\bfunc\s+{re.escape(name)}\s*\("
    elif suffix == ".rs":
        pattern = rf"\bfn\s+{re.escape(name)}\s*\("
    else:
        return source if suffix == ".sh" else None  # a selected shell script is one judging check
    match = re.search(pattern, source)
    if match is None:
        return None
    opening = source.find("{", match.end())
    if opening < 0:
        return None
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1:index]
    return None


def validate_test_wrong(evidence: Mapping[str, object], row: Mapping[str, object],
                        root: pathlib.Path) -> str | None:
    """Return a refusal reason, or None for unchanged original judging evidence."""
    if (set(evidence) != {"path", "test", "assertion", "why", "observed"}
            or any(not isinstance(value, str) or not value.strip() or len(value) > 2000
                   for value in evidence.values())):
        return "test_wrong evidence refused: provide path, test, assertion, why, and observed"
    path = str(evidence["path"])
    relative = pathlib.PurePosixPath(path)
    selectors = row.get("test_wrong_selectors")
    sources = row.get("test_wrong_sources")
    hashes = row.get("test_wrong_hashes")
    if (relative.is_absolute() or ".." in relative.parts or
            not isinstance(selectors, Mapping) or path not in selectors or
            not isinstance(sources, Mapping) or path not in sources or
            not isinstance(hashes, Mapping) or path not in hashes):
        return "test_wrong evidence refused: path is not a judging test"
    target = root / path
    if not target.is_file() or target.is_symlink():
        return "test_wrong evidence refused: judging test file is missing or invalid"
    try:
        current_hash = hashlib.sha256(target.read_bytes()).hexdigest()
    except OSError:
        return "test_wrong evidence refused: cannot read judging test"
    if current_hash != hashes[path]:
        return "test_wrong evidence refused: judging test was modified during the run"
    name = str(evidence["test"])
    selected_names = selectors[path]
    failures = row.get("test_wrong_failed_nodes")
    failed_names = failures.get(path, ()) if isinstance(failures, Mapping) else ()
    failed_at_start = name in failed_names
    if not isinstance(selected_names, list) or (selected_names and name not in selected_names
                                                and not failed_at_start):
        return "test_wrong evidence refused: named test is outside the judging selector"
    keyword = row.get("test_wrong_keyword")
    mark = row.get("test_wrong_mark")
    if (keyword is not None and (not isinstance(keyword, str) or
                                 not re.fullmatch(r"[A-Za-z0-9_]+", keyword) or
                                 keyword.lower() not in name.lower()) and not failed_at_start):
        return "test_wrong evidence refused: named test is outside the -k judging selector"
    if mark is not None and not failed_at_start:
        return "test_wrong evidence refused: -m selection needs a named starting failure"
    source = sources[path]
    if not isinstance(source, str):
        return "test_wrong evidence refused: original judging test is unavailable"
    body = (_python_body(source, name) if relative.suffix == ".py"
            else _named_body(source, name, relative.suffix))
    if body is None or str(evidence["assertion"]) not in body:
        return "test_wrong evidence refused: quoted assertion is not inside the named test in the original file"
    return None
