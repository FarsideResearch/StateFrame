"""Keep new request-building exception handlers from silently dropping evidence."""
from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Mapping, TypedDict, cast


ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(__file__).with_name("silent_swallow_guard.json")
BROAD_ERRORS = {
    "Exception", "BaseException", "ImportError", "OSError", "TypeError",
    "ValueError", "KeyError", "AttributeError", "RuntimeError", "IndexError",
}
DEFAULT_NAME = re.compile(r"_?(?:default|fallback)(?:_[a-z0-9_]+)?\Z")
EVIDENCE_CALL = re.compile(r"(?:journal|record|emit|log|_degrade)", re.IGNORECASE)


class GuardEntry(TypedDict):
    file: str
    function: str
    except_clause: str
    ordinal: int
    reason: str
    risk: str


class GuardConfig(TypedDict):
    module_globs: list[str]
    required_modules: list[str]
    allowlist: list[GuardEntry]


@dataclass(frozen=True)
class Finding:
    file: str
    function: str
    except_clause: str
    ordinal: int
    line: int

    @property
    def key(self) -> tuple[str, str, str, int]:
        return self.file, self.function, self.except_clause, self.ordinal


def _broad_handler(node: ast.ExceptHandler) -> bool:
    if node.type is None:
        return True
    types = node.type.elts if isinstance(node.type, ast.Tuple) else [node.type]
    return any(isinstance(item, ast.Name) and item.id in BROAD_ERRORS for item in types)


def _quiet_return(value: ast.expr | None) -> bool:
    if value is None:
        return True
    if isinstance(value, ast.Constant):
        return True
    if isinstance(value, ast.UnaryOp) and isinstance(value.op, (ast.UAdd, ast.USub)):
        return (
            isinstance(value.operand, ast.Constant)
            and isinstance(value.operand.value, (int, float))
        )
    if isinstance(value, (ast.List, ast.Tuple, ast.Dict, ast.Set)):
        return not getattr(value, "elts", None) and not getattr(value, "keys", None)
    return isinstance(value, ast.Name) and bool(DEFAULT_NAME.fullmatch(value.id))


def _fail_open(node: ast.ExceptHandler) -> bool:
    # Check the whole subtree before classifying the permitted no-op statements.
    # A nested journal or raise must never be mistaken for a silent fallback.
    for child in ast.walk(node):
        if isinstance(child, ast.Raise):
            return False
        if isinstance(child, ast.Call):
            name = child.func.attr if isinstance(child.func, ast.Attribute) else (
                child.func.id if isinstance(child.func, ast.Name) else ""
            )
            if EVIDENCE_CALL.search(name):
                return False
    return bool(node.body) and all(
        isinstance(stmt, (ast.Pass, ast.Continue))
        or isinstance(stmt, ast.Return) and _quiet_return(stmt.value)
        for stmt in node.body
    )


def scan_source(source: str, relpath: str) -> list[Finding]:
    """Return fail-open handlers with location-independent baseline identities."""
    tree = ast.parse(source, filename=relpath)
    findings: list[Finding] = []
    scope: list[str] = []
    ordinals: Counter[str] = Counter()

    class Visitor(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            scope.append(node.name)
            self.generic_visit(node)
            scope.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            scope.append(node.name)
            self.generic_visit(node)
            scope.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
            function = ".".join(scope) or "<module>"
            ordinals[function] += 1
            if _broad_handler(node) and _fail_open(node):
                clause = "except:" if node.type is None else f"except {ast.unparse(node.type)}:"
                if node.name:
                    clause = clause[:-1] + f" as {node.name}:"
                findings.append(Finding(relpath, function, clause, ordinals[function], node.lineno))
            self.generic_visit(node)

    Visitor().visit(tree)
    return findings


def load_config(path: Path = CONFIG) -> GuardConfig:
    return cast(GuardConfig, json.loads(path.read_text(encoding="utf-8")))


def guard_sources(
    root: Path,
    config: GuardConfig,
    overrides: Mapping[str, Path] | None = None,
) -> list[str]:
    """Report missing modules, newly silent handlers, and obsolete baseline debt."""
    problems: list[str] = []
    paths: dict[str, Path] = {}
    for pattern in config["module_globs"]:
        matches = sorted(root.glob(pattern))
        if not matches and not any(char in pattern for char in "*?["):
            problems.append(f"missing guarded module: {pattern}")
        for path in matches:
            paths[path.relative_to(root).as_posix()] = path
    for relpath in config["required_modules"]:
        if relpath not in paths:
            problems.append(f"missing guarded module: {relpath}")

    expected: dict[tuple[str, str, str, int], GuardEntry] = {}
    for entry in config["allowlist"]:
        key = (entry["file"], entry["function"], entry["except_clause"], entry["ordinal"])
        if key in expected:
            problems.append(f"duplicate allowlist entry: {key}")
        expected[key] = entry
        if entry["file"] not in paths and entry["file"] not in config["required_modules"]:
            problems.append(f"allowlist file outside guarded scope: {entry['file']}")
        if not entry["reason"] or not entry["risk"]:
            problems.append(f"incomplete allowlist entry: {key}")

    found: set[tuple[str, str, str, int]] = set()
    for relpath, path in sorted(paths.items()):
        source = (overrides or {}).get(relpath, path).read_text(encoding="utf-8")
        for finding in scan_source(source, relpath):
            if finding.key in found:
                problems.append(f"duplicate finding: {finding.key}")
            found.add(finding.key)
            if finding.key not in expected:
                problems.append(
                    f"new silent swallow: {finding.key} (line {finding.line}); "
                    "fix: fail loud with an explicit error or evidence, or add a "
                    "justified allowlist row to tests/silent_swallow_guard.json"
                )

    for key in sorted(expected.keys() - found):
        problems.append(
            "stale allowlist entry: delete this exact row from "
            f"tests/silent_swallow_guard.json: {json.dumps(expected[key], sort_keys=True)}"
        )
    return problems


def test_no_silent_swallow() -> None:
    assert not (problems := guard_sources(ROOT, load_config())), "\n".join(problems)


def test_planted_swallow_is_caught(tmp_path: Path) -> None:
    config = load_config()
    relpath = "src/stateframe/engine/kernel_request_auth_identity.py"
    original = ROOT / relpath
    scratch = tmp_path / original.name
    scratch.write_text(
        original.read_text(encoding="utf-8")
        + '\n\ndef _planted_silent_swallow():\n'
        + '    try:\n        x()\n    except Exception:\n        return ""\n',
        encoding="utf-8",
    )
    problems = guard_sources(ROOT, config, overrides={relpath: scratch})
    assert any("new silent swallow" in problem and "_planted_silent_swallow" in problem
               and "fix: fail loud" in problem and "justified allowlist row" in problem
               for problem in problems), problems


def test_scanner_distinguishes_silent_fallbacks_from_evidence() -> None:
    source = '''
def broad(default_value):
    try:
        x()
    except (TypeError, ValueError):
        return default_value

def bare():
    try:
        x()
    except:
        pass

def recorded():
    try:
        x()
    except Exception:
        log.warning("failed")
        return None

def raised():
    try:
        x()
    except BaseException:
        raise
'''
    findings = scan_source(source, "guarded.py")
    assert [(item.function, item.except_clause) for item in findings] == [
        ("broad", "except (TypeError, ValueError):"),
        ("bare", "except:"),
    ]


def test_stale_allowlist_and_missing_module_fail(tmp_path: Path) -> None:
    relpath = "src/stateframe/engine/sample.py"
    module = tmp_path / relpath
    module.parent.mkdir(parents=True)
    module.write_text(
        'def sample():\n    try:\n        x()\n    except Exception:\n        return ""\n',
        encoding="utf-8",
    )
    entry: GuardEntry = {
        "file": relpath,
        "function": "sample",
        "except_clause": "except Exception:",
        "ordinal": 1,
        "reason": "baseline debt 2026-11-05, see inventory",
        "risk": "med",
    }
    config: GuardConfig = {
        "module_globs": [relpath],
        "required_modules": [relpath],
        "allowlist": [entry],
    }
    assert guard_sources(tmp_path, config) == []
    stale = {**config, "allowlist": [{**entry, "ordinal": 2}]}
    assert any("stale allowlist entry" in problem and json.dumps(stale["allowlist"][0], sort_keys=True) in problem
               for problem in guard_sources(tmp_path, stale))
    assert any("missing guarded module" in problem
               for problem in guard_sources(tmp_path / "absent", config))


def test_removed_baseline_body_reports_exact_row(tmp_path: Path) -> None:
    relpath = "src/stateframe/engine/kernel_completion_status.py"
    original = ROOT / relpath
    source = original.read_text(encoding="utf-8")
    old_body = "        except Exception:\n            return None"
    assert source.count(old_body) == 1
    scratch = tmp_path / original.name
    scratch.write_text(source.replace(old_body, "        except Exception:\n            raise"), encoding="utf-8")

    config = load_config()
    entry = next(row for row in config["allowlist"] if row["file"] == relpath)
    problems = guard_sources(ROOT, config, overrides={relpath: scratch})
    assert problems == [
        "stale allowlist entry: delete this exact row from "
        f"tests/silent_swallow_guard.json: {json.dumps(entry, sort_keys=True)}"
    ]
