"""Keep test measurement-journal readers behind the shared live/sealed reader."""
from __future__ import annotations

import ast
import json
from pathlib import Path

from _journal import measurement_events, measurement_journal_paths
from stateframe.engine.kernel_journal_store import seal_measurement_run

TESTS = Path(__file__).resolve().parent
READ_CALLS = {"open", "read_text", "read_bytes", "glob", "rglob"}
PRODUCT_READERS = {"journal_paths", "open_journal"}
ALLOWLIST = {
    "acceptance/test_resume_completion.py::_interrupted_run": (
        "wait_for_saved reads run_dir/checkpoints/events.jsonl live; this is the checkpoint "
        "event log, not a measurement journal (rel330b c8031fe)."
    ),
}


def _call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return call.func.id if isinstance(call.func, ast.Name) else ""


def _strings(node: ast.AST | None) -> list[str]:
    if node is None:
        return []
    return [child.value for child in ast.walk(node)
            if isinstance(child, ast.Constant) and isinstance(child.value, str)]


def _names(node: ast.AST | None) -> set[str]:
    if node is None:
        return set()
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def _assigned_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, (ast.AnnAssign, ast.For, ast.AsyncFor)):
        targets = [node.target]
    else:
        return set()
    return {child.id for target in targets for child in ast.walk(target)
            if isinstance(child, ast.Name)}


def _raw_measurement_reads(root: Path) -> list[str]:
    violations = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "_journal.py":
            continue
        source = path.read_text(encoding="utf-8", errors="replace")
        if "events.jsonl" not in source and "measurement-runs" not in source:
            continue
        tree = ast.parse(source, filename=str(path))
        parent = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

        def enclosing_function(node: ast.AST) -> ast.AST | None:
            current = parent.get(node)
            while current is not None:
                if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    return current
                current = parent.get(current)
            return None

        scopes: dict[ast.AST | None, list[ast.AST]] = {}
        for node in ast.walk(tree):
            scopes.setdefault(enclosing_function(node), []).append(node)
        relative = path.relative_to(root).as_posix()
        for scope, nodes in scopes.items():
            key = f"{relative}::{scope.name}" if scope is not None else relative
            event_names: set[str] = set()
            checkpoint_names: set[str] = set()
            measurement_names: set[str] = set()
            changed = True
            while changed:
                changed = False
                for node in nodes:
                    if isinstance(node, (ast.Assign, ast.AnnAssign)):
                        value = node.value
                    elif isinstance(node, (ast.For, ast.AsyncFor)):
                        value = node.iter
                    else:
                        continue
                    target_names = _assigned_names(node)
                    referenced = _names(value)
                    values = _strings(value)
                    if (any("events.jsonl" in item for item in values)
                            or referenced & event_names):
                        before = len(event_names)
                        event_names.update(target_names)
                        changed |= len(event_names) != before
                    if (any("checkpoints" in item for item in values)
                            or referenced & checkpoint_names):
                        before = len(checkpoint_names)
                        checkpoint_names.update(target_names)
                        changed |= len(checkpoint_names) != before
                    if (any("measurement-runs" in item or item == "worker" for item in values)
                            or referenced & measurement_names):
                        before = len(measurement_names)
                        measurement_names.update(target_names)
                        changed |= len(measurement_names) != before

            for call in (node for node in nodes if isinstance(node, ast.Call)):
                name = _call_name(call)
                expressions = [call.func.value] if isinstance(call.func, ast.Attribute) else []
                expressions.extend(call.args)
                expressions.extend(keyword.value for keyword in call.keywords)
                strings = [value for expression in expressions for value in _strings(expression)]
                referenced = {value for expression in expressions for value in _names(expression)}
                event_path = (any("events.jsonl" in value for value in strings)
                              or bool(referenced & event_names))
                checkpoint_path = (any("checkpoints" in value for value in strings)
                                   or bool(referenced & checkpoint_names))
                measurement_path = (
                    any("measurement-runs" in value or value == "worker" for value in strings)
                    or bool(referenced & measurement_names)
                )
                if key in ALLOWLIST and event_path:
                    # The one live checkpoint-log read is documented above; it is not a measurement journal.
                    continue
                if checkpoint_path:
                    continue
                raw_path_access = name in READ_CALLS and event_path
                inline_product_reader = name in PRODUCT_READERS and measurement_path
                if raw_path_access or inline_product_reader:
                    violations.append(f"{relative}:{call.lineno}")
    return sorted(set(violations))

def test_measurement_reader_handles_live_and_sealed_journals(tmp_path):
    measurement_root = tmp_path / "worker" / "measurement-runs"
    live = measurement_root / "live"
    sealed = measurement_root / "sealed"
    live.mkdir(parents=True)
    sealed.mkdir(parents=True)
    live_events = [{"kind": "live_event"}, {"kind": "live_event_two"}]
    sealed_events = [{"kind": "sealed_event"}]
    (live / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in live_events),
        encoding="utf-8",
    )
    (sealed / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in sealed_events),
        encoding="utf-8",
    )
    seal_measurement_run(sealed)

    assert [path.relative_to(measurement_root).as_posix()
            for path in measurement_journal_paths(tmp_path)] == [
        "live/events.jsonl", "sealed/events.jsonl.xz",
    ]
    assert measurement_events(tmp_path) == [*live_events, *sealed_events]


def test_test_sources_use_the_shared_measurement_reader():
    violations = _raw_measurement_reads(TESTS)
    assert violations == [], f"read measurement journals through tests/_journal.py: {violations}"
    assert len(ALLOWLIST) == 1
    assert next(iter(ALLOWLIST.values()))


def test_planted_raw_events_glob_is_caught(tmp_path):
    source = tmp_path / "planted.py"
    source.write_text(
        'from pathlib import Path\n'
        'def test_raw_reader():\n'
        '    Path("run/worker/measurement-runs").glob("*/events.jsonl")\n',
        encoding="utf-8",
    )
    assert _raw_measurement_reads(tmp_path) == ["planted.py:3"]
