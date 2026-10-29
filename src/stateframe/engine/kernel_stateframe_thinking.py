"""StateFrame thinking views: structural shape, host timelines, and one fork.

This module is deliberately host-only.  It does not decide whether a brief is
correct, and it never turns a solver or a candidate result into a completion
verdict.  ``STATEFRAME_THINK=off`` is the native path: callers do not create a
``ThinkingLayer`` there and the payload helper is a no-op.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from kernel_test_python import test_python

MODE_ENV = "STATEFRAME_THINK"
VALID_MODES = {"off", "shadow", "assisted"}
SHAPE_MAX_TOKENS = 450
TIMELINE_MAX_TOKENS = 400
MAX_TIMELINE_STEPS = 16
MAX_FORK_SEQUENCES = 32
TRIGGERS = {"after_edit", "after_red", "before_done", "repetition", "manual"}
_GREEN = {"OK", "GREEN", "PASS", "PASSED", "SUCCESS", "APPLIED"}
_EDIT_KINDS = {"edit", "str_replace", "file_create"}


def thinking_mode() -> str:
    """Return the strict three-arm switch."""
    value = os.environ.get(MODE_ENV, "off")
    if value not in VALID_MODES:
        raise ValueError(f"{MODE_ENV} must be off, shadow or assisted")
    return value


def enabled() -> bool:
    return thinking_mode() != "off"


def _revision(source: str) -> str:
    return "sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest()


def _short(value: object, limit: int = 88) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _node_name(node: ast.AST) -> str:
    return str(getattr(node, "name", "<anonymous>"))


def _stub_body(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    body = list(node.body)
    if len(body) == 1 and isinstance(body[0], ast.Pass):
        return True
    if len(body) == 1 and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        return body[0].value.value is Ellipsis
    for item in body:
        if isinstance(item, ast.Raise):
            value = item.exc.func if isinstance(item.exc, ast.Call) else item.exc
            if isinstance(value, ast.Name) and value.id == "NotImplementedError":
                return True
    return False


def _ast_components(source: str) -> tuple[list[dict], list[str], list[str]]:
    """Collect every class/function/method, duplicates, and stub bodies."""
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return [{"name": "<syntax-error>", "kind": "module", "line": 0,
                 "detail": _short(error)}], [], []
    components: list[dict] = []
    duplicates: list[str] = []
    stubs: list[str] = []

    def visit(body: Sequence[ast.stmt], prefix: str = "") -> None:
        defs: list[ast.AST] = [node for node in body
                               if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
        counts: dict[str, int] = {}
        for node in defs:
            counts[_node_name(node)] = counts.get(_node_name(node), 0) + 1
        for node in defs:
            name = prefix + _node_name(node)
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            row = {"name": name, "kind": kind, "line": int(getattr(node, "lineno", 0)),
                   "end_line": int(getattr(node, "end_lineno", getattr(node, "lineno", 0))),
                   "stub_body": bool(kind == "function" and _stub_body(node))}
            components.append(row)
            if counts[_node_name(node)] > 1:
                duplicates.append(name)
            if row["stub_body"]:
                stubs.append(name)
            if isinstance(node, ast.ClassDef):
                visit(node.body, name + ".")
            else:
                visit(node.body, name + ".")

    visit(tree.body)
    return components, sorted(set(duplicates)), sorted(set(stubs))


_IMPORT_PROBE = r'''
import ast, contextlib, importlib.util, inspect, json, pathlib, sys, traceback
payload = json.load(sys.stdin)
path = pathlib.Path(payload["path"]).resolve()
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("_stateframe_fresh_target", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
result = {"status": "HOST_OBSERVED", "bindings": [], "error": None}

def _class_method_lines(value):
    lines = []
    for member in vars(value).values():
        if isinstance(member, (staticmethod, classmethod)):
            member = member.__func__
        elif isinstance(member, property):
            member = member.fget
        code = getattr(member, "__code__", None)
        if code is not None:
            lines.append(int(code.co_firstlineno))
    return lines

def _name_is_rebound(node, name):
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        target_names = {item.id for target in targets for item in ast.walk(target)
                        if isinstance(item, ast.Name)}
        value_names = {item.id for item in ast.walk(getattr(node, "value", node))
                       if isinstance(item, ast.Name)}
        return name in target_names or name in value_names
    if isinstance(node, ast.NamedExpr):
        return node.target.id == name or any(item.id == name for item in ast.walk(node.value)
                                             if isinstance(item, ast.Name))
    if isinstance(node, ast.Delete):
        return any(item.id == name for item in ast.walk(node)
                   if isinstance(item, ast.Name))
    return False

def _module_has_conditional_or_aliasing(tree, name):
    flow_nodes = tuple(node for node in
                       (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try,
                        getattr(ast, "Match", None)) if node is not None)
    for statement in tree.body:
        if isinstance(statement, flow_nodes):
            return True
        if isinstance(statement, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if getattr(statement, "name", None) == name and not isinstance(statement, ast.ClassDef):
                return True
            continue
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            if any((alias.asname or alias.name.split(".")[0]) == name
                   for alias in statement.names):
                return True
        elif _name_is_rebound(statement, name):
            return True
    return False

def _legacy_class_line(value, source_path, name):
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeError):
        return None
    candidates = [node for node in ast.walk(tree)
                  if isinstance(node, ast.ClassDef) and node.name == name]
    method_lines = _class_method_lines(value)
    if method_lines:
        matches = [node for node in candidates
                   if all(int(getattr(node, "lineno", 0)) <= line <=
                          int(getattr(node, "end_lineno", getattr(node, "lineno", 0)))
                          for line in method_lines)]
        return int(matches[0].lineno) if len(matches) == 1 else None
    top_level = [node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == name]
    if not top_level or _module_has_conditional_or_aliasing(tree, name):
        return None
    return int(top_level[-1].lineno)

def _binding_line(value, source_path, name):
    if inspect.isfunction(value):
        code = getattr(value, "__code__", None)
        return int(code.co_firstlineno) if code is not None else None
    if inspect.isclass(value):
        first = None if payload.get("mask_firstlineno") else getattr(value, "__firstlineno__", None)
        if first is not None:
            return int(first)
        return _legacy_class_line(value, source_path, name)
    try:
        return int(inspect.getsourcelines(value)[1])
    except (OSError, TypeError, IOError):
        return None

try:
    with contextlib.redirect_stdout(sys.stderr), contextlib.redirect_stderr(sys.stderr):
        spec.loader.exec_module(module)
    for name in sorted(key for key in vars(module) if not key.startswith("_")
                       and (inspect.isfunction(vars(module)[key])
                            or inspect.isclass(vars(module)[key]))):
        value = vars(module)[name]
        try:
            file_name = inspect.getsourcefile(value) or inspect.getfile(value)
            binding = str(pathlib.Path(file_name).resolve())
            if binding == str(path):
                binding = payload["display_path"]
            line = _binding_line(value, pathlib.Path(file_name).resolve(), name)
            location = f"{binding}:{line}" if line is not None else "UNKNOWN"
        except (OSError, TypeError, IOError):
            location = "<no-source-location>"
        result["bindings"].append({"name": name, "object": type(value).__name__,
                                   "location": location})
except BaseException as error:
    result["status"] = "UNKNOWN"
    result["error"] = f"{type(error).__name__}: {error}"
print(json.dumps(result, sort_keys=True))
'''


def fresh_isolated_import(source: str, path: str = "candidate.py", *, timeout_s: float = 3.0,
                          simulate_missing_firstlineno: bool = False) -> dict:
    """Import a disposable copy and report the live public function/class bindings.

    The product path is never imported.  A fresh interpreter and a temporary
    copy make import-time mutations and bytecode writes non-product effects.
    """
    display_path = str(path)
    with tempfile.TemporaryDirectory(prefix="stateframe-think-import-") as root:
        target = Path(root) / Path(path).name
        target.write_text(source, encoding="utf-8")
        payload = {"path": str(target), "display_path": display_path,
                   "mask_firstlineno": bool(simulate_missing_firstlineno)}
        try:
            from kernel_execution_policy import ExecutionPolicyError, run_sandboxed
            execution = run_sandboxed(
                [test_python(), "-I", "-c", _IMPORT_PROBE],
                root,
                input_data=json.dumps(payload),
                timeout=max(0.05, float(timeout_s)),
                output_limit=128 * 1024,
                readonly_workspace=True,
                cpu_seconds=min(30.0, max(1.0, float(timeout_s))),
            )
        except ExecutionPolicyError as error:
            return {"status": "UNKNOWN", "bindings": [],
                    "error": f"sandbox unavailable: {error}"}
        if execution.timed_out:
            return {"status": "UNKNOWN", "bindings": [], "error": "timeout"}
        completed = execution.completed
        output = completed.stdout.strip().splitlines()
        try:
            result = json.loads(output[-1])
        except (IndexError, json.JSONDecodeError):
            result = {"status": "UNKNOWN", "bindings": [],
                      "error": _short(completed.stderr or "fresh import produced no receipt")}
        result["returncode"] = completed.returncode
        result["sandbox"] = execution.capabilities.as_dict()
        return result


def _brief_requirements(brief: str, path: str) -> list[dict]:
    try:
        from kernel_stateframe_v2_s1 import _brief_requirements as s1_requirements
        return s1_requirements(brief, path)
    except (ImportError, AttributeError, TypeError):
        return []


def _receipt_revision(receipt: Mapping[str, object]) -> set[str]:
    values: set[str] = set()
    for key in ("current_revision", "after_revision", "product_revision", "current_bytes_sha256"):
        value = receipt.get(key)
        if isinstance(value, str):
            values.add(value)
    return values


def _evidence_rows(brief: str, path: str, source: str, components: list[dict],
                   duplicates: list[str], stubs: list[str], receipts: Sequence[object]) -> list[dict]:
    revision = _revision(source)
    names = {row["name"] for row in components}
    requirements = _brief_requirements(brief, path)
    rows: list[dict] = []
    for requirement in requirements:
        owner = requirement.get("owner")
        qualified = requirement["name"] if owner is None else f"{owner}.{requirement['name']}"
        exact = qualified in names
        fallback = requirement["name"] in names
        found = exact or fallback
        stale = False
        current_pass = False
        executed = False
        evidence: list[str] = []
        for raw in receipts:
            if not isinstance(raw, Mapping):
                continue
            target = str(raw.get("target_path", raw.get("path", path))).lstrip("./")
            if target != path.lstrip("./"):
                continue
            result = str(raw.get("result_class", raw.get("result", raw.get("status", "")))).upper()
            if result in _GREEN:
                executed = True
                receipt_id = str(raw.get("receipt_id", raw.get("event_id", "receipt")))
                evidence.append(receipt_id)
                if _receipt_revision(raw) and revision not in _receipt_revision(raw) \
                        and revision.removeprefix("sha256:") not in _receipt_revision(raw):
                    stale = True
                elif _receipt_revision(raw):
                    current_pass = True
        if not found:
            status = "FAIL_WITH_WITNESS"
            evidence.append("missing_public_binding")
        elif fallback and not exact and owner is not None:
            status = "INTERPRETATION_OPEN"
        elif qualified in stubs or requirement["name"] in stubs or qualified in duplicates:
            status = "FAIL_WITH_WITNESS"
        elif stale:
            status = "STALE"
        elif current_pass:
            status = "PASS_ON_LISTED_CASES"
        elif executed:
            status = "EXECUTED_WITHOUT_MAPPED_ASSERTION"
        else:
            status = "UNEXERCISED"
        rows.append({"clause": requirement.get("id"), "component": qualified,
                     "status": status, "evidence": evidence[:3],
                     "witness": ("missing_or_shadowed_binding" if status == "FAIL_WITH_WITNESS" else None)})
    return rows


def build_shape_view(*, brief: str, source: str, path: str = "candidate.py",
                     receipts: Sequence[object] = (), import_timeout_s: float = 3.0) -> dict:
    """Build the complete 3D structure/evidence view for one source module."""
    components, duplicates, stubs = _ast_components(source)
    live = fresh_isolated_import(source, path, timeout_s=import_timeout_s)
    rows = _evidence_rows(brief, path, source, components, duplicates, stubs, receipts)
    mapped = {row["component"] for row in rows}
    live_bindings = {
        str(row.get("name")): dict(row) for row in live.get("bindings", ())
        if isinstance(row, Mapping) and row.get("name")
    }
    complete = len(components) == 0 or bool(live.get("bindings")) or live.get("status") == "UNKNOWN"
    live_witness = []
    if any(str(name).startswith("QueueStats") for name in stubs) or "QueueStats" in duplicates:
        try:
            from kernel_jq_finish import live_binding_witness
            live_witness = live_binding_witness(source, path)
        except (ImportError, TypeError, ValueError):
            live_witness = []
    return {
        "schema": "stateframe-think/3d/1", "authority": "HOST_OBSERVED", "path": path,
        "revision": _revision(source), "ast_components": components,
        "duplicates": duplicates, "stubs": stubs, "live_import": live,
        "live_bindings": live_bindings,
        "live_binding_witness": live_witness,
        "evidence_rows": rows,
        "completeness": {
            "all_ast_components": len(components), "mapped_requirements": len(mapped),
            "requirements": len(rows), "statement": (
                "COMPLETE_FOR_AST_COMPONENTS_AND_S1_ROWS" if complete
                else "INCOMPLETE_LIVE_BINDING_RECEIPT")},
    }


def shape_view(*args, **kwargs) -> dict:
    """Compatibility spelling for callers that call the service by noun."""
    return build_shape_view(*args, **kwargs)


def render_shape(view: Mapping[str, object], *, max_tokens: int = SHAPE_MAX_TOKENS) -> str:
    lines = ["[StateFrame THINK 3D | HOST_OBSERVED]",
             f"MODULE {view.get('path')} rev={str(view.get('revision', ''))[:18]}"]
    components = view.get("ast_components", ())
    lines.append(f"AST components={len(components) if isinstance(components, Sequence) else 0} "
                 f"duplicates={','.join(view.get('duplicates', ()) or ()) or '-'} "
                 f"stubs={','.join(view.get('stubs', ()) or ()) or '-'}")
    live = view.get("live_import", {})
    bindings = live.get("bindings", ()) if isinstance(live, Mapping) else ()
    for binding in list(bindings)[:24]:
        if isinstance(binding, Mapping):
            lines.append(f"LIVE {binding.get('name')} -> {binding.get('object')} @ {binding.get('location')}")
    for row in list(view.get("evidence_rows", ()) or ())[:32]:
        lines.append(f"EVIDENCE {row.get('clause')} x {row.get('component')} "
                     f"{row.get('status')} refs={','.join(row.get('evidence', ()) or ()) or '-'}")
    # JobQueue completion uses the same fresh isolated import as the existing 3D view. The
    # witness is host-observed and only appears when the public QueueStats binding
    # is actually duplicated or stubbed; it never infers behaviour from source text.
    try:
        if any(str(name).startswith("QueueStats") for name in view.get("stubs", ()) or ()) \
                or "QueueStats" in (view.get("duplicates", ()) or ()):
            for witness in view.get("live_binding_witness", ()):
                lines.append("WITNESS " + witness)
    except (ImportError, TypeError, ValueError):
        pass
    completeness = view.get("completeness", {})
    lines.append("COMPLETENESS " + _short(completeness.get("statement", "UNKNOWN"), 120)
                 if isinstance(completeness, Mapping) else "COMPLETENESS UNKNOWN")
    text = "\n".join(lines)
    # The renderer uses a conservative character proxy, matching local receipt
    # budgets without importing the broader context renderer.
    limit = max(800, int(max_tokens) * 4)
    return text if len(text) <= limit else text[: max(0, limit - 14)] + "\n[TRUNCATED]"


_TIMELINE_WORKER = r'''
import builtins, contextlib, importlib.util, io, json, os, pathlib, sys, time as _time, types
payload = json.load(sys.stdin)
root = pathlib.Path.cwd().resolve()
clock = {"value": float(payload.get("clock_start", 0.0))}
_real_open = builtins.open
_real_io_open = io.open
_real_os_open = os.open
def _sandbox_path(value):
    try: return pathlib.Path(value).resolve()
    except (OSError, TypeError, ValueError): return root / "<invalid>"
def _write_mode(args, kwargs):
    mode = kwargs.get("mode", args[0] if args and isinstance(args[0], str) else "r")
    return any(flag in str(mode) for flag in ("w", "a", "x", "+"))
def _guarded_open(file, *args, **kwargs):
    if _write_mode(args, kwargs) and not _sandbox_path(file).is_relative_to(root):
        raise PermissionError("StateFrame timeline sandbox blocks product-file writes")
    return _real_open(file, *args, **kwargs)
def _guarded_io_open(file, *args, **kwargs):
    if _write_mode(args, kwargs) and not _sandbox_path(file).is_relative_to(root):
        raise PermissionError("StateFrame timeline sandbox blocks product-file writes")
    return _real_io_open(file, *args, **kwargs)
def _guarded_os_open(file, *args, **kwargs):
    flags = int(args[0]) if args else int(kwargs.get("flags", 0))
    writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC))
    if writing and not _sandbox_path(file).is_relative_to(root):
        raise PermissionError("StateFrame timeline sandbox blocks product-file writes")
    return _real_os_open(file, *args, **kwargs)
builtins.open = _guarded_open
io.open = _guarded_io_open
os.open = _guarded_os_open
def monotonic(): return clock["value"]
def walltime(): return clock["value"]
_time.monotonic = monotonic
_time.time = walltime
sys.dont_write_bytecode = True
def plain(value, depth=0):
    if depth > 3: return "<depth>"
    if value is None or isinstance(value, (bool, int, float, str)): return value
    if isinstance(value, (list, tuple)): return [plain(v, depth + 1) for v in value[:24]]
    if isinstance(value, dict):
        return {str(k): plain(value[k], depth + 1) for k in sorted(value, key=str)[:24]}
    data = getattr(value, "__dict__", None)
    if isinstance(data, dict):
        return {str(k): plain(v, depth + 1) for k, v in sorted(data.items()) if not str(k).startswith("_")}
    return {"type": type(value).__name__, "repr": repr(value)[:120]}
def resolve(value, objects):
    if isinstance(value, dict) and set(value) == {"$ref"}: return objects[value["$ref"]]
    if isinstance(value, list): return [resolve(v, objects) for v in value]
    if isinstance(value, dict): return {k: resolve(v, objects) for k, v in value.items()}
    return value
def snapshot(module, objects):
    return {"objects": {name: plain(value) for name, value in sorted(objects.items())},
            "module": {name: plain(value) for name, value in sorted(vars(module).items())
                       if not name.startswith("_") and not callable(value)
                       and not isinstance(value, (type, types.ModuleType))}}
result = {"status": "HOST_OBSERVED", "clock": payload.get("clock_used", {}), "trace": [], "error": None}
try:
    spec = importlib.util.spec_from_file_location("_stateframe_timeline_target", payload["path"])
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(sys.stderr): spec.loader.exec_module(module)
    objects = {}
    for index, raw in enumerate(payload.get("sequence", [])):
        action = dict(raw) if isinstance(raw, dict) else {"call": str(raw)}
        if "clock" in action: clock["value"] = float(action["clock"])
        elif "dt" in action: clock["value"] += float(action["dt"])
        before = snapshot(module, objects)
        result_value = None
        error = None
        try:
            args = resolve(action.get("args", []), objects)
            kwargs = resolve(action.get("kwargs", {}), objects)
            receiver = action.get("receiver", action.get("object"))
            if receiver is not None:
                target = getattr(objects[receiver], action.get("method", action.get("call")))
                result_value = target(*args, **kwargs)
            else:
                name = action.get("call", action.get("function", action.get("target")))
                target = getattr(module, name)
                result_value = target(*args, **kwargs)
            if action.get("assign"):
                objects[str(action["assign"])] = result_value
            result_record = {"status": "OK", "value": plain(result_value)}
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            result_record = {"status": "EXCEPTION", "error": error}
        after = snapshot(module, objects)
        result["trace"].append({"index": index, "state_before": before, "action": action,
                                 "result": result_record, "state_after": after})
except BaseException as exc:
    result["status"] = "UNKNOWN"
    result["error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(result, sort_keys=True, default=str))
'''


def _clock_usage(source: str) -> dict[str, bool]:
    return {"monotonic": bool(re.search(r"(?:time\s*\.\s*monotonic|from\s+time\s+import\s+monotonic)", source)),
            "time": bool(re.search(r"(?:time\s*\.\s*time|from\s+time\s+import\s+time)", source))}


def _run_timeline(source: str, path: str, sequence: Sequence[object], *, timeout_s: float = 4.0,
                  clock_start: float = 0.0) -> dict:
    """Execute a call sequence in a disposable subprocess."""
    usage = _clock_usage(source)
    with tempfile.TemporaryDirectory(prefix="stateframe-think-timeline-") as root:
        target = Path(root) / Path(path).name
        target.write_text(source, encoding="utf-8")
        payload = {"path": str(target), "sequence": list(sequence)[:MAX_TIMELINE_STEPS], "clock_start": clock_start,
                   "clock_used": usage}
        try:
            from kernel_execution_policy import ExecutionPolicyError, run_sandboxed
            execution = run_sandboxed(
                [test_python(), "-I", "-c", _TIMELINE_WORKER],
                root,
                input_data=json.dumps(payload),
                timeout=max(0.05, float(timeout_s)),
                output_limit=128 * 1024,
                readonly_workspace=True,
                cpu_seconds=min(30.0, max(1.0, float(timeout_s))),
            )
        except ExecutionPolicyError as error:
            return {"status": "UNKNOWN", "authority": "HOST_OBSERVED", "trace": [],
                    "clock_used": usage, "error": f"sandbox unavailable: {error}"}
        if execution.timed_out:
            return {"status": "UNKNOWN", "authority": "HOST_OBSERVED", "trace": [],
                    "clock_used": usage, "error": "timeout"}
        completed = execution.completed
        lines = completed.stdout.strip().splitlines()
        try:
            result = json.loads(lines[-1])
        except (IndexError, json.JSONDecodeError):
            result = {"status": "UNKNOWN", "trace": [],
                      "error": _short(completed.stderr or "timeline produced no receipt")}
        result["authority"] = "HOST_OBSERVED"
        result["clock_used"] = usage
        result["returncode"] = completed.returncode
        result["sandbox"] = execution.capabilities.as_dict()
        return result


def _reference_is_visible(reference: object, reference_source: str | None) -> bool:
    return reference is not None and reference_source in {"brief", "public_tests", "visible_brief", "visible_tests"}


def _result_projection(row: Mapping[str, object]) -> object:
    result = row.get("result", {})
    return result.get("value", result.get("status")) if isinstance(result, Mapping) else result


def _first_divergence(observed: Sequence[object], reference: Sequence[object]) -> dict | None:
    for index, row in enumerate(observed):
        expected = reference[index] if index < len(reference) else {"status": "MISSING"}
        actual_record = row.get("result", {}) if isinstance(row, Mapping) else row
        actual = _result_projection(row) if isinstance(row, Mapping) else row
        if isinstance(expected, Mapping):
            if "status" in expected:
                actual = actual_record.get("status") if isinstance(actual_record, Mapping) else actual
                wanted = expected["status"]
            else:
                wanted = expected.get("value", expected.get("result"))
        else:
            wanted = expected
        if actual != wanted:
            return {"index": index, "observed": actual, "reference": wanted,
                    "witness": row.get("action") if isinstance(row, Mapping) else None}
    if len(observed) != len(reference):
        return {"index": min(len(observed), len(reference)), "observed": "MISSING" if len(observed) < len(reference) else "EXTRA",
                "reference": reference[min(len(observed), len(reference))] if len(observed) < len(reference) else "END"}
    return None


def run_timeline(*, source: str, path: str = "candidate.py", sequence: Sequence[object],
                 reference: object = None, reference_source: str | None = None,
                 timeout_s: float = 4.0, clock_start: float = 0.0) -> dict:
    """Run a host timeline and compare only to a visible, supplied reference."""
    observed = _run_timeline(source, path, sequence, timeout_s=timeout_s, clock_start=clock_start)
    result = {"schema": "stateframe-think/4d/1", "authority": "HOST_OBSERVED",
              "clock_used": observed.get("clock_used", {}), "trace": observed.get("trace", []),
              "status": observed.get("status", "UNKNOWN"), "error": observed.get("error")}
    if result["status"] == "UNKNOWN":
        result["comparison"] = {"label": "HOST_OBSERVED", "verdict": "UNKNOWN"}
    elif _reference_is_visible(reference, reference_source):
        reference_rows = reference if isinstance(reference, Sequence) and not isinstance(reference, (str, bytes)) else []
        divergence = _first_divergence(result["trace"], reference_rows)
        result["reference"] = list(reference_rows)
        result["comparison"] = {"label": "HOST_COMPUTED(under assumptions)",
                                 "verdict": "FIRST_DIVERGENCE" if divergence else "NO_DIVERGENCE_IN_REFERENCE",
                                 "first_divergence": divergence}
    else:
        result["comparison"] = {"label": "HOST_OBSERVED", "verdict": "OBSERVED_TRACE_ONLY",
                                 "reference": "NOT_DERIVABLE_FROM_VISIBLE_BRIEF_OR_PUBLIC_TESTS"}
    return result


def timeline(*args, **kwargs) -> dict:
    return run_timeline(*args, **kwargs)


class HostTimelineExecutor:
    """Small object façade for callers that keep a host executor per run."""

    def __init__(self, *, timeout_s: float = 4.0):
        self.timeout_s = timeout_s

    def run(self, *, source: str, path: str = "candidate.py", sequence: Sequence[object],
            reference: object = None, reference_source: str | None = None,
            clock_start: float = 0.0) -> dict:
        return run_timeline(source=source, path=path, sequence=sequence,
                            reference=reference, reference_source=reference_source,
                            timeout_s=self.timeout_s, clock_start=clock_start)


def execute_timeline(*args, **kwargs) -> dict:
    return run_timeline(*args, **kwargs)


def boundary_sequences() -> list[list[dict]]:
    """Small public boundary templates; callers may add public-test sequences."""
    return [
        [{"call": "allow", "args": [], "clock": 0.0}],
        [{"call": "allow", "args": [0.5], "clock": 0.0}],
        [{"call": "allow", "args": [2.0], "clock": 0.0}],
        [],
    ]


def fork_record(profiles: Sequence[Mapping[str, object]]) -> dict:
    profiles = list(profiles)[:3]
    return {"schema": "stateframe-think/5d/1", "authority": "HOST_OBSERVED",
            "profiles": [copy.deepcopy(dict(profile)) for profile in profiles],
            "semantic_deltas": [str(profile.get("delta", profile.get("name", "unlabeled")))
                                 for profile in profiles]}


def find_interpretation_witness(*, source: str, profiles: Sequence[Mapping[str, object]],
                                sequences: Sequence[Sequence[object]], path: str = "candidate.py",
                                timeout_s: float = 4.0) -> dict:
    """Find the smallest input whose visible profile references distinguish.

    Every candidate is executed through the same 4D host executor.  Profile
    references are data, not model-generated predicates or executable strings.
    """
    profiles = list(profiles)[:3]
    bounded_sequences = list(sequences)[:MAX_FORK_SEQUENCES]
    for sequence in sorted((list(item) for item in bounded_sequences), key=lambda item: (len(item), repr(item))):
        runs = [run_timeline(source=source, path=path, sequence=sequence,
                             reference=profile.get("reference"),
                             reference_source=profile.get("reference_source"),
                             timeout_s=timeout_s) for profile in profiles]
        if any(run.get("status") == "UNKNOWN" for run in runs):
            return {"schema": "stateframe-think/5d/1", "status": "UNKNOWN",
                    "authority": "HOST_OBSERVED", "sequence": sequence, "runs": runs}
        projections = []
        for profile, run in zip(profiles, runs):
            reference = profile.get("reference")
            if _reference_is_visible(reference, profile.get("reference_source")):
                values = []
                for item in reference if isinstance(reference, Sequence) and not isinstance(reference, (str, bytes)) else ():
                    if isinstance(item, Mapping):
                        values.append(item.get("value", item.get("result", item.get("status"))))
                    else:
                        values.append(item)
                projections.append(tuple(values))
            else:
                projections.append(tuple(_result_projection(row) for row in run.get("trace", ())))
        if len(set(projections)) > 1:
            return {"schema": "stateframe-think/5d/1", "status": "WITNESS_FOUND",
                    "authority": "HOST_COMPUTED(under assumptions)", "sequence": sequence, "runs": runs}
    return {"schema": "stateframe-think/5d/1",
            "status": "NO_WITNESS_IN_SEARCHED_SCOPE", "authority": "HOST_OBSERVED",
            "searched_sequences": len(bounded_sequences)}


def divergence_finder(*args, **kwargs) -> dict:
    return find_interpretation_witness(*args, **kwargs)


def shape_module(*args, **kwargs) -> dict:
    return build_shape_view(*args, **kwargs)


def render_timeline(view: Mapping[str, object], *, max_tokens: int = TIMELINE_MAX_TOKENS) -> str:
    comparison = view.get("comparison", {})
    label = comparison.get("label", "HOST_OBSERVED") if isinstance(comparison, Mapping) else "HOST_OBSERVED"
    lines = [f"[StateFrame THINK 4D | {label}]", "CLOCK " + json.dumps(view.get("clock_used", {}), sort_keys=True)]
    if isinstance(comparison, Mapping):
        lines.append("COMPARISON " + str(comparison.get("verdict", "UNKNOWN")))
        if comparison.get("first_divergence"):
            lines.append("FIRST_DIVERGENCE " + _short(comparison["first_divergence"], 180))
    for row in list(view.get("trace", ()) or ())[:12]:
        if isinstance(row, Mapping):
            lines.append(f"k={row.get('index')} before={_short(row.get('state_before'), 100)} "
                         f"action={_short(row.get('action'), 100)} result={_short(row.get('result'), 100)} "
                         f"after={_short(row.get('state_after'), 100)}")
    text = "\n".join(lines)
    limit = max(700, int(max_tokens) * 4)
    return text if len(text) <= limit else text[: max(0, limit - 14)] + "\n[TRUNCATED]"


def render_fork(view: Mapping[str, object]) -> str:
    status = view.get("status", "UNKNOWN")
    return "\n".join(["[StateFrame THINK 5D | HOST_OBSERVED]",
                      f"STATUS {status}", f"WITNESS {view.get('sequence', 'none')}"])


class ThinkingLayer:
    """Run-scoped compute/exposure policy for the three thinking views."""

    def __init__(self, mode: str | None = None, journal: object = None):
        self.mode = thinking_mode() if mode is None else mode
        if self.mode not in VALID_MODES:
            raise ValueError("invalid StateFrame thinking mode")
        self.journal = journal
        self.shape: dict | None = None
        self.timeline: dict | None = None
        self.fork: dict | None = None
        self.rendered: str | None = None

    def _emit(self, kind: str, **fields: object) -> None:
        if self.journal is not None and hasattr(self.journal, "emit"):
            self.journal.emit(kind, **fields)

    def observe(self, *, brief: str, source: str, path: str,
                receipts: Sequence[object] = (), trigger: str | None = None,
                sequence: Sequence[object] | None = None, reference: object = None,
                reference_source: str | None = None,
                profiles: Sequence[Mapping[str, object]] = (),
                fork_sequences: Sequence[Sequence[object]] = ()) -> dict:
        if self.mode == "off":
            return {}
        self.shape = build_shape_view(brief=brief, source=source, path=path, receipts=receipts)
        self.shape["_source"] = source
        self._emit("thinking_3d_computed", path=path, revision=self.shape["revision"],
                   components=len(self.shape["ast_components"]), rows=len(self.shape["evidence_rows"]))
        if sequence is not None:
            self.timeline = run_timeline(source=source, path=path, sequence=sequence,
                                         reference=reference, reference_source=reference_source)
            self._emit("thinking_4d_computed", path=path, status=self.timeline.get("status"),
                       comparison=self.timeline.get("comparison"))
        if profiles and fork_sequences:
            self.fork = find_interpretation_witness(source=source, path=path, profiles=profiles,
                                                    sequences=fork_sequences)
            self._emit("thinking_5d_computed", status=self.fork.get("status"))
        self.rendered = None
        if self.mode == "assisted" and trigger in TRIGGERS:
            blocks = [render_shape(self.shape)]
            if self.timeline is not None:
                blocks.append(render_timeline(self.timeline))
            if self.fork is not None:
                blocks.append(render_fork(self.fork))
            self.rendered = "\n\n".join(blocks)
            self._emit("thinking_rendered", trigger=trigger, blocks=len(blocks), authority="HOST_OBSERVED")
        return {"mode": self.mode, "computed": True, "rendered": self.rendered is not None,
                "trigger": trigger}

    def before_done(self, **kwargs: object) -> dict:
        kwargs["trigger"] = "before_done"
        return self.observe(**kwargs)


def append_thinking_sidecar(messages: Sequence[dict], layer: object, *,
                            provider_request: bool = True) -> list[dict]:
    """Append the exact assisted block to either provider arm."""
    result = [copy.deepcopy(message) for message in messages]
    if provider_request and isinstance(layer, ThinkingLayer) and layer.mode == "assisted" and layer.rendered:
        result.append({"role": "user", "content": layer.rendered})
    return result


__all__ = [
    "MODE_ENV", "VALID_MODES", "TRIGGERS", "HostTimelineExecutor", "ThinkingLayer",
    "append_thinking_sidecar",
    "boundary_sequences", "build_shape_view", "divergence_finder", "enabled",
    "execute_timeline", "find_interpretation_witness", "fork_record", "fresh_isolated_import",
    "render_fork", "render_shape", "render_timeline", "run_timeline", "shape_module",
    "shape_view", "thinking_mode", "timeline",
]
