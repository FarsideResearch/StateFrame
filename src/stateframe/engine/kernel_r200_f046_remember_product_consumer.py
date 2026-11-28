"""Recall statically resolved Python callback consumers from verified submissions."""
from __future__ import annotations

import ast
import json
import os
from collections.abc import Mapping
from functools import wraps
from pathlib import Path, PurePosixPath
from threading import Lock

from kernel_request_evidence_closure import source_revision

SWITCH = "STATEFRAME_EFFECT_CONSUMER_MEMORY"
PREFIX = "[Effect consumer memory] "
MAX_ROWS = 32
MAX_ROW_TOKENS = 96
_MAX_SOURCE_BYTES = 256 * 1024
_MAX_FACT_BYTES = 384
_ATTR = "_effect_consumer_memory"
_LOCK = Lock()
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def enabled() -> bool:
    return os.environ.get(SWITCH, "off").strip().lower() in {"on", "true", "1", "yes"}


def _workspace(runtime: object) -> Path | None:
    root = getattr(getattr(runtime, "owner", None), "root", None)
    return Path(root) if isinstance(root, (str, Path)) else None


def _load(root: Path, path: str) -> tuple[str, ast.Module, bytes] | None:
    from kernel_green_lock import is_product_path
    from kernel_stateframe_headcarry import _read_optional_source
    from stateframe.workcopy import frame_source_path_allowed

    if (not path.endswith(".py") or not frame_source_path_allowed(path)
            or not is_product_path(path)):
        return None
    raw = _read_optional_source(root, path)
    if not raw or len(raw) > _MAX_SOURCE_BYTES:
        return None
    try:
        tree = ast.parse(raw.decode("utf-8"))
    except (SyntaxError, UnicodeError, RecursionError, ValueError):
        return None
    return source_revision(raw), tree, raw


def _walk(scope: ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
    """Walk a single lexical scope without attributing a child's effects to it."""
    pending = list(reversed(scope.body))
    while pending:
        node = pending.pop()
        yield node
        if not isinstance(node, (*_FUNCTIONS, ast.ClassDef, ast.Lambda,
                                 ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            pending.extend(reversed(list(ast.iter_child_nodes(node))))


def _bindings(scope):
    """Resolve unconditional direct bindings and mark competing bindings unknown."""
    direct = set(scope.body)
    names = {}
    instances = {
        stmt.targets[0]: stmt for stmt in scope.body
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name) and isinstance(stmt.value, ast.Call)
        and not stmt.value.args and not stmt.value.keywords
    }

    def bind(name, node):
        names[name] = node if name not in names else None

    if isinstance(scope, _FUNCTIONS):
        for arg in ast.walk(scope.args):
            if isinstance(arg, ast.arg):
                bind(arg.arg, None)
    for node in _walk(scope):
        if isinstance(node, (*_FUNCTIONS, ast.ClassDef)):
            bind(node.name, node if node in direct and not node.decorator_list else None)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bind(alias.asname or alias.name.split(".")[0],
                     (node, alias) if node in direct else None)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
            receiver = node.value
            while isinstance(receiver, ast.Attribute):
                receiver = receiver.value
            if isinstance(receiver, ast.Name):
                bind(receiver.id, None)
        elif isinstance(node, (ast.Name, ast.ExceptHandler, ast.MatchAs, ast.MatchStar,
                               ast.MatchMapping, ast.Global, ast.Nonlocal)):
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                bind(node.id, instances.get(node))
            elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name:
                bind(node.name, None)
            elif isinstance(node, ast.MatchMapping) and node.rest:
                bind(node.rest, None)
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                for name in node.names:
                    bind(name, None)
    return names


def _module_path(root: Path, origin: str, module: str, level: int) -> str | None:
    parent = PurePosixPath(origin).parent
    if level:
        if level > len(parent.parts) + 1:
            return None
        for _ in range(level - 1):
            parent = parent.parent
    else:
        # An absolute import under a src layout is rooted at src.
        parent = PurePosixPath("src" if origin.startswith("src/") else ".")
    if module:
        base = parent.joinpath(*module.split("."))
        candidates = (base.with_suffix(".py"), base / "__init__.py")
    else:
        # `from . import name` refers to the containing package. At the
        # repository root that path is `.`, which has no suffix-able name.
        candidates = (parent / "__init__.py",)
    paths = [p.as_posix() for p in candidates if p.name != ".py" and (root / p).is_file()]
    return paths[0] if len(paths) == 1 else None


def _unambiguous_self_package_import(origin, package, tree, node, alias) -> bool:
    """Resolve a package initializer's first direct relative import as a submodule."""
    if (package != origin or tree is None or node.module is not None or node.level < 1):
        return False
    imported_name = alias.asname or alias.name
    names = _bindings(tree)
    binding = names.get(imported_name)
    if (not isinstance(binding, tuple) or binding[0] is not node or binding[1] is not alias
            or "__getattr__" in names):
        return False
    for statement in tree.body:
        if statement is node:
            return True
        if (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)):
            continue
        if (isinstance(statement, ast.ImportFrom) and statement.level == 0
                and statement.module == "__future__"):
            continue
        return False
    return False


def _producer(root, origin, func, names, *, instance=True, origin_tree=None):
    """Resolve direct functions, imported modules and simple constructor receivers."""
    if isinstance(func, ast.Name):
        binding = names.get(func.id)
        if isinstance(binding, (*_FUNCTIONS, ast.ClassDef)) and binding.col_offset == 0:
            return origin, func.id, (binding.lineno,)
        if isinstance(binding, tuple) and isinstance(binding[0], ast.ImportFrom):
            node, alias = binding
            if node.module:
                path = _module_path(root, origin, node.module, node.level)
                if path:
                    return path, alias.name, tuple(range(node.lineno, node.end_lineno + 1))
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        binding = names.get(func.value.id)
        if instance and isinstance(binding, ast.Assign) and binding.end_lineno < func.lineno:
            constructor = _producer(root, origin, binding.value.func, names, instance=False,
                                    origin_tree=origin_tree)
            if constructor is not None:
                path, symbol, lines = constructor
                return path, f"{symbol}.{func.attr}", (*lines, binding.lineno)
        if isinstance(binding, tuple):
            node, alias = binding
            if isinstance(node, ast.Import):
                module, level = alias.name, 0
                if "." in module and not alias.asname:
                    return None
            else:
                module = ".".join(part for part in (node.module, alias.name) if part)
                level = node.level
                # A package member and a submodule can have the same name.
                package = _module_path(root, origin, node.module or "", level)
                if package is None or PurePosixPath(package).name != "__init__.py":
                    return None
                if package == origin:
                    if not _unambiguous_self_package_import(
                            origin, package, origin_tree, node, alias):
                        return None
                else:
                    loaded = _load(root, package)
                    if loaded is None or loaded[1].body:
                        return None
            path = _module_path(root, origin, module, level)
            if path:
                return path, func.attr, tuple(range(node.lineno, node.end_lineno + 1))
    return None


def _action(callback) -> tuple[str, int] | None:
    body = callback.body
    if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    if len(body) != 1 or not isinstance(body[0], (ast.Expr, ast.Return)):
        return None
    call = body[0].value
    if isinstance(call, ast.Await):
        call = call.value
    if not isinstance(call, ast.Call):
        return None
    # Render a concrete call shape without lifting literals from product source.
    target = call.func
    parts = []
    while isinstance(target, ast.Attribute):
        parts.append(target.attr)
        target = target.value
    if not isinstance(target, ast.Name):
        return None
    parts.append(target.id)
    name = ".".join(reversed(parts))
    if not name.isascii() or len(name) > 96:
        return None
    arguments = [arg.id if isinstance(arg, ast.Name) and arg.id.isascii() else "..."
                 for arg in call.args]
    if call.keywords:
        arguments.append("...")
    action = f"{name}({', '.join(arguments)})"
    return (action, call.lineno) if len(action) <= 120 else None


def _instance_attribute_overridden(initializer, member: str) -> bool:
    return any(
        isinstance(node, ast.Attribute) and node.attr == member
        and isinstance(node.ctx, (ast.Store, ast.Del))
        and isinstance(node.value, ast.Name) and node.value.id == "self"
        for node in ast.walk(initializer)
    )


def _definition(tree, symbol):
    if "*" in (names := _bindings(tree)):
        return None
    name, _, member = symbol.partition(".")
    node = names.get(name)
    scope = node
    if isinstance(node, ast.ClassDef) and not node.bases and not node.keywords:
        methods = _bindings(node)
        if set(methods) & {"__getattr__", "__getattribute__", "__new__"}:
            return None
        if member:
            initializer = methods.get("__init__")
            if (isinstance(initializer, _FUNCTIONS)
                    and _instance_attribute_overridden(initializer, member)):
                return None
            node = methods.get(member)
        else:
            node = methods.get("__init__")
    elif member:
        return None
    if not isinstance(node, _FUNCTIONS):
        return None
    return (scope if not member else node), node


def _signature(tree, symbol, keyword):
    definition = _definition(tree, symbol)
    if definition is None:
        return None
    _, node = definition
    if keyword in {arg.arg for arg in (*node.args.args, *node.args.kwonlyargs)}:
        return definition
    return None


def _async_call_executes(root, origin, module, node, call) -> bool:
    parents = {
        child: parent
        for parent in _walk(node)
        for child in ast.iter_child_nodes(parent)
    }
    parent = parents.get(call)
    if isinstance(parent, ast.Await):
        return True
    if not isinstance(parent, ast.Call):
        return False
    target = parent.func
    schedulers = {"create_task", "ensure_future"}
    local = _bindings(node)
    global_names = _bindings(module)

    def binding_for(name):
        return local[name] if name in local else global_names.get(name)

    if isinstance(target, ast.Attribute) and target.attr in schedulers:
        receiver = target.value
        if not isinstance(receiver, ast.Name):
            return False
        binding = binding_for(receiver.id)
        if (not isinstance(binding, tuple) or not isinstance(binding[0], ast.Import)
                or binding[1].name != "asyncio"):
            return False
        module_root = Path("src") if origin.startswith("src/") else Path(".")
        local_asyncio = root / module_root / "asyncio"
        if local_asyncio.is_dir() or local_asyncio.with_suffix(".py").is_file():
            return False
        scheduler = target.attr
    elif isinstance(target, ast.Name):
        binding = binding_for(target.id)
        if (not isinstance(binding, tuple) or not isinstance(binding[0], ast.ImportFrom)
                or binding[0].module != "asyncio" or binding[0].level != 0
                or binding[1].name not in schedulers):
            return False
        scheduler = binding[1].name
    else:
        return False
    return (scheduler in schedulers
            and (call in parent.args or any(item.value is call for item in parent.keywords)))


def _firing(signature, keyword, async_callback=False, *, root=None, origin="", module=None):
    if signature is None:
        return None
    scope, node = signature
    # Ignore an invocation when the parameter is rebound or escaped (including
    # into another scope). A unique direct call is the supported firing shape.
    loads = [item for item in ast.walk(node) if isinstance(item, ast.Name) and item.id == keyword]
    calls = [item for item in _walk(node) if isinstance(item, ast.Call)
             and isinstance(item.func, ast.Name) and item.func.id == keyword]
    guarded_names = set()
    for item in _walk(node):
        if not isinstance(item, ast.If):
            continue
        test = item.test
        if isinstance(test, ast.Name):
            guarded_names.add(test)
        elif (isinstance(test, ast.Compare) and len(test.ops) == 1
              and isinstance(test.ops[0], (ast.Is, ast.IsNot))
              and isinstance(test.comparators[0], ast.Constant)
              and test.comparators[0].value is None):
            guarded_names.add(test.left)
    if (len(calls) == 1 and set(loads) <= {calls[0].func, *guarded_names}
            and (not async_callback or (root is not None and module is not None
                                        and _async_call_executes(
                                            root, origin, module, node, calls[0])))):
        return calls[0], ()
    if len(loads) != 1 or not isinstance(scope, ast.ClassDef):
        return None
    # A constructor may store the callable for a later method on the same
    # instance. Require one field assignment and one direct invocation.
    for assignment in node.body:
        if (not isinstance(assignment, ast.Assign) or assignment.value is not loads[0]
                or len(assignment.targets) != 1 or not isinstance(assignment.targets[0], ast.Attribute)):
            continue
        target = assignment.targets[0]
        if not isinstance(target.value, ast.Name) or target.value.id != "self":
            return None
        attributes = [item for item in ast.walk(scope)
                      if isinstance(item, ast.Attribute) and item.attr == target.attr]
        if len(attributes) != 2:
            return None
        for method in _bindings(scope).values():
            if (not isinstance(method, _FUNCTIONS) or not method.args.args
                    or method.args.args[0].arg != "self"):
                continue
            for call in _walk(method):
                if (isinstance(call, ast.Call) and call.func in attributes
                        and call.func is not target and isinstance(call.func.value, ast.Name)
                        and call.func.value.id == "self"
                        and (not async_callback or (root is not None and module is not None
                                                    and _async_call_executes(
                                                        root, origin, module, method, call)))):
                    return call, (assignment, method)
    return None


def _covered(ranges, first: int, last: int) -> bool:
    for start, end in ranges:
        if start > first:
            break
        first = max(first, end + 1)
        if first > last:
            return True
    return False


def _registrations(root, path, revision, tree, ranges, load, *, require_signature=True):
    module = _bindings(tree)
    if "*" in module:
        return
    scopes = [tree, *(node for node in module.values() if isinstance(node, _FUNCTIONS))]
    for scope in scopes:
        local = _bindings(scope) if scope is not tree else module
        if "*" in local:
            continue
        names = {**module, **local}
        for call in _walk(scope):
            if (not isinstance(call, ast.Call) or not call.keywords
                    or not _covered(ranges, call.lineno, call.end_lineno)
                    or any(kw.arg is None for kw in call.keywords)
                    or len({kw.arg for kw in call.keywords}) != len(call.keywords)):
                continue
            producer = _producer(root, path, call.func, names, origin_tree=tree)
            if producer is None:
                continue
            producer_path, symbol, binding_lines = producer
            producer_source = load(producer_path)
            if (producer_source is None or max(binding_lines) > call.lineno
                    or not all(_covered(ranges, line, line) for line in binding_lines)):
                continue
            for kw in call.keywords:
                callback = names.get(kw.value.id) if isinstance(kw.value, ast.Name) else None
                if (not isinstance(callback, _FUNCTIONS) or not kw.arg
                        or callback.end_lineno >= call.lineno
                        or not _covered(ranges, callback.lineno, callback.end_lineno)
                        or (require_signature and _signature(
                            producer_source[1], symbol, kw.arg) is None)
                        or (not require_signature and _definition(
                            producer_source[1], symbol) is None)):
                    continue
                effect = _action(callback)
                if effect is not None:
                    action, action_line = effect
                    yield (path, revision, call.lineno, action_line, action,
                           producer_path, symbol, kw.arg, call.col_offset,
                           isinstance(callback, ast.AsyncFunctionDef))


def _verified_ranges(receipt: Mapping, load) -> dict[str, tuple[str, list[tuple[int, int]]]]:
    from itertools import accumulate

    locations = receipt.get("final_message_locations", {})
    proofs = receipt.get("delivery_proofs", {})
    submitted = receipt.get("submitted_http_bytes_digest")
    if not isinstance(submitted, str) or not submitted.startswith("sha256:"):
        return {}
    result = {}
    for path, row in receipt.get("paths", {}).items():
        if (not isinstance(path, str) or not path.endswith(".py") or not locations.get(path)
                or not proofs.get(path) or row.get("proof_status") != "VERIFIED"):
            continue
        source = load(path)
        if source is None or source[0] != row.get("revision"):
            continue
        revision, _, raw = source
        offsets = {offset: index for index, offset in enumerate(
            accumulate((len(line) for line in raw.splitlines(keepends=True)), initial=0))}
        spans = []
        for proof in proofs[path]:
            key = proof.get("key", {})
            first, last = key.get("byte_start"), key.get("byte_end")
            if (proof.get("wire_contains_exact") is True
                    and proof.get("transport_identity") == "VERIFIED"
                    and proof.get("final_message_location") in locations[path]
                    and proof.get("submitted_http_bytes_digest") == submitted
                    and key.get("canonical_path") == path and key.get("content_digest") == revision
                    and type(first) is int and type(last) is int and first < last
                    and first in offsets and last in offsets
                    and proof.get("delivered_digest") == source_revision(raw[first:last])):
                spans.append((offsets[first] + 1, offsets[last]))
        if spans:
            result[path] = (revision, sorted(spans))
    return result


def remember(runtime: object, receipt: Mapping) -> None:
    """Store plain revision-bound observations. The caller has verified submission."""
    root = _workspace(runtime)
    if root is None or getattr(runtime, "_unit_policy", None) is not None:
        return
    cache = {}

    def load(path):
        if path not in cache:
            cache[path] = _load(root, path)
        return cache[path]

    ranges = _verified_ranges(receipt, load)
    if not ranges:
        return
    with _LOCK:
        rows = dict(getattr(runtime, _ATTR, {}))
        for path, (revision, spans) in sorted(ranges.items()):
            source = load(path)
            if source is None or source[0] != revision:
                continue
            for binding in _registrations(root, path, revision, source[1], spans, load):
                key = (path, binding[2], binding[8], binding[5], binding[6], binding[7])
                previous = rows.pop(key, None)
                firing = previous[1] if previous is not None and previous[0] == binding else None
                rows[key] = (binding, firing)
                if len(rows) > MAX_ROWS:
                    del rows[next(iter(rows))]
        for key, (binding, firing) in rows.items():
            path, symbol, keyword = binding[5:8]
            if path not in ranges:
                continue
            revision, spans = ranges[path]
            source = load(path)
            if source is None or source[0] != revision:
                continue
            signature = _signature(source[1], symbol, keyword)
            firing_site = _firing(signature, keyword, binding[9],
                                  root=root, origin=path, module=source[1])
            if firing_site is None:
                continue
            scope, node = signature
            call, links = firing_site
            required = [(scope.lineno, scope.lineno), (node.lineno, node.body[0].lineno),
                        (call.lineno, call.end_lineno)]
            required.extend((link.lineno, link.end_lineno if isinstance(link, ast.Assign)
                             else link.body[0].lineno) for link in links)
            if all(_covered(spans, start, end) for start, end in required):
                rows[key] = (binding, (revision, call.lineno))
        setattr(runtime, _ATTR, rows)


def select(runtime: object, frame: Mapping) -> str | None:
    root = _workspace(runtime)
    site = frame.get("next_site")
    sites = frame.get("open_sites")
    if site is None and isinstance(sites, list) and sites:
        site = sites[0]
    if root is None or not isinstance(site, Mapping):
        return None
    path = site.get("path")
    rows = [row for row in getattr(runtime, _ATTR, {}).values() if row[0][5] == path]
    if not rows:
        return None
    current = _load(root, path)
    if current is None or site.get("revision") != current[0]:
        return None
    candidates = []
    lines = site.get("lines")
    for binding, firing in rows:
        consumer, revision, _, _, _, _, symbol, keyword, _, _ = binding
        definition = _definition(current[1], symbol)
        if definition is None:
            continue
        signature = _signature(current[1], symbol, keyword)
        node, _ = definition
        source = _load(root, consumer)
        if source is None or source[0] != revision:
            continue
        if (not isinstance(lines, Mapping) or type(lines.get("start")) is not int
                or type(lines.get("end")) is not int
                or lines["start"] > node.end_lineno or lines["end"] < node.lineno):
            continue
        full = ((1, len(source[2].splitlines())),)
        if binding in _registrations(root, consumer, revision, source[1], full,
                                     lambda target: current if target == path else _load(root, target),
                                     require_signature=False):
            candidates.append((binding, firing if signature is not None else None))
    identities = {(binding[5], binding[6], binding[7]) for binding, _ in candidates}
    if len(identities) > 1:
        anchor_line = site.get("line")
        firing_matches = []
        if type(anchor_line) is int:
            if 1 <= anchor_line <= len(current[2].splitlines()):
                firing_matches = [
                    candidate for candidate in candidates
                    if candidate[1] is not None and candidate[1][0] == current[0]
                    and candidate[1][1] == anchor_line
                ]
        else:
            firing_matches = [
                candidate for candidate in candidates
                if candidate[1] is not None and candidate[1][0] == current[0]
                and lines["start"] <= candidate[1][1] <= lines["end"]
            ]
        matching_identities = {(binding[5], binding[6], binding[7])
                               for binding, _ in firing_matches}
        if len(matching_identities) != 1:
            return None
        identity = next(iter(matching_identities))
        candidates = [candidate for candidate in candidates
                      if (candidate[0][5], candidate[0][6], candidate[0][7]) == identity]
    if not candidates or len({binding[4] for binding, _ in candidates}) != 1:
        return None
    binding, firing = min(candidates, key=lambda row: row[0])
    consumer, _, registration, action_line, action, _, symbol, keyword, _, _ = binding
    fields = {"binding": f"{consumer}:{registration}", "callback": f"{symbol}.{keyword}",
              "consumer": f"{consumer}:{action_line} {action}"}
    if firing is not None and firing[0] == current[0]:
        fields["firing"] = f"{path}:{firing[1]}"
    fact = PREFIX + json.dumps(fields, ensure_ascii=True, separators=(",", ":"))
    return fact if len(fact) <= _MAX_FACT_BYTES else None


def install() -> None:
    """Subscribe to the existing exact-wire submission hook, once per engine."""
    import kernel_request_source_window as source
    from kernel_frame_optional_allocation import install_effect_consumer_allocation

    with _LOCK:
        original = source.record_verified_source_messages
        if getattr(original, "_effect_consumer_hook", False):
            return

        @wraps(original)
        def observed(runtime, messages, receipt):
            added = original(runtime, messages, receipt)
            if added and enabled():
                remember(runtime, receipt)
            return added

        observed._effect_consumer_hook = True
        install_effect_consumer_allocation()
        source.record_verified_source_messages = observed
