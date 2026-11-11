"""Small AST scanners shared by the auth and unavailable-event class guards."""
from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping


ALLOWLIST = {
    "src/stateframe/engine/kernel_web_fetch.py:fetch_raw":
        ("Sends a bounded GET through vetted destinations with a bounded, validated caller-supplied Accept header; "
         "it fetches user-selected page content "
         "and does not invoke a model server."),
}
_UNSUPPORTED_HTTP_LIBRARIES = {"requests", "httpx", "aiohttp", "urllib3", "pycurl"}


@lru_cache(maxsize=2048)
def _parse_file(path: str, modified_ns: int, size: int) -> ast.AST:
    del modified_ns, size
    return ast.parse(Path(path).read_text(encoding="utf-8"), filename=path)


@dataclass(frozen=True)
class AuthScan:
    scanned: int
    keyed: int
    allowlisted: int
    violations: tuple[str, ...]

    @property
    def receipt(self) -> str:
        return (f"auth-guard scan: scanned={self.scanned} keyed={self.keyed} "
                f"allowlisted={self.allowlisted} violations={len(self.violations)}")


@dataclass(frozen=True)
class UnavailableScan:
    scanned: int
    with_reason: int
    violations: tuple[str, ...]

    @property
    def receipt(self) -> str:
        return (f"unavailable-reason scan: scanned={self.scanned} "
                f"with_reason={self.with_reason} violations={len(self.violations)}")


def _qualified_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _qualified_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _function_scope(node: ast.AST, parents: Mapping[ast.AST, ast.AST]) -> ast.AST | None:
    parent = parents.get(node)
    while parent is not None:
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return parent
        parent = parents.get(parent)
    return None


class _ScopeVisitor(ast.NodeVisitor):
    """Visit one function body without accidentally borrowing a nested function's auth."""

    def __init__(self, scope: ast.AST) -> None:
        self.scope = scope
        self.nodes: list[ast.AST] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node is self.scope:
            self.nodes.append(node)
            for item in node.body:
                self.visit(item)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if node is self.scope:
            self.nodes.append(node)
            for item in node.body:
                self.visit(item)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        if node is self.scope:
            self.nodes.append(node)
            self.visit(node.body)

    def generic_visit(self, node: ast.AST) -> None:
        self.nodes.append(node)
        super().generic_visit(node)


def _scope_nodes(scope: ast.AST | None, tree: ast.AST) -> list[ast.AST]:
    if scope is None:
        return list(ast.walk(tree))
    visitor = _ScopeVisitor(scope)
    visitor.visit(scope)
    return visitor.nodes


def _uses_key(value: ast.AST) -> bool:
    for node in ast.walk(value):
        if isinstance(node, ast.Name) and any(
                token in node.id.lower() for token in ("key", "token", "credential")):
            return True
        if isinstance(node, ast.Attribute) and any(
                token in node.attr.lower() for token in ("key", "token", "credential")):
            return True
        if (isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)
                and node.slice.value in {"key", "api_key", "auth_key", "token"}):
            return True
    return False


def _is_authorization_key(key: ast.AST | None) -> bool:
    return (isinstance(key, ast.Constant) and isinstance(key.value, str)
            and key.value.lower() == "authorization")


def _header_expr_is_keyed(value: ast.AST, scope_nodes: list[ast.AST], *,
                          before: int, helpers: Mapping[str, list[ast.AST]],
                          seen: frozenset[str] = frozenset()) -> bool:
    if isinstance(value, ast.Dict):
        for key, item in zip(value.keys, value.values):
            if _is_authorization_key(key) and _uses_key(item):
                return True
            if key is None and _header_expr_is_keyed(
                    item, scope_nodes, before=before, helpers=helpers, seen=seen):
                return True
    if isinstance(value, ast.IfExp):
        # An empty-key branch may omit auth; a non-key condition may not.
        return (_uses_key(value.test)
                and _header_expr_is_keyed(value.body, scope_nodes, before=before,
                                          helpers=helpers, seen=seen)) or (
                    _header_expr_is_keyed(value.body, scope_nodes, before=before,
                                          helpers=helpers, seen=seen)
                    and _header_expr_is_keyed(value.orelse, scope_nodes, before=before,
                                              helpers=helpers, seen=seen))
    if isinstance(value, ast.Call):
        name = _qualified_name(value.func).rsplit(".", 1)[-1]
        definitions = helpers.get(name, ())
        if definitions and name not in seen:
            for helper in definitions:
                helper_nodes = _scope_nodes(helper, helper)
                returns = [node.value for node in helper_nodes if isinstance(node, ast.Return)
                           and node.value is not None]
                if not returns or not all(
                        _header_expr_is_keyed(item, helper_nodes, before=item.lineno,
                                              helpers=helpers, seen=seen | {name})
                        for item in returns):
                    return False
            return True
    if isinstance(value, ast.Name):
        assignments = []
        for node in scope_nodes:
            if getattr(node, "lineno", 0) >= before:
                continue
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(target, ast.Name) and target.id == value.id for target in targets):
                    assignments.append(node)
        if assignments:
            latest = max(assignments, key=lambda item: item.lineno)
            if _header_expr_is_keyed(latest.value, scope_nodes, before=latest.lineno,
                                     helpers=helpers, seen=seen):
                return True
        for node in scope_nodes:
            if getattr(node, "lineno", 0) >= before:
                continue
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                            and target.value.id == value.id and _is_authorization_key(target.slice)
                            and _uses_key(node.value)
                            and (not assignments or node.lineno > latest.lineno)):
                        return True
            if isinstance(node, ast.Call) and _qualified_name(node.func).endswith(".add_header"):
                receiver = node.func.value
                if (isinstance(receiver, ast.Name) and receiver.id == value.id
                        and len(node.args) >= 2 and _is_authorization_key(node.args[0])
                        and _uses_key(node.args[1])
                        and (not assignments or node.lineno > latest.lineno)):
                    return True
    return False


def _request_variable(call: ast.Call, parents: Mapping[ast.AST, ast.AST]) -> str | None:
    parent = parents.get(call)
    if isinstance(parent, ast.Assign):
        for target in parent.targets:
            if isinstance(target, ast.Name):
                return target.id
    if isinstance(parent, ast.AnnAssign) and isinstance(parent.target, ast.Name):
        return parent.target.id
    return None


def _keyed_http_call(call: ast.Call, scope_nodes: list[ast.AST],
                     parents: Mapping[ast.AST, ast.AST], name: str,
                     helpers: Mapping[str, list[ast.AST]]) -> bool:
    short = name.rsplit(".", 1)[-1]
    if short == "request_bytes":
        key = call.args[2] if len(call.args) > 2 else next(
            (item.value for item in call.keywords if item.arg in {"key", "api_key"}), None)
        return isinstance(key, ast.AST) and _uses_key(key)
    if short == "Request":
        headers = next((item.value for item in call.keywords if item.arg == "headers"), None)
        if headers is not None and _header_expr_is_keyed(
                headers, scope_nodes, before=call.lineno, helpers=helpers):
            return True
        request_name = _request_variable(call, parents)
        if request_name:
            sends = [node.lineno for node in scope_nodes if isinstance(node, ast.Call)
                     and node.lineno > call.lineno and any(
                         isinstance(arg, ast.Name) and arg.id == request_name
                         for arg in node.args)]
            if sends:
                return _header_expr_is_keyed(ast.Name(id=request_name), scope_nodes,
                                             before=min(sends), helpers=helpers)
        return False
    if short in {"request", "putrequest"}:
        headers = next((item.value for item in call.keywords if item.arg == "headers"), None)
        if headers is not None and _header_expr_is_keyed(
                headers, scope_nodes, before=call.lineno, helpers=helpers):
            return True
        if short == "request":
            return False
        receiver = ast.unparse(call.func.value) if isinstance(call.func, ast.Attribute) else ""
        return any(
            isinstance(node, ast.Call)
            and _qualified_name(node.func).endswith(".putheader")
            and isinstance(node.func, ast.Attribute)
            and ast.unparse(node.func.value) == receiver
            and node.lineno > call.lineno
            and len(node.args) >= 2
            and _is_authorization_key(node.args[0])
            and _uses_key(node.args[1])
            for node in scope_nodes
        )
    if short in {"send", "sendall"}:
        if call.args and isinstance(call.args[0], ast.Subscript):
            payload = call.args[0].value
            if isinstance(payload, ast.Name):
                return any(
                    isinstance(node, (ast.Assign, ast.AnnAssign))
                    and node.lineno < call.lineno
                    and any(isinstance(target, ast.Name) and target.id == payload.id
                            for target in (node.targets if isinstance(node, ast.Assign)
                                           else [node.target]))
                    and "Authorization: Bearer" in ast.unparse(node.value)
                    and _uses_key(node.value)
                    for node in scope_nodes)
    if short in {"urlopen", "open"} and call.args:
        # A request built by an unknown helper is not covered by a local Request site.
        return False
    return False


def _is_request_object(value: ast.AST, scope_nodes: list[ast.AST],
                       imports: Mapping[str, str], before: int) -> bool:
    if isinstance(value, ast.Call) and _resolved_name(value, imports) == "urllib.request.Request":
        return True
    if isinstance(value, ast.Name):
        return any(
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and node.lineno < before
            and any(isinstance(target, ast.Name) and target.id == value.id
                    for target in (node.targets if isinstance(node, ast.Assign) else [node.target]))
            and isinstance(node.value, ast.Call)
            and _resolved_name(node.value, imports) == "urllib.request.Request"
            for node in scope_nodes
        )
    return False


def _mentions_curl(value: ast.AST, scope_nodes: list[ast.AST], before: int,
                   helpers: Mapping[str, list[ast.AST]],
                   seen: frozenset[str] = frozenset()) -> bool:
    if isinstance(value, ast.Name):
        assigned = _assigned_value(value.id, scope_nodes, before)
        return bool(assigned and _mentions_curl(assigned, scope_nodes, assigned.lineno,
                                                 helpers, seen))
    if isinstance(value, ast.Call):
        name = _qualified_name(value.func).rsplit(".", 1)[-1]
        if name in helpers and name not in seen:
            for helper in helpers[name]:
                nodes = _scope_nodes(helper, helper)
                if any(
                        isinstance(item, ast.Return) and item.value is not None
                        and _mentions_curl(item.value, nodes, item.lineno,
                                           helpers, seen | {name})
                        for item in nodes):
                    return True
            return False
    return any(
        isinstance(item, ast.Constant) and isinstance(item.value, str)
        and (item.value in {"curl", "/usr/bin/curl"}
             or item.value.startswith(("curl ", "/usr/bin/curl ")))
        for item in ast.walk(value))


def _http_call_candidate(call: ast.Call, imports: Mapping[str, str], name: str,
                         scope_nodes: list[ast.AST],
                         helpers: Mapping[str, list[ast.AST]]) -> bool:
    short = name.rsplit(".", 1)[-1]
    if name in {"urllib.request.Request", "urllib.request.urlopen",
                "urllib.request.urlretrieve"}:
        return True
    if short == "open" and isinstance(call.func, ast.Attribute):
        receiver = call.func.value
        if ("urllib" in imports or "urllib.request" in imports.values()) and isinstance(receiver, ast.Name):
            if receiver.id.lower().endswith("opener"):
                return True
            return any(
                isinstance(node, (ast.Assign, ast.AnnAssign)) and node.lineno < call.lineno
                and any(isinstance(target, ast.Name) and target.id == receiver.id
                        for target in (node.targets if isinstance(node, ast.Assign)
                                       else [node.target]))
                and isinstance(node.value, ast.Call)
                and _resolved_name(node.value, imports) == "urllib.request.build_opener"
                for node in scope_nodes)
    if short in {"request", "putrequest"} and any(
            module == "http" or module.startswith("http.client")
            for module in imports.values()):
        return True
    if name == "kernel_bounded_http.request_bytes":
        return True
    if short in {"send", "sendall"}:
        return any(isinstance(item, ast.Constant) and isinstance(item.value, str)
                   and "HTTP/1.1" in item.value for item in scope_nodes)
    if name in {"subprocess.run", "subprocess.Popen", "subprocess.call",
                "subprocess.check_call", "subprocess.check_output", "os.system"}:
        return bool(call.args and _mentions_curl(
            call.args[0], scope_nodes, call.lineno, helpers))
    return False


def _http_call_kind(call: ast.Call, scope_nodes: list[ast.AST],
                    scope: ast.AST | None, name: str,
                    imports: Mapping[str, str]) -> str | None:
    short = name.rsplit(".", 1)[-1]
    if short in {"urlopen", "open"} and call.args and _is_request_object(
            call.args[0], scope_nodes, imports, call.lineno):
        return None
    if short in {"Request", "urlopen", "urlretrieve", "open", "request", "putrequest", "request_bytes"}:
        return short
    if short in {"send", "sendall"} and scope is not None and any(
            isinstance(item, ast.Constant) and isinstance(item.value, str)
            and "HTTP/1.1" in item.value for item in ast.walk(scope)):
        return "raw_http"
    if name in {"subprocess.run", "subprocess.Popen", "subprocess.call",
                "subprocess.check_call", "subprocess.check_output", "os.system"}:
        return "curl"
    return None


def _resolved_name(call: ast.Call, imports: Mapping[str, str],
                   scope_nodes: list[ast.AST] | None = None) -> str:
    raw = _qualified_name(call.func)
    first, _, tail = raw.partition(".")
    if scope_nodes is not None and first not in imports:
        assigned = _assigned_value(first, scope_nodes, call.lineno)
        if isinstance(assigned, (ast.Name, ast.Attribute)):
            alias = _qualified_name(assigned)
            alias_first, _, alias_tail = alias.partition(".")
            raw = imports.get(alias_first, alias_first) + (
                "." + alias_tail if alias_tail else "") + ("." + tail if tail else "")
            first, _, tail = raw.partition(".")
    module = imports.get(first, first)
    return module + ("." + tail if tail else "")


def scan_auth_requests(paths: Iterable[Path], *, root: Path,
                       allowlist: Mapping[str, str] = ALLOWLIST) -> AuthScan:
    keyed = allowlisted = 0
    violations: list[str] = []
    scanned = 0
    root = root.resolve()
    for path in sorted(Path(item) for item in paths):
        resolved = path.resolve()
        source = resolved.read_text(encoding="utf-8")
        if not any(marker in source for marker in (
                "urllib", "http.client", "request_bytes", "HTTP/1.1", "curl",
                "requests", "httpx", "aiohttp")):
            continue
        stat = resolved.stat()
        tree = _parse_file(str(resolved), stat.st_mtime_ns, stat.st_size)
        nodes = list(ast.walk(tree))
        parents: dict[ast.AST, ast.AST] = {}
        for parent in nodes:
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent
        label = path.resolve().relative_to(root).as_posix()
        for node in nodes:
            modules = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                       else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            for module in modules:
                if module.split(".", 1)[0] in _UNSUPPORTED_HTTP_LIBRARIES:
                    scanned += 1
                    violations.append(f"{label}:<import>:{node.lineno} (unsupported HTTP library: {module})")
        imports = {
            (alias.asname or alias.name.split(".")[0]):
                (alias.name if alias.asname else alias.name.split(".")[0])
            for node in nodes if isinstance(node, ast.Import)
            for alias in node.names
        }
        imports.update({
            (alias.asname or alias.name): f"{node.module}.{alias.name}"
            for node in nodes if isinstance(node, ast.ImportFrom) and node.module
            for alias in node.names
        })
        potential_aliases = set()
        for node in nodes:
            if isinstance(node, ast.Assign) and isinstance(node.value, (ast.Name, ast.Attribute)):
                raw = _qualified_name(node.value)
                first, _, tail = raw.partition(".")
                resolved = imports.get(first, first) + ("." + tail if tail else "")
                if resolved in {"urllib.request.Request", "urllib.request.urlopen",
                                "urllib.request.urlretrieve", "kernel_bounded_http.request_bytes"}:
                    potential_aliases.update(target.id for target in node.targets
                                             if isinstance(target, ast.Name))
        helpers: dict[str, list[ast.AST]] = {}
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                helpers.setdefault(node.name, []).append(node)
        scope_cache: dict[ast.AST | None, list[ast.AST]] = {}
        for call in (node for node in nodes if isinstance(node, ast.Call)):
            name = _resolved_name(call, imports)
            if (name.rsplit(".", 1)[-1] == "open"
                    and "urllib" not in imports
                    and "urllib.request" not in imports.values()):
                continue
            if (name.rsplit(".", 1)[-1] not in {"send", "sendall", "open"}
                    and name not in {"subprocess.run", "subprocess.Popen", "subprocess.call",
                                     "subprocess.check_call", "subprocess.check_output", "os.system"}
                    and name not in potential_aliases
                    and not _http_call_candidate(call, imports, name, [], helpers)):
                continue
            scope = _function_scope(call, parents)
            if scope not in scope_cache:
                scope_cache[scope] = _scope_nodes(scope, tree)
            scope_nodes = scope_cache[scope]
            name = _resolved_name(call, imports, scope_nodes)
            if not _http_call_candidate(call, imports, name, scope_nodes, helpers):
                continue
            kind = _http_call_kind(call, scope_nodes, scope, name, imports)
            if kind is None:
                continue
            scanned += 1
            function = getattr(scope, "name", "<module>")
            site = f"{label}:{function}"
            if _keyed_http_call(call, scope_nodes, parents, name, helpers):
                keyed += 1
            elif site in allowlist:
                allowlisted += 1
            else:
                violations.append(f"{site}:{call.lineno} ({kind})")
    return AuthScan(scanned, keyed, allowlisted, tuple(violations))


def _assigned_value(name: str, scope_nodes: list[ast.AST], before: int) -> ast.AST | None:
    assignments = [node for node in scope_nodes
                   if isinstance(node, (ast.Assign, ast.AnnAssign))
                   and node.lineno < before
                   and any(isinstance(target, ast.Name) and target.id == name
                           for target in (node.targets if isinstance(node, ast.Assign)
                                          else [node.target]))]
    return max(assignments, key=lambda item: item.lineno).value if assignments else None


def _literal_event(value: ast.AST, scope_nodes: list[ast.AST], before: int) -> str | None:
    if isinstance(value, ast.Name):
        assigned = _assigned_value(value.id, scope_nodes, before)
        return _literal_event(assigned, scope_nodes, assigned.lineno) if assigned else None
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value if value.value.endswith("_unavailable") else None
    if isinstance(value, ast.Dict):
        for key, item in zip(value.keys, value.values):
            if _is_event_field(key):
                return _literal_event(item, scope_nodes, before)
    return None


def _call_event_name(call: ast.Call, scope_nodes: list[ast.AST]) -> str | None:
    name = _qualified_name(call.func).rsplit(".", 1)[-1]
    if name in {"emit", "_emit", "write"}:
        for arg in call.args[:2]:
            event = _literal_event(arg, scope_nodes, call.lineno)
            if event:
                return event
        for item in call.keywords:
            if item.arg in {"event", "kind", "name", "type"}:
                event = _literal_event(item.value, scope_nodes, call.lineno)
                if event:
                    return event
    if name == "append" and isinstance(call.func, ast.Attribute):
        receiver = call.func.value
        collection = receiver.id if isinstance(receiver, ast.Name) else receiver.attr if isinstance(receiver, ast.Attribute) else ""
        if collection in {"events", "event_log", "journal_events"} and call.args:
            return _literal_event(call.args[0], scope_nodes, call.lineno)
    return None


def _is_event_field(key: ast.AST | None) -> bool:
    return (isinstance(key, ast.Constant) and isinstance(key.value, str)
            and key.value.lower() in {"event", "kind", "name", "type"})


def _nonempty_reason(value: ast.AST) -> bool:
    if isinstance(value, ast.Constant):
        return (isinstance(value.value, str) and bool(value.value.strip())
                and value.value.strip().lower() not in {"unavailable", "unknown", "error", "n/a"})
    return True


def _call_has_reason(call: ast.Call, scope_nodes: list[ast.AST]) -> bool:
    for item in call.keywords:
        if item.arg == "reason":
            value = item.value
            if isinstance(value, ast.Name):
                value = _assigned_value(value.id, scope_nodes, call.lineno) or value
            if _nonempty_reason(value):
                return True
    for arg in (*call.args, *(item.value for item in call.keywords)):
        if isinstance(arg, ast.Name):
            arg = _assigned_value(arg.id, scope_nodes, call.lineno) or arg
        if isinstance(arg, ast.Dict):
            for key, value in zip(arg.keys, arg.values):
                if (isinstance(key, ast.Constant) and key.value == "reason"
                        and _nonempty_reason(
                            _assigned_value(value.id, scope_nodes, call.lineno) or value
                            if isinstance(value, ast.Name) else value)):
                    return True
    return False


def scan_unavailable_events(paths: Iterable[Path], *, root: Path) -> UnavailableScan:
    scanned = with_reason = 0
    violations: list[str] = []
    root = root.resolve()
    for path in sorted(Path(item) for item in paths):
        resolved = path.resolve()
        if "_unavailable" not in resolved.read_text(encoding="utf-8"):
            continue
        stat = resolved.stat()
        tree = _parse_file(str(resolved), stat.st_mtime_ns, stat.st_size)
        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent
        label = path.resolve().relative_to(root).as_posix()
        scope_cache: dict[ast.AST | None, list[ast.AST]] = {}
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            if _qualified_name(call.func).rsplit(".", 1)[-1] not in {"emit", "_emit", "write", "append"}:
                continue
            scope = _function_scope(call, parents)
            if scope not in scope_cache:
                scope_cache[scope] = _scope_nodes(scope, tree)
            event = _call_event_name(call, scope_cache[scope])
            if event is None:
                continue
            scanned += 1
            function = getattr(scope, "name", "<module>")
            site = f"{label}:{function}:{call.lineno} ({event})"
            if _call_has_reason(call, scope_cache[scope]):
                with_reason += 1
            else:
                violations.append(site)
    return UnavailableScan(scanned, with_reason, tuple(violations))


_RAW_FETCH_HEADERS = frozenset({"Host", "Accept", "Accept-Encoding", "User-Agent"})
_BROKER_ENV = {
    "LC_ALL": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
    "HOME": "/nonexistent", "TZ": "UTC",
}


def _function(tree: ast.AST, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    matches = [node for node in ast.walk(tree)
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
    return matches[0] if len(matches) == 1 else None


def _import_modules(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names)
    if isinstance(node, ast.ImportFrom) and node.module:
        return (node.module,)
    return ()


def _is_kernel_bounded_http(module: str) -> bool:
    return module == "kernel_bounded_http" or module.endswith(".kernel_bounded_http")


def _resolved_reference(value: ast.AST, imports: Mapping[str, str],
                        scope_nodes: list[ast.AST], before: int,
                        seen: frozenset[str] = frozenset()) -> str:
    """Resolve imported and local aliases without following arbitrary expressions."""
    raw = _qualified_name(value)
    first, _, tail = raw.partition(".")
    if not first or first in seen:
        return raw
    assigned = _assigned_value(first, scope_nodes, before)
    if isinstance(assigned, (ast.Name, ast.Attribute)):
        base = _resolved_reference(assigned, imports, scope_nodes, assigned.lineno,
                                   seen | {first})
    else:
        base = imports.get(first, first)
    return base + ("." + tail if tail else "")


def scan_fetch_raw_headers(source: str) -> tuple[str, ...]:
    """Pin fetch_raw's caller surface and its literal outgoing header map."""
    tree = ast.parse(source)
    function = _function(tree, "fetch_raw")
    if function is None:
        return ("fetch_raw definition missing or ambiguous",)
    violations = []
    args = (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs)
    if (any("header" in argument.arg.lower() for argument in args)
            or function.args.vararg or function.args.kwarg):
        violations.append("fetch_raw accepts caller-controlled headers")
    nodes = _scope_nodes(function, function)
    requests = [node for node in nodes if isinstance(node, ast.Call)
                and _qualified_name(node.func).rsplit(".", 1)[-1] == "request"]
    if len(requests) != 1:
        violations.append("fetch_raw must build one bounded request")
    else:
        headers = [item.value for item in requests[0].keywords if item.arg == "headers"]
        if len(headers) != 1 or not isinstance(headers[0], ast.Dict):
            violations.append("request headers must be a literal mapping")
        else:
            keys = [key.value for key in headers[0].keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)]
            if (len(keys) != len(headers[0].keys) or len(keys) != len(_RAW_FETCH_HEADERS)
                    or set(keys) != _RAW_FETCH_HEADERS):
                violations.append("request header set differs from the fixed four headers")
    return tuple(violations)


def scan_fetch_raw_auth_boundary(source: str) -> tuple[str, ...]:
    """Follow local calls from fetch_raw and forbid both model-auth sources."""
    tree = ast.parse(source)
    root = _function(tree, "fetch_raw")
    if root is None:
        return ("fetch_raw definition missing or ambiguous",)
    imports = {
        (alias.asname or alias.name.split(".", 1)[0]): alias.name
        for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    }
    imports.update({
        (alias.asname or alias.name): f"{node.module}.{alias.name}"
        for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
        for alias in node.names
    })
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    module_scope = [node for node in ast.walk(tree) if _function_scope(node, parents) is None]
    functions = {node.name: node for node in ast.walk(tree)
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    reachable: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    pending = [root]
    visited: set[int] = set()
    while pending:
        function = pending.pop()
        if id(function) in visited:
            continue
        visited.add(id(function))
        reachable.append(function)
        pending.extend(node for node in ast.walk(function)
                       if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                       and node is not function)
        for node in _scope_nodes(function, function):
            if not isinstance(node, ast.Call):
                continue
            callee = _qualified_name(node.func).rsplit(".", 1)[-1]
            if callee in functions:
                pending.append(functions[callee])

    violations = []
    top_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    if any(_is_kernel_bounded_http(module) for node in top_imports for module in _import_modules(node)):
        violations.append("kernel_web_fetch imports kernel_bounded_http")
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "_headers":
            violations.append("kernel_web_fetch reads Endpoint._headers")
            break
        if isinstance(node, ast.Constant) and node.value == "_headers":
            violations.append("kernel_web_fetch names Endpoint._headers")
            break
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        scope = _function_scope(node, parents)
        scope_nodes = module_scope if scope is None else _scope_nodes(scope, tree)
        name = _resolved_reference(node.func, imports, scope_nodes, node.lineno)
        short = name.rsplit(".", 1)[-1]
        if short in {"__import__", "import_module"} and node.args:
            target = node.args[0]
            if (isinstance(target, ast.Constant) and isinstance(target.value, str)
                    and _is_kernel_bounded_http(target.value)):
                violations.append("kernel_web_fetch dynamically imports kernel_bounded_http")
                break
    for function in reachable:
        for node in _scope_nodes(function, function):
            if isinstance(node, (ast.Import, ast.ImportFrom)) and any(
                    _is_kernel_bounded_http(module) for module in _import_modules(node)):
                violations.append("fetch_raw call graph imports kernel_bounded_http")
                break
            if isinstance(node, ast.Call):
                name = _qualified_name(node.func)
                if _is_kernel_bounded_http(name) or "kernel_bounded_http." in name:
                    violations.append("fetch_raw call graph calls kernel_bounded_http")
                    break
    return tuple(dict.fromkeys(violations))


def scan_mutation_broker_environment(source: str) -> tuple[str, ...]:
    """Pin the broker's exact child environment and its handoff to ExecCapability."""
    tree = ast.parse(source)
    violations = []
    env_assignments = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "_ENV" for target in node.targets):
            env_assignments.append(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "_ENV":
            env_assignments.append(node.value)
    if len(env_assignments) != 1:
        violations.append("broker environment definition missing or ambiguous")
    else:
        try:
            environment = ast.literal_eval(env_assignments[0])
        except (ValueError, TypeError):
            environment = None
        if environment != _BROKER_ENV:
            violations.append("broker environment differs from its fixed allowlist")
    env_stores = [node for node in ast.walk(tree) if isinstance(node, ast.Name)
                  and node.id == "_ENV" and isinstance(node.ctx, ast.Store)]
    env_targets = {id(node) for node in env_stores}
    os_names = {alias.asname or alias.name.split(".", 1)[0]
                for node in ast.walk(tree) if isinstance(node, ast.Import)
                for alias in node.names if alias.name == "os"}
    environ_names = {alias.asname or alias.name
                     for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                     and node.module == "os" for alias in node.names
                     if alias.name in {"environ", "getenv", "getenvb"}}
    if any((isinstance(node, ast.Attribute) and node.attr in {"environ", "getenv", "getenvb"}
            and isinstance(node.value, ast.Name) and node.value.id in os_names)
           or (isinstance(node, ast.Name) and node.id in environ_names)
           for node in ast.walk(tree)):
        violations.append("broker environment reads ambient credentials through os.environ or getenv")
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and _qualified_name(node.func).rsplit(".", 1)[-1] == "ExecCapability"]
    if len(calls) != 1:
        violations.append("broker environment handoff missing or ambiguous")
    else:
        call = calls[0]
        environment = call.args[6] if len(call.args) > 6 else next(
            (item.value for item in call.keywords if item.arg == "env"), None)
        if not isinstance(environment, ast.Name) or environment.id != "_ENV":
            violations.append("ExecCapability must receive the fixed broker environment")
        allowed_env_refs = {id(environment)} if isinstance(environment, ast.Name) else set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Name) or node.id != "_ENV":
                continue
            if id(node) in env_targets or id(node) in allowed_env_refs:
                continue
            violations.append("broker fixed environment is read or mutated outside its handoff")
            break
    if len(env_stores) != 1:
        violations.append("broker fixed environment is rebound")
    return tuple(violations)


def scan_provider_fetch_raw(source: str) -> tuple[str, ...]:
    """Reject static and literal dynamic reachability to the raw user-web transport."""
    tree = ast.parse(source)
    violations = []
    imports = {
        (alias.asname or alias.name.split(".", 1)[0]): alias.name
        for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    }
    imports.update({
        (alias.asname or alias.name): f"{node.module}.{alias.name}"
        for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
        for alias in node.names
    })
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    module_scope = [node for node in ast.walk(tree) if _function_scope(node, parents) is None]
    for _ in range(len(module_scope)):
        changed = False
        for node in module_scope:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)) or not isinstance(
                    node.value, (ast.Name, ast.Attribute)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            resolved = _resolved_reference(node.value, imports, module_scope, node.lineno)
            if not (resolved == "kernel_web_fetch.fetch_raw"
                    or resolved.endswith(".kernel_web_fetch.fetch_raw")):
                continue
            for target in targets:
                if isinstance(target, ast.Name) and imports.get(target.id) != resolved:
                    imports[target.id] = resolved
                    changed = True
        if not changed:
            break
    for node in ast.walk(tree):
        for module in _import_modules(node):
            if module == "kernel_web_fetch" or module.endswith(".kernel_web_fetch"):
                violations.append("provider imports kernel_web_fetch")
                break
        if not isinstance(node, ast.Call):
            continue
        scope = _function_scope(node, parents)
        scope_nodes = module_scope if scope is None else _scope_nodes(scope, tree)
        name = _resolved_reference(node.func, imports, scope_nodes, node.lineno)
        short = name.rsplit(".", 1)[-1]
        if short == "fetch_raw":
            violations.append("provider calls fetch_raw")
        elif short in {"__import__", "import_module"} and node.args:
            target = node.args[0]
            if isinstance(target, ast.Constant) and isinstance(target.value, str):
                if target.value == "kernel_web_fetch" or target.value.endswith(".kernel_web_fetch"):
                    violations.append("provider dynamically imports kernel_web_fetch")
        elif short == "getattr" and len(node.args) > 1:
            target = node.args[1]
            if isinstance(target, ast.Constant) and target.value == "fetch_raw":
                violations.append("provider resolves fetch_raw dynamically")
    return tuple(dict.fromkeys(violations))
