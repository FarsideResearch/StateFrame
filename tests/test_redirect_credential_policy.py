"""Inventory and behavior guards for credentials carried through HTTP redirects."""
from __future__ import annotations

import ast
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

from stateframe import endpoint
from stateframe.engine import kernel_bounded_http, kernel_redirect_policy
import kernel_request_diet


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"

# Independently inventoried from the complete src/stateframe redirect-handler search.
EXPECTED_REDIRECT_SITES = frozenset({
    ("stateframe/endpoint.py", "_LocalRedirects"),
    ("stateframe/engine/kernel_bounded_http.py", "_RoutedRedirects"),
    ("stateframe/engine/kernel_request_diet.py", "_TokenizeRedirects"),
})

# Keep this inventory independent of the implementation's registry.
EXPECTED_SECRET_HEADERS = (
    "Authorization",
    "Proxy-Authorization",
    "Cookie",
    "Cookie2",
    "X-Api-Key",
    "Api-Key",
    "X-Auth-Token",
)

REDIRECT_HANDLERS = (
    (("stateframe/endpoint.py", "_LocalRedirects"), "endpoint probe",
     lambda: endpoint._LocalRedirects(allow_remote=True)),
    (("stateframe/engine/kernel_bounded_http.py", "_RoutedRedirects"), "bounded provider",
     lambda: kernel_bounded_http._RoutedRedirects(allow_remote=True)),
    (("stateframe/engine/kernel_request_diet.py", "_TokenizeRedirects"), "tokenizer request",
     kernel_request_diet._TokenizeRedirects),
)

EXPECTED_REDIRECT_STATUSES = (301, 302, 303, 304, 305, 306, 307, 308)
EXPECTED_REDIRECT_METHODS = ("GET", "HEAD", "POST", "PUT", "DELETE", "PATCH")
EXPECTED_ORIGIN_CASES = (
    ("same-host-default-port", "http://a.example.test/start",
     "http://A.example.test:80/next", True),
    ("different-host", "http://a.example.test/start",
     "http://b.example.test/next", False),
    ("scheme-downgrade", "https://a.example.test/start",
     "http://a.example.test/next", False),
    ("ipv6-same-origin", "https://[::1]/start",
     "https://[0:0:0:0:0:0:0:1]:443/next", True),
    ("ipv6-different-port", "https://[::1]:8443/start",
     "https://[0:0:0:0:0:0:0:1]:8444/next", False),
)
EXPECTED_REDIRECT_BEHAVIOR = frozenset(
    (site, status, method, origin_case)
    for site in EXPECTED_REDIRECT_SITES
    for status in EXPECTED_REDIRECT_STATUSES
    for method in EXPECTED_REDIRECT_METHODS
    for origin_case, _source, _target, _same_origin in EXPECTED_ORIGIN_CASES
)


@dataclass(frozen=True)
class _ClassRecord:
    key: tuple[str, str]
    bases: tuple[str, ...]
    redirect_methods: tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...]
    aliases: dict[str, str]


def _expression_name(node: ast.AST, aliases: dict[str, str], module: str) -> str | None:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, f"{module}.{node.id}")
    if isinstance(node, ast.Attribute):
        parent = _expression_name(node.value, aliases, module)
        return f"{parent}.{node.attr}" if parent else None
    return None


def _import_aliases(tree: ast.Module) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                aliases[imported.asname or imported.name.split(".", 1)[0]] = (
                    imported.name if imported.asname else imported.name.split(".", 1)[0]
                )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                continue
            for imported in node.names:
                aliases[imported.asname or imported.name] = f"{module}.{imported.name}"
    return aliases


def _class_records(source_root: Path) -> tuple[_ClassRecord, ...]:
    records: list[_ClassRecord] = []
    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module = path.relative_to(source_root).with_suffix("").as_posix().replace("/", ".")
        aliases = _import_aliases(tree)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.AST):
                continue
            target_names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if _expression_name(node.value, aliases, module) == "urllib.request.HTTPRedirectHandler":
                for name in target_names:
                    aliases[name] = "urllib.request.HTTPRedirectHandler"

        class Collector(ast.NodeVisitor):
            def __init__(self) -> None:
                self.parents: list[str] = []

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                qualname = ".".join((*self.parents, node.name))
                bases = tuple(
                    name for base in node.bases
                    if (name := _expression_name(base, aliases, module)) is not None
                )
                redirect_methods = tuple(
                    child for child in node.body
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and child.name == "redirect_request"
                )
                records.append(_ClassRecord(
                    (path.relative_to(source_root).as_posix(), qualname),
                    bases,
                    redirect_methods,
                    aliases,
                ))
                self.parents.append(node.name)
                self.generic_visit(node)
                self.parents.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self.parents.append(node.name)
                self.generic_visit(node)
                self.parents.pop()

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self.parents.append(node.name)
                self.generic_visit(node)
                self.parents.pop()

        Collector().visit(tree)
    return tuple(records)


def _redirect_site_records(source_root: Path) -> tuple[_ClassRecord, ...]:
    records = _class_records(source_root)
    qualified_names = {
        ".".join(record.key[0].removesuffix(".py").split("/")) + f".{record.key[1]}": record
        for record in records
    }
    redirect_subclasses = {
        name for name, record in qualified_names.items()
        if "urllib.request.HTTPRedirectHandler" in record.bases
    }
    changed = True
    while changed:
        changed = False
        for name, record in qualified_names.items():
            if name in redirect_subclasses:
                continue
            if any(base in redirect_subclasses for base in record.bases):
                redirect_subclasses.add(name)
                changed = True
    return tuple(record for name, record in qualified_names.items()
                 if name in redirect_subclasses or record.redirect_methods)


def _control_context(
        node: ast.AST, parents: dict[ast.AST, ast.AST],
) -> tuple[tuple[ast.AST, str | None], ...]:
    controls = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith,
                ast.Match)
    found: list[tuple[ast.AST, str | None]] = []
    child = node
    current = parents.get(node)
    while current is not None:
        if isinstance(current, controls):
            branch = None
            if isinstance(current, ast.If):
                if child in current.body:
                    branch = "body"
                elif child in current.orelse:
                    branch = "orelse"
                else:
                    branch = "test"
            found.append((current, branch))
        child = current
        current = parents.get(current)
    return tuple(found)


def _routes_policy(method: ast.FunctionDef | ast.AsyncFunctionDef,
                   aliases: dict[str, str]) -> bool:
    parents = {child: parent for parent in ast.walk(method) for child in ast.iter_child_nodes(parent)}
    calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)]
    redirect_assignments = []
    policy_calls = []
    direct_redirect_returns = []
    returned_requests = []

    for call in calls:
        function_name = _expression_name(call.func, aliases, "")
        if (isinstance(call.func, ast.Attribute)
                and call.func.attr == "redirect_request"
                and isinstance(call.func.value, ast.Call)
                and isinstance(call.func.value.func, ast.Name)
                and call.func.value.func.id == "super"):
            parent = parents.get(call)
            if isinstance(parent, (ast.Assign, ast.AnnAssign)):
                targets = parent.targets if isinstance(parent, ast.Assign) else [parent.target]
                if any(isinstance(target, ast.Name) and target.id == "redirected"
                       for target in targets):
                    redirect_assignments.append((call, parent))
        if (function_name is not None
                and function_name.rsplit(".", 1)[-1] == "scope_redirect_credentials"
                and call.args and isinstance(call.args[0], ast.Name)
                and call.args[0].id == "redirected"):
            keywords = {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg}
            secret_headers = keywords.get("secret_headers")
            if ("credential_origin" in keywords and isinstance(secret_headers, ast.Name)
                    and aliases.get(secret_headers.id, secret_headers.id).endswith(
                        "REDIRECT_SECRET_HEADERS")):
                policy_calls.append(call)

    for node in ast.walk(method):
        if not isinstance(node, ast.Return):
            continue
        if isinstance(node.value, ast.Name) and node.value.id == "redirected":
            returned_requests.append(node)
        if (isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "redirect_request"
                and isinstance(node.value.func.value, ast.Call)
                and isinstance(node.value.func.value.func, ast.Name)
                and node.value.func.value.func.id == "super"):
            direct_redirect_returns.append(node)

    if not redirect_assignments or not policy_calls or not returned_requests or direct_redirect_returns:
        return False
    redirect_assignment_lines = [call.lineno for call, _parent in redirect_assignments]
    for returned in returned_requests:
        if not any(
                assignment_line < call.lineno < returned.lineno
                and _control_context(call, parents) == _control_context(returned, parents)
                for assignment_line in redirect_assignment_lines
                for call in policy_calls):
            return False
    return True


def _inventory_violations(source_root: Path) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    scanned = _redirect_site_records(source_root)
    scanned_sites = {record.key for record in scanned}
    violated = EXPECTED_REDIRECT_SITES ^ scanned_sites
    for record in scanned:
        if record.key not in EXPECTED_REDIRECT_SITES:
            continue
        if len(record.redirect_methods) != 1 or not _routes_policy(
                record.redirect_methods[0], record.aliases):
            violated.add(record.key)
    return scanned_sites, violated


def test_redirect_site_inventory_is_complete_and_routes_through_policy() -> None:
    scanned, violated = _inventory_violations(SOURCE_ROOT)
    summary = (f"GUARD redirect_credentials: expected={len(EXPECTED_REDIRECT_SITES)} "
               f"scanned={len(scanned)} violated={len(violated)}")
    print(summary, flush=True)
    print(f"SCANNED redirect_credentials_sites: {sorted(scanned)}", flush=True)
    details = f"{summary}; scanned_sites={sorted(scanned)} violated_sites={sorted(violated)}"
    assert len(EXPECTED_REDIRECT_SITES) == len(scanned), details
    assert not violated, details


def test_site_guard_rejects_a_decoy_policy_call_beside_an_unsafe_send() -> None:
    tree = ast.parse("""
def redirect_request(self, req, fp, code, msg, headers, newurl):
    scope_redirect_credentials(req, credential_origin=req.full_url,
                               secret_headers=REDIRECT_SECRET_HEADERS)
    return super().redirect_request(req, fp, code, msg, headers, newurl)
""")
    method = tree.body[0]
    assert isinstance(method, ast.FunctionDef)
    assert not _routes_policy(method, {})


def test_site_guard_rejects_policy_call_in_opposite_if_branch() -> None:
    tree = ast.parse("""
def redirect_request(self, req, fp, code, msg, headers, newurl):
    redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
    if req.full_url.startswith("https://"):
        redirected = scope_redirect_credentials(
            redirected, credential_origin=req.full_url,
            secret_headers=REDIRECT_SECRET_HEADERS,
        )
    else:
        return redirected
""")
    method = tree.body[0]
    assert isinstance(method, ast.FunctionDef)
    assert not _routes_policy(method, {})


def _header_names(request: urllib.request.Request) -> set[str]:
    return {
        name.casefold()
        for collection in (request.headers, request.unredirected_hdrs)
        for name in collection
        if isinstance(name, str)
    }


def _request_with_credentials(url: str) -> urllib.request.Request:
    values = {name: f"secret-{name.casefold()}" for name in EXPECTED_SECRET_HEADERS}
    request = urllib.request.Request(url, headers=values)
    for name, value in values.items():
        request.add_unredirected_header(name, value)
    return request


def _redirect_copy(request: urllib.request.Request, target_url: str) -> urllib.request.Request:
    redirected = urllib.request.Request(target_url, headers=dict(request.headers))
    redirected.unredirected_hdrs.update(request.unredirected_hdrs)
    if hasattr(request, kernel_redirect_policy.REDIRECT_CREDENTIAL_ORIGIN_ATTRIBUTE):
        setattr(
            redirected,
            kernel_redirect_policy.REDIRECT_CREDENTIAL_ORIGIN_ATTRIBUTE,
            getattr(request, kernel_redirect_policy.REDIRECT_CREDENTIAL_ORIGIN_ATTRIBUTE),
        )
    return redirected


def _assert_headers(request: urllib.request.Request, *, present: bool) -> None:
    found = _header_names(request)
    expected = {name.casefold() for name in EXPECTED_SECRET_HEADERS}
    if present:
        assert expected <= found, f"secret header missing: {sorted(expected - found)}"
    else:
        assert not (expected & found), f"secret headers remain: {sorted(expected & found)}"


def _redirect_outcome(
        handler: urllib.request.HTTPRedirectHandler,
        request: urllib.request.Request,
        status: int,
        target_url: str,
) -> tuple[tuple[object, ...], urllib.request.Request | None]:
    try:
        redirected = handler.redirect_request(
            request, None, status, "redirect", {"Location": target_url}, target_url,
        )
    except urllib.error.HTTPError as error:
        return ("http-error", error.code), None
    if redirected is None:
        return ("none",), None
    return ("request", redirected.get_method(), redirected.data), redirected


def _scan_redirect_behavior(handler_rows=REDIRECT_HANDLERS):
    expected = EXPECTED_REDIRECT_BEHAVIOR
    scanned = set()
    violations = []
    visited = 0
    for site, label, handler_factory in handler_rows:
        for status in EXPECTED_REDIRECT_STATUSES:
            for method in EXPECTED_REDIRECT_METHODS:
                for origin_case, source_url, target_url, same_origin in EXPECTED_ORIGIN_CASES:
                    case = (site, status, method, origin_case)
                    visited += 1
                    if case in scanned:
                        violations.append(f"{case}: duplicate scan via {label}")
                    scanned.add(case)

                    body = None if method in {"GET", "HEAD"} else f"body:{method}".encode()
                    base_request = _request_with_credentials(source_url)
                    base_request.data = body
                    base_request.method = method
                    try:
                        expected_result, _base_redirected = _redirect_outcome(
                            urllib.request.HTTPRedirectHandler(), base_request, status, target_url,
                        )
                    except Exception as error:
                        violations.append(f"{case}: base handler raised {type(error).__name__}")
                        continue

                    request = _request_with_credentials(source_url)
                    request.data = body
                    request.method = method
                    try:
                        actual_result, redirected = _redirect_outcome(
                            handler_factory(), request, status, target_url,
                        )
                    except Exception as error:
                        violations.append(f"{case}: raised {type(error).__name__}")
                        continue

                    if actual_result != expected_result:
                        violations.append(
                            f"{case}: outcome {actual_result!r} != base urllib outcome "
                            f"{expected_result!r}"
                        )
                    if expected_result[0] != "request" or actual_result[0] != "request":
                        continue
                    if redirected is None:
                        violations.append(f"{case}: follow outcome did not return a request")
                        continue
                    try:
                        _assert_headers(redirected, present=same_origin)
                    except AssertionError as error:
                        violations.append(f"{case}: {error}")
    return expected, scanned, visited, tuple(violations)


def _assert_behavior_guard(expected, scanned, visited, violations, *, name: str) -> None:
    summary = (f"GUARD {name}: expected={len(expected)} scanned={visited} "
               f"violated={len(violations)}")
    print(summary, flush=True)
    print(
        f"SCANNED {name}: {len(scanned)} unique cases; sites={len(EXPECTED_REDIRECT_SITES)} "
        f"statuses={len(EXPECTED_REDIRECT_STATUSES)} methods={len(EXPECTED_REDIRECT_METHODS)} "
        f"origins={len(EXPECTED_ORIGIN_CASES)}",
        flush=True,
    )
    details = f"{summary}; scanned_cases={len(scanned)} violations={violations[:8]!r}"
    assert expected == scanned, details
    assert visited == len(scanned), details
    assert not violations, details


def test_redirect_behavior_inventory_covers_status_method_and_origin_contracts() -> None:
    expected, scanned, visited, violations = _scan_redirect_behavior()
    assert len(expected) == 720, f"behavior inventory unexpectedly changed: {len(expected)}"
    _assert_behavior_guard(
        expected, scanned, visited, violations, name="redirect_credentials_behavior",
    )


class _BlanketRefusalRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def test_planted_blanket_refusal_control_is_rejected() -> None:
    routed_site = ("stateframe/engine/kernel_bounded_http.py", "_RoutedRedirects")
    planted_handlers = tuple(
        (site, label, _BlanketRefusalRedirects if site == routed_site else handler_factory)
        for site, label, handler_factory in REDIRECT_HANDLERS
    )
    expected, scanned, visited, violations = _scan_redirect_behavior(planted_handlers)
    assert expected == scanned
    assert visited == len(expected)
    assert violations, "planted blanket refusal was not detected"
    with pytest.raises(AssertionError, match="violated="):
        _assert_behavior_guard(
            expected, scanned, visited, violations,
            name="redirect_credentials_blanket_refusal_control",
        )


def _assert_policy_chain(policy) -> None:
    first = _request_with_credentials("http://a.example.test/start")
    same_origin = _redirect_copy(first, "http://A.example.test:80/same")
    policy(same_origin, credential_origin=first.full_url,
           secret_headers=EXPECTED_SECRET_HEADERS)
    _assert_headers(same_origin, present=True)

    credential_origin = getattr(
        same_origin,
        kernel_redirect_policy.REDIRECT_CREDENTIAL_ORIGIN_ATTRIBUTE,
        same_origin.full_url,
    )
    cross_origin = _redirect_copy(same_origin, "http://b.example.test/middle")
    policy(cross_origin, credential_origin=credential_origin,
           secret_headers=EXPECTED_SECRET_HEADERS)
    _assert_headers(cross_origin, present=False)

    for name in EXPECTED_SECRET_HEADERS:
        cross_origin.headers[name] = "reintroduced"
        cross_origin.unredirected_hdrs[name] = "reintroduced"
    credential_origin = getattr(
        cross_origin,
        kernel_redirect_policy.REDIRECT_CREDENTIAL_ORIGIN_ATTRIBUTE,
        cross_origin.full_url,
    )
    returned_to_origin = _redirect_copy(cross_origin, "http://a.example.test/final")
    policy(returned_to_origin, credential_origin=credential_origin,
           secret_headers=EXPECTED_SECRET_HEADERS)
    _assert_headers(returned_to_origin, present=False)


def test_shared_policy_rejects_credentials_in_both_header_stores_without_touching_body() -> None:
    request = _request_with_credentials("http://other.example.test/redirected")
    request.data = b"sealed request body"
    request.method = "POST"

    result = kernel_redirect_policy.scope_redirect_credentials(
        request,
        credential_origin="http://origin.example.test/start",
        secret_headers=EXPECTED_SECRET_HEADERS,
    )

    assert result is request
    _assert_headers(request, present=False)
    assert request.get_method() == "POST"
    assert request.data == b"sealed request body"


@pytest.mark.parametrize(("credential_origin", "target_url", "present"), [
    ("https://A.example.test:443/start", "https://a.example.test/redirected", True),
    ("https://a.example.test/start", "http://a.example.test/redirected", False),
    ("https://a.example.test/start", "https://b.example.test/redirected", False),
    ("https://a.example.test/start", "https://a.example.test:444/redirected", False),
])
def test_policy_compares_scheme_host_and_effective_port(
        credential_origin: str, target_url: str, present: bool) -> None:
    request = _request_with_credentials(target_url)
    kernel_redirect_policy.scope_redirect_credentials(
        request, credential_origin=credential_origin, secret_headers=EXPECTED_SECRET_HEADERS,
    )
    _assert_headers(request, present=present)


def _mutant_noop(request, *, credential_origin, secret_headers):
    return request


def _mutant_authorization_only(request, *, credential_origin, secret_headers):
    source = kernel_redirect_policy._origin(credential_origin)
    target = kernel_redirect_policy._origin(request.full_url)
    if source != target:
        for collection in (request.headers, request.unredirected_hdrs):
            for name in tuple(collection):
                if isinstance(name, str) and name.casefold() == "authorization":
                    del collection[name]
        setattr(request, kernel_redirect_policy.REDIRECT_CREDENTIAL_ORIGIN_ATTRIBUTE, None)
    return request


def _mutant_first_hop_only(request, *, credential_origin, secret_headers):
    source = kernel_redirect_policy._origin(credential_origin)
    target = kernel_redirect_policy._origin(request.full_url)
    if source != target:
        for collection in (request.headers, request.unredirected_hdrs):
            for name in tuple(collection):
                if isinstance(name, str) and name.casefold() in {
                    header.casefold() for header in secret_headers
                }:
                    del collection[name]
    setattr(request, kernel_redirect_policy.REDIRECT_CREDENTIAL_ORIGIN_ATTRIBUTE, source)
    return request


def _mutant_reinjection(request, *, credential_origin, secret_headers):
    kernel_redirect_policy.scope_redirect_credentials(
        request, credential_origin=credential_origin, secret_headers=secret_headers,
    )
    request.headers["X-Api-Key"] = "reintroduced"
    return request


@pytest.mark.parametrize(("name", "mutant"), [
    ("no-op", _mutant_noop),
    ("Authorization-only", _mutant_authorization_only),
    ("first-hop-only", _mutant_first_hop_only),
    ("reinjection", _mutant_reinjection),
])
def test_planted_policy_controls_are_rejected(name, mutant) -> None:
    with pytest.raises(AssertionError, match="secret header"):
        _assert_policy_chain(mutant)


class _RedirectServerHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args) -> None:
        pass

    def _handle_request(self) -> None:
        state = self.server.redirect_state
        body_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(body_length) if body_length else b""
        if self.path == "/late/start":
            state["late_redirect_arrived_at"] = time.monotonic()
        record = {
            name.casefold(): tuple(self.headers.get_all(name, ()))
            for name in EXPECTED_SECRET_HEADERS
        }
        state["records"][(self.server.server_port, self.path)] = record
        state["requests"][(self.server.server_port, self.path)] = {
            "method": self.command,
            "body": body,
            "authorization": self.headers.get("Authorization"),
        }

        if self.path == "/cross/start":
            status = 302
            location = state["b_url"] + "/cross/middle"
        elif self.path == "/cross/middle":
            status = 302
            location = state["a_url"] + "/cross/final"
        elif self.path == "/same/start":
            status = 302
            location = state["a_url"] + "/same/final"
        elif self.path == "/child/start":
            status = 302
            location = state["b_url"] + "/child/final"
        elif self.path == "/late/start":
            status = 302
            location = state["b_url"] + "/late/follow"
            while time.monotonic() < state["late_redirect_release_at"]:
                time.sleep(0.01)
            state["late_redirect_sent_at"] = time.monotonic()
        else:
            status = 200
            location = None

        response_body = b"ok" if self.path == "/child/final" else b""
        self.send_response(status)
        if location is not None:
            self.send_header("Location", location)
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        if response_body:
            self.wfile.write(response_body)

    def do_GET(self) -> None:  # noqa: N802
        self._handle_request()

    def do_POST(self) -> None:  # noqa: N802
        self._handle_request()


class _RedirectServers:
    def __enter__(self):
        self.state = {"records": {}, "requests": {}}
        self.a = ThreadingHTTPServer(("127.0.0.1", 0), _RedirectServerHandler)
        self.b = ThreadingHTTPServer(("127.0.0.1", 0), _RedirectServerHandler)
        self.a.redirect_state = self.state
        self.b.redirect_state = self.state
        self.state["a_url"] = f"http://127.0.0.1:{self.a.server_port}"
        self.state["b_url"] = f"http://127.0.0.1:{self.b.server_port}"
        self.threads = [threading.Thread(target=server.serve_forever, daemon=True)
                        for server in (self.a, self.b)]
        for thread in self.threads:
            thread.start()
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        for server in (self.a, self.b):
            server.shutdown()
            server.server_close()
        for thread in self.threads:
            thread.join(timeout=5)
            assert not thread.is_alive(), "redirect server did not stop"


def _open(opener: urllib.request.OpenerDirector, request: urllib.request.Request) -> None:
    with opener.open(request, timeout=5) as response:
        assert response.status == 200
        assert response.read() == b""


@pytest.mark.host_sandbox
@pytest.mark.parametrize(
    ("site", "handler_factory"),
    tuple((label, handler_factory) for _site, label, handler_factory in REDIRECT_HANDLERS),
)
def test_redirect_sites_strip_secrets_across_origins_and_keep_same_origin(site, handler_factory) -> None:
    with _RedirectServers() as servers:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), handler_factory(),
        )
        _open(opener, _request_with_credentials(servers.state["a_url"] + "/cross/start"))

        records = servers.state["records"]
        first = records[(servers.a.server_port, "/cross/start")]
        middle = records[(servers.b.server_port, "/cross/middle")]
        returned = records[(servers.a.server_port, "/cross/final")]
        for name in EXPECTED_SECRET_HEADERS:
            key = name.casefold()
            assert first[key], f"{site} did not send the initial {name} header"
            assert not middle[key], f"{site} forwarded {name} to the other origin"
            assert not returned[key], f"{site} restored {name} on a later redirect hop"

        _open(opener, _request_with_credentials(servers.state["a_url"] + "/same/start"))
        same_source = records[(servers.a.server_port, "/same/start")]
        same_target = records[(servers.a.server_port, "/same/final")]
        for name in EXPECTED_SECRET_HEADERS:
            key = name.casefold()
            assert same_source[key], f"{site} did not send initial same-origin {name}"
            assert same_target[key], f"{site} stripped same-origin {name}"


@pytest.mark.host_sandbox
def test_routed_redirect_policy_imports_in_isolated_http_child() -> None:
    wire = b'{"model":"redirect-child","messages":[]}'
    with _RedirectServers() as servers:
        deadline = time.monotonic() + 10.0
        packet = json.dumps({
            "wire": wire.decode("utf-8"),
            "endpoint": servers.state["a_url"] + "/child/start",
            "key": "fixture-key",
            "deadline": deadline,
            "request_deadline": deadline,
            "socket_timeout": 3.0,
            "send_stall_seconds": kernel_bounded_http.send_stall_seconds(),
            "report_body_sent": True,
            "gzip": False,
            "route_allow_remote": False,
        }).encode("utf-8")
        child = subprocess.run(
            [sys.executable, "-I", "-B", str(Path(kernel_bounded_http.__file__).resolve()),
             "--http-child"],
            input=packet,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=12.0,
            check=False,
        )

        first = servers.state["requests"][(servers.a.server_port, "/child/start")]
        redirected = servers.state["requests"][(servers.b.server_port, "/child/final")]

    assert child.returncode == 0, child.stderr.decode("utf-8", errors="replace")
    assert child.stdout == b"ok"
    assert first == {
        "method": "POST",
        "body": wire,
        "authorization": "Bearer fixture-key",
    }
    assert redirected == {
        "method": "GET",
        "body": b"",
        "authorization": None,
    }


@pytest.mark.host_sandbox
def test_routed_redirect_after_request_deadline_does_not_start_followup() -> None:
    wire = b'{"model":"late-redirect","messages":[]}'
    with _RedirectServers() as servers:
        request_deadline = time.monotonic() + 2.0
        deadline = request_deadline + 5.0
        servers.state["late_redirect_release_at"] = request_deadline + 0.05
        packet = json.dumps({
            "wire": wire.decode("utf-8"),
            "endpoint": servers.state["a_url"] + "/late/start",
            "key": "fixture-key",
            "deadline": deadline,
            "request_deadline": request_deadline,
            "socket_timeout": 1.0,
            "send_stall_seconds": kernel_bounded_http.send_stall_seconds(),
            "report_body_sent": True,
            "gzip": False,
            "route_allow_remote": False,
        }).encode("utf-8")
        child = subprocess.run(
            [sys.executable, "-I", "-B", str(Path(kernel_bounded_http.__file__).resolve()),
             "--http-child"],
            input=packet,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10.0,
            check=False,
        )

        first = servers.state["requests"].get((servers.a.server_port, "/late/start"))
        followup = servers.state["requests"].get((servers.b.server_port, "/late/follow"))
        arrived_at = servers.state.get("late_redirect_arrived_at")
        redirected_at = servers.state.get("late_redirect_sent_at")

    assert first == {
        "method": "POST",
        "body": wire,
        "authorization": "Bearer fixture-key",
    }
    assert arrived_at is not None and arrived_at < request_deadline
    assert redirected_at is not None and redirected_at > request_deadline
    assert followup is None, "a redirect after request_deadline started another request"
    assert child.returncode == 1
    assert child.stdout == b""
    diagnostic = json.loads(child.stderr.splitlines()[-1])
    assert diagnostic["error_type"] == "RequestBodyDeadlineTimeout"
