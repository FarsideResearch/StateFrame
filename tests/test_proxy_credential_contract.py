"""Inventory and behavior checks for the shared HTTP proxy credential contract."""
from __future__ import annotations

import ast
import base64
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import urllib.request

import pytest

from stateframe import _child
from stateframe.engine.kernel_redirect_policy import build_credential_scoped_opener


ROOT = Path(__file__).resolve().parents[1] / "src" / "stateframe"
HELPER = "engine/kernel_redirect_policy.py"
LEGACY = "engine/kernel_request_diet.py"  # Frozen tokenizer path; explicit direct routing stays intact.
# Independently inventoried by a source-wide helper/proxy constructor search and call-site review.
EXPECTED_SITES = {
    ("_child.py", "_install_direct_opener", "helper"),
    ("endpoint.py", "get_json", "helper"),
    ("endpoint.py", "get_slots", "helper"),
    ("engine/kernel_bounded_http.py", "_child", "helper"),
    ("engine/kernel_plain_http.py", "opener", "helper"),
    (HELPER, "build_credential_scoped_opener", "build_opener"),
    (HELPER, "build_credential_scoped_opener", "ProxyHandler"),
    (HELPER, "build_credential_scoped_opener", "_ProxyHeaderScrubber"),
    (HELPER, "build_credential_scoped_opener", "_CredentialScopedProxyHandler"),
    (LEGACY, "_build_tokenize_opener", "build_opener"),
    (LEGACY, "_build_tokenize_opener", "ProxyHandler"),
}
EXPECTED_MODES = {
    site: ("environment" if site[2] == "_CredentialScopedProxyHandler" else "direct")
    for site in EXPECTED_SITES
}


def _scope_import_aliases(nodes: list[ast.stmt]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    urllib_names = {
        "ProxyHandler", "OpenerDirector", "getproxies", "getproxies_environment", "build_opener",
    }
    policy_names = {"build_credential_scoped_opener", "_CredentialScopedProxyHandler"}

    class Collector(ast.NodeVisitor):
        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            module = node.module or ""
            for alias in node.names:
                is_urllib_api = module == "urllib.request" and alias.name in urllib_names
                is_policy_api = module.endswith("kernel_redirect_policy") and alias.name in policy_names
                if is_urllib_api or is_policy_api:
                    aliases[alias.asname or alias.name] = alias.name

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return

    collector = Collector()
    for node in nodes:
        collector.visit(node)
    return aliases


def _calls(path: str, source: str) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Find every opener and proxy constructor, including calls nested in functions."""
    tree = ast.parse(source)
    found: list[tuple[str, str, str]] = []
    violations: list[str] = []
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    aliases = _scope_import_aliases(tree.body)

    def under_environment_guard(node: ast.AST) -> bool:
        child = node
        parent = parents.get(child)
        while parent is not None:
            if (isinstance(parent, ast.If)
                    and isinstance(parent.test, ast.Name)
                    and parent.test.id == "use_environment"):
                return child in parent.body
            child = parent
            parent = parents.get(parent)
        return False

    class Visitor(ast.NodeVisitor):
        function = "<module>"

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            previous = self.function
            previous_aliases = self.aliases
            self.function = node.name
            self.aliases = {**previous_aliases, **_scope_import_aliases(node.body)}
            self.generic_visit(node)
            self.function = previous
            self.aliases = previous_aliases

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            previous_aliases = self.aliases
            self.aliases = {**previous_aliases, **_scope_import_aliases(node.body)}
            self.generic_visit(node)
            self.aliases = previous_aliases

        def __init__(self) -> None:
            self.aliases = aliases

        def visit_Call(self, node: ast.Call) -> None:
            name = self.aliases.get(node.func.id, node.func.id) if isinstance(node.func, ast.Name) else (
                node.func.attr if isinstance(node.func, ast.Attribute) else None
            )
            kind = "helper" if name == "build_credential_scoped_opener" else name
            if kind in {"getproxies", "getproxies_environment"}:
                kind = "proxy_environment_read"
            if kind in {"helper", "build_opener", "ProxyHandler",
                        "_ProxyHeaderScrubber", "_CredentialScopedProxyHandler", "OpenerDirector",
                        "proxy_environment_read"}:
                site = (path, self.function, kind)
                found.append(site)
                if site not in EXPECTED_SITES:
                    violations.append(f"{path}:{node.lineno}: unexpected {kind}")
                if kind == "helper":
                    mode = next((keyword.value.value for keyword in node.keywords
                                 if keyword.arg == "proxy_mode"
                                 and isinstance(keyword.value, ast.Constant)), None)
                    if mode != EXPECTED_MODES.get(site):
                        violations.append(f"{path}:{node.lineno}: wrong proxy mode {mode!r}")
                if kind == "ProxyHandler":
                    if (len(node.args) != 1 or node.keywords
                            or not isinstance(node.args[0], ast.Dict)
                            or node.args[0].keys):
                        violations.append(
                            f"{path}:{node.lineno}: ProxyHandler default reads the proxy environment"
                        )
                if kind == "proxy_environment_read":
                    violations.append(
                        f"{path}:{node.lineno}: unapproved proxy environment read"
                    )
                if (kind == "_CredentialScopedProxyHandler"
                        and not under_environment_guard(node)):
                    violations.append(
                        f"{path}:{node.lineno}: environment proxy handler must remain behind use_environment"
                    )
                if kind == "_ProxyHeaderScrubber" and under_environment_guard(node):
                    violations.append(
                        f"{path}:{node.lineno}: proxy header scrubber must cover direct mode"
                    )
            self.generic_visit(node)

    Visitor().visit(tree)
    return found, violations


def _inventory(sources: dict[str, str]) -> tuple[list[tuple[str, str, str]], list[str]]:
    scanned: list[tuple[str, str, str]] = []
    violated: list[str] = []
    for path, source in sources.items():
        sites, errors = _calls(path, source)
        scanned.extend(sites)
        violated.extend(errors)
    violated.extend(f"missing {site}" for site in sorted(EXPECTED_SITES - set(scanned)))
    violated.extend(f"duplicate {site}" for site, count in Counter(scanned).items()
                    if count > 1)
    return scanned, violated


def test_proxy_contract_source_inventory() -> None:
    sources = {path.relative_to(ROOT).as_posix(): path.read_text()
               for path in ROOT.rglob("*.py")}
    scanned, violated = _inventory(sources)
    summary = (f"GUARD proxy_contract: expected={len(EXPECTED_SITES)} "
               f"scanned={len(scanned)} violated={len(violated)}")
    print(summary, flush=True)
    print(f"SCANNED proxy_contract_sites: {sorted(scanned)}", flush=True)
    details = f"{summary}; scanned_sites={sorted(scanned)} violations={violated}"
    assert len(scanned) == len(EXPECTED_SITES), details
    assert set(scanned) == EXPECTED_SITES, details
    assert not violated, details
    assert set(EXPECTED_MODES) == EXPECTED_SITES


def test_proxy_inventory_rejects_planted_adhoc_handler() -> None:
    source = "def other():\n    urllib.request.ProxyHandler({})\n"
    _, violated = _calls("endpoint.py", source)
    assert any("unexpected ProxyHandler" in message for message in violated)
    alias_source = "from urllib.request import ProxyHandler as PH\ndef other():\n    PH({})\n"
    _, alias_violated = _calls("endpoint.py", alias_source)
    assert any("unexpected ProxyHandler" in message for message in alias_violated)
    nested_alias_source = (
        "def other():\n"
        "    from urllib.request import ProxyHandler as PH\n"
        "    PH({})\n"
    )
    _, nested_alias_violated = _calls("endpoint.py", nested_alias_source)
    assert any("unexpected ProxyHandler" in message for message in nested_alias_violated)


def test_proxy_inventory_rejects_direct_environment_discovery() -> None:
    source = """
def build_credential_scoped_opener(proxy_mode):
    if proxy_mode == "direct":
        from urllib.request import getproxies as read_proxies
        read_proxies()
"""
    _, violated = _calls(HELPER, source)
    assert any("unapproved proxy environment read" in message for message in violated)

    unguarded_handler = """
def build_credential_scoped_opener(proxy_mode):
    if proxy_mode == "direct":
        _CredentialScopedProxyHandler(origin)
"""
    _, handler_violated = _calls(HELPER, unguarded_handler)
    assert any("must remain behind use_environment" in message
               for message in handler_violated)

    helper_source = (ROOT / HELPER).read_text()
    constructor = "urllib.request.ProxyHandler({})"
    assert helper_source.count(constructor) == 1
    planted = helper_source.replace(constructor, "urllib.request.ProxyHandler()", 1)
    _, handler_violations = _calls(HELPER, planted)
    assert any("ProxyHandler default reads the proxy environment" in message
               for message in handler_violations)

    scrubber_call = "    opener.add_handler(_ProxyHeaderScrubber())"
    assert helper_source.count(scrubber_call) == 1
    scoped_scrubber = helper_source.replace(
        scrubber_call,
        "    if use_environment:\n        opener.add_handler(_ProxyHeaderScrubber())",
        1,
    )
    _, scrubber_violations = _calls(HELPER, scoped_scrubber)
    assert any("scrubber must cover direct mode" in message
               for message in scrubber_violations)


class _Server(BaseHTTPRequestHandler):
    def log_message(self, *_args: object) -> None:
        pass

    def do_GET(self) -> None:  # noqa: N802
        self.server.records.append((self.path, dict(self.headers)))
        if self.server.redirect:
            self.send_response(302)
            self.send_header("Location", self.server.redirect_target)
            self.send_header("Content-Length", "0")
            self.end_headers()
        else:
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")


class _Servers:
    def __enter__(self):
        self.origin = ThreadingHTTPServer(("127.0.0.1", 0), _Server)
        self.proxy = ThreadingHTTPServer(("127.0.0.1", 0), _Server)
        for server in (self.origin, self.proxy):
            server.records = []
            server.redirect = False
            server.redirect_target = "http://other.example.test/final"
        self.threads = [threading.Thread(target=server.serve_forever, daemon=True)
                        for server in (self.origin, self.proxy)]
        for thread in self.threads:
            thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        for server in (self.origin, self.proxy):
            server.shutdown()
            server.server_close()
        for thread in self.threads:
            thread.join(timeout=5)

    @property
    def origin_url(self) -> str:
        return f"http://127.0.0.1:{self.origin.server_port}"

    @property
    def proxy_url(self) -> str:
        return f"http://user:secret@127.0.0.1:{self.proxy.server_port}"


def _proxy_environment(monkeypatch, proxy_url: str) -> None:
    for name in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy",
                 "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("http_proxy", proxy_url)


def _assert_proxy_authorization_absent(headers: dict[str, str]) -> None:
    assert not any(name.casefold() == "proxy-authorization" for name in headers), (
        "Proxy-Authorization was sent after redirect stripping"
    )


@pytest.mark.host_sandbox
def test_direct_mode_ignores_proxy_environment(monkeypatch) -> None:
    with _Servers() as servers:
        _proxy_environment(monkeypatch, servers.proxy_url)
        monkeypatch.setattr(urllib.request, "getproxies", lambda: pytest.fail("read proxy env"))
        opener = build_credential_scoped_opener(proxy_mode="direct")
        with opener.open(servers.origin_url + "/direct", timeout=5) as response:
            assert response.read() == b"ok"
        assert len(servers.origin.records) == 1
        assert servers.proxy.records == []


@pytest.mark.host_sandbox
def test_environment_mode_uses_proxy_and_honors_no_proxy(monkeypatch) -> None:
    with _Servers() as servers:
        _proxy_environment(monkeypatch, servers.proxy_url)
        proxied_url = "http://origin.example.test/proxied"
        opener = build_credential_scoped_opener(
            proxy_mode="environment", credential_origin=proxied_url,
        )
        with opener.open(proxied_url, timeout=5) as response:
            assert response.read() == b"ok"
        assert len(servers.proxy.records) == 1
        assert servers.origin.records == []

        monkeypatch.setenv("no_proxy", "127.0.0.1")
        bypass = build_credential_scoped_opener(
            proxy_mode="environment", credential_origin=servers.origin_url,
        )
        request = urllib.request.Request(
            servers.origin_url + "/bypass", headers={"Proxy-Authorization": "bad"},
        )
        with bypass.open(request, timeout=5) as response:
            assert response.read() == b"ok"
        assert len(servers.origin.records) == 1
        _assert_proxy_authorization_absent(servers.origin.records[0][1])
        assert len(servers.proxy.records) == 1


@pytest.mark.host_sandbox
def test_direct_scrubber_preserves_default_request_headers(monkeypatch) -> None:
    with _Servers() as servers:
        _proxy_environment(monkeypatch, servers.proxy_url)
        scoped = build_credential_scoped_opener(proxy_mode="direct")
        baseline = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for opener in (scoped, baseline):
            with opener.open(servers.origin_url + "/default", timeout=5) as response:
                assert response.read() == b"ok"
        assert len(servers.origin.records) == 2
        assert servers.origin.records[0][1] == servers.origin.records[1][1]
        assert servers.proxy.records == []


@pytest.mark.host_sandbox
def test_direct_mode_scrubs_caller_proxy_authorization(monkeypatch) -> None:
    with _Servers() as servers:
        _proxy_environment(monkeypatch, servers.proxy_url)
        opener = build_credential_scoped_opener(proxy_mode="direct")
        request = urllib.request.Request(
            servers.origin_url + "/direct-header",
            headers={"Proxy-Authorization": "Basic proxy-only-secret"},
        )
        with opener.open(request, timeout=5) as response:
            assert response.read() == b"ok"
        assert len(servers.origin.records) == 1
        _assert_proxy_authorization_absent(servers.origin.records[0][1])
        assert servers.proxy.records == []


@pytest.mark.host_sandbox
def test_proxy_credentials_and_origin_secrets_stop_after_cross_origin_redirect(monkeypatch) -> None:
    with _Servers() as servers:
        _proxy_environment(monkeypatch, servers.proxy_url)
        origin_url = "http://origin.example.test/start"
        opener = build_credential_scoped_opener(
            proxy_mode="environment", credential_origin=origin_url,
        )
        request = urllib.request.Request(origin_url, headers={
            "Authorization": "Bearer origin-secret", "Cookie": "session=origin-secret",
            "X-Api-Key": "origin-secret",
        })
        # Only the first proxy response redirects; the second request reaches the same proxy.
        class RedirectOnce(_Server):
            def do_GET(self) -> None:  # noqa: N802
                if not self.server.records:
                    self.server.redirect = True
                else:
                    self.server.redirect = False
                super().do_GET()
        servers.proxy.RequestHandlerClass = RedirectOnce
        with opener.open(request, timeout=5) as response:
            assert response.read() == b"ok"
        first, second = servers.proxy.records
        expected = "Basic " + base64.b64encode(b"user:secret").decode("ascii")
        assert first[1]["Proxy-Authorization"] == expected
        assert first[1]["Authorization"] == "Bearer origin-secret"
        assert second[0] == "http://other.example.test/final"
        _assert_proxy_authorization_absent(second[1])
        for name in ("Proxy-Authorization", "Authorization", "Cookie", "X-Api-Key"):
            assert name not in second[1]


@pytest.mark.host_sandbox
def test_proxy_reinjection_control_after_redirect_stripping(monkeypatch) -> None:
    from stateframe.engine import kernel_redirect_policy

    monkeypatch.setattr(kernel_redirect_policy, "_proxy_without_credentials", lambda proxy: proxy)
    with _Servers() as servers:
        _proxy_environment(monkeypatch, servers.proxy_url)
        origin_url = "http://origin.example.test/start"
        opener = build_credential_scoped_opener(
            proxy_mode="environment", credential_origin=origin_url,
        )

        class RedirectOnce(_Server):
            def do_GET(self) -> None:  # noqa: N802
                self.server.redirect = not self.server.records
                super().do_GET()

        servers.proxy.RequestHandlerClass = RedirectOnce
        with opener.open(origin_url, timeout=5) as response:
            assert response.read() == b"ok"
        _first, second = servers.proxy.records
        assert second[0] == "http://other.example.test/final"
        assert "Proxy-Authorization" in second[1]
        with pytest.raises(AssertionError, match="Proxy-Authorization"):
            _assert_proxy_authorization_absent(second[1])


@pytest.mark.host_sandbox
def test_child_opener_forces_direct_helper_and_remains_installed(monkeypatch) -> None:
    with _Servers() as servers:
        _proxy_environment(monkeypatch, servers.proxy_url)
        previous = urllib.request._opener
        try:
            _child._install_direct_opener()
            child_opener = urllib.request._opener
            monkeypatch.setattr(urllib.request, "getproxies", lambda: pytest.fail("read proxy env"))
            helper = build_credential_scoped_opener(
                proxy_mode="environment", credential_origin=servers.origin_url,
            )
            assert urllib.request._opener is child_opener
            with helper.open(servers.origin_url + "/child", timeout=5) as response:
                assert response.read() == b"ok"
        finally:
            urllib.request._opener = previous
        assert len(servers.origin.records) == 1
        assert servers.proxy.records == []
