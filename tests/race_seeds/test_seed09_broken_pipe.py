"""Seed 09: a client that disconnects before reading the response must not register as an endpoint error.

Ported from the y4race harness (seeds/seed09_broken_pipe.py + mutants/seed09_no_swallow.py @ 9c6545d). Drives the
real BarrierEndpoint from the product fixture (tests/fixtures/barrier_endpoint.py). The parent request is held
inside the endpoint until the client has sent the full request and reset the connection (SO_LINGER 1,0). The
endpoint's response write therefore always lands on a dead socket, so the race is deterministic.

Mutant `no_swallow` (active only when ENGINECHECK_RACE_MUTANT=no_swallow): the endpoint stops tolerating client
disconnects. The mutated fixture is written into tmp_path and loaded from there for that test only; the product
file is never touched. Companion 09b is a plain test that must pass on HEAD with or without the mutant.
"""
from __future__ import annotations

import importlib.util
import json
import os
import socket
import struct
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import ModuleType

import pytest

import fixtures.barrier_endpoint as product_fixture

MUTANT_NAME = "no_swallow"
_MUTANT_ENV = "ENGINECHECK_RACE_MUTANT"
# Exact replacements that disable client-disconnect tolerance. Each must match exactly once in the fixture.
_SWALLOWS: tuple[tuple[str, str], ...] = (
    ("except (BrokenPipeError, ConnectionResetError) as error:", "except () as error:"),
    ("if isinstance(error, (BrokenPipeError, ConnectionResetError)):", "if isinstance(error, ()):"),
)

LOOPS = 20
SETTLE_SECONDS = 0.05
WAIT_SECONDS = 10.0
PARENT_SCRIPT: list[dict[str, object]] = [{"tool": "done", "arguments": {"summary": "scripted"}}]
TOOLS: list[dict[str, object]] = [{"type": "function", "function": {"name": "done", "parameters": {"type": "object"}}}]


def _mutate_fixture(source: str) -> str:
    """Apply each swallow replacement exactly once; refuse to run if the fixture drifted."""
    mutated = source
    for old, new in _SWALLOWS:
        count = mutated.count(old)
        if count != 1:
            raise ValueError(f"expected exactly one {old!r} in fixture, found {count}")
        mutated = mutated.replace(old, new)
    return mutated


def _endpoint_module(tmp_path: Path) -> ModuleType:
    """Return the product fixture, or the mutated copy when this seed's mutant is active.

    The mutant module is loaded from a spec, not registered in sys.modules, so nothing leaks past this call.
    """
    if os.environ.get(_MUTANT_ENV) != MUTANT_NAME:
        return product_fixture
    source = Path(product_fixture.__file__).read_text(encoding="utf-8")
    mutant_path = tmp_path / "barrier_endpoint.py"
    mutant_path.write_text(_mutate_fixture(source), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("barrier_endpoint", mutant_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load mutant fixture from {mutant_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _valid_body() -> bytes:
    """Shape the endpoint's do_POST accepts: object with messages list, tools offering the scripted step."""
    return json.dumps({
        "model": "scripted",
        "messages": [{"role": "user", "content": "Task: parent"}],
        "tools": TOOLS,
    }).encode("utf-8")


def _new_endpoint(module: ModuleType, *, hold: bool) -> object:
    hold_parent = 1 if hold else None
    return module.BarrierEndpoint([dict(step) for step in PARENT_SCRIPT], {}, hold_parent_request=hold_parent).start()


def _host_port(url: str) -> tuple[str, int]:
    netloc = url.split("//", 1)[1].split("/", 1)[0]
    host, port = netloc.rsplit(":", 1)
    return host, int(port)


def _send_then_reset(endpoint: object, body: bytes) -> None:
    """Send a complete POST, wait until the endpoint has it in hand, then RST the connection without reading."""
    host, port = _host_port(endpoint.url)  # type: ignore[attr-defined]
    head = (
        f"POST /v1/chat/completions HTTP/1.1\r\nHost: {host}:{port}\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n"
    ).encode("ascii")
    sock = socket.create_connection((host, port), timeout=WAIT_SECONDS)
    try:
        sock.sendall(head + body)
        if not endpoint.wait_for_requests("parent", 1, timeout=WAIT_SECONDS):  # type: ignore[attr-defined]
            raise RuntimeError("endpoint never received the disconnecting request")
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    finally:
        sock.close()
    time.sleep(SETTLE_SECONDS)  # let the RST reach the server socket before the held write is released


def _release_and_wait(endpoint: object) -> None:
    endpoint.release_parent_request()  # type: ignore[attr-defined]
    deadline = time.monotonic() + WAIT_SECONDS
    while not endpoint.intervals("parent"):  # type: ignore[attr-defined]
        if time.monotonic() >= deadline:
            raise RuntimeError("endpoint never finished the disconnected request")
        time.sleep(0.01)


def _later_request_tool(endpoint_url: str, body: bytes) -> str:
    """A normal request after the disconnect must still be served; returns the offered tool name."""
    request = urllib.request.Request(
        f"{endpoint_url}/chat/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=WAIT_SECONDS) as response:
        payload = json.loads(response.read())
    return str(payload["choices"][0]["message"]["tool_calls"][0]["function"]["name"])


@pytest.mark.race_seed(
    "09",
    mutant=MUTANT_NAME,
    expect="client disconnect recorded as endpoint error",
)
def test_seed09a_client_disconnect_before_read_is_not_endpoint_error(tmp_path: Path):
    module = _endpoint_module(tmp_path)
    body = _valid_body()
    for loop in range(LOOPS):
        endpoint = _new_endpoint(module, hold=True)
        try:
            _send_then_reset(endpoint, body)
            _release_and_wait(endpoint)
            errors = endpoint.errors  # type: ignore[attr-defined]
            assert not errors, f"loop {loop}: client disconnect recorded as endpoint error: {errors[0]}"
            try:
                endpoint.assert_consumed()  # type: ignore[attr-defined]
            except AssertionError as error:
                raise AssertionError(
                    f"loop {loop}: assert_consumed raised after client disconnect: {error}"
                ) from error
            tool = _later_request_tool(endpoint.url, body)  # type: ignore[attr-defined]
            assert tool == "done", f"loop {loop}: later request after disconnect got tool {tool!r}, expected 'done'"
        finally:
            endpoint.close()  # type: ignore[attr-defined]


def test_seed09b_companion_malformed_body_still_records_endpoint_error():
    endpoint = _new_endpoint(product_fixture, hold=False)
    try:
        request = urllib.request.Request(
            f"{endpoint.url}/chat/completions", data=b"{not json",  # type: ignore[attr-defined]
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as info:
            urllib.request.urlopen(request, timeout=WAIT_SECONDS)
        assert info.value.code == 400, f"malformed body answered with status {info.value.code}, expected 400"
        assert endpoint.errors, "malformed body was not recorded as an endpoint error"  # type: ignore[attr-defined]
        with pytest.raises(AssertionError, match="endpoint errors"):
            endpoint.assert_consumed()  # type: ignore[attr-defined]
    finally:
        endpoint.close()  # type: ignore[attr-defined]
