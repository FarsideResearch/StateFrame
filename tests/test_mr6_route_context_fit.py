"""MR-6(a): under STATEFRAME_ROUTING_SCHEMA=2 a non-main-frame route (decide2, unit subframes) may run a
smaller context, but must fit its OWN configured frame. Main-frame routes (read/plan/edit/finish) keep
the run-frame rule. Schema 1 keeps refusing any route smaller than the run's frame."""
from __future__ import annotations

from http.server import ThreadingHTTPServer
from threading import Thread

import pytest

from stateframe import runner
from stateframe.endpoint import Endpoint
from stateframe.scripted_endpoint import Script, make_handler
from _timeouts import scaled
from test_routes_cli import servers  # noqa: F401  (shared two-endpoint fixture)


def _options(default: str, small: str, route_cli: list[str], *, context: int = 65536) -> runner.RunOptions:
    options = runner.RunOptions(repo=None, task="task", endpoint=Endpoint(default, "base"),
                                route_cli=route_cli)
    options._context_tokens = context
    options._context_env = {}
    return options


def _slots(monkeypatch, small: str, *, small_slot: int, default_slot: int = 65536) -> None:
    monkeypatch.setattr(Endpoint, "get_slots", lambda self: [
        {"id": 0, "n_ctx": small_slot if self.base_url == small else default_slot}])


@pytest.fixture
def v2(monkeypatch):
    monkeypatch.setenv("STATEFRAME_ROUTING_SCHEMA", "2")


@pytest.fixture
def v1(monkeypatch):
    monkeypatch.delenv("STATEFRAME_ROUTING_SCHEMA", raising=False)


def test_v2_small_context_decide2_passes_preflight(servers, monkeypatch, v2):
    default, small = servers
    _slots(monkeypatch, small, small_slot=32768)
    options = _options(default, small, [f"decide2.endpoint={small}", "decide2.model=small",
                                        "decide2.context_tokens=32768"])
    runner._probe_routes(options, lambda *_: None)


def test_v2_small_context_unit_subframe_passes_preflight(servers, monkeypatch, v2):
    default, small = servers
    _slots(monkeypatch, small, small_slot=32768)
    options = _options(default, small, [f"unit.endpoint={small}", "unit.model=small",
                                        "unit.context_tokens=32768"])
    runner._probe_routes(options, lambda *_: None)


def test_v2_non_main_route_still_must_fit_its_own_slot(servers, monkeypatch, v2):
    default, small = servers
    _slots(monkeypatch, small, small_slot=16384)
    options = _options(default, small, [f"decide2.endpoint={small}", "decide2.model=small",
                                        "decide2.context_tokens=32768"])
    with pytest.raises(runner.RunError, match="route decide2 model small has a 16,384-token slot"):
        runner._probe_routes(options, lambda *_: None)


@pytest.mark.parametrize("role", ["read", "edit"])
def test_v2_main_frame_route_below_run_frame_is_refused(servers, monkeypatch, v2, role):
    default, small = servers
    _slots(monkeypatch, small, small_slot=32768)
    options = _options(default, small, [f"{role}.endpoint={small}", f"{role}.model=small",
                                        f"{role}.context_tokens=32768"])
    with pytest.raises(runner.RunError, match=rf"route {role}.context_tokens=32768 is smaller than this run's"):
        runner._probe_routes(options, lambda *_: None)


def test_v2_subframe_read_route_is_not_main_frame(servers, monkeypatch, v2):
    default, small = servers
    _slots(monkeypatch, small, small_slot=32768)
    options = _options(default, small, [f"unit.endpoint={small}", "unit.model=small",
                                        "unit.context_tokens=32768", "read.endpoint=" + default])
    runner._probe_routes(options, lambda *_: None)


def test_v1_small_decide2_route_is_refused(servers, monkeypatch, v1):
    default, small = servers
    _slots(monkeypatch, small, small_slot=32768)
    options = _options(default, small, [f"decide2.endpoint={small}", "decide2.model=small",
                                        "decide2.context_tokens=32768"])
    with pytest.raises(runner.RunError, match="route decide2.context_tokens=32768 is smaller than this run's"):
        runner._probe_routes(options, lambda *_: None)
