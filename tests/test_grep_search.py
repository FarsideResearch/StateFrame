"""Grep reports search evidence without inferring symbol absence."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

from kernel_navigation_pages import render_grep_result, search_repository


@pytest.fixture
def pricing_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "shop/pricing").mkdir(parents=True)
    (root / "shop/pricing/rules.py").write_text(
        "def qualifies(cart):\n"
        "    cart.minimum_spend = 25\n"
        "    return cart.total >= cart.minimum_spend\n",
        encoding="utf-8",
    )
    return root


@pytest.mark.parametrize("mode", [None, "regex"])
def test_default_regex_matches_pipe_alternatives(pricing_repo: Path, mode: str | None):
    page = search_repository(pricing_repo, "minimum_spend|min_spend", "shop/pricing", mode=mode)
    assert page.metadata["total"] == 2
    assert "shop/pricing/rules.py:2:    cart.minimum_spend = 25" in page.text


@pytest.mark.parametrize("mode,pattern", [
    ("literal", "minimum_spend|min_spend"),
    ("regex", r"minimum_spend\|min_spend"),
    ("literal", "missing_symbol"),
    ("path", "missing.py"),
])
def test_zero_hits_identify_search_without_absence_claim(pricing_repo: Path, mode: str, pattern: str):
    page = search_repository(pricing_repo, pattern, "shop/pricing", mode=mode)
    assert page.metadata["total"] == 0
    assert "does not exist" not in page.text.lower()
    assert "no hits" in page.text.lower()
    assert repr(pattern) in page.text
    assert mode in page.text
    assert "shop/pricing" in page.text
    if mode == "literal" and "|" in pattern:
        assert "'|' character was matched literally" in page.text


@pytest.mark.parametrize("coordinates", ["off", "on"])
@pytest.mark.parametrize("oracle", ["off", "on"])
def test_zero_hit_renderer_ignores_symbol_absence(pricing_repo: Path, monkeypatch, coordinates, oracle):
    monkeypatch.setenv("STATEFRAME_OWNER_COORDS", coordinates)
    monkeypatch.setenv("STATEFRAME_EXISTENCE_ORACLE", oracle)
    page = search_repository(pricing_repo, "minimum_spend|min_spend", mode="literal")
    reply = render_grep_result(
        page, symbol="minimum_spend", symbol_result="`minimum_spend` DOES NOT EXIST anywhere",
        workspace_root=pricing_repo,
    )
    assert reply == page.text
    assert "does not exist" not in reply.lower()


@pytest.mark.parametrize("oracle", ["off", "on"])
@pytest.mark.parametrize("pattern,mode", [
    ("minimum_spend|min_spend", "literal"), ("missing_symbol", "regex"),
])
def test_grep_observation_preserves_zero_hit_evidence(pricing_repo: Path, monkeypatch, oracle, pattern, mode):
    from kernel_bolt import KernelBolt

    monkeypatch.setenv("KERNEL_INTERVENTIONS", "1")
    monkeypatch.setenv("STATEFRAME_REPEAT_GATE", "on")
    monkeypatch.setenv("STATEFRAME_EXISTENCE_ORACLE", oracle)
    bolt = KernelBolt(pricing_repo, "qualifies", "grep-observation")
    bolt.last_grep_hits = 0
    events = []
    monkeypatch.setattr(bolt, "_rec", lambda *args: events.append(args))
    page = search_repository(pricing_repo, pattern, "shop/pricing", mode=mode)
    reply = bolt.observe(f"grep {pattern} shop/pricing", page.text)
    assert "does not exist" not in reply.lower()
    assert "no symbol index entry" not in reply.lower()
    assert reply == page.text
    assert not any(event[1] == "answered_dead_grep" for event in events)


@pytest.mark.parametrize("oracle", ["off", "on"])
@pytest.mark.parametrize("mode", ["literal", "regex"])
def test_zero_hit_renderer_keeps_positive_registry_location(pricing_repo: Path, monkeypatch, oracle, mode):
    from symbol_registry import SymbolRegistry

    monkeypatch.setenv("STATEFRAME_OWNER_COORDS", "on")
    monkeypatch.setenv("STATEFRAME_EXISTENCE_ORACLE", oracle)
    scope = pricing_repo / "elsewhere"
    scope.mkdir()
    page = search_repository(pricing_repo, "qualifies", "elsewhere", mode=mode)
    registry = SymbolRegistry().build(pricing_repo)
    reply = render_grep_result(
        page, symbol="qualifies", workspace_root=pricing_repo,
        symbol_definitions=registry.where("qualifies"),
        symbol_result=registry.answer("qualifies", workspace_root=pricing_repo),
    )
    assert page.metadata["total"] == 0
    assert reply.startswith(page.text)
    assert "shop/pricing/rules.py:1" in reply


@pytest.mark.parametrize("coordinates", ["off", "on"])
@pytest.mark.parametrize("oracle", ["off", "on"])
@pytest.mark.parametrize("symbol", ["minimum_spend", "missing_symbol"])
@pytest.mark.parametrize("mode", ["literal", "regex"])
def test_zero_hit_renderer_suppresses_negative_and_unknown_facts(
        pricing_repo: Path, monkeypatch, coordinates, oracle, symbol, mode):
    from symbol_registry import SymbolRegistry

    monkeypatch.setenv("STATEFRAME_OWNER_COORDS", coordinates)
    monkeypatch.setenv("STATEFRAME_EXISTENCE_ORACLE", oracle)
    scope = pricing_repo / "elsewhere"
    scope.mkdir()
    # An unparseable file makes a live index miss unknown instead of negative.
    (pricing_repo / "broken.py").write_text("def broken(\n", encoding="utf-8")
    registry = SymbolRegistry().build(pricing_repo)
    page = search_repository(pricing_repo, symbol, "elsewhere", mode=mode)
    reply = render_grep_result(
        page, symbol=symbol, symbol_definitions=registry.where(symbol),
        symbol_result=registry.answer(symbol, workspace_root=pricing_repo),
        workspace_root=pricing_repo,
    )
    assert reply == page.text


def test_renderer_keeps_positive_live_literal_occurrence(pricing_repo: Path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_EXISTENCE_ORACLE", "on")
    page = search_repository(pricing_repo, "minimum_spend", mode="literal")
    reply = render_grep_result(page, symbol="minimum_spend", workspace_root=pricing_repo)
    assert "shop/pricing/rules.py:2" in reply
    assert "2 matches for literal text" in reply


def test_renderer_suppresses_stale_literal_and_index_misses(pricing_repo: Path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_EXISTENCE_ORACLE", "on")
    page = search_repository(pricing_repo, "minimum_spend", mode="literal")
    (pricing_repo / "shop/pricing/rules.py").write_text("", encoding="utf-8")
    reply = render_grep_result(page, symbol="minimum_spend", workspace_root=pricing_repo)
    assert reply == page.text


@pytest.mark.parametrize("pattern,matches", [("minimum_spend", 2), ("= 25", 1)])
def test_zero_hit_bolt_keeps_positive_live_literal_occurrence(pricing_repo: Path, monkeypatch, pattern, matches):
    from kernel_bolt import KernelBolt

    monkeypatch.setenv("KERNEL_INTERVENTIONS", "1")
    monkeypatch.setenv("STATEFRAME_REPEAT_GATE", "on")
    monkeypatch.setenv("STATEFRAME_EXISTENCE_ORACLE", "on")
    bolt = KernelBolt(pricing_repo, "qualifies", "grep-occurrence")
    bolt.last_grep_hits = 0
    events = []
    monkeypatch.setattr(bolt, "_rec", lambda *args: events.append(args))
    reply = bolt.observe(f"grep {pattern}", "(no grep matches)")
    assert "shop/pricing/rules.py:2" in reply
    assert f"{matches} matches for literal text" in reply
    assert ("symbol", "answered_dead_grep", pattern) in events


@pytest.mark.parametrize("oracle", ["off", "on"])
def test_zero_hit_bolt_keeps_positive_registry_location(pricing_repo: Path, monkeypatch, oracle):
    from kernel_bolt import KernelBolt

    monkeypatch.setenv("KERNEL_INTERVENTIONS", "1")
    monkeypatch.setenv("STATEFRAME_REPEAT_GATE", "on")
    monkeypatch.setenv("STATEFRAME_EXISTENCE_ORACLE", oracle)
    scope = pricing_repo / "elsewhere"
    scope.mkdir()
    bolt = KernelBolt(pricing_repo, "qualifies", "grep-positive")
    bolt.last_grep_hits = 0
    events = []
    monkeypatch.setattr(bolt, "_rec", lambda *args: events.append(args))
    page = search_repository(pricing_repo, "qualifies", "elsewhere")
    reply = bolt.observe("grep qualifies elsewhere", page.text)
    assert reply.startswith(page.text)
    assert "shop/pricing/rules.py:1" in reply
    subject = "qualifies elsewhere" if oracle == "on" else "qualifies"
    assert ("symbol", "answered_dead_grep", subject) in events


class _StopAfterSearches(BaseException):
    """Capture actual tool replies before the worker tries to finish the task."""


@pytest.mark.parametrize("coordinates", ["off", "on"])
@pytest.mark.parametrize("oracle", ["off", "on"])
def test_native_grep_never_claims_attribute_absent(tmp_path: Path, monkeypatch, coordinates: str, oracle: str):
    from kernel_measurement import ACTIVE_REQUEST
    from kernel_pilot import ACTIVE_PARENT, ACTIVE_RUNTIME
    from kernel_stateframe_runtime import StateFrameRuntime
    from kernel_verdict_identity import _ACTIVE_PROVIDER
    from stateframe.engine.probes import react_repair
    from stateframe.scripted_endpoint import Script, serve
    from stateframe.task_row import build_row
    from stateframe.workcopy import create

    repo = tmp_path / "repo"
    (repo / "shop/pricing").mkdir(parents=True)
    (repo / "shop/pricing/rules.py").write_text(
        "def qualifies(cart):\n    cart.minimum_spend = 25\n    return cart.total >= cart.minimum_spend\n",
        encoding="utf-8",
    )
    (repo / "tests").mkdir()
    (repo / "tests/test_pricing.py").write_text(
        "def test_pricing():\n    assert False\n", encoding="utf-8")
    row = build_row(create(repo, tmp_path / "copy"), task="Inspect pricing minimums",
                    target="shop/pricing/rules.py", task_id="grep-pipe")
    output = tmp_path / "run/worker/measurement-runs"
    output.mkdir(parents=True)
    private_temp = tmp_path / "worker-tmp"
    private_temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(private_temp))
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "stateframe-home"))
    monkeypatch.setenv("STATEFRAME_OWNER_COORDS", coordinates)
    monkeypatch.setenv("STATEFRAME_EXISTENCE_ORACLE", oracle)
    monkeypatch.setenv("KERNEL_INTERVENTIONS", "1")
    monkeypatch.setenv("STATEFRAME_IDEALPATH", "off")
    monkeypatch.setattr(sys, "argv", [sys.argv[0]])
    original_init = StateFrameRuntime.__init__

    def scoped_init(self, owner, **kwargs):
        kwargs["output"] = output
        original_init(self, owner, **kwargs)

    monkeypatch.setattr(StateFrameRuntime, "__init__", scoped_init)
    monkeypatch.setattr(react_repair, "MAX_STEPS", 6)
    queries = [
        {"pattern": "minimum_spend|min_spend", "path": "shop/pricing", "mode": "literal"},
        {"pattern": "missing_symbol", "path": "shop/pricing"},
        {"pattern": "minimum_spend|min_spend", "path": "shop/pricing"},
        {"pattern": "qualifies", "path": "tests", "mode": "literal"},
    ]
    script = Script({"model": "scripted", "tokenize": {"mode": "ok"}, "steps": [
        {"tool": "grep", "required": True, "arguments": query} for query in queries
    ]})
    server = serve(script)
    monkeypatch.setattr(react_repair, "API", f"http://127.0.0.1:{server.server_port}/v1/chat/completions")
    monkeypatch.setattr(react_repair, "MODEL", "scripted")
    monkeypatch.setattr(react_repair, "KEY", "")
    original_ask = react_repair.ask_tools
    replies: list[str] = []
    sends = 0

    def capture_replies(messages, *args, **kwargs):
        nonlocal sends
        sends += 1
        replies.extend(str(message["content"]) for message in messages if message.get("role") == "tool")
        if sends == len(queries) + 1:
            raise _StopAfterSearches
        return original_ask(messages, *args, **kwargs)

    monkeypatch.setattr(react_repair, "ask_tools", capture_replies)
    contexts = (ACTIVE_PARENT, ACTIVE_RUNTIME, ACTIVE_REQUEST, _ACTIVE_PROVIDER)
    tokens = [context.set(None) for context in contexts]
    try:
        with pytest.raises(_StopAfterSearches):
            react_repair.work(row)
    finally:
        server.shutdown()
        server.server_close()
        for context, token in zip(contexts, tokens):
            context.reset(token)
    script.assert_consumed()
    searches = [reply for reply in replies if "[search page]" in reply]
    assert searches, replies
    for reply in searches:
        for absence in ("does not exist", "no symbol index entry", "not defined", "0 matches for literal"):
            assert absence not in reply.lower(), reply
    assert any("shop/pricing/rules.py:2:    cart.minimum_spend = 25" in reply for reply in searches)
    assert any("no hits" in reply.lower() and "literal" in reply for reply in searches)
    assert any("pattern='qualifies'" in reply and "/tests'" in reply
               and "shop/pricing/rules.py:1" in reply for reply in searches), searches
