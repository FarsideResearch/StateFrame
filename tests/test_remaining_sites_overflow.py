from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import kernel_remaining_sites as remaining_sites


def _site(path: str, line: int, *, role: str = "executable") -> SimpleNamespace:
    return SimpleNamespace(
        path=path,
        line=line,
        text="ticket_literal()",
        role=role,
        revision="a" * 64,
    )


def _runtime_with_patterns(tmp_path, monkeypatch, patterns, sites_by_literal):
    runtime = SimpleNamespace(owner=SimpleNamespace(root=tmp_path), journal=None)
    paths = ("app.py",)
    monkeypatch.setattr(remaining_sites.source_window, "editable_product_paths",
                        lambda _runtime: paths)
    monkeypatch.setattr(remaining_sites, "_parent_sources",
                        lambda _runtime, _paths: {"app.py": "parent"})
    monkeypatch.setattr(remaining_sites, "_ticket_patterns",
                        lambda *_args: patterns)
    monkeypatch.setattr(remaining_sites, "_read_source", lambda *_args: "current")
    monkeypatch.setattr(
        remaining_sites.source_window,
        "literal_site_inventory",
        lambda _root, _paths, literals, **_kwargs: list(sites_by_literal[literals[0]]),
    )
    return runtime


def test_capped_literal_makes_green_inventory_incomplete(tmp_path, monkeypatch):
    cap = remaining_sites.source_window._MAX_SITES
    crowded_comments = [_site("app.py", line, role="comment")
                        for line in range(1, cap + 1)]
    runtime = _runtime_with_patterns(
        tmp_path, monkeypatch, ("crowded_literal", "absent_literal"),
        {"crowded_literal": crowded_comments, "absent_literal": []},
    )

    assert remaining_sites.inventory_ticket_sites(runtime) is None


@pytest.mark.parametrize("current_revision", [None, "1234abcd"])
def test_empty_green_inventory_requires_matching_current_revision(
        tmp_path, monkeypatch, current_revision):
    runtime = SimpleNamespace(owner=SimpleNamespace(root=tmp_path))
    monkeypatch.setattr(remaining_sites, "inventory_ticket_sites", lambda _runtime: ())
    monkeypatch.setattr(remaining_sites, "current_workspace_revision",
                        lambda _runtime: current_revision)

    decision = remaining_sites.decide_green_nag(
        "green nudge", runtime, "deadbeef",
        environ={remaining_sites.GREEN_NAG_MODE_ENV: "on"},
    )

    assert decision.fired is False
    assert decision.message is None
    assert decision.remaining == 0


@pytest.mark.parametrize("tested_revision", [None, "", "not-a-revision"])
def test_empty_green_inventory_needs_a_valid_tested_revision(tmp_path, monkeypatch, tested_revision):
    runtime = SimpleNamespace(owner=SimpleNamespace(root=tmp_path))
    monkeypatch.setattr(remaining_sites, "inventory_ticket_sites", lambda _runtime: ())

    decision = remaining_sites.decide_green_nag(
        "green nudge", runtime, tested_revision,
        environ={remaining_sites.GREEN_NAG_MODE_ENV: "on"},
    )

    assert (decision.fired, decision.message) == (False, None)


def test_empty_green_inventory_fires_for_matching_current_revision(tmp_path, monkeypatch):
    runtime = SimpleNamespace(owner=SimpleNamespace(root=tmp_path))
    monkeypatch.setattr(remaining_sites, "inventory_ticket_sites", lambda _runtime: ())
    monkeypatch.setattr(remaining_sites, "current_workspace_revision",
                        lambda _runtime: "deadbeef")

    decision = remaining_sites.decide_green_nag(
        "green nudge", runtime, "deadbeef",
        environ={remaining_sites.GREEN_NAG_MODE_ENV: "on"},
    )

    assert decision.fired is True
    assert decision.message == "green nudge"
    assert decision.remaining == 0


def test_disabled_green_nag_keeps_legacy_decision(monkeypatch):
    monkeypatch.setattr(remaining_sites, "inventory_ticket_sites",
                        lambda _runtime: pytest.fail("disabled gate must not inventory"))

    decision = remaining_sites.decide_green_nag(
        "green nudge", object(), None,
        environ={remaining_sites.GREEN_NAG_MODE_ENV: "off"},
    )

    assert decision == remaining_sites.GreenNagDecision(
        "green nudge", False, None, False, False)


def test_disabled_remaining_sites_keeps_request_message_bytes_identical():
    messages = [{"role": "tool", "content": "result"}]
    before = json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode()

    changed = remaining_sites.maybe_append_remaining_sites(
        messages, object(), environ={remaining_sites.MODE_ENV: "off"})

    after = json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode()
    assert changed is False
    assert after == before
