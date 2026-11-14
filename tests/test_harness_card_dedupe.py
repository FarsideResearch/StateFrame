"""The harness card is a system-message prefix that appears exactly once per request.

Regression: the operating instructions embedded the budget arm's compact system prompt,
which already carried the card, so the card sat mid-prompt; frame hygiene then saw a system
message that did not *start* with the card and prepended a second copy on every request.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from stateframe import profiles

ROOT = Path(__file__).resolve().parents[1]
BLOCK_BYTES = 200


def repeated_blocks(text: str, *, size: int = BLOCK_BYTES) -> list[str]:
    """Return every ``size``-byte block that occurs twice in ``text`` without overlapping."""
    data = text.encode("utf-8")
    first_seen: dict[bytes, int] = {}
    repeats: list[str] = []
    for index in range(0, max(len(data) - size + 1, 0)):
        block = data[index:index + size]
        earlier = first_seen.setdefault(block, index)
        if earlier != index and index - earlier >= size:
            repeats.append(block.decode("utf-8", errors="replace"))
    return repeats


def _system_texts(request: dict) -> list[str]:
    texts: list[str] = []
    for message in request["messages"]:
        if message.get("role") != "system":
            continue
        content = message.get("content")
        texts.append(content if isinstance(content, str) else json.dumps(content, sort_keys=True))
    return texts


def _apply_profile(monkeypatch, name: str) -> None:
    for key, value in profiles.load(name).items():
        monkeypatch.setenv(key, value)


def test_repeated_blocks_detector_sees_a_duplicate_and_ignores_overlap():
    block = "x" * BLOCK_BYTES
    assert repeated_blocks(block + "-" + block)
    assert not repeated_blocks(block + block[:-1])  # one long run only overlaps itself
    assert not repeated_blocks("short text")


def test_system_prefix_hoists_an_embedded_card_instead_of_duplicating_it(monkeypatch):
    monkeypatch.setenv("STATEFRAME_HARNESS_CARD", "on")
    from kernel_harness_card import HARNESS_CARD, ensure_system_prefix, system_prefix

    embedded = "Do the task.\n" + HARNESS_CARD + "\nMore guidance."
    hoisted = system_prefix(embedded)
    assert hoisted.startswith(HARNESS_CARD)
    assert hoisted.count(HARNESS_CARD) == 1
    assert hoisted == HARNESS_CARD + "\nDo the task.\nMore guidance."
    # Already correct input is returned byte-for-byte.
    assert system_prefix(hoisted) == hoisted
    assert system_prefix("") == HARNESS_CARD
    projected = ensure_system_prefix([{"role": "system", "content": embedded},
                                      {"role": "user", "content": "Task: fix it"}])
    assert projected[0]["content"].count(HARNESS_CARD) == 1
    assert projected[0]["content"].startswith(HARNESS_CARD)
    assert projected[1] == {"role": "user", "content": "Task: fix it"}


@pytest.mark.parametrize("profile", sorted({profiles.DEFAULT, *profiles.available()}))
def test_prompt_guidance_carries_no_card_and_the_budget_system_prompt_carries_one(monkeypatch, profile):
    _apply_profile(monkeypatch, profile)
    from kernel_frame_budget import compact_system_prompt, prompt_guidance
    from kernel_harness_card import HARNESS_CARD, enabled

    guidance = prompt_guidance("v3")
    assert HARNESS_CARD not in guidance
    assert "You are a careful workspace editor." in guidance
    assert compact_system_prompt("g0").count(HARNESS_CARD) == (1 if enabled() else 0)


@pytest.mark.parametrize("profile", sorted({profiles.DEFAULT, *profiles.available()}))
def test_operating_instructions_through_frame_hygiene_hold_one_card(monkeypatch, profile):
    """The shipped system prompt, after the final presentation projection, has one card."""
    _apply_profile(monkeypatch, profile)
    from kernel_entry_authority import operating_instructions_for_ticket
    from kernel_frame_hygiene import project_messages
    from kernel_harness_card import HARNESS_CARD, enabled

    system = operating_instructions_for_ticket("Fix the pricing code so the tests pass.", None)
    prepared = project_messages([
        {"role": "system", "content": system},
        {"role": "user", "content": "Task: Fix the pricing code so the tests pass."},
    ])
    joined = "\n".join(_system_texts({"messages": prepared}))
    assert joined.count(HARNESS_CARD) == (1 if enabled() else 0)
    if enabled():
        assert joined.startswith(HARNESS_CARD)
    assert repeated_blocks(joined) == []


def _scripted_requests(monkeypatch, work: Path, profile: str) -> list[dict]:
    from stateframe import cli
    from stateframe.scripted_endpoint import Script, serve

    repo = work / "repo"
    shutil.copytree(ROOT / "examples" / "discount_bug", repo)
    script = Script(json.loads((ROOT / "examples" / "discount_bug.script.json").read_text()))
    monkeypatch.setenv("STATEFRAME_FL_CHECKPOINT_CAPTURE", "off")
    server = serve(script)
    try:
        cli.main([
            "run", "--repo", str(repo), "--task", "Fix the pricing code so the tests pass.",
            "--endpoint", f"http://127.0.0.1:{server.server_port}/v1", "--model", "scripted",
            "--run-dir", str(work / "run"), "--quiet", "--json", "--profile", profile,
            "--test-python", sys.executable,
        ])
    finally:
        server.shutdown()
        server.server_close()
    script.assert_consumed()
    return list(script.requests)


@pytest.mark.host_sandbox
@pytest.mark.parametrize("profile", sorted({profiles.DEFAULT, *profiles.available()}))
def test_shipped_requests_never_repeat_a_block_inside_the_system_message(tmp_path, monkeypatch, profile):
    """Class killer: no 200-byte block appears twice in any request's system message."""
    from kernel_harness_card import HARNESS_CARD

    requests = _scripted_requests(monkeypatch, tmp_path / profile, profile)
    assert requests, "scripted run produced no requests"
    card_on = profiles.load(profile).get("STATEFRAME_HARNESS_CARD", "off") == "on"
    for index, request in enumerate(requests, start=1):
        joined = "\n".join(_system_texts(request))
        repeats = repeated_blocks(joined)
        assert repeats == [], f"request {index}: repeated system block(s): {repeats[:1]!r}"
        assert joined.count(HARNESS_CARD) == (1 if card_on else 0), f"request {index}"
        if card_on:
            assert joined.startswith(HARNESS_CARD), f"request {index}: card is not the prefix"
