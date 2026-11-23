"""Blind holdouts for request-local final-layout projection reuse."""
from __future__ import annotations

import ast
import copy
import json
import random
import subprocess
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_UNCACHED = "7540148c9c7b58cd76dfec3b004dc3763451a3d3"


def _parent_projector():
    """Load the uncached final-layout projector from release 0.8.23, before the memo."""
    import kernel_cache_layout_candidate as candidate

    source = subprocess.run(
        ["git", "show", f"{_UNCACHED}:src/stateframe/engine/kernel_cache_layout_candidate.py"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout
    tree = ast.parse(source)
    wanted = {"_cacheable_frame_content", "_project_final_cache_layout",
              "_checked_final_cache_projection"}
    snippet = "\n\n".join(ast.get_source_segment(source, node) for node in tree.body
                            if isinstance(node, ast.FunctionDef) and node.name in wanted)
    namespace = {
        "__package__": "",
        "copy": copy,
        "json": json,
        "Mapping": Mapping,
        "SOURCE_COVERAGE_PREFIX": candidate.SOURCE_COVERAGE_PREFIX,
        "FinalCacheLayoutProjectionError": candidate.FinalCacheLayoutProjectionError,
    }
    exec(compile(snippet, "parent-kernel_cache_layout_candidate.py", "exec"), namespace)
    return namespace["_checked_final_cache_projection"]


def _body(exchange=False):
    payload = {"state": {"framefit5": {
        "owner_protocol": {
            "schema": "stateframe-framefit5-owner/1",
            "continuation_limits": "[CONTINUATION LIMITS | stateframe-continuation-limits/1] remaining=2",
            "evidence_pin": "[EVIDENCEPIN]current[/EVIDENCEPIN]",
        },
        "next_site": {"path": "pkg/a.py"},
        "last_verdict": {"line": "LAST FAIL"},
        "action_ledger": {"reads": [], "edits": []},
    }}}
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task: fix it"},
        {"role": "user", "content": "[StateFrame]\n" + json.dumps(
            payload, sort_keys=True, separators=(",", ":"))},
        {"role": "user", "content": "[Current source: pkg/a.py]\nrevision=sha256:123\nsource bytes"},
        {"role": "user", "content": "[CONTINUATION LIMITS | stateframe-continuation-limits/1] remaining=2"},
    ]
    if exchange:
        messages.extend([
            {"role": "assistant", "tool_calls": [{"id": "call-1", "type": "function",
                                                   "function": {"name": "read", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call-1", "content": "result"},
        ])
    messages.append({"role": "user", "content": "[WIRE VOLATILE TAIL]"})
    return {"messages": messages, "tools": []}


def _random_body(rng, index, *, vary_frame):
    body = _body(exchange=bool(index % 2))
    frame = body["messages"][2]
    if vary_frame:
        payload = json.loads(frame["content"].split("\n", 1)[1])
        framefit = payload["state"]["framefit5"]
        framefit["owner_protocol"]["evidence_pin"] = f"[EVIDENCEPIN]λ-{index}-雪[/EVIDENCEPIN]"
        framefit["random_unicode"] = ["é", "λ", "雪", f"line\n{index}"]
        frame["content"] = "[StateFrame]\n" + json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    rows = [
        {"role": "user", "content": f'note {index}: "λ" / 雪', "extra": {"n": index}},
        {"role": "user", "content": "[Read lockout] current instruction λ"},
        {"role": "user", "content": "[Source coverage]\nhealed 雪"},
    ]
    if index % 3 == 0:
        rows.append(copy.deepcopy(rows[0]))
    if index % 4 == 0:
        rows.pop(0)
    if index % 5 == 0:
        rows.reverse()
    split = rng.randrange(len(rows) + 1)
    messages = body["messages"]
    body["messages"] = messages[:2] + rows[:split] + [frame] + rows[split:] + messages[3:]
    return body


def _wire_bytes(body):
    return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def test_randomized_rows_match_parent_uncached_projection_bytes():
    import kernel_cache_layout_candidate as candidate
    from kernel_working_set_cache import transaction

    parent_project = _parent_projector()
    rng = random.Random(0x51A7E)
    bodies = [_random_body(rng, index, vary_frame=True) for index in range(36)]
    expected = [_wire_bytes(parent_project(copy.deepcopy(body))) for body in bodies]

    @transaction
    def memoized_builds():
        return [_wire_bytes(candidate._checked_final_cache_projection(copy.deepcopy(body)))
                for body in bodies]

    assert memoized_builds() == expected


def test_mutation_between_request_builds_is_seen():
    import kernel_cache_layout_candidate as candidate
    from kernel_working_set_cache import transaction

    body = _body()

    @transaction
    def build():
        return _wire_bytes(candidate._checked_final_cache_projection(copy.deepcopy(body)))

    first = build()
    payload = json.loads(body["messages"][2]["content"].split("\n", 1)[1])
    payload["state"]["framefit5"]["owner_protocol"]["evidence_pin"] = (
        "[EVIDENCEPIN]changed-between-builds[/EVIDENCEPIN]")
    body["messages"][2]["content"] = "[StateFrame]\n" + json.dumps(
        payload, sort_keys=True, separators=(",", ":"))
    second = build()

    assert first != second
    assert b"changed-between-builds" in second


def test_partition_render_calls_drop_for_repeated_frame_rows(monkeypatch):
    import kernel_cache_layout_candidate as candidate
    import kernel_stateframe_page_log as page_log
    from kernel_working_set_cache import transaction

    parent_project = _parent_projector()
    rng = random.Random(0xCACE)
    # Every body has the same StateFrame row but different surrounding row
    # positions, so only the frame partition is reusable.
    bodies = [_random_body(rng, index, vary_frame=False) for index in range(8)]
    calls = []
    original = page_log._partition_authority

    def counted(*args, **kwargs):
        calls.append(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(page_log, "_partition_authority", counted)
    expected = [_wire_bytes(parent_project(copy.deepcopy(body))) for body in bodies]
    parent_calls = len(calls)
    calls.clear()

    @transaction
    def memoized_builds():
        return [_wire_bytes(candidate._checked_final_cache_projection(copy.deepcopy(body)))
                for body in bodies]

    actual = memoized_builds()
    assert actual == expected
    assert len(calls) < parent_calls
