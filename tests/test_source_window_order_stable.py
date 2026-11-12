"""Equal-priority source carriers keep their first-seen order across requests."""

from __future__ import annotations

import json
from types import SimpleNamespace

from kernel_cache_layout_candidate import (
    _checked_final_cache_projection,
    render_final_cache_candidate,
)
from kernel_context import token_estimate
from kernel_frame_source_snapshot import editable_product_windows, fit_recent_read_windows
from kernel_request_budget import HARD_MAX_REQUEST_BYTES
from kernel_request_source_window import parse_model_source_window


def _request_body() -> dict[str, object]:
    frame = {"state": {"framefit5": {
        "mode": "FRAME_ONLY",
        "owner_protocol": {
            "schema": "stateframe-framefit5-owner/1",
            "continuation_limits": "[CONTINUATION LIMITS | stateframe-continuation-limits/1] remaining=2",
        },
        "read_facts": [],
        "action_ledger": {"reads": [], "edits": []},
    }}}
    return {"tools": [], "messages": [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task: preserve the source windows"},
        {"role": "user", "content": "[StateFrame]\n" + json.dumps(frame, sort_keys=True, separators=(",", ":"))},
        {"role": "user", "content": "[CONTINUATION LIMITS | stateframe-continuation-limits/1] remaining=2"},
        {"role": "user", "content": "[WIRE VOLATILE TAIL]"},
    ]}


def _source_rows(messages: list[dict[str, object]]) -> list[tuple[str, str]]:
    result = []
    for row in messages:
        content = row.get("content")
        metadata = parse_model_source_window(content)
        if metadata is not None and metadata.get("owner") == "recent_editable_read":
            result.append((metadata["path"], content))
    return result


def _serialized(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _first_difference(left: bytes, right: bytes) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def test_source_windows_are_append_only_across_consecutive_requests(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "off")
    paths = {name: f"pkg/{name}.py" for name in "abcd"}
    for name, path in paths.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"VALUE = '{name}'\n")

    # Only the workspace root and edit authority are supplied by the host.
    # File reading, source rendering, selection, fitting, and layout stay real.
    runtime = SimpleNamespace(owner=SimpleNamespace(root=tmp_path), edit_contract={})
    steps = [
        ("abc", "abc"),       # baseline
        ("abc", "cab"),       # same bytes, different bookmark input order
        ("abcd", "dcab"),     # new d must follow existing a, b, c
        ("acd", "cda"),       # evict b without sorting a, c, d
        ("acd", "adc"),       # change c's bytes in its old position
        ("abcd", "bdac"),     # reintroduced b is a new tail window
    ]
    projected = []
    observed = []
    for number, (editable, priority) in enumerate(steps):
        if number == 4:
            (tmp_path / paths["c"]).write_text("VALUE = 'c changed'\n")
        runtime.edit_contract = {"edit_authority": {
            "editable_paths": [paths[name] for name in editable],
        }}
        bookmarks = [{"path": paths[name], "rank": 2} for name in priority]
        windows, editable_paths = editable_product_windows(
            runtime, {"read_facts": [], "applied_edits": []}, bookmarks,
        )
        fitted = fit_recent_read_windows(
            _request_body(), windows, editable_paths=editable_paths,
            fits=lambda candidate: (
                len(_serialized(candidate)) <= HARD_MAX_REQUEST_BYTES
                and token_estimate(candidate) <= 33_344
            ),
        )
        # The explicit off switch returns the exact fitted request object and bytes.
        monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "off")
        before_off = _serialized(fitted)
        assert render_final_cache_candidate(
            None, runtime, fitted, meter=None, inputs=None,
        ) is fitted
        assert _serialized(fitted) == before_off

        monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "on")
        # This is the final layout's production projection, after the real
        # source selector and budget fitter. No ordering is stubbed.
        wire = _checked_final_cache_projection(fitted)
        rows = _source_rows(wire["messages"])
        assert len(rows) == len(editable), (number, rows)
        assert {path for path, _content in rows} == {paths[name] for name in editable}
        assert wire["messages"][2:2 + len(rows)] == [
            {"role": "user", "content": content} for _path, content in rows
        ]
        observed.append(rows)
        projected.append(wire["messages"])

    violations = []
    for number in range(1, len(steps)):
        previous, current = observed[number - 1], observed[number]
        prior_paths = [path for path, _content in previous]
        current_paths = [path for path, _content in current]
        expected = [path for path in prior_paths if path in current_paths]
        expected += [path for path in current_paths if path not in prior_paths]
        if current_paths != expected:
            violations.append(f"{number - 1}->{number}: {prior_paths} -> {current_paths}; expected {expected}")

        # The first changed/evicted/new window bounds the reusable serialized
        # message prefix. Input order among equal-priority sources must not
        # shorten it at all.
        prior_content = dict(previous)
        current_content = dict(current)
        shared = 0
        for path, content in previous:
            if current_content.get(path) != content:
                break
            shared += 1
        old_bytes = _serialized(projected[number - 1])
        new_bytes = _serialized(projected[number])
        difference = _first_difference(old_bytes, new_bytes)
        boundary = len(_serialized(projected[number - 1][:2 + shared])) - 1
        if difference is not None and difference < boundary:
            violations.append(
                f"{number - 1}->{number}: serialized difference at {difference}, "
                f"before expected source boundary {boundary}"
            )
        if number == 1 and difference is not None:
            violations.append(f"0->1: priority-only request changed at byte {difference}")
        if number == 4:
            assert prior_content[paths["c"]] != current_content[paths["c"]]
            assert prior_content[paths["a"]] == current_content[paths["a"]]
            assert prior_content[paths["d"]] == current_content[paths["d"]]

    assert not violations, "\n".join(violations)
