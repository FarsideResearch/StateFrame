"""Delivered source order stays stable while priority selects each fitted set."""

from __future__ import annotations

import json
from types import SimpleNamespace

from kernel_cache_layout_candidate import _checked_final_cache_projection
from kernel_context import token_estimate
from kernel_frame_source_snapshot import (
    editable_product_windows_for_request,
    fit_recent_read_windows,
)
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
        {"role": "user", "content": "Task: preserve source windows"},
        {"role": "user", "content": "[StateFrame]\n" + json.dumps(frame, sort_keys=True, separators=(",", ":"))},
        {"role": "user", "content": "[CONTINUATION LIMITS | stateframe-continuation-limits/1] remaining=2"},
        {"role": "user", "content": "[WIRE VOLATILE TAIL]"},
    ]}


def _runtime(root, paths: dict[str, str]) -> SimpleNamespace:
    return SimpleNamespace(
        owner=SimpleNamespace(root=root),
        edit_contract={"edit_authority": {"editable_paths": list(paths.values())}},
    )


def _visible_paths(body: dict[str, object], valid_paths: set[str]) -> list[str]:
    return [metadata["path"] for message in body["messages"]
            if (metadata := parse_model_source_window(message.get("content")))
            and metadata.get("owner") == "recent_editable_read"
            and metadata.get("path") in valid_paths]


def _serialized(body: object) -> bytes:
    return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _coverage(body: dict[str, object]) -> str:
    return next(message["content"] for message in body["messages"]
                if str(message.get("content", "")).startswith("[Source coverage]\n"))


def test_priority_reselection_appends_sources_and_preserves_survivor_order(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "off")
    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "on")

    paths = {name: f"pkg/{name}.py" for name in "abcde"}
    for name, path in paths.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"VALUE = '{name}'\n" + "#" * 1_700)
    valid_paths = set(paths.values())
    runtime = _runtime(tmp_path, paths)
    empty_runtime = _runtime(tmp_path, paths)

    def source_windows(active_runtime, bookmarks):
        frame = {"read_facts": [], "applied_edits": []}
        return editable_product_windows_for_request(active_runtime, frame, bookmarks)

    # Calibrate an actual byte ceiling to admit three equal whole-source windows;
    # the production fitter also enforces the request hard limits below.
    initial_bookmarks = [
        {"path": paths["a"], "rank": 1, "origin": "failed_check"},
        {"path": paths["b"], "rank": 2, "origin": "targeted_read"},
    ]
    initial_windows, editable = source_windows(empty_runtime, initial_bookmarks)
    three_window_request = fit_recent_read_windows(
        _request_body(), initial_windows[:3], editable_paths=editable,
        fits=lambda _candidate: True, runtime=empty_runtime,
    )
    byte_ceiling = len(_serialized(three_window_request))

    steps = [
        # First delivery: a, b, c.
        ([
            {"path": paths["a"], "rank": 1, "origin": "failed_check"},
            {"path": paths["b"], "rank": 2, "origin": "targeted_read"},
        ], ["a", "b", "c"]),
        # Re-rank the set: d is selected first, while a and c survive; b is evicted.
        ([
            {"path": paths["d"], "rank": 1, "origin": "targeted_read"},
            {"path": paths["a"], "rank": 2, "origin": "targeted_read"},
            {"path": paths["c"], "rank": 2, "origin": "targeted_read"},
        ], ["a", "c", "d"]),
        # New failed-check source e wins selection, then appends after c and d.
        ([
            {"path": paths["e"], "rank": 1, "origin": "failed_check"},
            {"path": paths["c"], "rank": 2, "origin": "targeted_read"},
            {"path": paths["d"], "rank": 2, "origin": "targeted_read"},
        ], ["c", "d", "e"]),
        # Reintroduced b is appended after surviving d and e.
        ([
            {"path": paths["b"], "rank": 1, "origin": "targeted_read"},
            {"path": paths["d"], "rank": 2, "origin": "targeted_read"},
            {"path": paths["e"], "rank": 2, "origin": "targeted_read"},
        ], ["d", "e", "b"]),
    ]

    for bookmarks, expected in steps:
        windows, editable = source_windows(runtime, bookmarks)
        fitted = fit_recent_read_windows(
            _request_body(), windows, editable_paths=editable, runtime=runtime,
            fits=lambda candidate: (
                len(_serialized(candidate))
                <= min(byte_ceiling, HARD_MAX_REQUEST_BYTES)
                and token_estimate(candidate) <= 33_344
            ),
        )
        wire = _checked_final_cache_projection(fitted)
        assert len(_serialized(wire)) <= HARD_MAX_REQUEST_BYTES
        assert token_estimate(wire) <= 33_344
        visible = _visible_paths(wire, valid_paths)
        assert [path.rsplit("/", 1)[-1][0] for path in visible] == expected

        coverage_lines = _coverage(fitted).splitlines()[1:]
        assert len(coverage_lines) == len(paths)
        assert all(any(line.startswith(path + " ") or line.startswith(path + ":")
                       for line in coverage_lines) for path in paths.values())
        assert {path for path in paths.values()
                if any(line.startswith(path + ": not shown") for line in coverage_lines)} == (
                    valid_paths - set(visible)
                )
