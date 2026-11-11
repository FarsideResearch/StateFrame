"""Priority selects the fitted set while delivered cache order stays stable."""

from __future__ import annotations

import json
from types import SimpleNamespace

from kernel_frame_source_snapshot import (
    editable_product_windows,
    fit_recent_read_windows,
    record_source_window_order,
)
from kernel_request_source_window import parse_model_source_window


def test_priority_selects_sources_while_cache_order_stays_stable(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "on")
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "off")

    paths = {
        "bookmark": "pkg/bookmark.py",
        "failure": "pkg/failure.py",
        "other": "pkg/other.py",
    }
    for path in paths.values():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("value = '" + "x" * 900 + "'\n")

    runtime = SimpleNamespace(
        owner=SimpleNamespace(root=tmp_path),
        edit_contract={"edit_authority": {"editable_paths": list(paths.values())}},
    )
    frame = {"mode": "FRAME_ONLY", "read_facts": [], "applied_edits": []}
    bookmarks = [
        {"path": paths["failure"], "rank": 1, "origin": "failed_check"},
        {"path": paths["bookmark"], "rank": 2, "origin": "targeted_read"},
    ]
    windows, editable = editable_product_windows(runtime, frame, bookmarks)
    assert [row["path"] for row in windows] == [
        paths["failure"], paths["bookmark"], paths["other"],
    ]

    # A previous delivery sets wire order; current priority still selects the set.
    record_source_window_order(runtime, {"messages": [
        {"role": "user", "content": windows[1]["content"]},
        {"role": "user", "content": windows[0]["content"]},
        {"role": "user", "content": windows[2]["content"]},
    ]})

    body = {"messages": [{"role": "user", "content": "[StateFrame]\n" + json.dumps({
        "state": {"framefit5": frame},
    })}], "tools": []}
    minimum = fit_recent_read_windows(
        body, windows[:2], editable_paths=editable,
        fits=lambda _candidate: True,
    )
    ceiling = len(json.dumps(minimum).encode()) + 50
    fitted = fit_recent_read_windows(
        body, windows, editable_paths=editable,
        fits=lambda candidate: len(json.dumps(candidate).encode()) <= ceiling,
    )

    visible = [
        metadata["path"] for message in fitted["messages"]
        if (metadata := parse_model_source_window(message.get("content")))
        and metadata.get("path") in paths.values()
    ]
    assert visible == [paths["bookmark"], paths["failure"]]  # Keep delivered order after priority selection.

    coverage = next(message["content"] for message in fitted["messages"]
                    if str(message.get("content", "")).startswith("[Source coverage]\n"))
    assert f"{paths['other']}: not shown" in coverage
