"""The default final layout preserves source-window order across request fits."""

from __future__ import annotations

import json
import pickle
from types import SimpleNamespace

from kernel_frame_source_snapshot import (
    editable_product_windows_for_request,
    fit_recent_read_windows,
    record_source_window_order,
)
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


def _runtime(root) -> SimpleNamespace:
    return SimpleNamespace(owner=SimpleNamespace(root=root), edit_contract={})


def _write_sources(root, paths: dict[str, str], *, changed: str | None = None) -> None:
    for name, path in paths.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        value = f"VALUE = '{name}'\n"
        if name == changed:
            value = f"VALUE = '{name} changed'\n"
        target.write_text(value)


def _source_rows(body: dict[str, object]) -> list[tuple[str, str]]:
    result = []
    for message in body["messages"]:
        metadata = parse_model_source_window(message.get("content"))
        if metadata is not None and metadata.get("owner") == "recent_editable_read":
            result.append((metadata["path"], message["content"]))
    return result


def _build_request(runtime, paths: dict[str, str], editable: str, priority: str):
    runtime.edit_contract = {"edit_authority": {
        "editable_paths": [paths[name] for name in editable],
    }}
    bookmarks = [{"path": paths[name], "rank": 1 if index == 0 else 2}
                 for index, name in enumerate(priority)]
    windows, editable_paths = editable_product_windows_for_request(
        runtime, {"read_facts": [], "applied_edits": []}, bookmarks,
    )
    fitted = fit_recent_read_windows(
        _request_body(), windows, editable_paths=editable_paths,
        fits=lambda _candidate: True, runtime=runtime,
    )
    record_source_window_order(runtime, fitted)
    return json.dumps(fitted, ensure_ascii=False, separators=(",", ":")).encode(), fitted


def test_eviction_removes_order_and_changed_bytes_keep_their_position(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "off")
    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "on")
    paths = {name: f"pkg/{name}.py" for name in "abcd"}
    _write_sources(tmp_path, paths)
    runtime = _runtime(tmp_path)
    steps = [
        ("abc", "abc", None, ["a", "b", "c"]),
        ("abc", "cab", None, ["a", "b", "c"]),  # re-ranking does not reorder delivered windows
        ("abcd", "dcab", None, ["a", "b", "c", "d"]),  # newly selected d appends at the tail
        ("acd", "cda", None, ["a", "c", "d"]),  # evict b without reordering survivors
        ("acd", "adc", "c", ["a", "c", "d"]),
        ("abcd", "bdac", None, ["a", "c", "d", "b"]),  # reintroduced b is a new tail window
    ]

    for editable, priority, changed, expected in steps:
        if changed:
            _write_sources(tmp_path, paths, changed=changed)
        _wire, fitted = _build_request(runtime, paths, editable, priority)
        rows = _source_rows(fitted)
        assert [path.rsplit("/", 1)[-1][0] for path, _content in rows] == expected
        if changed:
            assert "changed" in rows[1][1]


def test_checkpoint_restore_and_replay_rebuild_identical_source_order(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "off")
    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "on")
    paths = {name: f"pkg/{name}.py" for name in "abcd"}
    _write_sources(tmp_path, paths)

    runtime = _runtime(tmp_path)
    for editable, priority in (("abc", "abc"), ("abc", "cab")):
        _build_request(runtime, paths, editable, priority)
    restored = pickle.loads(pickle.dumps(runtime))
    expected, _body = _build_request(runtime, paths, "abcd", "bdac")
    resumed, _body = _build_request(restored, paths, "abcd", "bdac")

    replayed_runtime = _runtime(tmp_path)
    for editable, priority in (("abc", "abc"), ("abc", "cab"), ("abcd", "bdac")):
        replayed, _body = _build_request(replayed_runtime, paths, editable, priority)
    assert resumed == expected == replayed


def test_priority_selects_budgeted_windows_before_stable_order(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "off")
    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "on")
    paths = {name: f"pkg/{name}.py" for name in "abc"}
    for name, path in paths.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"VALUE = '{name}'\n" + "#" * 4_900)
    runtime = _runtime(tmp_path)
    _wire, fitted = _build_request(runtime, paths, "abc", "cab")
    assert [path.rsplit("/", 1)[-1][0] for path, _content in _source_rows(fitted)] == ["c", "a"]  # Priority selects the first delivered set.


def test_layout_off_and_stable_wire_keep_existing_priority_order(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    paths = {name: f"pkg/{name}.py" for name in "abc"}
    _write_sources(tmp_path, paths)

    for layout, stable in (("off", "off"), ("on", "on")):
        monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", layout)
        monkeypatch.setenv("STATEFRAME_STABLE_WIRE", stable)
        runtime = _runtime(tmp_path)
        _wire, fitted = _build_request(runtime, paths, "abc", "cab")
        assert [path.rsplit("/", 1)[-1][0] for path, _content in _source_rows(fitted)] == ["c", "a", "b"]
        assert not hasattr(runtime, "_framefit5_source_window_order")
