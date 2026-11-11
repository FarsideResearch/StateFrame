"""The final cache layout keeps fitted source windows in run order."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

from kernel_cache_layout_candidate import (
    FINAL_SOURCE_ORDER_STATE_ATTR,
    _order_final_source_windows,
    _persist_final_source_order_state,
)
from kernel_request_source_window import MODEL_WINDOW_SCHEMA


def _window(path: str, revision: str, value: str) -> dict[str, str]:
    metadata = {
        "schema": MODEL_WINDOW_SCHEMA,
        "path": path,
        "owner": "recent_editable_read",
        "revision": revision,
        "ranges": [{"start_line": 1, "end_line": 1}],
        "total_lines": 1,
        "status": "complete",
    }
    content = "\n".join((
        f"[Current source: {path}]",
        "[Model source window]",
        json.dumps(metadata, sort_keys=True, separators=(",", ":")),
        "[Source window content]",
        "--- source lines 1-1 ---",
        value,
        "[End source window]",
    ))
    return {"role": "user", "content": content}


def _body(names: tuple[str, ...], *, changed: str | None = None) -> dict[str, object]:
    windows = []
    for name in names:
        path = f"pkg/{name}.py"
        revision = f"revision-{name}"
        value = f"VALUE = '{name}'"
        if name == changed:
            revision = f"revision-{name}-changed"
            value = f"VALUE = '{name} changed'"
        windows.append(_window(path, revision, value))
    return {"messages": [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task: retain source order"},
        *windows,
    ]}


def _paths(body: dict[str, object]) -> list[str]:
    return [
        row["content"].splitlines()[0].removeprefix("[Current source: ").removesuffix("]")
        for row in body["messages"]
        if isinstance(row, dict)
        and isinstance(row.get("content"), str)
        and row["content"].startswith("[Current source:")
    ]


def _advance(runtime: object, body: dict[str, object]) -> dict[str, object]:
    ordered, state = _order_final_source_windows(body, runtime)
    assert isinstance(ordered, dict)
    _persist_final_source_order_state(runtime, state)
    return ordered


def test_eviction_removes_order_and_reintroduction_appends_after_current_windows():
    runtime = SimpleNamespace()
    assert _paths(_advance(runtime, _body(("a", "b", "c", "d")))) == [
        "pkg/a.py", "pkg/b.py", "pkg/c.py", "pkg/d.py",
    ]

    evicted = _advance(runtime, _body(("c", "a", "d"), changed="c"))
    assert _paths(evicted) == ["pkg/a.py", "pkg/c.py", "pkg/d.py"]
    assert "VALUE = 'c changed'" in evicted["messages"][3]["content"]

    reintroduced = _advance(runtime, _body(("b", "d", "a", "c")))
    assert _paths(reintroduced) == [
        "pkg/a.py", "pkg/c.py", "pkg/d.py", "pkg/b.py",
    ]


def test_checkpoint_resume_reproduces_next_source_order_bytes(monkeypatch):
    from kernel_continuation_restart import restore_runtime_field_values, runtime_fields

    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "on")
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "off")
    monkeypatch.setattr("kernel_continuation_restart.RUNTIME_FIELDS", ("sentinel",))

    original = SimpleNamespace(sentinel=None)
    _advance(original, _body(("a", "b")))
    saved = {name: copy.deepcopy(getattr(original, name)) for name in runtime_fields(original)}
    assert FINAL_SOURCE_ORDER_STATE_ATTR in saved

    resumed = SimpleNamespace(sentinel=None)
    restore_runtime_field_values(resumed, saved)
    assert getattr(resumed, FINAL_SOURCE_ORDER_STATE_ATTR) == \
        getattr(original, FINAL_SOURCE_ORDER_STATE_ATTR)

    next_body = _body(("c", "b", "a"))
    uninterrupted_wire = _advance(original, copy.deepcopy(next_body))
    resumed_wire = _advance(resumed, copy.deepcopy(next_body))
    assert json.dumps(resumed_wire, ensure_ascii=False, separators=(",", ":")).encode() == \
        json.dumps(uninterrupted_wire, ensure_ascii=False, separators=(",", ":")).encode()
    assert getattr(resumed, FINAL_SOURCE_ORDER_STATE_ATTR) == \
        getattr(original, FINAL_SOURCE_ORDER_STATE_ATTR)
