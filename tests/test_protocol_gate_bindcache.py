"""Protocol edits must invalidate the binding cache at the pre-dispatch boundary.

edit_source / rewrite_source publish product bytes, and binding() caches the
artifact revision keyed only by the host epoch. start_action's pre-dispatch
invalidation list previously named only legacy edit kinds, so a protocol edit
dispatched against a cached pre-edit binding.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from kernel_edit_tool_names import PROTOCOL_EDIT_TOOL_NAMES
from kernel_stateframe_runtime import StateFrameRuntime

STALE = {"epoch": 0, "inputs_digest": "fixture", "value": {"artifact": "pre-edit"}}


@pytest.fixture(autouse=True)
def qgates_off(monkeypatch):
    monkeypatch.delenv("STATEFRAME_QGATES", raising=False)


def _runtime():
    runtime = StateFrameRuntime.__new__(StateFrameRuntime)
    runtime.journal = SimpleNamespace(manifest={}, emit=lambda *_args, **_kwargs: None)
    runtime._binding_epoch = 0
    runtime._binding_cache = dict(STALE)
    # Stop start_action right after the pre-dispatch invalidation; the rest of the
    # dispatch path is out of scope for this gate.
    runtime.claim_grounding_before_dispatch = lambda _kind, _arguments: "STOP_AFTER_INVALIDATION"
    return runtime


@pytest.mark.parametrize("tool", sorted(PROTOCOL_EDIT_TOOL_NAMES))
def test_protocol_edit_invalidates_binding_cache_before_dispatch(tool):
    runtime = _runtime()
    runtime.start_action(tool, {"edits": []})
    assert runtime._binding_cache is None
    assert runtime._binding_epoch == 1


@pytest.mark.parametrize("tool", ["str_replace", "edit", "write"])
def test_legacy_edit_invalidates_binding_cache_before_dispatch(tool):
    runtime = _runtime()
    runtime.start_action(tool, {"path": "a.py"})
    assert runtime._binding_cache is None
    assert runtime._binding_epoch == 1


def test_read_keeps_binding_cache_before_dispatch():
    runtime = _runtime()
    runtime.start_action("read", {"path": "a.py"})
    assert runtime._binding_cache == STALE
    assert runtime._binding_epoch == 0
