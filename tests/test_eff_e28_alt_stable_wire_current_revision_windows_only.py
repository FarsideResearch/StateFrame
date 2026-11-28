import json
from types import SimpleNamespace

from test_stable_wire import _body, _invoke, _messages_with_prefix, _revision, _window


def test_revision_change_releases_obsolete_window(monkeypatch):
    old = _window(text="# old content\n" * 250)
    new = _window(revision=_revision("2"), text="x = 1\n")
    sizes = {}
    for mode in ("off", "on"):
        monkeypatch.setenv("STATEFRAME_STABLE_WIRE_CURRENT_ONLY", mode)
        runtime = SimpleNamespace(_stable_wire_state=None)
        _invoke(monkeypatch, _body([old]), runtime)
        result, capture, _ = _invoke(monkeypatch, _body([new]), runtime)
        assert capture.request_source_final_delivery == {"status": "COMPLETE"}
        expected = [new] if mode == "on" else [old, new]
        assert _messages_with_prefix(result["messages"], "[Current source:") == expected
        sizes[mode] = len(json.dumps(result).encode())
    assert sizes["off"] - sizes["on"] >= len(old["content"].encode())
