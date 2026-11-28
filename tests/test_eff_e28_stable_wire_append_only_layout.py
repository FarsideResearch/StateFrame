import json
import os
import re
from types import SimpleNamespace

from kernel_stable_wire import materialize_stable_frame
from test_stable_wire import _body, _invoke, _window


def tokens(body):
    wire = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
    return re.findall(r"\w+|[^\w\s]|\s+", wire)


def test_append_only_source_increases_token_lcp_without_losing_current_range(monkeypatch):
    source = _window(text="\n".join(f"line_{i} = {i}" for i in range(100)))
    current = _window(path="pkg/b.py", text="return 9\n")
    plain = [_body([source]), _body([current])]
    runtime = SimpleNamespace(_stable_wire_state=None)
    stable = []
    for body in plain:
        projected, _, _ = _invoke(monkeypatch, body, runtime)
        stable.append(projected)
        original_frame, = [m["content"] for m in body["messages"]
                           if m["content"].startswith("[StateFrame]")]
        assert materialize_stable_frame(projected["messages"]) == original_frame
    assert current in stable[-1]["messages"]
    assert len(os.path.commonprefix(list(map(tokens, stable)))) > (
        len(os.path.commonprefix(list(map(tokens, plain)))) + 100)
