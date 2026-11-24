"""Provider response Unicode is repaired at the recorded wire boundary."""
from __future__ import annotations

import hashlib
import json

import kernel_bounded_http
import pytest
from stateframe import _child
from stateframe.engine.kernel_wire_string_hygiene import sanitize_provider_response
from stateframe.replay_tape import Replayer, normalized_digest


def _record_response(tmp_path, monkeypatch, response: bytes, *, replayer=None) -> bytes:
    monkeypatch.setattr(_child, "HOST_STATE", dict(_child.HOST_STATE))
    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.setattr(kernel_bounded_http, "request_bytes",
                        lambda *_args, **_kwargs: response)
    endpoint = "http://127.0.0.1:8080/v1/chat/completions"
    _child.install_wire_recorder(run, endpoint, replayer)
    returned = kernel_bounded_http.request_bytes(
        b'{"model":"scripted","messages":[]}', endpoint, "")
    taped = (run / "wire/0001.response.json").read_bytes()
    assert returned == taped
    return taped


def test_live_recorder_replaces_a_surrogate_before_taping(tmp_path, monkeypatch):
    response = json.dumps({"choices": [{"message": {"content": "bad\ud800"}}]},
                          ensure_ascii=True).encode("ascii")
    taped = _record_response(tmp_path, monkeypatch, response)
    actual = json.loads(taped)["choices"][0]["message"]["content"]
    assert actual.encode("unicode_escape") == b"bad\\ufffd"


def test_live_recorder_replaces_a_raw_cesu_surrogate(tmp_path, monkeypatch):
    response = b'{"choices":[{"message":{"content":"\xed\xa0\x80"}}]}'
    taped = _record_response(tmp_path, monkeypatch, response)
    assert json.loads(taped)["choices"][0]["message"]["content"] == "\ufffd"


def test_chunked_stream_body_is_sanitized_after_assembly(tmp_path, monkeypatch):
    response = b''.join((b'{"choices":[{"message":{"content":"stream\\', b'uD800"}}]}'))
    taped = _record_response(tmp_path, monkeypatch, response)
    assert json.loads(taped)["choices"][0]["message"]["content"] == "stream\ufffd"


def test_all_provider_string_shapes_and_nested_argument_escapes_are_repaired():
    arguments = (r'{"plain":"\uD800","escaped":"\\ud800",'
                 r'"paired":"\ud83D\udE00","mixed":["\ud800","\udfff"],'
                 r'"nested":{"bad":"\ud801"},"row":"\ud800\ud801\udc01"}')
    parsed = {"choices": [{"message": {
        "content": "<tool_call>bad\ud800</tool_call>",
        "reasoning_content": "think\udfff",
        "tool_calls": [{"function": {"name": "run\ud800", "arguments": arguments}}],
    }}], "bad\ud800": "key", "metadata": {"arguments": r'{"literal":"\ud800"}'}}
    same, response = sanitize_provider_response(parsed, json.dumps(parsed).encode("ascii"))
    assert same is parsed
    cleaned = json.loads(response)
    message = cleaned["choices"][0]["message"]
    assert message["content"].encode("unicode_escape") == b"<tool_call>bad\\ufffd</tool_call>"
    assert message["reasoning_content"].encode("unicode_escape") == b"think\\ufffd"
    call = message["tool_calls"][0]["function"]
    assert call["name"].encode("unicode_escape") == b"run\\ufffd"
    args_text = call["arguments"]
    assert r"\\ud800" in args_text and r"\ud83D\udE00" in args_text
    args = json.loads(args_text)
    assert args["plain"] == "\ufffd"
    assert args["escaped"] == r"\ud800"
    assert args["paired"] == "😀"
    assert args["mixed"] == ["\ufffd", "\ufffd"]
    assert args["nested"]["bad"] == "\ufffd"
    assert args["row"] == "\ufffd\U00010401"
    assert cleaned["bad\ufffd"] == "key"
    assert cleaned["metadata"]["arguments"] == r'{"literal":"\ud800"}'


def test_ask_tools_parses_the_sanitized_recorder_body(tmp_path, monkeypatch):
    body = {"choices": [{"message": {"content": "ok", "tool_calls": [{
        "function": {"name": "run\ud800", "arguments": r'{"value":"\ud800"}'},
    }]}, "finish_reason": "stop"}]}
    taped = _record_response(tmp_path, monkeypatch,
                             json.dumps(body, ensure_ascii=True).encode("ascii"))
    from probes import react_repair

    class Reply:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def read(self): return taped

    monkeypatch.setattr(react_repair, "MODEL", "scripted")
    monkeypatch.setattr(react_repair, "API", "http://127.0.0.1:1/v1/chat/completions")
    monkeypatch.setattr(react_repair.urllib.request, "urlopen", lambda *_args, **_kwargs: Reply())
    content, calls = react_repair.ask_tools([{"role": "user", "content": "go"}], tools=[])
    assert content == "ok"
    assert calls[0]["function"]["name"] == "run\ufffd"
    assert json.loads(calls[0]["function"]["arguments"]) == {"value": "\ufffd"}


def test_replayed_response_with_surrogate_is_sanitized_before_recording(tmp_path, monkeypatch):
    response = json.dumps({"choices": [{"message": {"content": "stream\ud800"}}]},
                          ensure_ascii=True).encode("ascii")
    endpoint = "http://127.0.0.1:8080/v1/chat/completions"
    wire = b'{"model":"scripted","messages":[]}'
    source = tmp_path / "source"
    (source / "wire").mkdir(parents=True)
    request_path, response_path = source / "wire/0001.request.json", source / "wire/0001.response.json"
    request_path.write_bytes(wire)
    response_path.write_bytes(response)
    tape = [{"send": 1, "request": str(request_path), "response": str(response_path),
             "request_digest": normalized_digest(wire),
             "response_sha256": hashlib.sha256(response).hexdigest()}]
    monkeypatch.setattr(_child, "HOST_STATE", dict(_child.HOST_STATE))
    monkeypatch.setattr(kernel_bounded_http, "request_bytes",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("live send")))
    run = tmp_path / "resume"
    run.mkdir()
    _child.install_wire_recorder(run, endpoint, Replayer(tape))
    returned = kernel_bounded_http.request_bytes(wire, endpoint, "")
    assert returned == (run / "wire/0001.response.json").read_bytes()
    assert json.loads(returned)["choices"][0]["message"]["content"] == "stream\ufffd"
    assert response_path.read_bytes() == response


@pytest.mark.parametrize("payload", [
    {"choices": [{"message": {"content": "ordinary"}}]},
    {"choices": [{"message": {"tool_calls": None}}]},
    {"choices": [{"message": {"tool_calls": []}}]},
    {"choices": [{"message": {}}]},
    {"choices": [{"message": {"tool_calls": [
        {"function": {"name": "one", "arguments": "{}"}},
        {"function": {"name": "two", "arguments": "{}"}},
    ]}}]},
], ids=["content-only", "tool-calls-null", "tool-calls-empty", "tool-calls-missing", "multiple-calls"])
def test_ordinary_response_shapes_keep_the_same_object_and_bytes(payload):
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    parsed = json.loads(raw)
    same, untouched = sanitize_provider_response(parsed, raw)
    assert same is parsed
    assert same == payload
    assert untouched is raw
    assert untouched == raw


def test_deeply_nested_reply_never_raises():
    for depth in (100, 400, 800, 900, 950, 980, 990, 2000):
        raw = ('{"a":' + "[" * depth + '"\\ud800"' + "]" * depth + "}").encode()
        try:
            parsed = json.loads(raw)
        except RecursionError:
            parsed = None
        response = sanitize_provider_response(parsed, raw)[1]
        assert response == raw or b"\\ufffd" in response
