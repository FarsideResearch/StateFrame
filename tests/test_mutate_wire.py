import json

import pytest

from kernel_mutate_wire import WireError, build_stdin_doc, parse_stdout, request_key, strict_loads, validate_schema


def frame(value):
    body = json.dumps(value, separators=(",", ":")).encode()
    return b"SFMW1 " + str(len(body)).encode() + b"\n" + body


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"a":{"b":1,"b":2}}', b'{"x":NaN}', b'{"x":1e999}', b'{"x":12345678901234567890123456789012345678901}', b'"\\ud800"', b'[' * 17 + b'0' + b']' * 17])
def test_strict_json_rejects_bad_values(raw):
    with pytest.raises(WireError):
        strict_loads(raw)


def test_frame_requires_exact_boundary_and_call():
    valid = frame({"wire": 1, "call": "c", "kind": "result", "result": {"ok": True}})
    assert parse_stdout(valid, call="c").result == {"ok": True}
    header, body = valid.split(b"\n", 1)
    for raw in (valid + b"x", b"x" + valid, b"SFMW1 0" + header[6:] + b"\n" + body, valid[:4]):
        with pytest.raises(WireError):
            parse_stdout(raw, call="c")
    with pytest.raises(WireError):
        parse_stdout(valid, call="d")


def test_need_rejects_recorded_and_invalid_requests():
    value = {"wire": 1, "call": "c", "kind": "need", "fetch": [{"method": "GET", "url": "https://a.test", "accept": ""}]}
    raw = frame(value)
    assert len(parse_stdout(raw, call="c").fetch) == 1
    with pytest.raises(WireError):
        parse_stdout(raw, call="c", recorded_keys={request_key("https://a.test", "")})
    value["fetch"] = []
    with pytest.raises(WireError):
        parse_stdout(frame(value), call="c")
    value["fetch"] = [{"method": "GET", "url": "https://a.test/" + "x" * 2050, "accept": ""}]
    with pytest.raises(WireError):
        parse_stdout(frame(value), call="c")


def test_builder_and_schema_validation():
    data = build_stdin_doc(wire=1, call="c", launch="l", phase=0, role="payload", input={}, inputs=[], responses=[], grants={"network": False, "writes": False}, max_fetch=4)
    assert strict_loads(data)["call"] == "c"
    assert validate_schema({"x": 2}, {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"], "additionalProperties": False})
    with pytest.raises(WireError):
        validate_schema({"x": True}, {"type": "object", "properties": {"x": {"type": "integer"}}})
    with pytest.raises(WireError):
        validate_schema({}, {"$ref": "foo"})


def test_result_schema_rejects_hidden_unsupported_keyword():
    raw = frame({"wire": 1, "call": "c", "kind": "result", "result": {}})
    with pytest.raises(WireError):
        parse_stdout(raw, call="c", output_schema={"type": "object", "properties": {"missing": {"type": "string", "pattern": "x"}}})


def test_builder_rejects_bool_integer_and_response_mismatch():
    values = dict(wire=1, call="c", launch="l", phase=True, role="payload", input={}, inputs=[], responses=[], grants={"network": False, "writes": False}, max_fetch=4)
    with pytest.raises(WireError):
        build_stdin_doc(**values)
    values["phase"] = 0
    values["responses"] = [{"key": request_key("https://else.test", ""), "method": "GET", "url": "https://a.test", "accept": "", "status": 200, "headers": {}, "final_url": "https://a.test", "body": "/resp/a", "bytes": 1, "error": None}]
    with pytest.raises(WireError):
        build_stdin_doc(**values)


def test_frame_rejects_huge_header_and_unknown_key():
    with pytest.raises(WireError):
        parse_stdout(b"SFMW1 " + b"9" * 10000 + b"\n{}", call="c")
    with pytest.raises(WireError):
        parse_stdout(frame({"wire": 1, "call": "c", "kind": "result", "result": {}, "surprise": 1}), call="c")


def test_need_phase_and_total_fetch_limit():
    value = {"wire": 1, "call": "c", "kind": "need", "fetch": [{"method": "GET", "url": "https://a.test", "accept": ""}]}
    with pytest.raises(WireError):
        parse_stdout(frame(value), call="c", phase=3)
    with pytest.raises(WireError):
        parse_stdout(frame(value), call="c", recorded_keys={"sha256:" + str(x) * 64 for x in "1234"})


def test_url_control_char_and_enum_bool_integer_rejected():
    value = {"wire": 1, "call": "c", "kind": "need", "fetch": [{"method": "GET", "url": "https://a.test\r\nX: y", "accept": ""}]}
    with pytest.raises(WireError):
        parse_stdout(frame(value), call="c")
    with pytest.raises(WireError):
        validate_schema(True, {"type": "boolean", "enum": [1]})
    with pytest.raises(WireError):
        validate_schema({}, {"type": "object", "maxLength": 3})


def test_builder_pins_snapshot_paths_and_ids():
    values = dict(wire=1, call="c", launch="l", phase=0, role="payload", input={}, inputs=[
        {"id": "in0", "path": "/etc/passwd", "bytes": 1, "name": "sample"}], responses=[],
        grants={"network": False, "writes": False}, max_fetch=4)
    with pytest.raises(WireError):
        build_stdin_doc(**values)
    values["inputs"][0]["path"] = "/in/in0"
    values["inputs"].append(dict(values["inputs"][0]))
    with pytest.raises(WireError):
        build_stdin_doc(**values)
    values["inputs"] = []
    values["responses"] = [{"key": request_key("https://a.test", ""), "method": "GET", "url": "https://a.test", "accept": "", "status": 200, "headers": {}, "final_url": "https://a.test", "body": "/in/in0", "bytes": 1, "error": None}]
    with pytest.raises(WireError):
        build_stdin_doc(**values)


def test_invalid_url_and_surrogate_have_wire_errors():
    with pytest.raises(WireError):
        request_key("https://[broken", "")
    with pytest.raises(WireError):
        request_key("https://a.test", "\ud800")
