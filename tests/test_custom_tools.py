import json
from pathlib import Path

import pytest

from kernel_custom_tools import CustomToolError, load_tool_files
from kernel_custom_tools import _pattern


def definition():
    return {"name": "sample_tool", "description": "Print one argument",
            "parameters": {"type": "object", "properties": {
                "value": {"type": "string", "maxLength": 100}}, "required": ["value"]},
            "argv": ["/bin/echo", "{value}"], "network": "none", "writes": "none",
            "replay": "local", "timeout_seconds": 3}


def load(tmp_path, defs, **kwargs):
    path = tmp_path / "tools.json"
    path.write_text(json.dumps({"schema": "stateframe-tools/1", "tools": defs}))
    return load_tool_files([path], builtin_names=frozenset({"shell"}),
                           shell_network=kwargs.get("shell_network", "loopback"),
                           tokenizer=kwargs.get("tokenizer"))


def mutate(field, value):
    item = definition()
    parts = field.split(".")
    target = item
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    return item


CASES = [
    ("name", "A", "name: expected ^[a-z][a-z0-9_]{1,40}$"),
    ("name", "x", "name: expected ^[a-z][a-z0-9_]{1,40}$"),
    ("name", "a" * 42, "name: expected ^[a-z][a-z0-9_]{1,40}$"),
    ("name", "shell", "name: duplicates a built-in or earlier custom tool"),
    ("description", "", "description: expected 1 to 300 characters"),
    ("description", "a" * 301, "description: expected 1 to 300 characters"),
    ("parameters", {}, "parameters: expected object with type, properties, required"),
    ("parameters.type", "array", "parameters: type must be object with properties"),
    ("parameters.required", ["missing"], "parameters.required: unknown property"),
    ("parameters.required", ["value", "value"], "parameters.required: expected unique property names"),
    ("parameters.required", [], "argv: placeholder properties must be required"),
    ("parameters.properties.value.type", "object", "parameters.properties.value:type: expected string, integer, number, or boolean"),
    ("parameters.properties.value.maxLength", 0, "parameters.properties.value:maxLength: expected integer from 1 to 1000"),
    ("parameters.properties.value.maxLength", 1001, "parameters.properties.value:maxLength: expected integer from 1 to 1000"),
    ("parameters.properties.value.pattern", "(a+)+", "parameters.properties.value:pattern: repetition must be bounded at 100"),
    ("parameters.properties.value.pattern", "(?=a)", "parameters.properties.value:pattern: must use a bounded, timeout-free expression"),
    ("parameters.properties.value.pattern", "[", "parameters.properties.value:pattern: invalid expression: unterminated character set at position 0"),
    ("parameters.properties.value.pattern", "a{0,100}b{0,100}c{0,100}", "parameters.properties.value:pattern: more than 2 repetitions"),
    ("parameters.properties.value.enum", [], "parameters.properties.value:enum: expected 1 to 20 values of the declared type"),
    ("parameters.properties.value.enum", ["a"] * 21, "parameters.properties.value:enum: expected 1 to 20 values of the declared type"),
    ("parameters.properties.value.enum", ["a", "a"], "parameters.properties.value:enum: duplicate values"),
    ("argv", [], "argv: expected nonempty string items"),
    ("argv", ["/bin/echo", "pre{value}"], "argv: placeholders must occupy a whole item and name a property"),
    ("argv", ["{value}"], "argv: executable cannot be a placeholder"),
    ("argv", ["/not/an/executable"], "argv: executable not found"),
    ("network", "internet", "network: expected none, loopback, or full"),
    ("network", "full", "network: exceeds --shell-network"),
    ("writes", "host", "writes: expected none or workspace"),
    ("replay", "automatic", "replay: expected local or external"),
    ("timeout_seconds", 0, "timeout_seconds: expected finite number greater than 0 and at most 600"),
    ("timeout_seconds", 601, "timeout_seconds: expected finite number greater than 0 and at most 600"),
]


@pytest.mark.parametrize("field,value,fragment", CASES)
def test_invalid_definitions(tmp_path, field, value, fragment):
    item = mutate(field, value)
    with pytest.raises(CustomToolError) as caught:
        load(tmp_path, [item])
    assert str(caught.value) == f"{tmp_path / 'tools.json'}:{item['name']}:{fragment}"


def test_valid_argv_and_digest_changes(tmp_path):
    first = load(tmp_path, [definition()])
    tool = first.tools[0]
    assert tool.build_argv({"value": "a; rm -rf /"}) == [str(Path("/bin/echo").resolve()), "a; rm -rf /"]
    with pytest.raises(CustomToolError, match="violates string constraint"):
        tool.build_argv({"value": "a" * 101})
    with pytest.raises(CustomToolError, match="NUL byte is not allowed"):
        tool.build_argv({"value": "a\0b"})
    changed = definition()
    changed["description"] = "Different"
    assert load(tmp_path, [changed]).digest != first.digest


def test_duplicate_across_files(tmp_path):
    one = tmp_path / "one.json"
    two = tmp_path / "two.json"
    body = json.dumps({"schema": "stateframe-tools/1", "tools": [definition()]})
    one.write_text(body)
    two.write_text(body)
    with pytest.raises(CustomToolError, match="two.json:sample_tool:name: duplicates"):
        load_tool_files([one, two], builtin_names=frozenset(), shell_network="none")


def test_schema_budget_uses_tokenizer(tmp_path):
    with pytest.raises(CustomToolError, match="1501 tokens exceeds 1500"):
        load(tmp_path, [definition()], tokenizer=lambda _: 1501)


def test_full_network_cannot_claim_local_replay(tmp_path):
    item = definition()
    item["network"] = "full"
    with pytest.raises(CustomToolError) as caught:
        load(tmp_path, [item], shell_network="full")
    assert str(caught.value) == f"{tmp_path / 'tools.json'}:sample_tool:replay: full network tools must use external replay"


def test_executable_digest_changes(tmp_path):
    executable = tmp_path / "custom"
    executable.write_text("#!/bin/sh\necho one\n")
    executable.chmod(0o755)
    item = definition()
    item["argv"][0] = str(executable)
    before = load(tmp_path, [item])
    executable.write_text("#!/bin/sh\necho two\n")
    assert load(tmp_path, [item]).digest != before.digest
    with pytest.raises(CustomToolError, match="executable changed since load"):
        before.tools[0].verify_executable()


@pytest.mark.parametrize("body,fragment", [
    ('{"schema":"stateframe-tools/1","tools":[],"tools":[]}', "duplicate JSON key tools"),
    ('{"schema":"stateframe-tools/1","tools":[NaN]}', "non-finite JSON number NaN"),
])
def test_noncanonical_json_refused(tmp_path, body, fragment):
    path = tmp_path / "tools.json"
    path.write_text(body)
    with pytest.raises(CustomToolError, match=fragment):
        load_tool_files([path], builtin_names=frozenset(), shell_network="none")


@pytest.mark.parametrize("expression", [r"^[a-z]{1,40}$", r"(ab){1,10}c?", r"\[{2}\]",
    r"[(){}+*?]{1,8}", r"[^]a]{0,5}", r"\d{1,3}\.\w{1,20}", r"a{1,2}?b{1,2}"])
def test_bounded_public_regex_subset(expression):
    _pattern(expression, "test")


@pytest.mark.parametrize("expression", [r"(a{1,2}){1,2}", r"a|b", r"a{1,}",
    r"a{101}", r"(a?){1,2}", r"(((((((((a)))))))))", r"\x61+", r"a*b+"])
def test_regex_rejects_unbounded_or_ambiguous_work(expression):
    with pytest.raises(CustomToolError):
        _pattern(expression, "test")


@pytest.mark.parametrize('timeout', [0.1, 600])
def test_timeout_matches_execution_limit(tmp_path, timeout):
    from kernel_exec_capability import ExecLimits
    tools = load(tmp_path, [mutate('timeout_seconds', timeout)])
    assert ExecLimits(wall_seconds=tools.tools[0].timeout_seconds).wall_seconds == timeout
