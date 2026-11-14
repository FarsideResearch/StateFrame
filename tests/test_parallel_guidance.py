"""The parent parallel contract stays accurate to the admitted tool schema."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from kernel_parallel import (PARALLEL_TOOL, ParallelInputError, _UNIT, _complete_arguments,
                             _planner_clauses, _request)
from kernel_parallel_admission import DirectorySnapshot, admit
from kernel_parallel_records import ParallelLimits
from kernel_request_tool_surface import _parallel_terminal_drop_enabled, _without_parallel
from probes.react_repair import (
    PARALLEL_PARENT_GUIDANCE,
    _parallel_parent_system_instruction,
    _parallel_parent_tool_surface,
)


def _tool_names(tools):
    return [tool["function"]["name"] for tool in tools]


def _example(description: str, label: str) -> dict:
    line = next(line for line in description.splitlines() if line.startswith(label))
    raw = line.partition(":")[2].strip()
    if label == "Invalid:":
        raw = raw.partition("  (")[0]
    return json.loads(raw)


def test_parallel_description_guidance_is_exact_and_complete():
    expected = (
        "Use parallel as soon as you know the task has 2+ independent parts: each part edits its own files and has its own test selector.\n"
        "Each subframe starts with its owned files and its test already loaded, so do not read those files yourself before dispatching.\n"
        "Typical flow: read the task and the test layout, then call parallel. Edit serially only when parts share a file or depend on each other.\n"
        "\n"
        "Valid:   {\"id\":\"csv\",\"goal\":\"Update the CSV adapters\",\"owns\":[\"src/adapters4/csv_reader.py\",\"src/adapters4/csv_writer.py\"],\"creates\":[],\"check\":\"tests/adapters/test_csv.py\"}\n"
        "Invalid: {\"id\":\"csv\",\"creates\":[\"src/adapters4/csv_reader.py\"]}  (file already exists: put it in owns)\n"
        "Subframes must not share an owned or created path."
    )
    assert PARALLEL_TOOL["function"]["description"].endswith(expected)
    assert PARALLEL_PARENT_GUIDANCE == (
        "Independent parts (separate files + separate tests): dispatch them together with `parallel` before reading their sources."
    )
    assert _UNIT["properties"]["owns"]["description"] == (
        "Existing files this subframe may edit. Every file the subframe changes must be listed here or in creates."
    )
    assert _UNIT["properties"]["creates"]["description"] == (
        "NEW files this subframe adds. Only paths that do not exist yet; an existing file goes in owns, never creates."
    )
    assert _UNIT["properties"]["check"]["description"] == (
        "the test selector that proves this subframe (e.g. tests/adapters/test_csv.py)."
    )
    assert [variant["type"] for variant in _UNIT["properties"]["check"]["oneOf"]] == [
        "string", "array", "object",
    ]


def test_parent_instruction_only_appears_when_parallel_is_on_the_menu():
    parallel = {"type": "function", "function": {"name": "parallel"}}
    read = {"type": "function", "function": {"name": "read"}}

    assert _parallel_parent_system_instruction("rules", [read]) == "rules"
    assert _parallel_parent_system_instruction("rules", [read, parallel]) == (
        "rules\n" + PARALLEL_PARENT_GUIDANCE
    )


def test_valid_description_example_is_admitted_verbatim(tmp_path):
    example = _example(PARALLEL_TOOL["function"]["description"], "Valid:")
    files = {
        "src/adapters4/csv_reader.py": "VALUE = 1\n",
        "src/adapters4/csv_writer.py": "VALUE = 2\n",
        "src/adapters4/json_reader.py": "VALUE = 3\n",
        "tests/adapters/test_csv.py": "def test_csv(): assert True\n",
        "tests/adapters/test_json.py": "def test_json(): assert True\n",
    }
    for name, source in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    snapshot = DirectorySnapshot(tmp_path, 1)
    args = {"subframes": [example, {
        "id": "json", "goal": "Update the JSON reader",
        "owns": ["src/adapters4/json_reader.py"],
        "check": {"tests": ["tests/adapters/test_json.py"]},
    }]}
    completed, _actions = _complete_arguments(args, snapshot, ParallelLimits())
    request = _request(completed, "example", "2")
    admitted = admit(request, snapshot, _planner_clauses(completed, ()), ParallelLimits(),
                     protected=lambda _path: False, case_insensitive=False)
    assert admitted.status == "ADMITTED", admitted.refusal


@pytest.mark.parametrize("check,selectors", [
    ("tests/test_alpha.py", ("tests/test_alpha.py",)),
    (["tests/test_alpha.py", "tests/test_beta.py"],
     ("tests/test_alpha.py", "tests/test_beta.py")),
    ({"tests": ["tests/test_alpha.py"]}, ("tests/test_alpha.py",)),
])
def test_check_shapes_normalize_for_completion_and_direct_parsing(tmp_path, check, selectors):
    (tmp_path / "alpha.py").write_text("VALUE = 1\n")
    args = {"subframes": [{"id": "alpha", "outcome": "Update alpha", "owns": ["alpha.py"],
                           "creates": [], "check": check}]}
    snapshot = DirectorySnapshot(tmp_path, 1)
    completed, _actions = _complete_arguments(args, snapshot, ParallelLimits())

    assert completed["subframes"][0]["check"] == {"tests": list(selectors)}
    assert args["subframes"][0]["check"] == check
    assert _request(completed, "completed", "2").units[0].check.selectors == selectors
    assert _request(args, "direct", "2").units[0].check.selectors == selectors


@pytest.mark.parametrize("check", [
    "", " \t ", [""], [" \t"], [3], [None], {"tests": [""]},
    {"tests": [3]}, {"tests": "tests/test_alpha.py"}, {}, None,
])
def test_check_shapes_reject_empty_and_non_string_entries(tmp_path, check):
    args = {"subframes": [{"id": "alpha", "outcome": "Update alpha", "owns": ["alpha.py"],
                           "creates": [], "check": check}]}
    snapshot = DirectorySnapshot(tmp_path, 1)

    for parse in (lambda: _complete_arguments(args, snapshot, ParallelLimits()),
                  lambda: _request(args, "direct", "2")):
        with pytest.raises(ParallelInputError) as refusal:
            parse()
        assert refusal.value.unit_id == "alpha"
        assert refusal.value.field == "check.tests"


def test_schema_examples_match_create_and_own_admission(tmp_path):
    description = PARALLEL_TOOL["function"]["description"]
    valid = _example(description, "Valid:")
    invalid = _example(description, "Invalid:")
    assert valid == {
        "id": "csv",
        "goal": "Update the CSV adapters",
        "owns": ["src/adapters4/csv_reader.py", "src/adapters4/csv_writer.py"],
        "creates": [],
        "check": "tests/adapters/test_csv.py",
    }
    assert invalid == {"id": "csv", "creates": ["src/adapters4/csv_reader.py"]}
    files = {
        "src/adapters4/csv_reader.py": "VALUE = 1\n",
        "src/adapters4/csv_writer.py": "VALUE = 2\n",
        "tests/adapters/test_csv.py": "def test_csv(): assert True\n",
    }
    snapshot_root = tmp_path / "repo"
    for name, source in files.items():
        path = snapshot_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    snapshot = DirectorySnapshot(snapshot_root, 1)
    limits = ParallelLimits()

    def result(args):
        completed, _actions = _complete_arguments(args, snapshot, limits)
        request = _request(completed, "example", "2")
        requirements = _planner_clauses(completed, ())
        return admit(request, snapshot, requirements, limits, protected=lambda _path: False,
                     case_insensitive=False)

    valid_manifest = {"subframes": [
        {"id": "reader", "goal": "Update the CSV reader",
         "owns": [valid["owns"][0]], "creates": [],
         "check": {"tests": [valid["check"]]}},
        {"id": "writer", "goal": "Update the CSV writer",
         "owns": [valid["owns"][1]], "creates": [],
         "check": {"tests": [valid["check"]]}},
    ]}
    assert result(valid_manifest).status == "ADMITTED"
    invalid_manifest = {"subframes": [
        {"id": "reader", "goal": "Update the CSV reader", "owns": [],
         "creates": [invalid["creates"][0]],
         "check": {"tests": [valid["check"]]}},
        {"id": "writer", "goal": "Update the CSV writer",
         "owns": [valid["owns"][1]], "creates": [],
         "check": {"tests": [valid["check"]]}},
    ]}
    refused = result(invalid_manifest)
    assert refused.status == "REFUSED"
    assert refused.refusal.reason == "CREATE_EXISTS"
    assert refused.refusal.paths == ("src/adapters4/csv_reader.py",)


def test_parallel_terminal_menu_policy_drops_only_verified_complete_results():
    tools = [
        {"type": "function", "function": {"name": "read"}},
        {"type": "function", "function": {"name": "parallel"}},
    ]
    success = SimpleNamespace(
        admission=SimpleNamespace(status="ADMITTED"),
        outcomes=(SimpleNamespace(status="COMMITTED"),),
        final_check="PASS",
        integration="VERIFIED",
    )
    refused = SimpleNamespace(admission=SimpleNamespace(status="REFUSED"))
    incomplete = SimpleNamespace(
        admission=SimpleNamespace(status="ADMITTED"),
        outcomes=(SimpleNamespace(status="COMMITTED_UNVERIFIED"),),
        final_check="PASS",
        integration="UNVERIFIED",
    )

    assert "parallel" in _tool_names(_parallel_parent_tool_surface(tools, None))
    assert "parallel" not in _tool_names(_parallel_parent_tool_surface(tools, success))
    assert "parallel" in _tool_names(_parallel_parent_tool_surface(tools, refused))
    assert "parallel" in _tool_names(_parallel_parent_tool_surface(tools, incomplete))
    assert _parallel_parent_tool_surface(tools, success, enabled=False) == tools


def test_terminal_parent_filter_removes_stable_listed_parallel_schema(monkeypatch):
    runtime = SimpleNamespace(_parallel_parent_terminal=True)
    stale_stable_surface = [
        {"type": "function", "function": {"name": "read"}},
        {"type": "function", "function": {"name": "parallel"}},
    ]

    monkeypatch.setenv("STATEFRAME_PARALLEL_DROP_TOOL_AFTER_TERMINAL", "on")
    assert _parallel_terminal_drop_enabled(runtime)
    assert _tool_names(_without_parallel(stale_stable_surface)) == ["read"]

    monkeypatch.setenv("STATEFRAME_PARALLEL_DROP_TOOL_AFTER_TERMINAL", "off")
    assert not _parallel_terminal_drop_enabled(runtime)
