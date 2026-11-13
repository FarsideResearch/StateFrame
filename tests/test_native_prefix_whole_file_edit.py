"""Native-prefix proof must model whole-file `edit` authority.

Live run2 (sflive, 2026-11-11): the model batched three whole-file `edit` calls to
three disjoint files in one response.  The executor applied call 1 as a complete-file
replacement (EditAuthority mode=whole_file, kernel_edit_affordance:123 "omitted content
is removed"), but the prefix proof re-simulated call 1 with `definition_change`
(complete-definition splice, kernel_native_effect_chain:31-56), produced a different
revision, and refused call 2 with "prefix result differs from exact declared
replacement or changed another product".  Calls 2 and 3 were discarded.

RED: OrderedSources.apply("edit", ...) under whole-file authority must yield exactly
the declared function_source, matching what the executor writes.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from kernel_continuation_state import ContinuationError
from stateframe.engine.kernel_native_effect_chain import OrderedSources, revision

PATH = "toolkit/semver.py"
STUB = '"""Semantic version helpers."""\n\n\nclass Version:\n    """Parsed semantic version."""\n\n    def __init__(self, raw):\n        raise NotImplementedError\n'
WHOLE_FILE = (
    '"""Semantic version parsing and comparison (SemVer 2.0.0)."""\n\nimport re\n\n\n'
    'class Version:\n    """Parsed semantic version."""\n\n    _RE = re.compile(r"^(\\d+)\\.(\\d+)\\.(\\d+)$")\n\n'
    '    def __init__(self, raw):\n        self.raw = raw\n'
)


def _sources(authority_mode: str) -> OrderedSources:
    binding = {"product_files": {PATH: revision(STUB)}, "harness_files": {},
               "edit_authority": {"mode": authority_mode, "tools": ["edit", "str_replace"],
                                  "editable_paths": [PATH]}}
    payload = {"messages": [{"role": "user", "content": f"[Working source: {PATH}]\n{STUB}"}]}
    return OrderedSources([], {"sequence": 1}, payload, binding, final_delivery=None)


def test_whole_file_edit_prefix_simulation_matches_executor_replacement() -> None:
    sources = _sources("whole_file")
    assert sources.apply("edit", {"path": PATH, "function_source": WHOLE_FILE}, 0,
                         pending=False) == PATH
    # Executor semantics under whole-file authority: the file becomes function_source.
    assert sources.texts[PATH] == WHOLE_FILE, (
        "prefix proof spliced the definition instead of replacing the whole file; "
        "the resulting revision differs from the applied one, so the next batched edit "
        "is refused with 'prefix result differs from exact declared replacement'")
    assert revision(sources.texts[PATH]) == revision(WHOLE_FILE)


def test_whole_file_edit_prefix_accepts_multi_definition_file() -> None:
    """ini.py in run2: docstring + class + function.  Whole-file authority allows it;
    the complete-definition simulator refuses 'one unambiguous target'."""
    multi = STUB + "\n\ndef parse(text):\n    return Version(text)\n"
    sources = _sources("whole_file")
    sources.apply("edit", {"path": PATH, "function_source": multi}, 0, pending=False)
    assert sources.texts[PATH] == multi


# --- class tests: batched whole-file edits in one response -------------------------

MODULE_A = '"""A."""\n\n\ndef a():\n    return 1\n'
MODULE_B = '"""B."""\n\n\nclass B:\n    pass\n'
MODULE_C = '"""C."""\n\nimport os\n\n\ndef c():\n    return os.sep\n'
FILES = {"pkg/a.py": MODULE_A, "pkg/b.py": MODULE_B, "pkg/c.py": MODULE_C}
WHOLE_A = '"""A, rewritten."""\n\nimport math\n\n\nclass A:\n    pass\n\n\ndef a():\n    return math.pi\n'
WHOLE_B = '"""B, rewritten."""\n\n\nclass B:\n    value = 2\n\n\ndef make_b():\n    return B()\n'
WHOLE_C = '"""C, rewritten."""\n\n\nCONST = 3\n\n\ndef c():\n    return CONST\n'


def _multi_sources(authority_mode: str | None, *, tools: list | None = None) -> OrderedSources:
    binding = {"product_files": {path: revision(text) for path, text in FILES.items()},
               "harness_files": {}}
    if authority_mode is not None:
        binding["edit_authority"] = {"mode": authority_mode, "tools": ["edit", "str_replace"],
                                     "editable_paths": list(FILES)}
    payload = {"messages": [{"role": "user", "content": f"[Working source: {path}]\n{text}"}
                            for path, text in FILES.items()]}
    if tools is not None:
        payload["tools"] = tools
    return OrderedSources([], {"sequence": 1}, payload, binding, final_delivery=None)


def _tool_menu(source_description: str) -> list[dict]:
    return [{"type": "function", "function": {
        "name": "edit", "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "function_source": {"type": "string", "description": source_description}}}}},
        {"type": "function", "function": {"name": "str_replace", "parameters": {}}}]


def test_batched_whole_file_edits_to_disjoint_files_all_pass() -> None:
    """Three whole-file edits (two of them multi-definition) in one response: every
    prefix result must equal the declared complete file."""
    sources = _multi_sources("whole_file")
    batch = [("pkg/a.py", WHOLE_A), ("pkg/b.py", WHOLE_B), ("pkg/c.py", WHOLE_C)]
    for number, (path, text) in enumerate(batch):
        assert sources.apply("edit", {"path": path, "function_source": text}, number,
                             pending=False) == path
    for path, text in batch:
        assert sources.texts[path] == text
        assert revision(sources.texts[path]) == revision(text)


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("authority_mode", ["whole_file", "declared_product_files", None])
def test_whole_file_mode_survives_compacted_submitted_menu(monkeypatch, compact, authority_mode) -> None:
    """Record real host batch metadata, then replay it after journal serialization."""
    from kernel_capacity_retry import compact_tool_prose
    from kernel_measurement import ACTIVE_REQUEST
    from kernel_native_tools import NativeToolBatch
    from kernel_pilot import ACTIVE_RUNTIME
    from stateframe.engine.kernel_edit_affordance import WHOLE_FILE_SOURCE_DESCRIPTION

    monkeypatch.setenv("STATEFRAME_EMPIRICAL_FIXES", "off")
    monkeypatch.setenv("KERNEL_NATIVE_PREFIX_AUTHORITY", "0")
    monkeypatch.setenv("STATEFRAME_FRAME_EDIT_INTEGRITY", "off")
    events = []

    def emit(kind, **fields):
        event = {"kind": kind, "event_id": str(len(events)), **fields}
        events.append(event)
        return event

    tools = _tool_menu(WHOLE_FILE_SOURCE_DESCRIPTION)
    payload = {"tools": tools}
    if compact:
        compact_tool_prose(payload)
        assert "description" not in tools[0]["function"]["parameters"]["properties"]["function_source"]
    wire = json.dumps(payload).encode()
    binding = _multi_sources(None).origin
    runtime = SimpleNamespace(prepared_binding=binding, binding=lambda: binding,
                              edit_contract={"mode": authority_mode} if authority_mode else None)
    capture = SimpleNamespace(journal=SimpleNamespace(emit=emit),
                              request_id="origin", snapshot_id="snapshot")
    calls = [{"id": str(index), "type": "function", "function": {
        "name": "edit", "arguments": json.dumps({"path": path, "function_source": text,
                                                   "mode": "whole_file"})}}
        for index, (path, text) in enumerate([("pkg/a.py", WHOLE_A), ("pkg/b.py", WHOLE_B)])]
    runtime_token, capture_token = ACTIVE_RUNTIME.set(runtime), ACTIVE_REQUEST.set(capture)
    try:
        NativeToolBatch().next(lambda *_args, **_kwargs: ("", calls), [], tools=tools)
    finally:
        ACTIVE_REQUEST.reset(capture_token)
        ACTIVE_RUNTIME.reset(runtime_token)
    assert json.dumps(payload).encode() == wire
    batch = json.loads(json.dumps(next(e for e in events if e["kind"] == "native_batch_received")))
    assert batch["edit_authority_mode"] == (authority_mode or "definition")
    assert batch["origin_binding"] == binding  # metadata never changes workspace identity
    sources = _multi_sources(None, tools=json.loads(wire)["tools"])
    sources = OrderedSources([], {"sequence": 1}, sources.payload, batch["origin_binding"],
                             final_delivery=None, edit_mode=batch["edit_authority_mode"])
    if authority_mode != "whole_file":
        with pytest.raises(ContinuationError, match="one unambiguous target"):
            sources.apply("edit", {"path": "pkg/a.py", "function_source": WHOLE_A}, 0, pending=False)
        return
    sources.apply("edit", {"path": "pkg/a.py", "function_source": WHOLE_A}, 0, pending=False)
    sources.apply("edit", {"path": "pkg/b.py", "function_source": WHOLE_B}, 1, pending=False)
    assert sources.texts["pkg/a.py"] == WHOLE_A and sources.texts["pkg/b.py"] == WHOLE_B


def test_model_prose_and_arguments_cannot_select_whole_file_mode() -> None:
    from stateframe.engine.kernel_edit_affordance import WHOLE_FILE_SOURCE_DESCRIPTION

    for mode in (None, "top_level_function_or_class"):
        sources = _multi_sources(mode, tools=_tool_menu(WHOLE_FILE_SOURCE_DESCRIPTION))
        sources.payload["messages"].append({"role": "assistant", "content": json.dumps(
            {"edit_authority": {"mode": "whole_file"}})})
        with pytest.raises(ContinuationError, match="one unambiguous target"):
            sources.apply("edit", {"path": "pkg/a.py", "function_source": WHOLE_A,
                                   "edit_authority": {"mode": "whole_file"}}, 0, pending=False)


def test_definition_mode_menu_keeps_definition_splice() -> None:
    sources = _multi_sources(None, tools=_tool_menu(
        "Supply one complete top-level definition (function or class); "
        "the current source of that definition is the authoritative base."))
    sources.apply("edit", {"path": "pkg/a.py", "function_source": "def a():\n    return 2\n"},
                  0, pending=False)
    assert sources.texts["pkg/a.py"] == '"""A."""\n\n\ndef a():\n    return 2\n'
    with pytest.raises(ContinuationError, match="one unambiguous target"):
        sources.apply("edit", {"path": "pkg/b.py", "function_source": WHOLE_B}, 1, pending=False)


def test_whole_file_edits_mixed_with_definition_edits_in_one_response() -> None:
    """Definition-mode authority: definition edits splice; whole-file authority: files
    replace.  Both kinds in one response under their own frame's authority all pass."""
    definition = _multi_sources("top_level_function_or_class")
    definition.apply("edit", {"path": "pkg/a.py", "function_source": "def a():\n    return 2\n"},
                     0, pending=False)
    definition.apply("str_replace", {"path": "pkg/b.py", "old_str": "pass", "new_str": "x = 1"},
                     1, pending=False)
    assert definition.texts["pkg/a.py"] == '"""A."""\n\n\ndef a():\n    return 2\n'
    assert definition.texts["pkg/b.py"] == '"""B."""\n\n\nclass B:\n    x = 1\n'

    whole = _multi_sources("whole_file")
    whole.apply("edit", {"path": "pkg/a.py", "function_source": WHOLE_A}, 0, pending=False)
    whole.apply("str_replace", {"path": "pkg/b.py", "old_str": "pass", "new_str": "x = 1"},
                1, pending=False)
    whole.apply("edit", {"path": "pkg/c.py", "function_source": WHOLE_C}, 2, pending=False)
    assert whole.texts["pkg/a.py"] == WHOLE_A
    assert whole.texts["pkg/b.py"] == '"""B."""\n\n\nclass B:\n    x = 1\n'
    assert whole.texts["pkg/c.py"] == WHOLE_C


def test_mismatched_whole_file_result_is_still_refused() -> None:
    """The proof compares revisions; a result that differs from the declared complete file
    (what the executor wrote) must not verify. Unparsable candidates are refused."""
    sources = _multi_sources("whole_file")
    sources.apply("edit", {"path": "pkg/a.py", "function_source": WHOLE_A}, 0, pending=False)
    assert revision(sources.texts["pkg/a.py"]) != revision(WHOLE_A.rstrip("\n"))
    assert revision(sources.texts["pkg/a.py"]) != revision(WHOLE_A + "\n# extra\n")
    with pytest.raises(ContinuationError, match="invalid whole-file source"):
        sources.apply("edit", {"path": "pkg/c.py", "function_source": "def c(:\n"}, 2, pending=False)
    with pytest.raises(ContinuationError, match="repeated identical call"):
        sources.apply("edit", {"path": "pkg/a.py", "function_source": WHOLE_A}, 3, pending=False)


@pytest.mark.parametrize("replacement", ["", "\n", "   \n", "\t\n", "\r\n",
    "# replacement\n", "VALUE = 'café'\r\n", "def broken(:\n", "\x00"])
def test_whole_file_executor_proof_parity(tmp_path, replacement) -> None:
    """Comment-only __init__.py may become empty; preserve accepted bytes exactly."""
    from probes.react_repair import _replace_file

    path = "pkg/__init__.py"
    original = "# package marker\n"
    target = tmp_path / path
    target.parent.mkdir()
    target.write_bytes(original.encode())
    receipt = _replace_file(target, replacement, root=tmp_path)
    binding = {"product_files": {path: revision(original)}, "harness_files": {},
               "edit_authority": {"mode": "whole_file"}}
    payload = {"messages": [{"role": "user", "content": f"[Working source: {path}]\n{original}"}]}
    sources = OrderedSources([], {"sequence": 1}, payload, binding, final_delivery=None)
    try:
        sources.apply("edit", {"path": path, "function_source": replacement}, 0, pending=False)
    except ContinuationError:
        assert receipt.startswith("edit REJECTED:"), receipt
        assert target.read_bytes() == original.encode()
    else:
        assert receipt.startswith("edit applied."), receipt
        assert sources.texts[path].encode() == target.read_bytes() == replacement.encode()
