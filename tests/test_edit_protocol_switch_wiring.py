"""STATEFRAME_EDIT_PROTOCOL wiring: shipped-profile pairs, explicit legacy_safe identity,
multi-turn wire stability, boundary inputs, and the shell -> run spec -> engine child path.

The menu and executor units live in test_edit_protocol_menu.py, test_edit_source_tool.py
and test_rewrite_source_tool.py; this file pins how the switch composes with the shipped
profile and how it travels from the user's shell to the engine and its subframes.
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from kernel_edit_handles import DeliveredSpan, SpanKind, mint_handles
from kernel_edit_protocol_switch import MODE_ENV, handles_enabled, mode
from kernel_edit_source_tool import EditProtocolRunState, execute_edit_source, execute_rewrite_source
from kernel_edit_source_tool_surface import HIDDEN_LEGACY_TOOLS
from kernel_edit_tool_names import EDIT_SOURCE_TOOL, REWRITE_SOURCE_TOOL

BOUND_ENV = "STATEFRAME_BOUND_EDIT"
HANDLE_PATTERN = re.compile(r"\bq\d+h\d+\b")
CURRENT_SOURCE_MARK = "[CURRENT SOURCE"
PROTOCOL_TOOLS = (EDIT_SOURCE_TOOL, REWRITE_SOURCE_TOOL)
PAIRS = [(protocol, bound) for protocol in ("legacy_safe", "handles") for bound in ("off", "on")]
RULES = "def qualifies(cart):\n    cart.minimum_spend = 25\n    return cart.total >= cart.minimum_spend\n"
RULES_PATH = "shop/pricing/rules.py"


class _Stop(BaseException):
    """Stop the worker once the wanted number of requests has been captured."""


def _set_protocol(monkeypatch, protocol: str | None) -> None:
    if protocol is None:
        monkeypatch.delenv(MODE_ENV, raising=False)
    else:
        monkeypatch.setenv(MODE_ENV, protocol)


def _names(tools) -> list[str]:
    return [tool["function"]["name"] for tool in tools]


def _menus(protocol: str | None, monkeypatch) -> list[str]:
    """Serialized menus for every probe-mode combination under one protocol value."""
    from probes import react_repair
    _set_protocol(monkeypatch, protocol)
    return [json.dumps(react_repair._tools_for_probe_modes(probe_gate_on=gate, observation_probe_on=probe),
                       sort_keys=True)
            for gate in (False, True) for probe in (False, True)]


def _drive(tmp_path: Path, monkeypatch, *, protocol: str | None, steps: list[dict], stop_at: int,
           target: str = RULES_PATH, handle_kind: SpanKind | None = None,
           after_model_response=None):
    """Run the real worker against a scripted endpoint; capture each request's messages and tools."""
    import kernel_edit_request_map
    from kernel_measurement import ACTIVE_REQUEST
    from kernel_pilot import ACTIVE_PARENT, ACTIVE_RUNTIME
    from kernel_stateframe_runtime import StateFrameRuntime
    from kernel_verdict_identity import _ACTIVE_PROVIDER
    from stateframe.engine.probes import react_repair
    from stateframe.scripted_endpoint import Script, serve
    from stateframe.task_row import build_row
    from stateframe.workcopy import create

    repo = tmp_path / "repo"
    (repo / target).parent.mkdir(parents=True)
    (repo / target).write_text(RULES, encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests/test_pricing.py").write_text("def test_pricing():\n    assert False\n", encoding="utf-8")
    copy_root = tmp_path / "copy"
    row = build_row(create(repo, copy_root), task="Raise the minimum spend",
                    target=target, task_id="edit-proto")
    output = tmp_path / "run/worker/measurement-runs"
    output.mkdir(parents=True)
    private_temp = tmp_path / "worker-tmp"
    private_temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(private_temp))
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "stateframe-home"))
    monkeypatch.setenv("KERNEL_INTERVENTIONS", "1")
    monkeypatch.setenv("STATEFRAME_IDEALPATH", "off")
    _set_protocol(monkeypatch, protocol)
    monkeypatch.setattr(sys, "argv", [sys.argv[0]])
    # Unwrap an earlier drive's patches in the same test, or the captures would chain.
    original_init = getattr(StateFrameRuntime.__init__, "_wiring_original", StateFrameRuntime.__init__)

    def scoped_init(self, owner, **kwargs):
        kwargs["output"] = output
        original_init(self, owner, **kwargs)

    scoped_init._wiring_original = original_init
    monkeypatch.setattr(StateFrameRuntime, "__init__", scoped_init)
    # Continuation workspaces live under POC; keep them out of the engine tree.
    monkeypatch.setattr(react_repair, "POC", tmp_path / "engine")
    monkeypatch.setattr(react_repair, "MAX_STEPS", len(steps) + 4)
    server = serve(Script({"model": "scripted", "tokenize": {"mode": "ok"}, "steps": steps}))
    monkeypatch.setattr(react_repair, "API", f"http://127.0.0.1:{server.server_port}/v1/chat/completions")
    monkeypatch.setattr(react_repair, "MODEL", "scripted")
    monkeypatch.setattr(react_repair, "KEY", "")
    original_ask = getattr(react_repair.ask_tools, "_wiring_original", react_repair.ask_tools)
    sends: list[tuple[list[dict], list[dict]]] = []

    def capture(messages, *args, **kwargs):
        tools = kwargs.get("tools")
        if tools is None and len(args) >= 3:
            tools = args[2]
        sends.append((json.loads(json.dumps(messages)), json.loads(json.dumps(tools or []))))
        if len(sends) >= stop_at:
            raise _Stop
        response = original_ask(messages, *args, **kwargs)
        if handle_kind is not None:
            binding = kernel_edit_request_map.current_binding()
            assert binding is not None
            # This harness bypasses final wire composition. Supply the map
            # that the transport would bind for the delivered target.
            source = (binding.root / target).read_text(encoding="utf-8")
            binding.state.set_request(mint_handles(1, [
                DeliveredSpan.from_source(target, source, kind=handle_kind),
            ]))
        if after_model_response is not None:
            after_model_response(len(sends) - 1)
        return response

    capture._wiring_original = original_ask
    monkeypatch.setattr(react_repair, "ask_tools", capture)
    monkeypatch.setattr(kernel_edit_request_map, "_RUN_BINDING",
                        kernel_edit_request_map.current_binding())
    contexts = (ACTIVE_PARENT, ACTIVE_RUNTIME, ACTIVE_REQUEST, _ACTIVE_PROVIDER)
    tokens = [context.set(None) for context in contexts]
    try:
        with pytest.raises(_Stop):
            react_repair.work(row)
    finally:
        server.shutdown()
        server.server_close()
        for context, token in zip(contexts, tokens):
            context.reset(token)
    return sends, copy_root


def _tool_replies(messages: list[dict]) -> list[str]:
    return [str(message.get("content")) for message in messages if message.get("role") == "tool"]


def _normalized(sends, tmp_path: Path) -> list[str]:
    return [json.dumps({"messages": messages, "tools": tools}, sort_keys=True).replace(str(tmp_path), "<ROOT>")
            for messages, tools in sends]


def _edit_source_step(handle: str, old: str = "25", new: str = "30") -> dict:
    return {"tool": EDIT_SOURCE_TOOL, "required": True,
            "arguments": {"edits": [{"handle": handle, "old_text": old, "new_text": new}]}}


def _str_replace_step(old: str, new: str) -> dict:
    return {"tool": "str_replace", "required": True,
            "arguments": {"path": RULES_PATH, "old_str": old, "new_str": new}}


def _guarded_edit_step(editor: str, target: str) -> tuple[dict, SpanKind | None]:
    if editor == EDIT_SOURCE_TOOL:
        return _edit_source_step("q1h1"), SpanKind.SPAN
    if editor == REWRITE_SOURCE_TOOL:
        return ({"tool": editor, "required": True, "arguments": {
            "handle": "q1h1", "new_text": RULES.replace("25", "30"),
        }}, SpanKind.REWRITE)
    step = _str_replace_step("25", "30")
    step["arguments"]["path"] = target
    return step, None


@pytest.mark.parametrize("editor", ("str_replace", *PROTOCOL_TOOLS))
def test_edit_guards_pytest_collection(editor, tmp_path, monkeypatch):
    target = "shop/pricing/pytest_settings.py"
    step, kind = _guarded_edit_step(editor, target)
    sends, root = _drive(tmp_path, monkeypatch, protocol="handles" if kind else None,
                         steps=[step], stop_at=2, target=target, handle_kind=kind)

    assert "protected test tree: pytest_settings.py can change test collection" in _tool_replies(sends[-1][0])[-1]
    assert (root / target).read_text() == RULES


@pytest.mark.parametrize("editor", ("str_replace", *PROTOCOL_TOOLS))
def test_edit_refuses_protected_hook_path_before_pre_tool(editor, tmp_path, monkeypatch):
    from stateframe.engine.probes import react_repair

    hooks = Mock(protected_paths=(), state={})
    hooks.protected.side_effect = lambda path: path.as_posix().endswith("/" + RULES_PATH)
    hooks.pre_action.return_value = None
    monkeypatch.setattr(react_repair, "_hook_wiring", hooks, raising=False)
    step, kind = _guarded_edit_step(editor, RULES_PATH)

    sends, root = _drive(tmp_path, monkeypatch, protocol="handles" if kind else None,
                         steps=[step], stop_at=2, handle_kind=kind)

    hooks.pre_action.assert_not_called()
    assert "REFUSED HOOK_FILE" in _tool_replies(sends[-1][0])[-1]
    assert (root / RULES_PATH).read_text() == RULES


def _repo_file(tmp_path: Path, text: str, name: str = "pkg/mod.py") -> tuple[Path, Path]:
    root = tmp_path / "repo"
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    return root, path


def _state(name: str, text: str, *, kind: SpanKind = SpanKind.SPAN, ordinal: int = 5) -> EditProtocolRunState:
    state = EditProtocolRunState()
    state.set_request(mint_handles(ordinal, [DeliveredSpan.from_source(name, text, kind=kind)]))
    return state


def _edit(root: Path, state: EditProtocolRunState, edits: list[dict]):
    return execute_edit_source(root, {"edits": edits}, state=state, read_bytes=Path.read_bytes)


# --- T1: shipped profile x (legacy_safe | handles) x STATEFRAME_BOUND_EDIT (off | on) --------------

@pytest.mark.parametrize("protocol,bound", PAIRS)
def test_t1_shipped_profile_pair_menu(shipped_runtime_profile, monkeypatch, protocol, bound):
    monkeypatch.setenv(BOUND_ENV, bound)
    baseline = _menus(None, monkeypatch)
    menus = _menus(protocol, monkeypatch)
    if protocol == "legacy_safe":
        assert menus == baseline
        return
    for menu, base in zip(menus, baseline):
        tools, base_tools = json.loads(menu), json.loads(base)
        kept = [tool for tool in base_tools if tool["function"]["name"] not in HIDDEN_LEGACY_TOOLS]
        assert tools[:len(kept)] == kept, "non-editor tools must keep their bytes and order"
        assert _names(tools[len(kept):]) == list(PROTOCOL_TOOLS)
        assert not HIDDEN_LEGACY_TOOLS & set(_names(tools))
        assert HANDLE_PATTERN.search(menu) is None


@pytest.mark.parametrize("protocol,bound", PAIRS)
def test_t1_shipped_profile_pair_executor_behavior(shipped_runtime_profile, monkeypatch, tmp_path,
                                                   protocol, bound):
    """The executors do not read BOUND_EDIT; the legacy writer does not read the protocol."""
    monkeypatch.setenv(BOUND_ENV, bound)
    _set_protocol(monkeypatch, protocol)
    root, path = _repo_file(tmp_path, RULES, RULES_PATH)
    state = _state(RULES_PATH, RULES)
    handle = state.handle_map.records[0].handle
    out = _edit(root, state, [{"handle": handle, "old_text": "= 25", "new_text": "= 30"}])
    assert out.status == "APPLIED" and path.read_text() == RULES.replace("= 25", "= 30")

    from probes.react_repair import str_replace_edit
    legacy_root, legacy_path = _repo_file(tmp_path / "legacy", RULES, RULES_PATH)
    observed = str_replace_edit(legacy_path, "= 25", "= 40", root=legacy_root)
    _set_protocol(monkeypatch, None)
    unset_root, unset_path = _repo_file(tmp_path / "unset", RULES, RULES_PATH)
    expected = str_replace_edit(unset_path, "= 25", "= 40", root=unset_root)
    assert observed.replace(str(legacy_root), "<R>") == expected.replace(str(unset_root), "<R>")
    assert legacy_path.read_bytes() == unset_path.read_bytes()


@pytest.mark.parametrize("protocol,bound", PAIRS)
def test_t1_switch_pair_run_dispatch(monkeypatch, tmp_path, protocol, bound):
    # Run level at the engine defaults: the scripted worker harness cannot complete a request under the full
    # shipped profile (its frame-only projection needs a unit context); the shipped profile is covered at the
    # menu and executor level above, and dispatch reads only these two switches.
    monkeypatch.setenv(BOUND_ENV, bound)
    sends, copy_root = _drive(tmp_path, monkeypatch, protocol=protocol, stop_at=2,
                              steps=[_edit_source_step("q1h1")] if protocol == "handles"
                              else [_str_replace_step("= 25", "= 30")])
    first_menu = _names(sends[0][1])
    reply = _tool_replies(sends[-1][0])[-1]
    if protocol == "handles":
        assert set(PROTOCOL_TOOLS) <= set(first_menu) and not HIDDEN_LEGACY_TOOLS & set(first_menu)
        body = json.loads(reply)
        assert body["tool"] == EDIT_SOURCE_TOOL and [r["code"] for r in body["refusals"]] == ["STALE_HANDLE"]
        assert (copy_root / RULES_PATH).read_text() == RULES
    else:
        assert "str_replace" in first_menu and not set(PROTOCOL_TOOLS) & set(first_menu)
        if bound == "off":
            assert reply.startswith("str_replace applied"), reply[:120]
        else:  # the legacy writer's own bound-target refusal, unchanged by the protocol switch
            assert "bound edit target required" in reply and '"tool":"str_replace"' in reply


# --- T2 + T3: explicit legacy_safe == unset on the wire; handles wire is append-only --------------

def test_t2_shipped_profile_explicit_legacy_safe_is_identical_to_unset(shipped_runtime_profile, monkeypatch, tmp_path):
    """Shipped context: every menu and the legacy writer's observation and bytes match unset."""
    from probes.react_repair import str_replace_edit
    for bound in ("off", "on"):
        monkeypatch.setenv(BOUND_ENV, bound)
        assert _menus("legacy_safe", monkeypatch) == _menus(None, monkeypatch)
    monkeypatch.delenv(BOUND_ENV, raising=False)
    results = []
    for label, protocol in (("unset", None), ("explicit", "legacy_safe")):
        _set_protocol(monkeypatch, protocol)
        root, path = _repo_file(tmp_path / label, RULES, RULES_PATH)
        cases = [str_replace_edit(path, old, new, root=root).replace(str(root), "<R>")
                 for old, new in (("= 25", "= 30"), ("= 99", "= 1"), ("return", "return ("))]
        results.append((cases, path.read_bytes()))
    assert results[0] == results[1]


def test_t2_t3_explicit_legacy_safe_multi_turn_wire_is_byte_identical_to_unset(monkeypatch, tmp_path):
    steps = [_str_replace_step("= 25", "= 30"), _str_replace_step("= 99", "= 1")]
    runs = {}
    for label, protocol in (("unset", None), ("explicit", "legacy_safe")):
        base = tmp_path / label
        sends, copy_root = _drive(base, monkeypatch, protocol=protocol, steps=steps, stop_at=3)
        runs[label] = (_normalized(sends, base), (copy_root / RULES_PATH).read_bytes())
    assert len(runs["unset"][0]) == 3
    assert runs["explicit"] == runs["unset"]
    for serialized in runs["unset"][0]:
        assert EDIT_SOURCE_TOOL not in serialized and REWRITE_SOURCE_TOOL not in serialized


def test_t3_legacy_menu_equals_the_base_composition(monkeypatch):
    """legacy_safe adds nothing to the pre-protocol menu composition (surface bypassed)."""
    import kernel_edit_source_tool_surface as surface
    unset, explicit = _menus(None, monkeypatch), _menus("legacy_safe", monkeypatch)
    monkeypatch.setattr(surface, "tool_surface", lambda tools: tools)
    assert unset == explicit == _menus("handles", monkeypatch)


def test_t3_handles_multi_turn_wire_keeps_menu_and_prefix_stable(monkeypatch, tmp_path):
    sends, _ = _drive(tmp_path, monkeypatch, protocol="handles", stop_at=3,
                      steps=[_edit_source_step("q1h1"), _edit_source_step("q1h1")])
    assert len(sends) == 3
    tools = {json.dumps(tools, sort_keys=True) for _, tools in sends}
    assert len(tools) == 1, "the tool menu must be byte-identical on every request"
    (menu,) = tools
    assert HANDLE_PATTERN.search(menu) is None and CURRENT_SOURCE_MARK not in menu

    def history(messages):
        end = len(messages)
        while end and messages[end - 1].get("role") == "user":
            end -= 1
        return messages[:end]

    for earlier, later in zip(sends, sends[1:]):
        assert history(later[0])[:len(history(earlier[0]))] == history(earlier[0]), "history must be append-only"
    for messages, _ in sends:
        assert messages[0] == sends[0][0][0], "system message must not change between requests"
        last_user = max(i for i, m in enumerate(messages) if m.get("role") == "user")
        for index, message in enumerate(messages):
            if index != last_user:
                assert CURRENT_SOURCE_MARK not in str(message.get("content")), "map outside the final user message"
    replies = _tool_replies(sends[-1][0])
    assert len(replies) == 2
    for reply in replies:
        body = json.loads(reply)
        assert [r["code"] for r in body["refusals"]] == ["STALE_HANDLE"] and str(tmp_path) not in reply
    # The earlier reply is history: the later request carries it byte-identically.
    assert _tool_replies(sends[1][0]) == replies[:1]


# --- T5: boundary inputs ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("", "legacy_safe"), ("   ", "legacy_safe"), ("legacy_safe", "legacy_safe"),
    (" Legacy_Safe\t", "legacy_safe"), ("HANDLES\r\n", "handles"), ("handles", "handles"),
])
def test_t5_engine_switch_accepts_boundary_spellings(monkeypatch, raw, expected):
    monkeypatch.setenv(MODE_ENV, raw)
    assert mode() == expected and handles_enabled() is (expected == "handles")


@pytest.mark.parametrize("raw", ["﻿handles", "handles,legacy_safe", "légacy_safe", "legacy-safe",
                                 "on", "1", "h" * 4097])
def test_t5_engine_switch_refuses_unknown_values(monkeypatch, raw):
    monkeypatch.setenv(MODE_ENV, raw)
    with pytest.raises(ValueError, match=MODE_ENV):
        mode()


@pytest.mark.parametrize("raw", ["﻿handles", "handles,legacy_safe", "légacy_safe", "on"])
def test_t5_runner_refuses_unknown_values_at_launch(monkeypatch, raw):
    from stateframe import runner
    monkeypatch.setenv(MODE_ENV, raw)
    with pytest.raises(runner.RunError, match=MODE_ENV):
        runner.resolve_opt_in_switches()


@pytest.mark.parametrize("raw,expected", [("", None), ("  ", None), ("HANDLES", "handles"),
                                          (" legacy_safe ", "legacy_safe")])
def test_t5_runner_boundary_spellings(monkeypatch, raw, expected):
    from stateframe import runner
    monkeypatch.setenv(MODE_ENV, raw)
    assert runner.resolve_opt_in_switches().get(MODE_ENV) == expected


def test_t5_edit_source_empty_batch_and_exact_limit(tmp_path):
    text = "A = 1\nB = 2\nC = 3\nD = 4\nE = 5\n"
    root, path = _repo_file(tmp_path, text)
    state = _state("pkg/mod.py", text)
    handle = state.handle_map.records[0].handle
    empty = _edit(root, state, [])
    assert empty.status == "REJECTED" and empty.refusals == () and "at least one edit" in empty.detail
    edits = [{"handle": handle, "old_text": f"{name} = {n}", "new_text": f"{name} = {n * 10}"}
             for n, name in enumerate("ABCDE", start=1)]
    over = _edit(root, state, edits)
    assert over.status == "REJECTED" and "TOO_MANY_EDITS" in over.codes()
    assert path.read_text() == text
    exact = _edit(root, state, edits[:4])
    assert exact.status == "APPLIED"
    assert path.read_text() == "A = 10\nB = 20\nC = 30\nD = 40\nE = 5\n"


def test_t5_edit_source_crlf_old_text(tmp_path):
    text = "def a():\r\n    x = 1\r\n    return x\r\n"
    root, path = _repo_file(tmp_path, text)
    state = _state("pkg/mod.py", text)
    handle = state.handle_map.records[0].handle
    lf = _edit(root, state, [{"handle": handle, "old_text": "x = 1\n    return", "new_text": "x = 2\n    return"}])
    assert lf.status == "REJECTED" and lf.codes() == ("OLD_NOT_FOUND",)
    crlf = _edit(root, state, [{"handle": handle, "old_text": "x = 1\r\n    return",
                                "new_text": "x = 2\r\n    return"}])
    assert crlf.status == "APPLIED"
    assert path.read_bytes() == text.replace("x = 1", "x = 2").encode("utf-8")


def test_t5_edit_source_bom_and_non_ascii_bytes(tmp_path):
    text = "﻿LABEL = 'café ☕'\nOTHER = 'naïve'\n"
    root, path = _repo_file(tmp_path, text)
    state = _state("pkg/mod.py", text)
    handle = state.handle_map.records[0].handle
    missing = _edit(root, state, [{"handle": handle, "old_text": "cafe", "new_text": "x"}])
    assert missing.codes() == ("OLD_NOT_FOUND",)
    out = _edit(root, state, [{"handle": handle, "old_text": "café ☕", "new_text": "thé 🍵"}])
    assert out.status == "APPLIED"
    assert path.read_bytes() == text.replace("café ☕", "thé 🍵").encode("utf-8")
    assert json.loads(out.observation())["status"] == "APPLIED"


@pytest.mark.parametrize("args", [
    {"edits": [{"handle": "", "old_text": "A", "new_text": "B"}]},
    {"edits": [{"handle": None, "old_text": "A", "new_text": "B"}]},
])
def test_t5_edit_source_empty_or_non_string_handle_is_stale(tmp_path, args):
    root, path = _repo_file(tmp_path, "A = 1\n")
    out = execute_edit_source(root, args, state=_state("pkg/mod.py", "A = 1\n"), read_bytes=Path.read_bytes)
    assert out.status == "REJECTED" and out.codes() == ("STALE_HANDLE",)
    assert path.read_text() == "A = 1\n"


@pytest.mark.parametrize("args", [{}, {"handle": "q5h1"}, {"handle": "q5h1", "new_text": None},
                                  {"handle": 7, "new_text": ""}])
def test_t5_rewrite_source_malformed_args_are_rejected_without_writing(tmp_path, args):
    root, path = _repo_file(tmp_path, "A = 1\n")
    state = _state("pkg/mod.py", "A = 1\n", kind=SpanKind.REWRITE)
    out = execute_rewrite_source(root, args, state=state, read_bytes=Path.read_bytes)
    assert out.status == "REJECTED" and "must be strings" in out.detail
    assert path.read_text() == "A = 1\n"


# --- T7: shell -> run spec -> engine child -> subframe child -----------------------------------------

def _clear_opt_ins(monkeypatch) -> None:
    from stateframe import runner
    for name in runner.OPT_IN_SWITCHES:
        monkeypatch.delenv(name, raising=False)


def _prepare(tmp_path: Path):
    from stateframe import runner
    from stateframe.endpoint import Endpoint
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "test_sample.py").write_text("def test_sample(): assert True\n", encoding="utf-8")
    endpoint = Endpoint.from_args("http://127.0.0.1:8080/v1", "local-model", None)
    options = runner.RunOptions(repo, "check", endpoint, run_dir=tmp_path / "run")
    return runner.prepare(options, printer=lambda *_args: None)[2]


def test_t7_switch_is_registered_for_the_shell_and_read_by_the_engine():
    from stateframe import runner
    assert runner.OPT_IN_SWITCHES[MODE_ENV] == ("legacy_safe", "handles")
    assert MODE_ENV in runner.engine_switches()
    assert MODE_ENV in runner.SHELL_SETTINGS and MODE_ENV not in runner.HOST_ONLY_SWITCHES
    assert MODE_ENV not in runner.ignored_shell_settings({MODE_ENV: "handles"})


def test_t7_shell_value_reaches_the_run_spec(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    _clear_opt_ins(monkeypatch)
    monkeypatch.setenv(MODE_ENV, "Handles")
    spec = _prepare(tmp_path)
    assert spec["environment"][MODE_ENV] == "handles"
    assert spec["environment_sources"][MODE_ENV] == "ambient:" + MODE_ENV


def test_t7_unset_keeps_the_shipped_default(tmp_path, monkeypatch):
    from stateframe import profiles
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    _clear_opt_ins(monkeypatch)
    assert MODE_ENV not in profiles.load("r2hc3")
    spec = _prepare(tmp_path)
    assert MODE_ENV not in spec["environment"]


def test_t7_invalid_shell_value_refuses_the_launch(tmp_path, monkeypatch):
    from stateframe import runner
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    _clear_opt_ins(monkeypatch)
    monkeypatch.setenv(MODE_ENV, "handle")
    with pytest.raises(runner.RunError, match=MODE_ENV):
        _prepare(tmp_path)
    assert not (tmp_path / "run" / "spec.json").exists()


@pytest.mark.parametrize("recorded,expected", [("handles", "handles"), (None, "legacy_safe")])
def test_t7_engine_child_installs_the_recorded_value_and_drops_inherited(monkeypatch, recorded, expected):
    from stateframe import _child
    monkeypatch.setattr(_child.os, "environ", dict(os.environ))
    # An inherited shell value never leaks past the spec: only the recorded one counts.
    monkeypatch.setenv(MODE_ENV, "legacy_safe" if recorded else "handles")
    _child.install_environment({"environment": {MODE_ENV: recorded} if recorded else {}})
    assert mode() == expected


def test_t7_subframe_child_inherits_the_parent_value(tmp_path):
    from stateframe.parallel_worker import prepare_worker
    from test_parallel_worker_prep import _unit_fixture

    snapshot, run_dir, spec_path, spec, parent_spec, _parent_run = _unit_fixture(tmp_path)
    parent_spec["environment"][MODE_ENV] = "handles"
    prepare_worker(spec, spec_path, run_dir, snapshot, parent_spec=parent_spec)
    prepared = json.loads((run_dir / "spec.json").read_text())
    assert prepared["environment"][MODE_ENV] == "handles"


# --- T12: named regression for the empty-value fix ---------------------------------------------------

def test_t12_empty_engine_value_means_default_like_the_runner(monkeypatch):
    """Regression: an empty value raised ValueError mid-run while the runner treats it as unset."""
    from stateframe import runner
    monkeypatch.setenv(MODE_ENV, "")
    assert mode() == "legacy_safe"
    assert MODE_ENV not in runner.resolve_opt_in_switches()


def test_t12_protocol_modules_ship_in_the_release_manifest():
    manifest = json.loads((Path(__file__).resolve().parents[1] / "src/stateframe/engine/release-manifest.json")
                          .read_text(encoding="utf-8"))
    files = manifest["files"]
    listed = set(files) if isinstance(files, dict) else {entry.get("path") for entry in files}
    for module in ("kernel_edit_protocol_switch.py", "kernel_edit_source_tool.py",
                   "kernel_edit_source_tool_surface.py", "kernel_edit_tool_names.py"):
        assert module in listed, module
