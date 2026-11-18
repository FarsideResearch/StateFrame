"""Protocol edits (edit_source / rewrite_source) must refuse hook-protected files.

The legacy edit tools gate hook definitions and bundled inputs in ``_hook_gate``. The protocol
tools carry handles, not paths, so their refusal lives in ``_ep_not_editable`` inside
``react_repair.work``. Both closures are nested in ``work``; they are compiled out of its real
source (same technique as test_hook_wiring_core) so the production code under test is what runs.
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path
from types import SimpleNamespace

from probes import react_repair

from kernel_edit_handles import DeliveredSpan, SpanKind, mint_handles
from kernel_edit_source_tool import EditProtocolRunState, execute_edit_source, execute_rewrite_source

REFUSAL = "REFUSED HOOK_FILE: hook definition or bundled input is read-only"
HOOK_REL = "hooks/guard.json"
HOOK_TEXT = '{"schema": "stateframe-hooks/1", "hooks": []}\n'
PRODUCT_REL = "pkg/mod.py"
PRODUCT_TEXT = "def a():\n    return 1\n"


class _HookWiringStub:
    """Mirrors HookWiring.protected(): True for paths the hook bundle made read-only."""

    def __init__(self, protected: set[Path]):
        self._protected = protected
        self.calls: list[Path] = []

    def protected(self, path):
        self.calls.append(Path(path))
        return Path(path) in self._protected


def _work_def(name: str) -> ast.FunctionDef:
    tree = ast.parse(textwrap.dedent(inspect.getsource(react_repair.work)))
    defs = [node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(defs) == 1, f"expected one nested def named {name}"
    return defs[0]


def _compile_def(name: str, scope: dict) -> object:
    module = ast.fix_missing_locations(ast.Module(body=[_work_def(name)], type_ignores=[]))
    namespace = dict(scope)
    exec(compile(module, react_repair.__file__, "exec"), namespace)
    return namespace[name]


def _not_editable(root: Path, wiring: object | None):
    return _compile_def("_ep_not_editable", {
        "root": root,
        "baseline_tests": set(),
        "multifile": False,
        "_hook_wiring_for_run": wiring,
        "protected_test_path": react_repair.protected_test_path,
        "_scope_widen": lambda relative: None,
        "_outside_edit_scope": lambda relative: False,
    })


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "hooks").mkdir(parents=True)
    (root / "pkg").mkdir(parents=True)
    (root / HOOK_REL).write_text(HOOK_TEXT)
    (root / PRODUCT_REL).write_text(PRODUCT_TEXT)
    return root


def _state(kind: SpanKind, rel: str, text: str, ordinal: int = 3) -> EditProtocolRunState:
    span = DeliveredSpan.from_source(rel, text, kind=kind)
    state = EditProtocolRunState()
    state.set_request(mint_handles(ordinal, [span]))
    return state


def _read(p: Path) -> bytes:
    return p.read_bytes()


def test_edit_source_refuses_hook_file_and_keeps_bytes(tmp_path):
    root = _repo(tmp_path)
    hook = root / HOOK_REL
    before = hook.read_bytes()
    wiring = _HookWiringStub({root / HOOK_REL})
    state = _state(SpanKind.SPAN, HOOK_REL, HOOK_TEXT)
    handle = state.handle_map.records[0].handle
    out = execute_edit_source(
        root,
        {"edits": [{"handle": handle, "old_text": '"hooks": []', "new_text": '"hooks": [{}]'}]},
        state=state, read_bytes=_read, not_editable=_not_editable(root, wiring))
    assert out.status == "REJECTED", out.observation()
    assert REFUSAL in out.observation()
    assert out.changed_paths == ()
    assert hook.read_bytes() == before


def test_rewrite_source_refuses_hook_file_and_keeps_bytes(tmp_path):
    root = _repo(tmp_path)
    hook = root / HOOK_REL
    before = hook.read_bytes()
    wiring = _HookWiringStub({root / HOOK_REL})
    state = _state(SpanKind.REWRITE, HOOK_REL, HOOK_TEXT)
    handle = state.handle_map.records[0].handle
    out = execute_rewrite_source(
        root, {"handle": handle, "new_text": '{"schema": "x", "hooks": [{}]}\n'},
        state=state, read_bytes=_read, not_editable=_not_editable(root, wiring))
    assert out.status == "REJECTED", out.observation()
    assert REFUSAL in out.observation()
    assert out.changed_paths == ()
    assert hook.read_bytes() == before


def test_legacy_str_replace_on_hook_file_still_refused(tmp_path):
    root = _repo(tmp_path)
    wiring = _HookWiringStub({root / HOOK_REL})
    msgs: list[dict] = []
    observed: list[tuple] = []
    gate_scope = {
        "_hook_wiring_for_run": wiring,
        "root": root,
        "bolt": SimpleNamespace(receipt=lambda *a, **k: None),
        "measurement": SimpleNamespace(observe=lambda name, reply: observed.append((name, reply))),
        "msgs": msgs,
        "toolcall_mode": True,
        "_tool_reply_for_call": lambda messages, call_id, observation: {
            "role": "tool", "tool_call_id": call_id, "content": observation},
    }
    gate = _compile_def("_hook_gate", gate_scope)
    refused = gate("str_replace", {"path": HOOK_REL, "old": "a", "new": "b"}, "call-1")
    assert refused is True
    assert msgs[-1]["content"] == REFUSAL
    assert observed == [("str_replace", REFUSAL)]


def test_edit_source_on_product_file_still_applies(tmp_path):
    root = _repo(tmp_path)
    wiring = _HookWiringStub({root / HOOK_REL})
    state = _state(SpanKind.SPAN, PRODUCT_REL, PRODUCT_TEXT)
    handle = state.handle_map.records[0].handle
    out = execute_edit_source(
        root,
        {"edits": [{"handle": handle, "old_text": "return 1", "new_text": "return 10"}]},
        state=state, read_bytes=_read, not_editable=_not_editable(root, wiring))
    assert out.status == "APPLIED", out.observation()
    assert (root / PRODUCT_REL).read_text() == PRODUCT_TEXT.replace("return 1", "return 10")


def test_edit_source_on_product_file_without_hook_wiring_still_applies(tmp_path):
    root = _repo(tmp_path)
    state = _state(SpanKind.SPAN, PRODUCT_REL, PRODUCT_TEXT)
    handle = state.handle_map.records[0].handle
    out = execute_edit_source(
        root,
        {"edits": [{"handle": handle, "old_text": "return 1", "new_text": "return 10"}]},
        state=state, read_bytes=_read, not_editable=_not_editable(root, None))
    assert out.status == "APPLIED", out.observation()
