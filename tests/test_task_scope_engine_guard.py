"""Task file scope caps every native writer, including declared and absent paths."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from probes import react_repair as rr
from kernel_edit_handles import DeliveredSpan, SpanKind, mint_handles
from kernel_edit_source_tool import EditProtocolRunState, execute_edit_source
from kernel_plain_contracts import EditAuthority
from kernel_plain_hooks import HookSettings, PlainHooks
from kernel_plain_host import PlainHost
from stateframe.task_row import build_row
from stateframe.workcopy import create

SOURCE = "VALUE = 1\n"
INITIAL = {"answer.json": '{"value": 1}\n', "README.md": "value = 1\n", "notes.py": SOURCE}


class _Stop(BaseException):
    """Capture the next real request before the worker can finish or grade."""


def _drive(tmp_path, monkeypatch, tool, arguments, *, scoped=True, absent=(), handles=False,
           protected=(), hooks=False):
    from kernel_measurement import ACTIVE_REQUEST
    from kernel_pilot import ACTIVE_PARENT, ACTIVE_RUNTIME
    from kernel_stateframe_runtime import StateFrameRuntime
    from kernel_verdict_identity import _ACTIVE_PROVIDER
    import kernel_tool_tape

    repo = tmp_path / "source"
    repo.mkdir()
    for relative, text in INITIAL.items():
        if relative not in absent:
            (repo / relative).write_text(text)
    row = build_row(create(repo, tmp_path / "copy"), task="Update the answer",
                    runner_kind="generic", test_command="true", task_id="scope-guard")
    # Deliberately broader declared authority: the task cap must win even when
    # a checkpoint or caller supplies unrelated files in the editable set.
    if scoped:
        row["task_file_scope"] = ["answer.json"]
    row["protected_test_paths"] = list(protected)
    row["scoped_edits"] = False
    row["entry"] = {"path": "answer.json", "fn": ""}
    row["new_file_roots"] = ["answer.json", "notes.py"]
    output = tmp_path / "run/worker/measurement-runs"
    output.mkdir(parents=True)
    private_temp = tmp_path / "worker-tmp"
    private_temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(private_temp))
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("KERNEL_INTERVENTIONS", "1")
    monkeypatch.setenv("STATEFRAME_EFFECT_BOUNDARY", "off")
    monkeypatch.setenv("STATEFRAME_IDEALPATH", "off")
    monkeypatch.setenv("STATEFRAME_EDIT_SCOPE_WIDEN", "on")
    monkeypatch.setenv("STATEFRAME_EDIT_PROTOCOL", "handles" if handles else "legacy_safe")
    monkeypatch.setenv("STATEFRAME_TOKENLAB_ATOMIC_MULTI_FILE_APPLY", "on")
    monkeypatch.setattr(sys, "argv", [sys.argv[0]])
    monkeypatch.setattr(rr, "POC", tmp_path / "engine")
    monkeypatch.setattr(rr, "REPOS", {"build": repo})
    monkeypatch.setattr(rr, "MAX_STEPS", 4)
    hook_wiring = None
    if hooks:
        import kernel_hook_wiring
        from kernel_hooks import HookSet, create_bundle

        hook_run = tmp_path / "hook-run"
        hook_run.mkdir()
        hook_set = HookSet((), (), "empty")
        create_bundle(hook_set, hook_run / "hook-bundle")
        monkeypatch.setattr(kernel_hook_wiring, "_ACTIVE", None)
        hook_wiring = kernel_hook_wiring.HookWiring(hook_run, hook_set, {
            "repo": str(repo), "run_identity_seed": "scope-guard",
            "tool_settings": {"shell_network": "none"}, "hooks": {"start_context": []},
        }, replay=lambda: False)
        monkeypatch.setattr(rr, "_hook_wiring", hook_wiring, raising=False)
    monkeypatch.setattr(kernel_tool_tape, "_active_tape", kernel_tool_tape.ToolTape(tmp_path))
    monkeypatch.setattr(kernel_tool_tape, "_replay_active", False)
    original_init = StateFrameRuntime.__init__

    def scoped_init(self, owner, **kwargs):
        kwargs["output"] = output
        original_init(self, owner, **kwargs)

    monkeypatch.setattr(StateFrameRuntime, "__init__", scoped_init)
    requests = []
    snapshots = []

    def ask(messages, *args, **kwargs):
        requests.append(copy.deepcopy(messages))
        roots = list(private_temp.glob("react_*/repo"))
        assert len(roots) == 1
        root = roots[0]
        snapshots.append({p: (root / p).read_bytes() if (root / p).exists() else None
                          for p in (*INITIAL, "moved.py", "nested/notes.py")})
        assert not (root / "nested").exists()
        if len(requests) == 2:
            raise _Stop
        actual_args = copy.deepcopy(arguments)
        if handles:
            # Mint a real request handle for the forbidden file, as a read could.
            import kernel_edit_source_tool
            original = kernel_edit_source_tool.EXECUTORS[tool]

            def with_handle(root, args, *, state, **kw):
                kind = SpanKind.REWRITE if tool == "rewrite_source" else SpanKind.SPAN
                state.set_request(mint_handles(1, [DeliveredSpan.from_source(
                    "notes.py", SOURCE, kind=kind)]))
                handle = state.handle_map.records[0].handle
                if tool == "edit_source":
                    args = {"edits": [{"handle": handle, "old_text": "1", "new_text": "2"}]}
                else:
                    args = {"handle": handle, "new_text": "VALUE = 2\n"}
                return original(root, args, state=state, **kw)

            monkeypatch.setitem(kernel_edit_source_tool.EXECUTORS, tool, with_handle)
        return "", [{"id": "scope-call", "type": "function", "function": {
            "name": tool, "arguments": json.dumps(actual_args)}}]

    monkeypatch.setattr(rr, "ask_tools", ask)
    contexts = (ACTIVE_PARENT, ACTIVE_RUNTIME, ACTIVE_REQUEST, _ACTIVE_PROVIDER)
    tokens = [c.set(None) for c in contexts]
    try:
        with pytest.raises(_Stop):
            rr.work(row)
    finally:
        for context, token in zip(contexts, tokens):
            context.reset(token)
        if hook_wiring is not None and hook_wiring.runner is not None:
            hook_wiring.runner.close()
    replies = [str(m["content"]) for m in requests[-1] if m.get("role") == "tool"]
    assert replies, requests[-1]
    return replies[-1], requests[-1], snapshots


MUTATIONS = [
    ("file_create", {"path": "notes.py", "content": "VALUE = 2\n"}, ("notes.py",)),
    ("file_create", {"path": "README.md", "content": "changed\n"}, ()),
    ("file_delete", {"path": "README.md"}, ()),
    ("file_move", {"source": "README.md", "destination": "answer.json"}, ("answer.json",)),
    ("file_move", {"source": "answer.json", "destination": "moved.py"}, ()),
    ("str_replace", {"path": "README.md", "old_str": "1", "new_str": "2"}, ()),
    ("edit", {"path": "notes.py", "function_source": "VALUE = 2\n"}, ()),
    ("replace_module", {"path": "notes.py", "expected_revision": "unused",
                        "source": "VALUE = 2\n"}, ()),
    ("str_replace", {"edits": [{"path": "answer.json", "old": "1", "new": "2"},
                               {"path": "notes.py", "old": "1", "new": "2"}]}, ()),
    ("insert_before", {"path": "notes.py", "new_str": "VALUE = 2\n"}, ()),
    ("file_create", {"path": "nested/notes.py", "content": "VALUE = 2\n"}, ()),
]


@pytest.mark.parametrize("tool,arguments,absent", MUTATIONS)
def test_native_mutation_refused_before_any_write(tmp_path, monkeypatch, tool, arguments, absent):
    reply, _, snapshots = _drive(tmp_path, monkeypatch, tool, arguments, absent=absent)
    assert "OUTSIDE_TASK_SCOPE" in reply, reply
    assert "NOT_EDITABLE" in reply, reply
    assert "answer.json" in reply
    assert "--edit-scope widen" not in reply
    assert snapshots[0] == snapshots[1]


@pytest.mark.parametrize("tool", ["edit_source", "rewrite_source"])
def test_handle_refusal_survives_repair_in_next_model_request(tmp_path, monkeypatch, tool):
    args = {"edits": [{"handle": "q1h1", "old_text": "1", "new_text": "2"}]} if tool == "edit_source" else {
        "handle": "q1h1", "new_text": "VALUE = 2\n"}
    reply, messages, snapshots = _drive(tmp_path, monkeypatch, tool, args, handles=True)
    assert "NOT_EDITABLE" in reply
    assert "OUTSIDE_TASK_SCOPE" in reply and "answer.json" in reply
    # Request preparation can append host budget notices after the repair.
    repairs = [m["content"] for m in messages if m.get("role") == "user"
               and str(m.get("content", "")).startswith("[EDIT REFUSED] request")]
    assert len(repairs) == 1
    assert "OUTSIDE_TASK_SCOPE" in repairs[0]
    assert "answer.json" in repairs[0]
    assert snapshots[0] == snapshots[1]


@pytest.mark.parametrize("scoped", [False, True])
@pytest.mark.parametrize("create_new", [False, True])
def test_allowed_answer_create_and_edit(tmp_path, monkeypatch, scoped, create_new):
    args = {"path": "answer.json", "content": '{"value": 2}\n'} if create_new else {
        "path": "answer.json", "old_str": "1", "new_str": "2"}
    reply, _, snapshots = _drive(tmp_path, monkeypatch, "file_create" if create_new else "str_replace",
                                 args, scoped=scoped, absent=("answer.json",) if create_new else ())
    assert "OUTSIDE_TASK_SCOPE" not in reply
    assert snapshots[-1]["answer.json"] == b'{"value": 2}\n', reply


@pytest.mark.parametrize("tool,arguments,absent", [MUTATIONS[i] for i in (0, 2, 4, 5, 6, 8)])
def test_no_task_scope_preserves_mutations(tmp_path, monkeypatch, tool, arguments, absent):
    reply, _, snapshots = _drive(tmp_path, monkeypatch, tool, arguments, scoped=False, absent=absent)
    assert "OUTSIDE_TASK_SCOPE" not in reply
    assert snapshots[0] != snapshots[1], reply


def test_move_destination_refusal_survives_hook_preflight(tmp_path, monkeypatch):
    reply, _, snapshots = _drive(tmp_path, monkeypatch, "file_move", {
        "source": "answer.json", "destination": "moved.py"}, hooks=True)
    assert reply.startswith("REFUSED NEW_FILE_FORBIDDEN:")
    assert '"code":"NOT_EDITABLE"' in reply
    assert "OUTSIDE_TASK_SCOPE" in reply and "answer.json" in reply
    assert snapshots[0] == snapshots[1]


@pytest.mark.parametrize("tool,args,protected,code", [
    ("file_delete", {"path": "../README.md"}, (), "OUTSIDE_WORKSPACE"),
    ("file_delete", {"path": "README.md"}, ("README.md",), "PROTECTED_TEST"),
    ("file_move", {"source": "answer.json", "destination": "README.md"},
     ("README.md",), "PROTECTED_TEST"),
])
def test_file_operation_keeps_path_and_protection_reasons(tmp_path, monkeypatch, tool, args,
                                                         protected, code):
    reply, _, snapshots = _drive(tmp_path, monkeypatch, tool, args, protected=protected)
    assert reply.startswith(f"REFUSED {code}:")
    assert "OUTSIDE_TASK_SCOPE" not in reply
    assert snapshots[0] == snapshots[1]


def test_create_keeps_protected_file_reason(tmp_path, monkeypatch):
    reply, _, snapshots = _drive(tmp_path, monkeypatch, "file_create", {
        "path": "README.md", "content": "changed\n"}, protected=("README.md",))
    assert "protected pre-existing test path" in reply
    assert "OUTSIDE_TASK_SCOPE" not in reply
    assert snapshots[0] == snapshots[1]


@pytest.mark.parametrize("scoped", [False, True])
def test_protected_handle_keeps_its_reason(tmp_path, monkeypatch, scoped):
    reply, _, snapshots = _drive(tmp_path, monkeypatch, "edit_source", {"edits": []},
                                 scoped=scoped, handles=True, protected=("notes.py",))
    assert "NOT_EDITABLE" in reply
    assert "OUTSIDE_TASK_SCOPE" not in reply
    assert json.loads(reply)["detail"] == "NOT_EDITABLE: protected public test paths are immutable"
    assert snapshots[0] == snapshots[1]


def test_task_scope_disables_widening_without_admission(tmp_path, monkeypatch):
    from test_edit_scope_widen import _workspace
    root, row, fileset = _workspace(tmp_path)
    row["task_file_scope"] = ["answer.json"]
    monkeypatch.setenv("STATEFRAME_EDIT_SCOPE_WIDEN", "on")
    before = copy.deepcopy(row)
    before_files = dict(fileset.parent)
    admissions = []
    monkeypatch.setattr("kernel_scope_widen_guard.admit", lambda *args: admissions.append(args))
    reason = rr._edit_scope_widen_refusal(root, "app/rates.py", row, fileset, set())
    assert reason and "OUTSIDE_TASK_SCOPE" in reason
    assert rr._widen_edit_scope(root, "app/rates.py", row, fileset, set()) is None
    assert not admissions
    assert row == before and fileset.parent == before_files


@pytest.mark.parametrize("scoped", [False, True])
@pytest.mark.parametrize("tool,path", [("write_file", "README.md"), ("str_replace", "README.md"),
                                       ("write_file", "notes.py"), ("write_file", "answer.json")])
def test_plain_host_task_scope_reply_and_receipt(tmp_path, scoped, tool, path):
    (tmp_path / "README.md").write_text("value = 1\n")
    receipts = []
    journal = SimpleNamespace(receipt=lambda kind, receipt: receipts.append(receipt))
    authority = EditAuthority(tmp_path, ["answer.json", "README.md"])
    # PlainHost owns the row-level cap; EditAuthority's contract remains unchanged.
    kwargs = {"task_scope": ["answer.json"]} if scoped else {}
    host = PlainHost(authority, check_runner=None, journal=journal,
                     hooks=PlainHooks(HookSettings(), authority.declared, journal, out_cap=12000), **kwargs)
    args = {"path": path, "content": "value = 2\n", "old": "1", "new": "2"}
    reply = host.execute(tool, args)
    if scoped and path != "answer.json":
        assert "OUTSIDE_TASK_SCOPE" in reply and "answer.json" in reply
        assert "OUTSIDE_TASK_SCOPE" in receipts[-1].reason
        assert not receipts[-1].applied
        assert (tmp_path / "README.md").read_text() == "value = 1\n"
        assert not (tmp_path / "notes.py").exists()
    else:
        assert reply == "ok"


def test_edit_source_checks_every_resolved_batch_path(tmp_path):
    allowed = '{"value": 1}\n'
    (tmp_path / "answer.json").write_text(allowed)
    (tmp_path / "notes.py").write_text(SOURCE)
    state = EditProtocolRunState()
    state.set_request(mint_handles(1, [DeliveredSpan.from_source("answer.json", allowed),
                                      DeliveredSpan.from_source("notes.py", SOURCE)]))
    checked = []

    def refusal(relative):
        checked.append(relative)
        return "REFUSED OUTSIDE_TASK_SCOPE: Edit only answer.json" if relative == "notes.py" else None

    edits = [{"handle": rec.handle, "old_text": "1", "new_text": "2"} for rec in state.handle_map.records]
    out = execute_edit_source(tmp_path, {"edits": edits}, state=state, read_bytes=Path.read_bytes,
                              not_editable=refusal)
    assert checked == ["answer.json", "notes.py"]
    assert "NOT_EDITABLE" in out.codes() and "OUTSIDE_TASK_SCOPE" in out.detail
    assert (tmp_path / "answer.json").read_text() == allowed
    assert (tmp_path / "notes.py").read_text() == SOURCE
    replay = execute_edit_source(tmp_path, {"edits": edits}, state=state, read_bytes=Path.read_bytes,
                                 not_editable=refusal)
    assert replay.status == "REPLAYED" and replay.detail == out.detail
    assert replay.refusals == out.refusals


@pytest.mark.parametrize("path", ["notes.py", "./notes.py", "folder/../notes.py", "README.md"])
def test_shared_task_guard_normalizes_before_membership(path):
    from kernel_edit_source_batch import task_scope_refusal
    reason = task_scope_refusal(path, ["answer.json"])
    assert reason and "OUTSIDE_TASK_SCOPE" in reason and "answer.json" in reason


def test_shared_task_guard_allowed_and_empty_controls():
    from kernel_edit_source_batch import task_scope_refusal
    assert task_scope_refusal("./answer.json", ["answer.json"]) is None
    for scope in (None, [], ()):
        assert task_scope_refusal("notes.py", scope) is None
