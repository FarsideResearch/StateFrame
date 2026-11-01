from __future__ import annotations

import ast
import base64
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from kernel_done_is_submit_authority import capture_and_seal
from kernel_completion_owner import terminal_result
from kernel_effects import (
    EffectError,
    Reconciliation,
    diff,
    effect_policy_for_run,
    scan,
)
from kernel_file_ops import FileOperationRefused, admit_file_delete, admit_file_move
from kernel_hook_exec import (
    EventResult,
    HookRunResult,
    HookRunner,
    WriterEffect,
    WriterResult,
)
from kernel_hooks import (HookDefinition, HookOutcome, HookSet, create_bundle,
                          load_hook_files, parse_outcome)
from kernel_hook_wiring import DispatchOutcome, HookBlockedError, HookWiring
from kernel_measurement import _serialize_final_request_body
from kernel_request_budget import _section_kind
from kernel_request_diet import STAGE_SECTION_ALLOWLIST, _DROP_ORDER, _stage_allowlist, section_name
from kernel_stateframe_framefit5 import _is_framefit5_host_context_panel
from kernel_stop_reason_taxonomy import _classify_reason, classify_run_end
from kernel_terminal_disposition import mint, project
from kernel_terminal_lattice import join_terminal


def _real_job_dispatch_code():
    """Compile the job route and gate from the production work loop."""
    from probes import react_repair

    source = textwrap.dedent(inspect.getsource(react_repair.work))
    tree = ast.parse(source)
    gates = [node for node in ast.walk(tree)
             if isinstance(node, ast.FunctionDef) and node.name == "_hook_gate"]
    assert len(gates) == 1
    routes = []
    for node in ast.walk(tree):
        for _, value in ast.iter_fields(node):
            if not isinstance(value, list):
                continue
            for index, statement in enumerate(value):
                if (isinstance(statement, ast.Assign)
                        and any(isinstance(target, ast.Name)
                                and target.id == "_job_dispatch"
                                for target in statement.targets)):
                    routes.append(value[index:index + 5])
    assert len(routes) == 1
    assert len(routes[0]) == 5
    loop = ast.For(
        target=ast.Name(id="_", ctx=ast.Store()),
        iter=ast.Tuple(elts=[ast.Constant(value=1)], ctx=ast.Load()),
        body=routes[0], orelse=[],
    )
    return compile(ast.fix_missing_locations(ast.Module(
        body=[gates[0], loop], type_ignores=[],
    )), react_repair.__file__, "exec"), react_repair._tool_reply_for_call


def test_collapsed_job_launch_is_denied_before_execution(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("VALUE = 1\n")
    run = tmp_path / "run"
    run.mkdir()
    config = tmp_path / "hooks.json"
    config.write_text(json.dumps({"schema": "stateframe-hooks/1", "hooks": [
        {"name": "job_guard", "event": "pre_tool", "tools": ["job"],
         "argv": [sys.executable, "-c", "raise SystemExit(2)"]},
        {"name": "job_audit", "event": "post_tool", "tools": ["job"],
         "argv": [sys.executable, "-c", "pass"]},
    ]}))
    hooks = load_hook_files([config], shell_network="none", repo=repo)
    create_bundle(hooks, run / "hook-bundle")
    wiring = HookWiring(run, hooks, {
        "repo": str(repo), "run_identity_seed": "seed",
        "tool_settings": {"shell_network": "none"},
        "hooks": {"start_context": []},
    }, replay=lambda: False)
    launched = []
    journal_events = []
    observations = []

    def launch_job(name, args):
        launched.append((name, args))
        (repo / "launched").write_text("started\n")
        return "launched"

    runtime = SimpleNamespace(
        journal=SimpleNamespace(run_id="run-id", emit=lambda kind, **fields:
                                journal_events.append((kind, fields))),
        job_watch=object(),
        job_operation=launch_job,
        observe=lambda name, reply: observations.append((name, reply)),
    )
    bolt = SimpleNamespace(receipt=lambda *args, **kwargs: None)
    wiring.bind(repo, {"files": {"app.py": "VALUE = 1\n"}, "scoped_edits": False},
                set(), None, runtime)
    calls = []

    def execute_boundary(self, hook, argv, payload, capability):
        data = json.loads(payload)["data"]
        calls.append((hook.event, data))
        assert data["tool"] == "job"
        assert data["args"]["action"] == "launch"
        if hook.event == "post_tool":
            assert data["status"] == "denied"
            assert data["executed"] is False
        stderr = b"blocked" if hook.event == "pre_tool" else b""
        envelope = {
            "schema": "stateframe-hook-envelope/1",
            "exit_code": 2 if stderr else 0,
            "signal": None,
            "timed_out": False,
            "launch_error": None,
            "stdout_b64": "",
            "stderr_b64": base64.b64encode(stderr).decode("ascii"),
            "stdout_total": 0,
            "stderr_total": len(stderr),
            "cleanup_ok": True,
        }
        return parse_outcome(hook.event, envelope), envelope, 0.0

    monkeypatch.setattr(HookRunner, "_execute", execute_boundary)
    arguments = {"action": "launch", "cmd": "touch launched", "expected_s": 5,
                 "done_signal": "done"}
    call_id = "job-call"
    messages = [{"role": "assistant", "tool_calls": [{
        "id": call_id, "type": "function", "function": {
            "name": "job", "arguments": json.dumps(arguments),
        },
    }]}]
    code, tool_reply = _real_job_dispatch_code()
    scope = {
        "_hook_wiring_for_run": wiring,
        "_name": "job", "_args": arguments, "tool_call_id": call_id,
        "measurement": runtime, "bolt": bolt, "msgs": messages,
        "toolcall_mode": True, "_tool_reply_for_call": tool_reply,
    }
    try:
        exec(code, scope)
        refusal = messages[-1]["content"]
        assert refusal == "REFUSED HOOK_DENIED: job_guard: blocked"
        assert observations == [("job", refusal)]
        assert not launched
        assert not (repo / "launched").exists()
        outcome = wiring.dispatch_outcome("job", call_id, runtime)
        assert (outcome.status, outcome.executed) == ("denied", False)
        wiring.finalize_action("job", arguments, call_id, refusal,
                               runtime, bolt, set(), outcome=outcome)
        assert [event for event, _ in calls] == ["pre_tool", "post_tool"]
        records = [json.loads(line) for line in
                   (run / "hooks/records.jsonl").read_text().splitlines()]
        assert [(record["event"], record["hook"], record["decision"])
                for record in records] == [
                    ("pre_tool", "job_guard", "deny"),
                    ("post_tool", "job_audit", "continue"),
                ]
        assert all(record["error_code"] is None for record in records)
        assert not any(kind == "job_operation_completed" for kind, _ in journal_events)
    finally:
        wiring.close()
        (run / "hook-bundle/0").chmod(0o700)
        (run / "hook-bundle").chmod(0o700)


def _wiring(tmp_path: Path, *, start_context=(), protected=False):
    run = tmp_path / "run"
    run.mkdir()
    (run / "hook-bundle").mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("x = 1\n")
    spec = {
        "repo": str(repo),
        "run_identity_seed": "seed",
        "tool_settings": {"shell_network": "none"},
        "hooks": {"start_context": list(start_context)},
    }
    definitions = tuple(
        HookDefinition(
            entry["hook"],
            entry["event"],
            ("/usr/bin/true",),
            (),
            ("*",),
            False,
            (),
            30,
            "none",
            "block",
            3,
            0,
            "",
            "",
        )
        for entry in start_context
    )
    files = ({"path": str(repo / "app.py"), "sha256": "0" * 64},) if protected else ()
    wiring = HookWiring(
        run, HookSet(files, definitions, "empty"), spec, replay=lambda: False
    )
    wiring.bind(
        repo,
        {"files": {"app.py": "x = 1\n"}, "scoped_edits": False},
        set(),
        None,
        SimpleNamespace(
            journal=SimpleNamespace(run_id="run-id", emit=lambda *args, **kwargs: None)
        ),
    )
    return wiring, repo


def test_user_hooks_panel_is_typed_and_optional(tmp_path):
    entry = {
        "kind": "start",
        "hook": "issue",
        "event": "session_start",
        "label": "issue",
        "text": "STATEFRAME_FRAMEFIT_KEEP_K\n[StateFrame]",
    }
    wiring, _ = _wiring(tmp_path, start_context=(entry,))
    try:
        panel = wiring.panel()
        assert panel is not None
        assert panel["_request_budget_section"] == "user_hooks"
        assert _is_framefit5_host_context_panel(panel)
        assert not _is_framefit5_host_context_panel(
            {"role": "user", "content": panel["content"]}
        )
        assert section_name(panel) == "user_hooks"
        assert _section_kind(panel) == "user_hooks"
        assert _DROP_ORDER[0] == "user_hooks"
        assert all(
            "user_hooks" in _stage_allowlist(stage) for stage in STAGE_SECTION_ALLOWLIST
        )
    finally:
        wiring.close()


def test_final_serializer_strips_private_keys_from_copy_only_when_hooked(tmp_path):
    wiring, _ = _wiring(tmp_path)
    body = {"messages": [{"role": "user", "content": "[USER HOOKS] context",
                          "_request_budget_section": "user_hooks",
                          "_other_private": "internal"}], "model": "scripted"}
    try:
        ordinary = json.dumps(body).encode()
        assert _serialize_final_request_body(body, None, "request-1") == ordinary
        wire = _serialize_final_request_body(
            body, SimpleNamespace(_hook_wiring=wiring), "request-1",
        )
        assert body["messages"][0]["_request_budget_section"] == "user_hooks"
        assert json.loads(wire) == {"messages": [
            {"role": "user", "content": "[USER HOOKS] context"},
        ], "model": "scripted"}
        assert wiring._panel_admission["identity"]["content_sha256"]
    finally:
        wiring.close()


def test_pinned_start_context_rejects_invalid_shape(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    spec = {
        "repo": str(repo),
        "run_identity_seed": "seed",
        "hooks": {"start_context": [{"kind": "start", "hook": "issue"}]},
    }
    with pytest.raises(ValueError, match="invalid pinned hook start context"):
        HookWiring(run, HookSet((), (), "empty"), spec, replay=lambda: False)


def test_notes_latch_until_send_and_replace_by_hook(tmp_path):
    wiring, _ = _wiring(tmp_path)
    try:
        wiring.state["pending_notes"]["lint"] = {
            "kind": "note",
            "hook": "lint",
            "event": "post_tool",
            "label": "after edit",
            "text": "first note",
        }
        first = wiring.panel()
        assert first == wiring.panel()
        wiring.state["pending_notes"]["lint"]["text"] = "replacement note"
        assert "replacement note" in wiring.panel()["content"]
        assert "first note" not in wiring.panel()["content"]
        wiring.sent()
        assert wiring.panel() is None
    finally:
        wiring.close()


def test_note_delivery_records_actual_wire_and_consumes_once(tmp_path):
    wiring, _ = _wiring(tmp_path)
    try:
        wiring.state["pending_notes"]["lint"] = {
            "kind": "note",
            "hook": "lint",
            "event": "post_tool",
            "label": "after edit",
            "text": "formatted app.py",
        }
        wiring.state["pending_ids"]["lint"] = "event-lint"
        panel = wiring.panel()
        assert panel is not None
        assert wiring.panel() == panel
        body = {"messages": [panel]}
        wire = json.dumps({"messages": [{"role": "user", "content": panel["content"]}]}).encode()
        wiring.admit_panel(body, wire, "request-3")
        wiring.sent(wire, "request-3")
        delivery = json.loads((wiring.run_dir / "hooks/delivery.jsonl").read_text())
        assert delivery["event_ids"] == ["event-lint"]
        assert delivery["request_id"] == "request-3"
        assert delivery["admitted_bytes"] == len(panel["content"].encode())
        assert delivery["dropped"] is False
        assert wiring.state["consumed_ids"] == ["event-lint"]
        assert wiring.panel() is None
    finally:
        wiring.close()


def test_panel_delivery_ignores_user_prefix_collision_and_eviction(tmp_path):
    wiring, _ = _wiring(tmp_path)
    try:
        wiring.state["pending_notes"]["lint"] = {
            "kind": "note", "hook": "lint", "event": "post_tool",
            "label": "after edit", "text": "formatter note",
        }
        wiring.state["pending_ids"]["lint"] = "event-lint"
        panel = wiring.panel()
        assert panel is not None
        user = {"role": "user", "content": "[USER HOOKS] user-authored text"}
        wire = json.dumps({"messages": [user]}).encode()
        wiring.admit_panel({"messages": [user]}, wire, "request-4")
        wiring.sent(wire, "request-4")
        delivery = json.loads((wiring.run_dir / "hooks/delivery.jsonl").read_text())
        assert delivery["admitted_bytes"] == 0
        assert delivery["dropped"] is True
    finally:
        wiring.close()


def test_pre_action_is_idempotent_and_post_action_runs_once(tmp_path):
    wiring, repo = _wiring(tmp_path)
    result = EventResult("continue", None, None, "", (), ())
    try:
        with patch.object(HookRunner, "run_event", return_value=result) as boundary:
            assert wiring.pre_action("read", {"path": "app.py"}, "call-1") is None
            assert wiring.pre_action("read", {"path": "app.py"}, "call-1") is None
            assert boundary.call_count == 1
            (repo / "app.py").write_text("x = 2\n")
            wiring.finalize_action(
                "read",
                {"path": "app.py"},
                "call-1",
                "ok",
                SimpleNamespace(),
                SimpleNamespace(),
                set(),
                outcome=DispatchOutcome("ok", True, None, ()),
            )
            assert boundary.call_count == 2
            assert boundary.call_args.kwargs["match"]["changed_paths"] == ()
    finally:
        wiring.close()


def test_explicit_applied_paths_exclude_unrelated_workspace_changes(tmp_path):
    wiring, repo = _wiring(tmp_path)
    allowed = EventResult("continue", None, None, "", (), ())
    try:
        with patch.object(HookRunner, "run_event", return_value=allowed) as boundary:
            wiring.pre_action("edit", {"path": "app.py"}, "edit-1")
            wiring.note_applied("edit-1", ())
            (repo / "app.py").write_text("x = 2\n")
            wiring.finalize_action(
                "edit",
                {"path": "app.py"},
                "edit-1",
                "edit applied",
                SimpleNamespace(),
                SimpleNamespace(),
                set(),
                outcome=wiring.dispatch_outcome("edit", "edit-1", SimpleNamespace()),
            )
            assert boundary.call_args.kwargs["data"]["changed_paths"] == ()
    finally:
        wiring.close()


def test_native_receipt_scope_excludes_concurrent_mutation(tmp_path):
    wiring, repo = _wiring(tmp_path)
    (repo / "other.py").write_text("y = 1\n")
    allowed = EventResult("continue", None, None, "", (), ())
    try:
        with patch.object(HookRunner, "run_event", return_value=allowed) as boundary:
            wiring.pre_action("edit", {"path": "app.py"}, "edit-2")
            (repo / "app.py").write_text("x = 2\n")
            wiring.note_applied("edit-2", ("app.py",))
            (repo / "other.py").write_text("y = 2\n")
            outcome = wiring.dispatch_outcome("edit", "edit-2", SimpleNamespace())
            assert outcome.applied_paths == ("app.py",)
            wiring.finalize_action("edit", {"path": "app.py"}, "edit-2",
                                   "edit applied", SimpleNamespace(),
                                   SimpleNamespace(), set(), outcome=outcome)
            assert boundary.call_args.kwargs["match"]["changed_paths"] == ("app.py",)
            assert "other.py" not in boundary.call_args.kwargs["match"]["changed_paths"]
    finally:
        wiring.close()


def test_writer_skips_when_applied_effects_are_unavailable(tmp_path):
    wiring, repo = _wiring(tmp_path)
    writer = HookDefinition(
        "fmt", "post_tool", ("/usr/bin/false",), (), ("edit",), True,
        (), 30, "none", "block", 3, 0, "/usr/bin/false", "missing",
        writes="changed",
    )
    wiring.runner.hooks = HookSet((), (writer,), "writer")
    try:
        result = wiring.runner.run_event(
            "post_tool", boundary="post_tool:edit-3",
            data={"tool": "edit", "call_id": "edit-3", "args": {},
                  "status": "ok", "executed": True, "exit_code": None,
                  "output": "edited", "changed_paths": []},
            match={"tool": "edit", "changed_paths": (),
                   "applied_paths_available": False},
        )
        assert len(result.runs) == 1
        assert result.runs[0].skipped == "applied_effects_unavailable"
        assert result.writes == ()
        assert (repo / "app.py").read_text() == "x = 1\n"
    finally:
        wiring.close()


@pytest.mark.parametrize("result_class, expected", [
    ("RED", "failed"),
    ("HARNESS_ERROR", "failed"),
    ("APPLIED", "ok"),
    ("REFUSED", "refused"),
])
def test_post_status_uses_typed_receipt_not_reply_text(tmp_path, result_class, expected):
    wiring, _ = _wiring(tmp_path)
    allowed = EventResult("continue", None, None, "", (), ())
    try:
        with patch.object(HookRunner, "run_event", return_value=allowed):
            wiring.pre_action("run_tests", {}, "check-1")
        runtime = SimpleNamespace(step_ledger=SimpleNamespace(pending={
            "tool_call_id": "check-1",
            "receipts": [{"result_class": result_class}],
        }))
        outcome = wiring.dispatch_outcome("run_tests", "check-1", runtime)
        assert outcome.status == expected
        assert outcome.executed is (expected != "refused")
        with patch.object(HookRunner, "run_event", return_value=allowed) as boundary:
            wiring.finalize_action("run_tests", {}, "check-1",
                                   "REFUSED appears in successful tool data",
                                   runtime, SimpleNamespace(), set(), outcome=outcome)
            assert boundary.call_args.kwargs["data"]["status"] == expected
    finally:
        wiring.close()


def test_hook_journal_uses_nonreserved_identity_field(tmp_path):
    wiring, _ = _wiring(tmp_path)
    emitted = []
    wiring.runtime.journal.emit = lambda kind, **fields: emitted.append((kind, fields))
    outcome = HookOutcome("continue", "", "", None, None, None, False, 0, 0, 0, 0)
    try:
        wiring._record_runs(
            EventResult(
                "continue",
                None,
                None,
                "",
                (),
                (HookRunResult("guard", "pre_tool", "hook-id", outcome, False, None),),
            )
        )
        assert emitted[0][0] == "user_hook_run"
        assert emitted[0][1]["hook_event_id"] == "hook-id"
        assert "event_id" not in emitted[0][1]
    finally:
        wiring.close()


def test_hook_path_is_denied_by_effect_policy(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "guard.py").write_text("before\n")
    before = scan(root, exclude=lambda _: False)
    (root / "guard.py").write_text("after\n")
    effect = diff(before, scan(root, exclude=lambda _: False))[0]
    policy = effect_policy_for_run(
        root,
        {"files": {"guard.py": "before\n"}, "scoped_edits": False},
        set(),
        baseline_manifest=before,
        hook_paths=frozenset({"guard.py"}),
    )
    assert policy.authorize(effect)[0] is False


def test_hook_path_hardlink_is_denied_by_effect_policy(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    protected = root / "guard.py"
    protected.write_text("before\n")
    before = scan(root, exclude=lambda _: False)
    policy = effect_policy_for_run(
        root,
        {"files": {"alias.py": "before\n"}, "scoped_edits": False},
        set(),
        baseline_manifest=before,
        hook_paths=frozenset({"guard.py"}),
    )
    os.link(protected, root / "alias.py")
    (root / "alias.py").write_text("after\n")
    effect = next(
        effect
        for effect in diff(before, scan(root, exclude=lambda _: False))
        if effect.path == "alias.py"
    )
    assert policy.authorize(effect)[0] is False


def test_native_hook_path_predicate_denies_hardlink_alias(tmp_path):
    wiring, repo = _wiring(tmp_path, protected=True)
    alias = repo / "alias.py"
    os.link(repo / "app.py", alias)
    try:
        assert wiring.protected(repo / "app.py")
        assert wiring.protected(alias)
    finally:
        wiring.close()


def test_preexisting_protected_hardlink_fails_closed(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    protected = root / "guard.py"
    protected.write_text("before\n")
    os.link(protected, root / "alias.py")
    with pytest.raises(
        EffectError, match="protected hook file has hardlinks: guard.py"
    ):
        effect_policy_for_run(
            root,
            {"files": {"guard.py": "before\n"}, "scoped_edits": False},
            set(),
            hook_paths=frozenset({"guard.py"}),
        )


def test_native_file_admission_has_no_effect_and_refuses_invalid_path(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "source.py").write_text("VALUE = 1\n")

    def allow(_):
        return False

    def editable(path):
        return path == "source.py"

    assert (
        admit_file_delete(root, "source.py", protected=allow, editable=editable)[1]
        == "source.py"
    )
    assert (
        admit_file_move(
            root,
            "source.py",
            "dest.py",
            protected=allow,
            editable=editable,
            new_file_allowed=lambda _: True,
        )[1][1]
        == "dest.py"
    )
    assert (root / "source.py").read_text() == "VALUE = 1\n"
    assert not (root / "dest.py").exists()
    with pytest.raises(FileOperationRefused, match="PATH_NOT_FOUND"):
        admit_file_delete(root, "missing.py", protected=allow, editable=editable)


def test_post_action_marks_refusals_without_a_pre_action(tmp_path):
    wiring, _ = _wiring(tmp_path)
    result = EventResult("continue", None, None, "", (), ())
    try:
        with patch.object(HookRunner, "run_event", return_value=result) as boundary:
            wiring.finalize_action(
                "read",
                {"path": "app.py"},
                "call-refused",
                "[kernel-menu] unavailable",
                SimpleNamespace(),
                SimpleNamespace(),
                set(),
                outcome=DispatchOutcome("refused", False, None, ()),
            )
            assert boundary.call_args.kwargs["data"]["status"] == "refused"
            assert boundary.call_args.kwargs["data"]["executed"] is False
    finally:
        wiring.close()


def test_stop_payload_tracks_latest_check_facts(tmp_path):
    wiring, _ = _wiring(tmp_path)
    result = EventResult("continue", None, None, "", (), ())
    latest = {
        "receipt_id": "check-2",
        "status": "PASS",
        "exit_code": 0,
        "execution_complete": True,
        "stdout": "private check output",
    }
    try:
        with patch.object(HookRunner, "run_event", return_value=result) as boundary:
            wiring.stop(SimpleNamespace(_latest_check_receipt=latest), source="model")
            checks = boundary.call_args.kwargs["data"]["checks"]
            assert checks == {
                "receipt_id": "check-2",
                "status": "PASS",
                "exit_code": 0,
                "execution_complete": True,
            }
    finally:
        wiring.close()


def test_stop_gate_counts_vetoes_and_rechecks_each_submission(tmp_path):
    wiring, _ = _wiring(tmp_path)
    denied = EventResult("deny", "gate", "HOOK_DENIED", "run checks", (), ())
    allowed = EventResult("continue", None, None, "", (), ())
    try:
        with patch.object(
            HookRunner,
            "run_event",
            side_effect=[denied, denied, allowed, denied, denied],
        ) as boundary:
            assert wiring.stop(SimpleNamespace(), source="model")[1] is False
            assert wiring.stop(SimpleNamespace(), source="model")[1] is False
            assert (
                wiring.stop(SimpleNamespace(), source="model")[0].decision == "continue"
            )
            assert wiring.stop(SimpleNamespace(), source="model")[1] is False
            assert wiring.stop(SimpleNamespace(), source="model")[1] is True
            assert boundary.call_count == 5
            assert [call.kwargs["boundary"] for call in boundary.call_args_list] == [
                "stop:1",
                "stop:2",
                "stop:3",
                "stop:4",
                "stop:5",
            ]
    finally:
        wiring.close()


def test_host_submission_veto_prevents_candidate_storage(tmp_path):
    wiring, _ = _wiring(tmp_path)
    denied = EventResult("deny", "gate", "HOOK_DENIED", "run checks", (), ())
    runtime = SimpleNamespace(
        _hook_wiring=wiring,
        journal=SimpleNamespace(run_id="run-id", emit=lambda *args, **kwargs: None),
    )
    destination = tmp_path / "candidate"
    try:
        with patch.object(HookRunner, "run_event", return_value=denied):
            with pytest.raises(HookBlockedError, match="vetoed"):
                capture_and_seal(
                    runtime,
                    destination=destination,
                    task_id="task",
                    sources={"app.py": "x = 1\n"},
                    parent_sources={"app.py": "x = 1\n"},
                )
        assert not destination.exists()
        assert runtime.system_outcome == "hook_blocked"
    finally:
        wiring.close()


@pytest.mark.parametrize("reason", ["HOOK_BLOCKED", "HOOK_CONTAINMENT"])
def test_hook_terminal_reasons_keep_their_taxonomy(reason):
    classified, _ = _classify_reason({"result": reason}, None, ())
    assert classified == reason


@pytest.mark.parametrize("reason", ["HOOK_BLOCKED", "HOOK_CONTAINMENT"])
@pytest.mark.parametrize("prior", ["STEP_BUDGET_EXHAUSTED", "UNCERTIFIED_SUBMIT"])
def test_hook_terminal_survives_completion_projection_and_prior_budget(reason, prior):
    runtime = SimpleNamespace(
        system_outcome=reason.lower(),
        stateframe_terminal_lattice_mode="on",
        stateframe_completion_owner_mode="on",
    )
    join_terminal(runtime, {"result": prior}, writer="prior")

    terminal = terminal_result(runtime, {"result": reason, "steps": 4})
    projected = project(terminal, mint(runtime, terminal), runtime=runtime)
    classified = classify_run_end(
        projected,
        system_outcome=runtime.system_outcome,
        events=({"kind": "v2_terminal", "terminal": "INCOMPLETE"},),
        runtime=runtime,
    )

    assert projected["result"] == reason
    assert projected["accepted"] is False
    assert projected["terminal_disposition"]["stop_reason"] == reason
    assert projected["terminal_disposition"]["delivery"] == "NOT_STORED"
    assert classified["stop_reason"] == reason


@pytest.mark.parametrize("reason", ["HOOK_BLOCKED", "HOOK_CONTAINMENT"])
def test_hook_v2_terminal_event_classifies_without_result(reason):
    classified, evidence = _classify_reason(
        {}, None, ({"kind": "v2_terminal", "terminal": reason},),
    )
    assert classified == reason
    assert evidence == ["v2_terminal"]


def test_writer_receipt_invalidates_check_and_reaches_next_panel(tmp_path):
    wiring, repo = _wiring(tmp_path)
    allowed = EventResult("continue", None, None, "", (), ())
    receipts = []
    bolt = SimpleNamespace(
        root=repo,
        product_edits=0,
        last_test_exit=0,
        _last_green_revision="green",
        receipt=lambda *args, **kwargs: receipts.append((args, kwargs)),
    )
    runtime = SimpleNamespace(
        target=repo / "app.py",
        journal=SimpleNamespace(manifest={}, emit=lambda *args, **kwargs: None),
        _invalidate_binding_cache=lambda: None,
    )
    try:
        with patch.object(HookRunner, "run_event", return_value=allowed):
            wiring.pre_action("edit", {"path": "app.py"}, "edit-1")
        before = scan(repo, exclude=lambda _: False)
        (repo / "app.py").write_text("x = 2\n")
        after = scan(repo, exclude=lambda _: False)
        effect = diff(before, after)[0]
        writer = WriterResult(
            "fmt",
            (WriterEffect("app.py", effect.before.digest, effect.after.digest),),
            (),
            Reconciliation([effect], [], after),
        )
        completed = EventResult("continue", None, None, "", (), (), (writer,))
        with patch.object(HookRunner, "run_event", return_value=completed):
            wiring.finalize_action(
                "edit",
                {"path": "app.py"},
                "edit-1",
                "edit applied",
                runtime,
                bolt,
                set(),
                outcome=DispatchOutcome("ok", True, None, ("app.py",)),
            )
        assert bolt.product_edits == 1
        assert bolt.last_test_exit is None
        assert bolt._last_green_revision is None
        assert any(args[0] == "shell_effect" for args, _ in receipts)
        assert "fmt changed app.py" in wiring.panel()["content"]
    finally:
        wiring.close()


def test_hook_terminals_stay_authoritative_under_hard_budget_terminal_mode():
    # The shipped profile turns heuristic stop outcomes into advisories; a stop hook's veto limit and lost
    # hook containment are explicit policy stops and must still end the run.
    from types import SimpleNamespace

    import kernel_terminal_policy as policy

    runtime = SimpleNamespace(stateframe_hard_budget_only_terminals_mode="on")
    for outcome in ("hook_blocked", "hook_containment", "HOOK_BLOCKED"):
        assert policy.terminal_category(outcome, runtime) == "policy"
        assert policy.terminal_outcome_permitted(outcome, runtime)
    assert not policy.terminal_outcome_permitted("blocked", runtime)


def test_mcp_admission_keeps_declared_why_like_dispatch():
    # Dispatch sends MCP arguments as advertised; admission must judge the same arguments, or an MCP
    # tool that declares `why` would take the not-admitted route and run without its pre_tool hook.
    from kernel_mcp_tools import McpTool

    server = SimpleNamespace(executable=Path(sys.executable), argv=(sys.executable, "server.py"),
                             timeout_seconds=5, verify_executable=lambda: None)
    parameters = {"type": "object", "required": ["query", "why"],
                  "properties": {"query": {"type": "string", "maxLength": 200},
                                 "why": {"type": "string", "maxLength": 200}}}
    tool = McpTool(name="docs_search", server=server, remote_name="search", description="",
                   parameters=parameters, definition_digest="d")
    agent_tools = SimpleNamespace(custom={"docs_search": tool})
    wiring = object.__new__(HookWiring)
    assert wiring.admitted("docs_search", {"query": "q", "why": "look it up"}, None, agent_tools)
    assert not wiring.admitted("docs_search", {"query": "q"}, None, agent_tools)


def test_command_tool_admission_drops_engine_why():
    seen = []

    class Command:
        def build_argv(self, arguments):
            seen.append(dict(arguments))
            return ["/bin/true"]

        def verify_executable(self):
            return None

    agent_tools = SimpleNamespace(custom={"lint": Command()})
    wiring = object.__new__(HookWiring)
    assert wiring.admitted("lint", {"path": "a.py", "why": "check"}, None, agent_tools)
    assert seen == [{"path": "a.py"}]
