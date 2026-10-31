"""Narrow regressions for current checks and host-authorized edit advice."""

import ast
import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

from kernel_stateframe_action_ledger import build_action_context
from kernel_stateframe_f4 import project_f4_state
from kernel_stateframe_framefit5 import (
    _editable_site, _next_sites, _serialize_state_message, assemble_framefit5_request,
    build_typed_state, validate_final_product_boundary,
)
from kernel_stateframe_runtime import StateFrameRuntime
from kernel_context import token_estimate
from kernel_frame_allocation_capture import finalize_frame_candidate
from kernel_live_projection import WorkspaceViewStore
from kernel_request_evidence_closure import SpanNeed, build_source_closure


PRODUCT_REVISION = "sha256:changed-product"
ENTRY_REVISION = "sha256:unchanged-entry"
def _failing_frame():
    return {
        "artifact_revisions": {"pkg/__init__.py": ENTRY_REVISION,
                               "pkg/service.py": "sha256:changed-sibling"},
        "verification": {"revision": PRODUCT_REVISION, "status": "FAIL"},
        "readiness": {"revision": PRODUCT_REVISION, "public_check_status": "FAIL"},
        "test_receipts": [{"revision": PRODUCT_REVISION, "result_class": "RED",
                           "receipt_id": "current-failure"}],
    }


def test_runtime_f4_binds_failed_checks_to_changed_product(monkeypatch):
    import kernel_stateframe_f4

    frame = _failing_frame()
    baseline = project_f4_state(frame, path="pkg/__init__.py")
    assert baseline["state"]["public_checks"]["state"] == "NOT_RUN"
    projection = {}

    class Captured(Exception):
        pass

    original = kernel_stateframe_f4.project_f4_state

    def capture(*args, **kwargs):
        projection.update(original(*args, **kwargs))
        raise Captured

    monkeypatch.setattr(kernel_stateframe_f4, "project_f4_state", capture)
    runtime = SimpleNamespace(
        owner=SimpleNamespace(f2p_ids=("tests/test_product.py",)),
        binding=lambda: {"product_revision": PRODUCT_REVISION},
    )
    with pytest.raises(Captured):
        StateFrameRuntime._f4_view(runtime, [], frame, "", "pkg/__init__.py", [])
    assert projection["state"]["public_checks"]["state"] == "FAILED"
    assert projection["state"]["public_checks"]["revision"] == PRODUCT_REVISION
    assert projection["state"]["revision"] == baseline["state"]["revision"]


def test_live_red_dispatch_skips_blank_symbol_and_dispatches_valid_one(tmp_path, capsys):
    from probes import react_repair

    tree = ast.parse(textwrap.dedent(inspect.getsource(react_repair.work)))
    branches = [node for node in ast.walk(tree)
                if isinstance(node, ast.If) and isinstance(node.test, ast.Call)
                and isinstance(node.test.func, ast.Attribute)
                and node.test.func.attr == "startswith"
                and node.test.args and isinstance(node.test.args[0], ast.Constant)
                and node.test.args[0].value == "test"
                and any(isinstance(child, ast.Assign)
                        and any(isinstance(target, ast.Name)
                                and target.id == "edits_since_test" for target in child.targets)
                        for child in node.body)]
    assert len(branches) == 1
    code = compile(ast.fix_missing_locations(ast.Module(body=branches[0].body,
                                                      type_ignores=[])),
                   react_repair.__file__, "exec")
    target = tmp_path / "runner.py"
    target.write_text("def run_batch():\n    return []\n")
    test_file = tmp_path / "tests/test_batch.py"
    test_file.parent.mkdir()
    test_file.write_text("def test_batch():\n    assert False\n")
    seen = []

    class Nit:
        def observe_red(self, evidence):
            seen.append(evidence)
            return SimpleNamespace(render_red=lambda: "red detail", count=1)

    bolt = SimpleNamespace(last_test_exit=1, _green_gen=0, nit=Nit(),
                           organ_out=lambda _name, value: value,
                           _rec=lambda *_args: None)
    for symbol in ("", "run_batch"):
        scope = dict(bolt=bolt, edits_since_test=2, step=3, fn=symbol,
                     d={"rel": "runner.py", "sha": "12345678"},
                     target=target, root=tmp_path,
                     line="test tests/test_batch.py::test_batch", obs="assert False",
                     _red_confrontation_target=react_repair._red_confrontation_target)
        exec(code, scope)
        assert scope["edits_since_test"] == 0
        assert len(seen) == (1 if symbol else 0)
    assert seen[0].target_symbol == "run_batch"
    assert "red-confront failed" not in capsys.readouterr().out


def _read(path, content, call_id):
    revision = "sha256:" + hashlib.sha256(content.encode()).hexdigest()
    line_count = len(content.splitlines())
    return [
        {"role": "assistant", "tool_calls": [{"id": call_id, "type": "function",
            "function": {"name": "read", "arguments": json.dumps({"path": path})}}]},
        {"role": "tool", "tool_call_id": call_id,
         "content": f"[read page] path={path} revision={revision} lines=1-{line_count}/{line_count} "
                    f"chars={len(content)} clip=none\ncurrent source\n{content}"},
    ]


def _frame_request(tmp_path, path, allowed):
    contents = {"pkg/app.py": "VALUE = 1\n",
                "pkg/other.py": "OTHER = 1\n",
                "tests/test_app.py": "assert True\n"}
    for name, content in contents.items():
        file = tmp_path / name
        file.parent.mkdir(exist_ok=True)
        file.write_text(content)
    task = {"role": "user", "content": "Task: Change the product value."}
    messages = [task, *_read(path, contents[path], "read-1")]
    context = build_action_context(messages, workspace_root=tmp_path)
    context["journal_receipt_records"] = []
    runtime = SimpleNamespace(
        owner=SimpleNamespace(root=tmp_path, task_row={}),
        edit_contract={"edit_authority": {"editable_paths": allowed}},
        protected_tests={tmp_path / "tests/test_app.py": "baseline"},
        frame_ceiling=10_000,
        _framefit_action_context=context,
    )
    return messages, context, runtime


@pytest.mark.parametrize("path,allowed,expected", [
    ("tests/test_app.py", ["tests/test_app.py", "pkg/app.py"], None),
    # Advice is not an edit authorization check; the frozen startup contract
    # cannot exclude a file created or moved later in the run.
    ("pkg/other.py", ["pkg/app.py"], "pkg/other.py"),
    ("pkg/app.py", ["pkg/app.py"], "pkg/app.py"),
])
def test_frame_construction_and_final_validation_obey_protected_scope(
        tmp_path, monkeypatch, path, allowed, expected):
    import kernel_stateframe_framefit5 as framefit5

    messages, context, runtime = _frame_request(tmp_path, path, allowed)
    if expected:
        with monkeypatch.context() as baseline:
            baseline.setattr(framefit5, "_editable_site", lambda *_args: True)
            old_state = build_typed_state(messages, context, runtime)
            old_prepared = [{"role": "system", "content": "rules"}, messages[0],
                            {"role": "user", "content": _serialize_state_message(
                                {"state": {"framefit5": old_state}})}]
            old_assembled = assemble_framefit5_request(
                old_prepared, old_prepared[:2], live_frame=old_state, runtime=runtime)
            old_validation = validate_final_product_boundary(
                runtime, old_assembled, [], refresh_sources=True,
                refresh_headcarry=False)
    state = build_typed_state(messages, context, runtime)
    if expected:
        assert json.dumps(state, sort_keys=True) == json.dumps(old_state, sort_keys=True)
    assert (state["next_site"] or {}).get("path") == expected
    assert [site["path"] for site in state["open_sites"]] == ([expected] if expected else [])
    assert path in json.dumps(state["action_ledger"])
    prepared = [{"role": "system", "content": "rules"}, messages[0],
                {"role": "user", "content": _serialize_state_message(
                    {"state": {"framefit5": state}})}]
    assembled = assemble_framefit5_request(prepared, prepared[:2],
                                           live_frame=state, runtime=runtime)
    validation = validate_final_product_boundary(
        runtime, assembled, [], refresh_sources=True, refresh_headcarry=False)
    assert validation["next_site_path"] == expected
    if expected:
        assert json.dumps(assembled, sort_keys=True) == json.dumps(old_assembled, sort_keys=True)
        assert validation == old_validation


def test_missing_startup_edit_scope_still_recommends_site(tmp_path):
    messages, context, runtime = _frame_request(tmp_path, "pkg/app.py", ["pkg/app.py"])
    runtime.edit_contract = {"edit_authority": {}}
    state = build_typed_state(messages, context, runtime)
    assert state["next_site"]["path"] == "pkg/app.py"
    assert [site["path"] for site in state["open_sites"]] == ["pkg/app.py"]


def test_inventory_uses_next_original_candidate_after_forbidden(tmp_path, monkeypatch):
    import kernel_remaining_sites

    _, context, runtime = _frame_request(tmp_path, "pkg/app.py", ["pkg/app.py"])
    candidates = []
    for path, text in (("tests/test_app.py", "assert True"),
                       ("pkg/other.py", "OTHER = 1"),
                       ("pkg/app.py", "VALUE = 1")):
        raw = (tmp_path / path).read_bytes()
        candidates.append({"path": path, "line": 1, "text": text,
                           "revision": "sha256:" + hashlib.sha256(raw).hexdigest()})
    monkeypatch.setattr(kernel_remaining_sites, "remaining_site_inventory",
                        lambda _runtime: candidates)
    sites = _next_sites(runtime, context)
    assert [site["path"] for site in sites] == ["pkg/other.py", "pkg/app.py"]


def test_outside_workspace_symlink_is_not_edit_advice(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "pkg").mkdir()
    outside = tmp_path / "external.py"
    outside.write_text("VALUE = 1\n")
    (root / "pkg/escape.py").symlink_to(outside)
    runtime = SimpleNamespace(owner=SimpleNamespace(root=root), protected_tests={})
    assert _editable_site(runtime, root.resolve(), "pkg/escape.py") is False
    assert _next_sites(runtime, {"open_edit_site": {"path": "pkg/escape.py", "line": 1}}) == []


def _operation(name, arguments, receipt, call_id):
    return [
        {"role": "assistant", "tool_calls": [{"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}]},
        {"role": "tool", "tool_call_id": call_id, "content": receipt},
    ]


@pytest.mark.parametrize("operation", ["create", "move"])
def test_created_or_moved_destination_remains_edit_advice_after_resume(
        tmp_path, monkeypatch, operation):
    from kernel_file_ops import file_move
    import kernel_stateframe_framefit5

    root = tmp_path / "workspace"
    (root / "pkg").mkdir(parents=True)
    content = "VALUE = 1\n"
    destination = "pkg/new.py"
    created_this_run = set()
    messages = [{"role": "user", "content": "Task: Correct the new module."}]
    if operation == "create":
        (root / destination).write_text(content)
        messages += _operation("file_create", {"path": destination, "content": content},
                               f"created {destination}\n", "create-1")
        created_this_run.add(str(root / destination))
    else:
        source = root / "pkg/original.py"
        source.write_text(content)
        effect = file_move(root, "pkg/original.py", destination,
                           protected=lambda _path: False, editable=lambda _path: True,
                           new_file_allowed=lambda _path: True)
        assert effect.source == "pkg/original.py"
        assert effect.destination == destination
        created_this_run.add(str(root / effect.destination))
    messages += _read(destination, content, "read-1")
    context = build_action_context(messages, workspace_root=root)
    context["journal_receipt_records"] = []
    assert str(root / destination) in created_this_run
    if operation == "create":
        assert any(row.get("operation") == "create" for row in context["ledger"]["rows"])
    for resumed in (False, True):
        # Resume constructs a new runtime view over the retained workspace.
        restored = set(json.loads(json.dumps(sorted(created_this_run)))) if resumed else created_this_run
        runtime = _runtime_with_product(root)
        runtime.edit_contract = {"edit_authority": {"editable_paths": ["pkg/original.py"]}}
        runtime._framefit_action_context = context
        runtime.created_this_run = restored
        if resumed:
            assert destination in runtime.current_view("resume").product_files
            assert destination in runtime.binding()["product_files"]
        with monkeypatch.context() as patch:
            patch.setattr(kernel_stateframe_framefit5, "_editable_site",
                          lambda *_args: True)
            baseline = build_typed_state(messages, context, runtime)
        state = build_typed_state(messages, context, runtime)
        assert state == baseline
        assert state["next_site"]["path"] == destination
        assert [site["path"] for site in state["open_sites"]] == [destination]


def _runtime_with_product(root):
    runtime = StateFrameRuntime.__new__(StateFrameRuntime)
    runtime.owner = SimpleNamespace(root=root, task_row={}, f2p_ids=("tests/test_app.py",))
    runtime.journal = SimpleNamespace(manifest={"environment_revision": "fixture"},
                                      emit=lambda *_args, **_kwargs: None)
    runtime.projector_kind = "rich"
    runtime.context_scope = "fixture"
    runtime._workspace_view_store = WorkspaceViewStore(root, "fixture")
    runtime._binding_cache = None
    runtime._binding_epoch = 0
    runtime.frame_ceiling = 10_000
    runtime.edit_contract = {"edit_authority": {"editable_paths": ["pkg/__init__.py"]}}
    runtime.protected_tests = {root / "tests/test_app.py": "baseline"}
    runtime._empirical_frame_sources = {}
    return runtime


def _runtime_projection(runtime, frame, *, baseline, monkeypatch):
    import kernel_stateframe_f4

    class Captured(Exception):
        pass

    projection = {}
    original = kernel_stateframe_f4.project_f4_state

    def capture(*args, **kwargs):
        if baseline:
            kwargs.pop("current_product_revision", None)
        projection.update(original(*args, **kwargs))
        raise Captured

    with monkeypatch.context() as patch:
        patch.setattr(kernel_stateframe_f4, "project_f4_state", capture)
        with pytest.raises(Captured):
            StateFrameRuntime._f4_view(runtime, [], frame, "", "pkg/__init__.py", [])
    return projection


def _qualified_fixture_request(runtime, frame, messages, context, *, baseline, monkeypatch,
                               state_transform=None, validate=True, return_capture=False):
    import kernel_stateframe_framefit5

    with monkeypatch.context() as patch:
        if baseline:
            # Released selection did not filter edit advice at this boundary.
            patch.setattr(kernel_stateframe_framefit5, "_editable_site",
                          lambda *_args: True)
        state = build_typed_state(messages, context, runtime)
    if state_transform is not None:
        state_transform(state)
    projection = _runtime_projection(runtime, frame, baseline=baseline,
                                     monkeypatch=monkeypatch)
    projection["state"]["framefit5"] = state
    root = runtime.owner.root
    plan = build_source_closure(root, ["pkg/__init__.py"], "Fix value",
                                target_path="pkg/__init__.py").as_dict()
    unit = SpanNeed("pkg/__init__.py", 1, 1, "target", "fixture", 1).as_dict()
    plan["units"] = [{**unit, "unit_handle": "entry:1"}]
    plan["bounded_delivery"] = {"retained_handles": ["entry:1"],
                                "omitted_handles": []}
    runtime._request_source_closure = plan
    prepared = [
        {"role": "system", "content": "rules"}, messages[0],
        {"role": "user", "content": _serialize_state_message(projection)},
    ]
    assembled = assemble_framefit5_request(
        prepared, messages, source_plan=plan, sources=plan["source_contents"],
        runtime=runtime,
    )
    body = {"messages": assembled, "tools": []}
    capture = SimpleNamespace(
        request_id="fixture-1", journal=runtime.journal,
        _framefit5_semantic_messages=messages, _framefit5_task_message=messages[0],
        _request_source_closure_override=plan,
    )
    result = finalize_frame_candidate(runtime, capture, body, frame_counter=token_estimate)
    if validate:
        validation = validate_final_product_boundary(
            runtime, result["messages"], result["tools"],
            refresh_sources=False, refresh_headcarry=False,
        )
        assert validation["checked_before_provider"] is True
    return (result, capture) if return_capture else result


def _materialized_request(body):
    result = copy.deepcopy(body)
    state_messages = [message for message in result["messages"]
                      if isinstance(message.get("content"), str)
                      and message["content"].startswith("[StateFrame]\n")]
    assert len(state_messages) == 1
    state_messages[0]["content"] = json.loads(state_messages[0]["content"].split("\n", 1)[1])
    return result


def _different_paths(before, after, prefix=()):
    if isinstance(before, dict) and isinstance(after, dict):
        assert before.keys() == after.keys()
        return [path for key in before
                for path in _different_paths(before[key], after[key], (*prefix, key))]
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        return [path for index in range(len(before))
                for path in _different_paths(before[index], after[index], (*prefix, index))]
    return [] if before == after else [prefix]


@pytest.mark.parametrize("case", ["allowed", "protected", "near_budget", "multi_file"])
def test_portable_differential_changes_only_checks_and_protected_advice(
        tmp_path, monkeypatch, case):
    root = tmp_path / "workspace"
    (root / "pkg").mkdir(parents=True)
    (root / "tests").mkdir()
    root = root.resolve()
    entry = root / "pkg/__init__.py"
    entry.write_text("VALUE = 1\n")
    sibling = root / "pkg/sibling.py"
    sibling.write_text("SIBLING = 1\n")
    protected = root / "tests/test_app.py"
    protected.write_text("assert True\n")
    runtime = _runtime_with_product(root)
    initial = runtime.binding()["product_revision"]
    entry_revision = "sha256:" + hashlib.sha256(entry.read_bytes()).hexdigest()
    sibling.write_text("SIBLING = 2\n")
    runtime._workspace_view_store.note_mutation(("pkg/sibling.py",), "edit")
    runtime._invalidate_binding_cache()
    current = runtime.binding()["product_revision"]
    assert current != initial
    assert runtime.binding()["product_files"]["pkg/__init__.py"] == entry_revision
    frame = {
        "artifact_revisions": {"pkg/__init__.py": entry_revision,
                               "pkg/sibling.py": runtime.binding()["product_files"]["pkg/sibling.py"]},
        "verification": {"revision": current, "status": "FAIL"},
        "readiness": {"revision": current, "public_check_status": "FAIL"},
        "test_receipts": [{"revision": current, "result_class": "RED",
                           "receipt_id": "current-failure"}],
    }
    path = "tests/test_app.py" if case in {"protected", "near_budget"} else "pkg/__init__.py"
    content = (root / path).read_text()
    messages = [{"role": "user", "content": "Task: Fix value."},
                *_read(path, content, "read-1")]
    context = build_action_context(messages, workspace_root=root)
    context["journal_receipt_records"] = []
    runtime._framefit_action_context = context
    if case == "near_budget":
        runtime.frame_ceiling = 2_200
    baseline = _qualified_fixture_request(runtime, frame, messages, context,
                                          baseline=True, monkeypatch=monkeypatch)
    patched = _qualified_fixture_request(runtime, frame, messages, context,
                                         baseline=False, monkeypatch=monkeypatch)
    old_request = _materialized_request(baseline)
    new_request = _materialized_request(patched)
    state_index = next(index for index, message in enumerate(old_request["messages"])
                       if isinstance(message.get("content"), dict)
                       and message["content"].get("schema") == "stateframe-f4/1")
    old_state = old_request["messages"][state_index]["content"]
    new_state = new_request["messages"][state_index]["content"]
    old_checks = old_state["state"]["public_checks"]
    new_checks = new_state["state"]["public_checks"]
    assert old_checks["state"] == "NOT_RUN"
    assert new_checks["state"] == "FAILED"
    assert new_checks["revision"] == current
    site = old_state["state"]["framefit5"]["next_site"]
    forbidden = (root / path).resolve() in {
        protected_path.resolve() for protected_path in runtime.protected_tests
    }
    assert site["path"] == path
    assert forbidden == (case in {"protected", "near_budget"})
    assert (new_state["state"]["framefit5"]["next_site"] is None) == forbidden
    checks = ("messages", state_index, "content")
    permitted = [
        (*checks, "claims", 0, "evidence"),
        (*checks, "claims", 0, "revision"),
        (*checks, "claims", 0, "state"),
        (*checks, "state", "public_checks", "evidence"),
        (*checks, "state", "public_checks", "revision"),
        (*checks, "state", "public_checks", "state"),
    ]
    if forbidden:
        permitted += [
            (*checks, "state", "framefit5", "next_site"),
            (*checks, "state", "framefit5", "open_sites"),
            (*checks, "state", "framefit5", "open_work", "next_site_available"),
            (*checks, "state", "framefit5", "open_work", "next_site_source"),
            (*checks, "state", "framefit5", "open_work", "open_edit_site_count"),
        ]
    differences = _different_paths(old_request, new_request)
    assert differences
    assert all(any(path[:len(allowed)] == allowed for allowed in permitted)
               for path in differences), differences
    if case == "near_budget":
        assert token_estimate(patched) >= runtime.frame_ceiling * 0.8


def test_small_request_bookmark_switch_is_byte_identical(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    (root / "pkg").mkdir(parents=True)
    (root / "tests").mkdir()
    root = root.resolve()
    entry = root / "pkg/__init__.py"
    entry.write_text("VALUE = 1\n")
    (root / "tests/test_app.py").write_text("assert True\n")
    runtime = _runtime_with_product(root)
    revision = runtime.binding()["product_revision"]
    frame = {
        "artifact_revisions": {"pkg/__init__.py": "sha256:" + hashlib.sha256(entry.read_bytes()).hexdigest()},
        "verification": {"revision": revision, "status": "FAIL"},
        "readiness": {"revision": revision, "public_check_status": "FAIL"},
        "test_receipts": [{"revision": revision, "result_class": "RED",
                           "receipt_id": "current-failure"}],
    }
    messages = [{"role": "user", "content": "Task: Fix value."},
                *_read("pkg/__init__.py", entry.read_text(), "read-1")]
    context = build_action_context(messages, workspace_root=root)
    context["journal_receipt_records"] = []
    runtime._framefit_action_context = context
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "off")
    baseline = _qualified_fixture_request(runtime, frame, messages, context,
                                          baseline=False, monkeypatch=monkeypatch)
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    enabled = _qualified_fixture_request(runtime, frame, messages, context,
                                          baseline=False, monkeypatch=monkeypatch)
    assert json.dumps(enabled, sort_keys=True) == json.dumps(baseline, sort_keys=True)
    parent_source = subprocess.check_output([
        "git", "show", "a0ce931:src/stateframe/engine/kernel_frame_allocation_capture.py"
    ], cwd=Path(__file__).resolve().parents[1])
    parent_path = tmp_path / "parent_frame_allocation_capture.py"
    parent_path.write_bytes(parent_source)
    spec = importlib.util.spec_from_file_location("parent_frame_allocation_capture", parent_path)
    assert spec is not None and spec.loader is not None
    parent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parent)
    with monkeypatch.context() as patch:
        patch.setattr(sys.modules[__name__], "finalize_frame_candidate",
                      parent.finalize_frame_candidate)
        parent_request = _qualified_fixture_request(
            runtime, frame, messages, context, baseline=False, monkeypatch=monkeypatch)
    assert json.dumps(enabled, sort_keys=True) == json.dumps(parent_request, sort_keys=True)


def test_pressured_bookmarks_pass_final_allocation_and_validation(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    (root / "pkg").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "pkg/__init__.py").write_text("VALUE = 1\n")
    (root / "tests/test_app.py").write_text("assert True\n")
    names = [f"pkg/module_{index}.py" for index in range(10)]
    for index, name in enumerate(names):
        (root / name).write_text(f"def target(): return {index}\n" +
                                 ("OTHER = 42\n" if index == 0 else ""))
    root = root.resolve()
    runtime = _runtime_with_product(root)
    runtime.edit_contract = {"edit_authority": {"editable_paths": ["pkg/__init__.py", *names]}}
    revision = runtime.binding()["product_revision"]
    frame = {
        "artifact_revisions": {"pkg/__init__.py": "sha256:" + hashlib.sha256(
            (root / "pkg/__init__.py").read_bytes()).hexdigest()},
        "verification": {"revision": revision, "status": "FAIL"},
        "readiness": {"revision": revision, "public_check_status": "FAIL"},
        "test_receipts": [{"revision": revision, "result_class": "RED",
                           "receipt_id": "current-failure"}],
    }
    messages = [{"role": "user", "content": "Task: Fix the modules."}]
    for index, name in enumerate(names):
        messages += _read(name, (root / name).read_text(), f"read-{index}")
    messages += _read(names[0], (root / names[0]).read_text(), "read-extra")
    context = build_action_context(messages, workspace_root=root)
    context["journal_receipt_records"] = [{
        "kind": "action_receipt",
        "receipt": {"kind": "read", "target_path": name, "result_class": "OK",
                    "receipt_id": f"read-receipt:{index}", "workspace_revision": "1"},
    } for index, name in enumerate(names)]
    runtime._framefit_action_context = context
    receipt_id = "check-receipt:bookmark-fixture"
    runtime._check_output_receipts = {receipt_id: {
        "status": "FAIL", "revision": revision,
        "stdout": "Traceback (most recent call last):\n" + "".join(
            f'  File "{name}", line 1, in target\n'
            for name in names[:3]) + "AssertionError: wrong result\n",
    }}

    def mark_failure(state):
        assert len(state["read_facts"]) >= 10
        for fact in state["read_facts"]:
            fact["body_delivery"] = "not_shown"
        state["last_verdict"] = {
            "status": "FAIL", "tested_revision": revision,
            "current_revision": revision, "check_receipt_id": receipt_id,
            "line": f"LAST FAIL at rev {revision}",
        }
        state["action_ledger"]["journal_verdict"] = copy.deepcopy(state["last_verdict"])

    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    result, capture = _qualified_fixture_request(
        runtime, frame, messages, context, baseline=False, monkeypatch=monkeypatch,
        state_transform=mark_failure, validate=False, return_capture=True,
    )
    assert token_estimate(result) <= runtime.frame_ceiling
    state_message = next(message for message in result["messages"]
                         if str(message.get("content") or "").startswith("[StateFrame]\n"))
    projected = json.loads(state_message["content"].split("\n", 1)[1])["state"]["framefit5"]
    from kernel_reread_result_truth import reconcile_frame
    expected = copy.deepcopy(projected)
    reconcile_frame(expected, result["messages"])
    assert expected["read_facts"] == projected["read_facts"], [
        (before["path"], before.get("body_delivery"), after.get("body_delivery"))
        for before, after in zip(projected["read_facts"], expected["read_facts"])
        if before != after
    ]
    validate_final_product_boundary(
        runtime, result["messages"], result["tools"],
        refresh_sources=False, refresh_headcarry=False,
    )
    assert set(names[:3]) <= {row["path"] for row in projected["code_bookmarks"]
                              if row["rank"] == 1}
    delivery = {row["path"]: row["body_delivery"] for row in projected["read_facts"]}
    assert all(delivery[name] in {"present", "bookmarked"} for name in names[:3])
    from kernel_request_source_window import parse_model_source_window
    same_path_ranges = [metadata.get("ranges") for message in result["messages"]
                        if (metadata := parse_model_source_window(message.get("content")))
                        and metadata.get("path") == names[0]]
    assert any(any(span.get("end_line") == 2 for span in ranges)
               for ranges in same_path_ranges)
    assert capture._code_bookmark_transition[0] is True
    assert getattr(runtime, "_code_bookmarks_active", False) is False
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "off")
    disabled = _qualified_fixture_request(
        runtime, frame, messages, context, baseline=False, monkeypatch=monkeypatch,
        state_transform=mark_failure, validate=False)
    from kernel_reread_result_truth import refresh_messages
    refresh_messages(disabled["messages"])
    validate_final_product_boundary(
        runtime, disabled["messages"], disabled["tools"],
        refresh_sources=False, refresh_headcarry=False)
    baseline_ceiling = token_estimate(disabled)
    runtime.frame_ceiling = baseline_ceiling
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    import kernel_code_bookmarks
    original_collect = kernel_code_bookmarks.collect
    def padded_optional_metadata(bookmark_runtime, bookmark_frame):
        rows = original_collect(bookmark_runtime, bookmark_frame)
        for row in rows:
            row["origin"] += "x" * 2000
        return rows

    with monkeypatch.context() as patch:
        patch.setattr(kernel_code_bookmarks, "collect", padded_optional_metadata)
        tight, tight_capture = _qualified_fixture_request(
            runtime, frame, messages, context, baseline=False, monkeypatch=monkeypatch,
            state_transform=mark_failure, return_capture=True)
    assert token_estimate(tight) <= baseline_ceiling
    assert tight_capture._code_bookmark_transition[0] is False
    runtime.frame_ceiling = 10_000
    def reject_validation(*_args, **_kwargs):
        raise ValueError("reprepare")

    with monkeypatch.context() as patch:
        patch.setattr(sys.modules[__name__], "validate_final_product_boundary",
                      reject_validation)
        with pytest.raises(ValueError, match="reprepare"):
            _qualified_fixture_request(
                runtime, frame, messages, context, baseline=False, monkeypatch=monkeypatch,
                state_transform=mark_failure)
    assert getattr(runtime, "_code_bookmark_carry", None) is None
    retry = _qualified_fixture_request(
        runtime, frame, messages, context, baseline=False, monkeypatch=monkeypatch,
        state_transform=mark_failure)
    assert json.dumps(retry, sort_keys=True) == json.dumps(result, sort_keys=True)
    runtime._code_bookmarks_active, runtime._code_bookmark_carry = capture._code_bookmark_transition
    runtime.frame_ceiling = token_estimate(result) + 200
    near = _qualified_fixture_request(
        runtime, frame, messages, context, baseline=False, monkeypatch=monkeypatch,
        state_transform=mark_failure,
    )
    assert 0.9 * runtime.frame_ceiling <= token_estimate(near) <= runtime.frame_ceiling
    restarted = _runtime_with_product(root)
    restarted.edit_contract = copy.deepcopy(runtime.edit_contract)
    restarted.frame_ceiling = runtime.frame_ceiling
    restarted._framefit_action_context = context
    restarted._check_output_receipts = copy.deepcopy(runtime._check_output_receipts)
    replayed = _qualified_fixture_request(
        restarted, frame, messages, context, baseline=False, monkeypatch=monkeypatch,
        state_transform=mark_failure,
    )
    assert json.dumps(replayed, sort_keys=True) == json.dumps(near, sort_keys=True)

    # The old check is stale after an edit, but the qualified identity follows
    # its function into the next frame and remains checkpointable.
    changed = root / names[0]
    changed.write_text("\n" * 5 + changed.read_text())
    runtime._workspace_view_store.note_mutation((names[0],), "edit")
    runtime._invalidate_binding_cache()
    updated_revision = runtime.binding()["product_revision"]
    runtime.frame_ceiling = 10_000

    def mark_stale(state):
        state["last_verdict"] = {
            "status": "STALE", "tested_revision": revision,
            "current_revision": updated_revision,
            "check_receipt_id": receipt_id, "line": "STALE after edit",
        }
        state["action_ledger"]["journal_verdict"] = copy.deepcopy(state["last_verdict"])

    after_edit = _qualified_fixture_request(
        runtime, frame, messages, context, baseline=False, monkeypatch=monkeypatch,
        state_transform=mark_stale,
    )
    state_message = next(message for message in after_edit["messages"]
                         if str(message.get("content") or "").startswith("[StateFrame]\n"))
    current = json.loads(state_message["content"].split("\n", 1)[1])["state"]["framefit5"]
    assert any(row["path"] == names[0] and row["symbol"] == "target"
               and row["start_line"] == 6 for row in current["code_bookmarks"])
    assert runtime._code_bookmarks_active is True
    from kernel_reread_result_truth import checkpoint_replay, restore_replay
    checkpoint = checkpoint_replay(runtime)
    resumed = _runtime_with_product(root)
    resumed.edit_contract = copy.deepcopy(runtime.edit_contract)
    resumed.frame_ceiling = runtime.frame_ceiling
    resumed._framefit_action_context = context
    resumed._check_output_receipts = copy.deepcopy(runtime._check_output_receipts)
    restore_replay(resumed, checkpoint)
    resumed_request = _qualified_fixture_request(
        resumed, frame, messages, context, baseline=False, monkeypatch=monkeypatch,
        state_transform=mark_stale,
    )
    assert json.dumps(resumed_request, sort_keys=True) == json.dumps(after_edit, sort_keys=True), (
        [".".join(map(str, path)) for path in _different_paths(
            _materialized_request(resumed_request), _materialized_request(after_edit))]
    )
