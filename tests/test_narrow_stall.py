"""Narrow regressions for current checks and host-authorized edit advice."""

import ast
import copy
import hashlib
import inspect
import json
import textwrap
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernel_stateframe_action_ledger import build_action_context
from kernel_stateframe_f4 import project_f4_state
from kernel_stateframe_framefit5 import (
    _next_sites, _serialize_state_message, assemble_framefit5_request,
    build_typed_state, validate_final_product_boundary,
)
from kernel_stateframe_runtime import StateFrameRuntime


PRODUCT_REVISION = "sha256:changed-product"
ENTRY_REVISION = "sha256:unchanged-entry"
RUN = Path(
    "/Users/marida/Repos/learning-ai/.luna/product/_lab_evidence/a10_runs/"
    "a10_t2_sf020_t2_1031084555_65802"
)


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
    return [
        {"role": "assistant", "tool_calls": [{"id": call_id, "type": "function",
            "function": {"name": "read", "arguments": json.dumps({"path": path})}}]},
        {"role": "tool", "tool_call_id": call_id,
         "content": f"[read page] path={path} revision={revision} lines=1-1/1 "
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
    ("pkg/other.py", ["pkg/app.py"], None),
    ("pkg/app.py", ["pkg/app.py"], "pkg/app.py"),
])
def test_frame_construction_and_final_validation_obey_host_edit_scope(
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


def test_missing_host_edit_scope_recommends_no_site(tmp_path):
    messages, context, runtime = _frame_request(tmp_path, "pkg/app.py", ["pkg/app.py"])
    runtime.edit_contract = {"edit_authority": {}}
    state = build_typed_state(messages, context, runtime)
    assert state["next_site"] is None
    assert state["open_sites"] == []


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
    assert [site["path"] for site in sites] == ["pkg/app.py"]


def test_recorded_t2_frames_change_only_public_checks_and_forbidden_advice(monkeypatch):
    if not RUN.is_dir():
        pytest.skip("recorded T2 evidence is unavailable")
    journal_dir = next((RUN / "worker/measurement-runs").iterdir())
    events = [json.loads(line) for line in (journal_dir / "events.jsonl").read_text().splitlines()]
    snapshots = [event for event in events if event.get("kind") == "snapshot_committed"]
    submissions = {event["request_id"]: event["sequence"] for event in events
                   if event.get("kind") == "request_submitted"}
    monkeypatch.setenv("STATEFRAME_REMAINING_SITES", "off")
    checked = 0
    removed = 0
    changed = 0
    for request_path in sorted((RUN / "requests").glob("*.json")):
        request = json.loads(request_path.read_text())
        state_messages = [row for row in request["messages"]
                          if isinstance(row.get("content"), str)
                          and row["content"].startswith("[StateFrame]\n")]
        assert len(state_messages) == 1
        saved = json.loads(state_messages[0]["content"].split("\n", 1)[1])
        old_state = saved["state"]
        before = [row for row in snapshots
                  if row["sequence"] < submissions[request["request_id"]]]
        assert before
        digest = before[-1]["frame"]["$blob"]
        raw = zlib.decompress((journal_dir / "blobs" / (digest + ".zlib")).read_bytes())
        frame = json.loads(raw)
        revision = frame["readiness"]["revision"]
        old_projection = project_f4_state(frame, path="pkg/__init__.py")
        new_projection = project_f4_state(
            frame, path="pkg/__init__.py", current_product_revision=revision)
        assert old_state["public_checks"]["state"] == old_projection["state"]["public_checks"]["state"]
        old_without_checks = copy.deepcopy(old_projection)
        new_without_checks = copy.deepcopy(new_projection)
        old_without_checks["claims"][0] = new_without_checks["claims"][0]
        old_without_checks["state"]["public_checks"] = new_without_checks["state"]["public_checks"]
        assert old_without_checks == new_without_checks
        if old_projection["state"]["public_checks"] != new_projection["state"]["public_checks"]:
            changed += 1

        after = copy.deepcopy(old_state)
        site = after["framefit5"].get("next_site")
        authority = after["edit_authority"]["editable_paths"]
        runtime = SimpleNamespace(
            owner=SimpleNamespace(root=RUN / "workspace"),
            edit_contract={"edit_authority": {"editable_paths": authority}},
            protected_tests={RUN / "workspace" / path for path in
                             ("tests/test_batch_contract.py", "tests/test_consumers.py",
                              "tests/test_preservation.py",
                              "tests/test_registration_serialization.py")},
        )
        candidates = _next_sites(runtime, {"open_edit_site": site}) if site else []
        forbidden = site is not None and not candidates
        if candidates:
            assert candidates[0]["path"] == site["path"]
        if forbidden:
            after["framefit5"]["next_site"] = None
            after["framefit5"]["open_work"]["next_site_available"] = False
            removed += 1
        assert (after["framefit5"].get("next_site") is None or
                after["framefit5"]["next_site"]["path"] in authority)
        assert old_state["framefit5"]["action_ledger"] == after["framefit5"]["action_ledger"]
        checked += 1
    assert checked == 15
    assert changed > 0
    assert removed > 0
