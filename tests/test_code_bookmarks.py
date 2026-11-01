"""Code locations survive line shifts and respect final source delivery."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import copy
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernel_code_bookmarks import _source, collect, resolve, windows
from kernel_context import token_estimate
from kernel_frame_source_snapshot import _current_window, fit_recent_read_windows
from kernel_read_exec import execute_read
from kernel_request_source_window import verify_model_source_window
from kernel_reread_result_truth import reconcile_frame


def _revision(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _workspace(tmp_path: Path):
    root = tmp_path / "work"
    (root / "pkg").mkdir(parents=True)
    (root / "tests").mkdir()
    names = [f"pkg/module_{index}.py" for index in range(10)]
    for index, name in enumerate(names):
        (root / name).write_text(
            "from math import sqrt\n"
            f"FACTOR = {index + 1}\n"
            "class Worker:\n"
            "    def target(self, value):\n"
            "        return sqrt(value * FACTOR)\n"
            "\n"
            "def unrelated():\n"
            "    return 1\n"
        )
    receipt_id = "check-receipt:one"
    failure = "Traceback (most recent call last):\n" + "".join(
        f'  File "{name}", line 5, in target\n'
        "    return sqrt(value * FACTOR)\n"
        for name in names[:3]
    ) + "AssertionError: wrong value\n"
    runtime = SimpleNamespace(
        owner=SimpleNamespace(root=root, task_row={}),
        edit_contract={"edit_authority": {"editable_paths": names}},
        _check_output_receipts={receipt_id: {"status": "FAIL", "revision": "current",
                                            "stdout": failure}},
        _framefit_action_context={"ledger": {"rows": []}},
    )
    frame = {
        "last_verdict": {"status": "FAIL", "tested_revision": "current",
                         "current_revision": "current", "check_receipt_id": receipt_id},
        "read_facts": [{"path": name, "read_revision": _revision(root / name),
                        "current_revision": _revision(root / name),
                        "status": "UNCHANGED", "unchanged_since_read": True}
                       for name in names],
        "action_ledger": {"reads": []},
    }
    return root, runtime, frame, names


def test_failed_traceback_keeps_three_current_units_and_context(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    root, runtime, frame, names = _workspace(tmp_path)
    bookmarks = collect(runtime, frame)
    assert {row["path"] for row in bookmarks if row["rank"] == 1} == set(names[:3])
    assert all(row["symbol"] == "Worker.target" for row in bookmarks)
    frame["code_bookmarks"] = bookmarks
    selected = windows(root, bookmarks)
    assert len(selected) == 3
    for row in selected:
        content = row["content"]
        assert "from math import sqrt" in content
        assert "FACTOR = " in content
        assert "class Worker:" in content
        assert "whole file shown" in content
        assert "read the symbol or range for more" not in content
        assert verify_model_source_window(content, path=row["path"],
                                          source=(root / row["path"]).read_text())
    reconcile_frame(frame, [{"role": "user", "content": row["content"]}
                            for row in selected])
    deliveries = {row["path"]: row["body_delivery"] for row in frame["read_facts"]}
    assert all(deliveries[name] == "present" for name in names[:3])
    assert all(deliveries[name] == "not_shown" for name in names[3:])


def test_edit_discharges_bookmark_visibility_claim(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    _root, _runtime, frame, _names = _workspace(tmp_path)
    fact = frame["read_facts"][0]
    fact["current_revision"] = "sha256:edited"
    fact["body_delivery"] = "bookmarked"
    fact["visible_symbols"] = ["Worker.target"]
    fact["delivery_notice"] = "[bookmark shown] old source"
    reconcile_frame(frame, [])
    assert fact["body_delivery"] == "not_owed"
    assert "visible_symbols" not in fact
    assert "delivery_notice" not in fact


def test_bookmark_cap_evicts_oldest_within_rank(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    _root, runtime, frame, names = _workspace(tmp_path)
    runtime._check_output_receipts["check-receipt:one"]["stdout"] = (
        "Traceback (most recent call last):\n" + "".join(
            f'  File "{name}", line 5, in target\n'
            for name in names) + "AssertionError: wrong result\n"
    )
    bookmarks = collect(runtime, frame)
    assert len(bookmarks) == 8
    assert {row["path"] for row in bookmarks} == set(names[2:])
    assert all(row["rank"] == 1 for row in bookmarks)


def test_carried_symbol_follows_edit_with_stale_failure_and_resume(tmp_path, monkeypatch):
    from kernel_reread_result_truth import checkpoint_replay, restore_replay

    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    root, runtime, frame, names = _workspace(tmp_path)
    original = collect(runtime, frame)
    assert original
    runtime._code_bookmarks_active = True
    runtime._code_bookmark_carry = original
    path = root / names[0]
    path.write_text("\n" * 7 + path.read_text())
    frame["read_facts"][0]["current_revision"] = _revision(path)
    frame["last_verdict"]["current_revision"] = "after-edit"
    shifted = collect(runtime, frame)
    target = next(row for row in shifted if row["path"] == names[0])
    assert target["symbol"] == "Worker.target"
    assert target["start_line"] == original[0]["start_line"] + 7
    assert target["rank"] == 2
    assert any("unit 11-12" in row["content"] for row in windows(root, shifted)
               if row["path"] == names[0])

    snapshot = checkpoint_replay(runtime)
    fresh = SimpleNamespace(
        owner=runtime.owner, edit_contract=runtime.edit_contract,
        _framefit_action_context=runtime._framefit_action_context,
        _check_output_receipts=runtime._check_output_receipts,
    )
    restore_replay(fresh, snapshot)
    assert collect(fresh, frame) == shifted


def test_pytest_assertion_imports_locate_three_product_modules(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    root, runtime, frame, names = _workspace(tmp_path)
    test_path = root / "tests/test_flow.py"
    test_path.write_text(
        "import importlib\n"
        "def test_flow():\n"
        "    first = importlib.import_module('pkg.module_0')\n"
        "    from pkg import module_1 as second, module_2 as third\n"
        "    assert first.Worker is second.Worker is third.Worker\n"
    )
    frame["read_facts"].append({"path": "tests/test_flow.py",
                                "read_revision": _revision(test_path),
                                "current_revision": _revision(test_path)})
    runtime._check_output_receipts["check-receipt:one"]["stdout"] = (
        "E AssertionError: wrong binding\n"
        "tests/test_flow.py:5: AssertionError\n"
    )
    bookmarks = collect(runtime, frame)
    assert {row["path"] for row in bookmarks if row["rank"] == 1} == set(names[:3]) | {
        "tests/test_flow.py"}
    assert all(row["read_only"] for row in bookmarks if row["path"].startswith("tests/"))


def test_failed_public_test_stays_read_only_even_if_edit_scope_lists_it(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    root, runtime, frame, names = _workspace(tmp_path)
    test_path = root / "tests/test_flow.py"
    test_path.write_text("def test_flow():\n    assert False\n")
    frame["read_facts"].append({"path": "tests/test_flow.py",
                                "read_revision": _revision(test_path),
                                "current_revision": _revision(test_path)})
    runtime.edit_contract["edit_authority"]["editable_paths"].append("tests/test_flow.py")
    runtime._check_output_receipts["check-receipt:one"]["stdout"] = (
        "tests/test_flow.py:2: AssertionError\n")
    bookmarks = collect(runtime, frame)
    assert any(row["path"] == "tests/test_flow.py" and row["read_only"]
               for row in bookmarks)


def test_asserted_local_value_traces_to_producing_symbol(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    root, runtime, frame, names = _workspace(tmp_path)
    product = root / names[2]
    product.write_text("def serialize_batch(items):\n    return list(items)\n")
    frame["read_facts"][2]["read_revision"] = _revision(product)
    frame["read_facts"][2]["current_revision"] = _revision(product)
    test_path = root / "tests/test_flow.py"
    test_path.write_text(
        "import importlib\n"
        "def test_flow():\n"
        "    serializers = importlib.import_module('pkg.module_2')\n"
        "    serialized = serializers.serialize_batch([1])\n"
        "    assert serialized == [2]\n"
    )
    frame["read_facts"].append({"path": "tests/test_flow.py",
                                "read_revision": _revision(test_path),
                                "current_revision": _revision(test_path)})
    runtime._check_output_receipts["check-receipt:one"]["stdout"] = (
        "E AssertionError: incorrect serialization\n"
        "tests/test_flow.py:5: AssertionError\n"
    )
    bookmarks = collect(runtime, frame)
    assert any(row["path"] == names[2] and row["symbol"] == "serialize_batch"
               and row["rank"] == 1 for row in bookmarks)


def test_untrusted_traceback_path_cannot_expand_read_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    root, runtime, frame, _names = _workspace(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("def target():\n    return 'secret'\n")
    (root / "pkg/escape.py").symlink_to(outside)
    runtime._check_output_receipts["check-receipt:one"]["stdout"] = (
        'Traceback (most recent call last):\n'
        '  File "../outside.py", line 1, in target\n'
        '  File "pkg/escape.py", line 1, in target\n'
        'AssertionError: wrong value\n'
    )
    frame["read_facts"].append({"path": "pkg/escape.py",
                                "read_revision": _revision(outside),
                                "current_revision": _revision(outside)})
    assert collect(runtime, frame) == []


def test_stale_failure_receipt_does_not_seed_bookmarks(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    _root, runtime, frame, _names = _workspace(tmp_path)
    runtime._check_output_receipts["check-receipt:one"]["revision"] = "old"
    assert collect(runtime, frame) == []


def test_journal_failure_summary_recovers_bookmarks_without_private_output(
        tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    root, runtime, frame, names = _workspace(tmp_path)
    test_path = root / "tests/test_flow.py"
    test_path.write_text(
        "import importlib\n"
        "def test_flow():\n"
        "    product = importlib.import_module('pkg.module_0')\n"
        "    assert product.Worker is None\n"
    )
    frame["read_facts"].append({"path": "tests/test_flow.py",
                                "read_revision": _revision(test_path),
                                "current_revision": _revision(test_path)})
    runtime._check_output_receipts = {}
    runtime._framefit_action_context["journal_receipt_records"] = [{
        "receipt_id": "check-receipt:one", "status": "FAIL", "revision": "current",
        "summary": ("CHECK_FAILURE_SUMMARY assertion_context='assert product.Worker is None"
                    "\\ntests/test_flow.py:4: AssertionError' runner_tail='1 failed'"),
    }]
    bookmarks = collect(runtime, frame)
    assert any(row["path"] == names[0] and row["symbol"] == "Worker"
               and row["rank"] == 1 for row in bookmarks)
    runtime._check_output_receipts = {"check-receipt:one": {
        "status": "FAIL", "revision": "current",
        "stdout": f'  File "{names[1]}", line 5, in target\nAssertionError\n',
    }}
    assert collect(runtime, frame) == bookmarks


def test_measured_source_fitter_keeps_failure_units_under_ten_file_pressure(
        tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    root, runtime, frame, names = _workspace(tmp_path)
    bookmarks = collect(runtime, frame)
    prioritized = windows(root, bookmarks)
    ordinary = []
    for name in names[3:]:
        rendered = _current_window(root, name, 1, 8, _revision(root / name))
        assert rendered is not None
        ordinary.append({"path": name, "start_line": 1, "end_line": 8,
                         "content": rendered[0], "source_chars": len(rendered[1])})
    base = {"messages": [{"role": "system", "content": "rules"},
                         {"role": "user", "content": "fix"}], "tools": []}
    minimum = fit_recent_read_windows(base, prioritized, fits=lambda _body: True)
    ceiling = token_estimate(minimum) + 12
    selected = fit_recent_read_windows(base, [*prioritized, *ordinary],
                                       fits=lambda body: token_estimate(body) <= ceiling)
    assert token_estimate(selected) <= ceiling
    frame["code_bookmarks"] = bookmarks
    reconcile_frame(frame, selected["messages"])
    deliveries = {row["path"]: row["body_delivery"] for row in frame["read_facts"]}
    assert all(deliveries[name] in {"present", "bookmarked"} for name in names[:3])
    assert any(deliveries[name] == "not_shown" for name in names[3:])


def test_capacity_shedding_reconciles_partial_delivery_truth(tmp_path, monkeypatch):
    from kernel_stateframe_capacity import fit_protected_frame, protected_bytes
    from kernel_stateframe_framefit5 import _serialize_state_message, _state_payload

    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    root, runtime, frame, names = _workspace(tmp_path)
    bookmark = collect(runtime, frame)[0]
    frame["code_bookmarks"] = [bookmark]
    frame["mode"] = "FRAME_ONLY"
    frame["read_facts"] = [frame["read_facts"][0]]
    frame["action_ledger"] = {"reads": [], "edits": []}
    source_window = windows(root, [bookmark])[0]["content"]
    reconcile_frame(frame, [{"role": "user", "content": source_window}])
    payload = {"state": {"framefit5": frame}}
    body = {"messages": [{"role": "user", "content": _serialize_state_message(payload)},
                         {"role": "user", "content": source_window}], "tools": []}
    without = copy.deepcopy(body)
    state_message = without["messages"][0]
    prior_payload, prior_state = _state_payload(state_message)
    prior_state["framefit5"].pop("code_bookmarks")
    reconcile_frame(prior_state["framefit5"], without["messages"])
    state_message["content"] = _serialize_state_message(prior_payload)
    ceiling = (protected_bytes(body) + protected_bytes(without)) // 2
    assert protected_bytes(without) < ceiling < protected_bytes(body)
    fit_protected_frame(body, max_request_bytes=ceiling)
    final = _state_payload(body["messages"][0])[1]["framefit5"]
    assert "code_bookmarks" not in final
    assert final["read_facts"][0]["body_delivery"] == "present"


def test_capacity_reduces_old_read_facts_before_current_bookmark(tmp_path, monkeypatch):
    from kernel_stateframe_capacity import fit_protected_frame, protected_bytes
    from kernel_stateframe_framefit5 import _serialize_state_message, _state_payload

    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    root, runtime, frame, _names = _workspace(tmp_path)
    bookmark = collect(runtime, frame)[0]
    frame["code_bookmarks"] = [bookmark]
    frame["mode"] = "FRAME_ONLY"
    frame["action_ledger"] = {"reads": [], "edits": []}
    bookmarked_fact = next(row for row in frame["read_facts"]
                           if row["path"] == bookmark["path"])
    frame["read_facts"].remove(bookmarked_fact)
    frame["read_facts"].append(bookmarked_fact)
    source_window = windows(root, [bookmark])[0]["content"]
    body = {"messages": [
        {"role": "user", "content": _serialize_state_message({"state": {"framefit5": frame}})},
        {"role": "user", "content": source_window},
    ], "tools": []}
    before = protected_bytes(body)
    fit_protected_frame(body, max_request_bytes=before - 150)
    final = _state_payload(body["messages"][0])[1]["framefit5"]
    assert final["code_bookmarks"][0]["path"] == bookmark["path"]
    assert len(final["read_facts"]) < 10


def test_bookmark_follows_shift_and_deleted_symbol_degrades(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    root, runtime, frame, names = _workspace(tmp_path)
    bookmark = collect(runtime, frame)[0]
    path = root / bookmark["path"]
    path.write_text("\n" * 40 + path.read_text())
    current = resolve(root, bookmark)
    assert current is not None and current["status"] == "current"
    assert current["start_line"] == bookmark["start_line"] + 40
    shown = windows(root, [bookmark])[0]["content"]
    assert "unit 44-45" in shown
    path.write_text("from math import sqrt\nFACTOR = 1\nclass Worker: pass\n")
    missing = resolve(root, bookmark)
    assert missing is not None and missing["status"] == "unresolved"
    assert "Worker" in missing["outline"]


def test_missing_asserted_symbol_is_dropped_before_slot_selection(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    root, runtime, frame, names = _workspace(tmp_path)
    test_path = root / "tests/test_flow.py"
    test_path.write_text(
        "from pkg.module_0 import missing\n"
        "def test_flow():\n"
        "    assert missing(1) == 2\n"
    )
    frame["read_facts"].append({"path": "tests/test_flow.py",
                                "read_revision": _revision(test_path),
                                "current_revision": _revision(test_path)})
    runtime._check_output_receipts["check-receipt:one"]["stdout"] = (
        "tests/test_flow.py:3: AssertionError\n")
    decision = {}
    bookmarks = collect(runtime, frame, decision=decision)
    assert not any(row["path"] == names[0] and row["symbol"] == "missing"
                   for row in bookmarks)
    assert {"path": names[0], "symbol": "missing", "reason": "unresolved_symbol"} in decision["drops"]


def test_body_and_decorator_edits_change_unit_digest(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    root, runtime, frame, names = _workspace(tmp_path)
    path = root / names[0]
    path.write_text(path.read_text().replace(
        "    def target(self, value):", "    @staticmethod\n    def target(self, value):"))
    runtime._check_output_receipts["check-receipt:one"]["stdout"] = (
        f'Traceback (most recent call last):\n  File "{names[0]}", line 6, in target\n'
        "AssertionError: wrong value\n")
    bookmark = collect(runtime, frame)[0]
    assert bookmark["start_line"] == 4
    old_digest = bookmark["unit_sha256"]
    path.write_text(path.read_text().replace("@staticmethod", "@classmethod"))
    current = resolve(root, bookmark)
    assert current is not None and current["status"] == "current"
    assert current["unit_sha256"] != old_digest
    assert "@classmethod" in windows(root, [bookmark])[0]["content"]


def test_same_size_external_change_is_revalidated_by_content(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    root, runtime, frame, _names = _workspace(tmp_path)
    bookmark = collect(runtime, frame)[0]
    path = root / bookmark["path"]
    before = path.stat()
    path.write_text(path.read_text().replace("FACTOR = 1", "FACTOR = 2"))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    current = resolve(root, bookmark)
    assert current is not None and current["revision"] != bookmark["revision"]
    assert current["unit_sha256"] == bookmark["unit_sha256"]
    assert "FACTOR = 2" in windows(root, [bookmark])[0]["content"]


def test_oversize_unit_is_honest_slice(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    root, runtime, frame, names = _workspace(tmp_path)
    path = root / names[0]
    path.write_text("def target(value):\n" + "    value += 1\n" * 500 + "    return value\n")
    runtime._check_output_receipts["check-receipt:one"]["stdout"] = (
        f'Traceback (most recent call last):\n  File "{names[0]}", line 300, in target\n'
        "AssertionError: wrong value\n")
    bookmarks = collect(runtime, frame)
    selected = windows(root, bookmarks)
    assert selected and selected[0]["tier"] == "slice"
    assert "tier=slice" in selected[0]["content"]
    assert "(502 lines)" in selected[0]["content"]
    assert verify_model_source_window(selected[0]["content"], path=names[0],
                                      source=path.read_text())


def test_large_unit_slice_anchor_follows_line_shift(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    root, runtime, frame, names = _workspace(tmp_path)
    path = root / names[0]
    path.write_text("def target(value):\n" + "    value += 1\n" * 500 + "    return value\n")
    runtime._check_output_receipts["check-receipt:one"]["stdout"] = (
        f'Traceback (most recent call last):\n  File "{names[0]}", line 300, in target\n'
        "AssertionError: wrong value\n")
    bookmark = collect(runtime, frame)[0]
    path.write_text("\n" * 40 + path.read_text())
    shown = windows(root, [bookmark])[0]["content"]
    assert "tier=slice lines=310-370" in shown


def test_pressure_compacts_only_oversize_rank_one_slice(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    root, runtime, frame, names = _workspace(tmp_path)
    path = root / names[0]
    path.write_text("def target(value):\n" + "    value += 1\n" * 500 + "    return value\n")
    frame["read_facts"][0]["read_revision"] = _revision(path)
    frame["read_facts"][0]["current_revision"] = _revision(path)
    runtime._check_output_receipts["check-receipt:one"]["stdout"] = (
        f'Traceback (most recent call last):\n  File "{names[0]}", line 250, in target\n'
        "AssertionError: wrong value\n")
    bookmark = collect(runtime, frame)[0]
    assert bookmark["tier"] == "slice"
    selected = windows(root, [bookmark])
    assert "compact_content" in selected[0]
    body = {"messages": [{"role": "system", "content": "rules"}], "tools": []}
    compact = dict(selected[0], content=selected[0]["compact_content"],
                   source_chars=selected[0]["compact_source_chars"])
    compact_body = fit_recent_read_windows(body, [compact], fits=lambda _body: True)
    ceiling = token_estimate(compact_body) + 4
    fitted = fit_recent_read_windows(body, selected,
                                     fits=lambda proposal: token_estimate(proposal) <= ceiling)
    assert selected[0]["compact_content"] in [row["content"] for row in fitted["messages"]]
    frame["code_bookmarks"] = [bookmark]
    reconcile_frame(frame, fitted["messages"])
    assert frame["read_facts"][0]["body_delivery"] == "bookmarked"


@pytest.mark.parametrize("off", [False, True])
def test_activation_boundary_and_kill_switch_are_byte_identical(tmp_path, monkeypatch, off):
    root, runtime, frame, names = _workspace(tmp_path)
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "off" if off else "on")
    if not off:
        frame["read_facts"] = frame["read_facts"][:3]
    assert collect(runtime, frame) == []


@pytest.mark.parametrize("case,summary,phase,outcome,reason", [
    ("no_assertion_line", "AssertionError: wrong value", "ASSERTION", "BEHAVIOR_FAIL",
     "no_failed_location"),
    ("other_runner", "custom runner failed without a source location", "OBSERVATION",
     "BEHAVIOR_FAIL", "no_failed_location"),
    ("collection", "pkg/module_0.py:5: failed during collection", "COLLECTION",
     "INVALID_CHECK", "non_behavior_failure"),
    ("stdout_only", "check failed; stdout omitted from projection", "OBSERVATION",
     "BEHAVIOR_FAIL", "no_failed_location"),
    ("stderr_only", "check failed; stderr omitted from projection", "OBSERVATION",
     "BEHAVIOR_FAIL", "no_failed_location"),
])
def test_unsupported_failure_evidence_keeps_bookmarks_inactive(
        tmp_path, monkeypatch, case, summary, phase, outcome, reason):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    _root, runtime, frame, names = _workspace(tmp_path)
    receipt_id = frame["last_verdict"]["check_receipt_id"]
    runtime._framefit_action_context["journal_receipt_records"] = [{
        "kind": "check_receipt", "receipt_id": receipt_id, "revision": "current",
        "status": "FAIL", "outcome": outcome, "failure_phase": phase,
        "summary": summary,
    }]
    if case == "stdout_only":
        runtime._check_output_receipts[receipt_id]["stdout"] = (
            f"{names[0]}:5: AssertionError: private stdout location")
    if case == "stderr_only":
        runtime._check_output_receipts[receipt_id]["stderr"] = (
            f"{names[0]}:5: AssertionError: private stderr location")
    decision = {}
    assert collect(runtime, frame, decision) == []
    assert decision["reason"] == reason


def test_unchanged_read_keeps_requested_bytes_and_current_failure(tmp_path):
    path = tmp_path / "module.py"
    path.write_text("VALUE = 3\n")
    measurement = SimpleNamespace(
        owner=SimpleNamespace(workspace_revision=1), current_product_revision="rev1",
        journal=SimpleNamespace(emit=lambda *_args, **_kwargs: None),
        retained=None, step_ledger=None, _code_bookmarks_active=True,
        current_request_id="request-one",
        _latest_check_receipt={"receipt_id": "one", "revision": "rev1",
                                "status": "FAIL", "assertion_context_preview": "assert value == 4"},
    )
    measurement.host_check_receipts = [measurement._latest_check_receipt]
    arguments = {"path": "module.py", "start_line": 1, "end_line": 1}
    first_delivery: dict[str, object] = {}
    first = execute_read(tmp_path, path, "module.py", "module.py", 0,
                         args=arguments, measurement=measurement, step=1,
                         delivery=first_delivery)
    second_delivery: dict[str, object] = {}
    second = execute_read(tmp_path, path, "module.py", "module.py", 0,
                          args=arguments, measurement=measurement, step=2,
                          delivery=second_delivery)
    assert first[0] == second[0] == "OK"
    assert "VALUE = 3" in first[1] and "VALUE = 3" in second[1]
    assert "unchanged since request request-one" in second[1]
    assert "current failing assertion: assert value == 4" in second[1]
    assert "next useful action" not in second[1]
    assert second_delivery["requested_bytes_delivered"] > 0
    measurement._latest_check_receipt["assertion_context_preview"] = "assert value == 5"
    changed = execute_read(tmp_path, path, "module.py", "module.py", 0,
                           args=arguments, measurement=measurement, step=3)
    assert "current failing assertion: assert value == 4" not in changed[1]


def test_same_assertion_with_changed_failure_phase_is_not_reused(tmp_path):
    from kernel_reread_result_truth import result_note

    measurement = SimpleNamespace(
        owner=SimpleNamespace(workspace_revision=1), current_product_revision="rev1",
        _code_bookmarks_active=True, current_request_id="request-one",
        _latest_check_receipt={"receipt_id": "one", "revision": "rev1",
                               "status": "FAIL", "failure_phase": "assertion",
                               "assertion_context_preview": "assert value == 4"},
    )
    measurement.host_check_receipts = [measurement._latest_check_receipt]
    assert result_note(measurement, "module.py", "generation", "sha256:abc", 1) is None
    measurement._latest_check_receipt["failure_phase"] = "collection"
    note = result_note(measurement, "module.py", "generation", "sha256:abc", 2)
    assert note is not None and "current failing assertion" not in note
    followup = result_note(measurement, "module.py", "generation", "sha256:abc", 3)
    assert followup is not None
    assert "unchanged since request request-one" in followup
    assert "current failing assertion" not in followup


@pytest.mark.parametrize("outcome", [None, "BEHAVIOR_FAIL"])
def test_bookmark_read_note_checkpoint_roundtrip_is_identical(outcome):
    from kernel_reread_result_truth import checkpoint_replay, restore_replay, result_note

    original = SimpleNamespace(
        owner=SimpleNamespace(workspace_revision=1), current_product_revision="rev1",
        _code_bookmarks_active=True, current_request_id="request-one",
        _latest_check_receipt={"receipt_id": "one", "revision": "rev1",
                               "status": "FAIL", "outcome": outcome,
                               "failure_phase": "OBSERVATION", "runner_tail": "1 failed",
                               "assertion_context_preview": "assert value == 4"},
    )
    original.host_check_receipts = [original._latest_check_receipt]
    assert result_note(original, "module.py", "generation", "sha256:abc", 1) is None
    snapshot = checkpoint_replay(original)
    resumed = SimpleNamespace(owner=SimpleNamespace(workspace_revision=1),
                              current_product_revision="rev1", current_request_id="request-two")
    restore_replay(resumed, snapshot)
    assert result_note(resumed, "module.py", "generation", "sha256:abc", 2) == result_note(
        original, "module.py", "generation", "sha256:abc", 2)
    malformed = copy.deepcopy(snapshot)
    malformed["seen"].append(malformed["seen"][0])
    with pytest.raises(ValueError, match="invalid bookmark read replay"):
        restore_replay(SimpleNamespace(), malformed)


@pytest.mark.parametrize("active", [False, True])
def test_three_execute_reads_match_parent_when_inactive_or_off(tmp_path, monkeypatch, active):
    import kernel_reread_result_truth as current_truth
    current_note = current_truth.result_note

    parent_source = subprocess.check_output([
        "git", "show", "a0ce931:src/stateframe/engine/kernel_reread_result_truth.py"
    ], cwd=Path(__file__).resolve().parents[1])
    parent_path = tmp_path / "parent_reread_result_truth.py"
    parent_path.write_bytes(parent_source)
    spec = importlib.util.spec_from_file_location("parent_reread_result_truth", parent_path)
    assert spec is not None and spec.loader is not None
    parent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parent)
    path = tmp_path / "module.py"
    path.write_text("VALUE = 3\n")
    args = {"path": "module.py", "start_line": 1, "end_line": 1}

    def three(note):
        def compatible_note(*args, **kwargs):
            return note(*args) if note is parent.result_note else note(*args, **kwargs)

        monkeypatch.setattr(current_truth, "result_note", compatible_note)
        measurement = SimpleNamespace(owner=SimpleNamespace(workspace_revision=1),
                                      _code_bookmarks_active=active, retained=None,
                                      step_ledger=None,
                                      journal=SimpleNamespace(emit=lambda *_args, **_kwargs: None))
        return [execute_read(tmp_path, path, "module.py", "module.py", 0,
                             args=args, measurement=measurement, step=step)
                for step in (1, 2, 3)]

    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "off" if active else "on")
    expected = three(parent.result_note)
    actual = three(current_note)
    assert actual == expected
    assert "unchanged since action 1" in actual[2][1]


def test_stale_or_unauthenticated_failure_never_names_current_assertion(tmp_path):
    path = tmp_path / "module.py"
    path.write_text("VALUE = 3\n")
    receipt = {"receipt_id": "check-one", "revision": "rev1", "status": "FAIL",
               "assertion_context_preview": "assert value == 4"}
    measurement = SimpleNamespace(owner=SimpleNamespace(workspace_revision=1),
                                  current_product_revision="rev1",
                                  current_request_id="request-one",
                                  _code_bookmarks_active=True,
                                  _latest_check_receipt=receipt,
                                  host_check_receipts=[], retained=None, step_ledger=None,
                                  journal=SimpleNamespace(emit=lambda *_args, **_kwargs: None))
    args = {"path": "module.py", "start_line": 1, "end_line": 1}
    def read(step):
        return execute_read(tmp_path, path, "module.py", "module.py", 0,
                            args=args, measurement=measurement, step=step)[1]

    assert "VALUE = 3" in read(1)
    assert "current failing assertion" not in read(2)
    measurement.host_check_receipts = [receipt]
    assert "current failing assertion" not in read(3)
    assert "current failing assertion" in read(4)
    measurement.current_product_revision = "rev2"
    measurement.owner.workspace_revision = 2
    assert "current failing assertion" not in read(5)
    note = read(6)
    assert "current failing assertion" not in note and "next useful action" not in note


def test_replay_normalizes_before_bounding_and_never_replaces_check_receipt():
    from kernel_reread_result_truth import checkpoint_replay, restore_replay, result_note

    receipt = {"receipt_id": "one", "revision": "rev1", "status": "FAIL",
               "assertion_context_preview": "assert" + " \n" * 600 + "value == 4"}
    original = SimpleNamespace(owner=SimpleNamespace(workspace_revision=1),
                               current_product_revision="rev1", current_request_id="one",
                               _code_bookmarks_active=True, _latest_check_receipt=receipt,
                               host_check_receipts=[receipt], _code_bookmark_carry=[])
    assert result_note(original, "module.py", "1", "sha256:abc", 1) is None
    snapshot = checkpoint_replay(original)
    sentinel = {"receipt_id": "canonical", "revision": "rev1", "status": "PASS"}
    resumed = SimpleNamespace(owner=SimpleNamespace(workspace_revision=1),
                              current_product_revision="rev1", current_request_id="one",
                              _latest_check_receipt=sentinel)
    restore_replay(resumed, snapshot)
    assert resumed._latest_check_receipt is sentinel
    # A newer canonical PASS makes the replayed failure historical.
    assert "current failing assertion" not in result_note(
        resumed, "module.py", "1", "sha256:abc", 2)
    del resumed._latest_check_receipt
    assert result_note(resumed, "module.py", "1", "sha256:abc", 3) == result_note(
        original, "module.py", "1", "sha256:abc", 3)


def test_assertion_truncation_boundary_preserves_replay_and_next_read(tmp_path):
    from kernel_reread_result_truth import checkpoint_replay, restore_replay

    path = tmp_path / "module.py"
    path.write_text("VALUE = 3\n")
    receipt = {"receipt_id": "one", "revision": "rev1", "status": "FAIL",
               "assertion_context_preview": "a" * 199 + " b"}
    original = SimpleNamespace(
        owner=SimpleNamespace(workspace_revision=1), current_product_revision="rev1",
        current_request_id="request-one", _code_bookmarks_active=True,
        _code_bookmark_carry=[{"path": "module.py", "symbol": "target",
                               "read_only": False}],
        _latest_check_receipt=receipt, host_check_receipts=[receipt],
        retained=None, step_ledger=None,
        journal=SimpleNamespace(emit=lambda *_args, **_kwargs: None),
    )
    args = {"path": "module.py", "start_line": 1, "end_line": 1}
    assert execute_read(tmp_path, path, "module.py", "module.py", 0,
                        args=args, measurement=original, step=1)[0] == "OK"
    snapshot = checkpoint_replay(original)
    assert snapshot["seen"] and snapshot["bookmarks"]
    assert snapshot["failure"]["assertion"] == "a" * 199
    resumed = SimpleNamespace(
        owner=SimpleNamespace(workspace_revision=1), current_product_revision="rev1",
        current_request_id="request-two", retained=None, step_ledger=None,
        journal=SimpleNamespace(emit=lambda *_args, **_kwargs: None),
    )
    restore_replay(resumed, snapshot)
    assert resumed._code_bookmark_carry == snapshot["bookmarks"]
    original.current_request_id = "request-two"
    assert execute_read(tmp_path, path, "module.py", "module.py", 0,
                        args=args, measurement=resumed, step=2) == execute_read(
                            tmp_path, path, "module.py", "module.py", 0,
                            args=args, measurement=original, step=2)


def test_long_symbol_is_omitted_before_checkpoint(tmp_path, monkeypatch):
    from kernel_reread_result_truth import checkpoint_replay, restore_replay

    root, runtime, frame, names = _workspace(tmp_path)
    symbol = "a" * 257
    (root / names[0]).write_text(f"def {symbol}(): return 1\n")
    runtime._check_output_receipts["check-receipt:one"]["stdout"] = (
        f'Traceback (most recent call last):\n  File "{names[0]}", line 1, in {symbol}\n')
    bookmarks = collect(runtime, frame)
    assert all(row["path"] != names[0] for row in bookmarks)
    runtime._code_bookmarks_active = True
    runtime._code_bookmark_carry = bookmarks
    restore_replay(SimpleNamespace(), checkpoint_replay(runtime))


def test_bookmark_source_rejects_swapped_parent_and_fifo(tmp_path, monkeypatch):
    import kernel_code_bookmarks

    root = tmp_path / "work"
    outside = tmp_path / "outside"
    (root / "pkg").mkdir(parents=True)
    outside.mkdir()
    (root / "pkg/source.py").write_text("SAFE = 1\n")
    (outside / "source.py").write_text("SECRET = 1\n")
    native_open = os.open
    swapped = False

    def replace_parent(path, flags, *args, **kwargs):
        nonlocal swapped
        if path == "pkg" and not swapped:
            swapped = True
            (root / "pkg").rename(root / "old_pkg")
            (root / "pkg").symlink_to(outside, target_is_directory=True)
        return native_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(kernel_code_bookmarks.os, "open", replace_parent)
    assert _source(root, "pkg/source.py") is None
    monkeypatch.setattr(kernel_code_bookmarks.os, "open", native_open)
    (root / "pkg").unlink()
    (root / "pkg").mkdir()
    os.mkfifo(root / "pkg/source.py")
    assert _source(root, "pkg/source.py") is None


def test_outline_recursion_drops_unresolved_bookmarks(tmp_path, monkeypatch):
    import kernel_paged_source

    root, runtime, frame, _names = _workspace(tmp_path)
    def fail(_source_text):
        raise RecursionError("pathological syntax tree")
    monkeypatch.setattr(kernel_paged_source, "symbol_outline", fail)
    decision = {}
    bookmarks = collect(runtime, frame, decision=decision)
    assert not bookmarks
    assert decision["drops"]
    assert all(row["reason"] == "unresolved_symbol" for row in decision["drops"])


def test_file_move_does_not_guess_bookmark_identity(tmp_path):
    root, runtime, frame, names = _workspace(tmp_path)
    prior = collect(runtime, frame)
    runtime._code_bookmark_carry = prior
    runtime._code_bookmarks_active = True
    moved = "pkg/renamed.py"
    (root / names[0]).rename(root / moved)
    frame["read_facts"][0]["path"] = moved
    frame["read_facts"][0]["current_revision"] = _revision(root / moved)
    frame["last_verdict"]["current_revision"] = "after-move"
    runtime.edit_contract["edit_authority"]["editable_paths"].append(moved)
    current = collect(runtime, frame)
    assert all(row["path"] != moved or row["origin"] != "carried" for row in current)


def test_disjoint_ordinary_window_survives_bookmark_in_same_revision(tmp_path):
    path = tmp_path / "module.py"
    path.write_text("def target():\n    return 1\nOTHER = 2\n")
    revision = _revision(path)
    first = _current_window(tmp_path, "module.py", 1, 2, revision)
    third = _current_window(tmp_path, "module.py", 3, 3, revision)
    assert first is not None and third is not None
    bookmark = {"path": "module.py", "content": first[0], "start_line": 1,
                "end_line": 2, "source_chars": len(first[1]), "bookmark": True, "rank": 1}
    ordinary = {"path": "module.py", "content": third[0], "start_line": 3,
                "end_line": 3, "source_chars": len(third[1])}
    merged = [bookmark, ordinary]
    fitted = fit_recent_read_windows(
        {"messages": [{"role": "system", "content": "rules"}], "tools": []}, merged,
        fits=lambda _body: True)
    assert {bookmark["content"], ordinary["content"]} <= {
        row["content"] for row in fitted["messages"]}


def test_whole_file_bookmark_falls_back_to_window_under_pressure(tmp_path):
    path = tmp_path / "module.py"
    path.write_text("def target(): return 1\n" +
                    "".join(f"FILLER_{index} = {index}\n" for index in range(38)) +
                    "OTHER = 2\n")
    revision = _revision(path)
    bookmark = windows(tmp_path, [{"path": "module.py", "symbol": "target",
                                   "rank": 1, "revision": revision}])[0]
    disjoint = _current_window(tmp_path, "module.py", 40, 40, revision)
    assert disjoint is not None
    assert "compact_content" in bookmark
    ordinary_disjoint = {"path": "module.py", "content": disjoint[0],
                        "start_line": 40, "end_line": 40, "source_chars": len(disjoint[1])}
    base = {"messages": [{"role": "system", "content": "rules"}], "tools": []}
    compact = {**bookmark, "content": bookmark["compact_content"],
               "source_chars": bookmark["compact_source_chars"]}
    expected = fit_recent_read_windows(base, [compact, ordinary_disjoint],
                                       fits=lambda _body: True)
    ceiling = token_estimate(expected)
    fitted = fit_recent_read_windows(
        base, [bookmark, ordinary_disjoint],
        fits=lambda body: token_estimate(body) <= ceiling)
    contents = [row["content"] for row in fitted["messages"]]
    assert token_estimate(fitted) <= ceiling
    assert bookmark["compact_content"] in contents
    assert bookmark["content"] not in contents
    assert ordinary_disjoint["content"] in contents
    assert (f"kept 2 selected source windows; about "
            f"{bookmark['compact_source_chars'] + ordinary_disjoint['source_chars']} "
            "chars of source fit") in contents


@pytest.mark.parametrize("active", [True, False])
def test_worker_checkpoint_writes_restorable_bookmark_section(monkeypatch, active):
    import kernel_continuation_restart as restart
    import kernel_frame_series_replay
    import kernel_native_restart
    import kernel_working_notes_restart
    from kernel_reread_result_truth import restore_replay, result_note

    receipt = {"receipt_id": "one", "revision": "rev1", "status": "FAIL",
               "assertion_context_preview": "assert" + " \n" * 600 + "value == 4"}
    events = []
    runtime = SimpleNamespace(
        owner=SimpleNamespace(workspace_revision=1, current_step=1),
        current_product_revision="rev1", current_request_id="request-one",
        _code_bookmarks_active=active, _code_bookmark_carry=[],
        _latest_check_receipt=receipt, host_check_receipts=[receipt],
        step_ledger=SimpleNamespace(pending=None), live_projection=None,
        frame_series=None, decision_state=None, native_batch=None,
        generated=None, retained=None, actions=SimpleNamespace(actions=[]),
        _trimmed=set(), binding=lambda: {"product_revision": "rev1"},
        journal=SimpleNamespace(emit=lambda kind, **fields: events.append(
            {"kind": kind, **fields}) or events[-1]),
    )
    assert result_note(runtime, "module.py", "1", "sha256:abc", 1) is None
    monkeypatch.setenv("KERNEL_CONTINUATION_RESTART", "1")
    monkeypatch.setattr(restart, "_valid_worker_state_fields", lambda _value: True)
    monkeypatch.setattr(restart, "runtime_fields", lambda _value: ())
    monkeypatch.setattr(restart, "_checkpoint_ledger", lambda _value: {})
    monkeypatch.setattr(restart, "checkpoint_conversation", lambda _value: {})
    monkeypatch.setattr(restart, "capture_owner_restore", lambda _value: {})
    monkeypatch.setattr(restart, "_register_checkpoint", lambda *_args: None)
    for name in ("first_green_checkpoint_section", "post_green_checkpoint_section",
                 "resume_work_checkpoint_section"):
        monkeypatch.setattr(restart, name, lambda _value: None)
    monkeypatch.setattr(kernel_native_restart, "snapshot", lambda _value: None)
    monkeypatch.setattr(kernel_frame_series_replay, "snapshot_series", lambda _value: None)
    monkeypatch.setattr(kernel_working_notes_restart, "snapshot", lambda _value: None)
    restart.checkpoint(runtime, [], {})
    section = events[-1]["checkpoint"]["code_bookmark_replay"]
    assert section["active"] is active
    resumed = SimpleNamespace(owner=SimpleNamespace(workspace_revision=1),
                              current_product_revision="rev1", current_request_id="request-one")
    restore_replay(resumed, section)
    assert result_note(resumed, "module.py", "1", "sha256:abc", 2) == result_note(
        runtime, "module.py", "1", "sha256:abc", 2)


def test_current_multi_selector_failure_rebuilds_t2_bookmarks(tmp_path, monkeypatch):
    from kernel_framefit_receipt_history import ReceiptHistory

    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    root, runtime, frame, _names = _workspace(tmp_path)
    product = {
        "pkg/service.py": "def run_batch(items, work):\n    return []\n",
        "pkg/worker.py": ("class Worker:\n"
                          "    def process_batch(self, items, work):\n"
                          "        return []\n"),
        "pkg/registry.py": "def lookup(name):\n    return None\n",
    }
    tests = {
        "tests/test_consumers.py": (
            "from pkg import service, worker\n"
            "def test_consumers():\n"
            "    assert service.run_batch([1], str) == []\n"
            "    assert worker.Worker().process_batch([2], str) == []\n"
            "    assert calls == [(1,), (2,)]\n"),
        "tests/test_registration_serialization.py": (
            "from pkg import registry\n"
            "def test_registry():\n"
            "    assert registry.lookup('batch') is not None\n"),
    }
    for name, source in {**product, **tests}.items():
        path = root / name
        path.write_text(source)
        frame["read_facts"].append({"path": name, "read_revision": _revision(path),
                                    "current_revision": _revision(path),
                                    "body_delivery": "not_shown" if name in product else "present"})
    runtime.edit_contract["edit_authority"]["editable_paths"].extend(product)
    history = ReceiptHistory()
    failures = [
        ("tests/test_consumers.py", "consumer", 5),
        ("tests/test_registration_serialization.py", "registry", 3),
    ]
    for selector, identity, line in failures:
        history.observe({"kind": "check_receipt", "selector": selector,
                         "receipt_id": f"check-receipt:{identity}", "revision": "current",
                         "status": "FAIL",
                         "summary": f"CHECK_FAILURE_SUMMARY\\n{selector}:{line}: AssertionError"})
    runtime._check_output_receipts = {}
    runtime._framefit_action_context = history.context()
    frame["last_verdict"]["check_receipt_id"] = "check-receipt:registry"
    decision = {}
    bookmarks = collect(runtime, frame, decision=decision)
    symbols = {(row["path"], row["symbol"]) for row in bookmarks}
    assert decision == {"reason": "activated"}
    assert {("pkg/registry.py", "lookup"), ("pkg/service.py", "run_batch"),
            ("pkg/worker.py", "Worker.process_batch")} <= symbols
    assert len(bookmarks) == 6
    history.observe({"kind": "check_receipt", "selector": "tests/test_consumers.py",
                     "receipt_id": "check-receipt:consumer-green", "revision": "current",
                     "status": "PASS"})
    runtime._framefit_action_context = history.context()
    superseded = {(row["path"], row["symbol"]) for row in collect(runtime, frame)}
    assert ("pkg/registry.py", "lookup") in superseded
    assert ("pkg/service.py", "run_batch") not in superseded
    assert ("pkg/worker.py", "Worker.process_batch") not in superseded
    frame["last_verdict"]["current_revision"] = "later"
    assert collect(runtime, frame, decision=decision) == []
    assert decision == {"reason": "stale_revision"}


def test_ten_file_batch_reuses_only_unchanged_delivered_default_pages(
        tmp_path, monkeypatch):
    from kernel_read_exec import record_serialized_read_pages

    monkeypatch.setenv("STATEFRAME_DELIVER_REQUESTED_BYTES", "on")
    files = [tmp_path / f"module_{index}.py" for index in range(10)]
    for index, path in enumerate(files):
        path.write_text(f"VALUE = {index}\n")
    measurement = SimpleNamespace(
        owner=SimpleNamespace(workspace_revision=1), retained=None,
        step_ledger=None, _code_bookmarks_active=False,
        _frame_budget_request_count=13,
        journal=SimpleNamespace(emit=lambda *_args, **_kwargs: None),
    )
    previous = []
    for index, path in enumerate(files):
        status, observation, _ = execute_read(
            tmp_path, path, path.name, path.name, 0,
            args={"path": path.name}, measurement=measurement, step=index + 1,
            visible_messages=[], delivery={})
        assert status == "OK" and f"VALUE = {index}" in observation
        previous.append({"role": "tool", "content": observation})
    for index in range(4):
        rendered = _current_window(tmp_path, files[index].name, 1, 1,
                                   _revision(files[index]))
        assert rendered is not None
        previous[index] = {"role": "user", "content": rendered[0]}
    record_serialized_read_pages(measurement, previous, 13)
    measurement._frame_budget_request_count = 14
    for index in (8, 9):
        files[index].write_text(f"VALUE = {index + 100}\n")
    notices = 0
    bodies = 0
    for index, path in enumerate(files):
        delivery = {}
        status, observation, _ = execute_read(
            tmp_path, path, path.name, path.name, 0,
            args={"path": path.name}, measurement=measurement, step=index + 11,
            visible_messages=previous, delivery=delivery)
        assert status == "OK"
        if index < 8:
            notices += 1
            assert "unchanged since request 13" in observation
            assert "VALUE = " not in observation
            assert delivery["requested_bytes_delivered"] == 0
        else:
            bodies += 1
            assert f"VALUE = {index + 100}" in observation
            assert delivery["requested_bytes_delivered"] > 0
    assert (notices, bodies) == (8, 2)
    status, observation, _ = execute_read(
        tmp_path, files[0], files[0].name, files[0].name, 0,
        args={"path": files[0].name, "start_line": 1, "end_line": 1},
        measurement=measurement, step=21, visible_messages=previous, delivery={})
    assert status == "OK" and "VALUE = 0" in observation
