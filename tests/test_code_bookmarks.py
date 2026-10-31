"""Code locations survive line shifts and respect final source delivery."""

from __future__ import annotations

import hashlib
import os
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernel_code_bookmarks import collect, resolve, windows
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
        _check_output_receipts={receipt_id: {"status": "FAIL", "stdout": failure}},
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
        assert "tier=full" in content
        assert verify_model_source_window(content, path=row["path"],
                                          source=(root / row["path"]).read_text())
    reconcile_frame(frame, [{"role": "user", "content": row["content"]}
                            for row in selected])
    deliveries = {row["path"]: row["body_delivery"] for row in frame["read_facts"]}
    assert all(deliveries[name] == "bookmarked" for name in names[:3])
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
    assert any("lines=11-12" in row["content"] for row in windows(root, shifted)
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
    assert final["read_facts"][0]["body_delivery"] == "not_shown"


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
    assert "lines=44-45" in shown
    path.write_text("from math import sqrt\nFACTOR = 1\nclass Worker: pass\n")
    missing = resolve(root, bookmark)
    assert missing is not None and missing["status"] == "unresolved"
    assert "Worker" in missing["outline"]


def test_missing_asserted_symbol_renders_current_outline(tmp_path, monkeypatch):
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
    bookmark = next(row for row in collect(runtime, frame)
                    if row["path"] == names[0] and row["symbol"] == "missing")
    shown = windows(root, [bookmark])[0]["content"]
    assert "missing not resolved in current source" in shown
    assert "Worker.target" in shown


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


def test_unchanged_read_keeps_requested_bytes_and_current_failure(tmp_path):
    path = tmp_path / "module.py"
    path.write_text("VALUE = 3\n")
    measurement = SimpleNamespace(
        owner=SimpleNamespace(workspace_revision=1),
        journal=SimpleNamespace(emit=lambda *_args, **_kwargs: None),
        retained=None, step_ledger=None, _code_bookmarks_active=True,
        current_request_id="request-one",
        _latest_check_receipt={"receipt_id": "one", "revision": "rev1",
                               "status": "FAIL", "assertion_context_preview": "assert value == 4"},
    )
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
    assert "next useful action is an edit or a test run" in second[1]
    assert second_delivery["requested_bytes_delivered"] > 0
    measurement._latest_check_receipt["assertion_context_preview"] = "assert value == 5"
    changed = execute_read(tmp_path, path, "module.py", "module.py", 0,
                           args=arguments, measurement=measurement, step=3)
    assert "current failing assertion: assert value == 4" not in changed[1]


def test_same_assertion_with_changed_failure_phase_is_not_reused(tmp_path):
    from kernel_reread_result_truth import result_note

    measurement = SimpleNamespace(
        owner=SimpleNamespace(workspace_revision=1),
        _code_bookmarks_active=True, current_request_id="request-one",
        _latest_check_receipt={"receipt_id": "one", "revision": "rev1",
                               "status": "FAIL", "failure_phase": "assertion",
                               "assertion_context_preview": "assert value == 4"},
    )
    assert result_note(measurement, "module.py", "generation", "sha256:abc", 1) is None
    measurement._latest_check_receipt["failure_phase"] = "collection"
    note = result_note(measurement, "module.py", "generation", "sha256:abc", 2)
    assert note is not None and "current failing assertion" not in note
    followup = result_note(measurement, "module.py", "generation", "sha256:abc", 3)
    assert followup is not None
    assert "unchanged since request request-one" in followup
    assert "current failing assertion: assert value == 4" in followup


@pytest.mark.parametrize("outcome", [None, "BEHAVIOR_FAIL"])
def test_bookmark_read_note_checkpoint_roundtrip_is_identical(outcome):
    from kernel_reread_result_truth import checkpoint_replay, restore_replay, result_note

    original = SimpleNamespace(
        owner=SimpleNamespace(workspace_revision=1),
        _code_bookmarks_active=True, current_request_id="request-one",
        _latest_check_receipt={"receipt_id": "one", "revision": "rev1",
                               "status": "FAIL", "outcome": outcome,
                               "failure_phase": "OBSERVATION", "runner_tail": "1 failed",
                               "assertion_context_preview": "assert value == 4"},
    )
    assert result_note(original, "module.py", "generation", "sha256:abc", 1) is None
    snapshot = checkpoint_replay(original)
    resumed = SimpleNamespace(owner=SimpleNamespace(workspace_revision=1),
                              current_request_id="request-two")
    restore_replay(resumed, snapshot)
    assert result_note(resumed, "module.py", "generation", "sha256:abc", 2) == result_note(
        original, "module.py", "generation", "sha256:abc", 2)
    malformed = copy.deepcopy(snapshot)
    malformed["seen"].append(malformed["seen"][0])
    with pytest.raises(ValueError, match="invalid bookmark read replay"):
        restore_replay(SimpleNamespace(), malformed)
