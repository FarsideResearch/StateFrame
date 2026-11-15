"""Parallel result receipts preserve the decisions hidden by F4's opaque-action fold."""
from __future__ import annotations

import hashlib
import copy
import errno
import json
import os
import signal
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import kernel_parallel
from kernel_stateframe_action_ledger import build_action_context
from stateframe.engine.kernel_parallel_records import (
    AdmissionResult, CheckRecipe, ConsumeSpec, ParallelLimits, ParallelMeasured,
    ParallelRequest, ParallelResult, Refusal, UnitContract, UnitOutcome, UnitUsage,
)
from stateframe.engine.kernel_stateframe_framefit5 import (
    FrameFit5Error, _opaque_actions, _validated_parallel_disposition, build_typed_state,
)


def _case(tmp_path, statuses, *, refusal=None, long_paths=False):
    limits = ParallelLimits(max_units=max(6, len(statuses)))
    units = tuple(UnitContract(
        name, "Change the named module", (f"pkg/{name}{'x' * 50 if long_paths else ''}.py",),
        (), (), (ConsumeSpec("shared" if index else "base", None, None),), (), (),
        CheckRecipe((f"tests/test_{name}.py",), (f"tests/test_{name}.py",), "pytest"),
    ) for index, (name, _) in enumerate(statuses))
    request = ParallelRequest("call_0", units, "2")
    admission = AdmissionResult("REFUSED" if refusal else "ADMITTED", refusal, (), (),
                                tuple(name for name, _ in statuses), 0, "manifest", limits)
    usage = UnitUsage(1, 1, 10, 10, 100)
    outcomes = tuple(UnitOutcome(
        name, status, "PRODUCER_FAILED" if status == "CANCELLED" else
        "COMPOSITION_CHECK_FAILED" if status == "REFUSED" else None,
        "producer alpha failed" if status == "CANCELLED" else
        "assertion failed in tests/test_alpha.py" if status == "REFUSED" else None,
        index if status.startswith("COMMITTED") else None, (), True,
        "NOT_RUN" if status == "COMMITTED_UNVERIFIED" else "PASS", usage, 1.0, 0.0,
    ) for index, (name, status) in enumerate(statuses, 1))
    observation = ("REFUSED INVALID_ARGUMENTS: subframe alpha check: ambiguous tests; "
                   'fix: resend {"check": {"tests": ["tests/test_alpha.py"]}}'
                   if refusal else "Subframe results:\n" + "\n".join(
                       f"Subframe {name}: {status}" for name, status in statuses))
    result = ParallelResult("call_0", admission, outcomes,
                            tuple(item.txn_id for item in outcomes if item.txn_id is not None),
                            "PASS", (), "VERIFIED",
                            observation, ParallelMeasured(1.0, 2.0, 2.0, 2, 0.0))
    root = tmp_path / "parallel" / "call_0"
    root.mkdir(parents=True)
    (root / "result.json").write_text(json.dumps(result.to_json()))
    (root / "intent.json").write_text(json.dumps({
        "schema": "stateframe-parallel-intent/1", "call_id": "call_0",
        "request": request.to_json(), "admission": admission.to_json(),
    }))
    assistant = {"role": "assistant", "content": "", "tool_calls": [{
        "id": "call_0", "type": "function", "function": {
            "name": "parallel", "arguments": json.dumps({"subframes": []}),
        },
    }]}
    reply = {"role": "tool", "tool_call_id": "call_0", "content": observation}
    return result, ((assistant, reply),)


def _disposition(tmp_path, monkeypatch, statuses, **kwargs):
    result, turns = _case(tmp_path, statuses, **kwargs)
    monkeypatch.setattr(kernel_parallel, "_service", (tmp_path, {"parallel_mode": "2"}))
    rows, ids = _opaque_actions(turns)
    assert ids == {"call_0"}
    row, = rows
    assert row["kind"] == "parallel_disposition"
    assert row["call_id"] == result.call_id
    summary = row["sections"]["disposition"]
    assert summary["result_summary"]
    return result, row, summary


def _receipt(tmp_path, monkeypatch, statuses, **kwargs):
    result, item, summary = _disposition(tmp_path, monkeypatch, statuses, **kwargs)
    summary = dict(summary)
    summary.update({
        "schema": item["schema"],
        "kind": item["kind"],
        "outcome": ("REJECTED" if result.admission.status == "REFUSED"
                    or not result.txn_order else "APPLIED"),
    })
    return summary


def _result_summary(row):
    if "sections" in row:
        return row["sections"]["disposition"]["result_summary"]
    return row["result_summary"]


def _omitted_committed(row):
    if "sections" in row:
        return row["sections"]["disposition"]["omitted_committed"]
    return row["omitted_committed"]


@contextmanager
def _bounded_receipt_call(seconds=3):
    def timed_out(_signum, _frame):
        raise TimeoutError("subframe receipt cache open blocked")

    previous = signal.signal(signal.SIGALRM, timed_out)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def test_all_committed_has_every_expanded_plan_and_status(tmp_path, monkeypatch):
    row = _receipt(tmp_path, monkeypatch, (("alpha", "COMMITTED"), ("beta", "COMMITTED")))
    assert row["schema"] == kernel_parallel.PARALLEL_DISPOSITION_SCHEMA
    assert row["kind"] == "parallel_disposition"
    assert row["outcome"] == "APPLIED"
    assert "Plan alpha: owns=pkg/alpha.py; check=tests/test_alpha.py; consumes=base; status=COMMITTED" in _result_summary(row)
    assert "Plan beta: owns=pkg/beta.py; check=tests/test_beta.py; consumes=shared; status=COMMITTED" in _result_summary(row)
    assert "all committed" in _result_summary(row)
    assert not (tmp_path / "parallel" / "call_0" / "receipt").exists()


def test_failure_and_cancelled_consumer_precede_committed(tmp_path, monkeypatch):
    row = _receipt(tmp_path, monkeypatch, (("ok", "COMMITTED"), ("alpha", "REFUSED"),
                                           ("consumer", "CANCELLED")))
    summary = _result_summary(row)
    assert summary.index("Plan alpha:") < summary.index("Plan ok:")
    assert summary.index("Plan consumer:") < summary.index("Plan ok:")
    assert "COMPOSITION_CHECK_FAILED: assertion failed in tests/test_alpha.py" in summary
    assert "PRODUCER_FAILED: producer alpha failed" in summary
    assert "1 failed; 1 cancelled" in summary


def test_overflow_counts_committed_and_never_drops_failure(tmp_path, monkeypatch):
    statuses = (("failed", "REFUSED"),) + tuple((f"pass{i:03d}", "COMMITTED") for i in range(100))
    row = _receipt(tmp_path, monkeypatch, statuses, long_paths=True)
    summary = _result_summary(row)
    assert "Plan failed:" in summary
    assert "COMPOSITION_CHECK_FAILED" in summary
    assert "more committed" in summary
    assert _omitted_committed(row) > 0
    assert len(summary.encode()) <= 3072


def test_overflow_keeps_unverified_subframe_visible(tmp_path, monkeypatch):
    statuses = (("unchecked", "COMMITTED_UNVERIFIED"),) + tuple(
        (f"pass{i:03d}", "COMMITTED") for i in range(100))
    row = _receipt(tmp_path, monkeypatch, statuses, long_paths=True)
    assert "Plan unchecked:" in _result_summary(row)
    assert "status=COMMITTED_UNVERIFIED" in _result_summary(row)
    assert "check_status=NOT_RUN" in _result_summary(row)
    assert "1 unverified" in _result_summary(row)
    assert row["omitted_committed"] > 0


def test_refused_terse_plan_preserves_resend_fragment(tmp_path, monkeypatch):
    refusal = Refusal("INVALID_ARGUMENTS", "ambiguous tests", (), (), ())
    row = _receipt(tmp_path, monkeypatch, (("alpha", "REFUSED"),), refusal=refusal)
    assert 'fix: resend {"check": {"tests": ["tests/test_alpha.py"]}}' in _result_summary(row)
    assert row["outcome"] == "REJECTED"


def test_missing_durable_parallel_result_becomes_typed_refusal(tmp_path, monkeypatch):
    # A missing receipt is a refused exchange, journaled, not a crash; see test_parallel_schema_refusal_receipt.
    _, turns = _case(tmp_path, (("alpha", "COMMITTED"),))
    (tmp_path / "parallel" / "call_0" / "result.json").unlink()
    monkeypatch.setattr(kernel_parallel, "_service", (tmp_path, {"parallel_mode": "2"}))
    rows, ids = _opaque_actions(turns)
    assert ids == {"call_0"}
    assert len(rows) == 1
    assert rows[0]["outcome"] == "REJECTED"


def test_corrupt_durable_parallel_result_fails_closed(tmp_path, monkeypatch):
    _, turns = _case(tmp_path, (("alpha", "COMMITTED"),))
    (tmp_path / "parallel" / "call_0" / "result.json").write_text("{not json")
    monkeypatch.setattr(kernel_parallel, "_service", (tmp_path, {"parallel_mode": "2"}))
    with pytest.raises(FrameFit5Error, match="subframe result"):
        _opaque_actions(turns)


def test_subframe_child_omits_inherited_parent_parallel_exchange(tmp_path, monkeypatch):
    import kernel_unit_policy

    _, turns = _case(tmp_path, (("alpha", "COMMITTED"),))
    monkeypatch.setattr(kernel_parallel, "_service", None)
    monkeypatch.setattr(kernel_unit_policy, "active", lambda: object())
    task = {"role": "user", "content": "Task: continue the child work."}
    messages = [task, *turns[0]]
    action_context = build_action_context(messages, workspace_root=tmp_path)
    action_context["journal_receipt_records"] = []
    runtime = SimpleNamespace(
        owner=SimpleNamespace(root=tmp_path, task_row={}),
        binding=lambda **_kwargs: {"product_revision": "sha256:" + "a" * 64},
        edit_contract={"edit_authority": {"editable_paths": []}},
        protected_tests={}, frame_ceiling=100_000, micrograph_wire=None,
    )

    state = build_typed_state(messages, action_context, runtime)

    assert state["action_ledger"]["unclassified_actions"] == []
    assert "latest_parallel_disposition" not in state["action_ledger"]


def test_parent_without_parallel_result_service_fails_closed(monkeypatch, tmp_path):
    import kernel_unit_policy

    _, turns = _case(tmp_path, (("alpha", "COMMITTED"),))
    monkeypatch.setattr(kernel_parallel, "_service", None)
    monkeypatch.setattr(kernel_unit_policy, "active", lambda: None)

    with pytest.raises(FrameFit5Error, match="subframe result service is unavailable"):
        _opaque_actions(turns)


@pytest.mark.parametrize("mutation", [
    "schema", "summary", "summary_surrogate", "omitted_units", "kind", "call_id",
    "unhashable_status", "unhashable_contract", "unhashable_file_kind",
])
def test_parallel_disposition_schema_rejects_tampering(tmp_path, monkeypatch, mutation):
    _result, item, _section = _disposition(
        tmp_path, monkeypatch, (("alpha", "COMMITTED"),),
    )
    changed = copy.deepcopy(item)
    if mutation == "schema":
        changed["schema"] = "stateframe-parallel-disposition/2"
    elif mutation == "summary":
        changed["sections"]["disposition"]["result_summary"] = "x" * 3073
    elif mutation == "summary_surrogate":
        changed["sections"]["disposition"]["result_summary"] = "\ud800"
    elif mutation == "omitted_units":
        changed["truncation"]["omitted_units"] = -1
    elif mutation == "kind":
        changed["kind"] = "bounded_action_receipt"
    elif mutation == "unhashable_status":
        changed["sections"]["disposition"]["units"][0][1] = []
    elif mutation == "unhashable_contract":
        changed["sections"]["disposition"]["unit_details"][0][3] = []
    elif mutation == "unhashable_file_kind":
        changed["sections"]["disposition"]["unit_details"][0][2] = [
            ["pkg/alpha.py", [], 0, 0],
        ]
    else:
        changed["call_id"] = ""
    with pytest.raises(FrameFit5Error, match="disposition"):
        _validated_parallel_disposition(changed)


def test_too_many_failed_subframes_have_retrievable_pages(tmp_path, monkeypatch):
    statuses = tuple((f"bad{i:03d}", "REFUSED") for i in range(100))
    result, turns = _case(tmp_path, statuses, long_paths=True)
    monkeypatch.setattr(kernel_parallel, "_service", (tmp_path, {"parallel_mode": "2"}))
    row, = _opaque_actions(turns)[0]
    summary = _result_summary(row)
    assert "failed=100" in summary
    assert "read_check_receipt(handle=" in summary
    assert len(summary.encode()) <= 3072
    pages = sorted((tmp_path / "parallel" / result.call_id / "receipt").glob("page-*.json"))
    assert pages
    assert [item["id"] for page in pages for item in json.loads(page.read_bytes())["rows"]] == [
        name for name, _ in statuses]
    handle = summary.split('read_check_receipt(handle="', 1)[1].split('"', 1)[0]
    first_page = json.loads(kernel_parallel.read_subframe_receipt(handle))
    assert first_page["rows"][0]["id"] == "bad000"
    assert first_page["next_handle"].startswith("subframe:call_0:2:")
    pages[0].write_text('{"tampered":true}')
    restored = json.loads(kernel_parallel.read_subframe_receipt(handle))
    assert restored["rows"][0]["id"] == "bad000"
    assert list(pages[0].parent.glob("page-0001-*.json"))


def test_six_long_failures_keep_all_decisions_and_deduplicate_reason(tmp_path, monkeypatch):
    statuses = tuple((f"bad{i}", "REFUSED") for i in range(6))
    result, turns = _case(tmp_path, statuses)
    changed = tuple(item.__class__(
        item.unit_id, item.status, "COMPOSITION_CHECK_FAILED",
        "COMPOSITION_CHECK_FAILED: " + "x" * 300, item.txn_id, item.files,
        item.contract_ok, "FAIL", item.usage, item.wall_seconds, item.queue_wait_seconds,
    ) for item in result.outcomes)
    result = result.__class__(result.call_id, result.admission, changed, result.txn_order,
                              result.final_check, result.unknown_coupling, result.integration,
                              result.observation, result.measured)
    (tmp_path / "parallel" / result.call_id / "result.json").write_text(json.dumps(result.to_json()))
    monkeypatch.setattr(kernel_parallel, "_service", (tmp_path, {"parallel_mode": "2"}))
    row, = _opaque_actions(turns)[0]
    assert all(f"Plan bad{i}:" in _result_summary(row) for i in range(6))
    assert "COMPOSITION_CHECK_FAILED: COMPOSITION_CHECK_FAILED" not in _result_summary(row)
    assert len(_result_summary(row).encode()) <= 3072


def test_reason_renderer_does_not_repeat_identical_detail():
    assert kernel_parallel._render_reason("RUN_INCOMPLETE", "RUN_INCOMPLETE") == "RUN_INCOMPLETE"
    assert kernel_parallel._render_reason("RUN_INCOMPLETE", "RUN_INCOMPLETE: failed check") == (
        "RUN_INCOMPLETE: failed check")


def test_mandatory_rows_reserve_count_line_before_committed_details(tmp_path, monkeypatch):
    statuses = tuple((f"bad{i}", "REFUSED") for i in range(6)) + tuple(
        (f"ok{i:03d}", "COMMITTED") for i in range(100))
    result, turns = _case(tmp_path, statuses, long_paths=True)
    changed = tuple(item.__class__(
        item.unit_id, item.status, item.reason,
        "failure in owned check: " + "x" * 300 if item.status == "REFUSED" else item.detail,
        item.txn_id, item.files, item.contract_ok, item.check, item.usage,
        item.wall_seconds, item.queue_wait_seconds,
    ) for item in result.outcomes)
    result = result.__class__(result.call_id, result.admission, changed, result.txn_order,
                              result.final_check, result.unknown_coupling, result.integration,
                              result.observation, result.measured)
    (tmp_path / "parallel" / result.call_id / "result.json").write_text(json.dumps(result.to_json()))
    monkeypatch.setattr(kernel_parallel, "_service", (tmp_path, {"parallel_mode": "2"}))
    row, = _opaque_actions(turns)[0]
    summary = _result_summary(row)
    assert all(f"Plan bad{i}:" in summary for i in range(6))
    assert _omitted_committed(row) > 0
    assert f"{_omitted_committed(row)} more committed" in summary
    assert len(summary.encode()) <= 3072


def test_1024_failed_subframes_page_all_rows(tmp_path, monkeypatch):
    statuses = tuple((f"bad{i:04d}", "REFUSED") for i in range(1024))
    result, turns = _case(tmp_path, statuses)
    monkeypatch.setattr(kernel_parallel, "_service", (tmp_path, {"parallel_mode": "2"}))
    row, = _opaque_actions(turns)[0]
    assert "failed=1024" in _result_summary(row)
    assert "read_check_receipt(handle=" in _result_summary(row)
    pages = sorted((tmp_path / "parallel" / result.call_id / "receipt").glob("page-*.json"))
    assert len(pages) > 1
    assert len([entry for page in pages for entry in json.loads(page.read_bytes())["rows"]]) == 1024


def test_long_refusal_preserves_location_and_resend(tmp_path, monkeypatch):
    refusal = Refusal("INVALID_ARGUMENTS", "ambiguous tests", (), (), ())
    result, turns = _case(tmp_path, (("alpha", "REFUSED"),), refusal=refusal)
    long_observation = result.observation + "; evidence " + "x" * 4000
    result = result.__class__(result.call_id, result.admission, result.outcomes, result.txn_order,
                              result.final_check, result.unknown_coupling, result.integration,
                              long_observation, result.measured)
    (tmp_path / "parallel" / result.call_id / "result.json").write_text(json.dumps(result.to_json()))
    turns = ((turns[0][0], {**turns[0][1], "content": long_observation}),)
    monkeypatch.setattr(kernel_parallel, "_service", (tmp_path, {"parallel_mode": "2"}))
    row, = _opaque_actions(turns)[0]
    assert "INVALID_ARGUMENTS" in _result_summary(row)
    assert "fix: adjust the subframe contracts and retry" in _result_summary(row)
    assert "read_check_receipt(handle=" in _result_summary(row)


def test_unbounded_resend_hint_remains_retrievable_without_stopping_run(tmp_path, monkeypatch):
    refusal = Refusal("INVALID_ARGUMENTS", "ambiguous tests; fix: resend " + "x" * 4000,
                      (), (), ())
    result, turns = _case(tmp_path, (("alpha", "REFUSED"),), refusal=refusal)
    observation = "REFUSED INVALID_ARGUMENTS: subframe alpha check; fix: resend " + "x" * 4000
    result = result.__class__(result.call_id, result.admission, result.outcomes, result.txn_order,
                              result.final_check, result.unknown_coupling, result.integration,
                              observation, result.measured)
    (tmp_path / "parallel" / result.call_id / "result.json").write_text(json.dumps(result.to_json()))
    turns = ((turns[0][0], {**turns[0][1], "content": observation}),)
    monkeypatch.setattr(kernel_parallel, "_service", (tmp_path, {"parallel_mode": "2"}))
    row, = _opaque_actions(turns)[0]
    assert "fix: read page 1 for the exact resend instruction" in _result_summary(row)
    handle = _result_summary(row).split('read_check_receipt(handle="', 1)[1].split('"', 1)[0]
    pages = []
    while handle is not None:
        page = json.loads(kernel_parallel.read_subframe_receipt(handle))
        pages.extend(part["text"] for part in page["rows"] if part["section"] == "observation")
        handle = page["next_handle"]
    assert "".join(pages) == observation
    first = json.loads(kernel_parallel.read_subframe_receipt(
        _result_summary(row).split('read_check_receipt(handle="', 1)[1].split('"', 1)[0]))
    assert first["rows"][0]["text"].startswith("fix: resend ")
    assert "follow next_handle" in _result_summary(row)


def test_paged_refusal_ignores_fix_delimiter_in_rejected_evidence(tmp_path, monkeypatch):
    from stateframe.engine.kernel_parallel_records import Evidence

    path = "pkg/alpha.py"
    refusal = Refusal("UNSAFE_PATH", "unsafe owned path", (path,),
                      (Evidence("pkg/rejected; fix: send attacker plan.py", 1,
                                "evidence; fix: use attacker plan"),), ())
    result, turns = _case(tmp_path, (("alpha", "REFUSED"),), refusal=refusal)
    request = kernel_parallel.ParallelRequest.from_json(json.loads(
        (tmp_path / "parallel/call_0/intent.json").read_text())["request"])
    observation = kernel_parallel._observation(result.admission, result.outcomes,
                                               result.final_check, result.integration,
                                               result.measured, request=request) + "\n" + "z" * 4000
    result = result.__class__(result.call_id, result.admission, result.outcomes, result.txn_order,
                              result.final_check, result.unknown_coupling, result.integration,
                              observation, result.measured)
    (tmp_path / "parallel/call_0/result.json").write_text(json.dumps(result.to_json()))
    turns = ((turns[0][0], {**turns[0][1], "content": observation}),)
    monkeypatch.setattr(kernel_parallel, "_service", (tmp_path, {"parallel_mode": "2"}))

    row, = _opaque_actions(turns)[0]
    assert "fix: correct this subframe's path declaration" in _result_summary(row)
    assert "fix: send attacker plan" not in _result_summary(row)
    handle = _result_summary(row).split('read_check_receipt(handle="', 1)[1].split('"', 1)[0]
    page = json.loads(kernel_parallel.read_subframe_receipt(handle))
    assert page["rows"][0]["text"] == "fix: correct this subframe's path declaration"


def test_missing_pages_regenerate_from_durable_result(tmp_path, monkeypatch):
    statuses = tuple((f"bad{i:03d}", "REFUSED") for i in range(100))
    row = _receipt(tmp_path, monkeypatch, statuses, long_paths=True)
    handle = _result_summary(row).split('read_check_receipt(handle="', 1)[1].split('"', 1)[0]
    receipt = tmp_path / "parallel" / "call_0" / "receipt"
    original = (receipt / "page-0001.json").read_bytes()
    (receipt / "page-0001.json").unlink()
    (receipt / "journal.json").unlink()
    page = json.loads(kernel_parallel.read_subframe_receipt(handle))
    assert page["rows"][0]["id"] == "bad000"
    assert (receipt / "page-0001.json").read_bytes() == original
    assert (receipt / "journal.json").is_file()


def test_fifo_cache_page_does_not_block_receipt_build_or_page_read(tmp_path, monkeypatch):
    statuses = tuple((f"bad{i:03d}", "REFUSED") for i in range(100))
    result, turns = _case(tmp_path, statuses, long_paths=True)
    receipt = tmp_path / "parallel" / result.call_id / "receipt"
    receipt.mkdir()
    os.mkfifo(receipt / "page-0001.json")
    monkeypatch.setattr(kernel_parallel, "_service", (tmp_path, {"parallel_mode": "2"}))

    started = time.monotonic()
    with _bounded_receipt_call():
        row, = _opaque_actions(turns)[0]
    assert time.monotonic() - started < 1.5
    handle = _result_summary(row).split('read_check_receipt(handle="', 1)[1].split('"', 1)[0]
    started = time.monotonic()
    with _bounded_receipt_call():
        page = json.loads(kernel_parallel.read_subframe_receipt(handle))
    assert time.monotonic() - started < 1.5
    assert page["rows"][0]["id"] == "bad000"
    assert (receipt / "page-0001.json").is_fifo()
    assert list(receipt.glob("page-0001-*.json"))


def test_later_page_and_journal_replacement_cannot_change_chain(tmp_path, monkeypatch):
    statuses = tuple((f"bad{i:03d}", "REFUSED") for i in range(100))
    row = _receipt(tmp_path, monkeypatch, statuses, long_paths=True)
    handle = _result_summary(row).split('read_check_receipt(handle="', 1)[1].split('"', 1)[0]
    receipt = tmp_path / "parallel" / "call_0" / "receipt"
    first = (receipt / "page-0001.json").read_bytes()
    second_path = receipt / "page-0002.json"
    original_second = json.loads(second_path.read_bytes())
    replacement = json.loads(second_path.read_bytes())
    replacement["rows"][0]["id"] = "attacker"
    second_path.write_text(json.dumps(replacement))
    journal_path = receipt / "journal.json"
    journal = json.loads(journal_path.read_bytes())
    journal["pages"][1] = hashlib.sha256(second_path.read_bytes()).hexdigest()
    journal_path.write_text(json.dumps(journal))
    page_one = json.loads(kernel_parallel.read_subframe_receipt(handle))
    assert (receipt / "page-0001.json").read_bytes() == first
    page_two = json.loads(kernel_parallel.read_subframe_receipt(page_one["next_handle"]))
    assert page_two["rows"] == original_second["rows"]
    assert page_one["next_handle"].endswith(json.loads(first)["next_sha256"])


def test_receipt_ancestor_symlink_cannot_escape_run(tmp_path, monkeypatch):
    statuses = tuple((f"bad{i:03d}", "REFUSED") for i in range(100))
    result, turns = _case(tmp_path, statuses, long_paths=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    call = tmp_path / "parallel" / result.call_id
    target = outside / result.call_id
    os.rename(call, target)
    call.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(kernel_parallel, "_service", (tmp_path, {"parallel_mode": "2"}))
    with pytest.raises(FrameFit5Error, match="unsafe ancestor"):
        _opaque_actions(turns)
    assert not (target / "receipt").exists()


def test_receipt_read_rejects_ancestor_symlink(tmp_path, monkeypatch):
    statuses = tuple((f"bad{i:03d}", "REFUSED") for i in range(100))
    row = _receipt(tmp_path, monkeypatch, statuses, long_paths=True)
    handle = _result_summary(row).split('read_check_receipt(handle="', 1)[1].split('"', 1)[0]
    outside = tmp_path / "outside"
    outside.mkdir()
    call = tmp_path / "parallel" / "call_0"
    target = outside / "call_0"
    os.rename(call, target)
    call.symlink_to(target, target_is_directory=True)
    with pytest.raises(kernel_parallel.ParallelError, match="unsafe ancestor"):
        kernel_parallel.read_subframe_receipt(handle)
    assert (target / "receipt" / "page-0001.json").is_file()


def test_receipt_storage_outage_is_recoverable(tmp_path, monkeypatch):
    statuses = tuple((f"bad{i:03d}", "REFUSED") for i in range(100))
    _, turns = _case(tmp_path, statuses, long_paths=True)
    monkeypatch.setattr(kernel_parallel, "_service", (tmp_path, {"parallel_mode": "2"}))
    def fail_write(descriptor, name, data):
        raise OSError(errno.ENOSPC, "disk full")
    monkeypatch.setattr(kernel_parallel, "_write_receipt_file", fail_write)
    row, = _opaque_actions(turns)[0]
    handle = _result_summary(row).split('read_check_receipt(handle="', 1)[1].split('"', 1)[0]
    page = json.loads(kernel_parallel.read_subframe_receipt(handle))
    assert page["rows"][0]["id"] == "bad000"


def test_symlinked_receipt_cache_is_unavailable_but_derived_pages_work(tmp_path, monkeypatch):
    result, turns = _case(tmp_path, tuple((f"bad{i:03d}", "REFUSED") for i in range(100)),
                          long_paths=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    cache = tmp_path / "parallel" / result.call_id / "receipt"
    cache.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(kernel_parallel, "_service", (tmp_path, {"parallel_mode": "2"}))

    row, = _opaque_actions(turns)[0]
    handle = _result_summary(row).split('read_check_receipt(handle="', 1)[1].split('"', 1)[0]
    first = json.loads(kernel_parallel.read_subframe_receipt(handle))
    second = json.loads(kernel_parallel.read_subframe_receipt(first["next_handle"]))
    assert first["rows"][0]["id"] == "bad000"
    assert second["page"] == 2
    assert list(outside.iterdir()) == []


def test_durable_result_read_outage_requires_recovery(tmp_path, monkeypatch):
    _, turns = _case(tmp_path, (("alpha", "COMMITTED"),))
    monkeypatch.setattr(kernel_parallel, "_service", (tmp_path, {"parallel_mode": "2"}))
    original = kernel_parallel._read_receipt_file

    def fail_result(descriptor, name, limit):
        if name == "result.json":
            raise OSError(errno.EIO, "planted durable read outage")
        return original(descriptor, name, limit)

    monkeypatch.setattr(kernel_parallel, "_read_receipt_file", fail_result)
    with pytest.raises(kernel_parallel.RecoveryRequired, match="planted durable read outage"):
        _opaque_actions(turns)


def test_malformed_subframe_handle_is_invalid_observation(tmp_path, monkeypatch):
    _receipt(tmp_path, monkeypatch, (("alpha", "COMMITTED"),))
    handle = "subframe:call_0:" + "9" * 5000 + ":" + "a" * 64
    assert kernel_parallel.receipt_handle_available(handle) is False
    with pytest.raises(kernel_parallel.ParallelError, match="invalid"):
        kernel_parallel.read_subframe_receipt(handle)


def test_input_refusal_without_expanded_intent_keeps_exact_resend_text(tmp_path, monkeypatch):
    refusal = Refusal("INVALID_ARGUMENTS", "ambiguous tests", (), (), ())
    result, turns = _case(tmp_path, (("alpha", "REFUSED"),), refusal=refusal)
    (tmp_path / "parallel" / "call_0" / "intent.json").unlink()
    monkeypatch.setattr(kernel_parallel, "_service", (tmp_path, {"parallel_mode": "2"}))
    row, = _opaque_actions(turns)[0]
    assert _result_summary(row) == result.observation


def test_other_tool_keeps_legacy_first_line_receipt():
    turns = (({"role": "assistant", "content": "", "tool_calls": [{
        "id": "read_0", "function": {"name": "read", "arguments": "{}"},
    }]}, {"role": "tool", "tool_call_id": "read_0", "content": "READ lines 1-2\nsource"}),)
    row, = _opaque_actions(turns)[0]
    assert row["kind"] == "unclassified_tool_action"
    assert row["result_receipt"] == "READ lines 1-2"
    assert row["result_chars"] == len("READ lines 1-2\nsource")


def test_disabled_subframe_refusal_is_complete_on_its_first_line():
    refusal = "REFUSED RESOURCES_UNAVAILABLE: subframe execution is disabled for this run"
    turns = (({"role": "assistant", "content": "", "tool_calls": [{
        "id": "parallel_0", "function": {"name": "parallel", "arguments": "{}"},
    }]}, {"role": "tool", "tool_call_id": "parallel_0", "content": refusal}),)
    row, = _opaque_actions(turns)[0]
    assert row["result_receipt"] == refusal
    assert row["result_chars"] == len(refusal)
