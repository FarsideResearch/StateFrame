"""A deferral discovered after the first receipt is not a page lost by owner admission.

Regression (endurance s7_m1, v0.5.9): window fitting deferred the ticket target after
the frame receipt was already satisfied.  The post-owner re-verify compared the new
(unsatisfied, deferral-carrying) receipt against that stale satisfied receipt and raised
``final source delivery lost after owner admission`` although the admitted request
kept every page and the ``read`` tool.  The engine stopped with ENGINE_ERROR while
preparing request 14.

The reconciliation raises unless the regression is exactly a late or unqualifiable
deferral: every previously proven page is still delivered (per-path evidence, because an
unsatisfied receipt stamps FAILED on every row) or declared deferred by a notice that is
on the wire, ``read`` is admitted, and no other proof failure exists.
"""
from types import SimpleNamespace

import pytest

import kernel_request_source_window as window

TARGET = "core/__init__.py"
SIBLING = "core/service.py"
REVISION = "sha256:" + "0" * 64
READ_TOOL = {"type": "function", "function": {
    "name": "read", "parameters": {"type": "object", "properties": {}}}}
EDIT_TOOL = {"type": "function", "function": {
    "name": "edit", "parameters": {"type": "object", "properties": {}}}}


def _runtime():
    events = []
    journal = SimpleNamespace(emit=lambda kind, **fields: events.append((kind, fields)))
    return SimpleNamespace(journal=journal), events


def _complete_receipt(*paths):
    return {
        "status": "COMPLETE", "proof_status": "VERIFIED", "proof_failures": [],
        "deferred_paths": [], "deferred_notice_verified": False,
        "read_tool_available": True,
        "paths": {path: {"status": "COMPLETE", "proof_status": "VERIFIED", "revision": REVISION}
                  for path in paths},
    }


def _late_deferral_receipt(*, deferred, kept=(), read_tool_available=True, units=0):
    rows = {path: {"status": "COMPLETE", "proof_status": "VERIFIED", "revision": REVISION}
            for path in kept}
    for path in deferred:
        rows[path] = {
            "status": "DEFERRED", "proof_status": "VERIFIED", "revision": REVISION,
            "retrieval_route": {
                "status": "EXECUTABLE_BOUNDED_READ" if units else "NO_UNITS",
                "tool": "read", "path": path, "revision": REVISION,
                "unit_handles": [f"unit:{path}:{index}" for index in range(units)],
            },
        }
    return {
        "status": "DEFERRED", "proof_status": "VERIFIED", "proof_failures": [],
        "deferred_paths": list(deferred), "deferred_notice_verified": True,
        "read_tool_available": read_tool_available, "paths": rows,
    }


def _repo(tmp_path, *, ticket=None):
    root = tmp_path / "repo"
    (root / "core").mkdir(parents=True)
    (root / TARGET).write_text(
        "def handler(value):\n    return value + 1\n\n\ndef other(value):\n    return value - 1\n",
        encoding="utf-8",
    )
    (root / SIBLING).write_text(
        "".join(f"def handler_{index}(value):\n    return value + {index}\n\n"
                for index in range(60)),
        encoding="utf-8",
    )
    task_row = {"ticket_text": ticket} if ticket else {}
    events = []
    runtime = SimpleNamespace(
        target=root / TARGET, owner=SimpleNamespace(root=root, task_row=task_row),
        product_revision="0",
        journal=SimpleNamespace(emit=lambda kind, **fields: events.append((kind, fields))),
    )
    return root, runtime, events


def _sources(root):
    return {path: (root / path).read_text(encoding="utf-8") for path in (TARGET, SIBLING)}


def _verify(root, runtime, messages, tools):
    return window.verify_final_source_delivery(
        runtime._request_source_closure, messages,
        payload={"messages": messages, "tools": tools}, sources=_sources(root),
    )


def _kept_page_plus_deferral_wire(tmp_path, *, tools, notice=True):
    """Real wire: the target is windowed COMPLETE, the sibling deferred with no units."""
    root, runtime, events = _repo(tmp_path, ticket=f"fix handler in {TARGET}")
    runtime._request_deferred_source_paths = [SIBLING]
    messages = [{"role": "user", "content": f"please fix {TARGET}"}]
    result = window.bound_runtime_sources(
        messages, runtime, max_request_bytes=20_000, tools=tools)
    assert isinstance(result, dict), result
    plan = runtime._request_source_closure
    assert plan["deferred_paths"] == [SIBLING]
    assert [unit["path"] for unit in plan["units"]] == [TARGET]
    if notice:
        window.upsert_deferred_source_notice(messages, plan["deferred_paths"])
    return root, runtime, events, messages


def _envelope_deferral_wire(tmp_path, *, tools, notice):
    """Real wire: the envelope already exceeds the budget, the target is deferred unselected."""
    root, runtime, events = _repo(tmp_path)
    runtime.target = root / SIBLING
    messages = [{"role": "user", "content": "x" * 4000}]
    result = window.bound_runtime_sources(
        messages, runtime, max_request_bytes=512, tools=tools)
    assert isinstance(result, dict), result
    plan = runtime._request_source_closure
    assert plan["deferred_paths"] == [SIBLING]
    assert not any(
        window.parse_model_source_window(message.get("content")) is not None
        for message in messages if isinstance(message, dict)
    )
    if notice:
        window.upsert_deferred_source_notice(messages, plan["deferred_paths"])
    return root, runtime, events, messages


def _harden8_kinds(events):
    return [kind for kind, _ in events if kind.startswith("harden8")]


def test_late_deferral_is_advisory():
    # Same pages as before (the target moved into the deferral set) and read admitted:
    # the regression is a late deferral, journaled, never a crash.
    runtime, events = _runtime()
    capture = SimpleNamespace()
    previous = _complete_receipt(TARGET, SIBLING)
    final = _late_deferral_receipt(deferred=(TARGET,), kept=(SIBLING,), units=0)
    assert not window.source_delivery_satisfied(final)
    assert window.source_delivery_satisfied(previous)

    verdict = window.reconcile_owner_source_delivery(
        runtime, capture, previous, final, tools=[READ_TOOL, EDIT_TOOL])

    assert verdict["genuine_loss"] is False
    assert verdict["reason"] == "late_deferral_unsatisfied"
    assert verdict["lost_paths"] == []
    assert verdict["late_deferred_paths"] == [TARGET]
    assert verdict["read_tool_admitted"] is True
    assert verdict["deferred_rows"][0]["unit_count"] == 0
    assert capture.request_source_delivery_regression == verdict
    kinds = [kind for kind, _ in events]
    assert kinds == ["harden8_advisory_failure"]
    fields = events[0][1]
    assert fields["boundary"] == "completion_owner.source_delivery_receipt"
    assert fields["failure_class"] == "ADVISORY"
    assert fields["disposition"] == "degrade"
    assert "late_deferral_unsatisfied" in fields["message"]
    assert TARGET in fields["message"]


def test_late_deferral_real_loss_still_raises():
    # A page COMPLETE before admission that is neither delivered nor deferred afterwards
    # is a genuine loss: keep raising, and name the page.
    runtime, events = _runtime()
    capture = SimpleNamespace()
    previous = _complete_receipt(TARGET, SIBLING)
    final = _late_deferral_receipt(deferred=(TARGET,), kept=(), units=0)
    with pytest.raises(RuntimeError, match="lost after owner admission") as info:
        window.reconcile_owner_source_delivery(
            runtime, capture, previous, final, tools=[READ_TOOL])
    assert SIBLING in str(info.value)
    assert capture.request_source_delivery_regression["lost_paths"] == [SIBLING]
    assert events == []


def test_read_tool_dropped_by_admission_still_raises():
    # Deferred pages need the read route; an admission that dropped `read` lost it.
    runtime, _events = _runtime()
    capture = SimpleNamespace()
    previous = _complete_receipt(TARGET, SIBLING)
    final = _late_deferral_receipt(
        deferred=(TARGET,), kept=(SIBLING,), read_tool_available=False, units=3)
    with pytest.raises(RuntimeError, match="read tool dropped by admission"):
        window.reconcile_owner_source_delivery(
            runtime, capture, previous, final, tools=[EDIT_TOOL])
    assert capture.request_source_delivery_regression["read_tool_dropped"] is True


def test_satisfied_or_never_satisfied_receipts_are_not_reconciled():
    runtime, events = _runtime()
    capture = SimpleNamespace()
    assert window.reconcile_owner_source_delivery(
        runtime, capture, _complete_receipt(TARGET), _complete_receipt(TARGET)) is None
    unsatisfied_before = _late_deferral_receipt(deferred=(TARGET,), units=0)
    assert window.reconcile_owner_source_delivery(
        runtime, capture, unsatisfied_before,
        _late_deferral_receipt(deferred=(TARGET,), units=0)) is None
    assert window.reconcile_owner_source_delivery(
        runtime, capture, None, _late_deferral_receipt(deferred=(TARGET,), units=0)) is None
    assert events == []
    assert not hasattr(capture, "request_source_delivery_regression")


def test_unexplained_receipt_failure_still_raises():
    # Nothing delivered before, nothing deferred now, but the final receipt failed for a
    # reason the reconciliation does not qualify: keep the fail-closed stop.
    runtime, events = _runtime()
    capture = SimpleNamespace()
    final = dict(_complete_receipt(), status="UNKNOWN", proof_status="FAILED",
                 proof_failures=[{"kind": "workspace_authorization_unverified",
                                  "status": "UNVERIFIED"}])
    with pytest.raises(RuntimeError, match="workspace_authorization_unverified"):
        window.reconcile_owner_source_delivery(
            runtime, capture, _complete_receipt(), final, tools=[READ_TOOL])
    verdict = capture.request_source_delivery_regression
    assert verdict["reason"] == "final_receipt_unsatisfied"
    assert verdict["unexplained_failure_kinds"] == ["workspace_authorization_unverified"]
    assert events == []


def test_receipt_keys_are_normalized():
    previous = _complete_receipt("./core\\service.py")
    final = _late_deferral_receipt(deferred=(SIBLING,), units=0)
    verdict = window.classify_final_delivery_regression(previous, final, tools=[READ_TOOL])
    assert verdict["genuine_loss"] is False
    assert verdict["lost_paths"] == []
    assert verdict["deferred_paths"] == [SIBLING]


def test_envelope_over_budget_target_page_defers_without_crash(tmp_path):
    # Non-empty sibling of the zero-byte case: the envelope already exceeds the window
    # budget, so window fitting defers the target after the frame receipt was computed.
    # The real wire receipt carries the deferral as a typed proof failure and the notice
    # is on the wire; against a stale satisfied receipt the reconciliation must journal,
    # not raise. Only a page that is neither delivered nor deferred anywhere is a loss.
    tools = [READ_TOOL, EDIT_TOOL]
    root, runtime, events, messages = _envelope_deferral_wire(tmp_path, tools=tools, notice=True)
    final = _verify(root, runtime, messages, tools)
    assert final["deferred_notice_verified"] is True
    assert final["read_tool_available"] is True
    assert not window.source_delivery_satisfied(final)
    assert [failure["kind"] for failure in final["proof_failures"]] == [
        "deferred_source_path_unselected"]

    capture = SimpleNamespace()
    verdict = window.reconcile_owner_source_delivery(
        runtime, capture, _complete_receipt(SIBLING), final, tools=tools)
    assert verdict["genuine_loss"] is False
    assert verdict["reason"] == "deferral_unqualifiable"
    assert verdict["unqualified_deferred_paths"] == [SIBLING]
    assert verdict["deferred_notice_on_wire"] is True
    assert _harden8_kinds(events) == ["harden8_advisory_failure"]

    # Control: a second page that was COMPLETE before and is now nowhere is a loss.
    with pytest.raises(RuntimeError, match=TARGET):
        window.reconcile_owner_source_delivery(
            runtime, capture, _complete_receipt(SIBLING, TARGET), final, tools=tools)


def test_deferral_without_notice_on_wire_is_a_loss(tmp_path):
    # R2 body 1: the deferral was never written to the wire (no upsert ran). The model
    # has neither the page it had last request nor a statement that it was deferred, so
    # `deferred_source_notice_unverified` names the path but must not fold it into the
    # declared deferrals: raise, naming the page.
    tools = [READ_TOOL, EDIT_TOOL]
    root, runtime, events, messages = _envelope_deferral_wire(tmp_path, tools=tools, notice=False)
    final = _verify(root, runtime, messages, tools)
    kinds = {failure["kind"]: failure for failure in final["proof_failures"]}
    assert set(kinds) == {"deferred_source_notice_unverified", "deferred_source_path_unselected"}
    assert kinds["deferred_source_notice_unverified"]["notice_present"] is False
    assert kinds["deferred_source_notice_unverified"]["read_tool_available"] is True

    capture = SimpleNamespace()
    with pytest.raises(RuntimeError, match="pages absent from the admitted request") as info:
        window.reconcile_owner_source_delivery(
            runtime, capture, _complete_receipt(SIBLING), final, tools=tools)
    assert SIBLING in str(info.value)
    assert "without a notice on the wire" in str(info.value)
    verdict = capture.request_source_delivery_regression
    assert verdict["lost_paths"] == [SIBLING]
    assert verdict["undeclared_deferred_paths"] == [SIBLING]
    assert verdict["deferred_notice_on_wire"] is False
    assert _harden8_kinds(events) == []


def test_notice_on_wire_but_read_absent_is_a_loss(tmp_path):
    # Sibling of the above: the notice is present but admission dropped `read`, so the
    # deferred page has no retrieval route. Raise as a dropped read tool.
    tools = [EDIT_TOOL]
    root, runtime, _events, messages = _envelope_deferral_wire(tmp_path, tools=tools, notice=True)
    final = _verify(root, runtime, messages, tools)
    kinds = {failure["kind"]: failure for failure in final["proof_failures"]}
    assert kinds["deferred_source_notice_unverified"]["notice_present"] is True
    assert final["read_tool_available"] is False

    capture = SimpleNamespace()
    with pytest.raises(RuntimeError, match="read tool dropped by admission"):
        window.reconcile_owner_source_delivery(
            runtime, capture, _complete_receipt(SIBLING), final, tools=tools)
    assert capture.request_source_delivery_regression["read_tool_dropped"] is True


def test_kept_page_with_unqualified_deferral_is_advisory_on_real_wire(tmp_path):
    # R2 body 2: the real receipt stamps FAILED on every row once any proof failure
    # exists, so the kept target row reads PARTIAL/FAILED although it is complete on
    # the wire. Judged per path it is delivered; only the sibling is (declared) deferred.
    tools = [READ_TOOL, EDIT_TOOL]
    root, runtime, events, messages = _kept_page_plus_deferral_wire(tmp_path, tools=tools)
    final = _verify(root, runtime, messages, tools)
    assert not window.source_delivery_satisfied(final)
    assert [failure["kind"] for failure in final["proof_failures"]] == [
        "deferred_source_path_unselected"]
    target_row = final["paths"][TARGET]
    assert target_row["proof_status"] == "FAILED"
    assert target_row["status"] == "PARTIAL"
    assert target_row["delivered_units"] == target_row["selected_units"] == 1
    assert final["delivery_proofs"][TARGET]

    capture = SimpleNamespace()
    verdict = window.reconcile_owner_source_delivery(
        runtime, capture, _complete_receipt(TARGET), final, tools=tools)
    assert verdict["genuine_loss"] is False
    assert verdict["reason"] == "deferral_unqualifiable"
    assert verdict["lost_paths"] == []
    assert verdict["deferred_paths"] == [SIBLING]
    assert _harden8_kinds(events) == ["harden8_advisory_failure"]


def test_truncated_kept_page_on_real_wire_raises(tmp_path):
    # Same wire, but admission truncated the kept page: its proof fails for that path,
    # so the page is lost even though a late deferral is also present.
    tools = [READ_TOOL, EDIT_TOOL]
    root, runtime, events, messages = _kept_page_plus_deferral_wire(tmp_path, tools=tools)
    windows = [index for index, message in enumerate(messages)
               if str(message.get("content", "")).startswith(f"[Current source: {TARGET}]")]
    assert len(windows) == 1, messages
    content = messages[windows[0]]["content"]
    # Drop the last source line of the delivered window (admission strip/truncation).
    messages[windows[0]]["content"] = content[:content.rstrip("\n").rfind("\n")] + "\n"
    final = _verify(root, runtime, messages, tools)
    assert any(failure.get("path") == TARGET for failure in final["proof_failures"]), (
        final["proof_failures"])

    capture = SimpleNamespace()
    with pytest.raises(RuntimeError, match="pages absent from the admitted request") as info:
        window.reconcile_owner_source_delivery(
            runtime, capture, _complete_receipt(TARGET), final, tools=tools)
    assert TARGET in str(info.value)
    assert capture.request_source_delivery_regression["lost_paths"] == [TARGET]
    assert _harden8_kinds(events) == []


def _partial_target_plus_unselected_deferral_wire(tmp_path, *, tools, notice_paths=None):
    """Real wire: the target is deferred WITH selected units (the exact read route carries
    them) and a second declared deferral is never selected."""
    root, runtime, events = _repo(tmp_path, ticket=f"fix handler_7 in {SIBLING}")
    runtime.target = root / SIBLING
    runtime._request_deferred_source_paths = [TARGET]
    messages = [{"role": "user", "content": f"please fix handler_7 in {SIBLING}"}]
    result = window.bound_runtime_sources(
        messages, runtime, max_request_bytes=2_200, tools=tools)
    assert isinstance(result, dict), result
    plan = runtime._request_source_closure
    assert sorted(plan["deferred_paths"]) == sorted([TARGET, SIBLING]), plan["deferred_paths"]
    assert SIBLING in [unit["path"] for unit in plan["units"]], plan["units"]
    assert TARGET not in [unit["path"] for unit in plan["units"]], plan["units"]
    if notice_paths is None:
        notice_paths = plan["deferred_paths"]
    if notice_paths:
        window.upsert_deferred_source_notice(messages, notice_paths)
    return root, runtime, events, messages


def test_prior_verified_deferral_plus_late_second_deferral_is_advisory_on_real_wire(tmp_path):
    # R3 body 1: the previous receipt verified the target as DEFERRED with a read route.
    # Window fitting then deferred a second page that is never selected. The notice on
    # the wire names both pages and `read` is admitted, yet any proof failure empties
    # receipt["deferred_paths"], so the target used to be judged lost and the run stopped.
    # The declared deferral set is published regardless of proof status; folded in when
    # the notice is on the wire, the target is neither lost nor late: advisory only.
    tools = [READ_TOOL, EDIT_TOOL]
    root, runtime, events, messages = _partial_target_plus_unselected_deferral_wire(
        tmp_path, tools=tools)
    final = _verify(root, runtime, messages, tools)
    assert final["deferred_notice_verified"] is True
    assert final["read_tool_available"] is True
    assert not window.source_delivery_satisfied(final)
    assert [failure["kind"] for failure in final["proof_failures"]] == [
        "deferred_source_path_unselected"], final["proof_failures"]
    assert final["proof_failures"][0]["paths"] == [TARGET]
    assert final["deferred_paths"] == []
    assert sorted(final["declared_deferred_paths"]) == sorted([TARGET, SIBLING])
    assert final["paths"][SIBLING]["status"] == "PARTIAL"
    assert final["paths"][SIBLING]["selected_units"] == 1

    previous = _late_deferral_receipt(deferred=(SIBLING,), units=1)
    capture = SimpleNamespace()
    verdict = window.reconcile_owner_source_delivery(
        runtime, capture, previous, final, tools=tools)
    assert verdict["genuine_loss"] is False
    assert verdict["reason"] == "deferral_unqualifiable"
    assert verdict["lost_paths"] == []
    assert verdict["deferred_paths"] == sorted([TARGET, SIBLING])
    assert verdict["unqualified_deferred_paths"] == [TARGET]
    assert verdict["late_deferred_paths"] == [TARGET]
    assert verdict["deferred_notice_on_wire"] is True
    assert _harden8_kinds(events) == ["harden8_advisory_failure"]


def test_prior_verified_deferral_missing_from_notice_is_a_loss_on_real_wire(tmp_path):
    # Control for the above: the notice on the wire names only the new page. The target
    # the model was promised a read route for is neither delivered nor declared deferred
    # anywhere on this wire: raise, naming it.
    tools = [READ_TOOL, EDIT_TOOL]
    root, runtime, events, messages = _partial_target_plus_unselected_deferral_wire(
        tmp_path, tools=tools, notice_paths=[TARGET])
    final = _verify(root, runtime, messages, tools)
    kinds = {failure["kind"]: failure for failure in final["proof_failures"]}
    assert set(kinds) == {"deferred_source_notice_unverified", "deferred_source_path_unselected"}
    assert kinds["deferred_source_notice_unverified"]["notice_present"] is False
    assert sorted(final["declared_deferred_paths"]) == sorted([TARGET, SIBLING])

    previous = _late_deferral_receipt(deferred=(SIBLING,), units=1)
    capture = SimpleNamespace()
    with pytest.raises(RuntimeError, match="pages absent from the admitted request") as info:
        window.reconcile_owner_source_delivery(
            runtime, capture, previous, final, tools=tools)
    assert SIBLING in str(info.value)
    assert capture.request_source_delivery_regression["lost_paths"] == [SIBLING]
    assert _harden8_kinds(events) == []


def test_undeclared_deferral_message_names_the_pages():
    # R3 nit 2: a never-delivered page deferred with no notice on the wire has no lost
    # page and no unexplained kind; the catch-all must name the undeclared deferral
    # instead of "status=...".
    runtime, events = _runtime()
    capture = SimpleNamespace()
    final = dict(
        _complete_receipt(), status="PARTIAL", proof_status="FAILED",
        deferred_notice_verified=False,
        proof_failures=[
            {"kind": "deferred_source_notice_unverified", "paths": [TARGET],
             "notice_present": False, "read_tool_available": True},
            {"kind": "deferred_source_path_unselected", "paths": [TARGET]},
        ])
    with pytest.raises(RuntimeError, match="deferred without a notice on the wire") as info:
        window.reconcile_owner_source_delivery(
            runtime, capture, _complete_receipt(), final, tools=[READ_TOOL])
    assert TARGET in str(info.value)
    assert "status=" not in str(info.value)
    verdict = capture.request_source_delivery_regression
    assert verdict["reason"] == "final_receipt_unsatisfied"
    assert verdict["undeclared_deferred_paths"] == [TARGET]
    assert events == []
