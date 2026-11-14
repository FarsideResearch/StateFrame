"""Every plan deferral the live closure proof records gets a notice on the wire.

The ON path (kernel_working_set_live) marks a plan unit deferred when the FULL
carriers do not cover it with whole lines. The renderer used to declare only
``snapshot.deferred_paths`` plus non-FULL slots, so a plan unit carried FULL by
a clipped page was deferred in the receipt with no notice on the wire. Mainline's
late-deferral reconciliation (``reconcile_owner_source_delivery``) judges a
deferral by its notice and treats such a path as lost on re-verify.
"""
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import kernel_request_source_window as window
from kernel_read_receipts import ReceiptIdentity, line_digests_with_numbers
from kernel_working_set_allocator import FULL
from kernel_working_set_facts import _covered
from kernel_working_set_render import render_assignment
from kernel_working_set_snapshot import PreparationSnapshot, SourceSlot

TEXT = "alpha\nbeta\ngamma\n"
REVISION = "sha256:" + "1" * 64
READ_TOOL = {"type": "function", "function": {
    "name": "read", "parameters": {"type": "object", "properties": {}}}}


def _clipped_plan_slot(key="a"):
    """A captured page the closure plan merged into, clipped by max_chars."""
    digest = hashlib.sha256(TEXT.encode()).hexdigest()
    identity = ReceiptIdentity("a.py", REVISION, 1, 3, 0, len(TEXT.encode()), digest, None)
    digests, numbers = line_digests_with_numbers(TEXT, 1)
    page = (f"[read page] path=a.py revision={REVISION} lines=1-3/3 clip=max_chars\n"
            f"0 more; complete=true\n{TEXT}")
    standalone_prefix = f"[model source window] a.py revision={REVISION}\n"
    return SourceSlot(key, (key,), identity, TEXT, page, digests, numbers,
                      b'{"path":"a.py","start_line":1,"end_line":3}', 1, 1, "RESIDENT",
                      capture_ids=("cap-a",), standalone_page=standalone_prefix + TEXT,
                      standalone_payload_start=len(standalone_prefix.encode()))


def _snapshot(*slots):
    body = {"model": "local", "messages": [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task: fix"},
        {"role": "user", "content": "opaque newest suffix"}],
        "tools": [READ_TOOL], "max_tokens": 100}
    return PreparationSnapshot(json.dumps(body).encode(), tuple(slots), "request", 1,
                               100_000, current_revisions=(("a.py", REVISION),))


def _notice_paths(messages):
    return {path for notice in window.deferred_source_notice_paths(messages) for path in notice}


def test_clipped_full_plan_unit_is_a_deferral_with_a_notice_on_the_wire():
    slot = _clipped_plan_slot()
    snapshot = _snapshot(slot)
    # The live closure proof defers this unit: a clipped page is not whole lines.
    assert not _covered([slot], "a.py", REVISION, 1, 3)
    from kernel_working_set_render import uncovered_plan_unit_paths
    assert uncovered_plan_unit_paths(snapshot, [slot], {"a.py": REVISION}) == {"a.py"}
    rendered = render_assignment(snapshot, {"a": FULL}, "plain")
    messages = json.loads(rendered.wire)["messages"]
    assert "a.py" in _notice_paths(messages)


def test_whole_line_full_plan_unit_gets_no_notice():
    slot = _clipped_plan_slot()
    page = slot.page.replace(" clip=max_chars", "")
    slot = replace(slot, page=page)
    snapshot = _snapshot(slot)
    from kernel_working_set_render import uncovered_plan_unit_paths
    assert uncovered_plan_unit_paths(snapshot, [slot], {"a.py": REVISION}) == set()
    rendered = render_assignment(snapshot, {"a": FULL}, "plain")
    assert _notice_paths(json.loads(rendered.wire)["messages"]) == set()


def test_on_plan_deferral_passes_late_deferral_reverify():
    slot = _clipped_plan_slot()
    rendered = render_assignment(_snapshot(slot), {"a": FULL}, "plain")
    body = json.loads(rendered.wire)
    plan = {
        "units": [{"path": "a.py", "start_line": 1, "end_line": 3, "role": "closure"}],
        "revisions": {"a.py": REVISION},
        "source_contents": {"a.py": TEXT},
        "deferred_paths": ["a.py"],
        "workspace_authorization": {"scope": "test", "generation": "test"},
    }
    final = window.verify_final_source_delivery(
        plan, body["messages"], payload=rendered.wire, sources={"a.py": TEXT})
    unverified = [row for row in final["proof_failures"]
                  if row.get("kind") == "deferred_source_notice_unverified"]
    assert all(row.get("notice_present") is True for row in unverified)
    events = []
    runtime = SimpleNamespace(journal=SimpleNamespace(
        emit=lambda kind, **fields: events.append((kind, fields))))
    capture = SimpleNamespace()
    previous = {
        "status": "COMPLETE", "proof_status": "VERIFIED", "proof_failures": [],
        "deferred_paths": [], "deferred_notice_verified": False, "read_tool_available": True,
        "paths": {"a.py": {"status": "COMPLETE", "proof_status": "VERIFIED",
                           "revision": REVISION}},
    }
    # Must not raise: the deferral is declared on the wire and read is admitted.
    verdict = window.reconcile_owner_source_delivery(
        runtime, capture, previous, final, tools=body["tools"])
    if verdict is not None:
        assert verdict["genuine_loss"] is False
        assert verdict["undeclared_deferred_paths"] == []
        assert verdict["deferred_notice_on_wire"] is True
