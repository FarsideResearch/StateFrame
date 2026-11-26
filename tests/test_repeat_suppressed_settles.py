"""A read-only call the host suppressed as a repeat never ran; its effect is settled, not unknown.

An unknown read-only effect has no repair for search tools, so continuation blocked the whole run
on the first suppressed grep.
"""
from __future__ import annotations

import json
import sys

import pytest

from kernel_continuation_evidence import READ_RESULT_CLASSES, result_class_disposition


@pytest.mark.parametrize("tool", ["read", "text", "grep", "ls"])
def test_repeat_suppressed_read_only_call_is_settled_not_applied(tool):
    assert tool in READ_RESULT_CLASSES
    assert result_class_disposition(tool, "REPEAT_SUPPRESSED") == "settled_not_applied"


def test_unrecognised_read_result_stays_unknown():
    assert result_class_disposition("grep", "SOMETHING_NEW") == "unknown"


def test_admission_refusal_stays_scoped_to_read():
    assert result_class_disposition("grep", "REFUSED_ADMISSION") == "unknown"


@pytest.mark.host_sandbox
def test_suppressed_list_settles_once_before_next_action(tmp_path):
    from _journal import measurement_events
    from test_shell_e2e import finish, run_script, step

    hooks = tmp_path / "hooks.json"
    hooks.write_text(json.dumps({"schema": "stateframe-hooks/1", "hooks": [{
        "name": "audit-list", "event": "post_tool", "tools": ["list_files"],
        "argv": [sys.executable, "-c",
                 "import json, sys\n"
                 "data = json.load(sys.stdin)['data']\n"
                 "print(json.dumps([data['call_id'], data['status'], data['executed']]), file=sys.stderr)"],
    }]}))
    run, _ = run_script(tmp_path, [step("list_files")] * 11 + finish(),
                        flags=("--hooks", str(hooks)))
    events = measurement_events(run)
    [receipt] = [event["receipt"] for event in events if event["kind"] == "action_receipt"
                 and event["receipt"]["result_class"] == "REPEAT_SUPPRESSED"]
    assert receipt["kind"] == "list_files"
    [closure] = [event for event in events if event["kind"] == "action_outcome_closed"
                 and receipt["receipt_id"] in event["record"]["receipt_ids"]]
    [settlement] = [event for event in events if event.get("operation") == "settle_effect"
                    and event["data"]["effect_id"] == receipt["continuation_effect_id"]]
    assert settlement["data"]["receipt_ref"] == closure["event_id"]
    assert settlement["data"]["status"] == "not_applied"
    records = [json.loads(line) for line in (run / "hooks/records.jsonl").read_text().splitlines()]
    posts = [json.loads((run / "hooks/out" / f"{row['event_id']}.err").read_text())
             for row in records if row["hook"] == "audit-list"]
    call_id = closure["record"]["tool_call_id"]
    assert [post for post in posts if post[0] == call_id] == [[call_id, "refused", False]]
