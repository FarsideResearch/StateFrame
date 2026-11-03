"""A directory created by a successful shell command must not leave an unresolved effect.

Regression: `mkdir -p pkg/batch` through `shell` exited 0 and reported "created: pkg/batch", yet the effect stayed
status=unknown and blocked every later action before dispatch.
"""
import json

import pytest

from test_shell_e2e import run_script, step

pytestmark = pytest.mark.host_sandbox


def test_shell_mkdir_then_file_create_completes(tmp_path):
    run, requests = run_script(tmp_path, [
        step("shell", command="mkdir -p pkg/batch", why="Create the batch package directory"),
        step("file_create", path="pkg/batch/__init__.py", content="BATCH = 1\n"),
        step("run_tests", target="tests/test_value.py"),
        step("done", summary="Created the batch package."),
    ])
    assert "has no typed repair action" not in requests
    assert "Unresolved effect" not in requests
    assert (run / "workspace/pkg/batch/__init__.py").read_text() == "BATCH = 1\n"
    # A finished run seals its measurement journal as events.jsonl.xz; open_journal reads both forms.
    from stateframe.engine.kernel_journal_store import open_journal
    event_paths = [*run.rglob("events.jsonl"), *run.rglob("events.jsonl.xz")]
    assert event_paths
    events = []
    for path in event_paths:
        with open_journal(path.parent) as stream:
            events.extend(json.loads(line) for line in stream.read().splitlines() if line)
    shell_close = next(event for event in events
                       if event.get("kind") == "action_outcome_closed"
                       and event.get("record", {}).get("tool") == "shell")
    effect_id = shell_close["record"]["continuation_effect_id"]
    settlements = [event["data"]["status"] for event in events
                   if event.get("kind") == "continuation_transition"
                   and event.get("operation") == "settle_effect"
                   and event.get("data", {}).get("effect_id") == effect_id]
    assert settlements[-1] == "applied"
    assert any(event.get("kind") == "step_dispatch_binding"
               and event.get("tool") == "file_create"
               and event.get("accepted") is True for event in events)
