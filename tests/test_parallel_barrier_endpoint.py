"""The scripted endpoint routes the exact task text sent for each subframe."""
from __future__ import annotations

import pytest

from fixtures.barrier_endpoint import BarrierEndpoint


def _request(task: str, *, subframe: bool) -> dict:
    names = ["read", "done", "unit_blocked"] if subframe else ["read", "done", "parallel"]
    return {"messages": [{"role": "user", "content": task}],
            "tools": [{"function": {"name": name}} for name in names]}


def test_subframe_task_prefix_routes_exact_id():
    endpoint = BarrierEndpoint([], {"fix_tax": [{"tool": "read", "arguments": {"path": "tax.py"}}]})
    body = _request("Task: Subframe fix_tax: update tax\nOwned: tax.py", subframe=True)

    marker = endpoint._marker(body)

    assert marker == "fix_tax"
    assert endpoint._step(marker, ["read", "done", "unit_blocked"]) == {
        "tool": "read", "arguments": {"path": "tax.py"}}


def test_unmatched_subframe_task_is_endpoint_error():
    body = _request("Task: a renamed subframe task", subframe=True)

    with pytest.raises(ValueError, match="cannot identify subframe task"):
        BarrierEndpoint._marker(body)


def test_parent_task_with_subframe_text_stays_parent():
    body = _request("Task: fix Subframe fix_tax: after reading tests", subframe=False)
    assert BarrierEndpoint._marker(body) == "parent"
