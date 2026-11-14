"""Shipped-profile evidence for a capacity fallback with no launched subframes."""
from __future__ import annotations

from pathlib import Path

import pytest

from _journal import measurement_events, measurement_journal_paths
from barrier_endpoint import BarrierEndpoint
from e2e_support import (assert_shipped_wire, parent_wire_files, run_cli, summary_for,
                         tool_names, tool_step)


pytestmark = pytest.mark.host_sandbox


def test_all_refused_size_is_visible_in_parent_journal_and_summary(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("VALUE = '" + "a" * 100 + "'\n")
    (repo / "b.py").write_text("VALUE = '" + "b" * 100 + "'\n")
    units = [{"id": unit_id, "outcome": f"Update {path}", "owns": [path],
              "creates": [], "reads": [], "exports": [], "consumes": [],
              "requirements": [], "check": {"tests": []}}
             for unit_id, path in (("a", "a.py"), ("b", "b.py"))]
    steps = [tool_step("parallel", {"subframes": units}),
             tool_step("done", {"summary": "The subframes were refused by the size cap."})]
    with BarrierEndpoint(steps, {}, slots=2) as endpoint:
        completed = run_cli(
            isolated_engine_python, endpoint.url, repo, tmp_path / "run",
            "Record a subframe capacity refusal.", parallel="2",
            arguments=("--parallel-max-owned-bytes", "10"),
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        endpoint.assert_consumed()

    run = tmp_path / "run"
    requests = parent_wire_files(run)
    assert_shipped_wire(requests)
    assert "parallel" in tool_names(requests[0])
    summary = summary_for(run)
    assert summary["parallel"]["degraded"] == [{
        "call_id": summary["parallel"]["calls"][0]["call_id"],
        "reason": "SUBFRAME_CONTEXT_TOO_LARGE",
        "detail": "owned_source_bytes=111; max_owned_source_bytes=10 "
                  "(--parallel-max-owned-bytes / STATEFRAME_PARALLEL_MAX_OWNED_SOURCE_BYTES)",
        "launched_subframes": 0,
    }]
    journals = measurement_journal_paths(run)
    assert len(journals) == 1
    events = measurement_events(run)
    degraded = [event for event in events if event.get("kind") == "parallel_degraded"]
    assert len(degraded) == 1
    assert degraded[0]["reason"] == "UNIT_CONTEXT_TOO_LARGE"
    assert degraded[0]["detail"] == summary["parallel"]["degraded"][0]["detail"]
    assert degraded[0]["subframes"] == ["a", "b"]
    assert not list((run / "parallel").glob("*/units/*/unit-spec.json"))
