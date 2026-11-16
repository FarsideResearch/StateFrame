"""FU-070: the FILE floor byte check agrees with the allocator byte cap.

The allocator refuses a candidate only when serialized bytes are strictly greater than the
cap (kernel_working_set_prepare._violations: ``len(candidate.wire) > byte_cap``). A floor
exactly at the cap fits; one byte over it is the typed violation, refused at turn 1.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from test_fu070_alloc_checkpoint1 import (  # noqa: F401  (fixture re-export)
    _prepare, isolated_engine_python,
)
from e2e_support import run_cli, tool_step


pytestmark = pytest.mark.host_sandbox


def test_oversized_owned_file_pages_and_completes_without_floor_infeasible(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    # 2,800 lines (~63 KB): the owned file's plan closure is one whole-file unit with no single
    # read-page route. Subframe paging splits it into page-sized spans and retires delivered
    # pages from ordinal 2, so the child completes instead of FLOOR_INFEASIBLE mid-run.
    windows = ((1, 1000), (1001, 2000), (2001, 2800))
    child_steps = [tool_step("read", {"path": "owned_source.py", "start_line": start,
                                      "end_line": end}) for start, end in windows]
    repo, endpoint, run_dir = _prepare(tmp_path, isolated_engine_python, line_count=2800,
                                       child_steps=child_steps)
    with endpoint:
        run_cli(
            isolated_engine_python, endpoint.url, repo, run_dir,
            "Inspect the owned source with one subframe.", parallel="2",
            arguments=("--parallel-min-subframes", "1",
                       "--parallel-max-owned-bytes", "100000"),
            extra_environment={"FU070_ALLOCATOR_TEST": "on"}, timeout=600,
        )
    assert any(item["marker"] == "large_source" for item in endpoint.requests)
    unit_dirs = list((run_dir / "parallel").glob("*/units/large_source"))
    assert len(unit_dirs) == 1
    assert not list(unit_dirs[0].glob("unit-refused-request.json"))
    evidence = "\n".join(path.read_text(encoding="utf-8")
                         for path in unit_dirs[0].rglob("*.json"))
    assert "WORKING_SET_ALLOCATION_BLOCKED" not in evidence
    assert "FLOOR_INFEASIBLE" not in evidence
