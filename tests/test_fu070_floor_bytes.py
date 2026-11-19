"""FU-070: the FILE floor byte check agrees with the allocator byte cap.

The allocator refuses a candidate only when serialized bytes are strictly greater than the
cap (kernel_working_set_prepare._violations: ``len(candidate.wire) > byte_cap``). A floor
exactly at the cap fits; one byte over it is the typed violation, refused at turn 1.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_fu070_alloc_checkpoint1 import (  # noqa: F401  (fixture re-export)
    _prepare, isolated_engine_python,
)
from e2e_support import run_cli, tool_step


@pytest.mark.host_sandbox
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


def _with_active_unit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "kernel_unit_policy", SimpleNamespace(active=lambda: object()))


def test_floor_page_count_matches_plan_spans(monkeypatch: pytest.MonkeyPatch) -> None:
    # 100-character lines: a bare 64000-character page holds 640 of them, but the allocator
    # budget nets out the page header reserve (512 here) and holds 634. The floor must price
    # the pages the allocator emits (3), not the bare-ceiling pages (2).
    from kernel_source_view import file_floor_messages, parse_source_map
    from kernel_working_set_windows import _plan_spans

    path = "owned_source.py"
    source = "".join(f"{number:099d}\n" for number in range(1280))
    lines = source.splitlines(keepends=True)
    revision = "sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest()
    _with_active_unit(monkeypatch)
    planned = tuple((start, end) for start, end, _ in
                    _plan_spans(lines, 1, len(lines), path, revision, []))
    messages = file_floor_messages(path, source)
    *pages, source_map = messages
    floor_spans = tuple(tuple(int(number) for number in re.search(
        r"--- source lines (\d+)-(\d+) ---", page["content"]).groups()) for page in pages)
    assert floor_spans == planned == ((1, 634), (635, 1268), (1269, 1280))
    # The map line merges adjacent spans; it must still claim the whole file at this revision.
    assert parse_source_map(source_map["content"])[path] == (revision, ((1, 1280),))


def test_floor_of_an_empty_file_is_the_bare_map_line() -> None:
    from kernel_source_view import file_floor_messages, parse_source_map

    messages = file_floor_messages("empty.py", "")
    assert len(messages) == 1
    assert parse_source_map(messages[0]["content"])["empty.py"][1] == ()



def test_unit_gates_tolerate_a_policy_module_without_active(monkeypatch: pytest.MonkeyPatch) -> None:
    import kernel_working_set_render
    import kernel_working_set_windows

    lines = [f"{number:099d}\n" for number in range(1280)]
    for stub in (SimpleNamespace(), SimpleNamespace(active=None)):
        monkeypatch.setitem(sys.modules, "kernel_unit_policy", stub)
        assert kernel_working_set_windows._unit_active() is False
        assert kernel_working_set_render._unit_active() is False
        assert kernel_working_set_render._current_source_map([], {}) is None
        assert kernel_working_set_windows._plan_spans(
            lines, 1, len(lines), "owned_source.py", "sha256:" + "0" * 64, ["a"]) == (
                (1, len(lines), ["a"]),)
