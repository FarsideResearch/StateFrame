"""A repo with more files than the bounded closure walk must degrade, not refuse.

Regression: on mid-size repos (5 packages, ~1,100+ lines) the walk reached its
file cap and recorded the next file as a PARTIAL "closure file limit reached"
omission. The final source proof counted that unopened file as an undelivered
debt, so every run failed before its first send with
``closure_omissions_unreconciled``.
"""
from pathlib import Path

import pytest

from kernel_frame_allocation_capture import source_refusal
from kernel_request_evidence_closure import (
    CLOSURE_SCAN_LIMIT,
    build_source_closure,
    omission_blocks_delivery,
    record_final_source_delivery,
)

TARGET = "pkg0/mod0.py"


def _mid_size_repo(root: Path, packages: int = 5, modules: int = 6, functions: int = 6) -> list[str]:
    """Same shape as the reproducing repo: packages import their neighbour."""
    paths = []
    for p in range(packages):
        pkg = root / f"pkg{p}"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
        paths.append(f"pkg{p}/__init__.py")
        for m in range(modules):
            body = [f"from pkg{(p + 1) % packages}.mod0 import f0 as _peer\n" if p else ""]
            body += [f"\n\ndef f{f}(x):\n    y = x + {f}\n    return y * 2\n" for f in range(functions)]
            (pkg / f"mod{m}.py").write_text("".join(body))
            paths.append(f"pkg{p}/mod{m}.py")
    return paths


def _plan(tmp_path: Path) -> dict:
    root = tmp_path / "repo"
    paths = _mid_size_repo(root)
    plan = build_source_closure(root, paths, "Fix f1 in pkg0/mod0.py so tests pass.",
                                target_path=TARGET).as_dict()
    # As in the failing run, bounded source selection keeps one whole-file
    # unit of the target and nothing else.
    lines = len((root / TARGET).read_text().splitlines())
    plan["units"] = [{"path": TARGET, "start_line": 1, "end_line": lines,
                      "handle": "unit:target", "role": "module"}]
    return plan


def _delivered(plan: dict) -> dict:
    return {TARGET: [(unit["start_line"], unit["end_line"]) for unit in plan["units"]]}


def test_walk_marks_unopened_files_as_scan_limit(tmp_path):
    plan = _plan(tmp_path)
    limited = [row for row in plan["omissions"] if row.get("scope") == CLOSURE_SCAN_LIMIT]
    assert limited, "fixture must exceed the closure file cap"
    assert all(row["status"] == "PARTIAL" and row["continuation"].startswith("expand:file:")
               for row in limited)


def test_scan_limit_alone_keeps_final_delivery_complete(tmp_path):
    plan = _plan(tmp_path)
    receipt = record_final_source_delivery(plan, _delivered(plan), wire_proof_verified=True)
    assert receipt["status"] == "COMPLETE"


def test_refusal_does_not_count_scan_limit_rows(tmp_path):
    plan = _plan(tmp_path)
    receipt = {"status": "PARTIAL", "proof_status": "VERIFIED",
               "paths": {TARGET: {"status": "COMPLETE", "selected_units": len(plan["units"])}}}
    error = source_refusal(plan, receipt, {})
    assert "closure_omissions_unreconciled" not in error.source_diagnostic["failures"]
    assert error.source_diagnostic["omissions"] == []


@pytest.mark.parametrize("packages,modules", [(2, 5), (4, 4), (8, 4)])  # 12, 20, 40 files
def test_final_delivery_complete_across_repo_sizes(tmp_path, packages, modules):
    root = tmp_path / "repo"
    paths = _mid_size_repo(root, packages, modules)
    plan = build_source_closure(root, paths, "Fix f1 in pkg0/mod0.py so tests pass.",
                                target_path=TARGET).as_dict()
    lines = len((root / TARGET).read_text().splitlines())
    plan["units"] = [{"path": TARGET, "start_line": 1, "end_line": lines,
                      "handle": "unit:target", "role": "module"}]
    receipt = record_final_source_delivery(plan, _delivered(plan), wire_proof_verified=True)
    assert receipt["status"] == "COMPLETE"


def test_scan_limit_on_selected_path_still_blocks():
    row = {"path": "a.py", "status": "PARTIAL", "scope": CLOSURE_SCAN_LIMIT}
    assert omission_blocks_delivery(row, {"a.py"})
    assert not omission_blocks_delivery(row, {"b.py"})


def test_other_omissions_stay_fail_closed():
    assert omission_blocks_delivery({"path": "a.py", "status": "PARTIAL"}, set())
    assert omission_blocks_delivery({"path": "a.py", "status": "OUT_OF_SCOPE"}, set())
    assert omission_blocks_delivery({"status": "PARTIAL", "scope": CLOSURE_SCAN_LIMIT}, set())
    assert omission_blocks_delivery("not-a-row", set())
    assert not omission_blocks_delivery({"path": "a.py", "status": "NOT_SELECTED"}, set())
