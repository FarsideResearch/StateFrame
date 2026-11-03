"""Scoped writes must be refused before the candidate can lose their effects."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from acceptance._support import run_case
from probes import react_repair as rr
from test_edit_scope_widen import BROKEN_RATES, FIXED_RATES, _large_repo, _workspace


def _script(tmp_path: Path, tool: str, arguments: dict) -> Path:
    path = tmp_path / "refusal.script.json"
    path.write_text(json.dumps({"model": "scripted", "tokenize": {"mode": "ok"},
                                "steps": [{"tool": tool, "arguments": arguments},
                                          *({"tool": "done", "arguments": {"summary": "Done."}}
                                            for _ in range(3))]}))
    return path


@pytest.mark.parametrize("tool,arguments", [
    ("str_replace", {"path": "app/rates.py", "old_str": "p / 10", "new_str": "p / 100"}),
    ("edit", {"path": "app/rates.py", "function_source": FIXED_RATES}),
    ("file_create", {"path": "app/rates.py", "content": FIXED_RATES}),
    ("file_delete", {"path": "app/rates.py"}),
    ("file_move", {"source": "app/rates.py", "destination": "app/new_rates.py"}),
])
def test_default_scope_refuses_existing_product_file(tmp_path, tool, arguments):
    repo = _large_repo(tmp_path)
    case = run_case(tmp_path, str(_script(tmp_path, tool, arguments)), repo=repo,
                    task="Inspect app/pricing.py.", flags=("--edit", "app/pricing.py"))
    case.assert_offered(tool)
    case.assert_reply('\\"code\\":\\"NOT_EDITABLE\\",\\"path\\":\\"app/rates.py\\"')
    case.assert_reply("--edit app/rates.py")
    assert (repo / "app/rates.py").read_text() == BROKEN_RATES
    assert "app/rates.py" not in case.diff
    assert case.summary and "app/rates.py" not in case.summary["files_changed"]


# insert_before and replace_module are not offered in a shipped run, so they have no
# end-to-end case here; test_write_tool_inventory_requires_scope_guard covers them.


def test_write_tool_inventory_requires_scope_guard():
    offered = {item["function"]["name"] for item in [*rr.TOOLS, rr._JQ_REPLACE_MODULE_TOOL]}
    read_only = {"grep", "list_files", "read", "run_tests", "read_check_receipt",
                 "recall", "done", "git", "job_output", "job_wait", "job_stop",
                 "skill_load", "web_fetch"}
    separate_authority = {"shell", "authored_check"}
    assert offered - read_only - separate_authority == rr.SCOPED_WRITE_TOOLS


def test_scope_hint_is_in_typed_model_reply(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    reply = json.loads(rr._typed_edit_rejection(
        "NOT_EDITABLE", root / "app/rates.py", root=root,
        hint="Outside editable_paths: use --edit app/rates.py or --edit-scope widen.",
        include_hint=True))
    assert reply["code"] == "NOT_EDITABLE"
    assert reply["path"] == "app/rates.py"
    assert "--edit app/rates.py" in reply["alternatives"][0]["hint"]
    assert "--edit-scope widen" in reply["alternatives"][0]["hint"]


def test_atomic_batch_refuses_before_writing(tmp_path, monkeypatch):
    from kernel_coordinated_change_ledger import atomic_apply

    monkeypatch.setenv("STATEFRAME_TOKENLAB_ATOMIC_MULTI_FILE_APPLY", "on")
    root, _, _ = _workspace(tmp_path)
    result = atomic_apply(root, {"edits": [{"path": "app/rates.py",
                                             "old": "p / 10", "new": "p / 100"}]},
                          scope_refused=lambda path: path == "app/rates.py")
    assert result.status == "REJECTED"
    assert result.failures[0].code == "NOT_EDITABLE"
    assert (root / "app/rates.py").read_text() == BROKEN_RATES


def test_host_detects_changed_existing_file_it_would_drop(tmp_path):
    from stateframe import outcomes, runner, workcopy

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pricing.py").write_text("PRICE = 1\n")
    (repo / "rates.py").write_text("RATE = 1\n")
    work = workcopy.create(repo, tmp_path / "copy")
    candidate = {"rates.py": b"RATE = 2\n", "pricing.py": b"PRICE = 1\n"}
    assert runner._dropped_existing_candidate_paths(work, candidate, {"pricing.py"}) == ["rates.py"]
    assert runner._dropped_existing_candidate_paths(work, {"rates.py": b"RATE = 1\n"},
                                                    {"pricing.py"}) == []
    assert runner._dropped_existing_candidate_paths(work, {"new.py": b"NEW = 1\n"},
                                                    {"pricing.py"}) == []
    projected = outcomes.project({"engine_terminal": {"result": "SUBMITTED"},
                                  "child_exit_code": 0,
                                  "candidate_captured": True,
                                  "candidate_error": "sealed candidate changed existing files outside edit scope",
                                  "candidate_paths_dropped": ["rates.py"]})
    assert projected["runtime_outcome"] == "engine_error"
    assert projected["artifact"] == "partial"
