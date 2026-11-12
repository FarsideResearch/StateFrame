"""A zero-byte source file must prove COMPLETE through every final-source gate.

An empty ``__init__.py`` has no line, so the closure selects no unit for it and no
body goes on the wire. Its revision (sha256 of the empty bytes) is the whole proof.
Before the fix an empty entry target left the plan with zero selected units, so the
receipt stayed UNKNOWN and the run stopped before its first request.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from _stateframe_home import isolated_stateframe_home
from engine_run import run_monitored
from stateframe.scripted_endpoint import Script, serve

from kernel_request_evidence_closure import (
    build_source_closure, record_final_source_delivery, record_source_delivery,
    source_revision,
)
from kernel_request_source_window import (
    source_delivery_satisfied, verify_final_source_delivery,
)

EMPTY_REVISION = source_revision("")


def _plan(tmp_path: Path, files: dict[str, str], target: str) -> dict[str, object]:
    root = tmp_path / "repo"
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    plan = build_source_closure(
        root, tuple(files), "Implement the item in the README.", target_path=target,
    ).as_dict()
    plan["workspace_authorization"] = {"scope": str(root), "generation": "0"}
    plan["deferred_paths"] = []
    return plan


def _verify(plan, messages=()):
    return verify_final_source_delivery(
        plan, list(messages), payload={"messages": list(messages)}, sources={})


@pytest.mark.parametrize("target", ("pkg/__init__.py", "pkg/empty_module.py"))
def test_empty_target_receipt_is_complete_and_verified(tmp_path, target):
    plan = _plan(tmp_path, {"pkg/__init__.py": "", "pkg/empty_module.py": ""}, target)
    assert plan["units"] == []
    receipt = _verify(plan)
    assert receipt["proof_status"] == "VERIFIED"
    assert receipt["status"] == "COMPLETE"
    assert source_delivery_satisfied(receipt)
    row = receipt["paths"][target]
    assert row["status"] == "COMPLETE" and row["empty_source"] is True
    assert row["revision"] == EMPTY_REVISION
    assert plan["final_delivery_status"] == "COMPLETE"


def test_empty_target_beside_nonempty_unselected_sources_is_complete(tmp_path):
    plan = _plan(tmp_path, {"pkg/__init__.py": "", "pkg/core.py": "def f():\n    return 1\n"},
                 "pkg/__init__.py")
    receipt = _verify(plan)
    assert receipt["status"] == "COMPLETE" and source_delivery_satisfied(receipt)


def test_header_only_wire_message_for_empty_target_still_proves(tmp_path):
    plan = _plan(tmp_path, {"pkg/__init__.py": ""}, "pkg/__init__.py")
    receipt = _verify(plan, [{"role": "user", "content": "[Current source: pkg/__init__.py]\n"}])
    assert receipt["proof_status"] == "VERIFIED"
    assert source_delivery_satisfied(receipt)


def test_zero_byte_source_path_needs_matching_bytes_and_revision(tmp_path):
    from kernel_request_evidence_closure import zero_byte_source_path

    plan = _plan(tmp_path, {"pkg/__init__.py": "", "pkg/core.py": "x = 1\n"}, "pkg/core.py")
    assert zero_byte_source_path(plan, "pkg/__init__.py") is True
    assert zero_byte_source_path(plan, "pkg/core.py") is False
    assert zero_byte_source_path(plan, "pkg/missing.py") is False
    tampered = copy.deepcopy(plan)
    tampered["revisions"]["pkg/__init__.py"] = source_revision("x = 1\n")
    assert zero_byte_source_path(tampered, "pkg/__init__.py") is False
    forged = copy.deepcopy(plan)
    forged["source_contents"]["pkg/core.py"] = ""
    assert zero_byte_source_path(forged, "pkg/core.py") is False  # revision is not sha256("")


def test_tampered_empty_revision_is_not_complete(tmp_path):
    plan = _plan(tmp_path, {"pkg/__init__.py": ""}, "pkg/__init__.py")
    plan["revisions"]["pkg/__init__.py"] = source_revision("changed\n")
    receipt = _verify(plan)
    assert receipt["status"] != "COMPLETE"
    assert not source_delivery_satisfied(receipt)


def test_nonempty_target_with_nothing_delivered_is_still_not_complete(tmp_path):
    """The proof is not weakened: zero units never completes a non-empty target."""
    plan = _plan(tmp_path, {"pkg/__init__.py": "", "pkg/core.py": "x = 1\n"}, "pkg/core.py")
    plan["units"] = []
    receipt = _verify(plan)
    assert receipt["status"] != "COMPLETE"
    assert not source_delivery_satisfied(receipt)


def test_empty_sibling_cannot_complete_a_plan_whose_unit_is_undelivered(tmp_path):
    plan = _plan(tmp_path, {"pkg/__init__.py": "", "pkg/core.py": "x = 1\n"}, "pkg/core.py")
    plan["units"] = [{
        "path": "pkg/core.py", "start_line": 1, "end_line": 1, "role": "target_reservation",
        "reason": "source_reservation", "priority": 1, "handle": "unit:t:1-1",
    }]
    receipt = _verify(plan)
    assert receipt["paths"]["pkg/core.py"]["status"] != "COMPLETE"
    assert receipt["status"] != "COMPLETE"


def test_record_source_delivery_marks_only_zero_byte_paths(tmp_path):
    plan = _plan(tmp_path, {"pkg/__init__.py": "", "pkg/core.py": "x = 1\n"}, "pkg/core.py")
    assert record_source_delivery(plan, "pkg/__init__.py", ())["empty_source"] is True
    assert record_source_delivery(plan, "pkg/core.py", ())["empty_source"] is False
    receipt = record_final_source_delivery(plan, {}, wire_proof_verified=True)
    assert receipt["status"] != "COMPLETE"  # the non-empty target is PARTIAL


@pytest.mark.parametrize("text", ("\n", "   \n", "\n\n"))
def test_whitespace_only_target_still_reserves_a_unit_and_proves(tmp_path, text):
    """Newline/whitespace-only files have lines: they keep the ordinary unit path."""
    from kernel_request_evidence_closure import zero_byte_source_path
    from kernel_request_source_window import _ensure_target_unit, reserve_source_unit

    plan = _plan(tmp_path, {"pkg/__init__.py": text}, "pkg/__init__.py")
    assert zero_byte_source_path(plan, "pkg/__init__.py") is False
    reservation = reserve_source_unit(
        text, path="pkg/__init__.py", revision=source_revision(text), max_bytes=100_000,
        non_source_bytes=1_000, required_ranges=(), signature_ranges=()).as_dict()
    _ensure_target_unit(plan, "pkg/__init__.py", reservation)
    assert [unit["path"] for unit in plan["units"]] == ["pkg/__init__.py"]
    assert not source_delivery_satisfied(_verify(plan))  # undelivered unit never completes


# ---------------------------------------------------------------------------
# End to end: the run reaches its first request with empty files in the repo.
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RUN_TIMEOUT_SECONDS = 600
_SWITCHES = (
    pytest.param({}, id="default"),
    pytest.param({"STATEFRAME_FINAL_CACHE_LAYOUT": "on"}, id="final_cache_layout"),
    pytest.param({"STATEFRAME_STABLE_WIRE": "on"}, id="stable_wire"),
)


def _empty_file_repo(root: Path) -> None:
    (root / "pkg").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "pkg/__init__.py").write_text("")
    (root / "pkg/empty_module.py").write_text("")
    (root / "pkg/core.py").write_text(
        "from pkg import empty_module\n\n\ndef greet(name):\n    return 'hello ' + name\n")
    (root / "tests/test_core.py").write_text(
        "from pkg.core import greet\n\n\ndef test_greet():\n    assert greet('a') == 'hello a'\n")
    (root / "README.md").write_text("Make greet shout.\n")


@pytest.mark.host_sandbox
@pytest.mark.parametrize("target", ("pkg/__init__.py", "pkg/empty_module.py"))
@pytest.mark.parametrize("switches", _SWITCHES)
def test_scripted_run_with_empty_source_files_sends_its_first_request(
        tmp_path, switches, target):
    import subprocess

    repo = tmp_path / "repo"
    _empty_file_repo(repo)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "i"],
                   cwd=repo, check=True)
    server = serve(Script({"steps": [{"tool": "done", "arguments": {}}]}))
    run = tmp_path / "run"
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("STATEFRAME_")}
    environment.update({"PYTHONPATH": str(ROOT / "src"), **switches})
    environment["STATEFRAME_HOME"] = isolated_stateframe_home()
    try:
        result = run_monitored(
            [sys.executable, "-m", "stateframe", "run", "--repo", str(repo),
             "--task", "Implement the item in the README.",
             "--endpoint", f"http://127.0.0.1:{server.server_port}/v1",
             "--model", "scripted", "--run-dir", str(run), "--target", target,
             "--test-python", sys.executable, "--max-turns", "2", "--quiet"],
            env=environment, capture_output=True, text=True,
            expected_seconds=RUN_TIMEOUT_SECONDS, run_dir=run)
    finally:
        server.shutdown()
        server.server_close()
    log = (run / "engine.log").read_text() if (run / "engine.log").exists() else ""
    text = log + result.stdout + result.stderr
    assert "source proof" not in text, text[-3000:]
    assert "FrameRequestAllocationError" not in text, text[-3000:]
    assert len(list((run / "wire").glob("*.request.json"))) >= 1, text[-3000:]
