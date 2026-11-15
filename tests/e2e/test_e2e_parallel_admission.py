"""Wire-level admission refusals and user-facing cap names."""
from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

from barrier_endpoint import BarrierEndpoint
from e2e_support import (
    assert_shipped_wire,
    parallel_call_roots,
    parent_wire_files,
    read_json,
    run_cli,
    summary_for,
    tool_step,
    wire_message_text,
)
from stateframe.engine.kernel_parallel_admission import DirectorySnapshot


pytestmark = pytest.mark.host_sandbox


def _unit(unit_id: str, path: str, *, exports: list[dict] | None = None,
          consumes: list[dict] | None = None) -> dict:
    return {
        "id": unit_id,
        "outcome": f"Update {path}",
        "owns": [path],
        "creates": [],
        "exports": exports or [],
        "consumes": consumes or [],
        "reads": [],
        "requirements": [],
        "check": {"tests": []},
    }


def _parallel(units: list[dict]) -> dict:
    return tool_step("parallel", {"subframes": units})


def _assert_no_launches(call_root: Path) -> None:
    assert not list((call_root / "units").glob("*/unit-spec.json"))
    assert not list((call_root / "attempt-*").glob("**/rq/*.json"))


def _assert_parent_wire(run_dir: Path) -> None:
    requests = parent_wire_files(run_dir)
    assert_shipped_wire(requests)
    for path in requests:
        request = read_json(path)
        messages = request.get("messages")
        assert isinstance(messages, list), f"parent F4 wire request has no messages: {path}"
        contents = [item.get("content", "") for item in messages if isinstance(item, dict)]
        assert any(isinstance(content, str) and content.startswith("[StateFrame]\n")
                   and '"schema":"stateframe-f4/1"' in content.replace(" ", "")
                   for content in contents), f"shipped F4 state absent from parent wire bytes: {path}"
        assert any(isinstance(tool, dict) and isinstance(tool.get("function"), dict)
                   and tool["function"].get("name") == "parallel"
                   for tool in request.get("tools", [])), f"parallel schema absent from parent wire: {path}"


def test_overlap_undeclared_coupling_cycle_and_oversize_refuse_without_launches(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "pkg/a.py").write_text("def left():\n    return 1\n")
    (repo / "pkg/b.py").write_text("from pkg.a import left\n\ndef right():\n    return left()\n")
    (repo / "pkg/c.py").write_text("VALUE = 1\n")
    (repo / "pkg/large.py").write_text("VALUE = 1\n#" + "x" * 2048 + "\n")
    (repo / "tests/test_smoke.py").write_text("def test_smoke():\n    assert True\n")
    original_manifest = DirectorySnapshot(repo, 0).manifest_digest

    signature = {"kind": "function", "params": [], "returns": None, "decorators": []}
    left_export = {"id": "left.result", "path": "pkg/a.py", "symbol": "left",
                   "signature": signature}
    right_export = {"id": "right.result", "path": "pkg/b.py", "symbol": "right",
                    "signature": signature}
    calls = [
        _parallel([_unit("left", "pkg/a.py"), _unit("right", "pkg/a.py")]),
        # An undeclared import with one projectable owner is not refused: the shipped plan
        # completion links it into an explicit export and consume (see
        # test_e2e_parallel_live2 and tests/test_parallel_live_plans.py), and admission
        # refuses UNDECLARED_COUPLING only for a raw contract (tests/test_parallel_admission.py).
        _parallel([
            _unit("left", "pkg/a.py", exports=[left_export],
                  consumes=[{"export_id": "right.result"}]),
            _unit("right", "pkg/b.py", exports=[right_export],
                  consumes=[{"export_id": "left.result"}]),
        ]),
        _parallel([_unit("large", "pkg/large.py"), _unit("small", "pkg/c.py")]),
        tool_step("done", {"summary": "Admission refusals are recorded."}),
    ]
    parent_steps = [
        {"tool": step["tool"], "arguments": step["arguments"]}
        for step in calls
    ]
    with BarrierEndpoint(parent_steps, {}, slots=2) as endpoint:
        completed = run_cli(
            isolated_engine_python, endpoint.url, repo, tmp_path / "run",
            "Record why these requested subframes cannot be admitted.",
            parallel="2", arguments=("--parallel-max-subframes", "4",
                                      "--parallel-max-owned-bytes", "1024"),
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        endpoint.assert_consumed()

    run_dir = tmp_path / "run"
    summary = summary_for(run_dir)
    assert summary["parallel"]["totals"]["subframes"] == 0
    roots = parallel_call_roots(run_dir)
    assert len(roots) == 3
    # Call roots are named by a content hash, so their sort order is not the call order.
    results = {}
    for root in roots:
        item = read_json(root / "result.json")
        assert item["admission"]["status"] == "REFUSED"
        results[item["admission"]["refusal"]["reason"]] = item
    assert sorted(results) == ["DEPENDENCY_CYCLE", "OWNERSHIP_OVERLAP", "UNIT_CONTEXT_TOO_LARGE"]

    overlap = results["OWNERSHIP_OVERLAP"]["admission"]["refusal"]
    assert overlap["paths"] == ["pkg/a.py"]
    assert overlap["members"] == ["left", "right"]
    assert results["OWNERSHIP_OVERLAP"]["admission"]["manifest_digest"] == original_manifest

    cycle = results["DEPENDENCY_CYCLE"]["admission"]["refusal"]
    assert cycle["members"] == ["left", "right"]

    oversized = results["UNIT_CONTEXT_TOO_LARGE"]["admission"]["refusal"]
    assert oversized["paths"] == ["pkg/large.py"]
    assert f"owned_source_bytes={(repo / 'pkg/large.py').stat().st_size}" in oversized["detail"]
    assert "--parallel-max-owned-bytes" in oversized["detail"]
    for call_root in roots:
        _assert_no_launches(call_root)
    assert DirectorySnapshot(repo, 0).manifest_digest == original_manifest
    _assert_parent_wire(run_dir)
    assert all(request["marker"] == "parent" for request in endpoint.requests)


def test_subframe_cap_refusal_names_the_public_subframe_flag(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "tests").mkdir()
    units = []
    parent_steps = []
    for index in range(5):
        unit_id = f"part{index}"
        path = f"pkg/{unit_id}.py"
        (repo / path).write_text(f"VALUE_{index} = 0\n")
        units.append(_unit(unit_id, path))
    (repo / "tests/test_smoke.py").write_text("def test_smoke():\n    assert True\n")
    parent_steps.extend([
        {"tool": "parallel", "arguments": {"subframes": units}},
        {"tool": "done", "arguments": {"summary": "The subframe cap is recorded."}},
    ])
    with BarrierEndpoint(parent_steps, {}, slots=2) as endpoint:
        completed = run_cli(
            isolated_engine_python, endpoint.url, repo, tmp_path / "run",
            "Refuse work above the configured subframe cap.", parallel="2",
            arguments=("--parallel-max-subframes", "4", "--parallel-max-paths-total", "16"),
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        endpoint.assert_consumed()

    run_dir = tmp_path / "run"
    result, = [read_json(path / "result.json") for path in parallel_call_roots(run_dir)]
    refusal = result["admission"]["refusal"]
    assert refusal["reason"] == "TOO_MANY_UNITS"
    assert "max_subframes=4" in refusal["detail"]
    assert "--parallel-max-subframes" in refusal["detail"]
    assert "STATEFRAME_PARALLEL_MAX_SUBFRAMES" in refusal["detail"]
    observation = result["observation"]
    assert observation == (
        "REFUSED TOO_MANY_SUBFRAMES: max_subframes=4 "
        "(--parallel-max-subframes / STATEFRAME_PARALLEL_MAX_SUBFRAMES)"
        "; subframe all; field subframes; "
        "fix: remove subframes until the named maximum is met, or raise that maximum\n"
        + "\n".join(f"Plan part{index}: owns=pkg/part{index}.py; check=NOT_RUN; consumes=-"
                    for index in range(5)))
    assert "unit" not in observation.casefold()
    assert any(observation in wire_message_text(path) for path in parent_wire_files(run_dir))
    _assert_no_launches(parallel_call_roots(run_dir)[0])
    _assert_parent_wire(run_dir)


_DERIVED_DIGEST = re.compile(
    rb'(?P<prefix>\\?"[a-z_]*(?:schema|tool|surface|allocation|system_inputs)[a-z_]*digest\\?"'
    rb'\s*:\s*\\?"(?:sha256:)?)[0-9a-f]{64}(?=\\?")')
_EDIT_RECEIPT = re.compile(rb'(?P<prefix>\\?"edit_receipt\\?"\s*:\s*\\?")[0-9a-f]{20}(?=\\?")')


def _comparable_no_call_wire(path: Path, run_dir: Path, endpoint: str) -> dict:
    raw = path.read_bytes()  # request identity is the exact wire bytes
    raw = raw.replace(str(run_dir.resolve()).encode(), b"<RUN_DIR>").replace(
        endpoint.encode(), b"<ENDPOINT>")
    raw = _DERIVED_DIGEST.sub(lambda match: match.group("prefix") + b"<DERIVED_DIGEST>", raw)
    raw = _EDIT_RECEIPT.sub(lambda match: match.group("prefix") + b"<EDIT_RECEIPT>", raw)
    return json.loads(raw)


_PARALLEL_GUIDANCE = ("Independent parts (separate files + separate tests): dispatch them "
                      "together with `parallel` before reading their sources.\n")
_SUBFRAME_RECEIPT_DESCRIPTION = (
    "Retrieve a revision-bound check receipt with next_offset, or an authenticated subframe "
    "receipt page with next_handle. Follow next_handle to read all subframe pages.")
_PLAIN_RECEIPT_DESCRIPTION = (
    "Retrieve lossless public check stdout, stderr, or assertion context from a revision-bound "
    "receipt handle. Pages are bounded; use next_offset.")


def _off_receipt_description(tool: dict) -> dict:
    function = tool["function"]
    if function["name"] != "read_check_receipt":
        return tool
    assert function["description"] == _SUBFRAME_RECEIPT_DESCRIPTION
    return {**tool, "function": {**function, "description": _PLAIN_RECEIPT_DESCRIPTION}}


def _without_parallel_additions(message: dict) -> dict:
    content = message.get("content")
    if not isinstance(content, str):
        return message
    return {**message, "content": content.replace(_PARALLEL_GUIDANCE, "").replace('"parallel",', "")}


def test_auto_mode_without_parallel_call_only_adds_schema_and_derived_digests(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    repo = tmp_path / "repo"
    source = root / "examples" / "discount_bug"
    import shutil
    shutil.copytree(source, repo)
    script = json.loads((root / "examples" / "discount_bug.script.json").read_bytes())
    runs = {}
    for mode in ("off", "auto"):
        run_dir = tmp_path / f"run-{mode}"
        with BarrierEndpoint(script["steps"], {}) as endpoint:
            completed = run_cli(
                isolated_engine_python, endpoint.url, repo, run_dir,
                "Fix the discount calculation using the ordinary parent workflow.",
                parallel=mode, timeout=900,
            )
            assert completed.returncode == 0, completed.stdout + completed.stderr
            endpoint.assert_consumed()
            requests = parent_wire_files(run_dir)
            assert_shipped_wire(requests)
            runs[mode] = (run_dir, endpoint.url, requests)

    off_dir, off_endpoint, off_requests = runs["off"]
    auto_dir, auto_endpoint, auto_requests = runs["auto"]
    assert len(off_requests) == len(auto_requests) > 0
    for off_path, auto_path in zip(off_requests, auto_requests):
        off = _comparable_no_call_wire(off_path, off_dir, off_endpoint)
        auto = _comparable_no_call_wire(auto_path, auto_dir, auto_endpoint)
        off_tools = {tool["function"]["name"] for tool in off["tools"]}
        auto_tools = {tool["function"]["name"] for tool in auto["tools"]}
        assert auto_tools - off_tools == {"parallel"}
        assert off_tools - auto_tools == set()
        auto["tools"] = [_off_receipt_description(tool) for tool in auto["tools"]
                         if tool["function"]["name"] != "parallel"]
        # Auto mode adds exactly three things beside the schema: the parent guidance sentence,
        # the tool name in the state frame's menu lists, and the receipt-reader description
        # that mentions subframe pages.  Remove only those, then require byte equality.
        auto["messages"] = [_without_parallel_additions(message) for message in auto["messages"]]
        assert auto == off
        planted = {**auto, "model": "planted-different-model"}
        assert planted != off
