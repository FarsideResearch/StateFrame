"""Pre-run gate for the goal-only parallel subframe schema: OFF identity, ON wire stability, switch matrix."""
from __future__ import annotations

import json
import os
import shlex
import shutil
import sys
from pathlib import Path

import pytest
from _stateframe_home import isolated_stateframe_home
from engine_run import run_monitored

from stateframe.engine.kernel_parallel import PARALLEL_TOOL, _UNIT
from stateframe.scripted_endpoint import Script, serve

ROOT = Path(__file__).resolve().parents[1]
SHORT_FORM_SUBFRAMES = [
    {"id": "a", "goal": "Change the value in a.py.", "owns": ["a.py"]},
    {"id": "b", "goal": "Change the value in b.py.", "owns": ["b.py"]},
]


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "a.py").write_text("VALUE_A = 1\n", encoding="utf-8")
    (repo / "b.py").write_text("VALUE_B = 2\n", encoding="utf-8")
    (repo / "tests/test_a.py").write_text("from a import VALUE_A\n\ndef test_a():\n    assert VALUE_A\n",
                                          encoding="utf-8")
    (repo / "tests/test_b.py").write_text("from b import VALUE_B\n\ndef test_b():\n    assert VALUE_B\n",
                                          encoding="utf-8")
    return repo


def _run(tmp_path: Path, steps: list[dict[str, object]], *, extra_args: tuple[str, ...],
         allocator: str | None, name: str) -> tuple[Path, list[dict[str, object]]]:
    """Drive the real CLI against a scripted endpoint and return the run dir plus parsed wire requests."""
    repo = _repo(tmp_path / name)
    home = tmp_path / name / "home"
    home.mkdir(parents=True)
    run = tmp_path / name / "run"
    script = Script({"model": "scripted", "tokenize": {"mode": "ok"}, "slots": [65536], "steps": steps})
    server = serve(script)
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("STATEFRAME_")}
    environment.update({
        "PYTHONPATH": str(ROOT / "src"), "PYTHONHASHSEED": "0", "HOME": str(home),
        "STATEFRAME_HOME": isolated_stateframe_home(),
    })
    if allocator is not None:
        environment["STATEFRAME_WORKING_SET_ALLOCATOR"] = allocator
    command = [sys.executable, "-m", "stateframe", "run", "--repo", str(repo),
               "--task", "Change the values in a.py and b.py, each checked by its own test.",
               "--endpoint", f"http://127.0.0.1:{server.server_port}/v1",
               "--model", "scripted", "--run-dir", str(run),
               "--test-python", shutil.which("python3") or sys.executable,
               # The outer check runs inside bwrap, which cannot exec a uv venv interpreter
               # on the CI host; a system binary keeps the check path real without that dependency.
               "--test-cmd", shlex.join((shutil.which("true") or "true",)),
               # --max-turns is also the parallel request allowance; 0 means no cap, as in the e2e harness.
               "--max-turns", "0", "--quiet", *extra_args]
    try:
        completed = run_monitored(command, cwd=repo, env=environment, run_dir=run,
                                  expected_seconds=300, step=f"{name} parallel goal pre-run",
                                  check=False)
    finally:
        server.shutdown()
        server.server_close()
    wires = [json.loads(path.read_bytes())
             for path in sorted((run / "wire").glob("*.request.json"))]
    assert wires, completed.stdout[-3000:] + completed.stderr[-3000:]
    return run, wires


def _parallel_entries(request: dict[str, object]) -> list[dict[str, object]]:
    return [tool for tool in request["tools"]
            if isinstance(tool, dict) and tool.get("function", {}).get("name") == "parallel"]


# T2: default (parallel omitted, and --parallel off) must not offer the tool, so the first request bytes are unchanged.
@pytest.mark.host_sandbox
@pytest.mark.parametrize("extra_args", ((), ("--parallel", "off")), ids=("omitted", "off"))
def test_t2_parallel_off_first_request_has_no_parallel_tool(tmp_path: Path, extra_args) -> None:
    steps = [{"tool": "done", "required": True,
              "arguments": {"summary": "Checked the values."}}]
    run, wires = _run(tmp_path, steps, extra_args=extra_args, allocator=None,
                      name=f"t2-{len(extra_args)}")
    assert wires, "no provider request captured"
    first = wires[0]
    assert _parallel_entries(first) == []
    serialized = json.dumps(first["tools"])
    assert json.dumps(_UNIT) not in serialized
    assert "{id, goal, owns}" not in serialized
    assert not (run / "parallel").exists()


# T3: with parallel on, the tool definition is byte-identical across sends of one run.
@pytest.mark.host_sandbox
def test_t3_parallel_on_tool_definition_stable_across_sends(tmp_path: Path) -> None:
    steps = [
        {"tool": "read", "required": True, "arguments": {"path": "a.py"}},
        {"tool": "read", "required": True, "arguments": {"path": "b.py"}},
        {"tool": "done", "required": True, "arguments": {"summary": "Read both files."}},
    ]
    _run_dir, wires = _run(tmp_path, steps, extra_args=("--parallel", "2"), allocator="on", name="t3")
    assert len(wires) >= 2, "multi-turn run produced fewer than two sends"
    definitions = [_parallel_entries(request) for request in wires[:2]]
    assert all(len(entries) == 1 for entries in definitions), "parallel tool missing on a send"
    assert json.dumps(definitions[0], sort_keys=False) == json.dumps(definitions[1], sort_keys=False)
    # The wire copy is the provider-normalized tool (it adds additionalProperties); the goal short form must still be the required set.
    assert definitions[0][0]["function"]["parameters"]["properties"]["subframes"]["items"]["required"] == [
        "id", "goal", "owns"]
    assert definitions[0][0]["function"]["name"] == PARALLEL_TOOL["function"]["name"]


# T1: parallel on x working-set allocator on/off; the short form is admitted and the schema offered says so.
@pytest.mark.host_sandbox
@pytest.mark.parametrize("allocator", ("on", "off"), ids=("allocator-on", "allocator-off"))
def test_t1_short_form_subframes_admitted_under_allocator_matrix(tmp_path: Path, allocator: str) -> None:
    steps = [
        {"tool": "parallel", "required": True, "arguments": {"subframes": SHORT_FORM_SUBFRAMES}},
        {"tool": "done", "required": True, "arguments": {"summary": "Both values checked."}},
    ]
    run, wires = _run(tmp_path, steps, extra_args=("--parallel", "2"), allocator=allocator,
                      name=f"t1-{allocator}")
    assert wires, "no provider request captured"
    entries = _parallel_entries(wires[0])
    assert len(entries) == 1, "parallel tool not offered in the first send"
    assert entries[0]["function"]["parameters"]["properties"]["subframes"]["items"]["required"] == [
        "id", "goal", "owns"]
    roots = sorted(path for path in (run / "parallel").iterdir()
                   if path.is_dir() and (path / "intent.json").is_file())
    assert len(roots) == 1, f"expected one parallel call root, found {len(roots)}"
    intent = json.loads((roots[0] / "intent.json").read_bytes())
    admission = intent["admission"]
    assert admission["status"] == "ADMITTED", (
        f"short-form call refused under allocator={allocator}: {admission.get('refusal')}")
    assert [unit["id"] for unit in admission["units"]] == ["a", "b"]
    for unit in ("a", "b"):
        assert list((roots[0] / "units" / unit).glob("unit-spec.json")), (
            f"unit {unit} was not launched under allocator={allocator}")
