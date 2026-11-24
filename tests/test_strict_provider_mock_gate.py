"""A DeepInfra-shaped provider must admit a multi-send source-window run."""
from __future__ import annotations

import gzip
import io
import json
import os
import subprocess
import venv
from pathlib import Path
from types import SimpleNamespace

import pytest

from _journal import measurement_events
from engine_run import run_monitored
from kernel_completion_owner import (
    RequestActionContractError, assert_request_action_contract,
    establish_request_action_contract,
)
from kernel_measurement import RequestCapture
from kernel_read_fixes import ContinuationLimits
from kernel_request_reprepare import SourceRevisionReplanRefused, source_replan_preparation
from fixtures.barrier_endpoint import BarrierEndpoint
from tests.test_request_seal_wire_form import _run_scripted, assert_sealed_wire
from stateframe.scripted_endpoint import _strict_openai_error


ROOT = Path(__file__).resolve().parents[1]


class _StrictParallelEndpoint(BarrierEndpoint):
    """OpenAI-shaped mock that rejects llama-only fields and returns 404 probes."""

    def __init__(self, parent_steps: list[dict[str, object]],
                 subframe_steps: dict[str, list[dict[str, object]]]) -> None:
        super().__init__(parent_steps, subframe_steps)
        self.probe_statuses: list[tuple[str, int]] = []
        self.strict_errors: list[str] = []

    def _handler(self):
        base = super()._handler()
        endpoint = self

        class Handler(base):
            def do_GET(self) -> None:  # noqa: N802
                path = self.path.rstrip("/")
                if path in {"/props", "/slots"}:
                    with endpoint._condition:
                        endpoint.probe_statuses.append((path, 404))
                    self._send(404, {"error": {"message": "not found"}})
                    return
                super().do_GET()

            def do_POST(self) -> None:  # noqa: N802
                if self.path.rstrip("/") == "/v1/chat/completions":
                    raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                    decoded = (gzip.decompress(raw)
                               if self.headers.get("Content-Encoding") == "gzip" else raw)
                    error = _strict_openai_error(json.loads(decoded))
                    if error:
                        with endpoint._condition:
                            endpoint.strict_errors.append(error)
                        self._send(400, {"error": {"message": error}})
                        return
                    self.rfile = io.BytesIO(raw)
                super().do_POST()

        return Handler


@pytest.mark.host_sandbox
def test_strict_provider_six_sends_seal_provider_bytes(tmp_path: Path) -> None:
    steps: list[dict[str, object]] = [
        {"tool": "read", "arguments": {"path": path}}
        for path in ("app.py", "tests/test_value.py", "app.py", "tests/test_value.py")
    ] + [{"tool": "run_tests", "arguments": {"target": "tests/test_value.py"}},
         {"tool": "done", "arguments": {"summary": "VALUE is seven and tests pass."}}]
    process, run, script = _run_scripted(tmp_path, strict=True, steps=steps)
    output = process.stdout + process.stderr
    assert len(script.requests) >= 6, output
    assert (run / "tagged-source-window").is_file(), output
    assert "provider dialect: openai-strict" in output
    assert "measurement body/projection/provider identity changed" not in output
    assert process.returncode == 0, output
    script.assert_consumed()
    assert not script.strict_openai_errors
    assert_sealed_wire(run, minimum=6)


def _run_forced_subframes(
    tmp_path: Path, *, profile_environment: dict[str, str] | None = None,
) -> None:
    plan: dict[str, object] = {"subframes": [
        {"id": "left", "goal": "Update the left adapter.", "owns": ["left.py"],
         "creates": [], "reads": [], "exports": [], "consumes": [], "requirements": [],
         "check": {"tests": ["tests/test_left.py"]}},
        {"id": "right", "goal": "Update the right adapter.", "owns": ["right.py"],
         "creates": [], "reads": [], "exports": [], "consumes": [], "requirements": [],
         "check": {"tests": ["tests/test_right.py"]}},
    ]}
    def step(tool: str, arguments: dict[str, object]) -> dict[str, object]:
        return {"tool": tool, "arguments": arguments}

    # The forced manifest dispatches parallel internally; the provider resumes at done.
    parent_steps = [step("done", {
        "summary": "Both adapters are updated.",
    })]
    subframe_steps = {
        "left": [
            step("read", {"path": "left.py"}),
            step("str_replace", {"path": "left.py", "old_str": "VALUE = 1",
                                  "new_str": "VALUE = 3"}),
        ],
        "right": [
            step("read", {"path": "right.py"}),
            step("str_replace", {"path": "right.py", "old_str": "VALUE = 1",
                                  "new_str": "VALUE = 4"}),
        ],
    }
    endpoint = _StrictParallelEndpoint(parent_steps, subframe_steps)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "left.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "right.py").write_text("VALUE = 1\n", encoding="utf-8")
    test_root = repo / "tests"
    test_root.mkdir()
    (test_root / "test_left.py").write_text(
        "from left import VALUE\n\n\ndef test_left_value():\n    assert VALUE == 3\n",
        encoding="utf-8",
    )
    (test_root / "test_right.py").write_text(
        "from right import VALUE\n\n\ndef test_right_value():\n    assert VALUE == 4\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "forced-plan.json"
    manifest.write_text(json.dumps(plan, separators=(",", ":")), encoding="utf-8")
    run = tmp_path / "run"
    engine_environment = tmp_path / "engine-environment"
    venv.EnvBuilder(system_site_packages=True).create(engine_environment)
    engine_python = engine_environment / "bin" / "python"
    installation = subprocess.run(
        [str(engine_python), "-I", "-m", "pip", "install",
         "--disable-pip-version-check", "--no-index", "--no-build-isolation", "--no-deps",
         "--no-input", "--force-reinstall", str(ROOT)],
        capture_output=True, text=True, timeout=180, check=False,
    )
    assert installation.returncode == 0, installation.stdout + installation.stderr
    # Subframes lease provider slots under the user's home, so keep that home writable and isolated.
    home = tmp_path / "home"
    home.mkdir()
    environment = {name: value for name, value in os.environ.items()
                   if not name.startswith(("STATEFRAME_", "REACT_"))}
    environment.update(profile_environment or {})
    environment.update({
        "HOME": str(home), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
        "STATEFRAME_HOME": str(tmp_path / "state-home"),
        "STATEFRAME_PARALLEL_FORCE_MANIFEST": str(manifest),
    })
    with endpoint:
        command = [
            str(engine_python), "-m", "stateframe", "run", "--repo", str(repo),
            "--task", "Update left.py and right.py in separate subframes.",
            "--endpoint", endpoint.url, "--model", "scripted", "--run-dir", str(run),
            "--max-turns", "32", "--parallel", "2", "--shell", "off",
            "--python", str(engine_python), "--test-python", str(engine_python), "--quiet",
        ]
        process = run_monitored(
            command, cwd=ROOT, env=environment, capture_output=True, text=True,
            expected_seconds=600, run_dir=run, check=False,
        )
    output = process.stdout + process.stderr
    traceback = run / "child-traceback.txt"
    if traceback.is_file():
        output += traceback.read_text(encoding="utf-8")
    assert "provider dialect: openai-strict" in output
    assert not endpoint.strict_errors
    probe_paths = {path for path, status in endpoint.probe_statuses if status == 404}
    assert {"/props", "/slots"}.issubset(probe_paths)
    child_requests = {
        unit_id: [row for row in endpoint.requests if row.get("marker") == unit_id]
        for unit_id in ("left", "right")
    }
    assert all(len(requests) >= 2 for requests in child_requests.values()), output
    assert all(len(list((run / "parallel").glob(
        f"*/units/{unit_id}/wire/*.request.json"))) >= 2 for unit_id in child_requests), output
    assert all(_strict_openai_error(row["request"]) is None
               for row in endpoint.requests), output
    assert process.returncode == 0, output + f"\nMock errors: {endpoint.errors!r}"
    endpoint.assert_consumed()


@pytest.mark.host_sandbox
def test_strict_provider_forced_subframes_send_multiple_child_requests(tmp_path: Path) -> None:
    _run_forced_subframes(tmp_path)


@pytest.mark.host_sandbox
def test_strict_provider_charged_source_replan_sends_replacement(tmp_path: Path) -> None:
    steps: list[dict[str, object]] = [
        {"tool": "read", "arguments": {"path": "app.py"}},
        {"tool": "done", "arguments": {"summary": "VALUE is seven."}},
    ]
    process, run, script = _run_scripted(
        tmp_path, strict=True, steps=steps, inject_allocation_replan=True,
    )
    output = process.stdout + process.stderr
    traceback = run / "child-traceback.txt"
    if traceback.is_file():
        output += traceback.read_text(encoding="utf-8")
    assert (run / "injected-replan").is_file(), output
    assert (run / "injected-replan").read_text() == "1", output
    assert process.returncode == 0, output
    events = measurement_events(run)
    started = [row for row in events if row.get("kind") == "source_revision_replan_started"]
    adopted = [row for row in events if row.get("kind") == "source_revision_reprepared"]
    assert len(started) == len(adopted) == 1, output
    original_id = started[0]["request_id"]
    replacement_id = adopted[0]["replacement_request_id"]
    assert replacement_id != original_id
    submitted = [row for row in events if row.get("kind") == "request_submitted"]
    assert sum(row.get("request_id") == replacement_id for row in submitted) == 1
    assert all(row.get("request_id") != original_id for row in submitted)
    assert sum(row.get("kind") == "budget_unit_charged" and row.get("step") == 0
               for row in events) == 1
    assert not any(row.get("kind") == "source_revision_replan_refused" for row in events)
    assert len(script.requests) >= len(steps), output
    script.assert_consumed()
    assert_sealed_wire(run, minimum=len(steps))


def test_charged_replan_contract_requires_distinct_request_identity() -> None:
    class Journal:
        task_id = "charged-replan-contract"

        def emit(self, *_args: object, **_kwargs: object) -> None:
            pass

        def request_records(self, _request_id: str) -> list[object]:
            return []

    journal = Journal()
    runtime = SimpleNamespace(journal=journal, owner=SimpleNamespace(root="."),
                              system_outcome=None)
    budget = ContinuationLimits(32, mode="response", output_reservation_tokens=24000)
    runtime.continuation_limits = budget
    tools: list[dict[str, object]] = [{"type": "function", "function": {
        "name": "read", "description": "read",
        "parameters": {"type": "object", "properties": {}},
    }}]
    original = RequestCapture(journal, 1, "snapshot")
    assert original.request_id == RequestCapture(
        journal, 1, "snapshot",
        identity_content={"task_id": journal.task_id, "purpose": ""},
    ).request_id
    original_messages = [{"role": "user", "content": budget.render_continuation_limits()}]
    original_contract = establish_request_action_contract(
        runtime, original_messages, tools, request_id=original.request_id,
        preserve_input_surface=True, framefit5_preassembly=True,
    )
    assert budget.charge_response(step=1)
    assert budget.continuation_model_decisions_used == 1
    replacement_messages = [{"role": "user", "content": budget.render_continuation_limits()}]
    stale = establish_request_action_contract(
        runtime, replacement_messages, tools, request_id=original.request_id,
        preserve_input_surface=True, framefit5_preassembly=True,
    )
    assert stale["budget_wire_digest"] == original_contract["budget_wire_digest"]
    with pytest.raises(RequestActionContractError, match="serialized continuation limits were removed or changed"):
        assert_request_action_contract(
            None, replacement_messages, stale["admitted_tool_schemas"],
            request_id=original.request_id, frozen_contract=stale,
            framefit5_preassembly=True,
        )
    replacement = RequestCapture(journal, 1, "snapshot", identity_content={
        "task_id": journal.task_id, "purpose": "",
        "source_replan_of": original.request_id,
    })
    assert replacement.request_id != original.request_id
    fresh = establish_request_action_contract(
        runtime, replacement_messages, tools, request_id=replacement.request_id,
        preserve_input_surface=True, framefit5_preassembly=True,
    )
    assert fresh["budget_wire_digest"] != original_contract["budget_wire_digest"]
    assert_request_action_contract(
        None, replacement_messages, fresh["admitted_tool_schemas"],
        request_id=replacement.request_id, frozen_contract=fresh,
        framefit5_preassembly=True,
    )
    for changed in ([], [{"role": "user", "content": budget.render_continuation_limits() + " changed"}]):
        with pytest.raises(RequestActionContractError, match="serialized continuation limits were removed or changed"):
            assert_request_action_contract(
                None, changed, fresh["admitted_tool_schemas"],
                request_id=replacement.request_id, frozen_contract=fresh,
                framefit5_preassembly=True,
            )


def test_source_replan_origin_context_restores_prior_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kernel_request_reprepare.refresh_source_replan_inputs",
                        lambda _runtime, _capture: 0)
    runtime = SimpleNamespace(_source_replan_in_progress=False)
    with source_replan_preparation(runtime, SimpleNamespace(request_id="original")):
        assert runtime._source_replan_origin_request_id == "original"
        with source_replan_preparation(runtime, SimpleNamespace(request_id="nested")):
            assert runtime._source_replan_origin_request_id == "nested"
        assert runtime._source_replan_origin_request_id == "original"
    assert runtime._source_replan_in_progress is False
    assert not hasattr(runtime, "_source_replan_origin_request_id")
