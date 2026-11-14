"""Small model slots must fit safely or return an actionable pre-dispatch stop."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import replace

import pytest

from _journal import measurement_events
from _timeouts import scaled
from test_capacity_400_acceptance import CapacityEndpoint
from kernel_context_floor import context_floor
from kernel_endpoint_size import resolve_endpoint_size

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.host_sandbox


@pytest.mark.parametrize("method,wire_bytes,prompt_tokens", [
    ("tokenize", 60_000, 10_000),
    ("tokenize", 160_001, 1_000),
    ("calibrated_bytes", 40_000, 9_000),
    ("verified_byte_bound", 60_000, 10_000),
])
def test_context_floor_inverts_measured_prompt_and_byte_limits(method, wire_bytes, prompt_tokens):
    size = resolve_endpoint_size({"STATEFRAME_CONTEXT_TOKENS": "16384"})
    detail = {"measure_method": method, "request_bytes": wire_bytes,
              "prompt_tokens": prompt_tokens, "layout_token_reserve": 32}
    stop = context_floor(detail, size=size)
    assert stop is not None
    assert stop.minimum_context_tokens > size.context_tokens
    assert resolve_endpoint_size({
        "STATEFRAME_CONTEXT_TOKENS": str(stop.minimum_context_tokens),
    }).prompt_budget >= stop.minimum_prompt_tokens
    assert resolve_endpoint_size({
        "STATEFRAME_CONTEXT_TOKENS": str(stop.minimum_context_tokens - 1),
    }).prompt_budget < stop.minimum_prompt_tokens
    if method == "calibrated_bytes":
        assert int(stop.minimum_prompt_tokens * 2.46) >= wire_bytes
    else:
        assert stop.minimum_prompt_tokens * 16 >= wire_bytes
    assert stop.terminal_fields()["minimum_measure_method"] == method


def test_context_floor_does_not_hide_other_constraints():
    small = resolve_endpoint_size({"STATEFRAME_CONTEXT_TOKENS": "8192"})
    assert context_floor({"prompt_tokens": 1000}, size=small) is None
    assert context_floor({"prompt_tokens": 40_000}, size=resolve_endpoint_size({})) is None
    assert context_floor({"prompt_tokens": 10_000},
                         size=replace(small, budget_explicit=True)) is None


def test_zero_line_reservation_preserves_provable_deferred_obligation(tmp_path):
    from kernel_request_evidence_closure import build_source_closure, source_revision
    from kernel_request_source_window import (
        _ensure_target_unit, reserve_source_unit, upsert_deferred_source_notice,
        source_delivery_satisfied, verify_final_source_delivery,
    )

    source = "def value():\n    return 6\n"
    (tmp_path / "app.py").write_text(source)
    plan = build_source_closure(tmp_path, ("app.py",), "Fix the bug.", target_path="app.py").as_dict()
    assert plan["units"] == []
    reservation = reserve_source_unit(source, path="app.py", revision=source_revision(source),
                                      max_bytes=1, non_source_bytes=1000).as_dict()
    assert reservation["selected_lines"] == []
    _ensure_target_unit(plan, "app.py", reservation)
    assert len(plan["units"]) == 1
    assert plan["units"][0]["start_line"] == 1
    assert plan["units"][0]["end_line"] == 2
    plan["workspace_authorization"] = {"scope": str(tmp_path), "generation": "0"}
    plan["deferred_paths"] = ["app.py"]
    messages = []
    upsert_deferred_source_notice(messages, ["app.py"])
    body = {"messages": messages, "tools": [{"type": "function", "function": {"name": "read"}}]}
    receipt = verify_final_source_delivery(plan, messages, payload=body)
    assert source_delivery_satisfied(receipt), receipt
    assert receipt["deferred_paths"] == ["app.py"]
    assert not source_delivery_satisfied(verify_final_source_delivery(plan, [], payload={"messages": []}))


@pytest.mark.parametrize("source", ["", "def value():\n    return 6\n"])
def test_zero_line_reservation_cannot_invent_source_revision(tmp_path, source):
    from kernel_request_evidence_closure import build_source_closure
    from kernel_request_source_window import _ensure_target_unit
    (tmp_path / "app.py").write_text(source)
    plan = build_source_closure(tmp_path, ("app.py",), "Fix the bug.", target_path="app.py").as_dict()
    _ensure_target_unit(plan, "app.py", {"path": "app.py", "selected_lines": [],
                                      "total_lines": len(source.splitlines()), "revision": "wrong"})
    assert plan["units"] == []


@pytest.mark.parametrize("context", [16_384, 8_192])
@pytest.mark.parametrize("large_source,token_chars", [
    (False, 4), (True, 4), (False, 16),
], ids=["minimum", "deferred", "fits"])
@pytest.mark.parametrize("allocator", ["off", "on"], ids=["legacy", "allocator"])
def test_small_context_public_run(tmp_path: Path, context: int, large_source: bool,
                                  token_chars: int, allocator: str) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    source = "VALUE = 6\n"
    if large_source:
        source = ("def release_constraints():\n    VALUE = 6\n"
                  + "    # retained release constraint\n" * 1200 + "    return VALUE\n")
    (repo / "app.py").write_text(source)
    (repo / "tests").mkdir()
    test_source = ("from app import release_constraints\n"
                   "def test_value(): assert release_constraints() == 7\n" if large_source else
                   "from app import VALUE\ndef test_value(): assert VALUE == 7\n")
    (repo / "tests/test_app.py").write_text(test_source)
    run = tmp_path / "run"
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("STATEFRAME_", "REACT_"))}
    environment.update(PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
                       STATEFRAME_HOME=str(tmp_path / "home"),
                       STATEFRAME_CONTEXT_TOKENS=str(context),
                       STATEFRAME_WORKING_SET_ALLOCATOR=allocator)
    with CapacityEndpoint(()) as endpoint:
        parent = endpoint.server.RequestHandlerClass

        class Handler(parent):
            def do_GET(self) -> None:  # noqa: N802
                if self.path.rstrip("/") == "/props":
                    self._send(200, {"default_generation_settings": {"n_ctx": context},
                                     "model_path": "scripted-endpoint"})
                else:
                    super().do_GET()

            def do_POST(self) -> None:  # noqa: N802
                if token_chars != 4 and self.path.rstrip("/").endswith("/tokenize"):
                    raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                    content = str(json.loads(raw).get("content") or "")
                    self._send(200, {"tokens": list(range(max(1, len(content) // token_chars)))})
                else:
                    super().do_POST()

        endpoint.server.RequestHandlerClass = Handler
        result = subprocess.run(
            [sys.executable, "-B", "-m", "stateframe", "run", "--repo", str(repo),
             "--task", "Fix the bug in app.py.", "--endpoint", endpoint.url,
             "--model", "capacity-mock", "--run-dir", str(run), "--max-turns", "1",
             "--test-cmd", "python3 -m pytest -q tests/test_app.py", "--runner-kind", "pytest",
             "--test-python", sys.executable, "--target", "app.py", "--edit", "app.py",
             "--shell", "off", "--no-project-hooks", "--no-instructions", "--timeout", "180",
             "--quiet", "--json"], env=environment, cwd=ROOT, capture_output=True,
            text=True, timeout=scaled(240),
        )
        requests = tuple(endpoint.requests)
    summary = json.loads((run / "summary.json").read_text())
    events = measurement_events(run)
    output = result.stdout + result.stderr + (run / "engine.log").read_text()
    assert "Traceback (most recent call last)" not in output, output[-12000:]
    assert not str(summary.get("engine_result", "")).startswith("ERR:"), summary
    assert not any(row["kind"] == "run_exception" for row in events), events[-10:]
    legacy_should_serve = token_chars == 16 or context == 16_384
    if allocator == "off" and legacy_should_serve:
        assert requests, summary
        assert result.returncode == 0, summary
        assert summary["engine_result"] == "SUBMITTED", summary
        assert summary["runtime_outcome"] == "completed", summary
        if token_chars == 4:
            assert any(row["kind"] == "frame_layout_reserve_released" for row in events), events[-10:]
            assert any(row["kind"] == "final_cache_layout_failed_open" for row in events), events[-10:]
    elif allocator == "off":
        assert not requests, summary
        terminal = summary["engine_terminal"]
        assert result.returncode == 2, summary
        assert summary["runtime_outcome"] == "context_too_small", summary
        assert terminal["reason_code"] == "CONTEXT_TOO_SMALL", terminal
        assert terminal.get("resumable") is not True, terminal
        assert "suspension" not in terminal, terminal
        assert terminal["configured_context_tokens"] == context
        assert terminal["minimum_context_tokens"] > context
        assert terminal["message"] == (
            f"model context {context} is too small; StateFrame needs at least "
            f"{terminal['minimum_context_tokens']}")
        assert summary["provider_sends"] == 0
        exhausted = [row for row in events if row["kind"] == "frame_minimum_exhausted"]
        assert len(exhausted) == 1
        from kernel_context_floor import context_floor
        minimum = context_floor(exhausted[0]["minimum_detail"], size=resolve_endpoint_size({
            "STATEFRAME_CONTEXT_TOKENS": str(context),
        }))
        assert minimum is not None
        assert terminal["minimum_context_tokens"] == minimum.minimum_context_tokens
    elif requests:
        assert result.returncode == 0, summary
        assert summary["engine_result"] == "SUBMITTED", summary
        assert summary["runtime_outcome"] == "completed", summary
        if token_chars == 4 and context == 16_384:
            assert any(row["kind"] == "frame_layout_reserve_released" for row in events), events[-10:]
            assert any(row["kind"] == "final_cache_layout_failed_open" for row in events), events[-10:]
    else:
        # An ON allocator may serve the same case, or prove its source-qualified
        # minimum cannot fit this slot. Either result is a typed, pre-dispatch stop.
        terminal = summary["engine_terminal"]
        assert result.returncode == 2, summary
        assert summary["runtime_outcome"] == "context_too_small", summary
        assert terminal["reason_code"] == "CONTEXT_TOO_SMALL", terminal
        assert terminal["configured_context_tokens"] == context
        assert terminal["minimum_context_tokens"] > context
        assert terminal["message"] == (
            f"model context {context} is too small; StateFrame needs at least "
            f"{terminal['minimum_context_tokens']}")
        assert summary["provider_sends"] == 0
