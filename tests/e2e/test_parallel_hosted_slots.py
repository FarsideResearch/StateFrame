"""Provider slot compatibility through real shipped-profile subframe runs."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler
import gzip
import io
import json
from pathlib import Path
import shutil

import pytest

from barrier_endpoint import BarrierEndpoint
from e2e_support import (assert_shipped_wire, read_json, result_records,
                         run_cli, subframe_directories, subframe_wire_files)


pytestmark = pytest.mark.host_sandbox

OPENAI_CHAT_FIELDS = frozenset({
    "model", "messages", "tools", "tool_choice", "temperature", "top_p", "max_tokens",
    "max_completion_tokens", "seed", "stream", "stream_options", "stop", "n", "logprobs",
    "top_logprobs", "presence_penalty", "frequency_penalty", "logit_bias", "response_format",
    "parallel_tool_calls", "user", "reasoning_effort", "store", "metadata", "prediction",
    "modalities", "audio", "service_tier",
})
LLAMA_FIELDS = frozenset({
    "id_slot", "cache_prompt", "n_probs", "slot_id", "samplers", "grammar", "n_keep",
    "t_max_predict_ms", "top_k", "min_p", "continue_final_message", "add_generation_prompt",
})
PRODUCT_BASELINE_FIELDS = frozenset({
    "chat_template_kwargs", "reasoning_budget_tokens", "reasoning_mode",
})


class HostedEndpoint(BarrierEndpoint):
    """Reject unknown chat fields, allowing the product's existing reasoning fields."""

    def __init__(self, parent_steps: list[dict], subframe_steps: dict[str, list[dict]]) -> None:
        super().__init__(parent_steps, subframe_steps)
        self.rejected: list[dict] = []

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        base = super()._handler()
        endpoint = self

        class Handler(base):
            def do_GET(self) -> None:  # noqa: N802
                if self.path.rstrip("/") == "/slots":
                    self._send(404, {"error": {"message": "not found"}})
                    return
                super().do_GET()

            def do_POST(self) -> None:  # noqa: N802
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                decoded = gzip.decompress(raw) if self.headers.get("Content-Encoding") == "gzip" else raw
                if self.path.rstrip("/") == "/v1/chat/completions":
                    body = json.loads(decoded)
                    extra = set(body) - OPENAI_CHAT_FIELDS - PRODUCT_BASELINE_FIELDS
                    if extra:
                        with endpoint._condition:
                            endpoint.rejected.append({"body": body, "fields": sorted(extra)})
                        self._send(400, {"error": {"message": f"unsupported chat fields: {sorted(extra)}"}})
                        return
                self.rfile = io.BytesIO(raw)
                super().do_POST()

        return Handler


def _two_subframe_script() -> tuple[list[dict], dict[str, list[dict]]]:
    original = json.loads((Path(__file__).resolve().parents[2] /
                           "examples/parallel_fixes.script.json").read_bytes())
    parent = original["parent"][:1] + [{"tool": "done", "arguments": {
        "summary": "Discount and quantity calculations are fixed."}}]
    parent[0]["arguments"]["subframes"] = parent[0]["arguments"]["subframes"][:2]
    subframes = {name: original["subframes"][name] for name in ("discount", "quantity")}
    return parent, subframes


def _two_subframe_repo(tmp_path: Path) -> Path:
    source = Path(__file__).resolve().parents[2] / "examples/parallel_fixes"
    repo = tmp_path / "repo"
    for relative in ("pricing/__init__.py", "pricing/discount.py", "pricing/quantity.py",
                     "tests/test_discount.py", "tests/test_quantity.py"):
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
    return repo


def test_hosted_endpoint_without_slots_accepts_two_subframes(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    parent, script = _two_subframe_script()
    repo = _two_subframe_repo(tmp_path)
    run_dir = tmp_path / "run"
    with HostedEndpoint(parent, script) as endpoint:
        completed = run_cli(isolated_engine_python, endpoint.url, repo, run_dir,
                            "Fix discount and quantity using subframes.", timeout=600)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        endpoint.assert_consumed()
        assert endpoint.rejected == []
    result, = result_records(run_dir).values()
    assert [(outcome.unit_id, outcome.status) for outcome in result.outcomes] == [
        ("discount", "COMMITTED"), ("quantity", "COMMITTED")]
    assert result.txn_order == (1, 2)
    children = subframe_directories(run_dir)
    assert set(children) == {"discount", "quantity"}
    for name, child in children.items():
        paths = subframe_wire_files(child)
        assert_shipped_wire(paths, task_marker=f"Subframe {name}:")
        for path in paths:
            body = read_json(path)
            assert set(body) <= OPENAI_CHAT_FIELDS | PRODUCT_BASELINE_FIELDS
            assert set(body).isdisjoint(LLAMA_FIELDS)
        assert read_json(child / "parallel-slot.json")["pinned"] is False


def test_explicit_pool_uses_logical_leases_without_unsupported_provider_fields(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    parent, script = _two_subframe_script()
    run_dir = tmp_path / "run"
    with HostedEndpoint(parent, script) as endpoint:
        completed = run_cli(isolated_engine_python, endpoint.url, _two_subframe_repo(tmp_path),
                            run_dir, "Fix discount and quantity using subframes.",
                            arguments=("--parallel-slots", "0,1"), timeout=600)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        endpoint.assert_consumed()
        assert endpoint.rejected == []
    children = subframe_directories(run_dir)
    assert set(children) == {"discount", "quantity"}
    result, = result_records(run_dir).values()
    assert [(outcome.unit_id, outcome.status) for outcome in result.outcomes] == [
        ("discount", "COMMITTED"), ("quantity", "COMMITTED")]
    for name, child in children.items():
        assert read_json(child / "parallel-slot.json")["pinned"] is False
        for path in subframe_wire_files(child):
            body = read_json(path)
            assert set(body).isdisjoint(LLAMA_FIELDS)


def test_slots_endpoint_still_pins_subframe_requests(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    parent, script = _two_subframe_script()
    run_dir = tmp_path / "run"
    with BarrierEndpoint(parent, script, slots=2) as endpoint:
        completed = run_cli(isolated_engine_python, endpoint.url, _two_subframe_repo(tmp_path), run_dir,
                            "Fix discount and quantity using subframes.", timeout=600)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        endpoint.assert_consumed()
    children = subframe_directories(run_dir)
    assert set(children) == {"discount", "quantity"}
    result, = result_records(run_dir).values()
    assert [(outcome.unit_id, outcome.status) for outcome in result.outcomes] == [
        ("discount", "COMMITTED"), ("quantity", "COMMITTED")]
    for name, child in children.items():
        slot = read_json(child / "parallel-slot.json")["slot"]
        assert_shipped_wire(subframe_wire_files(child), task_marker=f"Subframe {name}:")
        assert all(read_json(path)["id_slot"] == slot for path in subframe_wire_files(child))
        assert all("cache_prompt" in read_json(path) for path in subframe_wire_files(child))
