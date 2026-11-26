"""Report projections at the provider boundary, using a scripted writer."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
from contextlib import suppress
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from fixtures.process_signals import record_identity, signal_run_owned, signal_run_verified
from stateframe import runner
from stateframe.scripted_endpoint import Script, make_handler
from tools.compare_mutation_off_requests import EXAMPLES, write_launcher
from tools.pinned_tree import pinned_tree
from tools.sandbox_temp import timeout_scale


ROOT = Path(__file__).resolve().parents[1]
PIN = "6e3b096cf3059f759138a88f7fdf46acd3d75baa"
SWITCH = "STATEFRAME_REPORT_DENOMINATORS"
SPEC = {
    "path": "input/reviewed.csv",
    "unit_key": "id",
    "cohort_field": "team",
    "eligibility": {"field": "scope", "equals": "eligible"},
    "outcome": {"field": "status", "equals": "open"},
    "missing_values": ["", "unreviewed"],
}
CSV = (
    b"id,team,scope,status\n"
    b"1,Billing,eligible,open\n"
    b"2,Billing,eligible,closed\n"
    b"3,Billing,eligible,unreviewed\n"
    b"4,Billing,outside,open\n"
    b"5,Billing,,open\n"
    b"6,Login,eligible,open\n"
    b"7,Login,eligible,unreviewed\n"
    b"8,Login,eligible,unreviewed\n"
)
EXPECTED = [
    {
        "source": SPEC["path"], "source_revision": "sha256:" + hashlib.sha256(CSV).hexdigest(),
        "cohort": name, "numerator": 1, "observed_denominator": observed,
        "eligible_population": 3, "exclusions": {
            "ineligible": ineligible, "eligibility_missing": unknown, "outcome_missing": missing,
        },
    }
    for name, observed, ineligible, unknown, missing in
    (("Billing", 2, 1, 1, 1), ("Login", 1, 0, 0, 2))
]


def _task(spec=SPEC):
    return EXAMPLES["brief_to_report"][0] + "\nReport denominators: " + json.dumps(spec)


@pytest.fixture
def provider():
    with ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Script({}))) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            yield server
        finally:
            server.shutdown()
            worker.join()


def _capture(provider, tree, work, label, *, switch=None, allocator="on", resume=None,
             pressure=False, preview_413=False, capacity_overflow=False):
    # Read the very bytes counted by the host so subsequent sends exercise live source
    # evidence and the reconstruction of the latest native exchange.
    paths = [SPEC["path"], "input/brief.md"]
    script = Script({
        "steps": [{"tool": "read", "arguments": {"path": path}} for path in paths],
        "tokenize": {"mode": "ok"},
    })
    received = []
    preview_413_hits = []
    capacity_rejections = []
    waiting = threading.Event()
    release = threading.Event()

    class Handler(make_handler(script)):
        def do_POST(self):
            if self.path == "/v1/chat/completions":
                incoming = self.rfile
                raw = incoming.read(int(self.headers["Content-Length"]))
                received.append(raw)
                if capacity_overflow and len(received) == 2:
                    capacity_rejections.append(len(received) - 1)
                    self._send(400, {"error": {
                        "message": "prompt is too long: 30000 tokens > 20000 maximum"}})
                    return
                if resume and len(received) == 2:
                    if resume == "checkpoint":
                        waiting.set()
                        release.wait(180 * timeout_scale())
                    self._send(429, {"error": {"code": "insufficient_quota", "message": "test interruption"}})
                    return
                self.rfile = io.BytesIO(raw)
                try:
                    super().do_POST()
                finally:
                    self.rfile = incoming
            else:
                super().do_POST()

        def _tokenize(self, raw):
            if preview_413 and b"Report denominators (task-declared CSV" in raw:
                preview_413_hits.append(len(raw))
                self._send(413, {"error": {
                    "message": "tokenizer request exceeds the 81920-byte limit"}})
            elif pressure and b"Report denominators (task-declared CSV" in raw:
                self._send(200, {"tokens": list(range(50_000))})
            else:
                super()._tokenize(raw)

    provider.RequestHandlerClass = Handler
    provider.preview_413_hits = preview_413_hits
    provider.capacity_rejections = capacity_rejections
    endpoint = f"http://127.0.0.1:{provider.server_port}/v1"
    repo, run_dir, home = (work / name for name in ("repo", "run", "home"))
    launcher = work / "launcher.py"
    shutil.copytree(ROOT / "examples/brief_to_report", repo)
    (repo / SPEC["path"]).write_bytes(CSV)
    write_launcher(launcher)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("STATEFRAME_", "REACT_", "PYTHONPATH"))}
    env.update(
        PYTHONPATH=os.pathsep.join(map(str, (tree / "src", tree / "src/stateframe/engine"))),
        PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0", STATEFRAME_HOME=str(home),
        STATEFRAME_WORKING_SET_ALLOCATOR=allocator,
        STATEFRAME_CHECKPOINT="off" if resume == "replay" else "save",
    )
    if switch is not None:
        env[SWITCH] = switch

    def run(*args):
        command = [sys.executable, str(launcher), *args, "--endpoint", endpoint, "--quiet"]
        timeout = 360 * timeout_scale()
        if args[0] == "run" and resume == "checkpoint":
            # Checkpoint recovery needs a recorded request still in flight.
            with subprocess.Popen(command, env=env, cwd=tree, text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
                try:
                    assert waiting.wait(timeout), "second provider request was never dispatched"
                    children = subprocess.check_output(["pgrep", "-P", str(process.pid)], text=True).split()
                    assert len(children) == 1, children
                    child = int(children[0])
                    identity = record_identity(child, launched_after=0, command_fragment="_child.py")
                    signal_run_verified(run_dir, child, identity, signal.SIGKILL,
                                        deadline=time.monotonic() + timeout, what="F023 checkpoint parity")
                finally:
                    release.set()
                    try:
                        stdout, stderr = process.communicate(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        with suppress(ProcessLookupError):
                            signal_run_owned(
                                run_dir, process, signal.SIGKILL, deadline=time.monotonic(),
                                what="F023 checkpoint parity cleanup", require_checkpoint_saved=False,
                            )
                        process.communicate()
                        raise
                result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        else:
            result = subprocess.run(command, env=env, cwd=tree, capture_output=True,
                                    text=True, timeout=timeout, check=False)
        assert (run_dir / "summary.json").is_file(), result.stdout + result.stderr
        return result

    result = run(
        "run", "--repo", str(repo), "--task", _task(),
        "--model", "scripted", "--run-dir", str(run_dir), "--test-python", sys.executable,
        "--test-cmd", EXAMPLES["brief_to_report"][1], "--no-run-env",
        "--max-turns", str(len(paths)),
    )
    if preview_413:
        assert preview_413_hits, "the report preview did not trigger the tokenizer's HTTP 413"
    if resume:
        initial = len(received)
        assert initial == 2 and len(script.requests) == 1, result.stdout + result.stderr
        result = run("resume", str(run_dir))
        spec = json.loads((run_dir / "spec.json").read_bytes())
        assert "resume" in spec, result.stdout + result.stderr
        resume_spec = spec["resume"]
        assert resume_spec["tape"] and not resume_spec.get("restart_live")
        assert bool(resume_spec.get("checkpoint")) == (resume == "checkpoint"), resume_spec
        assert len(received) == initial + 1, result.stdout + result.stderr
    summary = json.loads((run_dir / "summary.json").read_bytes())
    assert summary["engine_result"] == "RESPONSE_BUDGET_EXHAUSTED", result.stdout + result.stderr
    script.assert_consumed()
    assert len(received) >= len(script.steps)
    if preview_413:
        assert any(b"eligible,unreviewed" in payload for payload in received), (
            "the ordinary source evidence was lost after the optional preview failed")
    if capacity_overflow:
        assert len(capacity_rejections) == 1, "capacity requalification was not exercised"
        assert len(received) >= len(script.steps) + 1
        retry_index = capacity_rejections[0] + 1
        assert retry_index < len(received)
        assert received[capacity_rejections[0]] != received[retry_index], (
            "capacity requalification did not reseal the request")
    assert all(json.loads(payload)["tools"] for payload in received)
    for directory in (repo, run_dir, home):
        directory.rename(work / f"{label}-{directory.name}")
    return received


@pytest.mark.parametrize(("allocator", "resume"), [
    ("on", "checkpoint"), ("on", "replay"), ("off", "replay"),
])
def test_f023_numerator_observed_denominator_off_request_bytes_match_pinned_099(
        tmp_path, tmp_path_factory, provider, allocator, resume,
):
    baseline = pinned_tree(PIN, tmp_path_factory)
    options = {"allocator": allocator, "resume": resume}
    expected = _capture(provider, baseline, tmp_path, "baseline", **options)
    assert _capture(provider, ROOT, tmp_path, "unset", **options) == expected
    assert _capture(provider, ROOT, tmp_path, "off", switch="false", **options) == expected


def test_f023_numerator_observed_denominator_switch_defaults_off(monkeypatch):
    import kernel_stateframe_framefit5 as framefit

    monkeypatch.delenv(SWITCH, raising=False)
    assert runner.OPT_IN_SWITCHES[SWITCH] == ("off", "on")
    assert SWITCH in runner.engine_switches()
    assert SWITCH not in runner.resolve_opt_in_switches()
    assert not framefit._report_denominators_enabled()
    monkeypatch.setenv(SWITCH, "false")
    assert runner.resolve_opt_in_switches()[SWITCH] == "off"
    assert not framefit._report_denominators_enabled()
    monkeypatch.setenv(SWITCH, "on")
    assert runner.resolve_opt_in_switches()[SWITCH] == "on"
    assert framefit._report_denominators_enabled()


def test_f023_preview_tokenizer_413_does_not_block_ordinary_request(
        tmp_path, provider,
):
    import kernel_stateframe_framefit5 as framefit

    received = _capture(provider, ROOT, tmp_path, "preview-413", switch="on", preview_413=True)
    assert provider.preview_413_hits
    assert all(framefit._REPORT_PANEL.encode() not in payload for payload in received)


def test_f023_oversized_preview_413_does_not_poison_live_meter():
    import kernel_stateframe_framefit5 as framefit
    from kernel_request_budget import ENDPOINT_PROMPT_TOKEN_BUDGET
    from kernel_request_diet import RequestTokenMeter

    preview_sizes = []
    def tokenizer(content):
        size = len(content.encode("utf-8"))
        contains_panel = "Report denominators (task-declared CSV" in content
        if contains_panel and size > 80 * 1024:
            preview_sizes.append(size)
            raise urllib.error.HTTPError(
                "http://127.0.0.1/v1/tokenize", 413, "request too large", {}, io.BytesIO(b""))
        return [1]

    live_meter = RequestTokenMeter(tokenizer=tokenizer, template_overhead_tokens=0)
    live_meter._minimum_bytes_per_token = 1.0
    live_meter._calibration_samples = 1
    capture = SimpleNamespace(request_id="report", _tokenize_logical_request_id="report")
    live_meter.bind_request("report", "report")
    preview_meter = framefit._report_denominator_preview_meter(live_meter, capture)
    oversized_preview = {
        "messages": [{"role": "user", "content": framefit._REPORT_PANEL + "x" * (80 * 1024)}],
    }

    estimated, method = preview_meter.measure(oversized_preview)
    assert preview_sizes and min(preview_sizes) > 80 * 1024
    assert method == "calibrated_bytes" and estimated > ENDPOINT_PROMPT_TOKEN_BUDGET
    assert preview_meter._availability == "unavailable"
    assert live_meter._availability == "unknown"
    exact, exact_method = live_meter.measure({
        "messages": [{"role": "user", "content": "ordinary report request"}],
    })
    assert (exact, exact_method) == (1, "tokenize")
    assert live_meter._availability == "available"


def test_f023_previous_report_panel_is_optional_during_followup_assembly():
    import kernel_stateframe_framefit5 as framefit

    prior_panel = framefit._REPORT_PANEL + json.dumps(EXPECTED, separators=(",", ":"))
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "read-brief", "type": "function",
                "function": {"name": "read", "arguments": "{}"},
            }],
        },
        {"role": "tool", "tool_call_id": "read-brief", "name": "read",
         "content": "required brief excerpt: line 1"},
        {"role": "user", "content": (
            framefit.VOLATILE_TAIL_MARKER + "\n\n" + prior_panel
            + "\n\nordinary host fact"
        )},
    ]

    assembled = framefit.finalize_frame_only_wire_order(messages)
    tail = assembled[-1]["content"]
    assert framefit._REPORT_PANEL not in tail
    assert "ordinary host fact" in tail
    assert assembled[1]["content"] == "required brief excerpt: line 1"


def test_f023_large_report_panel_does_not_shrink_required_source_read():
    import kernel_stateframe_framefit5 as framefit

    report_rows = []
    for index in range(85):
        report_rows.append({
            "source": SPEC["path"], "source_revision": "sha256:" + "0" * 64,
            "cohort": f"Cohort-{index:02d}", "numerator": 1,
            "observed_denominator": 1, "eligible_population": 1,
            "exclusions": {
                "ineligible": 0, "eligibility_missing": 0, "outcome_missing": 0,
            },
        })
    brief = b"".join(
        f"brief-line-{index:03d} ".encode() + b"x" * 659 + b"\n"
        for index in range(80)
    )
    assert len(brief) == 54_000
    prior_panel = framefit._REPORT_PANEL + json.dumps(report_rows, separators=(",", ":"))
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "read-brief", "type": "function",
                "function": {"name": "read", "arguments": "{}"},
            }],
        },
        {"role": "tool", "tool_call_id": "read-brief", "name": "read",
         "content": brief.decode()},
        {"role": "user", "content": (
            framefit.VOLATILE_TAIL_MARKER + "\n\n" + prior_panel
            + "\n\nordinary host fact" + framefit.GREENFACT_TEXT
        )},
    ]

    assembled = framefit.finalize_frame_only_wire_order(messages)
    tail = assembled[-1]["content"]
    delivered = assembled[1]["content"]
    assert framefit._REPORT_PANEL not in tail
    assert "ordinary host fact" in tail
    assert framefit.GREENFACT_TEXT in tail
    assert len([
        index for index in range(80) if f"brief-line-{index:03d}" in delivered
    ]) == 80
    assert len(delivered.encode()) == len(brief)


@pytest.mark.parametrize("cache_layout", [False, True], ids=["plain", "final-cache"])
def test_f023_read_admission_does_not_price_prior_report_panel_against_next_read(
        tmp_path, monkeypatch, cache_layout,
):
    from dataclasses import replace
    from kernel_cache_layout_candidate import _project_final_cache_layout
    import kernel_read_admission as admission
    from kernel_source_delivery import deliver_source, normalize_read_request
    import kernel_stateframe_framefit5 as framefit
    import kernel_working_set_live as live
    from test_read_admission_qualification import seal_case

    monkeypatch.setenv(SWITCH, "on")
    record, capture, runtime = seal_case(tmp_path, monkeypatch, "K1")
    brief = b"".join(
        f"brief-line-{index:03d} ".encode() + b"x" * 659 + b"\n"
        for index in range(80)
    )
    assert len(brief) == 54_000
    path = "input/brief.md"
    args = {"path": path, "max_lines": 80, "max_chars": 64_000}
    call = {"id": "read-brief", "type": "function",
            "function": {"name": "read", "arguments": json.dumps(args)}}
    runtime.native_batch = SimpleNamespace(pending=[call], inflight=None)
    runtime._read_attempt_scope = SimpleNamespace(call_id=call["id"])

    panel_rows = [{
        "source": SPEC["path"], "source_revision": "sha256:" + "0" * 64,
        "cohort": f"Cohort-{index:02d}", "numerator": 1,
        "observed_denominator": 1, "eligible_population": 1,
        "exclusions": {"ineligible": 0, "eligibility_missing": 0, "outcome_missing": 0},
    } for index in range(85)]
    panel = framefit._REPORT_PANEL + json.dumps(panel_rows, separators=(",", ":"))

    def prior_request(extra):
        body = json.loads(json.dumps(record.body))
        tail = body["messages"][-1]
        content = tail["content"].removesuffix(framefit.GREENFACT_TEXT)
        tail["content"] = content + "\n\nordinary host fact" + extra + framefit.GREENFACT_TEXT
        if cache_layout:
            state_row = next(row for row in body["messages"]
                             if str(row.get("content")).startswith("[StateFrame]\n"))
            payload, frame = framefit._state_payload(state_row)
            frame.setdefault("owner_protocol", {})["continuation_limits"] = (
                "[CONTINUATION LIMITS | stateframe-continuation-limits/1] remaining=2"
            )
            state_row["content"] = framefit._serialize_state_message(payload)
            body = _project_final_cache_layout(body)
        capture.qualified_request = SimpleNamespace(body=body)
        admission.note_sealed_request(runtime, body["messages"], body["tools"])
        return body

    request = normalize_read_request(args)
    source = deliver_source(brief, path, 0, args=request)
    ordinary = prior_request("")
    snapshot = live.read_admission_snapshot(runtime, ordinary["messages"], byte_cap=200_000)
    assert snapshot is not None
    ordinal = max(slot.admission_ordinal for slot in snapshot.sources) + 1

    def floor_bytes(candidate):
        page = live.read_admission_slot(candidate, path=path, arguments=request.arguments,
                                       ordinal=ordinal)
        return len(live.read_admission_floor(snapshot, ((page, call["id"]),)).wire)

    # The ordinary wire fits all 80 lines. Charging the escaped prior panel
    # would leave room for only the 44-line page and its continuation.
    shortened = deliver_source(brief, path, 0, args=replace(request, max_lines=44))
    panel_bytes = len(json.dumps(panel).encode()) - 2  # Exclude JSON string quotes.
    cap = max(floor_bytes(source), floor_bytes(shortened) + panel_bytes)
    monkeypatch.setattr(admission, "request_byte_cap", lambda: cap)

    def read(body):
        return admission.admit(runtime, raw=brief, display=path, start=0,
                               request=request, source=source, visible_messages=body["messages"])

    assert read(ordinary).metadata["effective_range"]["end_line"] == 80
    enriched = prior_request(panel)
    sealed_wire = json.dumps(enriched).encode()
    delivered = read(enriched)
    assert delivered.metadata["effective_range"]["end_line"] == 80
    assert delivered.content == source.content
    assert json.dumps(enriched).encode() == sealed_wire
    assert panel in runtime._read_admission_provider_body["messages"][-1]["content"]
    admitted = live.read_admission_snapshot(runtime, enriched["messages"], byte_cap=cap)
    assert admitted is not None and admitted.fixed_body == snapshot.fixed_body

    runtime._active_capture = None
    retained = live.read_admission_snapshot(runtime, None, byte_cap=cap)
    assert retained is not None and retained.fixed_body == snapshot.fixed_body


def test_f023_calibrated_byte_fit_survives_exact_tokenizer_unavailable(monkeypatch):
    from dataclasses import dataclass

    import kernel_cache_layout_candidate as cache_layout
    import kernel_context
    import kernel_launch_identity_cost
    import kernel_provider_dialect
    import kernel_request_budget
    import kernel_request_diet
    import kernel_working_set_live
    import kernel_working_set_prepare
    import kernel_working_set_render
    import kernel_working_set_snapshot
    import kernel_stateframe_framefit5 as framefit

    @dataclass(frozen=True)
    class Slot:
        key: str

    @dataclass(frozen=True)
    class Snapshot:
        sources: tuple[Slot, ...]
        layouts: tuple[str, ...] = ()
        context_limit: int = 0
        output_reserve: int = 0
        safety_reserve: int = 0

    class Mode:
        rank = 1

    slot = Slot("brief")
    snapshot = Snapshot((slot,))
    preview_methods = []
    tokenizer_attempts = []

    def unavailable_tokenizer(_content):
        tokenizer_attempts.append(True)
        raise urllib.error.HTTPError(
            "https://example.invalid/tokenize", 503, "unavailable", {}, io.BytesIO(b""))

    live_meter = kernel_request_diet.RequestTokenMeter(
        tokenizer=unavailable_tokenizer, endpoint="https://example.invalid/v1",
        route_allow_remote=True, template_overhead_tokens=0,
    )
    live_meter._availability = "unavailable"
    live_meter._minimum_bytes_per_token = 1.0
    live_meter._calibration_samples = 1

    def record_price(_snapshot, _preview, measurement):
        preview_methods.append(measurement.method)
        return []

    monkeypatch.setattr(cache_layout, "compose_stable_guidance", lambda _capture, body: body)
    monkeypatch.setattr(cache_layout, "final_layout_mode", lambda: "off")
    monkeypatch.setattr(kernel_request_budget, "_lean_request_limits", lambda *_a, **_k: (100_000, 100_000))
    monkeypatch.setattr(kernel_request_budget, "_request_token_meter", lambda _runtime, _capture: live_meter)
    monkeypatch.setattr(kernel_request_diet, "request_tokenizer_credentials",
                        lambda _capture: (None, "https://example.invalid/v1"))
    monkeypatch.setattr(kernel_provider_dialect, "capabilities_for",
                        lambda *_a, **_k: SimpleNamespace(dialect="openai"))
    monkeypatch.setattr(kernel_launch_identity_cost, "reasoning_profile",
                        lambda **_k: SimpleNamespace(mode="none"))
    monkeypatch.setattr(kernel_working_set_live, "_source_plan_with_deferred_omissions",
                        lambda plan: plan)
    monkeypatch.setattr(kernel_working_set_snapshot, "freeze_preparation",
                        lambda *_a, **_k: snapshot)
    monkeypatch.setattr(kernel_working_set_render, "domain", lambda *_a: [Mode()])
    monkeypatch.setattr(kernel_working_set_render, "preview_assignment",
                        lambda *_a: SimpleNamespace(wire=b"{}"))
    monkeypatch.setattr(kernel_working_set_prepare, "Measurement",
                        lambda tokens, estimate, method: SimpleNamespace(
                            tokens=tokens, estimate=estimate, method=method))
    monkeypatch.setattr(kernel_working_set_prepare, "_violations", record_price)
    monkeypatch.setattr(kernel_context, "token_estimate", lambda _body: 1)

    capture = SimpleNamespace(
        request_id="draft", journal=SimpleNamespace(manifest={}),
    )
    runtime = SimpleNamespace(frame_ceiling=64)
    body = {"messages": [{"role": "user", "content": "ordinary draft request"}]}
    assert framefit._report_denominators_fit(body, runtime, capture, {})
    assert preview_methods == ["calibrated_bytes"]
    assert not tokenizer_attempts


def test_f023_outage_preview_matches_live_allocator(monkeypatch):
    from dataclasses import dataclass

    import kernel_cache_layout_candidate as cache_layout
    import kernel_context
    import kernel_launch_identity_cost
    import kernel_provider_dialect
    import kernel_request_budget
    import kernel_request_diet
    import kernel_working_set_live
    import kernel_working_set_prepare
    import kernel_working_set_render
    import kernel_working_set_snapshot
    import kernel_stateframe_framefit5 as framefit

    @dataclass(frozen=True)
    class Slot:
        key: str

    @dataclass(frozen=True)
    class Snapshot:
        body: dict[str, object]
        sources: tuple[Slot, ...]
        prompt_limit: int
        layouts: tuple[str, ...] = ()
        context_limit: int = 0
        output_reserve: int = 0
        safety_reserve: int = 0

    class Mode:
        rank = 1

    capture = SimpleNamespace(
        request_id="report", _tokenize_logical_request_id="report",
        journal=SimpleNamespace(manifest={}),
    )

    def outage_meter():
        meter = kernel_request_diet.RequestTokenMeter(
            tokenizer=lambda _content: [1], template_overhead_tokens=0,
        )
        meter._availability = "unavailable"
        meter._minimum_bytes_per_token = 4.0
        meter._calibration_samples = 2
        meter._namespace = "previous exact-tokenizer namespace"
        return meter

    ordinary_content = " " * 8192 + framefit.GREENFACT_TEXT
    panel_content = " " * 8192 + framefit._REPORT_PANEL + "[]" + framefit.GREENFACT_TEXT
    ordinary_body = {"messages": [{"role": "user", "content": ordinary_content}]}
    panel_body = {"messages": [{"role": "user", "content": panel_content}]}

    def calibrated_count(body):
        meter = outage_meter()
        count, method = meter.measure(body)
        assert method == "calibrated_bytes"
        return count

    ordinary_live_count = calibrated_count(ordinary_body)
    panel_live_count = calibrated_count(panel_body)
    assert ordinary_live_count < panel_live_count
    snapshot_limit = ordinary_live_count
    live_meter = outage_meter()

    monkeypatch.setattr(cache_layout, "compose_stable_guidance", lambda _capture, body: body)
    monkeypatch.setattr(cache_layout, "final_layout_mode", lambda: "off")
    monkeypatch.setattr(kernel_request_budget, "_lean_request_limits",
                        lambda *_a, **_k: (100_000, snapshot_limit))
    monkeypatch.setattr(kernel_request_budget, "_request_token_meter",
                        lambda _runtime, _capture: live_meter)
    monkeypatch.setattr(kernel_request_diet, "request_tokenizer_credentials",
                        lambda _capture: (None, "https://example.invalid/v1"))
    monkeypatch.setattr(kernel_provider_dialect, "capabilities_for",
                        lambda *_a, **_k: SimpleNamespace(dialect="openai"))
    monkeypatch.setattr(kernel_launch_identity_cost, "reasoning_profile",
                        lambda **_k: SimpleNamespace(mode="none"))
    monkeypatch.setattr(kernel_working_set_live, "_source_plan_with_deferred_omissions",
                        lambda plan: plan)
    monkeypatch.setattr(kernel_working_set_snapshot, "freeze_preparation",
                        lambda body, *_a, **_k: Snapshot(body, (Slot("brief"),), snapshot_limit))
    monkeypatch.setattr(kernel_working_set_render, "domain", lambda *_a: [Mode()])
    monkeypatch.setattr(kernel_working_set_render, "preview_assignment",
                        lambda snapshot, *_a: SimpleNamespace(
                            wire=json.dumps(snapshot.body).encode("utf-8")))
    monkeypatch.setattr(kernel_working_set_prepare, "_violations",
                        lambda snapshot, _preview, measurement:
                        ["prompt"] if measurement.prompt_tokens > snapshot.prompt_limit else [])
    monkeypatch.setattr(kernel_context, "token_estimate", lambda _body: 1)

    runtime = SimpleNamespace(frame_ceiling=64)
    assert framefit._report_denominators_fit(ordinary_body, runtime, capture, {})
    assert not framefit._report_denominators_fit(panel_body, runtime, capture, {})


def test_f023_off_request_bytes_match_pinned_099_after_capacity_requalification(
        tmp_path, tmp_path_factory, provider,
):
    baseline = pinned_tree(PIN, tmp_path_factory)
    expected = _capture(provider, baseline, tmp_path, "capacity-baseline", capacity_overflow=True)
    assert len(provider.capacity_rejections) == 1
    assert _capture(provider, ROOT, tmp_path, "capacity-unset", capacity_overflow=True) == expected
    assert _capture(provider, ROOT, tmp_path, "capacity-off", switch="false",
                    capacity_overflow=True) == expected


def test_f023_numerator_observed_denominator_on_path_is_bounded_and_idempotent(
        tmp_path, provider, monkeypatch,
):
    import kernel_stateframe_framefit5 as framefit

    received = _capture(provider, ROOT, tmp_path, "enabled", switch="on", resume="replay")
    panel = framefit._REPORT_PANEL
    enriched = []
    for payload in received:
        messages = json.loads(payload)["messages"]
        carriers = [row["content"] for row in messages if panel in str(row.get("content", ""))]
        if carriers:
            assert carriers == [messages[-1]["content"]]
            assert carriers[0].count(panel) == 1
            rows, _ = json.JSONDecoder().raw_decode(carriers[0].split(panel, 1)[1])
            assert rows == EXPECTED
            enriched.append(messages)
    assert len(enriched) >= 2
    assert any(any(row["role"] == "tool" and "eligible,unreviewed" in row["content"]
                   for row in messages) for messages in enriched)

    pressured = _capture(provider, ROOT, tmp_path, "pressure", switch="on", pressure=True)
    assert all(panel.strip().encode() not in payload for payload in pressured)
    assert any(b"eligible,unreviewed" in payload for payload in pressured)

    root = tmp_path / "enabled-repo"
    source = root / SPEC["path"]
    task = _task()
    assert framefit._report_denominator_rows(task, root) == EXPECTED
    crlf_source = CSV.replace(b"\n", b"\r\n")
    (source.parent / "crlf.csv").write_bytes(crlf_source)
    crlf = framefit._report_denominator_rows(_task(dict(SPEC, path="input/crlf.csv")), root)
    assert crlf == [
        dict(row, source="input/crlf.csv", source_revision="sha256:" + hashlib.sha256(crlf_source).hexdigest())
        for row in EXPECTED
    ]
    without_outcome = {key: value for key, value in SPEC.items() if key != "outcome"}
    unknown = framefit._report_denominator_rows(_task(without_outcome), root)
    assert len(unknown) == 2
    assert all(row["numerator"] is None and row["observed_denominator"] is None
               and row["eligible_population"] == 3 for row in unknown)
    (source.parent / "alias.csv").symlink_to(source)
    for spec in (dict(SPEC, path="../outside.csv"), dict(SPEC, path="input/alias.csv")):
        assert framefit._report_denominator_rows(_task(spec), root) == []

    capture = SimpleNamespace(request_id="report", final_tool_surface={"request_id": "report", "tools": []},
                              _capacity_retry_original_body={})
    runtime = SimpleNamespace(owner=SimpleNamespace(root=root), _active_capture=capture, current_request_id="report",
                              _framefit_action_context={"report_denominators": EXPECTED})
    messages = [{"role": "user", "content": framefit.VOLATILE_TAIL_MARKER}]
    monkeypatch.setattr(framefit, "_report_denominators_fit", lambda *args: True)
    framefit._attach_report_denominators(messages, runtime, {}, qualified_surface=True)
    enriched_once = messages[0]["content"]
    assert panel in enriched_once
    framefit._attach_report_denominators(messages, runtime, {}, qualified_surface=True)
    assert messages[0]["content"] == enriched_once
    with source.open("ab") as stream:
        stream.write(b"9,Login,eligible,open\n")
    framefit._attach_report_denominators(messages, runtime, {}, qualified_surface=True)
    assert messages[0]["content"] == framefit.VOLATILE_TAIL_MARKER
    with source.open("ab") as stream:
        stream.write(b"1,Login,eligible,closed\r\n")
    assert framefit._report_denominator_rows(task, root) == []
    source.write_bytes(CSV.ljust(framefit.MAX_SITE_BYTES + 1, b" "))
    assert framefit._report_denominator_rows(task, root) == []
