"""Exercise the remaining-sites experiment on a scripted provider wire."""
from __future__ import annotations

import json
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
from fixtures.process_signals import signal_run_owned
from engine_run import run_monitored, sandbox_test_python, start_monitored
from tools.pinned_tree import pinned_tree

from stateframe.scripted_endpoint import Script, make_handler, serve

ROOT = Path(__file__).resolve().parents[1]
SWITCH = "STATEFRAME_R200_R082_MINIMIZE_REMAINING_SITES"
FULL_SWITCH = "STATEFRAME_REMAINING_SITES"
PREFIX = "\n[host] Remaining ticket site: "
TASK = "Disable `LEGACY_MODE = True` in app.py and worker.py."


def _step(tool, **arguments):
    return {"tool": tool, "arguments": arguments}


def _scripted_run(folder, tree, *, compact=None, full="off"):
    folder.mkdir()
    repo = folder / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "app.py").write_text("LEGACY_MODE = True\n", encoding="utf-8")
    (repo / "worker.py").write_text("# LEGACY_MODE = True\nLEGACY_MODE = True\n", encoding="utf-8")
    (repo / "tests/test_app.py").write_text(
        "from app import LEGACY_MODE\n\ndef test_app():\n    assert LEGACY_MODE is False\n",
        encoding="utf-8",
    )
    steps = [
        _step("read", path="app.py"),
        _step("str_replace", path="app.py", old_str="LEGACY_MODE = True", new_str="LEGACY_MODE = False"),
        _step("read", path="worker.py"),
        _step("str_replace", path="worker.py", old_str="\nLEGACY_MODE = True", new_str="\nLEGACY_MODE = False"),
        _step("done", summary="Disabled both legacy modes."),
    ]
    script = Script({"model": "scripted", "steps": steps, "tokenize": {"mode": "ok"}})
    received = []
    handler = make_handler(script)
    pending = Event()
    release = Event()

    class WireHandler(handler):
        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return super().do_POST()
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            received.append(raw)
            if len(received) == 3:
                self._send(429, {"error": {"message": "rate limit"}})
                return
            if len(received) == 4:
                pending.set()
                release.wait()
                self.close_connection = True
                return
            if len(received) == 5:
                self._send(400, {"error": {"code": "context_length_exceeded"}})
                return
            reader = self.rfile
            try:
                self.rfile = BytesIO(raw)
                return super().do_POST()
            finally:
                self.rfile = reader

    server = serve(script)
    server.RequestHandlerClass = WireHandler
    profile = json.loads((tree / "src/stateframe/profiles/r2hc3.json").read_bytes())
    profile["switches"][FULL_SWITCH] = full
    profile["switches"]["STATEFRAME_CHECKPOINT_EVERY"] = "1"
    profile_path = folder / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("STATEFRAME_", "REACT_", "LLAMA_CPP_", "SF_"))}
    environment.update({
        "HOME": str(folder), "STATEFRAME_HOME": str(folder / "home"),
        "PYTHONPATH": os.pathsep.join(str(tree / path) for path in ("src", "src/stateframe/engine")),
        "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
        "STATEFRAME_TRANSPORT_BACKOFF_S": "0",
    })
    if compact is not None:
        environment[SWITCH] = compact
    run = folder / "run"
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        with sandbox_test_python(sys.executable) as test_python:
            command = [
                sys.executable, "-m", "stateframe", "run", "--repo", str(repo), "--task", TASK,
                "--target", "app.py", "--edit", "worker.py", "--check", "tests/test_app.py",
                "--endpoint", endpoint, "--model", "scripted",
                "--run-dir", str(run), "--test-python", test_python, "--no-run-env",
                "--profile", str(profile_path), "--quiet",
            ]
            with start_monitored(command, env=environment, expected_seconds=120) as first:
                while first.poll() is None and not pending.wait(0.1):
                    pass
                assert pending.is_set(), first.stdout_text + first.stderr_text
                signal_run_owned(run, first, signal.SIGTERM, deadline=time.monotonic() + 60,
                                 what="r082 first run before resume")
                assert first.wait() != 0, first.stdout_text + first.stderr_text
            release.set()
            assert len(received) == 4 and script.position == 2
            process = run_monitored([
                sys.executable, "-m", "stateframe", "resume", str(run),
                "--endpoint", endpoint, "--quiet",
            ], env=environment, capture_output=True, text=True, expected_seconds=120)
        traceback = run / "child-traceback.txt"
        assert process.returncode == 0, (process.stdout + process.stderr
                                          + (traceback.read_text() if traceback.exists() else ""))
        script.assert_consumed()
        assert len(received) == len(steps) + 3
        assert received[2] == received[3] == received[4]
        assert len(received[5]) < len(received[4])
        assert not (run / "resume-divergence.json").exists()
        return received
    finally:
        release.set()
        server.shutdown()
        server.server_close()


@pytest.mark.host_sandbox
@pytest.mark.parametrize(("full", "compact"), [("off", None), ("on", "off")])
def test_r082_minimize_remaining_sites_off_request_bytes_match_pinned_099(
        tmp_path, tmp_path_factory, full, compact):
    base = pinned_tree("29765b39820fd39800fd7292f52fb5bcb81e6069", tmp_path_factory)
    with ThreadPoolExecutor(max_workers=2) as executor:
        pinned = executor.submit(_scripted_run, tmp_path / "pinned", base, full=full)
        current = executor.submit(_scripted_run, tmp_path / "current", ROOT, compact=compact, full=full)
        assert current.result() == pinned.result()


def test_r082_minimize_remaining_sites_switch_defaults_off(monkeypatch):
    from kernel_remaining_sites import compact_enabled

    from stateframe import runner

    monkeypatch.delenv(SWITCH, raising=False)
    assert not compact_enabled()
    assert SWITCH not in runner.resolve_opt_in_switches()
    assert runner.OPT_IN_SWITCHES[SWITCH] == ("off", "on")
    profile = json.loads((ROOT / "src/stateframe/profiles/r2hc3.json").read_bytes())
    # The shipped profile leaves the switch unset so checkpoint manifests stay byte-identical to the previous release.
    assert SWITCH not in profile["switches"]
    monkeypatch.setenv(SWITCH, "false")
    assert not compact_enabled()
    assert runner.resolve_opt_in_switches()[SWITCH] == "off"
    monkeypatch.setenv(SWITCH, "on")
    assert compact_enabled()
    assert runner.resolve_opt_in_switches()[SWITCH] == "on"


@pytest.mark.host_sandbox
def test_r082_minimize_remaining_sites_on_path_is_bounded_and_idempotent(tmp_path):
    received = _scripted_run(tmp_path / "compact", ROOT, compact="on")
    assert [index for index, raw in enumerate(received) if PREFIX.strip().encode() in raw] == [2, 3, 4]
    messages = json.loads(received[3])["messages"]
    tool = [message for message in messages if message["role"] == "tool"][-1]
    assert tool["content"].count(PREFIX) == 1
    assert tool["content"].endswith(PREFIX + "worker.py:2")
    assert "status=PASS" in tool["content"]
    assert "Sites already changed:" not in tool["content"]


def test_compact_refresh_removes_only_the_owned_suffix(tmp_path, monkeypatch):
    import kernel_remaining_sites as sites
    from kernel_measurement import RequestCapture
    from kernel_request_budget import _wire_bytes

    monkeypatch.setattr("kernel_greenfact._green_revision", lambda runtime: (runtime.revision, None))
    monkeypatch.setattr("kernel_request_budget._request_token_meter",
                        lambda *_args, **_kwargs: SimpleNamespace(measure=lambda _body: (1, "tokenize")))
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "off")
    parent = "LEGACY_MODE = True\n"
    (tmp_path / "app.py").write_text("LEGACY_MODE = False\n", encoding="utf-8")
    worker = tmp_path / "worker.py"
    worker.write_text("# " + parent + parent, encoding="utf-8")
    runtime = SimpleNamespace(
        owner=SimpleNamespace(root=tmp_path, task_row={
            "msg": TASK, "target": "app.py",
            "files": {name: {"parent_src": parent} for name in ("app.py", "worker.py")},
        }),
        revision="tested", frame_ceiling=1000,
        host_check_receipts=[{"status": "PASS", "execution_complete": True,
                              "revision": "tested", "summary": "trusted pass"}],
    )
    quoted = "trusted pass" + PREFIX + "quoted.py:90"
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "check", "function": {"name": "run_tests"}}]},
        {"role": "tool", "tool_call_id": "check", "content": quoted},
    ]
    body = {"messages": messages}
    capture = object.__new__(RequestCapture)
    capture.request_budget_final_wire = {
        "status": "READY", "max_request_bytes": _wire_bytes(body), "max_prompt_tokens": 1000,
    }
    capture._framefit5_semantic_messages = messages
    profile = SimpleNamespace(mode="on")

    def refresh(request):
        return capture._refresh_compact_remaining_site(request, runtime, "mock", dialect="llama.cpp",
                                                       profile=profile)

    assert refresh(body) == body  # The entire budget is occupied by existing evidence.
    capture.request_budget_final_wire["max_request_bytes"] = 4000
    assert sites.render_remaining_sites(runtime, compact=True) == PREFIX + "worker.py:2"
    attached = refresh(body)
    assert attached["messages"][-1]["content"] == quoted + PREFIX + "worker.py:2"
    assert refresh(attached) == attached
    assert messages[-1]["content"] == quoted

    runtime.revision = "current"
    worker.write_text("# file changed after the check\n# " + parent + parent, encoding="utf-8")
    assert refresh(attached) == body
    assert capture._framefit5_semantic_messages == messages
    worker.write_text("# " + parent + "LEGACY_MODE = False\n", encoding="utf-8")
    assert sites.render_remaining_sites(runtime, compact=True) == ""


def test_compact_refresh_respects_learned_prompt_limit(monkeypatch):
    import kernel_remaining_sites as sites
    from kernel_measurement import RequestCapture
    from kernel_request_budget import ENDPOINT_PROMPT_TOKEN_BUDGET, _wire_bytes

    measured = []

    def measure(body):
        has_pointer = PREFIX.strip() in json.dumps(body)
        measured.append(has_pointer)
        return (5387 if has_pointer else 5370, "learned-tokenizer")

    monkeypatch.setattr("kernel_context.token_estimate", lambda _body: 100)
    monkeypatch.setattr("kernel_request_budget._request_token_meter",
                        lambda *_args, **_kwargs: SimpleNamespace(measure=measure))
    monkeypatch.setattr(sites, "compact_attachment", lambda _messages, _runtime:
                        ("check", PREFIX + "worker.py:2"))
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "off")

    messages = [
        {"role": "assistant", "tool_calls": [{"id": "check", "function": {"name": "run_tests"}}]},
        {"role": "tool", "tool_call_id": "check", "content": "trusted pass"},
    ]
    body = {"messages": messages}
    capture = object.__new__(RequestCapture)
    capture.request_budget_final_wire = {
        "status": "READY", "max_request_bytes": 1_000_000,
        "max_prompt_tokens": ENDPOINT_PROMPT_TOKEN_BUDGET,
    }
    capture._capacity_retry_limits = SimpleNamespace(max_prompt_tokens=5376)
    capture._framefit5_semantic_messages = messages
    runtime = SimpleNamespace(frame_ceiling=100_000)
    profile = SimpleNamespace(mode="on")

    assert measure(body)[0] <= capture._capacity_retry_limits.max_prompt_tokens
    refreshed = capture._refresh_compact_remaining_site(
        body, runtime, "mock", dialect="llama.cpp", profile=profile,
    )

    assert measured == [False, True]
    assert _wire_bytes(body) <= capture.request_budget_final_wire["max_request_bytes"]
    assert refreshed == body


def test_working_set_gate_respects_learned_prompt_limit(monkeypatch):
    from dataclasses import dataclass

    import kernel_remaining_sites as sites
    from kernel_request_budget import ENDPOINT_PROMPT_TOKEN_BUDGET
    import kernel_working_set_live as live_module
    import kernel_working_set_prepare as prepare_module
    import kernel_working_set_render as render_module
    import kernel_working_set_shadow as shadow_module
    import kernel_working_set_snapshot as snapshot_module

    @dataclass(frozen=True)
    class Snapshot:
        frame_limit: int
        layouts: tuple[str, ...]

    slot = SimpleNamespace(key="slot", richest_first=(("richest", 0),))
    monkeypatch.setattr(snapshot_module, "PreparationSnapshot", Snapshot)
    monkeypatch.setattr(shadow_module, "freeze_shadow",
                        lambda *_args, **_kwargs: Snapshot(frame_limit=1, layouts=("old",)))
    monkeypatch.setattr(render_module, "propose_slots", lambda _snapshot: (slot,))
    monkeypatch.setattr(render_module, "render_assignment", lambda *_args: object())
    monkeypatch.setattr(render_module, "materialize", lambda _rendered: SimpleNamespace(wire=b"wire"))
    monkeypatch.setattr(live_module, "live_measure", lambda _meter: lambda _wire:
                        SimpleNamespace(contract_valid=True, prompt_tokens=5387))
    monkeypatch.setattr(prepare_module, "_violations", lambda *_args: ())

    capture = SimpleNamespace(
        request_id="request",
        request_budget_final_wire={"max_prompt_tokens": ENDPOINT_PROMPT_TOKEN_BUDGET},
        _capacity_retry_limits=SimpleNamespace(max_prompt_tokens=5376),
    )
    runtime = SimpleNamespace(frame_ceiling=100_000)

    assert not sites.fits_working_set({}, runtime, capture, object(), "llama.cpp", "on")
