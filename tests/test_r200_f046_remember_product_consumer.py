"""Verified callback memory at the StateFrame provider boundary."""
from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from _timeouts import scaled
from engine_run import start_monitored
from fixtures.process_signals import record_identity, signal_run_owned, signal_run_verified
from kernel_request_evidence_closure import source_revision
from kernel_r200_f046_remember_product_consumer import (
    MAX_ROWS, PREFIX, SWITCH, enabled,
)
from stateframe import runner
from stateframe.scripted_endpoint import Script, ScriptStepUnavailable, serve
from tools.pinned_tree import pinned_tree

ROOT = Path(__file__).resolve().parents[1]
CONSUMER = """from producer import deliver

def receipt(payload):
    audit.store(payload)

def start(payload):
    deliver(payload, on_done=receipt)
"""
PRODUCER = """def deliver(payload, *, on_done):
    on_done(payload)
"""


def _read(path):
    return {"tool": "read", "arguments": {"path": path}}


def _wait_for_checkpoint_sends(run, process, *, minimum, deadline):
    events = run / "checkpoints" / "events.jsonl"
    while True:
        try:
            lines = events.read_bytes().splitlines(keepends=True)
        except FileNotFoundError:
            lines = ()
        for line in lines:
            if not line.endswith(b"\n"):
                continue
            try:
                event = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if (isinstance(event, dict) and event.get("event") == "checkpoint_saved"
                    and type(event.get("sends")) is int and event["sends"] >= minimum):
                return
        assert process.poll() is None, "engine exited before the required checkpoint was saved"
        remaining = deadline - time.monotonic()
        assert remaining > 0, f"checkpoint with at least {minimum} sends was not saved"
        time.sleep(min(0.05, remaining))


def _scenario(checkout, folder, *, switch=None, allocator="on", resume=None, requalify=False):
    repo = folder / "repo"
    repo.mkdir(parents=True)
    (repo / "consumer.py").write_text(CONSUMER)
    (repo / "producer.py").write_text(PRODUCER)
    run = folder / "run"
    blocked, release = threading.Event(), threading.Event()

    class Provider(Script):
        def next_call(self, offered):
            if resume and self.position == 2 and not release.is_set():
                blocked.set()
                release.wait(scaled(180))
                raise ScriptStepUnavailable("test interrupted after recording the pending request")
            return super().next_call(offered)

    script = Provider({"tokenize": {"mode": "ok"}, "steps": [
        _read("consumer.py"), _read("producer.py"),
        _read("producer.py"),
    ]})
    server = serve(script)
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    original_post = server.RequestHandlerClass.do_POST
    received = []

    def raw_post(handler):
        if handler.path.rstrip("/") != "/v1/chat/completions":
            return original_post(handler)
        stream = handler.rfile
        wire = stream.read(int(handler.headers["Content-Length"]))
        assert handler.headers.get("Content-Encoding") is None
        received.append(wire)
        if requalify and len(received) == 3:
            handler._send(400, {"error": {"code": "context_length_exceeded",
                                        "message": "maximum context length is 32768 tokens"}})
            return None
        handler.rfile = io.BytesIO(wire)
        try:
            return original_post(handler)
        finally:
            handler.rfile = stream

    server.RequestHandlerClass.do_POST = raw_post
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("STATEFRAME_", "REACT_"))}
    env.update(PYTHONPATH=str(checkout / "src"), PYTHONDONTWRITEBYTECODE="1",
               HOME=str(folder / "home"), STATEFRAME_HOME=str(folder / "state-home"),
               STATEFRAME_WORKING_SET_ALLOCATOR=allocator)
    if switch is not None:
        env[SWITCH] = switch
    if resume == "replay":
        env["STATEFRAME_CHECKPOINT"] = "off"
    elif resume == "checkpoint":
        env["STATEFRAME_CHECKPOINT_EVERY"] = "1"
    cli = [sys.executable, "-m", "stateframe"]
    options = ["--endpoint", endpoint, "--model", script.model, "--quiet", "--json"]
    command = [*cli, "run", "--repo", str(repo), "--run-dir", str(run),
               "--target", "producer.py", "--task", "Inspect the callback delivery wiring.",
               "--max-turns", "3", "--runner-kind", "generic", "--test-cmd", "true",
               "--keep-checkpoints", "--no-run-env", *options]
    subprocess_args = {"env": env, "cwd": folder, "text": True}
    process = None
    try:
        if resume is None:
            result = subprocess.run(command, capture_output=True, timeout=scaled(180),
                                    **subprocess_args)
        else:
            started = time.time()
            process = start_monitored(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **subprocess_args,
                run_dir=run, expected_seconds=scaled(180), step="F046 provider interruption",
            )
            deadline = time.monotonic() + scaled(150)
            while not blocked.wait(0.25):
                assert process.poll() is None, process.communicate()
                assert time.monotonic() < deadline, "pending producer request did not reach provider"
            pending = received[-1]
            if resume == "checkpoint":
                _wait_for_checkpoint_sends(run, process, minimum=2, deadline=deadline)
                assert received[-1] == pending
            (pid,) = subprocess.check_output(["pgrep", "-P", str(process.pid)], text=True).split()
            identity = record_identity(int(pid), launched_after=started, command_fragment="_child.py")
            signal_run_verified(run, int(pid), identity, signal.SIGKILL,
                                deadline=deadline, what="F046 provider interruption",
                                require_checkpoint_saved=resume == "checkpoint")
            release.set()
            process.communicate(timeout=scaled(30))
            result = subprocess.run([*cli, "resume", str(run), *options], capture_output=True,
                                    timeout=scaled(180), **subprocess_args)
            assert len(received) == 4, result.stdout + result.stderr
            assert received[-1] == pending
            received.pop()
            log = (run / "engine.log").read_text()
            if resume == "checkpoint":
                assert '"checkpoint_verified"' in log, log[-4000:]
                selected = json.loads((run / "spec.json").read_text())["resume"]["checkpoint"]
                assert selected["sends"] == 2
            else:
                assert '"replay_done"' in log, log[-4000:]
        assert (run / "summary.json").is_file(), result.stdout + result.stderr
        summary = json.loads((run / "summary.json").read_text())
        assert summary["engine_result"] != "ENGINE_ERROR", summary
        assert len(received) == (4 if requalify else 3), (summary, result.stdout, result.stderr)
        if requalify:
            assert received[2] != received[3]
        script.assert_consumed()
        return received
    finally:
        release.set()
        if process is not None and process.poll() is None:
            signal_run_owned(run, process, signal.SIGKILL, deadline=time.monotonic(),
                             what="F046 test cleanup", require_checkpoint_saved=False, group=True)
            process.communicate(timeout=scaled(10))
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize(("allocator", "resume"), [
    ("on", None), ("off", None), ("on", "replay"), ("on", "checkpoint"),
])
def test_f046_remember_product_consumer_off_request_bytes_match_pinned_099(
    tmp_path, tmp_path_factory, allocator, resume,
):
    baseline = pinned_tree("29765b39820fd39800fd7292f52fb5bcb81e6069", tmp_path_factory)
    requalify = allocator == "on" and resume is None
    expected = _scenario(baseline, tmp_path / "baseline", allocator=allocator, resume=resume,
                         requalify=requalify)
    for explicit in (None, "false"):
        current = _scenario(ROOT, tmp_path / str(explicit), switch=explicit,
                            allocator=allocator, resume=resume, requalify=requalify)
        assert current == expected  # HTTP entity bytes, including whitespace, ordering and schemas.
    assert all(json.loads(wire)["tools"] for wire in expected)
    assert b"on_done" in expected[-1]


def test_f046_remember_product_consumer_switch_defaults_off(monkeypatch):
    monkeypatch.delenv(SWITCH, raising=False)
    assert SWITCH in runner.OPT_IN_SWITCHES
    assert SWITCH in runner.engine_switches()
    assert SWITCH not in runner.resolve_opt_in_switches()
    assert not enabled()
    monkeypatch.setenv(SWITCH, "false")
    assert runner.resolve_opt_in_switches()[SWITCH] == "off"
    assert not enabled()
    monkeypatch.setenv(SWITCH, "on")
    assert runner.resolve_opt_in_switches()[SWITCH] == "on"
    assert enabled()


def _receipt(root, **ranges):
    receipt = {
        "status": "COMPLETE", "proof_status": "VERIFIED",
        "submitted_http_bytes_digest": source_revision("fixture"),
        "paths": {}, "delivery_proofs": {}, "final_message_locations": {},
    }
    for index, (path, spans) in enumerate(ranges.items()):
        raw = (root / path).read_bytes()
        lines = raw.splitlines(keepends=True)
        revision = source_revision(raw)
        location = f"$.messages[{index}].content"
        receipt["paths"][path] = {"revision": revision, "proof_status": "VERIFIED"}
        receipt["final_message_locations"][path] = [location]
        receipt["delivery_proofs"][path] = [
            {"key": {"canonical_path": path, "content_digest": revision,
                     "byte_start": len(b"".join(lines[:first - 1])),
                     "byte_end": len(b"".join(lines[:last]))},
             "wire_contains_exact": True, "transport_identity": "VERIFIED",
             "final_message_location": location,
             "submitted_http_bytes_digest": receipt["submitted_http_bytes_digest"],
             "delivered_digest": source_revision(b"".join(lines[first - 1:last]))}
            for first, last in spans
        ]
    return receipt


def _fact(root, runtime, path="producer.py", start=2, end=2, line=None):
    from kernel_r200_f046_remember_product_consumer import select

    site = {"path": path, "revision": source_revision((root / path).read_bytes()),
            "lines": {"start": start, "end": end}}
    if line is not None:
        site["line"] = line
    return select(runtime, {"next_site": site})


def test_f046_remember_product_consumer_on_path_is_bounded_and_idempotent(tmp_path):
    from kernel_r200_f046_remember_product_consumer import remember
    from kernel_stateframe_framefit5 import effect_consumer_tail, VOLATILE_TAIL_MARKER

    root = tmp_path
    (root / "consumer.py").write_text(CONSUMER)
    producer = root / "producer.py"
    producer.write_text(PRODUCER)
    runtime = SimpleNamespace(owner=SimpleNamespace(root=root))
    remember(runtime, _receipt(root, **{"consumer.py": [(1, 4)]}))
    assert _fact(root, runtime) is None
    forged = _receipt(root, **{"consumer.py": [(1, 7)]})
    forged["delivery_proofs"]["consumer.py"][0]["delivered_digest"] = source_revision("wrong")
    remember(runtime, forged)
    assert _fact(root, runtime) is None
    receipt = _receipt(root, **{"consumer.py": [(1, 7)], "producer.py": [(1, 2)]})
    remember(runtime, receipt)
    fact = _fact(root, runtime)
    assert fact is not None
    assert json.loads(fact.removeprefix(PREFIX)) == {
        "binding": "consumer.py:7", "callback": "deliver.on_done",
        "consumer": "consumer.py:4 audit.store(payload)", "firing": "producer.py:2",
    }
    remember(runtime, receipt)
    assert len(runtime._effect_consumer_memory) == 1
    tail = effect_consumer_tail(VOLATILE_TAIL_MARKER + "\n\nexisting evidence", fact)
    assert effect_consumer_tail(tail, fact) == tail
    assert effect_consumer_tail(tail, None) == VOLATILE_TAIL_MARKER + "\n\nexisting evidence"

    producer.write_text(PRODUCER.replace("    on_done", "    result = payload\n    on_done"))
    stale = _fact(root, runtime)
    assert stale and "audit.store" in stale and '"firing"' not in stale
    remember(runtime, _receipt(root, **{"producer.py": [(1, 3)]}))
    assert '"firing":"producer.py:3"' in _fact(root, runtime)
    producer.write_text(PRODUCER.replace("    on_done", "    if on_done is not None:\n        on_done"))
    remember(runtime, _receipt(root, **{"producer.py": [(1, 3)]}))
    assert '"firing":"producer.py:3"' in _fact(root, runtime)
    producer.write_text(PRODUCER + "\ndef deliver(payload, *, on_done):\n    return payload\n")
    assert _fact(root, runtime) is None
    producer.write_text(PRODUCER)
    (root / "consumer.py").write_text(CONSUMER.replace("audit.store", "audit.delete"))
    assert _fact(root, runtime) is None

    for index in range(MAX_ROWS + 4):
        path = f"consumer_{index:02}.py"
        (root / path).write_text(CONSUMER)
        remember(runtime, _receipt(root, **{path: [(1, 7)]}))
    assert len(runtime._effect_consumer_memory) == MAX_ROWS
    assert '"binding":"consumer_04.py:7"' in _fact(root, runtime)


def test_f046_resolves_constructor_registration_to_distant_method(tmp_path):
    from kernel_r200_f046_remember_product_consumer import remember

    (tmp_path / "consumer.py").write_text(CONSUMER.replace("deliver", "Worker").replace(
        "Worker(payload, on_done=receipt)", "Worker(on_done=receipt)"))
    (tmp_path / "producer.py").write_text(
        "class Worker:\n"
        "    def __init__(self, *, on_done):\n"
        "        self.callback = on_done\n"
        "\n"
        "    def run(self, payload):\n"
        "        self.callback(payload)\n"
    )
    runtime = SimpleNamespace(owner=SimpleNamespace(root=tmp_path))
    remember(runtime, _receipt(tmp_path, **{"consumer.py": [(1, 7)], "producer.py": [(1, 2), (5, 6)]}))
    assert '"firing"' not in _fact(tmp_path, runtime, start=6, end=6)
    remember(runtime, _receipt(tmp_path, **{"producer.py": [(1, 6)]}))
    assert '"firing":"producer.py:6"' in _fact(tmp_path, runtime, start=6, end=6)


def test_f046_instance_method_override_does_not_claim_class_firing(tmp_path):
    from kernel_r200_f046_remember_product_consumer import remember

    consumer = (
        "from producer import Worker\n"
        "\n"
        "def on_error(error):\n"
        "    audit.report(error)\n"
        "\n"
        "def start():\n"
        "    worker = Worker()\n"
        "    worker.deliver('payload', on_error=on_error)\n"
    )
    producer = (
        "class Worker:\n"
        "    def __init__(self):\n"
        "        self.deliver = self.enqueue\n"
        "\n"
        "    def deliver(self, payload, *, on_error):\n"
        "        on_error(payload)\n"
        "\n"
        "    def enqueue(self, payload, *, on_error):\n"
        "        self.queue.put((payload, on_error))\n"
    )
    (tmp_path / "consumer.py").write_text(consumer)
    (tmp_path / "producer.py").write_text(producer)
    runtime = SimpleNamespace(owner=SimpleNamespace(root=tmp_path))
    remember(runtime, _receipt(tmp_path, **{"consumer.py": [(1, 8)], "producer.py": [(1, 9)]}))

    assert _fact(tmp_path, runtime, start=6, end=6) is None


def test_f046_same_producer_callback_parameters_keep_their_consumer_identity(tmp_path):
    from kernel_r200_f046_remember_product_consumer import remember

    consumer = (
        "from producer import deliver\n"
        "\n"
        "def on_done(value):\n"
        "    audit.done(value)\n"
        "\n"
        "def on_error(error):\n"
        "    audit.error(error)\n"
        "\n"
        "def start():\n"
        "    deliver('payload', on_done=on_done, on_error=on_error)\n"
    )
    producer = (
        "def deliver(*, on_done, on_error):\n"
        "    on_done()\n"
        "    on_error()\n"
    )
    (tmp_path / "consumer.py").write_text(consumer)
    (tmp_path / "producer.py").write_text(producer)
    runtime = SimpleNamespace(owner=SimpleNamespace(root=tmp_path))
    remember(runtime, _receipt(tmp_path, **{"consumer.py": [(1, 10)], "producer.py": [(1, 3)]}))

    fact = _fact(tmp_path, runtime, start=3, end=3)
    assert fact is not None
    assert json.loads(fact.removeprefix(PREFIX)) == {
        "binding": "consumer.py:10", "callback": "deliver.on_error",
        "consumer": "consumer.py:7 audit.error(error)", "firing": "producer.py:3",
    }
    assert _fact(tmp_path, runtime, start=1, end=3) is None


def test_f046_from_package_export_does_not_claim_same_named_submodule_firing(
    tmp_path, monkeypatch,
):
    import importlib

    from kernel_r200_f046_remember_product_consumer import remember

    package = "callback_pkg_" + tmp_path.name.replace("-", "_")
    folder = tmp_path / package
    folder.mkdir()
    (folder / "__init__.py").write_text(
        "class ExportedClient:\n"
        "    def send(self, payload, *, on_done):\n"
        "        self.callback = on_done\n"
        "client = ExportedClient()\n"
    )
    (folder / "client.py").write_text(
        "def send(payload, *, on_done):\n"
        "    on_done(payload)\n"
    )
    (folder / "consumer.py").write_text(
        "from . import client\n"
        "\n"
        "def receipt(payload):\n"
        "    audit.store(payload)\n"
        "\n"
        "def start(payload):\n"
        "    client.send(payload, on_done=receipt)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    consumer_module = importlib.import_module(f"{package}.consumer")
    package_module = importlib.import_module(package)
    assert consumer_module.client is package_module.client
    consumer_module.start("payload")
    assert package_module.client.callback is consumer_module.receipt

    runtime = SimpleNamespace(owner=SimpleNamespace(root=tmp_path))
    remember(runtime, _receipt(tmp_path, **{
        f"{package}/consumer.py": [(1, 7)],
        f"{package}/client.py": [(1, 2)],
    }))
    assert _fact(tmp_path, runtime, f"{package}/client.py", start=1, end=2) is None


def test_f046_root_package_relative_client_import_is_resolved(tmp_path, monkeypatch):
    import importlib

    from kernel_r200_f046_remember_product_consumer import remember

    package = "callback_repo_" + tmp_path.name.replace("-", "_")
    root = tmp_path / package
    root.mkdir()
    (root / "__init__.py").write_text(
        "from . import client\n"
        "\n"
        "def receipt(payload):\n"
        "    audit.store(payload)\n"
        "\n"
        "def start(payload):\n"
        "    client.deliver(payload, on_done=receipt)\n"
    )
    (root / "client.py").write_text(PRODUCER)
    monkeypatch.syspath_prepend(str(tmp_path))
    package_module = importlib.import_module(package)
    package_module.audit = SimpleNamespace(store=lambda payload: None)
    package_module.start("payload")

    runtime = SimpleNamespace(owner=SimpleNamespace(root=root))
    receipt = _receipt(root, **{"__init__.py": [(1, 7)], "client.py": [(1, 2)]})
    remember(runtime, receipt)
    fact = _fact(root, runtime, "client.py", start=1, end=2)
    assert fact is not None
    assert json.loads(fact.removeprefix(PREFIX)) == {
        "binding": "__init__.py:7", "callback": "deliver.on_done",
        "consumer": "__init__.py:4 audit.store(payload)", "firing": "client.py:2",
    }


def test_f046_options_refactor_keeps_current_consumer_without_stale_firing(tmp_path):
    from kernel_r200_f046_remember_product_consumer import remember

    (tmp_path / "consumer.py").write_text(CONSUMER)
    producer = tmp_path / "producer.py"
    producer.write_text(PRODUCER)
    runtime = SimpleNamespace(owner=SimpleNamespace(root=tmp_path))
    remember(runtime, _receipt(tmp_path, **{
        "consumer.py": [(1, 7)], "producer.py": [(1, 2)],
    }))

    producer.write_text(
        "def deliver(payload, **options):\n"
        "    options['on_done'](payload)\n"
    )
    fact = _fact(tmp_path, runtime, start=1, end=2)
    assert fact is not None
    fields = json.loads(fact.removeprefix(PREFIX))
    assert fields["consumer"] == "consumer.py:4 audit.store(payload)"
    assert "firing" not in fields


def test_f046_bare_call_of_async_callback_is_not_recorded_as_firing(tmp_path):
    from kernel_r200_f046_remember_product_consumer import remember

    (tmp_path / "consumer.py").write_text(
        "from producer import deliver\n"
        "\n"
        "async def receipt(payload):\n"
        "    await audit.store(payload)\n"
        "\n"
        "def start(payload):\n"
        "    deliver(payload, on_done=receipt)\n"
    )
    (tmp_path / "producer.py").write_text(PRODUCER)
    runtime = SimpleNamespace(owner=SimpleNamespace(root=tmp_path))
    remember(runtime, _receipt(tmp_path, **{
        "consumer.py": [(1, 7)], "producer.py": [(1, 2)],
    }))

    fact = _fact(tmp_path, runtime)
    assert fact is not None
    fields = json.loads(fact.removeprefix(PREFIX))
    assert fields["consumer"] == "consumer.py:4 audit.store(payload)"
    assert "firing" not in fields


@pytest.mark.parametrize(("producer_source", "firing_line"), [
    ("async def deliver(payload, *, on_done):\n    await on_done(payload)\n", 2),
    ("import asyncio\n\n"
     "def deliver(payload, *, on_done):\n"
     "    asyncio.create_task(on_done(payload))\n", 4),
])
def test_f046_executed_async_callback_keeps_its_firing_site(
    tmp_path, producer_source, firing_line,
):
    from kernel_r200_f046_remember_product_consumer import remember

    (tmp_path / "consumer.py").write_text(
        "from producer import deliver\n"
        "\n"
        "async def receipt(payload):\n"
        "    await audit.store(payload)\n"
        "\n"
        "def start(payload):\n"
        "    deliver(payload, on_done=receipt)\n"
    )
    (tmp_path / "producer.py").write_text(producer_source)
    runtime = SimpleNamespace(owner=SimpleNamespace(root=tmp_path))
    end_line = len(producer_source.splitlines())
    remember(runtime, _receipt(tmp_path, **{
        "consumer.py": [(1, 7)], "producer.py": [(1, end_line)],
    }))

    fact = _fact(tmp_path, runtime, start=1, end=end_line)
    assert fact is not None
    fields = json.loads(fact.removeprefix(PREFIX))
    assert fields["consumer"] == "consumer.py:4 audit.store(payload)"
    assert fields["firing"] == f"producer.py:{firing_line}"


def test_f046_shadowed_create_task_does_not_count_as_async_scheduling(tmp_path):
    from kernel_r200_f046_remember_product_consumer import remember

    (tmp_path / "consumer.py").write_text(
        "from producer import deliver\n"
        "\n"
        "async def receipt(payload):\n"
        "    await audit.store(payload)\n"
        "\n"
        "def start(payload):\n"
        "    deliver(payload, on_done=receipt)\n"
    )
    (tmp_path / "producer.py").write_text(
        "def create_task(coroutine):\n"
        "    return None\n"
        "\n"
        "def deliver(payload, *, on_done):\n"
        "    create_task(on_done(payload))\n"
    )
    runtime = SimpleNamespace(owner=SimpleNamespace(root=tmp_path))
    remember(runtime, _receipt(tmp_path, **{
        "consumer.py": [(1, 7)], "producer.py": [(1, 5)],
    }))

    fact = _fact(tmp_path, runtime, start=4, end=5)
    assert fact is not None
    assert "firing" not in json.loads(fact.removeprefix(PREFIX))


def test_f046_exact_edit_line_selects_matching_callback_firing(tmp_path):
    from kernel_r200_f046_remember_product_consumer import remember

    (tmp_path / "consumer.py").write_text(
        "from producer import deliver\n"
        "\n"
        "def on_done(value):\n"
        "    audit.done(value)\n"
        "\n"
        "def on_error(error):\n"
        "    audit.error(error)\n"
        "\n"
        "def start():\n"
        "    deliver('payload', on_done=on_done, on_error=on_error)\n"
    )
    (tmp_path / "producer.py").write_text(
        "def deliver(*, on_done, on_error):\n"
        "    on_done()\n"
        "    on_error()\n"
    )
    runtime = SimpleNamespace(owner=SimpleNamespace(root=tmp_path))
    remember(runtime, _receipt(tmp_path, **{
        "consumer.py": [(1, 10)], "producer.py": [(1, 3)],
    }))

    fact = _fact(tmp_path, runtime, start=1, end=3, line=3)
    assert fact is not None
    assert json.loads(fact.removeprefix(PREFIX)) == {
        "binding": "consumer.py:10", "callback": "deliver.on_error",
        "consumer": "consumer.py:7 audit.error(error)", "firing": "producer.py:3",
    }


def test_f046_exact_unverified_callback_anchor_does_not_recall_neighbor(tmp_path):
    from kernel_r200_f046_remember_product_consumer import remember

    (tmp_path / "consumer.py").write_text(
        "from producer import deliver\n"
        "\n"
        "def on_done(value):\n"
        "    audit.done(value)\n"
        "\n"
        "def on_error(error):\n"
        "    audit.error(error)\n"
        "\n"
        "def start():\n"
        "    deliver('payload', on_done=on_done, on_error=on_error)\n"
    )
    (tmp_path / "producer.py").write_text(
        "def deliver(*, on_done, on_error):\n"
        "    on_done()\n"
        "    on_error()\n"
    )
    runtime = SimpleNamespace(owner=SimpleNamespace(root=tmp_path))
    # Only the success firing was verified; the exact edit anchor is the error firing.
    remember(runtime, _receipt(tmp_path, **{
        "consumer.py": [(1, 10)], "producer.py": [(1, 2)],
    }))

    assert _fact(tmp_path, runtime, "producer.py", start=1, end=3, line=3) is None


@pytest.mark.parametrize("host_identity", ["off", "shadow", "on"])
def test_f046_consumer_selection_uses_host_frame(monkeypatch, host_identity):
    from kernel_frame_optional_allocation import _effect_consumer_fact
    from kernel_host_authorship import author_row, bind, open_step
    from kernel_stateframe_envelope import serialize_stateframe_payload

    monkeypatch.setenv(SWITCH, "on")
    monkeypatch.setenv("STATEFRAME_HOST_ROW_IDENTITY", host_identity)
    monkeypatch.setattr("kernel_r200_f046_remember_product_consumer.select",
                        lambda _, frame: frame["next_site"]["path"])

    def frame_row(path):
        frame = {"mode": "FRAME_ONLY", "next_site": {"path": path}}
        return {"role": "user", "content": serialize_stateframe_payload({"state": {"framefit5": frame}})}

    task = frame_row("task-text.py")
    frame = frame_row("producer.py")
    captured = frame_row("captured-text.py")
    tail = {"role": "user", "content": "Tail"}
    capture = SimpleNamespace(_framefit5_newest_user_suffix=[captured])
    runtime = SimpleNamespace(_active_capture=capture)
    ledger = bind(capture, open_step(runtime, 1))
    for row, kind in ((task, "task"), (frame, "state"), (captured, "captured_user")):
        author_row(ledger, row["role"], row["content"], kind=kind, site="test")

    assert _effect_consumer_fact(runtime, capture, {"messages": [task, frame, captured, tail]}) == "producer.py"
    assert _effect_consumer_fact(runtime, capture, {"messages": [task, captured, tail]}) is None


def test_f046_allocation_prices_exact_tail_without_displacing_source():
    from dataclasses import replace

    from kernel_frame_optional_allocation import _with_allocated_consumer
    from kernel_stateframe_framefit5 import VOLATILE_TAIL_MARKER
    from kernel_working_set_prepare import Measurement, prepare_candidate
    from test_working_set_render import snapshot, source

    fact = PREFIX + (
        '{"binding":"consumer.py:7","callback":"deliver.on_done",'
        '"consumer":"consumer.py:4 audit.store(payload)"}'
    )
    body = {"messages": [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Task: inspect"},
        {"role": "user", "content": VOLATILE_TAIL_MARKER + "\n\nRequired evidence"},
    ], "tools": []}
    snap = snapshot(source(), body=body)
    prepared = prepare_candidate(snap, measure=lambda _: Measurement(500, 500, "tokenize"))
    assert _with_allocated_consumer(prepared, fact, lambda _: Measurement(
        597, 500, "tokenize")) is prepared
    tight = replace(snap, byte_cap=len(prepared.wire))
    tight_fit = prepare_candidate(tight, measure=lambda _: Measurement(500, 500, "tokenize"))
    assert tight_fit.wire == prepared.wire
    assert _with_allocated_consumer(tight_fit, fact, lambda _: pytest.fail("oversize was metered")) is tight_fit
    fitted = _with_allocated_consumer(prepared, fact, lambda _: Measurement(596, 500, "tokenize"))
    assert fitted.wire.replace(json.dumps("\n\n" + fact).encode()[1:-1], b"") == prepared.wire
    assert fitted.rendered.certificate.assignment == prepared.rendered.certificate.assignment


@pytest.mark.parametrize(("allocator", "resume"), [
    ("on", None), ("off", None), ("on", "replay"), ("on", "checkpoint"),
])
def test_f046_enabled_provider_replays_same_consumer(tmp_path, allocator, resume):
    requalify = allocator == "on" and resume is None
    provider = _scenario(ROOT, tmp_path / "enabled", switch="on", allocator=allocator,
                         resume=resume, requalify=requalify)
    assert all(PREFIX.encode() not in wire for wire in provider[:2])
    assert provider[2].count(PREFIX.encode()) == 1
    assert all(wire.count(PREFIX.encode()) <= 1 for wire in provider[3:])
    tail = json.loads(provider[2])["messages"][-1]["content"]
    assert "consumer.py:4 audit.store" in tail
    assert '"firing":"producer.py:2"' in tail
