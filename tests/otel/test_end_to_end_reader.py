"""Historical exports of real scripted CLI workloads."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import lzma
import os
from pathlib import Path
import shutil
import signal
import socket
import ssl
import subprocess
import sys
import time
from threading import Event
from types import SimpleNamespace

import pytest

from fixtures.process_signals import signal_run_owned
from _journal import journal_bytes, measurement_journal_paths, write_journal_bytes
from stateframe.scripted_endpoint import Script, serve
from stateframe.otel.types import Coverage, Cursor, Family, Namespace, SourceOccurrence
from tests.otlp_receiver import OTLPReceiver, assert_capture_clean, assert_no_duplicate_spans, assert_valid_traces


ROOT = Path(__file__).resolve().parents[2]
SOURCE_MARKER = "SYNTHETIC_SOURCE_MARKER_42"


class _WorkloadScript(Script):
    """Script parent calls normally and finish each admitted worker independently."""

    def __init__(self, document: dict[str, object]):
        self.worker_calls = 0
        self.worker_mode = document.get("worker_mode") is True
        super().__init__(document)

    def reset(self, document: dict[str, object]) -> None:
        self.worker_calls = 0
        self.worker_mode = document.get("worker_mode") is True
        Script.__init__(self, document)

    def next_call(self, offered: list[str]) -> dict:
        if self.worker_mode and "parallel" not in offered and "done" in offered:
            self.worker_calls += 1
            return {"tool": "done", "arguments": {"summary": "Unit complete."}, "delay_s": 0.05}
        return super().next_call(offered)


class _InterruptedScript(_WorkloadScript):
    def __init__(self, document):
        self.awaiting_reply = Event()
        self.release_reply = Event()
        super().__init__(document)

    def next_call(self, offered):
        self.awaiting_reply.set()
        if not self.release_reply.is_set():
            self.release_reply.wait()
            return {"tool": "done", "arguments": {"summary": "Interrupted."}}
        return super().next_call(offered)


def _isolated_python(root: Path) -> Path:
    # Model a private host path in this child only; the required shared test
    venv = root / "engine-env"
    subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", "--without-pip", str(venv)],
                   cwd=root, check=True, capture_output=True)
    python = venv / "bin" / "python"
    site_packages = next((venv / "lib").glob("python*/site-packages"))
    private_home = root / "isolated-home"
    private_home.mkdir(mode=0o700)
    (site_packages / "_e2e_startup.py").write_text(
        "import pwd\n"
        "_getpwuid = pwd.getpwuid\n"
        "def _isolated_home(uid):\n"
        "    record = _getpwuid(uid)\n"
        f"    return pwd.struct_passwd((*record[:5], {str(private_home)!r}, record.pw_shell))\n"
        "pwd.getpwuid = _isolated_home\n", encoding="utf-8",
    )
    (site_packages / "_e2e_startup.pth").write_text("import _e2e_startup\n", encoding="ascii")
    return python


def _workspace(root: Path, *, passing_check: bool = True, parallel: bool = False) -> None:
    (root / "src").symlink_to(ROOT / "src", target_is_directory=True)
    (root / "repo/shop").mkdir(parents=True)
    (root / "repo/shop/__init__.py").write_text("", encoding="utf-8")
    (root / "repo/shop/pricing.py").write_text(
        "def apply_discount(price, percent):\n"
        f"    # {SOURCE_MARKER}\n"
        "    return round(price * percent / 100, 2)\n",
        encoding="utf-8",
    )
    (root / "repo/left.py").write_text("VALUE = 0\n", encoding="utf-8")
    (root / "repo/right.py").write_text("VALUE = 0\n", encoding="utf-8")
    tests = root / "repo/tests"
    tests.mkdir()
    if parallel:
        check = (
            "from left import VALUE as LEFT\n"
            "from right import VALUE as RIGHT\n\n"
            "def test_units():\n    assert LEFT == RIGHT == 0\n"
        )
    elif passing_check:
        check = ("from shop.pricing import apply_discount\n\n"
                 "def test_discount():\n    assert apply_discount(100, 20) == 80\n")
    else:
        check = "def test_expected_failure():\n    assert False\n"
    (tests / "test_pricing.py").write_text(check, encoding="utf-8")


def _environment(model_url: str, workspace: Path) -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("STATEFRAME_", "REACT_", "OTEL_"))
    }
    environment.update(PYTHONPATH=".:src", PYTHONHASHSEED="1", STATEFRAME_ENDPOINT=model_url,
                       STATEFRAME_MODEL="scripted", HOME=str(workspace / "isolated-home"), STATEFRAME_HOME="product-home", STATEFRAME_CHECKPOINT="off")
    return environment


def _task_tail(model_url: str) -> list[str]:
    return [
        "--repo", "repo", "--task", "Inspect the pricing module, make the requested fix, and verify it.",
        "--endpoint", model_url, "--model", "scripted",
        "--shell", "off", "--shell-network", "none",
    ]


def _scenario_document(name: str) -> dict[str, object]:
    change = {"tool": "str_replace", "arguments": {
        "path": "shop/pricing.py",
        "old_str": "    return round(price * percent / 100, 2)",
        "new_str": "    return round(price * (100 - percent) / 100, 2)",
    }}
    done = {"tool": "done", "arguments": {"summary": "The pricing change is complete."}}
    if name == "parallel_workers":
        contracts = [
            {"id": unit, "goal": f"Inspect {unit}.", "owns": [f"{unit}.py"]}
            for unit in ("left", "right")
        ]
        return {
            "worker_mode": True,
            "steps": [
                {"tool": "parallel", "arguments": {"subframes": contracts}},
                {"tool": "done", "arguments": {"summary": "Both units completed."}},
            ],
        }
    return {
        "steps": ([change, done] if name == "two_turn_chat" else [
            {"tool": "read", "arguments": {"path": "shop/pricing.py"}},
            change,
            {"tool": "run_tests", "arguments": {"target": "tests/test_pricing.py"}},
            done,
        ]),
    }



def _run_tree(root):
    entries = {}
    for path in root.rglob("*"):
        info = path.lstat()
        record = (info.st_mode, info.st_size, info.st_mtime_ns)
        if path.is_symlink():
            record += (os.readlink(path),)
        elif path.is_file():
            record += (hashlib.sha256(path.read_bytes()).digest(),)
        entries[path.relative_to(root)] = record
    return entries


def _native(python, workspace, environment, *args, stdin=None):
    return subprocess.run([str(python), "-m", "stateframe", *args], cwd=workspace, env=environment,
                          input=stdin, capture_output=True, text=True)


def _interrupt(python, workspace, environment, arguments, script):
    with subprocess.Popen([str(python), "-m", "stateframe", *arguments], cwd=workspace,
                          env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          stdin=subprocess.DEVNULL, start_new_session=True) as process:
        try:
            while not script.awaiting_reply.wait(0.05):
                assert process.poll() is None, "run exited before recording the interrupted request"
            signal_run_owned(
                workspace / "run", process, signal.SIGINT,
                deadline=time.monotonic() + 5,
                what="held provider request", require_checkpoint_saved=False,
            )
            process.communicate()
            assert process.returncode != 0
        finally:
            script.release_reply.set()


def _workload(name, workspace, python):
    document = _scenario_document(name)
    if name == "chat":
        document = {"steps": _scenario_document("two_turn_chat")["steps"] + [
            {"tool": "done", "arguments": {"summary": "Follow-up complete."}},
        ]}
    script = _InterruptedScript(document) if name == "resume_interrupt" else _WorkloadScript(document)
    model = serve(script)
    model_url = f"http://127.0.0.1:{model.server_port}/v1"
    if name == "resume_interrupt":
        model.handle_error = lambda *_args: None
    environment = _environment(model_url, workspace)
    archived_spans = []
    interrupted_calls = 0
    run = workspace / "run"
    arguments = ["run", "--run-dir", "run", *_task_tail(model_url)]
    try:
        if name == "parallel_workers":
            arguments.extend(("--parallel", "2"))
        if name == "resume_interrupt":
            _interrupt(python, workspace, environment, arguments, script)
            interrupted_calls = len(script.requests)
            with OTLPReceiver() as receiver:
                archived_spans = _export(python, workspace, run, environment, receiver)
            script.reset(document)
            result = _native(python, workspace, environment, "resume", "run")
        elif name == "chat":
            result = _native(python, workspace, environment, "chat", "--session-dir", "session",
                             *_task_tail(model_url)[0:2], "--shell", "off", "--shell-network", "none",
                             stdin="Fix the pricing code.\nConfirm the result.\n/quit\n")
        else:
            result = _native(python, workspace, environment, *arguments)
        assert result.returncode == (1 if name == "failing_check" else 0), (
            name, result.returncode,
            (result.stderr + result.stdout)[-3000:].replace(str(workspace), "<workspace>").replace(str(ROOT), "<clone>"),
        )
        script.assert_consumed()
        if name == "parallel_workers":
            assert script.worker_calls >= 2
    finally:
        model.shutdown()
        model.server_close()
    runs = sorted((workspace / "session").glob("task-*")) if name == "chat" else [run]
    return environment, runs, archived_spans, interrupted_calls + len(script.requests)


def _otel(python, workspace, environment, *args, endpoint=None):
    env = dict(environment)
    if endpoint is not None:
        env["OTEL_EXPORTER_OTLP_ENDPOINT"] = endpoint
    entry = "from stateframe.otel.cli import main; raise SystemExit(main())"
    return subprocess.run([str(python), "-c", entry, *args], cwd=workspace, env=env,
                          capture_output=True, text=True)


def _spans(receiver):
    assert_no_duplicate_spans(receiver.requests)
    spans = []
    for request in receiver.requests:
        assert request.path == "/v1/traces"
        assert_valid_traces(request.decoded_json)
        spans.extend(span for resource in request.decoded_json["resourceSpans"]
                     for scope in resource["scopeSpans"] for span in scope["spans"])
    return spans


def _export(python, workspace, run, environment, receiver):
    before = _run_tree(run)
    result = _otel(python, workspace, environment, "export", str(run.relative_to(workspace)),
                   "--trust-source", "--protocol", "http/json", endpoint=receiver.url)
    assert result.returncode == 0, result.stderr
    status = _otel(python, workspace, environment, "status", str(run.relative_to(workspace)), "--json")
    assert status.returncode == 0, status.stderr
    spans = _spans(receiver)
    assert spans
    assert {span["traceId"] for span in spans} == {json.loads(status.stdout)["trace_id"]}
    assert before == _run_tree(run)
    payload = b"".join(request.raw_body for request in receiver.requests)
    assert SOURCE_MARKER.encode() not in payload
    assert os.fsencode(workspace) not in payload
    assert os.fsencode(ROOT) not in payload
    return spans


def _attrs(span):
    return {attr["key"]: next(iter(attr["value"].values())) for attr in span["attributes"]}


def _ids(spans):
    return {(span["traceId"], span["spanId"]) for span in spans}


def _native_journals(run):
    attempts = [*sorted((run / "attempts").glob("[0-9]*")), run]
    for attempt in attempts:
        patterns = ("worker/measurement-runs/*/events.jsonl*", "worker/plain/events.jsonl",
                    "parallel/*/units/*/journal/events.jsonl*",
                    "parallel/*/units/*/worker/measurement-runs/*/events.jsonl*")
        for pattern in patterns:
            for path in attempt.glob(pattern):
                if path.suffix in {".xz", ".jsonl"}:
                    with (lzma.open(path, "rb") if path.suffix == ".xz" else path.open("rb")) as stream:
                        yield path.relative_to(run), [json.loads(line) for line in stream]


def _assert_recorded_spans(run, spans):
    by_id = {span["spanId"]: span for span in spans}
    by_name = Counter(span["name"] for span in spans)
    durations = []
    submitted = actions = worker_streams = 0
    for path, rows in _native_journals(run):
        calls = sum(row["kind"] == "request_submitted" and row.get("dispatch") is not False
                    and not row.get("replayed") for row in rows)
        submitted += calls
        if "units" in path.parts and calls:
            worker_streams += 1
        for row in rows:
            if row["kind"] in {"response_received", "transport_error", "response"} and not row.get("replayed"):
                if "elapsed_ns" in row:
                    durations.append(row["elapsed_ns"])
                elif "seconds" in row:
                    durations.append(round(row["seconds"] * 1_000_000_000))
            if row["kind"] == "action_receipt" and "result_class" in row.get("receipt", {}):
                actions += 1
    assert by_name["chat"] == submitted
    assert by_name["stateframe.action"] >= actions > 0
    assert by_name["stateframe.check"] > 0
    calls = [span for span in spans if span["name"] == "chat"]
    assert sorted(int(_attrs(span)["stateframe.duration_ns"]) for span in calls
                  if "stateframe.duration_ns" in _attrs(span)) == sorted(durations)
    workers = set()
    for call in calls:
        parent = by_id[call["parentSpanId"]]
        if parent["name"] == "stateframe.subframe.context":
            workers.add(parent["spanId"])
            parent = by_id[by_id[parent["parentSpanId"]]["parentSpanId"]]
        assert parent["name"] == "stateframe.attempt.context"
        assert _attrs(call)["stateframe.attempt"] == _attrs(parent)["stateframe.attempt"]
    assert len(workers) == worker_streams
    return submitted, worker_streams


@pytest.mark.parametrize("name", ["run", "chat", "resume_interrupt", "parallel_workers",
                                  "failing_check", "collector_down"])
def test_end_to_end_reader(tmp_path, name):
    python = _isolated_python(tmp_path)
    _workspace(tmp_path, passing_check=name != "failing_check", parallel=name == "parallel_workers")
    environment, runs, archived_spans, expected_calls = _workload(name, tmp_path, python)
    assert len(runs) == (2 if name == "chat" else 1)
    model_calls = 0
    for run in runs:
        receiver_options = {}
        if name == "collector_down":
            # Bound without listening so the address cannot be claimed between the two calls.
            with socket.socket() as unavailable:
                unavailable.bind(("127.0.0.1", 0))
                port = unavailable.getsockname()[1]
                before = _run_tree(run)
                result = _otel(python, tmp_path, environment, "export", "run", "--trust-source",
                               "--protocol", "http/json", endpoint=f"http://127.0.0.1:{port}")
                assert result.returncode == 1
                assert result.stderr == "stateframe-otel: export not delivered (receiver failure: connection_failed)\n"
                status = _otel(python, tmp_path, environment, "status", "run", "--json")
                assert json.loads(status.stdout)["signals"]["traces"]["queue_items"] > 0
                assert before == _run_tree(run)
            receiver_options["port"] = port
        with OTLPReceiver(**receiver_options) as receiver:
            spans = _export(python, tmp_path, run, environment, receiver)
            call_count, worker_count = _assert_recorded_spans(run, spans)
            model_calls += call_count
            outcomes = {int(_attrs(span)["stateframe.attempt"]): _attrs(span) for span in spans
                        if span["name"] == "stateframe.attempt.outcome"}
            assert sum(span["name"] == "stateframe.attempt.outcome" for span in spans) == len(outcomes)
            if name == "parallel_workers":
                assert worker_count == 2
                parent_result = json.loads(next(run.glob("parallel/*/result.json")).read_text())
                assert [item["status"] for item in parent_result["outcomes"]] == ["REFUSED", "REFUSED"]
                workers = [span for span in spans if span["name"] == "stateframe.subframe"]
                assert len(workers) == 2
                assert {_attrs(span)["stateframe.outcome"] for span in workers} == {"ok"}
                assert {_attrs(span)["stateframe.parallel.disposition"] for span in workers} == {"refused"}
            if name == "failing_check":
                assert outcomes[1]["stateframe.outcome"] == "error"
                assert any(_attrs(span)["stateframe.outcome"] == "fail" and span["status"]["code"] == 2
                           for span in spans if span["name"] == "stateframe.check")
            if name == "resume_interrupt":
                assert outcomes[2]["stateframe.resume"] == "resumed"
                assert [int(_attrs(span)["stateframe.attempt"]) for span in spans
                        if span["name"] == "stateframe.attempt.context"] == [1, 2]
                assert _ids(archived_spans) <= _ids(spans)

            request_count = len(receiver.requests)
            before = _run_tree(run)
            result = _otel(python, tmp_path, environment, "export", str(run.relative_to(tmp_path)),
                           "--protocol", "http/json", endpoint=receiver.url)
            assert result.returncode == 0, result.stderr
            assert len(receiver.requests) == request_count
            assert _run_tree(run) == before

        if name == "run":
            prefix = tmp_path / "prefix"
            shutil.copytree(run, prefix, symlinks=True)
            journal = next(iter(measurement_journal_paths(prefix)))
            compressed = journal.suffix == ".xz"
            data = journal_bytes(journal, decompress=True)
            lines = data.splitlines(keepends=True)
            count = next(index for index, line in enumerate(lines, 1)
                         if json.loads(line)["kind"] == "response_received")
            retained = b"".join(lines[:count])
            write_journal_bytes(journal, lzma.compress(retained) if compressed else retained)
            with OTLPReceiver() as receiver:
                result = _otel(python, tmp_path, environment, "export", "prefix", "--trust-source",
                               "--protocol", "http/json", endpoint=receiver.url)
                assert result.returncode == 1
                assert "incomplete coverage" in result.stderr
                chats = [span for span in _spans(receiver) if span["name"] == "chat"]
                assert len(chats) == 1
                assert chats[0]["spanId"] in {span["spanId"] for span in spans if span["name"] == "chat"}
            status = _otel(python, tmp_path, environment, "status", "prefix", "--json")
            assert json.loads(status.stdout)["complete"] is False
            damaged = tmp_path / "malformed"
            shutil.copytree(run, damaged, symlinks=True)
            journal = next(iter(measurement_journal_paths(damaged)))
            compressed = journal.suffix == ".xz"
            lines = journal_bytes(journal, decompress=True).splitlines(keepends=True)
            receipt_index = next(index for index, line in enumerate(lines)
                                 if json.loads(line).get("kind") == "action_receipt")
            receipt = json.loads(lines[receipt_index])
            receipt["receipt"]["kind"] = []
            lines[receipt_index] = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            write_journal_bytes(journal, lzma.compress(b"".join(lines)) if compressed else b"".join(lines))
            with OTLPReceiver() as receiver:
                result = _otel(python, tmp_path, environment, "export", "malformed", "--trust-source",
                               "--protocol", "http/json", endpoint=receiver.url)
                assert result.returncode == 1
                assert any(span["name"] == "chat" for span in _spans(receiver))
            status = _otel(python, tmp_path, environment, "status", "malformed", "--json")
            assert json.loads(status.stdout)["complete"] is False
            moved = run.rename(tmp_path / "moved")
            copied = tmp_path / "copied"
            shutil.copytree(moved, copied, symlinks=True)
            for relocated in (moved, copied):
                with OTLPReceiver() as receiver:
                    assert _ids(_export(python, tmp_path, relocated, environment, receiver)) == _ids(spans)
    assert model_calls == expected_calls


def test_export_canaries_do_not_reach_wire_store_or_output(tmp_path, monkeypatch, capsys):
    from stateframe.otel import authority, companion
    from stateframe.otel.identity import from_seed

    openssl = shutil.which("openssl")
    if openssl is None:
        pytest.skip("OpenSSL is required to prepare the local TLS receiver")
    certificate = tmp_path / "receiver.pem"
    private_key = tmp_path / "receiver.key"
    subprocess.run([
        openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
        "-subj", "/CN=127.0.0.1", "-addext", "subjectAltName=IP:127.0.0.1",
        "-keyout", str(private_key), "-out", str(certificate),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    certificate.chmod(0o600)
    private_key.chmod(0o600)
    server_tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_tls.load_cert_chain(certificate, private_key)

    canaries = (
        "model-canary-5e41",
        "endpoint-canary-8bd2",
        "path-canary-312a",
        "tool-canary-6bc4",
        "check-command-canary-046f",
        "task-canary-3271",
        "file-content-canary-e014",
        "header-canary-4cd9",
        "run-directory-canary-d90a",
        "environment-canary-9a33",
    )
    (model, endpoint, path, tool, check, task, content, header, run_name, environment) = canaries
    key = b"k" * 32
    seed = b"s" * 16
    run_ref, _trace = from_seed(key, seed)
    namespace = Namespace(run_ref, 1, None, None)
    root = tmp_path / f"run-{run_name}"
    root.mkdir(mode=0o700)
    repository_path = tmp_path / f"repo-{path}"
    spec = {
        "schema": "stateframe-run/1",
        "run_identity_seed": seed.hex(),
        "attempt": 1,
        "model": model,
        "provider": "local",
        "chat_url": f"https://{endpoint}.invalid/v1",
        "repo": str(repository_path),
        "task": task,
        "api_key_env": "CANARY_ENV",
        "environment": {"CANARY_ENV": environment},
        "headers": {"x-source-header": header},
    }
    summary = {
        "runtime_outcome": "completed",
        "tests_before": {"exit_code": 0, "command": check, "selector": check},
        "tests_after": {"exit_code": 0, "command": check, "selector": check},
        "files_changed": [str(repository_path / f"{path}.py")],
    }

    def occurrence(index, family, row, stream):
        return SourceOccurrence(
            namespace, family, index, 1_730_000_000_000_000_000 + index, row,
            Cursor(stream, b"g" * 32, index, index, b"h" * 32),
        )

    occurrences = (
        occurrence(1, Family.SPEC, spec, b"s" * 16),
        occurrence(2, Family.JOURNAL, {
            "kind": "request_submitted", "request_id": "request", "step": 1,
            "event_id": "event", "task": task, "request_body": content,
        }, b"m" * 16),
        occurrence(3, Family.JOURNAL, {
            "kind": "response_received", "request_id": "request", "step": 1,
            "transport_attempt_id": "event", "elapsed_ns": 25_000_000,
            "response": {"model": model, "content": content},
        }, b"m" * 16),
        occurrence(4, Family.JOURNAL, {
            "kind": "check_receipt", "status": "pass", "execution_id": "check",
            "check_kind": "custom", "command": check, "selector": check, "output": content,
        }, b"c" * 16),
        occurrence(5, Family.TOOL, {
            "kind": "intent", "call_id": "tool-call", "tool": tool, "arguments": content,
        }, b"t" * 16),
        occurrence(6, Family.TOOL, {
            "kind": "complete", "call_id": "tool-call", "outcome": "ok", "result": content,
        }, b"t" * 16),
        occurrence(7, Family.SUMMARY, summary, b"u" * 16),
    )

    state_root = tmp_path / "export-state"
    state_root.mkdir(mode=0o700)
    view = SimpleNamespace(
        seed=seed, attempt=1, profile="restricted", environment={}, grants=(),
        authority_records=(), source_generation=None,
    )

    class Reader:
        def __init__(self, *_args):
            pass

        def poll(self):
            return occurrences, Coverage(True, ())

        def close(self):
            pass

    monkeypatch.setattr(companion, "protect_process", lambda **_kwargs: None)
    monkeypatch.setattr(companion, "policy_view", lambda _root: view)
    monkeypatch.setattr(companion, "_run_lock_held", lambda *_args: False)
    monkeypatch.setattr(companion, "admit", lambda *_args: Coverage(True, ()))
    monkeypatch.setattr(companion, "config_path_allowed", lambda *_args: True)
    monkeypatch.setattr(companion, "transport_paths_allowed", lambda *_args: True)
    monkeypatch.setattr(companion, "safe_home", lambda: tmp_path)
    monkeypatch.setattr(companion, "_state_root", lambda **_kwargs: state_root)
    monkeypatch.setattr(companion, "_identity_key", lambda *_args, **_kwargs: key)
    monkeypatch.setattr(companion, "Reader", Reader)
    monkeypatch.setattr(authority, "source_mode_problem", lambda _root: None)
    monkeypatch.setattr(authority, "_read_record", lambda record: spec if record.name == "spec.json" else summary)
    for name in tuple(os.environ):
        if name.startswith("OTEL_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CANARY_ENV", environment)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "http/json")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_HEADERS", f"x-export-canary={header}")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE", str(certificate))

    with OTLPReceiver(ssl_context=server_tls) as receiver:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", receiver.url + "/v1/traces")
        assert companion.export(root, trust_source=True) == 0
        assert len(receiver.requests) == 1
        assert receiver.requests[0].headers["x-export-canary"] == header
        assert_valid_traces(receiver.requests[0].decoded_json)
        assert_capture_clean(receiver.requests, canaries, allowed_headers={"x-export-canary": header})

    state_bytes = b"".join(path.read_bytes() for path in state_root.iterdir() if path.is_file())
    assert_capture_clean(state_bytes, canaries)
    captured = capsys.readouterr()
    assert_capture_clean(captured.out + captured.err, canaries)
    assert str(tmp_path) not in captured.out + captured.err
