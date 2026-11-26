"""Provider-boundary parity and source-bound API comparisons for R015."""
from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from _timeouts import scaled
from _stateframe_home import isolated_stateframe_home
from fixtures.process_signals import record_identity, signal_run_owned, signal_run_verified
from stateframe.scripted_endpoint import Script, make_handler
from tools.pinned_tree import pinned_tree

ROOT = Path(__file__).resolve().parents[1]
SWITCH = "STATEFRAME_API_ATTRIBUTE_ROWS"
HEADER = "[Repository API attribute rows]"
SOURCE = '''"""Choose between pricing.standard and pricing.discounted."""
def standard(price: int, /):
    """Return the price in cents."""
    return price


def discounted(
    price: int, *, percent: int = 10,
):
    """Apply percent before converting to dollars."""
    return price * (100 - percent) / 10_000


def quote(price):
    return standard(price)
'''
TASK = "Change quote in pricing.py to use the API that applies a percentage discount."


def _provider_run(tmp_path, checkout, setting, *, allocator="on", resume=None,
                  repo_sources=None, earlier_steps=()):
    @contextmanager
    def endpoint(script):
        class RawHandler(make_handler(script)):
            def do_POST(self):  # noqa: N802
                if self.path == "/v1/chat/completions":
                    raw = self.rfile.read(int(self.headers["Content-Length"]))
                    chat.append(gzip.decompress(raw) if self.headers.get("Content-Encoding") == "gzip"
                                else raw)
                    if resume and script.position == 1 and not release.is_set():
                        waiting.set()
                        if not release.wait(scaled(120)):
                            raise TimeoutError("test did not release the interrupted request")
                        return
                    self.rfile = io.BytesIO(raw)
                super().do_POST()

        server = ThreadingHTTPServer(("127.0.0.1", 0), RawHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/v1"
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    repo = tmp_path / "repo"
    run = tmp_path / "run"
    repo.mkdir(exist_ok=True)
    for path, source in (repo_sources or {"pricing.py": SOURCE}).items():
        (repo / path).write_text(source)
    chat = []
    waiting = threading.Event()
    release = threading.Event()

    script = Script({"tokenize": {"mode": "ok"}, "steps": [
        *earlier_steps,
        {"tool": "read", "arguments": {"path": "pricing.py"}},
        {"tool": "str_replace", "arguments": {
            "path": "pricing.py", "old_str": "    return standard(price)",
            "new_str": "    return discounted(price)",
        }},
        {"tool": "done", "arguments": {"summary": "Updated quote."}},
    ]})
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("STATEFRAME_", "REACT_"))}
    env.update(HOME=str(tmp_path / "home"), STATEFRAME_HOME=isolated_stateframe_home(),
               PYTHONPATH=str(checkout / "src"), PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0",
               STATEFRAME_WORKING_SET_ALLOCATOR=allocator, STATEFRAME_STABLE_TOOL_HEAD="off")
    if resume == "replay":
        env["STATEFRAME_CHECKPOINT"] = "off"  # rebuild confirmed sends from the tool tape
    if setting is not None:
        env[SWITCH] = setting
    with endpoint(script) as url:
        base_command = [sys.executable, "-m", "stateframe"]
        options = {"cwd": tmp_path, "env": env, "stdout": subprocess.PIPE,
                   "stderr": subprocess.PIPE, "text": True}
        command = [*base_command, "run", "--repo", str(repo), "--task", TASK,
                   "--target", "pricing.py", "--endpoint", url, "--model", script.model,
                   "--run-dir", str(run), "--test-cmd", "true", "--runner-kind", "generic",
                   "--test-python", sys.executable, "--no-run-env", "--quiet", "--json",
                   "--max-turns", str(len(script.steps) + 1)]
        if resume:
            process = subprocess.Popen(command, **options)
            try:
                deadline = time.monotonic() + scaled(300)
                while not waiting.wait(0.2):
                    if process.poll() is not None:
                        stdout, stderr = process.communicate()
                        pytest.fail(f"Run stopped before its in-flight request\n{stdout}{stderr}")
                    assert time.monotonic() < deadline, "Run did not reach its in-flight request"
                recorded = tuple(chat)
                children = subprocess.check_output(["pgrep", "-P", str(process.pid)], text=True).split()
                child, = map(int, children)
                identity = record_identity(child, launched_after=0, command_fragment="_child.py")
                signal_run_verified(
                    run, child, identity, signal.SIGKILL, deadline=time.monotonic() + scaled(90),
                    what="R015 raw provider-byte parity", require_checkpoint_saved=resume == "checkpoint",
                )
                process.communicate(timeout=scaled(90))
                assert (run / "wire" / "0002.request.json").read_bytes() == recorded[1]
                assert not (run / "wire" / "0002.response.json").exists()
                assert not (run / "wire" / "0002.error.txt").exists()
            finally:
                release.set()
                if process.poll() is None:
                    signal_run_owned(
                        run, process, signal.SIGKILL,
                        deadline=time.monotonic() + scaled(30),
                        what="R015 process cleanup",
                        require_checkpoint_saved=resume == "checkpoint",
                    )
                    process.communicate(timeout=scaled(20))
            command = [*base_command, "resume", str(run), "--endpoint", url, "--json"]
        completed = subprocess.run(command, **options, timeout=scaled(300))
        assert completed.returncode == 0, completed.stdout + completed.stderr
        script.assert_consumed()
    if resume:
        assert len(recorded) == 2 and len(chat) == 4
        assert chat[2] == recorded[1], "requalified in-flight send changed raw provider bytes"
        log = (run / "engine.log").read_text()
        resume_event = "checkpoint_verified" if resume == "checkpoint" else "replay_done"
        assert f'"kind":"{resume_event}"' in log, log
    else:
        assert len(chat) == len(script.steps)
    assert json.loads((run / "summary.json").read_text())["engine_result"] != "ENGINE_ERROR"
    return tuple(chat), run


@pytest.mark.parametrize(("allocator", "resume"), [("off", "replay"), ("on", "checkpoint")])
def test_r015_edit_depends_choice_off_request_bytes_match_pinned_099(
        tmp_path, tmp_path_factory, allocator, resume):
    baseline = pinned_tree("29765b39820fd39800fd7292f52fb5bcb81e6069", tmp_path_factory)
    expected, run = _provider_run(tmp_path, baseline, None, allocator=allocator, resume=resume)
    run.rename(tmp_path / "pinned-run")
    for setting in (None, "off"):
        actual, run = _provider_run(tmp_path, ROOT, setting, allocator=allocator, resume=resume)
        assert actual == expected
        run.rename(tmp_path / f"candidate-{setting}")


def test_r015_edit_depends_choice_switch_defaults_off(monkeypatch):
    from kernel_stateframe_framefit5 import _api_attribute_rows_enabled
    from stateframe.runner import OPT_IN_SWITCHES, engine_switches, resolve_opt_in_switches

    monkeypatch.delenv(SWITCH, raising=False)
    assert not _api_attribute_rows_enabled()
    assert SWITCH in OPT_IN_SWITCHES and SWITCH in engine_switches()
    assert SWITCH not in resolve_opt_in_switches()
    for setting in ("off", "false"):
        monkeypatch.setenv(SWITCH, setting)
        assert not _api_attribute_rows_enabled()
        assert resolve_opt_in_switches()[SWITCH] == "off"
    monkeypatch.setenv(SWITCH, "on")
    assert _api_attribute_rows_enabled()
    assert resolve_opt_in_switches()[SWITCH] == "on"


def test_r015_edit_depends_choice_on_path_is_bounded_and_idempotent(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import kernel_stateframe_framefit5 as framefit
    from kernel_cache_layout_view import semantic_messages
    from kernel_context import token_estimate
    from kernel_r200_r015_edit_depends_choice import MAX_BLOCK_BYTES, blocks, strip

    wire, _run = _provider_run(tmp_path, ROOT, "on")
    assert [body.count(HEADER.encode()) for body in wire] == [0, 1, 0]
    body = json.loads(wire[1])
    wire_messages = semantic_messages(body["messages"])
    block, = blocks(wire_messages[-1]["content"])
    assert len(block.encode()) <= MAX_BLOCK_BYTES
    columns_and_rows = json.loads(block.split("\n", 1)[1])
    assert columns_and_rows["columns"] == ["API", "Signature inputs", "Quoted contract", "Declaration"]
    revision = "sha256:" + hashlib.sha256(SOURCE.encode()).hexdigest()
    declarations = [
        {"path": "pricing.py", "revision": revision, "start_line": 7, "end_line": 10},
        {"path": "pricing.py", "revision": revision, "start_line": 2, "end_line": 3},
    ]
    assert columns_and_rows["rows"] == [
        ["pricing.discounted", "\n    price: int, *, percent: int = 10,\n",
         '"""Apply percent before converting to dollars."""', declarations[0]],
        ["pricing.standard", "price: int, /",
         '"""Return the price in cents."""', declarations[1]],
    ]

    root = tmp_path / "repo"
    runtime = SimpleNamespace(owner=SimpleNamespace(root=root), frame_ceiling=40000)
    contract = {"admitted_tool_schemas": body["tools"]}
    monkeypatch.setenv(SWITCH, "on")
    reassembled = framefit.assemble_framefit5_request(
        wire_messages, wire_messages, owner_contract=contract, runtime=runtime)
    assert blocks(reassembled[-1]["content"]) == [block]
    receipt = {
        "layout": "api_attribute_rows",
        "declarations": [{"api": row[0], **row[3]} for row in columns_and_rows["rows"]],
        "block_sha256": "sha256:" + hashlib.sha256(block.encode()).hexdigest(),
    }
    assert framefit._final_api_attribute_rows_receipt(body["messages"], runtime) == receipt
    assert framefit._final_api_attribute_rows_receipt(reassembled, runtime) == receipt
    corrupted = copy.deepcopy(reassembled)
    corrupted[-1]["content"] = corrupted[-1]["content"].replace(
        "Apply percent before converting to dollars.", "Return the price in cents.")
    with pytest.raises(framefit.FrameFit5Error, match="current declaration evidence"):
        framefit._final_api_attribute_rows_receipt(corrupted, runtime)

    under_pressure = copy.deepcopy(reassembled)
    tail = under_pressure[-1]["content"]
    baseline_tail = strip(tail)
    baseline = [*under_pressure[:-1], {**under_pressure[-1], "content": baseline_tail}]
    runtime.frame_ceiling = token_estimate({"messages": baseline, "tools": body["tools"]})
    framefit._render_api_attribute_rows(under_pressure, runtime, contract, None)
    assert under_pressure == baseline
    runtime.frame_ceiling = 40000

    stale = copy.deepcopy(reassembled)
    (root / "pricing.py").write_text(SOURCE.replace("Return the price in cents.", "Return the price."))
    framefit._render_api_attribute_rows(stale, runtime, contract, None)
    assert not blocks(stale[-1]["content"])
    assert stale[:-1] == reassembled[:-1]

    _check_cross_file_discovery(root, framefit._framefit5_delivered_source_ranges)


def _check_cross_file_discovery(root, delivered):
    from kernel_r200_r015_edit_depends_choice import MAX_BLOCK_BYTES, comparison
    from kernel_source_delivery import deliver_source

    frame = {"action_ledger": {"edits": []}}
    assistant = {"role": "assistant", "content": "Compare local.open_stream vs remote.open_stream.",
                 "tool_calls": []}
    messages = [assistant]
    for path, signature, clause in (
            ("local.py", "file_name, *, encoding='utf8'", "Open a local file."),
            ("remote.py", "url, *, timeout=5", "Open a remote URL.")):
        source = f'def open_stream({signature}):\n    """{clause}"""\n    pass\n'.encode()
        (root / path).write_bytes(source)
        args = {"path": path}
        assistant["tool_calls"].append(
            {"id": path, "function": {"name": "read", "arguments": json.dumps(args)}})
        messages.append({"role": "tool", "tool_call_id": path,
                         "content": deliver_source(source, path, 0, args=args).text})
    block, proof = comparison(messages, messages, frame, root, delivered)
    rows = json.loads(block.split("\n", 1)[1])["rows"]
    assert [row[:3] for row in rows] == [
        ["local.open_stream", "file_name, *, encoding='utf8'", '"""Open a local file."""'],
        ["remote.open_stream", "url, *, timeout=5", '"""Open a remote URL."""'],
    ]
    assert [row["path"] for row in proof["declarations"]] == ["local.py", "remote.py"]

    bound = copy.deepcopy(messages)
    remote = (root / "remote.py").read_bytes()
    bound[-1]["content"] = deliver_source(
        remote, "remote.py", 0, args={"path": "remote.py", "max_lines": 1}).text
    assert comparison(bound, bound, frame, root, delivered) is None
    frame["action_ledger"]["edits"] = [{"path": "caller.py"}]
    assert comparison(messages, messages, frame, root, delivered) is None
    frame["action_ledger"]["edits"] = []
    large = remote.replace(b"Open a remote URL.", b"x" * MAX_BLOCK_BYTES)
    (root / "remote.py").write_bytes(large)
    bound[-1]["content"] = deliver_source(large, "remote.py", 0, args={"path": "remote.py"}).text
    assert comparison(bound, bound, frame, root, delivered) is None


def _read_exchange(path, source):
    from kernel_source_delivery import deliver_source

    args = {"path": path}
    return [
        {"role": "assistant", "content": "Compare pricing.standard and pricing.discounted.",
         "tool_calls": [{"id": path, "function": {"name": "read", "arguments": json.dumps(args)}}]},
        {"role": "tool", "tool_call_id": path,
         "content": deliver_source(source.encode(), path, 0, args=args).text},
    ]


def test_r015_uses_a_prior_read_projected_as_current_source(tmp_path):
    from kernel_r200_r015_edit_depends_choice import comparison
    from kernel_request_source_window import bound_source_message
    from kernel_stateframe_framefit5 import _framefit5_delivered_source_ranges as delivered

    standard = 'def standard(price, /):\n    """Return the price in cents."""\n    return price\n'
    discounted = ('def discounted(price, *, percent):\n'
                  '    """Apply percent before converting to dollars."""\n'
                  '    return price * (100 - percent) / 10_000\n')
    (tmp_path / "standard.py").write_text(standard)
    (tmp_path / "pricing.py").write_text(discounted)
    prior = bound_source_message(
        "[Current source: standard.py]\n" + standard, path="standard.py", source=standard,
        runtime=None, max_chars=4096,
    )
    exchange = _read_exchange("pricing.py", discounted)
    exchange[0]["content"] = "Compare standard.standard and pricing.discounted."
    revision = "sha256:" + hashlib.sha256(standard.encode()).hexdigest()
    assert (1, 3) in delivered([{"role": "user", "content": prior}])[("standard.py", revision)]
    frame = {"action_ledger": {"edits": []}}
    messages = [{"role": "user", "content": prior}, *exchange]
    pair = comparison(messages, exchange, frame, tmp_path, delivered)
    assert pair is not None
    assert [row["api"] for row in pair[1]["declarations"]] == [
        "pricing.discounted", "standard.standard",
    ]
    assert comparison(exchange, exchange, frame, tmp_path, delivered) is None
    messages[0]["content"] = prior.replace("Return the price in cents.", "Return cents directly.")
    assert comparison(messages, exchange, frame, tmp_path, delivered) is None


def test_r015_unrelated_edit_leaves_the_api_choice_open(tmp_path):
    from kernel_r200_r015_edit_depends_choice import comparison
    from kernel_stateframe_framefit5 import _framefit5_delivered_source_ranges as delivered

    (tmp_path / "pricing.py").write_text(SOURCE)
    exchange = _read_exchange("pricing.py", SOURCE)
    edit = {"path": "caller.py", "operation": "str_replace",
            "old_text": 'label = "Price"', "new_text": 'label = "Quote"'}
    frame = {"action_ledger": {"edits": [edit]}, "applied_edits": [{"file": "caller.py"}]}
    assert comparison(exchange, exchange, frame, tmp_path, delivered) is not None
    edit.update(old_text="return standard(price)", new_text="return discounted(price)")
    assert comparison(exchange, exchange, frame, tmp_path, delivered) is None


def test_r015_omits_a_rebound_export(tmp_path):
    from kernel_r200_r015_edit_depends_choice import comparison
    from kernel_stateframe_framefit5 import _framefit5_delivered_source_ranges as delivered

    source = "import functools\n" + SOURCE.replace("percent: int = 10", "percent: int")
    rebound = source + "\ndiscounted = functools.partial(discounted, percent=10)\n"
    frame = {"action_ledger": {"edits": []}}
    (tmp_path / "pricing.py").write_text(source)
    exchange = _read_exchange("pricing.py", source)
    assert comparison(exchange, exchange, frame, tmp_path, delivered) is not None
    (tmp_path / "pricing.py").write_text(rebound)
    exchange = _read_exchange("pricing.py", rebound)
    assert comparison(exchange, exchange, frame, tmp_path, delivered) is None


def test_r015_provider_sees_separate_reads_after_an_unrelated_edit(tmp_path):
    from kernel_cache_layout_view import semantic_messages
    from kernel_r200_r015_edit_depends_choice import blocks

    standard = 'def standard(price: int, /):\n    """Return the price in cents."""\n    return price\n'
    pricing = SOURCE.replace(standard, "from standard import standard\n")
    pricing = pricing.replace("pricing.standard", "standard.standard")
    wire, run = _provider_run(
        tmp_path, ROOT, "on", repo_sources={"pricing.py": pricing, "standard.py": standard},
        earlier_steps=[
            {"tool": "file_create", "arguments": {"path": "note.txt", "content": "Quote labels\n"}},
            {"tool": "read", "arguments": {"path": "standard.py"}},
        ],
    )
    assert [body.count(HEADER.encode()) for body in wire] == [0, 0, 0, 1, 0]
    assert "+++ b/note.txt" in (run / "final.diff").read_text()
    messages = semantic_messages(json.loads(wire[3])["messages"])
    block, = blocks(messages[-1]["content"])
    assert [row[0] for row in json.loads(block.split("\n", 1)[1])["rows"]] == [
        "pricing.discounted", "standard.standard",
    ]


def test_r015_open_call_replacement_does_not_suppress_rows(tmp_path):
    from kernel_r200_r015_edit_depends_choice import comparison
    from kernel_stateframe_framefit5 import _framefit5_delivered_source_ranges as delivered

    (tmp_path / "pricing.py").write_text(SOURCE)
    exchange = _read_exchange("pricing.py", SOURCE)
    frame = {"action_ledger": {"edits": [{
        "path": "retry.py",
        "old_text": "retry_policy = RetryPolicy(",
        "new_text": "retry_policy = BackoffPolicy(",
    }]}}
    result = comparison(exchange, exchange, frame, tmp_path, delivered)
    assert result is not None
    block, receipt = result
    rows = json.loads(block.split("\n", 1)[1])["rows"]
    assert [row[0] for row in rows] == ["pricing.discounted", "pricing.standard"]
    assert [row["api"] for row in receipt["declarations"]] == [
        "pricing.discounted", "pricing.standard",
    ]
