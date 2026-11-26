"""Exercise the whole-command runner and its actual provider replies without inference."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import zlib
from http.server import ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from threading import Thread

import pytest

from _journal import measurement_events
from engine_run import run_monitored
from fixtures.final_wire_reads import _Journal
from stateframe.scripted_endpoint import Script, make_handler
from tools.pinned_tree import pinned_tree

ROOT = Path(__file__).resolve().parents[1]
SWITCH = "STATEFRAME_EXCEPTION_CAUSE_EXCERPT"
MARKER = "[Direct exception causes]"
DELIMITER = "\nThe above exception was the direct cause of the following exception:\n\n"


def _trace(path, line, function, exception, message, *, padding=""):
    source = f"    # {padding}\n" if padding else ""
    return (f'Traceback (most recent call last):\n  File "{path}", line {line}, in {function}\n'
            f"{source}{exception}: {message}\n")


PADDING = "x" * 4300
INNER = _trace("pkg/decoder.py", 7, "parse_port", "ValueError", "decoder detail", padding=PADDING)
INNER = INNER.replace('  File "pkg/decoder.py"',
                      '  File "config.py", line 3, in dispatch\n  File "pkg/decoder.py"')
INNER = INNER.replace("ValueError: ", '  File "vendor/adapter.py", line 4, in convert\nValueError: ')
MIDDLE = _trace("pkg/loader.py", 9, "load_port", "PortError", "loader detail")
OUTER = _trace("config.py", 11, "load_config", "ConfigLoadError", "invalid configuration")
CHAIN = DELIMITER.join((INNER, MIDDLE, OUTER))
LOGGED_OUTER = OUTER.replace(
    "ConfigLoadError:", "ERROR:worker:retry failed\nConfigLoadError:")
VENDORED_CHAIN = CHAIN.replace("vendor/adapter.py", "pkg/_vendor/decoder.py")


def _fixtures():
    tight = DELIMITER.join((
        INNER.replace("ValueError", "Origin" + "X" * 2200 + "Error"),
        MIDDLE.replace("PortError", "Middle" + "Y" * 2200 + "Error"),
        OUTER,
    ))
    return {
        "valid": ("unrelated stdout\n", CHAIN),
        "two": (DELIMITER.join((INNER, OUTER)), "unrelated stderr\n"),
        "visible": ("", CHAIN.replace(PADDING, "short source line")),
        "truncated": ("", "p" * (256 * 1024 - len(CHAIN.encode()) - 64) + "\n"
                      + CHAIN + "log\n" * 100),
        "multiple": ("", OUTER + CHAIN),
        "split": (INNER + DELIMITER, DELIMITER.join((MIDDLE, OUTER))),
        "interleaved": ("", CHAIN.replace(DELIMITER, "\nworker log line\n" + DELIMITER, 1)),
        "interleaved_outer": ("", DELIMITER.join((INNER, MIDDLE, LOGGED_OUTER))),
        "vendored": ("", VENDORED_CHAIN),
        "implicit": ("", CHAIN.replace(DELIMITER, "\nDuring handling of the above exception,"
                                                  " another exception occurred:\n\n", 1)),
        "ambiguous": ("", CHAIN.replace("pkg/decoder.py", "decoder.py")),
        "tight": ("", tight),
        "pass": ("check passed\n", ""),
    }


def _repository(root):
    repo = root / "repo"
    for path in ("pkg/decoder.py", "pkg/loader.py", "other/decoder.py", "vendor/adapter.py",
                 "pkg/_vendor/decoder.py", "config.py"):
        source = repo / path
        source.parent.mkdir(parents=True, exist_ok=True)
        header = 'MODE = "valid"\n' if path == "config.py" else ""
        source.write_text(header + "\n" * 11, encoding="utf-8")
    captures = {
        name: tuple(base64.b64encode(zlib.compress(text.encode())).decode() for text in streams)
        for name, streams in _fixtures().items()
    }
    (repo / "verify.py").write_text(
        "import base64\nimport sys\nimport zlib\nfrom config import MODE\n"
        f"stdout, stderr = {captures!r}[MODE]\n"
        "sys.stdout.buffer.write(zlib.decompress(base64.b64decode(stdout)))\n"
        "sys.stderr.buffer.write(zlib.decompress(base64.b64decode(stderr)))\n"
        "raise SystemExit(0 if MODE == 'pass' else 1)\n",
        encoding="utf-8",
    )
    return repo


def _steps(modes):
    previous = "valid"
    for mode in modes:
        if mode != previous:
            yield {"tool": "str_replace", "arguments": {
                "path": "config.py", "old_str": f'MODE = "{previous}"', "new_str": f'MODE = "{mode}"',
            }}
        yield {"tool": "run_tests", "arguments": {"target": ""}}
        previous = mode
    yield {"tool": "done", "arguments": {"summary": "Verified the configuration check."}}


def _run(root, repo, *, tree=ROOT, switch=None, modes=("valid",)):
    root.mkdir()
    steps = list(_steps(modes))
    script = Script({"steps": steps, "tokenize": {"mode": "ok"}})
    handler = make_handler(script)
    requests = []

    class RawHandler(handler):
        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return super().do_POST()
            original = self.rfile
            raw = original.read(int(self.headers["Content-Length"]))
            assert self.headers.get("Content-Encoding") != "gzip"
            requests.append(raw)
            self.rfile = BytesIO(raw)
            try:
                super().do_POST()
            finally:
                self.rfile = original

    server = ThreadingHTTPServer(("127.0.0.1", 0), RawHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    run = root / "run"
    env = {name: value for name, value in os.environ.items()
           if not name.startswith(("STATEFRAME_", "SF_", "REACT_", "LLAMA_CPP_"))}
    env.update(STATEFRAME_HOME=str(root / "home"), PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join(str(tree / path) for path in ("src", "src/stateframe/engine")))
    if switch is not None:
        env[SWITCH] = switch
    try:
        process = run_monitored(
            [sys.executable, "-m", "stateframe", "run", "--repo", str(repo),
             "--task", "Check the configuration failure and update config.py.",
             "--target", "config.py", "--test-cmd", "python3 -B verify.py",
             "--endpoint", f"http://127.0.0.1:{server.server_port}/v1", "--model", "scripted",
             "--run-dir", str(run), "--test-python", sys.executable, "--no-run-env",
             "--max-turns", str(len(steps)), "--quiet"],
            env=env, capture_output=True, text=True, expected_seconds=240,
        )
        assert process.returncode == (0 if modes[-1] == "pass" else 1), process.stdout + process.stderr
        script.assert_consumed()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    recorded = [path.read_bytes() for path in sorted((run / "wire").glob("*.request.json"))]
    assert requests == recorded
    return run, requests


def _deliveries(run, requests):
    """Match each scripted check to the next wire reply and its public call ID."""
    replies = []
    for send, after in enumerate(requests[1:], start=2):
        response = json.loads((run / "wire" / f"{send - 1:04}.response.json").read_bytes())
        call, = response["choices"][0]["message"]["tool_calls"]
        if call["function"]["name"] != "run_tests":
            continue
        messages = json.loads(after)["messages"]
        latest = next(message for message in reversed(messages) if message.get("tool_calls"))
        delivered, = latest["tool_calls"]
        assert delivered["function"] == call["function"]
        reply, = (item for item in messages if item.get("tool_call_id") == delivered["id"])
        replies.append((send, reply["content"]))
    return replies


def test_r022_failed_whole_command_off_request_bytes_match_pinned_099(tmp_path, tmp_path_factory):
    # The required test name is historical. The wire contract is pinned to 0.10.0.
    baseline = pinned_tree("fe31666bd5e3a8d1e109c5f3cd7638ab176adcbe", tmp_path_factory)
    repo = _repository(tmp_path)
    base_run, expected = _run(tmp_path / "pinned", repo, tree=baseline)
    unset_run, unset = _run(tmp_path / "unset", repo)
    off_run, off = _run(tmp_path / "off", repo, switch="off")
    false_run, false = _run(tmp_path / "false", repo, switch="false")

    assert len(expected) == 2
    assert len(_deliveries(base_run, expected)) == 1
    assert unset == off == false == expected
    fields = ("execution_id", "receipt_id", "revision", "outcome", "full_output_handle",
              "stdout_bytes", "stderr_bytes")
    base_receipts = _receipt_fields(base_run, fields)
    for run in (unset_run, off_run, false_run):
        assert _receipt_fields(run, fields) == base_receipts


def test_r022_failed_whole_command_switch_defaults_off(tmp_path):
    run, requests = _run(tmp_path / "default", _repository(tmp_path))
    assert SWITCH not in json.loads((run / "spec.json").read_bytes())["environment"]
    deliveries = _deliveries(run, requests)
    assert len(deliveries) == 1
    assert all(MARKER.encode() not in body for body in requests)
    (_, reply), = deliveries
    assert reply.startswith("check exit 1: check failed\n")
    assert "ConfigLoadError: invalid configuration" in reply
    assert 'File "pkg/decoder.py"' not in reply


def test_r022_failed_whole_command_on_path_is_bounded_and_idempotent(tmp_path):
    repo = _repository(tmp_path)
    modes = ("valid", "valid", "two", "visible", "truncated", "multiple", "split",
             "interleaved", "interleaved_outer", "vendored", "implicit", "ambiguous", "tight", "pass")
    run, on = _run(tmp_path / "on", repo, switch="on", modes=modes)
    _assert_enabled_run(run, on, modes)


def test_r022_recovers_cause_from_clipped_executable_source(tmp_path):
    pkg = tmp_path / "repo" / "pkg"
    pkg.mkdir(parents=True)
    repo = pkg.parent
    (pkg / "decoder.py").write_text(
        "import base64\n"
        "def decode_config(encoded):\n"
        "    return base64.b64decode(encoded, validate=True) if encoded else b'"
        + "A" * 4300 + "'\n",
        encoding="utf-8",
    )
    (repo / "config.py").write_text(
        "from binascii import Error\n"
        "from pkg.decoder import decode_config\n"
        "class ConfigLoadError(Exception):\n"
        "    pass\n"
        "def load_config():\n"
        "    try:\n"
        "        decode_config('!')\n"
        "    except Error as error:\n"
        "        raise ConfigLoadError('invalid configuration') from error\n",
        encoding="utf-8",
    )
    (repo / "verify.py").write_text("from config import load_config\nload_config()\n", encoding="utf-8")

    run, requests = _run(tmp_path / "on", repo, switch="on")
    deliveries = _deliveries(run, requests)
    (_, delivered), = deliveries
    journal = _Journal(run)
    (check, ordinary), = _check_executions(journal, requests, deliveries)
    execution_id, = check["canonical_check_execution_ids"]
    receipt, = (event for event in journal.events if event["kind"] == "check_receipt"
                and event["execution_id"] == execution_id)
    assert receipt["outcome"] == "BEHAVIOR_FAIL"
    assert receipt["revision"] == check["binding"]["product_revision"]
    assert ordinary.startswith("check exit 1: check failed\n" + "A" * 1000)
    assert "pkg/decoder.py" not in ordinary
    for message in ("binascii.Error: Only base64 data is allowed",
                    "ConfigLoadError: invalid configuration"):
        assert message in ordinary
        assert message in delivered
    assert delivered.count(MARKER) == 1
    assert ("binascii.Error  pkg/decoder.py:3  decode_config\n  directly caused\n"
            "config.ConfigLoadError  config.py:9  load_config") in delivered
    assert delivered.endswith(ordinary.split("\n", 2)[2])
    assert len(delivered.encode()) <= len(ordinary.encode())


def test_r022_preserves_clipped_exception_message(tmp_path):
    pkg = tmp_path / "repo" / "pkg"
    pkg.mkdir(parents=True)
    repo = pkg.parent
    port_range = "expected a port in the range 1–65535"
    (pkg / "decoder.py").write_text(
        "def parse_port(value):\n"
        f"    raise ValueError(f'Invalid port {{value!r}}; {port_range}')\n",
        encoding="utf-8",
    )
    # Clip the exception line while keeping its escaped JSON within the request budget.
    (repo / "config.py").write_text(
        "from pkg.decoder import parse_port\n"
        "class ConfigLoadError(Exception):\n"
        "    pass\n"
        "def load_config():\n"
        "    try:\n"
        "        parse_port('éA' * 2500)\n"
        "    except ValueError as error:\n"
        "        raise ConfigLoadError('invalid configuration') from error\n",
        encoding="utf-8",
    )
    (repo / "verify.py").write_text("from config import load_config\nload_config()\n", encoding="utf-8")

    run, requests = _run(tmp_path / "on", repo, switch="on")
    deliveries = _deliveries(run, requests)
    (_, delivered), = deliveries
    (_, ordinary), = _check_executions(_Journal(run), requests, deliveries)
    assert "éA" in ordinary
    assert port_range in ordinary
    assert "ValueError: Invalid port" not in ordinary
    assert "ConfigLoadError: invalid configuration" in ordinary
    assert port_range in delivered
    assert delivered == ordinary


def _project_cause_chain(tmp_path, stdout, stderr):
    from kernel_generic_check import _cause_chain

    repo = _repository(tmp_path)
    binding = {"product_files": {path.relative_to(repo).as_posix(): None for path in repo.rglob("*.py")}}
    return _cause_chain(repo, stdout, stderr, binding)


def test_r022_abstains_when_a_log_interrupts_the_outer_traceback(tmp_path):
    stdout, stderr = _fixtures()["interleaved_outer"]
    assert "ConfigLoadError: invalid configuration" in stderr[-4000:]
    assert _project_cause_chain(tmp_path, stdout, stderr) is None


def test_r022_skips_nested_python_vendor_frames(tmp_path):
    stdout, stderr = _fixtures()["vendored"]
    frames, _ = _project_cause_chain(tmp_path, stdout, stderr)
    assert [frame.location for frame in frames] == [
        "pkg/decoder.py:7", "pkg/loader.py:9", "config.py:11",
    ]


@pytest.mark.parametrize("clipped_line", ["short", "source-comment", "exception-message"])
def test_r022_preserves_the_failing_assertion_at_the_minimum_reply_cap(tmp_path, clipped_line):
    from kernel_frame_tool_result_budget import (
        bound_frame_tool_result, fit_check_cause_excerpt, frame_only_page_byte_limit,
    )
    from kernel_request_budget import HARD_MAX_REQUEST_BYTES

    assertion = "test_config_contract\nAssertionError: the public exception contract failed\n"
    stdout = assertion + "check progress " * 60 + "\n"
    stderr = CHAIN.replace(PADDING, "x" * 1800)
    port_range = "expected a port in the range 1–65535"
    if clipped_line == "source-comment":
        padded_middle = MIDDLE.replace("PortError: loader detail", "    # " + "m" * 600 + "\nPortError")
        stderr = stderr.replace(MIDDLE, padded_middle).replace("invalid configuration", "bad")
    elif clipped_line == "exception-message":
        padded_middle = MIDDLE.replace("loader detail", "m" * 600 + "; " + port_range)
        stderr = stderr.replace(MIDDLE, padded_middle).replace("invalid configuration", "bad")
    frames, source_line_ends = _project_cause_chain(tmp_path, stdout, stderr)
    assert len(frames) == 3
    observation = f"check exit 1: check failed\n{stdout}{stderr}"
    assert len(stdout + stderr) < 4000
    advisory = "\n[kernel] the failure anchors at config.py:11:load_config"
    trailer = "\n[kernel] NO required selector completed"
    messages = [{"role": "user", "content": "x" * HARD_MAX_REQUEST_BYTES}]
    assert frame_only_page_byte_limit(messages) == 1024
    ordinary = bound_frame_tool_result(observation + advisory, messages, kind="test", trailer=trailer)
    assert assertion in ordinary
    assert frames[-1].visible_in(ordinary)
    assert not frames[0].visible_in(ordinary)
    if clipped_line == "exception-message":
        assert port_range in ordinary
        assert "PortError:" not in ordinary

    delivered = fit_check_cause_excerpt(
        ordinary, observation + advisory + trailer, observation, frames,
        source_line_ends=source_line_ends,
    )

    assert assertion in delivered
    assert frames[-1].visible_in(delivered)
    assert delivered.endswith(advisory + trailer)
    assert len(delivered.encode()) <= len(ordinary.encode()) <= 1024
    head, separator, _ = ordinary.partition("\n[TEST OUTPUT CLIPPED:")
    assert separator and delivered.startswith(head + separator)
    if clipped_line == "source-comment":
        assert delivered.count(MARKER) == 1
        assert all(frame.visible_in(delivered) for frame in frames)
        assert len(delivered.encode()) == len(ordinary.encode())
    else:
        if clipped_line == "exception-message":
            assert port_range in delivered
        assert delivered == ordinary


def _assert_enabled_run(run, requests, modes):
    deliveries = _deliveries(run, requests)
    journal = _Journal(run)
    checks = _check_executions(journal, requests, deliveries)
    receipts = {event["execution_id"]: event for event in journal.events
                if event["kind"] == "check_receipt"}
    expected_rows = (
        "ValueError  pkg/decoder.py:7  parse_port\n  directly caused\n"
        "PortError  pkg/loader.py:9  load_port\n  directly caused\n"
        "ConfigLoadError  config.py:11  load_config"
    )
    for mode, (_, enhanced), (check, original) in zip(modes, deliveries, checks, strict=True):
        execution_id, = check["canonical_check_execution_ids"]
        receipt = receipts[execution_id]
        assert receipt["revision"] == check["binding"]["product_revision"]
        assert receipt["outcome"] == ("PASS" if mode == "pass" else "BEHAVIOR_FAIL")
        if mode == "truncated":
            assert receipt["stdout_bytes"] == 256 * 1024
        if mode not in {"valid", "two", "vendored"}:
            assert enhanced == original, mode
            assert MARKER not in enhanced
            if mode != "pass":
                assert "ConfigLoadError: invalid configuration" in original
            if mode == "interleaved_outer":
                assert "ERROR:worker:retry failed" in original
            continue
        rows = (expected_rows if mode != "two" else expected_rows.replace(
            "PortError  pkg/loader.py:9  load_port\n  directly caused\n", ""))
        assert enhanced.startswith(f"check exit 1: check failed\n{MARKER}\n{rows}\n")
        assert enhanced.count(MARKER) == 1
        assert len(enhanced.encode()) <= len(original.encode())
        assert 'File "pkg/decoder.py"' not in original
        assert "ConfigLoadError: invalid configuration" in enhanced
        assert "\n[kernel] the failure anchors at config.py:11:load_config" in original
        assert "\n[kernel] the failure anchors at config.py:11:load_config" in enhanced
        scope = "\n[kernel] NO required selector completed"
        assert enhanced.split(scope)[1] == original.split(scope)[1]

    first, second, third = (check for check, _ in checks[:3])
    assert first["binding"]["product_revision"] == second["binding"]["product_revision"]
    assert second["binding"]["product_revision"] != third["binding"]["product_revision"]
    assert first["canonical_check_execution_ids"] != second["canonical_check_execution_ids"]


def _check_executions(journal, requests, deliveries):
    sent = {event["payload_sha256"]: event for event in journal.events if event["kind"] == "request_body"}
    checks = [event for event in journal.events if event["kind"] == "test_subprocess_completed"]
    observations = {}
    for event in journal.events:
        if event["kind"] == "observation_committed":
            observation = journal.hydrate(event["observation"])
            if observation["action"] == "test":
                observations[observation["step"]] = observation["observation"]
    matched = []
    for send, _ in deliveries:
        # The previous send invoked the check, and this send carried its result.
        digest = "sha256:" + hashlib.sha256(requests[send - 2]).hexdigest()
        request = sent[digest]
        check, = (event for event in checks if event.get("request_id") == request["request_id"])
        matched.append((check, observations[request["step"]]))
    return matched


def _receipt_fields(run, fields):
    return [tuple(event[field] for field in fields)
            for event in measurement_events(run) if event["kind"] == "check_receipt"]
