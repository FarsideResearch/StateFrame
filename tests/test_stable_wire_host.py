"""Engine-level whole-message stable prefix property and its planted control."""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
import sys
from pathlib import Path

import pytest

from engine_run import run_monitored
from stateframe.scripted_endpoint import Script, serve

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.host_sandbox
RUN_TIMEOUT_SECONDS = 600
FRAME_PREFIX = "[StateFrame]\n"


def _index(rows, prefix):
    return next(index for index, row in enumerate(rows)
                if str(row.get("content", "")).startswith(prefix))


def _status_epoch(rows):
    content = rows[_index(rows, "[Source windows status]")]["content"]
    match = re.search(r"(?m)^Epoch (\d+)$", content)
    assert match is not None
    return int(match.group(1))


def _assert_same_epoch_stable_messages(previous, current):
    assert _status_epoch(previous) == _status_epoch(current)
    # Only messages ahead of the frame row (system, task, windows) are epoch-stable.
    # The frame sits after the windows and may change within an epoch.
    previous_end = _index(previous, FRAME_PREFIX)
    current_end = _index(current, FRAME_PREFIX)
    assert len(previous[:previous_end]) <= len(current[:current_end])
    for index, message in enumerate(previous[:previous_end]):
        assert current[index] == message


def _stable_wire_prefix(wire):
    """Return raw request bytes through the message before the frame row."""
    rows = json.loads(wire)["messages"]
    volatile = json.dumps(rows[_index(rows, FRAME_PREFIX)]).encode("utf-8")
    # kernel_measurement serializes the provider body with json.dumps(body).encode().
    marker = b", " + volatile
    offset = wire.find(marker)
    assert offset >= 0
    assert wire.find(marker, offset + 1) < 0
    return wire[:offset]


def _assert_same_epoch_stable_wire_prefix(previous_wire, current_wire):
    previous_prefix = _stable_wire_prefix(previous_wire)
    current_prefix = _stable_wire_prefix(current_wire)
    assert current_prefix.startswith(previous_prefix)


def _add_json_separator_space(wire):
    rows = json.loads(wire)["messages"]
    volatile_index = _index(rows, "[StateFrame volatile]")
    message = next(row for row in rows[:volatile_index]
                   if row.get("role") == "system" and "content" in row)
    encoded = json.dumps(message).encode("utf-8")
    separator = encoded.find(b'", "content":')
    assert separator >= 0
    row_start = wire.find(encoded)
    assert 0 <= row_start < len(_stable_wire_prefix(wire))
    insert_at = row_start + separator + 2
    return wire[:insert_at] + b" " + wire[insert_at:]


def _change_source_window_byte(wire):
    rows = json.loads(wire)["messages"]
    volatile_index = _index(rows, "[StateFrame volatile]")
    source = next(row for row in rows[:volatile_index]
                  if str(row.get("content", "")).startswith("[Current source:"))
    content = source["content"]
    path_start = content.index("[Current source: ") + len("[Current source: ")
    original = content[path_start].encode("utf-8")
    assert len(original) == 1
    replacement = b"Y" if original != b"Y" else b"Z"
    marker = b"[Current source: " + original
    offset = wire.find(marker)
    assert offset >= 0
    return wire[:offset + len(marker) - 1] + replacement + wire[offset + len(marker):]


def test_whole_message_control_allows_changed_frame_but_detects_corrupted_window():
    previous = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task: stable"},
        {"role": "user", "content": "[Current source: pkg/a.py]\nvalue = 1"},
        {"role": "user", "content": "[StateFrame]\n{\"stable\":1}"},
        {"role": "user", "content": "[StateFrame volatile]\n{}"},
        {"role": "user", "content": "[Source windows status]\nEpoch 1"},
    ]
    current = copy.deepcopy(previous)
    _assert_same_epoch_stable_messages(previous, current)

    # The frame row follows the windows, so a changed frame byte is allowed in-epoch.
    current[3]["content"] += " "
    _assert_same_epoch_stable_messages(previous, current)

    # Planted control: corrupting a delivered source window also breaks its whole message.
    current = copy.deepcopy(previous)
    current[2]["content"] += " "
    with pytest.raises(AssertionError):
        _assert_same_epoch_stable_messages(previous, current)


def test_tiny_site_scripted_run_preserves_whole_stable_messages(tmp_path):
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "examples/tiny_site", repo)
    script = Script(json.loads((ROOT / "examples/tiny_site.script.json").read_text()))
    server = serve(script)
    run = tmp_path / "run"
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    environment["STATEFRAME_FINAL_CACHE_LAYOUT"] = "on"
    environment["STATEFRAME_STABLE_WIRE"] = "on"
    environment["STATEFRAME_STABLE_WIRE_HEADROOM"] = "16000"
    environment["STATEFRAME_PAGE_LOG"] = "off"
    try:
        result = run_monitored(
            [sys.executable, "-m", "stateframe", "run", "--repo", str(repo),
             "--task", "Build the small static shop site described in the README.",
             "--endpoint", f"http://127.0.0.1:{server.server_port}/v1",
             "--model", "scripted", "--run-dir", str(run),
             "--test-python", sys.executable, "--max-turns", "16", "--quiet"],
            env=environment, capture_output=True, text=True, expected_seconds=RUN_TIMEOUT_SECONDS, run_dir=run)
        assert result.returncode == 0, result.stdout + result.stderr
        script.assert_consumed()
    finally:
        server.shutdown()
        server.server_close()

    wires = sorted((run / "wire").glob("*.request.json"))
    assert len(wires) >= 2
    wire_bytes = [path.read_bytes() for path in wires]
    requests = [json.loads(wire)["messages"] for wire in wire_bytes]
    # The final layout is built on a detached candidate. Its trim authority must
    # reach the published capture used by the real child recorder, not stay local.
    sends = [json.loads(line) for line in (run / "sends.jsonl").read_text().splitlines()]
    for path, rows, send in zip(wires, requests, sends, strict=True):
        record = json.loads(
            path.with_name(path.name.replace(".request.json", ".stable-wire.json")).read_bytes())
        assert record == send["stable_wire_reserves"]
        assert set(record) == {"reserves", "trim_count", "epoch", "initial_selection"}
        assert record["trim_count"] == len(record["reserves"])
        assert record["epoch"] == _status_epoch(rows)
    host_fact_order_checked = False
    for rows in requests:
        assert sum(str(row.get("content", "")).startswith("[Source windows status]")
                   for row in rows) == 1
        frame_index = _index(rows, "[StateFrame]\n")
        volatile_index = _index(rows, "[StateFrame volatile]")
        status_index = _index(rows, "[Source windows status]")
        assert frame_index + 1 == volatile_index
        assert volatile_index + 1 == status_index
        assert all(index < frame_index for index, row in enumerate(rows)
                   if str(row.get("content", "")).startswith("[Current source:"))
        assert all(row.get("name") != "stateframe_stable_wire_status"
                   for row in rows[:volatile_index])
        host_facts = [index for index, row in enumerate(rows)
                      if str(row.get("content", "")).startswith("[Host fact]")]
        if host_facts:
            assert status_index < host_facts[0]
            host_fact_order_checked = True

    same_epoch_pairs = []
    for previous_wire, current_wire, previous, current in zip(
            wire_bytes, wire_bytes[1:], requests, requests[1:]):
        previous_epoch = _status_epoch(previous)
        current_epoch = _status_epoch(current)
        if previous_epoch == current_epoch:
            _assert_same_epoch_stable_messages(previous, current)
            _assert_same_epoch_stable_wire_prefix(previous_wire, current_wire)
            same_epoch_pairs.append((previous_wire, current_wire, previous, current))

    # A repeated recorded capture supplies a same-epoch control if this run rebases
    # on every send; otherwise use the first real same-epoch consecutive pair.
    if same_epoch_pairs:
        previous_wire, current_wire, previous, current = same_epoch_pairs[0]
    else:
        previous_wire = wire_bytes[-1]
        current_wire = bytes(previous_wire)
        previous = json.loads(previous_wire)["messages"]
        current = json.loads(current_wire)["messages"]
    _assert_same_epoch_stable_messages(previous, current)
    _assert_same_epoch_stable_wire_prefix(previous_wire, current_wire)

    # A serialization-only change keeps parsed messages equal while breaking KV bytes.
    spaced_wire = _add_json_separator_space(current_wire)
    spaced_rows = json.loads(spaced_wire)["messages"]
    assert spaced_rows == current
    _assert_same_epoch_stable_messages(previous, spaced_rows)
    with pytest.raises(AssertionError):
        _assert_same_epoch_stable_wire_prefix(previous_wire, spaced_wire)

    # Changing a delivered source byte breaks both semantic and serialized checks.
    changed_source_wire = _change_source_window_byte(current_wire)
    changed_source_rows = json.loads(changed_source_wire)["messages"]
    with pytest.raises(AssertionError):
        _assert_same_epoch_stable_messages(previous, changed_source_rows)
    with pytest.raises(AssertionError):
        _assert_same_epoch_stable_wire_prefix(previous_wire, changed_source_wire)

    if any(str(row.get("content", "")).startswith("[Host fact]")
           for rows in requests for row in rows):
        assert host_fact_order_checked
