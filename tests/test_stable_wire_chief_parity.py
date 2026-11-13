"""Disabled request and tape bytes stay pinned to the candidate before the chief port."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.pinned_tree import pinned_tree


@pytest.mark.parametrize("final_layout", ["on", "off"])
def test_chief_port_disabled_request_and_tape_bytes_match_candidate(
        tmp_path, tmp_path_factory, final_layout):
    baseline = pinned_tree("37f0ac81a33eeee8e57747cfacdfdf64c4c48752", tmp_path_factory)
    root = Path(__file__).resolve().parents[1]
    script = r'''
import json, sys
from pathlib import Path
from types import SimpleNamespace
import kernel_bounded_http
import kernel_request_source_window as source
from kernel_cache_layout_candidate import render_final_cache_candidate
from kernel_measurement import ACTIVE_REQUEST
from stateframe import _child
from test_stable_wire import Meter, _final_layout_body
source.verify_final_source_delivery = lambda *a, **k: {"status": "COMPLETE"}
source.source_delivery_satisfied = lambda receipt: True
capture = SimpleNamespace(request_id="off", request_source_final_delivery=None)
body = render_final_cache_candidate(capture, SimpleNamespace(), _final_layout_body(),
    meter=Meter(), inputs={"token_ceiling":1000, "byte_ceiling":1000000,
                          "source_plan":{}, "sources":{}, "receipt":None})
wire = json.dumps(body).encode()
kernel_bounded_http.request_bytes = lambda *a, **k: b'{"choices":[{}]}'
_child.time.monotonic = lambda: 1.0
_child.emit = lambda *a, **k: None
run = Path(sys.argv[1]); run.mkdir()
_child.install_wire_recorder(run, "http://localhost/chat")
active = ACTIVE_REQUEST.set(capture)
try:
    kernel_bounded_http.request_bytes(wire, "http://localhost/chat", "")
finally:
    ACTIVE_REQUEST.reset(active)
print(json.dumps({str(p.relative_to(run)):p.read_bytes().hex()
                  for p in run.rglob("*") if p.is_file()}, sort_keys=True))
'''
    outputs = []
    for checkout, name in ((baseline, "baseline"), (root, "current")):
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("STATEFRAME_", "KERNEL_", "DS_", "REACT_", "PILOT_", "UNIT_PLAN_"))}
        env.update(STATEFRAME_STABLE_WIRE="off", STATEFRAME_FINAL_CACHE_LAYOUT=final_layout,
                   STATEFRAME_PAGE_LOG="off",
                   PYTHONPATH=os.pathsep.join(str(path) for path in (
                       checkout / "src/stateframe/engine", checkout / "src", root / "tests")))
        result = subprocess.run([sys.executable, "-c", script, str(tmp_path / name)],
                                env=env, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        outputs.append(json.loads(result.stdout))
    assert outputs[0] == outputs[1]
    assert set(outputs[1]) == {"wire/0001.request.json", "wire/0001.response.json", "sends.jsonl"}
    assert "stable_wire_reserves" not in bytes.fromhex(outputs[1]["sends.jsonl"]).decode()
