"""The scripted request must carry the cache layout on its sealed provider wire."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from kernel_cache_layout_view import CacheLayoutError, semantic_messages
from kernel_stateframe_page_log import _payload_has_authority_keys
from stateframe.scripted_endpoint import Script, serve
from tools.pinned_tree import pinned_tree

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.host_sandbox
# Per scripted run. tiny_site measured ~420 s solo for the whole parity test on the foundry host;
# a 240 s per-call limit flaked under full-suite load, so allow ~1.5x the solo time per call.
RUN_TIMEOUT_SECONDS = 600


def _parity_launchers(tmp_path):
    launcher = tmp_path / "parity_launcher.py"
    wrapper = tmp_path / "parity_child.py"
    launcher.write_text(
        "import secrets, subprocess, sys\n"
        "from stateframe import cli\n"
        "wrapper, *args = sys.argv[1:]\n"
        "original = secrets.token_hex\n"
        "used = False\n"
        "def fixed_hex(size=None):\n"
        "    global used\n"
        "    if size == 16 and not used:\n"
        "        used = True\n"
        "        return 'a' * 32\n"
        "    return original(size)\n"
        "secrets.token_hex = fixed_hex\n"
        "popen = subprocess.Popen\n"
        "def child(command, *positional, **keywords):\n"
        "    if len(command) >= 4 and str(command[2]).endswith('_child.py'):\n"
        "        command = [command[0], command[1], wrapper, command[2], command[3]]\n"
        "    return popen(command, *positional, **keywords)\n"
        "subprocess.Popen = child\n"
        "raise SystemExit(cli.main(args))\n")
    wrapper.write_text(
        "import os, runpy, sys\n"
        "child, run = sys.argv[1:]\n"
        "sys.path.insert(0, os.path.dirname(child))\n"
        "sys.argv = [child, run]\n"
        "runpy.run_path(child, run_name='__main__')\n")
    return launcher, wrapper


@pytest.mark.parametrize("example,task", [
    ("discount_bug", "Fix the pricing code so the tests pass."),
    ("tiny_site", "Build the small static shop site described in the README."),
    ("brief_to_report", "Write the report the brief asks for."),
])
def test_switch_off_bytes_match_base_release(tmp_path, tmp_path_factory, example, task):
    baseline = pinned_tree("e19c325", tmp_path_factory)
    engine_record = baseline / "src/stateframe/engine/ENGINE_SOURCE.json"
    assert engine_record.is_file()
    released = subprocess.run(
        ["git", "show", "e19c325:src/stateframe/engine/ENGINE_SOURCE.json"],
        cwd=ROOT, capture_output=True, check=True).stdout
    assert engine_record.read_bytes() == released
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "examples" / example, repo)
    launcher, wrapper = _parity_launchers(tmp_path)
    outputs = []
    for checkout, label in ((baseline, "baseline"), (ROOT, "current")):
        script = Script(json.loads((ROOT / "examples" / f"{example}.script.json").read_text()))
        server = serve(script)
        run = tmp_path / "run"
        environment = {**os.environ, "PYTHONPATH": str(checkout / "src")}
        environment.pop("STATEFRAME_FINAL_CACHE_LAYOUT", None)
        environment["STATEFRAME_PAGE_LOG"] = "off"
        try:
            result = subprocess.run(
                [sys.executable, str(launcher), str(wrapper), "run", "--repo", str(repo),
                 "--task", task, "--endpoint", f"http://127.0.0.1:{server.server_port}/v1",
                 "--model", "scripted", "--run-dir", str(run),
                 "--test-python", sys.executable, "--max-turns", "16", "--quiet"],
                env=environment, capture_output=True, text=True, timeout=RUN_TIMEOUT_SECONDS)
            assert result.returncode == 0, result.stdout + result.stderr
            script.assert_consumed()
            outputs.append([path.read_bytes() for path in sorted((run / "wire").glob("*.request.json"))])
            if label == "baseline":
                run.rename(tmp_path / "baseline_run")
        finally:
            server.shutdown()
            server.server_close()
    assert outputs[0] and outputs[1] == outputs[0]


def test_scripted_final_sealed_wire_and_planted_violation(tmp_path):
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "examples/discount_bug", repo)
    script = Script(json.loads((ROOT / "examples/discount_bug.script.json").read_text()))
    server = serve(script)
    run = tmp_path / "run"
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    environment["STATEFRAME_FINAL_CACHE_LAYOUT"] = "on"
    environment["STATEFRAME_PAGE_LOG"] = "off"
    try:
        result = subprocess.run(
            [sys.executable, "-m", "stateframe", "run", "--repo", str(repo),
             "--task", "Fix the pricing code so the tests pass.",
             "--endpoint", f"http://127.0.0.1:{server.server_port}/v1",
             "--model", "scripted", "--run-dir", str(run),
             "--test-python", sys.executable, "--max-turns", "16", "--quiet"],
            env=environment, capture_output=True, text=True, timeout=RUN_TIMEOUT_SECONDS)
        assert result.returncode == 0, result.stdout + result.stderr
        script.assert_consumed()
    finally:
        server.shutdown()
        server.server_close()

    wires = sorted((run / "wire").glob("*.request.json"))
    assert wires
    digests = set()
    for path in wires:
        raw = path.read_bytes()
        digests.add("sha256:" + hashlib.sha256(raw).hexdigest())
        rows = json.loads(raw)["messages"]
        state_index = next(i for i, row in enumerate(rows)
                           if str(row.get("content") or "").startswith("[StateFrame]\n"))
        assert state_index > 2
        assert any(str(row.get("content") or "").startswith("[Current source:")
                   for row in rows[2:state_index])
        assert sum(str(row.get("content") or "").count("[CONTINUATION LIMITS |")
                   for row in rows) == 1
        assert "[CONTINUATION LIMITS |" in rows[-1]["content"]
        frame = json.loads(rows[state_index]["content"].split("\n", 1)[1])
        assert not _payload_has_authority_keys(frame, final=True)
        assert semantic_messages(rows)[2]["content"].startswith("[StateFrame]\n")

    # A finished run seals its measurement journal as events.jsonl.xz; open_journal reads both forms.
    from stateframe.engine.kernel_journal_store import open_journal
    events = []
    for file in [*run.rglob("events.jsonl"), *run.rglob("events.jsonl.xz")]:
        with open_journal(file.parent) as stream:
            events.extend(json.loads(line) for line in stream.read().splitlines())
    final_events = [row for row in events if row.get("event") == "cache_layout_final_wire"
                    or row.get("kind") == "cache_layout_final_wire"]
    assert final_events
    assert digests.issubset({row.get("body_sha256") for row in final_events})
    assert all(row.get("authority_message_index") == len(row.get("message_order", ())) - 1
               for row in final_events)
    assert all(row["message_order"][0] == "system"
               and row["message_order"][-1] == "final_authority"
               and row["message_order"].index("source") < row["message_order"].index("state")
               for row in final_events)

    broken = copy.deepcopy(json.loads(wires[0].read_bytes())["messages"])
    broken[2]["content"] += "\n[CONTINUATION LIMITS | planted]"
    with pytest.raises(CacheLayoutError, match="physically final"):
        semantic_messages(broken)
