"""Resume after a run that stopped below its turn limit does not ask for a larger limit."""
import json
import shutil
import subprocess
import sys
from http.server import ThreadingHTTPServer
from threading import Thread

import pytest

from _journal import measurement_events
from stateframe.resume import assess
from stateframe.scripted_endpoint import Script, make_handler
from test_final_cache_layout_host import (
    ROOT, RUN_TIMEOUT_SECONDS, _parity_launchers, _scripted_environment,
)

pytestmark = pytest.mark.host_sandbox


@pytest.mark.parametrize("case, max_turns", [
    ("stop", "7"), ("stop-below-limit", "8"), ("stop-unbounded", "0"), ("turn-limit", "1"),
])
def test_resume_after_stop(tmp_path, case, max_turns):
    run, repo = tmp_path / "run", tmp_path / "repo"
    shutil.copytree(ROOT / "examples" / "discount_bug", repo)
    document = json.loads((ROOT / "examples" / "discount_bug.script.json").read_text())
    read, edit, check, done = document["steps"]
    reads = [read]
    for index in range(3):
        name = f"note{index}.txt"
        (repo / name).write_text(f"Note {index}\n")
        reads.append({"tool": "read", "arguments": {"path": name}})
    document["steps"] = [*reads, edit, check, done]
    script = Script(document)
    launcher, wrapper = _parity_launchers(tmp_path)
    wrapper.write_text(wrapper.read_text().removesuffix("runpy.run_path(child, run_name='__main__')\n") + f'''
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(child).parent / 'engine'))
import kernel_final_wire_allocation as allocation
import kernel_wire_truth_gate as truth

build = allocation.build_qualified_request
evaluate = truth.evaluate_wire
active = None

def build_request(capture, *args, **kwargs):
    global active
    previous, active = active, capture
    try:
        return build(capture, *args, **kwargs)
    finally:
        active = previous

def evaluate_request(*args, **kwargs):
    result = evaluate(*args, **kwargs)
    if {case!r} == 'turn-limit' or active is None or active.step != 5:
        return result
    return replace(result, verdict='BLOCKED', failed=(*result.failed, 'test_block'))

allocation.build_qualified_request = build_request
truth.evaluate_wire = evaluate_request
runpy.run_path(child, run_name='__main__')
''')
    environment = {**_scripted_environment(ROOT), "STATEFRAME_WORKING_SET_ALLOCATOR": "on"}
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(script))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"

    def cli(*args):
        result = subprocess.run(
            [sys.executable, str(launcher), str(wrapper), *args], env=environment,
            capture_output=True, text=True, timeout=RUN_TIMEOUT_SECONDS,
        )
        return result, result.stdout + result.stderr + (run / "engine.log").read_text()

    def terminal():
        return json.loads((run / "engine-result.json").read_text())["result"]

    def blocks():
        return sum(row["kind"] == "WIRE_TRUTH_BLOCKED" for row in measurement_events(run))

    try:
        _, detail = cli("run", "--repo", str(repo), "--task", "Fix the discount in shop/pricing.py.",
                        "--endpoint", endpoint, "--model", "scripted", "--run-dir", str(run),
                        "--test-python", sys.executable, "--max-turns", max_turns, "--quiet")
        before, sent = terminal(), len(script.requests)
        if case == "turn-limit":
            assert before["result"] == "RESPONSE_BUDGET_EXHAUSTED", before
            assert before["steps"] == sent == 1, detail
            result, detail = cli("resume", str(run), "--endpoint", endpoint, "--quiet")
            assert result.returncode == 2, detail
            assert "the last attempt stopped at its limit (RESPONSE_BUDGET_EXHAUSTED, step limit 1)" in detail
            assert not (run / "attempts").exists(), detail
            assert assess(run, max_turns=2)["max_turns"] == 2
            assert assess(run, max_turns=0)["max_turns"] == 0
            return
        assert before["reason_code"] == "boundary_unrecoverable", before
        assert blocks() and sent == 5, detail
        if case == "stop":
            refused, detail = cli("resume", str(run), "--endpoint", endpoint, "--quiet", "--max-turns", "8")
            assert refused.returncode == 2, detail
            assert "resume keeps the original budget" in detail, detail
            assert "to use a different limit, start a fresh run" in detail, detail
        _, detail = cli("resume", str(run), "--endpoint", endpoint, "--quiet")
        assert "the last attempt stopped at its limit" not in detail, detail
        assert (run / "attempts/1/engine-result.json").is_file(), detail
        after = terminal()
        assert (after["result"], after["reason_code"]) == (before["result"], before["reason_code"]), detail
        assert len(script.requests) == sent, detail
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
