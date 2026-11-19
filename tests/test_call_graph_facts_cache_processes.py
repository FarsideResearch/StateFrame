"""Exercise the same facts cache from two independent engine processes."""

import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

from _timeouts import scaled

WORKER = """
import json
import os
import sys
import time
from pathlib import Path

from kernel_invalidation import LedgerIndex
from mc_repo_intel_vendor import call_graph

root, phases, slot = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
timeout = float(sys.argv[4])
replace = call_graph.os.replace
extract_facts = call_graph._extract_python_facts
first_write = True
extracted = []

def wait_for(marker):
    deadline = time.monotonic() + timeout
    while not marker.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(str(marker))
        time.sleep(0.01)

def publish(temporary, destination):
    global first_write
    staged = first_write and "python-call-facts" in destination.parts
    if staged:
        first_write = False
        (phases / ("staged-" + slot)).write_text(
            json.dumps([str(temporary), str(destination)]), encoding="utf-8",
        )
        (phases / ("ready-" + slot)).touch()
        wait_for(phases / "publish")
    replace(temporary, destination)
    if staged:
        (phases / ("published-" + slot)).touch()
        wait_for(phases / "observed")

def extract(source, relpath, filename):
    extracted.append(relpath)
    return extract_facts(source, relpath, filename)

def dependencies():
    ledger = LedgerIndex()
    ledger.configure_call_graph(root)
    return ledger.dependencies_for("test_answer.py")

call_graph.os.replace = publish
call_graph._extract_python_facts = extract
cold = dependencies()
cold_extracted = extracted.copy()
(phases / ("done-" + slot)).touch()
wait_for(phases / "read")
call_graph.clear_build_memo()
call_graph._parse_cache.clear()
extracted.clear()
warm = dependencies()
print(json.dumps({"pid": os.getpid(), "cold": cold, "cold_extracted": cold_extracted,
                  "warm": warm, "warm_extracted": extracted}))
"""


def _wait_for_paths(paths, processes):
    deadline = time.monotonic() + scaled(40)
    while not all(path.exists() for path in paths):
        exits = [process.poll() for process in processes]
        assert all(code is None for code in exits), exits
        assert time.monotonic() < deadline, f"Timed out waiting for {paths}"
        time.sleep(0.01)


def test_two_engines_publish_and_read_shared_facts_atomically(tmp_path):
    tree = Path(__file__).resolve().parents[1]
    home = tmp_path / "home"
    phases = tmp_path / "phases"
    phases.mkdir()
    sources = {"answer.py": "def answer(): return 1\n",
               "test_answer.py": "from answer import answer\ndef test_answer(): return answer()\n"}
    sources.update({f"module_{i:02d}.py": f"def value(): return {i}\n" for i in range(24)})

    with ExitStack() as stack:
        processes = []
        for slot in ("one", "two"):
            root = tmp_path / slot
            root.mkdir()
            for name, source in sources.items():
                (root / name).write_text(source, encoding="utf-8")
            env = dict(os.environ, STATEFRAME_HOME=str(home),
                       STATEFRAME_REPO_CACHE_DIR=str(tmp_path / (slot + "-cache")),
                       STATEFRAME_TURN_REPO_INTEL="off", CODE_GRAPH_LSP="off",
                       PYTHONPATH=str(tree / "src/stateframe/engine"))
            process = subprocess.Popen(
                [sys.executable, "-B", "-c", WORKER, str(root), str(phases), slot, str(scaled(60))],
                env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            stack.enter_context(process)
            stack.callback(lambda child=process: child.kill() if child.poll() is None else None)
            processes.append(process)

        staged = [phases / f"staged-{slot}" for slot in ("one", "two")]
        _wait_for_paths([phases / f"ready-{slot}" for slot in ("one", "two")], processes)
        first, second = [[Path(value) for value in json.loads(path.read_text())]
                         for path in staged]
        assert first[0] != second[0]
        assert first[1] == second[1]
        assert first[1].is_relative_to(home / "repo-intel")
        assert not first[1].exists(), "A reader could see an entry before atomic publication"
        for temporary, _ in (first, second):
            assert json.loads(temporary.read_bytes())["extract"]["defs"][0]["file"] == "answer.py"
        (phases / "publish").touch()
        _wait_for_paths([phases / f"published-{slot}" for slot in ("one", "two")], processes)
        assert json.loads(first[1].read_bytes())["hash"] == hashlib.sha256(
            sources["answer.py"].encode(),
        ).hexdigest()
        (phases / "observed").touch()

        during_publish = set()
        deadline = time.monotonic() + scaled(40)
        done = [phases / f"done-{slot}" for slot in ("one", "two")]
        while not all(path.exists() for path in done):
            assert all(process.poll() is None for process in processes)
            assert time.monotonic() < deadline, "Cold cache writes did not finish"
            for entry in home.rglob("*.json"):
                during_publish.add(json.loads(entry.read_bytes())["hash"])
            time.sleep(0.001)
        entries = {json.loads(entry.read_bytes())["hash"] for entry in home.rglob("*.json")}
        assert entries == {hashlib.sha256(source.encode()).hexdigest() for source in sources.values()}
        assert during_publish <= entries
        assert not list(home.rglob("*.tmp"))
        (phases / "read").touch()

        results = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=scaled(40))
            assert process.returncode == 0, stderr
            results.append(json.loads(stdout))
        assert results[0]["pid"] != results[1]["pid"]
        for result in results:
            assert "answer.py" in result["cold_extracted"]
            assert result["cold"] == result["warm"] == ["answer.py"]
            assert result["warm_extracted"] == []
