"""Prove deferral and home inheritance through provider submissions from real children."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from _journal import measurement_events
from _stateframe_home import isolated_stateframe_home
from _timeouts import scaled
from test_workspace_state_oracle import scripted_provider
from tools.pinned_tree import pinned_tree

from stateframe import profiles

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
MINIMAL = "STATEFRAME_TURN_MINIMAL_LAUNCH"
INDEXED = "STATEFRAME_TURN_INDEXED_READS"
REQUEST_BUDGET = 8
TEST_PATH = "tests/test_pricing.py"
DRIVER = """
import sys
from pathlib import Path

from stateframe import cli, runner

runner.secrets.token_hex = lambda size=16: ("0123456789abcdef" * 4)[:2 * size]
probe = Path(sys.argv[1]).read_text(encoding="utf-8")
popen = runner.subprocess.Popen

def instrument(command, *args, **kwargs):
    if command[-1] == "stateframe._child":
        command = [*command[:3], probe + "\\n" + command[3], command[-1]]
    elif len(command) == 4 and str(command[2]).endswith("/_child.py"):
        child, run = command[2:]
        command = [*command[:2], "-c", probe + (
            "\\nimport runpy, sys\\n"
            f"sys.path.insert(0, {str(Path(child).parent)!r})\\n"
            f"sys.argv = [{child!r}, {run!r}]\\n"
            f"runpy.run_path({child!r}, run_name='__main__')"
        )]
    return popen(command, *args, **kwargs)

runner.subprocess.Popen = instrument
raise SystemExit(cli.main(sys.argv[2:]))
"""

PROBE = """
import atexit
import importlib.machinery
import json
import os
import sys
import time
from pathlib import Path

events = []
hold_until = config.get("hold_until")
demand_at = config.get("demand_at")
stop_at = config.get("stop_at")
wait_for_graph = config.get("wait_for_graph")
os.environ.update(CODE_GRAPH_LSP="off", EDIT_NTURN="off")

def wait_until(ready, seconds):
    deadline = time.monotonic() + seconds
    while not ready() and time.monotonic() < deadline:
        time.sleep(0.01)
    return ready()
block = config["fallback"]

class GraphProbe:
    def find_spec(self, fullname, path=None, target=None):
        if block and fullname.split(".")[0] in ("networkx", "tree_sitter", "tree_sitter_language_pack"):
            raise ImportError(fullname)
        if fullname not in ("mc_repo_intel_vendor.call_graph", "kernel_v46_mechanism_wiring"):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        load = spec.loader.exec_module

        def instrument(module):
            load(module)
            if fullname == "kernel_v46_mechanism_wiring":
                instrument_observations(module)
                return
            original = module._build_graph_uncached

            def build(*args, **kwargs):
                def observe(phase):
                    events.append([phase, time.monotonic_ns(), os.environ.get("STATEFRAME_HOME"),
                                   os.environ.get("STATEFRAME_TURN_INDEXED_READS")])

                first = not events
                observe("start")
                if first and hold_until:
                    # Release only once the chosen provider response is durable.
                    response = Path(f"wire/{hold_until}.response.json")
                    wait_until(response.exists, config["gate_seconds"])
                try:
                    return original(*args, **kwargs)
                finally:
                    observe("finish")

            module._build_graph_uncached = build

        spec.loader.exec_module = instrument
        return spec

sys.meta_path.insert(0, GraphProbe())

@atexit.register
def save_observations():
    Path("graph-observations.json").write_text(json.dumps(events), encoding="utf-8")

def instrument_observations(module):
    original = module.after_observation
    seeded = False
    waited = False

    def observe(runtime, action, observation):
        nonlocal seeded, waited
        if wait_for_graph and not waited and Path("wire/0001.response.json").exists():
            from mc_repo_intel_vendor import background
            root = Path(runtime.owner.root)
            assert wait_until(lambda: background.graph_snapshot(root)[0] is not None, 12)
            waited = True
        if demand_at and not seeded and Path(f"wire/{demand_at:04d}.response.json").exists():
            # Seed a check receipt at this exchange to demand the dependency
            # even on a repository with red preflight.
            from kernel_invalidation import runtime_invalidation_index
            ledger = runtime_invalidation_index(runtime)
            ledger.observe_receipt({"kind": "test", "result_class": "GREEN",
                                    "target_path": "tests/test_pricing.py"})
            seeded = True
        return original(runtime, action, observation)

    module.after_observation = observe

if stop_at or hold_until == "0002":
    def attach(frame, event, arg):
        if event != "call" or frame.f_code.co_name != "emit" or not frame.f_code.co_filename.endswith("/_child.py"):
            return
        sys.setprofile(None)
        original = frame.f_globals["emit"]

        def emit(kind, **fields):
            original(kind, **fields)
            if kind != "send":
                return
            send = fields["send"]
            if send == 2 and hold_until == "0002":
                assert wait_until(lambda: any(event[0] == "finish" for event in events), 25)
            if send == stop_at:
                save_observations()
                os._exit(91)

        frame.f_globals["emit"] = emit

    sys.setprofile(attach)
"""


def _run(tree, work, *, minimal="off", indexed=None, hold=False, fallback=False,
         sends=None, resume=False, gate_seconds=None):
    work.mkdir()
    repo = work / "repo"
    run = work / "run"
    home = Path(isolated_stateframe_home()) / work.parent.name / work.name
    shutil.copytree(EXAMPLES / "discount_bug", repo)
    steps = json.loads((EXAMPLES / "discount_bug.script.json").read_bytes())["steps"]
    if sends is not None:
        steps = steps[:sends]
    if resume:
        # A repository-wide short-name match exists outside the scoped imports.
        # Using scoped facts by accident during replay changes the tape.
        (repo / "shop" / "elsewhere.py").write_text("def raises(): pass\n", encoding="utf-8")
        steps.insert(1, {"tool": "read", "required": True, "arguments": {"path": TEST_PATH}})
    switches = profiles.load(profiles.DEFAULT)
    switches.update({MINIMAL: minimal, "STATEFRAME_TURN_REPO_INTEL": "off",
                     "STATEFRAME_PER_REQUEST_TIMEOUT_S": str(REQUEST_BUDGET)})
    if indexed is not None:
        switches[INDEXED] = indexed
    profile = work / "profile.json"
    profile.write_text(json.dumps({"schema": "stateframe-product-profile/1", "switches": switches}))
    probe = work / "probe.py"
    env = {name: value for name, value in os.environ.items() if not name.startswith("STATEFRAME_")}
    env.update(STATEFRAME_HOME=str(home), STATEFRAME_PROVIDER="llama.cpp", STATEFRAME_CHECKPOINT="off",
               PYTHONPATH=os.pathsep.join(str(tree / path) for path in ("src", "src/stateframe/engine")))
    config = {"fallback": fallback, "gate_seconds": gate_seconds or 35}
    if hold:
        config.update(hold_until="0002", demand_at=1)
    if resume:
        config.update(wait_for_graph=True, demand_at=2, stop_at=3)

    def launch(arguments):
        probe.write_text(f"config = {config!r}\n" + PROBE, encoding="utf-8")
        return subprocess.run([sys.executable, "-B", "-c", DRIVER, str(probe), *arguments],
                              cwd=tree, env=env, capture_output=True, text=True, check=False,
                              timeout=scaled(100))

    def diagnostic(result):
        log = run / "engine.log"
        return result.stdout + result.stderr + (log.read_text()[-6000:] if log.is_file() else "")

    with scripted_provider(steps) as (endpoint, trace):
        result = launch(["run", "--repo", str(repo), "--task", "Fix the pricing code so the tests pass.",
                         "--endpoint", endpoint, "--model", "scripted", "--run-dir", str(run), "--quiet",
                         "--test-python", sys.executable, "--no-run-env", "--profile", str(profile),
                         "--timeout", "85", "--max-turns", str(sends or 9)])
        if resume:
            assert result.returncode != 0 and len(trace.requests) == 3, diagnostic(result)
            config.update(wait_for_graph=False, stop_at=None, hold_until="0003")
            result = launch(["resume", str(run), "--endpoint", endpoint])
        assert len(trace.requests) == len(trace.responses) == len(steps), diagnostic(result)
        trace.script.assert_consumed()
        if sends is None:
            assert result.returncode == 0, diagnostic(result)
    events = json.loads((run / "graph-observations.json").read_bytes())
    submitted = [row for row in measurement_events(run) if row["kind"] == "request_submitted"]
    assert len(submitted) == len(steps)
    assert all(0 < row["socket_timeout_s"] <= REQUEST_BUDGET for row in submitted)
    assert all(0 < row["wall_timeout_s"] <= REQUEST_BUDGET for row in submitted)
    return trace, events, run, home


@pytest.mark.host_sandbox
@pytest.mark.parametrize("minimal,indexed", [("off", "off"), ("on", "on")])
@pytest.mark.parametrize("fallback", [False, True], ids=["installed", "stdlib"])
def test_following_send_uses_scoped_dependencies_while_full_graph_waits(
    tmp_path, minimal, indexed, fallback,
):
    trace, events, run, home = _run(ROOT, tmp_path / "current", minimal=minimal, indexed=indexed,
                                   hold=True, fallback=fallback, sends=2)
    starts = [event for event in events if event[0] == "start"]
    finishes = [event for event in events if event[0] == "finish"]
    first, second = (request[0] for request in trace.requests[:2])
    print(f"send 1 to send 2: {(second - first) / 1e9:.3f}s")
    assert starts and finishes
    assert (run / "wire/0001.request.json").read_bytes() == trace.requests[0][1]
    assert starts[0][1] < second < finishes[0][1]
    assert all(event[2:] == [str(home), indexed] for event in starts + finishes)

    first_tools = {tool["function"]["name"] for tool in json.loads(trace.requests[0][1])["tools"]}
    assert (first_tools == {"list_files", "read"}) == (minimal == "on")
    recorded = (run / "wire/0002.repo-dependencies.json").read_bytes()
    facts = json.loads(recorded)
    journal = measurement_events(run)
    visibility = next(row for row in journal if row["kind"] == "repository_dependencies_visible")
    assert visibility["send"] == 2 and visibility["sha256"] == hashlib.sha256(recorded).hexdigest()
    submissions = [row for row in journal if row["kind"] == "request_submitted"]
    assert submissions[1]["sequence"] < visibility["sequence"]
    assert any(row["path"] == TEST_PATH and row["dependencies"] == ["shop/pricing.py"]
               and row["source"] == "scoped" for row in facts)
    assert list((home / "repo-intel" / "python-call-facts").rglob("*.json"))
    assert not list((run / "worker").rglob("python-call-facts"))


@pytest.mark.host_sandbox
def test_first_provider_request_matches_wave_one(tmp_path, tmp_path_factory):
    base = pinned_tree("dc52265bc", tmp_path_factory)
    baseline, _, _, _ = _run(base, tmp_path / "baseline", sends=1)
    current, _, run, _ = _run(ROOT, tmp_path / "current", sends=1)

    first = current.requests[0][1]
    assert first == baseline.requests[0][1]
    assert (run / "wire/0001.request.json").read_bytes() == first


@pytest.mark.host_sandbox
def test_replay_reuses_the_facts_even_while_the_graph_is_stalled(tmp_path):
    trace, events, run, _ = _run(ROOT, tmp_path / "current", resume=True, fallback=True)
    archived = run / "attempts" / "1" / "wire"
    recorded = (archived / "0003.repo-dependencies.json").read_bytes()
    assert any(row["path"] == TEST_PATH and row["source"] == "graph"
               and "shop/elsewhere.py" in row["dependencies"] for row in json.loads(recorded))
    assert (run / "wire/0003.repo-dependencies.json").read_bytes() == recorded
    for send, (_, request) in enumerate(trace.requests[:3], 1):
        assert (run / "wire" / f"{send:04d}.request.json").read_bytes() == request
    assert [row.get("replayed", False) for row in
            map(json.loads, (run / "sends.jsonl").read_text().splitlines())] == [
                True, True, True, False, False,
            ]
    assert events[0][0] == "start"
