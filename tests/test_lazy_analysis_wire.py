"""Prove deferral and home inheritance through provider submissions from real children."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from _timeouts import scaled
from _stateframe_home import isolated_stateframe_home
from test_workspace_state_oracle import scripted_provider
from tools.pinned_tree import pinned_tree

from stateframe import profiles

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
MINIMAL = "STATEFRAME_TURN_MINIMAL_LAUNCH"
INDEXED = "STATEFRAME_TURN_INDEXED_READS"
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
import json
import os
import sys
import time
from pathlib import Path

events = []

def observe(code):
    if code.co_filename.endswith("/call_graph.py") and code.co_name in (
        "_build_graph_uncached", "_finalize",
    ):
        events.append([code.co_name, time.monotonic_ns(),
                       os.environ.get("STATEFRAME_HOME"),
                       os.environ.get("STATEFRAME_TURN_INDEXED_READS")])

@atexit.register
def save_observations():
    if hasattr(sys, "monitoring"):
        sys.monitoring.free_tool_id(5)
    else:
        sys.setprofile(None)
    Path("graph-observations.json").write_text(json.dumps(events), encoding="utf-8")

if hasattr(sys, "monitoring"):
    monitor = sys.monitoring
    monitor.use_tool_id(5, "dependency build observations")
    monitor.register_callback(5, monitor.events.PY_START, lambda code, offset: observe(code))
    monitor.set_events(5, monitor.events.PY_START)
else:
    def profile(frame, event, arg):
        if event == "call":
            observe(frame.f_code)

    sys.setprofile(profile)
"""


def _run(tree, work, *, minimal="off", indexed=None, home=None):
    work.mkdir()
    repo = work / "repo"
    run = work / "run"
    isolated_home = Path(isolated_stateframe_home())
    home = Path(home) if home is not None else isolated_home / work.parent.name / work.name
    shutil.copytree(EXAMPLES / "discount_bug", repo)
    steps = json.loads((EXAMPLES / "discount_bug.script.json").read_bytes())["steps"]
    switches = profiles.load(profiles.DEFAULT)
    switches.update({MINIMAL: minimal, "STATEFRAME_TURN_REPO_INTEL": "off"})
    if indexed is not None:
        switches[INDEXED] = indexed
    profile = work / "profile.json"
    profile.write_text(json.dumps({"schema": "stateframe-product-profile/1", "switches": switches}))
    probe = work / "probe.py"
    probe.write_text(PROBE, encoding="utf-8")
    env = {name: value for name, value in os.environ.items() if not name.startswith("STATEFRAME_")}
    env.update(STATEFRAME_HOME=str(home), STATEFRAME_PROVIDER="llama.cpp", STATEFRAME_CHECKPOINT="off",
               PYTHONPATH=os.pathsep.join(str(tree / path) for path in ("src", "src/stateframe/engine")))
    with scripted_provider(steps) as (endpoint, trace):
        result = subprocess.run(
            [sys.executable, "-B", "-c", DRIVER, str(probe), "run", "--repo", str(repo),
             "--task", "Fix the pricing code so the tests pass.", "--endpoint", endpoint,
             "--model", "scripted", "--run-dir", str(run), "--quiet", "--test-python",
             sys.executable, "--no-run-env", "--profile", str(profile), "--timeout", "240",
             "--max-turns", "8"],
            cwd=tree, env=env, capture_output=True, text=True, check=False, timeout=scaled(270),
        )
    log = (run / "engine.log").read_text(encoding="utf-8") if (run / "engine.log").exists() else ""
    assert len(trace.requests) == len(trace.responses) == len(steps), (
        result.stdout + result.stderr + log[-6000:]
    )
    trace.script.assert_consumed()
    assert result.returncode == 0, result.stdout + result.stderr + log[-3000:]
    events = json.loads((run / "graph-observations.json").read_bytes())
    return trace, events, run, home


def _assert_later_dependency_build(trace, events, home, indexed):
    builds = [event for event in events if event[0] == "_build_graph_uncached"]
    finalizers = [event for event in events if event[0] == "_finalize"]
    assert builds and finalizers
    # The edit may trigger an automatic check before the explicit test tool runs.
    assert trace.requests[1][0] < builds[0][1] <= finalizers[0][1] < trace.requests[3][0]
    assert all(event[2:] == [str(home), indexed] for event in builds + finalizers)


@pytest.mark.host_sandbox
@pytest.mark.parametrize("minimal", ["off", "on"])
@pytest.mark.parametrize("indexed", ["off", "on"])
def test_child_uses_installation_facts_cache_under_shipped_profile(
    tmp_path, minimal, indexed,
):
    trace, events, run, home = _run(ROOT, tmp_path / "current", minimal=minimal, indexed=indexed)
    _assert_later_dependency_build(trace, events, home, indexed)
    first_tools = {tool["function"]["name"] for tool in json.loads(trace.requests[0][1])["tools"]}
    assert (first_tools == {"list_files", "read"}) == (minimal == "on")
    facts = list((home / "repo-intel" / "python-call-facts").rglob("*.json"))
    assert facts
    assert any(symbol["file"] == "tests/test_pricing.py"
               for entry in facts for symbol in json.loads(entry.read_bytes())["extract"]["defs"])
    assert not list((run / "worker").rglob("python-call-facts"))


@pytest.mark.host_sandbox
def test_late_dependency_build_preserves_eager_provider_bytes(tmp_path, tmp_path_factory):
    base = pinned_tree("dfc7503df", tmp_path_factory)
    eager, eager_events, _, _ = _run(base, tmp_path / "base")
    base_builds = [event for event in eager_events if event[0] == "_build_graph_uncached"]
    assert base_builds and base_builds[0][1] < eager.requests[0][0]

    lazy, events, _, home = _run(ROOT, tmp_path / "current")
    _assert_later_dependency_build(lazy, events, home, profiles.load(profiles.DEFAULT).get(INDEXED))
    assert [body for _, body in lazy.requests] == [body for _, body in eager.requests]

    warm, warm_events, _, _ = _run(ROOT, tmp_path / "warm", home=home)
    _assert_later_dependency_build(warm, warm_events, home, profiles.load(profiles.DEFAULT).get(INDEXED))
    assert [body for _, body in warm.requests] == [body for _, body in eager.requests]


@pytest.mark.host_sandbox
def test_first_provider_request_matches_wave_one(tmp_path, tmp_path_factory):
    base = pinned_tree("dc52265bc", tmp_path_factory)
    baseline, _, _, _ = _run(base, tmp_path / "baseline")
    current, _, run, _ = _run(ROOT, tmp_path / "current")

    first = current.requests[0][1]
    assert first == baseline.requests[0][1]
    assert (run / "wire/0001.request.json").read_bytes() == first
