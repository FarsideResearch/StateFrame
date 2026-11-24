"""Explicitly disabled allocator preserves bytes from before allocator support."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from tools.pinned_tree import pinned_tree
from _stateframe_home import isolated_stateframe_home


def _provider_hashes(tree, work, allocator="off"):
    driver = '''
import json, sys
from pathlib import Path
import pytest
from test_request_determinism import _run_scripted_example, EXAMPLES
with pytest.MonkeyPatch.context() as patch:
    if sys.argv[3] == "default":
        patch.delenv("STATEFRAME_WORKING_SET_ALLOCATOR", raising=False)
    else:
        patch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", sys.argv[3])
    patch.setenv("SF_LAB_DETFIX3", "on")
    from stateframe import profiles, runner
    load_profile = profiles.load

    def load_off_profile(name):
        switches = load_profile(name)
        for switch in ("STATEFRAME_WORKING_SET_STABLE_FIRST", "STATEFRAME_CONSTANT_TOOL_MENU"):
            if switch in switches:
                switches[switch] = "off"
        return switches

    patch.setattr(profiles, "load", load_off_profile)
    runner.test_interpreter.choose = lambda *args, **kwargs: (sys.executable, "parity test interpreter")
    hashes, _, _ = _run_scripted_example(patch, "discount_bug", EXAMPLES["discount_bug"], Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(json.dumps(hashes))
'''
    work.mkdir(parents=True, exist_ok=True)
    destination = work / "hashes.json"
    env = {key: value for key, value in os.environ.items() if not key.startswith("STATEFRAME_")}
    env["STATEFRAME_HOME"] = isolated_stateframe_home()
    env["PYTHONPATH"] = os.pathsep.join(str(tree / path) for path in
        ("src", "src/stateframe/engine", "tests"))
    result = subprocess.run([sys.executable, "-c", driver, str(work / "run"), str(destination), allocator],
        cwd=tree, env=env, capture_output=True, text=True, timeout=600, check=False)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    hashes = json.loads(destination.read_text())
    assert hashes
    return hashes


# Each worker sees one immutable checkout per test session. Reuse the measured
# trace across direct-call parity wrappers as well as parametrized comparisons.
_PROVIDER_HASH_CACHE: dict[Path, tuple[str, ...]] = {}


def _cached_provider_hashes(tree, tmp_path_factory):
    tree = tree.resolve()
    hashes = _PROVIDER_HASH_CACHE.get(tree)
    if hashes is None:
        hashes = tuple(_provider_hashes(tree, tmp_path_factory.mktemp("off-provider-parity")))
        _PROVIDER_HASH_CACHE[tree] = hashes
    return hashes


def test_off_provider_bytes_match_release_before_allocator(tmp_path, tmp_path_factory):
    base = pinned_tree("56af0dc6d6c7b632808070e137b9d95f66db882e", tmp_path_factory)
    current = Path(__file__).resolve().parents[1]
    assert _cached_provider_hashes(base, tmp_path_factory) == _cached_provider_hashes(
        current, tmp_path_factory)


def test_off_parity_direct_callers_share_trace_results(tmp_path, monkeypatch):
    from types import SimpleNamespace
    module = sys.modules[__name__]
    calls = []
    reference = tmp_path / "reference"
    current = Path(__file__).resolve().parents[1]

    def hashes(tree, work):
        calls.append(tree)
        return ["trace-sha256"]

    monkeypatch.setattr(module, "_provider_hashes", hashes)
    monkeypatch.setattr(module, "pinned_tree", lambda *_: reference)
    monkeypatch.setattr(module, "_PROVIDER_HASH_CACHE", {}, raising=False)
    factory = SimpleNamespace(mktemp=lambda _: tmp_path)
    for _ in range(2):
        test_off_provider_bytes_match_release_before_allocator(tmp_path, factory)
    assert calls == [reference, current]
