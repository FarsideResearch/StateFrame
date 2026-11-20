"""Trace import execution across worker startup and the first edit."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from _timeouts import scaled


ROOT = Path(__file__).resolve().parents[1]
SOURCE = "def price():\n    return 10\n"
IMPLEMENTATION = "kernel_edit_source_tool"
EDIT_GUARD = "kernel_destructive_edit_guard"


def _trace(root, pilot, protocol, broken):
    from importlib.machinery import PathFinder
    import tempfile
    from types import ModuleType

    # Exercise the shared switch interface even if the host change has yet to land.
    cache = ModuleType("stateframe.send1_probe_cache")
    cache.enabled = lambda: pilot == "on"
    cache.lookup = lambda *_args, **_kwargs: None
    cache.publish = lambda *_args, **_kwargs: False
    sys.modules[cache.__name__] = cache
    selector = ModuleType("stateframe.send1_wave4")
    selector.enabled = cache.enabled
    sys.modules[selector.__name__] = selector
    executions = []

    class EditorImports:
        def find_spec(self, fullname, path=None, target=None):
            if fullname not in {IMPLEMENTATION, EDIT_GUARD}:
                return None
            if fullname == IMPLEMENTATION and broken:
                raise ModuleNotFoundError("editor dependency missing", name="editor_dependency")
            spec = PathFinder.find_spec(fullname, path, target)
            execute = spec.loader.exec_module

            def execute_module(module):
                executions.append(fullname)
                execute(module)

            spec.loader.exec_module = execute_module
            return spec

    sys.meta_path.insert(0, EditorImports())
    from probes import react_repair
    from kernel_stateframe_runtime import StateFrameRuntime, _compact_unified_diff
    from stateframe.task_row import build_row
    from stateframe.workcopy import create

    repo = root / "repo"
    repo.mkdir()
    (repo / "pricing.py").write_text(SOURCE, encoding="utf-8")
    (repo / "test_pricing.py").write_text(
        "from pricing import price\n\ndef test_price():\n    assert price() == 20\n",
        encoding="utf-8",
    )
    row = build_row(create(repo, root / "copy"), task="Change the price to 20.",
                    target="pricing.py", task_id="pricing")
    react_repair.POC = root / "worker"
    react_repair.POC.mkdir()
    tempfile.tempdir = str(root)

    class MenuBuilt(BaseException):
        pass

    result = {}
    build_menu = StateFrameRuntime.tool_menu

    def capture(runtime, tools, messages):
        menu = build_menu(runtime, tools, messages)
        result["registered"] = [tool["function"]["name"] for tool in tools]
        result["tools"] = [tool["function"]["name"] for tool in menu]
        result["before_dispatch"] = list(executions)
        raise MenuBuilt

    StateFrameRuntime.tool_menu = capture
    try:
        react_repair.work(row)
    except ModuleNotFoundError as error:
        result["error"] = [type(error).__name__, error.name]
    except MenuBuilt:
        candidate = repo / "pricing.py"
        if protocol == "legacy_safe":
            result["edit"] = react_repair.str_replace_edit(candidate, "return 10", "return 20",
                                                          root=repo)
            result["source"] = candidate.read_text(encoding="utf-8")
        result["diff"] = _compact_unified_diff(SOURCE, SOURCE.replace("10", "20"), "pricing.py")
        result["after_dispatch"] = list(executions)
    print(json.dumps(result))


def _run(tmp_path, pilot, protocol, broken=False):
    root = tmp_path / pilot
    root.mkdir()
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("STATEFRAME_", "KERNEL_", "REACT_", "DS_", "PILOT_"))}
    env.update(
        STATEFRAME_HOME=str(root / "home"),
        STATEFRAME_RESULTS_DIR=str(root / "results"),
        STATEFRAME_EDIT_PROTOCOL=protocol,
        PYTHONPATH=os.pathsep.join(str(path) for path in (ROOT / "src",
                                                       ROOT / "src/stateframe/engine", ROOT / "tests")),
    )
    result = subprocess.run(
        [sys.executable, __file__, str(root), pilot, protocol, str(int(broken))],
        env=env, cwd=ROOT, text=True, capture_output=True, timeout=scaled(90), check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.splitlines()[-1])


def test_legacy_menu_skips_handles_editor_and_edit_loads_the_guard(tmp_path):
    eager = _run(tmp_path, "off", "legacy_safe")
    deferred = _run(tmp_path, "on", "legacy_safe")
    assert eager["before_dispatch"] == [IMPLEMENTATION, EDIT_GUARD]
    assert eager["after_dispatch"] == eager["before_dispatch"]
    assert deferred["before_dispatch"] == []
    assert deferred["after_dispatch"] == [EDIT_GUARD]
    assert eager["tools"] == deferred["tools"]
    assert eager["edit"] == deferred["edit"]
    assert eager["source"] == deferred["source"] == SOURCE.replace("10", "20")
    assert eager["diff"] == deferred["diff"]
    assert "-    return 10\n+    return 20" in deferred["diff"]


@pytest.mark.parametrize("broken", [False, True], ids=["registered", "import_failure"])
def test_handles_keep_startup_registration_and_import_failure(tmp_path, broken):
    eager = _run(tmp_path, "off", "handles", broken)
    deferred = _run(tmp_path, "on", "handles", broken)
    assert eager == deferred
    if broken:
        assert deferred == {"error": ["ModuleNotFoundError", "editor_dependency"]}
    else:
        assert deferred["before_dispatch"] == [IMPLEMENTATION, EDIT_GUARD]
        assert {"edit_source", "rewrite_source"}.issubset(deferred["registered"])


if __name__ == "__main__":
    _trace(Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4] == "1")
