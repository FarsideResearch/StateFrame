"""Requests without selected skills retain the immediate base wire contract."""
from __future__ import annotations

import json
import os
import site
import subprocess
import sysconfig
from pathlib import Path

import pytest

from stateframe.replay_tape import first_difference
from ._support import (Endpoint, PYTHON, ROOT, assert_tree_imports, budget, copy_repo,
                       environment, pass_environment, tree_python)
from engine_run import run_monitored

pytest_plugins = ("tools.immediate_baseline",)


pytestmark = pytest.mark.host_sandbox

_MISSING_BASELINE = ("SF_TEST_BASELINE_IMMEDIATE must name the immediate base checkout "
                     "for no-skills parity")



def _raw_tools(request: bytes) -> bytes:
    """Extract the top-level tools value without changing its serialized bytes."""
    source = request.decode("utf-8")
    decoder = json.JSONDecoder()

    def skip_space(offset: int) -> int:
        while source[offset].isspace():
            offset += 1
        return offset

    offset = 1
    assert source.startswith("{")
    while True:
        offset = skip_space(offset)
        if source[offset] == "}":
            break
        key, offset = decoder.raw_decode(source, offset)
        assert isinstance(key, str)
        offset = skip_space(offset)
        assert source[offset] == ":"
        offset += 1
        offset = skip_space(offset)
        start = offset
        value, offset = decoder.raw_decode(source, offset)
        if key == "tools":
            assert isinstance(value, list)
            return source[start:offset].encode("utf-8")
        offset = skip_space(offset)
        if source[offset] == ",":
            offset += 1
    raise AssertionError("request has no tools array")


def _run_tree(root: Path, case: Path, repo: Path, script: Path, *,
              trace: Path) -> tuple[bytes, ...]:
    case.mkdir()
    tree_src = root / "src"
    python = tree_python(tree_src, case / "python")
    assert_tree_imports(python, tree_src, case)

    run_dir = case / "run"
    env = environment()
    env.pop("PYTHONPATH", None)
    env.pop("STATEFRAME_STABLE_TOOL_HEAD", None)
    env.pop("STATEFRAME_FINAL_CACHE_LAYOUT", None)
    launcher = case / "traced-run.py"
    launcher.write_text(
        "import json, runpy, secrets, sys\n"
        "_sequence = 0\n"
        "def fixed_hex(size=32):\n"
        "    global _sequence\n"
        "    _sequence += 1\n"
        "    return format(_sequence, 'x').zfill(size * 2)[-size * 2:]\n"
        "secrets.token_hex = fixed_hex\n"
        f"_trace = {str(trace)!r}\n"
        "try:\n"
        "    runpy.run_module('stateframe', run_name='__main__')\n"
        "finally:\n"
        "    with open(_trace, 'w', encoding='utf-8') as target:\n"
        "        json.dump(sorted(sys.modules), target)\n",
        encoding="utf-8",
    )
    with Endpoint(script) as endpoint:
        result = run_monitored(
            [python, str(launcher), "run", "--repo", str(repo),
             "--task", "Change app value and verify it", "--endpoint", endpoint.url,
             "--model", "scripted", "--run-dir", str(run_dir), "--test-python", PYTHON,
             "--test-cmd", "python -m unittest discover -s tests -q", "--max-turns", "8",
             "--shell", "off", "--quiet"],
            capture_output=True, text=True,
            env=env, expected_seconds=budget("no_skills_run"), check=False)
    requests = tuple(path.read_bytes() for path in sorted((run_dir / "wire").glob("*.request.json")))
    assert result.returncode == 0 and requests, result.stdout + result.stderr
    return requests


def test_tree_python_ignores_editable_pth_from_outer_site_packages(tmp_path, monkeypatch):
    def dummy_tree(name: str, marker: str) -> Path:
        source = tmp_path / name / "src"
        package = source / "stateframe"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (package / "workcopy.py").write_text(f"MARKER = {marker!r}\n")
        return source

    first_src = dummy_tree("first", "first-tree")
    second_src = dummy_tree("second", "second-tree")
    outer_site = tmp_path / "outer-site-packages"
    outer_site.mkdir()
    processed = tmp_path / "editable-pth-was-processed"
    (outer_site / "editable-other-tree.pth").write_text(
        f"import sys; sys.path.insert(0, {str(second_src)!r}); open({str(processed)!r}, 'w').close()\n")

    original_paths = sysconfig.get_paths()
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(outer_site)])
    monkeypatch.setattr(sysconfig, "get_paths", lambda: {
        **original_paths, "purelib": str(outer_site), "platlib": str(outer_site)})

    def assert_tree_import(tree_src: Path, work: Path, marker: str) -> None:
        python = tree_python(tree_src, work)
        result = subprocess.run(
            [python, "-I", "-B", "-c",
             "import pytest\nimport stateframe.workcopy\n"
             "print(pytest.__file__)\nprint(stateframe.workcopy.MARKER)\n"
             "print(stateframe.workcopy.__file__)"],
            capture_output=True, text=True, cwd=tmp_path, env=pass_environment(), check=True)
        pytest_file, output_marker, imported_file = result.stdout.splitlines()
        assert Path(pytest_file).resolve().is_relative_to(Path(python).parent.parent.resolve())
        assert output_marker == marker
        assert Path(imported_file).resolve().is_relative_to(tree_src.resolve())

    assert_tree_import(first_src, tmp_path / "first-work", "first-tree")
    assert_tree_import(second_src, tmp_path / "second-work", "second-tree")
    assert not processed.exists()


@pytest.mark.skipif(
    "SF_TEST_BASELINE_IMMEDIATE" not in os.environ
    and os.environ.get("SF_GATE_MUST_RUN", "").strip().lower() not in {"1", "true", "yes", "on"},
    reason=_MISSING_BASELINE,
)
def test_no_skills_wire_matches_immediate_base(tmp_path, baseline_checkout, must_run_witness):
    baseline = baseline_checkout
    script = tmp_path / "case.script.json"
    script.write_text(json.dumps({"model": "scripted", "tokenize": {"mode": "ok"}, "steps": [
        {"tool": "str_replace", "arguments": {"path": "app.py", "old_str": 'VALUE = "before"',
                                               "new_str": 'VALUE = "after"'}},
        {"tool": "run_tests", "arguments": {"target": "tests/test_app.py"}},
        {"tool": "done", "arguments": {"summary": "Changed the app value."}},
    ]}))
    repo_root = tmp_path / "shared"
    repo_root.mkdir()
    repo = copy_repo(repo_root)
    old_trace = tmp_path / "old-modules.json"
    current_trace = tmp_path / "current-modules.json"
    old = _run_tree(baseline, tmp_path / "old", repo, script, trace=old_trace)
    current = _run_tree(ROOT, tmp_path / "current", repo, script, trace=current_trace)
    from tools.compare_no_mcp_requests import unbacked_result_hash_changes
    assert len(current) == len(old)
    assert unbacked_result_hash_changes(tmp_path / "old", tmp_path / "current") == []
    for number, (expected, actual) in enumerate(zip(old, current), 1):
        assert _raw_tools(actual) == _raw_tools(expected), number
        assert actual == expected, (number, first_difference(expected, actual))
    must_run_witness.record_parity(
        baseline_trace=json.loads(old_trace.read_text(encoding="utf-8")),
        candidate_trace=json.loads(current_trace.read_text(encoding="utf-8")),
        expected_trace_prefixes=("stateframe",),
        comparison_completed=True,
    )
