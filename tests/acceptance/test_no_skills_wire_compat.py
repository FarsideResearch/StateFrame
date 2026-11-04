"""Requests without selected skills retain the 0.2.3 wire contract."""
from __future__ import annotations

import io
import json
import site
import subprocess
import sysconfig
import tarfile
from pathlib import Path

import pytest

from stateframe.replay_tape import first_difference, normalized_digest
from ._support import (Endpoint, PYTHON, ROOT, assert_tree_imports, budget, copy_repo,
                       environment, pass_environment, tree_python)


pytestmark = pytest.mark.host_sandbox

_BASELINE_COMMIT = "33c6c511611cc249b90b47930905a8e60648c05c"



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
              legacy_mode: bool = False,
              stable_tool_head: str | None) -> tuple[bytes, ...]:
    case.mkdir()
    tree_src = root / "src"
    python = tree_python(tree_src, case / "python")
    assert_tree_imports(python, tree_src, case)

    run_dir = case / "run"
    env = environment()
    env.pop("PYTHONPATH", None)
    env.pop("STATEFRAME_STABLE_TOOL_HEAD", None)
    if stable_tool_head is not None:
        env["STATEFRAME_STABLE_TOOL_HEAD"] = stable_tool_head
    with Endpoint(script) as endpoint:
        result = subprocess.run(
            [python, "-m", "stateframe", "run", "--repo", str(repo),
             "--task", "Change app value and verify it", "--endpoint", endpoint.url,
             "--model", "scripted", "--run-dir", str(run_dir), "--test-python", PYTHON,
             "--test-cmd", "python -m unittest discover -s tests -q", "--max-turns", "8",
             "--shell", "off", "--quiet",
             *(("--runner-kind", "pytest", "--protect-tests", "all") if legacy_mode else ())],
            capture_output=True, text=True,
            env=env, timeout=budget("no_skills_run"), check=False)
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


def test_no_skills_wire_matches_023(tmp_path):
    archive = subprocess.run(
        ["git", "-C", str(ROOT), "archive", _BASELINE_COMMIT, "src"],
        capture_output=True, check=True)
    baseline = tmp_path / "baseline-tree"
    baseline.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive.stdout), mode="r:") as contents:
        for member in contents:
            target = baseline / member.name
            assert target.resolve().is_relative_to(baseline) and not member.issym() and not member.islnk()
            contents.extract(member, baseline)
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
    old = _run_tree(baseline, tmp_path / "old", repo, script, stable_tool_head=None)
    # 0.2.5 intentionally treats unittest as a generic command and defaults to
    # editable tests. Pin the old runner, protection and tool-list modes for exact bytes.
    current = _run_tree(ROOT, tmp_path / "current", repo, script, legacy_mode=True,
                        stable_tool_head="off")
    from tools.compare_no_mcp_requests import mask_wire_ids, unbacked_result_hash_changes

    assert len(current) == len(old)
    # Wire IDs (edit receipts, wire-truth handles, request IDs) are derived differently
    # since the receipt-determinism change (11-03), so they are masked; a masked result
    # hash must still hash text that is sent and compared. Every other byte must match.
    assert unbacked_result_hash_changes(tmp_path / "old", tmp_path / "current") == []
    assert mask_wire_ids(current[0]) == mask_wire_ids(old[0]), first_difference(old[0], current[0])
    for number, (expected, actual) in enumerate(zip(old, current), 1):
        assert _raw_tools(actual) == _raw_tools(expected), number
        assert mask_wire_ids(actual) == mask_wire_ids(expected), (
            number, first_difference(expected, actual))
        assert normalized_digest(mask_wire_ids(actual)) == normalized_digest(
            mask_wire_ids(expected)), number
    default = _run_tree(ROOT, tmp_path / "default", repo, script, stable_tool_head=None)  # today's defaults, on purpose
    assert default[0] != old[0]
    assert b"no public selector was declared" in default[0]
