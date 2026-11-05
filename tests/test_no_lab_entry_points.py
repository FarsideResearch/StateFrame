"""Frozen release guard: corpus drivers must not ship in wheel or sdist."""

from __future__ import annotations

import ast
from pathlib import Path
import sys
import tarfile
import zipfile


LAB_ONLY = {
    "engine/probes/real_repo_baseline.py": {
        "ask", "prose_spans", "mutation_col", "is_prose_mutation",
        "harvest_tasks", "tokenize", "narrow", "menu_pick", "run_task",
    },
    "engine/probes/real_repo_baseline_v2.py": {
        "ask", "cascade", "_Norm", "norm_dump", "menu_pick", "run_task",
    },
    "engine/probes/real_repo_baseline_v3.py": {"run_task"},
    "engine/probes/stateframe_shadow.py": {
        "load_transcript", "_is_trim_marker", "reconstruct_messages",
        "iter_turns", "attach_kernel_events", "seal_frame",
        "append_idempotent", "project_file", "main",
    },
    "engine/func_index.py": {
        "class_pattern_rank_files", "file_cascade_pool", "shadow_compare",
    },
}

# These are the only public CLI entry modules, and neither runs a lab corpus.
ALLOWED_CLI_MAINS = {"cli.py", "__main__.py"}
LAB_CALLS = {"harvest_tasks", "shadow_compare", "project_file"}


def _is_main(node: ast.AST) -> bool:
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    test = node.test
    return (isinstance(test.left, ast.Name) and test.left.id == "__name__"
            and any(isinstance(part, ast.Constant) and part.value == "__main__"
                    for part in test.comparators))


def _check_sources(sources: dict[str, bytes], artifact: str) -> None:
    assert sources, f"{artifact}: no shipped Python modules found"
    seen: set[str] = set()
    for name, content in sources.items():
        rel = name.split("stateframe/", 1)[1]
        tree = ast.parse(content.decode("utf-8"), filename=name)
        if rel in LAB_ONLY:
            seen.add(rel)
            present = {node.name for node in tree.body
                       if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
            assert not (present & LAB_ONLY[rel]), (artifact, rel, sorted(present & LAB_ONLY[rel]))
        for node in tree.body:
            if not _is_main(node):
                continue
            if rel in LAB_ONLY:
                raise AssertionError((artifact, rel, "lab module retains __main__"))
            if rel in ALLOWED_CLI_MAINS:
                continue
            calls = {call.func.id for call in ast.walk(node)
                     if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)}
            assert not (calls & LAB_CALLS), (artifact, rel, sorted(calls & LAB_CALLS))
    assert seen == LAB_ONLY.keys(), (artifact, "target modules missing", sorted(LAB_ONLY.keys() - seen))


def test_built_artifacts_have_no_lab_entry_points(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "_build"))
    import sf_backend

    wheel = tmp_path / sf_backend.build_wheel(str(tmp_path))
    sdist = tmp_path / sf_backend.build_sdist(str(tmp_path))
    with zipfile.ZipFile(wheel) as archive:
        wheel_sources = {name: archive.read(name) for name in archive.namelist()
                         if name.startswith("stateframe/") and name.endswith(".py")}
    with tarfile.open(sdist, "r:gz") as archive:
        sdist_sources = {name: archive.extractfile(name).read()
                         for name in archive.getnames()
                         if "/src/stateframe/" in name and name.endswith(".py")}
    _check_sources(wheel_sources, "wheel")
    _check_sources(sdist_sources, "sdist")
