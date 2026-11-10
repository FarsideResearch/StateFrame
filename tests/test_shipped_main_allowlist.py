"""Every executable Python module in release artifacts needs an explicit review."""

from __future__ import annotations

import ast
from pathlib import Path
import sys
import tarfile
import zipfile


# Paths are relative to the wheel root or the sdist's project root. A main
# block is allowed only for the purpose recorded here; batch drivers live in
# maintainer task driver and are deliberately absent from both release artifacts.
ALLOWLIST = {
    "scripts/build_wheelhouse.py": "sdist installer dependency build tool",
    "stateframe/_child.py": "isolated engine child process",
    "stateframe/cli.py": "registered stateframe command",
    "stateframe/engine/blame_channel.py": "diagnostic command",
    "stateframe/engine/capability_release_runner.py": "release capability runner",
    "stateframe/engine/chains.py": "seam manifest diagnostic",
    "stateframe/engine/detreason/diagnosis.py": "operator diagnosis command",
    "stateframe/engine/detreason/plan_synth.py": "operator plan diagnostic",
    "stateframe/engine/detreason/priors.py": "operator prior diagnostic",
    "stateframe/engine/detreason/skeptic.py": "manual audit driver",
    "stateframe/engine/experiment_protocol.py": "experiment protocol command",
    "stateframe/engine/fixer.py": "direct fixer command",
    "stateframe/engine/kernel_bolt.py": "kernel self-test",
    "stateframe/engine/kernel_bounded_http.py": "HTTP and probe child process",
    "stateframe/engine/kernel_exec_supervisor.py": "execution supervisor child process",
    "stateframe/engine/kernel_hook_driver.py": "sandbox hook child process",
    "stateframe/engine/kernel_liveness.py": "liveness diagnostic",
    "stateframe/engine/kernel_release_switch_registry.py": "release registry maintenance command",
    "stateframe/engine/kernel_stateframe_thinking_router.py": "operator routing command",
    "stateframe/engine/probes/apply_corpus_fixes.py": "explicit corpus repair command",
    "stateframe/engine/prompt_ledger.py": "ledger diagnostic command",
    "stateframe/engine/protected_tree_guard.py": "protected tree check command",
    "stateframe/engine/sf_launch.py": "sandbox launch child process",
    "stateframe/engine/sidekick.py": "sidekick self-test",
    "stateframe/engine/symbol_registry.py": "symbol registry command",
    "stateframe/outer_checks.py": "outer check helper process",
    "stateframe/scripted_endpoint.py": "offline example endpoint command",
}


def _is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    test = node.test
    return (
        len(test.ops) == len(test.comparators) == 1
        and isinstance(test.ops[0], ast.Eq)
        and (
            isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "__main__"
        )
    )


def unlisted_mains(sources: dict[str, bytes]) -> list[str]:
    violations = []
    for name, source in sorted(sources.items()):
        tree = ast.parse(source, filename=name)
        if any(_is_main_guard(node) for node in ast.walk(tree)) and name not in ALLOWLIST:
            violations.append(name)
    return violations


def _artifact_sources(wheel: Path, sdist: Path) -> tuple[dict[str, bytes], dict[str, bytes]]:
    with zipfile.ZipFile(wheel) as archive:
        wheel_sources = {
            name: archive.read(name)
            for name in archive.namelist()
            if name.endswith(".py")
        }
    with tarfile.open(sdist, "r:gz") as archive:
        root = archive.getnames()[0].split("/", 1)[0] + "/"
        sdist_sources = {
            name.removeprefix(root).removeprefix("src/"): archive.extractfile(name).read()
            for name in archive.getnames()
            if name.endswith(".py")
        }
    return wheel_sources, sdist_sources


def test_shipped_main_allowlist(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "_build"))
    import sf_backend

    wheel = tmp_path / sf_backend.build_wheel(str(tmp_path))
    sdist = tmp_path / sf_backend.build_sdist(str(tmp_path))
    wheel_sources, sdist_sources = _artifact_sources(wheel, sdist)
    assert wheel_sources and sdist_sources
    wheel_violations = unlisted_mains(wheel_sources)
    sdist_violations = unlisted_mains(sdist_sources)
    assert not wheel_violations, f"wheel has unreviewed main blocks: {wheel_violations}"
    assert not sdist_violations, f"sdist has unreviewed main blocks: {sdist_violations}"
