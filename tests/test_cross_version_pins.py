"""Cross-version request-byte comparisons must state every switch whose default has changed.

0.3.13 turned the stable tool head on by default, and three comparisons broke because their newer
side silently ran the new default. Each comparison helper takes the switch as a keyword; every call
in a cross-version test must pass it explicitly, so a future default change is a one-line list
update instead of a hunt.
"""
from __future__ import annotations

import ast
from pathlib import Path

from tools.changed_defaults import CHANGED_DEFAULTS

TESTS = Path(__file__).resolve().parent
# Text that marks a file as running an older release tree next to the current one.
CROSS_VERSION_MARKERS = (
    "SF_TEST_PARITY_BASELINE",
    '"archive"',
    "compare_no_mcp_requests",
    "_archive_revision",
    "pinned_tree",
)
# Files that run an older tree but compare no request bytes, with the reason.
EXEMPT = {
    "acceptance/test_doc_command_sweep.py": "positive control: asserts the old release fails; no byte comparison",
}
KEYWORDS = {change.keyword for change in CHANGED_DEFAULTS.values()}


def _cross_version_files() -> list[Path]:
    return sorted(path for path in TESTS.rglob("*.py")
                  if path.name != Path(__file__).name
                  and path.relative_to(TESTS).as_posix() not in EXEMPT
                  and any(marker in path.read_text(encoding="utf-8") for marker in CROSS_VERSION_MARKERS))


def _runner_names(trees: list[ast.AST]) -> set[str]:
    """Functions that take a changed-default keyword: the comparison helpers."""
    names = set()
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                arguments = node.args.args + node.args.kwonlyargs
                if KEYWORDS & {argument.arg for argument in arguments}:
                    names.add(node.name)
    return names


def missing_pins(source: str, runners: set[str]) -> list[str]:
    """Calls to a comparison helper that omit a changed-default keyword."""
    problems = []
    for node in ast.walk(ast.parse(source)):
        # Helpers are called by bare name; attribute calls such as store.capture() are other APIs.
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        name = node.func.id
        if name not in runners:
            continue
        passed = {keyword.arg for keyword in node.keywords}
        for keyword in sorted(KEYWORDS - passed):
            problems.append(f"line {node.lineno}: {name}() does not state {keyword}")
    return problems


def test_changed_defaults_are_well_formed():
    assert CHANGED_DEFAULTS
    for name, change in CHANGED_DEFAULTS.items():
        assert name.startswith("STATEFRAME_") and change.pin and change.keyword.isidentifier()


def test_every_cross_version_comparison_states_changed_defaults():
    files = _cross_version_files()
    tool = TESTS / "tools/compare_no_mcp_requests.py"
    trees = [ast.parse(path.read_text(encoding="utf-8")) for path in (*files, tool)]
    runners = _runner_names(trees)
    assert {"capture", "_run_checkout", "_resume_checkout"} <= runners
    problems = [f"{path.relative_to(TESTS)} {problem}"
                for path in (*files, tool)
                for problem in missing_pins(path.read_text(encoding="utf-8"), runners)]
    assert problems == []


def test_planted_omission_is_caught():
    planted = (
        "def test_old_vs_new(tmp_path):\n"
        "    before = capture(baseline, python, launcher, script, repo, run_dir, stable_tool_head=None)\n"
        "    after = capture(ROOT, python, launcher, script, repo, run_dir)\n"
        "    assert before == after\n"
    )
    assert missing_pins(planted, {"capture"}) == ["line 3: capture() does not state stable_tool_head"]
