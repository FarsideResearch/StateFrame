"""Protocol edit-tool parity for kernel_stateframe_runtime.py (chunk K1).

Each literal listed in MIGRATED classifies what the model already did: an applied edit,
an edit call being recognised, or repair/budget accounting. Those literals must count
the protocol edit tools (edit_source, rewrite_source) alongside every legacy editor they
named before, otherwise a run on the protocol menu silently misses the edit.

Line numbers pin the literals in kernel_stateframe_runtime.py on the 0.7.79
release base.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from stateframe.engine import kernel_edit_tool_names as names

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "src" / "stateframe" / "engine" / "kernel_stateframe_runtime.py"
PROTOCOL_SPLICE = "PROTOCOL_EDIT_TOOL_NAMES"

# (line, legacy members the literal named before the protocol splice was added)
MIGRATED: tuple[tuple[int, tuple[str, ...]], ...] = (
    (1930, ("edit", "file_create", "str_replace")),
    (6705, ("create_file", "delete_file", "edit", "edit_file", "file_create", "rename_file", "replace", "replace_module", "shell", "str_replace", "write")),
    (9553, ("edit", "file_create", "str_replace")),
    (12020, ("edit", "file_create", "file_delete", "file_move", "replace", "replace_module", "shell", "str_replace", "write")),
    (13291, ("edit", "file_create", "file_delete", "file_move", "replace", "replace_module", "shell", "shell_effect", "str_replace", "write")),
    (14322, ("edit", "file_create", "file_delete", "file_move", "replace", "replace_module", "shell", "str_replace", "write")),
    (14539, ("edit", "file_create", "file_delete", "file_move", "replace", "replace_module", "shell", "str_replace", "write")),
    (14593, ("edit", "file_create", "replace", "replace_module", "shell", "str_replace", "write")),
    (14645, ("edit", "file_create", "str_replace")),
    (14658, ("edit", "file_create", "replace_module", "str_replace")),
    (14660, ("edit", "file_create", "replace", "replace_module", "str_replace", "write")),
    (14663, ("edit", "file_create", "replace", "replace_module", "str_replace", "write")),
    (14695, ("done", "edit", "file_create", "replace", "replace_module", "shell", "shell_effect", "str_replace", "write")),
    (14843, ("edit", "file_create", "file_delete", "file_move", "replace", "replace_module", "shell", "shell_effect", "str_replace", "write")),
    (15023, ("edit", "file_create", "replace", "replace_module", "shell_effect", "str_replace", "write")),
    (15097, ("edit", "file_create", "replace", "replace_module", "shell_effect", "str_replace", "write")),
    (15132, ("edit", "file_create", "replace", "replace_module", "shell_effect", "str_replace", "write")),
    (15152, ("edit", "file_create", "replace", "replace_module", "shell_effect", "str_replace", "write")),
    (15185, ("edit", "file_create", "replace_module", "shell_effect", "str_replace")),
    (15211, ("edit", "file_create", "replace", "replace_module", "shell_effect", "str_replace", "write")),
    (15243, ("edit", "file_create", "replace", "replace_module", "shell_effect", "str_replace", "write")),
    (15249, ("edit", "file_create", "replace", "replace_module", "shell_effect", "str_replace", "write")),
    (15261, ("edit", "file_create", "replace", "replace_module", "shell_effect", "str_replace", "write")),
    (15281, ("edit", "file_create", "str_replace")),
    (15305, ("edit", "file_create", "replace", "replace_module", "shell_effect", "str_replace", "write")),
    (15372, ("edit", "replace", "replace_module", "str_replace", "write")),
    (15500, ("edit", "replace", "replace_module", "str_replace", "write")),
    (15584, ("edit", "replace_module", "str_replace")),
    (15726, ("edit", "file_create", "file_delete", "file_move", "replace", "replace_module", "shell", "shell_effect", "str_replace", "write")),
    (15922, ("edit", "file_create", "str_replace")),
    (16511, ("edit", "file_create", "str_replace")),
    (16653, ("edit", "file_create", "str_replace")),
    (18610, ("edit", "file_create", "replace_module", "shell_effect", "str_replace")),
    (21051, ("edit", "str_replace")),
)


def _literal_at(line: int) -> tuple[frozenset[str], bool]:
    """Return (literal string members, whether the literal splices the protocol constant)."""
    tree = ast.parse(RUNTIME.read_text(encoding="utf-8"), filename=str(RUNTIME))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Set, ast.Tuple, ast.List)) or node.lineno != line:
            continue
        strings = frozenset(
            elt.value for elt in node.elts
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str))
        spliced = any(
            isinstance(elt, ast.Starred) and isinstance(elt.value, ast.Name)
            and elt.value.id == PROTOCOL_SPLICE
            for elt in node.elts)
        return strings, spliced
    raise AssertionError(f"no set/tuple/list literal at {RUNTIME.name}:{line}")


def test_migrated_table_covers_every_detect_literal() -> None:
    """34 DETECT literals; the 6 OFFER menus stay unchanged on purpose and are not listed."""
    assert len(MIGRATED) == 34
    assert len({line for line, _ in MIGRATED}) == len(MIGRATED)


def test_protocol_constant_is_the_shared_one() -> None:
    assert names.PROTOCOL_EDIT_TOOL_NAMES == {"edit_source", "rewrite_source"}


@pytest.mark.parametrize("line,legacy", MIGRATED, ids=[f"L{line}" for line, _ in MIGRATED])
def test_detect_literal_counts_protocol_edit_tools(line: int, legacy: tuple[str, ...]) -> None:
    strings, spliced = _literal_at(line)
    assert spliced, f"{RUNTIME.name}:{line} must splice *{PROTOCOL_SPLICE}"
    members = strings | (names.PROTOCOL_EDIT_TOOL_NAMES if spliced else frozenset())
    assert names.EDIT_SOURCE_TOOL in members
    assert names.REWRITE_SOURCE_TOOL in members
    missing = sorted(set(legacy) - members)
    assert not missing, f"{RUNTIME.name}:{line} dropped legacy members {missing}"
