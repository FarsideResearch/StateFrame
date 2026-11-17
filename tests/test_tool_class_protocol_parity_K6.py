"""Protocol edit-tool parity for the K6 edit-detection lists.

These lists classify what the model already did (counting edits, recognising
an edit call in history, evidence and ledger checks). Before this change they
named only legacy editors, so a run on the protocol menu (edit_source /
rewrite_source) silently missed its own edits. Each migrated list must now
name both protocol tools and keep every legacy member it had before.

Lists are located by their legacy member set rather than by line number, so
unrelated edits above them do not break the test.
"""
from __future__ import annotations

import ast
import importlib
import pathlib

import pytest

from kernel_edit_tool_names import EDIT_SOURCE_TOOL, PROTOCOL_EDIT_TOOL_NAMES, REWRITE_SOURCE_TOOL

ENGINE = pathlib.Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"
PROTOCOL = frozenset({EDIT_SOURCE_TOOL, REWRITE_SOURCE_TOOL})

# (module, legacy members the list carried before migration, expected spliced occurrences)
MIGRATED: tuple[tuple[str, frozenset[str], int], ...] = (
    ("kernel_decision_continuity_slots.py",
     frozenset({"edit", "str_replace", "file_create", "replace_module", "write", "replace"}), 1),
    ("kernel_decision_continuity_slots.py",
     frozenset({"edit", "str_replace", "write", "replace", "file_create"}), 1),
    ("kernel_delivery_manifest.py",
     frozenset({"edit", "edit_receipt", "file_create", "replace", "replace_module", "str_replace", "write"}), 1),
    ("kernel_empirical_fixes.py",
     frozenset({"test", "run_tests", "pytest", "verification", "edit", "str_replace",
                "file_create", "replace_module", "write"}), 1),
    ("kernel_empirical_fixes.py",
     frozenset({"edit", "str_replace", "file_create", "replace_module", "write"}), 3),
    ("kernel_evidence_relevance_shadow.py", frozenset({"edit", "str_replace"}), 1),
    ("kernel_first_green_disclosure.py",
     frozenset({"edit", "str_replace", "file_create", "replace_module", "write", "replace"}), 1),
    ("kernel_hook_wiring.py",
     frozenset({"edit", "str_replace", "insert_before", "replace_module", "file_create",
                "file_delete", "file_move", "shell"}), 1),
    ("kernel_host_facts.py",
     frozenset({"edit", "str_replace", "file_create", "replace_module", "write", "replace"}), 1),
    ("kernel_obligation_assess.py",
     frozenset({"edit", "str_replace", "file_create", "replace_module", "write", "replace"}), 1),
    ("kernel_obligations.py",
     frozenset({"edit", "file_edit", "file_create", "file_write", "patch", "replace", "str_replace", "write"}), 1),
    ("kernel_progress_facts.py",
     frozenset({"edit", "str_replace", "insert_before", "replace_module", "file_create", "write", "replace",
                "batch_edit", "edit_file", "create_file", "delete_file", "rename_file"}), 1),
    ("kernel_progress_ledger.py",
     frozenset({"edit", "str_replace", "file_create", "replace_module", "write", "replace", "patch"}), 1),
    ("kernel_reasoning_ledger.py", frozenset({"edit", "str_replace", "write", "insert"}), 1),
    ("kernel_retained_context.py", frozenset({"edit", "str_replace", "file_create"}), 1),
    ("kernel_retained_restart.py", frozenset({"edit", "str_replace", "file_create"}), 1),
    ("kernel_v46_mechanism_wiring.py",
     frozenset({"edit", "str_replace", "file_create", "replace_module", "write", "replace"}), 1),
    ("kernel_wire_truth_facts.py",
     frozenset({"edit", "str_replace", "write", "apply_patch", "done"}), 1),
    ("kernel_wire_truth_facts.py",
     frozenset({"edit", "str_replace", "write", "apply_patch"}), 2),
    ("kernel_work_carry.py",
     frozenset({"edit", "str_replace", "file_create", "replace_module", "write", "replace"}), 1),
)

MIGRATED_MODULES = tuple(dict.fromkeys(module for module, _legacy, _n in MIGRATED))


def _spliced_literal_sets(module: str) -> list[frozenset[str]]:
    """Member sets (constants plus protocol splice) of every literal list that splices the protocol names."""
    tree = ast.parse((ENGINE / module).read_text(encoding="utf-8"), filename=module)
    found: list[frozenset[str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Set, ast.Tuple, ast.List)):
            continue
        if not all(isinstance(e, (ast.Constant, ast.Starred)) for e in node.elts):
            continue
        spliced = any(isinstance(e, ast.Starred) and isinstance(e.value, ast.Name)
                      and e.value.id == "PROTOCOL_EDIT_TOOL_NAMES" for e in node.elts)
        if not spliced:
            continue
        constants = frozenset(e.value for e in node.elts
                              if isinstance(e, ast.Constant) and isinstance(e.value, str))
        found.append(constants)
    return found


@pytest.mark.parametrize("module,legacy,occurrences", MIGRATED,
                         ids=[f"{m}-{len(l)}legacy-x{n}" for m, l, n in MIGRATED])
def test_migrated_list_names_protocol_tools_and_keeps_legacy(
        module: str, legacy: frozenset[str], occurrences: int) -> None:
    matching = [constants for constants in _spliced_literal_sets(module) if constants == legacy]
    assert len(matching) == occurrences, (
        f"{module}: expected {occurrences} spliced list(s) with legacy members {sorted(legacy)}, "
        f"found {len(matching)}")
    for constants in matching:
        members = constants | PROTOCOL_EDIT_TOOL_NAMES
        assert PROTOCOL <= members, f"{module}: protocol tools missing from {sorted(members)}"
        assert legacy <= members, f"{module}: legacy members dropped from {sorted(members)}"


@pytest.mark.parametrize("module", MIGRATED_MODULES)
def test_migrated_module_imports_protocol_constant(module: str) -> None:
    """The splice needs the shared constant in scope; a bare name would fail only at runtime."""
    tree = ast.parse((ENGINE / module).read_text(encoding="utf-8"), filename=module)
    imported = any(
        isinstance(node, ast.ImportFrom) and node.module == "kernel_edit_tool_names"
        and any(alias.name == "PROTOCOL_EDIT_TOOL_NAMES" for alias in node.names)
        for node in tree.body
    )
    assert imported, f"{module} splices PROTOCOL_EDIT_TOOL_NAMES without importing it"


# Module-level constants that are evaluated at import time: check the live value, not only the source.
RUNTIME_CONSTANTS = (
    ("kernel_delivery_manifest", "_EDIT_KINDS"),
    ("kernel_first_green_disclosure", "_EDIT_KINDS"),
    ("kernel_host_facts", "_EDIT_KINDS"),
    ("kernel_obligations", "_EDIT_KINDS"),
    ("kernel_progress_facts", "_EDIT_KINDS"),
    ("kernel_progress_ledger", "_EDIT_KINDS"),
    ("kernel_v46_mechanism_wiring", "EDIT_KINDS"),
    ("kernel_work_carry", "_EDIT_TOOLS"),
)


@pytest.mark.parametrize("module,attribute", RUNTIME_CONSTANTS,
                         ids=[f"{m}.{a}" for m, a in RUNTIME_CONSTANTS])
def test_runtime_edit_constant_includes_protocol_tools(module: str, attribute: str) -> None:
    values = getattr(importlib.import_module(module), attribute)
    assert PROTOCOL <= frozenset(values)
