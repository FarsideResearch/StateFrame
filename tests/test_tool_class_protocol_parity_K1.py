"""Protocol edit-tool parity for kernel_stateframe_runtime.py (chunk K1).

Each literal listed in MIGRATED classifies what the model already did: an applied edit,
an edit call being recognised, or repair/budget accounting. Those literals must count
the protocol edit tools (edit_source, rewrite_source) alongside every legacy editor they
named before, otherwise a run on the protocol menu silently misses the edit.

Each site is keyed by its enclosing qualified name and exact legacy members.
The occurrence is counted in AST traversal order within identical keys.
"""
from __future__ import annotations

import ast
import pathlib
from collections import Counter
from collections.abc import Iterator
from typing import NamedTuple

import pytest

from stateframe.engine import kernel_edit_tool_names as names

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "src" / "stateframe" / "engine" / "kernel_stateframe_runtime.py"
PROTOCOL_SPLICE = "PROTOCOL_EDIT_TOOL_NAMES"

class _Site(NamedTuple):
    qualname: str
    legacy: frozenset[str]
    occurrence: int = 1

    def __str__(self) -> str:
        return f"{self.qualname} #{self.occurrence} with legacy members {sorted(self.legacy)}"


_EDIT_PAIR = frozenset({"edit", "str_replace"})
_EDITS_WITH_CREATE = _EDIT_PAIR | {"file_create"}
_EDITS_WITH_WRITE = _EDITS_WITH_CREATE | {"replace", "replace_module", "write"}
_FILE_MUTATIONS = _EDITS_WITH_WRITE | {"file_delete", "file_move", "shell"}
_FILE_EFFECTS = _FILE_MUTATIONS | {"shell_effect"}
_RECEIPT_EDITS = _EDITS_WITH_WRITE | {"shell_effect"}
_SOURCE_EFFECTS = _EDITS_WITH_CREATE | {"replace_module", "shell_effect"}

MIGRATED = (
    _Site("_apply_self_edit_public_view", _EDITS_WITH_CREATE),
    _Site("StateFrameRuntime._bind_target_from_action",
          _EDITS_WITH_WRITE | {"create_file", "delete_file", "edit_file", "rename_file", "shell"}),
    _Site("StateFrameRuntime._schedule_auto_verify", _EDITS_WITH_CREATE),
    _Site("StateFrameRuntime.completion_mutation_guard", _FILE_MUTATIONS),
    _Site("StateFrameRuntime.bounded_locate_before_dispatch", _FILE_EFFECTS),
    _Site("StateFrameRuntime._ripple_preview_before_write", _FILE_MUTATIONS),
    _Site("StateFrameRuntime.start_action", _FILE_MUTATIONS),
    _Site("StateFrameRuntime.start_action", _EDITS_WITH_WRITE | {"shell"}),
    _Site("StateFrameRuntime.start_action", _EDITS_WITH_CREATE),
    _Site("StateFrameRuntime.start_action", _EDITS_WITH_CREATE | {"replace_module"}),
    _Site("StateFrameRuntime.start_action", _EDITS_WITH_WRITE, 1),
    _Site("StateFrameRuntime.start_action", _EDITS_WITH_WRITE, 2),
    _Site("StateFrameRuntime._count_host_repair_action", _RECEIPT_EDITS | {"done", "shell"}),
    _Site("StateFrameRuntime._bind_other_end_contract_receipt", _FILE_EFFECTS),
    _Site("StateFrameRuntime.completed_receipt", _RECEIPT_EDITS, 1),
    _Site("StateFrameRuntime.completed_receipt", _RECEIPT_EDITS, 2),
    _Site("StateFrameRuntime.completed_receipt", _RECEIPT_EDITS, 3),
    _Site("StateFrameRuntime.completed_receipt", _RECEIPT_EDITS, 4),
    _Site("StateFrameRuntime.completed_receipt", _SOURCE_EFFECTS),
    _Site("StateFrameRuntime.completed_receipt", _RECEIPT_EDITS, 5),
    _Site("StateFrameRuntime.completed_receipt", _RECEIPT_EDITS, 6),
    _Site("StateFrameRuntime.completed_receipt", _RECEIPT_EDITS, 7),
    _Site("StateFrameRuntime.completed_receipt", _RECEIPT_EDITS, 8),
    _Site("StateFrameRuntime.completed_receipt", _EDITS_WITH_CREATE),
    _Site("StateFrameRuntime.completed_receipt", _RECEIPT_EDITS, 9),
    _Site("StateFrameRuntime._close_multi_definition_open_work",
          _EDITS_WITH_WRITE - {"file_create"}),
    _Site("StateFrameRuntime._retire_multi_definition_open_work",
          _EDITS_WITH_WRITE - {"file_create"}),
    _Site("StateFrameRuntime._record_multi_definition_open_work", _EDIT_PAIR | {"replace_module"}),
    _Site("StateFrameRuntime.close_step", _FILE_EFFECTS),
    _Site("StateFrameRuntime.close_step", _EDITS_WITH_CREATE),
    _Site("StateFrameRuntime._thinking_trigger", _EDITS_WITH_CREATE),
    _Site("StateFrameRuntime.turtle_pre_request", _EDITS_WITH_CREATE),
    _Site("StateFrameRuntime._terminal_last_edit", _SOURCE_EFFECTS),
    _Site("StateFrameRuntime._audit_repair_packet_request", _EDIT_PAIR),
)

# These six OFFER menus share the legacy edit pair used by DETECT literals.
_COMPLETION_MENU = _EDIT_PAIR | {"done", "read", "run_tests"}
_OFFER_SITES = frozenset({
    _Site("_lean_no_host_recovery_tools", _EDIT_PAIR | {
        "decide", "done", "file_create", "file_delete", "file_move", "grep", "job_kill",
        "job_output", "job_status", "launch_job", "list_files", "note_retrieve", "note_update",
        "read", "read_check_receipt", "recall", "recall_evidence", "references", "run_tests",
        "shell", "where_defined", "who_calls",
    }),
    _Site("StateFrameRuntime._get_sure_menu", _EDIT_PAIR),
    _Site("StateFrameRuntime._run_host_repair_checks", _COMPLETION_MENU),
    _Site("StateFrameRuntime._host_closure_for_done", _COMPLETION_MENU),
    _Site("StateFrameRuntime._host_closure_for_done", _EDIT_PAIR),
    _Site("StateFrameRuntime._terminal_tool_menu", _EDIT_PAIR | {"read"}),
})


def _scoped_literals(
    node: ast.AST, qualname: str = "<module>", prefix: str = "",
) -> Iterator[tuple[str, ast.Set | ast.Tuple | ast.List]]:
    if isinstance(node, ast.ClassDef):
        qualname = f"{prefix}{node.name}"
        prefix = f"{qualname}."
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        qualname = f"{prefix}{node.name}"
        prefix = f"{qualname}.<locals>."
    if isinstance(node, (ast.Set, ast.Tuple, ast.List)):
        yield qualname, node
    for child in ast.iter_child_nodes(node):
        yield from _scoped_literals(child, qualname, prefix)


def _detect_literals(path: pathlib.Path) -> dict[_Site, bool]:
    """Index tool literals by symbol and record whether each splices the protocol constant."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    occurrences: Counter[tuple[str, frozenset[str]]] = Counter()
    sites: dict[_Site, bool] = {}
    for scope, node in _scoped_literals(tree):
        legacy = frozenset(
            elt.value for elt in node.elts
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
        )
        spliced = any(
            isinstance(elt, ast.Starred) and isinstance(elt.value, ast.Name)
            and elt.value.id == PROTOCOL_SPLICE
            for elt in node.elts
        )
        # Scan both spliced literals and legacy edit pairs so a missing splice
        # or a newly added legacy-only detection literal is still visible.
        if not spliced and not _EDIT_PAIR <= legacy:
            continue
        occurrences[scope, legacy] += 1
        site = _Site(scope, legacy, occurrences[scope, legacy])
        if spliced or site not in _OFFER_SITES:
            sites[site] = spliced
    return sites


def _assert_inventory(path: pathlib.Path, sites: dict[_Site, bool]) -> None:
    expected = set(MIGRATED)
    missing = expected - sites.keys()
    untriaged = sites.keys() - expected
    assert not (missing or untriaged), "\n".join([
        *(f"{path.name}: no set/tuple/list literal at {site}" for site in sorted(map(str, missing))),
        *(f"{path.name}: untriaged DETECT literal at {site}" for site in sorted(map(str, untriaged))),
    ])


@pytest.fixture(scope="module")
def runtime_literals() -> dict[_Site, bool]:
    return _detect_literals(RUNTIME)


def test_migrated_table_covers_every_detect_literal(runtime_literals: dict[_Site, bool]) -> None:
    """34 DETECT literals; the 6 OFFER menus stay unchanged on purpose and are not listed."""
    assert len(MIGRATED) == 34
    assert len(set(MIGRATED)) == len(MIGRATED)
    _assert_inventory(RUNTIME, runtime_literals)


def test_protocol_constant_is_the_shared_one() -> None:
    assert names.PROTOCOL_EDIT_TOOL_NAMES == {"edit_source", "rewrite_source"}


def _assert_literal_parity(path: pathlib.Path, sites: dict[_Site, bool], site: _Site) -> None:
    label = f"{path.name}: {site}"
    assert site in sites, f"{path.name}: no set/tuple/list literal at {site}"
    assert sites[site], f"{label} must splice *{PROTOCOL_SPLICE}"
    members = site.legacy | names.PROTOCOL_EDIT_TOOL_NAMES
    assert names.EDIT_SOURCE_TOOL in members
    assert names.REWRITE_SOURCE_TOOL in members


@pytest.mark.parametrize("site", [pytest.param(site, id=str(site)) for site in MIGRATED])
def test_detect_literal_counts_protocol_edit_tools(
    site: _Site, runtime_literals: dict[_Site, bool],
) -> None:
    _assert_literal_parity(RUNTIME, runtime_literals, site)


def test_detect_literals_survive_line_shift(tmp_path: pathlib.Path) -> None:
    runtime = tmp_path / RUNTIME.name
    source = RUNTIME.read_text(encoding="utf-8")
    runtime.write_text("\n" * 5 + "# Shift all K1 sites.\n" + source, encoding="utf-8")
    sites = _detect_literals(runtime)
    _assert_inventory(runtime, sites)
    for site in MIGRATED:
        _assert_literal_parity(runtime, sites, site)


def test_renamed_detect_literal_reports_missing_site(tmp_path: pathlib.Path) -> None:
    runtime = tmp_path / RUNTIME.name
    source = RUNTIME.read_text(encoding="utf-8")
    declaration = "def _apply_self_edit_public_view("
    assert declaration in source
    runtime.write_text(
        source.replace(declaration, "def _renamed_self_edit_public_view(", 1),
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match=r"no set/tuple/list literal at _apply_self_edit_public_view\b"):
        _assert_inventory(runtime, _detect_literals(runtime))


@pytest.mark.parametrize("splice", [
    pytest.param(f", *{PROTOCOL_SPLICE}", id="spliced"),
    pytest.param("", id="legacy-only"),
])
def test_added_detect_literal_is_untriaged(tmp_path: pathlib.Path, splice: str) -> None:
    runtime = tmp_path / RUNTIME.name
    source = RUNTIME.read_text(encoding="utf-8")
    runtime.write_text(
        source + f'\n\ndef _new_edit_filter(kind):\n    return kind in {{"edit", "str_replace"{splice}}}\n',
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match=r"untriaged DETECT literal at _new_edit_filter\b"):
        _assert_inventory(runtime, _detect_literals(runtime))
