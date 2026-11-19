"""Protocol edit-tool parity for the K5 edit-detection literal lists.

Each list here classifies what the model already did or asked (edit counting,
receipt/ledger recognition, loop and rejection tracking, post-edit refresh).
When a run uses the protocol menu, the protocol edit tools (``edit_source``,
``rewrite_source``) must be counted alongside the legacy editors, or those
edits silently go unrecognised.

Each case identifies one literal by engine module, enclosing qualified scope, and exact legacy membership. The check is:
the list still names every legacy member, and it splices exactly the shared protocol tools on top of them.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from stateframe.engine import kernel_edit_tool_names as names
from _ast_site_keys import ProtocolLiteralSite, find_protocol_literal_sites, rename_protocol_literal_member

ENGINE = Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"
SENTINEL = "str_replace"


def _site(module: str, scope: str, legacy: set[str]) -> ProtocolLiteralSite:
    return ProtocolLiteralSite(module, scope, tuple(sorted(legacy)))


# Each literal is identified by engine module, enclosing qualified scope, and its exact legacy members.
MIGRATED_LISTS = (
    _site("kernel_edit_churn_budget", "observe_settled_action",
          {"edit", "file_create", "replace", "replace_module", "str_replace", "write"}),
    _site("kernel_groundtruth", "_assertion_from_action", {"edit", "str_replace"}),
    _site("kernel_groundtruth", "ConsumptionTracker.observe_receipt", {"edit", "str_replace", "test"}),
    _site("kernel_groundtruth", "ConsumptionTracker.observe_action", {"edit", "str_replace", "test"}),
    _site("kernel_intent_carry", "<module>",
          {"edit", "file_create", "replace", "replace_module", "str_replace", "write"}),
    _site("kernel_post_edit_refresh", "<module>",
          {"edit", "file_create", "file_delete", "file_move", "replace", "replace_module",
           "shell_effect", "str_replace", "write"}),
    _site("kernel_preedit_coverage_runtime", "<module>",
          {"edit", "file_create", "replace", "replace_module", "str_replace", "write"}),
    _site("kernel_rejected_candidate", "<module>", {"edit", "file_create", "str_replace"}),
    _site("kernel_rejection_template_restrict", "release_after_product_mutation",
          {"edit", "file_create", "insert_before", "replace", "replace_module", "str_replace", "write"}),
    _site("kernel_rejection_template_restrict", "release_noread_after_mutation",
          {"edit", "file_create", "insert_before", "replace", "replace_module", "str_replace", "write"}),
    _site("kernel_stateframe_action_ledger", "_is_open_site_action",
          {"edit", "edit_file", "file_create", "read", "str_replace"}),
    _site("kernel_stateframe_action_ledger", "_fallback_edit_open_site_locator",
          {"edit", "edit_file", "str_replace"}),
    _site("kernel_stateframe_action_ledger", "_bounded_action_outcome",
          {"edit", "edit_file", "str_replace"}),
    _site("kernel_stateframe_action_ledger", "_edit_fact",
          {"edit", "edit_file", "str_replace"}),
    _site("kernel_stateframe_action_ledger", "_edit_action_fact",
          {"edit", "edit_file", "str_replace"}),
    _site("kernel_stateframe_action_ledger", "build_action_context",
          {"edit", "edit_file", "file_create", "str_replace"}),
    _site("kernel_stateframe_f4", "<module>",
          {"apply_patch", "edit", "edit_file", "file_delete", "file_move", "shell", "str_replace",
           "write_file"}),
    _site("kernel_stateframe_thinking_router", "_row_is_applied_edit",
          {"edit", "file_create", "replace_module", "str_replace"}),
    _site("kernel_stateframe_v2_s1", "<module>", {"edit", "file_create", "str_replace"}),
    _site("kernel_stateframe_v2_s1", "_resolution_receipts",
          {"edit", "file_create", "probe", "read", "run_tests", "str_replace", "test"}),
    _site("kernel_working_set_snapshot", "freeze_preparation",
          {"edit", "file_create", "file_delete", "file_move", "insert_before", "str_replace"}),
)

MIGRATED_MODULES = tuple(sorted({site.module for site in MIGRATED_LISTS}))


def _paths() -> dict[str, Path]:
    return {module: ENGINE / f"{module}.py" for module in MIGRATED_MODULES}


def _inventory_error(paths: dict[str, Path]) -> str | None:
    found = Counter(site for path in paths.values() for site in find_protocol_literal_sites(path))
    expected = Counter(MIGRATED_LISTS)
    untriaged = sorted((found - expected).elements(), key=lambda site: site.label)
    stale = sorted((expected - found).elements(), key=lambda site: site.label)
    if not untriaged and not stale:
        return None
    details = []
    if untriaged:
        details.append("untriaged: " + "; ".join(site.label for site in untriaged))
    if stale:
        details.append("stale: " + "; ".join(site.label for site in stale))
    return " | ".join(details)


def test_migrated_list_count_is_pinned() -> None:
    assert len(MIGRATED_LISTS) == 21


@pytest.mark.parametrize(
    "site", MIGRATED_LISTS, ids=lambda site: site.label,
)
def test_detect_list_counts_protocol_edit_tools(site: ProtocolLiteralSite) -> None:
    path = ENGINE / f"{site.module}.py"
    matches = [candidate for candidate in find_protocol_literal_sites(path) if candidate == site]
    assert len(matches) == 1, f"{site.label}: expected exactly one legacy literal, found {len(matches)}"
    members = frozenset(site.legacy_members) | names.PROTOCOL_EDIT_TOOL_NAMES
    assert names.EDIT_SOURCE_TOOL in members, f"{site.label} does not count edit_source"
    assert names.REWRITE_SOURCE_TOOL in members, f"{site.label} does not count rewrite_source"
    assert frozenset(site.legacy_members) <= members, f"{site.label} dropped legacy members"
    assert members - frozenset(site.legacy_members) == names.PROTOCOL_EDIT_TOOL_NAMES, (
        f"{site.label} has unexpected members beyond legacy + protocol")


def test_migrated_literal_inventory_is_complete() -> None:
    error = _inventory_error(_paths())
    assert error is None, error


def test_protocol_literal_locator_survives_shift_and_rejects_removed_or_new_sites(
        tmp_path: Path) -> None:
    original_paths = _paths()
    shifted_paths: dict[str, Path] = {}
    for module, original_path in original_paths.items():
        copy_path = tmp_path / "shifted" / original_path.name
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        copy_path.write_text("\n" * 5 + "# shifted copy\n" + original_path.read_text(encoding="utf-8"),
                             encoding="utf-8")
        shifted_paths[module] = copy_path
    assert _inventory_error(shifted_paths) is None

    for site in MIGRATED_LISTS:
        original_path = original_paths[site.module]
        copy_path = tmp_path / "removed" / site.module / original_path.name
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        source = original_path.read_text(encoding="utf-8")
        changed = rename_protocol_literal_member(
            original_path, source, site, SENTINEL, "renamed_str_replace")
        copy_path.write_text(changed, encoding="utf-8")
        changed_paths = dict(original_paths)
        changed_paths[site.module] = copy_path
        error = _inventory_error(changed_paths)
        assert error is not None and f"stale: {site.label}" in error, error

    module_path = original_paths["kernel_edit_churn_budget"]
    copy_path = tmp_path / "added" / module_path.name
    copy_path.parent.mkdir(parents=True, exist_ok=True)
    source = module_path.read_text(encoding="utf-8")
    source += "\nUNREVIEWED_TOOL_LIST = {\"str_replace\", *PROTOCOL_EDIT_TOOL_NAMES}\n"
    copy_path.write_text(source, encoding="utf-8")
    changed_paths = dict(original_paths)
    changed_paths["kernel_edit_churn_budget"] = copy_path
    error = _inventory_error(changed_paths)
    assert error is not None and "untriaged: kernel_edit_churn_budget.<module>" in error, error
