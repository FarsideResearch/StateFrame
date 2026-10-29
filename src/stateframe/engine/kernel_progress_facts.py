"""Host-recorded edit and check facts for repair progress.

Model prose is intentionally not an input to this module. Callers provide typed
receipts and host-owned completion projections only.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
from dataclasses import asdict, dataclass, is_dataclass
import os
from pathlib import PurePosixPath


MODE_ENV = "STATEFRAME_HOST_TRUTH_PROGRESS"
_APPLIED_RESULTS = frozenset({
    "APPLIED", "CHANGED", "CREATED", "DELETED", "RENAMED", "SUCCESS", "OK",
})
_EDIT_KINDS = frozenset({
    "edit", "str_replace", "insert_before", "replace_module", "file_create",
    "write", "replace", "batch_edit", "edit_file", "create_file",
    "delete_file", "rename_file",
})
_CHECK_KINDS = frozenset({
    "test", "tests", "run_tests", "authored_check", "test_run", "check",
    "check_receipt", "test_subprocess_complete", "test_subprocess_completed",
    "auto_verify", "auto_verify_completed", "host_check", "host_obligation_check",
    "run_check",
})
_NON_EXECUTED_CHECK_RESULTS = frozenset({
    "AUTOMATIC_CHECK_NOT_FUNDED", "INVALID_CHECK", "NO_CHECK", "NOT_RUN", "SKIPPED",
    "NOT_EXECUTED", "REFUSED", "REJECTED", "UNRUNNABLE", "UNSUPPORTED_ACTION",
})


@dataclass(frozen=True, slots=True)
class ProgressFacts:
    """One consistent snapshot of host-recorded work and verification."""

    edits: int
    checks: int
    product_edits: int
    changed_files: tuple[str, ...]
    first_edit_step: int | None
    edits_since_check: int
    latest_check_status: str | None
    source: str
    edit_receipt_ids: tuple[str, ...] = ()
    check_receipt_ids: tuple[str, ...] = ()

    def to_record(self) -> dict[str, object]:
        """Return the host-owned checkpoint representation used across restarts."""
        return {"schema": "kernel-host-progress-facts/1", **asdict(self)}

    @property
    def acted(self) -> bool:
        return self.product_edits > 0 or self.checks > 0

    @property
    def latest_check_passed(self) -> bool:
        return (self.latest_check_status or "").upper() in {
            "GREEN", "PASS", "PASSED", "PASS_CASES", "TESTS_PASSED",
        }


def host_truth_progress_enabled(
        environ: Mapping[str, str] | None = None) -> bool:
    """Read the strict default-on release switch."""
    source = os.environ if environ is None else environ
    value = str(source.get(MODE_ENV, "on") or "on").strip().lower()
    if value not in {"on", "off"}:
        raise ValueError(f"{MODE_ENV} must be on or off")
    return value == "on"


def _mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        try:
            row = asdict(value)
        except (TypeError, ValueError):
            return None
        return row if isinstance(row, Mapping) else None
    for name in ("as_dict", "to_dict"):
        method = getattr(value, name, None)
        if callable(method):
            try:
                row = method()
            except (AttributeError, TypeError, ValueError):
                continue
            if isinstance(row, Mapping):
                return row
    return None


def _items(value: object) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping) or _mapping(value) is not None:
        return (value,)
    if isinstance(value, (str, bytes, bytearray)):
        return (value,)
    if isinstance(value, (Sequence, Set)):
        return tuple(value)
    return ()


def _status(row: Mapping[str, object]) -> str:
    for name in ("progress_result_class", "result_class", "status", "outcome", "result"):
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip().upper().replace("-", "_")
    return ""


def _kind(row: Mapping[str, object]) -> str:
    generic_kind = ""
    for name in ("progress_kind", "kind", "type", "tool", "operation", "producer"):
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            selected = value.strip().lower()
            if selected in {"tool", "action", "receipt", "event"}:
                generic_kind = selected
                continue
            return selected
    action_name = row.get("name")
    if isinstance(action_name, str) and action_name.strip():
        selected_name = action_name.strip().lower()
        if selected_name in _EDIT_KINDS or selected_name in _CHECK_KINDS:
            return selected_name
    return generic_kind


def _identity(row: Mapping[str, object], *, prefix: str) -> str | None:
    for name in ("receipt_id", "execution_id", "host_execution_id", "event_id", "id"):
        value = row.get(name)
        if value not in (None, ""):
            return f"{prefix}:{value}"
    return None


def _step(row: Mapping[str, object]) -> int | None:
    for name in ("turn", "step", "sequence"):
        value = row.get(name)
        if type(value) is int and value >= 0:
            return value
    return None


def _is_test_path(path: str) -> bool:
    normalized = path.replace("\\", "/").strip("/")
    parts = tuple(part.lower() for part in PurePosixPath(normalized).parts)
    if not parts:
        return False
    name = parts[-1]
    return (
        any(part in {"test", "tests", "__tests__"} for part in parts)
        or name == "conftest.py"
        or name.startswith("test_")
        or name.endswith("_test.py")
    )


def _paths_from_value(value: object, *, allow_strings: bool = False) -> set[str]:
    paths: set[str] = set()
    for raw in _items(value):
        row = _mapping(raw)
        if row is None:
            if allow_strings and isinstance(raw, str) and raw.strip():
                paths.add(raw.strip())
            continue
        for name in ("target_path", "file_path", "path", "file", "target"):
            path = row.get(name)
            if isinstance(path, str) and path.strip():
                paths.add(path.strip())
        for name in (
            "changed_paths", "changed_files", "changed_file_paths", "files_changed",
            "files", "paths", "applied_edits",
        ):
            nested = row.get(name)
            if nested is None:
                continue
            for item in _items(nested):
                if isinstance(item, str) and item.strip():
                    paths.add(item.strip())
                else:
                    child = _mapping(item)
                    if child is None:
                        continue
                    for path_name in ("target_path", "file_path", "path", "file"):
                        path = child.get(path_name)
                        if isinstance(path, str) and path.strip():
                            paths.add(path.strip())
    return paths


def _accepted_edit(row: Mapping[str, object], *, effect: bool = False) -> bool:
    status = _status(row)
    if status not in _APPLIED_RESULTS:
        return False
    if row.get("changed") is False or row.get("effect_applied") is False:
        return False
    if effect:
        if row.get("effect_applied") is not True:
            return False
        before = row.get("before_hash")
        after = row.get("after_hash")
        if isinstance(before, str) and isinstance(after, str) and before and after:
            if before == after:
                return False
    return True


def _accepted_check(row: Mapping[str, object]) -> bool:
    status = _status(row)
    if status in _NON_EXECUTED_CHECK_RESULTS:
        return False
    schema = str(row.get("schema") or "").lower()
    kind = _kind(row)
    has_identity = any(
        row.get(name) not in (None, "")
        for name in ("receipt_id", "execution_id", "host_execution_id", "check_id",
                     "instance_key", "execution_ids", "check_receipt_ids",
                     "test_receipt_id")
    )
    provenance = _mapping(row.get("provenance"))
    provenance_kind = str((provenance or {}).get("kind") or "").upper()
    execution = str(row.get("execution") or "").upper()
    return has_identity and (
        kind in _CHECK_KINDS
        or schema.startswith((
            "check-receipt/", "test-receipt/", "stateframe-host-check-receipt/",
            "stateframe-authored-check/",
        ))
        or any(name in row for name in ("execution_id", "execution_ids", "check_receipt_ids"))
        or row.get("authored_check") is True
        or provenance_kind == "AUTHORED_CHECK"
        or execution in {"EXECUTED", "SETUP_ERROR"}
    )


def _check_identities(row: Mapping[str, object]) -> tuple[str, ...]:
    identities: set[str] = set()
    execution_ids = row.get("execution_ids")
    if isinstance(execution_ids, Sequence) and not isinstance(
            execution_ids, (str, bytes, bytearray)):
        identities.update(f"execution:{item}" for item in execution_ids
                          if item not in (None, ""))
    for name in ("execution_id", "host_execution_id"):
        value = row.get(name)
        if value not in (None, ""):
            identities.add(f"execution:{value}")
    for name in ("receipt_id", "check_receipt_id", "test_receipt_id", "event_id", "id"):
        value = row.get(name)
        if value not in (None, ""):
            identities.add(f"receipt:{value}")
    for name in ("check_id", "instance_key"):
        value = row.get(name)
        if value not in (None, ""):
            identities.add(f"check:{value}")
    ids = row.get("check_receipt_ids")
    if isinstance(ids, Sequence) and not isinstance(ids, (str, bytes, bytearray)):
        identities.update(f"receipt:{item}" for item in ids if item not in (None, ""))
    return tuple(sorted(identities))


def _check_identity_groups(rows: Sequence[Mapping[str, object]]) -> tuple[frozenset[str], ...]:
    """Merge receipt rows that share any host identity alias."""
    groups: list[set[str]] = []
    for row in rows:
        merged = set(_check_identities(row))
        if not merged:
            continue
        unmatched: list[set[str]] = []
        for group in groups:
            if group.intersection(merged):
                merged.update(group)
            else:
                unmatched.append(group)
        unmatched.append(merged)
        groups = unmatched
    return tuple(frozenset(group) for group in groups)


def _file_create_facts(value: object) -> tuple[set[str], tuple[Mapping[str, object], ...]]:
    paths: set[str] = set()
    rows: list[Mapping[str, object]] = []
    for raw in _items(value):
        if isinstance(raw, str) and raw.strip():
            # `created_this_run` is a host-maintained set populated only after a
            # successful file_create, so path-only entries are already accepted.
            paths.add(raw.strip())
            continue
        row = _mapping(raw)
        if row is None or not _accepted_edit(row):
            continue
        rows.append(row)
        paths.update(_paths_from_value(row))
    return paths, tuple(rows)


def _latest_check_context(
        owner_rows: Sequence[Mapping[str, object]],
        canonical_rows: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object] | None, int | None]:
    candidates: list[tuple[int, int, int, Mapping[str, object], int | None]] = []
    for index, row in enumerate(canonical_rows):
        step = _step(row)
        if step is not None:
            candidates.append((step, 1, index, row, None))
    owner_checks: list[tuple[int, Mapping[str, object]]] = []
    for index, row in enumerate(owner_rows):
        if _kind(row) not in _CHECK_KINDS or not _accepted_check(row):
            continue
        owner_checks.append((index, row))
        step = _step(row)
        if step is not None:
            candidates.append((step, 0, index, row, index))

    if candidates:
        _step_number, _source_priority, _index, selected, owner_index = max(
            candidates, key=lambda candidate: candidate[:3])
    elif owner_checks:
        owner_index, selected = owner_checks[-1]
    elif canonical_rows:
        selected = canonical_rows[-1]
        owner_index = None
    else:
        return None, None

    if owner_index is None:
        identities = set(_check_identities(selected))
        if identities:
            matches = [
                index for index, row in owner_checks
                if identities.intersection(_check_identities(row))
            ]
            if matches:
                owner_index = matches[-1]
    return selected, owner_index


def _edit_count(rows: Sequence[Mapping[str, object]]) -> int:
    identities: set[str] = set()
    unidentified: list[Mapping[str, object]] = []
    for row in rows:
        identity = _identity(row, prefix="edit")
        if identity is not None:
            identities.add(identity)
        elif row not in unidentified:
            unidentified.append(row)
    return len(identities) + len(unidentified)


def _result(source: str, edits: object, checks: object, product_edits: object,
            first_edit_step: object, edits_since_check: object) -> ProgressFacts:
    return ProgressFacts(
        edits=max(0, edits) if type(edits) is int else 0,
        checks=max(0, checks) if type(checks) is int else 0,
        product_edits=max(0, product_edits) if type(product_edits) is int else 0,
        changed_files=(),
        first_edit_step=(first_edit_step if type(first_edit_step) is int
                         and first_edit_step >= 0 else None),
        edits_since_check=(max(0, edits_since_check)
                           if type(edits_since_check) is int else 0),
        latest_check_status=None,
        source=source,
    )


def derive_host_progress_facts(
        *,
        edit_receipts: object = (),
        effect_receipts: object = (),
        delivery_manifest: object = None,
        done_facts: object = None,
        product_edits: object = 0,
        file_create_results: object = (),
        test_receipts: object = (),
        host_check_receipts: object = (),
        check_receipts: object = (),
        owner_receipts: object = (),
        legacy_edits: object = 0,
        legacy_checks: object = 0,
        legacy_product_edits: object = 0,
        legacy_first_edit_step: object = None,
        legacy_edits_since_check: object = 0,
        prior_facts: object = None,
        enabled: bool | None = None,
) -> ProgressFacts:
    """Reduce typed host receipts into one progress authority.

    When the release switch is off, the returned fields preserve the legacy
    counters exactly. When enabled, model messages and observations are ignored.
    """
    selected = host_truth_progress_enabled() if enabled is None else enabled
    if selected is not True:
        return _result(
            "legacy", legacy_edits, legacy_checks, legacy_product_edits,
            legacy_first_edit_step, legacy_edits_since_check,
        )

    owner_rows = tuple(_mapping(item) for item in _items(owner_receipts))
    owner_rows = tuple(row for row in owner_rows if row is not None)
    prior = _mapping(prior_facts)
    if (prior is None or prior.get("schema") != "kernel-host-progress-facts/1"
            or prior.get("source") != "host"):
        prior = None
    prior_files = {
        path for path in _items(prior.get("changed_files") if prior else ())
        if isinstance(path, str) and path.strip()
    }
    prior_edit_ids = {
        value for value in _items(prior.get("edit_receipt_ids") if prior else ())
        if isinstance(value, str) and value
    }
    prior_check_ids = {
        value for value in _items(prior.get("check_receipt_ids") if prior else ())
        if isinstance(value, str) and value
    }
    prior_edits = prior.get("edits", 0) if prior else 0
    prior_checks = prior.get("checks", 0) if prior else 0
    prior_product_edits = prior.get("product_edits", 0) if prior else 0
    prior_first_edit_step = prior.get("first_edit_step") if prior else None
    prior_edits_since_check = prior.get("edits_since_check", 0) if prior else 0
    prior_latest_check_status = prior.get("latest_check_status") if prior else None
    edit_rows: list[Mapping[str, object]] = []
    effect_rows: list[Mapping[str, object]] = []
    owner_edit_rows: list[Mapping[str, object]] = []
    for row in owner_rows:
        if _kind(row) in _EDIT_KINDS and _accepted_edit(row):
            owner_edit_rows.append(row)
    for raw in _items(edit_receipts):
        row = _mapping(raw)
        if row is not None and _kind(row) in _EDIT_KINDS and _accepted_edit(row):
            edit_rows.append(row)
    for raw in _items(effect_receipts):
        row = _mapping(raw)
        if row is not None and _accepted_edit(row, effect=True):
            effect_rows.append(row)

    file_create_paths, file_create_rows = _file_create_facts(file_create_results)
    manifest = _mapping(delivery_manifest)
    done = _mapping(done_facts)
    manifest_files = _paths_from_value(manifest) if manifest is not None else set()
    done_files = _paths_from_value(done) if done is not None else set()
    receipt_paths = _paths_from_value(edit_rows)
    receipt_paths.update(_paths_from_value(effect_rows))
    receipt_paths.update(_paths_from_value(owner_edit_rows))
    current_changed_files = (
        file_create_paths | manifest_files | done_files | receipt_paths
    )
    changed_files_set = current_changed_files | prior_files
    changed_files = tuple(sorted(changed_files_set))

    current_edit_ids = {
        identity for row in (*edit_rows, *effect_rows, *owner_edit_rows, *file_create_rows)
        if (identity := _identity(row, prefix="edit")) is not None
    }
    edit_ids = current_edit_ids | prior_edit_ids
    explicit_product_edits = product_edits if type(product_edits) is int and product_edits > 0 else 0
    product_paths = {
        path for path in changed_files if not _is_test_path(path)
    }
    product_edit_count = max(
        explicit_product_edits,
        len(product_paths),
        prior_product_edits if type(prior_product_edits) is int else 0,
    )
    edit_count = max(
        len(changed_files), len(edit_ids),
        _edit_count((*edit_rows, *effect_rows, *owner_edit_rows, *file_create_rows)),
        explicit_product_edits,
        prior_edits if type(prior_edits) is int else 0,
    )

    canonical_check_rows: list[Mapping[str, object]] = []
    canonical_sources = (
        test_receipts,
        host_check_receipts,
        check_receipts,
        manifest.get("checks", ()) if manifest is not None else (),
        done.get("last_check_result") if done is not None else (),
    )
    for source in canonical_sources:
        for raw in _items(source):
            row = _mapping(raw)
            if row is None or not _accepted_check(row):
                continue
            ids = _check_identities(row)
            if not ids:
                continue
            canonical_check_rows.append(row)

    owner_check_rows: list[Mapping[str, object]] = []
    for row in owner_rows:
        kind = _kind(row)
        if kind in _CHECK_KINDS and _accepted_check(row):
            owner_check_rows.append(row)

    check_groups = _check_identity_groups((*canonical_check_rows, *owner_check_rows))
    current_check_ids = set().union(*check_groups) if check_groups else set()
    new_check_groups = [group for group in check_groups if not group.intersection(prior_check_ids)]
    check_ids = current_check_ids | prior_check_ids
    check_count = max(0, prior_checks) if type(prior_checks) is int else 0
    check_count += len(new_check_groups)

    current_edit_rows = (*edit_rows, *effect_rows, *owner_edit_rows, *file_create_rows)
    edit_steps = [
        turn for row in current_edit_rows
        if (turn := _step(row)) is not None
    ]
    first_edit_step = min(edit_steps) if edit_steps else None
    if type(prior_first_edit_step) is int and prior_first_edit_step >= 0:
        first_edit_step = (min(first_edit_step, prior_first_edit_step)
                           if first_edit_step is not None else prior_first_edit_step)

    latest_check, latest_owner_position = _latest_check_context(
        owner_rows, canonical_check_rows)
    latest_check_status = None
    if latest_check is not None:
        latest_check_status = _status(latest_check) or "UNKNOWN"
        latest_check_step = _step(latest_check)
        if latest_owner_position is not None:
            owner_after_check = [
                row for row in owner_rows[latest_owner_position + 1:]
                if _kind(row) in _EDIT_KINDS and _accepted_edit(row)
            ]
            if latest_check_step is None:
                edits_since_check = _edit_count(owner_after_check)
            else:
                prior_owner_edits = owner_rows[:latest_owner_position]
                prior_owner_ids = {
                    identity for row in prior_owner_edits
                    if _kind(row) in _EDIT_KINDS and _accepted_edit(row)
                    if (identity := _identity(row, prefix="edit")) is not None
                }
                prior_owner_objects = {id(row) for row in prior_owner_edits}
                after_latest_step = [
                    row for row in current_edit_rows
                    if ((current := _step(row)) is not None and current >= latest_check_step)
                    or (current is None and id(row) not in prior_owner_objects
                        and _identity(row, prefix="edit") not in prior_owner_ids)
                ]
                edits_since_check = max(
                    _edit_count(owner_after_check), _edit_count(after_latest_step),
                )
        elif latest_check_step is not None:
            after_latest_step = [
                row for row in current_edit_rows
                if (current := _step(row)) is None or current >= latest_check_step
            ]
            edits_since_check = _edit_count(after_latest_step)
        else:
            # There is no chronology linking this current check to the edit rows.
            # Keep verification freshness conservative until a host step is known.
            edits_since_check = edit_count
    elif isinstance(prior_latest_check_status, str):
        latest_check_status = prior_latest_check_status
        if prior is not None:
            edits_since_check = max(
                0, prior_edits_since_check if type(prior_edits_since_check) is int else 0,
            ) + max(
                len(current_edit_ids - prior_edit_ids),
                len(current_changed_files - prior_files),
            )
        else:
            edits_since_check = edit_count
    elif prior is not None:
        edits_since_check = max(
            0, prior_edits_since_check if type(prior_edits_since_check) is int else 0,
        ) + max(
            len(current_edit_ids - prior_edit_ids),
            len(current_changed_files - prior_files),
        )
    else:
        edits_since_check = edit_count

    return ProgressFacts(
        edits=edit_count,
        checks=check_count,
        product_edits=product_edit_count,
        changed_files=changed_files,
        first_edit_step=first_edit_step,
        edits_since_check=edits_since_check,
        latest_check_status=latest_check_status,
        source="host",
        edit_receipt_ids=tuple(sorted(edit_ids)),
        check_receipt_ids=tuple(sorted(check_ids)),
    )


__all__ = [
    "MODE_ENV", "ProgressFacts", "derive_host_progress_facts",
    "host_truth_progress_enabled",
]
