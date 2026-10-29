"""Q3 (E9): a row whose entry is absent and whose public checks pass on the parent
is NEW_BEHAVIOR, not a repro hunt.

When E1 says the row's entry is absent (``NOT_FOUND`` / ``REFERENCED_ONLY`` /
``kind: new_symbol``) or Q2 counts 0 product writers of the ticket's state, AND
the public checks PASS on the parent, the public tests cannot go red: the
witness is ``ABSENCE_ON_PARENT`` (an authored check that imports/calls the new
entry fails on the parent with ImportError/AttributeError).  Green public tests
are never a witness for such a row; certification needs an authored check.

Baseline PASS comes from the task row (``public_baseline``) or is observed by
the host: a SATISFIED public verdict while the workspace binding still equals
the first-request (parent) binding.

``STATEFRAME_ROW_ROLE`` (default ``on``; ``off`` keeps every row REPAIR).
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

ENV = "STATEFRAME_ROW_ROLE"
HEADER = "[ROW ROLE]"
NEW_BEHAVIOR = "NEW_BEHAVIOR"
REPAIR = "REPAIR"
ABSENCE_ON_PARENT = "ABSENCE_ON_PARENT"
PUBLIC_RED = "PUBLIC_RED"
_ABSENT = frozenset({"NOT_FOUND", "REFERENCED_ONLY"})
_MAX_TEST_FILES = 2000


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    values = os.environ if environ is None else environ
    value = str(values.get(ENV, "on") or "on").strip().lower()
    if value not in {"on", "off"}:
        raise ValueError(f"{ENV} must be on or off")
    return value == "on"


def classify(*, entry_status: object = None, owner_writers: int | None = None,
             baseline_public: object = None, new_symbol: bool = False) -> dict[str, str]:
    absent = str(entry_status or "") in _ABSENT or bool(new_symbol)
    no_writer = owner_writers == 0
    if enabled() and (absent or no_writer) and str(baseline_public or "").upper() == "PASS":
        return {"role": NEW_BEHAVIOR, "witness_kind": ABSENCE_ON_PARENT}
    return {"role": REPAIR, "witness_kind": PUBLIC_RED}


def _row(runtime: object) -> Mapping[str, object]:
    row = getattr(getattr(runtime, "owner", None), "task_row", {})
    return row if isinstance(row, Mapping) else {}


def _binding(runtime: object) -> object:
    binding = getattr(runtime, "binding", None)
    try:
        return binding() if callable(binding) else binding
    except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        return None


def _test_files(root: Path) -> frozenset[str]:
    from kernel_request_source_window import _FORBIDDEN_PARTS, _TEST_PATH
    found: set[str] = set()
    for current, dirs, files in os.walk(root):
        dirs[:] = [item for item in dirs if not item.startswith(".")
                   and item not in {"node_modules", "__pycache__", ".venv", "venv"}
                   and item.casefold() not in _FORBIDDEN_PARTS]
        for name in files:
            relative = (Path(current) / name).relative_to(root).as_posix()
            if _TEST_PATH.search(relative):
                found.add(relative)
                if len(found) >= _MAX_TEST_FILES:
                    return frozenset(found)
    return frozenset(found)


def observe(runtime: object) -> None:
    """Record the parent binding/test files once, and a PASS baseline when seen at the parent."""
    if not enabled():
        return
    if not hasattr(runtime, "_row_role_parent_binding"):
        root = getattr(getattr(runtime, "owner", None), "root", None)
        try:
            runtime._row_role_parent_binding = _binding(runtime)
            runtime._row_role_parent_tests = (_test_files(Path(root).resolve())
                                              if root is not None else frozenset())
        except (AttributeError, OSError, TypeError, ValueError):
            return
    if getattr(runtime, "_row_role_baseline", None) is not None:
        return
    declared = str(_row(runtime).get("public_baseline") or "").upper()
    if declared in {"PASS", "FAIL"}:
        runtime._row_role_baseline = declared
        return
    parent = getattr(runtime, "_row_role_parent_binding", None)
    if parent is None or _binding(runtime) != parent:
        return
    try:
        from kernel_completion_owner import _green_current
        if _green_current(runtime):
            runtime._row_role_baseline = "PASS"
    except (ImportError, AttributeError, TypeError, ValueError):
        return


def runtime_role(runtime: object) -> dict[str, str]:
    entry = getattr(runtime, "_site_facts_entry", None)
    entry = entry if isinstance(entry, Mapping) else {}
    try:
        from kernel_state_owner_inventory import owner_writer_count
        writers = owner_writer_count(runtime)
    except ImportError:
        writers = None
    baseline = getattr(runtime, "_row_role_baseline", None) or _row(runtime).get("public_baseline")
    return classify(entry_status=entry.get("status"), owner_writers=writers,
                    baseline_public=baseline, new_symbol=bool(entry.get("new_symbol")))


def render(role: Mapping[str, str]) -> str | None:
    if role.get("role") != NEW_BEHAVIOR:
        return None
    return (f"{HEADER} {NEW_BEHAVIOR}: the entry is absent, has 0 product writers, or its behavior "
            "is not yet observed by the product checks. Green public tests are not a witness. "
            f"{ABSENCE_ON_PARENT} is one valid parent fact; an authored check may instead call an "
            "existing callable and record the required behavior. Acceptance = a typed authored "
            "check with a current candidate observation.")


def _trusted_witness_receipt(receipt: Mapping[str, object]) -> bool:
    """Reject model claims and unstable/non-zero candidate executions."""
    actor = str(receipt.get("actor", receipt.get("author", ""))).strip().casefold()
    if actor in {"model", "assistant", "solver", "user"}:
        return False
    if receipt.get("host_admitted") is False or receipt.get("admitted") is False:
        return False
    provenance = receipt.get("provenance")
    if isinstance(provenance, Mapping):
        kind = str(provenance.get("kind", "")).strip().upper()
        if kind in {"MODEL", "MODEL_CLAIM", "ASSISTANT_MESSAGE", "USER_MESSAGE"}:
            return False
    if receipt.get("binding_stable") is not True:
        return False
    role = receipt.get("role", receipt.get("row_role"))
    # Authored checks certify both row families.  REPAIR rows use a
    # parent-fail/candidate-pass pair; NEW_BEHAVIOR rows use the same typed
    # witness for an absence-on-parent contract.  Preservation is intentionally
    # not accepted here because it cannot close a repair obligation.
    if role not in (None, "", NEW_BEHAVIOR, REPAIR):
        return False
    result = str(receipt.get("outcome", receipt.get("result", ""))).upper()
    exit_code = receipt.get("exit_code")
    if result in {"PASS", "PASSED", "GREEN"} and type(exit_code) is int and exit_code != 0:
        return False
    return True


def _partitions_complete(runtime: object, receipt: Mapping[str, object]) -> bool:
    required = receipt.get("required_partitions")
    if required in (None, (), []):
        required = getattr(runtime, "_witness_required_partitions", ())
    observed = receipt.get("observed_partitions", receipt.get("lifecycle_partitions", ()))
    if required in (None, (), []):
        return True
    if not isinstance(required, (list, tuple, set, frozenset)):
        return False
    if not isinstance(observed, (list, tuple, set, frozenset)):
        return False
    return set(str(item) for item in required) <= set(str(item) for item in observed)


def _authored_witness(runtime: object) -> bool:
    values = getattr(runtime, "_row_role_authored_receipts", ())
    candidates = values.values() if isinstance(values, Mapping) else values
    rows = [item for item in candidates if isinstance(item, Mapping)] if candidates is not None else []
    current = _binding(runtime)
    current_revision = _revision(current)
    by_identity: dict[tuple[str, str, str], dict[str, Mapping[str, object]]] = {}
    for receipt in rows:
        if not _trusted_witness_receipt(receipt):
            continue
        provenance = receipt.get("provenance")
        if not isinstance(provenance, Mapping) or str(provenance.get("kind", "")).upper() != "AUTHORED_CHECK":
            continue
        check_identity = str(receipt.get("check_identity") or receipt.get("check_hash")
                             or receipt.get("check_id") or receipt.get("selector") or "").strip()
        clause = str(receipt.get("clause_id") or receipt.get("requirement_id")
                     or receipt.get("obligation_id") or "").strip()
        oracle = str(receipt.get("oracle_hash") or provenance.get("oracle_hash")
                     or receipt.get("check_hash") or "").strip()
        if not check_identity or not clause or not oracle:
            continue
        result = str(receipt.get("outcome", receipt.get("result", ""))).upper()
        reached = receipt.get("reached_observation") is True
        integrity = receipt.get("integrity") is True
        if not _partitions_complete(runtime, receipt):
            continue
        revision = receipt.get("revision") or receipt.get("product_revision")
        if isinstance(receipt.get("after_binding"), Mapping):
            revision = receipt["after_binding"].get("product_revision", revision)
        if (not reached or not integrity or revision in (None, "")
                or (result in {"PASS", "PASSED", "GREEN"}
                    and not _same_revision(revision, current_revision))):
            continue
        bucket = by_identity.setdefault((check_identity, clause, oracle), {})
        if result in {"BEHAVIOR_FAIL", "FAIL", "FAILED"}:
            bucket["parent"] = receipt
        if result in {"PASS", "PASSED", "GREEN"}:
            bucket["candidate"] = receipt
    if any("parent" in pair and "candidate" in pair for pair in by_identity.values()):
        return True

    # A single typed receipt may carry the explicitly paired parent/candidate
    # results when the adapter runs both sides in one isolated check.
    for receipt in rows:
        if not isinstance(receipt, Mapping):
            continue
        if not _trusted_witness_receipt(receipt):
            continue
        provenance = receipt.get("provenance")
        if not isinstance(provenance, Mapping) or str(provenance.get("kind", "")).upper() != "AUTHORED_CHECK":
            continue
        parent = receipt.get("parent_result")
        candidate = receipt.get("candidate_result")
        candidate_revision = (
            candidate.get("revision", candidate.get("product_revision"))
            if isinstance(candidate, Mapping) else None
        )
        current_revision = _revision(_binding(runtime))
        if (receipt.get("binding_stable") is True
                and _partitions_complete(runtime, receipt)
                and isinstance(parent, Mapping) and str(parent.get("outcome", "")).upper() in {"BEHAVIOR_FAIL", "FAIL"}
                and isinstance(candidate, Mapping) and str(candidate.get("outcome", "")).upper() in {"PASS", "GREEN"}
                and receipt.get("reached_observation") is True
                and receipt.get("integrity") is True
                and receipt.get("clause_id")
                and receipt.get("check_identity")
                and receipt.get("oracle_hash")
                and candidate_revision not in (None, "")
                and candidate.get("binding_stable") is True
                and str(parent.get("revision", "")) not in {"", str(current_revision)}
                and not (isinstance(candidate.get("exit_code"), int) and candidate["exit_code"] != 0)
                and _same_revision(candidate_revision, current_revision)):
            return True
    return False


def _revision(value: object) -> object:
    if isinstance(value, Mapping):
        return value.get("product_revision", value.get("artifact", value.get("revision")))
    return value


def _same_revision(left: object, right: object) -> bool:
    if left in (None, "") or right in (None, ""):
        return False
    return left == right or str(left) == str(right)


def certification_blocked(runtime: object) -> bool:
    """True when a NEW_BEHAVIOR row has only public-green evidence (no authored check)."""
    if not enabled():
        return False
    if runtime_role(runtime).get("role") != NEW_BEHAVIOR:
        return False
    return not _authored_witness(runtime)


def draft_role(runtime: object) -> dict[str, str]:
    """Role for the attach-time S1 host draft (cached facts only; never computes)."""
    return runtime_role(runtime)


__all__ = ["ABSENCE_ON_PARENT", "ENV", "HEADER", "NEW_BEHAVIOR", "PUBLIC_RED", "REPAIR",
           "certification_blocked", "classify", "draft_role", "enabled", "observe", "render",
           "runtime_role"]
