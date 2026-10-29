"""r41b: a live counterexample on the current revision challenges ``done`` once.

r41 dissection (``dissection-r41-unchallenged-done.md``): first-green disclosure
showed open counterexample IDs to the model (broad collection), but admission
only refused ``done`` when one of the narrower gates (requirement alignment's
literal-repairable filter, readiness blockers, green-lock repair, get-sure)
fired.  The disclosed rows were typed repairable ADVISORIES, so the reducer
accepted, ``kernel_done_challenge_once.arbitrate`` took its natural-acceptance
fast path, and in 10/10 smokes no done was ever challenged.

This module supplies the ONE shared definition of "live counterexample on the
current revision" used by both arms when the switch is on:

* frame: ``StateFrameRuntime._completion_decision_core`` marks an otherwise
  admissible done inadmissible with ``OPEN_COUNTEREXAMPLE`` BEFORE the reducer
  seals, so the refusal flows through the existing one-shot challenge path of
  ``kernel_done_challenge_once.arbitrate`` (challenge once per revision, second
  done after one non-redundant action, run-wide cap/budget authoritative,
  sole-done menu submits).
* H+state+: ``kernel_stateframe_hstate_plus.done_decision`` uses these rows when
  its aligned-clause filter finds none, so its own budgeted challenge applies.

Switch: ``STATEFRAME_LIVE_CE_CHALLENGE=off|on`` (default off; off is
byte-identical: nothing here is called).
"""
from __future__ import annotations

import copy
import os
from collections.abc import Mapping, Sequence

MODE_ENV = "STATEFRAME_LIVE_CE_CHALLENGE"
REASON_CODE = "OPEN_COUNTEREXAMPLE"
# Not in kernel_done_challenge_once._COUNTEREXAMPLE_CODES: its challenge class is UNVERIFIED.
UNVERIFIED_CODE = "UNVERIFIED_DISCLOSED_CHECKS"
UNVERIFIED_LABEL = "[unverified checks]"
_BINDING_INVALID = frozenset({
    "not_entailed", "wrong_binding", "invalid_fixture", "wrong_observable",
    "conflicting_source", "check_invalid", "check_rebound", "self_contradictory",
    "unbound_entrypoint", "unbound_invocation", "contradicts_public_test",
    "unsatisfiable_contract_cell", "invalid_binding", "ambiguous", "proposed",
    # oracle grounding: the expected value is not bound to the ticket -> CHECK_INCONCLUSIVE
    "unbound_observable",
})
_SIGNATURE_MISMATCH = "SIGNATURE_MISMATCH"
_SIGNATURE_GAP = "SIGNATURE_GAP"
_PASS = frozenset({"PASS", "PASSED", "SATISFIED", "RESOLVED", "GREEN", "OK"})
_VIOLATION = frozenset({"COUNTEREXAMPLE", "VIOLATED", "VIOLATION"})


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = (source.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(runtime: object | None = None) -> bool:
    configured = getattr(runtime, "stateframe_live_ce_challenge_mode", None)
    if configured is not None:
        return str(configured).strip().lower() == "on"
    return mode() == "on"


def _text(value: object) -> str:
    return str(value).strip() if value not in (None, "") else ""


def _status(row: Mapping[str, object]) -> str:
    return _text(row.get("result_class") or row.get("status") or row.get("verdict")
                 or row.get("result") or row.get("underlying_status")).upper()


def _ident(row: Mapping[str, object]) -> str:
    return _text(row.get("check_id") or row.get("obligation_id") or row.get("clause_id")
                 or row.get("id"))


def _is_signature_gap(row: Mapping[str, object]) -> bool:
    return any(_text(row.get(key)).upper() == _SIGNATURE_GAP
               for key in ("classification", "demotion_reason", "classification_reason"))


def _binding_invalid(row: Mapping[str, object]) -> bool:
    for key in ("demotion_reason", "classification", "classification_reason"):
        value = _text(row.get(key))
        if value.upper() == _SIGNATURE_MISMATCH:
            return not _is_signature_gap(row)
        if value.lower() in _BINDING_INVALID:
            return True
    binding = _text(row.get("binding")).upper()
    if binding == _SIGNATURE_MISMATCH:
        return not _is_signature_gap(row)
    return binding in {"UNBOUND", "INVALID", "WRONG"}


def _as_violation(row: Mapping[str, object]) -> dict[str, object]:
    """Stamp the row so the challenge budget classifies it ``known_violation``."""
    result = copy.deepcopy(dict(row))
    ident = _ident(result)
    result.setdefault("check_id", ident)
    result.setdefault("clause_id", ident)
    result["underlying_status"] = _status(row) or "COUNTEREXAMPLE"
    result["status"] = "COUNTEREXAMPLE"
    result["live_counterexample"] = True
    return result


def _merge_witnesses(witnesses: Sequence[Mapping[str, object]]) -> list[dict]:
    """One merged witness per check: the green-menu row (observed/expected) and the collector row
    (classification/demotion) describe the same check; a demotion must never be lost."""
    merged: dict[str, dict] = {}
    for row in witnesses:
        if not isinstance(row, Mapping):
            continue
        key = _text(row.get("check_id")) or _text(row.get("obligation_id"))
        if key:
            merged[key] = {**merged.get(key, {}),
                           **{k: v for k, v in row.items() if v not in (None, "", [], {})}}
    return list(merged.values())


def demotion_reason(row: Mapping[str, object]) -> str:
    for key in ("demotion_reason", "classification", "classification_reason"):
        value = _text(row.get(key))
        if value.upper() == _SIGNATURE_GAP:
            return _SIGNATURE_GAP
        if value.lower() in _BINDING_INVALID:
            return value.upper()
    binding = _text(row.get("binding")).upper()
    if binding == _SIGNATURE_MISMATCH:
        return "UNBOUND"
    return "UNBOUND" if binding in {"UNBOUND", "INVALID", "WRONG"} else ""


def classify_disclosed(ids: Sequence[str], witnesses: Sequence[Mapping[str, object]]
                       ) -> tuple[list[dict], list[dict]]:
    """Split disclosed IDs into live counterexamples and host-demoted UNVERIFIED checks.

    The obligation id of an already-listed check is the same item (one row, not two).  An ID
    with no witness row stays a live counterexample (the disclosure named it as one)."""
    rows = _merge_witnesses(witnesses)
    by_id: dict[str, int] = {}
    for index, row in enumerate(rows):
        for key in (_text(row.get("check_id")), _text(row.get("obligation_id"))):
            if key:
                by_id.setdefault(key, index)
    live, unverified, used = [], [], set()
    for ident in ids:
        index = by_id.get(ident)
        if index is not None and index in used:
            continue
        row = dict(rows[index]) if index is not None else {"check_id": ident}
        if index is not None:
            used.add(index)
        row["check_id"] = _text(row.get("check_id")) or ident
        if _binding_invalid(row):
            # deep-dive 4b: keep the witness fields (path/target/observed/expected) through demotion
            # so a binder can use them; check_id/reason stay the authoritative keys.
            unverified.append({**{k: v for k, v in row.items() if k not in {"check_id", "reason"}},
                               "check_id": row["check_id"], "reason": demotion_reason(row)})
        else:
            live.append(row)
    return live, unverified


def _disclosed(runtime: object, frame: Mapping[str, object] | None
               ) -> tuple[list[dict], list[dict]]:
    """(live, unverified) for the IDs first-green disclosure showed for the CURRENT product
    revision, with witnesses captured at disclosure time (the open set is frame-scoped and is
    not recomputable at done time)."""
    try:
        from kernel_first_green_disclosure import disclosed_open
        ids, witnesses = disclosed_open(runtime, frame)
    except (ImportError, AttributeError, TypeError, ValueError, OSError):
        return [], []
    return classify_disclosed(ids, witnesses)


def _disclosed_rows(runtime: object, frame: Mapping[str, object] | None) -> list[dict]:
    return _disclosed(runtime, frame)[0]


def unverified_rows(runtime: object, frame: Mapping[str, object] | None = None) -> list[dict]:
    """Disclosed-but-host-demoted checks on the current revision (INVALID_BINDING,
    UNBOUND_OBSERVABLE, not_entailed, ...).  They feed the SAME one-shot done challenge as class
    UNVERIFIED, never COUNTEREXAMPLE (r41 dissection: winners were pushed back once on them)."""
    return _disclosed(runtime, frame)[1]


_REASON_TEXT = {
    "INVALID_BINDING": "the check could not bind to the code it names",
    "UNBOUND_OBSERVABLE": "its expected value is not stated in the ticket",
    "NOT_ENTAILED": "the ticket does not entail its expectation",
    "UNBOUND_ENTRYPOINT": "its entry point could not be found",
    "UNBOUND_INVOCATION": "its invocation could not be bound",
    "UNBOUND": "the check could not bind to the code it names",
}


# Tools the green-lock terminal menu keeps beside ``done`` while the unverified-check window is
# open (orchestrator option 1).  ``edit`` stays out: first-green rules decide edits.
UNVERIFIED_WINDOW_TOOLS = frozenset({"read", "probe"})
# Live smokescratch-r41b-ldev_01 (1481d2): with the window open and no done yet, the model spent
# all 24 responses on read/probe and never called done.  The window is bounded per revision:
# after this many window actions without a done it closes (done-only -> sole done submits).
UNVERIFIED_WINDOW_ACTION_CAP = 2


def _window_revision(runtime: object) -> str:
    try:
        from kernel_challenge_budget import artifact_revision
        return str(artifact_revision(runtime))
    except (ImportError, OSError, TypeError, ValueError):
        return ""


def note_window_action(runtime: object) -> int:
    """Count one read/probe taken through the window at the current revision."""
    key = _window_revision(runtime)
    counts = getattr(runtime, "_unverified_window_actions", None)
    if not isinstance(counts, dict):
        counts = {}
        setattr(runtime, "_unverified_window_actions", counts)
    counts[key] = int(counts.get(key, 0)) + 1
    return counts[key]


def unverified_window_open(runtime: object, frame: Mapping[str, object] | None = None) -> bool:
    """True while disclosed unverified checks are open on the current revision AND this
    revision's one-shot challenge is unspent and the run-wide cap/budget allows one.

    Once the challenge is spent (or the cap is hit) the terminal menu returns to done-only,
    and "a done-only menu always submits" (r39 deadlock protection) takes over unchanged."""
    if not enabled(runtime):
        return False
    try:
        from kernel_first_green_disclosure import enabled as fgd_enabled
        if not fgd_enabled(runtime):
            return False
    except (ImportError, AttributeError, TypeError, ValueError):
        return False
    if not unverified_rows(runtime, frame) or _disclosed_rows(runtime, frame):
        return False
    counts = getattr(runtime, "_unverified_window_actions", None)
    if isinstance(counts, dict) and int(counts.get(_window_revision(runtime), 0)) >= UNVERIFIED_WINDOW_ACTION_CAP:
        return False
    try:
        from kernel_challenge_budget import artifact_revision, budget_for, plan
        if budget_for(runtime) is not None:
            view = plan(runtime, artifact_revision(runtime), request=None) or {}
            return view.get("action") == "challenge"
    except (ImportError, OSError, TypeError, ValueError):
        return False
    try:
        from kernel_done_challenge_once import CHALLENGE_RUN_CAP, revision_key
        spent = int(getattr(runtime, "_done_challenge_count", 0) or 0)
        key = revision_key(None, runtime)
        return (spent < CHALLENGE_RUN_CAP
                and getattr(runtime, "_done_challenge_revision", None) != key)
    except (ImportError, AttributeError, TypeError, ValueError):
        return False


WINDOW_NOTICE_SUFFIX = ("You have up to 2 read/probe actions to check these before submitting. "
                        "Do not change correct code merely to agree with this notice.")


def window_notice(runtime: object, frame: Mapping[str, object] | None = None) -> str | None:
    """Orchestrator decision: show the UNVERIFIED challenge PROACTIVELY while the window is open
    (no refusal needed); None when the window is closed."""
    if not unverified_window_open(runtime, frame):
        return None
    rows = unverified_rows(runtime, frame)
    if not rows:
        return None
    suffix = f" {WINDOW_NOTICE_SUFFIX}" if budget_applies(rows, runtime) else ""  # M1
    return f"{unverified_detail(rows, runtime)}{suffix}"


def reason_text(reason: object) -> str:
    key = _text(reason).upper()
    return _REASON_TEXT.get(key, key.lower().replace("_", " ") or "the host could not verify it")


def budget_applies(rows: Sequence[Mapping[str, object]], runtime: object = None) -> bool:
    """M1: no budget for host-unobservable or NOT_ENTAILED rows."""
    from kernel_tokenlab_v12 import (NOT_ENTAILED_REASONS, UNBOUND_REASONS,
                                     challenge_split_enabled)
    if not challenge_split_enabled():
        return True
    non_actionable = UNBOUND_REASONS | NOT_ENTAILED_REASONS
    return any(_text(row.get("reason")).upper() not in non_actionable for row in rows)


def unverified_detail(rows: Sequence[Mapping[str, object]], runtime: object = None) -> str:
    from kernel_tokenlab_v12 import challenge_split_enabled, split_unverified
    if challenge_split_enabled():
        # M1: split by reason -- unobservable rows get the QUIET sentence, arity checks are bound by
        # the host from the current source, the rest keeps the legacy text.  Off: the legacy text below.
        text, _budget, _counts = split_unverified(rows, runtime, reason_text)
        return text
    items = "; ".join(f"{_text(row.get('check_id'))} ({reason_text(row.get('reason'))})"
                      for row in rows)
    return (f"{UNVERIFIED_LABEL} {items}: these checks could not be verified. This is not a "
            "demonstrated failure. Check the named behaviour yourself before submitting.")


def label_ids(ids: Sequence[str], witnesses: Sequence[Mapping[str, object]]) -> str:
    """Model-visible disclosure body: real counterexamples and unverified checks are labelled
    separately (the r41 disclosure called host-demoted check problems "open counterexamples")."""
    import json as _json
    live, unverified = classify_disclosed(ids, witnesses)
    parts = [f"open counterexamples {_json.dumps([row['check_id'] for row in live], ensure_ascii=False, separators=(',', ':'))}"]
    if unverified:
        parts.append(f"unverified checks {_json.dumps([row['check_id'] for row in unverified], ensure_ascii=False, separators=(',', ':'))}"
                     " (could not be verified; not demonstrated failures)")
    return "; ".join(parts)


def _green_menu_rows(runtime: object, frame: Mapping[str, object] | None) -> list[dict]:
    try:
        from kernel_green_menu_open_obligation import open_counterexamples
        values = open_counterexamples(runtime, frame)
    except (ImportError, AttributeError, TypeError, ValueError, OSError):
        return []
    result = []
    for item in values:
        to_dict = getattr(item, "to_dict", None)
        row = to_dict() if callable(to_dict) else item
        if isinstance(row, Mapping) and _ident(row):
            result.append(dict(row))
    return result


def _collector_rows(runtime: object, frame: Mapping[str, object] | None) -> list[dict]:
    try:
        from kernel_counterexample_visibility import collect_counterexamples
        values = collect_counterexamples(runtime, frame)
    except (ImportError, AttributeError, TypeError, ValueError, OSError):
        return []
    return [dict(row) for row in values if isinstance(row, Mapping) and _ident(row)]


def _merged_open_rows(runtime: object, frame: Mapping[str, object] | None) -> list[dict]:
    """Green-menu rows (witness) merged with collector rows (classification) per check id;
    binding-invalid/ungrounded checks dropped.  The green-menu row alone carries no demotion,
    so filtering it unmerged would admit INVALID_BINDING checks as violations."""
    merged: dict[str, dict] = {}
    for row in (*_green_menu_rows(runtime, frame), *_collector_rows(runtime, frame)):
        key = _text(row.get("check_id")) or _text(row.get("obligation_id"))
        if key:
            merged[key] = {**merged.get(key, {}),
                           **{k: v for k, v in row.items() if v not in (None, "", [], {})}}
    return [row for row in merged.values() if not _binding_invalid(row)]


def live_rows(runtime: object, frame: Mapping[str, object] | None = None,
              *, receipts: Sequence[Mapping[str, object]] = ()) -> tuple[dict[str, object], ...]:
    """Current-revision executed counterexamples, stamped as known violations.

    Source order matches first-green disclosure (``open_counterexample_ids``):
    the green-menu open set first (with readiness authority on: blockers plus
    typed repairable advisories), then the counterexample-visibility collector
    (binding-invalid rows excluded: a check problem is not a product failure).
    ``receipts`` lets H+state+ add its own receipt rows that carry an explicit
    violation verdict and a witness; a later PASS of the same id closes them.
    """
    rows = _disclosed_rows(runtime, frame) or _merged_open_rows(runtime, frame)
    closed_after: dict[str, int] = {}
    receipt_rows = [dict(item) for item in receipts if isinstance(item, Mapping)]
    # The live adversary is a host receipt producer.  Keep this additive so
    # older callers and the switch-off path remain byte-compatible while an
    # executed S3/S5 counterexample can reach this existing consumer.
    adversary_receipts = getattr(runtime, "_adversary_counterexample_receipts", ())
    receipt_rows.extend(dict(item) for item in adversary_receipts
                        if isinstance(item, Mapping))
    for position, row in enumerate(receipt_rows):
        if _ident(row) and _status(row) in _PASS:
            closed_after[_ident(row)] = position
    for position, row in enumerate(receipt_rows):
        if (_status(row) in _VIOLATION and _ident(row)
                and row.get("expected", row.get("want")) is not None
                and closed_after.get(_ident(row), -1) <= position
                and not _binding_invalid(row)):
            rows.append(row)
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        ident = _ident(row)
        if not ident or ident in seen:
            continue
        seen.add(ident)
        result.append(_as_violation(row))
    return tuple(result)


def diagnose(runtime: object, rows: Sequence[Mapping[str, object]],
             frame: Mapping[str, object] | None = None) -> dict[str, object]:
    """Plugged-in proof for every gate evaluation: distinguishes "no live counterexample"
    from "the sources were empty/broken" (a should-be-non-null check, not a guess)."""
    result: dict[str, object] = {"live_ids": [_ident(row) for row in rows]}
    record = getattr(runtime, "_fgd_disclosed_open", None)
    result["disclosure_record"] = isinstance(record, Mapping)
    if isinstance(record, Mapping):
        result["disclosed_ids"] = [str(item) for item in record.get("ids") or ()]
        result["disclosed_revision"] = str(record.get("revision"))
        result["witness_rows"] = len(record.get("rows") or ())
        result["capture_error"] = record.get("capture_error")
        dropped = []
        for row in record.get("rows") or ():
            if isinstance(row, Mapping) and _binding_invalid(row):
                dropped.append({"id": _ident(row),
                                "reason": _text(row.get("demotion_reason")
                                                or row.get("classification"))})
        result["dropped_binding_invalid"] = dropped[:12]
    try:
        from kernel_first_green_disclosure import _product_revision
        result["current_revision"] = str(_product_revision(runtime, frame))
    except (ImportError, AttributeError, TypeError, ValueError, OSError) as exc:
        result["current_revision"] = f"error:{type(exc).__name__}"
    return result


def refusal_detail(rows: Sequence[Mapping[str, object]]) -> str:
    """The witness text the frame refusal carries (the challenge suffix follows)."""
    parts = []
    for row in rows:
        observed = row.get("observed", row.get("actual"))
        expected = row.get("expected", row.get("want"))
        witness = ""
        if observed is not None or expected is not None:
            witness = f" (observed={_text(observed) or '?'} expected={_text(expected) or '?'})"
        parts.append(f"{_ident(row)}{witness}")
    return ("open counterexample on the current revision: " + "; ".join(parts)
            + ". A host check produced this witness.")


__all__ = ["MODE_ENV", "REASON_CODE", "UNVERIFIED_CODE", "UNVERIFIED_LABEL", "classify_disclosed",
           "enabled", "label_ids", "live_rows", "mode", "reason_text", "refusal_detail",
           "unverified_detail", "unverified_rows"]
