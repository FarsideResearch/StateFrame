"""Named-layer repair (r45; autopsy-a3b-r44c-ldev01.md + autopsy-ldev01-cross-release.md).

``STATEFRAME_NAMED_LAYER_REPAIR`` (strict ``off|on``, default ``off``; frame arms only).  With the
switch off nothing in this module is reached and every caller keeps its byte-identical r45 path.

ldev_01 lost four releases running on ONE missing layer (``app/cli.py``).  The kernel had the proof
in hand every time and overrode the model instead of acting on it:

A. Signature gap is a counterexample.  The host check executed ``app.cli:run`` with the ticket's
   fourth argument; the entry point RESOLVED and ``inspect.Signature.bind`` raised "too many
   positional arguments".  ``bind_target`` collapsed that into ``UNBOUND`` and the pipeline demoted
   it to ``INVALID_BINDING`` (not repairable).  On: a resolved target whose arguments do not bind
   is ``SIGNATURE_MISMATCH``; when the row is grounded in a ticket literal it becomes the repairable
   ``SIGNATURE_GAP`` advisory.  A genuine harness mis-binding (unresolved entry point, no literal)
   stays ``INVALID_BINDING``.
B. A confidently-untouched NAMED layer keeps edit/read admitted through the post-green rule
   (``kernel_post_green_menu.decide``, the one allow predicate readiness defers to).
C. get_sure renders a structural named-layer clause as UNPROBED bound to the layer row, never
   "INVALID CHECK, IGNORE".
D. Named-layer coverage is enforced on a done-only menu by reopening the post-green menu once per
   revision (``kernel_post_green_menu.reopen_instead_of_sole_submission``) with ``read <file>``.
"""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

MODE_ENV = "STATEFRAME_NAMED_LAYER_REPAIR"
PLUGGED_KEY = "stateframe_named_layer_repair"
SIGNATURE_MISMATCH = "SIGNATURE_MISMATCH"
SIGNATURE_GAP = "SIGNATURE_GAP"
RULE_NAMED_LAYER = "post_green_named_layer"
# Only these tools are admitted for an untouched named layer (the edit that adds it, the read that
# inspects it).  run_tests after the edit is admitted by the existing pending-verification rule.
NAMED_LAYER_ACTIONS = frozenset({"edit", "read"})


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = (source.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return mode(environ) == "on"


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


# --------------------------------------------------------------------------- A: signature gap

def _binding_rows(receipt: Mapping[str, object]) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    for key in ("observed", "observed_values"):
        value = receipt.get(key)
        if isinstance(value, Mapping):
            rows.append(value)
    counterexample = receipt.get("counterexample")
    if isinstance(counterexample, Mapping):
        rows.append(counterexample)
        for key in ("observed", "detail", "arity_error"):
            value = counterexample.get(key)
            if isinstance(value, Mapping):
                rows.append(value)
    return rows


def signature_mismatch(receipt: object) -> str | None:
    """The binding error when the executor resolved the target but the call did not bind."""
    if not isinstance(receipt, Mapping):
        return None
    for row in _binding_rows(receipt):
        if _text(row.get("binding")).upper() == SIGNATURE_MISMATCH:
            return _text(row.get("binding_error")) or "the call does not bind to the signature"
    return None


def literal_grounded(row: object) -> bool:
    """Host ticket-literal alignment (kernel_requirement_alignment.enrich_counterexample)."""
    if not isinstance(row, Mapping) or not bool(row.get("literal_grounded")):
        return False
    evidence = row.get("alignment_evidence")
    if not isinstance(evidence, Mapping):
        return False
    literals = evidence.get("matched_literals")
    if isinstance(literals, str):
        return bool(literals.strip())
    return (isinstance(literals, Sequence) and not isinstance(literals, (bytes, bytearray))
            and any(_text(item) for item in literals))


def is_signature_gap(row: object, receipt: object = None) -> bool:
    """A resolved entry point rejected a ticket-literal call: repairable, not a harness fault."""
    if not isinstance(row, Mapping):
        return False
    source = receipt if isinstance(receipt, Mapping) else row
    error = signature_mismatch(source)
    if error is None or not literal_grounded(row):
        return False
    # A literal match alone is not enough. Require (a) a SUPPORTED call form -- the entry
    # point rejected extra positional/keyword input the ticket supplies (not a harness mis-call such as a missing
    # required argument), and (b) provenance that the ticket REQUIRES it (an aligned requirement id).  Otherwise
    # the row stays advisory and cannot spend the one done challenge on a fabricated call.
    return _supported_call_form(error) and _ticket_provenance(row)


_SUPPORTED_CALL_ERRORS = ("too many positional arguments", "takes", "unexpected keyword argument",
                          "got an unexpected keyword")


def _supported_call_form(error: str) -> bool:
    text = _text(error).lower()
    if "missing" in text and "required" in text:
        return False
    return any(marker in text for marker in _SUPPORTED_CALL_ERRORS)


def _ticket_provenance(row: Mapping[str, object]) -> bool:
    if _text(row.get("alignment_requirement_id")):
        return True
    evidence = row.get("alignment_evidence")
    if isinstance(evidence, Mapping):
        return bool(_text(evidence.get("requirement_id")) or _text(evidence.get("clause_id"))
                    or _text(evidence.get("requirement")))
    return False


def signature_gap_text(row: Mapping[str, object], error: str) -> str:
    symbol = _text(row.get("target_symbol")) or _text(row.get("entry_point")) or "the entry point"
    target = _text(row.get("target_file"))
    where = f"{symbol} ({target})" if target else symbol
    return (f"{where} rejects the ticket's call: {error}. The ticket requires this call to "
            "work; change the signature (keep existing calls valid).")


def apply_signature_gap(row: Mapping[str, object], receipt: Mapping[str, object]) -> dict[str, object]:
    """Collector hook: reclassify a literal-grounded signature mismatch as ``SIGNATURE_GAP``."""
    result = dict(row)
    error = signature_mismatch(receipt)
    if error is None or not literal_grounded(result):
        return result
    result["demotion_reason"] = SIGNATURE_GAP
    result["classification"] = SIGNATURE_GAP
    result["binding"] = "BOUND"          # the entry point resolved; its signature is the product gap
    result["signature_gap"] = signature_gap_text(result, error)
    return result


# --------------------------------------------------------------------------- B/C/D: named layers

def untouched_rows(runtime: object) -> tuple[Mapping[str, object], ...]:
    """Confident named-layer rows unchanged since start (kernel_named_layer_coverage.missing_rows)."""
    provider = getattr(runtime, "layer_coverage_report", None)
    if not callable(provider):
        return ()
    try:
        report = provider()
    except (OSError, TypeError, ValueError, AttributeError):
        return ()
    from kernel_named_layer_coverage import missing_rows
    return missing_rows(report)


def untouched_files(runtime: object) -> tuple[str, ...]:
    files: list[str] = []
    for row in untouched_rows(runtime):
        for item in row.get("files", ()) or ():
            text = _text(item).replace("\\", "/")
            if text and text not in files:
                files.append(text)
    return tuple(files)


def _relative(runtime: object, path: object) -> str:
    text = _text(path).replace("\\", "/")
    root = _text(getattr(getattr(runtime, "owner", None), "root", "")).replace("\\", "/").rstrip("/")
    if root and text.startswith(root + "/"):
        text = text[len(root) + 1:]
    while text.startswith("./"):
        text = text[2:]
    return text


def admits(runtime: object, action: str, path: object = None) -> bool:
    """Rule B for the post-green allow predicate.  ``path=None`` = menu (some argument works)."""
    if action not in NAMED_LAYER_ACTIONS:
        return False
    files = untouched_files(runtime)
    if not files:
        return False
    if path is None:
        return True
    return _relative(runtime, path) in files


def layer_rows_for_clause(clause: str, rows: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    """Report rows whose layer this clause names (C)."""
    from kernel_named_layer_coverage import extract_named_layers
    try:
        layers = set(extract_named_layers(clause))
    except (TypeError, ValueError):
        return []
    return [row for row in rows if isinstance(row, Mapping) and row.get("layer") in layers
            and row.get("confidence") == "confident"]


def bind_invalid_named_clauses(table: object, report: object) -> object:
    """C: an INVALID get_sure row whose clause names a layer becomes UNPROBED bound to that row."""
    from kernel_stateframe_get_sure import INVALID_VERDICT, ClauseTable
    if not isinstance(table, ClauseTable) or not isinstance(report, Mapping):
        return table
    report_rows = [row for row in report.get("rows", ()) or () if isinstance(row, Mapping)]
    if not report_rows:
        return table
    changed = False
    rows: list[dict[str, object]] = []
    for row in table.rows:
        bound = (layer_rows_for_clause(_text(row.get("clause")), report_rows)
                 if row.get("verdict") == INVALID_VERDICT else [])
        if not bound:
            rows.append(row)
            continue
        changed = True
        rows.append({"id": row.get("id"), "clause": row.get("clause"), "verdict": "UNPROBED",
                     "layers": [{"layer": item.get("layer"), "files": list(item.get("files", ()) or ()),
                                 "diff_from_start": item.get("diff_label", "unknown")}
                                for item in bound]})
    if not changed:
        return table
    return ClauseTable(table.ticket, tuple(rows), table.revision)


def layer_text(row: Mapping[str, object]) -> str:
    """get_sure row suffix for a layer-bound UNPROBED clause."""
    parts = []
    for item in row.get("layers", ()) or ():
        if isinstance(item, Mapping):
            files = ", ".join(_text(name) for name in item.get("files", ()) or ()) or "unknown"
            parts.append(f"{item.get('layer')} -> {files} (diff-from-start: {item.get('diff_from_start')})")
    return ("named layer: " + "; ".join(parts)) if parts else ""


EVENT = "named_layer_repair_applied"


def emit(runtime: object, rule: str, stage: str = "acted", **fields: object) -> None:
    """Own event (fired proof).  ``rule``: signature_gap / post_green_admit / get_sure_bind / done_reopen;
    ``stage``: evaluated / applicable / acted.  Falls back to the active runtime's journal; never raises."""
    if not enabled():
        return
    journal = getattr(runtime, "journal", None)
    if journal is None:
        try:
            from kernel_pilot import ACTIVE_RUNTIME
            journal = getattr(ACTIVE_RUNTIME.get(), "journal", None)
        except (ImportError, LookupError):
            journal = None
    emit_fn = getattr(journal, "emit", None)
    if not callable(emit_fn):
        return
    try:
        emit_fn(EVENT, producer="kernel_named_layer_repair", dispatch=False, rule=rule, stage=stage, **fields)
    except Exception:  # noqa: BLE001 - telemetry never changes behaviour
        return


__all__ = [
    "EVENT", "emit",
    "MODE_ENV", "NAMED_LAYER_ACTIONS", "PLUGGED_KEY", "RULE_NAMED_LAYER", "SIGNATURE_GAP",
    "SIGNATURE_MISMATCH", "admits", "apply_signature_gap", "bind_invalid_named_clauses", "enabled",
    "is_signature_gap", "layer_rows_for_clause", "layer_text", "literal_grounded", "mode",
    "signature_gap_text", "signature_mismatch", "untouched_files", "untouched_rows",
]
