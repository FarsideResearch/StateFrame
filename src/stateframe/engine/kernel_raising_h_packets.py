"""Opt-in Raising-H failure packets (R-123).

Three mechanisms, all host-owned and behind ``STATEFRAME_RAISING_H_PACKETS=on``:

* evidence-rich failure packets: what failed, observed vs expected, where;
* partial-progress preservation: source snapshots are captured when a host
  test observation completes on a stable binding (revision-bound), later
  evidence merges into the same snapshot, and a snapshot with passing evidence
  is protected and never rotated out;
* complementary exploration: the next attempt names a different observable
  operation than the failed one.

Rendering is bounded section by section: every required section survives
compaction, the selected ``next_attempt`` is rendered verbatim, and anything
omitted is named by a revision-bound reference (snapshot id / receipt id).
Nothing here runs on the normal path while the switch is off.
"""
from __future__ import annotations

import copy
import difflib
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence


ENV = "STATEFRAME_RAISING_H_PACKETS"
SCHEMA = "stateframe-raising-h-packets/2"
FAILURE_SCHEMA = "stateframe-raising-h-failure/2"
PARTIAL_SCHEMA = "stateframe-raising-h-partial/2"
EXPLORATION_SCHEMA = "stateframe-raising-h-exploration/1"
HEADER = "[RAISING-H FAILURE PACKET]"
FOOTER = "[/RAISING-H FAILURE PACKET]"
REQUIRED_SECTIONS = ("WHAT FAILED", "OBSERVED VS EXPECTED", "WHERE", "CURRENT REVISION",
                     "PRESERVED PARTIAL PROGRESS", "COMPLEMENTARY EXPLORATION")
MAX_CANDIDATES = 4
MAX_PACKET_CHARS = 6000
MIN_PACKET_CHARS = 700
SNAPSHOT_REF_PREFIX = "raising-h-snapshot:"

_FAILURE_CLASSES = frozenset({
    "ERROR", "FAIL", "FAILED", "FAILURE", "RED", "COUNTEREXAMPLE",
    "REJECTED", "REFUSED", "REFUSED_PREDISPATCH", "BLOCKED", "NOT_RUN",
    "STALE", "CONFLICT", "MISMATCH", "TIMEOUT", "EXCEPTION",
    "HARNESS_ERROR", "SETUP_ERROR", "NOT_EXECUTED",
})
_PASS_CLASSES = frozenset({"PASS", "PASSED", "GREEN", "SATISFIED"})


def raising_h_mode() -> str:
    """Return the strict, opt-in switch value."""
    value = os.environ.get(ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{ENV} must be off or on")
    return value


def enabled() -> bool:
    return raising_h_mode() == "on"


def _clone(value):
    return copy.deepcopy(value)


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)


def _display(value) -> str:
    return value if isinstance(value, str) else _json(value)


def _digest(value) -> str:
    return "sha256:" + hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _text(value, default: str = "") -> str:
    return default if value is None else str(value)


def _where(failure: Mapping[str, object]) -> dict:
    value = failure.get("where")
    if isinstance(value, Mapping):
        return _clone(dict(value))
    result = {}
    for source, target in (("path", "path"), ("target_path", "path"),
                           ("symbol", "symbol"), ("line", "line"),
                           ("column", "column"), ("anchor", "anchor")):
        if source in failure and failure[source] not in (None, "") and target not in result:
            result[target] = _clone(failure[source])
    return result


def _location_value(value: object) -> dict:
    if isinstance(value, Mapping):
        return _clone(dict(value))
    if not isinstance(value, str) or not value.strip():
        return {}
    text = value.strip()
    result = {"anchor": text}
    match = re.match(r"^(?P<path>.+?)(?::(?P<line>\d+))?$", text)
    if match:
        result["path"] = match.group("path")
        if match.group("line") is not None:
            result["line"] = int(match.group("line"))
    return result


def failure_path(failure: Mapping[str, object]) -> str:
    """Return the stable causal path used to avoid repeating an attempt."""
    explicit = failure.get("failure_key")
    if explicit:
        return str(explicit)
    where = _where(failure)
    parts = (_text(where.get("path"), "unknown"), _text(where.get("symbol"), ""),
             _text(where.get("line"), ""),
             _text(failure.get("operation") or failure.get("kind"), ""))
    return ":".join(part for part in parts if part)


# ------------------------------------------------------- partial progress
def evidence_passes(item: object) -> bool:
    """Production success shapes: result_class/status/result PASS, or a
    stable test subprocess receipt with ``exit_code == 0``."""
    if not isinstance(item, Mapping):
        return False
    for key in ("result_class", "status", "result", "outcome"):
        if str(item.get(key) or "").upper() in _PASS_CLASSES:
            return True
    exit_code = item.get("exit_code")
    return (type(exit_code) is int and exit_code == 0
            and item.get("binding_stable", True) is not False)


def _evidence_identity(item: object) -> str:
    if isinstance(item, Mapping) and item.get("receipt_id") not in (None, ""):
        return str(item["receipt_id"])
    return _digest(item)


class PartialProgressLedger:
    """Revision-bound source snapshots retained across repair attempts.

    Unprotected snapshots rotate through a window of ``max_candidates``.
    Protected snapshots (any passing evidence) live outside the window and are
    never evicted: losing a correct partial edit is the failure being fixed.
    """

    def __init__(self, *, max_candidates: int = MAX_CANDIDATES):
        self.max_candidates = max(1, int(max_candidates))
        self._rotating: list[dict] = []
        self._protected: list[dict] = []

    def _find(self, identity: str) -> dict | None:
        return next((row for row in (*self._protected, *self._rotating)
                     if row["candidate_id"] == identity), None)

    def record_candidate(self, *, revision: str, files: Mapping[str, object],
                         evidence: Sequence[object] = (), label: str = "") -> dict:
        if not isinstance(files, Mapping):
            raise TypeError("files must be a mapping")
        safe_files = _clone(dict(files))
        identity = _digest({"revision": str(revision), "files": safe_files})
        candidate = self._find(identity)
        if candidate is None:
            candidate = {"schema": PARTIAL_SCHEMA, "candidate_id": identity,
                         "revision": str(revision), "files": safe_files,
                         "evidence": [], "label": str(label), "protected": False}
            self._rotating.append(candidate)
        seen = {_evidence_identity(row) for row in candidate["evidence"]}
        for item in evidence:
            if _evidence_identity(item) not in seen:
                candidate["evidence"].append(_clone(item))
                seen.add(_evidence_identity(item))
        if not candidate["protected"] and any(map(evidence_passes, candidate["evidence"])):
            candidate["protected"] = True
            self._rotating = [row for row in self._rotating if row is not candidate]
            self._protected.append(candidate)
        self._rotating = self._rotating[-self.max_candidates:]
        return _clone(candidate)

    def candidates(self) -> tuple[dict, ...]:
        return tuple(_clone(row) for row in (*self._protected, *self._rotating))

    def preserve_for_retry(self, *, current_revision: str, failure_path: str,
                           failure_key: str | None = None) -> dict:
        preserved = []
        for candidate in (*self._protected, *self._rotating):
            same = str(candidate.get("revision")) == str(current_revision)
            if same and not candidate.get("protected"):
                continue
            row = _clone(candidate)
            row["preserved"] = True
            row["preserve_reason"] = ("protected_passing_evidence" if candidate["protected"]
                                      else "retry_changed_revision")
            preserved.append(row)
        return {"schema": PARTIAL_SCHEMA, "current_revision": str(current_revision),
                "failure_path": str(failure_path),
                "failure_key": failure_key or str(failure_path),
                "preserved_candidates": preserved, "candidate_count": len(preserved)}


# ---------------------------------------------------- complementary route
def complementary_plan(history: Sequence[Mapping[str, object]], *,
                       current_failure: Mapping[str, object]) -> dict:
    """Choose an observable route with a different causal operation."""
    current = _clone(dict(current_failure))
    avoid = failure_path(current)
    prior = [row for row in history if isinstance(row, Mapping)]
    repeat_count = sum(1 for row in prior if failure_path(row) == avoid)
    operation = _text(current.get("operation") or current.get("kind"), "attempt")
    next_operation = ("source-boundary-observation"
                      if operation in {"public-check", "run_tests", "test"}
                      else "public-check-reproduction")
    return {
        "schema": EXPLORATION_SCHEMA, "avoid_path": avoid, "repeat_count": repeat_count,
        "history_size": len(prior),
        "next_attempt": {"path": f"alternate-observation:{avoid}",
                         "operation": next_operation, "evidence_source": "public-check",
                         "avoid_operation": operation,
                         "reason": "use a complementary observable path after the failed attempt"},
    }


# ------------------------------------------------------------- packets
def build_failure_packet(
    failure: Mapping[str, object], *, current_revision: str | None = None,
    preserved_progress: Mapping[str, object] | Sequence[object] | None = None,
    exploration: Mapping[str, object] | None = None,
    current_files: Mapping[str, str] | None = None,
) -> dict:
    """Build a packet from host evidence; observed/expected are copied verbatim.

    ``current_files`` lets each preserved snapshot carry a diff against the
    current source, which is the recovery surface rendered to the model.
    """
    if not isinstance(failure, Mapping):
        raise TypeError("failure must be a mapping")
    raw_failure = _clone(dict(failure))
    raw_failure.setdefault("what_failed", raw_failure.get("detail", "unspecified failure"))
    raw_failure.setdefault("observed", None)
    raw_failure.setdefault("expected", None)
    raw_failure["failure_key"] = failure_path(raw_failure) or _digest(
        {key: raw_failure.get(key) for key in ("what_failed", "observed", "expected")})
    if isinstance(preserved_progress, Mapping):
        preserved = _clone(dict(preserved_progress))
    elif isinstance(preserved_progress, Sequence) and not isinstance(preserved_progress, (str, bytes)):
        preserved = {"schema": PARTIAL_SCHEMA, "preserved_candidates": _clone(list(preserved_progress))}
    else:
        preserved = {"schema": PARTIAL_SCHEMA, "preserved_candidates": []}
    if isinstance(current_files, Mapping):
        for candidate in preserved.get("preserved_candidates", ()):
            if not isinstance(candidate, dict) or not isinstance(candidate.get("files"), Mapping):
                continue
            diffs = {}
            for name, content in candidate["files"].items():
                now = current_files.get(name)
                if isinstance(content, str) and isinstance(now, str) and content != now:
                    diffs[name] = "".join(difflib.unified_diff(
                        now.splitlines(True), content.splitlines(True),
                        f"current/{name}", f"snapshot/{name}", n=2))
            candidate["diff_vs_current"] = diffs
    packet = {
        "schema": FAILURE_SCHEMA, "packet_schema": SCHEMA, "failure": raw_failure,
        "where": _where(raw_failure),
        "current_revision": current_revision or raw_failure.get("current_revision"),
        "preserved_progress": preserved,
        "exploration": _clone(exploration) if exploration is not None else {
            "schema": EXPLORATION_SCHEMA, "avoid_path": raw_failure["failure_key"],
            "next_attempt": {"operation": "public-check-reproduction",
                             "evidence_source": "public-check"}},
    }
    packet["packet_id"] = _digest({key: value for key, value in packet.items()
                                   if key != "packet_id"})
    return packet


def _bounded(value: str, limit: int | None, reference: str) -> str:
    if limit is None or len(value) <= limit:
        return value
    return value[:max(0, limit)] + f"...[truncated {len(value) - limit} chars; full value: {reference}]"


def _candidate_lines(candidate: Mapping[str, object], *, diff_limit: int | None) -> list[str]:
    files = candidate.get("files", {})
    names = sorted(str(name) for name in files) if isinstance(files, Mapping) else []
    ref = SNAPSHOT_REF_PREFIX + _text(candidate.get("candidate_id"), "unknown")
    lines = [f"candidate={ref} revision={_text(candidate.get('revision'), 'unknown')} "
             f"protected={bool(candidate.get('protected'))} "
             f"reason={_text(candidate.get('preserve_reason'), 'recorded')} files={','.join(names)}"]
    passing = [str(row.get("receipt_id")) for row in candidate.get("evidence", ())
               if evidence_passes(row) and isinstance(row, Mapping) and row.get("receipt_id")]
    if passing:
        lines.append("passing_evidence=" + ",".join(passing[:4]))
    diffs = candidate.get("diff_vs_current")
    if isinstance(diffs, Mapping):
        if not diffs:
            lines.append("snapshot_matches_current_source=true")
        for name in sorted(diffs):
            lines.append(f"restore_diff[{name}] (apply to return to this snapshot):")
            lines.append(_bounded(str(diffs[name]), diff_limit, ref))
    elif isinstance(files, Mapping) and diff_limit != 0:
        for name in names:
            if isinstance(files.get(name), str):
                lines.append(f"preserved_source[{name}]=" + _bounded(_json(files[name]), diff_limit, ref))
    return lines


def _render(packet: Mapping[str, object], *, value_limit: int | None,
            diff_limit: int | None, drop_unprotected: bool) -> str:
    failure = packet.get("failure") if isinstance(packet.get("failure"), Mapping) else {}
    where = packet.get("where") if isinstance(packet.get("where"), Mapping) else {}
    progress = (packet.get("preserved_progress")
                if isinstance(packet.get("preserved_progress"), Mapping) else {})
    exploration = packet.get("exploration") if isinstance(packet.get("exploration"), Mapping) else {}
    attempt = exploration.get("next_attempt") if isinstance(exploration.get("next_attempt"), Mapping) else {}
    receipt_ref = "receipt " + _text(failure.get("receipt_id"), "unavailable")
    lines = [HEADER, f"schema={_text(packet.get('schema'), FAILURE_SCHEMA)}",
             f"packet_id={_text(packet.get('packet_id'))}",
             "WHAT FAILED",
             "what_failed=" + _bounded(_display(failure.get("what_failed")), value_limit, receipt_ref),
             f"failure_key={_text(failure.get('failure_key'))}",
             "OBSERVED VS EXPECTED",
             "observed=" + _bounded(_display(failure.get("observed")), value_limit, receipt_ref),
             "expected=" + _bounded(_display(failure.get("expected")), value_limit, receipt_ref),
             "WHERE", f"path={_text(where.get('path'), 'unknown')}"]
    lines += [f"{key}={where[key]}" for key in ("symbol", "line", "column", "anchor")
              if where.get(key) not in (None, "")]
    if where.get("path") not in (None, "") and where.get("line") not in (None, ""):
        lines.append(f"location={where['path']}:{where['line']}")
    lines += ["CURRENT REVISION", f"revision={_text(packet.get('current_revision'), 'unknown')}",
              "PRESERVED PARTIAL PROGRESS"]
    candidates = [row for row in progress.get("preserved_candidates", ()) if isinstance(row, Mapping)]
    if not candidates:
        lines.append("none_recorded=true")
    for candidate in candidates:
        if drop_unprotected and not candidate.get("protected"):
            lines.append(f"candidate={SNAPSHOT_REF_PREFIX}{_text(candidate.get('candidate_id'))} "
                         f"revision={_text(candidate.get('revision'))} protected=False detail_omitted=true")
            continue
        lines += _candidate_lines(candidate, diff_limit=diff_limit)
    lines += ["COMPLEMENTARY EXPLORATION",
              f"avoid_path={_text(exploration.get('avoid_path'), 'unknown')}",
              f"repeat_count={_text(exploration.get('repeat_count'), '0')}"]
    for key in ("path", "operation", "evidence_source", "avoid_operation", "reason"):
        if attempt.get(key) not in (None, ""):
            lines.append(f"next_{key}={attempt[key]}")
    lines.append(FOOTER)
    return "\n".join(lines)


def render_failure_packet(packet: Mapping[str, object], *, max_chars: int = MAX_PACKET_CHARS) -> str:
    """Render deterministically within ``max_chars``, compacting per field.

    Order of compaction: unprotected snapshot detail, then snapshot diffs,
    then long observed/expected values; every required section and the
    selected next attempt are always kept.  Returns ``""`` when even the
    compact form does not fit, so the caller can decline to insert it.
    """
    if not isinstance(packet, Mapping):
        raise TypeError("packet must be a mapping")
    attempts = [dict(value_limit=None, diff_limit=None, drop_unprotected=False),
                dict(value_limit=None, diff_limit=None, drop_unprotected=True)]
    attempts += [dict(value_limit=None, diff_limit=limit, drop_unprotected=True)
                 for limit in (2400, 1200, 600, 300)]
    attempts += [dict(value_limit=limit, diff_limit=300, drop_unprotected=True)
                 for limit in (800, 300, 120)]
    attempts.append(dict(value_limit=120, diff_limit=0, drop_unprotected=True))
    for options in attempts:
        text = _render(packet, **options)
        if len(text) <= max_chars:
            return text
    return ""


# ------------------------------------------------------ failure extraction
def _parse_observed_expected(detail: object) -> tuple[object, object]:
    text = str(detail or "")
    observed = re.search(r"\bobserved\s*[:=]\s*([^,;\s]+)", text, re.IGNORECASE)
    expected = re.search(r"\bexpected\s*[:=]\s*([^,;\s]+)", text, re.IGNORECASE)
    return (observed.group(1) if observed else None, expected.group(1) if expected else None)


def is_failure_receipt(receipt: Mapping[str, object]) -> bool:
    values = (receipt.get("result_class"), receipt.get("result"),
              receipt.get("status"), receipt.get("outcome"))
    if any(str(value or "").upper() in _FAILURE_CLASSES for value in values):
        return True
    exit_code = receipt.get("exit_code")
    return type(exit_code) is int and exit_code != 0


def failure_from_receipts(receipts: Sequence[Mapping[str, object]], *,
                          last_receipt: Mapping[str, object] | None = None,
                          host_payload: Mapping[str, object] | None = None) -> dict | None:
    """Extract the latest host failure without treating prose as authority."""
    rows = [row for row in receipts if isinstance(row, Mapping)]
    if isinstance(last_receipt, Mapping) and last_receipt:
        rows.append(last_receipt)
    selected = next((row for row in reversed(rows) if is_failure_receipt(row)), None)
    if selected is None and isinstance(host_payload, Mapping):
        selected = host_payload if any(key in host_payload for key in (
            "observed", "expected", "counterexample", "reason")) else None
    if selected is None:
        return None
    observed, expected = selected.get("observed"), selected.get("expected")
    if observed is None or expected is None:
        parsed_observed, parsed_expected = _parse_observed_expected(selected.get("detail"))
        observed = observed if observed is not None else parsed_observed
        expected = expected if expected is not None else parsed_expected
    host = host_payload if isinstance(host_payload, Mapping) else {}
    if observed is None:
        observed = host.get("observed", host.get("counterexample"))
    if expected is None:
        expected = host.get("expected", host.get("required"))
    if observed is None and type(selected.get("exit_code")) is int:
        observed = {"exit_code": selected["exit_code"], "selectors": selected.get("selectors")}
        expected = expected if expected is not None else {"exit_code": 0}
    location = selected.get("where")
    if not isinstance(location, (Mapping, str)):
        location = host.get("location_hint")
    location = _location_value(location)
    location.setdefault("path", selected.get("target_path", selected.get("path", "unknown")))
    anchor = selected.get("anchor")
    if anchor and "anchor" not in location:
        location["anchor"] = anchor
        match = re.match(r"(?P<symbol>[^:]+):(?P<line>\d+)$", str(anchor))
        if match:
            location.setdefault("symbol", match.group("symbol"))
            location.setdefault("line", int(match.group("line")))
    if type(selected.get("exit_code")) is int:
        # Host test subprocess receipt: the failed operation is the public check.
        what_failed = "public check exit " + str(selected["exit_code"])
        operation = "test"
    else:
        what_failed = selected.get("intent", selected.get("detail", selected.get("kind", "attempt")))
        operation = selected.get("intent", selected.get("kind", "attempt"))
    return {
        "what_failed": what_failed,
        "observed": _clone(observed), "expected": _clone(expected), "where": location,
        "operation": operation,
        "detail": selected.get("detail", ""),
        "result_class": selected.get("result_class", selected.get(
            "result", selected.get("status", "FAIL"))),
        "receipt_id": selected.get("receipt_id"),
    }


# Short aliases keep the producer/renderer easy to discover for integrations.
build_packet = build_failure_packet
render_packet = render_failure_packet
PartialProgress = PartialProgressLedger
next_exploration = complementary_plan
