"""Compact repeated prompt material while preserving task-state contracts.

Each optimization is independently controlled and leaves the tool menu and
completion evidence under their existing owners.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence

from kernel_stateframe_envelope import STATE_HEADER, serialize_stateframe_payload

MODE_ENV = "STATEFRAME_PROMPT_COMPACT"
MANIFEST_KEY = "stateframe_prompt_compact"
EVENT = "prompt_compact_applied"
PRODUCER = "kernel_prompt_compact"
SCHEMA = "stateframe-prompt-compact/1"

# Producer header constants (each cited; test_prompt_compact asserts equality with the producer).
CLAUSE_HEADER = "[Get-sure state]"                # kernel_stateframe_f4.CLAUSE_HEADER
HANDOFF_HEADER = "[Decision handoff]"             # kernel_stateframe_f4.HANDOFF_HEADER
HYPOTHESES_HEADER = "[HYPOTHESES_NOT_FACTS"       # kernel_hypothesis_projection.BLOCK_HEADER prefix
WORKING_SET_HEADER = "[Working sets | host-retained]"  # kernel_stateframe_runtime.F4_WORKING_SET_HEADER
CERTIFICATE_HEADER = "[VERIFICATION CERTIFICATE]"  # kernel_vacuity_sentinel.BLOCK_HEADER
SMOKE_HEADER = "[SHIP-TREE SMOKE]"                # kernel_ship_tree_smoke.BLOCK_HEADER
RAISING_H_HEADER = "[RAISING-H FAILURE PACKET]"   # kernel_raising_h_packets.HEADER
UNIFIED_STOP_HEADER = "[Unified stop authority]"  # kernel_completion_policy.RENDER_HEADER
EPOCH_DELTA_PREFIX = "[StateFrame epoch delta]"   # kernel_stateframe_epoch.DELTA_HEADER (sans newline)

SHORT_ID_CHARS = 12
_SHA_RE = re.compile(r"sha256:([0-9a-f]{%d})[0-9a-f]*(?:…)?" % SHORT_ID_CHARS)
_FULL_SHA_RE = re.compile(r"sha256:([0-9a-f]{64})(?![0-9a-f])")
# Prefixes that must stay full in the current request (alias collision check); set per request
# by ``compact_f4_messages`` / the producers, empty otherwise.
_COLLIDING: set[str] = set()
_EVIDENCE_KEYS = frozenset({"source", "state", "revision", "references"})

# Pending per-block measurements for the current request: block -> (bytes_before, bytes_after,
# confirm_text, dropped_marker).  Last write wins, so sizing trials are overwritten by the
# final render; drained (and cleared) at the confirmed-render point.
_PENDING: dict[str, tuple[int, int, str, str]] = {}


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = str(source.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return mode(environ) == "on"


def short_id(value: object) -> str:
    """``sha256:<hex>`` (or bare hex) -> its 12-hex prefix; anything else unchanged."""
    text = str(value)
    match = re.fullmatch(r"(?:sha256:)?([0-9a-f]{%d})[0-9a-f]*" % SHORT_ID_CHARS, text)
    if not match or match.group(1) in _COLLIDING:
        return text
    return match.group(1)


def shorten_hashes(text: str) -> str:
    return _SHA_RE.sub(lambda match: (match.group(0) if match.group(1) in _COLLIDING
                                      else match.group(1)), text)


def colliding_prefixes(texts: Sequence[str]) -> set[str]:
    """12-hex prefixes shared by two different full sha256 values across ``texts``."""
    seen: dict[str, str] = {}
    colliding: set[str] = set()
    for text in texts:
        for full in _FULL_SHA_RE.findall(text):
            prefix = full[:SHORT_ID_CHARS]
            if seen.setdefault(prefix, full) != full:
                colliding.add(prefix)
    return colliding


def set_collision_scope(texts: Sequence[str]) -> None:
    _COLLIDING.clear()
    _COLLIDING.update(colliding_prefixes(texts))


def _dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _note(block: str, before: str, after: str, *, dropped_marker: str = "") -> None:
    _PENDING[block] = (len(before.encode("utf-8")), len(after.encode("utf-8")), after, dropped_marker)


def reset_pending() -> None:
    _PENDING.clear()


# ---------------------------------------------------------------------------------------------
# 1. StateFrame
# ---------------------------------------------------------------------------------------------

def _group_evidence(rows: Sequence[Mapping[str, object]], top: str) -> list[dict[str, object]]:
    order: list[tuple] = []
    counts: dict[tuple, int] = {}
    for row in rows:
        revision = str(row.get("revision", top))
        refs = row.get("references") or []
        key = (str(row.get("source", "host")), str(row.get("state", "UNKNOWN")),
               "" if revision == top else short_id(revision),
               tuple(shorten_hashes(str(item)) for item in refs))
        if key not in counts:
            order.append(key)
            counts[key] = 0
        counts[key] += 1
    grouped: list[dict[str, object]] = []
    for key in order:
        source, state, revision, refs = key
        item: dict[str, object] = {"source": source, "state": state}
        if counts[key] > 1:
            item["n"] = counts[key]
        if revision:
            item["revision"] = revision
        if refs:
            item["references"] = list(refs)
        grouped.append(item)
    return grouped


def _is_evidence_list(value: object) -> bool:
    return (isinstance(value, list) and bool(value)
            and all(isinstance(row, Mapping) and set(row) <= _EVIDENCE_KEYS and "state" in row
                    for row in value))


def _compact_state(value: object, top: str, glossary: frozenset[str]) -> object:
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if key in {"framefit_action_ledger", "framefit_open_edit_site"}:
                # F3/F4 rows bind exact call receipts, source ranges, and bytes.
                # Generic F4 compaction must not shorten or regroup those contracts.
                result[key] = item
                continue
            if (key == "framefit5" and isinstance(item, Mapping)
                    and item.get("schema") == "stateframe-framefit5/1"):
                # FRAMEFIT5 receipts bind exact candidate bytes and revisions;
                # generic F4 compaction must not rewrite that typed contract.
                result[key] = item
                continue
            if key == "revision" and str(item) == top:
                continue
            if key == "establishes" and str(item) in glossary:
                continue
            if key == "evidence" and _is_evidence_list(item):
                result[key] = _group_evidence(item, top)
                continue
            result[key] = _compact_state(item, top, glossary)
        return result
    if isinstance(value, list):
        return [_compact_state(item, top, glossary) for item in value]
    if isinstance(value, str):
        return shorten_hashes(value)
    return value


def _glossary_values() -> frozenset[str]:
    from kernel_stateframe_f4 import F4_PREDICATE_GLOSSARY
    return frozenset(F4_PREDICATE_GLOSSARY.values())


def compact_state_text(text: str, *, smoke: str | None = None) -> str | None:
    """Compact one ``[StateFrame]\\n{json}`` block; ``None`` when it is not parseable."""
    if not text.startswith(STATE_HEADER + "\n"):
        return None
    try:
        payload = json.loads(text.split("\n", 1)[1])
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("state"), dict):
        return None
    top = str(payload.get("revision", ""))
    state = _compact_state(payload["state"], top, _glossary_values())
    if smoke is not None:
        state["smoke"] = smoke
    visible = {"rev": short_id(top), "schema": payload.get("schema"), "state": state}
    return serialize_stateframe_payload(visible)


def evidence_facts(state_text: str) -> list[tuple[str, str, str, str, tuple[str, ...]]]:
    """Sorted multiset of (path, source, state, short revision, refs) evidence facts in a
    full OR compact StateFrame block -- the no-information-loss invariant is that both give
    the same list."""
    payload = json.loads(state_text.split("\n", 1)[1])
    top = short_id(payload.get("revision", payload.get("rev", "")))
    facts: list[tuple[str, str, str, str, tuple[str, ...]]] = []

    def walk(node: object, path: str) -> None:
        if isinstance(node, Mapping):
            for key, item in node.items():
                if key == "evidence" and isinstance(item, list) and item and all(
                        isinstance(row, Mapping) and "state" in row
                        and set(row) <= _EVIDENCE_KEYS | {"n"} for row in item):
                    for row in item:
                        refs = tuple(shorten_hashes(str(ref)) for ref in row.get("references") or [])
                        rev = short_id(row.get("revision", top))
                        for _ in range(int(row.get("n", 1))):
                            facts.append((path, str(row.get("source")), str(row.get("state")),
                                          rev, refs))
                else:
                    walk(item, f"{path}.{key}")
        elif isinstance(node, list):
            for item in node:
                walk(item, path)

    walk(payload.get("state"), "state")
    return sorted(facts)


# ---------------------------------------------------------------------------------------------
# 2. Get-sure clause table + HYPOTHESES_NOT_FACTS
# ---------------------------------------------------------------------------------------------

_TICKET_LIST_LINE = "Ticket clauses (parsed once; stable numbering):"
_VERDICT_TABLE_LINE = "Clause verdict table (current):"
_TICKET_ROW = re.compile(r"^\[C\d+\] ")
_VERDICT_ROW = re.compile(r"^(C\d+) \| ([A-Z_]+) \| (.*)$")
_HYP_FIELDS = (" | status=", " | staleness=", " | basis_revision=", " | current_revision=",
               " | evidence=", " | provenance=")


def _split_hypothesis(line: str) -> tuple[str, dict[str, str]] | None:
    """Parse a ``_render_row`` line from the right (the claim text may itself contain ``|``)."""
    label, sep, rest = line.partition(": ")
    if not sep:
        return None
    fields: dict[str, str] = {}
    tail = rest
    for marker in reversed(_HYP_FIELDS):
        at = tail.rfind(marker)
        if at == -1:
            if marker == " | provenance=":
                continue
            return None
        fields[marker[3:-1]] = tail[at + len(marker):]
        tail = tail[:at]
    return label, {"claim": tail, **fields}


def _compact_hypotheses(lines: list[str]) -> list[str]:
    from kernel_hypothesis_projection import (
        ELIMINATED_HYPOTHESIS_LABEL, HYPOTHESIS_LABEL, STALE_HYPOTHESIS_LABEL,
    )
    out: list[str] = []
    counts: dict[str, str] = {}
    for line in lines:
        if line.startswith(HYPOTHESES_HEADER):
            out.append(line)
            continue
        if line.startswith("current_revision: "):
            # r46 journals: this revision (the hypothesis projector's product_revision) never
            # equalled the StateFrame rev (83/83 requests) -- different artifacts, so they are
            # NOT unified; the line is relabelled by domain so the request carries exactly one
            # value called "current" (the StateFrame rev).
            out.append("staleness_basis: product_revision "
                       + short_id(line[len("current_revision: "):]))
            continue
        key, sep, value = line.partition(": ")
        if sep and key in {"active_hypotheses", "stale_hypotheses", "eliminated_hypotheses"}:
            counts[key.split("_")[0]] = value
            if key == "eliminated_hypotheses":
                out.append("counts: " + " ".join(f"{name}={counts.get(name, '0')}"
                                                 for name in ("active", "stale", "eliminated")))
            continue
        if line.startswith(STALE_HYPOTHESIS_LABEL + ": "):
            continue  # counted in stale=N
        if line.startswith((HYPOTHESIS_LABEL + ": ", ELIMINATED_HYPOTHESIS_LABEL + ": ")):
            parsed = _split_hypothesis(line)
            if parsed is not None:
                label, fields = parsed
                out.append(f"{label}: {fields['claim']} | status={fields['status']} | "
                           f"staleness={fields['staleness']} | evidence={shorten_hashes(fields['evidence'])}")
                continue
        out.append(shorten_hashes(line))
    return out


def _compact_get_sure(lines: list[str], *, task_present: bool) -> list[str]:
    out: list[str] = []
    rows: list[tuple[str, str, str, str]] = []
    in_ticket = in_table = False
    for line in lines:
        if line == _TICKET_LIST_LINE:
            in_ticket = task_present
            if not task_present:
                out.append(line)
            continue
        if in_ticket and _TICKET_ROW.match(line):
            continue
        in_ticket = False
        if line == _VERDICT_TABLE_LINE:
            in_table = True
            out.append(line)
            continue
        match = _VERDICT_ROW.match(line) if in_table else None
        if match:
            clause, _sep, tail = match.group(3).partition(" | ")
            rows.append((match.group(1), match.group(2), clause, shorten_hashes(tail)))
            continue
        if in_table:
            out.extend(_render_rows(rows))
            rows, in_table = [], False
        out.append(shorten_hashes(line))
    if rows:
        out.extend(_render_rows(rows))
    return out


def _render_rows(rows: list[tuple[str, str, str, str]]) -> list[str]:
    tails: dict[str, list[str]] = {}
    for cid, _verdict, _clause, tail in rows:
        if tail:
            tails.setdefault(tail, []).append(cid)
    shared = {tail for tail, ids in tails.items() if len(ids) > 1}
    out = [f"{cid} {verdict}: {clause}" + (f" | {tail}" if tail and tail not in shared else "")
           for cid, verdict, clause, tail in rows]
    for tail, ids in tails.items():
        if tail in shared:
            out.append(f"{','.join(ids)}: {tail}")
    return out


def compact_clause_text(text: str, *, task_present: bool) -> str:
    lines = text.split("\n")
    at = next((index for index, line in enumerate(lines) if line.startswith(HYPOTHESES_HEADER)),
              len(lines))
    head = _compact_get_sure(lines[:at], task_present=task_present) if lines[0] == CLAUSE_HEADER \
        else [shorten_hashes(line) for line in lines[:at]]
    tail = _compact_hypotheses(lines[at:]) if at < len(lines) else []
    return "\n".join(head + tail)


# ---------------------------------------------------------------------------------------------
# 3. DELIVERY_MANIFEST model view
# ---------------------------------------------------------------------------------------------

def delivery_model_view(manifest: Mapping[str, object]) -> dict[str, object]:
    validation = manifest.get("validation") if isinstance(manifest.get("validation"), Mapping) else {}
    deltas = []
    for row in manifest.get("file_deltas") or ():
        if isinstance(row, Mapping):
            deltas.append({"path": row.get("path"), "status": row.get("status"),
                           "+": row.get("added_lines"), "-": row.get("deleted_lines")})
    view: dict[str, object] = {
        "files_changed": list(manifest.get("files_changed") or ()),
        "file_deltas": deltas,
        "checks": list(manifest.get("checks") or ()),
        "latest_check": manifest.get("latest_check"),
        "latest_check_revision": manifest.get("latest_check_revision"),
        "applied_edits": list(manifest.get("applied_edits") or ()),
        "open_items": list(manifest.get("open_items") or ()),
        "blocking_items": list(validation.get("blocking_items") or ()),
    }
    for key in ("behaviours_claimed", "omissions"):
        if manifest.get(key):
            view[key] = manifest.get(key)
    for key in ("done_receipt_id", "outcome"):
        if manifest.get(key) is not None:
            view[key] = manifest.get(key)
    return view


def compact_delivery_text(text: str) -> str:
    header, sep, rest = text.partition("\n")
    data_line, sep2, body = rest.partition("\n")
    if not (sep and sep2):
        return text
    try:
        manifest = json.loads(body)
    except (TypeError, ValueError):
        return text  # diet unchanged-notice or other non-JSON body: unchanged
    if not isinstance(manifest, Mapping):
        return text
    compact = header + "\n" + data_line + "\n" + _dumps(delivery_model_view(manifest))
    _note("delivery_manifest", text, compact)
    return compact


# ---------------------------------------------------------------------------------------------
# 4/5. Working sets, handoff, passive conclusions
# ---------------------------------------------------------------------------------------------

def compact_working_sets_text(text: str) -> str:
    """'' when nothing is active or resident; otherwise the block without ``snapshot_id``."""
    body = text.split("\n", 1)[1] if "\n" in text else ""
    try:
        view = json.loads(body)
    except (TypeError, ValueError):
        return re.sub(r'"snapshot_id":"[^"]*",?', "", text)
    if not isinstance(view, dict):
        return text
    directory = view.get("directory") if isinstance(view.get("directory"), list) else []
    resident = [row for row in directory
                if isinstance(row, Mapping) and str(row.get("residency", "cold")) != "cold"]
    if view.get("active") is None and not resident:
        return ""
    view.pop("snapshot_id", None)
    return WORKING_SET_HEADER + "\n" + _dumps(view)


_EMPTY_REJECTED = ("Rejected alternative: not yet supplied; reopen if the relevant code changes "
                   "or contrary evidence appears.")


def compact_handoff_text(text: str) -> str:
    return "\n".join(shorten_hashes(line) for line in text.split("\n") if line != _EMPTY_REJECTED)


_PASSIVE_ROW = re.compile(r"^- step (\d+), response=(\S+), basis=(\w+): (.*)$")
_PASSIVE_WITHHELD = re.compile(
    r"^\[reference only: recorded against an earlier binding; body withheld, id=([^\]]*)\]$")
PASSIVE_WITHHELD_NOTE = "basis=changed rows were recorded against an earlier binding; their bodies are withheld."
# STATEFRAME_TOKENLAB_PASSIVE_WITHHELD_DROP: withheld rows are dropped instead of rendered as
# reference rows (measured: every passive row in 702 ldev_12 v5 was withheld).  Strict off|on, default off.
PASSIVE_WITHHELD_DROP_ENV = "STATEFRAME_TOKENLAB_PASSIVE_WITHHELD_DROP"
PASSIVE_WITHHELD_DROP_EVENT = "passive_withheld_dropped"
PASSIVE_DROPPED_NOTE = "earlier conclusions recorded against a previous binding were withheld (not shown)."


def passive_withheld_drop_enabled(environ: Mapping[str, str] | None = None) -> bool:
    source = os.environ if environ is None else environ
    value = str(source.get(PASSIVE_WITHHELD_DROP_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{PASSIVE_WITHHELD_DROP_ENV} must be off or on")
    return value == "on"


def compact_passive_text(text: str) -> str:
    out: list[str] = []
    withheld = False
    drop = passive_withheld_drop_enabled()  # 
    dropped = kept_rows = 0
    for line in text.split("\n"):
        match = _PASSIVE_ROW.match(line)
        if not match:
            out.append(shorten_hashes(line))
            continue
        step, response, basis, body = match.groups()
        stale = _PASSIVE_WITHHELD.match(body)
        if stale and drop:
            dropped += 1
            continue
        kept_rows += 1
        if stale:
            withheld = True
            body = f"[withheld id={stale.group(1)}]"
        # ``response=<ref>`` is kept verbatim: kernel_passive_retention_delivery proves
        # delivery by that exact marker (a shortened ref would read as "never delivered").
        out.append(f"- step {step}, response={response}, basis={basis}: {body}")
    if withheld:
        first_row = next((index for index, line in enumerate(out) if line.startswith("- step ")), len(out))
        out.insert(first_row, PASSIVE_WITHHELD_NOTE)
    if dropped and not kept_rows:
        # nothing left to show: the whole block goes (the producer returns None on empty text)
        compact = ""
        _note("passive_conclusions", text, compact, dropped_marker=text)
        _PASSIVE_DROP_PENDING[0] = dropped
        return compact
    if dropped:
        out.append(PASSIVE_DROPPED_NOTE)
        _PASSIVE_DROP_PENDING[0] = dropped
    compact = "\n".join(out)
    _note("passive_conclusions", text, compact)
    return compact


_PASSIVE_DROP_PENDING: list[int] = [0]


def emit_passive_withheld_dropped(journal: object, *, request_id: object) -> bool:
    """own event of STATEFRAME_TOKENLAB_PASSIVE_WITHHELD_DROP: rows dropped for this request."""
    count, _PASSIVE_DROP_PENDING[0] = _PASSIVE_DROP_PENDING[0], 0
    if not count or not hasattr(journal, "emit"):
        return False
    journal.emit(PASSIVE_WITHHELD_DROP_EVENT, producer=PRODUCER, dispatch=False,
                 request_id=request_id, rows_dropped=count)
    return True


# ---------------------------------------------------------------------------------------------
# 6. System prompt glossary, certificate, smoke fold
# ---------------------------------------------------------------------------------------------

# The ``establishes`` meanings of kernel_stateframe_f4.F4_PREDICATE_GLOSSARY, said once and
# tersely (the system prompt is paid on every request too, so a verbatim copy would cost more
# than the per-request sentences it replaces).  test_prompt_compact pins one entry per predicate.
COMPACT_GLOSSARY: dict[str, str] = {
    "public_checks": "result of the named public checks on this revision only; not permission, not completion",
    "done_permission": "whether the dispatch gate allows done on this revision; not a test verdict or seal",
    "completion": "whether the host accepted the candidate; never rewrites check history",
    "submission": "terminal submission status only; not a completion seal",
    "blockers": "open host blockers on this revision; they explain a denied gate",
    "advisories": "visible host advisories; they grant nothing",
}


def system_addendum() -> str:
    """Constant: the StateFrame predicate glossary plus the compact-row legend."""
    return ("[StateFrame glossary] " + "; ".join(f"{name} = {text}" for name, text in COMPACT_GLOSSARY.items())
            + ". Evidence rows {source,state,n} = n identical host observations; a row shows a revision "
            "only when it differs from rev. 12-hex ids are sha256 prefixes.")


def _smoke_fold(text: str) -> str | None:
    match = re.fullmatch(re.escape(SMOKE_HEADER) + r" PASS: (\d+) touched module\(s\) import in a "
                         r"fresh interpreter\.", text)
    if match:
        return f"PASS({match.group(1)})"
    if text == f"{SMOKE_HEADER} nothing to check yet (no touched Python module).":
        return "NONE"
    if text.startswith(f"{SMOKE_HEADER} ENVIRONMENT"):
        # Baseline-identical failures: a non-blocking fact, never an obligation
        # (kernel_ship_tree_smoke.render_smoke ENVIRONMENT branch).
        count = sum(1 for line in text.splitlines()[1:] if line.startswith("- "))
        return f"ENVIRONMENT({count}): identical at the untouched parent; not blocking, no edit requested"
    return None  # FAIL keeps its own block (it lists the failing modules)


# ---------------------------------------------------------------------------------------------
# F4 request orchestration
# ---------------------------------------------------------------------------------------------

def _task_present(messages: Sequence[Mapping[str, object]]) -> bool:
    return any(isinstance(item, Mapping) and item.get("role") not in {"assistant", "tool"}
               and str(item.get("content") or "").startswith("Task:") for item in messages)


def compact_clause_block(content: str, *, task_present: bool) -> str:
    """F4's direct render path (runtime ``_f4_view``): compact the combined Get-sure +
    HYPOTHESES block once, before any sizing trial or fit sees it."""
    compact = compact_clause_text(content, task_present=task_present)
    _note("get_sure_hypotheses", content, compact)
    return compact


def compact_f4_messages(messages: Sequence[Mapping[str, object]], *,
                        clauses: bool = True, stateframe: bool = True,
                        protected_source_spans: Sequence[str] = ()) -> list[dict]:
    """Compact the composed F4 request (before fit/invariants).  Pure except for the
    per-request measurement buffer; roles, order and every non-owned block are unchanged.
    ``clauses=False`` when the clause block was already compacted at the render path;
    ``stateframe=False`` under STATEFRAME_FRAME_EPOCH, whose deltas address the full
    snapshot's paths (the snapshot and the smoke block are then left as rendered)."""
    original = [dict(message) for message in messages]
    set_collision_scope([str(item.get("content") or "") for item in original])
    history_start = next((index for index, item in enumerate(original)
                          if item.get("role") in {"assistant", "tool"}), len(original))
    contents = [str(item.get("content") or "") for item in original[:history_start]]
    task_present = _task_present(original)
    epoch = not stateframe or any(content.startswith(EPOCH_DELTA_PREFIX) for content in contents)
    smoke = None
    if not epoch:
        for content in contents:
            if content.startswith(SMOKE_HEADER):
                smoke = _smoke_fold(content)
                if smoke is not None:
                    _note("ship_tree_smoke", content, "", dropped_marker=content)
                break
    result: list[dict] = []
    for index, message in enumerate(original):
        if index >= history_start or message.get("role") != "user":
            result.append(message)
            continue
        content = str(message.get("content") or "")
        if any(span and span in content for span in protected_source_spans):
            result.append(message)
            continue
        compact: str | None = content
        block = ""
        if content.startswith(STATE_HEADER + "\n") and not epoch:
            block, compact = "stateframe", compact_state_text(content, smoke=smoke)
            if compact is None:
                compact = content
        elif content.startswith(SMOKE_HEADER) and smoke is not None:
            continue
        elif clauses and content.startswith((CLAUSE_HEADER, HYPOTHESES_HEADER)):
            block, compact = "get_sure_hypotheses", compact_clause_text(content, task_present=task_present)
        elif content.startswith(WORKING_SET_HEADER):
            block, compact = "working_sets", compact_working_sets_text(content)
        elif content.startswith(HANDOFF_HEADER):
            block, compact = "decision_handoff", compact_handoff_text(content)
        elif content.startswith(RAISING_H_HEADER):
            block, compact = "raising_h", shorten_hashes(content)
        elif content.startswith(UNIFIED_STOP_HEADER):
            block, compact = "unified_stop", shorten_hashes(content)
        elif content.startswith(CERTIFICATE_HEADER + " UNKNOWN:"):
            block, compact = "verification_certificate", ""
        if block and compact != content:
            _note(block, content, compact, dropped_marker=content if not compact else "")
        if compact:
            result.append({**message, "content": compact})
    addendum = system_addendum()
    for message in result:
        if message.get("role") == "system":
            content = str(message.get("content") or "")
            if addendum not in content:
                message["content"] = content + "\n" + addendum
                _note("system_glossary", content, message["content"])
            break
    return result


def confirmed_blocks(prepared: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Drain the buffer; keep the blocks whose compact form is proven in ``prepared``."""
    contents = [str(item.get("content") or "") for item in prepared if isinstance(item, Mapping)]
    confirmed: list[dict[str, object]] = []
    for block, (before, after, text, dropped) in sorted(_PENDING.items()):
        if text:
            ok = any(text in content for content in contents)
        else:
            ok = bool(dropped) and not any(dropped in content for content in contents)
        if ok:
            confirmed.append({"block": block, "bytes_before": before, "bytes_after": after})
    _PENDING.clear()
    return confirmed


def emit_applied(journal: object, prepared: Sequence[Mapping[str, object]], *,
                 request_id: object) -> dict[str, object] | None:
    blocks = confirmed_blocks(prepared)
    if not blocks or not hasattr(journal, "emit"):
        return None
    payload = {
        "compact_schema": SCHEMA, "request_id": request_id, "blocks": blocks,
        "bytes_before": sum(int(row["bytes_before"]) for row in blocks),
        "bytes_after": sum(int(row["bytes_after"]) for row in blocks),
    }
    journal.emit(EVENT, producer=PRODUCER, dispatch=False, **payload)
    return payload


__all__ = [
    "EVENT", "MANIFEST_KEY", "MODE_ENV", "PRODUCER", "SCHEMA", "colliding_prefixes",
    "compact_clause_block", "compact_clause_text",
    "compact_delivery_text", "compact_f4_messages", "compact_handoff_text", "compact_passive_text",
    "compact_state_text", "compact_working_sets_text", "confirmed_blocks", "delivery_model_view",
    "emit_applied", "enabled", "evidence_facts", "mode", "reset_pending", "set_collision_scope",
    "short_id", "shorten_hashes", "system_addendum",
]
