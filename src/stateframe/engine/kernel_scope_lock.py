"""Q7 (E10): post-green scope lock.

After ``done`` is refused for an EVIDENCE reason (missing checks, unverified
rows) rather than a failing check, the work is behaviourally done; what remains
is evidence.  A next edit that changes an existing numeric literal or constant
that no admitted cause row names is then held as ``SCOPE_UNSUPPORTED`` and
rendered as ``[SCOPE] <hunk> changes <old -> new>; no admitted row names it;
the refusal reason was <reason>``.  The model may add a row (new evidence) or
drop the hunk; the edit is never silently applied.  New lines (for example an
added immediate call) are not literal changes and are always admitted.

The lock sits in the central pre-write guard
(``kernel_destructive_edit_guard.guard_write``) that every product writer
crosses.  Admitted rows are the requirement-ledger clauses
(``runtime.requirements_ledger``) plus ``runtime.scope_lock_rows``.

``STATEFRAME_SCOPE_LOCK`` (default ``on``).
"""
from __future__ import annotations

import os
import re
import difflib
from collections import Counter
from collections.abc import Mapping, Sequence

ENV = "STATEFRAME_SCOPE_LOCK"
HEADER = "[SCOPE]"
SCOPE_UNSUPPORTED = "SCOPE_UNSUPPORTED"
EVIDENCE_CODES = frozenset({
    "MISSING_CHECKS", "NO_VERIFIER", "UNVERIFIED_DISCLOSED_CHECKS", "REQUIRED_EVIDENCE_GAP",
    "BEHAVIOR_WITNESS_REQUIRED", "behavior_witness_pending", "UNBOUND_OBSERVABLE",
    "CLAUSE_COVERAGE", "COVERAGE_OPEN", "UNVERIFIED",
})
FAILING_CODES = frozenset({
    "ORACLE_TESTS_RED", "PUBLIC_RED_DONE", "TESTS_NOT_GREEN", "NONZERO_EXIT", "OPEN_COUNTEREXAMPLE",
    "BOUND_COUNTEREXAMPLE", "COUNTEREXAMPLE", "UNIFIED_VIOLATED", "PRODUCT_VIOLATION",
    "SHIP_TREE_SMOKE", "READINESS_COUNTEREXAMPLE_OPEN", "UNRESOLVED_DEFECT",
})


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    values = os.environ if environ is None else environ
    value = str(values.get(ENV, "on") or "on").strip().lower()
    if value not in {"on", "off"}:
        raise ValueError(f"{ENV} must be on or off")
    return value == "on"


def literal_changes(before: str, after: str) -> list[dict[str, object]]:
    """Typed value deltas on stable lines; inserted lines are not value changes."""
    from kernel_aha10_honest_loop import typed_literal_changes
    result: list[dict[str, object]] = []
    for change in typed_literal_changes(before, after):
        old_line = str(change.get("old_line", ""))
        new_line = str(change.get("new_line", ""))
        if old_line.lstrip().startswith(("#", "//", "/*", "*")) \
                and new_line.lstrip().startswith(("#", "//", "/*", "*")):
            continue
        item = dict(change)
        item["span"] = {"start_line": change.get("line"), "end_line": change.get("line")}
        result.append(item)
    if result:
        return result

    # The Python helper intentionally ignores a mixed member/value hunk when
    # its parser cannot establish one language-specific AST.  Keep the scope
    # lock conservative at that boundary: compare stable replacement lines
    # lexically and report each changed value/member with the exact line span.
    token = re.compile(
        r"(?:\"(?:\\.|[^\"])*\"|'(?:\\.|[^'])*'|\b\d+(?:\.\d+)?\b|"
        r"[A-Za-z_$][\w$]*(?:\s*(?:\.|::|->)\s*[A-Za-z_$][\w$]*)*)"
    )
    old_lines, new_lines = before.splitlines(), after.splitlines()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
            None, old_lines, new_lines, autojunk=False).get_opcodes():
        if tag != "replace":
            continue
        for offset, (old_line, new_line) in enumerate(zip(old_lines[i1:i2], new_lines[j1:j2])):
            old_tokens, new_tokens = token.findall(old_line), token.findall(new_line)
            matcher = difflib.SequenceMatcher(None, old_tokens, new_tokens, autojunk=False)
            for token_tag, left, right, new_left, new_right in matcher.get_opcodes():
                if token_tag == "equal":
                    continue
                removed, added = old_tokens[left:right], new_tokens[new_left:new_right]
                for old_value, new_value in zip(removed, added):
                    if old_value == new_value:
                        continue
                    if not (any(char.isdigit() for char in old_value + new_value)
                            or "." in old_value or "." in new_value
                            or old_value[:1] in {'"', "'"} or new_value[:1] in {'"', "'"}):
                        continue
                    line_number = j1 + offset + 1
                    result.append({
                        "kind": "mixed_value_hunk", "name": old_value,
                        "old": old_value, "new": new_value, "line": line_number,
                        "old_line": old_line, "new_line": new_line,
                        "span": {"start_line": line_number, "end_line": line_number},
                    })
    return result


def structural_changes(before: str, after: str) -> list[dict[str, object]]:
    """Summarize generic import/enum/branch hunks for evidence-only edits."""
    patterns = {
        "import": re.compile(r"^\s*(?:from\s+\S+\s+import|import\s+\S+)", re.I),
        "enum": re.compile(r"\b(?:enum|Enum|interface|trait)\b"),
        "branch": re.compile(r"\b(?:if|elif|else|match|case|switch|catch)\b"),
    }
    old_lines, new_lines = before.splitlines(), after.splitlines()
    changes: list[dict[str, object]] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
            None, old_lines, new_lines, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        removed, added = old_lines[i1:i2], new_lines[j1:j2]
        for line in [*removed, *added]:
            kind = next((name for name, pattern in patterns.items() if pattern.search(line)), None)
            if kind is None:
                continue
            start_line = (j1 + 1) if added else (i1 + 1)
            end_line = start_line + max(len(removed), len(added), 1) - 1
            changes.append({"kind": kind, "line": line.strip(),
                            "span": {"start_line": start_line, "end_line": end_line},
                            "old_value": "\n".join(removed).strip(),
                            "new_value": "\n".join(added).strip(),
                            "old_line": "\n".join(removed).strip(),
                            "new_line": "\n".join(added).strip()})
            break
    return changes


_CALL_PATTERN = re.compile(
    r"(?<![\w])(?:[A-Za-z_$][\w$]*(?:\s*(?:\.|::|->)\s*[A-Za-z_$][\w$]*)*)\s*\([^()\n]{0,240}\)"
)
_NON_EFFECT_PREFIXES = ("def ", "async def ", "function ", "fn ", "func ", "class ")


def _effect_signatures(source: str) -> list[tuple[int, str]]:
    """Extract comparable call signatures without treating one language AST as universal."""
    result: list[tuple[int, str]] = []
    for line_number, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "//", "/*", "*")):
            continue
        if stripped.casefold().startswith(_NON_EFFECT_PREFIXES):
            continue
        for match in _CALL_PATTERN.finditer(line):
            signature = re.sub(r"\s+", "", match.group(0))
            if signature.casefold() in {"if()", "for()", "while()", "switch()", "catch()"}:
                continue
            result.append((line_number, signature))
    return result


def effect_changes(before: str, after: str) -> list[dict[str, object]]:
    """Detect inserted, duplicated, moved, or removed call effects generically.

    This is intentionally a conservative lexical boundary. Language adapters
    can provide a stronger signature, but a green comparison must never grant
    an unreceipted effect merely because a Python parser happened to accept it.
    """
    old = _effect_signatures(before)
    new = _effect_signatures(after)
    old_counts, new_counts = Counter(item[1] for item in old), Counter(item[1] for item in new)
    old_heads = {signature.split("(", 1)[0] for _line, signature in old}
    new_heads = {signature.split("(", 1)[0] for _line, signature in new}
    changes: list[dict[str, object]] = []
    for signature, count in (new_counts - old_counts).items():
        # A changed argument, including a mixed member/numeric hunk, is a
        # value change owned by literal_changes.  Only a genuinely new call
        # head is an independent structural effect at this boundary.
        # An EXACT repeat of an existing call is a duplicated effect, never an
        # argument change, so the head exemption applies only to new signatures.
        if signature not in old_counts and signature.split("(", 1)[0] in old_heads:
            continue
        kind = "duplicated_effect" if signature in old_counts else "inserted_effect"
        lines = [line for line, value in new if value == signature]
        changes.append({"kind": kind, "effect": signature, "line": lines[0] if lines else "",
                        "count": count, "old_count": old_counts.get(signature, 0),
                        "new_count": count + old_counts.get(signature, 0)})
    for signature, count in (old_counts - new_counts).items():
        if signature not in new_counts and signature.split("(", 1)[0] in new_heads:
            continue
        lines = [line for line, value in old if value == signature]
        changes.append({"kind": "removed_effect", "effect": signature, "line": lines[0] if lines else "",
                        "count": count, "old_count": count, "new_count": new_counts.get(signature, 0)})
    common = old_counts.keys() & new_counts.keys()
    old_positions = {signature: [line for line, value in old if value == signature] for signature in common}
    new_positions = {signature: [line for line, value in new if value == signature] for signature in common}
    if len(old) == len(new) and [value for _, value in old] != [value for _, value in new]:
        for signature in common:
            if old_positions[signature] != new_positions[signature]:
                changes.append({"kind": "moved_effect", "effect": signature,
                                "old_lines": old_positions[signature],
                                "new_lines": new_positions[signature], "line": new_positions[signature][0]})
    return changes


def _named(change: Mapping[str, object], rows: Sequence[str]) -> bool:
    """Identify the constrained subject, not authorization for its new value."""
    for row in rows:
        text = str(row).casefold()
        name = str(change.get("name", "")).casefold()
        if name and len(name) >= 3 and (
                re.search(rf"\b{re.escape(name)}\b", text)
                or name.replace("_", " ") in text):
            return True
    return False


def _normalized_span(value: object) -> tuple[object, object] | None:
    if isinstance(value, Mapping):
        start = value.get("start", value.get("start_line"))
        end = value.get("end", value.get("end_line", start))
        return (start, end) if start is not None else None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and value:
        return (value[0], value[1] if len(value) > 1 else value[0])
    return None


def _host_provenance(receipt: Mapping[str, object]) -> bool:
    provenance = str(receipt.get("provenance", receipt.get("producer", ""))).casefold()
    actor = str(receipt.get("actor", "")).casefold()
    return (receipt.get("host_admitted") is True
            or provenance in {"host", "host_receipt", "execution", "test"}) \
        and actor not in {"model", "assistant", "solver"}


def make_scope_authorization(*, clause_id: str, cause_id: str, path: str,
                             span: object, old_value: object, new_value: object,
                             current_revision: str, provenance: str = "host_receipt",
                             receipt_id: str | None = None) -> dict[str, object]:
    """Produce the typed host receipt required before an evidence-only edit."""
    if not all(str(value).strip() for value in (clause_id, cause_id, path, current_revision)):
        raise ValueError("scope authorization requires clause, cause, path, and revision")
    normalized = _normalized_span(span)
    if normalized is None:
        raise ValueError("scope authorization requires a concrete span")
    if provenance.casefold() not in {"host_receipt", "host", "execution"}:
        raise ValueError("scope authorization provenance must be host-owned")
    return {
        "receipt_id": receipt_id or f"scope:{clause_id}:{cause_id}:{path}:{normalized[0]}:{normalized[1]}",
        "status": "ADMITTED", "host_admitted": True, "provenance": provenance,
        "clause_id": clause_id, "cause_id": cause_id, "path": path, "span": normalized,
        "old_value": old_value, "new_value": new_value, "revision": current_revision,
    }


def admit_scope_authorization(runtime: object, authorization: Mapping[str, object]) -> dict[str, object]:
    """Record one host-produced authorization on the existing runtime ledger.

    This is the normal producer boundary.  It does not infer permission from a
    model row or a numeric mention; callers must provide the complete typed
    receipt emitted by :func:`make_scope_authorization`.
    """
    if not isinstance(authorization, Mapping) or not _host_provenance(authorization):
        raise ValueError("scope authorization must be host-provenance typed receipt")
    required = ("receipt_id", "clause_id", "cause_id", "path", "span",
                "old_value", "new_value", "revision")
    if any(str(authorization.get(key, "")).strip() == "" for key in required
           if key not in {"old_value", "new_value"}) \
            or "old_value" not in authorization or "new_value" not in authorization:
        raise ValueError("scope authorization is missing a required identity field")
    if _normalized_span(authorization.get("span")) is None:
        raise ValueError("scope authorization requires a concrete span")
    status = str(authorization.get("status", "")).upper()
    if status not in {"ADMITTED", "SUPPORTED", "HOST_CHECKED"}:
        raise ValueError("scope authorization status is not admitted")
    receipt = dict(authorization)
    prior = list(getattr(runtime, "scope_lock_change_receipts", ()) or ())
    if not any(isinstance(item, Mapping) and item.get("receipt_id") == receipt["receipt_id"]
               for item in prior):
        prior.append(receipt)
    setattr(runtime, "scope_lock_change_receipts", tuple(prior))
    authorizations = list(getattr(runtime, "scope_lock_authorizations", ()) or ())
    if not any(isinstance(item, Mapping) and item.get("receipt_id") == receipt["receipt_id"]
               for item in authorizations):
        authorizations.append(receipt)
    setattr(runtime, "scope_lock_authorizations", tuple(authorizations))
    setattr(runtime, "_scope_lock_last_authorization", receipt)
    emit = getattr(getattr(runtime, "journal", None), "emit", None)
    if callable(emit):
        emit("scope_lock_authorization_admitted", producer="kernel_scope_lock",
             receipt_id=receipt["receipt_id"], clause_id=receipt["clause_id"],
             cause_id=receipt["cause_id"], dispatch=False)
    return dict(receipt)


def dispatch_scope_authorization(runtime: object, arguments: Mapping[str, object]) -> dict[str, object]:
    """Execute the host-owned resolver rendered by :func:`evaluate`."""
    if not isinstance(arguments, Mapping):
        raise ValueError("scope authorization arguments must be a mapping")
    required = ("clause_id", "cause_id", "path", "span", "old_value",
                "new_value", "current_revision")
    if any(key not in arguments for key in required):
        raise ValueError("scope authorization resolver arguments are incomplete")
    receipt = make_scope_authorization(
        clause_id=str(arguments["clause_id"]),
        cause_id=str(arguments["cause_id"]),
        path=str(arguments["path"]),
        span=arguments["span"],
        old_value=arguments["old_value"],
        new_value=arguments["new_value"],
        current_revision=str(arguments["current_revision"]),
        provenance=str(arguments.get("provenance", "host_receipt")),
        receipt_id=str(arguments["receipt_id"]) if arguments.get("receipt_id") else None,
    )
    return admit_scope_authorization(runtime, receipt)


def _authorized_change(change: Mapping[str, object], *, admitted_changes: Sequence[Mapping[str, object]],
                       host_checked_values: Mapping[str, object], clause_id: str = "",
                       cause_id: str = "", current_revision: str = "") -> bool:
    from kernel_aha10_honest_loop import authorize_typed_change

    name = str(change.get("name", "")).casefold()
    new_value = change.get("new")
    old_value = change.get("old")
    change_path = str(change.get("path", ""))
    change_span = _normalized_span(change.get("span"))
    for receipt in admitted_changes:
        if not isinstance(receipt, Mapping):
            continue
        if not _host_provenance(receipt):
            continue
        subject = str(receipt.get("subject", receipt.get("name", ""))).casefold()
        subject_names = {subject, subject.rsplit(".", 1)[-1]}
        status = str(receipt.get("status", receipt.get("decision", ""))).upper()
        receipt_id = str(receipt.get("receipt_id", "")).strip()
        allowed = receipt.get("new_value", receipt.get("value"))
        if not (receipt_id and status in {"ADMITTED", "SUPPORTED", "HOST_CHECKED"}
                and (receipt.get("old_value", receipt.get("old", old_value)) == old_value
                     or str(receipt.get("old_value", receipt.get("old", old_value))) == str(old_value))
                and (allowed == new_value or str(allowed) == str(new_value))):
            continue
        receipt_path = str(receipt.get("path", ""))
        receipt_span = _normalized_span(receipt.get("span"))
        if (receipt_path or receipt_span) and not (
                "old_value" in receipt or "old" in receipt):
            continue
        exact_span = bool(change_path and change_span is not None
                          and receipt_path == change_path and receipt_span == change_span)
        if not authorize_typed_change(change, receipt=receipt) and not exact_span:
            continue
        if change_path and receipt_path != change_path:
            continue
        if change_span is not None and receipt_span != change_span:
            continue
        if current_revision and str(receipt.get("revision", "")) != current_revision:
            continue
        if clause_id and str(receipt.get("clause_id", "")) != clause_id:
            continue
        if cause_id and str(receipt.get("cause_id", "")) != cause_id:
            continue
        # Nameless literal spans are valid only when the typed receipt pins the
        # exact path/span; a row mentioning a number never grants permission.
        if name and name not in subject_names and not (change_path and change_span is not None):
            continue
        if not name and not (change_path and change_span is not None):
            continue
        return True
    # A host-checked scalar, a model-mentioned number, or a stale row is not an
    # authorization receipt.  Only the typed receipt loop above can grant an
    # evidence-only change.
    for key, value in host_checked_values.items():
        key_text = str(key).casefold()
        if name not in {key_text, key_text.rsplit(".", 1)[-1]}:
            continue
        if isinstance(value, Mapping):
            if (_host_provenance(value) and value.get("receipt_id")
                    and _authorized_change(change, admitted_changes=(value,),
                                           host_checked_values={}, clause_id=clause_id,
                                           cause_id=cause_id, current_revision=current_revision)):
                return True
    return False


def _authorized_structural(change: Mapping[str, object], *,
                            admitted_changes: Sequence[Mapping[str, object]],
                            host_checked_values: Mapping[str, object],
                            current_revision: str = "", clause_id: str = "",
                            cause_id: str = "", path: str = "") -> bool:
    kind = str(change.get("kind", "")).casefold()
    line = str(change.get("line", "")).casefold()
    for receipt in admitted_changes:
        if not isinstance(receipt, Mapping) or not _host_provenance(receipt):
            continue
        status = str(receipt.get("status", receipt.get("decision", ""))).upper()
        cause = str(receipt.get("structural_kind", receipt.get("kind", ""))).casefold()
        scope = str(receipt.get("hunk", receipt.get("subject", ""))).casefold()
        receipt_span = _normalized_span(receipt.get("span"))
        change_span = _normalized_span(change.get("span"))
        if (str(receipt.get("receipt_id", "")).strip()
                and status in {"ADMITTED", "SUPPORTED", "HOST_CHECKED"}
                and (not current_revision or str(receipt.get("revision", "")) == current_revision)
                and (not clause_id or str(receipt.get("clause_id", "")) == clause_id)
                and (not cause_id or str(receipt.get("cause_id", "")) == cause_id)
                and bool(path and str(receipt.get("path", "")) == path)
                and receipt_span is not None and receipt_span == change_span
                and (receipt.get("old_value", change.get("old_value")) == change.get("old_value"))
                and (receipt.get("new_value", change.get("new_value")) == change.get("new_value"))
                and kind in cause and (not scope or scope in line or line in scope)):
            return True
    # Structural changes also require a typed, revision/clause/path-bound
    # receipt.  A boolean host_checked_values flag is intentionally not enough.
    return False


def _authorized_effect(change: Mapping[str, object], *,
                       admitted_changes: Sequence[Mapping[str, object]],
                       current_revision: str = "", clause_id: str = "",
                       cause_id: str = "") -> bool:
    effect = str(change.get("effect", "")).casefold()
    for receipt in admitted_changes:
        if not isinstance(receipt, Mapping) or not _host_provenance(receipt):
            continue
        status = str(receipt.get("status", receipt.get("decision", ""))).upper()
        receipt_effect = str(receipt.get("effect", receipt.get("effect_signature",
                                      receipt.get("subject", "")))).casefold()
        if (not str(receipt.get("receipt_id", "")).strip()
                or status not in {"ADMITTED", "SUPPORTED", "HOST_CHECKED"}
                or (current_revision and str(receipt.get("revision", "")) != current_revision)
                or (clause_id and str(receipt.get("clause_id", "")) != clause_id)
                or (cause_id and str(receipt.get("cause_id", "")) != cause_id)):
            continue
        if effect and (effect == receipt_effect or effect in str(receipt.get("hunk", "")).casefold()
                       or effect in str(receipt.get("new_text", "")).casefold()):
            return True
    return False


def evaluate(before: str, after: str, *, reason_codes: Sequence[str],
             rows: Sequence[str], admitted_changes: Sequence[Mapping[str, object]] = (),
             host_checked_values: Mapping[str, object] | None = None,
             path: str = "", current_revision: str = "", clause_id: str = "",
             cause_id: str = "") -> dict[str, object] | None:
    """``SCOPE_UNSUPPORTED`` hold, or ``None`` when the edit is admitted."""
    codes = {str(item) for item in reason_codes}
    if not (codes & EVIDENCE_CODES) or (codes & FAILING_CODES):
        return None
    checked = host_checked_values if isinstance(host_checked_values, Mapping) else {}
    changes = literal_changes(before, after)
    for change in changes:
        if path:
            change["path"] = path
        if current_revision:
            change["revision"] = current_revision
    unsupported = [change for change in changes
                   if not _authorized_change(change, admitted_changes=admitted_changes,
                                             host_checked_values=checked, clause_id=clause_id,
                                             cause_id=cause_id, current_revision=current_revision)]
    for change in structural_changes(before, after):
        if path:
            change["path"] = path
        if current_revision:
            change["revision"] = current_revision
        if not _authorized_structural(change, admitted_changes=admitted_changes,
                                      host_checked_values=checked,
                                      current_revision=current_revision, clause_id=clause_id,
                                      cause_id=cause_id, path=path):
            unsupported.append(change)
    for change in effect_changes(before, after):
        if not _authorized_effect(change, admitted_changes=admitted_changes,
                                  current_revision=current_revision, clause_id=clause_id,
                                  cause_id=cause_id):
            unsupported.append(change)
    if not unsupported:
        return None
    reason = ", ".join(sorted(codes & EVIDENCE_CODES))
    lines = []
    for item in unsupported:
        if "old" in item:
            detail = (f"`{_clip(str(item['old_line']).strip())}` -> `{_clip(str(item['new_line']).strip())}` "
                      f"({item.get('name') or 'literal'}: {item.get('old')} -> {item.get('new')})")
        elif str(item.get("kind", "")).endswith("effect"):
            detail = f"{item.get('kind')} `{_clip(str(item.get('effect', '')) )}`"
        else:
            detail = f"structural {item.get('kind', 'hunk')} `{_clip(str(item.get('line', '')))}`"
        lines.append(f"{HEADER} edit changes {detail}; no current host receipt authorizes it; "
                     f"the refusal reason was {reason}.")
    has_effect = any(str(item.get("kind", "")).endswith("effect") for item in unsupported)
    repair = ("Add a current host cause/effect receipt that names this effect or drop the hunk."
              if has_effect else
              "Add a row that names this value (new evidence) or drop the hunk.")
    lines.append(f"{SCOPE_UNSUPPORTED}: the edit was held, not applied. {repair}")
    first = unsupported[0]
    return {"status": SCOPE_UNSUPPORTED, "changes": unsupported, "reason": reason,
            "text": "\n".join(lines), "repair_action": {
                "tool": "authorize_scope_change",
                "arguments": {"clause_id": clause_id, "cause_id": cause_id,
                               "path": path, "span": first.get("span"),
                               "old_value": first.get("old"), "new_value": first.get("new"),
                               "current_revision": current_revision,
                               "provenance": "host_receipt"},
            }}


def _clip(text: str, limit: int = 80) -> str:
    return text if len(text) <= limit else text[:limit - 3] + "..."


def runtime_rows(runtime: object) -> list[str]:
    rows: list[str] = [str(item) for item in getattr(runtime, "scope_lock_rows", ()) or ()]
    ledger = getattr(runtime, "requirements_ledger", None)
    if isinstance(ledger, Mapping):
        for requirement in ledger.get("requirements", ()) or ():
            if isinstance(requirement, Mapping):
                rows.extend(str(requirement.get(key) or "") for key in ("exact_clause", "clause", "text"))
    return [row for row in rows if row.strip()]


def runtime_change_receipts(runtime: object) -> tuple[Mapping[str, object], ...]:
    values = (
        *(getattr(runtime, "scope_lock_change_receipts", ()) or ()),
        *(getattr(runtime, "scope_lock_authorizations", ()) or ()),
    )
    result: list[Mapping[str, object]] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, Mapping):
            continue
        receipt_id = str(item.get("receipt_id", ""))
        if receipt_id and receipt_id in seen:
            continue
        if receipt_id:
            seen.add(receipt_id)
        result.append(item)
    return tuple(result)


def runtime_checked_values(runtime: object) -> Mapping[str, object]:
    values = getattr(runtime, "scope_lock_host_checked_values", {})
    return values if isinstance(values, Mapping) else {}


def guard_path(path: object, after: str) -> str | None:
    """Central-write hook for an existing file; reads it only when the lock can apply."""
    if not enabled() or not _evidence_refusal_open():
        return None
    try:
        from pathlib import Path
        target = Path(str(path))
        if not target.is_file():
            return None
        before = target.read_text(encoding="utf-8", errors="replace")
    except (OSError, TypeError, ValueError):
        return None
    return guard(path, before, after)


def _evidence_refusal_open() -> bool:
    """An evidence-only done refusal is open AND the public checks are green now (post-green)."""
    try:
        from kernel_pilot import ACTIVE_RUNTIME
        runtime = ACTIVE_RUNTIME.get()
        from kernel_completion_owner import _decision_codes, _green_current
        codes = set(_decision_codes(runtime, None)) if runtime is not None else set()
        if not (codes & EVIDENCE_CODES) or (codes & FAILING_CODES):
            return False
        return bool(_green_current(runtime))
    except (ImportError, LookupError, AttributeError, TypeError, ValueError):
        return False


def guard(path: object, before: str, after: str) -> str | None:
    """Central-write hook: the held ``[SCOPE]`` text, or ``None``."""
    if not enabled():
        return None
    try:
        from kernel_pilot import ACTIVE_RUNTIME
        runtime = ACTIVE_RUNTIME.get()
    except (ImportError, LookupError):
        return None
    if runtime is None:
        return None
    try:
        from kernel_completion_owner import _decision_codes
        codes = _decision_codes(runtime, None)
    except (ImportError, AttributeError, TypeError, ValueError):
        return None
    result = evaluate(
        before, after, reason_codes=codes, rows=runtime_rows(runtime),
        admitted_changes=runtime_change_receipts(runtime),
        host_checked_values=runtime_checked_values(runtime), path=str(path),
        current_revision=str(getattr(runtime, "scope_lock_current_revision", "") or ""),
        clause_id=str(getattr(runtime, "scope_lock_clause_id", "") or ""),
        cause_id=str(getattr(runtime, "scope_lock_cause_id", "") or ""),
    )
    try:
        runtime._scope_lock_last_result = result
        runtime._scope_lock_recovery_action = (
            result.get("repair_action") if isinstance(result, Mapping) else None
        )
    except (AttributeError, TypeError):
        pass
    if result is None:
        return None
    emit = getattr(getattr(runtime, "journal", None), "emit", None)
    if callable(emit):
        try:
            emit("scope_lock_held", producer="kernel_scope_lock", dispatch=False, path=str(path),
                 reason=result["reason"], changes=result["changes"][:6])
        except (TypeError, ValueError):
            pass
    return str(result["text"])


__all__ = ["ENV", "EVIDENCE_CODES", "HEADER", "SCOPE_UNSUPPORTED", "enabled", "evaluate",
           "admit_scope_authorization", "dispatch_scope_authorization", "make_scope_authorization",
           "guard", "literal_changes", "structural_changes", "effect_changes",
           "runtime_change_receipts", "runtime_checked_values",
           "runtime_rows"]
