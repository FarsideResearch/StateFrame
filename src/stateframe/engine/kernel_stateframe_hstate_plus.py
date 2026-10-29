"""History-arm H+state+ presentation and completion seams.

The module is intentionally history-specific.  It adds a replaceable or
append-only state tail, a small public clause ledger, and revision-scoped done
handling without changing the frame renderer or the ordinary history menu.
All host-only classifications stay out of the rendered tail.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from kernel_context import token_estimate
from kernel_measurement import digest
from kernel_verdict import (
    ACCEPTED,
    REFUSED,
    CompletionAssessment,
    CompletionBasis,
    CompletionDecision,
    CompletionReducer,
)


MODE_ENV = "STATEFRAME_HSTATE_PLUS"
TAIL_ENV = "STATEFRAME_HSTATE_PLUS_TAIL"
SCHEMA = "stateframe-hstate-plus/1"
TAIL_PLACEMENTS = frozenset({"replace", "append"})
_TEST_KINDS = frozenset({"test", "run_tests", "pytest", "probe"})
_PASS = frozenset({"PASS", "PASSED", "GREEN", "SUCCESS", "OK", "SATISFIED", "RESOLVED"})
_FAIL = frozenset({"FAIL", "FAILED", "RED", "ERROR", "COUNTEREXAMPLE", "VIOLATED",
                   "UNSATISFIED", "PRODUCT_RED", "REGRESSION"})
_INTERNAL_LABELS = (
    (re.compile(r"\bNOT_ENTAILED\b"), "unproven"),
    (re.compile(r"\bSHADOW\b"), "unclassified"),
    (re.compile(r"\bADVISORY\b"), "informational"),
)
_INTERNAL_KEYS = {"tier": "level", "admission": "host_classification",
                  "admitted": "host_accepted"}
# attempt assessment: a run-wide cap that edits cannot replenish (done -> cosmetic edit ->
# new revision -> challenge treadmill).  Held in host state on the runtime.
CHALLENGE_RUN_CAP = 2
NO_AGREE_TEXT = "Do not change correct code merely to agree with this challenge."
UNBOUND_TEXT = ("not a demonstrated failure: the host check's expected value is not "
                "bound to the ticket; check before editing")


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = source.get(MODE_ENV, "off").strip().lower() or "off"
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def tail_mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = source.get(TAIL_ENV, "append").strip().lower() or "append"
    if value not in TAIL_PLACEMENTS:
        raise ValueError(f"{TAIL_ENV} must be replace or append")
    return value


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return mode(environ) == "on"


def _value(value: object, *names: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("text", "content", "detail", "message", "description", "value"):
            result = _text(value.get(key))
            if result:
                return result
        return ""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return " ".join(item for item in (_text(item) for item in value) if item)
    return str(value).strip()


def _history_selection(runtime: object) -> bool:
    modes = getattr(runtime, "modes", None)
    if not isinstance(modes, Mapping) or "render_selection" not in modes:
        return True
    value = modes.get("render_selection")
    value = getattr(value, "value", value)
    return str(value).upper() not in {"ENFORCE", "BOTH"}


def history_arm(runtime: object, *, provider_request: bool = True) -> bool:
    """Return true only for the active provider-bound history presentation."""
    if not provider_request or not _history_selection(runtime):
        return False
    configured = getattr(runtime, "stateframe_hstate_plus_mode", None)
    if configured is None:
        try:
            configured = mode()
        except ValueError:
            return False
    return str(configured).strip().lower() == "on"


def _tail_text(blocks: Sequence[object]) -> str:
    from kernel_history_state_tail import render_state_tail

    return render_state_tail(blocks)


def _sanitize(value: object) -> object:
    if isinstance(value, str):
        result = value
        for pattern, replacement in _INTERNAL_LABELS:
            result = pattern.sub(replacement, result)
        return result
    if isinstance(value, Mapping):
        return {_INTERNAL_KEYS.get(str(key), str(key)): _sanitize(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize(item) for item in value)
    return value


def _render_witness(row: Mapping[str, object]) -> str:
    witness = row.get("witness")
    if not isinstance(witness, Mapping):
        witness = {}
    return (f"witness: observed={_text(witness.get('observed'))}; "
            f"expected={_text(witness.get('expected'))}; "
            f"target={_text(witness.get('target')) or 'host check'}")


def _render_verdict(row: Mapping[str, object]) -> str:
    verdict = str(row.get("verdict", "UNPROBED")).upper()
    if verdict not in {"VIOLATED", "SATISFIED", "UNPROBED", "UNKNOWN"}:
        verdict = "UNPROBED"
    line = f"{row.get('id', 'C?')} | {verdict} | {_text(row.get('clause'))}"
    if verdict == "VIOLATED":
        line += " | " + _render_witness(row)
    elif verdict == "UNKNOWN":
        reason = (_text(row.get("reason"))
                  if os.environ.get("STATEFRAME_ORACLE_GROUNDING", "off").strip().lower() == "on"
                  else "")
        line += " | " + (f"reason: {reason}; " if reason else "") + UNBOUND_TEXT
    elif verdict == "SATISFIED":
        line += (f" | receipt: {_text(row.get('receipt_id')) or 'host receipt'}; "
                 f"revision: {_text(row.get('revision')) or 'unknown'}")
    return line


def render_tail(*, checklist: Sequence[Mapping[str, object]] = (),
                verdicts: Sequence[Mapping[str, object]] = (),
                full_suite: Mapping[str, object] | None = None,
                applied: object = None,
                advisories: Sequence[object] = ()) -> str:
    """Render only public H+state+ facts; no internal tier vocabulary."""
    lines = [
        "[H+state+ checklist]",
        "Clause checklist (files and symbols named at turn 1):",
    ]
    if checklist:
        for row in checklist:
            targets = _text(row.get("targets")) or "target unresolved"
            lines.append(f"{row.get('id', 'C?')} | targets: {targets} | {_text(row.get('text'))}")
    else:
        lines.append("No public clause text is available yet.")
    lines.extend(("[H+state+ clause verdicts]", "Clause verdicts (current revision):"))
    if verdicts:
        lines.extend(_render_verdict(row) for row in verdicts)
    else:
        lines.append("C1 | UNPROBED | no clause probe is recorded")
    suite = dict(full_suite or {})
    status = str(suite.get("status", "UNPROBED")).upper() or "UNPROBED"
    if status not in _PASS | _FAIL | {"UNPROBED"}:
        status = "UNPROBED"
    scope = _text(suite.get("scope")) or "full suite"
    revision = _text(suite.get("revision")) or "unknown"
    lines.append(f"Last full suite run: {status}; scope={scope}; revision={revision}.")
    if applied:
        lines.extend(("[H+state+ applied receipt]", _text(applied)))
    clean_advisories = [_text(item) for item in advisories if _text(item)]
    if clean_advisories:
        lines.append("Informational notes (not a completion gate): " + "; ".join(clean_advisories[:4]))
    return _text(_sanitize("\n".join(lines)))


@dataclasses.dataclass(frozen=True)
class TailResult:
    messages: tuple[dict, ...]
    report: dict[str, object]


def _tail_report(messages: Sequence[Mapping[str, object]], tail: Mapping[str, object], *,
                 placement: str, replaced: int, active: bool) -> dict[str, object]:
    tail_tokens = token_estimate(tail) if active else 0
    return {
        "schema": SCHEMA,
        "placement": placement,
        "prefix_hash": digest(messages),
        "tail_hash": digest(tail) if active else None,
        "prefix_message_count": len(messages),
        "tail_tokens": tail_tokens,
        "replacement_tail_tokens": tail_tokens,
        "append_tail_tokens": tail_tokens,
        "prefill_tokens": token_estimate({"messages": [*messages, tail]}) if active else token_estimate({"messages": messages}),
        "tail_chars": len(str(tail.get("content", ""))) if active else 0,
        "replaced_tail_count": replaced if active and placement == "replace" else 0,
        "prefix_unchanged": True,
    }


def apply_tail(messages: Sequence[Mapping[str, object]], blocks: Sequence[object], *,
               enabled: bool, placement: str = "replace", provider_request: bool = True) -> TailResult:
    """Place one H+state+ message while preserving all prior history bytes."""
    if placement not in TAIL_PLACEMENTS:
        raise ValueError(f"{TAIL_ENV} must be replace or append")
    original = [copy.deepcopy(dict(message)) for message in messages]
    if not enabled or not provider_request:
        return TailResult(tuple(original), _tail_report(original, {}, placement=placement,
                                                        replaced=0, active=False))
    from kernel_history_state_tail import is_state_tail, strip_state_tail

    content = _tail_text(blocks)
    tail = {"role": "user", "content": content}
    if placement == "replace":
        prefix = strip_state_tail(original)
        result = prefix + [tail]
        replaced = len(original) - len(prefix)
    else:
        prefix = original
        result = original + [tail]
        replaced = 0
    report = _tail_report(prefix, tail, placement=placement, replaced=replaced, active=True)
    stable_prefix = strip_state_tail(original)
    stable_result_prefix = strip_state_tail(result[:-1])
    report["prefix_hash"] = digest(stable_prefix)
    report["stable_prefix_hash"] = report["prefix_hash"]
    report["prefix_message_count"] = len(stable_prefix)
    report["prefix_unchanged"] = stable_prefix == stable_result_prefix
    report["prior_tail_count"] = sum(1 for message in original if is_state_tail(message))
    replacement_prefix = strip_state_tail(original)
    report["replacement_prefill_tokens"] = token_estimate(
        {"messages": [*replacement_prefix, tail]})
    report["append_prefill_tokens"] = token_estimate(
        {"messages": [*original, tail]})
    report["append_state_tokens"] = token_estimate(tail)
    return TailResult(tuple(result), report)


def _ticket_text(runtime: object, frame: Mapping[str, object] | None = None) -> str:
    values: list[object] = []
    if isinstance(frame, Mapping):
        ledger = frame.get("requirements_ledger")
        if isinstance(ledger, Mapping):
            ticket = ledger.get("ticket")
            values.extend((ticket, ledger.get("ticket_text"), ledger.get("text")))
            for source in ledger.get("sources", ()) if isinstance(ledger.get("sources"), Sequence) else ():
                if isinstance(source, Mapping) and isinstance(source.get("text"), str):
                    values.append(source["text"])
    ledger = getattr(runtime, "requirements_ledger", None)
    if isinstance(ledger, Mapping):
        values.extend((ledger.get("ticket"), ledger.get("ticket_text"), ledger.get("text")))
    row = getattr(getattr(runtime, "owner", None), "task_row", {})
    if isinstance(row, Mapping):
        # Same reader the ledger uses: `brief` may name a public ticket FILE.
        try:
            from kernel_requirements_ledger import _ticket_from_row
            root = getattr(getattr(runtime, "owner", None), "root", None)
            values.append(_ticket_from_row(row, root=root)[1])
        except Exception:  # noqa: BLE001 - absent/unsupported rows fall through
            pass
        values.extend(row.get(key) for key in (
            "brief", "task_brief", "description", "request", "msg", "symptom",
        ))
    for value in values:
        text = _text(value)
        if text:
            return text
    return ""


def _sentences(ticket: str) -> tuple[str, ...]:
    if not ticket:
        return ()
    if os.environ.get("STATEFRAME_ORACLE_GROUNDING", "off").strip().lower() == "on":
        try:
            from kernel_oracle_grounding import extract_ticket_clauses
            return extract_ticket_clauses(ticket)
        except (ImportError, TypeError, ValueError):
            pass
    try:
        from kernel_requirement_alignment import extract_ticket_sentences
        values = extract_ticket_sentences(ticket)
        if values:
            return values
    except (ImportError, TypeError, ValueError):
        pass
    return tuple(item.strip() for item in re.split(r"(?<=[.!?])\s+|\n+", ticket) if item.strip())


def _target(row: Mapping[str, object], runtime: object) -> str:
    file_name = _text(row.get("target_file") or row.get("target_path") or row.get("file") or row.get("path"))
    symbol = _text(row.get("target_symbol") or row.get("entry_point") or row.get("symbol")
                   or row.get("function") or row.get("name"))
    if not file_name:
        try:
            from kernel_requirement_alignment import target_file_from_binding
            file_name = target_file_from_binding(row)
        except (ImportError, TypeError, ValueError):
            pass
    if not file_name:
        target = getattr(runtime, "target", None)
        root = getattr(getattr(runtime, "owner", None), "root", None)
        if isinstance(target, (str, os.PathLike)):
            try:
                file_name = str(Path(target).resolve().relative_to(Path(root).resolve())) if root else str(target)
            except (OSError, ValueError):
                file_name = str(target)
    if not symbol:
        symbol = _text(getattr(getattr(runtime, "owner", None), "fn", ""))
    return f"{file_name}:{symbol}" if file_name and symbol else file_name or symbol or "target unresolved"


_SYMBOL_FILE_CAP = 4
_SCAN_FILE_CAP = 400


def _product_files(runtime: object) -> tuple[Path, ...]:
    cached = getattr(runtime, "_hstate_plus_product_files", None)
    if isinstance(cached, tuple):
        return cached
    root = getattr(getattr(runtime, "owner", None), "root", None)
    files: list[Path] = []
    if root:
        base = Path(root)
        try:
            for path in sorted(base.rglob("*.py")):
                rel = path.relative_to(base)
                parts = rel.parts
                if any(part.startswith(".") or part in {"tests", "test", "__pycache__"}
                       for part in parts[:-1]) or rel.name.startswith("test_") \
                        or rel.name.endswith("_test.py") or rel.name == "conftest.py":
                    continue
                files.append(path)
                if len(files) >= _SCAN_FILE_CAP:
                    break
        except OSError:
            files = []
    result = tuple(files)
    try:
        setattr(runtime, "_hstate_plus_product_files", result)
    except AttributeError:
        pass
    return result


def _symbol_files(runtime: object, name: str) -> tuple[str, ...]:
    """Product files that DEFINE ``name`` (def/class/assignment); else files that use it."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{2,}", name or ""):
        return ()
    cache = getattr(runtime, "_hstate_plus_symbol_files", None)
    if not isinstance(cache, dict):
        cache = {}
        try:
            setattr(runtime, "_hstate_plus_symbol_files", cache)
        except AttributeError:
            pass
    if name in cache:
        return cache[name]
    root = Path(getattr(getattr(runtime, "owner", None), "root", "") or ".")
    defines = re.compile(rf"^\s*(?:async\s+def|def|class)\s+{re.escape(name)}\b|^\s*{re.escape(name)}\s*[:=]",
                         re.M)
    uses = re.compile(rf"\b{re.escape(name)}\b")
    defined: list[str] = []
    used: list[str] = []
    for path in _product_files(runtime):
        try:
            text = path.read_text(encoding="utf-8", errors="strict")
        except (OSError, UnicodeError):
            continue
        rel = path.relative_to(root).as_posix() if path.is_absolute() else path.as_posix()
        if defines.search(text):
            defined.append(rel)
        elif uses.search(text):
            used.append(rel)
    result = tuple((defined or used)[:_SYMBOL_FILE_CAP])
    cache[name] = result
    return result


def _ledger(runtime: object, frame: Mapping[str, object] | None = None) -> Mapping[str, object] | None:
    ledger = frame.get("requirements_ledger") if isinstance(frame, Mapping) else None
    if isinstance(ledger, Mapping):
        return ledger
    ledger = getattr(runtime, "requirements_ledger", None)
    if isinstance(ledger, Mapping):
        return ledger
    cached = getattr(runtime, "_hstate_plus_ledger", None)
    if isinstance(cached, Mapping):
        return cached
    if getattr(runtime, "_hstate_plus_ledger_attempted", False):
        return None
    try:
        setattr(runtime, "_hstate_plus_ledger_attempted", True)
    except AttributeError:
        return None
    owner = getattr(runtime, "owner", None)
    row = getattr(owner, "task_row", None)
    if not isinstance(row, Mapping):
        return None
    try:
        from kernel_requirements_ledger import build_ledger
        built = build_ledger(row, root=getattr(owner, "root", None))
    except Exception:  # noqa: BLE001 - the checklist degrades to the runtime target
        return None
    setattr(runtime, "_hstate_plus_ledger", built)
    return built


def checklist(runtime: object, frame: Mapping[str, object] | None = None) -> tuple[dict[str, object], ...]:
    cached = getattr(runtime, "_hstate_plus_checklist", None)
    if isinstance(cached, tuple) and cached:
        return cached
    ticket = _ticket_text(runtime, frame)
    sentences = _sentences(ticket)
    ledger = _ledger(runtime, frame)
    requirements = ledger.get("requirements", ()) if isinstance(ledger, Mapping) else ()
    bindings = ledger.get("bindings", ()) if isinstance(ledger, Mapping) else ()
    rows: list[dict[str, object]] = []
    for index, sentence in enumerate(sentences, 1):
        targets = []
        for requirement in requirements if isinstance(requirements, Sequence) else ():
            if not isinstance(requirement, Mapping):
                continue
            exact = _text(requirement.get("exact_clause") or requirement.get("clause"))
            if exact and exact not in sentence and sentence not in exact:
                continue
            candidates = []
            contract = requirement.get("contract")
            binding_ids = (contract.get("subject_binding_ids", ())
                           if isinstance(contract, Mapping) else ())
            if isinstance(bindings, Sequence) and isinstance(binding_ids, Sequence):
                for binding in bindings:
                    if not isinstance(binding, Mapping) or str(binding.get("id")) not in {
                            str(item) for item in binding_ids}:
                        continue
                    name = _text(binding.get("name"))
                    files = _symbol_files(runtime, name)
                    if files:
                        candidates.extend(f"{item}:{name}" for item in files)
                    elif name and binding.get("status") != "unresolved":
                        candidates.append(_target({"binding": binding,
                                                   "target_symbol": name}, runtime))
            if (not candidates or any(requirement.get(key) not in (None, "") for key in (
                    "target_file", "target_path", "file", "path", "target_symbol",
                    "entry_point", "symbol", "function"))):
                candidates.append(_target(requirement, runtime))
            for target in candidates:
                if target and target not in targets:
                    targets.append(target)
        # Identifiers quoted in the clause itself, resolved against product files; a
        # resolved surface outranks the generic runtime-target fallback.
        quoted: list[str] = []
        for name in re.findall(r"`([A-Za-z_][A-Za-z0-9_.]*)`", sentence):
            for item in _symbol_files(runtime, name.rsplit(".", 1)[-1]):
                label = f"{item}:{name.rsplit('.', 1)[-1]}"
                if label not in quoted and label not in targets:
                    quoted.append(label)
        if quoted:
            fallback = _target({}, runtime)
            targets = quoted + [item for item in targets if item != fallback]
        if not targets:
            targets.append(_target({}, runtime))
        rows.append({"id": f"C{index}", "text": sentence, "targets": ", ".join(targets[:6])})
    result = tuple(rows)
    if result and ticket:
        try:
            # Turn-one checklist: fixed for the run (stable tail bytes, cache friendly).
            setattr(runtime, "_hstate_plus_checklist", result)
        except AttributeError:
            pass
    return result


def _revision(runtime: object, explicit: object = None) -> str:
    if explicit not in (None, ""):
        return str(explicit)
    # r40 review: the green lock's _green_revision is the revision at which green
    # was OBSERVED (stale after an edit); the lock's own current product revision
    # is authoritative when present.
    current = getattr(getattr(runtime, "green_lock", None), "_current_revision", None)
    if callable(current):
        try:
            value = current()
            if value not in (None, ""):
                return str(value)
        except (AttributeError, KeyError, OSError, TypeError, ValueError):
            pass
    for value in (
        getattr(runtime, "current_revision", None), getattr(runtime, "product_revision", None),
        getattr(runtime, "revision", None), getattr(getattr(runtime, "green_lock", None), "_green_revision", None),
    ):
        if value not in (None, ""):
            return str(value)
    binding = getattr(runtime, "binding", None)
    if callable(binding):
        try:
            value = binding()
            if isinstance(value, Mapping):
                for key in ("product_revision", "revision", "snapshot_digest"):
                    if value.get(key) not in (None, ""):
                        return str(value[key])
                return "sha256:" + hashlib.sha256(
                    json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()
        except (AttributeError, OSError, TypeError, ValueError):
            pass
    return "unknown"


def _receipts(runtime: object, frame: Mapping[str, object] | None = None) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    owner = getattr(runtime, "owner", None)
    values = [getattr(owner, "_receipts", ()), getattr(runtime, "test_receipts", ()),
              getattr(runtime, "probe_receipts", ()), getattr(runtime, "host_check_receipts", ())]
    if isinstance(frame, Mapping):
        values.extend(frame.get(key, ()) for key in ("test_receipts", "probe_receipts", "receipts", "counterexamples"))
    for value in values:
        if isinstance(value, Mapping):
            value = value.values()
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            continue
        for item in value:
            if isinstance(item, Mapping):
                result.append(copy.deepcopy(dict(item)))
    return result


def _literal_values(value: object) -> tuple[object, ...]:
    result: list[object] = []
    if isinstance(value, Mapping):
        for child in value.values():
            result.extend(_literal_values(child))
        return tuple(result)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            result.extend(_literal_values(child))
        return tuple(result)
    if isinstance(value, (str, int, float, bool)):
        return (value,)
    return ()


def _sentence_literals(sentence: str) -> tuple[object, ...]:
    values: list[object] = []
    for match in re.findall(r"`([^`\n]+)`|'([^'\n]+)'|\"([^\"\n]+)\"", sentence):
        values.extend(item for item in match if item)
    values.extend(int(item) for item in re.findall(r"(?<![A-Za-z_])\d+(?![A-Za-z_])", sentence))
    return tuple(values)


def _same_literal(left: object, right: object) -> bool:
    if isinstance(left, (int, float)) and not isinstance(left, bool):
        try:
            return float(left) == float(right)
        except (TypeError, ValueError):
            return str(left) == str(right).strip()
    return _text(left).casefold() == _text(right).casefold()


def _clause_match(row: Mapping[str, object], clause: Mapping[str, object], count: int) -> bool:
    ident = _text(row.get("clause_id") or row.get("requirement_id") or row.get("check_id"))
    if ident:
        normalized = ident.upper().replace("CLAUSE-", "C").replace("CLAUSE_", "C")
        if normalized == str(clause.get("id", "")).upper():
            return True
    hint = set(re.findall(r"[A-Za-z0-9_]{3,}", _text(row.get("clause") or row.get("description")).lower()))
    text = set(re.findall(r"[A-Za-z0-9_]{3,}", _text(clause.get("text")).lower()))
    return count == 1 or bool(hint and text and len(hint & text) >= max(1, min(2, len(hint))))


def _aligned_failure(row: Mapping[str, object], clauses: Sequence[Mapping[str, object]], runtime: object) -> dict[str, object] | None:
    status = _text(row.get("result_class") or row.get("status") or row.get("result") or row.get("underlying_status")).upper()
    observed = _text(row.get("observed") or row.get("actual") or row.get("got") or row.get("detail"))
    expected = row.get("expected", row.get("expectation", row.get("want")))
    if not (status in _FAIL or "FAIL" in status or "ERROR" in status or "VIOLAT" in status
            or (observed and expected is not None and not _same_literal(observed, expected))):
        return None
    expected_values = _literal_values(expected)
    if not expected_values:
        return None
    for index, clause in enumerate(clauses, 1):
        if not _clause_match(row, clause, len(clauses)):
            continue
        sentence = _text(clause.get("text"))
        literals = _sentence_literals(sentence)
        matched = [value for value in expected_values
                   if any(_same_literal(value, literal) for literal in literals)]
        if not matched:
            continue
        result = copy.deepcopy(dict(row))
        result.update({
            "clause_id": clause.get("id", f"C{index}"),
            "clause": sentence,
            "alignment_evidence": {"method": "ticket_literals", "sentence": sentence,
                                    "matched_literals": copy.deepcopy(matched),
                                    "target": _target(row, runtime)},
            "literal_grounded": True,
            "repairable": True,
        })
        return result
    return None


def verdict_table(runtime: object, frame: Mapping[str, object] | None = None,
                  *, revision: object = None):
    from kernel_stateframe_get_sure import build_table

    ticket = _ticket_text(runtime, frame)
    current = _revision(runtime, revision)
    evidence = _receipts(runtime, frame)
    clauses = checklist(runtime, frame)
    # Give multi-clause evidence a stable public locator while retaining the
    # original receipt facts for witnesses and receipts.
    for row in evidence:
        for clause in clauses:
            if _clause_match(row, clause, len(clauses)):
                row.setdefault("clause_id", clause["id"])
                break
    table = build_table(ticket, evidence, revision=current)
    return table


def _aligned_violations(runtime: object, frame: Mapping[str, object] | None = None,
                        *, revision: object = None) -> tuple[dict[str, object], ...]:
    clauses = checklist(runtime, frame)
    return tuple(item for item in (
        _aligned_failure(row, clauses, runtime) for row in _receipts(runtime, frame)
    ) if item is not None and (
        not revision or _text(item.get("revision") or item.get("product_revision")) in {"", str(revision)}
    ))


def full_suite_status(runtime: object, frame: Mapping[str, object] | None = None,
                      *, revision: object = None) -> dict[str, object]:
    current = _revision(runtime, revision)
    candidates = []
    for row in _receipts(runtime, frame):
        selectors = row.get("selectors", row.get("selector", ()))
        selector_text = _text(selectors).lower()
        scope = _text(row.get("scope") or row.get("suite") or row.get("target"))
        if "full" in selector_text or "suite" in selector_text or "all" in selector_text \
                or "full" in scope.lower() or "suite" in scope.lower():
            candidates.append(row)
    if not candidates:
        return {"status": "UNPROBED", "scope": "full suite", "revision": current}
    row = candidates[-1]
    status = _text(row.get("result_class") or row.get("status") or row.get("result"))
    if not status and row.get("exit_code") is not None:
        status = "PASS" if row.get("exit_code") == 0 else "FAIL"
    status = status.upper() if status else "UNPROBED"
    if status in {"PASSED", "GREEN", "SUCCESS", "OK"}:
        status = "PASS"
    elif status in {"FAILED", "RED", "ERROR", "REGRESSION"}:
        status = "FAIL"
    return {"status": status if status in {"PASS", "FAIL"} else "UNPROBED",
            "scope": _text(row.get("scope") or row.get("suite") or row.get("selectors")) or "full suite",
            "revision": _text(row.get("revision") or row.get("product_revision") or current) or "unknown",
            "receipt_id": _text(row.get("receipt_id") or row.get("event_id"))}


def _basis(runtime: object, *, revision: object, evidence: object) -> CompletionBasis:
    creator = getattr(runtime, "_completion_basis", None)
    if callable(creator):
        try:
            result = creator(evidence=evidence)
            if isinstance(result, CompletionBasis):
                return result
        except (AttributeError, TypeError, ValueError):
            pass
    binding = getattr(runtime, "binding", None)
    try:
        binding_value = binding() if callable(binding) else getattr(runtime, "product_revision", revision)
    except (AttributeError, OSError, TypeError, ValueError):
        binding_value = revision
    return CompletionBasis.capture(str(revision), binding_value, evidence=evidence)


def _menu_names(runtime: object) -> tuple[str, ...]:
    value = getattr(runtime, "completion_menu", ())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(str(item) for item in value)
    return ("done", "read", "edit", "str_replace", "run_tests", "predict")


def _record(runtime: object, event: str, **fields: object) -> None:
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if callable(emit):
        emit(event, producer="kernel_stateframe_hstate_plus", dispatch=False, **fields)


def _challenge_text(rows: Sequence[Mapping[str, object]], *, budget_enabled: bool = False) -> str:
    """Render a challenge with a witness and the budget arm's exact ending."""
    clauses = []
    for row in rows:
        ident = _text(row.get("clause_id") or row.get("check_id")) or "clause"
        target = _target(row, object())
        observed = _text(row.get("observed") or row.get("actual") or row.get("got"))
        expected = _text(row.get("expected", row.get("expectation", row.get("want"))))
        clauses.append(f"{ident}: {_text(row.get('clause'))}; target: {target}; "
                       f"witness: observed={observed or '?'} expected={expected or '?'}")
    prefix = ("DONE CHALLENGE (COUNTEREXAMPLE): open violated aligned clause(s): "
              + "; ".join(clauses) + ". Repair or run the relevant check. ")
    if budget_enabled:
        from kernel_challenge_budget import CHALLENGE_ENDING
        return prefix + CHALLENGE_ENDING
    return prefix + NO_AGREE_TEXT + " The full menu remains available; a second done at this revision is accepted."


def _submission_record(runtime: object, *, revision: str, reason: str,
                       open_rows: Sequence[Mapping[str, object]]) -> None:
    """attempt assessment: attempt_status (submitted) is separate from assessment. An accepted
    override NEVER turns open clauses into passes: the assessment stays known_violation."""
    record = {
        "attempt_status": "submitted",
        "assessment": "known_violation" if open_rows else "no_known_violation",
        "override_reason": reason,
        "revision": revision,
        "open_clause_ids": [_text(row.get("clause_id") or row.get("check_id")) for row in open_rows],
    }
    setattr(runtime, "_hstate_plus_submission_record", record)
    _record(runtime, "hstate_plus_submission_record", **record)


def done_decision(runtime: object, *, revision: object = None, admissible: bool = True,
                  detail: str = "", commit=None, frame: Mapping[str, object] | None = None,
                  request_id: object = None):
    """Apply the one-challenge-per-revision rule, or return ``None`` if no aligned violation is open."""
    current = _revision(runtime, revision)
    budget_module = None
    budget = None
    budget_key = str(current)
    try:
        import kernel_challenge_budget as budget_module
        budget = budget_module.budget_for(runtime)
        if budget is not None:
            budget_key = budget_module.artifact_revision(runtime, fallback=current)
            terminal_plan = budget_module.plan(runtime, budget_key, request=request_id)
            if terminal_plan.get("action") in {"terminal", "terminal_replay"}:
                durable = budget_module.durable_decision(runtime)
                if durable is not None:
                    return durable
    except (ImportError, OSError, TypeError, ValueError):
        budget_module = None
        budget = None
    open_rows = _aligned_violations(runtime, frame, revision=current)
    if not open_rows and admissible:
        # r41b: the aligned-clause filter is narrower than "a host check found a
        # live counterexample on this revision"; with the switch on, those rows
        # get the same one-shot budgeted challenge (kernel_live_counterexample_challenge).
        try:
            from kernel_live_counterexample_challenge import enabled as _live_ce_enabled
            from kernel_live_counterexample_challenge import live_rows as _live_ce_rows
            if _live_ce_enabled(runtime):
                receipts = tuple(
                    row for row in _receipts(runtime, frame)
                    if _text(row.get("revision") or row.get("product_revision")) in {"", str(current)})
                open_rows = _live_ce_rows(runtime, frame, receipts=receipts)
                from kernel_live_counterexample_challenge import diagnose as _live_ce_diagnose
                _record(runtime, "live_counterexample_done_check",
                        receipts=len(receipts), **_live_ce_diagnose(runtime, open_rows, frame))
                if open_rows:
                    _record(runtime, "live_counterexample_done_gate",
                            check_ids=[_text(row.get("check_id")) for row in open_rows])
        except (ImportError, AttributeError, TypeError, ValueError):
            open_rows = ()
    if not open_rows:
        if budget is not None:
            # The previous challenge rows are historical evidence.  A repaired
            # artifact must be allowed to reach the ordinary clean reducer.
            setattr(runtime, "_hstate_plus_challenge_rows", ())
        return None
    if not admissible:
        # The caller already refuses this done (e.g. public tests red): the normal
        # history path owns it; a challenge is never spent on it.
        return None
    basis = _basis(runtime, revision=current, evidence=open_rows)
    if budget is not None:
        artifact_revision = budget_module.artifact_revision
        assessment_for = budget_module.assessment_for
        challenge_class = budget_module.challenge_class
        clause_ids = budget_module.clause_ids
        commit_challenge = budget_module.commit_challenge
        commit_submission = budget_module.commit_submission
        plan = budget_module.plan
        remember_decision = budget_module.remember_decision
        replay_decision = budget_module.replay_decision
        key = artifact_revision(runtime, basis=basis, fallback=current)
        budget_plan = plan(runtime, key, request=request_id)
        if budget_plan.get("action") in {"terminal", "terminal_replay"}:
            durable = budget_module.durable_decision(runtime, fallback_basis=basis)
            return durable if durable is not None else CompletionDecision(
                ACCEPTED, basis, accepted=True, sealed=True,
                detail="done accepted from durable terminal submission")
        if budget_plan.get("action") == "replay":
            cached = replay_decision(runtime, budget_plan)
            if cached is not None:
                return cached
            if budget_plan.get("previous") == "challenge":
                return CompletionDecision(
                    REFUSED, basis, reason_codes=("HSTATE_PLUS_DONE_CHALLENGE",),
                    detail=_challenge_text(open_rows, budget_enabled=True),
                    artifact_preserved=True)
            # A replay of a non-terminal challenge returns the same refusal and
            # never enters the reducer.  The accepted replay case is terminal
            # and was handled above.
            return CompletionDecision(
                REFUSED, basis, reason_codes=("HSTATE_PLUS_DONE_CHALLENGE",),
                detail=_challenge_text(open_rows, budget_enabled=True),
                artifact_preserved=True)
        evidence = budget_module.open_clause_rows(rows=open_rows)
        klass = challenge_class((), rows=evidence)
        ids = clause_ids(rows=evidence)
        assessment = assessment_for(klass, ids, rows=evidence)
        if budget_plan.get("action") in {"submit", "cap", "reconcile"}:
            reducer = getattr(runtime, "completion_reducer", None)
            if reducer is None:
                reducer = CompletionReducer()
                setattr(runtime, "completion_reducer", reducer)
            accepted = reducer.decide(CompletionAssessment(
                basis=basis, admissible=True,
                detail="done accepted after one challenge at this revision",
                tool_menu=_menu_names(runtime), selectors=tuple(
                    str(item) for item in getattr(getattr(runtime, "owner", None), "f2p_ids", ())),
                path=None, commit=commit))
            if accepted.accepted:
                status = "capped" if budget_plan["action"] == "cap" else "submitted"
                # STATEFRAME_SUBMIT_IDEMPOTENT only: plan() never says "reconcile" off.
                reconcile_kwargs = ({"override_reason": "submission_reconciled"}
                                    if budget_plan["action"] == "reconcile" else {})
                record = commit_submission(
                    runtime, budget_plan, attempt_status=status,
                    assessment=assessment, open_clause_ids=ids,
                    challenge_class=klass, terminal_result=accepted, **reconcile_kwargs)
                setattr(runtime, "_hstate_plus_submission_record", copy.deepcopy(record))
                setattr(runtime, "_hstate_plus_done_override", True)
                _record(runtime, "hstate_plus_done_override", revision=key,
                        reason=record["override_reason"], challenge_class=klass,
                        attempt_status=status, assessment=record["assessment"],
                        clause_ids=list(ids), targets=[_target(row, runtime) for row in open_rows])
                _record(runtime, "hstate_plus_submission_record", **record)
                remember_decision(runtime, budget_plan, accepted)
            return accepted
        if (budget_plan.get("action") == "challenge"
                and set(_menu_names(runtime)) <= {"done"}
                and _menu_names(runtime)):
            reducer = getattr(runtime, "completion_reducer", None)
            if reducer is None:
                reducer = CompletionReducer()
                setattr(runtime, "completion_reducer", reducer)
            accepted = reducer.decide(CompletionAssessment(
                basis=basis, admissible=True,
                detail="done accepted because done is the sole offered action",
                tool_menu=_menu_names(runtime), selectors=tuple(
                    str(item) for item in getattr(getattr(runtime, "owner", None), "f2p_ids", ())),
                path=None, commit=commit))
            if accepted.accepted:
                record = commit_submission(
                    runtime, {**budget_plan, "action": "submit"},
                    attempt_status="submitted", assessment=assessment,
                    open_clause_ids=ids, challenge_class=klass,
                    terminal_result=accepted, override_reason="sole_tool_submission")
                setattr(runtime, "_hstate_plus_submission_record", copy.deepcopy(record))
                setattr(runtime, "_hstate_plus_done_override", True)
                _record(runtime, "hstate_plus_done_override", revision=key,
                        reason="sole_tool_submission", challenge_class=klass,
                        attempt_status="submitted", assessment=record["assessment"],
                        clause_ids=list(ids), targets=[_target(row, runtime) for row in open_rows])
                _record(runtime, "hstate_plus_submission_record", **record)
                remember_decision(runtime, budget_plan, accepted)
            return accepted
        commit_challenge(runtime, budget_plan, challenge_class=klass)
        setattr(runtime, "_hstate_plus_challenged_revision", key)
        setattr(runtime, "_hstate_plus_challenge_count", int(budget_plan["spent"]) + 1)
        setattr(runtime, "_hstate_plus_challenge_rows", copy.deepcopy(open_rows))
        challenged = CompletionDecision(
            REFUSED, basis, reason_codes=("HSTATE_PLUS_DONE_CHALLENGE",),
            detail=_challenge_text(open_rows, budget_enabled=True), artifact_preserved=True)
        _record(runtime, "hstate_plus_done_challenge", revision=key,
                challenge_class=klass, run_challenge_count=int(budget_plan["spent"]) + 1,
                run_challenge_cap=int(budget_plan["cap"]), clause_ids=list(ids),
                targets=[_target(row, runtime) for row in open_rows])
        remember_decision(runtime, budget_plan, challenged)
        return challenged
    challenged = getattr(runtime, "_hstate_plus_challenged_revision", None)
    spent = int(getattr(runtime, "_hstate_plus_challenge_count", 0) or 0)
    cap_reached = spent >= CHALLENGE_RUN_CAP and str(challenged) != str(current)
    if str(challenged) == str(current) or cap_reached:
        reducer = getattr(runtime, "completion_reducer", None)
        if reducer is None:
            reducer = CompletionReducer()
            setattr(runtime, "completion_reducer", reducer)
        assessment = CompletionAssessment(
            basis=basis, admissible=True, detail="done challenged once; override accepted",
            tool_menu=_menu_names(runtime), selectors=tuple(
                str(item) for item in getattr(getattr(runtime, "owner", None), "f2p_ids", ())
            ), path=None, commit=commit,
        )
        decision = reducer.decide(assessment)
        setattr(runtime, "_hstate_plus_done_override", True)
        reason = "run_challenge_cap" if cap_reached else "second_done_same_revision"
        _record(runtime, "hstate_plus_done_override", revision=current, reason=reason,
                clause_ids=[_text(row.get("clause_id") or row.get("check_id")) for row in open_rows],
                targets=[_target(row, runtime) for row in open_rows])
        _submission_record(runtime, revision=current, reason=reason, open_rows=open_rows)
        return decision
    setattr(runtime, "_hstate_plus_challenged_revision", current)
    setattr(runtime, "_hstate_plus_challenge_count", spent + 1)
    setattr(runtime, "_hstate_plus_challenge_rows", copy.deepcopy(open_rows))
    text = _challenge_text(open_rows)
    decision = CompletionDecision(
        REFUSED, basis, reason_codes=("HSTATE_PLUS_DONE_CHALLENGE",), detail=text,
        artifact_preserved=True,
    )
    _record(runtime, "hstate_plus_done_challenge", revision=current, challenge_class="COUNTEREXAMPLE",
            run_challenge_count=spent + 1, run_challenge_cap=CHALLENGE_RUN_CAP,
            clause_ids=[_text(row.get("clause_id") or row.get("check_id")) for row in open_rows],
            targets=[_target(row, runtime) for row in open_rows])
    return decision


def tool_menu(runtime: object, tools: Sequence[Mapping[str, object]]) -> list[dict]:
    """Return the normal full history menu; predict is an additive capability."""
    result = [copy.deepcopy(dict(item)) for item in tools if isinstance(item, Mapping)]
    if not any(_text(_value(item.get("function"), "name")) == "predict" for item in result):
        try:
            from kernel_stateframe_get_sure import predict_tool
            result.append(predict_tool())
        except ImportError:
            pass
    names = tuple(_text(_value(item.get("function"), "name")) for item in result)
    runtime.completion_menu = names
    runtime._question_done_available = "done" in names
    _record(runtime, "hstate_plus_full_menu", offered=list(names))
    return result


def _canonical_test_arguments(arguments: Mapping[str, object] | None) -> str:
    args = dict(arguments or {})
    if "selectors" in args:
        args = {"selectors": args.get("selectors")}
    elif "selector" in args:
        args = {"selectors": args.get("selector")}
    return json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def green_lock_guard(runtime: object, kind: object, arguments: Mapping[str, object] | None = None) -> str | None:
    """Keep the green lock narrow: only an identical test rerun is refused."""
    name = str(kind or "").strip().lower()
    if name not in _TEST_KINDS or not history_arm(runtime):
        return None
    lock = getattr(runtime, "green_lock", None)
    if lock is None or str(getattr(lock, "mode", "off")) not in {"on", "terminal"}:
        return None
    try:
        lock.sync()
    except (AttributeError, TypeError, ValueError):
        pass
    revision = _revision(runtime)
    if not bool(getattr(lock, "_green", False)) or str(getattr(lock, "_green_revision", "")) != str(revision):
        return None
    prior = getattr(runtime, "_hstate_plus_last_test", None)
    if not isinstance(prior, Mapping):
        return None
    if str(prior.get("revision")) != str(revision) or prior.get("signature") != _canonical_test_arguments(arguments):
        return None
    from kernel_post_green_menu import pending_verification
    if pending_verification(runtime):
        return None
    result = _text(prior.get("result")) or "PASS"
    return f"GREEN LOCK: unchanged revision; result: {result}; test rerun is byte-identical."


def note_test_receipt(runtime: object, receipt: Mapping[str, object]) -> None:
    if not history_arm(runtime):
        return
    name = _text(receipt.get("kind") or receipt.get("tool")).lower()
    if name not in _TEST_KINDS:
        return
    status = _text(receipt.get("result_class") or receipt.get("status") or receipt.get("result")).upper()
    if not status and receipt.get("exit_code") is not None:
        status = "PASS" if receipt.get("exit_code") == 0 else "FAIL"
    if status not in _PASS:
        setattr(runtime, "_hstate_plus_last_test", None)
        return
    revision = _text(receipt.get("revision") or receipt.get("product_revision")
                     or receipt.get("after_revision")) or _revision(runtime)
    arguments = receipt.get("arguments") if isinstance(receipt.get("arguments"), Mapping) else receipt
    setattr(runtime, "_hstate_plus_last_test", {
        "revision": revision, "signature": _canonical_test_arguments(arguments),
        "result": "PASS", "receipt_id": _text(receipt.get("receipt_id") or receipt.get("event_id")),
    })


def note_applied_edit(runtime: object, receipt: Mapping[str, object]) -> str | None:
    if not history_arm(runtime) or _text(receipt.get("result_class")).upper() != "APPLIED":
        return None
    revision = _revision(runtime)
    path = _text(receipt.get("target_path") or receipt.get("path")) or "source"
    changed = _text(receipt.get("changed_region") or receipt.get("diff") or receipt.get("detail"))
    marker = f"APPLIED at revision {revision}; changed region ({path}):\n{changed or 'edit result recorded by host'}"
    setattr(runtime, "_hstate_plus_last_applied", marker)
    _record(runtime, "hstate_plus_applied_receipt", revision=revision, target=path,
            receipt_id=_text(receipt.get("receipt_id") or receipt.get("event_id")))
    return marker


def ground_verdicts(rows: Sequence[Mapping[str, object]], runtime: object,
                    frame: Mapping[str, object] | None = None) -> list[dict[str, object]]:
    """observable binding UNBOUND_OBSERVABLE: a VIOLATED verdict is rendered only when its
    witness is a public test result or a ticket-literal-aligned violation. A host
    check asserting an expectation the ticket does not state (ldev_09
    ``late_fee_cents`` vs ticket ``fee_cents``) is UNKNOWN: it never blocks and
    never passes."""
    oracle_grounding = False
    oracle_ground = None
    oracle_ticket = ""
    oracle_ledger = None
    if os.environ.get("STATEFRAME_ORACLE_GROUNDING", "off").strip().lower() == "on":
        try:
            from kernel_oracle_grounding import enabled as oracle_enabled, ground_verdict
            oracle_grounding = oracle_enabled()
            oracle_ground = ground_verdict
            oracle_ticket = _ticket_text(runtime, frame)
            oracle_ledger = _ledger(runtime, frame)
        except (ImportError, TypeError, ValueError):
            pass
    receipts = {(_text(row.get("receipt_id") or row.get("event_id"))): row
                for row in _receipts(runtime, frame)}
    aligned = {_text(item.get("clause_id")) for item in _aligned_violations(runtime, frame)}
    grounded: list[dict[str, object]] = []
    for row in rows:
        item = dict(row)
        receipt = receipts.get(_text(item.get("receipt_id")), {})
        if oracle_grounding and oracle_ground is not None:
            if "expected" not in item and isinstance(receipt, Mapping) and "expected" in receipt:
                item["expected"] = copy.deepcopy(receipt.get("expected"))
            item = oracle_ground(item, ticket=oracle_ticket, ledger=oracle_ledger)
        if str(item.get("verdict", "")).upper() == "VIOLATED":
            receipt = receipts.get(_text(item.get("receipt_id")), {})
            kind = _text(receipt.get("kind") or receipt.get("tool")).lower()
            public = kind in _TEST_KINDS and kind != "probe"
            if not public and _text(item.get("id")) not in aligned:
                item = {key: value for key, value in item.items() if key != "witness"}
                item["verdict"] = "UNKNOWN"
        grounded.append(item)
    return grounded


def tail_blocks(runtime: object, frame: Mapping[str, object], tools: object, *,
                base_blocks: Sequence[object] = (), working: str = "", path: str = "") -> list[str]:
    del tools, working, path
    table = verdict_table(runtime, frame, revision=_revision(runtime))
    checklist_rows = checklist(runtime, frame)
    rows = ground_verdicts([dict(row) for row in table.rows], runtime, frame)
    advisories = []
    for row in _receipts(runtime, frame):
        status = _text(row.get("result_class") or row.get("status")).upper()
        if status in _FAIL and not any(_clause_match(row, clause, len(checklist_rows)) for clause in checklist_rows):
            advisories.append("a failed host observation is not literally aligned to a checklist clause")
            break
    suite = full_suite_status(runtime, frame, revision=_revision(runtime))
    clean_base = [_sanitize(block) for block in base_blocks]
    return [*clean_base, render_tail(
        checklist=checklist_rows, verdicts=rows, full_suite=suite,
        applied=getattr(runtime, "_hstate_plus_last_applied", None), advisories=advisories,
    )]


def prediction(runtime: object, arguments: Mapping[str, object]) -> dict[str, object]:
    from kernel_stateframe_get_sure import predict_runtime
    return predict_runtime(runtime, arguments)


__all__ = [
    "ACCEPTED", "MODE_ENV", "REFUSED", "SCHEMA", "TAIL_ENV", "TailResult",
    "apply_tail", "checklist", "done_decision", "enabled", "full_suite_status",
    "CHALLENGE_RUN_CAP", "ground_verdicts", "green_lock_guard", "history_arm", "mode", "note_applied_edit", "note_test_receipt",
    "prediction", "render_tail", "tail_blocks", "tail_mode", "tool_menu", "verdict_table",
]
