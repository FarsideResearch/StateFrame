"""StateFrame shadow projector (side experiment, offline-only).

Folds an archived ReAct transcript (results/react_transcripts*/<fn>.json) into a
sequence of deterministic StateFrame snapshots — one per turn — persisted as JSONL
and NEVER shown to the model. This is the shadow-first half of the StateFrame Loop
verdict (sol_answers/stateframe_1.md): prove the kernel can reconstruct useful
state from recorded evidence before any of it becomes part of the live contract.

Hard boundaries (visibility contract):
  * imports nothing from react_repair.py and touches no live loop state
  * output is written under results/stateframes_shadow/ only
  * no LLM, no network, no sandbox mutation — pure fold over recorded messages

Persistence: append-only JSONL, one header + one snapshot per turn, keyed by
(source_sha256, turn_index). Re-running is idempotent; any conflict or malformed
existing line fails closed (raise, never repair).
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys
from dataclasses import dataclass, field
from typing import Iterator

SCHEMA = "poc.stateframe.shadow/2"

# Recorded transcripts do not carry a live source resolver or content revision.
# Keep captured evidence bounded, but make truncation explicit instead of turning
# it into a plausible-looking revision or source snapshot.
_MAX_SOURCE_CAPTURE_CHARS = 64 * 1024
_REVISION_UNAVAILABLE_REASON = "not_recorded_in_transcript"
_HISTORICAL_RESOLVER_UNAVAILABLE_REASON = "offline_projection_has_no_resolver"

# Tool names that mutate the sandbox vs. inspect it (mirrors the loop's TOOLS).
_EDIT_TOOLS = {"edit", "str_replace", "file_create"}
_TEST_TOOLS = {"run_tests", "test"}
_READ_TOOLS = {"read", "grep"}

# Observation prefixes that are negative work: the turn produced no forward motion.
_NEGATIVE_MARKERS = (
    "str_replace NO-OP", "edit NO-OP", "(no grep matches)", "(no matches)",
    "[kernel] REPEAT", "[kernel] you have repeated", "IsADirectoryError",
    "[kernel] you have been cycling", "edit REJECTED", "str_replace REJECTED",
    "file_create REJECTED", "[kernel] pre-existing test files are READ-ONLY",
)

_NEGATIVE_KIND_BY_MARKER = {
    "str_replace NO-OP": "EDIT_NOOP",
    "edit NO-OP": "EDIT_NOOP",
    "str_replace REJECTED": "EDIT_REJECTED",
    "edit REJECTED": "EDIT_REJECTED",
    "file_create REJECTED": "FILE_CREATE_REJECTED",
    "(no grep matches)": "DEAD_QUERY",
    "(no matches)": "DEAD_QUERY",
    "IsADirectoryError": "DEAD_QUERY",
    "[kernel] REPEAT": "REPEAT_BLOCKED",
    "[kernel] you have repeated": "REPEAT_BLOCKED",
    "[kernel] you have been cycling": "REPEAT_BLOCKED",
    "[kernel] pre-existing test files are READ-ONLY": "EDIT_REJECTED",
    "HARNESS_ERROR": "HARNESS_ERROR",
    "TEST_EDIT_AFTER_RED": "TEST_EDIT_AFTER_RED",
    "DONE_REFUSED": "DONE_REFUSED",
    "NO_ACTION_TURN": "NO_ACTION_TURN",
    "UNKNOWN_ACTION": "UNKNOWN_ACTION",
}

_RE_SUSPECT = re.compile(
    r"suspect function is\s*`([^`]+)`\s*in\s*`([^`]+)`", re.IGNORECASE)
_RE_FAILED = re.compile(r"^FAILED\s+(\S+)", re.MULTILINE)
_RE_NODEID = re.compile(r"(tests?/[\w/\.\-]+\.py::[\w\[\]\-\.]+)")
_RE_ERRLINE = re.compile(
    r"^E\s+([A-Za-z_][\w\.]*(?:Error|Exception|Failure)):?\s*(.*)$",
    re.MULTILINE,
)


def _sha256_bytes(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


def _canon(obj: object) -> str:
    """Canonical float-free serialization for digests and persisted records."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)


def _unavailable(reason: str) -> dict:
    """Return a typed absence marker; ``UNKNOWN`` is not a revision value."""
    return {"status": "unavailable", "value": None, "reason": reason}


def _revision_state(value: object, reason: str = _REVISION_UNAVAILABLE_REASON) -> dict:
    """Normalize an optional revision into a bound-or-unavailable state."""
    if isinstance(value, dict):
        status = value.get("status")
        revision = value.get("value")
        if status == "bound" and _known_revision(revision):
            return {"status": "bound", "value": revision.strip()}
        return _unavailable(str(value.get("reason") or reason))
    if _known_revision(value):
        return {"status": "bound", "value": value.strip()}
    return _unavailable(reason)


def _known_revision(value: object) -> bool:
    return (isinstance(value, str) and bool(value.strip())
            and value.strip().upper() not in {
                "UNKNOWN", "UNAVAILABLE", "MISSING", "NONE", "NULL", "N/A"})


def _revision_is_bound(value: object) -> bool:
    return (isinstance(value, dict) and value.get("status") == "bound"
            and _known_revision(value.get("value")))


def _mapping_value(mapping: object, keys: tuple[str, ...]) -> object:
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _source_revision(source: dict, keys: tuple[str, ...]) -> dict:
    """Read an explicitly recorded revision without inferring one from content."""
    containers: list[dict] = [source]
    for key in ("world", "environment", "metadata", "source"):
        nested = source.get(key)
        if isinstance(nested, dict):
            containers.append(nested)
    for container in containers:
        value = _mapping_value(container, keys)
        if value is not None:
            return _revision_state(value, "recorded_revision_is_unknown")
    return _unavailable(_REVISION_UNAVAILABLE_REASON)


def _event_id(kind: str, payload: object, position: int) -> str:
    """Return a content-and-position identity for one recorded event.

    ``position`` is the source-message or trace position, not the model step.  A
    single step may produce several receipts, so step numbers are deliberately
    never used as event identities.
    """
    return _sha256_bytes(_canon({"kind": kind, "position": position,
                                 "payload": payload}).encode())


def _source_identity(source: dict) -> str:
    """Identity stable across later transcript evidence appended to the source."""
    return _sha256_bytes(_canon({
        "fn": source.get("fn"),
        "rel_path": _source_rel_path(source),
        "brief": _brief_from_source(source),
    }).encode())


def _text_or_none(value: object) -> str | None:
    if isinstance(value, str):
        value = value.strip()
        return value or None
    if isinstance(value, pathlib.PurePath):
        value = str(value).strip()
        return value or None
    return None


def _brief_from_source(source: dict) -> str:
    messages = source.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = str(message.get("content") or "")
        if content.startswith("Bug report:") or "suspect function" in content:
            return content
    return ""


def _explicit_rel_path(source: dict) -> str | None:
    """Read a path explicitly carried by a transcript/source row.

    The transcript filename is deliberately not considered a target path. Nested
    source/task/row records are supported because archived source rows use more
    than one envelope shape.
    """
    containers: list[dict] = [source]
    for key in ("source", "task", "row"):
        nested = source.get(key)
        if isinstance(nested, dict):
            containers.append(nested)
    for container in containers:
        for key in ("rel_path", "rel", "path"):
            value = _text_or_none(container.get(key))
            if value and not pathlib.PurePath(value).is_absolute():
                return value
    return None


def _source_rel_path(source: dict) -> str | None:
    explicit = _explicit_rel_path(source)
    if explicit is not None:
        return explicit
    match = _RE_SUSPECT.search(_brief_from_source(source))
    return _text_or_none(match.group(2)) if match else None


def _objective_from_source(source: dict) -> dict:
    """Project only objective atoms present in the recorded source evidence."""
    brief = _brief_from_source(source)
    symbol = _text_or_none(source.get("fn"))
    path = _source_rel_path(source)
    suspect = _RE_SUSPECT.search(brief)
    if suspect:
        symbol = _text_or_none(suspect.group(1)) or symbol
        path = path or _text_or_none(suspect.group(2))

    signature = ""
    error_line = _RE_ERRLINE.search(brief)
    failed = _RE_FAILED.search(brief)
    if error_line:
        signature = (error_line.group(1) + ": " + error_line.group(2).strip()).strip()
    elif failed:
        signature = failed.group(1)

    acceptance: list[str] = []
    if failed:
        acceptance.append(failed.group(1))
    acceptance.extend(node for node in _RE_NODEID.findall(brief) if node not in acceptance)

    low = brief.lower()
    constraints: list[str] = []
    if "verify with the repository" in low or "existing tests" in low:
        constraints.append("verify_with_existing_tests")
    if "fix `" in low or "fix the" in low or "reproduce" in low:
        constraints.append("repair_named_function_only")

    target: dict[str, str] = {}
    if path is not None:
        target["path"] = path
    if symbol is not None:
        target["symbol"] = symbol

    objective: dict = {
        "target": target,
        "kind": "single_function_repair",
        "failure_signature": signature,
        "acceptance_tests": acceptance[:4],
        "constraints": constraints,
    }
    if symbol is not None:
        # Keep the v1 atom for consumers that still read objective.fn.
        objective["fn"] = symbol
    if brief:
        objective["source_digest"] = _sha256_bytes(brief.encode())[:23]
    return objective


@dataclass
class Turn:
    index: int
    assistant: dict
    observations: list[dict] = field(default_factory=list)
    position: int | None = None
    kernel_events: list[dict] = field(default_factory=list)

    @property
    def event_id(self) -> str:
        """Stable identity for this complete assistant/observation exchange."""
        position = self.index if self.position is None else self.position
        return _event_id("turn", {
            "assistant": self.assistant,
            "observations": self.observations,
            "kernel_events": self.kernel_events,
        }, position)


@dataclass(frozen=True)
class RenderBudget:
    """Caps for the model-facing copy of a stored StateFrame.

    The projector's persisted frame remains lossless.  These limits apply only
    when an explicit consumer asks for a bounded rendering.
    """

    max_verdicts: int = 16
    max_negative_work: int = 16

    def __post_init__(self) -> None:
        for name in ("max_verdicts", "max_negative_work"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


def _render_tail(items: object, limit: int) -> tuple[list[dict], int]:
    if not isinstance(items, list):
        return [], 0
    omitted = max(0, len(items) - limit)
    return (list(items[-limit:]) if limit else [], omitted)


def render_frame(stored_frame: dict, budget: RenderBudget | None = None) -> dict:
    """Return a bounded, detached rendering of an authoritative stored frame.

    This is intentionally opt-in: ``project_file`` persists the complete frame,
    while callers that need model context must explicitly request this lossy
    view.  Omission counts are carried only by the rendered object.
    """
    if not isinstance(stored_frame, dict):
        raise TypeError("stored_frame must be a dict")
    budget = budget or RenderBudget()
    rendered = json.loads(_canon(stored_frame))
    facts = rendered.get("facts")
    if not isinstance(facts, dict):
        raise ValueError("stored_frame.facts must be a dict")

    verdicts, omitted_verdicts = _render_tail(
        facts.get("verdicts"), budget.max_verdicts)
    negative_work, omitted_negative_work = _render_tail(
        rendered.get("negative_work"), budget.max_negative_work)
    facts["verdicts"] = verdicts
    rendered["negative_work"] = negative_work
    rendered["render_meta"] = {
        "kind": "stateframe_render",
        "version": 1,
        "omitted": {
            "verdicts": omitted_verdicts,
            "negative_work": omitted_negative_work,
        },
    }
    return rendered


def load_transcript(path: pathlib.Path) -> dict:
    raw = path.read_bytes()
    d = json.loads(raw)
    if not isinstance(d, dict) or not isinstance(d.get("messages"), list):
        raise ValueError(f"not a transcript dict with messages: {path}")
    d["_source_sha256"] = _sha256_bytes(raw)
    d["_source_path"] = str(path)
    rel_path = _source_rel_path(d)
    if rel_path is not None:
        d["_rel_path"] = rel_path
    return d


def _is_trim_marker(m: dict) -> bool:
    c = str(m.get("content", ""))
    return m.get("role") == "user" and c.startswith("[kernel] (") and \
        "trimmed to fit context" in c


def reconstruct_messages(
    transcript_path: pathlib.Path,
    messages: list[dict],
    *,
    max_step: int | None = None,
) -> list[dict]:
    """E316 reader half: splice back exchanges the live loop trimmed (persisted to
    results/react_events<TAG>/<fn>.jsonl by react_repair). Trim markers and
    previously-re-trimmed markers are dropped; order is chronological (trims are
    appended in trim order, each dump preserving message order). ``max_step``
    admits only explicitly step-attributed trim events at or before the cutoff;
    un-attributed events are excluded rather than leaked into an earlier prefix.

    R2 repair: only splice when `messages` itself carries evidence that trimming
    actually happened (a trim marker present). Without that, the events file for
    this filename/tag pairing has nothing to do with THIS `messages` view — the
    previous unconditional splice let evicted content from far later in the run
    (or, in the causal-cutoff regression, from an unrelated synthetic future)
    leak into an earlier/shorter view that never experienced any trim at all.
    """
    if not any(_is_trim_marker(m) for m in messages):
        return messages
    events_dir = transcript_path.parent.parent / \
        transcript_path.parent.name.replace("react_transcripts", "react_events")
    ev_path = events_dir / (transcript_path.stem + ".jsonl")
    if not ev_path.exists():
        return messages
    trimmed: list[dict] = []
    for line in ev_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)  # malformed events fail closed, same as snapshots
        if max_step is not None:
            event_step = rec.get("step")
            if not isinstance(event_step, int) or event_step > max_step:
                continue
        m = rec.get("msg", {})
        if rec.get("kind") == "trimmed" and not _is_trim_marker(m):
            trimmed.append(m)
    head, tail = messages[:2], [m for m in messages[2:] if not _is_trim_marker(m)]
    return head + trimmed + tail


def iter_turns(messages: list[dict]) -> Iterator[Turn]:
    """One turn = one assistant generation + the tool/user observations that follow,
    ending before the next assistant message. System messages and trim markers are
    context, not turns."""
    idx = 0
    cur: Turn | None = None
    for position, m in enumerate(messages):
        role = m.get("role")
        if role == "assistant":
            if cur is not None:
                yield cur
            cur = Turn(index=idx, assistant=m, position=position)
            idx += 1
        elif role in ("tool", "user") and cur is not None:
            cur.observations.append(m)
    if cur is not None:
        yield cur


def _kernel_event_turn_index(event: dict) -> int | None:
    """Return an explicit 0-based turn index, never infer one from event order."""
    step = event.get("step")
    if isinstance(step, int) and not isinstance(step, bool):
        return max(0, step - 1)
    turn = event.get("turn")
    if isinstance(turn, int) and not isinstance(turn, bool):
        return max(0, turn)
    return None


def attach_kernel_events(turns: list[Turn], trace: object) -> list[Turn]:
    """Attach only causally safe kernel events to reconstructed turns.

    Explicit ``step``/``turn`` coordinates are respected. Final-only events with
    no coordinate may use an ordered assistant-content prefix as evidence; an
    event without that anchor is retained on the terminal turn with an explicit
    unavailable attribution marker and is never visible in an earlier prefix.
    """
    if not turns or not isinstance(trace, list):
        return turns
    attached = [Turn(index=turn.index, assistant=turn.assistant,
                      observations=turn.observations, position=turn.position,
                      kernel_events=list(turn.kernel_events)) for turn in turns]
    terminal = len(attached) - 1
    last_content_match = -1
    for trace_position, raw_event in enumerate(trace):
        if not isinstance(raw_event, dict):
            continue
        event_turn = _kernel_event_turn_index(raw_event)
        event = json.loads(_canon(raw_event))
        event["event_id"] = _event_id("kernel", event, trace_position)
        if event_turn is None:
            # A run-final event may still carry an exact, ordered content anchor.
            # Treat that as explicit evidence, but never guess from event order;
            # an event with no anchor stays terminal and keeps a typed missing
            # step. This preserves the R2 cutoff for genuinely un-attributed
            # future evidence while recovering the R4 anchors the corpus has.
            detail = str(event.get("detail") or "")
            destination = None
            if detail:
                for candidate in range(last_content_match + 1, len(attached)):
                    content = str(attached[candidate].assistant.get("content") or "")
                    if content.startswith(detail):
                        destination = candidate
                        last_content_match = candidate
                        break
            if destination is None:
                destination = terminal
                event["turn"] = None
                event["step"] = None
                event["step_status"] = "unavailable"
            else:
                event["turn"] = destination
                event["step"] = destination + 1
                event["step_status"] = "attributed"
        else:
            destination = min(max(event_turn, 0), terminal)
            event["step_status"] = "attributed"
        attached[destination].kernel_events.append(event)
    return attached


def _captured_text(value: object, missing_reason: str) -> dict:
    if not isinstance(value, str):
        return _unavailable(missing_reason)
    digest = _sha256_bytes(value.encode())
    if len(value) > _MAX_SOURCE_CAPTURE_CHARS:
        return {
            "status": "observed_but_omitted",
            "value": value[:_MAX_SOURCE_CAPTURE_CHARS],
            "sha256": digest,
            "complete": False,
            "reason": "capture_limit_exceeded",
        }
    return {"status": "observed", "value": value, "sha256": digest,
            "complete": True}


def normalize_action(assistant: dict) -> dict | None:
    """Structured view of what the model tried to do this turn. Prose is never a
    fact; only the tool call (or its absence) is recorded."""
    calls = assistant.get("tool_calls") or []
    if not calls:
        return None
    first_call = calls[0] if isinstance(calls[0], dict) else {}
    fn = first_call.get("function", {})
    if not isinstance(fn, dict):
        fn = {}
    name = str(fn.get("name", "?"))
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except (ValueError, TypeError):
        args = {"_unparsed": True}
    if not isinstance(args, dict):
        args = {"_unparsed": True}
    target = args.get("path") or args.get("target") or args.get("pattern") or ""
    kind = ("edit" if name in _EDIT_TOOLS else
            "test" if name in _TEST_TOOLS else
            "read" if name in _READ_TOOLS else
            "finish" if name == "done" else "other")
    action = {"tool": name, "kind": kind, "target": str(target)[:200],
              "args_digest": _sha256_bytes(_canon(args).encode())[:23]}
    revision_evidence: dict[str, dict] = {}
    for field_name, aliases in {
        "product_revision": ("product_revision", "content_revision"),
        "test_harness_revision": ("test_harness_revision", "harness_revision"),
        "environment_revision": ("environment_revision", "environment_id"),
    }.items():
        value = _mapping_value(args, aliases)
        if value is not None:
            revision_evidence[field_name] = _revision_state(
                value, "recorded_revision_is_unknown")
    if revision_evidence:
        action["revision_evidence"] = revision_evidence
    if kind == "edit":
        action["edit_request"] = {
            "span": _captured_text(
                args.get("old_str", args.get("old_text")),
                "edit_request_has_no_source_span",
            ),
            "replacement": _captured_text(
                args.get("new_str", args.get("new_text")),
                "edit_request_has_no_replacement_text",
            ),
        }
    return action


def _line_starts(lines: list[str], marker: str) -> bool:
    marker = marker.lower()
    return any(line.strip().lower().startswith(marker) for line in lines)


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def normalize_observation(observations: list[dict], action_kind: str | None = None) -> dict:
    text = "\n".join(str(o.get("content", "")) for o in observations)
    test_exit: int | None = None
    match = re.search(r"pytest exit\s*(-?\d+)", text, re.IGNORECASE)
    if match:
        try:
            test_exit = int(match.group(1))
        except ValueError:
            test_exit = None
    lines = text.splitlines()
    edit_markers = {
        "str_replace NO-OP", "edit NO-OP", "edit REJECTED",
        "str_replace REJECTED", "file_create REJECTED",
        "[kernel] pre-existing test files are READ-ONLY",
    }
    read_markers = {"(no grep matches)", "(no matches)", "IsADirectoryError"}
    negative: list[str] = [
        mk for mk in _NEGATIVE_MARKERS
        if _line_starts(lines, mk)
        and not mk.startswith("[kernel]")
        and (action_kind is None
             or (action_kind == "edit" and mk in edit_markers)
             or (action_kind == "read" and mk in read_markers))
    ]
    if test_exit in (1, 4, 5):
        _append_unique(negative, f"pytest exit {test_exit}")
    low = text.lower()
    if action_kind == "test" and test_exit is None and (
            "traceback" in low or "error" in low):
        _append_unique(negative, "HARNESS_ERROR")
    if action_kind == "finish" and any(
            marker in low for marker in (
                "you have not edited anything", "you claim the bug is fixed",
                "`done` would submit", "done refused")):
        _append_unique(negative, "DONE_REFUSED")
    if action_kind == "no_action":
        _append_unique(negative, "NO_ACTION_TURN")
    if action_kind == "other":
        _append_unique(negative, "UNKNOWN_ACTION")
    if any(_line_starts(lines, marker) for marker in (
            "[kernel] (repeat", "[kernel] you have repeated",
            "[kernel] you have been cycling")):
        _append_unique(negative, "[kernel] REPEAT")
    kernel_lines = [ln.strip()[:160] for ln in lines
                    if ln.strip().startswith("[kernel]")][:4]
    # Exact success markers emitted by react_repair.py (verified against source:
    # "str_replace applied." L408, "edit applied." L469, "created <path>" L954).
    edit_applied = any(
        line.strip() in ("str_replace applied.", "edit applied.")
        or line.strip().startswith("created ") for line in lines)
    result = {"test_exit": test_exit,
              "edit_applied": edit_applied,
              "negative": negative, "kernel_lines": kernel_lines,
              "obs_digest": _sha256_bytes(text.encode())[:23],
              "observed": bool(observations),
              "content": text or None,
              "content_status": "observed" if text else "observed_but_omitted"}
    if action_kind == "read":
        result["source_snapshot"] = {
            "status": "observed" if text else "unavailable",
            "content": _captured_text(text or None, "read_receipt_has_no_content"),
            "source_revision": _unavailable(_REVISION_UNAVAILABLE_REASON),
        }
    return result


def _tri_logic(test_exit: int | None) -> str:
    """Only the narrow atom pytest_run_passed. Never infer target_fixed."""
    if test_exit == 0:
        return "TRUE"
    if test_exit == 1:
        return "FALSE"
    return "UNKNOWN"


def _verdict_status(exit_code: object) -> str:
    if type(exit_code) is not int:
        return "UNKNOWN"
    if exit_code == 0:
        return "PASS"
    if exit_code == 1:
        return "FAIL"
    if exit_code in (4, 5):
        return "NO_ORACLE"
    return "UNKNOWN"


def _latest_verdict(facts: dict) -> dict | None:
    verdicts = facts.get("verdicts")
    if not isinstance(verdicts, list) or not verdicts:
        return None
    latest = verdicts[-1]
    return latest if isinstance(latest, dict) else None


def _known_latest_verdict(facts: dict) -> dict | None:
    latest = _latest_verdict(facts)
    return latest if latest and _verdict_status(latest.get("exit")) != "UNKNOWN" else None


def _is_test_path(path: object) -> bool:
    normalized = str(path or "").replace("\\", "/").strip().lower().lstrip("./")
    base = normalized.rsplit("/", 1)[-1]
    return (normalized.startswith("tests/") or "/tests/" in normalized
            or base.startswith("test_") or base.endswith("_test.py")
            or base == "conftest.py")


def _negative_kind(marker: str) -> str:
    if marker in _NEGATIVE_KIND_BY_MARKER:
        return _NEGATIVE_KIND_BY_MARKER[marker]
    low = marker.lower()
    if low.startswith("pytest exit "):
        try:
            return "TEST_FAIL" if int(low.rsplit(" ", 1)[1]) == 1 else "TEST_NO_ORACLE"
        except ValueError:
            return "UNKNOWN_ACTION"
    if "no-op" in low:
        return "EDIT_NOOP"
    if "rejected" in low or "read-only" in low:
        return "EDIT_REJECTED"
    if "repeat" in low or "cycling" in low:
        return "REPEAT_BLOCKED"
    if "no grep matches" in low or "no matches" in low or "isadirectory" in low:
        return "DEAD_QUERY"
    return "UNKNOWN_ACTION"


def _append_negative(
    frame: dict,
    turn_index: int,
    marker: str,
    event_id: str | None = None,
) -> None:
    """Append to the carried ledger; never rebuild it from the current turn."""
    negative_work = frame["negative_work"]
    entry = {"turn": turn_index, "step": turn_index + 1,
             "kind": _negative_kind(marker), "marker": marker}
    if event_id is not None:
        entry["event_id"] = event_id
    negative_work.append(entry)


def _receipt_id(turn_event_id: str, kind: str, position: int,
                payload: object) -> str:
    return _event_id(f"receipt:{kind}", {
        "turn_event_id": turn_event_id,
        "payload": payload,
    }, position)


def _append_event_receipt(frame: dict, receipt: dict) -> None:
    receipts = frame["facts"].setdefault("event_receipts", [])
    event_id = receipt.get("event_id")
    if event_id and not any(
            isinstance(existing, dict) and existing.get("event_id") == event_id
            for existing in receipts):
        receipts.append(receipt)


def _hypothesis_state(hypotheses: list[dict]) -> dict:
    if not hypotheses:
        return {
            "status": "none_recorded",
            "recorded_events": 0,
            "step_attribution": "complete",
            "lifecycle": "not_observed",
        }
    steps = [item.get("step") for item in hypotheses]
    all_eliminated = all(item.get("lifecycle") == "eliminated"
                         for item in hypotheses)
    return {
        "status": "all_eliminated" if all_eliminated else "recorded",
        "recorded_events": len(hypotheses),
        "step_attribution": (
            "complete" if all(isinstance(step, int) for step in steps)
            else "partial" if any(isinstance(step, int) for step in steps)
            else "unavailable"),
        "lifecycle": "observed" if any(
            item.get("lifecycle") for item in hypotheses) else "unavailable",
    }


def _append_hypothesis(frame: dict, event: dict) -> None:
    name = str(event.get("event") or "")
    if name not in {"hypothesis", "hypothesis_opened", "hypothesis_eliminated"}:
        return
    event_id = str(event.get("event_id") or "")
    if not event_id:
        event_id = _event_id("kernel", event, len(frame.get("hypotheses", [])))
    hypotheses = frame.setdefault("hypotheses", [])
    if any(isinstance(item, dict) and item.get("event_id") == event_id
           for item in hypotheses):
        return
    lifecycle = "eliminated" if name == "hypothesis_eliminated" else "unverified"
    item = {
        "event_id": event_id,
        "kind": name,
        "claim": str(event.get("detail") or "")[:400],
        "status": "ELIMINATED" if lifecycle == "eliminated" else "UNVERIFIED",
        "lifecycle": lifecycle,
        "step": event.get("step"),
        # The fold checkpoint is known even when the run-final trace omitted the
        # event's causal step; keep both coordinates explicit.
        "frame_step": (frame.get("cursor") or {}).get("current_step"),
        "source_refs": [{"kind": "kernel_trace", "event_id": event_id}],
    }
    step_status = event.get("step_status")
    if step_status in {"unavailable", "attributed"}:
        item["step_status"] = step_status
    else:
        item["step_status"] = (
            "attributed" if isinstance(item["step"], int) else "unavailable")
    hypotheses.append(item)
    frame["hypothesis_state"] = _hypothesis_state(hypotheses)


def _certification_detail(facts: dict, world: dict) -> dict:
    """Evaluate certification only when the required binding is explicit."""
    verdicts = facts.get("verdicts")
    if not isinstance(verdicts, list) or not verdicts:
        if isinstance(facts.get("tests_run"), int) and facts["tests_run"] > 0:
            return {"status": "observed_but_omitted",
                    "reason": "test_action_recorded_without_exit_result"}
        return {"status": "not_observed", "reason": "no_test_verdict_recorded"}
    latest = _latest_verdict(facts)
    if latest is None:
        return {"status": "observed_but_omitted",
                "reason": "test_verdict_recorded_in_unreadable_shape"}
    if facts.get("tests_run", 0) > len(verdicts):
        return {"status": "observed_but_omitted",
                "reason": "test_action_recorded_without_exit_result"}
    latest_status = _verdict_status(latest.get("exit"))
    if latest_status in ("UNKNOWN", "NO_ORACLE"):
        return {"status": "unavailable", "reason": "test_result_unavailable",
                "result": latest_status}
    if latest_status != "PASS":
        return {"status": "observed_but_omitted",
                "reason": "required_check_did_not_pass", "result": latest_status}

    latest_turn = latest.get("turn")
    if type(latest_turn) is not int or latest_turn < 0:
        return {"status": "unavailable", "reason": "test_turn_unavailable"}
    edits = facts.get("edits_applied", [])
    if not isinstance(edits, list) or any(
            not isinstance(edit, dict) or type(edit.get("turn")) is not int
            or edit["turn"] < 0 for edit in edits):
        return {"status": "unavailable", "reason": "edit_order_unavailable"}
    later_edits = [edit for edit in facts.get("edits_applied", [])
                   if isinstance(edit, dict)
                   and isinstance(edit.get("turn"), int)
                   and edit["turn"] > latest_turn]
    if any(edit.get("role", "product") == "product" for edit in later_edits):
        return {"status": "stale", "reason": "product_changed_after_passing_check"}
    if any(edit.get("role", "test") == "test" for edit in later_edits):
        return {"status": "stale", "reason": "test_harness_changed_after_passing_check"}

    receipt_id = latest.get("receipt_id")
    observations = [entry for entry in facts.get("test_observations", [])
                    if isinstance(entry, dict) and receipt_id
                    and entry.get("receipt_id") == receipt_id]
    if len(observations) != 1:
        return {"status": "observed_but_omitted", "binding_status": "unbound",
                "reason": "passing_receipt_observation_missing_or_ambiguous"}
    observation = observations[0]
    if (type(observation.get("turn")) is not int
            or observation["turn"] != latest_turn
            or observation.get("selector") != latest.get("selector")
            or _verdict_status(observation.get("exit")) != "PASS"):
        return {"status": "unavailable", "reason": "receipt_verdict_conflict"}
    binding = observation.get("binding")
    required = ("product_revision", "test_harness_revision",
                "environment_revision")
    missing = [name for name in required
               if not isinstance(binding, dict)
               or not _revision_is_bound(binding.get(name))
               or not _revision_is_bound(world.get(name))]
    if missing:
        return {"status": "observed_but_omitted", "binding_status": "unbound",
                "reason": "required_revision_binding_missing", "missing": missing}
    changed = [name for name in required
               if binding[name]["value"] != world[name]["value"]]
    if changed:
        return {"status": "stale", "reason": "revision_binding_changed",
                "changed": changed, "receipt_id": receipt_id}
    selectors = world.get("required_selectors")
    # A generic green is a bound check, not product success. Require explicit
    # task-owned selectors, each with a passing receipt at the current binding.
    if not isinstance(selectors, list) or not selectors or not all(
            isinstance(selector, str) and selector.strip() for selector in selectors):
        return {"status": "observed_but_omitted", "reason": "required_coverage_unavailable",
                "bound_check_passed": True, "receipt_id": receipt_id}
    covered = set()
    latest_by_selector = {}
    for candidate in verdicts:
        if isinstance(candidate, dict):
            latest_by_selector[candidate.get("selector")] = candidate
    for verdict in latest_by_selector.values():
        if not isinstance(verdict, dict) or _verdict_status(verdict.get("exit")) != "PASS":
            continue
        linked = [entry for entry in facts.get("test_observations", [])
                  if isinstance(entry, dict) and verdict.get("receipt_id")
                  and entry.get("receipt_id") == verdict["receipt_id"]]
        if len(linked) != 1:
            continue
        entry = linked[0]
        turn = verdict.get("turn")
        if (type(turn) is int and 0 <= turn <= latest_turn
                and entry.get("turn") == turn
                and entry.get("selector") == verdict.get("selector")
                and _verdict_status(entry.get("exit")) == "PASS"
                and entry.get("binding") == {name: world[name] for name in required}
                and not any(edit["turn"] >= turn for edit in edits)):
            covered.add(verdict.get("selector"))
    missing_selectors = sorted(set(selectors) - covered)
    if missing_selectors:
        return {"status": "observed_but_omitted", "reason": "required_coverage_missing",
                "missing_selectors": missing_selectors, "bound_check_passed": True}
    return {"status": "certified", "reason": "bound_required_checks_passed",
            "test_turn": latest_turn, "receipt_id": receipt_id,
            "covered_selectors": sorted(covered)}


def _public_certification_value(certification: dict) -> bool:
    """The public boolean is strictly True only when certified.

    R3 contract: this value must NEVER assert True on an unbound/unknown
    revision, and must never leak an uppercased status string in its place.
    The detailed, distinct state (not_observed / observed_but_omitted /
    stale / unavailable / certified) lives in ``certification["status"]``
    (world["certification"]) for callers that need it — this boolean is
    the yes/no verdict only.
    """
    return certification.get("status") == "certified"


def _latest_read(facts: dict, target: str | None = None) -> dict | None:
    reads = facts.get("read_observations", [])
    if not isinstance(reads, list):
        return None
    candidates = [entry for entry in reads if isinstance(entry, dict)
                  and (target is None or entry.get("target") == target)]
    return candidates[-1] if candidates else None


def _build_action_prerequisites(frame: dict) -> dict:
    facts = frame["facts"]
    objective = frame.get("objective") or {}
    target_obj = objective.get("target") if isinstance(objective, dict) else {}
    target = target_obj.get("path") if isinstance(target_obj, dict) else None
    read = _latest_read(facts, target)
    edit_missing: list[str] = []
    if not read or read.get("content_status") != "observed":
        edit_missing.append("source_content")
    # The transcript has old/new edit arguments but no resolved source span or
    # authoritative revision. Do not promote either into a fabricated binding.
    edit_missing.extend(("precise_source_span", "source_revision"))
    edit = {
        "status": "available" if not edit_missing else "unavailable",
        "target": target,
        "observed_source": bool(read and read.get("content_status") == "observed"),
        "missing": edit_missing,
    }

    latest = _latest_verdict(facts)
    verify_missing = [name for name in (
        "product_revision", "test_harness_revision", "environment_revision")
        if not _revision_is_bound((frame.get("world") or {}).get(name))]
    verify = {
        "status": "available" if latest and not verify_missing else "unavailable",
        "selector": latest.get("selector") if latest else None,
        "result": _verdict_status(latest.get("exit")) if latest else None,
        "missing": verify_missing if verify_missing else [],
    }

    action = frame.get("action") or {}
    if not action.get("tool"):
        pending = {"status": "not_observed"}
        outcome_unknown = {"status": "not_observed"}
    elif not action.get("observation"):
        pending = {"status": "pending", "event_id": action.get("event_id")}
        outcome_unknown = {"status": "unknown", "reason": "receipt_not_observed"}
    elif action.get("kind") == "edit" and action.get("observation", {}).get(
            "edit_applied"):
        pending = {"status": "none"}
        outcome_unknown = {"status": "unknown",
                           "reason": "edit_requires_subsequent_verification"}
    else:
        pending = {"status": "none"}
        outcome_unknown = {"status": "not_applicable"}
    return {"edit": edit, "verify": verify, "resume": {
        "pending": pending,
        "outcome_unknown": outcome_unknown,
    }}


def _verified_fact_ledger(facts: dict) -> list[dict]:
    records: list[dict] = []
    for verdict in facts.get("verdicts", []):
        if not isinstance(verdict, dict):
            continue
        exit_code = verdict.get("exit")
        status = _verdict_status(exit_code)
        if status == "UNKNOWN":
            # Unknown exits remain in the raw verdict ledger, but are not
            # promoted into verified facts.
            continue
        records.append({
            "kind": "TEST",
            "step": verdict.get("turn", 0) + 1
            if isinstance(verdict.get("turn"), int) else None,
            "target": verdict.get("selector", ""),
            "status": status,
            "exit_code": exit_code,
        })
    for edit in facts.get("edits_applied", []):
        if not isinstance(edit, dict):
            continue
        turn = edit.get("turn")
        records.append({
            "kind": "EDIT_APPLIED",
            "step": turn + 1 if isinstance(turn, int) else edit.get("step"),
            "target": edit.get("target", ""),
            "status": "APPLIED",
            "role": edit.get("role", "test" if _is_test_path(edit.get("target"))
                             else "product"),
        })
    for receipt in facts.get("edit_receipts", []):
        if not isinstance(receipt, dict):
            continue
        records.append({
            # The rubric's union view recognizes receipt classes by kind and
            # translates them to its internal EDIT_RECEIPT atom.
            "kind": receipt.get("status", "UNKNOWN"),
            "step": receipt.get("step"),
            "target": receipt.get("target", ""),
            "status": receipt.get("status", "UNKNOWN"),
        })
    return sorted(records, key=lambda record: (
        record.get("step") if isinstance(record.get("step"), int) else -1,
        0 if record.get("kind") == "TEST" else 1))


def _policy_next_action(world: dict, facts: dict, latest: dict | None) -> str:
    if latest and _verdict_status(latest.get("exit")) == "UNKNOWN":
        latest = None
    dirty = world.get("test_world") == "DIRTY"
    last_status = _verdict_status(latest.get("exit")) if latest else None
    last_test_step = (latest.get("turn") + 1 if latest and
                      isinstance(latest.get("turn"), int) else None)
    product_steps = [
        edit.get("turn", -1) + 1 for edit in facts.get("edits_applied", [])
        if isinstance(edit, dict)
        and edit.get("role", "product") == "product"
        and isinstance(edit.get("turn"), int)
    ]
    last_product_edit_step = max(product_steps) if product_steps else None
    if dirty:
        return "inspect_product_test_boundary"
    if last_status in ("FAIL", "NO_ORACLE"):
        return "inspect_or_edit_product"
    if last_product_edit_step is not None and (
            last_test_step is None or last_product_edit_step > last_test_step):
        return "run_acceptance_test"
    if facts.get("repeat_blocks", 0) >= 2:
        return "change_diagnostic"
    if facts.get("done_refused") and not facts.get("edits_applied"):
        return "continue_or_record_no_change"
    if latest is None:
        return "run_acceptance_test"
    return "final_verification"


def _predicate(condition: bool | None) -> str:
    if condition is None:
        return "UNKNOWN"
    return "TRUE" if condition else "FALSE"


def _build_resume(frame: dict) -> dict:
    facts = frame["facts"]
    objective = frame["objective"]
    target = objective.get("target") if isinstance(objective.get("target"), dict) else {}
    cursor = frame["cursor"]
    world = frame["world"]
    latest = _latest_verdict(facts)
    action = frame.get("action") or {}
    negative_work = frame.get("negative_work") or []
    known_latest = _known_latest_verdict(facts)

    open_work: list[str] = []
    if world.get("test_world") == "DIRTY":
        open_work.append("test_world_dirty")
    if known_latest and _verdict_status(known_latest.get("exit")) in ("FAIL", "NO_ORACLE"):
        open_work.append("last_test_not_passing")
    if latest is None or known_latest is None:
        open_work.append("acceptance_verdict_pending")
    if latest is not None and (world.get("certification") or {}).get("status") != "certified":
        open_work.append("verification_not_certified")
    latest_step = (known_latest.get("turn") + 1 if known_latest and
                   isinstance(known_latest.get("turn"), int) else None)
    for edit in facts.get("edits_applied", []):
        if (isinstance(edit, dict) and edit.get("role", "product") == "product"
                and isinstance(edit.get("turn"), int)
                and (latest_step is None or edit["turn"] + 1 > latest_step)):
            if "product_edit_unverified" not in open_work:
                open_work.append("product_edit_unverified")

    unknowns: list[str] = []
    if not target.get("path"):
        unknowns.append("objective.target.path")
    if not target.get("symbol"):
        unknowns.append("objective.target.symbol")
    if not objective.get("acceptance_tests"):
        unknowns.append("objective.acceptance_tests")
    if not _revision_is_bound(world.get("product_revision")):
        unknowns.append("world.product_revision")
    if not _revision_is_bound(world.get("test_harness_revision")):
        unknowns.append("world.test_harness_revision")
    if not _revision_is_bound(world.get("environment_revision")):
        unknowns.append("world.environment_revision")
    if action.get("tool") and not action.get("target"):
        unknowns.append("action.target")
    if action.get("tool") and not world.get("last_observation"):
        unknowns.append("world.last_observation")
    if action.get("kind") == "test" and (latest is None or known_latest is None):
        unknowns.append("facts.verdicts.latest")

    predicates = {
        "R1": _predicate(
            None if not (target.get("path") and target.get("symbol")
                         and objective.get("acceptance_tests")) else True),
        "R2": _predicate(
            None if any(key not in cursor for key in
                        ("current_step", "next_step", "terminal"))
            or cursor.get("terminal") == "UNKNOWN"
            else bool(cursor.get("current_step") is not None
                      and cursor.get("next_step") is not None)),
        "R3": _predicate(
            None if (not _revision_is_bound(world.get("product_revision"))
                     or not _revision_is_bound(world.get("test_harness_revision"))
                     or not _revision_is_bound(world.get("environment_revision")))
            else (world.get("certification", {}).get("status") == "certified")),
        "R4": _predicate(
            None if latest is None or known_latest is None
            else bool(action.get("tool") and action.get("target")
                      and world.get("last_observation") and known_latest)),
        "R5": _predicate(isinstance(frame.get("verified_facts"), list)),
        "R6": _predicate(bool(isinstance(open_work, list)
                               and frame.get("obligations", {}).get("status")
                               == "projected")),
        "R7": _predicate(bool(isinstance(negative_work, list)
                               and (negative_work or not negative_work))),
        "R8": _predicate(bool(isinstance(unknowns, list)
                               and _policy_next_action(world, facts, known_latest))),
    }
    guardrails = [
        {"kind": entry.get("kind", "UNKNOWN_ACTION"),
         "step": entry.get("step", entry.get("turn", 0) + 1)
         if isinstance(entry.get("turn", 0), int) else None}
        for entry in negative_work if isinstance(entry, dict)
    ]
    resume = {
        "predicates": predicates,
        "open_work": open_work,
        "guardrails": guardrails,
        "unknowns": unknowns,
        "recommended_next_action": _policy_next_action(world, facts, known_latest),
    }
    # Keep the named rubric predicates directly addressable as well as grouped.
    resume.update(predicates)
    return resume


def _refresh_derived(frame: dict) -> None:
    facts = frame["facts"]
    latest = _latest_verdict(facts)
    last_exit = latest.get("exit") if latest else None
    facts["last_test_exit"] = last_exit
    frame["tri_logic"]["pytest_run_passed"] = _tri_logic(last_exit)
    frame["verified_facts"] = _verified_fact_ledger(facts)

    prior_world = frame.get("world") if isinstance(frame.get("world"), dict) else {}
    action = frame.get("action") if isinstance(frame.get("action"), dict) else {}
    observation = action.get("observation")
    observation = observation if isinstance(observation, dict) else {}
    evidence = action.get("revision_evidence")
    evidence = evidence if isinstance(evidence, dict) else {}

    revisions: dict[str, dict] = {}
    for field_name in ("product_revision", "test_harness_revision",
                       "environment_revision"):
        candidate = evidence.get(field_name)
        if candidate is not None:
            revisions[field_name] = _revision_state(
                candidate, _REVISION_UNAVAILABLE_REASON)
            continue
        edit_applied = bool(observation.get("edit_applied"))
        target = action.get("target", "")
        invalidated_by_edit = (
            action.get("kind") == "edit"
            and edit_applied
            and ((field_name == "test_harness_revision" and _is_test_path(target))
                 or (field_name == "product_revision" and not _is_test_path(target))))
        revisions[field_name] = (
            _unavailable("edit_revision_not_recorded") if invalidated_by_edit
            else _revision_state(prior_world.get(field_name),
                                 _REVISION_UNAVAILABLE_REASON))

    edits = [edit for edit in facts.get("edits_applied", [])
             if isinstance(edit, dict)]
    product_edits = [edit for edit in edits
                     if edit.get("role", "product") == "product"]
    test_edits = [edit for edit in edits
                  if edit.get("role", "test") == "test"]
    test_world = "DIRTY" if test_edits else "PRISTINE"
    world = {
        "last_action": dict(action),
        "last_observation": dict(observation),
        "product_revisions": product_edits,
        "test_harness_revisions": test_edits,
        "required_selectors": list(prior_world.get("required_selectors") or []),
        **revisions,
        "revision_binding_state": {
            "product": revisions["product_revision"],
            "test_harness": revisions["test_harness_revision"],
            "environment": revisions["environment_revision"],
        },
        "historical_resolver": prior_world.get(
            "historical_resolver",
            _unavailable(_HISTORICAL_RESOLVER_UNAVAILABLE_REASON),
        ),
        "source_snapshots": list(prior_world.get("source_snapshots") or []),
        "test_world": test_world,
        "last_test": dict(latest) if latest else None,
    }
    frame["world"] = world
    certification = _certification_detail(facts, world)
    world["certification"] = certification
    # Compatibility atom: only the fully bound predicate can be True. Every
    # other certification status is explicitly carried in ``certification``.
    world["valid_product_verification"] = _public_certification_value(certification)

    frame["action_prerequisites"] = _build_action_prerequisites(frame)
    frame["world"] = {
        **world,
    }
    frame["obligations"] = {"status": "projected",
                             "open_work": _build_resume(frame)["open_work"]}
    resume = _build_resume(frame)
    frame["resume"] = resume
    pending = frame["action_prerequisites"]["resume"]
    frame["pending"] = pending["pending"]
    frame["outcome_unknown"] = pending["outcome_unknown"]

    hypotheses = frame.get("hypotheses")
    if not isinstance(hypotheses, list):
        hypotheses = []
        frame["hypotheses"] = hypotheses
    frame["hypothesis_state"] = _hypothesis_state(hypotheses)
    frame["hypotheses_meta"] = {
        "total_recorded": len(hypotheses),
        "attributed": sum(1 for item in hypotheses
                           if isinstance(item, dict) and isinstance(item.get("step"), int)),
        "classification": frame["hypothesis_state"]["status"],
    }


def _kernel_hypothesis_events(source: dict) -> list[dict]:
    """Normalize trace-only hypothesis receipts for compatibility callers.

    The run-final trace is not itself a causal step stream. Explicit coordinates
    are retained; missing coordinates are marked unavailable. The projector's
    normal path uses :func:`attach_kernel_events`, which can additionally use an
    evidence-preserving assistant-content prefix match before folding.
    """
    kernel = source.get("kernel")
    if not isinstance(kernel, dict):
        return []
    trace = kernel.get("trace")
    if not isinstance(trace, list):
        return []
    events: list[dict] = []
    for position, raw_event in enumerate(trace):
        if not isinstance(raw_event, dict):
            continue
        if raw_event.get("event") not in {
                "hypothesis", "hypothesis_opened", "hypothesis_eliminated"}:
            continue
        event = dict(raw_event)
        event["event_id"] = _event_id("kernel", event, position)
        if _kernel_event_turn_index(event) is None:
            event["step_status"] = "unavailable"
        events.append(event)
    return events


def initial_frame(source: dict) -> dict:
    # State identity is causal: final-only kernel receipts appended later to the
    # recording must not change an earlier frame's embedded source identity.
    source_view: dict = {"fn": source.get("fn"), "sha256": _source_identity(source),
                         "unattributed_kernel_events": _unavailable(
                             "run_final_kernel_trace_has_no_prefix_attribution")}
    rel_path = _source_rel_path(source)
    if rel_path is not None:
        source_view["rel_path"] = rel_path
    # R2 repair: the previous `unattributed_kernel_events` count was derived once
    # from the FULL (run-final-only) kernel trace and then copied unchanged into
    # every per-turn frame. That made a later, unattributed kernel event change
    # what an EARLIER frame reported, which is exactly the causal-cutoff leak A4
    # found. Whole-run-scoped data does not belong in per-turn `source` state at
    # all; it is a constant, typed statement of unavailability instead so that
    # every frame of a given source is byte-identical regardless of position.
    source_view["kernel_trace_attribution"] = _unavailable(
        "run_final_kernel_trace_has_no_prefix_attribution")
    frame = {
        "source": source_view,
        "objective": _objective_from_source(source),
        "phase": "orienting",
        "action": {},
        "facts": {"edits_applied": [], "edit_receipts": [], "event_receipts": [],
                  "verdicts": [], "test_observations": [], "read_observations": [],
                  "last_test_exit": None, "tests_run": 0, "repeat_blocks": 0,
                  "done_refused": False, "files_inspected": []},
        "cursor": {"current_step": None, "next_step": None,
                   "total_steps_observed": 0, "terminal": "UNKNOWN",
                   "prefix_digest": None},
        "world": {"last_action": {}, "last_observation": {},
                  "required_selectors": list(source.get("required_selectors") or []),
                  "product_revisions": [], "test_harness_revisions": [],
                  "product_revision": _source_revision(
                      source, ("product_revision", "content_revision")),
                  "test_harness_revision": _source_revision(
                      source, ("test_harness_revision", "harness_revision")),
                  "environment_revision": _source_revision(
                      source, ("environment_revision", "environment_id")),
                  "revision_binding_state": {
                      "product": _source_revision(
                          source, ("product_revision", "content_revision")),
                      "test_harness": _source_revision(
                          source, ("test_harness_revision", "harness_revision")),
                      "environment": _source_revision(
                          source, ("environment_revision", "environment_id")),
                  },
                  "historical_resolver": _unavailable(
                      _HISTORICAL_RESOLVER_UNAVAILABLE_REASON),
                  "source_snapshots": [], "test_world": "PRISTINE",
                  "last_test": None, "valid_product_verification": False},
        "verified_facts": [],
        "hypotheses": [],
        "hypothesis_state": _hypothesis_state([]),
        "hypotheses_meta": {"total_recorded": 0, "attributed": 0,
                            "classification": "none_recorded"},
        "negative_work": [],
        "pending": {"status": "not_observed"},
        "outcome_unknown": {"status": "not_observed"},
        "obligations": {"status": "not_projected"},
        "tri_logic": {"pytest_run_passed": "UNKNOWN"},
        "kernel": {},
        # R1: content+position identity ledger for idempotent folding. Internal
        # (leading underscore) — not part of the published fact surface.
        "_event_ledger": [],
        # Retained as a stable internal compatibility slot; causal event
        # delivery is carried on Turn.kernel_events, never by whole-run data.
        "_kernel_hypothesis_events": [],
    }
    frame["resume"] = _build_resume(frame)
    return frame


def _event_identity(turn: Turn) -> str:
    """R1: stable identity for a folded event, derived from its content and its
    position WITHIN the turn's own observation set — deliberately NOT the turn's
    step/index number, since a re-delivery of the same underlying exchange can
    land at a different step position while still being the same event, and one
    step can itself emit several distinct receipts. Two turns with byte-identical
    assistant output and observations are the same event for folding purposes.
    """
    return turn.event_id


def fold_turn(frame: dict, turn: Turn) -> dict:
    """Deterministic fold: only code-observable facts move state. Assistant prose
    never becomes a verified fact.

    R1: idempotent per event identity — re-delivering an already-folded event
    (byte-identical assistant output + observations) must not create a second
    fact. `action`/`cursor` still advance to reflect stream position, but the
    fact-mutating body below is skipped entirely for a repeat delivery.
    """
    f = json.loads(_canon(frame))  # cheap deep copy, canonical types
    facts = f.setdefault("facts", {})
    facts.setdefault("edits_applied", [])
    facts.setdefault("edit_receipts", [])
    facts.setdefault("event_receipts", [])
    facts.setdefault("verdicts", [])
    facts.setdefault("test_observations", [])
    facts.setdefault("read_observations", [])
    facts.setdefault("last_test_exit", None)
    facts.setdefault("tests_run", 0)
    facts.setdefault("repeat_blocks", 0)
    facts.setdefault("done_refused", False)
    facts.setdefault("files_inspected", [])
    if not isinstance(f.get("negative_work"), list):
        raise ValueError("negative_work must be a list")

    ledger = f.setdefault("_event_ledger", [])
    if not isinstance(ledger, list):
        ledger = []
    event_id = _event_identity(turn)
    duplicate = event_id in ledger
    if not duplicate:
        ledger.append(event_id)
    f["_event_ledger"] = ledger

    receipt_position = 0

    def emit_receipt(kind: str, payload: object,
                     status: str = "observed") -> str:
        nonlocal receipt_position
        receipt_id = _receipt_id(event_id, kind, receipt_position, payload)
        receipt_position += 1
        _append_event_receipt(f, {
            "event_id": receipt_id,
            "turn": turn.index,
            "step": turn.index + 1,
            "kind": kind,
            "status": status,
        })
        return receipt_id

    action = normalize_action(turn.assistant) or {
        "tool": None, "kind": "no_action", "target": ""}
    obs = normalize_observation(turn.observations, action["kind"])
    action_record = {**action, "turn": turn.index, "event_id": event_id}
    if obs["observed"]:
        action_record["observation"] = obs
    f["action"] = action_record
    f["cursor"] = {
        **(f.get("cursor") or {}),
        "current_step": turn.index + 1,
        "next_step": turn.index + 2,
        "total_steps_observed": turn.index + 1,
        "terminal": "UNKNOWN",
    }

    if not duplicate:
        emit_receipt("turn", {"kind": action["kind"],
                               "target": action.get("target", "")})
        if action["kind"] == "read" and action.get("target"):
            seen = facts["files_inspected"]
            if action["target"] not in seen and len(seen) < 200:
                seen.append(action["target"])
            source_snapshot = obs.get("source_snapshot")
            if isinstance(source_snapshot, dict):
                source_snapshot = {
                    **source_snapshot, "target": action.get("target", ""),
                }
                read_receipt_id = emit_receipt(
                    "read", {"target": action.get("target", ""),
                             "digest": obs.get("obs_digest")})
                facts["read_observations"].append({
                    "turn": turn.index,
                    "target": action.get("target", ""),
                    "content_status": source_snapshot.get("status"),
                    "snapshot": source_snapshot,
                    "receipt_id": read_receipt_id,
                })
                snapshots = f.setdefault("world", {}).setdefault(
                    "source_snapshots", [])
                snapshots[:] = [entry for entry in snapshots
                                if not (isinstance(entry, dict)
                                        and entry.get("target") == action.get("target", ""))]
                snapshots.append(source_snapshot)
        if action["kind"] == "edit":
            if obs["edit_applied"] and not obs["negative"]:
                target = action.get("target", "")
                role = "test" if _is_test_path(target) else "product"
                edit_receipt_id = emit_receipt(
                    "edit", {"target": target, "role": role})
                facts["edits_applied"].append(
                    {"turn": turn.index, "step": turn.index + 1,
                     "target": target, "role": role,
                     "event_id": event_id,
                     "receipt_id": edit_receipt_id,
                     "edit_request": action.get("edit_request", {})})
                if role == "test" and any(
                        verdict.get("exit") == 1 for verdict in facts["verdicts"]
                        if isinstance(verdict, dict)):
                    _append_negative(
                        f, turn.index, "TEST_EDIT_AFTER_RED",
                        emit_receipt("negative", {"marker": "TEST_EDIT_AFTER_RED"}))
            for marker in obs["negative"]:
                kind = _negative_kind(marker)
                if kind in {"EDIT_REJECTED", "EDIT_NOOP", "FILE_CREATE_REJECTED"}:
                    edit_receipt_id = emit_receipt(
                        "edit_receipt", {"target": action.get("target", ""),
                                          "status": kind})
                    facts["edit_receipts"].append({
                        "turn": turn.index, "step": turn.index + 1,
                        "target": action.get("target", ""), "status": kind,
                        "event_id": edit_receipt_id,
                    })
        if action["kind"] == "test":
            facts["tests_run"] += 1
            evidence = action.get("revision_evidence")
            evidence = evidence if isinstance(evidence, dict) else {}
            binding = {
                name: (evidence.get(name)
                       if evidence.get(name) is not None else
                       (f.get("world") or {}).get(name))
                for name in ("product_revision", "test_harness_revision",
                             "environment_revision")
            }
            test_receipt_id = emit_receipt(
                "test", {"selector": action.get("target", ""),
                          "exit": obs["test_exit"]})
            facts["test_observations"].append({
                "turn": turn.index,
                "selector": action.get("target", ""),
                "exit": obs["test_exit"],
                "binding": json.loads(json.dumps(binding)),
                "receipt_id": test_receipt_id,
            })
            if obs["test_exit"] is not None:
                facts["verdicts"].append({
                    "receipt_id": test_receipt_id,
                    "turn": turn.index,
                    "selector": action.get("target", ""),
                    "exit": obs["test_exit"],
                })
        for mk in obs["negative"]:
            _append_negative(
                f, turn.index, mk,
                emit_receipt("negative", {"marker": mk}))
        facts["repeat_blocks"] += sum(
            1 for mk in obs["negative"] if _negative_kind(mk) == "REPEAT_BLOCKED")
        if any(_negative_kind(mk) == "DONE_REFUSED" for mk in obs["negative"]):
            facts["done_refused"] = True
        if obs["kernel_lines"]:
            f["kernel"] = {"last_lines": obs["kernel_lines"], "turn": turn.index}
        for event in turn.kernel_events:
            if isinstance(event, dict):
                kernel_id = event.get("event_id")
                if kernel_id:
                    _append_event_receipt(f, {
                        "event_id": kernel_id,
                        "turn": turn.index,
                        "step": event.get("step"),
                        "kind": f"kernel:{event.get('event', 'unknown')}",
                        "status": event.get("step_status", "unavailable"),
                    })
                _append_hypothesis(f, event)

    _refresh_derived(f)
    exit_ = f["facts"]["last_test_exit"]
    f["phase"] = ("verifying" if action["kind"] == "test" else
                   "finishing" if action["kind"] == "finish" else
                   "repairing" if facts["edits_applied"] else
                   "diagnosing" if exit_ is not None else "orienting")
    return f


def seal_frame(frame: dict, source_sha: str, turn_index: int) -> dict:
    digest = _sha256_bytes(_canon(frame).encode())
    return {"kind": "snapshot", "key": f"{source_sha}:{turn_index}",
            "turn": turn_index, "state_digest": digest, "state": frame}


def append_idempotent(out_path: pathlib.Path, records: list[dict]) -> int:
    """Append records not already present; identical existing records are skipped,
    conflicting or malformed existing lines FAIL CLOSED."""
    existing: dict[str, str] = {}
    if out_path.exists():
        for i, line in enumerate(out_path.read_text().splitlines()):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError as e:
                raise ValueError(f"malformed existing line {i} in {out_path}") from e
            key = rec.get("key") or f"__{rec.get('kind')}"
            existing[key] = _canon(rec)
    written = 0
    with out_path.open("a") as fh:
        for rec in records:
            key = rec.get("key") or f"__{rec.get('kind')}"
            enc = _canon(rec)
            if key in existing:
                if existing[key] != enc:
                    raise ValueError(f"snapshot conflict for {key} in {out_path}")
                continue
            fh.write(enc + "\n")
            fh.flush()
            import os
            os.fsync(fh.fileno())
            existing[key] = enc
            written += 1
    return written


def project_file(transcript_path: pathlib.Path, out_root: pathlib.Path) -> pathlib.Path:
    src = load_transcript(transcript_path)
    kernel = src.get("kernel") or {}
    trace_sha = _sha256_bytes(_canon(kernel).encode())
    out_dir = out_root / transcript_path.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (transcript_path.stem + ".jsonl")

    header: dict = {
        "kind": "header", "key": f"__header:{src['_source_sha256']}",
        "schema": SCHEMA, "source_path": src["_source_path"],
        "source_sha256": src["_source_sha256"],
        "kernel_trace_sha256": trace_sha,
        "trace_alignment": "run_final_only",
        "kernel_trace": kernel,
    }
    if src.get("_rel_path") is not None:
        header["rel_path"] = src["_rel_path"]
    records: list[dict] = [header]
    frame = initial_frame(src)
    max_step = src.get("steps")
    max_step = max_step if isinstance(max_step, int) and not isinstance(max_step, bool) else None
    turns = list(iter_turns(reconstruct_messages(
        transcript_path, src["messages"], max_step=max_step)))
    trace = kernel.get("trace") if isinstance(kernel, dict) else None
    turns = attach_kernel_events(turns, trace)
    for position, turn in enumerate(turns):
        frame = fold_turn(frame, turn)
        frame["cursor"]["terminal"] = position == len(turns) - 1
        _refresh_derived(frame)
        records.append(seal_frame(frame, src["_source_sha256"], turn.index))
    append_idempotent(out_path, records)
    return out_path


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: stateframe_shadow.py <transcript.json|transcript-dir> [...]")
        return 2
    out_root = pathlib.Path(__file__).resolve().parent.parent / "results" / "stateframes_shadow"
    targets: list[pathlib.Path] = []
    for a in argv[1:]:
        p = pathlib.Path(a)
        targets.extend(sorted(p.glob("*.json")) if p.is_dir() else [p])
    for t in targets:
        try:
            out = project_file(t, out_root)
            print(f"projected {t.name} -> {out}")
        except ValueError as e:
            print(f"FAILED CLOSED {t.name}: {e}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
