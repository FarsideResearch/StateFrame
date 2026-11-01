"""Exclusive, restartable audit writer for the continuation executive.

Explicit construction only: baseline AuditJournal remains unchanged. The hash
chain detects inconsistent records, not malicious rewriting of a whole journal.
A trusted expected_head is needed to detect a consistently truncated suffix.
Unanchored reopening requires an explicit recovery-only opt-in.
"""
from __future__ import annotations

import ast
import contextvars
import copy
import fcntl
import io
import inspect
import json
import os
import re
import stat
import threading
import time
import tokenize
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

from kernel_continuation_menu_authority import EDIT_TOOLS

from kernel_measurement import (
    AuditJournal, HostSpanRecorder, canonical_bytes, digest,
    externalize_journal_fields, profile_host_span, profiled_journal_span,
    resolve_journal_blob_refs, host_span_profiles_enabled,
    _runtime_stateframe_mode_snapshot,
)

FORMAT = "continuation-journal/1"
MAX_RECORD_BYTES = 16 * 1024 * 1024
MEMBOUND_ENV = "SF_LAB_RECORD_CACHE"
RECORD_CACHE_LIMIT = 2048
RECORD_CACHE_BYTE_LIMIT = 512 * 1024
MAX_FRAMEFIT_RECEIPT_RECORDS = 1024
MAX_FRAMEFIT_RECEIPT_FIELD_CHARS = 1024
FRAMEFIT5_JOURNAL_ORIGIN_ENV = "STATEFRAME_FRAMEFIT5_JOURNAL_ORIGIN"
_HOST_EVENT_SPAN_PAIRS = (
    ("runtime_snapshot_to_organ_wiring", "runtime_mode_snapshot",
     "organ_wiring_state", None),
    ("environment_observation_to_tokenlab_ledger", "environment_observation_changed",
     "tokenlab_v22_unified_ledger", "prepare"),
    ("outline_parser_to_census_outcome", "outline_parser_capability",
     "callsite_census_outcome", "callsite_census_prepare_request"),
    ("mechanism_notice_to_source_closure", "v46_mechanism_notice_rendered",
     "request_source_closure", "prepare"),
    ("continuation_projection_to_tool_menu", "continuation_menu_projection",
     "tool_menu_composed", "prepare"),
    ("unified_ledger_to_decision_cache", "tokenlab_v22_unified_ledger",
     "decision_cache_render", "prepare"),
)
_WARM_HOST_EVENT_SPAN_PAIRS = (
    ("frame_series_restrictions_to_tool_menu", "frame_series_restrictions",
     "tool_menu_composed", "tool_menu_composition"),
    ("readiness_menu_to_green_lock_authority", "readiness_menu_projected",
     "green_lock_menu_authority", "prepare"),
    ("vacuity_verdict_to_tokenlab_ledger", "vacuity_verdict",
     "tokenlab_v22_unified_ledger", "rulebook_mechanism_render"),
)


def framefit5_journal_origin_mode(environ: Mapping[str, str] | None = None) -> str:
    """Select receipt identity behavior; malformed or missing config stays legacy-off."""
    source = os.environ if environ is None else environ
    value = (source.get(FRAMEFIT5_JOURNAL_ORIGIN_ENV, "off") or "off").strip().lower()
    return value if value in {"off", "shadow", "on"} else "off"


RESERVED = frozenset({
    "schema", "kind", "run_id", "task_id", "event_id", "sequence",
    "timestamp_ns", "journal_format", "previous_hash", "record_hash",
})

_RECEIPT_KINDS = frozenset({
    "action_receipt", "action_receipt_amended", "action_outcome_closed",
    "completed_action", "observation_committed", "test_subprocess_completed",
    "completion_certification", "host_obligation_check", "host_repair_check",
    "observation_probe_completed",
})
_RECEIPT_REF_KEYS = frozenset({
    "receipt_id", "reference", "event_id", "record_id", "step_record_id",
    "verification_id", "test_receipt_id",
})


@dataclass(frozen=True, slots=True)
class ReceiptCursor:
    """Cursor over the durable journal sequence and derived receipt generation."""

    sequence: int
    generation: int


def _receipt_refs_from_record(record: Mapping[str, object]) -> tuple[str, ...]:
    if record.get("kind") not in _RECEIPT_KINDS:
        return ()
    found: list[str] = []

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in _RECEIPT_REF_KEYS and isinstance(item, str) and item.strip():
                    found.append(item.strip())
                elif key == "receipt_ids" and isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                    for ref in item:
                        if isinstance(ref, str) and ref.strip():
                            found.append(ref.strip())
                elif isinstance(item, (Mapping, list, tuple)):
                    visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(record)
    unique: list[str] = []
    seen: set[str] = set()
    for ref in found:
        if ref not in seen:
            seen.add(ref)
            unique.append(ref)
    return tuple(unique)


def journal_safe_fields(payload: Mapping[str, object]) -> dict[str, object]:
    """Copy of ``payload`` with journal-RESERVED keys renamed to ``receipt_<key>``.

    10-05 L109: receipts carrying ``schema``/``task_id`` were splatted into ``emit`` and
    every live task died before its first request.  Call sites that forward a dict they
    do not own wrap it here (the emit boundary); the source dict is never mutated.
    Guard: the reserved-event-field regression checks.
    """
    return {(f"receipt_{key}" if key in RESERVED else key): value for key, value in dict(payload).items()}


class JournalError(RuntimeError):
    """An unsafe journal must not authorize another action."""


_BLOCKER_ACTION_KINDS = frozenset({
    # ``action_outcome_closed`` is the authoritative per-action observation.
    # ``action_receipt`` and ``native_batch_action_closed`` mirror that outcome
    # and would otherwise double-count one action in a native batch.
    "action_outcome_closed",
})


def _blocker_text(record: Mapping[str, object]) -> str | None:
    """Extract one host-facing refusal from an arbitrary audit event."""
    candidates = []
    nested = record.get("record")
    if isinstance(nested, Mapping):
        candidates.append(nested.get("observation"))
    receipt = record.get("receipt")
    if isinstance(receipt, Mapping):
        candidates.append(receipt.get("detail"))
    candidates.extend((record.get("observation"), record.get("detail")))
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _continuation_state_marker(record: Mapping[str, object]) -> tuple | None:
    """Return a stable replay marker for continuation state, when present.

    Measurement journals contain many presentation events between actions.  The
    invariant is about a scheduler state that did not move, not about those
    renders, so only durable continuation state/transition markers participate.
    """
    kind = record.get("kind")
    if kind == "continuation_transition":
        return ("transition", record.get("after_hash"), record.get("sequence"))
    if kind == "continuation_stranded":
        unresolved_effects = record.get("unresolved_effects")
        if isinstance(unresolved_effects, list):
            unresolved_effects = tuple(unresolved_effects)
        return ("stranded", record.get("continuation_id"),
                record.get("state_revision"), unresolved_effects)
    if kind == "continuation_worker_stopped":
        state = record.get("state")
        return ("stopped", digest(state) if state is not None else None)
    if kind == "continuation_context":
        state = record.get("continuation_state")
        if state is not None:
            return ("context", digest(state))
    return None


def scan_repeated_blockers(path: str | os.PathLike[str], *, minimum: int = 5) -> list[dict]:
    """Replay an events.jsonl and report a stuck identical blocker.

    A finding requires ``minimum`` consecutive action outcomes carrying exactly
    the same blocker text while the last durable continuation-state marker and
    the action bindings remain unchanged.  This is intentionally a pure file
    reader: it does not open a live journal, acquire a writer lease, or mutate
    the evidence it is checking.
    """
    if type(minimum) is not int or minimum < 2:
        raise ValueError("minimum must be an integer >= 2")
    journal_path = Path(path)
    state_marker = None
    previous: tuple[str, tuple, str] | None = None
    streak: list[dict] = []
    findings: list[dict] = []

    def flush() -> None:
        nonlocal streak
        if len(streak) >= minimum:
            findings.append({
                "blocker": streak[0]["blocker"],
                "count": len(streak),
                "first_event_id": streak[0]["event_id"],
                "last_event_id": streak[-1]["event_id"],
                "event_ids": [item["event_id"] for item in streak],
                "state": streak[0]["state"],
            })
        streak = []

    blob_cache: dict[str, object] = {}
    with journal_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except (TypeError, ValueError, UnicodeError) as error:
                raise ValueError(f"invalid JSON at {journal_path}:{line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"journal event must be an object at {journal_path}:{line_number}")
            record = resolve_journal_blob_refs(record, journal_path, blob_cache)
            if not isinstance(record, dict):
                raise ValueError(f"journal event must resolve to an object at {journal_path}:{line_number}")
            marker = _continuation_state_marker(record)
            if marker is not None:
                state_marker = marker
            if record.get("kind") not in _BLOCKER_ACTION_KINDS:
                continue
            blocker = _blocker_text(record)
            if blocker is None:
                flush()
                previous = None
                continue
            nested = record.get("record")
            if isinstance(nested, Mapping):
                before = nested.get("before_binding")
                after = nested.get("after_binding")
                action_state = digest({"before": before, "after": after,
                                       "changes": nested.get("changes")})
            else:
                action_state = digest({"before": record.get("before_binding"),
                                       "after": record.get("after_binding"),
                                       "changes": record.get("changes")})
            state = (state_marker, action_state)
            current = {"blocker": blocker, "event_id": record.get("event_id", f"line:{line_number}"),
                       "state": state}
            key = (blocker, state, str(record.get("kind")))
            if previous is None or previous[:2] != key[:2]:
                flush()
                streak = [current]
            else:
                streak.append(current)
            previous = key
    flush()
    exact_event_ids = {
        event_id for finding in findings for event_id in finding.get("event_ids", ())
    }
    findings.extend(
        finding for finding in _scan_blocker_cycles(journal_path, minimum=minimum)
        if not exact_event_ids.intersection(finding.get("event_ids", ()))
    )
    return findings


def _action_record(record: Mapping[str, object]) -> Mapping[str, object]:
    nested = record.get("record")
    return nested if isinstance(nested, Mapping) else record


def _action_state(record: Mapping[str, object]) -> str:
    nested = _action_record(record)
    return digest({"before": nested.get("before_binding"),
                   "after": nested.get("after_binding"),
                   "changes": nested.get("changes")})


def _load_scanner_records(path: str | os.PathLike[str]) -> list[dict]:
    journal_path = Path(path)
    records: list[dict] = []
    blob_cache: dict[str, object] = {}
    with journal_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except (TypeError, ValueError, UnicodeError) as error:
                raise ValueError(f"invalid JSON at {journal_path}:{line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"journal event must be an object at {journal_path}:{line_number}")
            record = resolve_journal_blob_refs(record, journal_path, blob_cache)
            if not isinstance(record, dict):
                raise ValueError(f"journal event must resolve to an object at {journal_path}:{line_number}")
            records.append(record)
    return records


_CYCLE_RESULTS = frozenset({
    "REFUSED", "REFUSED_PREDISPATCH", "CHALLENGED", "BLOCKED", "STALLED",
    "REUSED_VERIFICATION",
})


def _cycle_token(record: Mapping[str, object]) -> tuple[str, str, str] | None:
    if record.get("kind") not in _BLOCKER_ACTION_KINDS:
        return None
    nested = _action_record(record)
    result = str(nested.get("result", nested.get("status", ""))).upper()
    if result not in _CYCLE_RESULTS:
        return None
    blocker = _blocker_text(record) or result
    return (str(nested.get("tool", "")), result, blocker)


def _cycle_finding(items: list[dict], *, start: int, end: int, period: int,
                   action_state: str) -> dict:
    selected = items[start:end]
    tokens = [item["token"] for item in selected]
    cycle = [tokens[index][1] + ":" + tokens[index][0]
             for index in range(period)]
    return {
        "pattern": "cycle",
        "period": period,
        "cycle": cycle,
        "blockers": [tokens[index][2] for index in range(period)],
        "blocker": tokens[0][2],
        "count": len(selected),
        "first_event_id": selected[0]["event_id"],
        "last_event_id": selected[-1]["event_id"],
        "event_ids": [item["event_id"] for item in selected],
        # Unlike the exact scanner's state tuple, this pass intentionally
        # omits the changing transition marker and retains the stable action
        # binding that proves the cycle did not make product progress.
        "state": (None, action_state),
    }


def _scan_blocker_cycles(path: str | os.PathLike[str], *, minimum: int) -> list[dict]:
    """Find alternating refusal/reuse cycles with periods two through four."""
    records = _load_scanner_records(path)
    findings: list[dict] = []
    segment: list[dict] = []

    def flush() -> None:
        nonlocal segment
        if len(segment) < minimum:
            segment = []
            return
        # A segment can contain a short prefix before the repeating cycle. Try
        # every start and retain the longest candidate, preferring lower period.
        candidates: list[tuple[int, int, int]] = []
        for start in range(len(segment)):
            for period in range(2, 5):
                if start + max(minimum, 2 * period) > len(segment):
                    continue
                end = start + period
                while end < len(segment) and segment[end]["token"] == segment[
                        start + ((end - start) % period)]["token"]:
                    end += 1
                count = end - start
                if count >= minimum and len({item["token"] for item in segment[start:end]}) > 1:
                    candidates.append((count, -period, start))
        if candidates:
            count, negative_period, start = max(candidates)
            period = -negative_period
            findings.append(_cycle_finding(
                segment, start=start, end=start + count, period=period,
                action_state=segment[start]["action_state"]))
        segment = []

    for record in records:
        if record.get("kind") not in _BLOCKER_ACTION_KINDS:
            continue
        token = _cycle_token(record)
        if token is None:
            flush()
            continue
        action_state = _action_state(record)
        if segment and action_state != segment[-1]["action_state"]:
            flush()
        segment.append({
            "token": token,
            "action_state": action_state,
            "event_id": record.get("event_id", f"record:{len(segment)}"),
        })
    flush()
    return findings


_REFUSAL_RESULTS = frozenset({
    "REFUSED", "REFUSED_PREDISPATCH", "CHALLENGED", "BLOCKED", "STALLED",
})
_EDIT_TOOL_NAMES = EDIT_TOOLS | frozenset({"replace_module", "write", "replace"})
_SOURCE_MARKER_RE = re.compile(
    r"\[(?:Working|Current) source:\s*([^\]]+)\]\s*\n?(.*?)(?=\n\[(?:Working|Current|Retained) source:|\n\[StateFrame\]|\Z)",
    re.DOTALL,
)
_TARGET_RE = re.compile(
    r"\b(?:target|file)\s+(?:is\s+)?(?:[A-Za-z_][\w.]*\s+in\s+)?([A-Za-z0-9_.\-/]+\.py)\b",
    re.IGNORECASE,
)


def _message_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        if isinstance(value.get("content"), (str, Mapping, list, tuple)):
            return _message_text(value.get("content"))
        if value.get("text") is not None:
            return str(value.get("text"))
        return " ".join(_message_text(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return "\n".join(_message_text(item) for item in value)
    return "" if value is None else str(value)


def _request_text(request: Mapping[str, object] | None) -> str:
    if not isinstance(request, Mapping):
        return ""
    messages = request.get("messages", request.get("request", ()))
    return _message_text(messages)


def _request_tools(request: Mapping[str, object] | None) -> list[Mapping[str, object]]:
    if not isinstance(request, Mapping):
        return []
    tools = request.get("tools", ())
    return [item for item in tools if isinstance(item, Mapping)] \
        if isinstance(tools, Sequence) and not isinstance(tools, (str, bytes)) else []


def _normal_path(value: object) -> str:
    return str(value).replace("\\", "/").strip().lstrip("./")


def _named_file(refusal: Mapping[str, object], request: Mapping[str, object] | None) -> str | None:
    text = _request_text(request)
    matches = _TARGET_RE.findall(text)
    for match in matches:
        candidate = _normal_path(match)
        if candidate and not candidate.startswith(("test/", "tests/")):
            return candidate
    for match in _SOURCE_MARKER_RE.finditer(text):
        candidate = _normal_path(match.group(1))
        if candidate:
            return candidate
    nested = _action_record(refusal)
    parameters = nested.get("action_parameters", {})
    if isinstance(parameters, Mapping):
        for key in ("path", "file_path", "target_path", "file", "target"):
            candidate = _normal_path(parameters.get(key, ""))
            if candidate and not candidate.startswith(("test/", "tests/")):
                return candidate
    return None


def _has_gap_text(text: str) -> bool:
    import re
    return bool(re.search(
        r"\b(?:gap|open obligation|unresolved|unmet|missing|stale|counterexample|"
        r"refus(?:e|ed|al)|required work|incomplete|no_tests)\b",
        text, re.IGNORECASE,
    ))


def _has_witness(text: str) -> bool:
    import re
    return bool(re.search(
        r"\b(?:witness|evidence|traceback|assert(?:ion)?error|pytest exit|"
        r"counterexample|expected|observed|keyerror|receipt|source metadata)\b",
        text, re.IGNORECASE,
    ))


def _has_current_source(text: str, named_file: str | None) -> bool:
    if not named_file:
        return False
    for match in _SOURCE_MARKER_RE.finditer(text):
        if _normal_path(match.group(1)) == named_file and match.group(2).strip():
            return True
    return False


def _edit_allows_file(request: Mapping[str, object] | None, named_file: str | None) -> bool:
    if not named_file:
        return False
    for tool in _request_tools(request):
        function = tool.get("function")
        if not isinstance(function, Mapping):
            continue
        name = str(function.get("name", "")).lower()
        if name not in _EDIT_TOOL_NAMES:
            continue
        description = str(function.get("description", ""))
        lowered_description = description.lower()
        if (name == "edit" and "only the entry function" in lowered_description
                and named_file not in description):
            continue
        parameters = function.get("parameters", {})
        if not isinstance(parameters, Mapping):
            return True
        properties = parameters.get("properties", {})
        path_property = properties.get("path") if isinstance(properties, Mapping) else None
        if not isinstance(path_property, Mapping):
            return True
        enum = path_property.get("enum")
        if isinstance(enum, Sequence) and not isinstance(enum, (str, bytes)):
            if named_file in {_normal_path(item) for item in enum}:
                return True
            continue
        # A free path parameter is an affordance for the named current file;
        # the serialized source block supplies the target binding context.
        return True
    return False


def audit_serialized_requests(path: str | os.PathLike[str]) -> list[dict]:
    """Audit the request immediately following every serialized refusal.

    This is an offline harness scanner.  It reports presence/absence only; it
    never treats a prompt claim as proof that source bytes or an edit route
    were actually available.
    """
    records = _load_scanner_records(path)
    requests = [
        (index, record) for index, record in enumerate(records)
        if record.get("kind") == "request_prepared"
    ]
    findings: list[dict] = []
    for index, refusal in enumerate(records):
        if refusal.get("kind") not in _BLOCKER_ACTION_KINDS:
            continue
        nested = _action_record(refusal)
        result = str(nested.get("result", nested.get("status", ""))).upper()
        if result not in _REFUSAL_RESULTS and not result.startswith("REFUSED"):
            continue
        next_request = next((record for request_index, record in requests
                             if request_index > index), None)
        text = _request_text(next_request)
        named_file = _named_file(refusal, next_request)
        gap_text = _has_gap_text(text)
        witness = _has_witness(text)
        current_source = _has_current_source(text, named_file)
        edit_tool = _edit_allows_file(next_request, named_file)
        row = {
            "refusal_event_id": refusal.get("event_id", f"record:{index}"),
            "next_request_event_id": (next_request.get("event_id")
                                       if isinstance(next_request, Mapping) else None),
            "named_file": named_file,
            "gap_text": gap_text,
            "next_request_gap_text": gap_text,
            "witness": witness,
            "next_request_witness": witness,
            "current_source": current_source,
            "current_source_bytes": current_source,
            "edit_tool": edit_tool,
            "edit_tool_allows_file": edit_tool,
        }
        findings.append(row)
    return findings




def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class _ReplayBlobCache(OrderedDict):
    """LRU bounded by verified logical bytes and entries, local to one read pass."""

    def __init__(self, byte_limit: int = 1024 * 1024, entry_limit: int = 2048):
        super().__init__()
        self.byte_limit = byte_limit
        self.entry_limit = entry_limit
        self.bytes = 0

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if key in self:
            self.bytes -= super().pop(key)[0]
        size = value[0]
        if size > self.byte_limit:
            return
        super().__setitem__(key, value)
        self.bytes += size
        while self.bytes > self.byte_limit or len(self) > self.entry_limit:
            _, removed = self.popitem(last=False)
            self.bytes -= removed[0]


class _CachedRecordView(Sequence[dict]):
    """A fixed journal prefix; hydrate and detach only rows actually read."""

    def __init__(self, journal: ContinuationJournal, sequence: int, head: str | None):
        self._journal = journal
        self._sequence = sequence
        self._head = head

    def __len__(self) -> int:
        return self._sequence

    def __iter__(self) -> Iterator[dict]:
        return self._iter_range(0, self._sequence)

    def _iter_range(self, start: int, stop: int) -> Iterator[dict]:
        journal = self._journal
        with journal._lock:
            journal.assert_owner()
            if hasattr(journal, "_record_offsets"):
                journal._ensure_record_index()
                if (journal._sequence < self._sequence or (self._sequence
                        and journal._record_hashes[self._sequence - 1] != self._head)):
                    raise JournalError("cached journal prefix changed")
                yield from journal._record_range(start, stop)
            else:
                yield from (self[index] for index in range(start, stop))

    def __getitem__(self, index: int | slice) -> dict | tuple[dict, ...]:
        if isinstance(index, slice):
            start, stop, step = index.indices(self._sequence)
            if step == 1:
                return tuple(self._iter_range(start, stop))
            return tuple(self[position] for position in range(*index.indices(self._sequence)))
        if not isinstance(index, int):
            raise TypeError("journal record index must be an integer or slice")
        if index < 0:
            index += self._sequence
        if index < 0 or index >= self._sequence:
            raise IndexError("journal record index out of range")
        journal = self._journal
        with journal._lock:
            journal.assert_owner()
            if hasattr(journal, "_record_offsets"):
                ContinuationJournal._ensure_record_index(journal)
                if (journal._sequence < self._sequence
                        or journal._record_hashes[self._sequence - 1] != self._head):
                    raise JournalError("cached journal prefix changed")
            cache_start = getattr(journal, "_record_cache_start", 0)
            cache_index = index - cache_start
            if 0 <= cache_index < len(journal._record_cache):
                return copy.deepcopy(journal._record_cache[cache_index])
            return ContinuationJournal._indexed_record(journal, index + 1)


class ContinuationJournal(AuditJournal):
    """Hold a cooperative POSIX writer lease until close/process termination.

    Resume validates the complete journal before appending an ownership event.
    assert_owner proves storage ownership ONLY, not permission to execute work.
    An executive must separately reconstruct/reconcile outstanding intentions
    before authorizing effects, even when a trusted head matches. This class
    does not execute, authorize or retry effects.
    """

    def __init__(self, parent: Path, task_id: str, manifest: Mapping[str, object]):
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("task_id must be nonempty")
        # Validate/clone before creating anything.
        frozen_manifest = json.loads(canonical_bytes(dict(manifest)))
        parent = Path(parent)
        if not parent.is_dir():
            raise ValueError("journal parent must already exist")
        run_id = uuid.uuid4().hex
        run_dir = parent / run_id
        run_dir.mkdir(mode=0o700)
        self._initialize(run_dir / "events.jsonl")
        try:
            self._open(create=True)
            self.run_id, self.task_id = run_id, task_id
            self.manifest = frozen_manifest
            self.config_hash = digest(self.manifest)
            self._start_cold_start_host_span()
            self.emit("run_started", manifest=self.manifest, config_hash=self.config_hash)
            self.emit(
                "runtime_mode_snapshot", producer="kernel_measurement",
                modes=_runtime_stateframe_mode_snapshot(),
            )
            _sync_directory(run_dir)
            _sync_directory(parent)
        except BaseException:
            self.close()
            raise

    def _initialize(self, path: Path) -> None:
        if path.name != "events.jsonl" or path.parent.is_symlink():
            raise JournalError("journal must be a real run directory/events.jsonl")
        self.path = path.absolute()
        self._lock = threading.RLock()
        self._fd: int | None = None
        self._pid = os.getpid()
        self._poisoned = False
        # 10-08 (L126e 928/929 died at assert_owner with no cause): why the writer is unusable.
        self._poison_cause: str | None = None
        self._observer_failures: list[dict[str, object]] = []
        self._reporting_observer_failures = False
        self._sequence = 0
        self._head: str | None = None
        self._size = 0
        self._blob_cache: dict[str, int] = {}
        # Keep recent validated records for cursor readers. Older records remain
        # available through the hash-verified journal; an unbounded list grows
        # by several GB on long runs. The lab-only legacy mode restores it.
        self._record_cache: list[dict] = []
        self._record_cache_start = 0
        self._record_cache_bytes = 0
        self._record_cache_sizes: list[int] = []
        # One compact pointer and one digest per durable line. Old cursors and
        # heads can be resolved without hydrating the preceding journal.
        self._record_offsets: list[int] = []
        self._record_hashes: list[str] = []
        self._record_cache_limit = (None if os.environ.get(MEMBOUND_ENV, "bounded").lower() == "legacy"
                                    else RECORD_CACHE_LIMIT)
        self._framefit_receipt_records: list[dict[str, object]] = []
        self._framefit_receipt_history = None
        self._framefit_pending_coverage_session = None
        self._framefit_receipt_cursor = 0
        self._framefit_receipt_overflow = False
        self._framefit_latest_check: dict[str, object] | None = None
        self._framefit_last_applied_edit_after_check: dict[str, object] | None = None
        self._framefit_journal_origin_mode = framefit5_journal_origin_mode()
        self._framefit_origin_shadow_mismatches: dict[str, dict[str, int]] = {}
        self._replaying = False
        self._notifying = False
        self._subscribers: dict[str, Callable[[dict], None]] = {}
        self._subscriber_kinds: dict[str, frozenset[str] | None] = {}
        self._subscriber_cursors: dict[str, int] = {}
        self._publication_stats = {
            "replayed": 0, "published": 0, "filtered": 0, "deep_copies": 0,
        }
        self.host_spans = HostSpanRecorder()
        self._cold_start_root = None
        self._cold_start_span_closed = False
        self._host_event_spans: dict[str, list[object]] = {}
        self._host_event_span_seen: set[str] = set()
        self._receipt_refs: list[tuple[int, str]] = []
        self._receipt_ref_set: set[str] = set()
        self._response_request_ids: set[str] = set()
        self._request_sequences: dict[str, list[int]] = {}
        self._tool_reconciliation_records: dict[tuple[str, str], dict[str, object]] = {}
        self._journal_applied_edit_rows: list[dict[str, object]] = []
        self._vacuity_latest_check: dict[str, object] | None = None
        self._vacuity_last_edit: dict[str, object] | None = None
        self._receipt_generation = 0
        self._outline_capability_reports: set[tuple[str, str, str, str]] = set()
        self._outline_capability_listener_token: contextvars.Token | None = None
        from mc_repo_intel_vendor.ts_dispatch import bind_outline_capability_listener

        self._outline_capability_listener_token = bind_outline_capability_listener(
            self._record_outline_parser_capability)

    def _record_outline_parser_capability(
        self,
        language: str,
        parser: str,
        capability: str,
        completeness: str,
    ) -> None:
        report = (language, parser, capability, completeness)
        with self._lock:
            if report in self._outline_capability_reports:
                return
            self._outline_capability_reports.add(report)
            try:
                self.emit(
                    "outline_parser_capability",
                    producer=(
                        "kernel_continuation_journal."
                        "ContinuationJournal._record_outline_parser_capability"
                    ),
                    language=language,
                    parser=parser,
                    capability=capability,
                    completeness=completeness,
                )
            except BaseException:
                self._outline_capability_reports.discard(report)
                raise

    def _start_cold_start_host_span(self) -> None:
        if self._cold_start_root is None:
            self._cold_start_root = self.host_spans.start_span(
                "first_turn_to_submit", force=True,
                boundary="request_preparation",
            )

    def _start_host_event_span(
            self, name: str, parent_name: str | None, *, repeatable: bool = False) -> None:
        if not repeatable and name in self._host_event_span_seen:
            return
        if not repeatable:
            self._host_event_span_seen.add(name)
        parent_span = (
            self.host_spans.find_active_ancestor(parent_name)
            if parent_name is not None else None
        )
        if parent_span is None and self._cold_start_root is not None:
            parent_span = self._cold_start_root.span_id
        handle = self.host_spans.start_span(
            name, boundary="request_preparation", parent_span=parent_span,
            force=True,
        )
        self._host_event_spans.setdefault(name, []).append(handle)

    def _finish_host_event_span(self, name: str) -> None:
        handles = self._host_event_spans.get(name)
        if not handles:
            return
        self.host_spans.finish_span(handles.pop())
        if not handles:
            self._host_event_spans.pop(name, None)

    def _finish_cold_start_host_spans(self) -> None:
        for name in tuple(self._host_event_spans):
            while self._host_event_spans.get(name):
                self._finish_host_event_span(name)
        self.host_spans.finish_span(self._cold_start_root)
        self._cold_start_root = None
        self._cold_start_span_closed = True

    def _finish_unclosed_warm_host_spans(self) -> None:
        for name, _start_kind, _end_kind, _parent_name in _WARM_HOST_EVENT_SPAN_PAIRS:
            while self._host_event_spans.get(name):
                self._finish_host_event_span(name)

    def _host_span_event_boundary(self, edge: str, kind: str) -> None:
        """Trace selected startup and warm-request gaps at their journal boundaries."""
        if not host_span_profiles_enabled():
            return
        if edge == "after" and kind == "run_started":
            self._start_cold_start_host_span()
            return
        if not self._cold_start_span_closed:
            for name, start_kind, end_kind, parent_name in _HOST_EVENT_SPAN_PAIRS:
                if edge == "after" and kind == start_kind:
                    self._start_host_event_span(name, parent_name)
                elif edge == "before" and kind == end_kind:
                    self._finish_host_event_span(name)
            if edge == "after" and kind == "request_submitted":
                self._finish_cold_start_host_spans()
            return
        for name, start_kind, end_kind, parent_name in _WARM_HOST_EVENT_SPAN_PAIRS:
            if edge == "after" and kind == start_kind:
                self._start_host_event_span(name, parent_name, repeatable=True)
            elif edge == "before" and kind == end_kind:
                self._finish_host_event_span(name)
        if edge == "after" and kind == "request_submitted":
            self._finish_unclosed_warm_host_spans()

    def _open(self, *, create: bool) -> None:
        flags = os.O_RDWR | os.O_APPEND | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
        if create:
            flags |= os.O_CREAT | os.O_EXCL
        fd = os.open(self.path, flags, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise JournalError("journal is not a regular file")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._size = os.fstat(fd).st_size
            self._fd = fd
        except BaseException:
            os.close(fd)
            raise

    @classmethod
    def resume(cls, path: Path, *, task_id: str,
               manifest: Mapping[str, object], expected_head: str | None = None,
               allow_unanchored_recovery: bool = False) -> ContinuationJournal:
        if type(allow_unanchored_recovery) is not bool:
            raise ValueError("allow_unanchored_recovery must be boolean")
        if expected_head is None and not allow_unanchored_recovery:
            raise JournalError("trusted head required; unanchored recovery must be explicit")
        if expected_head is not None and (
                not isinstance(expected_head, str) or len(expected_head) != 71
                or not expected_head.startswith("sha256:")
                or any(c not in "0123456789abcdef" for c in expected_head[7:])):
            raise ValueError("invalid trusted head")
        expected_manifest = json.loads(canonical_bytes(dict(manifest)))
        expected_config = digest(expected_manifest)
        obj = cls.__new__(cls)
        obj._initialize(Path(path))
        try:
            obj._open(create=False)
            first = None
            last = None
            for record in obj.records():
                if first is None:
                    first = record
                last = record
            if first is None or last is None:
                raise JournalError("empty journal")
            if (first.get("manifest") != expected_manifest
                    or first.get("config_hash") != expected_config
                    or first["task_id"] != task_id):
                raise JournalError("task/config identity mismatch")
            if expected_head is not None and last["record_hash"] != expected_head:
                raise JournalError("trusted journal head mismatch")
            obj.run_id, obj.task_id = first["run_id"], task_id
            obj.manifest = copy.deepcopy(first["manifest"])
            obj.config_hash = expected_config
            obj._sequence, obj._head = last["sequence"], last["record_hash"]
            obj.emit("continuation_writer_resumed", previous_head=obj._head,
                     anchor_status="TRUSTED_HEAD_MATCH" if expected_head is not None
                     else "EXPLICIT_UNANCHORED_RECOVERY",
                     execution_authorized=False)
            return obj
        except BaseException:
            obj.close()
            raise

    @contextmanager
    def ownership(self) -> Iterator[None]:
        """Serialize a host transition with journal observers and other appends."""
        with self._lock:
            self.assert_owner()
            yield

    def subscribe(self, name: str, observer: Callable[[dict], None], *,
                  kinds: Sequence[str] | None = None) -> None:
        """Replay then attach atomically; only one projection per named slot."""
        with self.ownership():
            if self._notifying or not name or name in self._subscribers or not callable(observer):
                raise JournalError("invalid or already attached projection")
            selected = None if kinds is None else frozenset(str(kind) for kind in kinds)
            if selected is not None and not all(selected):
                raise JournalError("subscriber kinds must be non-empty strings")
            for record in self.records():
                self._subscriber_cursors[name] = int(record["sequence"])
                if selected is None or record.get("kind") in selected:
                    self._publication_stats["replayed"] += 1
                    self._publication_stats["deep_copies"] += 1
                    self._notify_isolated(name, observer, record)
            self._subscribers[name] = observer
            self._subscriber_kinds[name] = selected
        self._report_observer_failures()

    def subscriber_cursor(self, name: str) -> int:
        with self._lock:
            return self._subscriber_cursors.get(name, 0)

    def publication_stats(self) -> dict[str, int]:
        """Return host publication counters without exposing observer state."""
        with self._lock:
            return dict(self._publication_stats)

    @property
    def receipt_generation(self) -> int:
        with self._lock:
            return self._receipt_generation

    def receipt_cursor(self) -> ReceiptCursor:
        with self._lock:
            return ReceiptCursor(self._sequence, self._receipt_generation)

    def _index_receipt_record(self, record: Mapping[str, object]) -> None:
        kind = record.get("kind")
        if kind == "native_batch_action_closed":
            call_id, step_id = record.get("tool_call_id"), record.get("step_record_id")
            if isinstance(call_id, str) and isinstance(step_id, str):
                self._tool_reconciliation_records[("native", call_id)] = {
                    "kind": kind, "tool_call_id": call_id, "step_record_id": step_id,
                }
        elif kind == "action_outcome_closed" and isinstance(record.get("record"), Mapping):
            source = record["record"]
            # Match reconcile_tool_call_results AND its outcome_from_receipt helper.
            # Keep separate last-wins call and step maps: replacing a step must
            # not erase the older direct-call binding (or vice versa).
            fields = ("tool", "observation", "result_class", "result",
                      "status", "hit_count", "match_count", "kind")
            projected = {key: copy.deepcopy(source[key]) for key in fields if key in source
                         and (key != "result" or isinstance(source[key], str))}
            outcome = source.get("tool_outcome")
            if isinstance(outcome, Mapping):
                projected["tool_outcome"] = {
                    key: copy.deepcopy(outcome[key])
                    for key in ("kind", "code", "count", "start_line", "end_line") if key in outcome
                }
            for field, namespace in (("step_record_id", "step"), ("tool_call_id", "call")):
                identity = source.get(field)
                if isinstance(identity, str):
                    self._tool_reconciliation_records[(namespace, identity)] = {
                        "kind": kind, "record": {**projected, field: identity},
                    }
        if kind == "check_receipt":
            self._vacuity_latest_check = {
                key: copy.deepcopy(record[key])
                for key in ("kind", "status", "revision", "tested_rev", "receipt_id")
                if key in record
            }
            self._vacuity_last_edit = None
        elif kind == "action_receipt":
            from kernel_vacuity_truth import journal_applied_edit_rows

            rows = journal_applied_edit_rows((record,))
            if rows:
                # emit() returns record to its caller. Own nested projection
                # values independently, copying only this newly admitted row.
                self._journal_applied_edit_rows.extend(copy.deepcopy(rows))
                receipt = record["receipt"]
                self._vacuity_last_edit = {
                    "kind": "action_receipt",
                    "receipt": {
                        key: copy.deepcopy(receipt[key])
                        for key in ("kind", "result_class", "receipt_id")
                        if key in receipt
                    },
                }
        request_id = record.get("request_id")
        if isinstance(request_id, str) and request_id:
            self._request_sequences.setdefault(request_id, []).append(int(record["sequence"]))
        if record.get("kind") == "response_received" and isinstance(request_id, str):
            self._response_request_ids.add(request_id)
        refs = _receipt_refs_from_record(record)
        if not refs:
            return
        sequence = int(record["sequence"])
        for ref in refs:
            if ref in self._receipt_ref_set:
                continue
            self._receipt_ref_set.add(ref)
            self._receipt_refs.append((sequence, ref))

    @staticmethod
    def _framefit_scalar(value: object) -> str | int | bool | None:
        if type(value) is str:
            return value[:MAX_FRAMEFIT_RECEIPT_FIELD_CHARS]
        if type(value) in {int, bool}:
            return value
        return None

    @staticmethod
    def _framefit_failure_context(record: Mapping[str, object]) -> str:
        """Keep bounded host check evidence through the receipt projection.

        Older journals put the preview only in the quoted summary.  Decode that
        host-authored field during replay so a resumed run sees the same location.
        """
        preview = record.get("assertion_context_preview")
        if isinstance(preview, str):
            return preview[:4096]
        summary = record.get("summary")
        if not isinstance(summary, str) or len(summary) > 65536:
            return ""
        marker = " assertion_context="
        if marker not in summary:
            return ""
        literal = summary.split(marker, 1)[1]
        try:
            first = next(tokenize.generate_tokens(io.StringIO(literal).readline))
            if first.type != tokenize.STRING:
                return ""
            decoded = ast.literal_eval(first.string)
        except (StopIteration, SyntaxError, ValueError, TypeError, MemoryError,
                RecursionError, tokenize.TokenError):
            return ""
        return decoded[:4096] if isinstance(decoded, str) else ""

    def _index_framefit_receipt(self, record: Mapping[str, object], *, coverage=None) -> None:
        """Keep receipt detail until the semantic history reducer is needed."""
        kind = record.get("kind")
        if kind not in {"check_receipt", "action_receipt"}:
            return

        event_id = self._framefit_scalar(record.get("event_id"))
        sequence = self._framefit_scalar(record.get("sequence"))
        if not isinstance(event_id, str):
            self._framefit_receipt_overflow = True
            return
        origin_fields_present = (
            "source_event_id" in record or "source_sequence" in record
        )
        if origin_fields_present and self._framefit_journal_origin_mode != "off":
            source_event_id = self._framefit_scalar(record.get("source_event_id"))
            source_sequence = self._framefit_scalar(record.get("source_sequence"))
            origin_valid = (
                isinstance(source_event_id, str)
                and bool(source_event_id)
                and len(source_event_id) <= 160
                and type(source_sequence) is int
                and source_sequence > 0
            )
            if self._framefit_journal_origin_mode == "on":
                if not origin_valid:
                    # Do not silently substitute a replay writer's identity for
                    # an explicitly supplied, malformed source-journal identity.
                    self._framefit_receipt_overflow = True
                    return
                event_id = source_event_id
                sequence = source_sequence
            elif not origin_valid:
                self._remember_framefit_origin_mismatch("source_identity", sequence)
            else:
                if source_event_id != event_id:
                    self._remember_framefit_origin_mismatch("event_id", sequence)
                if source_sequence != sequence:
                    self._remember_framefit_origin_mismatch("sequence", sequence)
        projected: dict[str, object] = {"kind": kind, "event_id": event_id}
        if type(sequence) is int:
            projected["sequence"] = sequence

        if kind == "check_receipt":
            from kernel_code_bookmarks import enabled as bookmarks_enabled
            bookmark_fields = bookmarks_enabled()
            fields = ("status", "revision", "tested_rev", "receipt_id", "selector")
            if bookmark_fields:
                fields += ("outcome", "failure_phase")
            for field in fields:
                value = self._framefit_scalar(record.get(field))
                if value is not None:
                    projected[field] = value
            if bookmark_fields and str(projected.get("status", "")).upper() == "FAIL":
                context = self._framefit_failure_context(record)
                if context:
                    projected["assertion_context_preview"] = context
            self._framefit_latest_check = copy.deepcopy(projected)
            self._framefit_last_applied_edit_after_check = None
        else:
            receipt = record.get("receipt")
            if not isinstance(receipt, Mapping):
                self._framefit_receipt_overflow = True
                return
            receipt_projection: dict[str, object] = {}
            for field in (
                    "kind", "result_class", "target_path", "receipt_id",
                    "workspace_revision", "request_id", "turn"):
                value = self._framefit_scalar(receipt.get(field))
                if value is not None:
                    receipt_projection[field] = value
            if not receipt_projection:
                self._framefit_receipt_overflow = True
                return
            projected["receipt"] = receipt_projection
            if (self._framefit_latest_check is not None
                    and str(receipt_projection.get("kind", "")).lower() == "edit"
                    and str(receipt_projection.get("result_class", "")).upper() == "APPLIED"):
                self._framefit_last_applied_edit_after_check = copy.deepcopy(projected)

        if (self._framefit_receipt_history is None
                and len(self._framefit_receipt_records) >= MAX_FRAMEFIT_RECEIPT_RECORDS):
            from kernel_framefit_receipt_history import ReceiptHistory

            from kernel_wire_identity import wire_run_id
            history = ReceiptHistory(public_run_id=wire_run_id(self))
            for previous in self._framefit_receipt_records:
                history.observe(previous, coverage=coverage)
            self._framefit_receipt_history = history
            self._framefit_receipt_records = []
        if self._framefit_receipt_history is not None:
            self._framefit_receipt_history.observe(projected, coverage=coverage)
        else:
            self._framefit_receipt_records.append(projected)

    def _sync_framefit_receipts(self) -> None:
        """Consume only new validated records; replay can rebuild the cursor itself."""
        history = self._framefit_receipt_history
        from kernel_framefit_receipt_history import coverage_session_for

        coverage = self._framefit_pending_coverage_session
        if coverage is None:
            coverage = (history.coverage_session(self) if history is not None
                        else coverage_session_for(self, None))
        for record in self.cached_records_since(self._framefit_receipt_cursor):
            if record["sequence"] <= self._framefit_receipt_cursor:
                continue
            self._advance_framefit_receipt(record, coverage=coverage)
        history = self._framefit_receipt_history
        if history is not None and coverage is not None:
            try:
                history.ensure_coverage(coverage)
                if self._framefit_receipt_overflow:
                    raise JournalError("journal FRAMEFIT5 receipt projection is incomplete")
                if coverage.digest is not None:
                    coverage.commit()
                self._framefit_pending_coverage_session = None
            except Exception:
                self._framefit_receipt_overflow = True
                raise

    def _advance_framefit_receipt(self, record: Mapping[str, object], *, coverage=None) -> None:
        try:
            indexer = self._index_framefit_receipt
            parameters = inspect.signature(indexer).parameters.values()
            accepts_coverage = any(
                parameter.name == "coverage"
                or parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters
            )
            if accepts_coverage:
                indexer(record, coverage=coverage)
            else:
                # Frozen C2 control reducers predate CoverageSession. Preserve
                # their exact refusal boundary while production reducers carry it.
                indexer(record)
        except Exception:
            # A semantic projection failure refuses typed construction, but must
            # not make an otherwise valid durable audit stream unreadable.
            self._framefit_receipt_overflow = True
        self._framefit_receipt_cursor = record["sequence"]

    def framefit_receipt_context(self) -> dict[str, object]:
        """Return receipt accounting with its non-wire journal coverage authority."""
        from kernel_stateframe_call_coverage import bind_index

        with self._lock:
            records = self.framefit_receipt_records()
            if self._framefit_receipt_overflow:
                raise JournalError("journal FRAMEFIT5 receipt projection is incomplete")
            if self._framefit_receipt_history is not None:
                context = self._framefit_receipt_history.context()
            else:
                context = {
                    "journal_receipt_records": records,
                    "journal_receipt_count": len(records),
                }
            return bind_index(context, self)

    def _remember_framefit_origin_mismatch(self, field: str, sequence: object) -> None:
        """Bound shadow evidence to one count/sequence range per mismatching field."""
        if self._framefit_journal_origin_mode != "shadow":
            return
        observed = sequence if type(sequence) is int and sequence > 0 else 0
        summary = self._framefit_origin_shadow_mismatches.get(field)
        if summary is None:
            self._framefit_origin_shadow_mismatches[field] = {
                "count": 1,
                "first_sequence": observed,
                "last_sequence": observed,
            }
            return
        summary["count"] += 1
        if not summary["first_sequence"] or (observed and observed < summary["first_sequence"]):
            summary["first_sequence"] = observed
        if observed > summary["last_sequence"]:
            summary["last_sequence"] = observed

    @property
    def framefit_receipt_overflowed(self) -> bool:
        with self._lock:
            self.assert_owner()
            self._sync_framefit_receipts()
            return self._framefit_receipt_overflow

    def framefit_receipt_records(self) -> tuple[dict[str, object], ...]:
        """Return witnesses; framefit_receipt_context() also carries history accounting."""
        with self._lock:
            self.assert_owner()
            self._sync_framefit_receipts()
            records = (self._framefit_receipt_history.context()["journal_receipt_records"]
                       if self._framefit_receipt_history is not None
                       else copy.deepcopy(self._framefit_receipt_records))
            shadow_mismatches = copy.deepcopy(self._framefit_origin_shadow_mismatches)
            self._framefit_origin_shadow_mismatches.clear()
            if self._framefit_receipt_overflow:
                latest_check = self._framefit_latest_check
                if latest_check is not None:
                    latest_id = latest_check.get("event_id")
                    if not any(row.get("event_id") == latest_id for row in records):
                        records.insert(0, copy.deepcopy(latest_check))
                    latest_edit = self._framefit_last_applied_edit_after_check
                    if (latest_edit is not None
                            and not any(row.get("event_id") == latest_edit.get("event_id")
                                        for row in records)):
                        records.append(copy.deepcopy(latest_edit))
        for field, summary in shadow_mismatches.items():
            try:
                self.emit(
                    f"framefit5_journal_origin_shadow_mismatch_{field}",
                    producer="kernel_continuation_journal",
                    field=field,
                    mismatch_count=summary["count"],
                    first_runtime_sequence=summary["first_sequence"],
                    last_runtime_sequence=summary["last_sequence"],
                    dispatch=False,
                )
            except Exception:
                # Shadow telemetry is advisory; a diagnostic failure must not
                # replace the legacy receipt projection used by this request.
                pass
        return tuple(records)

    def _rebuild_receipt_index(self, records: Iterable[Mapping[str, object]]) -> None:
        self._receipt_refs.clear()
        self._receipt_ref_set.clear()
        self._response_request_ids.clear()
        self._request_sequences = {}
        self._tool_reconciliation_records = {}
        self._journal_applied_edit_rows = []
        self._vacuity_latest_check = None
        self._vacuity_last_edit = None
        self._receipt_generation = 0
        for record in records:
            self._index_receipt_record(record)
            # Advance on every durable event, including an irrelevant one.
            # Derived receipt refs still fold only admitted fact identities.
            self._receipt_generation = int(record["sequence"])

    def journal_applied_edit_rows(self) -> tuple[dict[str, object], ...]:
        """Exact applied-edit ledger for delivery manifests, in journal order."""
        with self._lock:
            self.assert_owner()
            if (self._receipt_generation != self._sequence
                    or not hasattr(self, "_journal_applied_edit_rows")):
                for _ in self.records():
                    pass
            return tuple(copy.deepcopy(row) for row in self._journal_applied_edit_rows)

    def vacuity_truth_records(self) -> tuple[dict[str, object], ...]:
        """Exact minimal records needed to project the latest check truth."""
        with self._lock:
            self.assert_owner()
            if (self._receipt_generation != self._sequence
                    or not hasattr(self, "_vacuity_latest_check")):
                for _ in self.records():
                    pass
            return tuple(copy.deepcopy(row) for row in (
                self._vacuity_latest_check, self._vacuity_last_edit)
                         if row is not None)

    def request_records(self, request_id: str) -> tuple[dict, ...]:
        """Read the exact journal events for one request without a full scan."""
        if not isinstance(request_id, str) or not request_id:
            return ()
        with self._lock:
            self.assert_owner()
            if (self._receipt_generation != self._sequence
                    or not hasattr(self, "_request_sequences")):
                for _ in self.records():
                    pass
            return tuple(self._indexed_record(sequence)
                         for sequence in self._request_sequences.get(request_id, ()))

    def tool_reconciliation_records(self) -> tuple[dict[str, object], ...]:
        """Return only durable native-call and closed-outcome facts for LEAN/M3."""
        with self._lock:
            self.assert_owner()
            if (self._receipt_generation != self._sequence
                    or not isinstance(getattr(self, "_tool_reconciliation_records", None), dict)):
                for _ in self.records():
                    pass
            return tuple(copy.deepcopy(row) for row in self._tool_reconciliation_records.values())

    def has_response_received(self, request_id: str) -> bool:
        """Check the durable response index before a retry can resubmit a request."""
        if not isinstance(request_id, str) or not request_id:
            return False
        with self._lock:
            self.assert_owner()
            return request_id in self._response_request_ids

    def receipt_refs_at(self, head: object = None) -> tuple[str, ...]:
        """Return derived receipt references through a sequence/hash head."""
        with self._lock:
            if head is None:
                limit = self._sequence
            elif isinstance(head, ReceiptCursor):
                limit = head.sequence
            elif type(head) is int and head >= 0:
                limit = head
            elif isinstance(head, str):
                matching = next((record["sequence"] for record in self._record_cache
                                 if record.get("record_hash") == head), None)
                if matching is None:
                    self._ensure_record_index()
                    try:
                        matching = self._record_hashes.index(head) + 1
                    except ValueError:
                        pass
                    else:
                        self._indexed_record(matching)
                if matching is None:
                    raise JournalError("unknown journal head")
                limit = int(matching)
            else:
                raise ValueError("head must be a sequence, hash, cursor, or None")
            if limit > self._sequence:
                raise JournalError("journal head is beyond durable sequence")
            return tuple(ref for sequence, ref in self._receipt_refs if sequence <= limit)

    def new_receipts_after(self, cursor: object = None) -> tuple[str, ...]:
        """Return only new derived refs after a durable sequence cursor."""
        if cursor is None:
            sequence = 0
        elif isinstance(cursor, ReceiptCursor):
            sequence = cursor.sequence
        elif type(cursor) is int and cursor >= 0:
            sequence = cursor
        else:
            raise ValueError("cursor must be a non-negative sequence or ReceiptCursor")
        with self._lock:
            if sequence > self._sequence:
                raise JournalError("receipt cursor is beyond durable sequence")
            return tuple(ref for item_sequence, ref in self._receipt_refs
                         if item_sequence > sequence)

    @property
    def head(self) -> str | None:
        return self._head

    def assert_owner(self) -> None:
        """Reject closed/forked/poisoned writers and inode/size drift.

        Same-size in-place tampering is outside the cooperative-writer lease;
        full chain validation occurs at replay, not before every dispatch.
        """
        if self._fd is None or os.getpid() != self._pid or self._poisoned:
            state = ("closed" if self._fd is None else "inherited" if os.getpid() != self._pid
                     else "poisoned")
            cause = getattr(self, "_poison_cause", None) if state == "poisoned" else None
            raise JournalError(f"worker is closed, inherited or poisoned ({state}"
                               + (f": {cause}" if cause else "") + ")")
        try:
            opened = os.fstat(self._fd)
            current = self.path.lstat()
        except OSError as error:
            self._poison(f"ownership check failed: {type(error).__name__}")
            raise JournalError("journal ownership cannot be verified") from error
        if (not stat.S_ISREG(current.st_mode)
                or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
                or opened.st_size != self._size):
            self._poison("journal changed outside its writer")
            raise JournalError("journal changed outside its writer")

    def records(self, *, rebuild: bool = True) -> Iterator[dict]:
        """Validate the whole stream; optionally leave live projections untouched."""
        with self._lock:
            self.assert_owner()
            if self._replaying:
                raise JournalError("nested replay is forbidden")
            self._replaying = True
            try:
                yield from self._records(rebuild=rebuild)
            finally:
                self._replaying = False

    def _records(self, *, rebuild: bool = True) -> Iterator[dict]:
        with self._lock:
            self.assert_owner()
            validated: deque[tuple[dict, int]] = deque()
            cache_start = 0
            cache_bytes = 0
            offsets: list[int] = []
            hashes: list[str] = []
            first_record: dict | None = None
            legacy = self._record_cache_limit is None
            blob_cache = {} if legacy else _ReplayBlobCache()
            projection = None
            coverage_index = None
            if rebuild and not legacy:
                # Publish derived state only after the entire pass validates.
                projection = copy.copy(self)
                projection._receipt_refs = []
                projection._receipt_ref_set = set()
                projection._response_request_ids = set()
                projection._rebuild_receipt_index(())
                projection._reset_framefit_receipt_projection()
                projection._framefit_origin_shadow_mismatches = copy.deepcopy(
                    self._framefit_origin_shadow_mismatches)
            fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as handle:
                reader, writer = os.fstat(handle.fileno()), os.fstat(self._fd)
                if (not stat.S_ISREG(reader.st_mode)
                        or (reader.st_dev, reader.st_ino) != (writer.st_dev, writer.st_ino)):
                    self._poison("journal replaced while opening reader")
                    raise JournalError("journal replaced while opening reader")
                sequence, previous, task_id = 0, None, None
                while True:
                    offset = handle.tell()
                    line = handle.readline(MAX_RECORD_BYTES + 1)
                    if not line:
                        break
                    sequence += 1
                    if len(line) > MAX_RECORD_BYTES or not line.endswith(b"\n"):
                        raise JournalError("oversize or torn journal record")
                    try:
                        record = json.loads(line)
                        canonical = canonical_bytes(record) + b"\n"
                    except (ValueError, TypeError, UnicodeError) as error:
                        raise JournalError("invalid journal JSON") from error
                    if not isinstance(record, dict) or canonical != line:
                        raise JournalError("record must be canonical JSON object")
                    if sequence == 1:
                        task_id = record.get("task_id")
                        if (record.get("kind") != "run_started"
                                or not isinstance(task_id, str) or not task_id):
                            raise JournalError("missing run identity")
                    if (record.get("schema") != "stateframe-audit/1"
                            or (sequence > 1 and record.get("kind") == "run_started")
                            or record.get("journal_format") != FORMAT
                            or type(record.get("sequence")) is not int
                            or record["sequence"] != sequence
                            or record.get("run_id") != self.path.parent.name
                            or record.get("task_id") != task_id
                            or record.get("event_id") != f"{self.path.parent.name}:{sequence}"
                            or not isinstance(record.get("kind"), str) or not record["kind"]
                            or type(record.get("timestamp_ns")) is not int
                            or record["timestamp_ns"] <= 0
                            or record.get("previous_hash") != previous):
                        raise JournalError("invalid journal lineage")
                    supplied = record.get("record_hash")
                    body = {key: value for key, value in record.items() if key != "record_hash"}
                    if supplied != digest(body):
                        raise JournalError("journal hash mismatch")
                    previous = supplied
                    offsets.append(offset)
                    hashes.append(supplied)
                    try:
                        resolved = resolve_journal_blob_refs(record, self.path, blob_cache)
                    except (OSError, ValueError) as error:
                        raise JournalError("invalid journal blob reference") from error
                    if not isinstance(resolved, dict):
                        raise JournalError("journal blob did not resolve to an object")
                    if rebuild:
                        if first_record is None:
                            first_record = resolved
                            if projection is not None:
                                from kernel_stateframe_call_coverage import CoverageIndex, _CONTEXT_INDEXES

                                projection.run_id = resolved["run_id"]
                                projection.task_id = resolved["task_id"]
                                coverage_index = CoverageIndex(projection)
                                existing = getattr(self, "_framefit_call_coverage_index", None)
                                if existing is not None:
                                    _CONTEXT_INDEXES[existing.context_handle] = existing
                        if projection is not None:
                            from kernel_stateframe_call_coverage import EVENT

                            projection._index_receipt_record(resolved)
                            projection._receipt_generation = sequence
                            projection._advance_framefit_receipt(resolved)
                            if resolved.get("kind") == EVENT:
                                try:
                                    coverage_index._accept(resolved.get("coverage_proof"))
                                except Exception:
                                    projection._framefit_receipt_overflow = True
                            coverage_index.sequence = sequence
                        record_bytes = len(canonical_bytes(resolved))
                        validated.append((resolved, record_bytes))
                        cache_bytes += record_bytes
                        if self._record_cache_limit is not None:
                            while (len(validated) > self._record_cache_limit
                                   or cache_bytes > RECORD_CACHE_BYTE_LIMIT):
                                _, removed_bytes = validated.popleft()
                                cache_start += 1
                                cache_bytes -= removed_bytes
                    # Replay consumers own their rows too. Keeping the same
                    # object here would let them corrupt a later cached read.
                    yield copy.deepcopy(resolved) if rebuild and not legacy else resolved
                self.assert_owner()
            self._record_offsets = offsets
            self._record_hashes = hashes
            if rebuild:
                self._record_cache = [row for row, _size in validated]
                self._record_cache_sizes = [size for _row, size in validated]
                self._record_cache_start = cache_start
                self._record_cache_bytes = cache_bytes
                if self._sequence == 0 and first_record is not None:
                    # A resumed writer has no in-memory head until first replay.
                    # Bind the verified run identity before receipt replay: the
                    # call-coverage index derives its authority from this ID.
                    # Keep the full stream until those projections are built.
                    self._sequence, self._head = sequence, previous
                    self.run_id = first_record["run_id"]
                    self.task_id = first_record["task_id"]
                if legacy:
                    self._rebuild_receipt_index(self._record_cache)
                    if sequence == self._sequence:
                        self._rebuild_framefit_receipt_projection(self._record_cache)
                    else:
                        self._reset_framefit_receipt_projection()
                else:
                    for name in ("_receipt_refs", "_receipt_ref_set", "_response_request_ids",
                                 "_request_sequences", "_tool_reconciliation_records",
                                 "_journal_applied_edit_rows", "_vacuity_latest_check",
                                 "_vacuity_last_edit", "_receipt_generation"):
                        setattr(self, name, getattr(projection, name))
                    if sequence == self._sequence:
                        for name in ("_framefit_receipt_records", "_framefit_receipt_history",
                                     "_framefit_receipt_cursor", "_framefit_receipt_overflow",
                                     "_framefit_latest_check", "_framefit_last_applied_edit_after_check",
                                     "_framefit_origin_shadow_mismatches"):
                            setattr(self, name, getattr(projection, name))
                        self._framefit_pending_coverage_session = None
                        if coverage_index is not None:
                            # Preserve existing contexts referencing the live index.
                            existing = getattr(self, "_framefit_call_coverage_index", None)
                            if existing is not None:
                                existing.__dict__.update(coverage_index.__dict__)
                                coverage_index = existing
                            coverage_index.journal = self
                            self._framefit_call_coverage_index = coverage_index
                            from kernel_stateframe_call_coverage import _CONTEXT_INDEXES
                            _CONTEXT_INDEXES[coverage_index.context_handle] = coverage_index
                        if self._framefit_receipt_history is not None:
                            from kernel_framefit_receipt_history import coverage_session_for
                            coverage = coverage_session_for(self, None)
                            self._framefit_receipt_history.ensure_coverage(coverage)
                            if coverage.digest is not None:
                                self._framefit_pending_coverage_session = coverage
                    else:
                        self._reset_framefit_receipt_projection()
                self._trim_record_cache()

    def validate_checkpoint_journal(self) -> None:
        """Verify a trimmed checkpoint's full journal without retaining its heap."""
        with self._lock:
            self.assert_owner()
            if self._replaying:
                raise JournalError("nested replay is forbidden")
            self._replaying = True
            try:
                count, head = 0, None
                for record in self._records(rebuild=False):
                    count, head = record["sequence"], record["record_hash"]
                if count != self._sequence or head != self._head:
                    raise JournalError("checkpoint journal head does not match saved state")
            finally:
                self._replaying = False

    def _trim_record_cache(self) -> None:
        limit = self._record_cache_limit
        if (not hasattr(self, "_record_cache_sizes")
                or len(self._record_cache_sizes) != len(self._record_cache)):
            # Compatibility with checkpoints made before bounded retention.
            self._record_cache_sizes = [len(canonical_bytes(row)) for row in self._record_cache]
            self._record_cache_bytes = sum(self._record_cache_sizes)
        if limit is not None:
            while (len(self._record_cache) > limit
                   or self._record_cache_bytes > RECORD_CACHE_BYTE_LIMIT):
                self._record_cache.pop(0)
                self._record_cache_start += 1
                self._record_cache_bytes -= self._record_cache_sizes.pop(0)

    def _ensure_record_index(self) -> None:
        if (len(getattr(self, "_record_offsets", ())) != self._sequence
                or len(getattr(self, "_record_hashes", ())) != self._sequence
                or (self._sequence and self._record_hashes[-1] != self._head)):
            for _ in self.records(rebuild=False):
                pass
            if (len(self._record_offsets) != self._sequence
                    or len(self._record_hashes) != self._sequence
                    or (self._sequence and self._record_hashes[-1] != self._head)):
                raise JournalError("journal index does not match durable head")

    def _indexed_record(self, sequence: int) -> dict:
        """Hydrate one verified line without materializing its predecessors."""
        return next(self._indexed_records(sequence, sequence))

    def _indexed_records(self, first: int, last: int) -> Iterator[dict]:
        """Hydrate a contiguous range with one open and a bounded shared blob LRU."""
        self._ensure_record_index()
        if first > last:
            return
        blob_cache = _ReplayBlobCache()
        fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as handle:
            reader, writer = os.fstat(handle.fileno()), os.fstat(self._fd)
            if ((reader.st_dev, reader.st_ino) != (writer.st_dev, writer.st_ino)
                    or not stat.S_ISREG(reader.st_mode)):
                raise JournalError("journal replaced while reading indexed record")
            handle.seek(self._record_offsets[first - 1])
            for sequence in range(first, last + 1):
                self.assert_owner()
                if handle.tell() != self._record_offsets[sequence - 1]:
                    raise JournalError("indexed journal offset mismatch")
                line = handle.readline(MAX_RECORD_BYTES + 1)
                yield self._resolve_indexed_line(line, sequence, blob_cache)

    def _resolve_indexed_line(self, line: bytes, sequence: int, blob_cache: dict) -> dict:
        if len(line) > MAX_RECORD_BYTES or not line.endswith(b"\n"):
            raise JournalError("oversize or torn indexed journal record")
        try:
            record = json.loads(line)
            canonical = canonical_bytes(record) + b"\n"
        except (ValueError, TypeError, UnicodeError) as error:
            raise JournalError("invalid indexed journal JSON") from error
        expected_previous = self._record_hashes[sequence - 2] if sequence > 1 else None
        expected_hash = self._record_hashes[sequence - 1]
        if (not isinstance(record, dict) or canonical != line
                or type(record.get("sequence")) is not int
                or record["sequence"] != sequence
                or record.get("schema") != "stateframe-audit/1"
                or record.get("journal_format") != FORMAT
                or record.get("run_id") != self.run_id
                or record.get("task_id") != self.task_id
                or record.get("event_id") != f"{self.run_id}:{sequence}"
                or record.get("previous_hash") != expected_previous
                or record.get("record_hash") != expected_hash
                or record.get("record_hash") != digest(
                    {key: value for key, value in record.items() if key != "record_hash"})):
            raise JournalError("indexed journal hash or lineage mismatch")
        try:
            resolved = resolve_journal_blob_refs(record, self.path, blob_cache)
        except (OSError, ValueError) as error:
            raise JournalError("invalid indexed journal blob reference") from error
        if not isinstance(resolved, dict):
            raise JournalError("indexed journal blob did not resolve to an object")
        return resolved

    def _record_range(self, start: int, stop: int) -> Iterator[dict]:
        """Zero-based half-open range: stream evicted prefix, detach cached suffix."""
        index = start
        while index < stop:
            self.assert_owner()
            # A consumer may append under this reentrant lock between yields.
            # Recheck the suffix position so eviction cannot shift the snapshot.
            cache_start = getattr(self, "_record_cache_start", 0)
            if cache_start + len(self._record_cache) != self._sequence:
                yield from self._indexed_records(index + 1, stop)
                return
            if index < cache_start:
                disk_stop = min(stop, cache_start)
                yield from self._indexed_records(index + 1, disk_stop)
                index = disk_stop
            else:
                yield copy.deepcopy(self._record_cache[index - cache_start])
                index += 1

    def _reset_framefit_receipt_projection(self) -> None:
        self._framefit_receipt_records = []
        self._framefit_receipt_history = None
        self._framefit_pending_coverage_session = None
        self._framefit_receipt_cursor = 0
        self._framefit_receipt_overflow = False
        self._framefit_latest_check = None
        self._framefit_last_applied_edit_after_check = None

    def _rebuild_framefit_receipt_projection(
            self, records: Iterable[Mapping[str, object]] | None = None) -> None:
        """Replay the bounded projection after the journal index is authoritative."""
        self._reset_framefit_receipt_projection()
        from kernel_framefit_receipt_history import coverage_session_for

        try:
            coverage = coverage_session_for(self, None)
        except Exception:
            # Audit replay stays readable; typed construction fails closed.
            self._framefit_receipt_overflow = True
            return
        for record in self._record_cache if records is None else records:
            self._advance_framefit_receipt(record, coverage=coverage)
        if (self._framefit_receipt_history is not None
                and coverage is not None and coverage.digest is not None):
            self._framefit_pending_coverage_session = coverage

    def cached_records(self) -> Sequence[dict]:
        """Return a fixed prefix that detaches only records the caller reads."""
        with self._lock:
            self.assert_owner()
            if getattr(self, "_record_cache_limit", None) is None:
                if len(self._record_cache) != self._sequence:
                    for _record in self.records():
                        pass
                return tuple(self._record_cache)
            # The R0 probe supplies an in-memory journal double without a disk
            # index; real journals must prove index coverage before exposing it.
            if hasattr(self, "_record_offsets"):
                ContinuationJournal._ensure_record_index(self)
            return _CachedRecordView(self, self._sequence, getattr(self, "_head", None))

    def cached_records_since(self, sequence: int) -> tuple[dict, ...]:
        """Return validated records after ``sequence``, replaying old cursors."""
        if type(sequence) is not int or sequence < 0:
            raise ValueError("sequence must be a non-negative integer")
        with self._lock:
            self.assert_owner()
            if self._record_cache_limit is None:
                if len(self._record_cache) != self._sequence:
                    for _record in self.records():
                        pass
                return tuple(self._record_cache[sequence:])
            return tuple(self._record_range(sequence, self._sequence))

    @profiled_journal_span("journal_emit")
    def emit(self, kind: str, **fields: object) -> dict[str, object]:
        identity_fire = False
        identity_error: dict[str, str] | None = None
        try:
            from kernel_verdict_identity import prepare_journal_event
            fields, identity_fire = prepare_journal_event(self, kind, fields)
        except ImportError as error:
            # Keep this journal usable as a standalone continuation primitive;
            # the identity organ is an explicit opt-in kernel seam.
            identity_error = {
                "event_schema": "stateframe-harden8-authority-error/1",
                "failure_class": "ADVISORY",
                "boundary": "journal.identity_adapter",
                "disposition": "degrade",
                "error_type": type(error).__name__,
            }
            fields.setdefault("harden8_advisory_failure", identity_error)
        with self._lock:
            self.assert_owner()
            if self._replaying or self._notifying:
                raise JournalError("cannot append while replaying or publishing")
            if kind == "run_started" and self._sequence != 0:
                raise ValueError("run_started is only valid as the first event")
            if not isinstance(kind, str) or not kind or RESERVED.intersection(fields):
                # N1 class: a telemetry write must never kill a run. The r2EY ALL arm died at
                # step 0 on this raise. Record a typed rejection instead and continue; the reserved journal
                # fields are never overwritten by caller data.
                rejected_fields = sorted(str(name)[:64] for name in RESERVED.intersection(fields))
                if isinstance(kind, str) and kind:
                    # Valid kind, reserved caller key(s): keep the event (no lost telemetry) with each reserved key
                    # renamed to field_<key>, and flag it so the cut gate (count must be 0) names the producer.
                    fields = {
                        **{key: value for key, value in fields.items() if key not in RESERVED},
                        **{f"field_{key}": fields[key] for key in rejected_fields},
                        "reserved_fields_renamed": rejected_fields,
                    }
                else:
                    fields = {
                        "producer": "kernel_continuation_journal",
                        "rejected_kind": type(kind).__name__ if not isinstance(kind, str) else "",
                        "reserved_fields": rejected_fields,
                        "dispatch": False,
                    }
                    kind = "journal_rejected_event"
            record = {
                **fields, "schema": "stateframe-audit/1", "kind": kind,
                "run_id": self.run_id, "task_id": self.task_id,
                "event_id": f"{self.run_id}:{self._sequence + 1}",
                "sequence": self._sequence + 1, "timestamp_ns": time.time_ns(),
                "journal_format": FORMAT, "previous_hash": self._head,
            }
            # Freeze nested caller-owned values before hashing/writing.
            encoded_record = canonical_bytes(record)
            record = json.loads(encoded_record)
            with profile_host_span(self, "journal_blob_io"):
                stored_record = externalize_journal_fields(
                    record, self.path, self._blob_cache)
            stored_record["record_hash"] = digest(stored_record)
            record["record_hash"] = stored_record["record_hash"]
            line = canonical_bytes(stored_record) + b"\n"
            if len(line) > MAX_RECORD_BYTES:
                raise ValueError("journal record exceeds limit")
            try:
                view = memoryview(line)
                offset = self._size
                with self.host_spans.measure("append_fsync", sequence=record["sequence"]):
                    while view:
                        count = os.write(self._fd, view)
                        if count <= 0:
                            raise OSError("journal write made no progress")
                        view = view[count:]
                    os.fsync(self._fd)
                self._size += len(line)
                self._sequence, self._head = record["sequence"], record["record_hash"]
                if len(getattr(self, "_record_offsets", ())) == record["sequence"] - 1:
                    self._record_offsets.append(offset)
                    self._record_hashes.append(record["record_hash"])
                if self._record_cache_start + len(self._record_cache) == record["sequence"] - 1:
                    # record_hash is added after the initial canonical encoding.
                    record_bytes = len(encoded_record) + len(canonical_bytes({
                        "record_hash": record["record_hash"]})) - 1
                    self._record_cache.append(copy.deepcopy(record))
                    self._record_cache_sizes.append(record_bytes)
                    self._record_cache_bytes += record_bytes
                    self._trim_record_cache()
                else:
                    # This should only be reachable after an interrupted
                    # validation pass; force the next indexed read to rebuild.
                    self._record_cache = []
                    self._record_cache_start = record["sequence"]
                    self._record_cache_bytes = 0
                    self._record_cache_sizes = []
                # FRAMEFIT5 consumes the validated cache via its existing cursor
                # API when requested; unrelated events need no receipt work.
                self._index_receipt_record(record)
                self._receipt_generation = record["sequence"]
                self._notifying = True
                generated = getattr(self, "generated_artifacts", None)
                if generated is not None:
                    self._notify_isolated(
                        "generated_artifacts", generated.observe, record,
                        on_unobserved=generated.mark_unobserved)
                with self.host_spans.measure("subscriber_fanout", sequence=record["sequence"]):
                    for name, observer in self._subscribers.items():
                        if self._subscriber_cursors.get(name, 0) != record["sequence"] - 1:
                            raise JournalError("subscriber sequence gap")
                        self._subscriber_cursors[name] = record["sequence"]
                        selected = self._subscriber_kinds.get(name)
                        if selected is None or record.get("kind") in selected:
                            self._publication_stats["published"] += 1
                            self._publication_stats["deep_copies"] += 1
                            self._notify_isolated(name, observer, record)
                        else:
                            self._publication_stats["filtered"] += 1
            except BaseException as error:
                self._poison(f"{type(error).__name__} appending {kind!r} after seq "
                             f"{self._sequence}: {str(error)[:160]}")
                raise
            finally:
                self._notifying = False
            self._report_observer_failures()
            if identity_fire:
                from kernel_verdict_identity import journal_identity_fired
                journal_identity_fired(self, kind)
            return record

    def _poison(self, cause: str) -> None:
        """Poison the writer and remember WHY (first cause wins; every later assert quotes it)."""
        self._poisoned = True
        if not getattr(self, "_poison_cause", None):
            self._poison_cause = cause

    def _notify_isolated(self, name: str, observer, record: Mapping[str, object],
                         *, on_unobserved=None) -> None:
        """Deliver an already-DURABLE record to one observer; its failure cannot poison the writer.

        10-08: fanout ran inside the append's poison handler, so any observer exception (including
        an observer that itself calls emit() and hits "cannot append while publishing") poisoned a
        healthy fsynced journal and every later emit died at assert_owner with no cause. The
        failure is journaled as journal_observer_failed right after the append instead.
        """
        try:
            observer(copy.deepcopy(record))
        except Exception as error:  # noqa: BLE001 - isolated, then journaled
            self._publication_stats["observer_failures"] = (
                self._publication_stats.get("observer_failures", 0) + 1)
            failure = {
                "subscriber": str(name), "event_kind": str(record.get("kind")),
                "event_sequence": record.get("sequence"), "error_type": type(error).__name__,
                "error": str(error)[:200],
            }
            self._observer_failures.append(failure)
            if callable(on_unobserved):
                try:
                    on_unobserved(
                        record, error_type=failure["error_type"], error=failure["error"])
                except Exception as marker_error:  # noqa: BLE001 - preserve the original failure
                    self._observer_failures.append({
                        "subscriber": f"{name}_unobserved_marker",
                        "event_kind": str(record.get("kind")),
                        "event_sequence": record.get("sequence"),
                        "error_type": type(marker_error).__name__,
                        "error": str(marker_error)[:200],
                    })

    def _report_observer_failures(self) -> None:
        if not self._observer_failures or self._reporting_observer_failures:
            return
        failures, self._observer_failures = self._observer_failures, []
        self._reporting_observer_failures = True
        try:
            for failure in failures[:20]:
                self.emit("journal_observer_failed", **failure)
        finally:
            self._reporting_observer_failures = False

    def host_span_snapshot(self) -> tuple[dict[str, object], ...]:
        return self.host_spans.snapshot()

    def emit_batch(self, events: Sequence[tuple[str, Mapping[str, object]]]) -> list[dict[str, object]]:
        """Append a paired durable batch under one writer lease.

        The journal remains the serialization boundary: observers cannot see an
        interleaved assignment/replay pair from another writer.  Each record is
        still fsynced by ``emit`` so a torn tail is rejected on resume.
        """
        if not isinstance(events, Sequence) or isinstance(events, (str, bytes, bytearray)):
            raise JournalError("journal batch must be a sequence")
        normalized = []
        for event in events:
            if (not isinstance(event, tuple) or len(event) != 2
                    or not isinstance(event[0], str)
                    or not isinstance(event[1], Mapping)):
                raise JournalError("journal batch event is malformed")
            normalized.append((event[0], dict(event[1])))
        with self._lock:
            self.assert_owner()
            return [self.emit(kind, **fields) for kind, fields in normalized]

    def close(self) -> None:
        with self._lock:
            if os.getpid() == self._pid:
                try:
                    from kernel_host_cache import flush_journal_cache_summaries

                    flush_journal_cache_summaries(self)
                except Exception:
                    # Request cache telemetry cannot prevent journal closure.
                    pass
            try:
                if self._cold_start_root is not None:
                    self._finish_cold_start_host_spans()
                if self._fd is not None:
                    # Closing an inherited fd must not explicitly unlock the parent's lease.
                    os.close(self._fd)
                    self._fd = None
            finally:
                token = self._outline_capability_listener_token
                self._outline_capability_listener_token = None
                if token is not None:
                    from mc_repo_intel_vendor.ts_dispatch import reset_outline_capability_listener

                    reset_outline_capability_listener(token)

    def __enter__(self) -> ContinuationJournal:
        self.assert_owner()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
