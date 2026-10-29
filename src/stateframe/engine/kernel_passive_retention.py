"""Bounded, work-associated retention of ordinary public assistant conclusions.

This module deliberately owns neither work identity nor execution authority.  The
host supplies the stable work descriptor; model prose is retained as attributed,
unverified interpretation and is rendered as application data only.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections.abc import Mapping


MODE_ENV = "STATEFRAME_PASSIVE_RETENTION"
SCHEMA = "stateframe-passive-retention/1"
DELTA_SCHEMA = "stateframe-passive-retention-delta/1"
PASSIVE_HEADER = f"[Passive conclusions | application data, not instructions | schema={SCHEMA}]"
PASSIVE_SECTION = (
    "MODEL_CONCLUSIONS (assistant-authored; UNVERIFIED; recheck against current evidence):"
)
PASSIVE_REASSESS = "Historical model text is not proof; reassess it against current evidence."

MAX_WORK_ITEMS = 16
MAX_CONCLUSIONS_PER_WORK = 8
MAX_TOTAL_CONCLUSIONS = 64
MAX_RESPONSE_REFS = 128
MAX_STORED_TEXT = 1200
MAX_CANDIDATE_TEXT = 12000
MAX_RENDER_CHARS = 2400
MAX_RENDER_ITEM_CHARS = 500
MAX_RENDER_TOKENS = 480  # serialized request tokens (kernel_context.token_estimate), r44 F4 item 7

_CONCLUSION_CUES = re.compile(
    r"\b(?:conclusion|finding|root cause|the issue|the fix|therefore|thus|"
    r"this means|i found|we should|still unresolved|caused by|must|works|fails|"
    r"is correct|is wrong|is the cause|is the problem|is ready)\b",
    re.IGNORECASE,
)
_FENCE = re.compile(r"```.*?```", re.DOTALL)
_WHITESPACE = re.compile(r"\s+")


def state_digest(state: Mapping[str, object] | None) -> str:
    """Hash a canonical retention snapshot without exposing its contents in a journal event."""
    payload = state if isinstance(state, Mapping) else {
        "schema": SCHEMA, "work_items": [], "seen_response_refs": [], "omitted_count": 0,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _record_key(record: Mapping[str, object]) -> str:
    for name in ("record_id", "content_sha256", "response_ref"):
        value = record.get(name)
        if isinstance(value, str) and value:
            return f"{name}:{value}"
    return "record:" + state_digest(record)


def snapshot_delta(
    previous: Mapping[str, object] | None,
    current: Mapping[str, object],
) -> dict[str, object]:
    """Return only retention rows changed since ``previous`` plus an integrity digest."""
    if not isinstance(current, Mapping) or current.get("schema") != SCHEMA:
        raise ValueError("passive retention delta requires a current snapshot")
    previous = previous if isinstance(previous, Mapping) and previous.get("schema") == SCHEMA else None
    previous_rows = {
        str(row.get("work_id")): row
        for row in (previous.get("work_items", []) if previous else [])
        if isinstance(row, Mapping) and isinstance(row.get("work_id"), str)
    }
    changed_rows: list[dict[str, object]] = []
    current_rows = current.get("work_items", [])
    if not isinstance(current_rows, list):
        raise ValueError("passive retention snapshot has malformed work items")
    for value in current_rows:
        if not isinstance(value, Mapping):
            continue
        work_id = value.get("work_id")
        if not isinstance(work_id, str):
            continue
        old = previous_rows.get(work_id)
        old_records = {
            _record_key(record): record
            for record in (old.get("conclusions", []) if isinstance(old, Mapping) else [])
            if isinstance(record, Mapping)
        }
        conclusions = value.get("conclusions", [])
        if not isinstance(conclusions, list):
            raise ValueError("passive retention snapshot has malformed conclusions")
        added = [
            copy.deepcopy(record) for record in conclusions
            if isinstance(record, Mapping)
            and old_records.get(_record_key(record)) != record
        ]
        metadata = {
            name: copy.deepcopy(value.get(name, default))
            for name, default in (("work_id", work_id), ("label", ""), ("focus", ""), ("binding", {}))
        }
        old_metadata = {
            name: old.get(name, default) for name, default in metadata.items()
        } if isinstance(old, Mapping) else None
        if old is None or metadata != old_metadata or added:
            metadata["conclusions"] = added
            changed_rows.append(metadata)
    current_ids = {
        str(row.get("work_id")) for row in current_rows
        if isinstance(row, Mapping) and isinstance(row.get("work_id"), str)
    }
    previous_refs = set(previous.get("seen_response_refs", []) if previous else ())
    refs = [
        str(ref) for ref in current.get("seen_response_refs", [])
        if isinstance(ref, str) and ref not in previous_refs
    ]
    previous_omitted = int(previous.get("omitted_count", 0) or 0) if previous else 0
    current_omitted = int(current.get("omitted_count", 0) or 0)
    return {
        "schema": DELTA_SCHEMA,
        "base_digest": state_digest(previous) if previous else None,
        "state_digest": state_digest(current),
        "work_items": changed_rows,
        "removed_work_ids": sorted(set(previous_rows) - current_ids),
        "seen_response_refs": refs,
        "omitted_count_delta": max(0, current_omitted - previous_omitted),
    }


def apply_snapshot_delta(
    previous: Mapping[str, object] | None,
    delta: Mapping[str, object],
) -> dict[str, object]:
    """Apply one compact retention delta and verify its resulting snapshot digest."""
    if not isinstance(delta, Mapping) or delta.get("schema") != DELTA_SCHEMA:
        raise ValueError("unsupported passive retention delta schema")
    if previous is not None and delta.get("base_digest") != state_digest(previous):
        raise ValueError("passive retention delta base digest mismatch")
    state = copy.deepcopy(dict(previous)) if isinstance(previous, Mapping) else {
        "schema": SCHEMA, "work_items": [], "seen_response_refs": [], "omitted_count": 0,
    }
    rows = state.get("work_items")
    if not isinstance(rows, list):
        raise ValueError("passive retention state has malformed work items")
    for work_id in delta.get("removed_work_ids", []):
        rows[:] = [row for row in rows if not isinstance(row, Mapping) or row.get("work_id") != work_id]
    for value in delta.get("work_items", []):
        if not isinstance(value, Mapping) or not isinstance(value.get("work_id"), str):
            raise ValueError("passive retention delta has malformed work item")
        current = next((row for row in rows if isinstance(row, Mapping)
                        and row.get("work_id") == value["work_id"]), None)
        if current is None:
            current = {"work_id": value["work_id"], "label": "", "focus": "", "binding": {}, "conclusions": []}
            rows.append(current)
        for name in ("label", "focus", "binding"):
            if name in value:
                current[name] = copy.deepcopy(value[name])
        records = current.setdefault("conclusions", [])
        if not isinstance(records, list):
            raise ValueError("passive retention state has malformed conclusions")
        by_key = {_record_key(record): index for index, record in enumerate(records)
                  if isinstance(record, Mapping)}
        for record in value.get("conclusions", []):
            if not isinstance(record, Mapping):
                raise ValueError("passive retention delta has malformed conclusion")
            key = _record_key(record)
            if key in by_key:
                records[by_key[key]] = copy.deepcopy(dict(record))
            else:
                by_key[key] = len(records)
                records.append(copy.deepcopy(dict(record)))
    refs = state.setdefault("seen_response_refs", [])
    if not isinstance(refs, list):
        raise ValueError("passive retention state has malformed response refs")
    for ref in delta.get("seen_response_refs", []):
        if isinstance(ref, str) and ref not in refs:
            refs.append(ref)
    state["omitted_count"] = int(state.get("omitted_count", 0) or 0) + int(
        delta.get("omitted_count_delta", 0) or 0
    )
    if delta.get("state_digest") != state_digest(state):
        raise ValueError("passive retention delta result digest mismatch")
    return state


def mode() -> str:
    """Return the strict opt-in switch value."""
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be 'off' or 'on'")
    return value


def enabled() -> bool:
    return mode() == "on"


def _text(value: object, *, limit: int = 240) -> str:
    if not isinstance(value, str):
        return ""
    return _WHITESPACE.sub(" ", value.replace("\x00", " ")).strip()[:limit]


_QUALIFIER_LEAD = re.compile(
    r"^(?:however|but|except|unless|only|note|caveat|although|though|this applies|"
    r"that said|with the exception|still)\b",
    re.IGNORECASE,
)


def _passages(content: str) -> list[str]:
    """Split public prose into whole authored passages (paragraphs/bullets).

    A passage is never split at sentence boundaries: a qualification written in
    the same passage ("... However, only ASCII is fixed.") must travel with the
    conclusion it limits.  A following passage that opens with an explicit
    qualifier is attached to the preceding passage for the same reason.
    """
    # Markdown code is not an assistant conclusion, and extracting it would make
    # an implementation detail look like a durable interpretation.
    without_code = _FENCE.sub(" ", content)
    passages: list[str] = []
    for paragraph in re.split(r"\n\s*\n|\n(?=\s*[-*•])", without_code):
        paragraph = paragraph.strip().lstrip("-*• ").strip()
        normalized = _WHITESPACE.sub(" ", paragraph.replace("\x00", " ")).strip()
        if not normalized:
            continue
        if passages and _QUALIFIER_LEAD.match(normalized):
            passages[-1] = f"{passages[-1]} {normalized}"
            continue
        passages.append(normalized)
    return passages


def _public_candidates(content: str) -> list[str]:
    candidates: list[str] = []
    for passage in _passages(content):
        # Oversized passages are kept whole here; storage replaces them with an
        # intact-passage reference instead of clipping away qualifications.
        if len(passage) < 16 or passage.endswith((":", "?")):
            continue
        if passage not in candidates:
            candidates.append(passage)
    if not candidates:
        return []
    cued = [candidate for candidate in candidates if _CONCLUSION_CUES.search(candidate)]
    if cued:
        # First two explicit conclusion passages in response order, each whole.
        return cued[:2]
    # Ordinary prose often states its conclusion without a marker.  The final
    # complete passage is the least speculative compact fallback.
    return [candidates[-1]]


def extract_conclusions(content: object) -> tuple[str, ...]:
    """Extract at most two bounded public conclusion passages from one response."""
    if not isinstance(content, str) or not content.strip():
        return ()
    return tuple(_public_candidates(content)[:2])


def _lookup(value: object, key: str) -> object:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _first_text(*values: object, limit: int = 240) -> str:
    for value in values:
        text = _text(value, limit=limit)
        if text:
            return text
    return ""


def _carry_work_id(source: object) -> str:
    carry = _lookup(source, "work_carry")
    return _text(_lookup(carry, "work_id"), limit=128) if carry is not None else ""


def resolve_host_work_id(owner: object, frame: Mapping[str, object] | None = None,
                         active_work_id: object = None) -> str:
    """Resolve work identity with the SAME precedence as the work_carry lane.

    Contract (lanes/work_carry/src/kernel_stateframe_runtime.py
    ``_work_carry_config``): ``task_row.work_carry`` wins, then
    ``frame.work_carry``, then flat task-row keys, then ``owner.task_id``.  An
    explicit host ``active_work_id`` still outranks all of them.  Falling back
    to ``task_id`` while a nested carry names a different work item would
    replay one work item's conclusions into another (cross-work recall).
    """
    row = _lookup(owner, "task_row")
    candidates: list[object] = [active_work_id, _carry_work_id(row), _carry_work_id(frame)]
    for source in (row, owner, frame):
        if source is None:
            continue
        for key in ("active_work_id", "work_id", "work_item_id", "objective_id", "objective_ref"):
            candidates.append(_lookup(source, key))
    if isinstance(frame, Mapping):
        decision_state = frame.get("decision_state")
        if isinstance(decision_state, Mapping):
            public = decision_state.get("public")
            if isinstance(public, Mapping):
                for key in ("active_work_id", "work_id", "active_item_id"):
                    candidates.append(public.get(key))
    candidates.append(_lookup(owner, "task_id"))
    for candidate in candidates:
        value = _text(candidate, limit=128)
        if value:
            return value
    return "unassigned-work"


_work_id = resolve_host_work_id


def work_context(owner: object, *, frame: Mapping[str, object] | None = None,
                 path: object = "", binding: Mapping[str, object] | None = None,
                 active_work_id: object = None) -> dict[str, object]:
    """Build a host-owned stable work descriptor for capture and rendering."""
    row = _lookup(owner, "task_row")
    work_id = _work_id(owner, frame, active_work_id)
    label = _first_text(
        _lookup(row, "objective"), _lookup(row, "brief"), _lookup(row, "title"),
        _lookup(row, "task"), _lookup(owner, "fn"), limit=240,
    )
    focus = _first_text(path, _lookup(row, "path"), _lookup(owner, "fn"), limit=180)
    safe_binding = copy.deepcopy(dict(binding)) if isinstance(binding, Mapping) else {}
    return {"work_id": work_id, "label": label, "focus": focus, "binding": safe_binding}


def _checked_work(work: Mapping[str, object]) -> dict[str, object]:
    work_id = _text(work.get("work_id"), limit=128)
    if not work_id:
        raise ValueError("passive retention requires a stable work_id")
    binding = work.get("binding")
    return {
        "work_id": work_id,
        "label": _text(work.get("label"), limit=240),
        "focus": _text(work.get("focus"), limit=180),
        "binding": copy.deepcopy(dict(binding)) if isinstance(binding, Mapping) else {},
    }


def _capture_value(capture: object, name: str) -> object:
    if isinstance(capture, Mapping):
        return capture.get(name)
    return getattr(capture, name, None)


def _short_reference(value: str) -> str:
    return value if len(value) <= 48 else "…" + value[-47:]


class PassiveRetentionStore:
    """A bounded journal-backed collection partitioned by host work identity."""

    def __init__(self, state: Mapping[str, object] | None = None):
        self._work_items: list[dict[str, object]] = []
        self._seen_response_refs: list[str] = []
        self._omitted_count = 0
        self.last_render_stats: dict[str, int] = {}
        if state is not None:
            self._restore(state)

    @classmethod
    def from_state(cls, state: Mapping[str, object]) -> "PassiveRetentionStore":
        return cls(state)

    def _restore(self, state: Mapping[str, object]) -> None:
        if not isinstance(state, Mapping) or state.get("schema") != SCHEMA:
            raise ValueError("unsupported passive retention schema")
        rows = state.get("work_items", [])
        refs = state.get("seen_response_refs", [])
        if not isinstance(rows, list) or not isinstance(refs, list):
            raise ValueError("malformed passive retention state")
        if len(rows) > MAX_WORK_ITEMS or len(refs) > MAX_RESPONSE_REFS:
            raise ValueError("passive retention state exceeds bounds")
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("malformed passive retention work row")
            work = _checked_work(row)
            conclusions = row.get("conclusions", [])
            if not isinstance(conclusions, list) or len(conclusions) > MAX_CONCLUSIONS_PER_WORK:
                raise ValueError("malformed passive retention conclusions")
            checked: list[dict[str, object]] = []
            for record in conclusions:
                if not isinstance(record, Mapping):
                    raise ValueError("malformed passive retention conclusion")
                checked.append(self._checked_record(record, work["work_id"]))
            self._work_items.append({**work, "conclusions": checked})
        self._seen_response_refs = [
            ref for ref in refs if isinstance(ref, str) and ref
        ][:MAX_RESPONSE_REFS]
        self._omitted_count = max(0, int(state.get("omitted_count", 0) or 0))
        if sum(len(row["conclusions"]) for row in self._work_items) > MAX_TOTAL_CONCLUSIONS:
            raise ValueError("passive retention state exceeds total conclusion bound")

    @staticmethod
    def _checked_record(record: Mapping[str, object], work_id: object) -> dict[str, object]:
        response_ref = _text(record.get("response_ref"), limit=240)
        request_id = _text(record.get("request_id"), limit=240)
        text = record.get("text")
        if not response_ref or not request_id or not isinstance(text, str):
            raise ValueError("malformed passive retention identity")
        if record.get("authority") != "MODEL_INTERPRETATION" or record.get("status") != "UNVERIFIED":
            raise ValueError("passive retention must remain unverified model interpretation")
        step = record.get("source_step")
        if type(step) is not int or step < 0:
            raise ValueError("malformed passive retention source step")
        origin_binding = record.get("origin_binding")
        if not isinstance(origin_binding, Mapping):
            raise ValueError("malformed passive retention binding")
        checked = {
            "record_id": _text(record.get("record_id"), limit=360),
            "response_ref": response_ref,
            "request_id": request_id,
            "source_step": step,
            "work_id": _text(work_id, limit=128),
            "authority": "MODEL_INTERPRETATION",
            "status": "UNVERIFIED",
            "text": text,
            "origin_binding": copy.deepcopy(dict(origin_binding)),
            "content_sha256": _text(record.get("content_sha256"), limit=128),
            "omitted": bool(record.get("omitted", False)),
            "original_characters": int(record.get("original_characters", len(text))),
        }
        if len(checked["text"]) > MAX_STORED_TEXT:
            raise ValueError("passive retention text exceeds bound")
        return checked

    def _find_work(self, work_id: str) -> dict[str, object] | None:
        return next((row for row in self._work_items if row["work_id"] == work_id), None)

    @staticmethod
    def _storage_record(work: Mapping[str, object], capture: object,
                        text: str, index: int) -> dict[str, object]:
        response_ref = _text(_capture_value(capture, "response_event_id"), limit=240)
        request_id = _text(_capture_value(capture, "request_id"), limit=240)
        source_step = _capture_value(capture, "step")
        if type(source_step) is not int or source_step < 0:
            raise ValueError("passive retention requires a valid source step")
        original_characters = len(text)
        omitted = original_characters > MAX_STORED_TEXT
        stored_text = (
            "[Intact passage omitted: stored conclusion exceeds the passive retention bound.]"
            if omitted else text
        )
        return {
            "record_id": f"{work['work_id']}:{response_ref}:{index}",
            "response_ref": response_ref,
            "request_id": request_id,
            "source_step": source_step,
            "work_id": work["work_id"],
            "authority": "MODEL_INTERPRETATION",
            "status": "UNVERIFIED",
            "text": stored_text,
            "origin_binding": copy.deepcopy(work.get("binding", {})),
            "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "omitted": omitted,
            "original_characters": original_characters,
        }

    def ingest_capture(self, capture: object, work: Mapping[str, object]) -> dict[str, object]:
        """Ingest one authenticated public response exactly once."""
        checked_work = _checked_work(work)
        response_ref = _text(_capture_value(capture, "response_event_id"), limit=240)
        request_id = _text(_capture_value(capture, "request_id"), limit=240)
        source_step = _capture_value(capture, "step")
        content = _capture_value(capture, "response_content")
        if not response_ref or not request_id or type(source_step) is not int or source_step < 0:
            return {"accepted": False, "added": 0, "reason": "missing_response_identity"}
        if response_ref in self._seen_response_refs:
            return {"accepted": False, "added": 0, "reason": "duplicate_response"}
        self._seen_response_refs.append(response_ref)
        self._seen_response_refs = self._seen_response_refs[-MAX_RESPONSE_REFS:]
        candidates = extract_conclusions(content)
        if not candidates:
            return {"accepted": True, "added": 0, "reason": "no_public_conclusion"}
        row = self._find_work(checked_work["work_id"])
        if row is None:
            if len(self._work_items) >= MAX_WORK_ITEMS:
                self._omitted_count += len(candidates)
                return {"accepted": True, "added": 0, "omitted": len(candidates),
                        "reason": "work_item_bound"}
            row = {**checked_work, "conclusions": []}
            self._work_items.append(row)
        else:
            for key in ("label", "focus", "binding"):
                if checked_work.get(key):
                    row[key] = copy.deepcopy(checked_work[key])
        existing_hashes = {record["content_sha256"] for record in row["conclusions"]}
        added = 0
        omitted = 0
        for index, text in enumerate(candidates):
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if content_hash in existing_hashes:
                continue
            total = sum(len(item["conclusions"]) for item in self._work_items)
            if len(row["conclusions"]) >= MAX_CONCLUSIONS_PER_WORK or total >= MAX_TOTAL_CONCLUSIONS:
                omitted += 1
                continue
            record = self._storage_record(checked_work, capture, text, index)
            row["conclusions"].append(record)
            existing_hashes.add(content_hash)
            added += 1
        self._omitted_count += omitted
        return {"accepted": True, "added": added, "omitted": omitted,
                "reason": "stored" if added else "already_present_or_bounded"}

    def snapshot(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "work_items": copy.deepcopy(self._work_items),
            "seen_response_refs": list(self._seen_response_refs),
            "omitted_count": self._omitted_count,
        }

    @staticmethod
    def _render_text(record: Mapping[str, object]) -> str:
        if record.get("omitted") or len(str(record.get("text", ""))) > MAX_RENDER_ITEM_CHARS:
            return (
                f"[Intact passage omitted: {record.get('original_characters', '?')} chars; "
                f"response={_short_reference(str(record.get('response_ref', 'unknown')))}.]"
            )
        return str(record.get("text", ""))

    def render_message(self, work: Mapping[str, object], *,
                       known_claim_ids: frozenset[str] | set[str] = frozenset(),
                       known_refs: frozenset[str] | set[str] = frozenset(),
                       ) -> dict[str, str] | None:
        """Render the bounded passive block for one work item.

        r44 F4 fixes (all behind this lane's switch):
        R1 -- a record whose exact claim identity (``claim_id``) or response/record
        id is already carried by another rendered host block (decision handoff,
        exchange ring, settled_state) is skipped instead of repeated as UNVERIFIED
        prose; ``last_render_stats['skipped_duplicate']`` reports the count.
        R3 -- a record whose origin binding differs from the current binding is
        rendered reference-only (id + basis=changed), never with its stale body,
        matching conclusion_cache's STALE/UNKNOWN withholding policy.
        Item 7 -- the render bound is measured in serialized request tokens
        (``kernel_context.token_estimate``), not whitespace words.
        """
        from kernel_context import token_estimate

        self.last_render_stats = {"skipped_duplicate": 0, "stale_reference_only": 0,
                                  "rendered": 0, "omitted": 0}
        checked_work = _checked_work(work)
        row = self._find_work(checked_work["work_id"])
        if row is None or not row["conclusions"]:
            return None
        known_ids = set(known_claim_ids or ())
        refs = set(known_refs or ())
        records = []
        for record in row["conclusions"]:
            if (record.get("response_ref") in refs or record.get("record_id") in refs
                    or record_is_known(record, known_ids)):
                self.last_render_stats["skipped_duplicate"] += 1
                continue
            records.append(record)
        if not records:
            return None
        lines = [PASSIVE_HEADER, f"Work: {checked_work['work_id']}"]
        if checked_work.get("label"):
            lines.append(f"Focus: {checked_work['label']}")
        if checked_work.get("focus"):
            lines.append(f"Path: {checked_work['focus']}")
        lines.extend((PASSIVE_SECTION, PASSIVE_REASSESS))
        def fits(candidate_lines: list[str]) -> bool:
            # Item 7: bounded in serialized request tokens, not whitespace words.
            text = "\n".join(candidate_lines)
            return (len(text) <= MAX_RENDER_CHARS
                    and token_estimate({"role": "user", "content": text}) <= MAX_RENDER_TOKENS)

        current_binding = checked_work.get("binding", {})
        remaining = len(records)
        rendered = 0
        for record in records:
            same = record.get("origin_binding", {}) == current_binding
            basis = "same" if same else "changed"
            if same:
                body = self._render_text(record)
            else:
                body = (f"[reference only: recorded against an earlier binding; body withheld, "
                        f"id={str(record.get('content_sha256', ''))[:12] or 'unknown'}]")
                self.last_render_stats["stale_reference_only"] += 1
            line = (
                f"- step {record['source_step']}, response={_short_reference(record['response_ref'])}, "
                f"basis={basis}: {body}"
            )
            if not fits([*lines, line]):
                break
            lines.append(line)
            rendered += 1
        omitted = remaining - rendered
        if omitted:
            # The omission marker is inside the same serialized budget: drop the
            # newest rendered line(s) until block + marker fit.
            while True:
                marker = (f"- {remaining - rendered} retained conclusion(s) omitted from this "
                          "compact view; recall is bounded.")
                if fits([*lines, marker]) or not rendered:
                    break
                lines.pop()
                rendered -= 1
            omitted = remaining - rendered
            if rendered and fits([*lines, marker]):
                lines.append(marker)
        self.last_render_stats["rendered"] = rendered
        self.last_render_stats["omitted"] = omitted
        if not rendered:
            return None
        content = "\n".join(lines)
        from kernel_prompt_compact import compact_passive_text, enabled as prompt_compact_enabled
        if prompt_compact_enabled():
            # r47 F4/L2: short response ids + one withheld-body note (fit bound above was
            # measured on the longer full form, so the compact block always fits too).
            content = compact_passive_text(content)
            if not content:
                return None  # every row withheld and dropped: no block
        return {"role": "user", "content": content}


_CLAIM_LABEL = re.compile(
    r"^(?:conclusion|finding|root cause|hypothesis|the fix|the issue|result|summary)\s*:\s*",
    re.IGNORECASE,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+")


def claim_id(text: object) -> str:
    """Exact identity of one claim: whitespace/case-normalised, one leading cue
    label and trailing punctuation removed, then hashed.  Equality only -- this
    is an identity, never a similarity score."""
    if not isinstance(text, str):
        return ""
    value = _WHITESPACE.sub(" ", text.replace("\x00", " ")).strip()
    value = _CLAIM_LABEL.sub("", value).strip().rstrip(".!;:, ").casefold()
    if len(value) < 8:
        return ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _record_claim_ids(text: str) -> tuple[str, set[str]]:
    whole = claim_id(text)
    parts = {claim_id(part) for part in _SENTENCE_SPLIT.split(text) if part.strip()}
    parts.discard("")
    return whole, parts


def record_is_known(record: Mapping[str, object], known_ids: set[str]) -> bool:
    """True when the record's claim is already carried verbatim by a host block:
    the whole passage's id, or every sentence id of the passage, is known."""
    if not known_ids or record.get("omitted"):
        return False
    whole, parts = _record_claim_ids(str(record.get("text", "")))
    if whole and whole in known_ids:
        return True
    return bool(parts) and parts <= known_ids


def harvest_claim_ids(*values: object, limit: int = 4096) -> tuple[set[str], set[str]]:
    """Collect exact claim ids and literal refs from host-owned state snapshots
    (decision continuity slots/ring, decision-cache records)."""
    ids: set[str] = set()
    refs: set[str] = set()
    stack = list(values)
    seen = 0
    while stack and seen < limit:
        value = stack.pop()
        seen += 1
        if isinstance(value, str):
            text = value.strip()
            if not text:
                continue
            refs.add(text)
            for part in (text, *_SENTENCE_SPLIT.split(text)):
                identity = claim_id(part)
                if identity:
                    ids.add(identity)
        elif isinstance(value, Mapping):
            stack.extend(value.values())
        elif isinstance(value, (list, tuple, set, frozenset)):
            stack.extend(value)
    return ids, refs
