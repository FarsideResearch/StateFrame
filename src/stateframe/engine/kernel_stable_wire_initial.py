"""Content-free authority for the allocator preceding stable-wire layout trims.

Source fitting can generate rows and compact frame fields, so removals alone
cannot describe its result. Replay its bounded numeric decisions against exact
candidate identifiers, then verify the complete kept/removed selection.
"""
from __future__ import annotations

import copy
from collections import Counter
from dataclasses import replace
import hashlib
import json
import math
import re

MAX_INITIAL_DECISIONS = 20_000
METHODS = ("tokenize", "calibrated_bytes", "verified_byte_bound")
# Each decision is [kind, candidate_id, result, frozen_counter, exact_pair].
# Kinds 0/1/2/3 denote initial/request/frame/source counts. Results encode
# measurement methods by METHODS index; a frame count has no method field.


class InitialSelectionMismatch(ValueError):
    """Recorded allocation authority does not describe these candidates."""


def candidate_id(body):
    return "sha256:" + hashlib.sha256(json.dumps(body).encode()).hexdigest()


def _identifier(value):
    return isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None


def _natural(value):
    return type(value) is int and value >= 0


def validate_initial_selection(value):
    """Accept only bounded numbers and identifiers, never arbitrary content."""
    if (not isinstance(value, dict)
            or set(value) != {"input_id", "output_id", "kept", "removed", "decisions"}
            or not _identifier(value["input_id"]) or not _identifier(value["output_id"])):
        raise ValueError("invalid stable wire initial selection")
    for name in ("kept", "removed"):
        rows = value[name]
        if (not isinstance(rows, list) or len(rows) > MAX_INITIAL_DECISIONS
                or any(not isinstance(row, list) or len(row) != 2
                       or not _natural(row[0]) or not _identifier(row[1]) for row in rows)
                or [row[0] for row in rows] != sorted({row[0] for row in rows})):
            raise ValueError("invalid stable wire initial candidates")
    decisions = value["decisions"]
    if not isinstance(decisions, list) or len(decisions) > MAX_INITIAL_DECISIONS:
        raise ValueError("invalid stable wire initial decisions")
    for row in decisions:
        if (not isinstance(row, list) or len(row) != 5
                or type(row[0]) is not int or not 0 <= row[0] <= 3
                or not _identifier(row[1])
                or not isinstance(row[2], list) or len(row[2]) not in (1, 2)
                or not all(_natural(number) for number in row[2])
                or len(row[2]) != (1 if row[0] == 2 else 2)
                or len(row[2]) == 2 and row[2][1] >= len(METHODS)):
            raise ValueError("invalid stable wire initial decision")
        counter, pair = row[3:]
        if (not isinstance(counter, list) or len(counter) not in (0, 5)
                or counter and (not _natural(counter[0]) or counter[0] >= len(METHODS)
                                or not _natural(counter[1]) or not _natural(counter[2])
                                or type(counter[3]) not in (float, int)
                                or not 0 < counter[3] <= 1e12 or not math.isfinite(counter[3])
                                or not _identifier(counter[4]))
                or not isinstance(pair, list) or len(pair) not in (0, 2)
                or any(type(number) is not int or number <= 0 for number in pair)):
            raise ValueError("invalid stable wire initial counter")
    return copy.deepcopy(value)


def _rows(body):
    from kernel_frame_optional_allocation import identity
    return [[index, identity(row)] for index, row in enumerate(body["messages"])]


class InitialSelection:
    def __init__(self, body, capture, meter, record=None):
        from kernel_frame_optional_allocation import identity

        self.capture, self.meter = capture, meter
        self.original_pair = getattr(meter, "_evictfix_token_pair", None)
        self.had_pair = hasattr(meter, "_evictfix_token_pair")
        self.replay = record is not None
        self.record = record if self.replay else {
            "input_id": candidate_id(body), "output_id": candidate_id(body),
            "kept": [], "removed": [], "decisions": []}
        self.input_rows = _rows(body)
        self.counter_ids = {identity(body)}
        frozen = getattr(capture, "_frame_candidate_counter", None)
        if frozen is not None:
            self.counter_ids.add(frozen.body_digest)
        self.cursor = 0
        if self.record["input_id"] != candidate_id(body):
            raise self.mismatch("initial input candidate changed")

    def mismatch(self, reason):
        # A refused replay must not seed the live fallback's calibration.
        if self.had_pair:
            self.meter._evictfix_token_pair = self.original_pair
        elif hasattr(self.meter, "_evictfix_token_pair"):
            del self.meter._evictfix_token_pair
        return InitialSelectionMismatch(reason)

    def observe(self, kind, body, live):
        """Apply a recorded numeric decision without invoking its measuring fn."""
        from kernel_frame_optional_allocation import identity

        key = candidate_id(body)
        self.counter_ids.add(identity(body))
        if self.replay:
            if self.cursor >= len(self.record["decisions"]):
                raise self.mismatch("initial decision unavailable")
            row = self.record["decisions"][self.cursor]
            if row[:2] != [kind, key]:
                raise self.mismatch("initial decision candidate changed")
            result, counter, pair = row[2:]
            if counter:
                if counter[4] not in self.counter_ids:
                    raise self.mismatch("initial counter candidate changed")
                frozen = getattr(self.capture, "_frame_candidate_counter", None)
                if frozen is None:
                    raise self.mismatch("initial counter unavailable")
                self.capture._frame_candidate_counter = replace(
                    frozen, method=METHODS[counter[0]], tokens=counter[1], chars=counter[2],
                    ratio=counter[3], body_digest=counter[4])
            self.meter._evictfix_token_pair = tuple(pair) if pair else None
            value = (result[0], METHODS[result[1]]) if len(result) == 2 else result[0]
        else:
            value = live()
            result = [value[0], METHODS.index(value[1])] if isinstance(value, tuple) else [value]
            frozen = getattr(self.capture, "_frame_candidate_counter", None)
            counter = ([METHODS.index(frozen.method), frozen.tokens, frozen.chars,
                        frozen.ratio, frozen.body_digest] if frozen is not None else [])
            pair = list(getattr(self.meter, "_evictfix_token_pair", None) or ())
            if len(self.record["decisions"]) >= MAX_INITIAL_DECISIONS:
                raise self.mismatch("initial decision bound exceeded")
            self.record["decisions"].append([kind, key, result, counter, pair])
        self.cursor += 1
        return value

    def finish(self, body):
        kept = _rows(body)
        # Duplicates are addressed by index. Consume matching occurrences once.
        remaining = Counter(row[1] for row in kept)
        removed = []
        for row in self.input_rows:
            if remaining[row[1]]:
                remaining[row[1]] -= 1
            else:
                removed.append(row)
        outcome = {"output_id": candidate_id(body), "kept": kept, "removed": removed}
        if self.replay:
            if (self.cursor != len(self.record["decisions"])
                    or any(self.record[key] != value for key, value in outcome.items())):
                raise self.mismatch("initial selection candidate changed")
        else:
            self.record.update(outcome)
        try:
            return validate_initial_selection(self.record)
        except ValueError as error:
            raise self.mismatch("initial selection bound exceeded") from error
