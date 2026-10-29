"""request assignment paired experiment protocol and opt-in request assignment block.

The protocol is deliberately a batch tool, not a run launcher. It produces
pre-assigned ON/OFF blocks, reuses one compatible OFF block for many
candidates, and evaluates synthetic or recorded outcomes at fixed looks. The
runtime-facing part is a small, independently switchable renderer used to
prove that the assignment which the planner produced reaches ``prepare()``'s
provider-bound request.

No third-party statistics package is required. The paired p-value is the
exact one-sided McNemar/binomial tail over discordant pairs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path


MODE_ENV = "STATEFRAME_EXPERIMENT_PROTOCOL"
ARM_ENV = "STATEFRAME_EXPERIMENT_ARM"
PHASE_ENV = "STATEFRAME_EXPERIMENT_PHASE"
CANDIDATE_ENV = "STATEFRAME_EXPERIMENT_CANDIDATE"
PAIR_ENV = "STATEFRAME_EXPERIMENT_PAIR_ID"
BLOCK_ENV = "STATEFRAME_EXPERIMENT_BLOCK_ID"
CONTROL_ENV = "STATEFRAME_EXPERIMENT_CONTROL_KEY"
CONFIG_ENV = "STATEFRAME_EXPERIMENT_CONFIG_HASH"
SHARED_OFF_ENV = "STATEFRAME_EXPERIMENT_SHARED_OFF"

SCHEMA = "stateframe-experiment-protocol/1"
PLAN_SCHEMA = "stateframe-experiment-plan/1"
ANALYSIS_SCHEMA = "stateframe-experiment-analysis/1"
PROTOCOL_HEADER = "[Experiment protocol]"

DELIVERY_PAIRS = 12
FUTILITY_PAIRS = 24
ADVANCE_PAIRS = 48
FUTILITY_DELTA = -0.10
ADVANCE_DELTA = 0.10
ADVANCE_P_VALUE = 0.20
MAX_CONFIRMATION_TICKETS = 2
CONFIRMATION_MAX_PAIRS = 450
CONFIRMATION_EARLY_ALPHA = 0.005
CONFIRMATION_FINAL_ALPHA = 0.020

_VALID_ARMS = frozenset({"on", "off"})
_VALID_PHASES = frozenset({"screen", "confirmation"})
_SLUG_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def mode(environ: Mapping[str, str] | None = None) -> str:
    """Return the validated protocol switch; absent and ``off`` are identical."""
    source = os.environ if environ is None else environ
    value = str(source.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return mode(environ) == "on"


def configuration(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    return {"experiment_protocol": mode(environ)}


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), default=str).encode("utf-8")


def _digest_label(value: object, *, length: int = 16) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()[:length]


def _slug(value: object, *, fallback: str = "item") -> str:
    text = str(value or "").strip()
    text = _SLUG_RE.sub("-", text).strip("-")
    return text[:80] or fallback


def _one_line(value: object, *, fallback: str = "unassigned", limit: int = 120) -> str:
    text = " ".join(str(value or "").split()).strip()
    return (text or fallback)[:limit]


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if number < 1:
        raise ValueError(f"{name} must be a positive integer")
    return number


def _task_id(value: object) -> str:
    if isinstance(value, Mapping):
        value = value.get("task_id", value.get("id", value.get("name")))
    result = _one_line(value, fallback="", limit=160)
    if not result:
        raise ValueError("every task needs a non-empty task_id")
    return result


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, Sequence) or isinstance(values, (bytes, bytearray)):
        raise ValueError("applicability must be a string or sequence of strings")
    result = tuple(_one_line(item, fallback="", limit=160)
                   for item in values if _one_line(item, fallback="", limit=160))
    return tuple(dict.fromkeys(result))


@dataclass(frozen=True)
class Candidate:
    """One registered mechanism.  Its config hash is frozen before outcomes."""

    candidate_id: str
    family: str
    mechanism: str
    config_hash: str
    endpoint: str = "success"
    control_key: str = ""
    applicability: tuple[str, ...] = ()
    budget: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: object) -> "Candidate":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("candidate specifications must be mappings")
        candidate_id = _one_line(value.get("candidate_id", value.get("id")),
                                 fallback="", limit=120)
        if not candidate_id:
            raise ValueError("candidate_id is required")
        family = _one_line(value.get("family"), fallback=candidate_id, limit=120)
        mechanism = _one_line(value.get("mechanism"), fallback=candidate_id, limit=160)
        endpoint = _one_line(value.get("endpoint"), fallback="success", limit=80)
        raw_hash = value.get("config_hash")
        if raw_hash in (None, ""):
            raw_hash = "sha256:" + _digest_label({
                "candidate_id": candidate_id,
                "family": family,
                "mechanism": mechanism,
                "endpoint": endpoint,
                "applicability": value.get("applicability", ()),
                "metadata": value.get("metadata", {}),
            }, length=64)
        config_hash = _one_line(raw_hash, fallback="", limit=160)
        if not config_hash:
            raise ValueError("config_hash is required")
        control_key = _one_line(
            value.get("control_key"), fallback=f"{family}:{endpoint}", limit=160)
        budget = value.get("budget")
        if budget is not None:
            budget = _positive_int(budget, "candidate budget")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("candidate metadata must be a mapping")
        return cls(
            candidate_id=candidate_id,
            family=family,
            mechanism=mechanism,
            config_hash=config_hash,
            endpoint=endpoint,
            control_key=control_key,
            applicability=_string_tuple(value.get("applicability")),
            budget=budget,
            metadata=copy.deepcopy(dict(metadata)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "family": self.family,
            "mechanism": self.mechanism,
            "config_hash": self.config_hash,
            "endpoint": self.endpoint,
            "control_key": self.control_key,
            "applicability": list(self.applicability),
            "budget": self.budget,
            "metadata": copy.deepcopy(dict(self.metadata)),
        }


CandidateSpec = Candidate


@dataclass(frozen=True)
class PlannedPair:
    pair_id: str
    candidate_id: str
    pair_index: int
    task_id: str
    replicate: int
    on_block_id: str
    off_block_id: str
    control_key: str
    arm_order: tuple[str, str]
    seed: str

    def to_dict(self) -> dict[str, object]:
        return {
            "pair_id": self.pair_id,
            "candidate_id": self.candidate_id,
            "pair_index": self.pair_index,
            "task_id": self.task_id,
            "replicate": self.replicate,
            "on_block_id": self.on_block_id,
            "off_block_id": self.off_block_id,
            "control_key": self.control_key,
            "arm_order": list(self.arm_order),
            "seed": self.seed,
        }


@dataclass(frozen=True)
class PlannedBlock:
    block_id: str
    arm: str
    candidate_id: str | None
    family: str
    task_id: str
    replicate: int
    control_key: str
    pair_ids: tuple[str, ...]
    seed: str
    shared_off: bool
    environment: Mapping[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "block_id": self.block_id,
            "arm": self.arm,
            "candidate_id": self.candidate_id,
            "family": self.family,
            "task_id": self.task_id,
            "replicate": self.replicate,
            "control_key": self.control_key,
            "pair_ids": list(self.pair_ids),
            "seed": self.seed,
            "shared_off": self.shared_off,
            "environment": dict(self.environment),
        }


@dataclass(frozen=True)
class ConfirmationTicket:
    ticket_id: str
    release_id: str
    candidate_id: str
    max_pairs: int = CONFIRMATION_MAX_PAIRS
    early_alpha: float = CONFIRMATION_EARLY_ALPHA
    final_alpha: float = CONFIRMATION_FINAL_ALPHA
    fresh_data: bool = True

    @property
    def early_look_pairs(self) -> int:
        return self.max_pairs // 2

    def to_dict(self) -> dict[str, object]:
        return {
            "ticket_id": self.ticket_id,
            "release_id": self.release_id,
            "candidate_id": self.candidate_id,
            "max_pairs": self.max_pairs,
            "early_look_pairs": self.early_look_pairs,
            "early_alpha": self.early_alpha,
            "final_alpha": self.final_alpha,
            "fresh_data": self.fresh_data,
        }


@dataclass(frozen=True)
class BatchPlan:
    release_id: str
    seed: str
    candidates: tuple[Candidate, ...]
    pairs: tuple[PlannedPair, ...]
    blocks: tuple[PlannedBlock, ...]
    confirmation_tickets: tuple[ConfirmationTicket, ...] = ()
    warnings: tuple[str, ...] = ()
    # Fix (post-review r44, finding 3): the plan's own phase (and, for
    # confirmation, its unique run id) are persisted so analyze_confirmation
    # can reject a screen plan or a stale/reused confirmation run outright,
    # instead of trusting whatever pair/block ids the caller happens to pass.
    phase: str = "screen"
    confirmation_run_id: str | None = None

    @property
    def protocol(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "delivery_pairs": DELIVERY_PAIRS,
            "futility_pairs": FUTILITY_PAIRS,
            "advance_pairs": ADVANCE_PAIRS,
            "futility_delta": FUTILITY_DELTA,
            "advance_delta": ADVANCE_DELTA,
            "advance_p_value": ADVANCE_P_VALUE,
            "max_confirmation_tickets": MAX_CONFIRMATION_TICKETS,
            "shared_off_controls": True,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": PLAN_SCHEMA,
            "release_id": self.release_id,
            "seed": self.seed,
            "protocol": self.protocol,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "pairs": [pair.to_dict() for pair in self.pairs],
            "blocks": [block.to_dict() for block in self.blocks],
            "confirmation_tickets": [ticket.to_dict() for ticket in self.confirmation_tickets],
            "warnings": list(self.warnings),
            "phase": self.phase,
            "confirmation_run_id": self.confirmation_run_id,
        }

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "BatchPlan":
        if value.get("schema") not in {None, PLAN_SCHEMA}:
            raise ValueError("unsupported experiment plan schema")
        candidates = tuple(Candidate.from_value(item)
                           for item in value.get("candidates", ()))
        pairs = tuple(
            PlannedPair(
                pair_id=str(item["pair_id"]), candidate_id=str(item["candidate_id"]),
                pair_index=int(item["pair_index"]), task_id=str(item["task_id"]),
                replicate=int(item["replicate"]), on_block_id=str(item["on_block_id"]),
                off_block_id=str(item["off_block_id"]), control_key=str(item["control_key"]),
                arm_order=tuple(item.get("arm_order", ("on", "off"))),
                seed=str(item["seed"]),
            )
            for item in value.get("pairs", ()) if isinstance(item, Mapping)
        )
        blocks = tuple(
            PlannedBlock(
                block_id=str(item["block_id"]), arm=str(item["arm"]),
                candidate_id=(str(item["candidate_id"]) if item.get("candidate_id") is not None else None),
                family=str(item["family"]), task_id=str(item["task_id"]),
                replicate=int(item["replicate"]), control_key=str(item["control_key"]),
                pair_ids=tuple(str(pair_id) for pair_id in item.get("pair_ids", ())),
                seed=str(item["seed"]), shared_off=bool(item.get("shared_off", False)),
                environment={str(key): str(val) for key, val in
                             dict(item.get("environment", {})).items()},
            )
            for item in value.get("blocks", ()) if isinstance(item, Mapping)
        )
        tickets = tuple(
            ConfirmationTicket(
                ticket_id=str(item["ticket_id"]), release_id=str(item["release_id"]),
                candidate_id=str(item["candidate_id"]), max_pairs=int(item["max_pairs"]),
                early_alpha=float(item.get("early_alpha", CONFIRMATION_EARLY_ALPHA)),
                final_alpha=float(item.get("final_alpha", CONFIRMATION_FINAL_ALPHA)),
                fresh_data=bool(item.get("fresh_data", True)),
            )
            for item in value.get("confirmation_tickets", ()) if isinstance(item, Mapping)
        )
        if len(tickets) > MAX_CONFIRMATION_TICKETS:
            raise ValueError("confirmation tickets exceed the per-release maximum")
        # Fix (post-review r44, finding 4): validate id uniqueness on LOAD
        # too, not only at construction time -- a hand-edited or corrupted
        # plan file must not silently pass an aliased pair/block through.
        _assert_unique_ids(pairs, blocks)
        phase = str(value.get("phase", "screen") or "screen")
        if phase not in _VALID_PHASES:
            raise ValueError("phase must be screen or confirmation")
        confirmation_run_id = value.get("confirmation_run_id")
        return cls(
            release_id=str(value.get("release_id", "release")),
            seed=str(value.get("seed", "label")), candidates=candidates,
            pairs=pairs, blocks=blocks, confirmation_tickets=tickets,
            warnings=tuple(str(item) for item in value.get("warnings", ())),
            phase=phase,
            confirmation_run_id=(str(confirmation_run_id)
                                 if confirmation_run_id is not None else None),
        )


def _block_environment(*, arm: str, phase: str, candidate_id: str,
                       pair_id: str, block_id: str, control_key: str,
                       config_hash: str, shared_off: bool) -> dict[str, str]:
    return {
        MODE_ENV: "on",
        ARM_ENV: arm,
        PHASE_ENV: phase,
        CANDIDATE_ENV: candidate_id,
        PAIR_ENV: pair_id,
        BLOCK_ENV: block_id,
        CONTROL_ENV: control_key,
        CONFIG_ENV: config_hash,
        SHARED_OFF_ENV: "1" if shared_off else "0",
    }


def _confirmation_values(
    release_id: str,
    candidate_id: str,
    requested: int | Sequence[ConfirmationTicket | Mapping[str, object]],
) -> tuple[ConfirmationTicket, ...]:
    if isinstance(requested, bool):
        raise ValueError("confirmation tickets must be an integer or ticket sequence")
    if isinstance(requested, int):
        count = requested
        if count < 0:
            raise ValueError("confirmation tickets cannot be negative")
        if count > MAX_CONFIRMATION_TICKETS:
            raise ValueError("confirmation tickets exceed the per-release maximum")
        return tuple(
            ConfirmationTicket(
                ticket_id=f"{_slug(release_id, fallback='release')}-confirm-{index}",
                release_id=release_id, candidate_id=candidate_id,
            )
            for index in range(1, count + 1)
        )
    values = tuple(requested)
    if len(values) > MAX_CONFIRMATION_TICKETS:
        raise ValueError("confirmation tickets exceed the per-release maximum")
    result = []
    for value in values:
        if isinstance(value, ConfirmationTicket):
            result.append(value)
        elif isinstance(value, Mapping):
            result.append(ConfirmationTicket(
                ticket_id=str(value["ticket_id"]), release_id=str(value.get("release_id", release_id)),
                candidate_id=str(value.get("candidate_id", candidate_id)),
                max_pairs=_positive_int(value.get("max_pairs", CONFIRMATION_MAX_PAIRS), "max_pairs"),
                early_alpha=float(value.get("early_alpha", CONFIRMATION_EARLY_ALPHA)),
                final_alpha=float(value.get("final_alpha", CONFIRMATION_FINAL_ALPHA)),
                fresh_data=bool(value.get("fresh_data", True)),
            ))
        else:
            raise ValueError("confirmation tickets must contain ticket objects")
    return tuple(result)


def build_batch_plan(
    candidates: Sequence[Candidate | Mapping[str, object]],
    tasks: Sequence[object],
    *,
    replicates: int = 1,
    pair_count: int | None = None,
    release_id: str = "screen",
    seed: str = "screen-seed",
    phase: str = "screen",
    confirmation_tickets: int | Sequence[ConfirmationTicket | Mapping[str, object]] = 0,
    confirmation_candidate_id: str | None = None,
    confirmation_run_id: str | None = None,
) -> BatchPlan:
    """Build a deterministic paired batch without launching any work.

    A pair is one candidate ON block plus one OFF block.  OFF blocks are keyed
    by ``(control_key, task_id, replicate)`` and therefore appear once even
    when several compatible candidates reference them.

    ``phase`` and (for confirmation) ``confirmation_run_id`` are baked into
    every pair/block id so a confirmation plan can never collide with, or be
    satisfied by, outcomes recorded against a screen plan for the same
    release (post-review r44, finding 3).
    """
    if phase not in _VALID_PHASES:
        raise ValueError("phase must be screen or confirmation")
    if phase == "confirmation":
        if not str(confirmation_run_id or "").strip():
            raise ValueError("confirmation phase requires a confirmation_run_id")
        confirmation_run_id = _one_line(confirmation_run_id, fallback="run", limit=120)
    else:
        confirmation_run_id = None
    release_id = _one_line(release_id, fallback="screen", limit=120)
    seed = _one_line(seed, fallback="screen-seed", limit=160)
    replicate_count = _positive_int(replicates, "replicates")
    candidate_values = tuple(Candidate.from_value(value) for value in candidates)
    if not candidate_values:
        raise ValueError("at least one candidate is required")
    candidate_ids = [candidate.candidate_id for candidate in candidate_values]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate_id values must be unique")
    task_values = tuple(dict.fromkeys(_task_id(value) for value in tasks))
    if not task_values:
        raise ValueError("at least one task is required")
    slots = tuple((task_id, replicate)
                  for replicate in range(1, replicate_count + 1)
                  for task_id in task_values)
    if pair_count is None:
        requested_pairs = len(slots)
    else:
        requested_pairs = _positive_int(pair_count, "pair_count")
        if requested_pairs > len(slots):
            raise ValueError("pair_count exceeds task x replicate slots")

    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidate_values}
    pairs: list[PlannedPair] = []
    on_blocks: list[PlannedBlock] = []
    off_records: dict[tuple[str, str, int], dict[str, object]] = {}
    for candidate in candidate_values:
        ordered_slots = tuple(sorted(
            slots,
            key=lambda slot: _digest_label((seed, candidate.candidate_id, *slot), length=64),
        ))[:requested_pairs]
        for pair_index, (task_id, replicate) in enumerate(ordered_slots, start=1):
            # Fix (post-review r44, findings 3 & 4): a sanitized slug alone
            # is not a unique identity -- distinct raw values (punctuation
            # difference, or the same release used for a different phase/
            # confirmation run) can slug to the same string. Append a digest
            # of the complete UNSANITIZED identity tuple so a collision in
            # the human-readable prefix can never alias two distinct blocks.
            phase_tag = phase if phase == "screen" else f"confirmation-{confirmation_run_id}"
            pair_identity = (release_id, phase, confirmation_run_id,
                             candidate.candidate_id, task_id, replicate, pair_index)
            on_identity = (release_id, phase, confirmation_run_id,
                          "on", candidate.candidate_id, task_id, replicate)
            off_identity = (release_id, phase, confirmation_run_id,
                           "off", candidate.control_key, task_id, replicate)
            pair_id = (
                f"{_slug(release_id)}-{_slug(phase_tag)}-pair-{_slug(candidate.candidate_id)}-"
                f"{pair_index:03d}-{_digest_label(pair_identity, length=10)}"
            )
            on_block_id = (
                f"{_slug(release_id)}-{_slug(phase_tag)}-on-{_slug(candidate.candidate_id)}-"
                f"{_slug(task_id)}-{replicate}-{_digest_label(on_identity, length=10)}"
            )
            off_key = (phase, confirmation_run_id, candidate.control_key, task_id, replicate)
            off_block_id = (
                f"{_slug(release_id)}-{_slug(phase_tag)}-off-{_slug(candidate.control_key)}-"
                f"{_slug(task_id)}-{replicate}-{_digest_label(off_identity, length=10)}"
            )
            pair_seed = _digest_label((seed, pair_id), length=24)
            arm_order = ("on", "off") if int(pair_seed[-1], 16) % 2 else ("off", "on")
            pairs.append(PlannedPair(
                pair_id=pair_id, candidate_id=candidate.candidate_id,
                pair_index=pair_index, task_id=task_id, replicate=replicate,
                on_block_id=on_block_id, off_block_id=off_block_id,
                control_key=candidate.control_key, arm_order=arm_order, seed=pair_seed,
            ))
            on_blocks.append(PlannedBlock(
                block_id=on_block_id, arm="on", candidate_id=candidate.candidate_id,
                family=candidate.family, task_id=task_id, replicate=replicate,
                control_key=candidate.control_key, pair_ids=(pair_id,), seed=pair_seed,
                shared_off=False,
                environment=_block_environment(
                    arm="on", phase=phase, candidate_id=candidate.candidate_id,
                    pair_id=pair_id, block_id=on_block_id, control_key=candidate.control_key,
                    config_hash=candidate.config_hash, shared_off=False,
                ),
            ))
            record = off_records.setdefault(off_key, {
                "block_id": off_block_id,
                "task_id": task_id,
                "replicate": replicate,
                "control_key": candidate.control_key,
                "family": candidate.family,
                "pair_ids": [],
            })
            if "endpoint" in record and candidate.endpoint != str(record["endpoint"]):
                raise ValueError("candidates sharing an OFF control must share an endpoint")
            record.setdefault("endpoint", candidate.endpoint)
            record["pair_ids"].append(pair_id)

    off_blocks: list[PlannedBlock] = []
    for record in off_records.values():
        block_id = str(record["block_id"])
        pair_ids = tuple(str(item) for item in record["pair_ids"])
        off_blocks.append(PlannedBlock(
            block_id=block_id, arm="off", candidate_id=None,
            family=str(record["family"]), task_id=str(record["task_id"]),
            replicate=int(record["replicate"]), control_key=str(record["control_key"]),
            pair_ids=pair_ids, seed=_digest_label((seed, block_id), length=24),
            shared_off=len(pair_ids) > 1,
            environment=_block_environment(
                arm="off", phase=phase, candidate_id="shared-off",
                pair_id="shared-off", block_id=block_id,
                control_key=str(record["control_key"]), config_hash="off-control",
                shared_off=len(pair_ids) > 1,
            ),
        ))

    warnings = []
    if requested_pairs < DELIVERY_PAIRS:
        warnings.append(f"plan has fewer than delivery look ({DELIVERY_PAIRS}) pairs per candidate")
    elif requested_pairs < FUTILITY_PAIRS:
        warnings.append(f"plan has fewer than futility look ({FUTILITY_PAIRS}) pairs per candidate")
    elif requested_pairs < ADVANCE_PAIRS:
        warnings.append(f"plan has fewer than advance look ({ADVANCE_PAIRS}) pairs per candidate")
    confirmation_id = confirmation_candidate_id or candidate_values[0].candidate_id
    if confirmation_id not in candidate_by_id:
        raise ValueError("confirmation_candidate_id must name a candidate")
    tickets = _confirmation_values(release_id, confirmation_id, confirmation_tickets)
    all_blocks = tuple([*on_blocks, *off_blocks])
    _assert_unique_ids(pairs, all_blocks)
    return BatchPlan(
        release_id=release_id, seed=seed, candidates=candidate_values,
        pairs=tuple(pairs), blocks=all_blocks,
        confirmation_tickets=tickets, warnings=tuple(warnings),
        phase=phase, confirmation_run_id=confirmation_run_id,
    )


def _assert_unique_ids(pairs: Sequence["PlannedPair"], blocks: Sequence["PlannedBlock"]) -> None:
    """Validate id uniqueness on construction AND on load (finding 4)."""
    pair_ids = [pair.pair_id for pair in pairs]
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("pair_id collision detected while building the plan")
    block_ids = [block.block_id for block in blocks]
    if len(set(block_ids)) != len(block_ids):
        raise ValueError("block_id collision detected while building the plan")


plan_batch = build_batch_plan
make_batch_plan = build_batch_plan


def allocate_confirmation_tickets(
    release_id: str,
    candidate_id: str,
    *,
    count: int = 1,
    existing: Sequence[ConfirmationTicket | Mapping[str, object]] = (),
) -> tuple[ConfirmationTicket, ...]:
    """Allocate no more than two confirmation tickets for one release."""
    existing_tickets = _confirmation_values(release_id, candidate_id, tuple(existing))
    if count < 0:
        raise ValueError("confirmation tickets cannot be negative")
    if len(existing_tickets) + count > MAX_CONFIRMATION_TICKETS:
        raise ValueError("confirmation tickets exceed the per-release maximum")
    start = len(existing_tickets) + 1
    new_tickets = tuple(
        ConfirmationTicket(
            ticket_id=f"{_slug(release_id)}-confirm-{index}",
            release_id=str(release_id), candidate_id=str(candidate_id),
        )
        for index in range(start, start + count)
    )
    return (*existing_tickets, *new_tickets)


@dataclass(frozen=True)
class PairObservation:
    pair_id: str
    candidate_id: str
    pair_index: int
    on_success: bool
    off_success: bool
    on_delivered: bool = True
    off_delivered: bool = True


class OutcomeError(ValueError):
    """Raised internally for malformed outcome rows."""


def _first_value(row: Mapping[str, object], names: Sequence[str]) -> object:
    for name in names:
        if name in row:
            return row[name]
    return None


def _bool_value(value: object, *, name: str, default: bool | None = None) -> bool:
    if value is None:
        if default is not None:
            return default
        raise OutcomeError(f"{name} is missing")
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "pass", "passed", "success", "ok", "green"}:
        return True
    if text in {"0", "false", "no", "n", "fail", "failed", "failure", "red"}:
        return False
    raise OutcomeError(f"{name} must be boolean-like")


def _success_value(row: Mapping[str, object], *, name: str) -> bool:
    value = _first_value(row, ("success", "passed", "accepted", "outcome", "result", "status"))
    if isinstance(value, Mapping):
        value = _first_value(value, ("success", "passed", "accepted", "outcome", "result", "status"))
    return _bool_value(value, name=name)


def _delivery_value(row: Mapping[str, object], *, arm: str) -> bool:
    # Fix (post-review r44, finding 2): delivery/effective-config evidence
    # must be explicit. Defaulting a missing field to True let every
    # success-only outcome row silently pass the delivery checkpoint
    # The experiment contract requires this validation.
    # A missing value now raises OutcomeError -> the caller returns
    # INVALID_DATA (or WAITING, before the delivery look is reached).
    names = (("on_delivered", "on_effective", "effective_config", "visible_change", "delivered")
             if arm == "on" else ("off_delivered", "off_effective", "delivered"))
    value = _first_value(row, names)
    return _bool_value(value, name=f"{arm}_delivered")


def _pair_index(row: Mapping[str, object]) -> int:
    value = _first_value(row, ("pair_index", "pair_number", "sequence", "index"))
    if value is None and isinstance(row.get("pair"), int):
        value = row["pair"]
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise OutcomeError("pair_index is required") from error
    if result < 1:
        raise OutcomeError("pair_index must be positive")
    return result


def _direct_observation(row: Mapping[str, object]) -> PairObservation | None:
    has_on = any(key in row for key in ("on_success", "on", "success_on", "passed_on"))
    has_off = any(key in row for key in ("off_success", "off", "success_off", "passed_off"))
    if not (has_on and has_off):
        return None
    pair_index = _pair_index(row)
    pair_id = str(row.get("pair_id", row.get("id", f"pair-{pair_index}")))
    candidate_id = str(row.get("candidate_id", "unassigned"))
    on_value = _first_value(row, ("on_success", "on", "success_on", "passed_on"))
    off_value = _first_value(row, ("off_success", "off", "success_off", "passed_off"))
    return PairObservation(
        pair_id=pair_id, candidate_id=candidate_id, pair_index=pair_index,
        on_success=_bool_value(on_value, name="on_success"),
        off_success=_bool_value(off_value, name="off_success"),
        on_delivered=_delivery_value(row, arm="on"),
        off_delivered=_delivery_value(row, arm="off"),
    )


def _block_observation(row: Mapping[str, object], *, pair: PlannedPair,
                       arm: str) -> tuple[bool, bool]:
    return _success_value(row, name=f"{arm}_success"), _delivery_value(row, arm=arm)


def _observations(
    outcomes: Sequence[Mapping[str, object]],
    *,
    candidate_id: str | None,
    plan: BatchPlan | None,
) -> tuple[list[PairObservation], str | None]:
    direct: dict[str, PairObservation] = {}
    by_pair_arm: dict[tuple[str, str], Mapping[str, object]] = {}
    try:
        for row in outcomes:
            if not isinstance(row, Mapping):
                raise OutcomeError("outcome rows must be mappings")
            observation = _direct_observation(row)
            if observation is not None:
                if observation.pair_id in direct:
                    raise OutcomeError(f"duplicate pair outcome: {observation.pair_id}")
                direct[observation.pair_id] = observation
                continue
            block_id = row.get("block_id")
            arm = str(row.get("arm", "")).strip().lower()
            if block_id is None or arm not in _VALID_ARMS:
                raise OutcomeError("outcome needs paired fields or block_id + arm")
            key = (str(block_id), arm)
            if key in by_pair_arm:
                raise OutcomeError(f"duplicate block outcome: {block_id}/{arm}")
            by_pair_arm[key] = row
    except OutcomeError as error:
        return [], str(error)

    if plan is None:
        selected = direct.values()
        if candidate_id is not None:
            selected = [item for item in selected if item.candidate_id == candidate_id]
        return sorted(selected, key=lambda item: item.pair_index), None
    candidate_pairs = [pair for pair in plan.pairs
                       if candidate_id is None or pair.candidate_id == candidate_id]
    if candidate_id is None:
        candidate_ids = {pair.candidate_id for pair in plan.pairs}
        if len(candidate_ids) == 1:
            candidate_id = next(iter(candidate_ids))
            candidate_pairs = [pair for pair in plan.pairs if pair.candidate_id == candidate_id]
    observations: list[PairObservation] = []
    for pair in candidate_pairs:
        existing = direct.get(pair.pair_id)
        if existing is not None:
            # Fix (post-review r44, finding 1): the row's own pair_index/
            # candidate_id are attacker/producer-supplied and must never
            # override the plan's frozen registration -- otherwise
            # rotating supplied indices across the same 48 outcomes can
            # reorder which pairs occupy the early delivery/futility looks.
            # Re-derive identity strictly from the matched PLANNED pair;
            # reject a row whose supplied candidate_id disagrees outright
            # rather than silently trusting either value.
            if existing.candidate_id != pair.candidate_id:
                return [], (
                    f"pair {pair.pair_id} outcome candidate_id "
                    f"{existing.candidate_id!r} does not match the registered "
                    f"candidate {pair.candidate_id!r}"
                )
            observations.append(PairObservation(
                pair_id=pair.pair_id, candidate_id=pair.candidate_id,
                pair_index=pair.pair_index, on_success=existing.on_success,
                off_success=existing.off_success, on_delivered=existing.on_delivered,
                off_delivered=existing.off_delivered,
            ))
            continue
        # Fix (post-review r44, finding 1): join strictly by registered
        # block ID AND arm. The previous arm-blind `by_block` fallback let a
        # row submitted under the wrong `arm` satisfy either side of a pair
        # whenever its block_id happened to equal `on_block_id`/
        # `off_block_id`, misreporting which arm actually produced it.
        on_row = by_pair_arm.get((pair.on_block_id, "on"))
        off_row = by_pair_arm.get((pair.off_block_id, "off"))
        if on_row is None or off_row is None:
            continue
        try:
            on_success, on_delivered = _block_observation(on_row, pair=pair, arm="on")
            off_success, off_delivered = _block_observation(off_row, pair=pair, arm="off")
        except OutcomeError as error:
            return [], str(error)
        observations.append(PairObservation(
            pair_id=pair.pair_id, candidate_id=pair.candidate_id,
            pair_index=pair.pair_index, on_success=on_success,
            off_success=off_success, on_delivered=on_delivered,
            off_delivered=off_delivered,
        ))
    return sorted(observations, key=lambda item: item.pair_index), None


def _contiguous_prefix(observations: Sequence[PairObservation]) -> tuple[list[PairObservation], int]:
    by_index: dict[int, PairObservation] = {}
    for observation in observations:
        if observation.pair_index in by_index:
            raise OutcomeError(f"duplicate pair index: {observation.pair_index}")
        by_index[observation.pair_index] = observation
    prefix: list[PairObservation] = []
    expected = 1
    while expected in by_index:
        prefix.append(by_index[expected])
        expected += 1
    return prefix, expected - 1


def _stats(observations: Sequence[PairObservation]) -> dict[str, object]:
    count = len(observations)
    on_successes = sum(item.on_success for item in observations)
    off_successes = sum(item.off_success for item in observations)
    wins = sum(item.on_success and not item.off_success for item in observations)
    losses = sum(item.off_success and not item.on_success for item in observations)
    ties = count - wins - losses
    delta = ((on_successes - off_successes) / count) if count else 0.0
    p_value = exact_p_value(wins, losses)
    return {
        "on_successes": on_successes,
        "off_successes": off_successes,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "delta": delta,
        "p_value": p_value,
        "discordant": wins + losses,
    }


def exact_p_value(on_only: int, off_only: int) -> float:
    """One-sided exact McNemar p-value for ON beating OFF."""
    on_only = int(on_only)
    off_only = int(off_only)
    if on_only < 0 or off_only < 0:
        raise ValueError("discordant counts cannot be negative")
    discordant = on_only + off_only
    if discordant == 0:
        return 1.0
    return sum(math.comb(discordant, index) for index in range(on_only, discordant + 1)) / (2 ** discordant)


@dataclass(frozen=True)
class ScreenAnalysis:
    candidate_id: str
    decision: str
    phase: str
    stop_reason: str
    pairs_observed: int
    required_pairs: int
    look_pairs: int | None
    delivery_passed: bool | None
    on_successes: int
    off_successes: int
    wins: int
    losses: int
    ties: int
    delta: float | None
    p_value: float | None
    ignored_pairs: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": ANALYSIS_SCHEMA,
            "candidate_id": self.candidate_id,
            "decision": self.decision,
            "phase": self.phase,
            "stop_reason": self.stop_reason,
            "pairs_observed": self.pairs_observed,
            "required_pairs": self.required_pairs,
            "look_pairs": self.look_pairs,
            "delivery_passed": self.delivery_passed,
            "on_successes": self.on_successes,
            "off_successes": self.off_successes,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "delta": self.delta,
            "p_value": self.p_value,
            "ignored_pairs": self.ignored_pairs,
        }

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]


def _screen_result(candidate_id: str, *, decision: str, phase: str,
                   reason: str, prefix: Sequence[PairObservation], required: int,
                   look: int | None = None, delivery: bool | None = None,
                   ignored: int = 0, stats: Mapping[str, object] | None = None) -> ScreenAnalysis:
    stats = stats or {}
    return ScreenAnalysis(
        candidate_id=candidate_id, decision=decision, phase=phase,
        stop_reason=reason, pairs_observed=len(prefix), required_pairs=required,
        look_pairs=look, delivery_passed=delivery,
        on_successes=int(stats.get("on_successes", 0)),
        off_successes=int(stats.get("off_successes", 0)),
        wins=int(stats.get("wins", 0)), losses=int(stats.get("losses", 0)),
        ties=int(stats.get("ties", 0)),
        delta=(float(stats["delta"]) if "delta" in stats else None),
        p_value=(float(stats["p_value"]) if "p_value" in stats else None),
        ignored_pairs=ignored,
    )


def analyze_screen(
    outcomes: Sequence[Mapping[str, object]],
    candidate_id: str | None = None,
    plan: BatchPlan | Mapping[str, object] | None = None,
) -> ScreenAnalysis:
    """Evaluate fixed delivery/futility/advance looks for one candidate.

    Rows are sorted by their preassigned ``pair_index`` and only a contiguous
    prefix is eligible for a look.  Arrival order can therefore never advance
    a candidate past a missing earlier block.
    """
    plan_obj = BatchPlan.from_dict(plan) if isinstance(plan, Mapping) else plan
    if candidate_id is None and plan_obj is not None and len(plan_obj.candidates) == 1:
        candidate_id = plan_obj.candidates[0].candidate_id
    candidate_label = str(candidate_id or "unassigned")
    observations, error = _observations(
        list(outcomes), candidate_id=candidate_id, plan=plan_obj)
    if error:
        return _screen_result(candidate_label, decision="INVALID_DATA", phase="DATA",
                              reason=error, prefix=(), required=DELIVERY_PAIRS)
    try:
        prefix, pairs_observed = _contiguous_prefix(observations)
    except OutcomeError as error:
        return _screen_result(candidate_label, decision="INVALID_DATA", phase="DATA",
                              reason=str(error), prefix=(), required=DELIVERY_PAIRS)
    ignored = max(0, len(observations) - len(prefix))
    planned_count = None
    if plan_obj is not None:
        planned_count = sum(pair.candidate_id == candidate_label for pair in plan_obj.pairs)
    if pairs_observed < DELIVERY_PAIRS:
        return _screen_result(candidate_label, decision="WAITING", phase="DELIVERY",
                              reason="WAITING_FOR_PREASSIGNED_DELIVERY_BLOCKS",
                              prefix=prefix, required=DELIVERY_PAIRS, ignored=ignored)
    first_delivery = prefix[:DELIVERY_PAIRS]
    if not all(item.on_delivered and item.off_delivered for item in first_delivery):
        bad_on = any(not item.on_delivered for item in first_delivery)
        return _screen_result(candidate_label, decision="DELIVERY_FAILED", phase="DELIVERY",
                              reason="ON_NOT_DELIVERED" if bad_on else "OFF_NOT_DELIVERED",
                              prefix=first_delivery, required=DELIVERY_PAIRS,
                              look=DELIVERY_PAIRS, delivery=False, ignored=ignored,
                              stats=_stats(first_delivery))
    if planned_count is not None and planned_count < FUTILITY_PAIRS:
        return _screen_result(candidate_label, decision="PLAN_TOO_SMALL", phase="FUTILITY",
                              reason="PLAN_HAS_FEWER_THAN_FUTILITY_LOOK_BLOCKS",
                              prefix=first_delivery, required=FUTILITY_PAIRS,
                              look=DELIVERY_PAIRS, delivery=True, ignored=ignored,
                              stats=_stats(first_delivery))
    if pairs_observed < FUTILITY_PAIRS:
        return _screen_result(candidate_label, decision="CONTINUE", phase="DELIVERY",
                              reason="DELIVERY_PASSED_WAITING_FOR_FUTILITY_LOOK",
                              prefix=first_delivery, required=FUTILITY_PAIRS,
                              look=DELIVERY_PAIRS, delivery=True, ignored=ignored,
                              stats=_stats(first_delivery))
    futility_look = prefix[:FUTILITY_PAIRS]
    futility_stats = _stats(futility_look)
    if float(futility_stats["delta"]) <= FUTILITY_DELTA:
        return _screen_result(candidate_label, decision="FUTILITY", phase="FUTILITY",
                              reason="DELTA_AT_OR_BELOW_MINUS_TEN_POINTS",
                              prefix=futility_look, required=FUTILITY_PAIRS,
                              look=FUTILITY_PAIRS, delivery=True, ignored=ignored,
                              stats=futility_stats)
    if planned_count is not None and planned_count < ADVANCE_PAIRS:
        return _screen_result(candidate_label, decision="PLAN_TOO_SMALL", phase="ADVANCE",
                              reason="PLAN_HAS_FEWER_THAN_ADVANCE_LOOK_BLOCKS",
                              prefix=futility_look, required=ADVANCE_PAIRS,
                              look=FUTILITY_PAIRS, delivery=True, ignored=ignored,
                              stats=futility_stats)
    if pairs_observed < ADVANCE_PAIRS:
        return _screen_result(candidate_label, decision="CONTINUE", phase="FUTILITY",
                              reason="FUTILITY_NOT_TRIGGERED_WAITING_FOR_ADVANCE_LOOK",
                              prefix=futility_look, required=ADVANCE_PAIRS,
                              look=FUTILITY_PAIRS, delivery=True, ignored=ignored,
                              stats=futility_stats)
    advance_look = prefix[:ADVANCE_PAIRS]
    advance_stats = _stats(advance_look)
    if (float(advance_stats["delta"]) >= ADVANCE_DELTA
            and float(advance_stats["p_value"]) <= ADVANCE_P_VALUE):
        decision = "ADVANCE"
        reason = "LARGE_EFFECT_AND_PAIRED_P_VALUE_PASSED"
    else:
        decision = "NO_ADVANCE"
        reason = "ADVANCE_CRITERIA_NOT_MET"
    return _screen_result(candidate_label, decision=decision, phase="ADVANCE",
                          reason=reason, prefix=advance_look, required=ADVANCE_PAIRS,
                          look=ADVANCE_PAIRS, delivery=True, ignored=ignored,
                          stats=advance_stats)


def analyze_confirmation(
    outcomes: Sequence[Mapping[str, object]],
    ticket: ConfirmationTicket | Mapping[str, object],
    *,
    plan: BatchPlan | Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Evaluate a frozen winner on fresh data at two preassigned looks."""
    if isinstance(ticket, Mapping):
        ticket = ConfirmationTicket(
            ticket_id=str(ticket["ticket_id"]), release_id=str(ticket["release_id"]),
            candidate_id=str(ticket["candidate_id"]), max_pairs=int(ticket.get("max_pairs", CONFIRMATION_MAX_PAIRS)),
            early_alpha=float(ticket.get("early_alpha", CONFIRMATION_EARLY_ALPHA)),
            final_alpha=float(ticket.get("final_alpha", CONFIRMATION_FINAL_ALPHA)),
            fresh_data=bool(ticket.get("fresh_data", True)),
        )
    plan_obj = BatchPlan.from_dict(plan) if isinstance(plan, Mapping) else plan
    # Fix (post-review r44, finding 3): "fresh confirmation" was only a
    # label -- a screen plan (same pair/block ids, since the old scheme
    # ignored phase entirely) was silently accepted and its screening rows
    # reproduced CONFIRMED_EARLY. Reject anything that is not an explicit
    # confirmation-phase plan outright, before touching any observation.
    if plan_obj is not None and plan_obj.phase != "confirmation":
        return {"schema": ANALYSIS_SCHEMA, "ticket_id": ticket.ticket_id,
                "decision": "INVALID_DATA",
                "reason": "confirmation requires a confirmation-phase plan, not screen"}
    observations, error = _observations(
        list(outcomes), candidate_id=ticket.candidate_id, plan=plan_obj)
    if error:
        return {"schema": ANALYSIS_SCHEMA, "ticket_id": ticket.ticket_id,
                "decision": "INVALID_DATA", "reason": error}
    prefix, _ = _contiguous_prefix(observations)
    early_n = ticket.early_look_pairs
    if len(prefix) < early_n:
        return {"schema": ANALYSIS_SCHEMA, "ticket_id": ticket.ticket_id,
                "candidate_id": ticket.candidate_id, "decision": "WAITING",
                "look": early_n, "fresh_data_required": ticket.fresh_data}
    early = _stats(prefix[:early_n])
    if float(early["delta"]) <= 0:
        decision = "FUTILITY"
        reason = "EARLY_DELTA_NON_POSITIVE"
        look = early_n
        stats = early
    elif float(early["p_value"]) <= ticket.early_alpha:
        decision = "CONFIRMED_EARLY"
        reason = "EARLY_EFFICACY_P_VALUE_PASSED"
        look = early_n
        stats = early
    elif len(prefix) < ticket.max_pairs:
        return {"schema": ANALYSIS_SCHEMA, "ticket_id": ticket.ticket_id,
                "candidate_id": ticket.candidate_id, "decision": "CONTINUE",
                "look": early_n, "required_pairs": ticket.max_pairs,
                "early": early, "fresh_data_required": ticket.fresh_data}
    else:
        final = _stats(prefix[:ticket.max_pairs])
        if float(final["p_value"]) <= ticket.final_alpha:
            decision = "CONFIRMED"
            reason = "FINAL_EFFICACY_P_VALUE_PASSED"
        else:
            decision = "NOT_CONFIRMED"
            reason = "FINAL_EFFICACY_P_VALUE_NOT_MET"
        look = ticket.max_pairs
        stats = final
    return {
        "schema": ANALYSIS_SCHEMA,
        "ticket_id": ticket.ticket_id,
        "candidate_id": ticket.candidate_id,
        "decision": decision,
        "reason": reason,
        "look": look,
        "fresh_data_required": ticket.fresh_data,
        "stats": stats,
    }


def analyze_batch(
    outcomes: Sequence[Mapping[str, object]],
    plan: BatchPlan | Mapping[str, object],
) -> dict[str, object]:
    plan_obj = BatchPlan.from_dict(plan) if isinstance(plan, Mapping) else plan
    results = [analyze_screen(outcomes, candidate.candidate_id, plan_obj)
               for candidate in plan_obj.candidates]
    off_by_control: dict[str, set[str]] = defaultdict(set)
    for block in plan_obj.blocks:
        if block.arm == "off":
            off_by_control[block.control_key].add(block.block_id)
    return {
        "schema": ANALYSIS_SCHEMA,
        "release_id": plan_obj.release_id,
        "protocol": plan_obj.protocol,
        "candidates": [result.to_dict() for result in results],
        "shared_off": {
            "off_block_count": sum(len(blocks) for blocks in off_by_control.values()),
            "controls": {key: sorted(blocks) for key, blocks in sorted(off_by_control.items())},
        },
        "confirmation_tickets": [ticket.to_dict()
                                  for ticket in plan_obj.confirmation_tickets],
    }


@dataclass(frozen=True)
class RuntimeAssignment:
    phase: str
    candidate_id: str
    arm: str
    pair_id: str
    block_id: str
    control_key: str
    config_hash: str
    shared_off: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol_schema": SCHEMA,
            "phase": self.phase,
            "candidate_id": self.candidate_id,
            "arm": self.arm,
            "pair_id": self.pair_id,
            "block_id": self.block_id,
            "control_key": self.control_key,
            "config_hash": self.config_hash,
            "shared_off": self.shared_off,
        }


def runtime_assignment(environ: Mapping[str, str] | None = None) -> RuntimeAssignment:
    source = os.environ if environ is None else environ
    if mode(source) != "on":
        raise ValueError("runtime assignment requires the protocol switch to be on")
    arm = str(source.get(ARM_ENV, "on") or "on").strip().lower()
    phase = str(source.get(PHASE_ENV, "screen") or "screen").strip().lower()
    if arm not in _VALID_ARMS:
        raise ValueError(f"{ARM_ENV} must be on or off")
    if phase not in _VALID_PHASES:
        raise ValueError(f"{PHASE_ENV} must be screen or confirmation")
    shared = str(source.get(SHARED_OFF_ENV, "0") or "0").strip().lower()
    return RuntimeAssignment(
        phase=phase,
        candidate_id=_one_line(source.get(CANDIDATE_ENV), fallback="unassigned"),
        arm=arm,
        pair_id=_one_line(source.get(PAIR_ENV), fallback="unassigned"),
        block_id=_one_line(source.get(BLOCK_ENV), fallback="unassigned"),
        control_key=_one_line(source.get(CONTROL_ENV), fallback="default"),
        config_hash=_one_line(source.get(CONFIG_ENV), fallback="unregistered"),
        shared_off=shared in {"1", "true", "yes", "on"},
    )


def render_protocol_block(assignment: RuntimeAssignment | Mapping[str, object]) -> str:
    if isinstance(assignment, Mapping):
        assignment = RuntimeAssignment(
            phase=str(assignment.get("phase", "screen")),
            candidate_id=str(assignment.get("candidate_id", "unassigned")),
            arm=str(assignment.get("arm", "on")).lower(),
            pair_id=str(assignment.get("pair_id", "unassigned")),
            block_id=str(assignment.get("block_id", "unassigned")),
            control_key=str(assignment.get("control_key", "default")),
            config_hash=str(assignment.get("config_hash", "unregistered")),
            shared_off=bool(assignment.get("shared_off", False)),
        )
    return "\n".join((
        PROTOCOL_HEADER,
        f"schema={SCHEMA}; phase={assignment.phase}; candidate={_one_line(assignment.candidate_id)}; arm={assignment.arm.upper()}",
        f"pair={_one_line(assignment.pair_id)}; block={_one_line(assignment.block_id)}; control={_one_line(assignment.control_key)}",
        f"config_hash={_one_line(assignment.config_hash)}; shared_off={'yes' if assignment.shared_off else 'no'}",
        "gates=delivery@12; futility@24 (delta <= -0.10); advance@48 (delta >= +0.10, paired_p <= 0.20)",
    ))


def append_protocol_block(
    messages: Sequence[Mapping[str, object]],
    assignment: RuntimeAssignment | Mapping[str, object],
) -> list[dict[str, object]]:
    """Append exactly one assignment block without changing existing messages."""
    result = [copy.deepcopy(dict(message)) for message in messages]
    if any(str(message.get("content", "")).startswith(PROTOCOL_HEADER)
           for message in result):
        return result
    result.append({"role": "user", "content": render_protocol_block(assignment)})
    return result


render_runtime_block = render_protocol_block


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_candidate_arg(value: str) -> dict[str, str]:
    if "|" in value:
        parts = value.split("|")
    else:
        parts = value.split(":")
        # Config hashes are commonly written as ``sha256:<digest>``.  Keep
        # that prefix together while still permitting an optional control key.
        if len(parts) > 4 and parts[3].lower() in {"sha256", "sha1", "md5"}:
            parts = [*parts[:3], ":".join(parts[3:5]), *parts[5:]]
    if len(parts) < 4:
        raise argparse.ArgumentTypeError(
            "--candidate expects candidate_id|family|mechanism|config_hash[|control_key]"
        )
    result = {
        "candidate_id": parts[0], "family": parts[1],
        "mechanism": parts[2], "config_hash": parts[3],
    }
    if len(parts) > 4:
        result["control_key"] = ":".join(parts[4:])
    return result


def _write_json(path: str | None, value: object) -> None:
    rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path:
        Path(path).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser("plan", help="write a preassigned screen batch")
    plan_parser.add_argument("--candidate", action="append", type=_parse_candidate_arg, required=True)
    plan_parser.add_argument("--task", action="append", required=True)
    plan_parser.add_argument("--replicates", type=int, default=1)
    plan_parser.add_argument("--pairs", type=int, default=None)
    plan_parser.add_argument("--release", default="screen")
    plan_parser.add_argument("--seed", default="screen-seed")
    plan_parser.add_argument("--confirmation-tickets", type=int, default=0)
    plan_parser.add_argument("--confirmation-candidate", default=None)
    plan_parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    if args.command == "plan":
        plan = build_batch_plan(
            args.candidate, args.task, replicates=args.replicates,
            pair_count=args.pairs, release_id=args.release, seed=args.seed,
            confirmation_tickets=args.confirmation_tickets,
            confirmation_candidate_id=args.confirmation_candidate,
        )
        _write_json(args.output, plan.to_dict())
        return 0
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ADVANCE_DELTA", "ADVANCE_PAIRS", "ADVANCE_P_VALUE", "ANALYSIS_SCHEMA",
    "ARM_ENV", "BatchPlan", "BLOCK_ENV", "CONFIG_ENV", "CANDIDATE_ENV",
    "Candidate", "CandidateSpec", "ConfirmationTicket", "CONTROL_ENV",
    "DELIVERY_PAIRS", "FUTILITY_DELTA", "FUTILITY_PAIRS", "MAX_CONFIRMATION_TICKETS",
    "MODE_ENV", "PAIR_ENV", "PLAN_SCHEMA", "PROTOCOL_HEADER", "RuntimeAssignment",
    "SCHEMA", "ScreenAnalysis", "allocate_confirmation_tickets", "analyze_batch",
    "analyze_confirmation", "analyze_screen", "append_protocol_block", "build_batch_plan",
    "configuration", "enabled", "exact_p_value", "make_batch_plan", "mode",
    "plan_batch", "render_protocol_block", "render_runtime_block", "runtime_assignment",
]
