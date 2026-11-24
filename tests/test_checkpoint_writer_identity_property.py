"""Seeded checks for stable background-writer failure identity."""

from __future__ import annotations

import json
import os
import random
from collections.abc import Callable
from pathlib import Path

import pytest

import kernel_checkpoint


_SEED_COUNT = 256
_GENERIC = "unexpected writer error"
_SAFE_REASONS = (
    _GENERIC,
    "checkpoint refused",
    "checkpoint write failed: ENOSPC",
)
_SAFE_TYPES = (
    "TypeError",
    "ValueError",
    "AttributeError",
    "RuntimeError",
    "OverflowError",
    "RecursionError",
)


def _token(rng: random.Random, label: str) -> str:
    return f"{label}-{rng.getrandbits(96):024x}"


def _expected_reason(event: dict[str, object]) -> str:
    reason = event.get("failure_reason")
    if type(reason) is not str or reason not in _SAFE_REASONS:
        return _GENERIC
    error_type = event.get("writer_error_type")
    if reason == _GENERIC and type(error_type) is str and error_type in _SAFE_TYPES:
        return f"{reason}: {error_type}"
    return reason


def _latest_failure(
        events: list[dict[str, object]],
        accepts: Callable[[dict[str, object]], bool],
) -> dict[str, object] | None:
    for event in reversed(events):
        if (event.get("event") == "checkpoint_failed"
                and event.get("background") is True and accepts(event)):
            return event
    return None


def _mutant_step_only(events: list[dict[str, object]], step: int, _generation: str) -> str:
    event = _latest_failure(events, lambda row: row.get("step") == step)
    return _expected_reason(event) if event is not None else _GENERIC


def _mutant_generation_only(
        events: list[dict[str, object]], _step: int, generation: str) -> str:
    event = _latest_failure(events, lambda row: row.get("writer_generation") == generation)
    return _expected_reason(event) if event is not None else _GENERIC


def _mutant_payload_text(events: list[dict[str, object]], step: int, generation: str) -> str:
    event = _latest_failure(
        events,
        lambda row: row.get("step") == step and row.get("writer_generation") == generation,
    )
    value = event.get("error") if event is not None else _GENERIC
    return value if type(value) is str else _GENERIC


def _mutant_lifecycle_override(
        events: list[dict[str, object]], _step: int, _generation: str) -> str:
    event = events[-1]
    reason = event.get("failure_reason")
    error_type = event.get("writer_error_type")
    return f"{reason}: {error_type}"


def _matches_invariant(observed: str, expected: str, forbidden: tuple[str, ...]) -> bool:
    return observed == expected and all(value not in observed for value in forbidden)


def _other_safe_type(rng: random.Random, excluded: object) -> str:
    choices = [error_type for error_type in _SAFE_TYPES if error_type != excluded]
    return rng.choice(choices)


def test_seeded_writer_reason_stays_with_its_step_and_generation(tmp_path: Path) -> None:
    rng = random.Random(0x5EED_C0DE)
    checkpoints = tmp_path / "run" / "checkpoints"
    checkpoints.mkdir(parents=True)
    journal = checkpoints / "events.jsonl"
    killed: dict[str, int | None] = {
        "step_only": None,
        "generation_only": None,
        "payload_text": None,
        "lifecycle_override": None,
    }

    for seed in range(_SEED_COUNT):
        step = rng.randrange(2, 1 << 20)
        writer_pid = 2000 + seed
        generation = _token(rng, "writer")
        other_generation = _token(rng, "other-writer")
        payload_text = _token(rng, "payload")
        payload_path = _token(rng, "path")
        publication_text = _token(rng, "publication")

        reason = rng.choice(_SAFE_REASONS)
        error_type = rng.choice((*_SAFE_TYPES, payload_text, payload_path))
        excluded_type = error_type if error_type in _SAFE_TYPES else None
        target = {
            "event": "checkpoint_failed",
            "step": step,
            "background": True,
            "writer_generation": generation,
            "writer_pid": writer_pid,
            "failure_reason": reason,
            "writer_error_type": error_type,
            "error": payload_text,
            "path": payload_path,
        }
        alternatives = [candidate for candidate in _SAFE_REASONS if candidate != reason]
        wrong_step = {
            "event": "checkpoint_failed",
            "step": step + 1,
            "background": True,
            "writer_generation": generation,
            "writer_pid": writer_pid,
            "failure_reason": alternatives[0],
            "writer_error_type": _other_safe_type(rng, excluded_type),
            "error": _token(rng, "wrong-step-payload"),
            "path": _token(rng, "wrong-step-path"),
        }
        other_writer = {
            "event": "checkpoint_failed",
            "step": step,
            "background": True,
            "writer_generation": other_generation,
            "writer_pid": writer_pid + 1,
            "failure_reason": _GENERIC,
            "writer_error_type": _other_safe_type(rng, excluded_type),
            "error": _token(rng, "other-writer-payload"),
            "path": _token(rng, "other-writer-path"),
        }
        later_records = [
            {
                "event": "checkpoint_saved",
                "step": step,
                "writer_generation": other_generation,
                "writer_pid": writer_pid + 1,
                "failure_reason": alternatives[0],
                "writer_error_type": rng.choice(_SAFE_TYPES),
            },
            {
                "event": "checkpoint_retry",
                "step": step,
                "writer_generation": generation,
                "writer_pid": writer_pid,
                "failure_reason": payload_text,
                "writer_error_type": payload_path,
            },
            {
                "event": "checkpoint_publication",
                "step": step,
                "writer_generation": other_generation,
                "writer_pid": writer_pid + 1,
                "failure_reason": publication_text,
                "writer_error_type": payload_text,
                "path": payload_path,
            },
        ]
        events = [target, wrong_step, other_writer, *later_records]
        journal.write_text("".join(json.dumps(event) + "\n" for event in events))

        expected = _expected_reason(target)
        info = journal.stat()
        observed = kernel_checkpoint._writer_failure_reason(
            checkpoints.parent, step, generation, writer_pid, 0,
            (info.st_dev, info.st_ino),
        )
        forbidden = (
            payload_text,
            payload_path,
            publication_text,
            generation,
            other_generation,
            wrong_step["writer_error_type"],
            other_writer["writer_error_type"],
        )
        assert _matches_invariant(observed, expected, forbidden), f"seed {seed}: {observed!r}"

        mutants = (
            ("step_only", _mutant_step_only),
            ("generation_only", _mutant_generation_only),
            ("payload_text", _mutant_payload_text),
            ("lifecycle_override", _mutant_lifecycle_override),
        )
        for name, mutant in mutants:
            if not _matches_invariant(mutant(events, step, generation), expected, forbidden):
                if killed[name] is None:
                    killed[name] = seed

    assert all(seed is not None for seed in killed.values()), killed


def _failure_row(step: int, identity: str, writer_pid: int, reason: str) -> dict[str, object]:
    return {
        "event": "checkpoint_failed",
        "step": step,
        "background": True,
        "writer_generation": identity,
        "writer_pid": writer_pid,
        "failure_reason": reason,
    }


def _journal_file_id(path: Path) -> tuple[int, int]:
    info = path.stat()
    return info.st_dev, info.st_ino


def test_failure_lookup_scopes_rows_to_offset_step_identity_and_pid(tmp_path: Path) -> None:
    step, offset, writer_pid = 19, 1024, 7401
    identity = f"{step}:{offset}"
    run = tmp_path / "run"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    journal = checkpoints / "events.jsonl"

    prior = json.dumps(_failure_row(step, identity, writer_pid,
                                    "checkpoint refused")).encode() + b"\n"
    assert len(prior) < offset
    journal.write_bytes(prior + b" " * (offset - len(prior)))
    file_id = _journal_file_id(journal)

    def lookup() -> str:
        return kernel_checkpoint._writer_failure_reason(
            run, step, identity, writer_pid, offset, file_id,
        )

    assert lookup() == "unexpected writer error"  # The matching row is before the offset.

    with journal.open("ab") as stream:
        for row in (
            _failure_row(step + 1, identity, writer_pid, "checkpoint refused"),
            _failure_row(step, identity, writer_pid + 1, "checkpoint write failed: ENOSPC"),
        ):
            stream.write(json.dumps(row).encode() + b"\n")
    assert lookup() == "unexpected writer error"

    with journal.open("ab") as stream:
        stream.write(json.dumps(_failure_row(
            step, identity, writer_pid, "checkpoint write failed: ENOSPC",
        )).encode() + b"\n")
    assert lookup() == "checkpoint write failed: ENOSPC"


@pytest.mark.parametrize("damage", ("truncate", "replace"))
def test_failure_lookup_rejects_truncated_or_replaced_journal(
        tmp_path: Path, damage: str) -> None:
    step, offset, writer_pid = 23, 1024, 7501
    identity = f"{step}:{offset}"
    checkpoints = tmp_path / "run" / "checkpoints"
    checkpoints.mkdir(parents=True)
    journal = checkpoints / "events.jsonl"
    journal.write_bytes(b" " * offset)
    file_id = _journal_file_id(journal)
    row = json.dumps(_failure_row(
        step, identity, writer_pid, "checkpoint write failed: ENOSPC",
    )).encode() + b"\n"

    if damage == "truncate":
        journal.write_bytes(row)
    else:
        replacement = checkpoints / "replacement.jsonl"
        replacement.write_bytes(b" " * offset + row)
        assert _journal_file_id(replacement) != file_id
        replacement.replace(journal)

    assert kernel_checkpoint._writer_failure_reason(
        checkpoints.parent, step, identity, writer_pid, offset, file_id,
    ) == "unexpected writer error"


@pytest.mark.skipif(not hasattr(os, "fork"), reason="background writer requires fork")
def test_success_row_carries_deterministic_identity_and_child_pid(tmp_path: Path) -> None:
    run = tmp_path / "run"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    journal = checkpoints / "events.jsonl"
    journal.write_text('{"event":"prior"}\n')
    offset = journal.stat().st_size
    step = 29
    final = checkpoints / "0029-1"
    temporary = checkpoints / ".0029-1.tmp"
    temporary.mkdir()
    previous_state = kernel_checkpoint._STATE.copy()
    kernel_checkpoint._STATE.clear()
    try:
        kernel_checkpoint._fork_writer(
            run, step, final,
            lambda: {"sends": 1, "state_bytes": 12, "chunks": []}, temporary,
        )
        writer_pid = kernel_checkpoint._STATE["child"]
        assert writer_pid is not None
        outcome = kernel_checkpoint.finish_background_writer()
        assert outcome is not None and outcome["outcome"] == "saved" and outcome["exit_code"] == 0
        saved_rows = map(json.loads, reversed(journal.read_text().splitlines()))
        saved = next(row for row in saved_rows if row.get("event") == "checkpoint_saved")
        assert saved["writer_generation"] == f"{step}:{offset}"
        assert saved["writer_pid"] == writer_pid
    finally:
        child = kernel_checkpoint._STATE.get("child")
        if child:
            try:
                os.waitpid(child, 0)
            except ChildProcessError:
                pass
        kernel_checkpoint._STATE.clear()
        kernel_checkpoint._STATE.update(previous_state)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="background writer requires fork")
def test_unwritable_events_journal_does_not_prevent_checkpoint_save(tmp_path: Path) -> None:
    run = tmp_path / "run"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "events.jsonl").mkdir()
    final = checkpoints / "0003-1"
    temporary = checkpoints / ".0003-1.tmp"
    temporary.mkdir()
    previous_state = kernel_checkpoint._STATE.copy()
    kernel_checkpoint._STATE.clear()

    def write_checkpoint() -> dict[str, object]:
        final.mkdir()
        (final / "state.pkl").write_bytes(b"saved")
        return {"sends": 1, "state_bytes": 5, "chunks": []}

    try:
        kernel_checkpoint._fork_writer(run, 3, final, write_checkpoint, temporary)
        writer_pid = kernel_checkpoint._STATE["child"]
        assert writer_pid is not None
        outcome = kernel_checkpoint.finish_background_writer()
        assert outcome is not None and outcome["outcome"] == "saved" and outcome["exit_code"] == 0
        assert (final / "state.pkl").read_bytes() == b"saved"
    finally:
        child = kernel_checkpoint._STATE.get("child")
        if child:
            try:
                os.waitpid(child, 0)
            except ChildProcessError:
                pass
        kernel_checkpoint._STATE.clear()
        kernel_checkpoint._STATE.update(previous_state)
