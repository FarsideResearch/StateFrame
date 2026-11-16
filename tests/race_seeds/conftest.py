"""Registers the race_seed marker and validates it at collection time.

tests/enginecheck/seeds.py runs `pytest --collect-only tests/race_seeds` with ENGINECHECK_SEED_DUMP set; the hook below
then writes the validated seeds as JSON. A malformed marker raises UsageError, so collection fails instead of a seed
being skipped. Keep this file free of enginecheck imports: archived known-bad trees carry only tests/enginecheck.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

_DUMP_ENV = "ENGINECHECK_SEED_DUMP"
_ALLOWED_KWARGS = frozenset({"known_bad", "mutant", "expect"})
_SEED_ID = re.compile(r"[A-Za-z0-9_.-]+")
_SHA = re.compile(r"[0-9a-f]{7,40}")
_MUTANT = re.compile(r"[A-Za-z0-9_.-]+")


def _problem(mark: pytest.Mark) -> str | None:
    if len(mark.args) != 1 or not isinstance(mark.args[0], str) or not _SEED_ID.fullmatch(mark.args[0]):
        return "first argument must be one seed id matching [A-Za-z0-9_.-]+"
    unknown = sorted(set(mark.kwargs) - _ALLOWED_KWARGS)
    if unknown:
        return f"unknown keyword(s) {unknown}"
    known_bad = mark.kwargs.get("known_bad")
    mutant = mark.kwargs.get("mutant")
    if (known_bad is None) == (mutant is None):
        return "exactly one of known_bad (commit sha) or mutant (name) is required"
    if known_bad is not None and not (isinstance(known_bad, str) and _SHA.fullmatch(known_bad)):
        return "known_bad must be a lowercase hex commit sha (7-40 chars)"
    if mutant is not None and not (isinstance(mutant, str) and _MUTANT.fullmatch(mutant)):
        return "mutant must match [A-Za-z0-9_.-]+"
    expect = mark.kwargs.get("expect")
    if not isinstance(expect, str) or not expect.strip():
        return "expect is required: the intended failure (exception name or message substring)"
    return None


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "race_seed(seed_id, *, known_bad=<sha> | mutant=<name>, expect=<failure>): race regression seed (enginecheck.seeds)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    seeds: list[dict[str, str | None]] = []
    problems: list[str] = []
    for item in items:
        for mark in item.iter_markers("race_seed"):
            problem = _problem(mark)
            if problem is not None:
                problems.append(f"{item.nodeid}: {problem}")
                continue
            seeds.append(
                {
                    "seed_id": mark.args[0],
                    "node_id": item.nodeid,
                    "known_bad": mark.kwargs.get("known_bad"),
                    "mutant": mark.kwargs.get("mutant"),
                    "expect": mark.kwargs["expect"],
                }
            )
    if problems:
        raise pytest.UsageError("invalid race_seed marker: " + "; ".join(problems))
    dump = os.environ.get(_DUMP_ENV)
    if dump:
        Path(dump).write_text(json.dumps(seeds), encoding="utf-8")
