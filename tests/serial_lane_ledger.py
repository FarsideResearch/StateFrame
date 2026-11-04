"""Read and validate per-session serial-lane execution records."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


_FIELDS = ("collected", "ran", "deferred")


class LedgerError(ValueError):
    """The ledger is unreadable or contains a malformed session record."""


@dataclass(frozen=True)
class LedgerTotals:
    collected: int = 0
    ran: int = 0
    deferred: int = 0
    sessions: int = 0


def read_ledger(path: str | Path) -> LedgerTotals:
    """Sum session records; a missing or empty ledger represents zero collected cases."""
    ledger_path = Path(path)
    totals = {field: 0 for field in _FIELDS}
    sessions = 0
    try:
        stream = ledger_path.open("r", encoding="utf-8")
    except FileNotFoundError:
        return LedgerTotals()
    except OSError as exc:
        raise LedgerError(f"cannot read serial_lane ledger {ledger_path}: {exc}") from exc

    with stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LedgerError(
                    f"invalid JSON on serial_lane ledger line {line_number}: {exc.msg}"
                ) from exc
            if type(record) is not dict or set(record) != set(_FIELDS):
                raise LedgerError(
                    f"serial_lane ledger line {line_number} must contain exactly: "
                    + ", ".join(_FIELDS)
                )
            for field in _FIELDS:
                value = record[field]
                if type(value) is not int or value < 0:
                    raise LedgerError(
                        f"serial_lane ledger line {line_number} has invalid {field}; "
                        "expected a non-negative integer"
                    )
                totals[field] += value
            if record["deferred"] > record["collected"]:
                raise LedgerError(
                    f"serial_lane ledger line {line_number} defers more cases than it collected"
                )
            sessions += 1
    return LedgerTotals(**totals, sessions=sessions)


def failure_message(totals: LedgerTotals) -> str | None:
    """Return a failure when corpus items were collected but none ran in any session."""
    if totals.collected > 0 and totals.ran == 0:
        return (
            "serial_lane ledger failure: "
            f"{totals.collected} corpus cases collected, 0 ran across "
            f"{totals.sessions} pytest sessions ({totals.deferred} deferred)"
        )
    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path, help="per-session JSONL ledger path")
    args = parser.parse_args(argv)
    try:
        totals = read_ledger(args.ledger)
    except LedgerError as exc:
        print(f"serial_lane ledger error: {exc}", file=sys.stderr)
        return 2

    failure = failure_message(totals)
    if failure is not None:
        print(failure, file=sys.stderr)
        return 1
    if totals.collected == 0:
        print("serial_lane ledger: no corpus cases collected")
    else:
        print(
            f"serial_lane ledger: {totals.collected} corpus cases collected, "
            f"{totals.ran} ran, {totals.deferred} deferred across {totals.sessions} pytest sessions"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
