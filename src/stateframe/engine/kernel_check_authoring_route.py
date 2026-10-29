"""Offer solver check authoring while host obligations lack any check.

Measured (batch 800 sample task, seq 591/869): public tests PASS but host obligations
sat at ``UNKNOWN / MISSING_CHECKS`` and the continuation gate withheld
``authored_check`` / ``read_check_receipt``, so the solver could only submit
uncertified. This module is the single predicate the continuation uses to admit
those two tools. The authored check itself is solver-written from ticket and
repo only; its honesty gates live in ``kernel_authored_check`` (parent FAIL +
candidate PASS on host-frozen bytes) and ``kernel_obligation_assess.
certifying_authored_receipt`` (revision-bound, clause-bound).
"""
from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping

CHECK_AUTHORING_TOOLS = frozenset({"authored_check", "read_check_receipt"})


def _row(item: object) -> Mapping[str, object] | None:
    if isinstance(item, Mapping):
        return item
    if isinstance(item, str) and item.lstrip().startswith("{"):
        try:
            value = ast.literal_eval(item)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            return None
        return value if isinstance(value, Mapping) else None
    return None


def missing_check_obligations(runtime: object) -> tuple[str, ...]:
    """Obligation ids readiness blocks on as ``UNKNOWN / MISSING_CHECKS``."""
    try:
        from kernel_readiness_authority import runtime_snapshot
        snapshot = runtime_snapshot(runtime)
    except (ImportError, AttributeError, TypeError, ValueError):
        return ()
    blocking = getattr(snapshot, "host_blocking", ()) or ()
    if isinstance(blocking, (str, bytes)) or not isinstance(blocking, Iterable):
        return ()
    ids: list[str] = []
    for item in blocking:
        row = _row(item)
        if (row is not None and row.get("verdict") == "UNKNOWN"
                and row.get("reason") == "MISSING_CHECKS" and row.get("obligation_id")):
            ids.append(str(row["obligation_id"]))
    return tuple(dict.fromkeys(ids))


def authoring_open(runtime: object) -> bool:
    return bool(missing_check_obligations(runtime))


__all__ = ["CHECK_AUTHORING_TOOLS", "authoring_open", "missing_check_obligations"]
