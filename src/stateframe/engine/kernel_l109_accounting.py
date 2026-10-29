"""Bounded late-stage request accounting for host-backed verification.

The reducer records settled work, not screen renders.  A receipt can make a
check/edit productive; a bounded pending check can reserve capacity.  A
repeated refusal or render cannot manufacture progress.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence


SCHEMA = "kernel-l109-accounting/1"


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class LateStageAccounting:
    """Small host-owned meter with idempotent settled request/action receipts."""

    def __init__(self, *, capacity: int = 1, pending_limit: int = 8) -> None:
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        if type(pending_limit) is not int or pending_limit < 0:
            raise ValueError("pending_limit must be a non-negative integer")
        self.capacity = capacity
        self.pending_limit = pending_limit
        self.settled_requests: set[str] = set()
        self.settled_actions: set[str] = set()
        self.receipts: set[str] = set()
        self.pending = 0
        self.productive = 0

    def settle(
        self,
        request_id: object,
        *,
        action: str,
        receipt_ids: Sequence[object] = (),
        pending_checks: int = 0,
        refusal: bool = False,
    ) -> dict[str, object]:
        request = str(request_id).strip()
        action_name = str(action).strip().lower()
        if not request or not action_name:
            raise ValueError("request_id and action are required")
        if type(pending_checks) is not int or pending_checks < 0:
            raise ValueError("pending_checks must be a non-negative integer")
        key = _digest({"request": request, "action": action_name})
        if request in self.settled_requests or key in self.settled_actions:
            return self.snapshot(productive=False, duplicate=True, request_id=request)
        fresh_receipts = {
            str(value).strip() for value in receipt_ids
            if str(value).strip() and str(value).strip() not in self.receipts
        }
        reserve = min(pending_checks, self.pending_limit)
        productive = bool(fresh_receipts) or (not refusal and action_name in {"check", "edit"})
        if reserve:
            self.pending = min(self.pending_limit, self.pending + reserve)
        if fresh_receipts:
            self.pending = max(0, self.pending - len(fresh_receipts))
        self.settled_requests.add(request)
        self.settled_actions.add(key)
        self.receipts.update(fresh_receipts)
        if productive:
            self.productive += 1
        return self.snapshot(
            productive=productive, duplicate=False, request_id=request,
            fresh_receipts=tuple(sorted(fresh_receipts)), reserved=reserve,
        )

    def snapshot(self, *, productive: bool = False, duplicate: bool = False,
                 request_id: str = "", fresh_receipts: Sequence[str] = (),
                 reserved: int = 0) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "productive": productive,
            "duplicate": duplicate,
            "request_id": request_id,
            "fresh_receipts": list(fresh_receipts),
            "reserved_pending": reserved,
            "pending": self.pending,
            "productive_count": self.productive,
            "capacity": self.capacity,
        }


__all__ = ["LateStageAccounting", "SCHEMA"]
