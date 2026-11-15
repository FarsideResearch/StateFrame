"""I-6 integration contract for crew-11; NO HOOK PRESENT in this base tree.

When a production view ladder is merged, its entry point must consult
kernel_read_receipts.enabled(), whose switch is STATEFRAME_WORKING_SET_ALLOCATOR.
Allocator ON owns source membership: return the original body before either the
pre-trim selector or apply_views can run, regardless of the ladder's own switch.
Allocator OFF leaves the ladder's own switch and pre-trim/apply order intact.

This helper is test-only specification, not evidence of a production hook guard.
The merged hook needs an integration test comparing ON/ladder-on with
ON/ladder-off provider bytes through the real qualification path.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import kernel_read_receipts

Body = TypeVar("Body")


def apply_view_ladder_contract(
    body: Body,
    *,
    ladder_enabled: bool,
    pre_trim: Callable[[Body], Body],
    apply_views: Callable[[Body], Body],
) -> Body:
    """Specify a strict allocator no-op encompassing both ladder mutations."""
    if kernel_read_receipts.enabled() or not ladder_enabled:
        return body
    return apply_views(pre_trim(body))
