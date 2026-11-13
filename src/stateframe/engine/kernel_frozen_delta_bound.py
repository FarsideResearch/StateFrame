"""Conservative segment accounting for an exact-counted final layout in an outage."""
from __future__ import annotations

import json
import math
from collections.abc import Mapping

from kernel_frame_allocation_capture import FrozenRequestCounter, _calibrated_ratio
from kernel_frame_optional_allocation import identity
from kernel_request_diet import FALLBACK_MAX_TOKENS_PER_WIRE_BYTE

# tests/tools/frozen_delta_splices.py, seed 12059, 20,000 replacements/backend:
# deterministic byte BPE max 16 extra tokens; offline llama.cpp GPT-2 max 6,
# Llama BPE max 5, DeepSeek Coder max 4. No weights or network were used.
# Charge the full two-ended replacement excess per point, then double it.
FROZEN_DELTA_SEAM_TOKENS = 32


def _wire_segment(row: Mapping[str, object]) -> int:
    # Include escaping, message metadata and an array separator in every new
    # segment. Unlike net growth, a smaller replacement pays its full new size.
    return len(json.dumps(row, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")).encode("utf-8")) + 1


def _segments(base: Mapping[str, object], projected: Mapping[str, object]
              ) -> tuple[int, int] | None:
    if ({key: value for key, value in base.items() if key != "messages"}
            != {key: value for key, value in projected.items() if key != "messages"}):
        return None
    before, after = base.get("messages"), projected.get("messages")
    if (not isinstance(before, list) or not isinstance(after, list)
            or not before or not after
            or any(not isinstance(row, Mapping) for row in [*before, *after])):
        return None

    # The final layout has a lossless, validated index map. Recognize ONLY the
    # exact renderer output for this frozen base, never an arbitrary permutation
    # or a merely similar authority overlay. Retained source carriers are whole
    # segments even when the renderer moves the frame past them.
    from kernel_cache_layout_view import FINAL_LAYOUT_NAME, semantic_index_map
    if after[-1].get("name") == FINAL_LAYOUT_NAME:
        from kernel_cache_layout_candidate import (
            FinalCacheLayoutProjectionError,
            _checked_final_cache_projection,
        )
        try:
            if projected != _checked_final_cache_projection(base):
                return None
            indexes = semantic_index_map(after)
        except (FinalCacheLayoutProjectionError, ValueError, TypeError, KeyError, IndexError):
            return None
        retained = {index: indexes[index] for index, row in enumerate(after)
                    if index in indexes and row == before[indexes[index]]}
        changed = [row for index, row in enumerate(after) if index not in retained]
        removed = len(before) - len(retained)
        moved = sum(index != old_index for index, old_index in retained.items())
        return sum(_wire_segment(row) for row in changed), 2 * (len(changed) + removed + moved)

    # Ordinary projections may append messages or replace string-content
    # segments in place. Schema, role, tools, deletions and reorderings are not
    # delta-accountable here and retain the whole-body calibrated guard.
    if len(after) < len(before):
        return None
    delta_bytes = 0
    replacements = 0
    original_rows = {identity(row) for row in before}
    for index, old in enumerate(before):
        new = after[index]
        if new == old:
            continue
        if (not isinstance(old.get("content"), str)
                or not isinstance(new.get("content"), str)
                or {key: value for key, value in old.items() if key != "content"}
                != {key: value for key, value in new.items() if key != "content"}
                or identity(new) in original_rows):
            return None
        delta_bytes += _wire_segment(new)
        replacements += 1
    appended = after[len(before):]
    delta_bytes += sum(_wire_segment(row) for row in appended)
    return delta_bytes, 2 * (replacements + len(appended))


def frozen_delta_bound(base: Mapping[str, object], projected: Mapping[str, object],
                       meter: object, frozen: FrozenRequestCounter
                       ) -> dict[str, int | float | str] | None:
    """Return a bound only when the caller still holds the exact frozen base.

    Observed mean or minimum density cannot bound unseen dense text. Cap the
    ratio by the meter's existing conservative token/byte envelope as well as
    every available run minimum. Retained bytes keep their frozen exact count;
    only added/replaced UTF-8 segments pay this bound and the splice margin.
    """
    if (frozen.method != "tokenize" or identity(base) != frozen.body_digest
            or getattr(meter, "measure_method", None) != "calibrated_bytes"):
        return None
    segments = _segments(base, projected)
    if segments is None:
        return None
    ratios = [frozen.ratio, _calibrated_ratio(meter),
              1 / FALLBACK_MAX_TOKENS_PER_WIRE_BYTE]
    observed = getattr(meter, "_minimum_bytes_per_token", None)
    if observed is not None:
        ratios.append(observed)
    if any(not isinstance(ratio, (int, float)) or isinstance(ratio, bool)
           or not math.isfinite(ratio) or ratio <= 0 for ratio in ratios):
        return None
    ratio = min(ratios)
    delta_bytes, splices = segments
    delta_tokens = math.ceil(delta_bytes / ratio)
    seam_tokens = splices * FROZEN_DELTA_SEAM_TOKENS
    return {"path": "delta_bound", "frozen_tokens": frozen.tokens,
            "delta_bytes": delta_bytes, "conservative_ratio": ratio,
            "delta_tokens": delta_tokens, "splice_points": splices,
            "seam_tokens": seam_tokens,
            "prompt_tokens": frozen.tokens + delta_tokens + seam_tokens}
