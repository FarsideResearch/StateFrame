"""Content-free stable-wire trim history, shared by the engine and tape loader."""
from __future__ import annotations

from collections.abc import Mapping
import re

# Trims a stable-wire send may ask for before it falls back to the plain layout.
MAX_STABLE_TRIMS = 3
MAX_TRIM_CANDIDATES = 20_000


def validate_reserves_record(value: object) -> dict[str, object] | None:
    """Copy a bounded cumulative trim record; None denotes an older tape."""
    if value is None:
        return None
    if (not isinstance(value, Mapping)
            or not {"reserves", "trim_count", "epoch"} <= set(value)
            or set(value) - {"reserves", "trim_count", "epoch", "selections", "initial_selection"}
            or type(value["trim_count"]) is not int
            or not 0 <= value["trim_count"] <= MAX_STABLE_TRIMS
            or type(value["epoch"]) is not int or value["epoch"] < 1
            or not isinstance(value["reserves"], list)
            or len(value["reserves"]) != value["trim_count"]):
        raise ValueError("invalid stable wire reserves record")
    previous = [0, 0]
    reserves = []
    for row in value["reserves"]:
        if (not isinstance(row, list) or len(row) != 2
                or any(type(number) is not int or number < 0 for number in row)
                or any(number < before for number, before in zip(row, previous))):
            raise ValueError("invalid stable wire cumulative reserves")
        reserves.append(list(row))
        previous = row
    result = {"reserves": reserves, "trim_count": value["trim_count"], "epoch": value["epoch"]}
    if "selections" in value:
        selections = value["selections"]
        if not isinstance(selections, list) or len(selections) != len(reserves):
            raise ValueError("invalid stable wire trim selections")
        copied = []
        for selection in selections:
            if not isinstance(selection, list) or len(selection) > MAX_TRIM_CANDIDATES:
                raise ValueError("invalid stable wire trim selection")
            rows = []
            seen = set()
            for row in selection:
                if (not isinstance(row, Mapping)
                        or set(row) != {"kind", "index", "candidate_id"}
                        or row["kind"] != "message"
                        or type(row["index"]) is not int or row["index"] < 0
                        or row["index"] in seen
                        or not isinstance(row["candidate_id"], str)
                        or re.fullmatch(r"sha256:[0-9a-f]{64}", row["candidate_id"]) is None):
                    raise ValueError("invalid stable wire trim candidate")
                seen.add(row["index"])
                rows.append(dict(row))
            copied.append(rows)
        result["selections"] = copied
    if "initial_selection" in value:
        try:
            from .kernel_stable_wire_initial import validate_initial_selection
        except ImportError:
            from kernel_stable_wire_initial import validate_initial_selection
        result["initial_selection"] = validate_initial_selection(value["initial_selection"])
    return result
