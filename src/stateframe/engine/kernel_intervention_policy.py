"""Explicit default-on adaptive interventions, separate from passive state/audit."""
import os


def interventions_enabled() -> bool:
    value = os.environ.get("KERNEL_INTERVENTIONS", "1")
    if value not in {"0", "1"}:
        raise ValueError("KERNEL_INTERVENTIONS must be 0 or 1")
    return value == "1"
