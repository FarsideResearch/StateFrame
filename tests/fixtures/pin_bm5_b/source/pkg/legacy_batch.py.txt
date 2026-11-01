from dataclasses import dataclass


@dataclass(frozen=True)
class BatchResult:
    """Legacy dashboard summary; it is not an execution result."""

    label: str
    completed: int
