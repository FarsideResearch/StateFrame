from dataclasses import dataclass


@dataclass(frozen=True)
class BatchResult:
    index: int
    item: object
    ok: bool
    value: object
    error: object
