"""Minimal unit 2 stand-in for unit 1's kernel_execution_boundary module.

The isolated engine seam can use this when the real module is not present yet.
Keep the public names and signatures aligned with the fixed boundary interface.
"""
from dataclasses import dataclass

BACKEND_PLATFORM = "platform"
BACKEND_CONTAINER = "container"
BOUNDARY_STATEMENT = "test container boundary"


@dataclass(frozen=True)
class Detection:
    classification: str
    signals: tuple[str, ...]
    namespaces: dict[str, str]
    detector_version: int


class BoundaryRefused(Exception):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class Boundary:
    backend: str
    detection: Detection

    def record(self) -> dict:
        return {"backend": self.backend}


_active = None


def activate(boundary: Boundary) -> None:
    global _active
    if _active is not None:
        raise RuntimeError("execution boundary already activated")
    _active = boundary


def active() -> Boundary:
    return _active or Boundary(BACKEND_PLATFORM, Detection("not_container", (), {}, 1))
