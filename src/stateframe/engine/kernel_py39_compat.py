"""Small Python 3.9 compatibility boundary for optional v46 modules."""

from __future__ import annotations

import dataclasses
import enum
import sys
import typing


def install() -> None:
    """Provide only stdlib features used by v46 that arrived after Python 3.9."""
    if sys.version_info >= (3, 10):
        return

    if not hasattr(enum, "StrEnum"):
        class StrEnum(str, enum.Enum):
            def __str__(self) -> str:
                return self.value

        enum.StrEnum = StrEnum

    if not hasattr(typing, "TypeAlias"):
        typing.TypeAlias = object

    if getattr(dataclasses.dataclass, "_stateframe_py39", False):
        return
    original = dataclasses.dataclass

    def compatible_dataclass(cls=None, /, **kwargs):
        kwargs.pop("slots", None)
        kwargs.pop("weakref_slot", None)
        if cls is None:
            return lambda target: original(target, **kwargs)
        return original(cls, **kwargs)

    compatible_dataclass._stateframe_py39 = True
    dataclasses.dataclass = compatible_dataclass


install()
