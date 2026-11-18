"""Boundary inputs for unit-level checks of read, parse, page, fit and edit helpers.

The engine-level corpus in enginecheck.boundary runs whole sessions. This module hands the same contents to a
single function, so a helper that receives text or bytes can be checked against every boundary case in one
parametrized test. The contract it checks: the helper returns, or raises one of the exception types it
documents. Any other exception is a crash at the boundary.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import pytest

from enginecheck.boundary import BOUNDARY_CONTENTS, limit_values

# Cases enginecheck does not carry: line-ending and whitespace shapes that broke edit and read helpers before.
EXTRA_CONTENTS: dict[str, bytes] = {
    "lone_cr": b"a = 1\rb = 2\r",
    "bom_crlf": b"\xef\xbb\xbfa = 1\r\nb = 2\r\n",
    "whitespace_only": b"   \t  \n\n",
    "trailing_blank_lines": b"a = 1\n\n\n\n",
    "many_short_lines": b"x\n" * 5000,
}

BOUNDARY_BYTES: dict[str, bytes] = {**BOUNDARY_CONTENTS, **EXTRA_CONTENTS}


def as_text(content: bytes) -> str:
    """Decode the way the engine's readers do when they keep undecodable bytes: surrogateescape, never lossy."""
    return content.decode("utf-8", errors="surrogateescape")


BOUNDARY_TEXTS: dict[str, str] = {name: as_text(content) for name, content in BOUNDARY_BYTES.items()}


def boundary_bytes_params() -> list:
    """pytest params of (name, bytes), ids = case names."""
    return [pytest.param(name, content, id=name) for name, content in BOUNDARY_BYTES.items()]


def boundary_text_params() -> list:
    """pytest params of (name, str), ids = case names."""
    return [pytest.param(name, text, id=name) for name, text in BOUNDARY_TEXTS.items()]


def sized_text(size: int, *, line: int = 80) -> str:
    """ASCII text of exactly `size` characters in lines of at most `line` characters (newline included)."""
    if size <= 0:
        return ""
    row = "y" * (line - 1) + "\n"
    full, rest = divmod(size, line)
    return row * full + "y" * rest


def limit_texts(limit: int) -> dict[str, str]:
    """Texts of 0, 1, limit-1, limit and limit+1 characters, keyed by size label."""
    return {f"size_{value}": sized_text(value) for value in limit_values(limit)}


@dataclass(frozen=True)
class Outcome:
    """What a boundary call did: returned a value, or raised an allowed exception."""

    value: object = None
    raised: BaseException | None = None


def call_bounded(fn: Callable[..., object], *args: object,
                 allowed: Iterable[type[BaseException]] = (), **kwargs: object) -> Outcome:
    """Call fn and return its Outcome. Raises AssertionError naming the case if fn raises anything not allowed.

    `allowed` lists the exception types the helper documents for bad input. Leave it empty when the helper
    promises never to raise.
    """
    allowed_types = tuple(allowed)
    try:
        return Outcome(value=fn(*args, **kwargs))
    except allowed_types as error:
        return Outcome(raised=error)
    except Exception as error:  # the point of the check: report every undocumented exception
        name = getattr(fn, "__qualname__", repr(fn))
        raise AssertionError(f"{name} crashed at a boundary input: {type(error).__name__}: {str(error)!a}") from error
