"""Quote paths and regular expressions for macOS sandbox profiles (SBPL)."""
from __future__ import annotations

import json
from pathlib import Path


_ERE_METACHARACTERS = frozenset(".[]()*+?{}|^$\\")


def _check_text(text: str) -> None:
    if not isinstance(text, str):
        raise TypeError("SBPL text must be a string")
    if any(ord(char) < 0x20 or ord(char) == 0x7f or 0xd800 <= ord(char) <= 0xdfff
           for char in text):
        raise ValueError("SBPL text contains a control character or unpaired surrogate")


def sbpl_string(text: str) -> str:
    """Return an ordinary SBPL string, with UTF-8 characters left intact."""
    _check_text(text)
    return json.dumps(text, ensure_ascii=False)


def sbpl_regex_literal(pattern: str) -> str:
    """Return a raw SBPL regex literal; its backslashes are regex syntax."""
    _check_text(pattern)
    if '"' in pattern:
        raise ValueError("SBPL raw regex cannot safely contain a double quote")
    return '#"' + pattern + '"'


def sbpl_path_regex(prefix_path: str | Path, suffix_regex: str) -> str:
    """Anchor a literal path prefix to a trusted POSIX ERE suffix."""
    prefix = str(prefix_path)
    _check_text(prefix)
    escaped = "".join("\\" + char if char in _ERE_METACHARACTERS else char
                      for char in prefix)
    return sbpl_regex_literal("^" + escaped + suffix_regex)
