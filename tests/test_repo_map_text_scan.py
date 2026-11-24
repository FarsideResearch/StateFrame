"""The text scan makes one prefix match per line and keeps its exact output."""
from __future__ import annotations

import random

import pytest

from stateframe.engine.repo_intel_core import repo_map

PREFIXES = {
    ".py": ("def ", "async def ", "class "),
    **dict.fromkeys((".ts", ".tsx", ".js", ".jsx"), (
        "export ", "function ", "class ", "interface ", "type ", "const ", "async function ")),
}


def _reference(source, ext):
    sigs = []
    for line in source.splitlines():
        stripped = line.strip()
        if any(stripped.startswith(k) for k in PREFIXES.get(ext, ())):
            sigs.append(stripped[:repo_map._MAX_SIG_CHARS])
        if len(sigs) >= repo_map._MAX_SIGS_PER_FILE:
            break
    return sigs


@pytest.mark.parametrize("ext", [*PREFIXES, ".vue", ".go", ""])
def test_text_scan_matches_reference(ext):
    rng = random.Random(7)
    words = ["def ", "async def ", "class ", "export ", "const ", "type", "x = 1", "", "  ",
             " class ", "async function ", "function", "é", "n" * 400, "\x0c"]
    for size in (0, 1, 5, 40, 400):
        source = "".join(rng.choice(words) + rng.choice(words) + rng.choice(["\n", "\r\n", " "])
                         for _ in range(size))
        assert repo_map._fallback_extract_sigs(source, ext) == _reference(source, ext)


@pytest.mark.parametrize("ext", list(PREFIXES))
def test_text_scan_makes_one_match_per_line(ext):
    class Line(str):
        calls = 0

        def strip(self, *args):
            return self

        def startswith(self, *args):
            Line.calls += 1
            return str.startswith(self, *args)

    class Source(str):
        def splitlines(self, *args):
            return [Line("value += 1")] * 50

    assert repo_map._fallback_extract_sigs(Source(), ext) == []
    assert Line.calls == 50
