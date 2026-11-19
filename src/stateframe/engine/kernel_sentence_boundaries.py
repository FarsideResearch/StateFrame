"""Shared punctuation boundaries for ticket and obligation sentence splitting."""
from __future__ import annotations


_TERMINALS = frozenset(".!?")
_CLOSERS = frozenset(")]}'\"")
# These forms introduce the text that follows them, so their final period is
# internal even when the next word is capitalized (for example, ``e.g. JSON``).
_NONTERMINAL_ABBREVIATIONS = frozenset({
    "a.k.a", "approx", "cf", "e.g", "i.e", "incl", "viz", "vs", "w.r.t",
})


def _opens_quote(text: str, index: int, char: str) -> bool:
    if char != "'" or index == 0 or index + 1 >= len(text):
        return True
    return not (text[index - 1].isalnum() and text[index + 1].isalnum())


def _is_nonterminal_abbreviation(text: str, terminal_index: int) -> bool:
    """Match complete abbreviations, never suffixes inside ordinary words."""
    start = terminal_index
    while start > 0 and (text[start - 1].isalnum() or text[start - 1] in "._"):
        start -= 1
    return text[start:terminal_index].lower() in _NONTERMINAL_ABBREVIATIONS


def sentence_ranges(text: str) -> tuple[tuple[int, int], ...]:
    """Return source ranges ending at terminal runs followed by whitespace or EOF.

    Punctuation inside backticks or quotes and digit-dot-digit decimals stays
    untouched. Closing brackets and quotes immediately after a terminal run
    remain part of that sentence. Periods inside dotted references, URLs, and
    complete nonterminal abbreviations do not create boundaries.
    """
    ranges: list[tuple[int, int]] = []
    start = 0
    in_backticks = False
    quote = ""
    index = 0

    while index < len(text):
        char = text[index]
        if char == "`":
            in_backticks = not in_backticks
            index += 1
            continue
        if in_backticks:
            index += 1
            continue

        if quote:
            if char == quote and (index == 0 or text[index - 1] != "\\"):
                quote = ""
            index += 1
            continue
        if char in "'\"" and _opens_quote(text, index, char):
            quote = char
            index += 1
            continue

        if char not in _TERMINALS:
            index += 1
            continue

        if (char == "." and index > 0 and index + 1 < len(text)
                and text[index - 1].isdigit() and text[index + 1].isdigit()):
            index += 1
            continue

        terminal_end = index + 1
        while terminal_end < len(text) and text[terminal_end] in _TERMINALS:
            terminal_end += 1
        closing_end = terminal_end
        while closing_end < len(text) and text[closing_end] in _CLOSERS:
            closing_end += 1

        followed_by_boundary = closing_end == len(text) or text[closing_end].isspace()
        single_abbreviation_period = terminal_end == index + 1 and char == "."
        abbreviation_continues = (single_abbreviation_period
                                  and _is_nonterminal_abbreviation(text, index))
        if followed_by_boundary and not abbreviation_continues:
            if text[start:closing_end].strip():
                ranges.append((start, closing_end))
            start = closing_end

        # A closing quote after punctuation is a closer even when adjacent
        # text means the punctuation does not end the sentence.
        index = closing_end

    if text[start:].strip():
        ranges.append((start, len(text)))
    return tuple(ranges)
