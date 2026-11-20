"""Reject a private product label in any shipped text, regardless of punctuation."""

from __future__ import annotations

from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor
import hashlib
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
_TARGET_LENGTH = 14
_TARGET_DIGEST = "ee8b0b1a6855e94df36492e9d05962d9d0a3aa75d861b03355e4851e94bc5698"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _matching_lines(text: str, length: int, digest: str, first_word_length: int) -> list[int]:
    """Match a normalized digest at source word starts with allowed separators."""
    normalized_lines: list[str] = []
    source_positions: list[int] = []
    line_starts = [0]
    source_offset = 0
    for source_line in text.splitlines(keepends=True):
        normalized_chars: list[str] = []
        for index, char in enumerate(source_line):
            if char.isalnum():
                lowered = char.lower()
                normalized_chars.extend(lowered)
                source_positions.extend([source_offset + index] * len(lowered))
        normalized_line = "".join(normalized_chars)
        normalized_lines.append(normalized_line)
        line_starts.append(line_starts[-1] + len(normalized_line))
        source_offset += len(source_line)
    normalized_text = "".join(normalized_lines)
    target_digest = bytes.fromhex(digest)
    matches: list[int] = []
    for start in range(len(normalized_text) - length + 1):
        window = normalized_text[start : start + length]
        if hashlib.sha256(window.encode("utf-8")).digest() == target_digest:
            source_start = source_positions[start]
            if source_start:
                previous = text[source_start - 1]
                first = text[source_start]
                if previous.isalnum() and not (previous.islower() and first.isupper()):
                    continue
            split = start + first_word_length
            if any(
                source_positions[index + 1] != source_positions[index] + 1
                for index in range(start, start + length - 1)
                if index != split - 1
            ):
                continue
            separator = text[source_positions[split - 1] + 1 : source_positions[split]]
            if any(not (char.isspace() or char in "-_./") for char in separator):
                continue
            matches.append(bisect_right(line_starts, start))
    return matches


def _shipped_paths():
    for directory in ("src", "docs"):
        for path in sorted((ROOT / directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                yield path
    yield ROOT / "pyproject.toml"
    yield from sorted(path for path in ROOT.glob("README*") if path.is_file())


def _scan_file(path: Path) -> list[str]:
    data = path.read_bytes()
    if b"\x00" in data:
        return []
    try:
        content = data.decode("utf-8")
    except UnicodeDecodeError:
        return []
    return [
        f"{path.relative_to(ROOT)}:{line}: sha256:{_TARGET_DIGEST[:12]}"
        for line in _matching_lines(content, _TARGET_LENGTH, _TARGET_DIGEST, 7)
    ]


def test_normalized_digest_matches_separator_and_identifier_forms():
    target = "alphabeta"
    digest = _digest(target)
    for text in (
        "alpha_beta_x", "x_alpha_beta", "AlphaBeta", "fooAlphaBeta",
        "AlphaBetaTest", "alpha-beta", "alpha beta", "alpha.beta", "alpha/beta",
    ):
        assert _matching_lines(text, len(target), digest, 5) == [1]
    assert _matching_lines("x\nalpha_beta", len(target), digest, 5) == [2]
    assert _matching_lines("alpha\nbeta", len(target), digest, 5) == [1]
    for text in ("galpha beta", "galpha-beta", "alpha gamma beta"):
        assert _matching_lines(text, len(target), digest, 5) == []


def test_namespace_fallbacks_use_neutral_sample_name():
    import kernel_semantics

    assert kernel_semantics.NamespaceFacts().base_name == "sample_test_f2"
    context = kernel_semantics._SourceContext(None)
    assert kernel_semantics._namespace_analysis(context, None).prefix == "sample_test_f2"


def test_shipped_text_has_no_private_product_label():
    paths = list(_shipped_paths())
    assert ROOT / "src/stateframe/engine/ENGINE_SOURCE.json" in paths
    worker_count = min(8, len(paths), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=worker_count) as workers:
        hits = [
            hit
            for file_hits in workers.map(_scan_file, paths, chunksize=4)
            for hit in file_hits
        ]
    assert not hits, "Private label in shipped text:\n" + "\n".join(hits)
