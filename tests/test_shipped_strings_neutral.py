"""Reject a private product label in any shipped text, regardless of punctuation."""

from __future__ import annotations

from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor
import hashlib
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
_TARGET_LENGTH = 14
_TARGET_DIGEST = "ee8b0b1a6855e94df36492e9d05962d9d0a3aa75d861b03355e4851e94bc5698"
_TARGETS = (
    (_TARGET_LENGTH, _TARGET_DIGEST, 7, False),
    (10, "d4f70cba07334e84f3368b8056d5e05a48deefb7adb141684238efc4658475dd", 3, False),
    (12, "ef8453bef6cee2ae3b7aa720b3594e26f071b35237d19e7c2e80a12e97a6fb46", (6, 8), False),
    (11, "b0aba49976a70ff280a7e9b8785b29d51cf615c7701bc3018c877c7092aafdd1", 5, False),
    (9, "6eed3508eee654f48cc4d57910ead9310e4b2b386248d56bd40bbf16fcd9a77f", None, True),
    (4, "5f4f045d317c6bb4cce5e8c489dc0e6d8a31522ec0eaff5310d58c74033e7966", None, True),
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _matching_lines(
        text: str, length: int, digest: str,
        first_word_length: int | tuple[int, ...] | None,
        trailing_underscore: bool = False,
) -> list[int]:
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
            splits = (
                (first_word_length,) if isinstance(first_word_length, int)
                else first_word_length or ()
            )
            split_indexes = {start + offset for offset in splits}
            if any(
                source_positions[index + 1] != source_positions[index] + 1
                for index in range(start, start + length - 1)
                if index + 1 not in split_indexes
            ):
                continue
            for offset in splits:
                split = start + offset
                separator = text[source_positions[split - 1] + 1 : source_positions[split]]
                if any(not (char.isspace() or char in "-_./") for char in separator):
                    break
            else:
                if trailing_underscore:
                    following = source_positions[start + length - 1] + 1
                    if following >= len(text) or text[following] != "_":
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
        f"{path.relative_to(ROOT)}:{line}: sha256:{digest[:12]}"
        for length, digest, first_word_length, trailing_underscore in _TARGETS
        for line in _matching_lines(content, length, digest, first_word_length,
                                    trailing_underscore)
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


def test_digest_prefix_matcher_preserves_prefix_length():
    import kernel_ripple_preview

    synthetic = frozenset({(6, _digest("alpha_"))})
    assert kernel_ripple_preview._matches_digest_prefix("alpha_job", synthetic)
    assert kernel_ripple_preview._matches_digest_prefix("alpha_", synthetic)
    assert not kernel_ripple_preview._matches_digest_prefix("alpha", synthetic)
    assert not kernel_ripple_preview._matches_digest_prefix("beta_job", synthetic)
    assert not kernel_ripple_preview._matches_digest_prefix("\udcff_job", synthetic)
    assert sorted(length for length, _ in kernel_ripple_preview._EXCLUDED_BIN_PREFIX_DIGESTS) == [5, 10]


def test_private_path_part_stays_excluded():
    import kernel_authored_check_context

    part = bytes((98, 101, 108, 116, 95, 113, 117, 101, 117, 101)).decode("ascii")
    assert _digest(part) in kernel_authored_check_context._FORBIDDEN_PART_DIGESTS
    for candidate in (part, part.upper()):
        with pytest.raises(kernel_authored_check_context.ContextError):
            kernel_authored_check_context._path(f"src/{candidate}/unit.py")
    assert kernel_authored_check_context._path("src/public/unit.py") == "src/public/unit.py"
    assert kernel_authored_check_context._path("src/\udcff/unit.py") == "src/\udcff/unit.py"


def test_rule_discovery_accepts_neutral_keyword(tmp_path):
    from stateframe.engine.detreason.premortem import discover_rule_ids

    rules = tmp_path / "rules.txt"
    rules.write_text("### 10. Later\n### 2. Earlier\n### 2. Duplicate\n", encoding="utf-8")
    assert discover_rule_ids(rules_path=str(rules)) == ("2", "10")


def test_digest_scanner_requires_prefix_boundary():
    digest = _digest("alpha")
    assert _matching_lines("alpha_task", 5, digest, None, True) == [1]
    assert _matching_lines("x_alpha_task", 5, digest, None, True) == [1]
    for text in ("alphabet", "alpha-task", "xalpha_task"):
        assert _matching_lines(text, 5, digest, None, True) == []
    compound = _digest("alphabetagamma")
    assert _matching_lines("alpha_beta_gamma", 14, compound, (5, 9)) == [1]
    assert _matching_lines("alpha_beta delta", 14, compound, (5, 9)) == []


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
