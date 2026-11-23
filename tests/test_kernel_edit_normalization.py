"""Byte-level precision, work bounds, and publication parity for normalization."""
from __future__ import annotations

from functools import lru_cache
from os.path import commonprefix
from random import Random

import pytest

import kernel_edit_normalization as normalizer


def _normalize(base, expected, **payload):
    path = "module.py"
    result = normalizer.normalize_edit(
        {"allowed_paths": [path]}, base, {"path": path, "candidate": expected, **payload},
    )
    assert isinstance(result, normalizer.NormalizedEdit), result
    assert result.candidate_bytes == normalizer.apply_hunks(base, result.hunks) == expected
    assert result.changed_spans == tuple(
        normalizer.ChangedSpan(hunk.start, hunk.end) for hunk in result.hunks
    )
    return result


def _reference(base, candidate):
    """Find an optimal match path recursively and take the gaps between matches."""
    prefix = len(commonprefix([base, candidate]))
    suffix = len(commonprefix([base[prefix:][::-1], candidate[prefix:][::-1]]))
    old, new = base[prefix:len(base) - suffix], candidate[prefix:len(candidate) - suffix]

    @lru_cache(maxsize=None)
    def matches(i, j):
        if i == len(old) or j == len(new):
            return ()
        if old[i] == new[j]:
            return ((i, j), *matches(i + 1, j + 1))
        return max(matches(i + 1, j), matches(i, j + 1), key=len)

    matched = matches(0, 0)
    anchors = ((-1, -1), *matched, (len(old), len(new)))
    hunks = tuple(
        normalizer.EditHunk(prefix + i + 1, prefix + next_i, old[i + 1:next_i], new[j + 1:next_j])
        for (i, j), (next_i, next_j) in zip(anchors, anchors[1:])
        if next_i != i + 1 or next_j != j + 1
    )
    return hunks, prefix + len(matched) + suffix


def test_repeated_copy_preserves_all_matchable_bytes():
    base, candidate = b"aaa", b"babaab"
    result = _normalize(base, candidate)
    assert result.preservation_proof["unchanged_byte_count"] == 3
    assert result.hunks == (
        normalizer.EditHunk(0, 0, b"", b"b"),
        normalizer.EditHunk(1, 1, b"", b"b"),
        normalizer.EditHunk(3, 3, b"", b"b"),
    )


@pytest.mark.parametrize("seed", [0, 711, 8191])
def test_small_edits_match_exact_reference(seed):
    rng = Random(seed)
    alphabet = b"\x00ab\n\xff"
    for _ in range(200):
        prefix, base, candidate, suffix = (
            bytes(rng.choices(alphabet, k=rng.randrange(size))) for size in (6, 12, 12, 6)
        )
        base, candidate = prefix + base + suffix, prefix + candidate + suffix
        if base == candidate:
            continue
        expected_hunks, unchanged = _reference(base, candidate)
        result = _normalize(base, candidate)
        assert result.hunks == expected_hunks
        assert result.preservation_proof["exact_bytes"] is True
        assert result.preservation_proof["unchanged_byte_count"] == unchanged


class _CountedBytes(bytes):
    def __new__(cls, data, reads):
        value = super().__new__(cls, data)
        value.reads = reads
        return value

    def __getitem__(self, key):
        value = super().__getitem__(key)
        if isinstance(key, slice):
            return _CountedBytes(value, self.reads)
        self.reads[0] += 1
        return value


def _assert_coarse(result, start, old, new):
    assert result.hunks == (normalizer.EditHunk(start, start + len(old), old, new),)
    proof = result.to_dict()["preservation_proof"]
    assert proof["span_precision"] == "coarse"
    assert proof["exact_bytes"] is False
    assert proof["round_trip"] is True
    assert "unchanged_byte_count" not in proof
    if result.mode == "complete-file":
        assert proof["all_hunks_retained"] is False
        assert proof["broad_change"] is True


@pytest.mark.parametrize(("old_size", "coarse"), [(255, False), (256, False), (257, True)])
def test_exact_cell_limit_on_repetitive_middle(old_size, coarse):
    old, new = b"x" + b"a" * (old_size - 2) + b"x", b"y" + b"a" * 254 + b"y"
    prefix, suffix = b"p" * 1024, b"s" * 1024
    reads = [0]
    result = _normalize(_CountedBytes(prefix + old + suffix, reads),
                        _CountedBytes(prefix + new + suffix, reads))
    trim_reads = 2 * (len(prefix) + len(suffix) + 2)
    if coarse:
        _assert_coarse(result, len(prefix), old, new)
        assert reads[0] == trim_reads
    else:
        assert result.preservation_proof["exact_bytes"] is True
        assert result.preservation_proof["unchanged_byte_count"] == len(prefix + suffix) + old_size - 2
        assert trim_reads + len(old) * len(new) <= reads[0] <= 2 * len(old) * len(new) + trim_reads + 2048


@pytest.mark.parametrize("size", [2048, 4096, 8192])
def test_large_repetitive_middle_takes_only_edge_comparisons(size):
    prefix, suffix = b"p" * size, b"s" * size
    old, new = b"x" + b"a" * size + b"x", b"y" + b"a" * size + b"z"
    reads = [0]
    result = _normalize(_CountedBytes(prefix + old + suffix, reads),
                        _CountedBytes(prefix + new + suffix, reads))
    _assert_coarse(result, len(prefix), old, new)
    assert reads[0] == 2 * (len(prefix) + len(suffix) + 2)


@pytest.mark.parametrize("delete", [False, True])
def test_large_one_sided_middle_is_exact(delete):
    middle = b"a" * 70_000
    old, new = (middle, b"") if delete else (b"", middle)
    result = _normalize(b"p" + old + b"s", b"p" + new + b"s")
    assert result.hunks == (normalizer.EditHunk(1, 1 + len(old), old, new),)
    assert result.preservation_proof["exact_bytes"] is True
    assert result.preservation_proof["unchanged_byte_count"] == 2


@pytest.mark.parametrize("large", [False, True])
def test_definition_spans_stay_inside_the_target(large):
    prefix, suffix = b"# header\n", b"\n\nVALUE = 1\n"
    opening, closing = b'def f():\n    return "', b'"'
    shared = b"a" * 300
    old = opening + b"x" + shared + b"x" + closing
    new = opening + b"y" + shared + b"z" + closing if large else old + b"\n"
    result = _normalize(prefix + old + suffix, prefix + new + suffix, candidate=new,
                        mode="complete-definition", supports_definition=True, target={"name": "f"})
    proof = result.preservation_proof
    assert proof["target_span"] == {"start": len(prefix), "end": len(prefix + old)}
    assert proof["exterior_unchanged"] is True
    if large:
        _assert_coarse(result, len(prefix + opening), b"x" + shared + b"x", b"y" + shared + b"z")
    else:
        assert result.hunks == (normalizer.EditHunk(len(prefix + old), len(prefix + old), b"", b"\n"),)
        assert proof["exact_bytes"] is True


@pytest.mark.parametrize("switch", [None, "off", "on"])
@pytest.mark.parametrize("mode", ["complete-file", "complete-definition"])
def test_publication_keeps_candidate_bytes(monkeypatch, tmp_path, switch, mode):
    from probes.react_repair import EDIT_NORMALIZATION_ENV, _normalize_candidate_for_publication

    path = tmp_path / "module.py"
    base = b'# prefix\r\ndef f():\r\n    return 1\r\n# suffix\r\n'
    replacement = b'def f():\r\n    return 2'
    composed = base.replace(b"def f():\r\n    return 1", replacement)
    if switch == "on":
        path.write_bytes(base)
    else:
        # Disabled publication succeeds without reading the authoritative file.
        assert not path.exists()
    if switch is None:
        monkeypatch.delenv(EDIT_NORMALIZATION_ENV, raising=False)
    else:
        monkeypatch.setenv(EDIT_NORMALIZATION_ENV, switch)
    definition = mode == "complete-definition"
    assert _normalize_candidate_for_publication(
        path, replacement if definition else composed, mode=mode, root=tmp_path,
        target="f" if definition else None, composed_candidate=composed if definition else None,
    ) == (composed, None)
