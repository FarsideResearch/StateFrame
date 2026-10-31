import os

import pytest

from kernel_job_output import JobOutputCollector, LiteralMatcher


def test_literal_boundaries_and_final_utf8(tmp_path):
    collector = JobOutputCollector(tmp_path / "ring", 32)
    matcher = collector.watch("ready €")
    collector.feed(b"xxxrea")
    collector.feed(b"dy \xe2\x82")
    assert not matcher.matched
    collector.feed(b"\xac")
    collector.finish()
    assert matcher.matched
    collector.close()


def test_ring_overflow_and_disk_bound(tmp_path):
    collector = JobOutputCollector(tmp_path / "ring", 17)
    payload = bytes(range(256)) * 1000
    collector.feed(payload)
    assert collector.snapshot() == payload[-17:]
    assert collector.total_bytes == len(payload)
    assert collector.lost_bytes == len(payload) - 17
    data, cursor, lost = collector.read_since(0)
    assert data == payload[-17:] and cursor == len(payload) and lost == len(payload) - 17
    assert (tmp_path / "ring").stat().st_size == 17
    collector.close()


def test_single_character_matcher_is_bounded():
    matcher = LiteralMatcher("z")
    for _ in range(100):
        matcher.feed("x" * 65536)
    assert matcher._tail == ""
    assert not matcher.matched
    assert matcher.feed("z")


def test_nonblocking_match_before_exit(tmp_path):
    read, write = os.pipe()
    collector = JobOutputCollector(tmp_path / "ring")
    matcher = collector.watch("READY")
    try:
        assert not collector.drain(read)
        os.write(write, b"READY")
        os.close(write)
        write = -1
        assert collector.drain(read)
        assert matcher.matched
    finally:
        os.close(read)
        if write >= 0:
            os.close(write)
        collector.close()


def test_newline_free_fifty_megabytes(tmp_path):
    collector = JobOutputCollector(tmp_path / "ring")
    matcher = collector.watch("finished")
    for _ in range(800):
        collector.feed(b"x" * 65536)
    collector.feed(b"finished")
    assert matcher.matched
    assert collector.total_bytes == 50 * 1024 * 1024 + 8
    assert len(collector.snapshot()) == 1 << 20
    assert (tmp_path / "ring").stat().st_size == 1 << 20
    collector.close()


def test_matcher_and_cursor_validation(tmp_path):
    with pytest.raises(ValueError):
        LiteralMatcher("")
    collector = JobOutputCollector(tmp_path / "ring")
    with pytest.raises(ValueError):
        collector.read_since(1)
    collector.close()


def test_sealed_ring_reopens_in_chronological_order(tmp_path):
    collector = JobOutputCollector(tmp_path / "ring", 17)
    collector.feed(b"0123456789abcdefghijklmnopqrstuv")
    expected = collector.snapshot()
    collector.seal()
    assert not collector._ring and collector._file.closed
    assert collector.snapshot() == expected
    assert collector.read_since(0)[0] == expected
    collector.close()
