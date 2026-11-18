"""Regression coverage for observation noise and bounded investigation."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from kernel_tool_call_loop_breaker import UnboundedProgressWatch, _digest, stop_unbounded_no_progress


@pytest.mark.parametrize("mode", ["off", "on"])
@pytest.mark.parametrize("kind", ["shell", "web_fetch", "read"])
def test_volatile_output_stalls(mode, kind, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", mode)
    watch = UnboundedProgressWatch()
    receipts = []
    for index in range(33):
        text = (f"2026-11-18T12:00:00.{index:06d}Z pid={index + 1} "
                f"duration={index}.1ms /tmp/tmp{index}/result status=unchanged")
        receipts.append({"kind": kind, "target": "same", "detail": text,
                         "fingerprint": hashlib.sha256(text.encode()).hexdigest()[:16],
                         "result_class": "HOST_OBSERVED" if kind == "read" else "NO_OP"})
        assert watch.observe({"product_revision": "r1"}, receipts, boundary=index) is (index == 32)


def test_normalization_keeps_semantic_numbers():
    for left, right in [("exit=1", "exit=2"), ("a.py:12", "a.py:13"),
                        ("expected 7 got 8", "expected 7 got 9"),
                        ("version 1.2", "version 1.3")]:
        assert _digest(left) != _digest(right)
    assert _digest("Date: Wed, 18 Nov 2026 12:00:01 GMT") == _digest("Date: Wed, 18 Nov 2026 12:00:02 GMT")


def test_shell_receipt_preserves_full_normalized_evidence():
    from kernel_bolt import KernelBolt
    bolt = KernelBolt.__new__(KernelBolt)
    bolt._receipts = []
    prefix = "same preamble " * 20
    for suffix in ("status=waiting pid=1", "status=waiting pid=2", "status=ready pid=3"):
        text = prefix + suffix
        bolt.receipt("shell", "shell", text, "NO_OP", text)
    rows = bolt._receipts
    assert rows[0]["detail"] == rows[2]["detail"]
    assert rows[0]["fingerprint"] != rows[1]["fingerprint"]
    assert rows[0]["progress_output_digest"] == rows[1]["progress_output_digest"]
    assert rows[0]["progress_output_digest"] != rows[2]["progress_output_digest"]
    watch = UnboundedProgressWatch()
    watch.observe({"product_revision": "r1"}, rows[:1], boundary=0)
    watch.observe({"product_revision": "r1"}, rows[:2], boundary=1)
    assert watch.streak == 1
    watch.observe({"product_revision": "r1"}, rows, boundary=2)
    assert watch.streak == 0


@pytest.mark.parametrize("mode", ["off", "on"])
@pytest.mark.parametrize("kind", ["shell", "web", "web_fetch", "web_search", "fetch", "read"])
@pytest.mark.parametrize("template", [
    "2026-11-18T12:00:{index:02d}Z status=waiting",
    "Date: Wed, 18 Nov 2026 12:00:{index:02d} GMT\nstatus=waiting",
    "duration={index}.1ms status=waiting",
    "elapsed: {index}.5 seconds status=waiting",
    "pid={index} ppid={index} status=waiting",
    "/tmp/tmp{index}/result.txt status=waiting",
    "/var/tmp/tmp{index}/result.txt status=waiting",
    "request_id=0x{index:08x} status=waiting",
    "trace_id={index:032x} status=waiting",
    "port={port} status=waiting",
    "http://127.0.0.1:{port}/health status=waiting",
    "http://localhost:{port}/health status=waiting",
    "http://[::1]:{port}/health status=waiting",
])
def test_each_volatile_token_charges_streak(mode, kind, template, monkeypatch):
    """The same operation cannot turn runtime noise into fresh evidence."""
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", mode)
    watch = UnboundedProgressWatch(threshold=3)
    receipts = []
    for index in range(4):
        text = template.format(index=index + 1, port=40001 + index)
        receipts.append({
            "kind": kind, "target": "same-operation", "detail": text,
            "fingerprint": hashlib.sha256(text.encode()).hexdigest()[:16],
            "result_class": "HOST_OBSERVED" if kind == "read" else "NO_OP",
        })
        assert watch.observe({"product_revision": "r1"}, receipts, boundary=index) is (index == 3)
        assert watch.streak == index


@pytest.mark.parametrize("kind", ["shell", "web", "web_fetch", "web_search", "fetch"])
def test_full_receipt_noise_charges_and_semantic_tail_resets(kind):
    """Exercise the real producer, including bytes outside its display preview."""
    from kernel_bolt import KernelBolt

    bolt = KernelBolt.__new__(KernelBolt)
    bolt._receipts = []
    watch = UnboundedProgressWatch(threshold=3)
    prefix = "stable preamble " * 20
    for index in range(4):
        status = "ready" if index == 3 else "waiting"
        text = prefix + f"pid={index + 1} request_id=0x{index + 1:08x} port={41000 + index} status={status}"
        bolt.receipt(kind, kind, text, "NO_OP", text)
        assert not watch.observe({"product_revision": "r1"}, bolt._receipts, boundary=index)
        assert watch.streak == (0 if index == 3 else index)
    assert len({row["detail"] for row in bolt._receipts}) == 1


@pytest.mark.parametrize("left,right", [
    ("expected 0xabc got 0xdef", "expected 0xabc got 0x123"),
    ("sha256=abcdef1234567890", "sha256=1234567890abcdef"),
    ("a.py:12:3", "a.py:12:4"),
    ("version 0.7.47", "version 0.7.48"),
    ("/tmp/config-a.json", "/tmp/config-b.json"),
    ("expected 7 got 8", "expected 7 got 9"),
    ("status=waiting pid=1", "status=ready pid=2"),
    ("exit=1", "exit=2"),
])
def test_semantic_output_change_resets_streak(left, right):
    watch = UnboundedProgressWatch()
    receipts = [{"kind": "shell", "result_class": "NO_OP", "detail": left}]
    assert not watch.observe({"product_revision": "r1"}, receipts, boundary=0)
    receipts.append(dict(receipts[0]))
    assert not watch.observe({"product_revision": "r1"}, receipts, boundary=1)
    assert watch.streak == 1
    receipts.append({"kind": "shell", "result_class": "NO_OP", "detail": right})
    assert not watch.observe({"product_revision": "r1"}, receipts, boundary=2)
    assert watch.streak == 0


@pytest.mark.parametrize("digest", [None, "", "bad", "A" * 64, 1])
def test_invalid_producer_digest_falls_back_to_normalized_detail(digest):
    watch = UnboundedProgressWatch()
    receipts = []
    for boundary in range(3):
        receipts.append({"kind": "shell", "result_class": "NO_OP",
                         "fingerprint": str(boundary), "progress_output_digest": digest,
                         "detail": f"pid={boundary + 1} status=waiting"})
        assert not watch.observe({"product_revision": "r1"}, receipts, boundary=boundary)
    assert watch.streak == 2


@pytest.mark.parametrize("field,value", [("exit_code", 1), ("result_class", "OK"), ("target", "other")])
def test_output_identity_preserves_result_exit_code_and_target(field, value):
    watch = UnboundedProgressWatch()
    receipt = {"kind": "shell", "result_class": "NO_OP", "exit_code": 0,
               "target": "same", "detail": "same"}
    receipts = [dict(receipt), dict(receipt)]
    watch.observe({"product_revision": "r1"}, receipts[:1], boundary=0)
    watch.observe({"product_revision": "r1"}, receipts, boundary=1)
    assert watch.streak == 1
    receipts.append({**receipt, field: value})
    assert not watch.observe({"product_revision": "r1"}, receipts, boundary=2)
    assert watch.streak == 0
