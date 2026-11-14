"""Localize uses the full query text and configurable inventory limits.

Guards two regressions: a silent 2,400-character slice of the brief/query that
could drop the only search anchor, and fixed inventory caps (4,000 files and
512 KB per source) with no way for a host to raise them.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import kernel_localize as kl

FILLER = "The invoice totals drift when the nightly ledger is reconciled against the bank export. "
ANCHOR_FILE = "src/zebra_quartz_handler.py"


def _tiny_repo(root: Path) -> Path:
    (root / "src").mkdir(parents=True)
    (root / ANCHOR_FILE).write_text("def zebra_quartz_handler():\n    return 1\n", encoding="utf-8")
    (root / "src" / "other.py").write_text("def other():\n    return 2\n", encoding="utf-8")
    return root


def _long_text_with_late_anchor() -> str:
    text = FILLER * 40 + "The failure is in zebra_quartz_handler after reconciliation."
    assert text.index("zebra_quartz_handler") > 2400
    return text


# Item A: the full query text is used, never a prefix.

def _distractor_repo(root: Path, count: int = 8) -> Path:
    """Anchor file plus distractors that share no vocabulary with the filler, so only the anchor carries the identifier."""
    (root / "src").mkdir(parents=True)
    (root / ANCHOR_FILE).write_text("def zebra_quartz_handler():\n    return 1\n", encoding="utf-8")
    for index in range(count):
        (root / "src" / f"billing_queue_{index}.py").write_text(
            f"def billing_queue_{index}():\n    return queue_size_{index}\n", encoding="utf-8")
    return root


@pytest.mark.parametrize("position", ["brief", "query"])
def test_anchor_after_former_2400_char_slice_ranks_first(tmp_path, position):
    repo = _distractor_repo(tmp_path / "repo")
    late = _long_text_with_late_anchor()
    if position == "brief":
        result = kl.localize(repo, late)
    else:
        result = kl.localize(repo, "nightly ledger drift", query=late)
    # Eight distractors: the anchor file must win on the identifier that sits after char 2400.
    assert {"zebra", "quartz", "handler"} <= set(result["anchors"]["terms"])
    assert result["suspects"][0]["path"] == ANCHOR_FILE


def test_query_echo_is_capped_and_scan_reports_full_length(tmp_path):
    repo = _tiny_repo(tmp_path / "repo")
    late = _long_text_with_late_anchor()
    brief = "nightly ledger drift"
    result = kl.localize(repo, brief, query=late)
    assert result["query"] == late[:400]
    assert result["query_echo_truncated"] is True
    assert result["query_truncated"] is False
    assert result["query_chars"] == len(brief) + 1 + len(late)
    assert result["query_terms_total"] == result["query_terms_used"]


def test_short_query_echo_is_not_flagged(tmp_path):
    repo = _tiny_repo(tmp_path / "repo")
    result = kl.localize(repo, "", query="zebra_quartz_handler crashes")
    assert result["query"] == "zebra_quartz_handler crashes"
    assert result["query_echo_truncated"] is False


def test_one_megabyte_query_is_fast_and_result_stays_small(tmp_path):
    """Wire cost and CPU: a 1 MB query with ~45k distinct quoted literals stays bounded."""
    import json
    import time

    repo = _distractor_repo(tmp_path / "repo")
    items = "".join(f'"item_{index:06d} marker" ' for index in range(50_000))
    query = items + " zebra_quartz_handler"
    assert len(query) >= 1_000_000
    started = time.perf_counter()
    result = kl.localize(repo, "crash report", query=query)
    elapsed = time.perf_counter() - started
    assert elapsed < 10.0
    assert len(json.dumps(result, default=str).encode("utf-8")) < 64 * 1024
    assert result["query_echo_truncated"] is True
    assert len(result["query"]) == 400
    assert result["query_truncated"] is True
    assert result["anchors_truncated"] is True
    assert len(result["anchors"]["literals"]) == kl.MAX_ANCHOR_VALUES
    assert result["anchor_totals"]["literals"] > kl.MAX_ANCHOR_VALUES
    # 50k distinct code-shaped identifiers overflow the 64-term cap, so the late anchor may be
    # dropped here. That is the explicit query_truncated contract, not a silent loss.
    assert len(result["anchors"]["terms"]) == result["query_terms_used"] == kl.MAX_QUERY_TERMS


def test_anchor_cap_keeps_specific_values_first_and_reports_totals():
    values = tuple("plain words " + "q" * (index + 1) for index in range(40)) + ("src/zebra_quartz.py",)
    kept, total = kl._bounded_anchors(values)
    assert total == 41
    assert len(kept) == kl.MAX_ANCHOR_VALUES
    assert "src/zebra_quartz.py" in kept  # path separator marks it specific, so it survives the cap


def test_anchor_value_over_length_cap_is_dropped_and_counted():
    kept, total = kl._bounded_anchors(("x" * 241, "short_one"))
    assert total == 2
    assert kept == ("short_one",)


def test_term_cap_is_explicit_and_keeps_specific_anchor(tmp_path):
    repo = _tiny_repo(tmp_path / "repo")
    prose = " ".join(f"word{chr(97 + i % 26)}{chr(97 + (i // 26) % 26)}x" for i in range(200))
    query = prose + " zebra_quartz_handler"
    result = kl.localize(repo, "", query=query)
    assert result["query_truncated"] is True
    assert result["query_terms_total"] > result["query_terms_used"] == kl.MAX_QUERY_TERMS
    assert ANCHOR_FILE in [row.get("path") for row in result["suspects"]]


def test_scan_keeps_all_terms_when_under_cap():
    anchors, bounds = kl._query_scan("Traceback in zebra_quartz_handler", "")
    assert bounds.truncated is False
    assert "zebra" in anchors["terms"] and "quartz" in anchors["terms"]


# Item B: configurable inventory limits read at call time.

def _write_files(root: Path, count: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (root / f"mod_{index:05d}.py").write_text(f"VALUE_{index} = {index}\n", encoding="utf-8")


def test_defaults_are_50000_files_and_4_mib_sources(monkeypatch):
    monkeypatch.delenv(kl.ENV_MAX_FILES, raising=False)
    monkeypatch.delenv(kl.ENV_MAX_SOURCE_BYTES, raising=False)
    assert kl.max_files_limit() == 50_000
    assert kl.max_source_bytes_limit() == 4_194_304
    assert kl.LOCALIZE_MAX_FILES_HARD == 1_000_000
    assert kl.LOCALIZE_MAX_SOURCE_BYTES_HARD == 67_108_864


def test_4001_tiny_files_are_fully_inventoried(tmp_path, monkeypatch):
    monkeypatch.delenv(kl.ENV_MAX_FILES, raising=False)
    _write_files(tmp_path / "many", 4001)
    inventory = kl.build_inventory(tmp_path / "many")
    assert len(inventory.files) == 4001
    assert inventory.omitted == ()
    assert inventory.coverage()["scope_status"] == "COMPLETE"


def test_env_override_of_10_gives_typed_file_limit(tmp_path, monkeypatch):
    monkeypatch.setenv(kl.ENV_MAX_FILES, "10")
    _write_files(tmp_path / "many", 12)
    inventory = kl.build_inventory(tmp_path / "many")
    assert len(inventory.files) == 10
    assert {"path": "<remaining>", "reason": "FILE_LIMIT"} in [dict(item) for item in inventory.omitted]
    assert inventory.coverage()["scope_status"] == "INCOMPLETE"


def test_explicit_max_files_argument_still_wins_and_is_clamped(tmp_path, monkeypatch):
    monkeypatch.delenv(kl.ENV_MAX_FILES, raising=False)
    _write_files(tmp_path / "many", 5)
    assert len(kl.build_inventory(tmp_path / "many", max_files=2).files) == 2
    assert kl.build_inventory(tmp_path / "many", max_files=10**9).files  # clamped, not rejected


def test_600_kb_source_is_now_included(tmp_path, monkeypatch):
    monkeypatch.delenv(kl.ENV_MAX_SOURCE_BYTES, raising=False)
    root = tmp_path / "big"
    root.mkdir()
    (root / "large.py").write_text("x = 1\n" * 100_000, encoding="utf-8")  # 600,000 bytes
    inventory = kl.build_inventory(root)
    assert [item.path for item in inventory.files] == ["large.py"]
    assert not any(item.get("reason") == "OVERSIZED" for item in inventory.omitted)


def test_source_bytes_override_keeps_typed_oversized_omission(tmp_path, monkeypatch):
    monkeypatch.setenv(kl.ENV_MAX_SOURCE_BYTES, "1000")
    root = tmp_path / "big"
    root.mkdir()
    (root / "large.py").write_text("x = 1\n" * 100_000, encoding="utf-8")
    inventory = kl.build_inventory(root)
    assert inventory.files == ()
    assert [dict(item)["reason"] for item in inventory.omitted] == ["OVERSIZED"]
    assert inventory.coverage()["scope_status"] == "INCOMPLETE"


@pytest.mark.parametrize("raw", ["", "   ", "abc", "1e6", "0", "-5", "nan", "inf", "1.5"])
def test_invalid_env_values_fall_back_to_defaults(monkeypatch, raw):
    monkeypatch.setenv(kl.ENV_MAX_FILES, raw)
    monkeypatch.setenv(kl.ENV_MAX_SOURCE_BYTES, raw)
    assert kl.max_files_limit() == 50_000
    assert kl.max_source_bytes_limit() == 4_194_304


def test_env_values_above_hard_max_are_clamped(monkeypatch):
    monkeypatch.setenv(kl.ENV_MAX_FILES, str(10**12))
    monkeypatch.setenv(kl.ENV_MAX_SOURCE_BYTES, str(10**12))
    assert kl.max_files_limit() == 1_000_000
    assert kl.max_source_bytes_limit() == 67_108_864


def test_env_values_are_read_at_call_time(monkeypatch):
    monkeypatch.setenv(kl.ENV_MAX_FILES, "7")
    assert kl.max_files_limit() == 7
    monkeypatch.setenv(kl.ENV_MAX_FILES, "9")
    assert kl.max_files_limit() == 9
