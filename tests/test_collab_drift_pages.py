"""COLLAB-DRIFT pages: a source page whose file changed on disk is refreshed when the model is working in it.

Drives the production path the runtime uses: RealExchangeLedger.record (admits a read exchange) ->
refresh_source_pages (once per turn) -> source_pages / source_working_set_notices. OFF and STRICT=on must keep
today's eviction behaviour and notice text byte for byte.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from kernel_paged_source import source_revision
from kernel_stateframe_f4 import MAX_SOURCE_WORKING_SET_BYTES, RealExchangeLedger

PATH = "pkg/mod.py"
SOURCE = "def a():\n    return 1\n\n\ndef b():\n    return 2\n"
CALL_ID = "call_read_1"


@pytest.fixture
def drift_on(monkeypatch):
    monkeypatch.setenv("STATEFRAME_COLLAB_DRIFT", "on")
    monkeypatch.delenv("STATEFRAME_STRICT_DRIFT", raising=False)


@pytest.fixture
def drift_off(monkeypatch):
    monkeypatch.delenv("STATEFRAME_COLLAB_DRIFT", raising=False)
    monkeypatch.delenv("STATEFRAME_STRICT_DRIFT", raising=False)


def _write(root: Path, text: str) -> Path:
    path = root / PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    return path


def _read_pair(text: str, start: int, end: int, *, assistant_text: str = "") -> list[dict[str, object]]:
    """One admitted read exchange, shaped like the provider transcript the ledger receives."""
    lines = text.splitlines(keepends=True)
    body = "".join(lines[start - 1:end])
    header = (f"[read page] path={PATH} revision={source_revision(text)} lines={start}-{end}/{len(lines)} "
              f"chars={len(body)} truncated=false decoded_with_replacement=false")
    call = {
        "id": CALL_ID, "type": "function",
        "function": {"name": "read", "arguments": json.dumps({"path": PATH, "start_line": start, "end_line": end})},
    }
    assistant = {"role": "assistant", "content": assistant_text, "tool_calls": [call]}
    result = {"role": "tool", "tool_call_id": CALL_ID, "content": f"{header}\n[notice]\n{body}"}
    return [assistant, result]


def _ledger_with_read(root: Path, text: str, start: int, end: int, *, cited: bool) -> RealExchangeLedger:
    _write(root, text)
    ledger = RealExchangeLedger()
    assert ledger.record(_read_pair(text, start, end, assistant_text=f"Using {PATH}." if cited else ""),
                         provider_call_id=CALL_ID)
    return ledger


def _page(ledger: RealExchangeLedger) -> dict[str, object]:
    pages = ledger.source_pages()
    assert len(pages) == 1, pages
    return pages[0]


# -- production refresh path, switch ON --------------------------------------------------------------------

def test_active_page_refreshed_in_place_when_file_changed_above_it(tmp_path, drift_on):
    ledger = _ledger_with_read(tmp_path, SOURCE, 1, 2, cited=True)
    _write(tmp_path, "# header\n" + SOURCE)
    assert ledger.refresh_source_pages(tmp_path)
    page = _page(ledger)
    assert (page["start_line"], page["end_line"]) == (2, 3)
    assert page["content"] == "def a():\n    return 1\n"
    assert page["revision"] == source_revision("# header\n" + SOURCE)
    notices = ledger.source_working_set_notices()
    assert notices == (
        f"[Source page refreshed] {PATH} lines 1-2 changed on disk since you read it; current bytes shown",
    )


def test_active_page_whose_own_lines_changed_shows_new_bytes_and_notice(tmp_path, drift_on):
    ledger = _ledger_with_read(tmp_path, SOURCE, 1, 2, cited=True)
    changed = SOURCE.replace("return 1", "return 10")
    _write(tmp_path, changed)
    assert ledger.refresh_source_pages(tmp_path)
    page = _page(ledger)
    assert page["content"] == "def a():\n    return 10\n"
    assert "return 1\n" not in str(page["content"])
    assert page["revision"] == source_revision(changed)
    assert ledger.source_working_set_notices() == (
        f"[Source page refreshed] {PATH} lines 1-2 changed on disk since you read it; current bytes shown",
    )


def test_same_size_content_change_with_preserved_mtime_is_detected(tmp_path, drift_on):
    ledger = _ledger_with_read(tmp_path, SOURCE, 1, 2, cited=True)
    path = tmp_path / PATH
    before = path.stat()
    changed = SOURCE.replace("return 1", "return 2")
    assert len(changed) == len(SOURCE)
    _write(tmp_path, changed)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert ledger.refresh_source_pages(tmp_path)
    assert _page(ledger)["content"] == "def a():\n    return 2\n"


def test_non_active_changed_page_is_evicted_with_changed_on_disk_notice(tmp_path, drift_on):
    ledger = _ledger_with_read(tmp_path, SOURCE, 1, 2, cited=False)
    _write(tmp_path, SOURCE.replace("return 1", "return 10"))
    ledger.refresh_source_pages(tmp_path)
    assert ledger.source_pages() == ()
    notices = ledger.source_working_set_notices()
    assert len(notices) == 1
    assert notices[0].startswith("[Source page evicted]\n")
    assert f"path={PATH} lines=1-2 reason=source_revision_changed" in notices[0]
    assert "changed on disk since you read it" in notices[0]
    assert "Retrieve current bytes" in notices[0]


def test_refreshed_page_over_working_set_byte_cap_falls_back_to_eviction(tmp_path, drift_on):
    ledger = _ledger_with_read(tmp_path, SOURCE, 1, 1, cited=True)
    oversize = "def a():  " + "x" * MAX_SOURCE_WORKING_SET_BYTES + "\n"
    _write(tmp_path, oversize + SOURCE[len("def a():\n"):])
    ledger.refresh_source_pages(tmp_path)
    assert ledger.source_pages() == ()
    notices = ledger.source_working_set_notices()
    assert len(notices) == 1
    assert notices[0].startswith("[Source page evicted]\n")
    assert "changed on disk since you read it" in notices[0]


def test_refresh_notice_survives_snapshot_restore(tmp_path, drift_on):
    ledger = _ledger_with_read(tmp_path, SOURCE, 1, 2, cited=True)
    _write(tmp_path, SOURCE.replace("return 1", "return 10"))
    ledger.refresh_source_pages(tmp_path)
    restored = RealExchangeLedger.from_state(ledger.snapshot())
    assert restored.source_working_set_notices() == ledger.source_working_set_notices()
    assert len(restored.source_working_set_notices()) == 1


# -- switch OFF and STRICT=on: today's behaviour, byte for byte --------------------------------------------

_TODAY_EVICTION = (
    "[Source page evicted]\n"
    f"path={PATH} lines=1-2 reason=source_revision_changed\n"
    f"Retrieve current bytes before relying on this page with: read(path=\"{PATH}\", start_line=1, end_line=2)."
)


def test_switch_off_active_changed_page_is_evicted_as_today(tmp_path, drift_off):
    ledger = _ledger_with_read(tmp_path, SOURCE, 1, 2, cited=True)
    _write(tmp_path, SOURCE.replace("return 1", "return 10"))
    ledger.refresh_source_pages(tmp_path)
    assert ledger.source_pages() == ()
    assert ledger.source_working_set_notices() == (_TODAY_EVICTION,)


def test_strict_on_forces_today_eviction_even_with_collab_on(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_COLLAB_DRIFT", "on")
    monkeypatch.setenv("STATEFRAME_STRICT_DRIFT", "on")
    ledger = _ledger_with_read(tmp_path, SOURCE, 1, 2, cited=True)
    _write(tmp_path, SOURCE.replace("return 1", "return 10"))
    ledger.refresh_source_pages(tmp_path)
    assert ledger.source_pages() == ()
    assert ledger.source_working_set_notices() == (_TODAY_EVICTION,)
