"""Large receipt fingerprints are reused by content and stay byte-identical."""
from __future__ import annotations

import copy
import dataclasses
import json
import pickle
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"))

import kernel_green_menu_open_obligation as menu  # noqa: E402


def _plain(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))


def _receipt(files=300):
    return {"receipt_id": "r", "exit": 0, "ok": True,
            "binding": {f"src/é{i}.py": "a" * 64 for i in range(files)},
            "rows": [{"path": f"p{i}", "size": i, "ratio": 0.5, "none": None} for i in range(files)]}


@pytest.fixture(autouse=True)
def _empty(monkeypatch):
    monkeypatch.setattr(menu, "_FINGERPRINTS", {})


def _count(monkeypatch):
    calls = []
    real = json.dumps
    monkeypatch.setattr(menu.json, "dumps", lambda *a, **k: calls.append(1) or real(*a, **k))
    return calls


def test_unchanged_large_receipt_is_encoded_once(monkeypatch):
    receipt = _receipt()
    expected = _plain(receipt)
    calls = _count(monkeypatch)
    assert [menu._fingerprint(receipt) for _ in range(20)] == [expected] * 20
    assert len(calls) <= 2


@pytest.mark.parametrize("change", [
    lambda r: r["binding"].update({"src/é7.py": "b" * 64}),
    lambda r: r["rows"][5].update(size=6),
    lambda r: r["rows"][5].update(size=True),
    lambda r: r["rows"][5].update(ratio=-0.0),
    lambda r: r.update(ok=1),
    lambda r: r["rows"].pop(),
])
def test_in_place_change_is_always_seen(change):
    receipt = _receipt()
    menu._fingerprint(receipt)
    change(receipt)
    assert menu._fingerprint(receipt) == _plain(receipt)
    menu._fingerprint(receipt)
    assert menu._fingerprint(copy.deepcopy(receipt)) == _plain(receipt)


def test_values_marshal_refuses_are_never_reused(monkeypatch):
    class Odd:
        text = "one"

        def __str__(self):
            return self.text

    receipt = {**_receipt(), "odd": Odd()}
    first = menu._fingerprint(receipt)
    Odd.text = "two"
    assert menu._fingerprint(receipt) != first
    assert menu._FINGERPRINTS == {}


def test_small_rows_dataclasses_and_unsortable_rows_keep_their_text():
    @dataclasses.dataclass
    class Row:
        binding: dict

    assert menu._fingerprint({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    row = Row(_receipt()["binding"])
    assert menu._fingerprint(row) == menu._fingerprint(row) == _plain({"binding": row.binding})
    unsortable = {1: "x" * 9000, "a": 2}
    assert menu._fingerprint(unsortable) == repr(unsortable)


def test_text_store_is_bounded(monkeypatch):
    size = len(_plain(_receipt()))
    monkeypatch.setattr(menu, "_FINGERPRINT_CHARS", size * 2)
    for index in range(10):
        receipt = {**_receipt(), "exit": index}
        assert menu._fingerprint(receipt) == _plain(receipt)
        assert sum(map(len, menu._FINGERPRINTS.values())) <= size * 2
    monkeypatch.setattr(menu, "_FINGERPRINT_CHARS", 10)
    menu._FINGERPRINTS.clear()
    menu._fingerprint(_receipt())
    assert menu._FINGERPRINTS == {}


@pytest.mark.parametrize("prefix", ["", "engine.", "stateframe.engine."])
def test_text_store_never_enters_a_checkpoint(prefix):
    import kernel_checkpoint

    name = prefix + "kernel_green_menu_open_obligation"
    before = kernel_checkpoint._module_state(name, menu)
    menu._fingerprint(_receipt())
    assert menu._FINGERPRINTS
    after = kernel_checkpoint._module_state(name, menu)
    assert "_FINGERPRINTS" not in after
    assert pickle.dumps(sorted(after)) == pickle.dumps(sorted(before))
