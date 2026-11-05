"""Second-pass semantic refusals must not mask lease loss after materialization."""
from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable

import pytest

from kernel_continuation_journal import ContinuationJournal, JournalError


_REPLACE_JOURNAL = (
    "import os, pathlib, sys\n"
    "path = pathlib.Path(sys.argv[1])\n"
    "replacement = path.with_name(f'.{path.name}.external-{os.getpid()}')\n"
    "replacement.write_bytes(path.read_bytes())\n"
    "os.replace(replacement, path)\n"
)


def _external_replace(path) -> None:
    """Replace the journal inode from a real child process without its flock."""
    subprocess.run(
        [sys.executable, "-c", _REPLACE_JOURNAL, str(path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )


class _TriggerRecord(dict[str, object]):
    def __init__(self, record: dict[str, object], trigger: Callable[[], None]):
        super().__init__(record)
        self._trigger = trigger

    def get(self, key: str, default: object = None) -> object:
        self._trigger()
        return super().get(key, default)


def test_second_pass_owner_loss_precedes_semantic_refusal_after_materialization(
        tmp_path, monkeypatch):
    from kernel_unknown_outcomes import ExternalEffectAdapter, PRESENT

    journal = ContinuationJournal(
        tmp_path, "external-effect-second-pass-owner", {"origin": "second-pass-owner"},
    )
    try:
        adapter = ExternalEffectAdapter(journal)
        adapter.begin("second-pass-key", {"value": 1}, destination_name="mail")
        _effect_id, intent = adapter._intent("second-pass-key")
        adapter._record_outcome(intent, outcome=PRESENT, source="lookup")

        original_records = journal.records
        scans = 0
        second_pass_materialized = False
        external_replacement_done = False

        def replace_after_materialization() -> None:
            nonlocal external_replacement_done
            if external_replacement_done:
                return
            assert second_pass_materialized
            _external_replace(journal.path)
            external_replacement_done = True

        def controlled_records(*args, **kwargs):
            nonlocal scans, second_pass_materialized
            scans += 1
            rows = original_records(*args, **kwargs)
            if scans != 2:
                return rows
            materialized = list(rows)
            second_pass_materialized = True
            return [_TriggerRecord(row, replace_after_materialization) for row in materialized]

        monkeypatch.setattr(journal, "records", controlled_records)

        with pytest.raises(JournalError) as raised:
            adapter._replay()

        assert type(raised.value) is JournalError
        assert scans == 2
        assert second_pass_materialized
        assert external_replacement_done
    finally:
        journal.close()
