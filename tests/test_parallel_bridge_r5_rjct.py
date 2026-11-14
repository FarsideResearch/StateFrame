"""Answer ACK accounting must outlive bounded diagnostic rejection history."""
import hashlib
from pathlib import Path

import pytest

import stateframe.parallel_channel as channel
from stateframe.parallel_journal import JournalBusy, Ticket


class _Journal:
    fault = None

    def __init__(self):
        self.tickets = []

    def submit(self, kind, payload):
        ticket = Ticket(str(len(self.tickets)), kind, payload)
        self.tickets.append(ticket)
        return ticket

    def take(self, ticket):
        if not ticket.done.is_set():
            return None
        result = ticket.result
        ticket.result = None
        return result


def test_rejected_answer_ack_survives_diagnostic_cache_churn(tmp_path, monkeypatch):
    journal = _Journal()
    monkeypatch.setattr(channel, "Journal", lambda: journal)
    monkeypatch.setattr(channel, "_RETRY_BACKOFF_SECONDS", 0.0)
    writer = channel.ChannelWrites()
    callbacks = []
    records = {}
    admission_limit = channel._MAX_PENDING
    admitted = 0
    refused_path = None
    first_path = tmp_path / "review/an/001.json"

    # Each visible write fails its bounded retry budget. The diagnostic cache may
    # evict old failures, but admission must retain their ACK observers separately.
    for index in range(admission_limit + 1):
        path = tmp_path / f"review/an/{index + 1:03d}.json"
        value = {"answer": index}
        records[path] = (value, hashlib.sha256(channel.canonical_json(value) + b"\n").hexdigest())
        ticket_count = len(journal.tickets)
        with pytest.raises(JournalBusy):
            writer.write(path, value, replace_existing=False, wait=False,
                         on_ack=lambda index=index: callbacks.append(index))

        if len(journal.tickets) == ticket_count:
            refused_path = path
            continue

        admitted += 1
        for attempt in range(3):
            ticket = journal.tickets[-1]
            assert ticket.kind == "write"
            ticket.result = {"ok": False, "error_class": "OSError",
                             "error": "directory fsync failed"}
            ticket.done.set()
            writer.drain()
            if attempt < 2:
                writer.drain()

    monkeypatch.setattr(channel, "_record", lambda path, _label: records[path])
    with pytest.raises(JournalBusy):
        writer.read(first_path, "answer", wait=False)
    assert journal.tickets[-1].kind == "sync"
    assert callbacks == [], "a visible answer is not accepted before its revalidation ACK"

    journal.tickets[-1].result = {"ok": True}
    journal.tickets[-1].done.set()
    writer.drain()
    assert writer.read(first_path, "answer", wait=False) == {"answer": 0}
    assert callbacks == [0], "revalidated answer must retain and fire its original ACK observer"
    assert writer.read(first_path, "answer", wait=False) == {"answer": 0}
    assert callbacks == [0], "the original ACK observer must fire exactly once"

    assert admitted == admission_limit
    assert refused_path == tmp_path / f"review/an/{admission_limit + 1:03d}.json"
    assert len(writer._rejected) <= admission_limit
