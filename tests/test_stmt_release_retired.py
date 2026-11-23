"""The retired resource release signal must not offer close-before-use edits."""
from __future__ import annotations

import edit_sets
import pytest
from kernel_continuation_journal import ContinuationJournal


@pytest.mark.parametrize(
    ("source", "close_call"),
    [
        (
            "def read_payload(path):\n"
            "    handle = open(path)\n"
            "    return handle.read()",
            "handle.close()",
        ),
        (
            "def query():\n"
            "    session = SessionLocal()\n"
            "    return session.execute('select 1').fetchone()",
            "session.close()",
        ),
        (
            "def take(pool):\n"
            "    lease = pool.acquire()\n"
            "    return lease.value()",
            "lease.release()",
        ),
    ],
    ids=("open-read", "session-result", "pool-lease"),
)
def test_resource_release_candidates_are_not_offered(source: str, close_call: str):
    candidates = edit_sets.c2_statement_insert(source.splitlines(), 0)

    assert all(candidate.pattern != "stmt_release" for candidate in candidates)
    assert all(close_call not in candidate[2] for candidate in candidates)


def test_other_statement_insert_kinds_keep_their_order():
    source = (
        "def handle(session, row, should_return):\n"
        "    session.add(row)\n"
        "    if should_return:\n"
        "        return result\n"
        "    result = 1\n"
        "\n"
        "def peer(session, row):\n"
        "    session.add(row)\n"
        "    session.refresh(row)"
    )
    file_lines = source.splitlines()
    candidates = edit_sets.c2_statement_insert(file_lines[:5], 0,
                                              file_lines=file_lines)

    assert [candidate.pattern for candidate in candidates] == [
        "stmt_commit", "stmt_bind", "stmt_peer",
    ]


def test_legacy_release_kind_card_still_loads_from_journal(tmp_path):
    card = edit_sets.EditSet(
        (0, 0), "handle", "handle.close()",
        [edit_sets.SpanEdit("handle", "handle.close()")],
        "stmt_release", ["handle"],
    )
    manifest = {"origin": "legacy-release-card"}
    journal = ContinuationJournal(tmp_path, "legacy-release-card", manifest)
    try:
        journal.emit("stmt_release", card={
            "kind": card.pattern,
            "description": card.describe(),
        })
        path = journal.path
    finally:
        journal.close()

    loaded = ContinuationJournal.resume(
        path, task_id="legacy-release-card", manifest=manifest,
        expected_head=journal.head,
    )
    received = []
    try:
        loaded.subscribe("legacy-card-reader", received.append,
                         kinds=("stmt_release",))
    finally:
        loaded.close()

    assert len(received) == 1
    assert received[0]["kind"] == "stmt_release"
    assert received[0]["card"]["kind"] == "stmt_release"
