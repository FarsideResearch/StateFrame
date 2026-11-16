"""kernel_edit_repair_hook: whole-file handles, runtime-held ledger, cache-safe placement."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ENGINE = Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from kernel_edit_handles import CURRENT_SOURCE_PREFIX  # noqa: E402
from kernel_edit_refusal_codes import HANDLE_PATTERN, Refusal, RefusalCode  # noqa: E402
from kernel_edit_repair_hook import (  # noqa: E402
    LEDGER_ATTR, PENDING_ATTR, build_repair, journal_fields, load_ledger, place_repair_message,
    repair_minter, store_ledger,
)
from kernel_edit_repair_request import BLOCKED_AT, EscalationLevel, FingerprintLedger  # noqa: E402

PATH = "pkg/mod.py"
SOURCE = b"def alpha():\n    return 1\n\ndef beta():\n    return 2\n"
PROPOSAL = [{"handle": "q4h1", "old_text": "    return 3\n", "new_text": "    return 30\n"}]


def refusal(code=RefusalCode.OLD_NOT_FOUND, index=0, old="    return 3\n"):
    return Refusal.build(code, edit_index=index, path=PATH, old_text=old)


def test_minter_covers_whole_file_with_one_span_handle():
    hm = repair_minter(5, PATH, SOURCE.decode("utf-8"))
    assert hm.request_ordinal == 5
    assert [r.handle for r in hm.records] == ["q5h1"]
    rec = hm.records[0]
    assert rec.span.start_line == 1 and rec.span.end_line == 5
    assert rec.revision == "sha256:" + hashlib.sha256(SOURCE).hexdigest()
    assert "not shown: none" in hm.render()


def test_ledger_round_trips_through_runtime_attribute():
    rt = SimpleNamespace()
    assert load_ledger(rt).counts == ()
    store_ledger(rt, FingerprintLedger.from_counts({"a" * 64: 2}))
    assert getattr(rt, LEDGER_ATTR) == {"counts": {"a" * 64: 2}}
    assert load_ledger(rt).count("a" * 64) == 2


@pytest.mark.parametrize("code", list(RefusalCode))
def test_every_code_builds_a_repair_with_fresh_handles_and_bytes(code):
    rt = SimpleNamespace()
    rep = build_repair(runtime=rt, refusals=[refusal(code)], current_bytes=SOURCE, path=PATH,
                       proposal=PROPOSAL, request_ordinal=7)
    assert rep.codes == (code,)
    assert rep.handles == ("q7h1",)
    assert rep.current_source == SOURCE.decode("utf-8")
    text = rep.render()
    assert SOURCE.decode("utf-8") in text and "q7h1" in text and CURRENT_SOURCE_PREFIX in text
    # The builder renders the rejected proposal verbatim as JSON, so compare the JSON spelling.
    assert json.dumps(PROPOSAL[0]["old_text"])[1:-1] in text
    assert HANDLE_PATTERN.search(rep.observation()) is None
    assert load_ledger(rt).count(rep.fingerprints[0]) == 1


def test_ladder_escalates_across_calls_and_is_replay_deterministic():
    rt = SimpleNamespace()
    levels = []
    for ordinal in range(1, BLOCKED_AT + 1):
        rep = build_repair(runtime=rt, refusals=[refusal()], current_bytes=SOURCE, path=PATH,
                           proposal=PROPOSAL, request_ordinal=ordinal)
        levels.append(rep.level)
    assert levels == [EscalationLevel.RETRY, EscalationLevel.NARROW, EscalationLevel.BLOCKED]
    # replay: same ledger snapshot + same ordinal -> identical payload digest
    snap = dict(getattr(rt, LEDGER_ATTR))
    a = build_repair(runtime=SimpleNamespace(**{LEDGER_ATTR: dict(snap)}), refusals=[refusal()],
                     current_bytes=SOURCE, path=PATH, proposal=PROPOSAL, request_ordinal=9)
    b = build_repair(runtime=SimpleNamespace(**{LEDGER_ATTR: dict(snap)}), refusals=[refusal()],
                     current_bytes=SOURCE, path=PATH, proposal=PROPOSAL, request_ordinal=9)
    assert a.digest() == b.digest() and a.handles == b.handles == ("q9h1",)


def test_place_repair_message_appends_new_final_user_message_only():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"}, {"role": "tool", "content": "obs", "tool_call_id": "c1"}]
    before = [dict(m) for m in msgs]
    place_repair_message(msgs, "[EDIT REFUSED] request q3 ...\n" + CURRENT_SOURCE_PREFIX)
    assert msgs[:-1] == before
    assert msgs[-1] == {"role": "user", "content": "[EDIT REFUSED] request q3 ...\n" + CURRENT_SOURCE_PREFIX}
    with pytest.raises(ValueError):
        place_repair_message(msgs, "again " + CURRENT_SOURCE_PREFIX)   # final msg already carries a map
    with pytest.raises(ValueError):
        place_repair_message([{"role": "user", "content": "x"}], "")


def test_journal_fields_carry_no_bytes_or_handles():
    rep = build_repair(runtime=SimpleNamespace(), refusals=[refusal()], current_bytes=SOURCE,
                       path=PATH, proposal=PROPOSAL, request_ordinal=2)
    fields = journal_fields(rep)
    assert set(fields) == {"codes", "fingerprints", "level", "repeat_count", "blocked",
                           "request_ordinal", "path", "digest"}
    flat = repr(fields)
    assert "return 1" not in flat and HANDLE_PATTERN.search(flat) is None
    assert fields["level"] == "RETRY" and fields["blocked"] is False


def test_blocked_repair_trips_live_breaker_only_when_blocked():
    from kernel_edit_repair_hook import trip_loop_breaker_if_blocked
    from kernel_tool_call_loop_breaker import ToolCallLoopBreaker
    rt = SimpleNamespace(_tool_call_loop_breaker=ToolCallLoopBreaker(threshold=5))
    reps = [build_repair(runtime=rt, refusals=[refusal()], current_bytes=SOURCE, path=PATH,
                         proposal=PROPOSAL, request_ordinal=i) for i in range(1, BLOCKED_AT + 1)]
    assert trip_loop_breaker_if_blocked(rt, reps[0], "edit_source") is None
    assert trip_loop_breaker_if_blocked(rt, reps[1], "edit_source") is None
    notice = trip_loop_breaker_if_blocked(rt, reps[2], "edit_source")
    assert notice and rt._tool_call_loop_breaker.tripped
    assert trip_loop_breaker_if_blocked(SimpleNamespace(), reps[2], "edit_source") is None  # no breaker: no-op
