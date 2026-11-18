import os

from stateframe import replay_tape
from stateframe.inspect_run import _wire_files


def test_replay_and_inspect_keep_sends_after_four_digit_boundary(tmp_path):
    run = tmp_path
    wire = run / "wire"
    wire.mkdir()
    request = run / "request"
    response = run / "response"
    request.write_text('{"messages":[]}')
    response.write_text('{"choices":[{"message":{"role":"assistant","content":"ok"}}]}')

    for number in range(1, 10002):
        os.link(request, wire / f"{number:04d}.request.json")
        if number <= 10000:
            os.link(response, wire / f"{number:04d}.response.json")

    tape = replay_tape.build(wire, run / "sends.jsonl")
    pending = replay_tape.pending_request(wire)

    assert len(tape) == 10000
    assert tape[-1]["send"] == 10000
    assert pending is not None
    assert pending["send"] == 10001
    assert len(_wire_files(run)) == 10001
