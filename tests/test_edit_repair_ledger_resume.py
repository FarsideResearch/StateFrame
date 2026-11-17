"""The repair fingerprint ledger survives snapshot/restore and keeps escalating.

Owner: probes/react_repair.py writes it into the worker checkpoint and restores it on resume;
kernel_continuation_restart.py validates it as an optional worker field.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ENGINE = Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from kernel_edit_refusal_codes import Refusal, RefusalCode  # noqa: E402
from kernel_edit_repair_hook import SNAPSHOT_KEY, build_repair, restore_ledger, snapshot_ledger  # noqa: E402
from kernel_edit_repair_request import EscalationLevel  # noqa: E402

SRC = b"x = 1\n"
PROP = [{"handle": "q1h1", "old_text": "y", "new_text": "z"}]
REF = [Refusal.build(RefusalCode.OLD_NOT_FOUND, edit_index=0, path="m.py", old_text="y")]


def test_snapshot_none_when_empty_and_restore_resumes_ladder():
    a = SimpleNamespace()
    assert snapshot_ledger(a) is None
    build_repair(runtime=a, refusals=REF, current_bytes=SRC, path="m.py", proposal=PROP, request_ordinal=1)
    state = snapshot_ledger(a)
    assert state is not None and SNAPSHOT_KEY == "edit_repair_ledger"
    b = SimpleNamespace()
    restore_ledger(b, state)
    rep = build_repair(runtime=b, refusals=REF, current_bytes=SRC, path="m.py", proposal=PROP, request_ordinal=2)
    assert rep.level is EscalationLevel.NARROW and rep.repeat_count == 2


def test_restore_ignores_garbage():
    b = SimpleNamespace()
    restore_ledger(b, "nope")
    restore_ledger(b, {"counts": "nope"})
    assert snapshot_ledger(b) is None


REACT = Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine" / "probes" / "react_repair.py"
RESTART = ENGINE / "kernel_continuation_restart.py"


def test_owner_writes_and_restores_the_ledger():
    src = REACT.read_text(encoding="utf-8")
    write = src.index("_checkpoint_state[_EDIT_REPAIR_KEY] = _edit_repair_ledger")
    assert "edit_repair_ledger" in src[write - 400:write + 120]
    read = src.index("restore_ledger(measurement, _restored.get(_EDIT_REPAIR_KEY))")
    assert "edit_repair_ledger" in src[read:read + 120]


def test_restart_validation_accepts_a_ledger_and_refuses_garbage():
    import kernel_continuation_restart as restart
    base = {name: None for name in restart.WORKER_FIELDS}
    assert restart._valid_worker_state_fields(base)
    assert restart._valid_worker_state_fields({**base, SNAPSHOT_KEY: {"counts": {"a" * 64: 2}}})
    for garbage in ("nope", {"counts": "x"}, {"counts": {}}, {"counts": {"a": 0}}, {"counts": {"a": 1}, "x": 1}):
        assert not restart._valid_worker_state_fields({**base, SNAPSHOT_KEY: garbage}), garbage
