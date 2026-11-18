"""Digest identity caches are process observations, never checkpoint authority."""
from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
from stateframe.engine import kernel_checkpoint as checkpoint

ALIASES = (
    "kernel_file_digest_memo",
    "engine.kernel_file_digest_memo",
    "stateframe.engine.kernel_file_digest_memo",
)


@pytest.mark.parametrize("alias", ALIASES)
def test_checkpoint_excludes_digest_observations(alias):
    module = ModuleType(alias)
    module._MEMO = {"old.py": ((1, 2, 3, 4, 5), "stale")}
    module._LAST_CLOCK_NS = 13_000_000_000
    state = checkpoint._module_state(alias, module)
    assert "_MEMO" not in state
    assert "_LAST_CLOCK_NS" not in state


@pytest.mark.parametrize("alias", ALIASES)
@pytest.mark.parametrize("section", ["modules", "patched", "absent"])
def test_restore_discards_legacy_digest_observations(tmp_path, monkeypatch, alias, section):
    module = ModuleType(alias)
    module._MEMO = {"live.py": ((1, 2, 3, 4, 5), "old-process")}
    module._LAST_CLOCK_NS = 15_000_000_000
    monkeypatch.setitem(sys.modules, alias, module)

    class RestoreReachedParticipants(Exception):
        pass

    class Participants:
        def items(self):
            raise RestoreReachedParticipants

    payload = {"modules": {}, "classes": {}, "patched": {}, "participants": Participants()}
    if section != "absent":
        payload[section][alias] = {
            "_MEMO": {"saved.py": ((1, 2, 3, 4, 5), "stale")},
            "_LAST_CLOCK_NS": 13_000_000_000,
        }
    monkeypatch.setattr(checkpoint, "_validate", lambda path: ({}, b""))
    monkeypatch.setattr(checkpoint, "_restore_files", lambda *args: None)
    monkeypatch.setattr(checkpoint, "_frame_maps", lambda frame: ({}, {}))
    monkeypatch.setattr(checkpoint, "_Unpickler", lambda *args: SimpleNamespace(load=lambda: payload))
    with pytest.raises(RestoreReachedParticipants):
        checkpoint.restore(None, tmp_path / "checkpoint", tmp_path)
    assert module._MEMO == {}
    assert module._LAST_CLOCK_NS is None
