"""Child process for the checkpoint kill/resume replay test (never a StateFrame run).

usage: checkpoint_kill_child.py run|resume <run_dir> <turns> [kill_spec]
kill_spec: none | mid:K (SIGKILL while the K-th save writes its state) | prepublish:K (SIGKILL after
the K-th save is complete but before it is published) | between:K (SIGKILL after save K, before turn K+1).
Prints the final digest of the loop state and the env tree.
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import os
import signal
import sys
import types
from pathlib import Path

from stateframe.engine import kernel_checkpoint as cp


def digest(state: dict, run: Path) -> str:
    h = hashlib.sha256(json.dumps(state, sort_keys=True).encode())
    for path in sorted((run / "env").rglob("*")):
        if path.is_file():
            h.update(str(path.relative_to(run)).encode() + path.read_bytes())
    return h.hexdigest()


def main(argv: list[str]) -> None:
    action, run, turns = argv[0], Path(argv[1]), int(argv[2])
    spec = argv[3] if len(argv) > 3 else "none"
    kind, _, at = spec.partition(":")
    at_save = int(at) if at else -1
    frame = types.SimpleNamespace(f_code=main.__code__, f_locals={"state": {"counter": 0, "log": []}})
    cp._is_state_module = lambda *_a: False
    cp._context_items = lambda: []
    cp.contextvars.copy_context = contextvars.Context
    cp.install_continuous_clock = lambda _saved: None
    cp._STATE["setup"] = {"code": frame.f_code, "line": 0, "funcs": frozenset()}
    cp.register_signing_key(b"checkpoint-test-key-0123456789abcdef")
    saves = {"n": 0}
    if kind == "mid":
        real_write = cp._StateWriter.write

        def write(self, chunk):
            written = real_write(self, chunk)
            if saves["n"] == at_save:  # part of the state image is on disk, nothing is published
                self.stream.write(b"torn")
                self.stream.flush()
                signal.raise_signal(signal.SIGKILL)
            return written
        cp._StateWriter.write = write
    if kind == "prepublish":
        real_publish = cp._publish_checkpoint

        def publish(*args):
            if saves["n"] == at_save:
                signal.raise_signal(signal.SIGKILL)
            return real_publish(*args)
        cp._publish_checkpoint = publish
    (run / "env").mkdir(parents=True, exist_ok=True)
    if action == "resume":
        latest = cp.latest(run)
        if latest is None:
            print("NOCHECKPOINT")
            return
        cp.restore(frame, latest, run)
    state = frame.f_locals["state"]
    while state["counter"] < turns:
        state["counter"] += 1
        k = state["counter"]
        state["log"].append(f"turn-{k}-" + "x" * (k * 37 % 211))
        (run / "env" / f"f{k % 3}.txt").write_text(f"{k}\n" * (k + 5))
        saves["n"] += 1
        cp.save(frame, k + 1, run, background=False)
        if kind == "between" and saves["n"] == at_save:
            signal.raise_signal(signal.SIGKILL)
    print(digest(state, run))


if __name__ == "__main__":
    main(sys.argv[1:])
