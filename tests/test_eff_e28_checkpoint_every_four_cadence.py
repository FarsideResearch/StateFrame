import hashlib
import sys
from types import SimpleNamespace

import pytest

from scripts import checkpoint_restore_probe as probe

from stateframe.engine import kernel_checkpoint as checkpoint
from test_checkpoint_pool_prune_objects import _synthetic_checkpoint_environment  # noqa: F401


def publications(monkeypatch, root, cadence, *, steps=8, measurement=None):
    """Publish real generations, count stored state bytes before pruning any of them."""
    root.mkdir()
    monkeypatch.setenv(checkpoint.EVERY_ENV, str(cadence))
    monkeypatch.setenv(checkpoint.MODE_ENV, "save")
    monkeypatch.setattr(checkpoint, "_async_enabled", lambda: False)
    monkeypatch.setattr(checkpoint, "_low_disk_space", lambda _: None)
    monkeypatch.setattr(checkpoint, "_STATE", dict(checkpoint._STATE, disabled=False,
                       saved=0, restored_step=None, last_sends=None, extra_check_save=False))
    checkpoint._STATE["setup"] = {"code": sys._getframe().f_code, "line": 0, "funcs": frozenset()}
    published = {}
    for step in range(1, steps + 1):
        monkeypatch.setattr(checkpoint, "_sends", lambda _, n=step: n)
        checkpoint.save_point(sys._getframe(), step, root / "worker")
        if checkpoint._STATE["saved"] > len(published):
            state = (checkpoint.latest(root) / "state.pkl").read_bytes()
            published[step] = (len(state), hashlib.sha256(state).hexdigest())
    return published


def test_every_four_publishes_expected_boundaries(monkeypatch, tmp_path):
    control = publications(monkeypatch, tmp_path / "dense", 1)
    sparse = publications(monkeypatch, tmp_path / "sparse", 4)
    assert list(control) == list(range(1, 9))
    assert list(sparse) == [4, 8]
    assert sparse == {step: control[step] for step in sparse}
    assert sum(n for n, _ in sparse.values()) <= 0.4 * sum(n for n, _ in control.values())


def test_passing_scoped_check_gets_off_cycle_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("STATEFRAME_CHECKPOINT_AFTER_CHECK", "on")
    receipt = {"turn": 0, "kind": "test", "result_class": "PARTIAL_PASS"}
    published = publications(
        monkeypatch, tmp_path / "scoped", 4, steps=2,
        measurement=SimpleNamespace(last_completed_receipt=receipt),
    )

    assert list(published) == [1]


@pytest.mark.host_sandbox
def test_sparse_checkpoint_resume_matches_pending_send(tmp_path, monkeypatch):
    check_sparse_resume(tmp_path, monkeypatch, 4)


def check_sparse_resume(tmp_path, monkeypatch, cadence):
    monkeypatch.setenv("STATEFRAME_CHECKPOINT_AFTER_CHECK", "off")
    point = cadence + 2
    run, original, disk, endpoint, url = probe.capture(tmp_path, point, point + 2, True, "edit", cadence)
    try:
        result = probe.resume(tmp_path, run, point, original, True, endpoint, url, cadence)
    finally:
        probe.stop(endpoint)
    assert disk["step"] % cadence == 0
    assert result["normalized_identical"] and result["checkpoint_verified"], result
    assert result["checkpoint_restored"] and not result["full_replay_fallback"]
    assert not result["divergence"] and not result["checkpoint_refused"]
    archived = run / "attempts" / "1"
    for request in (archived / "wire").glob("*.request.json"):
        assert (run / "wire" / request.name).read_bytes() == request.read_bytes()
    effects = sorted((archived / "workspace").glob("step_*.txt"))
    assert effects
    for effect in effects:
        assert (run / "workspace" / effect.name).read_bytes() == effect.read_bytes()


def test_control_publishes_base_state_bytes(tmp_path, tmp_path_factory):
    import os
    import subprocess
    from pathlib import Path

    from _stateframe_home import isolated_stateframe_home
    from tools.pinned_tree import pinned_tree

    driver = """
import random, sys
from pathlib import Path
from pytest import MonkeyPatch
from test_checkpoint_pool_prune_objects import _save, _synthetic_checkpoint_environment
random.seed(28)
with MonkeyPatch.context() as patch:
    _synthetic_checkpoint_environment.__wrapped__(patch)
    run = Path(sys.argv[1])
    run.mkdir()
    state = (_save(run, 1) / "state.pkl").read_bytes()
    (run / "published.pkl").write_bytes(state)
"""
    candidate = Path(__file__).resolve().parents[1]
    for label, tree in (("base", pinned_tree("1886f919", tmp_path_factory)), ("candidate", candidate)):
        env = {k: v for k, v in os.environ.items() if not k.startswith("STATEFRAME_")}
        env["STATEFRAME_HOME"] = isolated_stateframe_home()
        env["PYTHONPATH"] = os.pathsep.join(str(tree / subdir)
                                            for subdir in ("tests", "src", "src/stateframe/engine"))
        env.update(STATEFRAME_CHECKPOINT_EVERY="1", STATEFRAME_CHECKPOINT_AFTER_CHECK="off",
                   STATEFRAME_WORKSPACE_OBJECTS="off", STATEFRAME_FL_WRITER_ANCHOR="off")
        subprocess.run([sys.executable, "-c", driver, str(tmp_path / label)],
                       cwd=tree, env=env, check=True)
    assert (tmp_path / "candidate/published.pkl").read_bytes() == (tmp_path / "base/published.pkl").read_bytes()
