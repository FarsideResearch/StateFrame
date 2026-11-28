from types import SimpleNamespace

from test_checkpoint_pool_prune_objects import _synthetic_checkpoint_environment  # noqa: F401
from test_eff_e28_checkpoint_every_four_cadence import publications


def test_completed_check_gets_one_off_cycle_checkpoint(tmp_path, monkeypatch):
    measurement = SimpleNamespace(last_completed_receipt={"turn": 1, "kind": "test",
                                                         "result_class": "GREEN"})
    for mode, expected in (("off", [4, 8]), ("on", [2, 4, 8])):
        monkeypatch.setenv("STATEFRAME_CHECKPOINT_AFTER_CHECK", mode)
        saves = publications(monkeypatch, tmp_path / mode, 4, measurement=measurement)
        assert list(saves) == expected
        # Interruption before step three can restore the completed check on this arm.
        tail_steps = 2 - max([0] + [step for step in saves if step <= 2])
        assert tail_steps == (2 if mode == "off" else 0)
