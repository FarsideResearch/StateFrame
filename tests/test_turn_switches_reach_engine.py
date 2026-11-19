"""Every engine switch named STATEFRAME_TURN_* in the switch registry must be settable from the shell."""
import json
from pathlib import Path

from stateframe import runner

ROOT = Path(__file__).resolve().parents[1]


def test_every_turn_switch_is_an_opt_in_switch():
    registry = (ROOT / "src/stateframe/engine/switch-registry.json").read_text()
    names = {token.strip('"') for token in registry.replace(",", " ").split() if token.strip('"').startswith("STATEFRAME_TURN_")}
    missing = sorted(name for name in names if name not in runner.OPT_IN_SWITCHES and name not in getattr(runner, "CHECKPOINT_ENV", ()))
    assert missing == [], f"shell cannot set: {missing}"
