"""User-settable engine switches must accept exactly the values their protocol spec declares."""
from __future__ import annotations

import pytest

from stateframe import runner
from stateframe.engine import kernel_protocol_spec

SPEC = {ref.name: ref for ref in kernel_protocol_spec._SWITCH_REFS}


@pytest.mark.parametrize("name", ["STATEFRAME_UNIT_PLAN_V2", "STATEFRAME_SEAM_RECALL"])
def test_unit_plan_switches_are_user_settable(name):
    assert name in runner.OPT_IN_SWITCHES, f"{name} is in the protocol spec but users cannot set it"


def test_opt_in_values_match_protocol_spec():
    # Class guard: a runner entry with a value list must not drift from the spec it passes through.
    drift = {
        name: (tuple(values), SPEC[name].values)
        for name, values in runner.OPT_IN_SWITCHES.items()
        if name in SPEC and values and set(values) != set(SPEC[name].values)
    }
    assert drift == {}
