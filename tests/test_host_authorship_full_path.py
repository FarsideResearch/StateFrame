"""Host-row identity on a full scripted run: off and shadow leave every request byte alone."""
from __future__ import annotations

import hashlib

import pytest

from host_authorship_fixtures import run_scripted_example

pytestmark = pytest.mark.host_sandbox


def test_scripted_run_bytes_match_in_off_and_shadow_and_on_journals_manifests(tmp_path):
    arms = {label: run_scripted_example(tmp_path, "discount_bug", mode, label=label)
            for label, mode in (("unset", None), ("off", "off"), ("shadow", "shadow"), ("on", "on"))}
    assert arms["unset"]["wires"]
    # The default is off, and shadow only observes: all three send the same bytes.
    assert arms["off"]["wires"] == arms["unset"]["wires"]
    assert arms["shadow"]["wires"] == arms["unset"]["wires"]
    for label in ("unset", "off", "shadow"):
        assert arms[label]["manifests"] == [], label
    for label in ("unset", "off"):
        assert arms[label]["shadow"] == [] and arms[label]["missing"] == [], label
    # On: one manifest beside each final serialization, each naming the bytes it describes.
    on = arms["on"]
    assert len(on["wires"]) == len(arms["unset"]["wires"])
    assert len(on["manifests"]) >= len(on["wires"])
    described = {event["body_sha256"] for event in on["manifests"]}
    assert all(isinstance(value, str) and len(value) == 64 for value in described)
    assert described & {hashlib.sha256(wire).hexdigest() for wire in on["wires"]}
    for event in on["manifests"]:
        assert isinstance(event["step"], int) and isinstance(event["request_id"], str)
        assert isinstance(event["rows"], list)
