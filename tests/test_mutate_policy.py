import pytest

from kernel_mutate_policy import Budget, MutationPolicy, policy_from
from stateframe.mutate_host import validate_ceiling


def test_policy_requires_network_ceiling_and_valid_values():
    assert not policy_from({}).enabled
    with pytest.raises(ValueError):
        policy_from({"level": "bogus"})
    requested = policy_from({"level": "authored", "network": "public-web", "shell_network": "none"})
    assert requested.network == "public-web"
    with pytest.raises(ValueError, match="mutation network requires --shell-network full"):
        validate_ceiling({"network": "public-web"}, {"shell_network": "none"})
    validate_ceiling({"network": "public-web"}, {"shell_network": "full"})
    assert policy_from({"level": "authored", "network": "public-web", "shell_network": "full"}).enabled
    with pytest.raises(ValueError):
        policy_from({"shell_network": "mystery"})


def test_budget_reservation_is_durable_and_exhaustion_sticks():
    state = Budget().fold([])
    assert state.reserve("invocations", 39)
    assert not state.reserve("invocations", 2)
    assert not state.reserve("invocations", 1)
    events = state.events
    resumed = Budget().fold(events)
    assert resumed.disabled
    assert not resumed.reserve("invocations", 1)


def test_unsettled_launch_charges_full_reservation():
    state = Budget().fold([{"kind": "BUDGET_RESERVE", "name": "payload_wall_s", "amount": 10, "id": "x"}])
    assert state.remaining()["payload_wall_s"] == 110
    settled = Budget().fold([*state.events, {"kind": "BUDGET_SETTLE", "id": "x", "amount": 2}])
    assert settled.remaining()["payload_wall_s"] == 118


def test_settlement_overrun_disables_mutations():
    state = Budget().fold([])
    assert state.reserve("payload_wall_s", 10, id="l1")
    state.settle("l1", 121)
    assert state.disabled
    assert not state.reserve("invocations", 1)


def test_exact_reservation_and_corrupt_replay_are_fail_closed():
    state = Budget().fold([])
    assert state.reserve("invocations", 40)
    assert state.disabled
    with pytest.raises(ValueError):
        Budget().fold([{"kind": "BUDGET_RESERVE", "name": "invocations", "amount": -1}])
    with pytest.raises(ValueError):
        Budget().fold([{"kind": "BUDGET_SETTLE", "id": "missing", "amount": 1}])
    with pytest.raises(ValueError):
        Budget().fold([{"kind": "BUDGET_EXHAUSTED", "name": "revisions_per_unit", "scope": []}])


def test_direct_policy_constructor_refuses_unknown_level():
    with pytest.raises(ValueError):
        MutationPolicy(level="mystery")


def test_per_unit_and_episode_limits_are_scoped():
    state = Budget().fold([])
    with pytest.raises(ValueError):
        state.reserve("revisions_per_unit", 1)
    assert state.reserve("revisions_per_unit", 2, scope="unit-a")
    assert state.reserve("revisions_per_unit", 2, scope="unit-b")
    assert state.remaining(scope="unit-a")["revisions_per_unit"] == 1
    assert Budget().fold(state.events).remaining(scope="unit-b")["revisions_per_unit"] == 1
    assert state.reserve("sends_per_episode", 1, scope="episode-1")
    assert state.reserve("sends_per_episode", 1, scope="episode-2")
    assert state.reserve("revisions_per_unit", 1, scope="unit-a")
    assert state.disabled
    assert Budget().fold(state.events).disabled


def test_revision_admission_charges_are_one_durable_event():
    event = {"kind": "BUDGET_ADMISSION", "id": "submit-1", "scope": "reader",
             "charges": {"authored_source_bytes": 64, "revisions_total": 1,
                         "revisions_per_unit": 1}}
    state = Budget().fold([event])
    assert state.remaining(scope="reader")["revisions_per_unit"] == 2
    assert state.remaining()["authored_source_bytes"] == Budget().limits["authored_source_bytes"] - 64
    with pytest.raises(ValueError):
        Budget().fold([event, event])
