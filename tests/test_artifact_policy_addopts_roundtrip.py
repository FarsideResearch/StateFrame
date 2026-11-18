"""TestArtifactPolicy must carry sanitized_addopts through to_dict and from_mapping.

A resumed run reloads the policy from its serialized payload. Dropping or mangling sanitized_addopts on that
trip would silently change the pytest command that gets replayed, so the round trip and the loader's defensive
defaults are pinned here.
"""
from __future__ import annotations

import kernel_test_artifact_policy as artifact_policy

# Module alias: binding the class name at module scope would make pytest try to collect it as a test class.
Policy = artifact_policy.TestArtifactPolicy

ADDOPTS = ("-p", "no:cacheprovider")


def test_sanitized_addopts_survive_to_dict_and_from_mapping() -> None:
    policy = Policy(sanitized_addopts=ADDOPTS)

    reloaded = Policy.from_mapping(policy.to_dict())

    assert reloaded == policy
    assert reloaded is not None
    assert reloaded.sanitized_addopts == ADDOPTS


def test_payload_without_sanitized_addopts_loads_as_empty_tuple() -> None:
    payload = Policy(pytest_config="pytest.ini", coverage_requested=True).to_dict()
    payload.pop("sanitized_addopts", None)

    reloaded = Policy.from_mapping(payload)

    assert reloaded is not None
    assert reloaded.sanitized_addopts == ()


def test_malformed_sanitized_addopts_loads_as_empty_tuple() -> None:
    # A bare string, and a list with a non-str item, are both malformed and must be ignored, never crash.
    for junk in ("-p no:cacheprovider", ["-p", 3], [None]):
        payload = Policy(pytest_config="pytest.ini").to_dict()
        payload["sanitized_addopts"] = junk

        reloaded = Policy.from_mapping(payload)

        assert reloaded is not None, junk
        assert reloaded.sanitized_addopts == (), junk


def test_empty_addopts_leave_the_serialized_policy_unchanged() -> None:
    """With no replayed addopts the payload has no sanitized_addopts key, so request bytes match earlier releases."""
    assert "sanitized_addopts" not in Policy().to_dict()
    assert Policy.from_mapping(Policy().to_dict()).sanitized_addopts == ()
