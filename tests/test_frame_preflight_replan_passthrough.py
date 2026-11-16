"""Frame-only preflight must pass replan signals through, and the authority message must name owner_protocol keys."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import capability_request_preflight as preflight
from kernel_frame_optional_allocation import FrameAllocationReplanNeeded
from kernel_frame_allocation_validation import _authority_mismatch_message
from kernel_request_seal import RequestSealError


def _framefit_runtime() -> SimpleNamespace:
    return SimpleNamespace(stateframe_framefit_mode="on", current_request_id="req-test")


def _force_apply_failure(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    """Make the first step of the preflight try block raise the given error."""

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise error

    monkeypatch.setattr(preflight, "_apply_request_host_text", _raise)


def test_replan_signal_passes_through_frame_only_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    replan = FrameAllocationReplanNeeded(
        "scoped allocation frame differs from its unit authority (fields=['owner_protocol'])"
    )
    _force_apply_failure(monkeypatch, replan)

    with pytest.raises(FrameAllocationReplanNeeded) as excinfo:
        preflight.preflight_presentation_or_degrade([], {}, _framefit_runtime())

    assert excinfo.value is replan
    assert not isinstance(excinfo.value, RequestSealError)


def test_other_errors_still_wrap_as_request_seal_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_apply_failure(monkeypatch, ValueError("unrelated preflight failure"))

    with pytest.raises(RequestSealError, match="frame-only request preflight failed") as excinfo:
        preflight.preflight_presentation_or_degrade([], {}, _framefit_runtime())

    assert isinstance(excinfo.value.__cause__, ValueError)


def test_authority_message_lists_differing_owner_protocol_keys() -> None:
    frame = {
        "owner_protocol": {"continuation_limits": {"max": 4}, "stable": 1, "only_frame": True},
        "read_facts": [],
    }
    projection = {
        "owner_protocol": {"continuation_limits": {"max": 5}, "stable": 1, "only_projection": "x"},
        "read_facts": [],
    }

    message = _authority_mismatch_message(["owner_protocol"], frame, projection)

    assert message == (
        "scoped allocation frame differs from its unit authority "
        "(fields=['owner_protocol'], owner_protocol_keys="
        "['continuation_limits', 'only_frame', 'only_projection'])"
    )
    # Key names only; no values leak into the message.
    assert "max" not in message
    assert "only_projection': 'x'" not in message


def test_authority_message_unchanged_for_other_fields_and_non_mapping_owner() -> None:
    frame = {"owner_protocol": {"a": 1}, "read_facts": [1]}
    projection = {"owner_protocol": {"a": 2}, "read_facts": [2]}

    other_only = _authority_mismatch_message(["read_facts"], frame, projection)
    assert other_only == "scoped allocation frame differs from its unit authority (fields=['read_facts'])"

    non_mapping = _authority_mismatch_message(
        ["owner_protocol"], {"owner_protocol": "text"}, {"owner_protocol": {"a": 1}}
    )
    assert non_mapping == (
        "scoped allocation frame differs from its unit authority (fields=['owner_protocol'])"
    )
