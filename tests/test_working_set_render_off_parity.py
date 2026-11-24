"""Whole-request OFF parity against the release before allocator support."""


def test_off_wire_bytes_match_release_before_allocator(tmp_path, tmp_path_factory):
    from test_working_set_off_parity_base import (
        test_off_provider_bytes_match_release_before_allocator,
    )

    test_off_provider_bytes_match_release_before_allocator(tmp_path, tmp_path_factory)
