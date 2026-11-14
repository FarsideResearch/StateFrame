"""Whole-request OFF parity against the assigned cutover base."""
import pytest


@pytest.mark.parametrize("reference", ['56af0dc6d6', 'd7fd3aad93'])
def test_off_wire_bytes_match_1de34aa4(tmp_path, tmp_path_factory, reference):
    from test_working_set_astra_off_parity import test_off_provider_bytes_match_cutover_base

    test_off_provider_bytes_match_cutover_base(tmp_path, tmp_path_factory, reference)
