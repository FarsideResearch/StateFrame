"""A wire recorder installed by one test must not intercept sends in the next one."""
from __future__ import annotations

import kernel_bounded_http
import pytest

from stateframe import _child

REAL = kernel_bounded_http.request_bytes


@pytest.mark.expected_installer_leak
def test_a_installs_a_routed_recorder(tmp_path):
    _child.install_wire_recorder(tmp_path, "http://127.0.0.1:1/v1/chat/completions",
                                 routed_endpoints={"http://127.0.0.1:1/v1/chat/completions"})
    assert kernel_bounded_http.request_bytes is not REAL


def test_b_sees_the_real_sender_again():
    # Runs after test_a in file order; the autouse fixture in conftest.py restored the sender.
    assert kernel_bounded_http.request_bytes is REAL
