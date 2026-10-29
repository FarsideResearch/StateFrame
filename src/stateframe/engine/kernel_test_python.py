"""Select the host-declared interpreter for project checks and installs."""
from __future__ import annotations

import os
import sys


def test_python() -> str:
    """Use the product's test interpreter only when it is an executable absolute file."""
    candidate = os.environ.get("STATEFRAME_TEST_PYTHON")
    if candidate and os.path.isabs(candidate) and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    return sys.executable
