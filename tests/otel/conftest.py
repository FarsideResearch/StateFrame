"""Private evidence fixtures independent of the pytest temp tree's permissions."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest


@pytest.fixture
def private_root():
    # The reader checks every ancestor. chmod on a leaf under tmp_path cannot
    # make a shared pytest base directory or its ancestors trustworthy.
    with TemporaryDirectory(prefix="stateframe-otel-") as directory:
        yield Path(directory)
