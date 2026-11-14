"""Expose the L4 integration files without collecting fixture repositories."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "tests" / "fixtures"))
sys.path.insert(0, str(_ROOT / "tests" / "fixtures" / "parallel"))


@pytest.hookimpl(tryfirst=True)
def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    """Override the broad fixture ignore only for the L4 integration modules."""
    e2e_root = Path(__file__).resolve().parent
    candidate = Path(str(collection_path))
    if (candidate.parent == e2e_root and candidate.suffix == ".py"
            and candidate.name.startswith("test_parallel_")):
        return False
    return None


@pytest.fixture(scope="session")
def isolated_engine_python(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Install this checkout once into a fresh interpreter for CLI runs."""
    from e2e_support import create_engine_venv

    return create_engine_venv(tmp_path_factory.mktemp("parallel-engine") / "venv")
