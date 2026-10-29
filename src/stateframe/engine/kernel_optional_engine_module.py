"""Load operator-configured engine extensions only at their point of use."""

from __future__ import annotations

import importlib
import os
from types import ModuleType


def optional_module(variable: str) -> ModuleType | None:
    name = os.environ.get(variable, "").strip()
    if not name:
        return None
    try:
        return importlib.import_module(name)
    except ImportError:
        return None
