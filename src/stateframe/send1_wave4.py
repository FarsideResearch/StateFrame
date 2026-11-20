"""Select the startup pilot without importing any of its implementations."""
from contextlib import contextmanager
from contextvars import ContextVar
import os

from . import _STARTUP_SPEC_KEY as SPEC_KEY, _STARTUP_SWITCH as SWITCH

_selection = ContextVar("send1_wave4_selection", default=None)
_child_selected = False


def enabled() -> bool:
    selected = _selection.get()
    if selected is not None:
        return selected
    return _child_selected or os.environ.get(SWITCH, "").strip().lower() == "on"


@contextmanager
def launch_selection(selected: bool):
    token = _selection.set(selected)
    try:
        yield
    finally:
        _selection.reset(token)


@contextmanager
def child_selection():
    """Use the recorded decision across all threads in the engine process."""
    global _child_selected
    previous = _child_selected
    _child_selected = True
    try:
        yield
    finally:
        _child_selected = previous
