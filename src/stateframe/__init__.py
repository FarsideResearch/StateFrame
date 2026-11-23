"""StateFrame: an agent engine whose model requests carry a typed state frame instead of chat history."""
__version__ = "0.8.22"

_STARTUP_SWITCH = "STATEFRAME_SEND1_WAVE4"
_STARTUP_SPEC_KEY = "send1_wave4"


def _startup_optimization_enabled() -> bool:
    import os
    import sys

    # A child or host scope can already hold a recorded selection. Otherwise
    # leave the selector unloaded until a launch opts in.
    module = "stateframe.send1_wave4"
    if module not in sys.modules and os.environ.get(_STARTUP_SWITCH, "").strip().lower() != "on":
        return False
    try:
        from stateframe.send1_wave4 import enabled
    except ModuleNotFoundError as error:
        if error.name != module:
            raise
        return False
    return enabled()
