"""Import-compatible entry point for the history compactor.

The ReAct script is also run directly from ``probes/`` in historical commands,
so the implementation lives beside that script while this module keeps the
lane's normal ``PYTHONPATH=.../src`` import contract intact.
"""

from probes.history_compaction import *  # noqa: F401,F403
from probes.history_compaction import __all__
