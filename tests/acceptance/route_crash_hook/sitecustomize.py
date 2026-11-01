"""Test-only crash barriers injected through --python by test_routes_host."""
import builtins
import os
from pathlib import Path
import threading


def _block() -> None:
    Path(os.environ["ROUTE_CRASH_MARKER"]).touch()
    threading.Event().wait(120)


if os.environ.get("ROUTE_CRASH_POINT") == "after_route:2":
    _replace = os.replace

    def _replace_then_block(source, target):
        result = _replace(source, target)
        if str(target).endswith("/0002.route.json"):
            _block()
        return result

    os.replace = _replace_then_block

elif os.environ.get("ROUTE_CRASH_POINT") == "after_discard":
    _import = builtins.__import__

    def _patch_journal(module):
        journal = getattr(module, "AuditJournal", None)
        if journal is None or getattr(journal, "_route_crash_wrapped", False):
            return
        original = journal.emit

        def emit_then_block(self, kind, **fields):
            result = original(self, kind, **fields)
            if kind == "route_escalation_discarded":
                _block()
            return result

        journal.emit = emit_then_block
        journal._route_crash_wrapped = True

    def _import_then_patch(name, globals=None, locals=None, fromlist=(), level=0):
        module = _import(name, globals, locals, fromlist, level)
        if name == "kernel_measurement":
            _patch_journal(module)
        return module

    builtins.__import__ = _import_then_patch
