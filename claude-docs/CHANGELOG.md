## 2026-11-05

### StateFrame test orphan guard — leak reporting
- Accept Python signal enums at the retained Linux pidfd and macOS audit-token boundaries, preserving bound-handle cleanup.
- Clear stale `ctypes` errno before the macOS identity probe so an inspection gap cannot be mistaken for an exited process.
- Promote guard findings into pytest's teardown report so a leak or unverifiable cleanup cannot be reported as a passing item.
- Files touched: `tests/_orphan_guard.py`, `tests/test_orphan_guard_unit.py`, `CHANGELOG.md`.
