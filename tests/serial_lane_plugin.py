"""Keep corpus cases serial and record their per-session execution."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

import pytest


_SERIAL_LANE_CONFIG: pytest.Config | None = None
_SERIAL_LANE_COLLECTED_PROPERTY = "serial_lane_collected"


def configure_serial_lane(config: pytest.Config) -> None:
    global _SERIAL_LANE_CONFIG
    if getattr(config, "_serial_lane_initialized", False):
        return
    _SERIAL_LANE_CONFIG = config
    config._serial_lane_initialized = True
    config._serial_lane_collected = 0
    config._serial_lane_deferred = 0
    config._serial_lane_ran = 0
    config._serial_lane_item_ids = set()
    config._serial_lane_reported_nodeids = set()
    config._serial_lane_deferred_nodeids = set()
    config._serial_lane_ran_nodeids = set()
    config.addinivalue_line(
        "markers", "serial_lane: runs a recorded corpus case that must execute outside xdist workers"
    )


def _serial_lane_xdist_active(config: pytest.Config | None) -> bool:
    """Use pytest's worker config, not inherited worker environment variables."""
    if config is None:
        return False
    if hasattr(config, "workerinput"):
        return True
    option = getattr(config, "option", None)
    return getattr(option, "numprocesses", None) not in (None, 0, "0")


def _append_serial_lane_ledger(
    path: str | Path, *, collected: int, ran: int, deferred: int
) -> None:
    """Append one complete logical pytest-session record."""
    record = {"collected": collected, "ran": ran, "deferred": deferred}
    if any(type(value) is not int or value < 0 for value in record.values()):
        raise ValueError("serial_lane ledger counts must be non-negative integers")
    payload = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        remaining = memoryview(payload)
        while remaining:
            written = os.write(fd, remaining)
            if written == 0:
                raise OSError("could not append serial_lane ledger record")
            remaining = remaining[written:]
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def collect_serial_lane_items(
    items: list[pytest.Item], config: pytest.Config | None = None
) -> None:
    """Count serial corpus items and defer them whenever this session uses xdist."""
    serial_items = [item for item in items if item.get_closest_marker("serial_lane") is not None]
    count = len(serial_items)
    deferred = count if _serial_lane_xdist_active(config) else 0
    for item in serial_items:
        if not any(
            name == _SERIAL_LANE_COLLECTED_PROPERTY and value is True
            for name, value in item.user_properties
        ):
            item.user_properties.append((_SERIAL_LANE_COLLECTED_PROPERTY, True))
    if config is not None:
        config._serial_lane_collected = count
        config._serial_lane_deferred = deferred
        config._serial_lane_item_ids = {id(item) for item in serial_items}
    if not deferred:
        return

    reason = f"serial_lane: {count} corpus cases deferred to the serial step"
    skip = pytest.mark.skip(reason=reason)
    for item in serial_items:
        item.add_marker(skip)


def record_call(item: pytest.Item, report: pytest.TestReport) -> None:
    config = item.config
    if report.when == "call" and id(item) in config._serial_lane_item_ids:
        config._serial_lane_ran += 1


def record_logreport(report: pytest.TestReport) -> None:
    """Count distinct serial items from reports relayed to the xdist controller."""
    config = _SERIAL_LANE_CONFIG
    if config is None or hasattr(config, "workerinput") or not _serial_lane_xdist_active(config):
        return
    if "serial_lane" not in report.keywords or not any(
        name == _SERIAL_LANE_COLLECTED_PROPERTY and value is True
        for name, value in report.user_properties
    ):
        return

    config._serial_lane_reported_nodeids.add(report.nodeid)
    if report.when == "call":
        config._serial_lane_ran_nodeids.add(report.nodeid)
    elif report.when == "setup" and report.skipped:
        config._serial_lane_deferred_nodeids.add(report.nodeid)


def finish_session(session: pytest.Session) -> None:
    """Write one ledger record per logical run, from the controller under xdist."""
    config = session.config
    if hasattr(config, "workerinput"):
        return

    if _serial_lane_xdist_active(config):
        collected = max(config._serial_lane_collected, len(config._serial_lane_reported_nodeids))
        ran = max(config._serial_lane_ran, len(config._serial_lane_ran_nodeids))
        deferred = max(config._serial_lane_deferred, len(config._serial_lane_deferred_nodeids))
    else:
        collected = config._serial_lane_collected
        ran = config._serial_lane_ran
        deferred = config._serial_lane_deferred

    ledger = os.environ.get("SF_SERIAL_LANE_LEDGER")
    if ledger:
        _append_serial_lane_ledger(ledger, collected=collected, ran=ran, deferred=deferred)


def terminal_summary(
    terminalreporter: pytest.TerminalReporter, exitstatus: int, config: pytest.Config
) -> None:
    """Report the distinct number of corpus cases deferred to the serial step."""
    if hasattr(config, "workerinput"):
        return
    deferred = max(config._serial_lane_deferred, len(config._serial_lane_deferred_nodeids))
    if deferred:
        terminalreporter.write_line(
            f"serial_lane: {deferred} corpus cases deferred to the serial step"
        )
