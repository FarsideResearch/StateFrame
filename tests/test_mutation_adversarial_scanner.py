"""Self-tests for the adversarial suite's zero-leftover scanner."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import signal
import sys
import time

import pytest
from fixtures.process_signals import signal_run_owned

from kernel_exec_tracker import ProcessTable
from mutate_leftovers import assert_no_leftovers, snapshot_leftovers


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="Linux /proc supplies recorded runtime process facts",
)
def test_mutation_leftover_scanner_detects_recorded_process_and_runtime_cwd(tmp_path):
    mutation_root = tmp_path / "runtime" / "run-acceptance"
    mutation_root.mkdir(parents=True)
    ready = mutation_root / "child-ready"
    source = (
        "from pathlib import Path\n"
        "import sys, time\n"
        "Path(sys.argv[1]).write_text('ready', encoding='utf-8')\n"
        "time.sleep(30)\n"
    )
    before = snapshot_leftovers(mutation_root)
    process = subprocess.Popen(
        [sys.executable, "-c", source, str(ready)], cwd=mutation_root,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        close_fds=True, start_new_session=True,
    )
    ledger = mutation_root / "launches.jsonl"
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), "scanner child did not reach its ready barrier"
        identity = ProcessTable().identity(process.pid)
        assert identity is not None
        row = {"schema": "stateframe-mutation-launch/1", "state": "RELEASED",
               "identity": {"pid": identity.pid,
                            "start": [identity.start[0], identity.start[1]],
                            "boot_id": "test-boot"}}
        ledger.write_text(json.dumps(row) + "\n", encoding="utf-8")

        live = snapshot_leftovers(mutation_root)
        with pytest.raises(AssertionError, match="leftovers"):
            assert_no_leftovers(before, live)
        assert live.runtime_processes
        cwd_only = replace(live, launch_identities=frozenset(), processes=frozenset())
        with pytest.raises(AssertionError, match="runtime root"):
            assert_no_leftovers(before, cwd_only)
    finally:
        if process.poll() is None:
            signal_run_owned(mutation_root, process, signal.SIGKILL,
                             deadline=time.monotonic(),
                             what="scanner fixture cleanup", require_checkpoint_saved=False)
        process.wait(timeout=5)
        # Remember the durable identity before removing the synthetic ledger row.
        snapshot_leftovers(mutation_root)
        ledger.unlink(missing_ok=True)
        ready.unlink(missing_ok=True)

    after = snapshot_leftovers(mutation_root)
    # Concurrent host processes may create unrelated matching /tmp names.
    after = replace(after, tmp_entries=before.tmp_entries)
    assert_no_leftovers(before, after)
