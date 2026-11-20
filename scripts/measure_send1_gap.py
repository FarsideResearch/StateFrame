"""Measure a fresh launch through the first request and transport handoff.

Run from a prepared repo copy. The observer leaves the launch command, its
environment and its output files intact. Example::

    python scripts/measure_send1_gap.py --run-dir /scratch/run --state cold --trace -- COMMAND ...

For matched comparisons use --plan with a JSON array of launch objects. Each
object has label (baseline or candidate), state (cold or warm), run_dir and
command (an argv array). Optional cwd and env fields override the inherited
working directory and environment. List adjacent pairs in execution order.
Use fresh run dirs and prepare the repo copies and installation seed state
before calling this script. Cold/warm describes the declared installation
seed state. --trace uses Linux strace and can perturb elapsed time.

Wall intervals are exclusive. CPU, runqueue counters, and syscall spans
describe observations that can overlap. CPU deltas cover only sample pairs
inside a phase. Process CPU includes its threads. Host pressure includes other
workloads. A file mtime precedes atomic publication, and first poll visibility
follows it. Transport time uses a recorded monotonic handoff when available.
The taped body and a matching socket connect can bound that handoff.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import lzma
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from stateframe.engine.kernel_journal_store import journal_paths, open_journal


SECOND = 1_000_000_000
MARKER = Path("requests/0001.json")
WIRE = Path("wire/0001.request.json")
JOURNAL_PHASES = (
    "run_started", "tool_menu_composed", "state_owner_inventory", "request_prepared",
    "qualified_request_sealed", "request_final_wire_prepared", "request_submitted",
)
SYSCALLS = (
    "clone", "clone3", "fork", "vfork", "execve", "execveat", "flock",
    "rename", "renameat", "renameat2", "connect", "clock_nanosleep",
    "nanosleep", "fsync", "fdatasync", "wait4",
)
WAIT_SYSCALLS = {"clock_nanosleep", "nanosleep", "fsync", "fdatasync"}
SPAWN_SYSCALLS = {"clone", "clone3", "fork", "vfork"}
TRACE_LINE = re.compile(r"^(\d+)\s+(\d+\.\d+) (.*)$")
CALL = re.compile(r"^(\w+)\(")
RESUMED = re.compile(r"^<\.\.\. (\w+) resumed>(.*)$")
RESULT = re.compile(r"\)\s+=\s+(-?\d+).* <(\d+(?:\.\d+)?)>$")


@dataclass(frozen=True)
class Marker:
    ns: int
    source: str
    uncertainty_ns: int = 0


@dataclass(frozen=True)
class Clock:
    mono_ns: int
    wall_ns: int
    error_ns: int

    @classmethod
    def read(cls):
        before = time.monotonic_ns()
        wall = time.time_ns()
        after = time.monotonic_ns()
        return cls((before + after) // 2, wall, (after - before + 1) // 2)

    @property
    def offset(self):
        return self.wall_ns - self.mono_ns


class ClockMapping:
    """Convert realtime observations only while the two clocks agree."""

    def __init__(self, clocks):
        self.clocks = clocks
        self.walls = [clock.wall_ns for clock in clocks]
        self.errors = []
        for before, after in zip(clocks, clocks[1:]):
            # Allow clock slewing and the uncertainty of both clock reads.
            error = before.error_ns + after.error_ns + max(
                50_000, (after.mono_ns - before.mono_ns) // 1000)
            if after.wall_ns <= before.wall_ns or abs(after.offset - before.offset) > error:
                self.errors = []
                break
            self.errors.append(error)

    def marker(self, wall_ns, source, resolution_ns=0):
        index = bisect_right(self.walls, wall_ns) - 1
        if 0 <= index < len(self.errors):
            before, after = self.clocks[index:index + 2]
            offset = before.offset + (
                (after.offset - before.offset) * (wall_ns - before.wall_ns)
                // (after.wall_ns - before.wall_ns)
            )
            return Marker(wall_ns - offset, source, self.errors[index] + resolution_ns)
        return None


@dataclass(frozen=True)
class Call:
    tid: int
    name: str
    args: str
    result: int
    start_ns: int
    end_ns: int
    uncertainty_ns: int


def decimal_ns(value):
    whole, _, fraction = value.partition(".")
    return int(whole) * SECOND + int((fraction + "000000000")[:9])


def trace_calls(path, clocks):
    """Read completed syscalls. Interrupted lines are joined by TID."""
    pending = {}
    with path.open(errors="replace") as stream:
        for line in stream:
            match = TRACE_LINE.match(line)
            if match is None:
                continue
            tid, timestamp, text = match.groups()
            tid = int(tid)
            returned = None
            if text.endswith("<unfinished ...>"):
                pending[tid] = (timestamp, text.removesuffix("<unfinished ...>"))
                continue
            resumed = RESUMED.match(text)
            if resumed:
                previous = pending.pop(tid, None)
                if previous is None or not previous[1].startswith(resumed[1] + "("):
                    continue
                returned = clocks.marker(decimal_ns(timestamp), "syscall_return", 1000)
                timestamp, prefix = previous
                text = prefix + resumed[2]
            call, result = CALL.match(text), RESULT.search(text)
            if call is None or result is None:
                continue
            start = clocks.marker(decimal_ns(timestamp), "syscall", 1000)
            if start is not None:
                end = returned.ns if returned else start.ns + decimal_ns(result[2])
                yield Call(tid, call[1], text[call.end():result.start()], int(result[1]),
                           start.ns, end, max(start.uncertainty_ns,
                                              returned.uncertainty_ns if returned else 0))


def read_text(path):
    try:
        return path.read_text(errors="replace")
    except OSError:
        return None


def records(path):
    """Read complete existing journal records, including a sealed journal."""
    try:
        with open_journal(path, binary=True) as stream:
            for line in stream:
                if not line.endswith(b"\n"):
                    break
                try:
                    record = json.loads(line)
                except (ValueError, UnicodeError):
                    continue
                if isinstance(record, dict):
                    yield record
    except (OSError, EOFError, lzma.LZMAError):
        return


def file_identity(path):
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def process_spawn(call):
    return call.name in SPAWN_SYSCALLS and call.result > 0 and "CLONE_THREAD" not in call.args


def collect_markers(run, launch, clocks, calls, published):
    """Read monotonic and calibrated markers with their distinct evidence sources."""
    markers = {"launch": Marker(launch, "monotonic_before_popen")}
    for key, (mtime, seen) in published.items():
        markers[key] = (clocks.marker(mtime, "mtime_before_publish")
                        or Marker(seen, "first_poll_seen"))
        markers[key + "_observed"] = Marker(seen, "first_poll_seen")
    lock = (read_text(run / ".engine.lock") or "").strip()
    child = int(lock) if lock.isascii() and lock.isdecimal() else None
    pid = next((call.tid for call in calls
                if call.name in {"execve", "execveat"} and call.result == 0), None)
    if child is not None:
        spawn = next((call for call in calls if process_spawn(call) and call.result == child), None)
        if spawn:
            # A vfork can return after the child's exec. Its entry is the
            # observed spawn boundary, not the instant the child first ran.
            markers["child_spawn"] = Marker(spawn.start_ns, "spawn_syscall_entry", spawn.uncertainty_ns)
        for call in calls:
            if call.tid != child or call.result != 0:
                continue
            if call.name in {"execve", "execveat"}:
                markers.setdefault("child_exec", Marker(call.end_ns, "exec_syscall_return", call.uncertainty_ns))
            if call.name == "flock" and ".engine.lock>" in call.args and "LOCK_EX" in call.args:
                markers.setdefault("child_lock", Marker(call.end_ns, "successful_flock", call.uncertainty_ns))
    for call in calls:
        if call.name in {"rename", "renameat", "renameat2"} and call.result == 0:
            for key, path in (("request_marker", MARKER), ("transport_body", WIRE)):
                destination = json.dumps(str(run / path))
                arguments = call.args.rstrip()
                if call.name == "renameat2":
                    arguments = arguments.rsplit(", ", 1)[0]
                if arguments.endswith(f", {destination}"):
                    markers.setdefault(key + "_syscall", Marker(
                        call.end_ns, "atomic_rename_return", call.uncertainty_ns))
    for key in ("request_marker", "transport_body"):
        if key + "_syscall" in markers:
            markers[key] = markers.pop(key + "_syscall")

    journals = journal_paths(run / "worker/measurement-runs")
    for journal in journals:
        for event in records(journal):
            kind = event.get("kind")
            stamp = event.get("timestamp_ns")
            if kind not in JOURNAL_PHASES or type(stamp) is not int:
                continue
            if event.get("step") not in (None, 0) or event.get("provider_request") is False:
                continue
            marker = clocks.marker(stamp, "journal_timestamp")
            if marker is not None and marker.ns >= launch:
                markers.setdefault(kind, marker)
        if "run_started" in markers:
            break

    try:
        spec = json.loads(read_text(run / "spec.json") or "{}")
        start = spec.get("turn_launch", {}).get("prepare_started_ns")
        handoff = next((entry.get("prepare_to_first_transport_ns")
                        for entry in records(run / "sends.jsonl") if entry.get("send") == 1), None)
        if type(start) is int and type(handoff) is int and handoff >= 0 and start >= launch:
            markers["transport_entry"] = Marker(start + handoff, "recorded_monotonic_handoff")

        # A matching TCP connect following the taped body is a physical
        # boundary. Keep it distinct from entry to the application transport.
        endpoint = urlsplit(spec.get("chat_url", ""))
        host = endpoint.hostname
        port = endpoint.port or (443 if endpoint.scheme == "https" else 80)
        body = markers.get("transport_body")
        if child and body and body.source == "atomic_rename_return" and host:
            # The HTTP operation can run in a dedicated process spawned
            # by the engine after body publication.
            callers = {child} | {
                call.result for call in calls if process_spawn(call)
                and call.tid == child and call.start_ns >= body.ns
            }
            connect = min((call for call in calls if call.name == "connect" and call.tid in callers
                           and call.start_ns >= body.ns and f'"{host}"' in call.args
                           and f"htons({port})" in call.args),
                          default=None, key=lambda call: call.start_ns)
            if connect:
                markers["transport_connect"] = Marker(
                    connect.start_ns, "matching_connect_entry", connect.uncertainty_ns)
    except (ValueError, AttributeError, TypeError):
        pass
    return markers, pid, child


def cpu_stat(path):
    value = read_text(path)
    if value is None:
        return None
    fields = value.rsplit(")", 1)[-1].split()
    try:
        return (int(fields[19]), int(fields[2]), int(fields[11]) + int(fields[12]),
                int(fields[39]) if len(fields) > 39 else None)
    except (ValueError, IndexError):
        return None


def sched_stat(path):
    try:
        runtime, delay, slices = map(int, (read_text(path) or "").split())
        return runtime, delay, slices
    except ValueError:
        return None


def pressure(proc):
    totals = {}
    for resource in ("cpu", "io", "memory"):
        for line in (read_text(proc / "pressure" / resource) or "").splitlines():
            match = re.search(r"^(some|full) .*\btotal=(\d+)$", line)
            if match:
                totals[f"{resource}_{match[1]}_us"] = int(match[2])
    return totals


@dataclass
class Sample:
    start_ns: int
    end_ns: int
    tasks: dict
    pressure: dict


def sample_tree(root_pid, known, proc, ticks_ns):
    before = time.monotonic_ns()
    tasks = {}
    pending = [root_pid, *known]
    seen = set()
    delay_enabled = (read_text(proc / "sys/kernel/task_delayacct") or "").strip() == "1"
    schedstats_enabled = (read_text(proc / "sys/kernel/sched_schedstats") or "").strip() == "1"
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        root = proc / str(pid)
        process = cpu_stat(root / "stat")
        if process is None or (pid in known and process[0] != known[pid]):
            continue
        identity, _, ticks, _ = process
        known[pid] = identity
        tasks[pid, None, identity] = {"cpu_ns": ticks * ticks_ns}
        try:
            threads = list((root / "task").iterdir())
        except OSError:
            continue
        for thread in threads:
            pending.extend(int(n) for n in (read_text(thread / "children") or "").split()
                           if n.isdecimal())
            stat = cpu_stat(thread / "stat")
            if stat is None:
                continue
            started, _, ticks, io = stat
            counts = {"cpu_ns": ticks * ticks_ns}
            sched = sched_stat(thread / "schedstat")
            if sched is not None:
                runtime, delay, slices = sched
                if runtime or delay or slices:
                    counts.update(scheduled_cpu_ns=runtime, timeslices=slices)
                if schedstats_enabled or delay:
                    counts["runqueue_delay_ns"] = delay
            if delay_enabled and io is not None:
                counts["block_io_delay_ns"] = io * ticks_ns
            tasks[pid, int(thread.name), started] = counts
    host_pressure = pressure(proc)
    return Sample(before, time.monotonic_ns(), tasks, host_pressure)


def count_deltas(before, after, accum):
    for key in before.keys() & after.keys():
        if after[key] >= before[key]:
            accum[key] = accum.get(key, 0) + after[key] - before[key]


def union_duration(spans):
    end = 0
    total = 0
    for start, stop in sorted(spans):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def attribute_intervals(markers, samples, calls, end, parent_pid, child_pid, period_ns):
    """Partition wall once. Report counter deltas and concurrent work separately."""
    launch = markers["launch"].ns
    spawn = markers.get("child_spawn")
    lock = markers.get("child_lock")
    ordered = sorted((mark.ns, name) for name, mark in markers.items()
                     if name != "launch" and launch <= mark.ns <= end)
    boundaries = [(launch, "launch"), *ordered, (end, "observation_end")]
    phases = []
    sample_pairs = list(zip(samples, samples[1:]))
    for (start, left), (stop, right) in zip(boundaries, boundaries[1:]):
        tasks = defaultdict(dict)
        host_pressure = {}
        covered = []
        for before, after in sample_pairs:
            if start <= before.start_ns <= after.end_ns <= stop:
                covered.append((before.end_ns, after.start_ns))
                count_deltas(before.pressure, after.pressure, host_pressure)
                for key in before.tasks.keys() & after.tasks.keys():
                    pid, tid, identity = key
                    counters = tasks[pid, tid, identity]
                    counters["covered_ns"] = counters.get("covered_ns", 0) + after.start_ns - before.end_ns
                    count_deltas(before.tasks[key], after.tasks[key], counters)
        waits = defaultdict(list)
        parent_waits = []
        journal_syncs = []
        for call in calls:
            lo, hi = max(start, call.start_ns), min(stop, call.end_ns)
            if lo >= hi:
                continue
            if call.name in WAIT_SYSCALLS and call.result == 0:
                waits[call.tid, call.name].append((lo, hi))
                if call.name in {"fsync", "fdatasync"} and "/events.jsonl>" in call.args:
                    journal_syncs.append((lo, hi))
            if call.name == "wait4" and call.tid == parent_pid and call.result == child_pid:
                parent_waits.append((lo, hi))
        coverage_ns = union_duration(covered)
        scope = "unresolved"
        if spawn and stop <= spawn.ns:
            scope = "before_child_spawn"
        elif spawn and lock and start >= spawn.ns and stop <= lock.ns:
            scope = "child_spawn_to_lock"
        elif lock and start >= lock.ns:
            scope = "after_child_lock"
        primary = None
        if scope == "before_child_spawn":
            primary = parent_pid
        elif scope != "unresolved":
            primary = child_pid
        phases.append({
            "from": left, "to": right, "wall_ns": stop - start,
            "scope": scope,
            "coverage_ns": coverage_ns, "sample_pairs": len(covered),
            "cpu_and_scheduler": [
                {"pid": pid, "tid": tid, "start_ticks": identity,
                 "role": "parent" if pid == parent_pid else "child" if pid == child_pid else "helper",
                 **counters}
                for (pid, tid, identity), counters in sorted(
                    tasks.items(), key=lambda row: (row[0][0], row[0][1] or 0, row[0][2]))
            ],
            "host_pressure_us": host_pressure or None,
            "observed_syscalls": [
                {"tid": tid, "name": name, "wall_ns": union_duration(spans)}
                for (tid, name), spans in sorted(waits.items())
            ],
            "observed_syscall_union_ns": union_duration(
                [span for spans in waits.values() for span in spans]) if calls else None,
            "primary_syscall_union_ns": union_duration([
                span for (tid, _), spans in waits.items() if tid == primary for span in spans
            ]) if calls and primary is not None else None,
            "observed_journal_sync_ns": union_duration(journal_syncs) if calls else None,
            "observed_parent_wait_for_child_ns": union_duration(parent_waits) if parent_waits else None,
        })
    return {
        "phases": phases,
        "observation_wall_ns": end - launch,
        "launch_to_marker_ns": (markers["request_marker"].ns - launch
                                if "request_marker" in markers else None),
        "spawn_to_lock_ns": (lock.ns - spawn.ns if spawn and lock and lock.ns >= spawn.ns else None),
        "sampling": {
            "ticks": len(samples), "period_ns": period_ns,
            "max_sample_ns": max((tick.end_ns - tick.start_ns for tick in samples), default=None),
            "max_tick_gap_ns": max((b.start_ns - a.start_ns for a, b in sample_pairs), default=None),
            "covered_ns": sum(phase["coverage_ns"] for phase in phases),
        },
    }


def stop_trial(process, known, proc):
    """Stop the measured command and any still-observed private process groups."""
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    for pid, identity in reversed(list(known.items())):
        stat = cpu_stat(proc / str(pid) / "stat")
        if stat is not None and stat[0] == identity:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def observe(case, trace, period, timeout, tail, proc=Path("/proc")):
    run = (Path(case.get("cwd", ".")) / case["run_dir"]).resolve()
    if run.exists():
        raise ValueError(f"fresh run directory required: {run}")
    command = case["command"]
    env = {**os.environ, **case.get("env", {})}
    run.parent.mkdir(parents=True, exist_ok=True)
    period_ns = round(period * SECOND)
    samples = []
    known = {}
    published = {}
    missed = 0
    load = read_text(proc / "loadavg")
    ticks_ns = SECOND // os.sysconf("SC_CLK_TCK")
    with tempfile.TemporaryDirectory(prefix="send-gap-", dir=run.parent) as scratch:
        trace_path = Path(scratch) / "syscalls"
        if trace:
            command = ["strace", "-f", "-qq", "-ttt", "-T", "-y", "-s", "0",
                       "--seccomp-bpf", "--kill-on-exit", "-e", "trace=" + ",".join(SYSCALLS),
                       "-o", str(trace_path), "--", *command]
        log_fd = os.open(str(run) + ".observer.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(log_fd, "wb") as log:
            clocks = [Clock.read()]
            launch = time.monotonic_ns()
            process = subprocess.Popen(command, cwd=case.get("cwd"), env=env,
                                       stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            target = launch
            deadline = launch + round(timeout * SECOND)
            body_seen = None
            try:
                while True:
                    tick = sample_tree(process.pid, known, proc, ticks_ns)
                    samples.append(tick)
                    skipped = max(0, (tick.start_ns - target) // period_ns)
                    missed += skipped
                    target += (skipped + 1) * period_ns
                    clocks.append(Clock.read())
                    for key, path in (("request_marker", MARKER), ("transport_body", WIRE)):
                        if key in published:
                            continue
                        try:
                            mtime = (run / path).stat().st_mtime_ns
                        except FileNotFoundError:
                            continue
                        seen = time.monotonic_ns()
                        published[key] = (mtime, seen)
                        if key == "transport_body":
                            body_seen = seen
                    now = time.monotonic_ns()
                    if process.poll() is not None:
                        status = "complete" if len(published) == 2 else "early_exit"
                        break
                    if len(published) == 2 and now - body_seen >= round(tail * SECOND):
                        status = "complete"
                        break
                    if now >= deadline:
                        status = "timeout"
                        break
                    time.sleep(max(0, min(target, deadline) - time.monotonic_ns()) / SECOND)
            finally:
                end = time.monotonic_ns()
                clocks.append(Clock.read())
                observer_stopped = process.poll() is None
                stop_trial(process, known, proc)
        mapping = ClockMapping(clocks)
        calls = list(trace_calls(trace_path, mapping)) if trace_path.exists() else []

    markers, traced_pid, child_pid = collect_markers(run, launch, mapping, calls, published)
    markers = {name: mark for name, mark in markers.items() if launch <= mark.ns <= end}
    # First-poll timestamps bound publication latency. They do not divide
    # phase wall time when a better timestamp exists.
    phase_marks = {name: mark for name, mark in markers.items() if not name.endswith("_observed")}
    if trace:
        for sample in samples:
            sample.tasks = {key: counters for key, counters in sample.tasks.items()
                            if key[0] != process.pid}
    result = attribute_intervals(phase_marks, samples, calls, end,
                                 traced_pid if trace else process.pid, child_pid, period_ns)
    result["sampling"].update(missed_ticks=missed, cpu_tick_ns=ticks_ns,
                              wall_clock_calibrated=bool(mapping.errors),
                              schedstats_setting=(read_text(proc / "sys/kernel/sched_schedstats") or "").strip() or None,
                              block_io_setting=(read_text(proc / "sys/kernel/task_delayacct") or "").strip() or None)
    seed = read_text(run / "env-seed.json")
    try:
        seed = json.loads(seed) if seed else None
    except ValueError:
        seed = None
    return {
        **result, "status": status, "label": case.get("label"), "seed_state": case["state"],
        "seed_evidence": seed, "trace_requested": trace, "trace_observed": bool(calls),
        "loadavg_before": load.strip() if load else None, "loadavg_after": read_text(proc / "loadavg"),
        "launcher_exit_code": process.returncode, "observer_stopped_launcher": observer_stopped,
        "run_dir": str(run),
        "markers": {name: {"elapsed_ns": mark.ns - launch, "source": mark.source,
                           "uncertainty_ns": mark.uncertainty_ns}
                    for name, mark in markers.items()},
        "request_marker": file_identity(run / MARKER), "transport_body": file_identity(run / WIRE),
        "transport_entry_bounds_ns": {
            "after": markers["transport_body"].ns - launch,
            "before": markers["transport_connect"].ns - launch,
        } if {"transport_body", "transport_connect"} <= markers.keys() else None,
    }


def compare_pairs(results):
    pairs = []
    for baseline, candidate in zip(results[::2], results[1::2]):
        before = baseline["launch_to_marker_ns"]
        after = candidate["launch_to_marker_ns"]
        pairs.append({
            "seed_state": baseline["seed_state"],
            "baseline_run": baseline["run_dir"],
            "candidate_run": candidate["run_dir"],
            "marker_delta_ns": after - before if before is not None and after is not None else None,
            "request_marker_equal": equal_capture(baseline, candidate, "request_marker"),
            "transport_body_equal": equal_capture(baseline, candidate, "transport_body"),
        })
    return pairs


def equal_capture(before, after, key):
    if before[key] is None or after[key] is None:
        return None
    return before[key] == after[key]


def parse_plan(path):
    plan = json.loads(path.read_text())
    if not isinstance(plan, list) or not plan or len(plan) % 2:
        raise ValueError("plan must contain adjacent baseline/candidate pairs")
    for baseline, candidate in zip(plan[::2], plan[1::2]):
        if (
            not isinstance(baseline, dict) or not isinstance(candidate, dict)
            or baseline.get("label") != "baseline" or candidate.get("label") != "candidate"
            or baseline.get("state") not in ("cold", "warm")
            or baseline["state"] != candidate.get("state")
        ):
            raise ValueError("each pair must be baseline then candidate with matching cold/warm state")
    for entry in plan:
        if (
            not isinstance(entry.get("run_dir"), str) or not entry["run_dir"]
            or not isinstance(entry.get("command"), list) or not entry["command"]
            or not all(isinstance(arg, str) for arg in entry["command"])
            or not isinstance(entry.get("cwd", ""), str)
            or not isinstance(entry.get("env", {}), dict)
            or not all(isinstance(val, str) for val in entry.get("env", {}).values())
        ):
            raise ValueError("plan entry requires a run_dir and an argv array, with optional cwd and env")
    return plan


FIXTURE = """
import fcntl, json, os, pathlib, socket, subprocess, sys, threading, time

run = pathlib.Path(sys.argv[1])
if len(sys.argv) == 2:
    run.mkdir()
    start = time.monotonic_ns()
    (run / "spec.json").write_text(json.dumps({"turn_launch": {"prepare_started_ns": start}}))
    time.sleep(0.08)
    subprocess.run([sys.executable, sys.argv[0], str(run), "child"], check=True)
else:
    time.sleep(0.12)
    lock = (run / ".engine.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    lock.write(str(os.getpid()) + "\\n")
    lock.flush()
    journal = run / "worker/measurement-runs/fixture/events.jsonl"
    journal.parent.mkdir(parents=True)
    def emit(kind):
        with journal.open("a") as stream:
            stream.write(json.dumps({"kind": kind, "timestamp_ns": time.time_ns()}) + "\\n")
            stream.flush()
            os.fsync(stream.fileno())
    emit("run_started")
    ready = threading.Event()
    def helper():
        ready.set()
        time.sleep(0.50)
    concurrent = threading.Thread(target=helper)
    concurrent.start()
    ready.wait()
    time.sleep(0.18)
    emit("request_prepared")
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    port = server.getsockname()[1]
    spec_path = run / "spec.json"
    spec = json.loads(spec_path.read_text())
    spec["chat_url"] = f"http://127.0.0.1:{port}/test"
    spec_path.write_text(json.dumps(spec))
    for name, data in (("requests/0001.json", b'{"marker":1}'),
                       ("wire/0001.request.json", b'{"wire":1}')):
        path = run / name
        path.parent.mkdir()
        pending = path.with_suffix(".tmp")
        pending.write_bytes(data)
        pending.replace(path)
    start = spec["turn_launch"]["prepare_started_ns"]
    entered = time.monotonic_ns()
    subprocess.run([sys.executable, "-c",
                    "import socket, sys\\nsocket.create_connection(('127.0.0.1', int(sys.argv[1]))).close()",
                    str(port)], check=True)
    server.close()
    (run / "sends.jsonl").write_text(json.dumps(
        {"send": 1, "prepare_to_first_transport_ns": entered - start}) + "\\n")
    concurrent.join()
"""


class AttributionTests(unittest.TestCase):
    def test_observer_does_not_count_parallel_sleep_twice(self):
        if not shutil.which("strace"):
            self.skipTest("strace unavailable")
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "launch.py"
            fixture.write_text(FIXTURE)
            cases = []
            for variant in ("baseline", "candidate"):
                run = str(Path(directory) / variant)
                cases.append({"run_dir": run, "state": "cold", "label": variant,
                              "command": [sys.executable, str(fixture), run]})
            plain = observe(cases[0], False, 0.01, 15, 1)
            traced = observe(cases[1], True, 0.01, 15, 1)
            self.assertEqual(traced["status"], "complete")
            self.assertEqual(compare_pairs([plain, traced])[0]["request_marker_equal"], True)
            self.assertEqual(compare_pairs([plain, traced])[0]["transport_body_equal"], True)
            self.assertGreaterEqual(traced["spawn_to_lock_ns"], 120_000_000)
            prep = next(p for p in traced["phases"] if p["to"] == "child_spawn")
            self.assertGreaterEqual(prep["wall_ns"], 80_000_000)
            self.assertGreaterEqual(prep["observed_syscall_union_ns"], 60_000_000)
            overlap = next(p for p in traced["phases"] if p["to"] == "request_prepared")
            self.assertGreaterEqual(overlap["wall_ns"], 180_000_000)
            sleeps = [s for s in overlap["observed_syscalls"] if s["name"] == "clock_nanosleep"]
            self.assertGreaterEqual(len(sleeps), 2)
            self.assertGreater(sum(s["wall_ns"] for s in sleeps), overlap["observed_syscall_union_ns"])
            self.assertLessEqual(overlap["observed_syscall_union_ns"], overlap["wall_ns"])
            self.assertGreater(overlap["observed_parent_wait_for_child_ns"], 150_000_000)
            self.assertGreater(sum(p["observed_journal_sync_ns"] for p in traced["phases"]), 0)
            self.assertEqual(traced["markers"]["request_marker"]["source"], "atomic_rename_return")
            self.assertEqual(traced["markers"]["transport_entry"]["source"], "recorded_monotonic_handoff")
            bounded = traced["transport_entry_bounds_ns"]
            handoff = traced["markers"]["transport_entry"]["elapsed_ns"]
            self.assertLessEqual(bounded["after"], handoff)
            self.assertLessEqual(handoff, bounded["before"])
            self.assertEqual(sum(p["wall_ns"] for p in traced["phases"]),
                             traced["observation_wall_ns"])

    def test_timeout_with_unavailable_counters(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            data = observe({"run_dir": str(run), "state": "unknown",
                            "command": [sys.executable, "-c", "import time; time.sleep(10)"]},
                           False, 0.01, 0.08, 0, proc=run / "no-proc")
            self.assertEqual(data["status"], "timeout")
            self.assertIsNone(data["request_marker"])
            self.assertIsNone(data["spawn_to_lock_ns"])
            self.assertIsNone(data["phases"][0]["host_pressure_us"])
            self.assertEqual(data["phases"][0]["cpu_and_scheduler"], [])
            self.assertIsNone(data["phases"][0]["observed_syscall_union_ns"])

    def test_clock_step_is_not_reported_as_a_wall_boundary(self):
        mapping = ClockMapping([
            Clock(10_000_000, 100_000_000, 100), Clock(20_000_000, 120_000_000, 100),
        ])
        self.assertIsNone(mapping.marker(115_000_000, "fixture"))

    def test_observer_reports_missed_ticks_after_a_slow_scan(self):
        def slow_sample(*args):
            sample = sample_tree_unpatched(*args)
            time.sleep(0.035)
            return sample

        sample_tree_unpatched = sample_tree
        with tempfile.TemporaryDirectory() as directory:
            case = {"run_dir": str(Path(directory) / "run"), "state": "unknown",
                    "command": [sys.executable, "-c", "import time; time.sleep(10)"]}
            with patch(__name__ + ".sample_tree", side_effect=slow_sample):
                result = observe(case, False, 0.01, 0.15, 0)
            self.assertGreaterEqual(result["sampling"]["missed_ticks"], 2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--state", choices=("cold", "warm", "unknown"), default="unknown")
    parser.add_argument("--trace", action="store_true", help="trace lifecycle and selected syscalls on Linux")
    parser.add_argument("--period", type=float, default=0.02, help="sampling interval in seconds")
    parser.add_argument("--timeout", type=float, default=120, help="maximum observation time per launch")
    parser.add_argument("--tail", type=float, default=0.1, help="seconds to observe after transport publication")
    parser.add_argument("--expected-marker-sha256")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(AttributionTests)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    if args.trace and (not sys.platform.startswith("linux") or not shutil.which("strace")):
        parser.error("--trace requires Linux and strace")
    if not all(math.isfinite(n) for n in (args.period, args.timeout, args.tail)):
        parser.error("sampling and timeout durations must be finite")
    if args.period < 0.001 or args.timeout <= 0 or args.tail < 0:
        parser.error("--period must be at least 0.001, --timeout positive and --tail nonnegative")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        if args.plan and not (args.run_dir or command):
            plan = parse_plan(args.plan)
        elif args.run_dir and command and not args.plan:
            plan = [{"run_dir": str(args.run_dir), "state": args.state, "command": command}]
        else:
            parser.error("provide --plan or --run-dir with a launch command")
        results = [observe(case, args.trace, args.period, args.timeout, args.tail) for case in plan]
    except (OSError, ValueError) as error:
        parser.exit(2, f"{parser.prog}: {error}\n")
    pairs = compare_pairs(results) if args.plan else []
    print(json.dumps({"runs": results, "pairs": pairs}, indent=2, sort_keys=True))
    failed = any(result["status"] != "complete" for result in results)
    failed |= any(pair[key] is not True for pair in pairs
                  for key in ("request_marker_equal", "transport_body_equal"))
    if args.expected_marker_sha256:
        failed |= any((result["request_marker"] or {}).get("sha256") != args.expected_marker_sha256
                      for result in results)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
