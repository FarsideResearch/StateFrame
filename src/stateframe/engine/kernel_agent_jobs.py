"""Read-only background jobs with frozen request snapshots and bounded waits."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
import sys
import time

from kernel_job_output import JobOutputCollector
from kernel_shell_tool import ShellCleanupFailed


@dataclass
class Job:
    command: str
    handle: object
    output: JobOutputCollector | None
    state: str = "running"
    exit_code: int | None = None
    cursor: int = 0
    events: list[str] = field(default_factory=list)


class AgentJobs:
    def __init__(self, shell, deadline: float):
        self.shell = shell
        self.deadline = deadline
        self.jobs: dict[str, Job] = {}
        self.sequence = 0
        self.frozen = None
        self.pending = {}

    def _drain(self, job):
        if job.handle is None or job.state != "running":
            return
        job.output.feed(job.handle.read_available())
        code = job.handle.poll()
        if code is not None:
            # poll is leader exit; stop verifies descendant quiescence as well.
            report = job.handle.stop()
            if not report.ok:
                raise ShellCleanupFailed(f"REFUSED CLEANUP_FAILED: {report.detail}")
            job.output.feed(job.handle.read_available())
            job.exit_code, job.state = code, "exited"
            job.events.append("exited")
            job.output.seal()

    def capability(self, ident, timeout=600):
        scratch = self.shell.run_dir / "jobs" / ident
        scratch.mkdir(parents=True, exist_ok=True)
        cap = self.shell.capability
        env = dict(cap.env)
        env.update(HOME=str(scratch), TMPDIR=str(scratch))
        return replace(cap, purpose="job", workspace_writable=False, write_roots=(scratch,), env=env,
                       limits=replace(cap.limits, wall_seconds=timeout))

    def launch(self, args):
        timeout = args.get("timeout_seconds", 600)
        if type(timeout) is not int or not 1 <= timeout <= 600:
            return "REFUSED DENIED_PROGRAM: timeout_seconds must be an integer from 1 to 600"
        for job in self.jobs.values():
            self._drain(job)
        if sum(job.state == "running" for job in self.jobs.values()) >= 4:
            return "REFUSED JOB_LIMIT: at most four running jobs"
        ident = f"j{len(self.jobs) + 1}"
        cap = self.capability(ident, timeout)
        handle = self.shell.session.spawn(["/bin/sh", "-c", args["command"]], cap, cwd=self.shell.root)
        spool = self.shell.run_dir / "job-output"
        spool.mkdir(mode=0o700, exist_ok=True)
        collector = JobOutputCollector(spool / (ident + ".ring"))
        self.jobs[ident] = Job(args["command"], handle, collector)
        return f"started job {ident}: {args['command'][:200]}\n{self.shell.settings.shell} | network {cap.network}"

    def replay_launch(self, args, observation):
        if observation.startswith("started job "):
            ident = observation.split(":", 1)[0].split()[-1]
            self.jobs[ident] = Job(args["command"], None, None, "lost on restart")

    def _tail(self, job, limit=4096, lines=200):
        if job.output is None:
            return ""
        raw = job.output.snapshot()
        text = raw.decode("utf-8", "replace")
        text = "\n".join(text.splitlines()[-lines:])
        from kernel_shell_policy import normalize_output
        shown = normalize_output(text.encode()[-limit:], replacements={str(self.shell.run_dir): "<run>"})
        omitted = max(0, job.output.total_bytes - len(shown.encode()))
        lost = job.output.lost_bytes + getattr(job.handle, "lost_bytes", 0)
        if omitted or lost:
            shown = f"[{omitted} bytes omitted; {lost} bytes lost]\n" + shown
        return shown

    def execute(self, name, args):
        ident = args.get("job")
        if not isinstance(ident, str) or ident not in self.jobs:
            return "REFUSED UNKNOWN_JOB: job does not exist in this run"
        job = self.jobs[ident]
        self._drain(job)
        if name == "job_stop":
            report = job.handle.stop() if job.handle is not None else None
            if report is not None and not report.ok:
                raise ShellCleanupFailed(f"REFUSED CLEANUP_FAILED: {report.detail}")
            self._drain(job)
            job.state = "stopped" if job.handle is not None else "lost on restart"
            return f"job {ident} stopped ({report.killed if report else 0} processes)"
        if name == "job_output":
            lines = args.get("tail_lines", 200)
            if type(lines) is not int or not 1 <= lines <= 200:
                return "REFUSED BAD_CONDITION: tail_lines must be an integer from 1 to 200"
            return f"job {ident}: {job.state} (exit {job.exit_code})\n{self._tail(job, lines=lines)}"
        until, timeout = args.get("until"), args.get("timeout_seconds", 120)
        text, port = args.get("text"), args.get("port")
        if (until not in {"exit", "output", "endpoint"} or type(timeout) is not int or not 1 <= timeout <= 600
                or (until == "output" and (not isinstance(text, str) or not text or len(text.encode()) > 65536))
                or (until == "endpoint" and (type(port) is not int or not 1 <= port <= 65535))):
            return "REFUSED BAD_CONDITION: invalid wait condition, text, port or timeout_seconds"
        started = time.monotonic()
        deadline = min(self.deadline, started + timeout)
        matcher = job.output.watch(text) if until == "output" and job.output is not None else None
        result, event = "timeout", ""
        try:
            while True:
                self._drain(job)
                if matcher is not None and matcher.matched:
                    result, event = "met", "output matched"
                    break
                if until == "exit" and job.state in {"exited", "stopped"}:
                    result = "met"
                    break
                if job.state != "running":
                    result = "exited"
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                if until == "endpoint" and self._probe(port, remaining):
                    result, event = "met", "endpoint reachable"
                    break
                time.sleep(min(0.02, max(0, deadline - time.monotonic())))
        finally:
            if matcher is not None:
                job.output.unwatch(matcher)
        if event:
            job.events.append(event)
        return (f"job {ident}: {result} (exit {job.exit_code}) after {time.monotonic()-started:.2f}s"
                f"{'; ' + event if event else ''}\n{self._tail(job)}")

    def _probe(self, port, remaining):
        # Session.run shares the Linux supervisor's network namespace. Never
        # probe the host namespace for a Linux job.
        cap = replace(self.shell.capability, purpose="probe", workspace_writable=False,
                      write_roots=(), limits=replace(self.shell.capability.limits,
                                                     wall_seconds=min(1, remaining)))
        source = ("import socket,sys\ntry:\n s=socket.create_connection(('127.0.0.1',int(sys.argv[1])),.2)\n"
                  " s.close()\nexcept OSError:\n sys.exit(1)\n")
        result = self.shell.session.run([sys.executable, "-I", "-c", source, str(port)], cap, cwd=self.shell.root)
        if not result.cleanup.ok:
            raise ShellCleanupFailed(f"REFUSED CLEANUP_FAILED: {result.cleanup.detail}")
        return result.exit_code == 0

    def snapshot(self, observe):
        if self.frozen is None:
            self.sequence += 1
            def build():
                rows = []
                for ident, job in self.jobs.items():
                    self._drain(job)
                    tail = ""
                    if job.output is not None:
                        raw, cursor, lost = job.output.read_since(job.cursor)
                        from kernel_shell_policy import normalize_output
                        tail = normalize_output(raw[-1024:], replacements={str(self.shell.run_dir): "<run>"})
                        self.pending[ident] = (cursor, len(job.events))
                        if lost:
                            tail = f"[{lost} bytes lost]\n" + tail
                        tail = tail.encode()[-1024:].decode("utf-8", "ignore")
                    rows.append({"id": ident, "command": job.command[:80], "state": job.state,
                                 "exit_code": job.exit_code, "events": job.events[-4:], "tail": tail})
                # Retain active jobs and the newest completed summaries; bounded
                # history remains on the tool tape outside the prompt.
                active = [r for r in rows if r["state"] in {"running", "lost on restart"}]
                completed = [r for r in rows if r["state"] not in {"running", "lost on restart"}][-4:]
                for row in completed:
                    row["tail"] = ""
                return {"sequence": self.sequence, "jobs": active + completed,
                        "omitted_completed": max(0, len(rows)-len(active)-len(completed))}
            self.frozen = observe(self.sequence, build)
        return copy.deepcopy(self.frozen)

    def commit(self, journal):
        if self.frozen is None or getattr(self, "committed", False):
            return
        journal.emit("job_snapshot_committed", sequence=self.sequence)
        for ident, (cursor, events) in self.pending.items():
            self.jobs[ident].cursor = cursor
            del self.jobs[ident].events[:events]
        self.pending.clear()
        # Keep the snapshot for fitting/retries of this request. The next menu
        # boundary explicitly releases it after the committed request.
        self.committed = True

    def next_request(self, key):
        if key != getattr(self, "request_key", None) and getattr(self, "committed", False):
            self.frozen = None
            self.committed = False
        self.request_key = key

    def stop_all(self):
        if not self.jobs:
            return ""
        report = self.shell.session.stop_all()
        if not report.ok:
            raise ShellCleanupFailed(f"REFUSED CLEANUP_FAILED: {report.detail}")
        for job in self.jobs.values():
            self._drain(job)
        return f"stopped all jobs ({report.killed} processes)"


def render_snapshot(snapshot):
    lines = [f"[JOBS sequence={snapshot['sequence']}]"]
    for row in snapshot["jobs"]:
        lines.append(f"{row['id']} {row['state']} exit={row['exit_code']} {row['command']}")
        if row["events"]:
            lines.append("; ".join(row["events"]))
        if row["tail"]:
            lines.append(row["tail"])
    if snapshot.get("omitted_completed"):
        lines.append(f"{snapshot['omitted_completed']} older completed jobs omitted")
    return "\n".join([*lines, "[/JOBS]"])
