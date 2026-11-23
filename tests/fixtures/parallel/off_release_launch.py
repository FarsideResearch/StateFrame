"""Supply repeatable host inputs to isolated release-parity CLI runs."""

import random
import runpy
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace


class RunClock:
    def __init__(self, *measurement_functions):
        self.measurements = {fn.__code__ for fn in measurement_functions}

    def monotonic(self):
        # Reporting uses a fixed clock. Deadlines and supervision use the real clock.
        return 1000.0 if sys._getframe(1).f_code in self.measurements else time.monotonic()

    def __getattr__(self, name):
        return getattr(time, name)


class RunDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls.fromtimestamp(1_700_000_000, tz)


def install_child_clock():
    run_path = runpy.run_path

    def run_child(path, *, run_name):
        runpy.run_path = run_path
        assert run_name == "__main__"
        child = run_path(path, run_name="release_parity_child")
        # Install clock inputs before calling the regular child CLI with its real argv.
        sys.path.insert(0, str(Path(path).with_name("engine")))
        from kernel_exec_session import ExecSession

        run_command = ExecSession.run

        def run_with_duration(self, *args, **kwargs):
            return replace(run_command(self, *args, **kwargs), wall_seconds=0.25)

        ExecSession.run = run_with_duration
        # Newer children run and time their work in _main; older ones do it in main.
        child["main"].__globals__["time"] = RunClock(child["main"], child.get("_main", child["main"]))
        raise SystemExit(child["_run_cli"](sys.argv))

    runpy.run_path = run_child


def launch_host(cli_args):
    from stateframe import cli, runner

    runner.datetime = RunDateTime
    runner.time = RunClock(runner.run, runner.summarize)
    rng = random.Random(0)
    runner.secrets = SimpleNamespace(token_hex=lambda length: rng.randbytes(length).hex())

    preamble = f"import runpy;runpy.run_path({__file__!r})['install_child_clock']();"

    def popen(command, **kwargs):
        if command[-1] == "stateframe._child":
            assert command[2] == "-c"
            command = [*command[:3], preamble + command[3], *command[4:]]
        elif len(command) > 2 and command[-2] == str(runner.CHILD):
            bootstrap = ("import sys;from pathlib import Path;sys.argv.pop(0);"
                         "sys.path[0]=str(Path(sys.argv[0]).parent);"
                         "runpy.run_path(sys.argv[0],run_name='__main__')")
            command = [*command[:-2], "-c", preamble + bootstrap, *command[-2:]]
        return subprocess.Popen(command, **kwargs)

    class RunnerSubprocess:
        Popen = staticmethod(popen)

        def __getattr__(self, name):
            return getattr(subprocess, name)

    runner.subprocess = RunnerSubprocess()
    sys.argv = [sys.argv[0], *cli_args]
    return cli.main()


if __name__ == "__main__":
    raise SystemExit(launch_host(sys.argv[1:]))
