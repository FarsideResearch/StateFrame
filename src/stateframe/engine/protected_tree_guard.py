"""Fail before protected writes, including subprocesses; never mutate either tree.

Run Python tools through: python3 protected_tree_guard.py -m pytest ...
The OS envelope protects native/subprocess writes. The Python audit hook also
records caught write attempts so a test cannot swallow a denial and report green.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import runpy
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PROTECTED = (ROOT / "results", ROOT / "results.SNAPSHOT-20260920")


class WriteGuard:
    def __init__(self, roots=PROTECTED):
        self.roots = tuple(Path(root).resolve() for root in roots)
        self.violations: list[str] = []

    def check(self, path, *, dir_fd=None):
        if isinstance(path, int) or path is None:
            return
        candidate = Path(os.fsdecode(path))
        if not candidate.is_absolute() and dir_fd not in (None, -1):
            from corpus_of_record import directory_path
            candidate = Path(directory_path(dir_fd)) / candidate
        candidate = candidate.resolve()
        if any(candidate == root or candidate.is_relative_to(root) for root in self.roots):
            self.violations.append(str(candidate))
            raise PermissionError(f"protected-tree write denied before mutation: {candidate}")

    def audit(self, event, args):
        if event == "open":
            path, mode, flags = args
            if (flags or 0) & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                self.check(path)
        elif event in {"os.remove", "os.rmdir", "os.mkdir", "os.chmod", "os.chown", "os.utime", "os.truncate"}:
            index = {"os.remove": 1, "os.rmdir": 1, "os.mkdir": 2, "os.chmod": 2,
                     "os.chown": 3, "os.utime": 3}.get(event)
            self.check(args[0], dir_fd=args[index] if index is not None and len(args) > index else None)
        elif event in {"os.rename", "os.link"}:
            self.check(args[0], dir_fd=args[2] if len(args) > 2 else None)
            self.check(args[1], dir_fd=args[3] if len(args) > 3 else None)
        elif event == "os.symlink":
            self.check(args[1], dir_fd=args[2] if len(args) > 2 else None)


def guarded_command(command, roots=PROTECTED):
    roots = tuple(dict.fromkeys(Path(root).resolve() for root in (*PROTECTED, *roots)))
    if platform.system() == "Darwin":
        backend = shutil.which("sandbox-exec")
        if not backend:
            raise RuntimeError("sandbox-exec is required for indirect-worker protection")
        paths = " ".join(f"(subpath {json.dumps(str(root))})" for root in roots)
        return [backend, "-p", f"(version 1) (allow default) (deny file-write* {paths})", *command]
    backend = shutil.which("bwrap")
    if not backend:
        raise RuntimeError("bwrap is required; no unguarded fallback")
    args = [backend, "--die-with-parent", "--bind", "/", "/", "--dev", "/dev"]
    for root in roots:
        args += ["--ro-bind", str(root), str(root)]
    return [*args, "--", *command]


def main():
    if sys.argv[1:2] != ["--inside"]:
        return subprocess.call(guarded_command([sys.executable, str(Path(__file__)), "--inside", *sys.argv[1:]]))
    guard = WriteGuard()
    sys.addaudithook(guard.audit)
    from corpus_of_record import install_read_routing
    install_read_routing()
    args = sys.argv[2:]
    exit_code = 0
    try:
        if args[:1] == ["-m"] and len(args) >= 2:
            sys.argv = [args[1], *args[2:]]
            runpy.run_module(args[1], run_name="__main__", alter_sys=True)
        elif args:
            sys.argv = args
            runpy.run_path(args[0], run_name="__main__")
        else:
            raise ValueError("provide -m MODULE or a Python script")
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else (1 if exc.code else 0)
    finally:
        if guard.violations:
            print("PROTECTED WRITE ATTEMPTS: " + json.dumps(guard.violations), file=sys.stderr)
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
