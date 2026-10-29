"""Execute one immutable public probe in a private, read-only product snapshot."""
from __future__ import annotations

import hashlib
import os
import signal
from pathlib import Path
import subprocess
import tempfile
import time

from capability_study_support import test_process_lock
from kernel_test_isolation import isolated_command
from kernel_test_python import test_python
from kernel_public_probe_sandbox import RUNNER_SOURCE


def execute_probe(root: Path, product_files: dict, source: str, output: Path, *,
                  deadline: float | None = None) -> dict:
    if not product_files or len(product_files) > 100:
        raise ValueError('public probe requires a bounded product snapshot')
    contents = {}
    for relative in product_files:
        path = Path(relative)
        if (path.is_absolute() or '..' in path.parts or path.as_posix() != relative
                or not path.parts
                or path.name in {'conftest.py', 'sitecustomize.py', 'usercustomize.py', 'pytest.py'}):
            raise ValueError('unsafe public product path')
        original = root / path
        if original.is_symlink() or not original.resolve().is_relative_to(root.resolve()):
            raise ValueError('product snapshot escapes workspace')
        if original.stat().st_size > 4_000_000 - sum(map(len, contents.values())):
            raise ValueError('public product snapshot exceeds size bound')
        contents[relative] = original.read_bytes()
    if sum(map(len, contents.values())) > 4_000_000:
        raise ValueError('public product snapshot exceeds size bound')
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
    if any(product_files[name] != 'sha256:' + value for name, value in hashes.items()):
        raise ValueError('product changed while taking public probe snapshot')
    with tempfile.TemporaryDirectory(prefix='public-probe-', dir=output) as temporary:
        scratch = Path(temporary)
        for name, data in contents.items():
            path = scratch / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        test = scratch / 'tests/test_public_probe.py'
        test.parent.mkdir()
        test.write_text(source)
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in scratch.rglob('*.py')}
        # Bound log writes inside the child as well as wall time; never buffer arbitrary output in RAM.
        logfile = scratch / '.test-tmp/output.txt'
        logfile.parent.mkdir(parents=True, exist_ok=True)
        lock = Path(os.environ.get('CAPABILITY_TEST_LOCK', str(output / 'public-probe-pytest.lock')))
        timed_out = False
        with test_process_lock(lock, output / 'public-probe-test-lock.jsonl', deadline=deadline):
            allowance = 15 if deadline is None else min(15, deadline - time.monotonic())
            if allowance <= 0:
                raise TimeoutError('public probe deadline expired before execution')
            with logfile.open('xb') as stream:
                from kernel_execution_policy import sandboxed_popen
                with sandboxed_popen(
                    [
                        test_python(), '-I', '-c', RUNNER_SOURCE, '-q', '-x', '--noconftest',
                        '-c', '/dev/null', '--rootdir', str(scratch), '-p', 'no:cacheprovider',
                        '-o', f'pythonpath={scratch}', 'tests/test_public_probe.py',
                    ],
                    scratch,
                    env=dict(os.environ),
                    stdin=subprocess.DEVNULL,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    readonly_workspace=True,
                    cpu_seconds=15,
                    memory_bytes=1024 * 1024 * 1024,
                    file_bytes=1024 * 1024,
                    isolation_builder=isolated_command,
                )[0] as process:
                    try:
                        code = process.wait(timeout=allowance)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                        code, timed_out = -1, True
        if any(not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != value
               for path, value in before.items()):
            raise ValueError('public probe modified read-only sources')
        with logfile.open('rb') as stream:
            size = logfile.stat().st_size
            stream.seek(max(0, size - 6000))
            text = stream.read(6000).decode(errors='replace')
    return {'exit_code': code, 'timeout': timed_out, 'output': text,
            'omitted_output_bytes': max(0, size - 6000), 'candidate_file_hashes': hashes,
            'authority': 'UNVERIFIED_MODEL_PROPOSAL',
            'result': 'PROBE_GREEN' if code == 0 else 'PROBE_RED' if code == 1 else 'PROBE_UNAVAILABLE'}
