import subprocess
import sys

import pytest

from kernel_check_permits import acquire, active, install


def test_install_active_and_timeout(tmp_path):
    install(tmp_path / 'permits', 1)
    assert active() == (tmp_path / 'permits', 1)
    with acquire(tmp_path / 'permits', 1, timeout=0.1) as number:
        assert number == 0
        with pytest.raises(TimeoutError, match='no check permit'):
            with acquire(tmp_path / 'permits', 1, timeout=0.01):
                pass


def test_two_processes_are_exclusive_and_kill_releases(tmp_path):
    root = tmp_path / 'permits'
    script = tmp_path / 'hold.py'
    source = '''import sys, time
sys.path.insert(0, sys.argv[2])
from kernel_check_permits import acquire
from pathlib import Path
with acquire(Path(sys.argv[1]), 1, timeout=float(sys.argv[3])) as number:
    print(number, flush=True)
    if sys.argv[4] == 'hold':
        time.sleep(30)
'''
    script.write_text(source)
    engine = str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src/stateframe/engine')
    first = subprocess.Popen([sys.executable, '-I', str(script), str(root), engine, '2', 'hold'], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
    try:
        assert first.stdout.readline().strip() == '0'
        second = subprocess.run([sys.executable, '-I', str(script), str(root), engine, '0.1', 'once'],
                                capture_output=True, text=True)
        assert second.returncode != 0
        assert 'TimeoutError' in second.stderr
        with pytest.raises(TimeoutError):
            with acquire(root, 1, timeout=0.02):
                pass
    finally:
        first.kill()
        first.wait(timeout=5)
    with acquire(root, 1, timeout=1) as number:
        assert number == 0


def test_two_permits_allow_two_holders_and_refuse_third(tmp_path):
    root = tmp_path / 'two'
    script = tmp_path / 'hold_two.py'
    script.write_text('import sys, time\n'
                      'sys.path.insert(0, sys.argv[2])\n'
                      'from kernel_check_permits import acquire\n'
                      'from pathlib import Path\n'
                      'with acquire(Path(sys.argv[1]), 2, timeout=float(sys.argv[3])) as number:\n'
                      '    print(number, flush=True)\n'
                      '    if sys.argv[4] == "hold": time.sleep(30)\n')
    engine = str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src/stateframe/engine')
    holders = []
    try:
        for _ in range(2):
            child = subprocess.Popen([sys.executable, '-I', str(script), str(root), engine, '2', 'hold'],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            holders.append(child)
        numbers = [child.stdout.readline().strip() for child in holders]
        assert set(numbers) == {'0', '1'}
        third = subprocess.run([sys.executable, '-I', str(script), str(root), engine, '0.1', 'once'],
                               capture_output=True, text=True)
        assert third.returncode != 0
        assert 'TimeoutError' in third.stderr
    finally:
        for child in holders:
            child.kill()
            child.wait(timeout=5)
    with acquire(root, 2, timeout=1) as number:
        assert number in (0, 1)


def test_acquire_closes_descriptor_once_when_body_raises(tmp_path, monkeypatch):
    import kernel_check_permits as permits
    original = permits.os.close
    closed = []
    def tracking(fd):
        closed.append(fd)
        return original(fd)
    monkeypatch.setattr(permits.os, 'close', tracking)
    with pytest.raises(RuntimeError, match='injected'):
        with acquire(tmp_path / 'permits', 1, timeout=1):
            raise RuntimeError('injected')
    assert len(closed) == 1
    with acquire(tmp_path / 'permits', 1, timeout=1):
        pass
