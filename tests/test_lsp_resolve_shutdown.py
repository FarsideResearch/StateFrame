"""A timed-out code-graph lookup must let the engine interpreter exit."""
import os
import select
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from _timeouts import scaled


def test_stalled_lookup_allows_interpreter_exit(tmp_path):
    engine = Path(__file__).resolve().parents[1] / "src/stateframe/engine"
    (tmp_path / "sample.py").write_text("target()\n", encoding="utf-8")
    program = textwrap.dedent("""\
        import faulthandler
        import os
        import sys
        from pathlib import Path
        from repo_intel_core import lsp_resolve as resolver

        # Model a backend that has started reading a reply from a live, silent peer.
        def stalled_lookup(*args):
            Path("lookup-started").touch()
            os.read(0, 1)

        resolver.is_enabled = lambda: True
        resolver._resolve_sites_blocking = stalled_lookup
        sites = [resolver.CallSiteRef(1, 0, "target")]
        assert resolver.resolve_file_full(".", "sample.py", sites, timeout=0.1) == {}
        assert Path("lookup-started").exists()
        faulthandler.dump_traceback_later(float(sys.argv[1]))
        print("resolver returned", flush=True)
    """)
    # Keep the writer open across wait(). communicate() would send EOF and mask the bug.
    with subprocess.Popen(
        [sys.executable, "-c", program, str(scaled(1))], cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(engine)},
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    ) as child:
        try:
            readable, _, _ = select.select([child.stdout], [], [], scaled(30))
            assert readable and child.stdout.readline() == b"resolver returned\n"
            try:
                assert child.wait(timeout=scaled(3)) == 0
            except subprocess.TimeoutExpired:
                pytest.fail("interpreter exit timed out after the resolver returned")
        finally:
            child.stdin.close()
            if child.poll() is None:
                child.kill()
