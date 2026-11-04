"""Repository tests import the checkout's product package and engine modules."""
import sys
import importlib.metadata
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_RETIRED_PARITY_BASELINES = {
    "test_hook_wiring_parity.py": "immediate base release",
    "test_final_cache_layout_host.py": "e19c325",
    "test_edit_scope_widen.py": "0b057d3",
}
# Fixture repositories under tests/e2e carry their own tests; they run inside engine runs only.
collect_ignore_glob = ["e2e/*", "fixtures/*"]  # recorded repo copies in fixtures/ are data, never tests
for path in (ROOT / "src", ROOT / "src" / "stateframe" / "engine"):
    sys.path.insert(0, str(path))


@pytest.fixture(autouse=True, scope="session")
def _isolated_stateframe_home(tmp_path_factory):
    """Runs under test build env seeds and user state; keep them out of the real ~/.stateframe."""
    import os

    if os.environ.get("STATEFRAME_HOME"):
        yield
        return
    os.environ["STATEFRAME_HOME"] = str(tmp_path_factory.mktemp("stateframe-home"))
    try:
        yield
    finally:
        os.environ.pop("STATEFRAME_HOME", None)


@pytest.fixture(autouse=True)
def reject_retired_shared_parity_baseline(request):
    """Keep old foundry mounts from silently routing tests to the wrong source tree."""
    reference = _RETIRED_PARITY_BASELINES.get(request.node.path.name)
    if reference is not None and "SF_TEST_PARITY_BASELINE" in os.environ:
        pytest.fail(
            f"SF_TEST_PARITY_BASELINE is retired: {request.node.nodeid} "
            f"pins its own reference ({reference}); unset it",
            pytrace=False,
        )


# The session home is shared by every test in a worker. Powerups and skills in it change what the current tree
# sends, so a test that leaves one behind changes later tests' request bytes (11-04: the 0.2.8 parity test failed
# only in full runs). A test that needs such state sets its own STATEFRAME_HOME with monkeypatch.
_SESSION_HOME_STATE = ("powerups", "powerups.json", "skills")


def _session_home_state_snapshot(home):
    import hashlib
    from pathlib import Path

    state = {}
    for name in _SESSION_HOME_STATE:
        root = Path(home) / name
        paths = [root] if root.is_file() or root.is_symlink() else sorted(root.rglob("*")) if root.is_dir() else []
        for path in paths:
            relative = str(path.relative_to(home))
            if path.is_symlink():
                state[relative] = "link:" + str(path.readlink())
            elif path.is_file():
                state[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            else:
                state[relative] = "dir"
    return state


@pytest.fixture(autouse=True)
def guard_session_home_state(request):
    import os

    home = os.environ.get("STATEFRAME_HOME")
    before = _session_home_state_snapshot(home) if home else None
    yield
    # Compare against the same home: a test that monkeypatched its own home has restored the session one by now.
    if before is not None and os.environ.get("STATEFRAME_HOME") == home:
        after = _session_home_state_snapshot(home)
        if after != before:
            changed = sorted(set(before) ^ set(after) | {k for k in before.keys() & after.keys() if before[k] != after[k]})
            pytest.fail(f"{request.node.nodeid} left session STATEFRAME_HOME state changed: {', '.join(changed[:8])}; "
                        "set its own STATEFRAME_HOME with monkeypatch", pytrace=False)


# _child installers patch these process globals directly. A test must register the
# pristine value with monkeypatch before invoking an installer (or restore it in finally).
_MISSING = object()
_INSTALLER_ATTRIBUTES = (
    ("kernel_bounded_http", (), "request_bytes"),
    ("kernel_request_diet", ("RequestTokenMeter",), "measure"),
    ("kernel_bridge_evidence", (), "capture_candidate"),
    ("kernel_execution_policy", (), "run_sandboxed"),
    ("kernel_measurement", (), "_v30_accounting_patch"),
    ("probes.react_repair", (), "ask_tools"),
    ("probes.react_repair", (), "_materialize_public_tests"),
    ("probes.react_repair", (), "run_tests"),
    ("uuid", (), "uuid4"),
)


def _imported_installer_attributes():
    """Snapshot loaded modules without importing the engine as a fixture side effect."""
    found = {}
    for module_name, owner_path, attribute in _INSTALLER_ATTRIBUTES:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        owner = module
        for part in owner_path:
            owner = getattr(owner, part, None)
            if owner is None:
                break
        if owner is not None:
            found[(module_name, owner_path, attribute)] = (
                module, owner, getattr(owner, attribute, _MISSING))
    return found


def _original_from_child_wrapper(value):
    """Recover a pre-install callable when its module was first imported in this test."""
    while getattr(value, "__module__", None) == "stateframe._child":
        closure = dict(zip(value.__code__.co_freevars, value.__closure__ or ()))
        original = closure.get("original", closure.get("original_ask"))
        if original is None:
            break
        value = original.cell_contents
    return value


# Import-time installers set watched globals the moment their module is first imported. That state is the
# process baseline, not a test leak, so those modules are imported before the pristine snapshot. Collection no
# longer reliably imports them first (11-03, v0.3.29: a test that first imported kernel_stateframe_runtime was
# reported as leaking kernel_measurement._v30_accounting_patch). See tests/_import_time_installers.py.
@pytest.fixture(scope="session")
def import_time_installer_state():
    """Run every import-time installer once, so the pristine snapshot includes its process-global state."""
    import importlib

    from _import_time_installers import IMPORT_TIME_INSTALLER_MODULES

    for module_name in IMPORT_TIME_INSTALLER_MODULES:
        importlib.import_module(module_name)


@pytest.fixture(scope="session")
def pristine_installer_attributes(import_time_installer_state):
    # Session fixture setup follows collection, when test-module imports are complete, and runs after the
    # import-time installers so their state is the baseline.
    return _imported_installer_attributes()


@pytest.fixture(autouse=True)
def guard_child_installer_globals(request, pristine_installer_attributes):
    """Name the leaking test, then repair globals before another test can inherit them."""
    baseline = dict(pristine_installer_attributes)
    for name, value in _imported_installer_attributes().items():
        baseline.setdefault(name, value)
    child = sys.modules.get("stateframe._child")
    host_state = child.HOST_STATE if child is not None else None
    host_entries = dict(host_state) if host_state is not None else None

    yield

    leaked = []
    for (module_name, owner_path, attribute), (module, owner, original) in baseline.items():
        if sys.modules.get(module_name) is not module:
            continue
        if getattr(owner, attribute, _MISSING) is not original:
            leaked.append(f"{module_name}.{'.'.join((*owner_path, attribute))}")
            if original is _MISSING:
                delattr(owner, attribute)
            else:
                setattr(owner, attribute, original)
    # A test can import an engine module inside its body, after the setup snapshot.
    # Child wrappers retain their prior callable in a closure, so detect and unwind them.
    for name, (_, owner, value) in _imported_installer_attributes().items():
        if name in baseline:
            continue
        original = _original_from_child_wrapper(value)
        if original is not value:
            module_name, owner_path, attribute = name
            leaked.append(f"{module_name}.{'.'.join((*owner_path, attribute))}")
            setattr(owner, attribute, original)
    if child is not None and sys.modules.get("stateframe._child") is child:
        current = child.HOST_STATE
        if current is not host_state or (host_entries is not None and
                                         (current.keys() != host_entries.keys() or
                                          any(current[key] is not value for key, value in host_entries.items()))):
            leaked.append("stateframe._child.HOST_STATE")
            host_state.clear()
            host_state.update(host_entries)
            child.HOST_STATE = host_state
    elif child is None:
        newly_imported_child = sys.modules.get("stateframe._child")
        if newly_imported_child is not None and newly_imported_child.HOST_STATE:
            leaked.append("stateframe._child.HOST_STATE")
            newly_imported_child.HOST_STATE.clear()
    # A test that leaks on purpose (to prove this guard restores state) declares it; it must still leak.
    if request.node.get_closest_marker("expected_installer_leak") is not None:
        if not leaked:
            pytest.fail(f"{request.node.nodeid} is marked expected_installer_leak but leaked nothing", pytrace=False)
    elif leaked:
        pytest.fail(f"{request.node.nodeid} leaked child installer state: {', '.join(leaked)}", pytrace=False)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "host_sandbox: starts an OS sandbox (sandbox-exec/bwrap) or a local socket service")
    config.addinivalue_line(
        "markers", "expected_installer_leak: the test leaks _child installer state on purpose; the guard restores it")


@pytest.fixture(autouse=True)
def _restore_transport_sender():
    """_child.install_wire_recorder replaces kernel_bounded_http.request_bytes for the whole
    process. Put the real sender back after each test so a recorder installed by one test
    (for example a routed one) never intercepts a later test's sends."""
    import kernel_bounded_http

    original = kernel_bounded_http.request_bytes
    yield
    kernel_bounded_http.request_bytes = original


@pytest.fixture
def clean_test_environment(tmp_path):
    """Real pytest and pip, with no ambient editable installs or startup hooks."""
    def create(destination):
        subprocess.run([sys.executable, "-I", "-m", "venv", str(destination)],
                       check=True, capture_output=True, timeout=60)
        site = next(destination.glob("lib/python*/site-packages"))
        # Copy only installed distribution files. In particular, do not inherit
        # the host's editable StateFrame .pth or grant its source tree authority.
        for name in ("pytest", "pluggy", "iniconfig", "packaging", "pygments"):
            distribution = importlib.metadata.distribution(name)
            for relative in distribution.files:
                if ".." in relative.parts or relative.suffix == ".pyc":
                    continue
                source = Path(distribution.locate_file(relative))
                if source.is_file():
                    target = site / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
        return destination / "bin/python"
    return create
