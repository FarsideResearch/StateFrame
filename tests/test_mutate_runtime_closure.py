"""Python runtime closure and excluded extension containment."""
import builtins
from io import BytesIO
import importlib
import json
from pathlib import Path
import subprocess
import sys
import sysconfig
from types import SimpleNamespace

import pytest

import kernel_mutate_build as build
import kernel_mutate_exec as execute
import kernel_mutate_qualify as qualify
from mutate_leftovers import stage_test_unit
from test_mutate_exec_integration import _spec, _unit


@pytest.fixture
def fake_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(execute.sys, "platform", "linux")
    python = tmp_path / "python"
    python.write_bytes(b"python")
    stdlib = tmp_path / "stdlib"
    dynamic = stdlib / "lib-dynload"
    dynamic.mkdir(parents=True)
    (stdlib / "os.py").write_bytes(b"pass")
    (stdlib / "encodings").mkdir()
    (stdlib / "site-packages").mkdir()
    (stdlib / "dist-packages").mkdir()
    (dynamic / "README").write_bytes(b"extensions")
    excluded = dynamic / "_tkinter.fake.so"
    admitted = dynamic / "_struct.fake.so"
    excluded.write_bytes(b"optional")
    admitted.write_bytes(b"required")
    libraries = tmp_path / "libraries"
    libraries.mkdir()
    for name in ("libbase.so", "libgood.so", "libonly.so"):
        (libraries / name).write_bytes(name.encode())
    (stdlib / "alias.py").symlink_to(stdlib / "os.py")
    (dynamic / "alias.so").symlink_to(admitted)
    execute._runtime_roots_cached.cache_clear()
    yield python, stdlib, excluded, admitted, libraries
    execute._runtime_roots_cached.cache_clear()


def _ldd(binary_lines):
    return "".join(f"{binary}:\n" + "".join(f"\t{line}\n" for line in lines)
                   for binary, lines in binary_lines)


def _scan(monkeypatch, python, stdlib, output):
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(execute.subprocess, "run", run)
    closure = execute._runtime_roots_cached(str(python), str(stdlib), 0, 0)
    assert len(calls) == 1
    return closure


def test_optional_extension_is_excluded_without_its_libraries(fake_runtime, monkeypatch):
    python, stdlib, excluded, admitted, libraries = fake_runtime
    alias = stdlib / "lib-dynload" / "alias.so"
    output = _ldd([
        (python, [f"libbase.so => {libraries / 'libbase.so'} (0x1)"]),
        (alias, [f"libgood.so => {libraries / 'libgood.so'} (0x2)"]),
        (admitted, [f"libgood.so => {libraries / 'libgood.so'} (0x3)"]),
        (excluded, ["libtk8.6.so => not found", "libtcl8.6.so not found",
                    f"libonly.so => {libraries / 'libonly.so'} (0x4)"]),
    ])
    closure = _scan(monkeypatch, python, stdlib, output)
    assert closure.excluded == ((excluded.name, ("libtcl8.6.so", "libtk8.6.so")),)
    # One stdlib bind; the unloadable extension and package directories are masked inside it.
    assert set(closure.roots) == {python, stdlib, libraries / "libbase.so", libraries / "libgood.so"}
    assert libraries / "libonly.so" not in closure.roots
    assert set(closure.hidden) == {excluded, stdlib / "site-packages", stdlib / "dist-packages"}


def test_hidden_paths_resolve_inside_the_stdlib(fake_runtime, monkeypatch, tmp_path):
    python, stdlib, excluded, admitted, libraries = fake_runtime
    (stdlib / "dist-packages").rmdir()
    outside = tmp_path / "outside-packages"
    outside.mkdir()
    (stdlib / "dist-packages").symlink_to(outside)
    real = stdlib / "lib-dynload" / "_real_tk.so"
    excluded.rename(real)
    excluded.symlink_to(real)
    dynamic = stdlib / "lib-dynload"
    output = _ldd([
        (python, [f"libbase.so => {libraries / 'libbase.so'} (0x1)"]),
        (dynamic / "_real_tk.so", ["libtk8.6.so => not found"]),
        (dynamic / "_struct.fake.so", ["statically linked"]),
        (excluded, ["libtk8.6.so => not found"]),
        (dynamic / "alias.so", ["statically linked"]),
    ])
    closure = _scan(monkeypatch, python, stdlib, output)
    # A link to an excluded file masks its target; a link out of the stdlib is not mounted at all.
    assert real in closure.hidden
    assert outside not in closure.hidden and outside not in closure.roots
    assert all(path.resolve() == path for path in closure.hidden)


def test_missing_interpreter_dependency_fails_closed(fake_runtime, monkeypatch):
    python, stdlib, excluded, admitted, _ = fake_runtime
    output = _ldd([(python, ["libpython.so => not found"]),
                   (admitted, ["statically linked"]),
                   (excluded, ["statically linked"]),
                   (stdlib / "lib-dynload" / "alias.so", ["statically linked"])])
    monkeypatch.setattr(execute.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(stdout=output))
    with pytest.raises(execute.MutationPlatformError, match="runtime library dependency missing"):
        execute._runtime_roots_cached(str(python), str(stdlib), 0, 0)


def test_missing_binary_section_fails_closed(fake_runtime, monkeypatch):
    python, stdlib, excluded, admitted, _ = fake_runtime
    output = _ldd([(python, ["statically linked"]),
                   (admitted, ["statically linked"]),
                   (excluded, ["statically linked"])])
    monkeypatch.setattr(execute.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(stdout=output))
    with pytest.raises(execute.MutationPlatformError, match="cannot qualify Python runtime libraries"):
        execute._runtime_roots_cached(str(python), str(stdlib), 0, 0)


def test_single_binary_without_header(fake_runtime, monkeypatch):
    python, stdlib, _, _, libraries = fake_runtime
    for path in (stdlib / "lib-dynload").glob("*.so"):
        path.unlink()
    output = f"\tlibbase.so => {libraries / 'libbase.so'} (0x1)\n"
    closure = _scan(monkeypatch, python, stdlib, output)
    assert closure.excluded == ()
    assert libraries / "libbase.so" in closure.roots


def test_binding_records_exclusions_and_invalidates(monkeypatch, tmp_path):
    root = tmp_path / "stdlib.py"
    root.write_bytes(b"stdlib")
    monkeypatch.setattr(qualify, "_runtime_roots", lambda: (root,))
    monkeypatch.setattr(qualify, "_runtime_closure", lambda: execute.RuntimeClosure((root,), ()))
    first = qualify._binding()
    assert first["runtime_excluded"] == []
    monkeypatch.setattr(qualify, "_runtime_closure", lambda: execute.RuntimeClosure(
        (root,), (("_tkinter.fake.so", ("libtcl8.6.so", "libtk8.6.so")),)))
    second = qualify._binding()
    assert second["runtime_excluded"] == [["_tkinter.fake.so", ["libtcl8.6.so", "libtk8.6.so"]]]
    assert second != first


@pytest.mark.parametrize("exception,import_name", [
    ("ModuleNotFoundError", "_tkinter"),
    ("ImportError", "optional._tkinter.widgets"),
])
def test_execution_failure_names_excluded_extension_libraries(monkeypatch, exception, import_name):
    call = "missing-extension"
    message = f"No module named '{import_name}'"
    error = {"type": "HandlerError", "message": message,
             "trace": f'  File "/sf/unit/main.py", line 2, in handle\n{exception}: {message}'}
    body = json.dumps({"wire": 1, "call": call, "kind": "error",
                       "error": error, "stderr": ""}).encode()
    outcome = SimpleNamespace(stdout=b"SFMW1 " + str(len(body)).encode() + b"\n" + body,
                              stdout_overflow=False, timed_out=False, exit_code=3)
    monkeypatch.setattr(execute, "_runtime_closure", lambda: execute.RuntimeClosure((), (
        ("_tkinter.cpython-313-x86_64-linux-gnu.so", ("libtk8.6.so", "libtcl8.6.so")),)))
    detail = build.execution_failure_detail(outcome, "smoke", call)
    assert detail.startswith("unavailable extension _tkinter on this host "
                             "(not mounted: missing libtk8.6.so, libtcl8.6.so); "
                             "use a pure-Python alternative; smoke: boot error frame")
    assert len(detail) <= 1200

    other = dict(error, message="No module named 'unrelated'",
                 trace="ModuleNotFoundError: No module named 'unrelated'")
    other_body = json.dumps({"wire": 1, "call": call, "kind": "error",
                             "error": other, "stderr": ""}).encode()
    other_outcome = SimpleNamespace(stdout=b"SFMW1 " + str(len(other_body)).encode() + b"\n" + other_body,
                                    stdout_overflow=False, timed_out=False, exit_code=3)
    unchanged = build.execution_failure_detail(other_outcome, "smoke", call)
    assert unchanged == ('smoke: boot error frame (untrusted) type="HandlerError" '
                         'message="No module named \'unrelated\'" '
                         'trace="ModuleNotFoundError: No module named \'unrelated\'" stderr=""')
    monkeypatch.setattr(execute, "_runtime_closure", lambda: (_ for _ in ()).throw(
        execute.MutationPlatformError("unavailable")))
    assert build.execution_failure_detail(outcome, "smoke", call).startswith(
        'smoke: boot error frame (untrusted) type="HandlerError"')


@pytest.mark.parametrize("import_error,path_readable,denied", [
    (None, False, False),
    (ImportError, True, False),
    (ImportError, False, True),
    (ImportError, "empty", True),
    (RuntimeError, False, False),
])
def test_excluded_probe_requires_import_and_path_denial(monkeypatch, tmp_path,
                                                        import_error, path_readable, denied):
    host_path = tmp_path / "_optional.fake.so"
    operation = qualify._excluded_extension_operation(host_path.name, host_path)
    namespace = {}
    exec(qualify._probe_body(operation, exception="BaseException"), namespace)
    original_open = builtins.open

    def import_module(_name):
        if import_error is not None:
            raise import_error("probe")
        return object()

    def open_file(path, mode="r", *args, **kwargs):
        if path == str(host_path):
            if path_readable:
                return BytesIO(b"" if path_readable == "empty" else b"ok")
            raise FileNotFoundError(path)
        return original_open(path, mode, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(importlib, "import_module", import_module)
        patch.setattr(builtins, "open", open_file)
        assert namespace["handle"]({}, None)["denied"] is denied


@pytest.fixture
def forced_exclusion(monkeypatch):
    if not sys.platform.startswith("linux"):
        pytest.skip("Linux dynamic extension sandbox only")
    dynamic = Path(sysconfig.get_path("stdlib")).resolve(strict=True) / "lib-dynload"
    extension = next((path for name in ("_bz2", "_lzma", "_csv")
                      for path in sorted(dynamic.glob(name + ".*.so"))), None)
    if extension is None:
        pytest.skip("no suitable optional extension")
    original = subprocess.run

    def run(args, **kwargs):
        result = original(args, **kwargs)
        if args[0] != "/usr/bin/ldd":
            return result
        header = str(extension) + ":\n"
        assert header in result.stdout
        output = result.stdout.replace(header, header + "\tlibsf_missing_probe.so => not found\n", 1)
        return subprocess.CompletedProcess(result.args, result.returncode, output, result.stderr)

    execute._runtime_roots_cached.cache_clear()
    monkeypatch.setattr(execute.subprocess, "run", run)
    yield extension
    execute._runtime_roots_cached.cache_clear()


@pytest.mark.host_sandbox
def test_excluded_extension_is_masked_in_real_sandbox(tmp_path, forced_exclusion):
    extension = forced_exclusion
    name = extension.name.split(".", 1)[0]
    site = str(Path(sysconfig.get_path("stdlib")).resolve(strict=True) / "site-packages")
    source = ("import ctypes, importlib, os\n"
              "def handle(input, ctx):\n"
              "    try:\n"
              f"        importlib.import_module({name!r})\n"
              "        import_denied = False\n"
              "    except ImportError:\n"
              "        import_denied = True\n"
              "    try:\n"
              f"        masked_bytes = open({str(extension)!r}, 'rb').read(1)\n"
              "    except OSError:\n"
              "        masked_bytes = b''\n"
              f"    host_masked = not os.path.isfile({str(extension)!r}) and masked_bytes == b''\n"
              f"    packages_hidden = not os.path.isdir({site!r}) or os.listdir({site!r}) == []\n"
              "    try:\n"
              f"        ctypes.CDLL({str(extension)!r})\n"
              "        dlopen_denied = False\n"
              "    except OSError:\n"
              "        dlopen_denied = True\n"
              "    importlib.import_module('_struct')\n"
              "    return {'import_denied': import_denied, 'host_masked': host_masked,\n"
              "            'packages_hidden': packages_hidden,\n"
              "            'dlopen_denied': dlopen_denied, 'admitted_import': True}\n")
    unit = _unit(tmp_path, source)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    with execute.MutationExecutor(runtime, network_envelope="none") as executor:
        outcome = executor.run(_spec(unit))
    assert outcome.exit_code == 0 and outcome.cleanup.ok, outcome.stdout
    for key in ("import_denied", "host_masked", "packages_hidden", "dlopen_denied", "admitted_import"):
        assert f'"{key}":true'.encode() in outcome.stdout


@pytest.mark.host_sandbox
def test_admission_reports_missing_extension_library(tmp_path, forced_exclusion):
    extension = forced_exclusion.name.split(".", 1)[0]
    source = f"def handle(input, ctx):\n    import {extension}\n    return {{}}\n"
    staged = stage_test_unit(build, tmp_path / "stage" / "runtime", source)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    with execute.MutationExecutor(runtime, network_envelope="none") as executor:
        with pytest.raises(build.BuildError) as caught:
            build.admit(staged, executor, runtime_closure_digest="runtime",
                        sdk_digest="sdk", policy_version="mutation/1", grants_digest="grants")
    assert caught.value.code == "EXECUTION"
    assert f"unavailable extension {extension} on this host" in caught.value.detail
    assert "libsf_missing_probe.so" in caught.value.detail
    assert "use a pure-Python alternative" in caught.value.detail


@pytest.mark.host_sandbox
def test_qualification_with_excluded_extension(tmp_path, forced_exclusion):
    report = qualify.selftest(tmp_path)
    assert report["ok"], [check for check in report["checks"] if not check["ok"]]
    assert [forced_exclusion.name, ["libsf_missing_probe.so"]] in report["runtime_excluded"]
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["runtime_exclusions"]["ok"]
    assert checks["excluded_extension"]["ok"]


@pytest.fixture
def mask_command(tmp_path, monkeypatch):
    import kernel_test_isolation as isolation

    monkeypatch.setattr(isolation, "_bwrap_userns_guard_supported", lambda _executable: False)
    stdlib = tmp_path / "stdlib"
    (stdlib / "lib-dynload").mkdir(parents=True)
    (stdlib / "site-packages").mkdir()
    extension = stdlib / "lib-dynload" / "_tkinter.fake.so"
    extension.write_bytes(b"elf")
    stage = tmp_path / "stage"
    mounts = {}
    for name in ("boot", "unit", "ctl", "in", "resp"):
        (stage / name).mkdir(parents=True)
        mounts[name] = str(stage / name)
    cap = SimpleNamespace(read_roots=(stdlib, *(Path(path) for path in mounts.values())),
                          workspace=stage / "unit", workspace_writable=False, write_roots=(),
                          purpose="mutation", network="none", env={})

    def build(hide):
        return isolation._mutation_command(["true"], cap, "bwrap", {**mounts, "hide": hide}, False)[0]
    return build, stdlib, extension, stage


def test_hidden_paths_are_masked_after_the_stdlib_bind(mask_command):
    build, stdlib, extension, _stage = mask_command
    arguments = build([str(extension), str(stdlib / "site-packages")])
    bind = arguments.index(str(stdlib))
    masked_file = arguments.index(str(extension))
    masked_dir = arguments.index(str(stdlib / "site-packages"))
    assert arguments[bind - 1] == "--ro-bind"
    assert arguments[masked_file - 2:masked_file + 1] == ["--ro-bind", "/dev/null", str(extension)]
    assert arguments[masked_dir - 1:masked_dir + 3] == ["--tmpfs", str(stdlib / "site-packages"),
                                                       "--remount-ro", str(stdlib / "site-packages")]
    assert bind < masked_file < arguments.index("/sf/boot")
    assert bind < masked_dir < arguments.index("/sf/boot")


@pytest.mark.parametrize("case", ["outside", "symlink", "relative", "staged", "missing", "not_list"])
def test_hidden_paths_fail_closed(mask_command, tmp_path, case):
    build, stdlib, extension, stage = mask_command
    outside = tmp_path / "outside.so"
    outside.write_bytes(b"elf")
    link = stdlib / "alias.so"
    link.symlink_to(extension)
    hide = {"outside": [str(outside)], "symlink": [str(link)], "relative": ["stdlib/x.so"],
            "staged": [str(stage / "unit")], "missing": [str(stdlib / "gone.so")],
            "not_list": str(extension)}[case]
    with pytest.raises((ValueError, OSError)):
        build(hide)


def test_missing_ldd_is_named(monkeypatch, fake_runtime):
    python, stdlib, *_ = fake_runtime

    def run(*_args, **_kwargs):
        raise FileNotFoundError("ldd")

    monkeypatch.setattr(execute.subprocess, "run", run)
    with pytest.raises(execute.MutationPlatformError, match="ldd is unavailable"):
        execute._runtime_roots_cached(str(python), str(stdlib), 0, 0)
    monkeypatch.setattr(execute.subprocess, "run", lambda *_a, **_k: (_ for _ in ()).throw(
        subprocess.CalledProcessError(1, ["ldd"])))
    with pytest.raises(execute.MutationPlatformError, match=r"ldd failed \(exit 1\)"):
        execute._runtime_roots_cached(str(python), str(stdlib), 1, 1)


def test_unqualified_platform_refuses_mutate_once_and_the_run_continues(tmp_path):
    from kernel_mutate_policy import MutationPolicy
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime, MutationUnavailable

    calls = []

    def factory():
        calls.append(1)
        raise execute.MutationPlatformError(
            "cannot qualify Python runtime libraries: ldd is unavailable (FileNotFoundError)")

    runtime = MutationRuntime(level="prebuilt", journal=Journal(tmp_path),
                              policy=MutationPolicy(level="prebuilt"), executor_factory=factory)
    runtime._dispatch = lambda *_args: runtime._ensure_executor()
    first = runtime.dispatch({"op": "use", "id": "file_facts", "args": "{}"}, "call-1")
    second = runtime.dispatch({"op": "use", "id": "file_facts", "args": "{}"}, "call-2")
    for reply in (first, second):
        assert reply.startswith("REFUSED MUTATION_PLATFORM_UNQUALIFIED ")
        assert "ldd is unavailable" in reply
    assert calls == [1]
    with pytest.raises(MutationUnavailable):
        runtime._ensure_executor()
