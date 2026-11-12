import json
import sys
import os
import hashlib
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from kernel_mutate_build import BuildError, admission_digest, admit, smoke, source_digest, stage_unit, validate
from kernel_mutate_wire import request_key, strict_loads


def manifest():
    return {"schema": "stateframe-mutation-unit/1", "name": "x", "revision": 1, "parent": None, "entry": "main.py", "summary": "x", "kind": "reader", "input_schema": {"type": "object"}, "output_schema": {"type": "object"}, "requires": {"network": False, "writes": False}, "smoke": [{"input": {}, "expect": {}}], "limits": {"wall_s": 10, "cpu_s": 5, "mem_mb": 256}}


def test_stage_digest_and_read_only(tmp_path):
    files = {"main.py": b"def handle(input, ctx): return {}\n"}
    unit = stage_unit(files, manifest(), tmp_path)
    assert unit.source_digest == source_digest(files, manifest())
    assert not ((unit.dir / "main.py").stat().st_mode & 0o222)
    assert unit.dir.name == unit.source_digest.removeprefix("sha256:")
    assert stage_unit(files, manifest(), tmp_path).dir == unit.dir


def test_source_and_admission_digest_canonical_formula():
    files = {"main.py": b"pass\n"}
    value = manifest()
    def canonical(item):
        return json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()

    rows = [["main.py", 5, hashlib.sha256(b"pass\n").hexdigest()]]
    source = "sha256:" + hashlib.sha256(canonical(value) + canonical(rows)).hexdigest()
    assert source_digest(files, value) == source
    parts = [source, "runtime", "sdk", "v1", "grants", "smoke"]
    assert admission_digest(*parts) == "sha256:" + hashlib.sha256(canonical(parts)).hexdigest()


@pytest.mark.parametrize("name,data", [("../x.py", b"x"), ("Main.py", b"x"), ("main.py", b"x" * (65 << 10))])
def test_stage_rejects_bad_bundle(tmp_path, name, data):
    with pytest.raises(BuildError):
        stage_unit({name: data}, manifest(), tmp_path)


def test_authored_smoke_required(tmp_path):
    value = manifest()
    value["smoke"] = [{"input": {}, "expect": None}]
    with pytest.raises(BuildError):
        stage_unit({"main.py": b"pass"}, value, tmp_path)
    value["smoke"] = [{"input": {}, "expect": "scalar"}]
    with pytest.raises(BuildError):
        stage_unit({"main.py": b"pass"}, value, tmp_path)


def test_manifest_name_is_catalog_slug(tmp_path):
    value = manifest()
    value["name"] = "../other"
    with pytest.raises(BuildError):
        stage_unit({"main.py": b"pass"}, value, tmp_path)


def test_admit_uses_executor_run_for_validator_and_smoke(tmp_path, monkeypatch):
    fake_module = types.ModuleType("kernel_mutate_exec")
    @dataclass
    class MutationLimits:
        wall_s: float = 10
        cpu_s: float = 5
        mem_mb: int = 256
    @dataclass
    class LaunchSpec:
        role: str
        call: str
        launch: str
        unit_dir: object
        inputs: dict
        responses: dict
        stdin_doc: bytes
        limits: MutationLimits
    fake_module.MutationLimits = MutationLimits
    fake_module.LaunchSpec = LaunchSpec
    monkeypatch.setitem(sys.modules, "kernel_mutate_exec", fake_module)
    class Executor:
        roles = []
        def run(self, spec):
            self.roles.append(spec.role)
            result = {"ok": True, "errors": [], "hints": []} if spec.role == "validator" else {}
            body = json.dumps({"wire": 1, "call": spec.call, "kind": "result", "result": result}).encode()
            return types.SimpleNamespace(stdout=b"SFMW1 " + str(len(body)).encode() + b"\n" + body,
                stdout_overflow=False, timed_out=False, exit_code=0, cleanup=types.SimpleNamespace(ok=True))
    executor = Executor()
    unit = stage_unit({"main.py": b"def handle(input, ctx): return {}\n"}, manifest(), tmp_path)
    admitted = admit(unit, executor, runtime_closure_digest="a", sdk_digest="b", policy_version="1", grants_digest="c")
    assert admitted.admission_digest.startswith("sha256:")
    assert executor.roles == ["validator", "smoke"]


def test_admit_retries_an_explicit_timeout_with_a_fresh_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    fake_module = types.ModuleType("kernel_mutate_exec")

    @dataclass
    class MutationLimits:
        wall_s: float = 10
        cpu_s: float = 5
        mem_mb: int = 256

    @dataclass
    class LaunchSpec:
        role: str
        call: str
        launch: str
        unit_dir: object
        inputs: dict
        responses: dict
        stdin_doc: bytes
        limits: MutationLimits

    fake_module.MutationLimits = MutationLimits
    fake_module.LaunchSpec = LaunchSpec
    monkeypatch.setitem(sys.modules, "kernel_mutate_exec", fake_module)

    class Executor:
        def __init__(self):
            self.specs = []

        def run(self, spec):
            self.specs.append(spec)
            if len(self.specs) == 1:
                return types.SimpleNamespace(stdout=b"", stdout_overflow=False, timed_out=True,
                                             exit_code=None, cleanup=types.SimpleNamespace(ok=True))
            result = {"ok": True, "errors": [], "hints": []} if spec.role == "validator" else {}
            body = json.dumps({"wire": 1, "call": spec.call, "kind": "result",
                               "result": result}).encode()
            return types.SimpleNamespace(stdout=b"SFMW1 " + str(len(body)).encode() + b"\n" + body,
                                         stdout_overflow=False, timed_out=False, exit_code=0,
                                         cleanup=types.SimpleNamespace(ok=True))

    executor = Executor()
    unit = stage_unit({"main.py": b"def handle(input, ctx): return {}\n"}, manifest(), tmp_path)
    admitted = admit(unit, executor, runtime_closure_digest="a", sdk_digest="b", policy_version="1",
                     grants_digest="c", timeout_retries=1, wall_timeout_s=60)

    assert admitted.admission_digest.startswith("sha256:")
    assert [spec.role for spec in executor.specs] == ["validator", "validator", "smoke"]
    assert executor.specs[0].launch != executor.specs[1].launch
    assert all(spec.limits.wall_s == 60 for spec in executor.specs)


def test_admit_does_not_retry_a_non_timeout_execution_failure(tmp_path, monkeypatch):
    fake_module = types.ModuleType("kernel_mutate_exec")
    fake_module.MutationLimits = lambda **fields: fields
    fake_module.LaunchSpec = lambda **fields: types.SimpleNamespace(**fields)
    monkeypatch.setitem(sys.modules, "kernel_mutate_exec", fake_module)

    class Executor:
        def __init__(self):
            self.calls = 0

        def run(self, _spec):
            self.calls += 1
            return types.SimpleNamespace(stdout=b"", stdout_overflow=False, timed_out=False,
                                         exit_code=137, cleanup=types.SimpleNamespace(ok=True))

    executor = Executor()
    unit = stage_unit({"main.py": b"def handle(input, ctx): return {}\n"}, manifest(), tmp_path)
    with pytest.raises(BuildError) as error:
        admit(unit, executor, runtime_closure_digest="a", sdk_digest="b", policy_version="1",
              grants_digest="c", timeout_retries=1, wall_timeout_s=60)

    assert error.value.code == "EXECUTION"
    assert executor.calls == 1


def test_stage_rejects_stdlib_shadow(tmp_path):
    value = manifest()
    value["entry"] = "json.py"
    with pytest.raises(BuildError):
        stage_unit({"json.py": b"pass"}, value, tmp_path)


def test_stage_rejects_inapplicable_schema_keyword(tmp_path):
    value = manifest()
    value["output_schema"] = {"type": "object", "maxLength": 10}
    with pytest.raises(BuildError):
        stage_unit({"main.py": b"pass"}, value, tmp_path)


def test_smoke_passes_fixture_snapshots_to_executor(tmp_path, monkeypatch):
    fake_module = types.ModuleType("kernel_mutate_exec")
    fake_module.MutationLimits = lambda **fields: fields
    fake_module.LaunchSpec = lambda **fields: types.SimpleNamespace(**fields)
    monkeypatch.setitem(sys.modules, "kernel_mutate_exec", fake_module)
    sample = tmp_path / "sample.txt"
    sample.write_bytes(b"sample")
    unit = stage_unit({"main.py": b"def handle(input, ctx): return {}\n"}, manifest(), tmp_path)
    class Executor:
        def run(self, spec):
            assert spec.inputs == {"in0": sample}
            assert strict_loads(spec.stdin_doc)["inputs"] == [{"id": "in0", "path": "/in/in0", "bytes": 6, "name": "sample.txt"}]
            body = json.dumps({"wire": 1, "call": spec.call, "kind": "result", "result": {}}).encode()
            return types.SimpleNamespace(stdout=b"SFMW1 " + str(len(body)).encode() + b"\n" + body,
                stdout_overflow=False, timed_out=False, exit_code=0, cleanup=types.SimpleNamespace(ok=True))
    assert smoke(unit, Executor(), fixtures=[{"input": {}, "expect": {}, "inputs": {"in0": sample}}]).ok


def test_smoke_expected_mapping_is_recursive_subset(tmp_path, monkeypatch):
    fake_module = types.ModuleType("kernel_mutate_exec")
    fake_module.MutationLimits = lambda **fields: fields
    fake_module.LaunchSpec = lambda **fields: types.SimpleNamespace(**fields)
    monkeypatch.setitem(sys.modules, "kernel_mutate_exec", fake_module)
    value = manifest()
    value["output_schema"] = {"type": "object", "properties": {"outer": {"type": "object"}}}
    value["smoke"] = [{"input": {}, "expect": {"outer": {"x": 1}}}]
    unit = stage_unit({"main.py": b"def handle(input, ctx): return {}\n"}, value, tmp_path)
    class Executor:
        def run(self, spec):
            body = json.dumps({"wire": 1, "call": spec.call, "kind": "result", "result": {"outer": {"x": 1, "y": 2}}}).encode()
            return types.SimpleNamespace(stdout=b"SFMW1 " + str(len(body)).encode() + b"\n" + body,
                stdout_overflow=False, timed_out=False, exit_code=0, cleanup=types.SimpleNamespace(ok=True))
    assert smoke(unit, Executor()).ok


def test_restaging_refuses_special_member(tmp_path):
    files = {"main.py": b""}
    unit = stage_unit(files, manifest(), tmp_path)
    unit.dir.chmod(0o700)
    (unit.dir / "main.py").unlink()
    os.mkfifo(unit.dir / "main.py")
    with pytest.raises(BuildError):
        stage_unit(files, manifest(), tmp_path)


def test_smoke_refuses_symlink_response_fixture(tmp_path, monkeypatch):
    fake_module = types.ModuleType("kernel_mutate_exec")
    fake_module.MutationLimits = lambda **fields: fields
    fake_module.LaunchSpec = lambda **fields: types.SimpleNamespace(**fields)
    monkeypatch.setitem(sys.modules, "kernel_mutate_exec", fake_module)
    unit = stage_unit({"main.py": b"pass"}, manifest(), tmp_path)
    body = tmp_path / "body"
    body.write_bytes(b"response")
    linked = tmp_path / "linked"
    linked.symlink_to(body)
    digest = "a" * 64
    response = {"key": request_key("https://a.test", ""), "method": "GET", "url": "https://a.test", "accept": "", "status": 200, "headers": {}, "final_url": "https://a.test", "body": "/resp/" + digest, "bytes": 8, "error": None}
    class Executor:
        def run(self, spec):
            raise AssertionError("unsafe fixture reached executor")
    with pytest.raises(BuildError):
        smoke(unit, Executor(), fixtures=[{"input": {}, "expect": {}, "responses": [response], "response_files": {digest: linked}}])


def test_prebuilt_fixture_symlink_is_rejected_before_read(tmp_path, monkeypatch):
    unit_dir = tmp_path / "prebuilt"
    fixtures = unit_dir / "fixtures"
    fixtures.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.write_bytes(b"secret")
    (fixtures / "page.html").symlink_to(outside)
    document = manifest()
    document["smoke"] = [{"input": {}, "inputs": [],
                          "responses": [{"url": "https://a.test", "body": "fixtures/page.html"}],
                          "expect": {}}]
    unit = stage_unit({"main.py": b"pass"}, document, tmp_path, fixture_root=unit_dir)
    original = Path.read_bytes
    def guarded(path):
        if path == fixtures / "page.html":
            raise AssertionError("symlink fixture was read")
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", guarded)
    with pytest.raises(BuildError):
        smoke(unit, object())


def test_failed_cleanup_is_terminal_containment_error(tmp_path, monkeypatch):
    fake_module = types.ModuleType("kernel_mutate_exec")
    fake_module.MutationLimits = lambda **fields: fields
    fake_module.LaunchSpec = lambda **fields: types.SimpleNamespace(**fields)
    class MutationContainmentError(BaseException):
        code = "MUTATION_CONTAINMENT"
    fake_module.MutationContainmentError = MutationContainmentError
    monkeypatch.setitem(sys.modules, "kernel_mutate_exec", fake_module)
    unit = stage_unit({"main.py": b"pass"}, manifest(), tmp_path)
    class Executor:
        def run(self, spec):
            return types.SimpleNamespace(stdout=b"", stdout_overflow=False,
                timed_out=False, exit_code=0, cleanup=types.SimpleNamespace(ok=False))
    with pytest.raises(MutationContainmentError):
        validate(unit, Executor())


@pytest.mark.parametrize("change,needle", [
    (lambda value: value.update(smoke=[{"expect": {}}]),
     "smoke[0]: needs input (object) and expect (object)"),
    (lambda value: value.update(input_schema={"type": "string", "pattern": "x"}),
     "input_schema: unsupported keyword pattern"),
])
def test_manifest_refusal_names_rule_and_location(tmp_path, change, needle):
    value = manifest()
    change(value)
    with pytest.raises(BuildError) as caught:
        stage_unit({"main.py": b"pass\n"}, value, tmp_path)
    assert needle in caught.value.detail
    assert len(caught.value.detail) <= 1200


def test_file_refusal_names_filename_rule_and_detail_is_safe(tmp_path):
    with pytest.raises(BuildError) as caught:
        stage_unit({"Bad\n/path.py": b"pass\n"}, manifest(), tmp_path)
    assert "files: name must match [a-z0-9_]{1,40}.py" in caught.value.detail
    assert "\\n" in caught.value.detail
    assert "\n" not in caught.value.detail
    message = BuildError("EXECUTION", "/home/user/private/secret.py: boom\n" + "x" * 3000)
    assert "/home/user" not in message.detail
    assert "\\n" in message.detail
    assert len(message.detail) <= 1200
    hostile = BuildError("EXECUTION", "bad\ud800\u2028line")
    assert "\\ud800" in hostile.detail and "\\u2028" in hostile.detail
    spaced = BuildError("EXECUTION", 'File "/home/user/private folder/secret.py", line 2')
    assert "private folder" not in spaced.detail
    assert "secret.py" not in spaced.detail


def _diagnostic_executor(result=None, *, error=None, stderr="", exit_code=0,
                         timed_out=False, stdout_overflow=False):
    class Executor:
        def run(self, spec):
            body = {"wire": 1, "call": spec.call,
                    "kind": "error" if error is not None else "result"}
            if error is not None:
                body.update(error=error, stderr=stderr)
            else:
                body["result"] = result if result is not None else {}
            encoded = json.dumps(body).encode()
            return types.SimpleNamespace(
                stdout=b"SFMW1 " + str(len(encoded)).encode() + b"\n" + encoded,
                stdout_overflow=stdout_overflow, timed_out=timed_out,
                exit_code=exit_code, cleanup=types.SimpleNamespace(ok=True))
    return Executor()


def test_validator_failure_lists_at_most_five_source_locations(tmp_path):
    unit = stage_unit({"main.py": b"pass\n"}, manifest(), tmp_path)
    errors = [{"file": "main.py", "line": index, "col": 2,
               "msg": "invalid syntax"} for index in range(1, 8)]
    executor = _diagnostic_executor({"ok": False, "errors": errors, "hints": []})
    with pytest.raises(BuildError) as caught:
        admit(unit, executor, runtime_closure_digest="rt", sdk_digest="sdk",
              policy_version="p", grants_digest="g")
    assert caught.value.code == "STATIC"
    assert "main.py:1:2: invalid syntax" in caught.value.detail
    assert "main.py:5:2" in caught.value.detail
    assert "main.py:6:2" not in caught.value.detail


def test_execution_error_frame_shows_type_message_trace_and_stderr_tail(tmp_path):
    unit = stage_unit({"main.py": b"pass\n"}, manifest(), tmp_path)
    trace = "".join(f'  File "/sf/unit/main.py", line {index}, in handle\n' for index in range(20))
    error = {"type": "HandlerError", "message": "ValueError: bad record",
             "trace": trace}
    executor = _diagnostic_executor(error=error, stderr="z" * 700 + "stderr-tail", exit_code=3)
    with pytest.raises(BuildError) as caught:
        smoke(unit, executor)
    assert caught.value.code == "EXECUTION"
    assert "HandlerError" in caught.value.detail
    assert "ValueError: bad record" in caught.value.detail
    assert "/sf/unit/main.py" in caught.value.detail
    assert "stderr-tail" in caught.value.detail
    assert "line 1," not in caught.value.detail
    assert len(caught.value.detail) <= 1200


@pytest.mark.parametrize("options,needle", [
    ({"timed_out": True}, "timed out"),
    ({"stdout_overflow": True}, "output overflow"),
    ({"exit_code": 9}, "crashed"),
])
def test_execution_failure_mode_is_plain(tmp_path, options, needle):
    unit = stage_unit({"main.py": b"pass\n"}, manifest(), tmp_path)
    with pytest.raises(BuildError) as caught:
        smoke(unit, _diagnostic_executor({}, **options))
    assert needle in caught.value.detail


def test_smoke_mismatch_shows_sorted_expected_actual_and_first_path(tmp_path):
    value = manifest()
    value["smoke"] = [{"input": {}, "expect": {"outer": {"count": 2}}}]
    unit = stage_unit({"main.py": b"pass\n"}, value, tmp_path)
    with pytest.raises(BuildError) as caught:
        smoke(unit, _diagnostic_executor({"outer": {"count": 1, "extra": True}}))
    assert caught.value.code == "SMOKE_MISMATCH"
    assert "outer.count" in caught.value.detail
    assert 'expected: {"outer":{"count":2}}' in caught.value.detail
    assert 'actual: {"outer":{"count":1,"extra":true}}' in caught.value.detail
