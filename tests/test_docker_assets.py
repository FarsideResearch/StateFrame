import gzip
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def project_version():
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]


def write_test_wheel(wheel_dir, version, *, missing_license=None):
    wheel_dir.mkdir(parents=True, exist_ok=True)
    wheel_path = wheel_dir / f"stateframe-{version}-py3-none-any.whl"
    entries = {
        f"stateframe-{version}.dist-info/METADATA": (
            f"Metadata-Version: 2.1\nName: stateframe\nVersion: {version}\n\n"
        ).encode(),
        "stateframe/THIRD_PARTY_LICENSES.md": b"third-party terms",
    }
    entries.update({
        f"stateframe-{version}.dist-info/licenses/LICENSE": b"license text",
        f"stateframe-{version}.dist-info/licenses/NOTICE": b"notice text",
        f"stateframe-{version}.dist-info/licenses/COMMERCIAL-LICENSE.md": b"commercial terms",
    })
    if missing_license is not None:
        entries.pop(f"stateframe-{version}.dist-info/licenses/{missing_license}", None)
    with zipfile.ZipFile(wheel_path, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return wheel_path


def load_wheel_verifier():
    module_path = ROOT / "docker/verify_wheel.py"
    spec = importlib.util.spec_from_file_location("docker_wheel_verifier", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dockerfile_pins_base_checks_wheel_and_runs_nonroot():
    dockerfile = (ROOT / "docker/Dockerfile").read_text(encoding="utf-8")
    wheel_check = (ROOT / "docker/verify_wheel.py").read_text(encoding="utf-8")
    assert re.search(
        r"^ARG PYTHON_IMAGE=python:3\.13-slim-bookworm$",
        dockerfile,
        re.MULTILINE,
    )
    assert dockerfile.count("FROM ${PYTHON_IMAGE}") == 2
    assert "verify_wheel.py /tmp/wheels" in dockerfile
    assert r"python:3\.13-slim-bookworm@sha256:[0-9a-f]{64}" in wheel_check
    assert "WHEEL_SHA256" in dockerfile and "hashlib.sha256" in wheel_check
    assert "RUN python -" in dockerfile and "<<'PY'" in dockerfile
    assert "COPY dist/*.whl" in dockerfile
    assert "COPY docker/verify_wheel.py" in dockerfile
    assert "pip install --no-cache-dir" in dockerfile
    assert "USER 65532:0" in dockerfile
    assert 'ENTRYPOINT ["/usr/bin/tini", "-g", "--", "stateframe"]' in dockerfile
    assert "install -d -o root -g 0 -m 2775 /home/sf /workspace" in dockerfile
    apt_install_lines = [
        line.strip().removesuffix("\\").rstrip()
        for line in dockerfile.splitlines()
        if "apt-get install" in line
    ]
    assert apt_install_lines == [
        "&& apt-get install -y --no-install-recommends ca-certificates git bubblewrap tini",
    ]
    assert "VOLUME " not in dockerfile
    assert "ARG APP_VERSION=" not in dockerfile
    assert "ARG WHEEL_SHA256=" not in dockerfile
    assert "STATEFRAME_API_KEY" not in dockerfile
    assert "/usr/share/licenses/stateframe" in dockerfile
    assert not re.search(r"(?m)^COPY\s+(?:src/|\.\s)", dockerfile)
    assert not re.search(r"(?m)^ENV\s+.*(?:STATEFRAME_ENDPOINT|STATEFRAME_MODEL|API_KEY)", dockerfile)
    assert f"APP_VERSION={project_version()}" not in dockerfile


def test_wheel_verifier_checks_hash_metadata_and_copies_all_licenses(tmp_path, monkeypatch):
    version = project_version()
    wheel_dir = tmp_path / "wheels"
    wheel = write_test_wheel(wheel_dir, version)
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    licenses_dir = tmp_path / "licenses"
    args = [
        "verify_wheel.py", str(wheel_dir),
        "--python-image", f"python:3.13-slim-bookworm@sha256:{'a' * 64}",
        "--app-version", version, "--sha256", digest,
        "--licenses-dir", str(licenses_dir),
    ]
    verifier = load_wheel_verifier()

    monkeypatch.setattr(sys, "argv", args)
    assert verifier.main() == 0
    assert {path.name for path in licenses_dir.iterdir()} == {
        "LICENSE", "NOTICE", "COMMERCIAL-LICENSE.md", "THIRD_PARTY_LICENSES.md",
    }
    assert (licenses_dir / "THIRD_PARTY_LICENSES.md").read_bytes() == b"third-party terms"

    monkeypatch.setattr(sys, "argv", [*args[:-3], "0" * 64, *args[-2:]])
    with pytest.raises(SystemExit, match="wheel SHA-256 does not match"):
        verifier.main()


def test_wheel_verifier_rejects_missing_license(tmp_path, monkeypatch):
    version = project_version()
    wheel_dir = tmp_path / "wheels"
    wheel = write_test_wheel(wheel_dir, version, missing_license="NOTICE")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    monkeypatch.setattr(sys, "argv", [
        "verify_wheel.py", str(wheel_dir),
        "--python-image", f"python:3.13-slim-bookworm@sha256:{'b' * 64}",
        "--app-version", version, "--sha256", digest,
        "--licenses-dir", str(tmp_path / "licenses"),
    ])
    with pytest.raises(SystemExit, match="missing one or more required license files"):
        load_wheel_verifier().main()


def test_image_build_refuses_missing_digest_and_limits_context():
    build = (ROOT / "docker/build.sh").read_text(encoding="utf-8")
    ignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    environment = os.environ.copy()
    environment.pop("PYTHON_IMAGE", None)
    environment["TMPDIR"] = str(ROOT.parent / "tmp")
    result = subprocess.run(
        [str(ROOT / "docker/build.sh"), "HEAD"], cwd=ROOT, env=environment,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "PYTHON_IMAGE is required" in result.stderr
    assert "PYTHON_IMAGE is required" in build
    assert "python:3\\.13-slim-bookworm@sha256:[0-9a-f]{64}" in build
    assert "BUILDER:-docker buildx" in build
    assert "BuildKit is required" in build
    assert '"${builder_cmd[@]}" build' in build
    assert "--load" in build
    assert "BUILD_CPUS" in build and "--cpu-quota" in build
    assert "BUILD_MEMORY" in build and "--memory" in build
    assert "API_KEY" not in build
    assert "archive --format=tar" in build
    assert "pip wheel --no-deps --no-build-isolation" in build
    assert '"sf-eks-stateframe:$version"' in build
    assert '"project"]["version"' in build and "pyproject.toml" in build
    assert f"sf-eks-stateframe:{project_version()}" not in build
    assert "!dist/*.whl" in ignore
    assert "!requirements.txt" in ignore
    assert "!docker/Dockerfile" in ignore
    assert "!docker/verify_wheel.py" in ignore


def test_kubernetes_job_is_restricted_and_has_both_storage_variants():
    job_path = ROOT / "docker/k8s/base/job.yaml"
    network_path = ROOT / "docker/k8s/base/networkpolicy.yaml"
    job = yaml.safe_load(job_path.read_text(encoding="utf-8"))
    network = yaml.safe_load(network_path.read_text(encoding="utf-8"))
    pod = job["spec"]["template"]["spec"]
    container = pod["containers"][0]

    assert job["spec"]["backoffLimit"] == 0
    assert pod["restartPolicy"] == "Never"
    assert pod["terminationGracePeriodSeconds"] == 30
    assert pod["automountServiceAccountToken"] is False
    assert pod["securityContext"]["runAsNonRoot"] is True
    assert pod["securityContext"]["runAsUser"] == 10001
    assert pod["securityContext"]["runAsGroup"] == 0
    assert pod["securityContext"]["seccompProfile"]["type"] == "RuntimeDefault"
    assert container["image"] == "sf-eks-stateframe:replace-with-build-tag"
    assert container["imagePullPolicy"] == "IfNotPresent"
    assert container["command"] == ["/" + "usr/bin/tini", "-g", "--", "python", "-c"]
    assert container["securityContext"]["allowPrivilegeEscalation"] is False
    assert container["securityContext"]["readOnlyRootFilesystem"] is True
    assert container["securityContext"]["capabilities"]["drop"] == ["ALL"]
    assert "requests" in container["resources"] and "limits" in container["resources"]
    api_key_env = next(item for item in container["env"] if item["name"] == "API_KEY")
    assert api_key_env["valueFrom"]["secretKeyRef"]["name"] == "sf-eks-credentials"
    assert "value" not in api_key_env
    assert api_key_env["valueFrom"]["secretKeyRef"]["optional"] is True
    remote_endpoint_env = next(item for item in container["env"]
                               if item["name"] == "STATEFRAME_ALLOW_REMOTE_ENDPOINT")
    assert remote_endpoint_env["value"] == "1"
    extra_args_env = next(item for item in container["env"]
                          if item["name"] == "STATEFRAME_EXTRA_ARGS")
    assert extra_args_env["value"] == (
        "--container-is-sandbox --shell unrestricted --shell-network full"
    )
    assert any("emptyDir" in volume for volume in pod["volumes"])

    assert network["spec"]["policyTypes"] == ["Egress"]
    rules = network["spec"]["egress"]
    assert len(rules) == 2
    dns = rules[0]
    assert dns["to"] == [{
        "namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": "kube-system"}},
        "podSelector": {"matchLabels": {"k8s-app": "kube-dns"}},
    }]
    assert {(port["protocol"], port["port"]) for port in dns["ports"]} == {
        ("UDP", 53), ("TCP", 53),
    }
    endpoint = rules[1]
    assert endpoint["to"] == [{"ipBlock": {"cidr": "198.51.100.1/32"}}]
    assert endpoint["ports"] == [{"protocol": "TCP", "port": 443}]
    assert "ingress" not in network["spec"]

    for base in (ROOT / "docker/k8s/base", ROOT / "docker/k8s/pvc"):
        kustomization = yaml.safe_load((base / "kustomization.yaml").read_text(encoding="utf-8"))
        for resource in kustomization["resources"]:
            resource_path = base / resource
            assert resource_path.is_file() or (
                resource_path.is_dir() and (resource_path / "kustomization.yaml").is_file()
            )
        for patch in kustomization.get("patches", []):
            assert (base / patch["path"]).is_file()
    for path in (ROOT / "docker/k8s").rglob("*.yaml"):
        list(yaml.safe_load_all(path.read_text(encoding="utf-8")))

    smoke = (ROOT / "docker/smoke/smoke.sh").read_text(encoding="utf-8")
    assert 'image=${STATEFRAME_IMAGE:-}' in smoke
    assert "STATEFRAME_IMAGE must be set to the exact versioned tag printed by docker/build.sh" in smoke
    assert "pyproject.toml" not in smoke
    for setting in ("--read-only", "--tmpfs /tmp", "--cap-drop ALL",
                    "--security-opt no-new-privileges", "--user 54321:0"):
        assert setting in smoke
    assert "--shell unrestricted --shell-sandbox on" in smoke
    assert "--shell-sandbox off" not in smoke
    assert "--sandbox=pod" not in smoke
    assert "REFUSED SANDBOX_UNAVAILABLE" in smoke and "FAIL:" in smoke
    assert "container_owned" in smoke and "volume_owned" in smoke and "network_owned" in smoke
    assert 'owner="sf-eks-smoke-$$-$RANDOM"' in smoke
    assert "sf-eks-smoke.owner=$owner" in smoke
    assert 'made_cli )) && container_owned "$cli_name"' in smoke
    assert 'made_server )) && container_owned "$server_name"' in smoke
    assert 'made_work )) && volume_owned "$work_volume"' in smoke
    assert 'made_home )) && volume_owned "$home_volume"' in smoke
    assert 'made_network )) && network_owned "$network"' in smoke
    assert "--init" not in smoke
    assert "STATEFRAME_ENDPOINT=http://$server_name:8765/v1" in smoke
    assert "-e STATEFRAME_ALLOW_REMOTE_ENDPOINT=1" in smoke
    assert "SMOKE_EXTRA_ARGS" in smoke
    assert 'run_args+=("${smoke_extra_args[@]}")' in smoke
    readme = (ROOT / "docker/README.md").read_text(encoding="utf-8")
    assert ("`STATEFRAME_ALLOW_REMOTE_ENDPOINT=1` allows the runner to send repository content "
            "to any non-loopback HTTP or HTTPS endpoint and to follow non-loopback redirects.") in readme
    for doc_path in (ROOT / "docker/README.md", job_path):
        content = doc_path.read_text(encoding="utf-8")
        assert f"sf-eks-stateframe:{project_version()}" not in content


def test_kubernetes_kustomize_renders_base_and_pvc_profiles():
    if shutil.which("kubectl") is None:
        pytest.skip("kubectl is not installed")

    environment = os.environ.copy()
    environment["KUBECONFIG"] = "/dev/null"
    expected_extra_args = "--container-is-sandbox --shell unrestricted --shell-network full"
    for variant in ("base", "pvc"):
        directory = f"docker/k8s/{variant}"
        result = subprocess.run(
            ["kubectl", "kustomize", directory], cwd=ROOT, env=environment,
            capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0, result.stderr
        resources = [item for item in yaml.safe_load_all(result.stdout) if item]
        job = next(item for item in resources if item["kind"] == "Job")
        pod = job["spec"]["template"]["spec"]
        container = pod["containers"][0]

        assert pod["automountServiceAccountToken"] is False
        assert pod["securityContext"]["runAsNonRoot"] is True
        assert pod["securityContext"]["runAsUser"] == 10001
        assert pod["securityContext"]["seccompProfile"]["type"] == "RuntimeDefault"
        assert container["securityContext"]["runAsNonRoot"] is True
        assert container["securityContext"]["runAsUser"] == 10001
        assert container["securityContext"]["capabilities"]["drop"] == ["ALL"]
        assert container["securityContext"]["readOnlyRootFilesystem"] is True
        assert container["securityContext"]["seccompProfile"]["type"] == "RuntimeDefault"
        extra_args = next(item["value"] for item in container["env"]
                          if item["name"] == "STATEFRAME_EXTRA_ARGS")
        assert extra_args == expected_extra_args

        volumes = {item["name"]: item for item in pod["volumes"]}
        mounts = {item["name"]: item["mountPath"] for item in container["volumeMounts"]}
        assert {"home", "workspace"}.issubset(volumes)
        for name, claim in (("home", "sf-eks-home"), ("workspace", "sf-eks-work")):
            if variant == "pvc":
                assert volumes[name]["persistentVolumeClaim"]["claimName"] == claim
                assert "emptyDir" not in volumes[name]
            else:
                assert "emptyDir" in volumes[name]
        assert mounts["home"] == "/home/sf"
        assert mounts["workspace"] == "/workspace"
        home_env = {item["name"]: item.get("value") for item in container["env"]}
        assert home_env["HOME"] == "/home/sf"
        assert home_env["STATEFRAME_HOME"] == "/home/sf/.stateframe"

        network = next(item for item in resources if item["kind"] == "NetworkPolicy")
        dns_rule = next(rule for rule in network["spec"]["egress"]
                        if any(port["port"] == 53 for port in rule["ports"]))
        assert dns_rule["to"] == [{
            "namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": "kube-system"}},
            "podSelector": {"matchLabels": {"k8s-app": "kube-dns"}},
        }]
        assert {(port["protocol"], port["port"]) for port in dns_rule["ports"]} == {
            ("UDP", 53), ("TCP", 53),
        }


def test_kubernetes_job_builds_run_arguments_without_missing_optional_key(monkeypatch):
    job = yaml.safe_load((ROOT / "docker/k8s/base/job.yaml").read_text(encoding="utf-8"))
    container = job["spec"]["template"]["spec"]["containers"][0]
    script = container["args"][0]
    default_extra_args = next(item["value"] for item in container["env"]
                              if item["name"] == "STATEFRAME_EXTRA_ARGS")
    calls = []
    monkeypatch.setattr(os, "execvp", lambda executable, argv: calls.append((executable, argv)))
    monkeypatch.setenv("STATEFRAME_TASK", "run the configured task")
    monkeypatch.setenv("STATEFRAME_EXTRA_ARGS", "--max-turns 9")
    monkeypatch.delenv("API_KEY", raising=False)

    exec(compile(script, "docker/k8s/base/job.yaml", "exec"), {})

    executable, argv = calls[-1]
    assert executable == "stateframe"
    assert argv == [
        "stateframe", "run", "--repo", "/workspace/repo",
        "--task", "run the configured task", "--in-place", "--max-turns", "9",
    ]
    assert "STATEFRAME_TASK" not in os.environ
    assert "STATEFRAME_EXTRA_ARGS" not in os.environ

    monkeypatch.setenv("API_KEY", "test-value")
    monkeypatch.setenv("STATEFRAME_TASK", "run the configured task")
    monkeypatch.setenv("STATEFRAME_EXTRA_ARGS", "--max-turns 9")
    exec(compile(script, "docker/k8s/base/job.yaml", "exec"), {})
    _, argv = calls[-1]
    assert argv[argv.index("--api-key-env") + 1] == "API_KEY"
    assert "test-value" not in argv

    monkeypatch.setenv("STATEFRAME_TASK", "run the configured task")
    monkeypatch.setenv("STATEFRAME_EXTRA_ARGS", default_extra_args)
    exec(compile(script, "docker/k8s/base/job.yaml", "exec"), {})
    _, argv = calls[-1]
    assert argv == [
        "stateframe", "run", "--repo", "/workspace/repo",
        "--task", "run the configured task", "--in-place",
        "--api-key-env", "API_KEY", "--container-is-sandbox",
        "--shell", "unrestricted", "--shell-network", "full",
    ]


def test_smoke_extra_args_are_optional_run_arguments():
    smoke = (ROOT / "docker/smoke/smoke.sh").read_text(encoding="utf-8")
    append_line = 'run_args+=("${smoke_extra_args[@]}")'
    parse_start = smoke.index("smoke_extra_args=()")
    parse_end = smoke.index(append_line, parse_start) + len(append_line)
    parse_block = smoke[parse_start:parse_end]
    command = (
        "run_args=(run --repo /workspace/repo)\n"
        + parse_block
        + '\nprintf \'%s\\0\' "${run_args[@]}"\n'
    )

    for value, expected in (
        ("", ["run", "--repo", "/workspace/repo"]),
        ("--max-turns 9", ["run", "--repo", "/workspace/repo", "--max-turns", "9"]),
        ("--max-turns\n9", ["run", "--repo", "/workspace/repo", "--max-turns", "9"]),
    ):
        environment = os.environ.copy()
        environment["SMOKE_EXTRA_ARGS"] = value
        result = subprocess.run(
            ["bash", "-c", command], env=environment,
            capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.split("\0")[:-1] == expected


def test_smoke_fake_endpoint_serves_probes_gzip_and_both_chat_modes(tmp_path):
    module_path = ROOT / "docker/smoke/fake_server.py"
    spec = importlib.util.spec_from_file_location("smoke_fake_server", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = module.State(tmp_path, None)
    handler_type = module.handler_for(state)
    offered = [
        {"type": "function", "function": {"name": name}}
        for name in ("read", "str_replace", "run_tests", "done")
    ]

    def call(method, path, body=None, *, compressed=False):
        raw = b""
        headers = {}
        if body is not None:
            raw = json.dumps(body).encode()
        if compressed:
            raw = gzip.compress(raw)
            headers["Content-Encoding"] = "gzip"
        if body is not None:
            headers["Content-Length"] = str(len(raw))
        handler = handler_type.__new__(handler_type)
        handler.path = path
        handler.headers = headers
        handler.rfile = io.BytesIO(raw)
        handler.wfile = io.BytesIO()
        handler.response_headers = {}
        handler.status_code = None
        handler.send_response = lambda status, *_args: setattr(handler, "status_code", status)
        handler.send_header = lambda name, value: handler.response_headers.__setitem__(name, value)
        handler.end_headers = lambda: None
        if method == "GET":
            handler.do_GET()
        else:
            handler.do_POST()
        return handler.status_code, handler.response_headers, handler.wfile.getvalue()

    status, _, payload = call("GET", "/v1/models")
    assert status == 200 and json.loads(payload)["data"][0]["id"] == module.MODEL
    status, _, payload = call("GET", "/props")
    assert status == 200 and json.loads(payload)["default_generation_settings"]["n_ctx"] == 65536
    status, _, payload = call("GET", "/slots")
    assert status == 200 and json.loads(payload)[0]["n_ctx"] == 65536
    status, _, payload = call("GET", "/health")
    assert status == 200 and json.loads(payload) == {"ok": True}
    status, _, payload = call("POST", "/tokenize", {"content": "count this"})
    assert status == 200 and json.loads(payload)["tokens"] == [0, 1]
    status, _, payload = call("POST", "/tokenize", {"content": "count this"}, compressed=True)
    assert status == 200 and json.loads(payload)["tokens"] == [0, 1]
    assert state.requests == 0

    status, _, payload = call(
        "POST", "/v1/chat/completions", {"tools": offered, "stream": False},
    )
    result = json.loads(payload)
    first_call = result["choices"][0]["message"]["tool_calls"][0]
    assert status == 200
    assert first_call["id"] == "call_1"
    assert first_call["function"]["name"] == "read"
    assert json.loads(first_call["function"]["arguments"]) == {"path": "calc.py"}
    assert result["choices"][0]["finish_reason"] == "tool_calls"
    assert state.position == 1

    status, headers, payload = call(
        "POST", "/v1/chat/completions", {"tools": offered, "stream": True},
    )
    streamed = payload.decode()
    assert status == 200
    assert headers["Content-Type"].startswith("text/event-stream")
    assert headers["Connection"] == "close"
    assert "data: [DONE]" in streamed
    chunks = [
        json.loads(line.removeprefix("data: "))
        for line in streamed.splitlines()
        if line.startswith("data: {")
    ]
    assert len(chunks) == 3
    tool_delta = chunks[0]["choices"][0]["delta"]["tool_calls"][0]
    assert tool_delta["id"] == "call_2"
    assert tool_delta["function"]["name"] == "str_replace"
    arguments = "".join(
        chunk["choices"][0]["delta"].get("tool_calls", [{}])[0]
        .get("function", {}).get("arguments", "")
        for chunk in chunks
    )
    assert json.loads(arguments) == {
        "path": "calc.py", "old_str": "return 41", "new_str": "return 42",
    }
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert state.position == 2
