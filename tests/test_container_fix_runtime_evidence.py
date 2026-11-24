from types import SimpleNamespace
from unittest.mock import mock_open

import pytest

from stateframe.engine import kernel_execution_boundary as boundary


def test_unselected_boundary_stays_on_platform_backend():
    detection = boundary.Detection("container", ("cgroup:docker",), {"pid": "4026533000"}, 1)
    result = boundary.authorize(False, conflicts=(), detection=detection, uid=1000)
    assert result.backend == boundary.BACKEND_PLATFORM
    assert result.record()["policy"] == "platform"
    assert result.record()["statement"] == ""


def test_runtime_root_mount_counts_as_container_evidence(monkeypatch):
    def fixture_stat(path, **_kwargs):
        if path == "/fixture-proc/self/ns/pid":
            return SimpleNamespace(st_ino=4026533000)
        raise FileNotFoundError(path)

    def fixture_open(path, *args, **kwargs):
        data = {
            "/fixture-proc/self/mountinfo":
                "36 25 0:32 /var/lib/docker/btrfs/subvolumes/abc / rw - btrfs /dev/vda rw\n",
            "/fixture-proc/self/cgroup": "0::/\n",
        }
        return mock_open(read_data=data[path])()

    monkeypatch.setattr(boundary, "os", SimpleNamespace(stat=fixture_stat, path=boundary.os.path))
    monkeypatch.setattr(boundary, "open", fixture_open, raising=False)
    result = boundary.detect("/fixture-proc", "/fixture-fs")
    assert result.classification == "container", result


def test_plain_host_bwrap_root_mount_is_not_container(monkeypatch):
    def fixture_stat(path, **_kwargs):
        if path == "/fixture-proc/self/ns/pid":
            return SimpleNamespace(st_ino=4026533000)
        raise FileNotFoundError(path)

    def fixture_open(path, *args, **kwargs):
        data = {
            "/fixture-proc/self/mountinfo":
                "36 25 0:32 / / rw - overlay overlay rw,lowerdir=/home/user/rootfs,"
                "upperdir=/tmp/bwrap-root/upper,workdir=/tmp/bwrap-root/work\n",
            "/fixture-proc/self/cgroup": "0::/user.slice/user-1000.slice/user@1000.service/bwrap.scope\n",
        }
        return mock_open(read_data=data[path])()

    monkeypatch.setattr(boundary, "os", SimpleNamespace(stat=fixture_stat, path=boundary.os.path))
    monkeypatch.setattr(boundary, "open", fixture_open, raising=False)
    result = boundary.detect("/fixture-proc", "/fixture-fs")
    assert result.classification == "not_container", result


def test_host_service_with_docker_prefix_is_not_runtime_evidence(monkeypatch):
    def fixture_stat(path, **_kwargs):
        if path == "/fixture-proc/self/ns/pid":
            return SimpleNamespace(st_ino=4026533000)
        raise FileNotFoundError(path)

    def fixture_open(path, *args, **kwargs):
        data = {
            "/fixture-proc/self/mountinfo": "36 25 0:32 / / rw - ext4 /dev/vda rw\n",
            "/fixture-proc/self/cgroup": "0::/system.slice/docker-build.service\n",
        }
        return mock_open(read_data=data[path])()

    monkeypatch.setattr(boundary, "os", SimpleNamespace(stat=fixture_stat, path=boundary.os.path))
    monkeypatch.setattr(boundary, "open", fixture_open, raising=False)
    result = boundary.detect("/fixture-proc", "/fixture-fs")
    assert result.classification == "not_container", result


@pytest.mark.parametrize(
    ("path", "signal"),
    [
        ("/docker/" + "a" * 64, "cgroup:docker"),
        ("/system.slice/docker-" + "a" * 64 + ".scope", "cgroup:docker"),
        ("/kubepods/burstable/pod12345678_1234_1234_1234_123456789abc/" + "a" * 64,
         "cgroup:kubepods"),
        ("/kubepods-burstable-pod12345678_1234_1234_1234_123456789abc.slice",
         "cgroup:kubepods"),
        ("/containerd/" + "a" * 64, "cgroup:containerd"),
        ("/system.slice/cri-containerd-" + "a" * 64 + ".scope", "cgroup:containerd"),
        ("/crio/" + "a" * 64, "cgroup:crio"),
        ("/system.slice/crio-" + "a" * 64 + ".scope", "cgroup:crio"),
        ("/podman/" + "a" * 64, "cgroup:podman"),
        ("/system.slice/libpod-" + "a" * 64 + ".scope", "cgroup:podman"),
    ],
)
def test_cgroup_runtime_forms_remain_supported(path, signal):
    assert boundary._cgroup_signals("0::" + path + "\n") == [signal]


@pytest.mark.parametrize(
    "path",
    [
        "/system.slice/docker-build.service",
        "/system.slice/docker-not-a-container.scope",
        "/system.slice/DOCKER-" + "a" * 64 + ".scope",
        "/kubepods-company.slice",
        "/system.slice/kubepods-burstable-build.service",
        "/system.slice/containerd.service",
        "/system.slice/containerd-shim.service",
        "/system.slice/cri-containerd-build.service",
        "/system.slice/crio-build.service",
        "/system.slice/cri-o.service",
        "/system.slice/podman.service",
        "/system.slice/libpod-build.service",
    ],
)
def test_host_service_names_are_not_runtime_evidence(path):
    assert boundary._cgroup_signals("0::" + path + "\n") == []
