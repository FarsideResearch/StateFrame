import os
from types import SimpleNamespace

import pytest

from stateframe.engine import kernel_execution_boundary as boundary
_REAL_STAT = os.stat

def _tree(tmp_path):
    proc = tmp_path / "proc"
    ns = proc / "self" / "ns"
    ns.mkdir(parents=True)
    for name in ("pid", "mnt", "net", "user"):
        (ns / name).write_text(name)
    (proc / "self").mkdir(exist_ok=True)
    (proc / "self" / "mountinfo").write_text("")
    (proc / "self" / "cgroup").write_text("0::/\n")
    fs = tmp_path / "root"
    (fs / "run").mkdir(parents=True)
    return proc, fs

def _detect(proc, fs): return boundary.detect(str(proc), str(fs))

def _noninitial_pid(proc):
    (proc / "self" / "ns" / "pid").write_text("private namespace fixture")

def _mounts(proc, *mounts):
    escape = lambda value: value.replace(" ", r"\040")
    lines = [f"36 25 0:32 {escape(root)} {escape(point)} rw - {fs} {source} {options}\n"
             for root, point, fs, source, options in mounts]
    (proc / "self" / "mountinfo").write_text("".join(lines))

def _refusal_code(action):
    with pytest.raises(boundary.BoundaryRefused) as error:
        action()
    return error.value.code

def _pretend_root_owned_regular(monkeypatch, marker):
    marker_path = str(marker)
    def root_owned_stat(path, *args, **kwargs):
        info = _REAL_STAT(path, *args, **kwargs)
        if os.fspath(path) == marker_path:
            return SimpleNamespace(st_uid=0, st_mode=info.st_mode, st_ino=info.st_ino)
        return info
    monkeypatch.setattr(boundary.os, "stat", root_owned_stat)

def _pid_inode(proc, inode, monkeypatch):
    pid_path = str(proc / "self" / "ns" / "pid")
    def stat(path, *args, **kwargs):
        if os.fspath(path) == pid_path:
            return SimpleNamespace(st_ino=inode)
        return _REAL_STAT(path, *args, **kwargs)
    monkeypatch.setattr(boundary.os, "stat", stat)

def test_root_overlay_and_injected_mounts_are_runtime_evidence(tmp_path, monkeypatch):
    proc, fs = _tree(tmp_path)
    _noninitial_pid(proc)
    lower = r"/var/lib/containerd/io.containerd.snapshotter.v1.overlayfs/snapshots/4/layer\040one"
    _mounts(proc, ("/", "/", "overlay", "overlay", f"rw,lowerdir={lower}"),
            ("/var/lib/docker/containers/id/hosts", "/etc/hosts", "ext4", "none", "rw"),
            ("/var/lib/docker/containers/id/hostname", "/etc/hostname", "ext4", "none", "rw"))
    result = _detect(proc, fs)
    assert result.classification == "container"
    assert "mount:containerd" in result.signals
    proc, fs = _tree(tmp_path / "kube")
    _noninitial_pid(proc)
    _mounts(proc, ("/var/lib/kubelet/pods/uid/etc-hosts", "/etc/hosts", "ext4", "none", "rw"))
    assert _detect(proc, fs).classification == "container"

def test_host_mounts_and_user_bind_mounts_are_not_runtime_signals(tmp_path, monkeypatch):
    proc, fs = _tree(tmp_path)
    _mounts(proc, ("/", "/var/lib/docker/rootfs/overlayfs/id", "overlay", "overlay",
                   "rw,lowerdir=/var/lib/docker/overlay2/id/diff"),
            ("/", "/var/lib/containerd/snapshots/id/fs", "overlay", "overlay",
             "rw,lowerdir=/var/lib/containerd/snapshots/id/lower"))
    _pid_inode(proc, 4026531836, monkeypatch)
    assert _detect(proc, fs).classification == "not_container"
    _pid_inode(proc, 12345, monkeypatch)  # Host bwrap has a private PID namespace only.
    assert _detect(proc, fs).classification == "not_container"
    proc, fs = _tree(tmp_path / "bind")
    _noninitial_pid(proc)
    _mounts(proc, ("/var/lib/docker/containers/id/rootfs", "/work", "ext4",
                   "/srv/project-var/lib/docker/containers/id/rootfs", "rw"))
    result = _detect(proc, fs)
    assert result.classification == "not_container"
    assert not result.signals

def test_cgroup_signal_and_unknown_detection(tmp_path):
    proc, fs = _tree(tmp_path)
    (proc / "self" / "ns" / "pid").unlink()
    assert _detect(proc, fs).classification == "unknown"
    proc, fs = _tree(tmp_path / "cgroup")
    _noninitial_pid(proc)
    (proc / "self" / "cgroup").write_text("0::/some-prefix-containerd.service\n")
    assert _detect(proc, fs).classification == "not_container"
    (proc / "self" / "cgroup").write_text("0::/kubepods.slice/pod-a\n")
    assert _detect(proc, fs).classification == "container"

def test_dockerenv_marker_alone_identifies_outer_container(tmp_path, monkeypatch):
    proc, fs = _tree(tmp_path)
    _noninitial_pid(proc)
    marker = fs / ".dockerenv"
    marker.write_text("")
    _pretend_root_owned_regular(monkeypatch, marker)
    assert _detect(proc, fs).classification == "container"

def test_unreadable_proc_and_malformed_mountinfo_are_safe(tmp_path, monkeypatch):
    proc, fs = _tree(tmp_path)
    (proc / "self" / "ns" / "pid").unlink()
    assert _detect(proc, fs).classification == "unknown"
    proc, fs = _tree(tmp_path / "malformed")
    _noninitial_pid(proc)
    (proc / "self" / "mountinfo").write_text("not a mountinfo record\n")
    assert _detect(proc, fs).classification == "not_container"
    marker = fs / ".dockerenv"
    marker.write_text("")
    _pretend_root_owned_regular(monkeypatch, marker)
    (proc / "self" / "mountinfo").chmod(0)
    assert _detect(proc, fs).classification == "unknown"

def test_untrusted_fake_markers_and_environment_are_ignored(tmp_path, monkeypatch):
    proc, fs = _tree(tmp_path)
    _noninitial_pid(proc)
    marker = fs / ".dockerenv"
    marker.write_text("fake")
    if os.geteuid() == 0:
        os.chown(marker, 65534, 65534)
    assert _detect(proc, fs).classification == "not_container"
    marker.unlink()
    marker.symlink_to(fs / "run" / "target")
    (fs / "run" / "target").write_text("fake")
    monkeypatch.setenv("container", "podman")
    monkeypatch.setenv("CONTAINER_IS_SANDBOX", "1")
    assert _detect(proc, fs).classification == "not_container"

def test_authorize_conflicts_uid_and_detection(capsys):
    detected = boundary.Detection("container", ("marker:/.dockerenv",), {"pid": "9"}, 1)
    assert boundary.authorize(False, conflicts=("yolo",), detection=detected, uid=1000).backend == boundary.BACKEND_PLATFORM
    for conflict in ("yolo", "shell-sandbox-off"):
        assert _refusal_code(lambda: boundary.authorize(True, conflicts=(conflict,), detection=detected, uid=1000)) == "CONTAINER_CONFLICT"
    assert _refusal_code(lambda: boundary.authorize(True, conflicts=("mutations",), detection=detected, uid=1000)) == "CONTAINER_MUTATIONS"
    assert _refusal_code(lambda: boundary.authorize(True, conflicts=("mutate-like-name",), detection=detected, uid=1000)) == "CONTAINER_CONFLICT"
    assert _refusal_code(lambda: boundary.authorize(True, conflicts=(), detection=detected, uid=0)) == "CONTAINER_ROOT_UID"
    for classification, expected in (("not_container", "CONTAINER_NOT_DETECTED"), ("unknown", "CONTAINER_UNKNOWN")):
        item = boundary.Detection(classification, (), {}, 1)
        assert _refusal_code(lambda: boundary.authorize(True, conflicts=(), detection=item, uid=1000)) == expected
    boundary.authorize(True, conflicts=(), detection=detected, uid=1000)
    assert capsys.readouterr().out == ""

def test_records_resume_validation_and_namespace_ids(tmp_path):
    detection = boundary.Detection("container", ("marker:/.dockerenv",), {"pid": "100", "mnt": "200"}, 1)
    current = boundary.authorize(True, conflicts=(), detection=detection, uid=1000)
    recorded = current.record()
    assert recorded["statement"] == boundary.BOUNDARY_STATEMENT
    changed_namespaces = boundary.Boundary(
        boundary.BACKEND_CONTAINER,
        boundary.Detection("container", detection.signals, {"pid": "999", "mnt": "888"}, 1),
    )
    assert boundary.check_resume(recorded, changed_namespaces) is None
    changed_class = boundary.Boundary(boundary.BACKEND_CONTAINER, boundary.Detection("not_container", (), {}, 1))
    assert _refusal_code(lambda: boundary.check_resume(recorded, changed_class)) == "CONTAINER_RESUME_MISMATCH"
    changed_backend = boundary.Boundary(boundary.BACKEND_PLATFORM, detection)
    assert _refusal_code(lambda: boundary.check_resume(recorded, changed_backend)) == "CONTAINER_RESUME_MISMATCH"
    platform = boundary.Boundary(boundary.BACKEND_PLATFORM, boundary.Detection("not_container", (), {}, 1))
    assert _refusal_code(lambda: boundary.check_resume(platform.record(), current)) == "CONTAINER_RESUME_MISMATCH"
    tampered = dict(recorded)
    tampered["statement"] = "changed"
    with pytest.raises((TypeError, ValueError)):
        boundary.Boundary.from_record(tampered)
    for bad in (None, [], {**recorded, "backend": []}, {**recorded, "signals": ["bad\r\nrecord"]},
                {**platform.record(), "classification": "container"}):
        with pytest.raises((TypeError, ValueError)):
            boundary.Boundary.from_record(bad)
    assert _refusal_code(lambda: boundary.check_resume({"corrupt": True}, current)) == "CONTAINER_RESUME_MISMATCH"
    assert boundary.check_resume(None, boundary.Boundary(boundary.BACKEND_PLATFORM, detection)) is None

def test_activation_is_once_per_process(monkeypatch):
    # Set through monkeypatch first, so teardown restores the inactive state.
    monkeypatch.setattr(boundary, "_active_boundary", None)
    assert boundary.active().backend == boundary.BACKEND_PLATFORM
    detection = boundary.Detection("container", (), {"pid": "1"}, 1)
    chosen = boundary.Boundary(boundary.BACKEND_CONTAINER, detection)
    boundary.activate(chosen)
    assert boundary.active() == chosen
    with pytest.raises(RuntimeError):
        boundary.activate(chosen)
