import errno
import os

import pytest

from stateframe import outer_checks
from stateframe.engine import kernel_execution_boundary as boundary


def _container_record(signals: tuple[str, ...]) -> dict[str, object]:
    detection = boundary.Detection(
        "container", signals, {"pid": "4026533000"}, boundary.DETECTOR_VERSION
    )
    return boundary.Boundary(boundary.BACKEND_CONTAINER, detection).record()


def _detector_signals() -> set[str]:
    return {
        "marker:/.dockerenv",
        "marker:/run/.containerenv",
        *(signal for _, signal in boundary._RUNTIME_MOUNT_ROOTS),
        *(signal for _, signal in boundary._CGROUP_RUNTIME_FORMS),
    }


def test_protected_test_hash_closes_descriptor_when_fstat_fails(monkeypatch, tmp_path):
    (tmp_path / "test_payload.py").write_text("value = 1\n", encoding="utf-8")
    opened: list[int] = []
    real_open = os.open
    real_fstat = os.fstat

    def track_open(path, flags, *args, **kwargs):
        descriptor = real_open(path, flags, *args, **kwargs)
        opened.append(descriptor)
        return descriptor

    def fail_fstat(_descriptor):
        raise OSError("simulated descriptor inspection failure")

    monkeypatch.setattr(outer_checks.os, "open", track_open)
    monkeypatch.setattr(outer_checks.os, "fstat", fail_fstat)

    tampered = outer_checks._protected_test_tampering(
        tmp_path, {"test_payload.py": "0" * 64}
    )

    assert tampered == ["test_payload.py"]
    assert len(opened) == 1
    with pytest.raises(OSError) as error:
        real_fstat(opened[0])
    assert error.value.errno == errno.EBADF


@pytest.mark.parametrize("signal", sorted(_detector_signals()))
def test_boundary_record_accepts_each_detector_signal(signal):
    record = _container_record((signal,))

    restored = boundary.Boundary.from_record(record)

    assert restored.detection.signals == (signal,)


def test_boundary_record_rejects_signal_not_emitted_by_detector():
    record = _container_record(("cgroup:docker",))
    record["signals"] = ["unrecognized-runtime-signal"]

    with pytest.raises(ValueError, match="invalid execution boundary record"):
        boundary.Boundary.from_record(record)
