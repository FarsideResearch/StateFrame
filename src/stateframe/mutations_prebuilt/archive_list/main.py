"""List bounded zip and tar archives without extracting any member."""
import io
import ntpath
from pathlib import PurePosixPath
import tarfile
import zipfile


MAX_ARCHIVE_BYTES = 4 << 20
MAX_MEMBERS = 2000
MAX_EXPANDED_BYTES = 64 << 20
MAX_MEMBER_READ = 1 << 20
MAX_TEXT_CHARS = 64 << 10


def _unsafe(name):
    normalized = name.replace("\\", "/")
    drive, _ = ntpath.splitdrive(normalized)
    parts = PurePosixPath(normalized).parts
    return (bool(drive) or normalized.startswith("/") or ".." in parts or
            "\x00" in name)


def _entry(name, size, kind):
    if len(name.encode("utf-8")) > 255:
        raise ValueError("member_name_limit")
    return {"name": name, "size_bytes": max(0, int(size)), "kind": kind,
            "unsafe_name": _unsafe(name)}


def _text(data):
    decoded = data.decode("utf-8", errors="replace")
    return decoded[:MAX_TEXT_CHARS], len(decoded) > MAX_TEXT_CHARS


def _result(error=""):
    return {"members": [], "bytes_declared": 0, "selected_name": "",
            "selected_text": "", "selected_truncated": False, "error": error}


def _read_zip(data, selected):
    result = _result()
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_MEMBERS:
                result["error"] = "member_count_limit"
                return result
            total = 0
            selected_info = None
            for info in infos:
                total += info.file_size
                if total > MAX_EXPANDED_BYTES:
                    result["error"] = "expanded_size_limit"
                    return result
                name = info.filename
                kind = "directory" if info.is_dir() else "file"
                result["members"].append(_entry(name, info.file_size, kind))
                if selected is not None and name == selected and not info.is_dir():
                    selected_info = info
            result["bytes_declared"] = total
            if selected_info is not None:
                with archive.open(selected_info) as member:
                    content = member.read(MAX_MEMBER_READ)
                value, clipped = _text(content)
                result["selected_name"] = selected_info.filename
                result["selected_text"] = value
                result["selected_truncated"] = (selected_info.file_size > len(content) or
                                                 clipped)
            return result
    except (OSError, zipfile.BadZipFile, RuntimeError, ValueError):
        result["error"] = "invalid_archive"
        return result


def _read_tar(data, selected):
    result = _result()
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r|*") as archive:
            total = 0
            selected_name = ""
            selected_text = ""
            selected_truncated = False
            count = 0
            for info in archive:
                count += 1
                if count > MAX_MEMBERS:
                    result["error"] = "member_count_limit"
                    return result
                size = max(0, info.size)
                total += size
                if total > MAX_EXPANDED_BYTES:
                    result["error"] = "expanded_size_limit"
                    return result
                kind = ("directory" if info.isdir() else "link" if info.issym() or info.islnk()
                        else "file")
                result["members"].append(_entry(info.name, size, kind))
                if selected is not None and info.name == selected and info.isfile():
                    member = archive.extractfile(info)
                    if member is not None:
                        content = member.read(MAX_MEMBER_READ)
                        member.close()
                        selected_text, clipped = _text(content)
                        selected_name = info.name
                        selected_truncated = size > len(content) or clipped
            result["bytes_declared"] = total
            result["selected_name"] = selected_name
            result["selected_text"] = selected_text
            result["selected_truncated"] = selected_truncated
            return result
    except (OSError, tarfile.TarError, EOFError, ValueError):
        result["error"] = "invalid_archive"
        return result


def handle(input, ctx):
    input_id = input.get("input_id")
    if not isinstance(input_id, str) or input_id not in ctx.inputs:
        raise ValueError("input_id is required")
    data = ctx.input_bytes(input_id, limit=MAX_ARCHIVE_BYTES + 1)
    if len(data) > MAX_ARCHIVE_BYTES:
        return _result("archive_size_limit")
    selected = input.get("member")
    if selected is not None and (not isinstance(selected, str) or
                                 len(selected.encode("utf-8")) > 255):
        return _result("invalid_member")
    if zipfile.is_zipfile(io.BytesIO(data)):
        return _read_zip(data, selected)
    return _read_tar(data, selected)
