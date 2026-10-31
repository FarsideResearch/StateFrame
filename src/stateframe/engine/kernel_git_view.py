"""Pinned, configuration-independent read access to repository history."""
from __future__ import annotations

import difflib
import os
from pathlib import Path
import re
import stat
from typing import Callable, Mapping

from stateframe.workcopy import IGNORE_DIRS, _ignored

_OID = re.compile(r"[0-9a-fA-F]{40,64}\Z")
_REF = re.compile(r"(?:refs/)?[A-Za-z0-9_./-]+(?:[~^][0-9]*)*\Z")
_MAX_OUTPUT = 64 * 1024


class GitRequestError(ValueError):
    """A git request is unsafe or outside the read-only surface."""


class GitObjectError(RuntimeError):
    """The pinned repository object store cannot satisfy a read."""


def excluded_path(path: str) -> bool:
    """Apply the work-copy's directory and file exclusions to a repo path."""
    parts = Path(path).parts
    return (not parts or parts[-1] in IGNORE_DIRS or
            any(part in IGNORE_DIRS or _ignored(part, True) for part in parts[:-1]) or
            _ignored(parts[-1], False))


def _safe_path(value: object) -> str:
    if not isinstance(value, str) or not value or value.startswith("-") or "\x00" in value:
        raise GitRequestError("path must be a nonempty relative repository path")
    path = Path(value)
    if path.is_absolute() or any(part in ("..", ".") for part in path.parts) or ":" in value or "\\" in value:
        raise GitRequestError("path must be a plain relative repository path")
    if excluded_path(value):
        raise GitRequestError(f"path is excluded from the work copy: {value}")
    return value


def _safe_rev(value: object) -> str:
    if not isinstance(value, str) or not value or value.startswith("-") or not (_OID.fullmatch(value) or _REF.fullmatch(value)):
        raise GitRequestError("revision must be a hex object ID or a ref name")
    if ".." in value or "@{" in value or value.endswith(".") or value.endswith("/") or "//" in value:
        raise GitRequestError("invalid revision name")
    return value


def _git_metadata(repo: Path) -> tuple[Path, Path]:
    dot = repo / ".git"
    if dot.is_dir() and not dot.is_symlink():
        gitdir = dot.resolve()
    elif dot.is_file() and not dot.is_symlink():
        line = dot.read_text(encoding="utf-8").strip()
        if not line.startswith("gitdir: "):
            raise GitRequestError("invalid .git file")
        gitdir = (repo / line[8:]).resolve(strict=True)
    else:
        raise GitRequestError("not a git repository")
    common_file = gitdir / "commondir"
    common = (gitdir / common_file.read_text().strip()).resolve(strict=True) if common_file.is_file() else gitdir
    if not (common / "objects").is_dir():
        raise GitRequestError("git object directory is missing")
    return gitdir, common


def _alternates(objects: Path, seen: set[Path]) -> list[Path]:
    objects = objects.resolve(strict=True)
    if objects in seen:
        return []
    seen.add(objects)
    paths = [objects]
    file = objects / "info" / "alternates"
    if file.is_file():
        for line in file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                alternate = Path(line)
                paths += _alternates(alternate if alternate.is_absolute() else objects / alternate, seen)
    return paths


def _refs(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    refdir = root / "refs"
    if refdir.is_dir():
        for path in refdir.rglob("*"):
            if path.is_file() and not path.is_symlink():
                name = path.relative_to(root).as_posix()
                _validate_ref(name)
                result[name] = path.read_text().strip()
    packed = root / "packed-refs"
    if packed.is_file():
        for line in packed.read_text().splitlines():
            if line and line[0] not in "#^":
                oid, name = line.split(" ", 1)
                _validate_ref(name)
                result.setdefault(name, oid)
    return result


def _validate_ref(name: str) -> None:
    parts = name.split("/")
    if (not name.startswith("refs/") or len(parts) < 3
            or any(not p or p.startswith(".") or p.endswith((".", ".lock")) for p in parts)
            or ".." in name or "@{" in name
            or any(ord(c) < 33 or ord(c) == 127 or c in "~^:?*[\\" for c in name)):
        raise GitRequestError(f"invalid git ref name: {name!r}")


def _write_ref(private: Path, name: str, value: str) -> None:
    _validate_ref(name)
    fd = os.open(private, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in name.split("/")[:-1]:
            try:
                os.mkdir(component, mode=0o700, dir_fd=fd)
            except FileExistsError:
                pass
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        output = os.open(name.split("/")[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=fd)
        with os.fdopen(output, "w") as stream:
            stream.write(value.lower() + "\n")
    finally:
        os.close(fd)


def _resolve_head(head: str, refs: Mapping[str, str]) -> str:
    seen: set[str] = set()
    while head.startswith("ref: "):
        name = head[5:].strip()
        if name in seen or name not in refs:
            raise GitRequestError("HEAD does not resolve to a commit")
        seen.add(name)
        head = refs[name]
    if not _OID.fullmatch(head):
        raise GitRequestError("HEAD has an invalid object ID")
    return head.lower()


class GitView:
    def __init__(self, *, git_dir: Path, workspace: Path, baseline: Path, head_oid: str,
                 object_dirs: tuple[Path, ...], secret_filter: Callable[[str], bool]):
        self.git_dir = git_dir
        self.workspace = workspace
        self.baseline = baseline
        self.head_oid = head_oid
        self.object_dirs = object_dirs
        self.secret_filter = secret_filter

    @classmethod
    def build(cls, source_repo: Path, run_dir: Path, *, secret_filter: Callable[[str], bool]) -> GitView | None:
        source_repo = source_repo.resolve()
        try:
            gitdir, common = _git_metadata(source_repo)
        except GitRequestError as exc:
            if str(exc) == "not a git repository":
                return None
            raise
        run_dir = run_dir.resolve()
        private = run_dir / "git-view"
        private.mkdir(parents=True, exist_ok=False)
        (private / "objects" / "info").mkdir(parents=True)
        (private / "empty-hooks").mkdir()
        (private / "scratch-home").mkdir()
        object_dirs = tuple(_alternates(common / "objects", set()))
        (private / "objects" / "info" / "alternates").write_text("".join(f"{path}\n" for path in object_dirs))
        refs = _refs(common)
        if gitdir != common:
            refs.update(_refs(gitdir))
        head = (gitdir / "HEAD").read_text().strip()
        oid = _resolve_head(head, refs)
        (private / "HEAD").write_text(f"{oid}\n")
        for name, value in refs.items():
            if not name.startswith("refs/") or not _OID.fullmatch(value):
                continue
            _write_ref(private, name, value)
        (private / "config").write_text(
            "[core]\n\tbare = false\n\tfsmonitor = false\n\tpager = cat\n\thooksPath = empty-hooks\n"
            "[diff]\n\texternal =\n")
        view = cls(git_dir=private, workspace=run_dir / "workspace", baseline=run_dir / "parent",
                   head_oid=oid, object_dirs=object_dirs, secret_filter=secret_filter)
        return view

    def env(self) -> dict[str, str]:
        return {"PATH": os.defpath, "GIT_DIR": str(self.git_dir), "GIT_WORK_TREE": str(self.workspace),
                "GIT_NO_LAZY_FETCH": "1", "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_TERMINAL_PROMPT": "0", "GIT_PAGER": "cat", "GIT_OPTIONAL_LOCKS": "0",
                "GIT_ATTR_NOSYSTEM": "1", "LC_ALL": "C", "HOME": str(self.git_dir / "scratch-home")}

    def _scope(self) -> list[str]:
        # Git pathspec exclusions are a first barrier; filter_output is the second.
        from stateframe.workcopy import SECRET_GLOBS, IGNORE_GLOBS
        patterns = [f":(exclude,glob)**/{name}/**" for name in sorted(IGNORE_DIRS)]
        patterns += [f":(exclude,glob)**/{name}" for name in (*SECRET_GLOBS, *IGNORE_GLOBS)]
        return [".", *patterns]

    def _path(self, value: object) -> str:
        path = _safe_path(value)
        if self.secret_filter(path):
            raise GitRequestError(f"path is excluded from the work copy: {path}")
        return path

    def argv(self, action: str, params: Mapping[str, object]) -> list[str]:
        if action not in {"status", "diff", "log", "show", "blame"}:
            raise GitRequestError(f"unsupported git action: {action}")
        allowed = {"status": {"against"}, "diff": {"against", "paths"},
                   "log": {"max_count", "path", "since_rev", "rev"},
                   "show": {"rev", "path"}, "blame": {"path", "start_line", "end_line", "rev"}}[action]
        extra = set(params) - allowed
        if extra:
            raise GitRequestError(f"unsupported {action} parameters: {', '.join(sorted(extra))}")
        base = ["git", f"--git-dir={self.git_dir}", f"--work-tree={self.workspace}",
                "--no-pager", "-c", "core.fsmonitor=false", "-c", "core.hooksPath=empty-hooks"]
        if action in {"status", "diff"}:
            against = params.get("against", "baseline")
            if against not in {"baseline", "HEAD"}:
                raise GitRequestError("against must be baseline or HEAD")
            if action == "status":
                return base + ["status", "--porcelain=v1", "--untracked-files=all", "--", *self._scope()]
            paths = params.get("paths", [])
            if not isinstance(paths, list) or any(not isinstance(p, str) for p in paths):
                raise GitRequestError("paths must be a list of repository paths")
            selected = [self._path(p) for p in paths]
            if against == "HEAD":
                return base + ["ls-tree", "-lrz", "--full-tree", self.head_oid]
            return base + ["diff", "--no-ext-diff", "--no-textconv", "--no-color", self.head_oid,
                           "--", *(selected or self._scope())]
        if action == "log":
            count = params.get("max_count", 20)
            if type(count) is not int or not 1 <= count <= 50:
                raise GitRequestError("max_count must be an integer from 1 to 50")
            rev = (_safe_rev(params["since_rev"]) + ".." + self.head_oid
                   if "since_rev" in params else _safe_rev(params.get("rev", self.head_oid)))
            path = params.get("path")
            return base + ["log", "--no-color", f"--max-count={count}", "--format=medium", rev,
                           "--", *([self._path(path)] if path is not None else self._scope())]
        if action == "show":
            # Git's commit peel rejects raw blob/tree IDs before rendering any
            # bytes. Paths may only address an authorized commit tree.
            rev = _safe_rev(params.get("rev", self.head_oid)) + "^{commit}"
            path = params.get("path")
            if path is not None:
                return base + ["show", "--no-ext-diff", "--no-textconv", "--no-color",
                               f"{rev}:{self._path(path)}"]
            return base + ["show", "--no-ext-diff", "--no-textconv", "--no-color", rev,
                           "--", *self._scope()]
        path = self._path(params.get("path"))
        start = params.get("start_line")
        end = params.get("end_line")
        if (start is None) != (end is None) or (start is not None and
                (type(start) is not int or type(end) is not int or start < 1 or end < start)):
            raise GitRequestError("blame requires a positive inclusive line range")
        line = ["-L", f"{start},{end}"] if start is not None else []
        return base + ["blame", *line,
                       _safe_rev(params.get("rev", self.head_oid)), "--", path]

    def filter_output(self, action: str, text: str) -> str:
        if action == "status":
            return "".join(line for line in text.splitlines(keepends=True)
                           if len(line) > 3 and not self._excluded(line[3:].strip().strip('"')))
        if action in {"diff", "show"}:
            sections = re.split(r"(?=^diff --git )", text, flags=re.MULTILINE)
            kept = []
            for section in sections:
                if section.startswith("diff --git "):
                    match = re.match(r"diff --git a/(.*?) b/(.*?)\n", section)
                    # Git quotes unusual names; without a plain path header we
                    # cannot establish that the section is authorized.
                    if not match or self._excluded(match.group(1)) or self._excluded(match.group(2)):
                        continue
                kept.append(section)
            text = "".join(kept)
        return text

    def _excluded(self, path: str) -> bool:
        return excluded_path(path) or self.secret_filter(path)

    def head_files(self, session_runner: Callable) -> tuple[dict[str, bytes], tuple[str, ...]]:
        """Read only baseline/working paths from the pinned tree, without an index."""
        base = ["git", f"--git-dir={self.git_dir}", f"--work-tree={self.workspace}", "--no-pager"]
        roots = (*self.object_dirs, self.git_dir, self.workspace)
        code, raw = session_runner(base + ["ls-tree", "-lrz", "--full-tree", self.head_oid],
                                   self.env(), roots)
        if code:
            raise GitObjectError(f"cannot list pinned HEAD tree: {raw.decode(errors='replace')[:300]}")
        if raw and not raw.endswith(b"\0"):
            raise GitObjectError("pinned tree listing is incomplete")
        eligible = _files(self.baseline).keys() | _files(self.workspace).keys()
        blobs: dict[str, bytes] = {}
        submodules: list[str] = []
        for entry in raw.split(b"\0"):
            if not entry:
                continue
            try:
                metadata, encoded_path = entry.split(b"\t", 1)
                mode, kind, oid, size = metadata.split()
                path = encoded_path.decode("utf-8", errors="surrogateescape")
            except (ValueError, UnicodeError) as exc:
                raise GitObjectError("invalid pinned tree entry") from exc
            if self._excluded(path):
                continue
            if kind == b"commit":
                submodules.append(path)
            elif kind == b"blob" and path in eligible and mode != b"120000":
                if size == b"BAD":
                    raise GitObjectError(f"pinned object unavailable for {path}")
                try:
                    blob_size = int(size)
                except ValueError as exc:
                    raise GitObjectError(f"invalid pinned blob size for {path}") from exc
                if blob_size > (1 << 20):
                    continue
                code, data = session_runner(base + ["cat-file", "blob", oid.decode("ascii")],
                                            self.env(), roots)
                if code:
                    raise GitObjectError(f"pinned object unavailable for {path}: {data.decode(errors='replace')[:300]}")
                blobs[path] = data
        return blobs, tuple(sorted(submodules))


def _files(root: Path) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    total = 0
    if not root.is_dir():
        return result
    for directory, dirs, files in os.walk(root, followlinks=False):
        here = Path(directory)
        dirs[:] = sorted(d for d in dirs if not _ignored(d, True) and not (here / d).is_symlink())
        for name in files:
            path = here / name
            rel = path.relative_to(root).as_posix()
            if not excluded_path(rel) and path.is_file() and not path.is_symlink():
                if len(result) >= 5000:
                    raise GitRequestError("workspace exceeds Git view file count limit")
                try:
                    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                    try:
                        info = os.fstat(descriptor)
                        if not stat.S_ISREG(info.st_mode) or info.st_size > (1 << 20):
                            raise GitRequestError(f"workspace file exceeds Git view size limit: {rel}")
                        chunks: list[bytes] = []
                        length = 0
                        while length <= (1 << 20):
                            chunk = os.read(descriptor, (1 << 20) + 1 - length)
                            if not chunk:
                                break
                            chunks.append(chunk)
                            length += len(chunk)
                        data = b"".join(chunks)
                    finally:
                        os.close(descriptor)
                except OSError as exc:
                    raise GitRequestError(f"cannot read workspace file {rel}: {exc}") from exc
                if len(data) > (1 << 20):
                    raise GitRequestError(f"workspace file exceeds Git view size limit: {rel}")
                total += len(data)
                if total > (64 << 20):
                    raise GitRequestError("workspace exceeds Git view total size limit")
                result[rel] = data
    return result


def engine_diff(baseline_manifest_or_dir: Mapping[str, bytes] | Path, workspace: Path,
                *, secret_filter: Callable[[str], bool] = lambda _: False,
                paths: set[str] | None = None) -> str:
    """Unified diff over baseline paths union additions, retaining deleted paths."""
    before = (_files(baseline_manifest_or_dir) if isinstance(baseline_manifest_or_dir, Path)
              else dict(baseline_manifest_or_dir))
    after = _files(workspace)
    chunks: list[str] = []
    for rel in sorted(before.keys() | after.keys()):
        if excluded_path(rel) or secret_filter(rel) or (paths is not None and rel not in paths) or before.get(rel) == after.get(rel):
            continue
        old = before.get(rel, b"").decode("utf-8", errors="replace")
        new = after.get(rel, b"").decode("utf-8", errors="replace")
        chunks.extend(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
                      fromfile=f"a/{rel}" if rel in before else "/dev/null",
                      tofile=f"b/{rel}" if rel in after else "/dev/null"))
    return "".join(chunks)


def _bounded(text: str) -> tuple[str, int, bool]:
    raw = text.encode("utf-8")
    if len(raw) <= _MAX_OUTPUT:
        return text, len(raw), False
    half = _MAX_OUTPUT // 2
    return (raw[:half].decode("utf-8", errors="replace") +
            f"\n[git output truncated: {len(raw)} bytes total]\n" +
            raw[-half:].decode("utf-8", errors="replace"), len(raw), True)


def handle_git_call(view: GitView | None, session_runner: Callable, arguments: Mapping[str, object]) -> tuple[str, dict]:
    """Validate and execute one observation through the caller's sandboxed runner."""
    if view is None:
        return "git unavailable: source is not a git repository", {"ok": False, "refusal": "not_git_repo"}
    try:
        action = arguments.get("action")
        if not isinstance(action, str):
            raise GitRequestError("action is required")
        params = {key: value for key, value in arguments.items() if key != "action"}
        argv = view.argv(action, params)
        if action in {"status", "diff"}:
            against = params.get("against", "baseline")
            submodules: tuple[str, ...] = ()
            if against == "HEAD":
                before, submodules = view.head_files(session_runner)
            else:
                before = _files(view.baseline)
            selected = set(params.get("paths", [])) if action == "diff" and params.get("paths") else None
            patch = engine_diff(before, view.workspace, secret_filter=view.secret_filter, paths=selected)
            if action == "status":
                after = _files(view.workspace)
                patch = "".join(f"{('D' if p not in after else 'A' if p not in before else 'M')} {p}\n"
                                for p in sorted(before.keys() | after.keys())
                                if before.get(p) != after.get(p) and not view._excluded(p))
            if submodules:
                patch += "Submodules not traversed: " + ", ".join(submodules) + "\n"
            output = patch
        else:
            code, raw = session_runner(argv, view.env(), (*view.object_dirs, view.git_dir, view.workspace))
            if code:
                detail = raw.decode("utf-8", errors="replace")[:500]
                raise GitObjectError(f"git {action} failed ({code}): {detail}")
            output = view.filter_output(action, raw.decode("utf-8", errors="replace"))
        shown, total, truncated = _bounded(output)
        return shown, {"ok": True, "action": action, "head_oid": view.head_oid,
                       "total_bytes": total, "truncated": truncated, "submodules_traversed": False,
                       "replay_class": "observation"}
    except GitRequestError as exc:
        return f"git request refused: {exc}", {"ok": False, "refusal": "invalid_request", "detail": str(exc)}
    except GitObjectError as exc:
        return f"git object unavailable: {exc}", {"ok": False, "refusal": "object_unavailable", "detail": str(exc)}


def legacy_git_query(repository: Path, args: list[str], *, timeout: float = 10) -> str | None:
    """Repo-intel queries use the same private config and offline process boundary.

    Remote-config cache keys deliberately become unavailable: a sanitized view
    has no remotes. Optional indexing must not consult executable repo config.
    """
    import shutil
    import tempfile
    from kernel_safe_probe import run_probe

    repository = Path(repository).resolve()
    while not (repository / ".git").exists():
        if repository == repository.parent:
            return None
        repository = repository.parent
    executable = shutil.which("git", path=os.defpath)
    if executable is None:
        raise GitObjectError("git executable unavailable")
    with tempfile.TemporaryDirectory(prefix="stateframe-git-intel-") as directory:
        view = GitView.build(repository, Path(directory), secret_filter=excluded_path)
        if view is None:
            return None
        view.workspace = repository
        result = run_probe(
            [executable, "--no-pager", *args], repository, timeout=timeout,
            env=view.env(), read_roots=(view.git_dir, *view.object_dirs))
        if result.timed_out or result.exit_code != 0:
            return None
        if result.output_truncated:
            raise GitObjectError("repo-intel git output exceeded its bounded collector")
        return result.output.decode("utf-8", "replace")
