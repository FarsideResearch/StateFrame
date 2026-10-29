"""Request sidecars for the expert-gap host facts (Q2, Q3, Q4).

Called from ``StateFrameRuntime._apply_rulebook_mechanisms`` beside the P2/P3
site facts, before the request token estimate, so every block counts against
the request budget.  The Q2 inventory and Q4 table describe the PARENT and are
computed once per run; the Q3 row-role baseline is observed per request.
Failures remain explicitly unavailable and never project as an all-clear.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import textwrap
from collections.abc import Mapping
from pathlib import Path, PurePosixPath


_MAX_SOURCE_BYTES = 256 * 1024
_MAX_CLOSURE_FILES = 12
_MAX_DEPENDENCIES_PER_FILE = 320
_MAX_UNIQUE_READS = 1024
_MAX_TOTAL_READ_BYTES = 8 * 1024 * 1024
_REVISION = re.compile(r"^sha256:[0-9a-f]{64}$")
_FORBIDDEN_PARTS = {".git", "backups", "evaluator", "grader", "graders", "gold", "hidden"}


def _safe_relative_path(value: object) -> bool:
    if (not isinstance(value, str) or not value or len(value) > 512
            or "\\" in value or "\x00" in value):
        return False
    path = PurePosixPath(value)
    return (not path.is_absolute() and path.as_posix() == value
            and bool(path.parts)
            and len(path.parts) <= 64
            and all(part not in {"", ".", ".."} for part in path.parts)
            and not any(part.casefold() in _FORBIDDEN_PARTS for part in path.parts))


def _directory_chain(root_fd: int, parts: tuple[str, ...]) -> tuple[tuple[int, int], ...] | None:
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    nofollow_flag = getattr(os, "O_NOFOLLOW", 0)
    cloexec_flag = getattr(os, "O_CLOEXEC", 0)
    if not directory_flag or not nofollow_flag:
        return None
    descriptor = -1
    identities: list[tuple[int, int]] = []
    try:
        descriptor = os.dup(root_fd)
        root_stat = os.fstat(descriptor)
        if not stat.S_ISDIR(root_stat.st_mode):
            return None
        identities.append((root_stat.st_dev, root_stat.st_ino))
        for part in parts:
            next_descriptor = os.open(
                part, os.O_RDONLY | directory_flag | nofollow_flag | cloexec_flag,
                dir_fd=descriptor,
            )
            next_stat = os.fstat(next_descriptor)
            if not stat.S_ISDIR(next_stat.st_mode):
                os.close(next_descriptor)
                return None
            os.close(descriptor)
            descriptor = next_descriptor
            identities.append((next_stat.st_dev, next_stat.st_ino))
        return tuple(identities)
    except OSError:
        return None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _open_root(root: object) -> tuple[Path, int] | None:
    """Walk the supplied absolute root without resolving symlink aliases."""
    if not isinstance(root, (str, os.PathLike)):
        return None
    try:
        path = Path(root)
        if (not path.is_absolute() or ".." in path.parts or len(path.parts) < 2
                or len(path.parts) > 128):
            return None
        directory_flag = getattr(os, "O_DIRECTORY", 0)
        nofollow_flag = getattr(os, "O_NOFOLLOW", 0)
        if not directory_flag or not nofollow_flag:
            return None
        descriptor = os.open(path.anchor, os.O_RDONLY | directory_flag | nofollow_flag)
        transferred = False
        try:
            for part in path.parts[1:]:
                if part in {"", ".", ".."}:
                    return None
                named = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
                if not stat.S_ISDIR(named.st_mode):
                    return None
                next_descriptor = os.open(
                    part, os.O_RDONLY | directory_flag | nofollow_flag,
                    dir_fd=descriptor,
                )
                opened = os.fstat(next_descriptor)
                if (not stat.S_ISDIR(opened.st_mode)
                        or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)):
                    os.close(next_descriptor)
                    return None
                os.close(descriptor)
                descriptor = next_descriptor
            current = os.fstat(descriptor)
            if not stat.S_ISDIR(current.st_mode):
                return None
            transferred = True
            return path, descriptor
        except OSError:
            return None
        finally:
            if not transferred:
                os.close(descriptor)
    except (OSError, TypeError, ValueError):
        return None


def _read_root_file(root_fd: int, relative: str, *, byte_limit: int) -> bytes | None:
    """Read one regular file through no-follow directory descriptors."""
    if not _safe_relative_path(relative):
        return None
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    nofollow_flag = getattr(os, "O_NOFOLLOW", 0)
    nonblock_flag = getattr(os, "O_NONBLOCK", 0)
    cloexec_flag = getattr(os, "O_CLOEXEC", 0)
    if not directory_flag or not nofollow_flag or not nonblock_flag:
        return None
    parts = relative.split("/")
    chain_before = _directory_chain(root_fd, tuple(parts[:-1]))
    if chain_before is None:
        return None
    parent_fd = -1
    leaf_fd = -1
    try:
        parent_fd = os.dup(root_fd)
        for part in parts[:-1]:
            next_fd = os.open(part, os.O_RDONLY | directory_flag | nofollow_flag | cloexec_flag,
                              dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = next_fd
        named_before = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(named_before.st_mode) or named_before.st_size < 0 or named_before.st_size > byte_limit:
            return None
        leaf_fd = os.open(parts[-1], os.O_RDONLY | nofollow_flag | nonblock_flag | cloexec_flag,
                           dir_fd=parent_fd)
        before = os.fstat(leaf_fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_size < 0
                or before.st_size > byte_limit
                or (before.st_dev, before.st_ino) != (named_before.st_dev, named_before.st_ino)):
            return None
        chunks: list[bytes] = []
        remaining = byte_limit + 1
        while remaining:
            chunk = os.read(leaf_fd, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(leaf_fd)
        named = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
        if (len(raw) > byte_limit or not stat.S_ISREG(named.st_mode)
                or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                or (after.st_dev, after.st_ino) != (named.st_dev, named.st_ino)
                or len(raw) != after.st_size
                or _directory_chain(root_fd, tuple(parts[:-1])) != chain_before):
            return None
        return raw
    except OSError:
        return None
    finally:
        if leaf_fd >= 0:
            os.close(leaf_fd)
        if parent_fd >= 0:
            os.close(parent_fd)


def _safe_existing_file(root_fd: int, relative: str) -> str | None:
    if not _safe_relative_path(relative):
        return None
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    nofollow_flag = getattr(os, "O_NOFOLLOW", 0)
    cloexec_flag = getattr(os, "O_CLOEXEC", 0)
    parent_fd = -1
    try:
        parent_fd = os.dup(root_fd)
        parts = relative.split("/")
        for part in parts[:-1]:
            next_fd = os.open(part, os.O_RDONLY | directory_flag | nofollow_flag | cloexec_flag,
                              dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = next_fd
        entry = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
        return relative if stat.S_ISREG(entry.st_mode) else None
    except OSError:
        return None
    finally:
        if parent_fd >= 0:
            os.close(parent_fd)


def _source_digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _current_candidate_revision(runtime: object) -> str | None:
    binding = getattr(runtime, "binding", None)
    if not callable(binding):
        return None
    try:
        value = binding()
    except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        return None
    if not isinstance(value, Mapping):
        return None
    revision = value.get("product_revision")
    return revision if isinstance(revision, str) and _REVISION.fullmatch(revision) else None


def _fresh_closure(runtime: object, root: Path, root_fd: int, target_path: str,
                   closure_module: object, ticket: object) -> tuple[dict[str, object], dict[str, str], dict[str, str]] | None:
    get_plan = getattr(closure_module, "closure_plan_for_runtime", None)
    if not callable(get_plan):
        return None
    try:
        plan = get_plan(runtime, ticket)
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return None
    if not isinstance(plan, Mapping):
        return None
    if (plan.get("schema") != "stateframe-request-evidence-closure/1"
            or plan.get("target_path") != target_path
            or plan.get("ticket_sha256") != getattr(ticket, "sha256", None)):
        return None
    paths_raw = plan.get("paths")
    if (not isinstance(paths_raw, (list, tuple)) or not paths_raw
            or len(paths_raw) > _MAX_CLOSURE_FILES
            or any(not _safe_relative_path(item) for item in paths_raw)
            or len(set(paths_raw)) != len(paths_raw) or target_path not in paths_raw):
        return None
    paths = tuple(paths_raw)
    for key in ("omissions", "skipped_files"):
        rows = plan.get(key)
        if not isinstance(rows, (list, tuple)) or rows:
            return None
    coverage = plan.get("coverage")
    if not isinstance(coverage, Mapping):
        return None
    complete = coverage.get("completeness")
    if (not isinstance(complete, Mapping) or complete.get("status") != "COMPLETE"
            or type(complete.get("selected_files")) is not int
            or complete.get("selected_files") != len(paths)
            or type(complete.get("declared_files")) is not int
            or complete.get("declared_files", 0) > _MAX_CLOSURE_FILES):
        return None
    for key in ("skipped_file", "omission", "lexical_site"):
        row = coverage.get(key)
        if not isinstance(row, Mapping) or row.get("status") != "COMPLETE":
            return None
        if type(row.get("count", 0)) is not int or row.get("count", 0) != 0:
            return None
    revisions_in = plan.get("revisions")
    deps_in = plan.get("dependency_revisions")
    adapters = plan.get("adapters")
    source_contents = plan.get("source_contents")
    if not all(isinstance(item, Mapping) for item in (revisions_in, deps_in, adapters, source_contents)):
        return None
    if any(set(item.keys()) != set(paths) for item in (revisions_in, deps_in, adapters, source_contents)):
        return None
    if any(not isinstance(adapters.get(path), Mapping)
           or adapters[path].get("completeness") != "KNOWN"
           or adapters[path].get("status") in {"UNKNOWN", "UNRUNNABLE", "UNAVAILABLE", "PARTIAL"}
           for path in paths):
        return None

    read_cache: dict[str, bytes] = {}
    total_bytes = 0

    def read_file(path: str) -> bytes | None:
        nonlocal total_bytes
        if path in read_cache:
            return read_cache[path]
        if len(read_cache) >= _MAX_UNIQUE_READS:
            return None
        remaining = _MAX_TOTAL_READ_BYTES - total_bytes
        if remaining < 0:
            return None
        raw = _read_root_file(root_fd, path, byte_limit=min(_MAX_SOURCE_BYTES, remaining))
        if raw is None or total_bytes + len(raw) > _MAX_TOTAL_READ_BYTES:
            return None
        total_bytes += len(raw)
        read_cache[path] = raw
        return raw

    fresh_revisions: dict[str, str] = {}
    fresh_dependencies: dict[str, str] = {}
    candidate_finder = getattr(closure_module, "_dependency_candidates", None)
    adapter_for = getattr(closure_module, "language_adapter_for", None)
    if not callable(candidate_finder) or not callable(adapter_for):
        return None
    for path in paths:
        raw = read_file(path)
        if raw is None:
            return None
        try:
            source = raw.decode("utf-8")
        except UnicodeError:
            return None
        current_revision = _source_digest(raw)
        if (revisions_in.get(path) != current_revision
                or source_contents.get(path) != source):
            return None
        fresh_revisions[path] = current_revision
        try:
            adapter = adapter_for(path)
            dependencies = candidate_finder(
                root, path, source, adapter, root_resolved=root,
                safe_path=lambda candidate: _safe_existing_file(root_fd, candidate),
            )
        except (OSError, RecursionError, RuntimeError, SyntaxError, TypeError, ValueError):
            return None
        if (not isinstance(dependencies, tuple)
                or len(dependencies) > _MAX_DEPENDENCIES_PER_FILE
                or any(_safe_existing_file(root_fd, dependency) != dependency for dependency in dependencies)):
            return None
        digest = hashlib.sha256()
        closure_paths = tuple(sorted(set((path, *dependencies))))
        for dependency in closure_paths:
            raw_dependency = read_file(dependency)
            if raw_dependency is None:
                return None
            digest.update(dependency.encode("utf-8") + b"\0")
            digest.update(raw_dependency)
        current_dependency_revision = "sha256:" + digest.hexdigest()
        if deps_in.get(path) != current_dependency_revision:
            return None
        fresh_dependencies[path] = current_dependency_revision

    identity = plan.get("identities")
    if not isinstance(identity, Mapping) or set(identity.keys()) != set(paths):
        return None
    if any(not isinstance(identity.get(path), Mapping)
           or identity[path].get("current_revision") != fresh_revisions[path]
           for path in paths):
        return None
    identity_coverage = coverage.get("current_parent_identity")
    if (not isinstance(identity_coverage, Mapping)
            or identity_coverage.get("status") != "CURRENT"
            or identity_coverage.get("values") != dict(identity)):
        return None
    parser_coverage = coverage.get("parser_capability")
    if not isinstance(parser_coverage, Mapping) or parser_coverage.get("status") != "KNOWN":
        return None
    dependency_coverage = coverage.get("dependency_revision")
    if (not isinstance(dependency_coverage, Mapping)
            or dependency_coverage.get("status") != "CURRENT"
            or dependency_coverage.get("values") != dict(deps_in)):
        return None
    authorization = plan.get("workspace_authorization")
    if not isinstance(authorization, Mapping):
        return None
    try:
        expected_authorization = {
            "scope": str(getattr(runtime, "workspace_scope", root)),
            "generation": str(getattr(
                runtime, "workspace_revision", getattr(runtime, "product_revision", "0")
            )),
        }
    except (TypeError, ValueError):
        return None
    if (set(authorization.keys()) != {"scope", "generation"}
            or any(not isinstance(value, str) or not value or len(value) > 1024
                   for value in expected_authorization.values())
            or authorization != expected_authorization):
        return None
    return dict(plan), fresh_revisions, fresh_dependencies


def _emit(runtime: object, event: str, **fields: object) -> None:
    emit = getattr(getattr(runtime, "journal", None), "emit", None)
    if callable(emit):
        try:
            emit(event, producer="kernel_expert_gap_sidecars", dispatch=False, **fields)
        except (TypeError, ValueError):
            pass


def _clear_timeline_wire_expectation(runtime: object) -> None:
    try:
        setattr(runtime, "_timing_timeline_wire_expectation", None)
    except (AttributeError, TypeError):
        pass


def _timeline_wire_proof_fields(record: Mapping[str, object],
                                expected: object) -> dict[str, object] | None:
    if (not isinstance(expected, tuple) or len(expected) != 2
            or not isinstance(expected[0], str) or not isinstance(expected[1], str)):
        return None
    payload_utf8 = record.get("payload_utf8")
    payload_hash = record.get("payload_hash")
    if not isinstance(payload_utf8, str) or not isinstance(payload_hash, str):
        return None
    try:
        wire = payload_utf8.encode("utf-8")
        body = json.loads(payload_utf8)
    except (UnicodeError, ValueError):
        return None
    body_hash = "sha256:" + hashlib.sha256(wire).hexdigest()
    messages = body.get("messages") if isinstance(body, Mapping) else None
    if (body_hash != payload_hash or not isinstance(messages, list)
            or not any(isinstance(message, Mapping)
                       and isinstance(message.get("content"), str)
                       and expected[0] in message["content"] for message in messages)):
        return None
    request_id = record.get("request_id")
    submission_event_id = record.get("event_id")
    submission_sequence = record.get("sequence")
    if (not isinstance(request_id, str) or not request_id
            or not isinstance(submission_event_id, str) or not submission_event_id
            or type(submission_sequence) is not int or submission_sequence < 1):
        return None
    return {
        "request_id": request_id,
        "submission_event_id": submission_event_id,
        "submission_sequence": submission_sequence,
        "body_sha256": body_hash,
        "request_submitted_payload_hash": payload_hash,
        "facts_sha256": expected[1],
    }


def _emit_timeline_wire_proof(original, runtime: object,
                              kind: str, **fields: object):
    record = original(kind, **fields)
    if kind != "request_submitted" or not isinstance(record, Mapping):
        return record
    proof = _timeline_wire_proof_fields(
        record, getattr(runtime, "_timing_timeline_wire_expectation", None))
    if proof is not None:
        try:
            original("timing_timeline_on_wire", producer="kernel_expert_gap_sidecars",
                     dispatch=False, **proof)
        except Exception:
            # Proof publication is advisory; never turn a serialized request into a failed send.
            pass
    return record


def _install_timeline_wire_hook(journal: object, runtime: object, original) -> None:
    def emit_with_timeline_proof(kind: str, **fields: object):
        return _emit_timeline_wire_proof(original, runtime, kind, **fields)

    try:
        setattr(journal, "_timing_timeline_wire_hook_owner", id(runtime))
        setattr(journal, "emit", emit_with_timeline_proof)
    except (AttributeError, TypeError):
        try:
            delattr(journal, "_timing_timeline_wire_hook_owner")
        except (AttributeError, TypeError):
            pass
        try:
            setattr(journal, "emit", original)
        except (AttributeError, TypeError):
            pass


def _arm_timeline_wire_proof(runtime: object, facts_text: str) -> None:
    """Record only exact facts that a post-serialization journal hook can verify."""
    journal = getattr(runtime, "journal", None)
    original = getattr(journal, "emit", None)
    if not callable(original):
        return
    hook_owner = getattr(journal, "_timing_timeline_wire_hook_owner", None)
    if hook_owner is not None and hook_owner != id(runtime):
        return
    facts_digest = "sha256:" + hashlib.sha256(facts_text.encode("utf-8")).hexdigest()
    try:
        setattr(runtime, "_timing_timeline_wire_expectation", (facts_text, facts_digest))
    except (AttributeError, TypeError):
        return
    if hook_owner != id(runtime):
        _install_timeline_wire_hook(journal, runtime, original)


def _timeline(runtime: object) -> str | None:
    """Q5 source-only timeline; OFF remains a no-inspection/no-mutation path."""
    from kernel_timing_timeline import analyse, mode, render
    if mode() != "on":
        return None
    _clear_timeline_wire_expectation(runtime)
    try:
        from kernel_paged_source import symbol_outline
        import kernel_request_evidence_closure as closure_module
        from kernel_request_evidence_closure import ticket_identity_for_runtime
        from kernel_request_source_window import editable_product_paths, entry_symbol
        owner = getattr(runtime, "owner", None)
        root_open = _open_root(getattr(owner, "root", None))
        if root_open is None:
            return None
        root, root_fd = root_open
        try:
            root_input = Path(getattr(owner, "root"))
            target_raw = getattr(runtime, "target", None)
            if not isinstance(target_raw, (str, os.PathLike)):
                return None
            target = Path(target_raw)
            if not target.is_absolute():
                target = root_input / target
            try:
                relative = target.absolute().relative_to(root_input.absolute()).as_posix()
            except ValueError:
                try:
                    relative = target.absolute().relative_to(root).as_posix()
                except ValueError:
                    return None
            if not _safe_relative_path(relative) or not relative.endswith(".py"):
                return None
            ticket = ticket_identity_for_runtime(runtime)
            if ticket is None or not isinstance(getattr(ticket, "sha256", None), str):
                return None
            declared = editable_product_paths(runtime)
            stored_plan = getattr(runtime, "_request_source_closure", None)
            stored_paths = stored_plan.get("paths") if isinstance(stored_plan, Mapping) else None
            if (not isinstance(declared, tuple) or not declared
                    or len(declared) > _MAX_CLOSURE_FILES
                    or not isinstance(stored_paths, (tuple, list))
                    or any(not _safe_relative_path(path) or path not in stored_paths
                           for path in declared)):
                return None
            fresh = _fresh_closure(runtime, root, root_fd, relative,
                                   closure_module, ticket)
            if fresh is None:
                return None
            plan, revisions, dependency_revisions = fresh
            raw_source = _read_root_file(root_fd, relative, byte_limit=_MAX_SOURCE_BYTES)
            if raw_source is None or _source_digest(raw_source) != revisions.get(relative):
                return None
            try:
                source = raw_source.decode("utf-8")
            except UnicodeError:
                return None
            current_revision = _current_candidate_revision(runtime)
            resumed = getattr(runtime, "_template_conformance_resumed", None)
            run_id = getattr(owner, "run_id", None)
            symbol = entry_symbol(runtime)
            if (current_revision is None or type(resumed) is not bool
                    or not isinstance(run_id, str) or not run_id or len(run_id) > 256
                    or not isinstance(symbol, str) or not symbol):
                return None
            rows = symbol_outline(source)
            matches = [row for row in rows
                       if isinstance(row, Mapping)
                       and row.get("kind") in {"function", "async_function"}
                       and row.get("qualified_name") == symbol]
            if len(matches) != 1:
                return None
            start = matches[0].get("start_line")
            end = matches[0].get("end_line")
            if (type(start) is not int or type(end) is not int or start < 1 or end < start
                    or end > len(source.splitlines())):
                return None
            source_lines = source.splitlines(keepends=True)
            owner_line = source_lines[start - 1]
            indentation = owner_line[:len(owner_line) - len(owner_line.lstrip(" \t"))]
            column_offset = len(indentation.encode("utf-8"))
            owner_source = textwrap.dedent("".join(source_lines[start - 1:end]))
            cache_key = (
                relative, symbol, start, end, ticket.sha256, current_revision,
                tuple(sorted(revisions.items())), tuple(sorted(dependency_revisions.items())),
                plan.get("workspace_authorization", {}).get("scope"),
                plan.get("workspace_authorization", {}).get("generation"),
                run_id, resumed,
            )
            cached = getattr(runtime, "_timing_timeline_cache", None)
            if (isinstance(cached, tuple) and len(cached) == 2
                    and cached[0] == cache_key):
                block = cached[1] if isinstance(cached[1], str) else None
                if block:
                    _arm_timeline_wire_proof(runtime, block)
                return block
            facts = analyse(
                owner_source, owner=symbol, path=relative, start_line=start,
                source_revision=revisions[relative], owner_span=(start, end),
                column_offset=column_offset,
            )
            block = render(facts)
            try:
                runtime._timing_timeline_cache = (cache_key, block)
            except (AttributeError, TypeError):
                pass
            if block:
                _arm_timeline_wire_proof(runtime, block)
            return block
        finally:
            os.close(root_fd)
    except (AttributeError, ImportError, KeyError, OSError, RecursionError,
            RuntimeError, SyntaxError, TypeError, UnicodeError, ValueError):
        return None


def blocks(runtime: object) -> list[str]:
    result: list[str] = []
    try:
        from kernel_lifecycle_table import runtime_table
        from kernel_row_role import observe, render, runtime_role
        from kernel_state_owner_inventory import render as render_inventory
        from kernel_state_owner_inventory import runtime_inventory
        first = not hasattr(runtime, "_state_owner_inventory")
        inventory = runtime_inventory(runtime)
        if inventory is not None:
            result.append(render_inventory(inventory))
            if first:
                _emit(runtime, "state_owner_inventory", **inventory.summary())
        table = runtime_table(runtime, inventory=inventory)
        if table:
            result.append(table)
            if first:
                _emit(runtime, "lifecycle_state_table", rendered=True)
        observe(runtime)
        role = runtime_role(runtime)
        if role != getattr(runtime, "_row_role_last", None):
            try:
                runtime._row_role_last = role
            except (AttributeError, TypeError):
                pass
            _emit(runtime, "row_role", **role)
        rendered = render(role)
        if rendered:
            result.append("[CURRENT FACTS]\n" + rendered)
        timeline = _timeline(runtime)
        if timeline:
            result.append(timeline)
    except (AttributeError, ImportError, KeyError, OSError, RecursionError, RuntimeError,
            SyntaxError, TypeError, ValueError) as error:
        if not getattr(runtime, "_expert_gap_failed", False):
            try:
                runtime._expert_gap_failed = True
            except (AttributeError, TypeError):
                pass
            _emit(runtime, "expert_gap_sidecars_failed", error_type=type(error).__name__)
        result.append(
            "[EXPERT GAP SIDECARS] status=UNAVAILABLE; host sidecar facts could not be "
            "verified for this revision. Treat dependent gates as UNKNOWN and use bounded "
            "read/check recovery."
        )
        return result
    return result


__all__ = ["blocks"]
