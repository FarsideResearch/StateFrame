"""Explicit provenance boundary for information supplied by the harness.

This is an information-flow contract, not an OS sandbox for arbitrary Python.
Public task fields remain public in every condition; private annotations never
become public merely because a compiler accepts an arbitrary metadata mapping.
"""
from __future__ import annotations

import copy
import hashlib
import os
import shlex
from pathlib import Path, PurePosixPath

VERSION = "information-policy/1"
# PUBLIC_PATHS_UNLOCALIZED (Stage 6 live rows, capability_release_runner.child) discloses
# strictly less than FILE_AND_SYMBOL: declared public paths only, no evaluator symbol.
TARGET_DISCLOSURES = frozenset({"FILE_AND_SYMBOL", "PUBLIC_PATHS_UNLOCALIZED"})
FIELDS = {"information_policy_version", "initial_oracle_symptom",
          "live_grader_feedback", "target_disclosure"}
PUBLIC_FIELDS = {
    "task_id", "repo", "sha", "fn", "rel", "path", "msg", "symptom",
    "parent_src", "test_src", "source_dataset", "base_commit",
    "instance_id", "test_path", "public_test_path", "FAIL_TO_PASS", "PASS_TO_PASS", "fail_to_pass",
    "pass_to_pass", "files", "public_tests", "public_test_files", "public_test_source",
    "public_test_src", "test_source", "changed_functions", "shape", "public_check_paths",
    "allowed_product_paths", "entry", "working_path", "_skip_db", "_oracle_preflight",
    "_bridge_ordinary_brief", "_bridge_capture", "information_policy",
}
_PUBLIC_NON_PRODUCT_PATH_PARTS = frozenset({
    ".git", ".pytest_cache", "__pycache__", "tests", "test", "probes",
    "scratch", "tmp", "private", "hidden_tests",
})


def _public_relative_path(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    raw = value.strip().replace("\\", "/")
    if (not raw or len(raw) > 1024 or raw.startswith("/") or "\x00" in raw
            or (len(raw) > 1 and raw[1] == ":")):
        return None
    if raw.startswith("./"):
        raw = raw[2:]
    relative = PurePosixPath(raw)
    parts = relative.parts
    if (not parts or any(part in {"..", "."} for part in parts)
            or any(part.casefold() in _PUBLIC_NON_PRODUCT_PATH_PARTS
                   or part.startswith(".") for part in parts)):
        return None
    return relative.as_posix()


def _public_path_strings(value: object) -> list[str]:
    if isinstance(value, dict):
        candidates = list(value)
    elif isinstance(value, (list, tuple)):
        candidates = list(value[:256])
    else:
        return []
    result: list[str] = []
    for candidate in candidates:
        path = _public_relative_path(candidate)
        if path is not None and path not in result:
            result.append(path)
        if len(result) >= 128:
            break
    return result


def _public_manifest_paths(value: object, *, depth: int = 0) -> list[str]:
    if depth > 5:
        return []
    found: list[str] = []
    if isinstance(value, dict):
        source_path = _public_relative_path(value.get("source_path"))
        if source_path is not None:
            found.append(source_path)
        for key in ("SNAPSHOT_MANIFEST", "snapshot_manifest", "manifest",
                    "sources", "entries", "files"):
            if key in value:
                found.extend(_public_manifest_paths(value[key], depth=depth + 1))
    elif isinstance(value, (list, tuple)):
        for item in value[:256]:
            found.extend(_public_manifest_paths(item, depth=depth + 1))
    return list(dict.fromkeys(found))[:128]


def _public_task_path_metadata(row: dict) -> dict[str, object]:
    result: dict[str, object] = {}
    scope = row.get("scope")
    if isinstance(scope, dict):
        public_scope: dict[str, object] = {}
        declared = _public_path_strings(scope.get("declared_paths"))
        if declared:
            public_scope["declared_paths"] = declared
        source_paths = _public_manifest_paths(scope)
        if source_paths:
            public_scope["SNAPSHOT_MANIFEST"] = [
                {"source_path": path} for path in source_paths
            ]
        if public_scope:
            result["scope"] = public_scope
    manifest_paths = _public_manifest_paths(row.get("SNAPSHOT_MANIFEST"))
    if manifest_paths:
        result["SNAPSHOT_MANIFEST"] = [
            {"source_path": path} for path in manifest_paths
        ]
    return result


def text_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


class InformationPolicy:
    def __init__(self, row: dict):
        supplied = row.get("information_policy")
        baseline = {"information_policy_version": VERSION,
                    "initial_oracle_symptom": os.environ.get("INITIAL_ORACLE_SYMPTOM", "ON"),
                    "live_grader_feedback": os.environ.get("LIVE_GRADER_FEEDBACK", "DIAGNOSTIC"),
                    "target_disclosure": "FILE_AND_SYMBOL"}
        if supplied is not None:
            if not isinstance(supplied, dict) or set(supplied) != FIELDS:
                raise ValueError("information_policy requires exactly the four declared policy fields")
            for env, key in (("INITIAL_ORACLE_SYMPTOM", "initial_oracle_symptom"),
                             ("LIVE_GRADER_FEEDBACK", "live_grader_feedback")):
                if env in os.environ and os.environ[env] != supplied[key]:
                    raise ValueError(f"contradictory information policy: {env}")
            baseline = dict(supplied)
        if (baseline["information_policy_version"] != VERSION
                or baseline["initial_oracle_symptom"] not in {"ON", "OFF"}
                or baseline["live_grader_feedback"] not in {"FINAL_ONLY", "STATUS_ONLY", "DIAGNOSTIC"}
                or baseline["target_disclosure"] not in TARGET_DISCLOSURES):
            raise ValueError("unsupported information policy")
        self.requested = baseline
        self.effective = dict(baseline)
        self.payload = str(row.get("_oracle_symptom") or "")
        self.source_revision = str(row.get("_symptom_source_revision") or text_hash(self.payload))
        self.ordinary_brief = ""
        self.exposures: list[str] = []
        self.initial_exposures: list[str] = []
        self.live_exposures: list[str] = []
        self.live_payloads: list[str] = []
        self.isolation_backend: str | None = None
        self.strict = self.off or self.feedback != "DIAGNOSTIC" or bool(row.get("_bridge_capture"))

    @property
    def off(self) -> bool:
        return self.effective["initial_oracle_symptom"] == "OFF"

    @property
    def feedback(self) -> str:
        return self.effective["live_grader_feedback"]

    def worker_row(self, row: dict) -> dict:
        from kernel_public_evidence import declared_public_check_paths
        declared_public_check_paths(row)
        # Gold source/test-patch fields are retained solely for final grading.
        # Existing AllowedTaskView excludes them. Drop free-form derived fields,
        # including nested metadata, advice and prior capsules, at ingress.
        if self.off:
            projected = {key: copy.deepcopy(value) for key, value in row.items()
                         if key in PUBLIC_FIELDS}
            # ``entry`` is a public controller orientation hint, not an oracle.
            # Keep only its path/function identity; nested annotations are not
            # allowed to smuggle a reference implementation across the boundary.
            entry = projected.get("entry")
            if isinstance(entry, dict) and isinstance(entry.get("path"), str):
                projected["entry"] = {"path": entry["path"], **(
                    {"fn": entry["fn"]} if isinstance(entry.get("fn"), str)
                    and entry.get("fn") else {})}
            elif entry is not None:
                projected.pop("entry", None)
            # A manifest-shaped ``files`` mapping is public parent material.  A
            # fixed source is grading/reference material even when it is nested
            # below that mapping, so omit it in strict worker rows.  FileSet
            # treats an omitted fixed source as the public parent for diagnostics.
            files = projected.get("files")
            if isinstance(files, dict):
                safe_files = {}
                for path, payload in files.items():
                    if not isinstance(payload, dict):
                        safe_files[path] = copy.deepcopy(payload)
                        continue
                    public_payload = {key: copy.deepcopy(value) for key, value in payload.items()
                                      if key not in {"fixed_src", "gold_src", "fixed_source",
                                                     "reference_src"}}
                    safe_files[path] = public_payload
                projected["files"] = safe_files
            elif isinstance(files, list):
                safe_files = []
                for item in files:
                    if not isinstance(item, dict):
                        safe_files.append(copy.deepcopy(item))
                        continue
                    safe_files.append({key: copy.deepcopy(value) for key, value in item.items()
                                       if key not in {"fixed_src", "gold_src", "fixed_source",
                                                      "reference_src"}})
                projected["files"] = safe_files
            if os.environ.get("KERNEL_REAL04_TARGET_METADATA", "off").strip().lower() != "off":
                from kernel_real04_bugs import preserve_target_metadata
                projected = preserve_target_metadata(projected, row)
            # Declared source paths are public task scope needed by the host's
            # completion gate. Keep only path identities; arbitrary annotations
            # stay controller-private.
            projected.update(_public_task_path_metadata(row))
            return projected
        return row  # Baseline keeps the historical row identity and mutation semantics.

    def manifest(self) -> dict:
        return {**self.effective,
                "policy_requested": dict(self.requested), "policy_effective": dict(self.effective),
                "ordinary_brief_hash": text_hash(self.ordinary_brief),
                "symptom_source_revision": self.source_revision,
                "symptom_payload_hash": text_hash(self.payload),
                "exposure_event_ids": list(self.exposures),
                "information_exposed": {"initial_symptom": bool(self.initial_exposures),
                                        "live_grader_feedback": bool(self.live_exposures)},
                "memory_policy": "DISABLED_UNPROVENANCED" if self.strict else "HISTORICAL",
                "test_isolation_backend": self.isolation_backend,
                "test_isolation_binary_sha256": (hashlib.sha256(Path(self.isolation_backend).read_bytes()).hexdigest()
                                                  if self.isolation_backend else None),
                "artifact_boundary": "WORKSPACE_TOOLS_AND_OS_ISOLATED_TESTS" if self.strict else "HISTORICAL"}

    def check_path(self, root: Path, value: str) -> bool:
        if not self.strict:
            return True
        raw = str(value).split("::", 1)[0]
        if ":" in raw and raw.rsplit(":", 1)[1].isdigit():
            raw = raw.rsplit(":", 1)[0]
        return (root / raw).resolve().is_relative_to(root.resolve())

    def done_feedback(self, diagnostic: str) -> str:
        if self.feedback == "DIAGNOSTIC":
            result = diagnostic
        elif self.feedback == "STATUS_ONLY":
            result = ("[grader] UNAVAILABLE: completion unverified." if "UNAVAILABLE" in diagnostic
                      else "[grader] RED: completion refused.")
        else:
            raise ValueError("FINAL_ONLY cannot deliver live grader feedback")
        if result not in self.live_payloads:
            self.live_payloads.append(result)
        return result

    def command_allowed(self, root: Path, command: str) -> bool:
        if not self.strict:
            return True
        try:
            words = shlex.split(command)
        except ValueError:
            return False
        if not words:
            return True
        if words[0] in {"read", "edit", "str_replace", "file_create"}:
            paths = words[1:2]
        elif words[0] == "test":
            paths = [word for word in words[1:] if not word.startswith("-")]
        else:
            paths = []  # grep checks its final resolved scope separately.
        return all(self.check_path(root, path) for path in paths)
