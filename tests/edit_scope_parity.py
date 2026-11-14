"""Edit-scope surface builder shared by the parity and size tests.

Builds the model-visible edit contract, tool surface and frame prerequisites for a declared
product set in every contract mode, so a refactor of the edit affordance can be compared by digest.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os

import kernel_edit_affordance as affordance

BASE_TOOLS = [
    {"type": "function", "function": {
        "name": "edit", "description": "edit", "parameters": {
            "type": "object", "properties": {"path": {"type": "string"},
                                              "function_source": {"type": "string"}},
            "required": ["path", "function_source"]}}},
    {"type": "function", "function": {
        "name": "str_replace", "description": "sr", "parameters": {
            "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "file_create", "description": "fc", "parameters": {
            "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
]


def paths_for(count: int) -> list[str]:
    return [f"pkg/sub{index % 17}/mod_{index:05d}.py" for index in range(count)]


def surfaces(count: int, *, size_lines: int = 20) -> dict[str, object]:
    """Contract + tools + frame prerequisites for each mode over ``count`` product files."""
    os.environ["KERNEL_EDIT_AFFORDANCE"] = "1"
    os.environ["STATEFRAME_VIEW"] = "operational"
    declared = paths_for(count)
    sizes = {path: {"lines": size_lines, "bytes": size_lines * 10} for path in declared}
    cases = {
        "whole_file": dict(multi_file=True, entry_path=None, entry_fn=None, product_sizes=sizes),
        "declared_unmeasured": dict(multi_file=True, entry_path=None, entry_fn=None),
        "entry_function": dict(multi_file=True, entry_path=declared[0], entry_fn="run"),
        "top_level": dict(multi_file=False, entry_path=declared[0], entry_fn="run", target_fn="run"),
        "qualified": dict(multi_file=False, entry_path=declared[0], entry_fn="C.run",
                          target_fn="C.run"),
    }
    result: dict[str, object] = {}
    for name, kwargs in cases.items():
        contract = affordance.contract(product_paths=declared, **kwargs)
        row: dict[str, object] = {"contract": contract}
        for authority in (True, False):
            tools = affordance.tool_surface(copy.deepcopy(BASE_TOOLS), contract, authority=authority)
            frame: dict[str, object] = {"action_prerequisites": {}}
            affordance.apply_frame(frame, contract, declared[0], authority=authority)
            row[f"tools_{authority}"] = tools
            row[f"frame_{authority}"] = frame
            row[f"instructions_{authority}"] = affordance.instructions(
                "The edit tool replaces a complete top-level function or class.", contract,
                authority=authority)
        result[name] = row
    return result


def serialize(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def digest(value: object) -> str:
    return hashlib.sha256(serialize(value).encode()).hexdigest()
