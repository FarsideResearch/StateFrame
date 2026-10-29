"""Expose the existing edit executor's scope without changing its authority."""
from __future__ import annotations
import ast
import copy
import os
from collections.abc import Mapping, Sequence
from pathlib import Path


def _product_paths(paths: object) -> list[str]:
    """Normalize the declared product surface without treating test files as targets."""
    if isinstance(paths, str):
        paths = (paths,)
    if not isinstance(paths, Sequence) or isinstance(paths, (bytes, bytearray)):
        return []
    result: list[str] = []
    for raw in paths:
        value = str(raw).replace("\\", "/").strip().lstrip("./")
        if not value or value.startswith("/"):
            continue
        parts = value.split("/")
        if (parts[0] in {"test", "tests"} or value.startswith("test_")
                or "/tests/" in "/" + value or "/test_" in "/" + value
                or parts[-1] == "conftest.py"):
            continue
        if value not in result:
            result.append(value)
    return sorted(result)


def measure_product_sizes(root: Path, paths: Sequence[str]) -> dict[str, dict[str, int | None]]:
    """Measure declared files from the live workspace before selecting edit scope."""
    result: dict[str, dict[str, int | None]] = {}
    workspace = root.resolve()
    for path in sorted(set(paths)):
        try:
            candidate = (workspace / path).resolve(strict=True)
            if not candidate.is_relative_to(workspace):
                raise ValueError('declared path escapes the workspace')
            source = candidate.read_bytes()
        except (OSError, RuntimeError, ValueError):
            result[path] = {'lines': None, 'bytes': None}
        else:
            result[path] = {'lines': len(source.splitlines()), 'bytes': len(source)}
    return result


def _whole_file_eligibility(
        declared: Sequence[str], sizes: Mapping[str, Mapping[str, int | None]] | None) -> dict:
    files = {}
    for path in sorted(declared):
        measured = sizes.get(path, {}) if isinstance(sizes, Mapping) else {}
        lines = measured.get('lines') if isinstance(measured, Mapping) else None
        bytes_count = measured.get('bytes') if isinstance(measured, Mapping) else None
        files[path] = {'lines': lines, 'bytes': bytes_count}
    return {
        'files': files,
        'eligible': bool(files) and all(
            isinstance(item['lines'], int) and not isinstance(item['lines'], bool)
            and 0 <= item['lines'] <= 400
            and isinstance(item['bytes'], int) and not isinstance(item['bytes'], bool)
            and 0 <= item['bytes'] <= 16384 for item in files.values()),
    }

# Single-authority wire contract (r38).  capability_release_runner's preflight asserts these exact
# strings; keep ONE definition so the runner and the renderer cannot drift (cdd3c5 crashed every run
# at step 0 on 09-25 because the runner still asserted the removed legacy instruction).
AUTHORITY_EDIT_DESCRIPTION = (
    'Use the single EditAuthority in the current frame for editable paths; '
    'the entry shortcut names a preferred starting point, while function_source '
    'may replace a top-level definition in any editable path.')
AUTHORITY_WHOLE_FILE_EDIT_DESCRIPTION = (
    'Use the single EditAuthority in the current frame for editable paths; '
    'function_source is the complete desired file for the selected path. '
    'Omitted file content is intentionally removed.')
AUTHORITY_SYSTEM_LINE = 'The single EditAuthority in the current frame describes editable paths and shortcuts.'


def enabled() -> bool:
    value = os.environ.get('KERNEL_EDIT_AFFORDANCE', '0')
    if value not in {'0', '1'}:
        raise ValueError('KERNEL_EDIT_AFFORDANCE must be 0 or 1')
    return value == '1'


def _with_edit_authority(scope: dict, editable_paths: Sequence[str],
                         shortcuts: Sequence[dict]) -> dict:
    """Attach the single wire-format scope authority to a legacy contract."""
    mode = str(scope.get('mode', '')).replace('_', '-').lower()
    normalization_mode = 'complete-file' if mode == 'whole-file' else 'complete-definition'
    normalization = {
        'schema': 'edit-normalization/1',
        'mode': normalization_mode,
        'requires_authoritative_base': True,
        'definition_parser_required': normalization_mode == 'complete-definition',
        'explicit_text_fallback': 'str_replace',
    }
    scope['edit_normalization'] = copy.deepcopy(normalization)
    scope['edit_authority'] = {
        'editable_paths': list(dict.fromkeys(str(path) for path in editable_paths if str(path))),
        'shortcuts': copy.deepcopy(list(shortcuts)),
        'normalization': copy.deepcopy(normalization),
    }
    return scope

def contract(*, multi_file: bool, entry_path: str | None,
             entry_fn: str | None, target_fn: str = "",
             product_paths: Sequence[str] = (),
             product_sizes: Mapping[str, Mapping[str, int | None]] | None = None) -> dict | None:
    if not enabled():
        return None
    if os.environ.get('STATEFRAME_VIEW', 'existing') != 'operational':
        raise ValueError('edit affordance requires operational frames')
    declared = _product_paths(product_paths)
    if entry_path and entry_path not in declared:
        declared.append(str(entry_path).replace("\\", "/").lstrip("./"))
        declared.sort()
    no_entry_definition = multi_file and not (entry_path and entry_fn)
    eligibility = (_whole_file_eligibility(declared, product_sizes)
                   if no_entry_definition else None)
    if no_entry_definition and eligibility['eligible']:
        scope = {'mode': 'whole_file', 'instruction':
            'edit replaces the complete file at path with function_source, including '
            'imports, classes and functions. Supply the entire desired file content; '
            'omitted content is removed. Every declared product file is editable: use '
            'str_replace for a smaller exact change. Existing acceptance tests remain protected.'}
        edit_paths = list(declared)
        scope.update(_typed_editability(declared, edit_paths, edit_paths))
        return _with_edit_authority(scope, declared, ({
            'tool': 'edit', 'mode': 'whole_file', 'paths': edit_paths,
            'label': 'edit replaces the complete file at the selected path',
        },))
    if multi_file:
        from kernel_model_profile import edit_scope_relaxed
        if no_entry_definition or edit_scope_relaxed():
            scope = {'mode': 'declared_product_files',
                     'instruction':
                     'edit replaces a complete top-level function or class in any '
                     'declared product file. Supply the complete replacement definition. '
                     'Use str_replace for smaller exact changes. Existing acceptance tests '
                     'remain protected.'}
            scope.update(_typed_editability(declared, declared, []))
            if no_entry_definition:
                scope['whole_file_eligibility'] = eligibility
            scope = _with_edit_authority(scope, declared, ())
            if no_entry_definition:
                scope['edit_authority']['whole_file_eligibility'] = copy.deepcopy(eligibility)
                # In definition mode the required base is the selected definition's
                # current source, even though normalization also reads its containing file.
                for normalization in (scope['edit_normalization'],
                                      scope['edit_authority']['normalization']):
                    normalization['authoritative_base_source'] = 'current top-level definition'
            return scope
        scope = {'mode': 'declared_product_files', 'path': entry_path, 'function': entry_fn,
                'instruction': 'edit replaces the complete top-level function or class '
                'named by function_source in any declared product file. Supply exactly '
                'one replacement definition. The entry function is a shortcut for the '
                'preferred starting point, not an edit boundary. Use str_replace for '
                'smaller exact changes. Existing acceptance tests remain protected.'}
        scope.update(_typed_editability(declared, declared, []))
        scope['edit_fn_shortcut'] = {'path': str(entry_path), 'function': str(entry_fn)}
        return _with_edit_authority(scope, declared, ({
            'tool': 'edit', 'path': str(entry_path), 'function': str(entry_fn),
            'label': f'edit replaces a top-level definition in any editable path; '
                     f'{entry_fn} in {entry_path} is the preferred entry shortcut',
        },))
    if '.' in target_fn:
        scope = {'mode': 'qualified_function', 'function': target_fn, 'instruction':
                f'edit replaces the qualified function {target_fn} with function_source. '
                'Supply the complete unindented function definition. '
                'All declared product files remain editable with str_replace. '
                'Existing acceptance tests remain protected.'}
        scope.update(_typed_editability(declared, [], []))
        return _with_edit_authority(scope, declared, ({
            'tool': 'edit', 'function': str(target_fn),
            'label': f'edit replaces the qualified function {target_fn}',
        },))
    scope = {'mode': 'top_level_function_or_class', 'instruction':
            'edit replaces a complete top-level function or class with function_source. '
            'Use str_replace for smaller exact changes in any declared product file. '
            'Existing acceptance tests remain protected.'}
    scope.update(_typed_editability(declared, declared, []))
    return _with_edit_authority(scope, declared, ({
        'tool': 'edit', 'mode': 'top_level_function_or_class', 'paths': declared,
        'label': 'edit replaces the complete file operation only when the authority '
                 'selects whole-file mode; otherwise it replaces a top-level function or class',
    },))


def _typed_editability(declared: Sequence[str], edit_paths: Sequence[str],
                       whole_file_paths: Sequence[str]) -> dict:
    paths = list(dict.fromkeys(str(path) for path in declared if str(path)))
    edit_paths = list(dict.fromkeys(str(path) for path in edit_paths if str(path)))
    whole_file_paths = list(dict.fromkeys(str(path) for path in whole_file_paths if str(path)))
    return {
        # These fields are intentionally explicit.  A model should not have to
        # infer a whitelist from a task's entry-function label.
        'editable_paths': paths,
        'edit_fn_shortcut': ({'path': edit_paths[0], 'function': ''}
                             if len(edit_paths) == 1 and not whole_file_paths else None),
        'tool_matrix': {
            'edit': {'operation': 'replace_entry_function_or_top_level',
                     'paths': edit_paths,
                     'whole_file_paths': whole_file_paths},
            'str_replace': {'operation': 'replace_exact_text', 'paths': paths},
            'typed_plan': {
                'operations': ['replace'],
                'composition': 'exact old-text match against current source',
            },
        },
    }

def tool_surface(tools: list[dict], scope: dict | None, *, authority: bool = False) -> list[dict]:
    if scope is None:
        from kernel_bound_edit import enabled as bound_edit_enabled, target_property
        if not bound_edit_enabled():
            return tools
        result = copy.deepcopy(tools)
        for tool in result:
            if tool.get('function', {}).get('name') != 'edit':
                continue
            parameters = tool['function'].setdefault('parameters', {})
            properties = parameters.setdefault('properties', {})
            properties['target_ref'] = target_property()
            required = list(parameters.get('required', ()))
            if 'target_ref' not in required:
                required.append('target_ref')
            parameters['required'] = required
            tool['function']['description'] = (
                '[STATEFRAME_BOUND_EDIT=on] ' + tool['function'].get('description', '')
                + ' Pass the exact current target_ref from the repair packet.')
        return result
    result = copy.deepcopy(tools)
    from kernel_bound_edit import enabled as bound_edit_enabled, target_property
    bound_on = bound_edit_enabled()
    wire_authority = scope.get('edit_authority', {}) if authority else None
    for tool in result:
        name = tool['function']['name']
        if name == 'edit':
            if wire_authority is not None:
                if scope.get('mode') == 'whole_file':
                    tool['function']['description'] = (
                        AUTHORITY_EDIT_DESCRIPTION + ' ' + AUTHORITY_WHOLE_FILE_EDIT_DESCRIPTION)
                elif scope.get('mode') == 'qualified_function':
                    tool['function']['description'] = (
                        AUTHORITY_EDIT_DESCRIPTION + ' Supply the complete unindented qualified '
                        'function definition. The current source of that qualified function '
                        'is the authoritative base.')
                else:
                    tool['function']['description'] = (
                        AUTHORITY_EDIT_DESCRIPTION + ' Supply one complete top-level definition '
                        '(function or class). The current source of that definition is the '
                        'authoritative base.')
                normalization = scope.get('edit_normalization', {})
                if isinstance(normalization, dict):
                    tool['function']['description'] += (
                        f" Normalization scope is {normalization.get('mode', 'declared')}"
                        "; it preserves every byte and hunk, and never repairs semantics.")
            else:
                tool['function']['description'] = scope['instruction']
            if scope['mode'] == 'whole_file':
                source_description = 'Entire desired file content.'
            elif scope['mode'] == 'qualified_function':
                source_description = (
                    'Complete unindented qualified function definition; the current source '
                    'of that qualified function is the authoritative base.')
            else:
                source_description = (
                    'Supply one complete top-level definition (function or class); '
                    'the current source of that definition is the authoritative base.')
            tool['function']['parameters']['properties']['function_source'][
                'description'] = source_description
        elif name == 'str_replace' and scope.get('editable_paths'):
            if wire_authority is not None:
                tool['function']['description'] = (
                    'Exact text replacement in a path listed by the single EditAuthority. '
                    'The old text must match the current file exactly once.')
            else:
                paths = ', '.join(scope['editable_paths'])
                tool['function']['description'] = (
                    f'Exact text replacement in any declared product file. Editable paths: {paths}. '
                    'The old text must match the current file exactly once.')
            properties = tool['function'].get('parameters', {}).get('properties', {})
            if isinstance(properties, dict) and 'path' in properties:
                properties['path']['description'] = (
                    'Declared product file path; the single EditAuthority lists eligible paths.')
        elif name == 'file_create' and scope.get('editable_paths'):
            tool['function']['description'] = (
                'Create a new product file when the task permits it. Existing product files '
                'are edited with edit or str_replace.')
        issued_target = scope.get('host_issued_target_ref')
        if (bound_on and name == 'str_replace'
                and isinstance(issued_target, dict)
                and issued_target.get('schema') in {
                    'revision-bound-edit/1', 'stateframe-bound-edit/1',
                }):
            parameters = tool['function'].get('parameters', {})
            properties = parameters.setdefault('properties', {})
            properties['target_ref'] = target_property()
            required = list(parameters.get('required', ()))
            if 'target_ref' not in required:
                required.append('target_ref')
            parameters['required'] = required
        if bound_on and name == 'edit':
            parameters = tool['function'].get('parameters', {})
            properties = parameters.setdefault('properties', {})
            properties['target_ref'] = target_property()
            required = list(parameters.get('required', ()))
            if 'target_ref' not in required:
                required.append('target_ref')
            parameters['required'] = required
            tool['function']['description'] = (
                '[STATEFRAME_BOUND_EDIT=on] ' + tool['function'].get('description', '')
                + ' Pass the exact current target_ref from the repair packet.')
        if name == 'str_replace' and authority:
            parameters = tool['function'].setdefault('parameters', {})
            properties = parameters.setdefault('properties', {})
            properties.setdefault('operation', {
                'type': 'string', 'enum': ['replace'],
                'description': 'Exact text replacement; insertion and move need a host-issued anchor.',
            })
    return result

def instructions(text: str, scope: dict | None, *, authority: bool = False) -> str:
    if scope is None:
        return text
    if authority:
        for old in (
            'The edit tool replaces a complete top-level function, not a class.',
            'Use str_replace for classes; edit replaces a complete top-level function only.',
            'The edit tool replaces a complete top-level function or class.',
            'Use str_replace for changes inside classes or functions; edit replaces a complete top-level function or class.',
        ):
            text = text.replace(old, AUTHORITY_SYSTEM_LINE)
        if scope.get('mode') == 'whole_file':
            text += (
                '\nThe edit replaces the complete file only when the single EditAuthority '
                'selects whole-file mode.')
        return text
    for old in (
        'The edit tool replaces a complete top-level function, not a class.',
        'Use str_replace for classes; edit replaces a complete top-level function only.',
        'The edit tool replaces a complete top-level function or class.',
        'Use str_replace for changes inside classes or functions; edit replaces a complete top-level function or class.',
    ):
        text = text.replace(old, scope['instruction'])
    return text

def apply_frame(frame: dict, scope: dict | None, path: str, *, authority: bool = False) -> None:
    if scope is None:
        return
    if authority:
        frame.setdefault('action_prerequisites', {})
        frame['action_prerequisites'].pop('edit', None)
        authority_value = scope.get('edit_authority', {
            'editable_paths': list(scope.get('editable_paths', ())),
            'shortcuts': [],
        })
        frame['action_prerequisites']['edit_scope'] = copy.deepcopy(authority_value)
        frame['action_prerequisites']['typed_edit_contract'] = {
            'operations': ['replace'],
            'legacy_old_str': 'exactly-one-match compatibility path',
        }
        return
    description = scope['instruction']
    frame['action_prerequisites']['edit'] = description
    frame['action_prerequisites']['edit_scope'] = copy.deepcopy(scope)


def qualified_source(source: str, name: str) -> str | None:
    """Resolve an explicit qualified symbol, without a leaf-name fallback."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    def visit(node: ast.AST, prefix: tuple[str, ...]) -> str | None:
        current = prefix
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            current = (*prefix, node.name)
            if '.'.join(current) == name:
                return ast.get_source_segment(source, node)
        for child in ast.iter_child_nodes(node):
            found = visit(child, current)
            if found is not None:
                return found
        return None
    return visit(tree, ())
