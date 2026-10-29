"""Opt-in one-decision tool pause after three unchanged identical outcomes."""
from __future__ import annotations

import copy
import json
import re

_AVAILABILITY_PREFIX = '\nCurrent tool availability: '
_AVAILABILITY_LINE = re.compile(r'\nCurrent tool availability: [^\n]*')


def restrict_repeated_tool(tools: list[dict], records: list[dict]) -> tuple[list[dict], dict | None]:
    recent = records[-3:]
    if len(recent) != 3:
        return tools, None
    last = recent[-1]
    identity = (last['tool'], last['arguments_hash'], last['result'])
    if any((row['tool'], row['arguments_hash'], row['result']) != identity
           or row['before_binding'] != row['after_binding']
           or row['after_binding'] != last['after_binding'] for row in recent):
        return tools, None
    remaining = [tool for tool in tools if tool['function']['name'] != last['tool']]
    if not remaining or len(remaining) == len(tools):
        return tools, None
    return remaining, {'tool': last['tool'], 'duration': 'this_decision_only',
        'reason': 'Three identical actions produced the same outcome without changing the workspace. '
                  'Choose a different tool for this decision; the paused tool becomes available afterward.',
        'supporting_steps': [row['step'] for row in recent]}


def synchronize_tool_instructions(messages: list[dict], tools: list[dict], restriction: dict) -> list[dict]:
    result = copy.deepcopy(messages)
    marker = '\nAvailable tools:\n'
    for message in result:
        if message.get('role') != 'system':
            continue
        content = str(message.get('content') or '')
        if marker in content:
            prefix, tail = content.split(marker, 1)
            _, end = json.JSONDecoder().raw_decode(tail)
            content = prefix + marker + json.dumps(tools) + tail[end:]
        # b1380: this ran every call, so the system prompt accumulated one
        # "Current tool availability: null" line per turn, and the model read "null" as "no tools available"
        # while edit/str_replace were on its menu. Keep exactly one line, and only for a real restriction.
        content = _AVAILABILITY_LINE.sub('', content)
        if restriction is not None:
            content += _AVAILABILITY_PREFIX + json.dumps(restriction)
        message['content'] = content
        break
    return result
