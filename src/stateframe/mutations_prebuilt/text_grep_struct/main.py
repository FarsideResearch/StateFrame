"""Bounded regular-expression previews using a step-limited Thompson NFA."""
import io


MAX_INPUT = 4 << 20
MAX_PATTERN = 256
MAX_STATES = 512
MAX_STEPS = 200000
MAX_MATCHES = 100
MAX_MATCH_CHARS = 256


class _PatternError(ValueError):
    pass


class _StepBudget(Exception):
    pass


class _Parser:
    def __init__(self, pattern):
        if len(pattern) > MAX_PATTERN:
            raise _PatternError("pattern_too_long")
        self.ignore_case = pattern.startswith("(?i)")
        self.pattern = pattern[4:] if self.ignore_case else pattern
        self.position = 0
        self.depth = 0

    def parse(self):
        tree = self._expression()
        if self.position != len(self.pattern):
            raise _PatternError("invalid_pattern")
        return tree

    def _expression(self):
        alternatives = [self._sequence()]
        while self.position < len(self.pattern) and self.pattern[self.position] == "|":
            self.position += 1
            alternatives.append(self._sequence())
        return alternatives[0] if len(alternatives) == 1 else ("alt", tuple(alternatives))

    def _sequence(self):
        nodes = []
        while self.position < len(self.pattern) and self.pattern[self.position] not in ")|":
            nodes.append(self._repeat())
        return "seq", tuple(nodes)

    def _repeat(self):
        node = self._atom()
        if self.position < len(self.pattern) and self.pattern[self.position] in "*+?":
            operator = self.pattern[self.position]
            self.position += 1
            node = (operator, node)
            if self.position < len(self.pattern) and self.pattern[self.position] in "*+?":
                raise _PatternError("invalid_pattern")
        return node

    def _atom(self):
        if self.position >= len(self.pattern):
            raise _PatternError("invalid_pattern")
        char = self.pattern[self.position]
        self.position += 1
        if char == "(":
            self.depth += 1
            if self.depth > 16:
                raise _PatternError("pattern_depth_limit")
            if self.pattern.startswith("?:", self.position):
                self.position += 2
            elif self.position < len(self.pattern) and self.pattern[self.position] == "?":
                raise _PatternError("unsupported_group")
            value = self._expression()
            if self.position >= len(self.pattern) or self.pattern[self.position] != ")":
                raise _PatternError("invalid_pattern")
            self.position += 1
            self.depth -= 1
            return value
        if char == ")" or char in "*+?":
            raise _PatternError("invalid_pattern")
        if char == ".":
            return "char", ("any", None)
        if char == "^":
            return "start", None
        if char == "$":
            return "end", None
        if char == "[":
            return "char", ("class", self._character_class())
        if char == "\\":
            return "char", self._escape()
        return "char", ("literal", char)

    def _escaped_char(self):
        if self.position >= len(self.pattern):
            raise _PatternError("invalid_pattern")
        value = self.pattern[self.position]
        self.position += 1
        if value in {"n", "r", "t"}:
            return {"n": "\n", "r": "\r", "t": "\t"}[value]
        if value.isalnum():
            raise _PatternError("unsupported_escape")
        return value

    def _escape(self):
        if self.position >= len(self.pattern):
            raise _PatternError("invalid_pattern")
        value = self.pattern[self.position]
        if value in "dDwWsS":
            self.position += 1
            return "category", value
        if value.isdigit():
            raise _PatternError("backreferences_unsupported")
        return "literal", self._escaped_char()

    def _class_char(self):
        if self.position >= len(self.pattern):
            raise _PatternError("invalid_pattern")
        if self.pattern[self.position] == "\\":
            self.position += 1
            if self.position < len(self.pattern) and self.pattern[self.position] in "dDwWsS":
                raise _PatternError("character_class_escape_unsupported")
            return self._escaped_char()
        value = self.pattern[self.position]
        self.position += 1
        return value

    def _character_class(self):
        negated = (self.position < len(self.pattern) and self.pattern[self.position] == "^")
        if negated:
            self.position += 1
        ranges = []
        while self.position < len(self.pattern) and self.pattern[self.position] != "]":
            first = self._class_char()
            if (self.position + 1 < len(self.pattern) and
                    self.pattern[self.position] == "-" and
                    self.pattern[self.position + 1] != "]"):
                self.position += 1
                last = self._class_char()
                if ord(first) > ord(last):
                    raise _PatternError("invalid_character_range")
                ranges.append((first, last))
            else:
                ranges.append((first, first))
        if self.position >= len(self.pattern) or not ranges:
            raise _PatternError("invalid_pattern")
        self.position += 1
        return negated, tuple(ranges)


class _Builder:
    def __init__(self):
        self.states = []

    def _emit(self, operation, argument=None, out=None, out2=None):
        if len(self.states) >= MAX_STATES:
            raise _PatternError("pattern_state_limit")
        self.states.append({"op": operation, "arg": argument, "out": out, "out2": out2})
        return len(self.states) - 1

    def _patch(self, outs, target):
        for index, branch in outs:
            self.states[index][branch] = target

    def compile(self, node):
        kind = node[0]
        if kind == "char":
            index = self._emit("char", node[1])
            return index, [(index, "out")]
        if kind in {"start", "end"}:
            index = self._emit(kind)
            return index, [(index, "out")]
        if kind == "seq":
            children = node[1]
            if not children:
                index = self._emit("epsilon")
                return index, [(index, "out")]
            start, outs = self.compile(children[0])
            for child in children[1:]:
                next_start, next_outs = self.compile(child)
                self._patch(outs, next_start)
                outs = next_outs
            return start, outs
        if kind == "alt":
            fragments = [self.compile(child) for child in node[1]]
            start, outs = fragments[-1]
            for next_start, next_outs in reversed(fragments[:-1]):
                split = self._emit("split", out=next_start, out2=start)
                start = split
                outs.extend(next_outs)
            return start, outs
        if kind == "*":
            inner_start, inner_outs = self.compile(node[1])
            split = self._emit("split", out=inner_start)
            self._patch(inner_outs, split)
            return split, [(split, "out2")]
        if kind == "+":
            inner_start, inner_outs = self.compile(node[1])
            split = self._emit("split", out=inner_start)
            self._patch(inner_outs, split)
            return inner_start, [(split, "out2")]
        if kind == "?":
            inner_start, inner_outs = self.compile(node[1])
            split = self._emit("split", out=inner_start)
            inner_outs.append((split, "out2"))
            return split, inner_outs
        raise _PatternError("invalid_pattern")


def _matches(spec, char, ignore_case):
    kind, value = spec
    candidate = char.lower() if ignore_case else char
    if kind == "any":
        return char != "\n"
    if kind == "literal":
        expected = value.lower() if ignore_case else value
        return candidate == expected
    if kind == "class":
        negated, ranges = value
        found = any((start.lower() if ignore_case else start) <= candidate <=
                    (end.lower() if ignore_case else end) for start, end in ranges)
        return not found if negated else found
    if kind == "category":
        digit = char.isdecimal()
        word = char == "_" or char.isalnum()
        space = char.isspace()
        return {"d": digit, "D": not digit, "w": word, "W": not word,
                "s": space, "S": not space}[value]
    return False


def _compile(pattern):
    parser = _Parser(pattern)
    tree = parser.parse()
    builder = _Builder()
    start, outs = builder.compile(tree)
    accept = builder._emit("match")
    builder._patch(outs, accept)
    return builder.states, start, parser.ignore_case


def _budget_step(budget):
    budget[0] += 1
    if budget[0] > MAX_STEPS:
        raise _StepBudget


def _closure(states, starts, position, line_length, budget):
    pending = list(starts)
    found = set()
    while pending:
        index = pending.pop()
        if index is None or index in found:
            continue
        _budget_step(budget)
        found.add(index)
        state = states[index]
        operation = state["op"]
        if operation == "epsilon":
            pending.append(state["out"])
        elif operation == "split":
            pending.extend((state["out"], state["out2"]))
        elif operation == "start" and position == 0:
            pending.append(state["out"])
        elif operation == "end" and position == line_length:
            pending.append(state["out"])
    return found


def _match_at(states, start_state, line, start, ignore_case, budget):
    active = _closure(states, [start_state], start, len(line), budget)
    latest = None
    if any(states[index]["op"] == "match" for index in active) and start < len(line):
        latest = start
    for position in range(start, len(line)):
        next_states = []
        for index in active:
            state = states[index]
            if state["op"] == "char":
                _budget_step(budget)
                if _matches(state["arg"], line[position], ignore_case):
                    next_states.append(state["out"])
        if not next_states:
            break
        active = _closure(states, next_states, position + 1, len(line), budget)
        if any(states[index]["op"] == "match" for index in active):
            latest = position + 1
    return latest


def _line_matches(states, start_state, line, ignore_case, budget, line_number, output):
    cursor = 0
    while cursor < len(line) and len(output) < MAX_MATCHES:
        found = None
        for start in range(cursor, len(line)):
            end = _match_at(states, start_state, line, start, ignore_case, budget)
            if end is not None and end > start:
                found = (start, end)
                break
        if found is None:
            return
        start, end = found
        output.append({"line": line_number, "column": start + 1,
                       "match": line[start:end][:MAX_MATCH_CHARS]})
        cursor = end


def handle(input, ctx):
    input_id = input.get("input_id")
    pattern = input.get("pattern")
    if not isinstance(input_id, str) or input_id not in ctx.inputs:
        raise ValueError("input_id is required")
    if not isinstance(pattern, str):
        return {"matches": [], "steps": 0, "step_limit": MAX_STEPS,
                "error": "invalid_pattern"}
    raw = ctx.input_bytes(input_id, limit=MAX_INPUT)
    text = raw.decode("utf-8", errors="replace")
    matches = []
    budget = [0]
    try:
        states, start_state, ignore_case = _compile(pattern)
        for line_number, line in enumerate(io.StringIO(text), 1):
            line = line.rstrip("\r\n")
            _line_matches(states, start_state, line, ignore_case, budget,
                          line_number, matches)
            if len(matches) >= MAX_MATCHES:
                break
        error = ""
    except _StepBudget:
        error = "step_budget_exceeded"
    except _PatternError as exc:
        error = str(exc)
    return {"matches": matches, "steps": min(budget[0], MAX_STEPS),
            "step_limit": MAX_STEPS, "error": error}
