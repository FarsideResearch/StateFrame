"""Small bounded previews for common structured text formats."""
import configparser
import csv
import io
import json
import tomllib


MAX_INPUT = 4 << 20
MAX_ROWS = 20
MAX_SCAN_ROWS = 10000
MAX_COLUMNS = 100
MAX_CELL_CHARS = 2048


def _cell(value):
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return text[:MAX_CELL_CHARS]


def _empty(format_name, error="", line=0, column=0, truncated=False):
    return {"format": format_name, "columns": [], "rows": [], "row_count": 0,
            "top_level_type": "", "error": error, "error_line": line,
            "error_column": column, "truncated": truncated}


def _json_rows(value):
    if isinstance(value, dict):
        columns = list(value)[:MAX_COLUMNS]
        return "object", columns, [[_cell(value[key]) for key in columns]]
    if isinstance(value, list):
        if value and all(isinstance(row, dict) for row in value[:MAX_ROWS]):
            columns = []
            for row in value[:MAX_ROWS]:
                for key in row:
                    if key not in columns and len(columns) < MAX_COLUMNS:
                        columns.append(str(key))
            rows = [[_cell(row.get(key)) for key in columns] for row in value[:MAX_ROWS]]
            return "array", columns, rows
        return "array", [], [[_cell(row)] for row in value[:MAX_ROWS]]
    return type(value).__name__, [], [[_cell(value)]]


def handle(input, ctx):
    input_id = input.get("input_id")
    format_name = input.get("format")
    if not isinstance(input_id, str) or input_id not in ctx.inputs:
        raise ValueError("input_id is required")
    if format_name not in {"csv", "tsv", "json", "jsonl", "toml", "ini"}:
        return _empty(str(format_name or ""), "unsupported_format")
    raw = ctx.input_bytes(input_id, limit=MAX_INPUT + 1)
    truncated = len(raw) > MAX_INPUT
    raw = raw[:MAX_INPUT]
    try:
        text = raw.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        result = _empty(format_name, "invalid_encoding", raw[:exc.start].count(b"\n") + 1,
                        exc.start - raw.rfind(b"\n", 0, exc.start))
        result["truncated"] = truncated
        return result

    try:
        if format_name in {"csv", "tsv"}:
            delimiter = "\t" if format_name == "tsv" else ","
            reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
            first = next(reader, None)
            if first is None:
                result = _empty(format_name, truncated=truncated)
                result["top_level_type"] = "table"
                return result
            columns = [_cell(value) for value in first[:MAX_COLUMNS]]
            rows = []
            row_count = 0
            for row in reader:
                row_count += 1
                if row_count <= MAX_ROWS:
                    rows.append([_cell(value) for value in row[:MAX_COLUMNS]])
                if row_count >= MAX_SCAN_ROWS:
                    truncated = True
                    break
            return {"format": format_name, "columns": columns, "rows": rows,
                    "row_count": row_count, "top_level_type": "table", "error": "",
                    "error_line": 0, "error_column": 0, "truncated": truncated}
        if format_name == "json":
            value = json.loads(text)
            top_level_type, columns, rows = _json_rows(value)
            return {"format": format_name, "columns": columns, "rows": rows,
                    "row_count": len(value) if isinstance(value, list) else len(rows),
                    "top_level_type": top_level_type, "error": "", "error_line": 0,
                    "error_column": 0, "truncated": truncated or
                    (isinstance(value, list) and len(value) > MAX_ROWS)}
        if format_name == "jsonl":
            rows = []
            row_count = 0
            for line_number, line in enumerate(io.StringIO(text), 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    return {"format": format_name, "columns": [], "rows": rows,
                            "row_count": row_count, "top_level_type": "jsonl",
                            "error": "invalid_json", "error_line": line_number,
                            "error_column": exc.colno, "truncated": truncated}
                row_count += 1
                if row_count <= MAX_ROWS:
                    rows.append([_cell(value)])
                if row_count >= MAX_SCAN_ROWS:
                    truncated = True
                    break
            return {"format": format_name, "columns": [], "rows": rows,
                    "row_count": row_count, "top_level_type": "jsonl", "error": "",
                    "error_line": 0, "error_column": 0, "truncated": truncated}
        if format_name == "toml":
            value = tomllib.loads(text)
            top_level_type, columns, rows = _json_rows(value)
            return {"format": format_name, "columns": columns, "rows": rows,
                    "row_count": len(rows), "top_level_type": top_level_type,
                    "error": "", "error_line": 0, "error_column": 0,
                    "truncated": truncated}
        parser = configparser.ConfigParser(interpolation=None, strict=True)
        parser.read_string(text)
        columns = []
        rows = []
        for section in parser.sections():
            for key, value in parser.items(section):
                if len(rows) >= MAX_ROWS:
                    truncated = True
                    break
                rows.append([_cell(section), _cell(key), _cell(value)])
            if len(rows) >= MAX_ROWS:
                break
        return {"format": format_name, "columns": ["section", "key", "value"],
                "rows": rows, "row_count": len(rows), "top_level_type": "ini",
                "error": "", "error_line": 0, "error_column": 0,
                "truncated": truncated}
    except json.JSONDecodeError as exc:
        return _empty(format_name, "invalid_json", exc.lineno, exc.colno, truncated)
    except (csv.Error, configparser.Error, tomllib.TOMLDecodeError, RecursionError, ValueError):
        return _empty(format_name, "parse_error", truncated=truncated)
