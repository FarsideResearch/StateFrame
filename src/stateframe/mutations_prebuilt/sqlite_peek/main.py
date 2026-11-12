"""Read a bounded sample from a main-file-only SQLite snapshot."""
import sqlite3
from urllib.parse import quote


MAX_TABLES = 50
MAX_PROGRESS_CALLBACKS = 100
MAX_COLUMNS = 100
MAX_ROWS = 20


def _cell(value):
    if value is None:
        return "null"
    if isinstance(value, bytes):
        return f"<blob {len(value)} bytes>"
    return str(value)[:2048]


def _quote_identifier(value):
    return '"' + value.replace('"', '""') + '"'


def handle(input, ctx):
    input_id = input.get("input_id")
    if not isinstance(input_id, str) or input_id not in ctx.inputs:
        raise ValueError("input_id is required")
    path = ctx.inputs[input_id]
    supplied_wal = input.get("wal_present")
    wal_present = (supplied_wal if isinstance(supplied_wal, bool) else
                   path.with_name(path.name + "-wal").exists())
    uri = "file:" + quote(path.resolve().as_posix(), safe="/") + "?mode=ro&immutable=1"
    connection = None
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=0.2)
        connection.execute("PRAGMA query_only=ON")
        progress = {"callbacks": 0}

        def progress_handler():
            progress["callbacks"] += 1
            return int(progress["callbacks"] > MAX_PROGRESS_CALLBACKS)

        connection.set_progress_handler(progress_handler, 1000)
        readable = {"sqlite_master", "sqlite_schema"}

        def authorizer(action, first, _second, _database, _trigger):
            if action == sqlite3.SQLITE_SELECT:
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_READ and first in readable:
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_FUNCTION:
                return sqlite3.SQLITE_OK
            if hasattr(sqlite3, "SQLITE_RECURSIVE") and action == sqlite3.SQLITE_RECURSIVE:
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_READ and first in tables_for_authorizer:
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY

        tables_for_authorizer = set()
        connection.set_authorizer(authorizer)
        cursor = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name LIMIT ?", (MAX_TABLES,))
        names = [row[0] for row in cursor.fetchall()]
        tables_for_authorizer.update(names)
        tables = []
        for name in names:
            query = "SELECT * FROM " + _quote_identifier(name) + f" LIMIT {MAX_ROWS}"
            rows_cursor = connection.execute(query)
            columns = [description[0] for description in rows_cursor.description or ()]
            rows = [[_cell(value) for value in row[:MAX_COLUMNS]]
                    for row in rows_cursor.fetchmany(MAX_ROWS)]
            tables.append({"name": name, "columns": columns[:MAX_COLUMNS], "rows": rows})
        return {"tables": tables, "wal_present": wal_present, "error": ""}
    except (sqlite3.DatabaseError, OSError, ValueError):
        return {"tables": [], "wal_present": wal_present, "error": "database_error"}
    finally:
        if connection is not None:
            connection.close()
