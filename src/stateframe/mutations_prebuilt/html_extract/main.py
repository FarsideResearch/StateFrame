"""Extract bounded readable content from a fetched HTML response."""
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit


MAX_BODY_BYTES = 10 << 20
MAX_TEXT_BYTES = 64 << 10
MAX_LINKS = 200
MAX_TABLES = 10
MAX_TABLE_ROWS = 200
MAX_TABLE_COLUMNS = 32
MAX_CELL_BYTES = 2048
MAX_TABLE_TEXT_BYTES = 256 << 10
MAX_TITLE_BYTES = 2048


def _append(parts, value, used, limit):
    if used >= limit or not value:
        return used
    encoded = value.encode("utf-8", errors="replace")
    remaining = limit - used
    if len(encoded) > remaining:
        encoded = encoded[:remaining]
        while encoded:
            try:
                value = encoded.decode("utf-8", errors="strict")
                break
            except UnicodeDecodeError:
                encoded = encoded[:-1]
        else:
            value = ""
    parts.append(value)
    return used + len(value.encode("utf-8"))


def _bounded(value, limit):
    parts = []
    _append(parts, value, 0, limit)
    return "".join(parts)


class _Extractor(HTMLParser):
    def __init__(self, base_url):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.text_parts = []
        self.text_bytes = 0
        self.title_parts = []
        self.title_bytes = 0
        self.title_depth = 0
        self.hidden_depth = 0
        self.links = []
        self.tables = []
        self.table_depth = 0
        self.active_table = None
        self.current_row = None
        self.current_cell = None
        self.table_text_bytes = 0
        self.truncated = False

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self.hidden_depth += 1
        if self.hidden_depth:
            return
        if tag == "title":
            self.title_depth += 1
        if tag == "a" and len(self.links) < MAX_LINKS:
            href = dict(attrs).get("href")
            if href:
                candidate = urljoin(self.base_url, href)
                parts = urlsplit(candidate)
                if (parts.scheme in {"http", "https"} and parts.hostname and
                        len(candidate) <= 2048):
                    self.links.append(candidate)
        if tag == "table":
            if self.table_depth == 0:
                if len(self.tables) < MAX_TABLES:
                    self.active_table = len(self.tables)
                    self.tables.append([])
                else:
                    self.active_table = None
                    self.truncated = True
            self.table_depth += 1
            return
        if self.table_depth == 1 and self.active_table is not None:
            if tag == "tr":
                rows = self.tables[self.active_table]
                if len(rows) < MAX_TABLE_ROWS:
                    self.current_row = []
                else:
                    self.current_row = None
                    self.truncated = True
            elif tag in {"td", "th"} and self.current_row is not None:
                if len(self.current_row) < MAX_TABLE_COLUMNS:
                    self.current_cell = []
                else:
                    self.current_cell = None
                    self.truncated = True

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"}:
            self.hidden_depth = max(0, self.hidden_depth - 1)
            return
        if self.hidden_depth:
            return
        if tag == "title" and self.title_depth:
            self.title_depth -= 1
        if tag in {"td", "th"} and self.current_row is not None:
            if self.current_cell is not None:
                self.current_row.append("".join(self.current_cell))
            self.current_cell = None
        if tag == "tr" and self.current_row is not None and self.active_table is not None:
            self.tables[self.active_table].append(self.current_row)
            self.current_row = None
        if tag == "table" and self.table_depth:
            self.table_depth -= 1
            if self.table_depth == 0:
                self.active_table = None
                self.current_row = None
                self.current_cell = None

    def handle_data(self, data):
        if self.hidden_depth:
            return
        self.text_bytes = _append(self.text_parts, data, self.text_bytes, MAX_TEXT_BYTES)
        if self.text_bytes >= MAX_TEXT_BYTES:
            self.truncated = True
        if self.title_depth:
            self.title_bytes = _append(self.title_parts, data, self.title_bytes,
                                       MAX_TITLE_BYTES)
        if self.current_cell is not None and self.table_text_bytes < MAX_TABLE_TEXT_BYTES:
            current = "".join(self.current_cell)
            updated = _bounded(current + data, MAX_CELL_BYTES)
            self.table_text_bytes += max(0, len(updated.encode("utf-8")) -
                                         len(current.encode("utf-8")))
            if len(updated) < len(current + data):
                self.truncated = True
            self.current_cell[:] = [updated]
        elif self.current_cell is not None:
            self.truncated = True

    def finish(self):
        visible = " ".join("".join(self.text_parts).split())
        title = " ".join("".join(self.title_parts).split())
        tables = []
        for rows in self.tables[:MAX_TABLES]:
            tables.append([row[:MAX_TABLE_COLUMNS] for row in rows[:MAX_TABLE_ROWS]])
        return {
            "title": _bounded(title, MAX_TITLE_BYTES),
            "text": _bounded(visible, MAX_TEXT_BYTES),
            "links": self.links[:MAX_LINKS],
            "tables": tables,
            "truncated": self.truncated,
        }


def _empty(error):
    return {"title": "", "text": "", "links": [], "tables": [],
            "truncated": False, "error": error}


def handle(input, ctx):
    url = input.get("url")
    if not isinstance(url, str) or not url or len(url) > 2048:
        return _empty("invalid_url")
    response = ctx.fetch(url, accept="text/html")
    if response.error is not None:
        return _empty("fetch_error")
    if response.status < 200 or response.status >= 300:
        return _empty("http_status")
    body = response.body[:MAX_BODY_BYTES]
    truncated = len(response.body) > MAX_BODY_BYTES
    content_type = response.headers.get("content-type", "")
    charset = "utf-8"
    for parameter in content_type.split(";")[1:]:
        key, _, value = parameter.strip().partition("=")
        if key.lower() == "charset" and value:
            charset = value.strip().strip('"')
    try:
        decoded = body.decode(charset, errors="replace")
    except LookupError:
        decoded = body.decode("utf-8", errors="replace")
    base_url = response.url or url
    parser = _Extractor(base_url)
    parser.feed(decoded)
    parser.close()
    result = parser.finish()
    result["truncated"] = bool(result["truncated"] or truncated)
    result["error"] = ""
    return result
