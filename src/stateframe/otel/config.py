"""Bounded operator configuration for the optional host exporter."""
from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import os
from pathlib import Path
import re
import stat
import tomllib
from typing import Mapping
from urllib.parse import unquote_to_bytes, urlsplit, urlunsplit


from . import DETAIL_ENV, DETAIL_FLAGS
from .levels import Levels


DEFAULT_ENDPOINT = "http://127.0.0.1:4318"
DEFAULT_TIMEOUT_MS = 10_000
MIN_TIMEOUT_MS = 100
MAX_TIMEOUT_MS = 30_000
MAX_CONFIG_BYTES = 32 * 1024
MAX_HEADER_BYTES = 8 * 1024
MAX_HEADERS = 32
MAX_TLS_FILE_BYTES = 1024 * 1024

_TOKEN = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_BAD_PERCENT = re.compile(r"%(?![0-9A-Fa-f]{2})")
_RATIO = re.compile(r"^(?:0(?:\.[0-9]{1,9})?|1(?:\.0{1,9})?)$")
_RESERVED_HEADERS = frozenset({
    "host", "content-length", "content-type", "content-encoding", "transfer-encoding",
    "connection", "keep-alive", "upgrade", "te", "trailer", "proxy-authorization",
    "proxy-connection", "expect", "accept", "accept-encoding",
})
_GENERIC_KEYS = frozenset({
    "endpoint", "protocol", "compression", "max_export_time_milliseconds", "headers", "certificate",
    "client_certificate", "client_key",
})
_SIGNAL_KEYS = _GENERIC_KEYS | frozenset({"exporter"})
_TOML_SETTING_ALIASES = {
    "timeout_ms": "max_export_time_milliseconds",
}
_UNSUPPORTED_ENV = frozenset({
    "OTEL_RESOURCE_ATTRIBUTES", "OTEL_RESOURCE_DETECTORS", "TRACEPARENT",
    "OTEL_ATTRIBUTE_COUNT_LIMIT", "OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT",
    "OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT", "OTEL_SPAN_EVENT_COUNT_LIMIT",
    "OTEL_SPAN_LINK_COUNT_LIMIT", "OTEL_BSP_MAX_EXPORT_BATCH_SIZE",
    "OTEL_BSP_MAX_QUEUE_SIZE", "OTEL_BSP_SCHEDULE_DELAY", "OTEL_BLRP_MAX_EXPORT_BATCH_SIZE",
    "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", "OTEL_EXPORTER_OTLP_LOGS_PROTOCOL",
    "OTEL_EXPORTER_OTLP_LOGS_COMPRESSION", "OTEL_EXPORTER_OTLP_LOGS_TIMEOUT",
    "OTEL_EXPORTER_OTLP_LOGS_HEADERS", "OTEL_EXPORTER_OTLP_LOGS_CERTIFICATE",
    "OTEL_EXPORTER_OTLP_LOGS_CLIENT_CERTIFICATE", "OTEL_EXPORTER_OTLP_LOGS_CLIENT_KEY",
})


@dataclass(frozen=True, slots=True)
class Settings:
    """The exact telemetry setting surface accepted by the CLI."""

    config: Path | None = None
    traces: str | None = None
    protocol: str | None = None
    compression: str | None = None
    timeout_ms: int | None = None
    sample: str | None = None
    sample_ratio: str | None = None
    include_cost: bool | None = None


@dataclass(frozen=True, slots=True, repr=False)
class ConfigResult:
    traces: object | None
    sampler: str
    sample_ratio: float
    diagnostics: tuple[str, ...] = ()
    sdk_disabled: bool = False
    levels: Levels = Levels()
    price_rates: tuple[int | float, int | float] | None = None

    def __repr__(self) -> str:
        return (
            "ConfigResult(traces=" + ("configured" if self.traces is not None else "disabled")
            + f", sampler={self.sampler!r}, sample_ratio={self.sample_ratio!r}, diagnostics={self.diagnostics!r})"
        )


class _InvalidConfig(ValueError):
    pass


def resolve(cli: Settings, env: Mapping[str, str], credentials: tuple[str, ...] = ()) -> ConfigResult:
    """Resolve and validate settings without retaining environment or file text."""
    diagnostics: set[str] = set()
    try:
        if not isinstance(cli, Settings) or not isinstance(env, Mapping):
            raise _InvalidConfig
        for key, value in env.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise _InvalidConfig
        sdk_disabled = _env_true(env.get("OTEL_SDK_DISABLED"))
        sdk_value = env.get("OTEL_SDK_DISABLED", "")
        if sdk_value and sdk_value.strip().lower() not in {"true", "false"}:
            diagnostics.add("invalid_config")
        if sdk_disabled:
            return ConfigResult(None, "always_on", 1.0,
                                tuple(sorted(diagnostics)), sdk_disabled=True)

        config = _read_config(cli.config) if cli.config is not None else {}
        unsupported = any(env.get(name) not in (None, "") for name in _UNSUPPORTED_ENV)
        unsupported = unsupported or _unsupported_standard_settings(env)
        if unsupported:
            diagnostics.add("unsupported_setting")

        levels, price_rates = resolve_detail(cli, env, config.get("detail", {}))
        sampler, ratio = _resolve_sampling(cli, env, config, diagnostics)

        trace_selector = _select_selector(cli.traces, env.get("OTEL_TRACES_EXPORTER"),
                                          config.get("traces", {}).get("exporter"))
        if trace_selector == "invalid":
            diagnostics.add("unsupported_setting")
        if env.get("OTEL_LOGS_EXPORTER") not in (None, "", "none"):
            diagnostics.add("unsupported_setting")

        traces = _transport_for(cli, env, config, diagnostics) if trace_selector == "otlp" else None
        if traces is not None:
            from .authority import has_application_alias
            if has_application_alias(env, traces, credentials):
                raise _InvalidConfig
        return ConfigResult(traces, sampler, ratio, tuple(sorted(diagnostics)),
                            levels=levels, price_rates=price_rates)
    except (_InvalidConfig, OSError, UnicodeError, tomllib.TOMLDecodeError, RecursionError, TypeError, ValueError):
        return ConfigResult(None, "always_on", 1.0,
                            tuple(sorted(diagnostics | {"invalid_config"})))


def _transport_for(cli: Settings, env: Mapping[str, str],
                   config: dict[str, object], diagnostics: set[str]):
    generic = config.get("otlp", {})
    specific = config.get("traces", {})
    if not isinstance(generic, dict) or not isinstance(specific, dict):
        raise _InvalidConfig
    endpoint, signal_specific = next(
        (candidate, is_specific) for candidate, is_specific in (
            (env.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"), True),
            (env.get("OTEL_EXPORTER_OTLP_ENDPOINT"), False),
            (specific.get("endpoint"), True), (generic.get("endpoint"), False),
            (DEFAULT_ENDPOINT, False),
        ) if candidate not in (None, "")
    )
    endpoint = _normalize_endpoint(endpoint, signal_specific=signal_specific)
    protocol = _choose(cli.protocol,
        env.get("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"),
        env.get("OTEL_EXPORTER_OTLP_PROTOCOL"), specific.get("protocol"), generic.get("protocol"),
        "http/protobuf")
    compression = _choose(cli.compression,
        env.get("OTEL_EXPORTER_OTLP_TRACES_COMPRESSION"),
        env.get("OTEL_EXPORTER_OTLP_COMPRESSION"), specific.get("compression"), generic.get("compression"),
        "none")
    timeout = _choose(cli.timeout_ms,
        env.get("OTEL_EXPORTER_OTLP_TRACES_TIMEOUT"),
        env.get("OTEL_EXPORTER_OTLP_TIMEOUT"),
        specific.get("max_export_time_milliseconds"), generic.get("max_export_time_milliseconds"), DEFAULT_TIMEOUT_MS)

    if not isinstance(endpoint, str):
        raise _InvalidConfig
    if protocol not in {"http/protobuf", "http/json"}:
        diagnostics.add("unsupported_setting")
        raise _InvalidConfig
    if compression not in {"none", "gzip"}:
        raise _InvalidConfig
    if isinstance(timeout, str):
        if cli.timeout_ms is not None:
            raise _InvalidConfig
        timeout = _parse_ascii_int(timeout)
    timeout = _strict_int(timeout, MIN_TIMEOUT_MS, MAX_TIMEOUT_MS)

    headers_raw = _choose(None,
        env.get("OTEL_EXPORTER_OTLP_TRACES_HEADERS"),
        env.get("OTEL_EXPORTER_OTLP_HEADERS"),
        specific.get("headers"), generic.get("headers"), "")
    headers = _parse_headers(headers_raw)
    parts = urlsplit(endpoint)
    if parts.scheme == "http" and headers:
        raise _InvalidConfig

    ca = _choose(None,
        env.get("OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE"),
        env.get("OTEL_EXPORTER_OTLP_CERTIFICATE"),
        specific.get("certificate"), generic.get("certificate"), None)
    cert = _choose(None,
        env.get("OTEL_EXPORTER_OTLP_TRACES_CLIENT_CERTIFICATE"),
        env.get("OTEL_EXPORTER_OTLP_CLIENT_CERTIFICATE"),
        specific.get("client_certificate"), generic.get("client_certificate"), None)
    key_file = _choose(None,
        env.get("OTEL_EXPORTER_OTLP_TRACES_CLIENT_KEY"),
        env.get("OTEL_EXPORTER_OTLP_CLIENT_KEY"),
        specific.get("client_key"), generic.get("client_key"), None)
    ca_path = _tls_file(ca, private=False) if ca is not None else None
    cert_path = _tls_file(cert, private=False) if cert is not None else None
    key_path = _tls_file(key_file, private=True) if key_file is not None else None
    if (cert_path is None) != (key_path is None) or ((ca_path or cert_path or key_path) and parts.scheme != "https"):
        raise _InvalidConfig

    try:
        from .transport import TransportSettings
    except ImportError as exc:
        diagnostics.add("unsupported_setting")
        raise _InvalidConfig from exc
    return TransportSettings(endpoint, protocol, headers, compression, timeout, ca_path, cert_path, key_path)


def _read_config(path: Path) -> dict[str, object]:
    if not isinstance(path, Path) or not path.is_absolute():
        raise _InvalidConfig
    try:
        original = path.lstat()
    except OSError as exc:
        raise _InvalidConfig from exc
    resolved = path.resolve(strict=True)
    if path != resolved or not stat.S_ISREG(original.st_mode) or not _safe_parent_chain(path):
        raise _InvalidConfig
    repository = Path(__file__).resolve().parents[3]
    if _is_relative_to(resolved, repository):
        raise _InvalidConfig
    info = resolved.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
        raise _InvalidConfig
    if info.st_mode & 0o077 or info.st_size > MAX_CONFIG_BYTES:
        raise _InvalidConfig
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(resolved, flags)
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino, opened.st_uid, opened.st_nlink) != (
            info.st_dev, info.st_ino, info.st_uid, info.st_nlink
        ) or not stat.S_ISREG(opened.st_mode):
            raise _InvalidConfig
        chunks: list[bytes] = []
        remaining = MAX_CONFIG_BYTES + 1
        while remaining:
            block = os.read(fd, min(remaining, 4096))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        raw = b"".join(chunks)
        if len(raw) > MAX_CONFIG_BYTES:
            raise _InvalidConfig
    finally:
        os.close(fd)
    document = tomllib.loads(raw.decode("utf-8", "strict"))
    if type(document) is not dict or set(document) - {"otlp", "traces", "sampling", "detail"}:
        raise _InvalidConfig
    for name, value in document.items():
        if type(value) is not dict:
            raise _InvalidConfig
        allowed = _GENERIC_KEYS if name == "otlp" else _SIGNAL_KEYS
        if name == "detail":
            if set(value) - {"include_cost", "price_per_million_input_tokens", "price_per_million_output_tokens"}:
                raise _InvalidConfig
            continue
        if name == "sampling":
            allowed = frozenset({"sampler", "ratio"})
        for alias, key in _TOML_SETTING_ALIASES.items():
            if alias in value and key in allowed:
                if key in value:
                    raise _InvalidConfig
                value[key] = value.pop(alias)
        if set(value) - allowed:
            raise _InvalidConfig
        for key, item in value.items():
            if key == "max_export_time_milliseconds":
                if type(item) is not int:
                    raise _InvalidConfig
            elif not isinstance(item, str):
                raise _InvalidConfig
    return document


def resolve_detail(cli: Settings, env: Mapping[str, str], table: dict) -> tuple[Levels, tuple | None]:
    flags = {}
    for name, env_name in DETAIL_ENV.items():
        value = getattr(cli, name)
        if value is None:
            ambient = env.get(env_name)
            if ambient not in (None, ""):
                if ambient.lower() not in {"true", "false"}:
                    raise _InvalidConfig
                value = ambient.lower() == "true"
            else:
                value = table.get(name, False)
        if type(value) is not bool:
            raise _InvalidConfig
        flags[name] = value
    rates = None
    if any(name in table for name in ("price_per_million_input_tokens", "price_per_million_output_tokens")):
        from math import isfinite
        first, last = table.get("price_per_million_input_tokens"), table.get("price_per_million_output_tokens")
        if any(type(rate) not in {int, float} or not 0 <= rate <= 1e12 or not isfinite(rate)
               for rate in (first, last)):
            raise _InvalidConfig
        rates = (first, last)
    return Levels(**flags), rates


def _choose(cli_value, env_specific, env_generic, config_specific, config_generic, default):
    for candidate in (cli_value, env_specific, env_generic, config_specific, config_generic):
        if candidate is not None and candidate != "":
            return candidate
    return default


def _select_selector(cli_value, env_value, config_value) -> str:
    value = _choose(cli_value, env_value, None, config_value, None, "otlp")
    return value if value in {"otlp", "none"} else "invalid"


def _normalize_endpoint(value: str, *, signal_specific: bool) -> str:
    if len(value) > 2048 or any(ord(char) <= 32 or ord(char) == 127 for char in value):
        raise _InvalidConfig
    if _BAD_PERCENT.search(value):
        raise _InvalidConfig
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError as exc:
        raise _InvalidConfig from exc
    scheme = parts.scheme.lower()
    if (scheme not in {"http", "https"} or not parts.hostname
            or parts.username is not None or parts.password is not None or "@" in parts.netloc
            or "?" in value or "#" in value):
        raise _InvalidConfig
    if not parts.netloc or port is not None and not 1 <= port <= 65535:
        raise _InvalidConfig
    if not parts.hostname.isascii() or "%" in parts.hostname:
        raise _InvalidConfig
    host = parts.hostname.lower().rstrip(".")
    if host in {"metadata.google.internal", "instance-data"}:
        raise _InvalidConfig
    address = None
    effective = None
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if host.replace(".", "").isdigit():
            raise _InvalidConfig
    else:
        effective = address.ipv4_mapped if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped else address
        if effective.is_link_local or effective.is_unspecified or effective.is_multicast or effective == ipaddress.ip_address("255.255.255.255"):
            raise _InvalidConfig
        if str(effective) in {"169.254.169.254", "100.100.100.200"}:
            raise _InvalidConfig
    if scheme == "http" and (effective is None or not effective.is_loopback):
        raise _InvalidConfig
    path = parts.path
    if signal_specific:
        path = path or "/"
    else:
        path = path.rstrip("/") + "/v1/traces"
    return urlunsplit((scheme, parts.netloc, path, "", ""))


def _parse_headers(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, str):
        raise _InvalidConfig
    if len(value.encode("utf-8")) > MAX_HEADER_BYTES:
        raise _InvalidConfig
    if value == "":
        return ()
    pairs = value.split(",")
    if len(pairs) > MAX_HEADERS:
        raise _InvalidConfig
    result: list[tuple[str, str]] = []
    names: set[str] = set()
    total = 0
    for pair in pairs:
        if "=" not in pair:
            raise _InvalidConfig
        name, raw = pair.split("=", 1)
        if not name or not _TOKEN.fullmatch(name) or name.lower() in _RESERVED_HEADERS:
            raise _InvalidConfig
        normalized = name.lower()
        if normalized in names:
            raise _InvalidConfig
        names.add(normalized)
        if _BAD_PERCENT.search(raw):
            raise _InvalidConfig
        try:
            value_bytes = unquote_to_bytes(raw)
            decoded = value_bytes.decode("ascii")
        except (UnicodeError, ValueError) as exc:
            raise _InvalidConfig from exc
        if any((ord(char) < 32 and char != " ") or ord(char) > 126 for char in decoded):
            raise _InvalidConfig
        total += len(name.encode("ascii")) + len(decoded.encode("ascii"))
        if total > MAX_HEADER_BYTES:
            raise _InvalidConfig
        result.append((name, decoded))
    return tuple(result)


def _tls_file(value: object, *, private: bool) -> Path:
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise _InvalidConfig
    path = Path(value)
    if not path.is_absolute():
        raise _InvalidConfig
    try:
        info = path.lstat()
    except OSError as exc:
        raise _InvalidConfig from exc
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1
            or info.st_size > MAX_TLS_FILE_BYTES or stat.S_IMODE(info.st_mode) & 0o077):
        raise _InvalidConfig
    if private and stat.S_IMODE(info.st_mode) != 0o600:
        raise _InvalidConfig
    resolved = path.resolve(strict=True)
    if resolved != path or not _safe_parent_chain(path):
        raise _InvalidConfig
    return resolved


def _safe_parent_chain(path: Path) -> bool:
    from .authority import unsafe_parent_mode

    current = path.parent
    while True:
        try:
            info = current.lstat()
        except OSError:
            return False
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            return False
        if unsafe_parent_mode(current, info):
            return False
        if current.parent == current:
            return True
        current = current.parent


def _resolve_sampling(cli: Settings, env: Mapping[str, str], config: dict[str, object],
                      diagnostics: set[str]) -> tuple[str, float]:
    sampling = config.get("sampling", {})
    if not isinstance(sampling, dict):
        raise _InvalidConfig
    sampler = _choose(cli.sample, env.get("OTEL_TRACES_SAMPLER"), None,
                      sampling.get("sampler"), None, "always_on")
    ratio = _choose(cli.sample_ratio, env.get("OTEL_TRACES_SAMPLER_ARG"), None,
                    sampling.get("ratio"), None, "1")
    if sampler in {"parentbased_always_on", "parentbased_always_off", "parentbased_traceidratio"}:
        diagnostics.add("unsupported_setting")
        return "always_off", 0.0
    if sampler not in {"always_on", "always_off", "traceidratio"}:
        diagnostics.add("unsupported_setting")
        return "always_off", 0.0
    if sampler == "always_off":
        return sampler, 0.0
    if sampler == "always_on":
        return sampler, 1.0
    if not isinstance(ratio, str) or not _RATIO.fullmatch(ratio):
        diagnostics.add("invalid_config")
        return "always_on", 1.0
    return sampler, float(ratio)


def _unsupported_standard_settings(env: Mapping[str, str]) -> bool:
    if env.get("OTEL_SERVICE_NAME") not in (None, "", "stateframe"):
        return True
    if env.get("OTEL_PROPAGATORS") not in (None, "", "none"):
        return True
    return False


def _env_true(value: str | None) -> bool:
    return isinstance(value, str) and value.lower() == "true"


def _parse_ascii_int(value: object) -> int:
    if not isinstance(value, str) or not value.isascii() or not value.isdigit():
        raise _InvalidConfig
    return int(value, 10)


def _strict_int(value: object, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise _InvalidConfig
    return value


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False
