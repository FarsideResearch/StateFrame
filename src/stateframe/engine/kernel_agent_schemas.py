"""Frozen schemas for host-authorized execution and observation tools."""

from copy import deepcopy

try:
    from kernel_reason_log import emit_reason
except ImportError:
    from .kernel_reason_log import emit_reason


class SchemaMeteringTimeout(TimeoutError):
    """Startup schema tokenization exhausted its attempts on a timeout."""


def _log_schema_fallback(what, method, tokens, *, configured):
    """Byte-measured schemas with no tokenizer configured are by design (INFO); a configured
    tokenizer that failed is a degrade (WARN). journal=False: request path, no journal event."""
    emit_reason("fallback", "schema_measured_configured" if configured else "schema_measured_unconfigured",
                "agent_schemas", what, f"method {method}", level="WARN" if configured else "INFO",
                journal=False, tokens=tokens)


def schema(name, description, properties, required):
    return {"type": "function", "function": {"name": name, "description": description,
        "parameters": {"type": "object", "properties": {**properties, "why": {"type": "string"}},
                       "required": required, "additionalProperties": False}}}


JOB_NAMES = frozenset({"job_wait", "job_output", "job_stop"})
OBSERVATION_NAMES = JOB_NAMES | {"git", "web_fetch", "skill_load"}
BUILTIN_SCHEMAS = (
    schema("skill_load", "Load one selected skill's instructions into the current user facts. "
           "At most two bodies remain resident; loading another evicts the oldest.", {
        "name": {"type": "string"}, "why": {"type": "string"}}, ["name"]),
    schema("job_wait", "Wait for exit, literal output, or a reachable loopback endpoint. Not check evidence.", {
        "job": {"type": "string"}, "until": {"type": "string", "enum": ["exit", "output", "endpoint"]},
        "text": {"type": "string", "maxLength": 65536},
        "port": {"type": "integer", "minimum": 1, "maximum": 65535},
        "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 600}}, ["job", "until"]),
    schema("job_output", "Read bounded retained output from a job.", {
        "job": {"type": "string"}, "tail_lines": {"type": "integer", "minimum": 1, "maximum": 200}}, ["job"]),
    schema("job_stop", "Stop a job and its supervised descendants.", {"job": {"type": "string"}}, ["job"]),
    schema("git", "Read sanitized source history or changes in this run. No source-read credit.", {
        "action": {"type": "string", "enum": ["status", "diff", "log", "show", "blame"]},
        "against": {"type": "string", "enum": ["baseline", "HEAD"]}, "path": {"type": "string"},
        "rev": {"type": "string"}, "max_count": {"type": "integer", "minimum": 1, "maximum": 50},
        "start_line": {"type": "integer", "minimum": 1}, "end_line": {"type": "integer", "minimum": 1}}, ["action"]),
    schema("web_fetch", "Fetch bounded public text as untrusted external content.", {
        "url": {"type": "string"}, "max_chars": {"type": "integer", "minimum": 1, "maximum": 12000}}, ["url"]),
)


def measure_schemas(custom_schemas=(), *, mcp_schemas=(), endpoint=None, tokenizer=None,
                    auth_key=None, generation_endpoint=None):
    """Use the request meter's tokenizer transport, recording fallback explicitly."""
    return prepare_schema_measurement(
        custom_schemas, mcp_schemas=mcp_schemas, endpoint=endpoint, tokenizer=tokenizer,
        auth_key=auth_key, generation_endpoint=generation_endpoint,
    )()


def prepare_schema_measurement(custom_schemas=(), *, mcp_schemas=(), endpoint=None, tokenizer=None,
                               auth_key=None, generation_endpoint=None, deadline=None):
    """Capture the schema inputs and bound meter before startup configures the live menu."""
    from kernel_request_diet import RequestTokenMeter
    from kernel_shell_tool import SHELL_SCHEMA

    meter = RequestTokenMeter(
        tokenizer, endpoint=endpoint, template_overhead_tokens=0,
        auth_key=auth_key, generation_endpoint=generation_endpoint,
    )

    # Capture both bound methods before the live request meter gains its warning wrapper.
    measure = meter.measure
    live_count = estimate_count = measure
    if deadline is not None:
        meter.bind_request("startup-schema-budget", "startup-schema-budget", deadline=deadline)
        estimate_count = RequestTokenMeter(template_overhead_tokens=0).measure

        def live_count(body):
            count, method = measure(body)
            # Cold provider probes have shorter deadlines than the bind. Their
            # retries can exhaust before the absolute deadline expires.
            if method == "calibrated_bytes" and meter.unavailable_reason() == "timeout":
                raise SchemaMeteringTimeout("tokenizer timed out in startup schema metering")
            return count, method

    builtin_schemas = deepcopy((SHELL_SCHEMA, *BUILTIN_SCHEMAS))
    additional_schemas = deepcopy([*custom_schemas, *mcp_schemas])
    configured = endpoint is not None or tokenizer is not None

    def measure_schemas_once(*, estimate=False):
        count = estimate_count if estimate else live_count
        return _measure_schemas(builtin_schemas, additional_schemas, count, configured=configured)

    return measure_schemas_once


def _measure_schemas(builtin_schemas, additional_schemas, measure, *, configured):
    from kernel_custom_tools import CustomToolError

    result = {}
    for item in builtin_schemas:
        count, method = measure({"messages": [], "tools": [item]})
        result[item["function"]["name"]] = {"tokens": count, "method": method}
        if method != "tokenize":
            _log_schema_fallback(f"schema {item['function']['name']} measured without tokenizer",
                                 method, count, configured=configured)
        if count > 1200:
            raise CustomToolError(f"built-in {item['function']['name']} schema exceeds 1200 tokens: {count}")
    if additional_schemas:
        count, method = measure({"messages": [], "tools": additional_schemas})
        result["custom_total"] = {"tokens": count, "method": method}
        if method != "tokenize":
            _log_schema_fallback("custom tool schemas measured without tokenizer", method, count,
                                 configured=configured)
        if count > 1500:
            raise CustomToolError(f"custom tool schemas: {count} tokens exceeds 1500")
    return result
