"""Frozen schemas for host-authorized execution and observation tools."""


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


def measure_schemas(custom_schemas=(), *, endpoint=None, tokenizer=None):
    """Use the request meter's tokenizer transport, recording fallback explicitly."""
    from kernel_request_diet import RequestTokenMeter
    from kernel_shell_tool import SHELL_SCHEMA
    from kernel_custom_tools import CustomToolError

    meter = RequestTokenMeter(tokenizer, endpoint=endpoint, template_overhead_tokens=0)
    result = {}
    for item in (SHELL_SCHEMA, *BUILTIN_SCHEMAS):
        count, method = meter.measure({"messages": [], "tools": [item]})
        result[item["function"]["name"]] = {"tokens": count, "method": method}
        if count > 1200:
            raise CustomToolError(f"built-in {item['function']['name']} schema exceeds 1200 tokens: {count}")
    if custom_schemas:
        count, method = meter.measure({"messages": [], "tools": list(custom_schemas)})
        result["custom_total"] = {"tokens": count, "method": method}
        if count > 1500:
            raise CustomToolError(f"custom tool schemas: {count} tokens exceeds 1500")
    return result
