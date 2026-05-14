# LLM.md — CrewAI Ollama Cloud Provider

> Documentation for AI coding assistants working on this project.

## Installation & Setup

```bash
# From PyPI
pip install crewai-ollama-cloud

# From source (development)
git clone <repo-url>
cd crewai-ollama-cloud
pip install -e ".[dev]"
```

**Requirements:** Python ≥3.10, CrewAI ≥0.80.0, httpx ≥0.25.0

Dev extras include: pytest, pytest-asyncio, ruff.

## How to Use in Code

### Basic Usage

```python
from crewai import Agent, Task, Crew
from crewai_ollama_cloud import OllamaCloudProvider

# Local Ollama
llm = OllamaCloudProvider(model="llama3.1:8b", base_url="http://localhost:11434")

# Ollama Cloud
llm = OllamaCloudProvider(
    model="deepseek-v4-flash",
    base_url="https://ollama.com",
    api_key="sk-xxxx",
    temperature=0.7,
    stream=True,
)

agent = Agent(role="Analyst", goal="Analyze", backstory="...", llm=llm)
task = Task(description="Summarize report", expected_output="Summary")
crew = Crew(agents=[agent], tasks=[task])
result = crew.kickoff()
```

### Runtime Configuration Overrides

```python
llm = OllamaCloudProvider(model="llama3.1:8b", temperature=0.3)

# Override between calls — takes effect immediately
llm.temperature = 0.9
result = llm.call("Write creatively")

llm.temperature = 0.1
result = llm.call("Be precise")
```

### Model Discovery

```python
from crewai_ollama_cloud import list_ollama_models

models = list_ollama_models("http://localhost:11434")
for m in models:
    print(f"{m.name} — {m.parameter_size} — {m.family}")

# Cloud with auth
models = list_ollama_models("https://ollama.com", api_key="sk-xxxx")
```

### Tool Calling

```python
@tool("search")
def search(query: str) -> str:
    return f"Found: {query}"

llm = OllamaCloudProvider(model="llama3.1:8b", base_url="http://localhost:11434")
agent = Agent(role="Researcher", goal="...", backstory="...", llm=llm, tools=[search])
```

### Structured Output

```python
from pydantic import BaseModel

class Summary(BaseModel):
    key_points: list[str]
    sentiment: str

llm = OllamaCloudProvider(model="llama3.1:8b", temperature=0)
result = llm.call("Analyze Q3 results", response_model=Summary)
# result.key_points, result.sentiment
```

---

## Project Overview

`crewai-ollama-cloud` is a custom CrewAI LLM provider that speaks **Ollama's native REST API**
(`POST /api/chat`). It extends `BaseLLM` from CrewAI and uses `httpx` for HTTP
calls. No OpenAI SDK, no LiteLLM, no proxy.

- **Package**: `crewai_ollama_cloud`
- **Source**: `src/crewai_ollama_cloud/`
- **Tests**: `tests/` (39 tests)
- **CrewAI version**: 1.14.4

## Module Map

```
src/crewai_ollama_cloud/
├── __init__.py          # Public: OllamaCloudProvider, list_ollama_models, OllamaModelInfo
├── provider.py          # OllamaCloudProvider(BaseLLM) — the LLM engine
└── discovery.py         # list_ollama_models(), OllamaModelInfo

tests/
├── __init__.py
└── test_provider.py     # 39 tests: init, capabilities, body, streaming, tools, discovery, async
```

## Architecture: Call Chain

```
Agent.executor
  └─ self.llm.call(messages, tools, available_functions, from_task, from_agent, ...)
       └─ OllamaCloudProvider.call()
            ├─ self._emit_call_started_event()
            ├─ self._format_messages(messages)        # string → [LLMMessage]
            ├─ self._invoke_before_llm_call_hooks()
            ├─ if stream: self._streaming_call()
            │    ├─ self._build_body(messages, tools, response_model)
            │    ├─ self._client.stream("POST", "/api/chat", json=body)
            │    ├─ NDJSON iter_lines() → content + thinking + tool_calls
            │    ├─ self._emit_stream_chunk_event() per token
            │    ├─ self._track_token_usage_internal() at done chunk
            │    └─ self._emit_call_completed_event()
            ├─ else: self._non_streaming_call()
            │    ├─ self._build_body(...)
            │    ├─ self._client.post("/api/chat", json=body)
            │    └─ self._process_response()
            └─ on error: self._emit_call_failed_event()
```

## BaseLLM Methods Used

| Method | When | Notes |
|--------|------|-------|
| `_format_messages(messages)` | Call start | Converts `str` → `[LLMMessage]` |
| `_invoke_before_llm_call_hooks(messages, from_agent)` | Pre-call | Returns `False` to block |
| `_invoke_after_llm_call_hooks(messages, text, from_agent)` | Post-response | Only when `from_agent is None` |
| `_emit_call_started_event(...)` | Call start | Observability |
| `_emit_call_completed_event(...)` | Success | Includes response, usage, call_type |
| `_emit_call_failed_event(...)` | Exception | Includes error string |
| `_emit_stream_chunk_event(...)` | Per token | `call_type=LLMCallType.LLM_CALL` |
| `_emit_thinking_chunk_event(...)` | Per thinking token | For DeepSeek-R1 etc. |
| `_track_token_usage_internal(usage)` | After response | `{prompt_tokens, completion_tokens, total_tokens}` |
| `_handle_tool_execution(...)` | Tool calls | Returns tool result string |
| `_apply_stop_words(text)` | Post-processing | Cuts at stop sequences |

## Pydantic Validators

Two-step validation in `OllamaCloudProvider`:

1. **`@model_validator(mode="before")`** `_normalize_fields`:
   - Forces `provider="ollama_cloud"`
   - Falls back to `OLLAMA_API_KEY` env var
   - Strips `/v1` suffix from `base_url`

2. **`@model_validator(mode="after")`** `_init_client`:
   - Builds `httpx.Client` with auth headers + timeout

## Ollama Native API Protocol

### Request: `POST /api/chat`

```json
{
  "model": "llama3.1:8b",
  "messages": [{"role": "user", "content": "Hello"}],
  "options": {"temperature": 0.7, "num_predict": 1024, "stop": ["END"]},
  "stream": true,
  "keep_alive": "5m",
  "think": false
}
```

### Streaming Response (NDJSON)

```json
{"model":"llama3.1:8b","message":{"role":"assistant","content":"Hello"},"done":false}
{"model":"llama3.1:8b","message":{"role":"assistant","content":" world"},"done":false}
{"model":"llama3.1:8b","message":{"role":"assistant","content":""},"done":true,"prompt_eval_count":5,"eval_count":7}
```

### Tool Call Response

```json
{"message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"get_weather","arguments":{"city":"Berlin"}}}]},"done":true}
```

## Parameter Mapping

| CrewAI field | Ollama body |
|-------------|-------------|
| `temperature` | `options.temperature` |
| `max_tokens` | `options.num_predict` |
| `top_p` | `options.top_p` |
| `top_k` | `options.top_k` |
| `stop` | `options.stop` |
| `think` | `think` (top-level) |
| `response_model` | `format` (JSON schema) |
| `keep_alive` | `keep_alive` |

## Testing Patterns

No real HTTP — all mocked via `unittest.mock.patch`:

### Mock Non-Streaming POST

```python
mock_response = MagicMock(spec=httpx.Response)
mock_response.json.return_value = {...}
mock_response.raise_for_status = MagicMock()
with patch.object(provider._client, "post", return_value=mock_response):
    result = provider.call("Hello")
```

### Mock Streaming

```python
mock_stream = MagicMock()
mock_stream.__enter__ = MagicMock(return_value=mock_stream)
mock_stream.__exit__ = MagicMock(return_value=None)
mock_stream.iter_lines.return_value = [json_line1, json_line2, ...]
mock_stream.raise_for_status = MagicMock()
with patch.object(provider._client, "stream", return_value=mock_stream):
    result = provider.call("Hello")
```

### Tool Call Event Format

CrewAI's `LLMStreamChunkEvent` requires:
```python
tool_call = {
    "index": 0,
    "function": {
        "name": "get_weather",
        "arguments": '{"city": "Berlin"}'  # JSON string, not dict
    }
}
```

## Common Pitfalls

1. **`_format_messages` expects list**, not single dict: wrap with `[...]`
2. **Ollama tool args are dict**, CrewAI event wants JSON string: `json.dumps(args)`
3. **`base_url` must not end with `/v1`** — validator strips it
4. **Empty `options` dict should be omitted** from request body
5. **Only call `_invoke_after_llm_call_hooks` when `from_agent is None`**

## Adding Features

- **New Ollama param**: Add field → update `_build_body()` → update `to_config_dict()` → add test
- **New capability**: Override method → add capability test
- **New context window**: Add to `_OLLAMA_CONTEXT_SIZES` → test

## Dependencies

Only two direct dependencies:
- **crewai** (≥0.80.0): `BaseLLM`, events, exceptions
- **httpx** (≥0.25.0): HTTP + streaming

No `openai`, no `litellm`, no `requests`.
