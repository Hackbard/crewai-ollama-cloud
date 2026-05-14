# CrewAI Ollama Cloud Provider

[![CI](https://github.com/Hackbard/crewai-ollama-cloud/actions/workflows/ci.yml/badge.svg)](https://github.com/Hackbard/crewai-ollama-cloud/actions)
[![Ruff](https://img.shields.io/badge/ruff-clean-brightgreen)](https://github.com/astral-sh/ruff)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![CrewAI](https://img.shields.io/badge/crewai-1.14.4-orange)](https://crewai.com)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

A custom [CrewAI](https://crewai.com) LLM provider that speaks **native Ollama protocol** —
`POST /api/chat` with NDJSON streaming. No OpenAI shim, no LiteLLM, no proxy needed.
Works with local Ollama, self-hosted instances, and [ollama.com](https://ollama.com) Cloud API.

## Why?

CrewAI's built-in Ollama support routes through the OpenAI-compatible shim
(`/v1/chat/completions`). This provider talks the **real Ollama protocol** —
`/api/chat` with native JSON, NDJSON streaming, and Ollama's native tool
calling and thinking formats.

If you're running Ollama Cloud models (``gpt-oss:120b-cloud``, ``kimi-k2.6-cloud``,
etc.) or just want direct API access without translation layers, this is for you.

## Features

| Feature | Support |
|---------|---------|
| **Native `/api/chat`** | ✅ real Ollama protocol, not OpenAI-compatible |
| **NDJSON streaming** | ✅ token-by-token, thinking/reasoning tokens |
| **Tool calling** | ✅ native Ollama tool calls (v0.3+) |
| **Structured output** | ✅ JSON schema via `format` parameter |
| **Thinking models** | ✅ `think` parameter for DeepSeek-R1, Kimi, etc. |
| **Cloud auth** | ✅ `Authorization: Bearer` for ollama.com |
| **Model discovery** | ✅ `list_ollama_models()` |
| **Config overrides** | ✅ runtime temperature, max_tokens, etc. |
| **Context windows** | ✅ auto-detection for popular models |
| **Stop words** | ✅ `options.stop` |
| **Keep alive** | ✅ `keep_alive` parameter |
| **Multimodal** | ✅ image support for vision models |
| **CrewAI events** | ✅ full observability integration |

## Installation

```bash
pip install crewai-ollama-cloud
```

Requires: Python ≥3.10, CrewAI ≥0.80.0, httpx ≥0.25.0

### Environment Setup

```bash
# Optional: set your Ollama Cloud API key
export OLLAMA_API_KEY="sk-xxxx"
```

For local Ollama, no API key is needed.

## Quick Start

```python
from crewai import Agent, Task, Crew
from crewai_ollama_cloud import OllamaCloudProvider

# Ollama Cloud
llm = OllamaCloudProvider(
    model="deepseek-v4-flash",
    base_url="https://ollama.com",
    api_key="sk-xxxx",  # or set OLLAMA_API_KEY env var
    temperature=0.7,
    stream=True,
)

# Or local Ollama
# llm = OllamaCloudProvider(model="llama3.1:8b", base_url="http://localhost:11434")

agent = Agent(role="Analyst", goal="Analyze data", backstory="Expert", llm=llm)
task = Task(description="Summarize Q1 report", expected_output="Summary")
crew = Crew(agents=[agent], tasks=[task])

result = crew.kickoff()
print(result)
```

## Configuration Reference

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | *(required)* | Ollama model name (e.g. `"llama3.1:8b"`, `"deepseek-v4-flash"`) |
| `base_url` | `str` | `"http://localhost:11434"` | Ollama host URL (no trailing `/v1`) |
| `api_key` | `str` or `None` | env `OLLAMA_API_KEY` | API key for cloud instances |
| `temperature` | `float` or `None` | `None` | Sampling temperature (0–2) |
| `max_tokens` | `int` or `None` | `None` | Max tokens to generate |
| `top_p` | `float` or `None` | `None` | Nucleus sampling |
| `top_k` | `int` or `None` | `None` | Top-k sampling |
| `stop` | `list[str]` | `[]` | Stop sequences |
| `stream` | `bool` | `False` | Enable NDJSON streaming |
| `timeout` | `float` | `120.0` | HTTP timeout in seconds |
| `keep_alive` | `str` | `"5m"` | Model keep-alive duration |
| `think` | `bool` | `False` | Enable thinking/reasoning tokens |
| `additional_params` | `dict` | `{}` | Extra parameters merged into request body |

### Ollama Parameter Mapping

When calling the API, CrewAI parameters are mapped to Ollama's native format:

| CrewAI field | Ollama request field |
|-------------|---------------------|
| `temperature` | `options.temperature` |
| `max_tokens` | `options.num_predict` |
| `top_p` | `options.top_p` |
| `top_k` | `options.top_k` |
| `stop` | `options.stop` |
| `think` | `think` (top-level) |
| `response_model` | `format` (JSON schema) |
| `keep_alive` | `keep_alive` (top-level) |

### Runtime Overrides

All configuration fields can be changed at runtime between calls:

```python
llm = OllamaCloudProvider(model="llama3.1:8b", temperature=0.3)

# Warm up: creative mode
llm.temperature = 0.9
result = llm.call("Write a poem")

# Switch to precise mode for next call
llm.temperature = 0.1
llm.top_p = 0.95
result = llm.call("Calculate 2+2")
```

## Model Discovery

```python
from crewai_ollama_cloud import list_ollama_models, OllamaModelInfo

# List models on a local GPU rig
models = list_ollama_models("http://localhost:11434")

# List cloud models
models = list_ollama_models("https://ollama.com", api_key="sk-xxxx")

for m in models:
    print(f"{m.name:35s} | {m.parameter_size:6s} | {m.family:10s} | {m.size_gb:5.1f} GB")
# Output:
# llama3.1:8b                         | 8b     | llama      |  4.7 GB
# mistral:7b                          | 7b     | mistral    |  4.1 GB
# deepseek-v4-flash                   | 70b    | deepseek   | 40.5 GB
```

The `OllamaModelInfo` object contains:

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Full model name |
| `digest` | `str` | SHA256 digest |
| `size` | `int` | Size in bytes |
| `modified_at` | `str` or `None` | Last modified timestamp |
| `family` | `str` | Inferred model family |
| `parameter_size` | `str` | Parameter count (e.g. `"8b"`, `"70b"`) |
| `size_gb` | `float` | Size in gigabytes |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OLLAMA_API_KEY` | API key for authenticated Ollama instances (e.g. cloud) |

## Stream Output

When `stream=True`, the provider uses Ollama's native NDJSON streaming.
Tokens are emitted via CrewAI's `LLMStreamChunkEvent`:

```python
llm = OllamaCloudProvider(model="llama3.1:8b", stream=True)

# Each token triggers a stream chunk event
result = llm.call("Tell me about black holes")
# Events:
#   chunk: "Black"
#   chunk: " holes"
#   chunk: " are"
#   ...
```

For thinking models (`think=True`, like `deepseek-r1`), reasoning tokens
are separated from final output and emitted as thinking chunk events.

## Tool Calling

Ollama v0.3+ supports native tool calling. The provider converts CrewAI
`BaseTool` objects to Ollama's native tool format:

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get weather for a city",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {"type": "string", "description": "City name"}
      },
      "required": ["city"]
    }
  }
}
```

Tool execution results are returned directly.

## Structured Output

To get JSON responses, use `response_model`:

```python
from pydantic import BaseModel

class Summary(BaseModel):
    key_points: list[str]
    sentiment: str

llm = OllamaCloudProvider(model="llama3.1:8b", temperature=0)
result = llm.call("Analyze Q3 results", response_model=Summary)
# result.key_points = ["Revenue up 15%", ...]
# result.sentiment = "positive"
```

## Context Windows

The provider auto-detects context window sizes for known models:

| Model | Context Size |
|-------|-------------|
| llama3:70b | 8,192 |
| llama3.1:8b | 131,072 |
| llama3.1:70b | 131,072 |
| llama3.1:405b | 131,072 |
| llama3.2:1b/3b | 131,072 |
| llama3.3:70b | 131,072 |
| mistral:7b | 8,192 |
| mixtral:8x7b | 32,768 |
| qwen2.5:7b/32b | 32,768 |
| deepseek-r1:7b/8b | 131,072 |
| Unknown models | 4,096 (default) |

## Error Handling

| Error | Provider Behavior |
|-------|------------------|
| HTTP 4xx/5xx | `HTTPStatusError` → `LLMCallFailedEvent` |
| Context overflow | `LLMContextLengthExceededError` (CrewAI native) |
| Connection failure | `Exception` → `LLMCallFailedEvent` |

## Architecture

```
┌────────────────┐
│  CrewAI Agent  │
└───────┬────────┘
        │ Agent.llm.call(messages, tools, ...)
        ▼
┌─────────────────────────────┐
│  OllamaCloudProvider        │
│  (extends BaseLLM)          │
│                             │
│  call() / acall()           │
│   ├─ _format_messages()     │
│   ├─ _build_body()          │
│   ├─ BEFORE hooks           │
│   ├─ httpx POST /api/chat   │───────┐
│   ├─ _process_response()    │       │
│   ├─ AFTER hooks            │       │
│   └─ event emission         │       │
└─────────────────────────────┘       │
                                      ▼
                            ┌─────────────────┐
                            │  Ollama Instance │
                            │  (local/remote)  │
                            │                 │
                            │  POST /api/chat  │
                            │  ← JSON / NDJSON │
                            └─────────────────┘
```

**Zero translation layers.** httpx → `/api/chat` → Ollama. That's the whole call path.

## Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

39 tests cover: initialization, capabilities, request body building, non-streaming
calls, streaming calls with thinking tokens, tool calls, stop words, context overflow
handling, auth headers, async call delegation, model discovery.

## License

MIT — see [LICENSE](LICENSE) file.
