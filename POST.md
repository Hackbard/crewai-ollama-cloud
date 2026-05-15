# crewai-ollama-cloud — A Native Ollama Provider for CrewAI

**GitHub:** https://github.com/Hackbard/crewai-ollama-cloud

---

## The Problem

CrewAI's built-in Ollama support has a catch: it routes through the **OpenAI-compatible shim** (`/v1/chat/completions`). That's not the real Ollama API — it's a translation layer bolted onto Ollama to make it look like OpenAI. Works… mostly. But it means:

- **No access to Ollama-native features** — thinking/reasoning tokens, `keep_alive`, native tool calls, fine-grained `options` like `num_predict`, `mirostat`, etc.
- **Indirect error handling** — context overflows get mangled through the OpenAI error format, making debugging harder.
- **No Ollama Cloud support** — ollama.com's hosted models (like `gpt-oss:120b-cloud` or `kimi-k2.6-cloud`) expect native API auth flows, not the OpenAI shim.
- **LiteLLM dependency** — another layer between you and the model, adding latency and complexity.

If you're running models locally, the shim works. But if you want the **full Ollama feature set** — especially on cloud — you're out of luck.

## The Solution

`crewai-ollama-cloud` is a **custom LLM provider for CrewAI** that talks directly to Ollama's **native REST API** (`POST /api/chat`). No shims, no LiteLLM, no OpenAI-format translation.

```python
from crewai_ollama_cloud import OllamaCloudProvider

llm = OllamaCloudProvider(
    model="deepseek-v4-flash",
    base_url="https://ollama.com",
    api_key="sk-xxxx",
    temperature=0.7,
    stream=True,
    think=True,          # DeepSeek-R1 / Kimi reasoning tokens
    keep_alive="10m",     # Keep model loaded between calls
)
```

**Every call goes:** CrewAI → httpx → `POST /api/chat` → Ollama. That's it.

## What You Get

| Feature | Built-in Ollama (shim) | crewai-ollama-cloud |
|---|---|---|
| `/api/chat` endpoint | ❌ `/v1/chat/completions` | ✅ Native |
| Thinking / reasoning tokens | ❌ | ✅ `think` parameter |
| NDJSON streaming | ⚠️ OpenAI-format SSE | ✅ True Ollama NDJSON |
| Native tool calls | ⚠️ Translated | ✅ Ollama v0.3+ format |
| `keep_alive` | ❌ | ✅ |
| `num_predict` (`max_tokens`) | ⚠️ Mapped | ✅ Native `options` |
| Ollama Cloud (ollama.com) | ❌ | ✅ Bearer auth |
| Structured output (JSON schema) | ⚠️ via OpenAI format | ✅ Ollama `format` param |
| Model discovery (`/api/tags`) | ❌ | ✅ `list_ollama_models()` |
| Runtime config overrides | ❌ | ✅ `llm.temperature = 0.9` |

## Example: Thinking Models

This is the killer feature for me. Models like **DeepSeek-R1** and **Kimi** use reasoning tokens — a "thinking" phase before the final answer. The OpenAI shim either drops these or confuses CrewAI's output handling.

With the native provider, thinking tokens are properly separated:

```python
llm = OllamaCloudProvider(
    model="deepseek-r1:8b",
    base_url="http://localhost:11434",
    think=True,
    stream=True,
)

# Thinking tokens go to CrewAI as thinking_chunk events
# Final output goes to stream_chunk events
# Both are tracked separately — no confusion
```

## Zero Bloat

The whole provider is **~500 lines** of Python. Dependencies: `crewai` (≥0.80.0) and `httpx` (≥0.25.0). No OpenAI SDK. No LiteLLM. No proxy.

39 unit tests, ruff-clean, GitHub Actions CI on Python 3.10–3.12.

## Who Is This For?

- **Ollama Cloud users** — finally a direct way to use ollama.com models in CrewAI
- **DeepSeek-R1 / Kimi users** — proper thinking token handling
- **Self-hosting with auth** — any HTTPS Ollama instance with API key
- **Anyone who wants full control** over Ollama's native parameters

## Try It

```bash
pip install crewai-ollama-cloud
```

```python
from crewai import Agent, Task, Crew
from crewai_ollama_cloud import OllamaCloudProvider

llm = OllamaCloudProvider(
    model="llama3.1:8b",
    base_url="http://localhost:11434",
    stream=True,
)

agent = Agent(role="Analyst", goal="Analyze", backstory="...", llm=llm)
task = Task(description="Summarize", expected_output="Summary")
crew = Crew(agents=[agent], tasks=[task])
result = crew.kickoff()
```

**Repo:** https://github.com/Hackbard/crewai-ollama-cloud  
**Issues / PRs welcome.**

---

*Built by [Hackbard](https://github.com/Hackbard). MIT licensed.*
