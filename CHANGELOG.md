# Changelog

## [0.1.0] — 2025-05-14

### Added
- `OllamaCloudProvider` — native `POST /api/chat` via httpx
- NDJSON streaming with thinking/reasoning token support
- Native Ollama tool calling (v0.3+ format)
- Structured output via JSON schema (`format` parameter)
- Model discovery (`list_ollama_models()`)
- Cloud auth (`Authorization: Bearer`, `OLLAMA_API_KEY` env)
- Runtime config overrides (temperature, max_tokens, etc.)
- Context window auto-detection for 15+ popular Ollama models
- Multimodal detection for vision models
- Full CrewAI event emission + hook integration
- Async call support (`acall`) delegating to sync
- 39 unit tests covering init, capabilities, body, streaming, tools, discovery, async
- Ruff linting clean
- GitHub Actions CI (Python 3.10–3.12, ruff + pytest)
- Comprehensive README with badges
- LLM.md for AI assistant context
