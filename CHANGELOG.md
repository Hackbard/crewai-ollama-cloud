# Changelog

## [0.2.1] — 2026-06-05

### Fixed
- **Completes native tool calling with CrewAI's native-tools executor.** Two
  further incompatibilities surfaced during end-to-end agent runs:
  - `_convert_tools_ollama` assumed CrewAI's `{tool_name: BaseTool}` mapping and
    iterated `.items()` on OpenAI-formatted tool dicts the native executor passes
    — producing bogus tools literally named `type`/`function`, so the model
    emitted unusable tool calls ("Tool not found"). OpenAI-format tool dicts are
    now forwarded unchanged (Ollama's tool format is OpenAI-compatible).
  - `_normalize_ollama_messages` re-sent assistant `tool_calls` with `arguments`
    as a JSON *string* and `content: null`, which Ollama's `/api/chat` rejects
    with HTTP 400 ("Value looks like object…"). Arguments are now parsed to
    objects and null content is coerced to `""`.
- Together with 0.2.0, a full multi-turn native tool loop now works end-to-end.

## [0.2.0] — 2026-06-05

### Fixed
- **Native tool calling with CrewAI's experimental agent executor.** When the
  model returned tool calls but `available_functions` was not supplied (exactly
  how CrewAI's native-tools executor invokes the LLM — it expects the tool-call
  list back to execute itself), the provider collapsed the empty `content` to
  `""`, causing CrewAI to raise `Invalid response from LLM call - None or empty.`
  The provider now **surfaces the raw tool-call list** to the caller in that case
  (both streaming and non-streaming paths), so agents can actually use tools.

### Added
- Tests covering tool-call passthrough when `available_functions` is `None`.

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
