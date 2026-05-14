"""Ollama Cloud Provider for CrewAI — native REST API.

Talks directly to Ollama's real ``POST /api/chat`` endpoint. No OpenAI
shim, no LiteLLM — pure Ollama protocol with streaming NDJSON, tool calls,
structured output, and thinking support.

Supports:
- Local Ollama (``http://localhost:11434``)
- Ollama Cloud (``https://ollama.com`` with API key)
- Any self-hosted Ollama instance

Usage::

    from crewai_ollama_cloud import OllamaCloudProvider

    llm = OllamaCloudProvider(
        model="llama3.1:8b",
        base_url="http://localhost:11434",   # no /v1 — native API
        api_key="sk-...",                     # only for cloud
        temperature=0.7,
        stream=True,
    )
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any, Literal

import httpx
from pydantic import BaseModel, PrivateAttr, Field, model_validator

from crewai.events.types.llm_events import LLMCallType
from crewai.llms.base_llm import BaseLLM, llm_call_context
from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededError,
)

if TYPE_CHECKING:
    from crewai.agents.agent_builder.base_agent import BaseAgent
    from crewai.task import Task
    from crewai.tools.base_tool import BaseTool
    from crewai.utilities.types import LLMMessage

logger = logging.getLogger(__name__)

# ── context window sizes for common Ollama models ──────────────────────
_OLLAMA_CONTEXT_SIZES: dict[str, str | int] = {
    "llama3:70b": 8192,
    "llama3.1:8b": 131072,
    "llama3.1:70b": 131072,
    "llama3.1:405b": 131072,
    "llama3.2:1b": 131072,
    "llama3.2:3b": 131072,
    "llama3.3:70b": 131072,
    "mistral:7b": 8192,
    "mistral-nemo": 1024000,
    "mixtral:8x7b": 32768,
    "gemma2:9b": 8192,
    "gemma2:27b": 8192,
    "qwen2.5:7b": 32768,
    "qwen2.5:32b": 32768,
    "deepseek-r1:7b": 131072,
    "deepseek-r1:8b": 131072,
    "command-r": 131072,
    "phi3:mini": 4096,
}
_DEFAULT_CONTEXT_SIZE = 4096


class OllamaCloudProvider(BaseLLM):
    """Ollama LLM provider via native ``POST /api/chat``.

    Talks the real Ollama protocol: JSON request/response, NDJSON streaming,
    native tool calling, structured output via JSON schema, and thinking
    (reasoning) support for models like ``deepseek-r1``.
    """

    llm_type: Literal["ollama_cloud"] = "ollama_cloud"
    model_config = {"arbitrary_types_allowed": True, "populate_by_name": True}

    # ── fields ──────────────────────────────────────────────────────────
    model: str
    temperature: float | None = None
    base_url: str = "http://localhost:11434"
    api_key: str | None = None
    provider: str = Field(default="ollama_cloud")
    stream: bool = False
    max_tokens: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop: list[str] = Field(default_factory=list)
    timeout: float = 120.0
    keep_alive: str = "5m"
    think: bool = False  # enable thinking/reasoning tokens
    additional_params: dict[str, Any] = Field(default_factory=dict)

    _client: httpx.Client = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _normalize_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        # Force provider name
        data["provider"] = "ollama_cloud"
        # API key: explicit > OLLAMA_API_KEY env
        if not data.get("api_key"):
            data["api_key"] = os.getenv("OLLAMA_API_KEY")
        # Clean base_url: strip /v1 etc. we talk native API
        url = (data.get("base_url") or "http://localhost:11434").rstrip("/")
        # Remove /v1 suffix if present
        if url.endswith("/v1"):
            url = url[:-3].rstrip("/")
        data["base_url"] = url
        return data

    @model_validator(mode="after")
    def _init_client(self) -> OllamaCloudProvider:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout,
        )
        return self

    # ── public API ──────────────────────────────────────────────────────

    def call(
        self,
        messages: str | list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: BaseAgent | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        with llm_call_context():
            try:
                self._emit_call_started_event(
                    messages=messages, tools=tools, callbacks=callbacks,
                    available_functions=available_functions,
                    from_task=from_task, from_agent=from_agent,
                )

                formatted = self._format_messages(messages)

                if not self._invoke_before_llm_call_hooks(formatted, from_agent):
                    raise ValueError("LLM call blocked by before_llm_call hook")

                if self.stream:
                    return self._streaming_call(
                        formatted, tools, available_functions,
                        from_task, from_agent, response_model,
                    )
                return self._non_streaming_call(
                    formatted, tools, available_functions,
                    from_task, from_agent, response_model,
                )

            except Exception as e:
                logger.error("OllamaCloudProvider call failed: %s", e)
                self._emit_call_failed_event(
                    error=str(e), from_task=from_task, from_agent=from_agent,
                )
                raise

    async def acall(
        self,
        messages: str | list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: BaseAgent | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Async call — delegates to sync with thread-safe httpx client."""
        return self.call(
            messages=messages, tools=tools, callbacks=callbacks,
            available_functions=available_functions,
            from_task=from_task, from_agent=from_agent,
            response_model=response_model,
        )

    # ── non-streaming ───────────────────────────────────────────────────

    def _non_streaming_call(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None,
        available_functions: dict[str, Any] | None,
        from_task: Task | None,
        from_agent: BaseAgent | None,
        response_model: type[BaseModel] | None,
    ) -> str | Any:
        body = self._build_body(messages, tools, response_model)
        body["stream"] = False

        try:
            resp = self._client.post("/api/chat", json=body)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            if _is_context_overflow_msg(str(e)):
                raise LLMContextLengthExceededError(str(e)) from e
            raise

        data: dict[str, Any] = resp.json()
        return self._process_response(data, available_functions, from_task, from_agent)

    # ── streaming (NDJSON) ──────────────────────────────────────────────

    def _streaming_call(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None,
        available_functions: dict[str, Any] | None,
        from_task: Task | None,
        from_agent: BaseAgent | None,
        response_model: type[BaseModel] | None,
    ) -> str:
        body = self._build_body(messages, tools, response_model)
        body["stream"] = True

        full_text = ""
        usage: dict[str, Any] = {}

        try:
            with self._client.stream("POST", "/api/chat", json=body) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk: dict[str, Any] = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Collect text tokens
                    msg = chunk.get("message", {})
                    content = msg.get("content", "")
                    thinking = msg.get("thinking", "")

                    if content:
                        full_text += content
                        self._emit_stream_chunk_event(
                            chunk=content,
                            from_task=from_task,
                            from_agent=from_agent,
                            call_type=LLMCallType.LLM_CALL,
                        )

                    if thinking:
                        self._emit_thinking_chunk_event(
                            chunk=thinking,
                            from_task=from_task,
                            from_agent=from_agent,
                        )

                    # Track tool calls from stream
                    tool_calls = msg.get("tool_calls")
                    if tool_calls:
                        tc = tool_calls[0]
                        args = tc["function"].get("arguments", {})
                        self._emit_stream_chunk_event(
                            chunk="",
                            from_task=from_task,
                            from_agent=from_agent,
                            tool_call={
                                "index": 0,
                                "function": {
                                    "name": tc["function"]["name"],
                                    "arguments": (
                                        json.dumps(args)
                                        if not isinstance(args, str)
                                        else args
                                    ),
                                },
                            },
                            call_type=LLMCallType.TOOL_CALL,
                        )

                    # Final chunk -> usage
                    if chunk.get("done"):
                        eval_count = chunk.get("eval_count", 0)
                        prompt_eval_count = chunk.get("prompt_eval_count", 0)
                        usage = {
                            "prompt_tokens": prompt_eval_count,
                            "completion_tokens": eval_count,
                            "total_tokens": prompt_eval_count + eval_count,
                        }

                        # Handle tool calls from final chunk
                        if msg.get("tool_calls") and available_functions:
                            tool_call = msg["tool_calls"][0]
                            fn_name = tool_call["function"]["name"]
                            fn_args = tool_call["function"]["arguments"]
                            if isinstance(fn_args, str):
                                fn_args = json.loads(fn_args)
                            result = self._handle_tool_execution(
                                function_name=fn_name,
                                function_args=fn_args,
                                available_functions=available_functions,
                                from_task=from_task,
                                from_agent=from_agent,
                            )
                            if usage:
                                self._track_token_usage_internal(usage)
                            return result if result is not None else ""

        except httpx.HTTPStatusError as e:
            if _is_context_overflow_msg(str(e)) and full_text:
                return full_text.strip()
            raise

        if usage:
            self._track_token_usage_internal(usage)

        text = full_text.strip()
        text = self._apply_stop_words(text)

        if from_agent is None and isinstance(text, str):
            text = self._invoke_after_llm_call_hooks(messages, text, from_agent)

        self._emit_call_completed_event(
            response=text,
            call_type=LLMCallType.LLM_CALL,
            from_task=from_task,
            from_agent=from_agent,
            messages=messages,
            usage=usage,
        )
        return text

    # ── response processing ─────────────────────────────────────────────

    def _process_response(
        self,
        data: dict[str, Any],
        available_functions: dict[str, Any] | None,
        from_task: Task | None,
        from_agent: BaseAgent | None,
    ) -> str | Any:
        """Process a non-streaming Ollama /api/chat response."""
        usage = {
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
            "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
        }
        self._track_token_usage_internal(usage)

        msg = data.get("message", {})
        tool_calls = msg.get("tool_calls")

        if tool_calls and available_functions:
            tool_call = tool_calls[0]
            fn_name = tool_call["function"]["name"]
            fn_args = tool_call["function"]["arguments"]
            if isinstance(fn_args, str):
                fn_args = json.loads(fn_args)
            return self._handle_tool_execution(
                function_name=fn_name,
                function_args=fn_args,
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
            )

        text = msg.get("content", "") or ""
        text = self._apply_stop_words(text)

        if from_agent is None and isinstance(text, str):
            text = self._invoke_after_llm_call_hooks(
                [msg], text, from_agent,
            )

        self._emit_call_completed_event(
            response=text,
            call_type=LLMCallType.LLM_CALL,
            from_task=from_task,
            from_agent=from_agent,
            usage=usage,
        )
        return text

    # ── request body builder ────────────────────────────────────────────

    def _build_body(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None,
        response_model: type[BaseModel] | None,
    ) -> dict[str, Any]:
        """Build the JSON body for POST /api/chat."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": _normalize_ollama_messages(messages),
        }

        # Options sub-object
        options: dict[str, Any] = {}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.max_tokens is not None:
            options["num_predict"] = self.max_tokens
        if self.top_p is not None:
            options["top_p"] = self.top_p
        if self.top_k is not None:
            options["top_k"] = self.top_k
        if self.stop_sequences:
            options["stop"] = self.stop_sequences
        if options:
            body["options"] = options

        body["keep_alive"] = self.keep_alive

        if self.think:
            body["think"] = True

        # Tool calling — Ollama native format
        if tools:
            body["tools"] = _convert_tools_ollama(tools)

        # Structured output via JSON schema
        if response_model is not None:
            from crewai.utilities.pydantic_schema_utils import generate_model_description
            body["format"] = generate_model_description(response_model)

        # Merge additional params
        for key, value in self.additional_params.items():
            if key not in body:
                body[key] = value

        return body

    # ── capabilities ────────────────────────────────────────────────────

    def supports_function_calling(self) -> bool:
        return True

    def supports_stop_words(self) -> bool:
        return self._supports_stop_words_implementation()

    def get_context_window_size(self) -> int:
        for prefix, size in _OLLAMA_CONTEXT_SIZES.items():
            if self.model.startswith(prefix) or self.model == prefix:
                return int(size)
        return _DEFAULT_CONTEXT_SIZE

    def supports_multimodal(self) -> bool:
        vision_kw = ("vision", "multimodal", "llava", "bakllava", "minicpm", "cogvlm")
        return any(kw in self.model.lower() for kw in vision_kw)

    def to_config_dict(self) -> dict[str, Any]:
        config = super().to_config_dict()
        if self.stream:
            config["stream"] = self.stream
        if self.max_tokens is not None:
            config["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            config["top_p"] = self.top_p
        if self.top_k is not None:
            config["top_k"] = self.top_k
        if self.timeout != 120.0:
            config["timeout"] = self.timeout
        if self.think:
            config["think"] = self.think
        return config


# ── helpers ─────────────────────────────────────────────────────────────

def _normalize_ollama_messages(messages: list[LLMMessage]) -> list[dict[str, Any]]:
    """Ensure messages are in Ollama's expected flat-dict format."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        entry: dict[str, Any] = {"role": msg["role"], "content": msg.get("content", "")}
        # Keep images if present (multimodal)
        if "images" in msg:
            entry["images"] = msg["images"]
        # Keep tool_calls if present
        if "tool_calls" in msg:
            entry["tool_calls"] = msg["tool_calls"]
        result.append(entry)
    return result


def _convert_tools_ollama(tools: list[dict[str, BaseTool]]) -> list[dict[str, Any]]:
    """Convert CrewAI tool format to Ollama native tools format."""
    converted: list[dict[str, Any]] = []
    for tool_dict in tools:
        for tool_name, tool_obj in tool_dict.items():
            # Build JSON Schema from CrewAI BaseTool
            schema: dict[str, Any] = {
                "type": "object",
                "properties": {},
                "required": [],
            }
            if hasattr(tool_obj, "args_schema") and tool_obj.args_schema:
                s = tool_obj.args_schema.model_json_schema()
                schema["properties"] = s.get("properties", {})
                schema["required"] = s.get("required", [])

            converted.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": getattr(tool_obj, "description", tool_name),
                    "parameters": schema,
                },
            })
    return converted


def _is_context_overflow_msg(text: str) -> bool:
    markers = (
        "context length", "maximum context", "reduce the length",
        "too many tokens", "context window", "input is too long",
        "exceeds the maximum",
    )
    return any(m in text.lower() for m in markers)
