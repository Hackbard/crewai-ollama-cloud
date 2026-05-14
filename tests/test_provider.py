"""Tests for OllamaCloudProvider — native REST API."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from crewai_ollama_cloud import OllamaCloudProvider
from crewai_ollama_cloud.discovery import list_ollama_models
from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededError,
)


# ── provider initialization tests ──────────────────────────────────────

class TestOllamaCloudProviderInit:
    """Test provider instantiation and configuration."""

    def test_default_url(self):
        provider = OllamaCloudProvider(model="llama3:8b")
        assert provider.base_url == "http://localhost:11434"

    def test_cloud_url(self):
        provider = OllamaCloudProvider(
            model="llama3:8b",
            base_url="https://ollama.my-cloud.com",
        )
        assert provider.base_url == "https://ollama.my-cloud.com"

    def test_strips_v1_suffix(self):
        provider = OllamaCloudProvider(
            model="llama3:8b",
            base_url="http://localhost:11434/v1",
        )
        assert provider.base_url == "http://localhost:11434"

    def test_api_key_explicit(self):
        with patch.dict("os.environ", {}, clear=True):
            provider = OllamaCloudProvider(model="llama3:8b", api_key="sk-test")
            assert provider.api_key == "sk-test"

    def test_api_key_from_env(self):
        with patch.dict("os.environ", {"OLLAMA_API_KEY": "sk-env"}, clear=True):
            provider = OllamaCloudProvider(model="llama3:8b")
            assert provider.api_key == "sk-env"

    def test_api_key_none(self):
        with patch.dict("os.environ", {}, clear=True):
            provider = OllamaCloudProvider(model="llama3:8b")
            assert provider.api_key is None

    def test_temperature(self):
        provider = OllamaCloudProvider(model="llama3:8b", temperature=0.7)
        assert provider.temperature == 0.7

    def test_stream_enabled(self):
        provider = OllamaCloudProvider(model="llama3:8b", stream=True)
        assert provider.stream is True

    def test_max_tokens(self):
        provider = OllamaCloudProvider(model="llama3:8b", max_tokens=1024)
        assert provider.max_tokens == 1024

    def test_stop_sequences(self):
        provider = OllamaCloudProvider(model="llama3:8b", stop=["\n", "END"])
        assert "\n" in provider.stop_sequences
        assert "END" in provider.stop_sequences

    def test_think_mode(self):
        provider = OllamaCloudProvider(model="deepseek-r1:8b", think=True)
        assert provider.think is True

    def test_provider_name(self):
        provider = OllamaCloudProvider(model="llama3:8b")
        assert provider.provider == "ollama_cloud"
        assert provider.llm_type == "ollama_cloud"

    def test_keep_alive(self):
        provider = OllamaCloudProvider(model="llama3:8b")
        assert provider.keep_alive == "5m"


# ── capabilities tests ─────────────────────────────────────────────────

class TestOllamaCloudProviderCapabilities:

    def test_function_calling_supported(self):
        provider = OllamaCloudProvider(model="llama3:8b")
        assert provider.supports_function_calling() is True

    def test_stop_words_supported(self):
        provider = OllamaCloudProvider(model="llama3:8b", stop=["END"])
        assert provider.supports_stop_words() is True

    def test_stop_words_empty(self):
        provider = OllamaCloudProvider(model="llama3:8b")
        assert provider.supports_stop_words() is False

    def test_context_window_known_model(self):
        provider = OllamaCloudProvider(model="llama3.1:8b")
        assert provider.get_context_window_size() == 131072

    def test_context_window_unknown_model(self):
        provider = OllamaCloudProvider(model="unknown-model")
        assert provider.get_context_window_size() == 4096

    def test_multimodal_vision_model(self):
        provider = OllamaCloudProvider(model="llava:13b")
        assert provider.supports_multimodal() is True

    def test_multimodal_text_model(self):
        provider = OllamaCloudProvider(model="llama3:8b")
        assert provider.supports_multimodal() is False

    def test_to_config_dict(self):
        provider = OllamaCloudProvider(
            model="llama3:8b",
            base_url="http://localhost:11434",
            temperature=0.5,
            stop=["END"],
            stream=True,
            max_tokens=2048,
            top_p=0.9,
        )
        config = provider.to_config_dict()
        assert config["model"] == "ollama_cloud/llama3:8b"
        assert config["temperature"] == 0.5
        assert config["base_url"] == "http://localhost:11434"
        assert "END" in config["stop"]
        assert config["stream"] is True
        assert config["max_tokens"] == 2048
        assert config["top_p"] == 0.9


# ── request body tests ─────────────────────────────────────────────────

class TestRequestBody:
    """Test that _build_body produces correct Ollama API JSON."""

    def test_simple_body(self):
        provider = OllamaCloudProvider(model="llama3:8b")
        body = provider._build_body(
            messages=[{"role": "user", "content": "Hello"}],
            tools=None,
            response_model=None,
        )
        assert body["model"] == "llama3:8b"
        assert body["messages"] == [{"role": "user", "content": "Hello"}]
        assert body["keep_alive"] == "5m"

    def test_with_options(self):
        provider = OllamaCloudProvider(
            model="llama3:8b",
            temperature=0.3,
            max_tokens=512,
            top_p=0.9,
            stop=["END"],
        )
        body = provider._build_body(
            messages=[{"role": "user", "content": "Hi"}],
            tools=None,
            response_model=None,
        )
        assert body["options"]["temperature"] == 0.3
        assert body["options"]["num_predict"] == 512
        assert body["options"]["top_p"] == 0.9
        assert body["options"]["stop"] == ["END"]

    def test_with_think(self):
        provider = OllamaCloudProvider(model="deepseek-r1:8b", think=True)
        body = provider._build_body(
            messages=[{"role": "user", "content": "reason"}],
            tools=None,
            response_model=None,
        )
        assert body["think"] is True

    def test_messages_preserved(self):
        provider = OllamaCloudProvider(model="llama3:8b")
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        body = provider._build_body(messages=msgs, tools=None, response_model=None)
        assert len(body["messages"]) == 2
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][1]["role"] == "user"


# ── non-streaming call tests ───────────────────────────────────────────

class TestNonStreamingCall:

    def test_simple_call(self):
        provider = OllamaCloudProvider(model="llama3:8b", stream=False)

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "model": "llama3:8b",
            "message": {"role": "assistant", "content": "Hello from Ollama!"},
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 5,
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider._client, "post", return_value=mock_response) as spy:
            result = provider.call("Hello")
            assert result == "Hello from Ollama!"

            # Verify POST to correct endpoint
            call_args = spy.call_args
            assert call_args[0][0] == "/api/chat"

        # Token tracking
        usage = provider.get_token_usage_summary()
        assert usage.total_tokens == 15

    def test_tool_call_non_streaming(self):
        provider = OllamaCloudProvider(model="llama3.1:8b", stream=False)

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "model": "llama3.1:8b",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "Berlin"},
                        }
                    }
                ],
            },
            "done": True,
            "prompt_eval_count": 20,
            "eval_count": 8,
        }
        mock_response.raise_for_status = MagicMock()

        def weather_fn(city: str) -> str:
            return f"Sunny in {city}"

        with patch.object(provider._client, "post", return_value=mock_response):
            result = provider.call(
                "What's the weather?",
                available_functions={"get_weather": weather_fn},
            )
            assert result == "Sunny in Berlin"

    def test_stop_words_applied(self):
        provider = OllamaCloudProvider(model="llama3:8b", stop=["END"], stream=False)

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "model": "llama3:8b",
            "message": {"role": "assistant", "content": "First part. END Rest is cut"},
            "done": True,
            "prompt_eval_count": 5,
            "eval_count": 6,
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider._client, "post", return_value=mock_response):
            result = provider.call("Go")
            assert result == "First part."

    def test_context_overflow(self):
        provider = OllamaCloudProvider(model="llama3:8b", stream=False)

        error_response = MagicMock(spec=httpx.Response)
        error_response.status_code = 400
        error_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "context length exceeded", request=MagicMock(), response=error_response
        )

        with patch.object(provider._client, "post", return_value=error_response):
            with pytest.raises(LLMContextLengthExceededError):
                provider.call("Very long input...")


# ── streaming call tests ───────────────────────────────────────────────

class TestStreamingCall:

    def test_streaming_concatenates(self):
        provider = OllamaCloudProvider(model="llama3:8b", stream=True)

        raw_lines = [
            json.dumps({"model": "llama3:8b", "message": {"role": "assistant", "content": "Hello"}, "done": False}),
            json.dumps({"model": "llama3:8b", "message": {"role": "assistant", "content": " from"}, "done": False}),
            json.dumps({"model": "llama3:8b", "message": {"role": "assistant", "content": " Ollama!"}, "done": False}),
            json.dumps({"model": "llama3:8b", "message": {"role": "assistant", "content": ""}, "done": True,
                        "prompt_eval_count": 3, "eval_count": 7}),
        ]

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=None)
        mock_stream.iter_lines.return_value = raw_lines
        mock_stream.raise_for_status = MagicMock()

        with patch.object(provider._client, "stream", return_value=mock_stream):
            result = provider.call("Hi")
            assert result == "Hello from Ollama!"

    def test_streaming_with_thinking(self):
        provider = OllamaCloudProvider(model="deepseek-r1:8b", think=True, stream=True)

        raw_lines = [
            json.dumps({"model": "deepseek-r1:8b", "message": {"role": "assistant", "content": "", "thinking": "Hmm..."}, "done": False}),
            json.dumps({"model": "deepseek-r1:8b", "message": {"role": "assistant", "content": "42"}, "done": False}),
            json.dumps({"model": "deepseek-r1:8b", "message": {"role": "assistant", "content": ""}, "done": True,
                        "prompt_eval_count": 5, "eval_count": 3}),
        ]

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=None)
        mock_stream.iter_lines.return_value = raw_lines
        mock_stream.raise_for_status = MagicMock()

        with patch.object(provider._client, "stream", return_value=mock_stream):
            result = provider.call("What is the answer?")
            assert result == "42"

    def test_streaming_stop_words(self):
        provider = OllamaCloudProvider(model="llama3:8b", stop=["STOP"], stream=True)

        raw_lines = [
            json.dumps({"model": "llama3:8b", "message": {"role": "assistant", "content": "Keep this. STOP cut"}, "done": False}),
            json.dumps({"model": "llama3:8b", "message": {"role": "assistant", "content": ""}, "done": True,
                        "prompt_eval_count": 2, "eval_count": 2}),
        ]

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=None)
        mock_stream.iter_lines.return_value = raw_lines
        mock_stream.raise_for_status = MagicMock()

        with patch.object(provider._client, "stream", return_value=mock_stream):
            result = provider.call("Go")
            assert result == "Keep this."

    def test_streaming_tool_call(self):
        provider = OllamaCloudProvider(model="llama3.1:8b", stream=True)

        raw_lines = [
            json.dumps({
                "model": "llama3.1:8b", "done": True,
                "message": {
                    "role": "assistant", "content": "",
                    "tool_calls": [{"function": {"name": "calc", "arguments": {"x": 2}}}],
                },
                "prompt_eval_count": 10, "eval_count": 5,
            }),
        ]

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=None)
        mock_stream.iter_lines.return_value = raw_lines
        mock_stream.raise_for_status = MagicMock()

        def calc_fn(x: int) -> str:
            return str(x * 2)

        with patch.object(provider._client, "stream", return_value=mock_stream):
            result = provider.call(
                "calc 2", available_functions={"calc": calc_fn},
            )
            assert result == "4"


# ── discovery tests ────────────────────────────────────────────────────

class TestModelDiscovery:

    def test_list_models_success(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3:8b", "digest": "abc", "size": 4666166976, "modified_at": "2024-01-01T00:00:00Z"},
                {"name": "mistral:7b", "digest": "def", "size": 4369166336, "modified_at": "2024-01-02T00:00:00Z"},
            ]
        }
        with patch("httpx.get", return_value=mock_response):
            models = list_ollama_models("http://localhost:11434")
            assert len(models) == 2
            assert models[0].name == "llama3:8b"
            assert models[0].family == "llama"
            assert models[1].family == "mistral"
            assert models[0].parameter_size == "8b"

    def test_list_models_connection_error(self):
        with patch("httpx.get", side_effect=Exception("Connection refused")):
            with pytest.raises(Exception, match="Connection refused"):
                list_ollama_models("http://localhost:11434", timeout=1)

    def test_list_models_missing_key(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"not_models": []}
        with patch("httpx.get", return_value=mock_response):
            with pytest.raises(ValueError, match="expected 'models' key"):
                list_ollama_models("http://localhost:11434")


# ── auth header test ───────────────────────────────────────────────────

class TestAuthHeaders:
    def test_cloud_auth_header(self):
        provider = OllamaCloudProvider(model="llama3:8b", api_key="sk-cloud", base_url="https://ollama.com")
        assert provider._client.headers.get("Authorization") == "Bearer sk-cloud"

    def test_local_no_auth(self):
        with patch.dict("os.environ", {}, clear=True):
            provider = OllamaCloudProvider(model="llama3:8b", base_url="http://localhost:11434")
            assert "Authorization" not in provider._client.headers


# ── async call test ────────────────────────────────────────────────────

class TestAsyncCall:
    async def test_acall_delegates_to_call(self):
        provider = OllamaCloudProvider(model="llama3:8b", stream=False)

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "model": "llama3:8b",
            "message": {"role": "assistant", "content": "Async works"},
            "done": True,
            "prompt_eval_count": 2,
            "eval_count": 2,
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider._client, "post", return_value=mock_response):
            result = await provider.acall("Hello")
            assert result == "Async works"
