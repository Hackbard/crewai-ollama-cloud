"""Ollama model discovery utilities.

Provides functions to list available models from an Ollama instance
and introspect model capabilities.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class OllamaModelInfo:
    """Information about an Ollama model.

    Attributes:
        name: Model name (e.g. ``"llama3:8b"``).
        digest: SHA256 digest of the model.
        size_bytes: Size of the model in bytes.
        size_human: Human-readable size string.
        modified_at: Last modified timestamp.
        family: Model family if detectable (e.g. ``"llama"``, ``"mistral"``).
        parameter_size: Parameter size label (e.g. ``"8b"``, ``"70b"``).
        multimodal: Whether the model supports multimodal input.
    """

    name: str
    digest: str = ""
    size_bytes: int = 0
    size_human: str = ""
    modified_at: str = ""
    family: str = ""
    parameter_size: str = ""
    multimodal: bool = False


def list_ollama_models(
    base_url: str = "http://localhost:11434",
    api_key: str | None = None,
    timeout: float = 10.0,
) -> list[OllamaModelInfo]:
    """List all available models from an Ollama instance.

    Uses Ollama's native REST API (not the OpenAI-compatible endpoint)
    to fetch the full model catalog with metadata.

    Args:
        base_url: Base URL of the Ollama instance.
                  Defaults to ``http://localhost:11434``.
        api_key: API key for authenticated instances (e.g. Ollama Cloud).
                 Also reads ``OLLAMA_API_KEY`` env var.
        timeout: Request timeout in seconds.

    Returns:
        List of :class:`OllamaModelInfo` for each available model.

    Raises:
        httpx.HTTPError: If the request fails (connection error, timeout).
        ValueError: If the response format is unexpected.

    Example:
        >>> models = list_ollama_models("https://ollama.my-company.com")
        >>> for m in models:
        ...     print(f"{m.name} ({m.parameter_size})")
    """
    api_key = api_key or os.environ.get("OLLAMA_API_KEY")
    url = f"{base_url.rstrip('/')}/api/tags"
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = httpx.get(url, timeout=timeout, headers=headers)
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPError as e:
        logger.error("Failed to connect to Ollama at %s: %s", base_url, e)
        raise
    except ValueError as e:
        raise ValueError(
            f"Invalid JSON response from Ollama at {base_url}: {e}"
        ) from e

    if not isinstance(data, dict) or "models" not in data:
        raise ValueError(
            f"Unexpected response format from Ollama at {base_url}: "
            f"expected 'models' key in JSON object"
        )

    models: list[OllamaModelInfo] = []
    for raw in data["models"]:
        if not isinstance(raw, dict):
            continue

        name = raw.get("name", "unknown")
        info = OllamaModelInfo(
            name=name,
            digest=raw.get("digest", ""),
            size_bytes=raw.get("size", 0),
            size_human=_format_bytes(raw.get("size", 0)),
            modified_at=raw.get("modified_at", ""),
            family=_detect_family(name),
            parameter_size=_detect_param_size(name),
            multimodal=_detect_multimodal(name, raw.get("details", {})),
        )
        models.append(info)

    logger.info("Discovered %d models from %s", len(models), base_url)
    return models


def _format_bytes(size: int) -> str:
    """Format bytes into a human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def _detect_family(name: str) -> str:
    """Detect model family from the model name."""
    lower = name.lower()
    families = [
        "llama", "mistral", "gemma", "phi", "qwen", "deepseek",
        "falcon", "yi", "command-r", "dolphin", "orca", "vicuna",
        "wizard", "codellama", "starcoder", "neural-chat",
    ]
    for f in families:
        if f in lower:
            return f
    # Fallback: use the first segment before colon or slash
    base = name.split(":")[0].split("/")[-1]
    return base


def _detect_param_size(name: str) -> str:
    """Detect parameter size from the model name."""
    lower = name.lower()
    # Common patterns: 7b, 8b, 13b, 70b, 1.5b, etc.
    match = re.search(r"(\d+\.?\d*)\s*[bB]", lower)
    if match:
        return f"{match.group(1)}b"
    return ""


_MULTIMODAL_PATTERNS = (
    "vision", "multimodal", "llava", "bakllava",
    "minicpm-v", "cogvlm",
)


def _detect_multimodal(name: str, details: dict[str, Any]) -> bool:
    """Detect if a model supports multimodal inputs."""
    lower = name.lower()
    if any(pat in lower for pat in _MULTIMODAL_PATTERNS):
        return True
    # Check details for vision support
    families = details.get("families", [])
    if isinstance(families, list):
        for f in families:
            if isinstance(f, str) and any(pat in f.lower() for pat in _MULTIMODAL_PATTERNS):
                return True
    return False
