"""CrewAI Ollama Cloud Provider.

A custom LLM provider for CrewAI that connects to cloud-hosted
or local Ollama instances with full streaming, tool-calling,
and model discovery support.
"""

from crewai_ollama_cloud.provider import OllamaCloudProvider
from crewai_ollama_cloud.discovery import list_ollama_models, OllamaModelInfo

__all__ = [
    "OllamaCloudProvider",
    "list_ollama_models",
    "OllamaModelInfo",
]
