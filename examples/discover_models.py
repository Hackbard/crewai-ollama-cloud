"""Example: Model discovery and provider inspection.

Run with:
    PYTHONPATH=src python examples/discover_models.py
"""

from crewai_ollama_cloud import list_ollama_models, OllamaCloudProvider


def main():
    # Discover available models on your Ollama instance
    base_url = "http://localhost:11434"

    print(f"Discovering models from {base_url}...\n")
    try:
        models = list_ollama_models(base_url)
        print(f"Found {len(models)} models:\n")
        for m in models:
            caps = []
            if m.multimodal:
                caps.append("vision")
            cap_str = f" [{', '.join(caps)}]" if caps else ""
            print(
                f"  {m.name:30s} {m.parameter_size:>6s}  "
                f"family={m.family:12s} {m.size_human:>10s}{cap_str}"
            )
    except Exception as e:
        print(f"Could not connect to Ollama: {e}")
        print("Make sure Ollama is running and accessible.")

    # Show provider metadata
    print("\nProvider capabilities:")
    for model_name in ["llama3.1:8b", "mistral:7b", "unknown-model"]:
        provider = OllamaCloudProvider(model=model_name, base_url=f"{base_url}/v1")
        print(
            f"  {model_name:20s}: "
            f"context={provider.get_context_window_size()}, "
            f"multimodal={provider.supports_multimodal()}, "
            f"tool_calling={provider.supports_function_calling()}"
        )


if __name__ == "__main__":
    main()
