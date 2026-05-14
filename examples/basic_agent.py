"""Example: Using OllamaCloudProvider with CrewAI.

Run with:
    PYTHONPATH=src python examples/basic_agent.py
"""

from crewai import Agent, Task, Crew

# Import our custom provider
from crewai_ollama_cloud import OllamaCloudProvider


def main():
    # Configure Ollama Cloud provider
    # For local Ollama, omit api_key and use localhost:
    llm = OllamaCloudProvider(
        model="llama3.2:3b",
        base_url="http://localhost:11434/v1",
        temperature=0.7,
        stream=True,
    )

    # Create an agent with our provider
    agent = Agent(
        role="Helpful Assistant",
        goal="Provide concise, accurate answers",
        backstory=(
            "You are a helpful AI assistant powered by Ollama. "
            "You communicate clearly and directly."
        ),
        llm=llm,
        verbose=True,
    )

    # Define a task
    task = Task(
        description="What are the three laws of robotics?",
        expected_output="A concise summary of Asimov's three laws of robotics.",
    )

    # Run the crew
    crew = Crew(agents=[agent], tasks=[task])
    result = crew.kickoff()
    print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
