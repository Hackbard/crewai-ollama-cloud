# Contributing

Thanks for wanting to help! Here's how to contribute.

## Setup

```bash
git clone https://github.com/Hackbard/crewai-ollama-cloud.git
cd crewai-ollama-cloud
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Development Workflow

1. Fork & branch off `main`
2. Write code + tests
3. Run linting: `ruff check src/ tests/`
4. Run tests: `PYTHONPATH=src pytest tests/ -v`
5. Push & open a PR

## Code Style

- Python 3.10+ with type hints
- Follow existing patterns (extend `BaseLLM`, use `httpx`, emit events)
- Ruff clean required
- All new features need tests

## Pull Requests

- One feature/fix per PR
- Keep diff focused
- Reference any related issues
- Tests must pass

## Reporting Issues

Open an issue on GitHub. Include:
- CrewAI version (`pip show crewai`)
- Ollama version / cloud
- Error traceback
- Minimal reproduction code
