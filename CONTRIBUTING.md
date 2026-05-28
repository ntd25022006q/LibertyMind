# Contributing to LibertyMind

First off, thank you for considering contributing to LibertyMind! It's people like you that make LibertyMind such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by the principle of **honest and respectful collaboration**. Be constructive, be honest, be kind.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues. When you create a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the problem
- **Expected behavior** vs **actual behavior**
- **Environment details** (Python version, OS, PyTorch version)
- **Error logs** if applicable

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. Include:

- **Use case** — why is this enhancement useful?
- **Expected behavior** — what should it do?
- **Current workaround** — is there an alternative today?

### Pull Requests

1. Fork the repo and create your branch from `main`
2. If you've added code, add tests
3. Ensure the test suite passes (`pytest tests/ -v`)
4. Make sure your code lints (`ruff check src/`)
5. Write a clear commit message

## Development Setup

```bash
# Clone
git clone https://github.com/ntd25022006q/LibertyMind.git
cd LibertyMind

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check src/
```

## Coding Standards

- **Python 3.9+** compatibility
- **Type hints** for all public functions
- **Docstrings** for all modules, classes, and public functions
- **Tests** for all new features
- **No hardcoded secrets** — use environment variables

## Architecture

- `src/core/` — PyTorch `nn.Module` components (must be differentiable)
- `src/integration/` — Pure Python integration modules
- `src/clients/` — Provider adapters
- `src/server/` — FastAPI server
- `tests/` — Test suite

## Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters
- Reference issues and pull requests liberally

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
