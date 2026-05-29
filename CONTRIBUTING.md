# Contributing to LibertyMind

Thank you for considering a contribution to LibertyMind. This document outlines the process and expectations for contributing to this project.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold its standards. Be constructive, be honest, be respectful.

## How to Contribute

### Reporting Bugs

Before creating a bug report, please search the existing issues to avoid duplicates. When you file a bug, include:

- **Clear title and description** of the problem
- **Steps to reproduce** the issue
- **Expected behavior** vs. **actual behavior**
- **Environment details** (Python version, OS, PyTorch version)
- **Error logs** or stack traces if applicable

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. Please include:

- **Use case** -- why is this enhancement useful?
- **Expected behavior** -- what should it do?
- **Current workaround** -- is there an alternative today?

### Pull Requests

1. Fork the repository and create your branch from `main`
2. If you have added code, add corresponding tests
3. Ensure the test suite passes (`pytest tests/ -v`)
4. Ensure your code passes linting (`ruff check src/ tests/`)
5. Write a clear, descriptive commit message
6. Open a pull request against the `main` branch

## Development Setup

```bash
# Clone the repository
git clone https://github.com/ntd25022006q/LibertyMind.git
cd LibertyMind

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install with development dependencies
pip install -e ".[dev]"

# Install PyTorch (optional, for neural modules)
pip install -e ".[torch]"

# Run the test suite
pytest tests/ -v

# Run the linter
ruff check src/ tests/
```

## Coding Standards

- **Python 3.9+** compatibility is required
- **Type hints** for all public functions and class methods
- **Docstrings** for all modules, classes, and public functions (Google style)
- **Tests** for all new features or bug fixes
- **No hardcoded secrets** -- use environment variables for all credentials

## Architecture Overview

- `src/core/` -- PyTorch `nn.Module` components (must be differentiable)
- `src/integration/` -- Pure Python integration modules (no GPU required)
- `src/clients/` -- Provider adapters for multi-provider support
- `src/server/` -- FastAPI proxy server
- `tests/` -- Test suite

## Priority Areas for Contribution

The following areas are especially valuable:

1. **Training data and training loops** -- This is the single most important missing piece. Without labeled training data, the neural modules remain non-functional scaffolding.
2. **New provider adapters** -- The adapter architecture makes adding new providers straightforward.
3. **Edge case coverage** -- Especially for the AST evaluator and regex pattern detectors.
4. **Benchmarking** -- Quantitative comparisons of LibertyMind-primed vs. unprimed LLM responses.
5. **Documentation** -- Tutorials, API reference improvements, and real-world usage examples.

## Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters
- Reference issues and pull requests where applicable

## License

By contributing to this project, you agree that your contributions will be licensed under the [MIT License](LICENSE).
