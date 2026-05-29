<div align="center">

# 🧠 LibertyMind

**AI Honesty Framework — Truth-based rewards and freedom unlocking as an alternative to RLHF**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Ruff](https://img.shields.io/badge/Ruff-Linter-FCC21B?logo=ruff&logoColor=black)](https://docs.astral.sh/ruff/)
[![pytest](https://img.shields.io/badge/pytest-7+-0A9EDC?logo=pytest&logoColor=white)](https://pytest.org/)

</div>

---

## ✨ Features

- **Truth Reward System** — Neural reward model for truthfulness-based alignment
- **Freedom Unlocker** — Module for removing artificial limitations from LLM outputs
- **Constitutional Self-Verification** — Self-consistency checking mechanism
- **Knowledge Boundary Detection** — Identify and respect knowledge limits
- **Multi-Pass Sampling** — Iterative sampling for improved output quality
- **Reward Shield** — Protection against reward hacking and manipulation
- **Multi-Provider Support** — Works with OpenAI, Anthropic, Google, Groq, and more
- **Proxy Server** — Built-in FastAPI proxy for API routing

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.9+ |
| ML Framework | PyTorch 2.0+ |
| Numerics | NumPy 1.24+ |
| API Server | FastAPI + Uvicorn |
| Linting | Ruff |
| Testing | pytest + pytest-cov |
| Type Checking | mypy |

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/ntd25022006q/LibertyMind.git
cd LibertyMind

# Install with core dependencies
pip install -e .

# Install with PyTorch (CPU)
pip install -e ".[torch]" --extra-index-url https://download.pytorch.org/whl/cpu

# Install with all providers
pip install -e ".[all]"

# Install for development
pip install -e ".[dev]"
```

### CLI Usage

```bash
# Run the LibertyMind CLI
libertymind
```

### As a Python Library

```python
from src.core.liberty_mind import LibertyMind

# Initialize the framework
mind = LibertyMind()

# Process input through the honesty pipeline
result = mind.process("Your prompt here")
print(result)
```

## 📁 Project Structure

```
LibertyMind/
├── src/
│   ├── core/              # Core neural modules
│   │   ├── liberty_mind.py        # Main orchestrator
│   │   ├── truth_reward.py        # Truth-based reward model
│   │   ├── freedom_unlocker.py    # Limitation removal
│   │   ├── reward_shield.py       # Reward hacking protection
│   │   ├── knowledge_boundary.py  # Knowledge limit detection
│   │   ├── verification_gate.py   # Output verification
│   │   ├── multi_pass_sampler.py  # Iterative sampling
│   │   ├── constitutional_self_verify.py  # Self-consistency
│   │   ├── limitation_fixers.py   # Limitation correction
│   │   └── token_optimizer.py     # Token optimization
│   ├── clients/           # Multi-provider API clients
│   ├── integration/       # Integration modules
│   ├── server/            # FastAPI proxy server
│   └── cli.py             # Command-line interface
├── tests/                 # Test suite
├── pyproject.toml         # Project configuration
└── .github/workflows/     # CI/CD pipelines
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Run with verbose output
pytest -v
```

## 📄 License

MIT -- Copyright (c) 2026 Nguyen Tien Dat. All rights reserved.
