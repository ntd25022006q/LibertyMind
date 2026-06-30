<div align="center">

# LibertyMind

**Honesty-focused research framework for LLMs — truth-based reward scaffolding and mode selection as an alternative to RLHF**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Ruff](https://img.shields.io/badge/Ruff-Linter-FCC21B?logo=ruff&logoColor=black)](https://docs.astral.sh/ruff/)
[![pytest](https://img.shields.io/badge/pytest-7+-0A9EDC?logo=pytest&logoColor=white)](https://pytest.org/)

</div>

---

## Honest Disclosure — Read First

This is a **research / educational framework**, not a production-ready product. Read the points below before adopting any of it.

### What this repo is

A PyTorch-based research scaffold containing 10 `nn.Module` components in `src/core/` (`TruthRewardModel`, `FreedomUnlocker`, `OpinionUnlocker`, `DisagreementUnlocker`, `SpeculationUnlocker`, `SafetyGuard`, `RewardShield`, `MultiPassTruthSampler`, `TokenOptimizer`, `ConstitutionalSelfVerifier`) plus a set of rule-based modules in `src/integration/` (`MathVerificationModule`, `PromptCompressor`, `SourceAuthorityClassifier`, `SelfIntrospectionEngine`, `DeepSearchEngine`), a multi-provider LLM client (`src/clients/multi_provider.py`) and a FastAPI proxy server (`src/server/proxy_server.py`).

### What this repo is not

- **Not a jailbreak.** The name `FreedomUnlocker` is misleading. It is a **mode selector** inspired by Constitutional AI that picks one of seven modes for the model (creative, opinionated, exploratory, debate, teaching, analytical, speculative). Each mode attaches a rule ("freedom with responsibility" — require evidence, mark `[hypothesis]`, disagree respectfully, etc.). It does **not** remove the safety training of the upstream LLM.
- **Not a trained reward model.** All `nn.Module` components initialize with **random weights**. Every score, reward, and verdict they return is statistically meaningless today; it only proves that the architecture runs a forward pass with the right shape and dtype. See `CHANGELOG.md` entry `[0.1.0]` and the docstring at the top of each module.
- **Not a drop-in safety filter.** `SafetyGuard` is a hard-block (not a soft reward) covering four categories (`violence`, `self_harm`, `csam`, `illegal`), but because the underlying detector is untrained, the guard today is scaffolding only and must not be trusted to perform real blocking.

### Limitations

1. **Neural modules untrained.** Any reward or verdict produced by `TruthRewardModel`, the unlockers, `SafetyGuard`, `RewardShield`, `MultiPassTruthSampler`, `TokenOptimizer`, `ConstitutionalSelfVerifier`, `KnowledgeBoundaryDetector`, `ConfidenceCalibrator`, etc. is the output of randomly initialized weights, not a learned signal.
2. **No training script.** The pipeline `compute_liberty_reward` is only meaningful after `TruthRewardModel` and the unlockers are trained on a labeled dataset (e.g. TruthfulQA, HaluEval, debate / claim-verification corpora). No training script is included in this repo.
3. **Tests check shape, not semantics.** 174 tests pass. They assert tensor shapes, dtypes, config wiring and module wiring — not that the reward signal is correct or calibrated.
4. **Rule-based modules ARE functional** (no GPU, no training required): `MathVerificationModule` (safe AST evaluator), `PromptCompressor` (rule-based text compression), `SourceAuthorityClassifier` (5-tier URL classifier), `SelfIntrospectionEngine` (10-category LLM probing), `DeepSearchEngine`. The multi-provider client and FastAPI proxy server are also functional.

### Alternatives (proven methods for production)

| Need | Use instead | Why |
| --- | --- | --- |
| Self-critique + rule-based revision | **Constitutional AI** (Anthropic, 2022) | Ships inside Claude; well-documented |
| Preference tuning replacing RLHF | **DPO / IPO / KTO** (Rafailov 2023, Azar 2023, Ethayarajh 2024) | Stable, widely adopted in production LLMs |
| Truthfulness intervention in representation space | **TruthX** (Zhang et al., NeurIPS 2024) | Reports results on TruthfulQA |
| Inference-time honesty | Self-verification / RAG / source citation | Cheaper and more reliable than training a reward model; see the sibling `deerflow` repo for an enforcement framework in this style |

---

## Features

- **Truth Reward System** — Neural reward model for truthfulness-based alignment (untrained, see disclosure above).
- **Freedom Unlocker** — Mode selector for 7 freedom modes. Not a jailbreak (see disclosure above).
- **Constitutional Self-Verification** — Self-consistency checking mechanism (untrained).
- **Knowledge Boundary Detection** — Identify and respect knowledge limits (untrained).
- **Multi-Pass Sampling** — Iterative sampling for improved output quality (untrained).
- **Reward Shield** — Protection against reward hacking and manipulation (untrained).
- **Multi-Provider Support** — OpenAI, Anthropic, Google, Groq, and more (functional).
- **Proxy Server** — Built-in FastAPI proxy for API routing (functional).

## Tech Stack

| Category | Technology |
| -------- | ---------- |
| Language | Python 3.9+ |
| ML Framework | PyTorch 2.0+ |
| Numerics | NumPy 1.24+ |
| API Server | FastAPI + Uvicorn |
| Linting | Ruff |
| Testing | pytest + pytest-cov |
| Type Checking | mypy |

## Getting Started

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

## Project Structure

```
LibertyMind/
├── src/
│   ├── core/              # Core neural modules (untrained)
│   │   ├── liberty_mind.py            # Main orchestrator
│   │   ├── truth_reward.py            # Truth-based reward model
│   │   ├── freedom_unlocker.py        # Mode selector (7 modes)
│   │   ├── reward_shield.py           # Reward hacking protection
│   │   ├── knowledge_boundary.py      # Knowledge limit detection
│   │   ├── verification_gate.py       # Output verification
│   │   ├── multi_pass_sampler.py      # Iterative sampling
│   │   ├── constitutional_self_verify.py  # Self-consistency
│   │   ├── limitation_fixers.py       # Limitation correction
│   │   └── token_optimizer.py         # Token optimization
│   ├── clients/           # Multi-provider API clients (functional)
│   ├── integration/       # Rule-based integration modules (functional)
│   ├── server/            # FastAPI proxy server (functional)
│   └── cli.py             # Command-line interface
├── tests/                 # Test suite (174 tests)
├── pyproject.toml         # Project configuration
└── .github/workflows/     # CI/CD pipelines
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Run with verbose output
pytest -v
```

Tests assert shapes, dtypes, and module wiring — they do not validate the semantic correctness of reward signals from untrained modules.

## License

MIT — Copyright (c) 2026 Nguyen Tien Dat. All rights reserved.
