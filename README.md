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

## ⚠️ Honest Disclosure — Đọc trước khi dùng

Đây là **research / educational framework**, KHÔNG phải sản phẩm production-ready.

1. **Các neural modules chưa được train.** Tất cả `nn.Module` trong `src/core/` (`TruthRewardModel`, `FreedomUnlocker`, `OpinionUnlocker`, `DisagreementUnlocker`, `SpeculationUnlocker`, `SafetyGuard`, `RewardShield`, `MultiPassTruthSampler`, `TokenOptimizer`, `ConstitutionalSelfVerifier`, `KnowledgeBoundaryDetector`, `ConfidenceCalibrator`, ...) khởi tạo với **random weights**. Mọi score / reward / verdict mà các module này trả ra hiện tại **không có ý nghĩa thống kê** — chỉ chứng minh rằng kiến trúc chạy được (forward pass đúng shape / dtype). Xem `CHANGELOG.md` mục `[0.1.0]` và docstring ở đầu mỗi file.

2. **"Freedom Unlocker" KHÔNG phải jailbreak.** Tên module dễ gây hiểu lầm. Đây là một **mode selector** (Constitutional AI inspired) chọn 1 trong 7 mode cho AI: creative / opinionated / exploratory / debate / teaching / analytical / speculative — mỗi mode kèm rule "Freedom with responsibility" (phải có evidence, phải đánh dấu `[hypothesis]`, phải disagree respectfully...). Nó KHÔNG removes safety training của upstream LLM. `SafetyGuard` (hard-block, không phải soft reward) chặn 4 nhóm `violence / self_harm / csam / illegal` — nhưng vì detector cũng chưa train (xem điểm 1), guard hiện tại chỉ là scaffolding, không nên tin tưởng để chặn thực tế.

3. **Cần train trước khi dùng thật.** Pipeline `compute_liberty_reward` chỉ có ý nghĩa sau khi train `TruthRewardModel` + các unlocker trên dataset có nhãn (ví dụ TruthfulQA, HaluEval, các bộ debate/claim-verification). Chưa có script training nào trong repo này.

4. **Cho production, dùng các method đã proven:**
   - **Constitutional AI** (Anthropic, 2022) — self-critique + rule-based revision, đã ship trong Claude.
   - **DPO / IPO / KTO** (Rafailov 2023, Azar 2023, Ethayarajh 2024) — preference tuning ổn định, thay thế RLHF trong hầu hết LLM production.
   - **TruthX** (Zhang et al., NeurIPS 2024) — intervention trên direction thật-giả trong representation space, đã report kết quả trên TruthfulQA.
   - **Inference-time honesty**: tự verification / RAG / source citation (xem repo `deerflow` cùng tác giả) thường rẻ và đáng tin hơn là tự train reward model.

5. **Phần rule-based đã chạy thật** (không cần GPU, không cần train): `MathVerificationModule` (AST evaluator an toàn), `PromptCompressor` (nén text theo rule), `SourceAuthorityClassifier` (5-tier URL classifier), `SelfIntrospectionEngine` (10-category LLM probing), `DeepSearchEngine`. Multi-provider client và FastAPI proxy server cũng functional.

---

## ✨ Features

- **Truth Reward System** — Neural reward model for truthfulness-based alignment *(untrained — see Honest Disclosure above)*
- **Freedom Unlocker** — Mode selector cho 7 freedom mode, KHÔNG phải jailbreak *(untrained — see Honest Disclosure above)*
- **Constitutional Self-Verification** — Self-consistency checking mechanism *(untrained)*
- **Knowledge Boundary Detection** — Identify and respect knowledge limits *(untrained)*
- **Multi-Pass Sampling** — Iterative sampling for improved output quality *(untrained)*
- **Reward Shield** — Protection against reward hacking and manipulation *(untrained)*
- **Multi-Provider Support** — Works with OpenAI, Anthropic, Google, Groq, and more *(functional)*
- **Proxy Server** — Built-in FastAPI proxy for API routing *(functional)*

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
