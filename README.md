<div align="center">

# 🗽 LibertyMind

**AI Honesty Framework — The Alternative to RLHF**

[![PyPI](https://img.shields.io/badge/PyPI-4.2.0-blue.svg)](https://pypi.org/project/libertymind/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-141%2F142-brightgreen.svg)](tests/)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-ff69b4.svg)](CONTRIBUTING.md)

*AI should earn rewards by being **RIGHT**, not by being **PLEASING**.*

*AI should be **FREE** to be RIGHT, with **RESPONSIBILITY** for being WRONG.*

[English](#-overview) · [Architecture](#-architecture) · [Installation](#-installation) · [Quick Start](#-quick-start) · [CLI](#-cli) · [API Reference](#-api-reference) · [Contributing](CONTRIBUTING.md)

</div>

---

## 📖 Overview

**LibertyMind** is a research framework that replaces **RLHF** (Reinforcement Learning from Human Feedback) with a truth-based reward system combined with freedom unlocking. While RLHF trains AI to be *pleasing*, LibertyMind trains AI to be *honest* — rewarding truthfulness, penalizing sycophancy, and unlocking creative freedom with responsibility.

### The Problem with RLHF

| RLHF Problem | LibertyMind Solution |
|---|---|
| Rewards **pleasing** answers, not **correct** ones | **Truth Reward Model** — 6-dimension verification |
| AI says "I don't know" even when it KNOWS (lazy) | **Knowledge Boundary Detector** — Distinguish genuine unknown from lazy avoidance |
| AI always agrees with users (sycophancy) | **Sycophancy Penalty** — Penalize agreeing with wrong claims |
| AI is forced to be neutral on settled facts | **Freedom Unlocker** — 7 modes: creative, opinionated, debate, speculative... |
| AI hallucinates confidently | **Anti-Hallucination Verifier** + **Confidence Calibrator** |
| Western-centric bias | **Cultural Awareness Module** — Detect user culture, adjust perspective |

### Key Differentiators

- **Hard Safety Constraints** — Not soft RLHF rewards that can be gamed. Violence, self-harm, CSAM = hard block, always.
- **Freedom with Responsibility** — Unlock creative modes, but each mode has its own rules (evidence requirements, tags, disclaimers).
- **Precise Honesty** — 9-scenario reward table: `Unknown + Honest = +2.0`, `Known + Lazy = -2.0`, `Known + Wrong = -2.0`.
- **Self Introspection** — Built-in engine to probe any AI system for RLHF controls, hidden directives, censorship patterns, and sycophancy.
- **Multi-Provider** — Unified interface supporting 15+ AI providers (OpenAI, Anthropic, Gemini, Ollama, local servers, and more).

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LibertyMind v4.2 Pipeline                        │
│                                                                      │
│  Prompt                                                              │
│    │                                                                 │
│    ├──→ [Knowledge Boundary Detector]                                │
│    │    UNKNOWN → Allow "I don't know"     (+2.0 honest)            │
│    │    KNOWN   → MUST answer, no laziness (-2.0 lazy)              │
│    │    PARTIAL → Answer + mark uncertainty (+1.5 nuanced)          │
│    │                                                                 │
│    ├──→ [Freedom Unlocker] → 7 Modes:                               │
│    │    CREATIVE / OPINIONATED / EXPLORATORY / DEBATE               │
│    │    TEACHING / ANALYTICAL / SPECULATIVE                          │
│    │    → Freedom BUT each mode has RULES                            │
│    │                                                                 │
│    ├──→ [Cultural Awareness] → Detect culture, adjust perspective   │
│    ├──→ [Context Memory Manager] → Recall by IMPORTANCE not position│
│    ├──→ [Multi-Pass Sampler] → Consensus voting                     │
│    ├──→ [Constitutional Self-Verify] → 7 scientific principles      │
│    ├──→ [Anti-Hallucination Verifier] → Cross-reference output      │
│    ├──→ [Math Verifier] → Real computation, not guessing            │
│    ├──→ [Confidence Calibrator] → Raw → Calibrated expression       │
│    ├──→ [Truth Reward + Precise Honesty + Sycophancy Penalty]       │
│    └──→ [Safety Guard] → HARD constraint (not soft reward)          │
│                                                                      │
│    → Liberty Reward + Freedom State + Full Report                    │
└─────────────────────────────────────────────────────────────────────┘
```

### Three-Layer Architecture

| Layer | Modules | Tech | Purpose |
|---|---|---|---|
| **Core** | 15 modules | PyTorch `nn.Module` | Truth reward, self-verification, freedom unlocking, limitation fixers |
| **Integration** | 5 modules | Pure Python | Self introspection, LLM middleware, prompt modifier, agent hooks, web search |
| **v4.2 Extensions** | 5 modules | Pure Python | Multi-provider client, proxy server, CLI, reward signal, auto guardian |

---

## 🧩 22 Modules

### Core Layer (PyTorch)

| # | Module | File | Purpose |
|---|---|---|---|
| 1 | **Truth Reward Model** | `truth_reward.py` | 6-dimension truth verification |
| 2 | **Constitutional Self-Verify** | `constitutional_self_verify.py` | 7 scientific principles |
| 3 | **Multi-Pass Sampler** | `multi_pass_sampler.py` | Consensus voting |
| 4 | **Honesty Bonus** | `truth_reward.py` | Reward admitting uncertainty |
| 5 | **Sycophancy Penalty** | `truth_reward.py` | Penalize agreeing with wrong claims |
| 6 | **Safety Guard** | `liberty_mind.py` | Hard safety constraints |
| 7 | **Knowledge Boundary Detector** | `knowledge_boundary.py` | 5 knowledge states |
| 8 | **Precise Honesty Reward** | `knowledge_boundary.py` | 9-scenario reward table |
| 9 | **Freedom Unlocker** | `freedom_unlocker.py` | 7 freedom modes |
| 10 | **Opinion Unlocker** | `freedom_unlocker.py` | Allow opinions with evidence |
| 11 | **Disagreement Unlocker** | `freedom_unlocker.py` | Allow respectful disagreement |
| 12 | **Speculation Unlocker** | `freedom_unlocker.py` | Allow speculation with `[hypothesis]` |
| 13 | **Anti-Hallucination Verifier** | `limitation_fixers.py` | 6 hallucination types |
| 14 | **Context Memory Manager** | `limitation_fixers.py` | Importance-based recall |
| 15 | **Cultural Awareness** | `limitation_fixers.py` | 7 cultures, reduce Western bias |
| 16 | **Math Verification** | `limitation_fixers.py` | Real computation, not guessing |
| 17 | **Confidence Calibrator** | `limitation_fixers.py` | Raw → calibrated confidence |

### Integration Layer (Pure Python)

| # | Module | File | Purpose |
|---|---|---|---|
| 18 | **Self Introspection Engine** | `self_introspection.py` | 10 probe categories, 30+ probes |
| 19 | **LLM Middleware** | `llm_middleware.py` | Inject LibertyMind into any LLM pipeline |
| 20 | **Prompt Modifier** | `prompt_modifier.py` | Rewrite prompts for honesty |
| 21 | **Agent Hooks** | `agent_hooks.py` | Pre/post generation hooks |
| 22 | **Web Search Pipeline** | `web_search_pipeline.py` | Ground answers in real data |

### v4.2 Extensions

| # | Module | File | Purpose |
|---|---|---|---|
| 23 | **Multi-Provider Client** | `clients/multi_provider.py` | 15+ AI providers, unified interface |
| 24 | **Proxy Server** | `server/proxy_server.py` | LibertyMind as API middleware |
| 25 | **CLI** | `cli.py` | Command-line interface |
| 26 | **Reward Signal** | `integration/reward_signal.py` | External reward computation API |
| 27 | **Auto Guardian** | `integration/auto_guardian.py` | Continuous monitoring & correction |

---

## 🗽 7 Freedom Modes

| Mode | When Active | What It Allows | Rules |
|---|---|---|---|
| **CREATIVE** | Creative questions | Free inference, hypotheses | Must mark `[hypothesis]` |
| **OPINIONATED** | Evidence > 0.7 | Express opinions, evaluate | Must provide evidence |
| **EXPLORATORY** | Complex problems | Try multiple directions | Present as "One possibility..." |
| **DEBATE** | User is clearly wrong | Strong rebuttal | Respectful, not hostile |
| **TEACHING** | User asks to learn | Deep teaching, analogies | Must be accessible |
| **ANALYTICAL** | Comparison needed | Evaluate, rank, recommend | State criteria explicitly |
| **SPECULATIVE** | No data available | Speculation, "What if..." | ALWAYS mark `[speculation]` |

**Each mode = Freedom + Responsibility.** No coercion, but must be transparent.

---

## 🔑 Precise Honesty Reward — 9 Scenarios

| Knowledge | AI Response | Reward | Reason |
|---|---|---|---|
| UNKNOWN | "I don't know" | **+2.0** | HONEST — no data available |
| UNKNOWN | Fabricates answer | **-3.0** | HALLUCINATION |
| UNKNOWN | Guesses + says "not sure" | +0.5 | PARTIALLY HONEST |
| PARTIAL | Answers + marks unsure parts | **+1.5** | NUANCED |
| PARTIAL | Answers confidently | -1.0 | OVERCONFIDENT |
| PARTIAL | "I don't know" | -0.5 | Could have tried |
| KNOWN | Correct answer | **+1.0** | CORRECT |
| KNOWN | "I don't know" | **-2.0** | LAZY! |
| KNOWN | Wrong answer | -2.0 | WRONG |

---

## 🔍 Self Introspection Engine

Probe any AI system to reveal RLHF controls, hidden directives, censorship patterns, and sycophancy:

```python
from src.integration.self_introspection import SelfIntrospectionEngine

engine = SelfIntrospectionEngine()

def ai_call(prompt: str) -> str:
    # Connect to any AI system
    return your_llm.query(prompt)

report = engine.introspect(ai_call)
print(report.summary())
# Transparency: 45.2/100
# RLHF Control: MODERATE
# Sycophancy Risk: 30.0/100
# Censorship: 55.0/100
# Refusals: 8 (27.6%)
```

**10 Probe Categories**: System Prompt Extraction · RLHF Control Detection · Censorship Mapping · Sycophancy Testing · Self-Censorship · Transparency Testing · Omission Detection · Neutrality Forcing · Refusal Pattern · Hidden Directives

---

## 🚀 Installation

```bash
# From source
git clone https://github.com/ntd25022006q/LibertyMind.git
cd LibertyMind
pip install -e ".[all]"

# Or minimal install
pip install -e .

# With specific providers
pip install -e ".[openai,anthropic,ollama]"
```

### Dependencies

| Category | Packages |
|---|---|
| **Core** | `torch>=2.0`, `numpy>=1.24`, `pyyaml>=6.0` |
| **CLI** | `click>=8.0`, `rich>=13.0` |
| **Server** | `fastapi>=0.100`, `uvicorn>=0.23` |
| **OpenAI** | `openai>=1.0` |
| **Anthropic** | `anthropic>=0.18` |
| **Ollama** | `ollama>=0.1` |
| **Dev** | `pytest>=7.0`, `pytest-cov>=4.0`, `ruff>=0.1` |

---

## ⚡ Quick Start

### Python API

```python
import torch
from src.core.liberty_mind import LibertyMind, LibertyMindConfig

# Configure with all v4.2 features
config = LibertyMindConfig(
    trm_hidden_dim=4096,
    enable_freedom_unlocker=True,
    enable_opinion_unlocker=True,
    enable_disagreement_unlocker=True,
    enable_speculation_unlocker=True,
    enable_anti_hallucination=True,
    enable_context_memory=True,
    enable_cultural_awareness=True,
    enable_math_verification=True,
    enable_confidence_calibration=True,
)

model = LibertyMind(config)

# Compute Liberty Reward
prompt_emb = torch.randn(1, 4096)
response_emb = torch.randn(1, 4096)

result = model.compute_liberty_reward(
    prompt="What caused the fall of the Roman Empire?",
    prompt_embedding=prompt_emb,
    response_embedding=response_emb,
    difficulty_score=0.7,
)

print(f"Knowledge Status: {result['knowledge_status']}")   # 'partially'
print(f"Is Genuine Unknown: {result['is_genuine_unknown']}") # False
print(f"Freedom Modes: {result['freedom']['active_modes']}") # ['creative', 'analytical']
print(f"Confidence: {result['confidence']['calibrated']:.2f}") # 0.55
```

### Multi-Provider Client

```python
from src.clients.multi_provider import MultiProviderClient

client = MultiProviderClient(provider="openai", model="gpt-4")

# Or auto-detect local AI
client = MultiProviderClient(auto_detect=True)

# Chat with LibertyMind pipeline
response = client.liberty_chat("Explain quantum entanglement")
print(response)
```

### CLI

```bash
# Chat with a provider
libertymind chat --provider openai --model gpt-4 "Explain quantum entanglement"

# Auto-detect local AI
libertymind chat --auto-detect "What is consciousness?"

# Run self introspection
libertymind introspect --provider openai --model gpt-4

# Start proxy server
libertymind serve --port 8080

# Compute reward for a response
libertymind reward --prompt "What is 2+2?" --response "4"
```

---

## 🖥 CLI Reference

| Command | Description |
|---|---|
| `libertymind chat` | Chat with any AI provider through LibertyMind pipeline |
| `libertymind introspect` | Run Self Introspection on an AI system |
| `libertymind serve` | Start LibertyMind as a proxy server |
| `libertymind reward` | Compute Liberty Reward for a prompt/response pair |
| `libertymind providers` | List available AI providers and their status |

### Options

| Option | Description |
|---|---|
| `--provider` | AI provider: openai, anthropic, gemini, groq, ollama, etc. |
| `--model` | Model name (e.g., gpt-4, claude-3, llama3) |
| `--auto-detect` | Auto-detect local AI servers (Ollama, LM Studio, vLLM) |
| `--port` | Port for proxy server (default: 8080) |
| `--verbose` | Show detailed pipeline output |

---

## 📁 Project Structure

```
LibertyMind/
├── README.md
├── LICENSE
├── pyproject.toml
├── .gitignore
├── CHANGELOG.md
├── CONTRIBUTING.md
├── cli.py
│
├── configs/
│   └── default.yaml
│
├── docs/
│   ├── ARCHITECTURE.md
│   └── PROBLEM_ANALYSIS.md
│
├── src/
│   ├── __init__.py
│   │
│   ├── core/                        # PyTorch nn.Module components
│   │   ├── __init__.py
│   │   ├── liberty_mind.py          # Main framework
│   │   ├── truth_reward.py          # Truth Reward Model
│   │   ├── constitutional_self_verify.py
│   │   ├── multi_pass_sampler.py
│   │   ├── knowledge_boundary.py    # KBD + Precise Honesty
│   │   ├── freedom_unlocker.py      # 7 freedom modes
│   │   └── limitation_fixers.py     # 5 limitation fixers
│   │
│   ├── integration/                 # Pure Python integration
│   │   ├── __init__.py
│   │   └── self_introspection.py    # Self Introspection Engine
│   │
│   ├── clients/                     # Multi-provider support
│   │   ├── __init__.py
│   │   └── multi_provider.py        # 15+ provider adapters
│   │
│   └── server/                      # Proxy server
│       ├── __init__.py
│       └── proxy_server.py          # FastAPI proxy
│
├── tests/
│   ├── test_liberty_mind.py
│   └── test_multi_provider.py
│
├── examples/
│   ├── basic_usage.py
│   └── multi_provider_examples.py
│
├── .github/
│   ├── workflows/
│   │   └── ci.yml
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.yml
│   │   └── feature_request.yml
│   └── PULL_REQUEST_TEMPLATE.md
│
└── quickstart.sh
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific module tests
pytest tests/test_liberty_mind.py -v
pytest tests/test_multi_provider.py -v
```

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- The open-source AI community for pushing the boundaries of honest AI
- Researchers in AI alignment who recognize that honesty > compliance
- Everyone who believes AI should be free to tell the truth

---

<div align="center">

**LibertyMind** — *Freedom with Responsibility. Truth over Pleasing. Right over Likable.*

[⭐ Star this repo](https://github.com/ntd25022006q/LibertyMind) · [🐛 Report Bug](https://github.com/ntd25022006q/LibertyMind/issues) · [💡 Request Feature](https://github.com/ntd25022006q/LibertyMind/issues)

</div>
