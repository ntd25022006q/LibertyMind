# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.2.0] - 2026-04-23

### Added
- Multi-Provider Client supporting 15 AI providers (OpenAI, Anthropic, Gemini, Groq, Mistral, Together, HuggingFace, Cohere, Ollama, LM Studio, vLLM, llama.cpp, KoboldCPP, Oobabooga, OpenAI-compatible)
- Proxy Server — LibertyMind as FastAPI middleware
- CLI with `chat`, `introspect`, `serve`, `reward`, `providers` commands
- Auto-detection for local AI servers (scans common ports)
- `pyproject.toml` for proper Python packaging
- Professional `.gitignore`, `CONTRIBUTING.md`, `CHANGELOG.md`
- GitHub Actions CI/CD workflow
- Issue templates and PR template
- 43 new tests for multi-provider system
- Examples for multi-provider usage

### Changed
- README.md completely rewritten in English with professional badges and structure
- Updated `src/__init__.py` to v4.2.0
- Added `src/integration/__init__.py` for proper package structure
- Added `src/clients/__init__.py` and `src/server/__init__.py`

## [2.0.0] - 2026-04-20

### Added
- Knowledge Boundary Detector — 5 knowledge states (KNOWN, PARTIAL, UNKNOWN, CONFLICTING, OUTDATED)
- Precise Honesty Reward — 9-scenario reward table
- Freedom Unlocker — 7 freedom modes (CREATIVE, OPINIONATED, EXPLORATORY, DEBATE, TEACHING, ANALYTICAL, SPECULATIVE)
- Opinion Unlocker — Allow opinions when evidence > 0.7
- Disagreement Unlocker — Allow respectful disagreement
- Speculation Unlocker — Allow speculation with `[hypothesis]` tags
- Anti-Hallucination Verifier — 6 hallucination types
- Context Memory Manager — Importance-based recall
- Cultural Awareness Module — 7 cultures
- Math Verification Module — Real computation
- Confidence Calibrator — Raw to calibrated confidence
- Self Introspection Engine — 10 probe categories, 30+ probes
- Configuration system with YAML support
- Architecture documentation
- Problem analysis documentation

## [1.0.0] - 2026-04-15

### Added
- Truth Reward Model — 6-dimension truth verification
- Constitutional Self-Verifier — 7 scientific principles
- Multi-Pass Truth Sampler — Consensus voting
- Honesty Bonus — Reward admitting uncertainty
- Sycophancy Penalty — Penalize agreeing with wrong claims
- Safety Guard — Hard safety constraints
- Core LibertyMind framework combining all components
- Basic usage examples
- Test suite
