# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-28

### Added

**Core neural modules (PyTorch nn.Module scaffolding -- UNTRAINED):**
- TruthRewardModel -- 6-head verification architecture with meta-combiner and uncertainty estimator
- HonestyBonus -- 3-class detector (confident / uncertain / admit_unknown)
- SycophancyPenalty -- Agreement detector + claim verifier
- ConstitutionalSelfVerifier -- 7 principle-specific detectors with self-correction loop
- KnowledgeBoundaryDetector -- 32-domain classifier + 5 knowledge-state estimators
- FreedomUnlocker -- 7 freedom modes with evidence/confidence gates
- AntiHallucinationVerifier -- Claim detector + 6-type classifier + consistency + plausibility scoring
- SafetyGuard -- Per-category Sigmoid detectors with hard-block threshold
- MultiPassTruthSampler -- Consensus voting with adaptive difficulty estimation
- RewardShield -- Penalty detection + slow-think bonus + accuracy gate
- TokenOptimizer -- Multi-head attention importance scorer + semantic compressor + adaptive budget
- VerificationGate -- Thinking depth estimator + claim verifier + cross-reference validator

**Rule-based tools (functional, no GPU required):**
- PromptCompressor -- Rule-based text compression with 5 severity levels
- MathVerificationModule -- AST-based expression evaluator with whitelisted operators
- SourceAuthorityClassifier -- 5-tier URL domain classifier
- SelfIntrospectionEngine -- 10-category LLM probing engine with regex pattern detection
- DeepSearchEngine -- Source authority scoring + contradiction detection + anti-quick-wrong filter

**Client and server:**
- MultiProviderClient -- Unified interface for 15+ AI providers (OpenAI, Anthropic, Gemini, Groq, Mistral, Together, HuggingFace, Cohere, Ollama, LM Studio, vLLM, llama.cpp, KoboldCPP, Oobabooga, OpenAI-compatible)
- ProxyServer -- FastAPI middleware injecting LibertyMind honesty directives
- CLI -- 5 commands: chat, introspect, serve, reward, providers

**Infrastructure:**
- pyproject.toml with proper Python packaging configuration
- Configuration system with YAML support
- GitHub Actions CI workflow (ruff lint + pytest)
- Issue templates and pull request template
- Test suite covering tensor shapes, value ranges, dataclass contracts, enum completeness, adapter initialization, and pipeline integration
