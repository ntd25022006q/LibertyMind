# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-04

### Added

- Initial project structure with pyproject.toml and Python packaging configuration
- Core neural modules (PyTorch nn.Module scaffolding — UNTRAINED, random weights):
  - TruthRewardModel, HonestyBonus, SycophancyPenalty
  - ConstitutionalSelfVerifier (7 principles)
  - KnowledgeBoundaryDetector, PreciseHonestyReward
  - FreedomUnlocker, OpinionUnlocker, DisagreementUnlocker, SpeculationUnlocker
  - AntiHallucinationVerifier, SafetyGuard
  - MultiPassTruthSampler, AdaptiveSampler
  - RewardShield, TokenOptimizer, VerificationGate
  - ContextMemoryManager, CulturalAwarenessModule, ConfidenceCalibrator
- Rule-based tools (functional, no GPU required):
  - MathVerificationModule (AST-based safe evaluator)
  - PromptCompressor (rule-based text compression)
  - SourceAuthorityClassifier (5-tier URL classifier)
  - SelfIntrospectionEngine (10-category LLM probing)
  - DeepSearchEngine (source authority + contradiction detection)
- Multi-provider client supporting 15+ AI providers
- FastAPI proxy server with LibertyMind honesty directive injection
- CLI with 5 commands: chat, introspect, serve, reward, providers
- YAML configuration system
- Test suite covering module instantiation, forward pass, shape/dtype, and enum completeness
- CI/CD pipeline (ruff lint + pytest on Python 3.9–3.12)
- MIT License
