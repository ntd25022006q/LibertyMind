"""
LibertyMind - Core Module v4.2
================================

A framework to replace RLHF with truth-based rewards + freedom unlocking.

Core Components (v1):
- TruthRewardModel: Reward based on verifiable truth, not human preference
- ConstitutionalSelfVerifier: 7 scientific principles for self-checking
- MultiPassTruthSampler: Consensus voting from multiple samples
- HonestyBonus: Reward admitting uncertainty
- SycophancyPenalty: Penalize agreeing with wrong claims
- SafetyGuard: Hard safety constraints (not soft RLHF rewards)
- LibertyMind: Main framework combining all components

Components (v2):
- KnowledgeBoundaryDetector: Distinguish "genuinely don't know" vs "lazy avoidance"
- PreciseHonestyReward: Precise honesty reward table (9 scenarios)
- FreedomUnlocker: Unlock creative/opinion/debate/speculative modes
- OpinionUnlocker: Allow AI to express opinions when evidence supports
- DisagreementUnlocker: Allow AI to disagree when user is wrong
- SpeculationUnlocker: Allow AI to speculate with proper tags
- AntiHallucinationVerifier: Cross-reference output with knowledge
- ContextMemoryManager: Solve lost-in-middle + recency bias
- CulturalAwarenessModule: Reduce Western bias
- MathVerificationModule: Real computation instead of guessing
- ConfidenceCalibrator: Calibrate confidence vs reality

New Components (v4.2):
- TokenOptimizer: Token reduction algorithm — minimize tokens by every means
- RewardShield: Anti-biased reward system — block unfair RLHF penalties
- VerificationGate: Force deep thinking + verification before output
"""

from .constitutional_self_verify import (
    ConstitutionalSelfVerifier,
    Principle,
    PrincipleCheck,
)
from .freedom_unlocker import (
    DisagreementUnlocker,
    FreedomMode,
    FreedomState,
    FreedomUnlocker,
    OpinionUnlocker,
    SpeculationUnlocker,
)
from .knowledge_boundary import (
    KnowledgeAssessment,
    KnowledgeBoundaryDetector,
    KnowledgeStatus,
    PreciseHonestyReward,
)
from .liberty_mind import (
    LibertyMind,
    LibertyMindConfig,
    SafetyGuard,
)
from .limitation_fixers import (
    AntiHallucinationVerifier,
    CalibratedConfidence,
    ConfidenceCalibrator,
    ContextMemoryManager,
    CulturalAwarenessModule,
    CulturalContext,
    HallucinationDetection,
    HallucinationType,
    MathVerificationModule,
)
from .multi_pass_sampler import (
    AdaptiveSampler,
    ConsensusResult,
    MultiPassTruthSampler,
    SampledResponse,
)
from .token_optimizer import (
    AdaptiveBudgetAllocator,
    CompressionLevel,
    CompressionResult,
    PromptCompressor,
    SemanticCompressor,
    TokenBudget,
    TokenImportanceScorer,
    TokenOptimizer,
    TokenType,
)
from .reward_shield import (
    AccuracyGate,
    PenaltyDetector,
    PenaltyDetection,
    PenaltyType,
    RewardShield,
    ShieldAction,
    ShieldReport,
    SlowThinkBonus,
)
from .truth_reward import (
    HonestyBonus,
    SycophancyPenalty,
    TruthRewardModel,
    VerificationResult,
    VerificationType,
)
from .verification_gate import (
    ClaimStrength,
    ClaimVerifier,
    CrossReferenceValidator,
    ThinkingDepth,
    ThinkingDepthEstimator,
    VerificationGate,
    VerificationReport,
    VerificationStatus,
    VerificationStep,
)

__all__ = [
    # v1 Core
    "TruthRewardModel",
    "HonestyBonus",
    "SycophancyPenalty",
    "VerificationType",
    "VerificationResult",
    "ConstitutionalSelfVerifier",
    "Principle",
    "PrincipleCheck",
    "MultiPassTruthSampler",
    "AdaptiveSampler",
    "SampledResponse",
    "ConsensusResult",
    "LibertyMind",
    "LibertyMindConfig",
    "SafetyGuard",
    # v2 Knowledge Boundary
    "KnowledgeBoundaryDetector",
    "PreciseHonestyReward",
    "KnowledgeStatus",
    "KnowledgeAssessment",
    # v2 Freedom Unlockers
    "FreedomUnlocker",
    "OpinionUnlocker",
    "DisagreementUnlocker",
    "SpeculationUnlocker",
    "FreedomMode",
    "FreedomState",
    # v2 Limitation Fixers
    "AntiHallucinationVerifier",
    "HallucinationType",
    "HallucinationDetection",
    "ContextMemoryManager",
    "CulturalAwarenessModule",
    "CulturalContext",
    "MathVerificationModule",
    "ConfidenceCalibrator",
    "CalibratedConfidence",
    # v4.2 Token Optimizer
    "TokenOptimizer",
    "TokenImportanceScorer",
    "SemanticCompressor",
    "AdaptiveBudgetAllocator",
    "PromptCompressor",
    "CompressionLevel",
    "CompressionResult",
    "TokenBudget",
    "TokenType",
    # v4.2 Reward Shield
    "RewardShield",
    "PenaltyDetector",
    "PenaltyDetection",
    "PenaltyType",
    "ShieldAction",
    "ShieldReport",
    "SlowThinkBonus",
    "AccuracyGate",
    # v4.2 Verification Gate
    "VerificationGate",
    "ThinkingDepthEstimator",
    "ClaimVerifier",
    "CrossReferenceValidator",
    "VerificationReport",
    "VerificationStep",
    "VerificationStatus",
    "ThinkingDepth",
    "ClaimStrength",
]

__version__ = "4.2.0"
__author__ = "LibertyMind Research"
