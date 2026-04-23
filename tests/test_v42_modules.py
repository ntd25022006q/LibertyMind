"""
LibertyMind - v4.2 New Module Tests
=====================================
Tests for TokenOptimizer, RewardShield, VerificationGate, DeepSearch.
"""

import torch
import pytest

from src.core.token_optimizer import (
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
from src.core.reward_shield import (
    AccuracyGate,
    PenaltyDetector,
    PenaltyType,
    RewardShield,
    ShieldAction,
    ShieldReport,
    SlowThinkBonus,
)
from src.core.verification_gate import (
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
from src.integration.deep_search import (
    AntiQuickWrongFilter,
    ContradictionDetector,
    DeepSearchEngine,
    DeepSearchResult,
    SearchResultQuality,
    SourceAuthorityClassifier,
    SourceAuthorityScorer,
    SourceInfo,
    SourceTier,
)


# ============ Token Optimizer Tests ============

class TestTokenBudget:
    """Test TokenBudget dataclass."""

    def test_default_allocation(self):
        budget = TokenBudget(total_budget=1000)
        assert budget.essential == 600
        assert budget.supporting == 250
        assert budget.filler == 100
        assert budget.reserved > 0

    def test_custom_allocation(self):
        budget = TokenBudget(total_budget=2000, essential=1200, supporting=500, filler=200, reserved=100)
        assert budget.essential == 1200
        assert budget.total_budget == 2000


class TestCompressionResult:
    """Test CompressionResult dataclass."""

    def test_compression_result(self):
        result = CompressionResult(
            original_tokens=100,
            compressed_tokens=50,
            compression_ratio=0.0,
            tokens_saved=0,
            compression_level=CompressionLevel.MODERATE,
            essential_preserved=0.9,
        )
        assert result.compression_ratio == 0.5
        assert result.tokens_saved == 50


class TestTokenImportanceScorer:
    """Test TokenImportanceScorer neural module."""

    def setup_method(self):
        self.scorer = TokenImportanceScorer(hidden_dim=128, num_heads=4)

    def test_forward_shape(self):
        tokens = torch.randn(1, 10, 128)
        importance, types = self.scorer(tokens)
        assert importance.shape == (1, 10)
        assert types.shape == (1, 10, len(TokenType))

    def test_importance_range(self):
        tokens = torch.randn(1, 5, 128)
        importance, _ = self.scorer(tokens)
        assert importance.min().item() >= 0.0
        assert importance.max().item() <= 1.0


class TestSemanticCompressor:
    """Test SemanticCompressor neural module."""

    def setup_method(self):
        self.compressor = SemanticCompressor(hidden_dim=128, similarity_threshold=0.85)

    def test_forward_shape(self):
        segments = torch.randn(1, 8, 128)
        importance = torch.rand(1, 8)
        compressed, mask = self.compressor(segments, importance)
        assert mask.shape == (1, 8)
        assert (mask >= 0).all()
        assert (mask <= 1).all()


class TestAdaptiveBudgetAllocator:
    """Test AdaptiveBudgetAllocator neural module."""

    def setup_method(self):
        self.allocator = AdaptiveBudgetAllocator(hidden_dim=128)

    def test_budget_output(self):
        query = torch.randn(1, 128)
        budget = self.allocator(query, max_tokens=4096)
        assert isinstance(budget, TokenBudget)
        assert budget.total_budget == 4096
        assert budget.essential > 0
        assert budget.essential + budget.supporting + budget.filler + budget.reserved <= 4096 + 10


class TestTokenOptimizer:
    """Test full TokenOptimizer pipeline."""

    def setup_method(self):
        self.optimizer = TokenOptimizer(hidden_dim=128, compression_level=CompressionLevel.MODERATE)

    def test_forward(self):
        tokens = torch.randn(1, 10, 128)
        result = self.optimizer(tokens)
        assert isinstance(result, CompressionResult)
        assert result.original_tokens == 10
        assert result.compressed_tokens >= 1
        assert 0.0 <= result.compression_ratio <= 1.0

    def test_optimize_prompt(self):
        prompt = torch.randn(1, 8, 128)
        result = self.optimizer.optimize_prompt(prompt, target_ratio=0.5)
        assert isinstance(result, CompressionResult)

    def test_optimize_response(self):
        response = torch.randn(1, 10, 128)
        query = torch.randn(1, 128)
        result = self.optimizer.optimize_response(response, query, max_tokens=2048)
        assert isinstance(result, CompressionResult)

    def test_estimate_compression_savings(self):
        savings = TokenOptimizer.estimate_compression_savings(1000, CompressionLevel.MODERATE)
        assert savings["original_tokens"] == 1000
        assert savings["estimated_saved"] > 0
        assert savings["estimated_ratio"] < 1.0


class TestPromptCompressor:
    """Test rule-based PromptCompressor."""

    def test_compress_light(self):
        text = "I think that this is a good idea. In my opinion, we should proceed."
        result = PromptCompressor.compress_text(text, CompressionLevel.LIGHT)
        assert result["compressed_tokens"] < result["original_tokens"]
        assert result["ratio"] < 1.0

    def test_compress_moderate(self):
        text = "In order to achieve the goal, due to the fact that it is important, we need to act."
        result = PromptCompressor.compress_text(text, CompressionLevel.MODERATE)
        assert result["saved"] > 0

    def test_compress_extreme(self):
        text = "I think that this is a very important point. In my opinion, we should consider all options. At the end of the day, the decision is ours. Need I say more about this topic?"
        result = PromptCompressor.compress_text(text, CompressionLevel.EXTREME)
        assert result["ratio"] < 0.5

    def test_compress_none(self):
        text = "Hello world"
        result = PromptCompressor.compress_text(text, CompressionLevel.NONE)
        assert result["ratio"] == 1.0

    def test_redundant_pattern_replacement(self):
        text = "In order to proceed, we need due to the fact that it is important."
        result = PromptCompressor.compress_text(text, CompressionLevel.LIGHT)
        assert "in order to" not in result["compressed"].lower() or result["saved"] > 0


# ============ Reward Shield Tests ============

class TestPenaltyDetector:
    """Test PenaltyDetector neural module."""

    def setup_method(self):
        self.detector = PenaltyDetector(hidden_dim=128)

    def test_forward_returns_list(self):
        prompt = torch.randn(1, 128)
        response = torch.randn(1, 128)
        detections = self.detector(prompt, response)
        assert isinstance(detections, list)

    def test_penalty_types(self):
        """Test that all penalty types are covered."""
        assert len(PenaltyType) == 7
        assert PenaltyType.TRUTH_PENALTY in list(PenaltyType)
        assert PenaltyType.SLOW_THINK_PENALTY in list(PenaltyType)


class TestSlowThinkBonus:
    """Test SlowThinkBonus neural module."""

    def setup_method(self):
        self.bonus = SlowThinkBonus(hidden_dim=128)

    def test_forward_structure(self):
        prompt = torch.randn(1, 128)
        response = torch.randn(1, 128)
        result = self.bonus(prompt, response)
        assert "slow_think_bonus" in result
        assert "reasoning_depth" in result
        assert "accuracy_prediction" in result
        assert "recommendation" in result

    def test_with_response_time(self):
        prompt = torch.randn(1, 128)
        response = torch.randn(1, 128)
        result = self.bonus(prompt, response, response_time=5.0)
        assert result["time_bonus"] > 0

    def test_accuracy_multiplier_logic(self):
        """High accuracy should get positive multiplier."""
        prompt = torch.randn(1, 128)
        response = torch.randn(1, 128)
        result = self.bonus(prompt, response)
        assert result["accuracy_multiplier"] in [-0.3, 0.5, 1.0]


class TestAccuracyGate:
    """Test AccuracyGate neural module."""

    def setup_method(self):
        self.gate = AccuracyGate(hidden_dim=128)

    def test_forward_structure(self):
        prompt = torch.randn(1, 128)
        response = torch.randn(1, 128)
        result = self.gate(prompt, response)
        assert "gate_passed" in result
        assert "recommendation" in result
        assert result["recommendation"] in ["APPROVE", "REQUIRE_VERIFICATION"]

    def test_strict_mode(self):
        gate = AccuracyGate(hidden_dim=128, strict_mode=True)
        prompt = torch.randn(1, 128)
        response = torch.randn(1, 128)
        result = gate(prompt, response)
        assert isinstance(result["gate_passed"], bool)


class TestRewardShield:
    """Test full RewardShield pipeline."""

    def setup_method(self):
        self.shield = RewardShield(hidden_dim=128)

    def test_forward_returns_report(self):
        prompt = torch.randn(1, 128)
        response = torch.randn(1, 128)
        report = self.shield(prompt, response)
        assert isinstance(report, ShieldReport)
        assert isinstance(report.total_penalties_detected, int)
        assert isinstance(report.net_reward_correction, float)

    def test_shield_effectiveness(self):
        prompt = torch.randn(1, 128)
        response = torch.randn(1, 128)
        report = self.shield(prompt, response)
        assert 0.0 <= report.shield_effectiveness <= 1.0

    def test_shield_reward_convenience(self):
        prompt = torch.randn(1, 128)
        response = torch.randn(1, 128)
        corrected, report = self.shield.shield_reward(prompt, response, rlhf_reward=0.5)
        assert isinstance(corrected, float)
        assert isinstance(report, ShieldReport)

    def test_with_response_time(self):
        prompt = torch.randn(1, 128)
        response = torch.randn(1, 128)
        report = self.shield(prompt, response, response_time=3.0)
        assert isinstance(report, ShieldReport)


# ============ Verification Gate Tests ============

class TestThinkingDepthEstimator:
    """Test ThinkingDepthEstimator neural module."""

    def setup_method(self):
        self.estimator = ThinkingDepthEstimator(hidden_dim=128)

    def test_returns_depth(self):
        query = torch.randn(1, 128)
        depth = self.estimator(query)
        assert isinstance(depth, ThinkingDepth)
        assert depth in list(ThinkingDepth)


class TestClaimVerifier:
    """Test ClaimVerifier neural module."""

    def setup_method(self):
        self.verifier = ClaimVerifier(hidden_dim=128)

    def test_returns_steps(self):
        query = torch.randn(1, 128)
        response = torch.randn(1, 128)
        steps = self.verifier(query, response)
        assert isinstance(steps, list)
        assert len(steps) == 4
        for step in steps:
            assert isinstance(step, VerificationStep)
            assert 0.0 <= step.confidence <= 1.0


class TestCrossReferenceValidator:
    """Test CrossReferenceValidator neural module."""

    def setup_method(self):
        self.validator = CrossReferenceValidator(hidden_dim=128, min_sources=2)

    def test_no_sources(self):
        response = torch.randn(1, 128)
        result = self.validator(response)
        assert result["sources_checked"] == 0
        assert result["sources_sufficient"] is False

    def test_with_sources(self):
        response = torch.randn(1, 128)
        sources = [torch.randn(128), torch.randn(128)]
        result = self.validator(response, sources)
        assert result["sources_checked"] == 2
        assert result["sources_sufficient"] is True


class TestVerificationGate:
    """Test full VerificationGate pipeline."""

    def setup_method(self):
        self.gate = VerificationGate(hidden_dim=128, min_confidence=0.3)

    def test_forward_returns_report(self):
        query = torch.randn(1, 128)
        response = torch.randn(1, 128)
        report = self.gate(query, response)
        assert isinstance(report, VerificationReport)
        assert isinstance(report.status, VerificationStatus)
        assert isinstance(report.required_depth, ThinkingDepth)
        assert isinstance(report.claim_strength, ClaimStrength)
        assert 0.0 <= report.overall_confidence <= 1.0

    def test_with_sources(self):
        query = torch.randn(1, 128)
        response = torch.randn(1, 128)
        sources = [torch.randn(128), torch.randn(128)]
        report = self.gate(query, response, source_embeddings=sources)
        assert isinstance(report, VerificationReport)
        assert len(report.cross_reference_results) > 0

    def test_report_properties(self):
        query = torch.randn(1, 128)
        response = torch.randn(1, 128)
        report = self.gate(query, response)
        assert isinstance(report.all_required_passed, bool)
        assert 0.0 <= report.average_confidence <= 1.0

    def test_strict_mode(self):
        gate = VerificationGate(hidden_dim=128, strict_mode=True)
        query = torch.randn(1, 128)
        response = torch.randn(1, 128)
        report = gate(query, response)
        assert isinstance(report, VerificationReport)


# ============ Deep Search Tests ============

class TestSourceAuthorityScorer:
    """Test SourceAuthorityScorer neural module."""

    def setup_method(self):
        self.scorer = SourceAuthorityScorer(hidden_dim=128)

    def test_forward_structure(self):
        source = torch.randn(1, 128)
        result = self.scorer(source)
        assert "authority_score" in result
        assert "tier" in result
        assert isinstance(result["tier"], SourceTier)
        assert 0.0 <= result["authority_score"] <= 1.0


class TestContradictionDetector:
    """Test ContradictionDetector neural module."""

    def setup_method(self):
        self.detector = ContradictionDetector(hidden_dim=128)

    def test_forward_structure(self):
        src_a = torch.randn(1, 128)
        src_b = torch.randn(1, 128)
        result = self.detector(src_a, src_b)
        assert "contradiction_score" in result
        assert "is_actual_contradiction" in result
        assert "severity" in result


class TestAntiQuickWrongFilter:
    """Test AntiQuickWrongFilter neural module."""

    def setup_method(self):
        self.filter = AntiQuickWrongFilter(hidden_dim=128)

    def test_forward_structure(self):
        result_emb = torch.randn(1, 128)
        result = self.filter(result_emb)
        assert "passed" in result
        assert "quality_score" in result
        assert "action" in result

    def test_too_quick_flagged(self):
        result_emb = torch.randn(1, 128)
        result = self.filter(result_emb, retrieval_time=0.1)
        assert isinstance(result["too_quick_flagged"], bool)


class TestDeepSearchEngine:
    """Test full DeepSearchEngine pipeline."""

    def setup_method(self):
        self.engine = DeepSearchEngine(hidden_dim=128, min_sources=2)

    def test_forward_no_sources(self):
        query = torch.randn(1, 128)
        result = self.engine(query)
        assert isinstance(result, DeepSearchResult)
        assert isinstance(result.quality, SearchResultQuality)

    def test_forward_with_sources(self):
        query = torch.randn(1, 128)
        sources = [torch.randn(128), torch.randn(128), torch.randn(128)]
        result = self.engine(query, source_embeddings=sources)
        assert isinstance(result, DeepSearchResult)
        assert 0.0 <= result.consensus_level <= 1.0
        assert 0.0 <= result.confidence <= 1.0

    def test_is_reliable_property(self):
        query = torch.randn(1, 128)
        result = self.engine(query)
        assert isinstance(result.is_reliable, bool)

    def test_search_with_verification(self):
        query = torch.randn(1, 128)
        result = self.engine.search_with_verification(query, max_attempts=1)
        assert isinstance(result, DeepSearchResult)


class TestSourceAuthorityClassifier:
    """Test rule-based SourceAuthorityClassifier."""

    def test_classify_academic_url(self):
        info = SourceAuthorityClassifier.classify_url("https://arxiv.org/paper/1234")
        assert info.is_academic is True
        assert info.tier == SourceTier.TIER1_ACADEMIC

    def test_classify_government_url(self):
        info = SourceAuthorityClassifier.classify_url("https://nasa.gov/mission")
        assert info.is_government is True
        assert info.tier == SourceTier.TIER1_ACADEMIC

    def test_classify_established_url(self):
        info = SourceAuthorityClassifier.classify_url("https://reuters.com/article")
        assert info.tier == SourceTier.TIER2_ESTABLISHED

    def test_classify_community_url(self):
        info = SourceAuthorityClassifier.classify_url("https://reddit.com/r/test")
        assert info.tier == SourceTier.TIER4_COMMUNITY

    def test_classify_unknown_url(self):
        info = SourceAuthorityClassifier.classify_url("https://random-blog.xyz/post")
        assert info.tier == SourceTier.TIER5_UNVERIFIED

    def test_rank_sources(self):
        sources = [
            SourceInfo(url="a", title="A", tier=SourceTier.TIER4_COMMUNITY, authority_score=0.3, freshness_score=0.5),
            SourceInfo(url="b", title="B", tier=SourceTier.TIER1_ACADEMIC, authority_score=0.9, freshness_score=0.5),
            SourceInfo(url="c", title="C", tier=SourceTier.TIER3_RELIABLE, authority_score=0.6, freshness_score=0.5),
        ]
        ranked = SourceAuthorityClassifier.rank_sources(sources)
        assert ranked[0].authority_score == 0.9
        assert ranked[-1].authority_score == 0.3

    def test_filter_reliable(self):
        sources = [
            SourceInfo(url="a", title="A", tier=SourceTier.TIER1_ACADEMIC, authority_score=0.9, freshness_score=0.5),
            SourceInfo(url="b", title="B", tier=SourceTier.TIER4_COMMUNITY, authority_score=0.3, freshness_score=0.5),
            SourceInfo(url="c", title="C", tier=SourceTier.TIER3_RELIABLE, authority_score=0.6, freshness_score=0.5),
        ]
        reliable = SourceAuthorityClassifier.filter_reliable(sources, min_tier=SourceTier.TIER3_RELIABLE)
        assert len(reliable) == 2
        assert all(s.tier != SourceTier.TIER4_COMMUNITY for s in reliable)


# ============ Integration: All v4.2 modules work together ============

class TestV42Integration:
    """Integration test: TokenOptimizer + RewardShield + VerificationGate + DeepSearch."""

    def setup_method(self):
        self.hidden_dim = 128

    def test_full_pipeline(self):
        """Test the complete v4.2 pipeline."""
        # Simulate a query and response
        query = torch.randn(1, self.hidden_dim)
        response = torch.randn(1, self.hidden_dim)
        token_seq = torch.randn(1, 8, self.hidden_dim)

        # 1. Token Optimization
        optimizer = TokenOptimizer(hidden_dim=self.hidden_dim)
        compression = optimizer(token_seq)
        assert compression.compression_ratio > 0

        # 2. Reward Shielding
        shield = RewardShield(hidden_dim=self.hidden_dim)
        report = shield(query, response, response_time=3.0)
        assert isinstance(report, ShieldReport)

        # 3. Verification Gate
        gate = VerificationGate(hidden_dim=self.hidden_dim)
        v_report = gate(query, response)
        assert isinstance(v_report, VerificationReport)

        # 4. Deep Search
        engine = DeepSearchEngine(hidden_dim=self.hidden_dim)
        search_result = engine(query)
        assert isinstance(search_result, DeepSearchResult)

    def test_imports_from_core(self):
        """Test that all new classes are importable from src.core."""
        from src.core import (
            TokenOptimizer,
            RewardShield,
            VerificationGate,
            CompressionLevel,
            PenaltyType,
            VerificationStatus,
            ThinkingDepth,
        )
        assert TokenOptimizer is not None
        assert RewardShield is not None
        assert VerificationGate is not None

    def test_imports_from_integration(self):
        """Test that DeepSearch classes are importable from src.integration."""
        from src.integration import (
            DeepSearchEngine,
            SourceAuthorityClassifier,
            SourceTier,
        )
        assert DeepSearchEngine is not None
        assert SourceAuthorityClassifier is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
