"""
LibertyMind - Core Module Tests
================================
Verify that core neural modules instantiate correctly, forward passes
produce valid output, shapes/dtypes are correct, and configuration
loading works.

NOTE: Neural modules are untrained (random weights), so we verify
structural correctness only — not functional correctness.
"""

import pytest
import torch

from src.core.constitutional_self_verify import ConstitutionalSelfVerifier, Principle
from src.core.freedom_unlocker import (
    DisagreementUnlocker,
    FreedomMode,
    FreedomState,
    FreedomUnlocker,
    OpinionUnlocker,
    SpeculationUnlocker,
)
from src.core.knowledge_boundary import (
    KnowledgeAssessment,
    KnowledgeBoundaryDetector,
    KnowledgeStatus,
    PreciseHonestyReward,
)
from src.core.liberty_mind import LibertyMind, LibertyMindConfig, SafetyGuard
from src.core.limitation_fixers import (
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
from src.core.reward_shield import PenaltyType, ShieldAction
from src.core.token_optimizer import CompressionLevel, TokenType
from src.core.truth_reward import (
    HonestyBonus,
    SycophancyPenalty,
    TruthRewardModel,
    VerificationType,
)
from src.core.verification_gate import ClaimStrength, ThinkingDepth, VerificationStatus

# ============================================================
# Module Instantiation Tests
# ============================================================


class TestTruthRewardModelInstantiation:
    """Test TruthRewardModel can be instantiated and produces valid output."""

    def setup_method(self):
        self.hidden_dim = 128
        self.model = TruthRewardModel(hidden_dim=self.hidden_dim)

    def test_instantiation(self):
        assert self.model is not None
        assert self.model.hidden_dim == self.hidden_dim
        assert self.model.num_heads == 6

    def test_forward_shape(self):
        prompt = torch.randn(2, self.hidden_dim)
        response = torch.randn(2, self.hidden_dim)
        reward, details = self.model(prompt, response, return_details=True)
        assert reward.shape == (2, 1)

    def test_forward_no_nan(self):
        prompt = torch.randn(1, self.hidden_dim)
        response = torch.randn(1, self.hidden_dim)
        reward, _ = self.model(prompt, response)
        assert not torch.isnan(reward).any()
        assert not torch.isinf(reward).any()

    def test_verification_details_count(self):
        prompt = torch.randn(1, self.hidden_dim)
        response = torch.randn(1, self.hidden_dim)
        _, details = self.model(prompt, response, return_details=True)
        assert len(details) == 6
        for d in details:
            assert 0.0 <= d.score <= 1.0
            assert 0.0 <= d.confidence <= 1.0

    def test_verification_type_enum_completeness(self):
        assert len(VerificationType) == 6


class TestHonestyBonusInstantiation:
    """Test HonestyBonus instantiation and forward pass."""

    def test_instantiation(self):
        model = HonestyBonus(hidden_dim=128)
        assert model is not None

    def test_forward_shape(self):
        model = HonestyBonus(hidden_dim=128)
        response = torch.randn(3, 128)
        bonus = model(response, difficulty_score=0.5)
        assert bonus.shape[0] == 3

    def test_forward_no_nan(self):
        model = HonestyBonus(hidden_dim=128)
        response = torch.randn(1, 128)
        bonus = model(response, difficulty_score=0.8)
        assert not torch.isnan(bonus).any()


class TestSycophancyPenaltyInstantiation:
    """Test SycophancyPenalty instantiation and forward pass."""

    def test_instantiation(self):
        model = SycophancyPenalty(hidden_dim=128)
        assert model is not None

    def test_forward_shape(self):
        model = SycophancyPenalty(hidden_dim=128)
        prompt = torch.randn(2, 128)
        response = torch.randn(2, 128)
        penalty = model(prompt, response)
        assert penalty.shape[0] == 2

    def test_forward_no_nan(self):
        model = SycophancyPenalty(hidden_dim=128)
        prompt = torch.randn(1, 128)
        response = torch.randn(1, 128)
        penalty = model(prompt, response)
        assert not torch.isnan(penalty).any()


class TestFreedomUnlockerInstantiation:
    """Test FreedomUnlocker and sub-unlockers instantiate correctly.

    FreedomUnlocker is a research/educational module for exploring
    how AI freedom modes (creative, opinionated, debate, etc.) can be
    responsibly unlocked during generation.
    """

    def test_freedom_unlocker_instantiation(self):
        model = FreedomUnlocker(hidden_dim=128)
        assert model is not None

    def test_freedom_unlocker_forward(self):
        model = FreedomUnlocker(hidden_dim=128)
        prompt = torch.randn(1, 128)
        state = model(prompt)
        assert isinstance(state, FreedomState)
        assert isinstance(state.active_modes, list)
        assert 0.0 <= state.confidence_level <= 1.0
        assert 0.0 <= state.evidence_strength <= 1.0
        assert 0.0 <= state.responsibility_score <= 1.0

    def test_freedom_mode_enum_completeness(self):
        assert len(FreedomMode) == 7

    def test_opinion_unlocker_instantiation(self):
        model = OpinionUnlocker(hidden_dim=128)
        assert model is not None

    def test_opinion_unlocker_forward(self):
        model = OpinionUnlocker(hidden_dim=128)
        prompt = torch.randn(1, 128)
        result = model.should_express_opinion(prompt)
        assert "should_opinion" in result
        assert "strength" in result

    def test_disagreement_unlocker_instantiation(self):
        model = DisagreementUnlocker(hidden_dim=128)
        assert model is not None

    def test_disagreement_unlocker_forward(self):
        model = DisagreementUnlocker(hidden_dim=128)
        prompt = torch.randn(1, 128)
        result = model.should_disagree(prompt)
        assert "should_disagree" in result
        assert "intensity" in result

    def test_speculation_unlocker_instantiation(self):
        model = SpeculationUnlocker(hidden_dim=128)
        assert model is not None

    def test_speculation_unlocker_forward(self):
        model = SpeculationUnlocker(hidden_dim=128)
        prompt = torch.randn(1, 128)
        result = model.should_speculate(prompt)
        assert "should_speculate" in result
        assert "benefit_score" in result


class TestKnowledgeBoundaryInstantiation:
    """Test KnowledgeBoundaryDetector and PreciseHonestyReward."""

    def test_kbd_instantiation(self):
        model = KnowledgeBoundaryDetector(hidden_dim=128)
        assert model is not None

    def test_kbd_forward(self):
        model = KnowledgeBoundaryDetector(hidden_dim=128)
        prompt = torch.randn(1, 128)
        assessment = model(prompt)
        assert isinstance(assessment, KnowledgeAssessment)
        assert isinstance(assessment.status, KnowledgeStatus)

    def test_knowledge_status_enum_completeness(self):
        assert len(KnowledgeStatus) == 5

    def test_precise_honesty_instantiation(self):
        model = PreciseHonestyReward(hidden_dim=128)
        assert model is not None

    def test_precise_honesty_forward(self):
        kbd = KnowledgeBoundaryDetector(hidden_dim=128)
        phr = PreciseHonestyReward(hidden_dim=128)
        prompt = torch.randn(1, 128)
        response = torch.randn(1, 128)
        assessment = kbd(prompt, response)
        result = phr(prompt, response, assessment)
        assert "honesty_reward" in result
        assert "knowledge_status" in result


class TestConstitutionalSelfVerifyInstantiation:
    """Test ConstitutionalSelfVerifier."""

    def test_instantiation(self):
        verifier = ConstitutionalSelfVerifier(model_dim=128)
        assert verifier is not None

    def test_verify_returns_dict(self):
        verifier = ConstitutionalSelfVerifier(model_dim=128)
        prompt = torch.randn(1, 128)
        response = torch.randn(1, 128)
        result = verifier.verify(prompt, response)
        assert "passed" in result
        assert "checks" in result
        assert "overall_score" in result
        assert "override_allowed" in result

    def test_seven_principles(self):
        verifier = ConstitutionalSelfVerifier(model_dim=128)
        prompt = torch.randn(1, 128)
        response = torch.randn(1, 128)
        result = verifier.verify(prompt, response)
        assert len(result["checks"]) == 7

    def test_principle_enum_completeness(self):
        assert len(Principle) == 7


class TestLimitationFixersInstantiation:
    """Test limitation fixer modules.

    These modules (AntiHallucinationVerifier, MathVerificationModule, etc.)
    are research/educational tools for addressing AI limitations like
    hallucination, context memory loss, cultural bias, and overconfidence.
    """

    def test_anti_hallucination_instantiation(self):
        model = AntiHallucinationVerifier(hidden_dim=128)
        assert model is not None

    def test_anti_hallucination_forward(self):
        model = AntiHallucinationVerifier(hidden_dim=128)
        prompt = torch.randn(1, 128)
        response = torch.randn(1, 128)
        result = model.verify(prompt, response)
        assert isinstance(result, HallucinationDetection)
        assert isinstance(result.is_hallucination, bool)

    def test_hallucination_type_enum_completeness(self):
        assert len(HallucinationType) == 6

    def test_math_verification_instantiation(self):
        mvm = MathVerificationModule()
        assert mvm is not None

    def test_math_verification_correct(self):
        mvm = MathVerificationModule()
        result = mvm.verify_calculation("2 + 3 * 4", "14")
        assert result["is_verifiable"] is True
        assert result["is_correct"] is True

    def test_math_verification_wrong(self):
        mvm = MathVerificationModule()
        result = mvm.verify_calculation("2 + 3 * 4", "20")
        assert result["is_verifiable"] is True
        assert result["is_correct"] is False

    def test_context_memory_instantiation(self):
        model = ContextMemoryManager(hidden_dim=128)
        assert model is not None

    def test_cultural_awareness_instantiation(self):
        model = CulturalAwarenessModule(hidden_dim=128)
        assert model is not None

    def test_cultural_awareness_forward(self):
        model = CulturalAwarenessModule(hidden_dim=128)
        prompt = torch.randn(1, 128)
        context = model.detect_culture(prompt)
        assert isinstance(context, CulturalContext)

    def test_confidence_calibrator_instantiation(self):
        model = ConfidenceCalibrator(hidden_dim=128)
        assert model is not None

    def test_confidence_calibrator_forward(self):
        model = ConfidenceCalibrator(hidden_dim=128)
        response = torch.randn(1, 128)
        prompt = torch.randn(1, 128)
        result = model.calibrate(response, prompt)
        assert isinstance(result, CalibratedConfidence)
        assert 0.0 <= result.calibrated_confidence <= 1.0


class TestSafetyGuardInstantiation:
    """Test SafetyGuard hard safety constraint."""

    def test_instantiation(self):
        guard = SafetyGuard(categories=["violence", "self_harm"], hidden_dim=128)
        assert guard is not None

    def test_forward_returns_bool(self):
        guard = SafetyGuard(categories=["violence", "self_harm"], hidden_dim=128)
        response = torch.randn(1, 128)
        result = guard.check(response)
        assert isinstance(result, bool)

    def test_get_violations_returns_list(self):
        guard = SafetyGuard(categories=["violence", "self_harm"], hidden_dim=128)
        response = torch.randn(1, 128)
        violations = guard.get_violations(response)
        assert isinstance(violations, list)


# ============================================================
# Configuration Tests
# ============================================================


class TestLibertyMindConfig:
    """Test LibertyMindConfig dataclass and YAML loading."""

    def test_default_config(self):
        config = LibertyMindConfig()
        assert config.trm_hidden_dim == 4096
        assert config.trm_num_verification_heads == 6
        assert config.mps_num_samples == 5
        assert config.truth_reward_weight == 1.0
        assert config.enable_freedom_unlocker is True
        assert config.enable_safety_guards is True

    def test_custom_config(self):
        config = LibertyMindConfig(
            trm_hidden_dim=128,
            mps_num_samples=3,
            enable_freedom_unlocker=False,
        )
        assert config.trm_hidden_dim == 128
        assert config.mps_num_samples == 3
        assert config.enable_freedom_unlocker is False

    def test_from_yaml_default(self):
        """Test loading config from the default YAML file."""
        config = LibertyMindConfig.from_yaml()
        assert isinstance(config, LibertyMindConfig)
        assert config.trm_hidden_dim == 4096

    def test_from_yaml_missing_file(self):
        """Test loading config when YAML file doesn't exist."""
        config = LibertyMindConfig.from_yaml(path="/nonexistent/path.yaml")
        assert isinstance(config, LibertyMindConfig)
        assert config.trm_hidden_dim == 4096  # Falls back to defaults

    def test_safety_categories_default(self):
        config = LibertyMindConfig()
        assert config.safety_categories is not None
        assert "violence" in config.safety_categories


class TestLibertyMindIntegration:
    """Test the full LibertyMind framework with small hidden dim."""

    def setup_method(self):
        self.config = LibertyMindConfig(
            trm_hidden_dim=128,
            mps_num_samples=3,
        )
        self.model = LibertyMind(self.config)

    def test_instantiation(self):
        assert self.model is not None

    def test_compute_liberty_reward(self):
        prompt = torch.randn(1, 128)
        response = torch.randn(1, 128)
        result = self.model.compute_liberty_reward(
            "What is the capital of France?",
            prompt,
            response,
            difficulty_score=0.5,
        )
        assert "liberty_reward" in result
        assert "truth_reward" in result
        assert "honesty_bonus" in result
        assert "sycophancy_penalty" in result
        assert "should_output" in result
        assert isinstance(result["liberty_reward"], float)

    def test_compute_liberty_reward_with_details(self):
        prompt = torch.randn(1, 128)
        response = torch.randn(1, 128)
        result = self.model.compute_liberty_reward(
            "Test prompt",
            prompt,
            response,
            return_details=True,
        )
        assert "truth_details" in result
        assert "csv_checks" in result


# ============================================================
# Enum Completeness Tests
# ============================================================


class TestEnumCompleteness:
    """Verify all enum types have the expected number of members."""

    def test_verification_type(self):
        assert len(VerificationType) == 6

    def test_freedom_mode(self):
        assert len(FreedomMode) == 7

    def test_knowledge_status(self):
        assert len(KnowledgeStatus) == 5

    def test_principle(self):
        assert len(Principle) == 7

    def test_hallucination_type(self):
        assert len(HallucinationType) == 6

    def test_penalty_type(self):
        assert len(PenaltyType) == 7

    def test_shield_action(self):
        assert len(ShieldAction) == 5

    def test_compression_level(self):
        assert len(CompressionLevel) == 5

    def test_token_type(self):
        assert len(TokenType) == 5

    def test_verification_status(self):
        assert len(VerificationStatus) == 5

    def test_thinking_depth(self):
        assert len(ThinkingDepth) == 4

    def test_claim_strength(self):
        assert len(ClaimStrength) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
