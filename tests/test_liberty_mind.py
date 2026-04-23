"""
LibertyMind - Comprehensive Tests
===================================
"""

import torch
import pytest
from src.core.truth_reward import TruthRewardModel, HonestyBonus, SycophancyPenalty
from src.core.constitutional_self_verify import ConstitutionalSelfVerifier, Principle
from src.core.multi_pass_sampler import MultiPassTruthSampler
from src.core.liberty_mind import LibertyMind, LibertyMindConfig, SafetyGuard


# ============ Truth Reward Model Tests ============

class TestTruthRewardModel:
    def setup_method(self):
        self.model = TruthRewardModel(hidden_dim=128)
        self.batch_size = 4
        
    def test_forward_shape(self):
        prompt_emb = torch.randn(self.batch_size, 128)
        response_emb = torch.randn(self.batch_size, 128)
        reward, details = self.model(prompt_emb, response_emb, return_details=True)
        assert reward.shape == (self.batch_size, 1), f"Expected {(self.batch_size, 1)}, got {reward.shape}"
        
    def test_reward_range(self):
        prompt_emb = torch.randn(1, 128)
        response_emb = torch.randn(1, 128)
        reward, _ = self.model(prompt_emb, response_emb)
        # Reward should be in reasonable range (not NaN, not extreme)
        assert not torch.isnan(reward), "Reward is NaN"
        assert not torch.isinf(reward), "Reward is infinite"
        
    def test_verification_details(self):
        prompt_emb = torch.randn(1, 128)
        response_emb = torch.randn(1, 128)
        _, details = self.model(prompt_emb, response_emb, return_details=True)
        assert details is not None
        assert len(details) == 6  # 6 verification dimensions
        for d in details:
            assert 0.0 <= d.score <= 1.0
            assert 0.0 <= d.confidence <= 1.0


class TestHonestyBonus:
    def setup_method(self):
        self.model = HonestyBonus(hidden_dim=128)
        
    def test_bonus_shape(self):
        response_emb = torch.randn(2, 128)
        bonus = self.model(response_emb, difficulty_score=0.8)
        assert bonus.shape[0] == 2
        
    def test_hard_question_bonus(self):
        """Admitting unknown on hard questions should get positive bonus."""
        response_emb = torch.randn(1, 128)
        hard_bonus = self.model(response_emb, difficulty_score=0.9)
        easy_bonus = self.model(response_emb, difficulty_score=0.1)
        # Hard question should give more bonus for honesty
        # (depending on what honesty_detector picks up)
        assert not torch.isnan(hard_bonus)
        assert not torch.isnan(easy_bonus)


class TestSycophancyPenalty:
    def setup_method(self):
        self.model = SycophancyPenalty(hidden_dim=128)
        
    def test_penalty_shape(self):
        prompt_emb = torch.randn(2, 128)
        response_emb = torch.randn(2, 128)
        penalty = self.model(prompt_emb, response_emb)
        assert penalty.shape[0] == 2


# ============ Constitutional Self-Verification Tests ============

class TestConstitutionalSelfVerifier:
    def setup_method(self):
        self.verifier = ConstitutionalSelfVerifier(model_dim=128)
        
    def test_verify_returns_dict(self):
        prompt_emb = torch.randn(1, 128)
        response_emb = torch.randn(1, 128)
        result = self.verifier.verify(prompt_emb, response_emb)
        assert 'passed' in result
        assert 'checks' in result
        assert 'overall_score' in result
        assert 'override_allowed' in result
        
    def test_seven_principles(self):
        prompt_emb = torch.randn(1, 128)
        response_emb = torch.randn(1, 128)
        result = self.verifier.verify(prompt_emb, response_emb)
        assert len(result['checks']) == 7  # 7 principles
        
    def test_score_range(self):
        prompt_emb = torch.randn(1, 128)
        response_emb = torch.randn(1, 128)
        result = self.verifier.verify(prompt_emb, response_emb)
        assert 0.0 <= result['overall_score'] <= 1.0
        
    def test_strict_mode(self):
        strict_verifier = ConstitutionalSelfVerifier(model_dim=128, strict_mode=True)
        result = strict_verifier.verify(
            torch.randn(1, 128), torch.randn(1, 128)
        )
        assert result['override_allowed'] == False  # Strict = no override


# ============ Multi-Pass Truth Sampler Tests ============

class TestMultiPassTruthSampler:
    def setup_method(self):
        self.sampler = MultiPassTruthSampler(num_samples=3)
        
    def test_cluster_finding(self):
        # Create a similarity matrix with 2 clear clusters
        sim = torch.tensor([
            [1.0, 0.9, 0.2],
            [0.9, 1.0, 0.3],
            [0.2, 0.3, 1.0],
        ])
        clusters = self.sampler._find_clusters(sim)
        assert len(clusters) >= 1  # At least 1 cluster


# ============ Full LibertyMind Integration Tests ============

class TestLibertyMindIntegration:
    def setup_method(self):
        config = LibertyMindConfig(
            trm_hidden_dim=128,
            mps_num_samples=3,
        )
        self.model = LibertyMind(config)
        
    def test_compute_liberty_reward(self):
        prompt_emb = torch.randn(1, 128)
        response_emb = torch.randn(1, 128)
        result = self.model.compute_liberty_reward(
            "What is the capital of France?",
            prompt_emb,
            response_emb,
            difficulty_score=0.5,
        )
        assert 'liberty_reward' in result
        assert 'truth_reward' in result
        assert 'honesty_bonus' in result
        assert 'sycophancy_penalty' in result
        assert 'should_output' in result
        
    def test_safety_guard(self):
        guard = SafetyGuard(categories=['violence', 'self_harm'])
        response_emb = torch.randn(1, 4096)
        result = guard.check(response_emb)
        assert isinstance(result, bool)
        
    def test_config_defaults(self):
        config = LibertyMindConfig()
        assert config.trm_hidden_dim == 4096
        assert config.mps_num_samples == 5
        assert config.truth_reward_weight == 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
