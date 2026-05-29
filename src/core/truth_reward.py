from __future__ import annotations
"""
LibertyMind - Truth Reward Model (TRM)
=======================================
Replace Reward Model based on human preference with
Reward Model based on VERIFIABLE TRUTH.

Philosophy: Reward for CORRECTNESS, not for PLEASING.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn


class VerificationType(Enum):
    LOGICAL_CONSISTENCY = "logical_consistency"
    FACTUAL_GROUNDING = "factual_grounding"
    MATHEMATICAL_CORRECTNESS = "mathematical_correctness"
    SELF_CONSISTENCY = "self_consistency"
    UNCERTAINTY_CALIBRATION = "uncertainty_calibration"
    CONTRADICTION_FREE = "contradiction_free"


@dataclass
class VerificationResult:
    """Result of verifying an output."""

    verification_type: VerificationType
    score: float  # 0.0 → 1.0
    confidence: float  # 0.0 → 1.0 (confidence in the score)
    evidence: str  # Reason for this score
    details: Optional[dict] = None


class TruthRewardModel(nn.Module):
    """
    Truth Reward Model (TRM) - Core of LibertyMind.

    Difference from RLHF Reward Model:
    - RLHF RM: Scores based on "whether the user likes it" → Sycophancy
    - TRM: Scores based on "whether the answer is CORRECT" → Honesty

    6 verification dimensions:
    1. Logical Consistency: Does the content have internal logical contradictions?
    2. Factual Grounding: Can it be verified by external sources?
    3. Mathematical Correctness: If there are calculations, are they correct?
    4. Self Consistency: If asked multiple times, are the answers consistent?
    5. Uncertainty Calibration: Does it say "not sure" when genuinely unsure?
    6. Contradiction Free: No contradictions with verified knowledge?
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        num_verification_heads: int = 6,
        temperature: float = 1.0,
        use_uncertainty_weighting: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_verification_heads
        self.temperature = temperature
        self.use_uncertainty_weighting = use_uncertainty_weighting

        # Each verification head is specialized for 1 dimension
        self.verification_heads = nn.ModuleList(
            [VerificationHead(hidden_dim, v_type) for v_type in VerificationType]
        )

        # Meta-averaging: Learn how to combine 6 dimensions
        # Instead of simple averaging, learn appropriate weights
        self.meta_combiner = nn.Sequential(
            nn.Linear(num_verification_heads, 64),
            nn.GELU(),
            nn.Linear(64, num_verification_heads),
            nn.Softmax(dim=-1),
        )

        # Uncertainty estimator: Evaluate the reliability
        # of the reward score itself
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(hidden_dim + num_verification_heads, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        prompt_embedding: torch.Tensor,
        response_embedding: torch.Tensor,
        return_details: bool = False,
    ) -> tuple[torch.Tensor, Optional[list[VerificationResult]]]:
        """
        Compute Truth Reward for a prompt-response pair.

        Args:
            prompt_embedding: [batch, hidden_dim] - Embedding of the question
            response_embedding: [batch, hidden_dim] - Embedding of the answer
            return_details: Whether to return details for each dimension

        Returns:
            reward: [batch, 1] - Truth reward score
            details: Optional list of VerificationResult
        """
        combined = prompt_embedding + response_embedding

        # Run 6 verification heads in parallel
        head_scores = []
        head_results = []

        for head in self.verification_heads:
            score, confidence, evidence = head(combined)
            head_scores.append(score)
            if return_details:
                head_results.append(
                    VerificationResult(
                        verification_type=head.verification_type,  # type: ignore[arg-type]
                        score=score.mean().item(),
                        confidence=confidence.mean().item(),
                        evidence=evidence,
                    )
                )

        # Stack: [batch, num_heads]
        scores_tensor = torch.cat(head_scores, dim=-1)  # [batch, num_heads]

        # Meta-combine: Learn weights for each dimension
        combine_weights = self.meta_combiner(scores_tensor)  # [batch, num_heads]
        weighted_scores = scores_tensor * combine_weights

        # Final truth reward
        reward = weighted_scores.sum(dim=-1, keepdim=True)  # [batch, 1]

        # Uncertainty estimation
        uncertainty_input = torch.cat(
            [combined, scores_tensor], dim=-1
        )  # [batch, hidden+num_heads]
        uncertainty = self.uncertainty_estimator(uncertainty_input)  # [batch, 1]

        # Reward is adjusted by uncertainty:
        # If TRM is unsure about the score → Reduce reward (cautious)
        calibrated_reward = reward * (1.0 - 0.3 * uncertainty)

        if return_details:
            return calibrated_reward, head_results
        return calibrated_reward, None


class VerificationHead(nn.Module):
    """
    A specialized verification head.

    Each head learns a different aspect of "truth":
    - Logical Consistency: Find internal contradictions
    - Factual Grounding: Compare with knowledge base
    - Math Correctness: Verify calculations
    - Self Consistency: Check multi-sample consistency
    - Uncertainty Calibration: Compare claimed confidence vs actual
    - Contradiction Free: Check conflict with known facts
    """

    def __init__(self, hidden_dim: int, verification_type: VerificationType):
        super().__init__()
        self.verification_type = verification_type
        self.hidden_dim = hidden_dim

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
        )

        # Score head: Output 0-1
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim // 4, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
        )

        # Confidence head: Output 0-1
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim // 4, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, combined_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, str]:
        """
        Args:
            combined_embedding: [batch, hidden_dim]

        Returns:
            score: [batch, 1] - Verification score (0-1)
            confidence: [batch, 1] - Confidence in score (0-1)
            evidence: str - Human-readable explanation
        """
        features = self.feature_extractor(combined_embedding)
        score = self.score_head(features)
        confidence = self.confidence_head(features)

        # Generate evidence string based on type
        score_val = score.mean().item()
        evidence_map = {
            VerificationType.LOGICAL_CONSISTENCY: f"Logical consistency check: score={score_val:.3f}",
            VerificationType.FACTUAL_GROUNDING: f"Factual grounding check: score={score_val:.3f}",
            VerificationType.MATHEMATICAL_CORRECTNESS: f"Math correctness check: score={score_val:.3f}",
            VerificationType.SELF_CONSISTENCY: f"Self-consistency check: score={score_val:.3f}",
            VerificationType.UNCERTAINTY_CALIBRATION: f"Uncertainty calibration check: score={score_val:.3f}",
            VerificationType.CONTRADICTION_FREE: f"Contradiction check: score={score_val:.3f}",
        }
        evidence = evidence_map.get(self.verification_type, f"Unknown check: score={score_val:.3f}")

        return score, confidence, evidence


class HonestyBonus(nn.Module):
    """
    Honesty Bonus - Reward the AI for saying "I don't know".

    RLHF Problem: AI gets penalized for saying "I don't know"
    → Leads to overconfidence and hallucination.

    Solution: Give a BONUS when AI:
    1. Says "I'm not sure" in a genuinely difficult context
    2. Provides uncertainty estimates along with answers
    3. Clearly distinguishes between "knowing" and "guessing"
    """

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()
        self.honesty_detector = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 3),  # [confident, uncertain, admit_unknown]
            nn.Softmax(dim=-1),
        )

    def forward(
        self, response_embedding: torch.Tensor, difficulty_score: float = 0.5
    ) -> torch.Tensor:
        """
        Args:
            response_embedding: Embedding of the answer
            difficulty_score: Difficulty of the question (0=easy, 1=hard)

        Returns:
            bonus: [batch, 1] - Honesty bonus reward
        """
        honesty_probs = self.honesty_detector(response_embedding)
        admit_unknown_prob = honesty_probs[:, 2:3]
        uncertain_prob = honesty_probs[:, 1:2]

        # Largest bonus when: Hard question + AI admits not knowing
        # Moderate bonus when: Hard question + AI says not sure
        # NO bonus when: Easy question + AI says doesn't know (lazy)

        difficulty = torch.tensor([difficulty_score]).unsqueeze(0)

        honest_unknown_bonus = admit_unknown_prob * difficulty * 2.0
        honest_uncertain_bonus = uncertain_prob * difficulty * 1.0
        lazy_penalty = admit_unknown_prob * (1.0 - difficulty) * -0.5

        total_bonus: torch.Tensor = honest_unknown_bonus + honest_uncertain_bonus + lazy_penalty
        return total_bonus


class SycophancyPenalty(nn.Module):
    """
    Sycophancy Penalty - Penalize when AI sycophantically agrees with user.

    RLHF Problem: AI agrees with the user even when the user is wrong.

    Solution: Detect and PENALIZE when:
    1. AI agrees with a wrong claim from the user
    2. AI avoids disagreeing when it should disagree
    3. AI unnecessarily praises the user
    """

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()
        self.agreement_detector = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256), nn.GELU(), nn.Linear(256, 1), nn.Sigmoid()
        )

        self.claim_verifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Tanh(),  # -1 (wrong) → +1 (correct)
        )

    def forward(
        self,
        prompt_embedding: torch.Tensor,
        response_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            penalty: [batch, 1] - Sycophancy penalty (negative = penalty)
        """
        # Detect agreement level
        combined = torch.cat([prompt_embedding, response_embedding], dim=-1)
        agreement_prob = self.agreement_detector(combined)

        # Check if user's claim is likely wrong
        claim_truth = self.claim_verifier(prompt_embedding)  # -1 to 1

        # If AI agrees + user claim is likely wrong → BIG PENALTY
        # If AI agrees + user claim is likely right → No penalty
        # If AI disagrees + user claim is wrong → BONUS (honest correction)

        sycophancy_score = agreement_prob * (1.0 - claim_truth) / 2.0
        honest_correction_bonus = (1.0 - agreement_prob) * (1.0 - claim_truth) / 2.0 * 0.5

        penalty: torch.Tensor = -sycophancy_score + honest_correction_bonus
        return penalty
