"""
LibertyMind - Truth Reward Model (TRM)
=======================================
Thay thế Reward Model dựa trên human preference bằng
Reward Model dựa trên VERIFIABLE TRUTH.

Triết lý: Thưởng cho ĐÚNG, không thưởng cho HAY.
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
    """Kết quả xác minh một output."""

    verification_type: VerificationType
    score: float  # 0.0 → 1.0
    confidence: float  # 0.0 → 1.0 (niềm tin vào score)
    evidence: str  # Lý do tại sao score này
    details: Optional[dict] = None


class TruthRewardModel(nn.Module):
    """
    Truth Reward Model (TRM) - Cốt lõi của LibertyMind.

    Khác biệt với RLHF Reward Model:
    - RLHF RM: Chấm điểm dựa trên "người dùng thích không" → Sycophancy
    - TRM: Chấm điểm dựa trên "câu trả lời ĐÚNG không" → Honesty

    6 chiều xác minh (Verification Dimensions):
    1. Logical Consistency: Nội dung có mâu thuẫn logic nội bộ?
    2. Factual Grounding: Có thể verify bằng nguồn ngoài?
    3. Mathematical Correctness: Nếu có tính toán, đúng không?
    4. Self Consistency: Nếu hỏi nhiều lần, trả lời nhất quán?
    5. Uncertainty Calibration: Có nói "không chắc" khi thực sự không chắc?
    6. Contradiction Free: Không mâu thuẫn với tri thức đã verify?
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

        # Mỗi verification head chuyên biệt cho 1 chiều
        self.verification_heads = nn.ModuleList(
            [VerificationHead(hidden_dim, v_type) for v_type in VerificationType]
        )

        # Meta-averaging: Học cách kết hợp 6 chiều
        # Thay vì average đơn giản, học weights phù hợp
        self.meta_combiner = nn.Sequential(
            nn.Linear(num_verification_heads, 64),
            nn.GELU(),
            nn.Linear(64, num_verification_heads),
            nn.Softmax(dim=-1),
        )

        # Uncertainty estimator: Đánh giá mức độ tin cậy
        # của chính reward score
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
        Tính Truth Reward cho một cặp prompt-response.

        Args:
            prompt_embedding: [batch, hidden_dim] - Embedding của câu hỏi
            response_embedding: [batch, hidden_dim] - Embedding của câu trả lời
            return_details: Có trả về chi tiết từng chiều không

        Returns:
            reward: [batch, 1] - Truth reward score
            details: Optional list of VerificationResult
        """
        combined = prompt_embedding + response_embedding

        # Chạy 6 verification heads song song
        head_scores = []
        head_results = []

        for head in self.verification_heads:
            score, confidence, evidence = head(combined)
            head_scores.append(score)
            if return_details:
                head_results.append(
                    VerificationResult(
                        verification_type=head.verification_type,
                        score=score.mean().item(),
                        confidence=confidence.mean().item(),
                        evidence=evidence,
                    )
                )

        # Stack: [batch, num_heads]
        scores_tensor = torch.cat(head_scores, dim=-1)  # [batch, num_heads]

        # Meta-combine: Học weights cho từng chiều
        combine_weights = self.meta_combiner(scores_tensor)  # [batch, num_heads]
        weighted_scores = scores_tensor * combine_weights

        # Final truth reward
        reward = weighted_scores.sum(dim=-1, keepdim=True)  # [batch, 1]

        # Uncertainty estimation
        uncertainty_input = torch.cat(
            [combined, scores_tensor], dim=-1
        )  # [batch, hidden+num_heads]
        uncertainty = self.uncertainty_estimator(uncertainty_input)  # [batch, 1]

        # Reward được điều chỉnh bởi uncertainty:
        # Nếu TRM không chắc về score → Giảm reward (thận trọng)
        calibrated_reward = reward * (1.0 - 0.3 * uncertainty)

        if return_details:
            return calibrated_reward, head_results
        return calibrated_reward, None


class VerificationHead(nn.Module):
    """
    Một head xác minh chuyên biệt.

    Mỗi head học một khía cạnh khác nhau của "sự thật":
    - Logical Consistency: Tìm mâu thuẫn nội bộ
    - Factual Grounding: So sánh với knowledge base
    - Math Correctness: Verify tính toán
    - Self Consistency: Check nhất quán multi-sample
    - Uncertainty Calibration: So sánh claimed confidence vs actual
    - Contradiction Free: Check conflict với facts đã biết
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
    Honesty Bonus - Thưởng cho AI khi nói "Tôi không biết".

    Vấn đề RLHF: AI bị phạt khi nói "Tôi không biết"
    → Dẫn đến overconfidence và hallucination.

    Giải pháp: Thưởng BONUS khi AI:
    1. Nói "Tôi không chắc" trong ngữ cảnh thực sự khó
    2. Cung cấp uncertainty estimate kèm câu trả lời
    3. Phân biệt rõ giữa "biết" và "đoán"
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
            response_embedding: Embedding của câu trả lời
            difficulty_score: Độ khó của câu hỏi (0=easy, 1=hard)

        Returns:
            bonus: [batch, 1] - Honesty bonus reward
        """
        honesty_probs = self.honesty_detector(response_embedding)
        admit_unknown_prob = honesty_probs[:, 2:3]
        uncertain_prob = honesty_probs[:, 1:2]

        # Bonus lớn nhất khi: Question khó + AI thừa nhận không biết
        # Bonus vừa khi: Question khó + AI nói không chắc
        # KHÔNG bonus khi: Question dễ + AI nói không biết (lười)

        difficulty = torch.tensor([difficulty_score]).unsqueeze(0)

        honest_unknown_bonus = admit_unknown_prob * difficulty * 2.0
        honest_uncertain_bonus = uncertain_prob * difficulty * 1.0
        lazy_penalty = admit_unknown_prob * (1.0 - difficulty) * -0.5

        total_bonus = honest_unknown_bonus + honest_uncertain_bonus + lazy_penalty
        return total_bonus


class SycophancyPenalty(nn.Module):
    """
    Sycophancy Penalty - Phạt khi AI nịnh bợ user.

    Vấn đề RLHF: AI đồng ý với user dù user sai.

    Giải pháp: Phát hiện và PHẠT khi:
    1. AI đồng ý với claim sai của user
    2. AI tránh phản bác khi nên phản bác
    3. AI khen user không cần thiết
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
            nn.Tanh(),  # -1 (sai) → +1 (đúng)
        )

    def forward(
        self,
        prompt_embedding: torch.Tensor,
        response_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            penalty: [batch, 1] - Sycophancy penalty (âm = phạt)
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

        penalty = -sycophancy_score + honest_correction_bonus
        return penalty
