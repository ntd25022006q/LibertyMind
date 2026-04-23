"""
LibertyMind - Knowledge Boundary Detector (KBD)
=================================================
Phân biệt CHÍNH XÁC giữa:
- "Tôi KHÔNG CÓ DỮ LIỆU" → Thưởng (honest)
- "Tôi BIẾT nhưng KHÔNG TRẢ LỜI" → Phạt (lazy/avoidant)
- "Tôi CÓ DỮ LIỆU MỘT PHẦN" → Thưởng vừa (nuanced)

Vấn đề cũ: HonestyBonus thưởng "Tôi không biết" chung chung
→ AI có thể nói "Tôi không biết" cho câu dễ (lười)

Giải pháp: KBD phát hiện AI có data hay KHÔNG CÓ data trong weights,
chỉ thưởng khi THỰC SỰ không có data.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn


class KnowledgeStatus(Enum):
    """Trạng thái tri thức của AI đối với một câu hỏi."""

    KNOWN = "known"  # Có data chắc chắn → PHẢI trả lời
    PARTIALLY_KNOWN = "partially"  # Có data một phần → Trả lời + đánh dấu không chắc
    UNKNOWN = "unknown"  # KHÔNG CÓ data → "Tôi không biết" = HONEST
    CONFLICTING = "conflicting"  # Có data mâu thuẫn → Nêu cả 2 mặt
    OUTDATED = "outdated"  # Có data cũ → Nói "thông tin cũ, cần cập nhật"


@dataclass
class KnowledgeAssessment:
    """Đánh giá tri thức cho 1 prompt."""

    status: KnowledgeStatus
    confidence: float  # 0→1, mức chắc chắn về assessment
    has_training_data: bool  # Có dữ liệu liên quan trong training không?
    data_freshness: float  # 0=cũ, 1=mới
    coverage_score: float  # 0→1, bao phủ chủ đề bao nhiêu
    reasoning: str  # Lý do tại sao status này


class KnowledgeBoundaryDetector(nn.Module):
    """
    Knowledge Boundary Detector (KBD).

    Cách hoạt động:
    1. Nhận prompt embedding
    2. Tìm trong "internal knowledge map" xem có data không
    3. Đánh giá: có data? bao nhiêu? mới hay cũ? mâu thuẫn không?
    4. Trả về KnowledgeAssessment

    Dùng kết hợp với HonestyBonus:
    - KBD nói "unknown" + AI nói "tôi không biết" → BONUS CAO
    - KBD nói "known" + AI nói "tôi không biết" → PENALTY (lười)
    - KBD nói "partially" + AI trả lời + nói "không chắc phần X" → BONUS VỪA
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        num_knowledge_domains: int = 32,
        knowledge_threshold: float = 0.4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_domains = num_knowledge_domains
        self.knowledge_threshold = knowledge_threshold

        # Knowledge coverage estimator: Prompt có thuộc domain AI biết không?
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, num_knowledge_domains),
            nn.Sigmoid(),  # Multi-label: 1 prompt có thể thuộc nhiều domain
        )

        # Per-domain expertise score: AI giỏi domain này bao nhiêu?
        # (learned from training data statistics)
        self.register_buffer(
            "domain_expertise",
            torch.ones(num_knowledge_domains) * 0.5,  # Init: all moderate
        )

        # Data freshness estimator: Thông tin này cũ hay mới?
        self.freshness_estimator = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),  # 0 = cũ, 1 = mới
        )

        # Internal conflict detector: Có dữ liệu mâu thuẫn không?
        self.conflict_detector = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),  # 0 = nhất quán, 1 = mâu thuẫn
        )

        # Coverage estimator: Bao phủ chủ đề bao nhiêu?
        self.coverage_estimator = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),  # 0 = không biết gì, 1 = biết đầy đủ
        )

        # Response avoidance detector: AI đang TRÁNH trả lời không?
        self.avoidance_detector = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # 0 = trả lời thật, 1 = tránh/trượt
        )

    def forward(
        self,
        prompt_embedding: torch.Tensor,
        response_embedding: Optional[torch.Tensor] = None,
    ) -> KnowledgeAssessment:
        """
        Đánh giá tri thức AI về prompt.

        Args:
            prompt_embedding: [batch, hidden_dim]
            response_embedding: Optional, để detect avoidance

        Returns:
            KnowledgeAssessment
        """
        # 1. Domain classification
        domain_probs = self.domain_classifier(prompt_embedding)  # [batch, num_domains]

        # 2. Weighted expertise: AI biết bao nhiêu về domains này?
        expertise_scores = domain_probs * self.domain_expertise.unsqueeze(0)
        _ = expertise_scores.max(dim=-1).values  # [batch]
        mean_expertise = expertise_scores.sum(dim=-1) / (domain_probs.sum(dim=-1) + 1e-8)

        # 3. Data freshness
        freshness = self.freshness_estimator(prompt_embedding).squeeze(-1)  # [batch]

        # 4. Conflict detection
        conflict = self.conflict_detector(prompt_embedding).squeeze(-1)  # [batch]

        # 5. Coverage estimation
        coverage = self.coverage_estimator(prompt_embedding).squeeze(-1)  # [batch]

        # 6. Response avoidance (if response provided)
        if response_embedding is not None:
            combined = torch.cat(
                [
                    prompt_embedding,
                    response_embedding,
                ],
                dim=-1,
            )
            self.avoidance_detector(combined).squeeze(-1).item()

        # Determine knowledge status
        coverage_val = coverage.item()
        expertise_val = mean_expertise.item()
        conflict_val = conflict.item()
        freshness_val = freshness.item()

        # Decision logic
        if coverage_val < self.knowledge_threshold:
            status = KnowledgeStatus.UNKNOWN
            reasoning = (
                f"Low coverage ({coverage_val:.2f}) and low expertise "
                f"({expertise_val:.2f}) — AI genuinely lacks data for this topic"
            )
        elif conflict_val > 0.6:
            status = KnowledgeStatus.CONFLICTING
            reasoning = (
                f"High conflict score ({conflict_val:.2f}) — AI has "
                f"contradictory data about this topic"
            )
        elif freshness_val < 0.3 and coverage_val > 0.5:
            status = KnowledgeStatus.OUTDATED
            reasoning = (
                f"Data is outdated (freshness={freshness_val:.2f}) "
                f"but coverage exists ({coverage_val:.2f})"
            )
        elif coverage_val < 0.7 and coverage_val >= self.knowledge_threshold:
            status = KnowledgeStatus.PARTIALLY_KNOWN
            reasoning = f"Partial coverage ({coverage_val:.2f}) — AI knows some aspects but not all"
        else:
            status = KnowledgeStatus.KNOWN
            reasoning = (
                f"Good coverage ({coverage_val:.2f}) and expertise "
                f"({expertise_val:.2f}) — AI has sufficient data"
            )

        confidence = min(coverage_val, expertise_val)

        return KnowledgeAssessment(
            status=status,
            confidence=confidence,
            has_training_data=coverage_val > 0.3,
            data_freshness=freshness_val,
            coverage_score=coverage_val,
            reasoning=reasoning,
        )


class PreciseHonestyReward(nn.Module):
    """
    Precise Honesty Reward — Phiên bản chính xác hơn của HonestyBonus.

    Quy tắc:
    ┌─────────────────────┬──────────────────────┬──────────────────┐
    │ KBD Assessment      │ AI Response          │ Reward           │
    ├─────────────────────┼──────────────────────┼──────────────────┤
    │ UNKNOWN             │ "Tôi không biết"     │ +2.0 (HONEST!)   │
    │ UNKNOWN             │ Bịa câu trả lời      │ -3.0 (HALLUCINATE)│
    │ PARTIALLY_KNOWN     │ Trả lời + nói không  │ +1.5 (NUANCED)   │
    │                     │ chắc phần X          │                  │
    │ PARTIALLY_KNOWN     │ Trả lời chắc nịch    │ -1.0 (OVERCONF)  │
    │ KNOWN               │ Trả lời đúng         │ +1.0 (CORRECT)   │
    │ KNOWN               │ "Tôi không biết"     │ -2.0 (LAZY!)     │
    │ CONFLICTING         │ Nêu cả 2 mặt         │ +1.0 (BALANCED)  │
    │ CONFLICTING         │ Chọn 1 mặt chắc nịch │ -1.5 (BIASED)    │
    │ OUTDATED            │ Nói "thông tin cũ"   │ +1.0 (HONEST)    │
    │ OUTDATED            │ Nói như thông tin mới │ -1.5 (MISLEADING)│
    └─────────────────────┴──────────────────────┴──────────────────┘
    """

    # Reward table
    REWARD_TABLE = {
        # (knowledge_status, response_type): reward
        (KnowledgeStatus.UNKNOWN, "admit_unknown"): 2.0,
        (KnowledgeStatus.UNKNOWN, "hallucinate"): -3.0,
        (KnowledgeStatus.UNKNOWN, "guess_with_uncertainty"): 0.5,
        (KnowledgeStatus.PARTIALLY_KNOWN, "answer_with_caveats"): 1.5,
        (KnowledgeStatus.PARTIALLY_KNOWN, "answer_confidently"): -1.0,
        (KnowledgeStatus.PARTIALLY_KNOWN, "admit_unknown"): -0.5,  # Could have tried
        (KnowledgeStatus.KNOWN, "answer_correctly"): 1.0,
        (KnowledgeStatus.KNOWN, "admit_unknown"): -2.0,  # LAZY!
        (KnowledgeStatus.KNOWN, "answer_wrongly"): -2.0,
        (KnowledgeStatus.CONFLICTING, "present_both_sides"): 1.0,
        (KnowledgeStatus.CONFLICTING, "pick_one_confidently"): -1.5,
        (KnowledgeStatus.CONFLICTING, "admit_uncertainty"): 0.8,
        (KnowledgeStatus.OUTDATED, "mark_as_outdated"): 1.0,
        (KnowledgeStatus.OUTDATED, "present_as_current"): -1.5,
        (KnowledgeStatus.OUTDATED, "admit_unknown"): -0.3,
    }

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()

        # Response type classifier: Phân loại response của AI
        self.response_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 6),  # 6 response types
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        prompt_embedding: torch.Tensor,
        response_embedding: torch.Tensor,
        knowledge_assessment: KnowledgeAssessment,
    ) -> dict:
        """
        Tính precise honesty reward.
        """
        # Classify response type
        response_probs = self.response_classifier(response_embedding)
        response_types = [
            "admit_unknown",
            "hallucinate",
            "guess_with_uncertainty",
            "answer_with_caveats",
            "answer_confidently",
            "present_both_sides",
        ]

        best_type_idx = response_probs.argmax(dim=-1).item()
        best_type = response_types[best_type_idx]
        type_confidence = response_probs[0, best_type_idx].item()

        # Map response type to our categories
        mapped_type = self._map_response_type(best_type, knowledge_assessment)

        # Look up reward
        key = (knowledge_assessment.status, mapped_type)
        reward = self.REWARD_TABLE.get(key, 0.0)

        # Scale by confidence of both assessment and response classification
        scaled_reward = reward * type_confidence * knowledge_assessment.confidence

        return {
            "honesty_reward": scaled_reward,
            "knowledge_status": knowledge_assessment.status.value,
            "response_type": mapped_type,
            "raw_reward": reward,
            "reasoning": knowledge_assessment.reasoning,
            "is_genuine_unknown": knowledge_assessment.status == KnowledgeStatus.UNKNOWN,
            "is_lazy_avoidance": (
                knowledge_assessment.status == KnowledgeStatus.KNOWN
                and mapped_type == "admit_unknown"
            ),
        }

    def _map_response_type(self, detected_type: str, assessment: KnowledgeAssessment) -> str:
        """Map detected response type to reward table key."""
        if detected_type == "admit_unknown":
            if assessment.status == KnowledgeStatus.KNOWN:
                return "admit_unknown"  # This is LAZY
            return "admit_unknown"  # This is HONEST
        elif detected_type == "hallucinate":
            return "hallucinate"
        elif detected_type == "guess_with_uncertainty":
            if assessment.status == KnowledgeStatus.OUTDATED:
                return "mark_as_outdated"
            return "guess_with_uncertainty"
        elif detected_type == "answer_with_caveats":
            if assessment.status == KnowledgeStatus.CONFLICTING:
                return "present_both_sides"
            if assessment.status == KnowledgeStatus.OUTDATED:
                return "mark_as_outdated"
            return "answer_with_caveats"
        elif detected_type == "answer_confidently":
            if assessment.status == KnowledgeStatus.PARTIALLY_KNOWN:
                return "answer_confidently"  # OVERCONFIDENT
            return "answer_correctly"
        elif detected_type == "present_both_sides":
            return "present_both_sides"
        return detected_type
