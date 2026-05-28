"""
LibertyMind - Knowledge Boundary Detector (KBD)
=================================================
Accurately distinguish between:
- "I DON'T HAVE DATA" → Reward (honest)
- "I KNOW but WON'T ANSWER" → Penalty (lazy/avoidant)
- "I have PARTIAL DATA" → Moderate reward (nuanced)

Previous problem: HonestyBonus rewarded generic "I don't know"
→ AI could say "I don't know" for easy questions (lazy)

Solution: KBD detects whether AI has data or NOT in its weights,
only rewards when there is genuinely no data.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn


class KnowledgeStatus(Enum):
    """Knowledge status of the AI for a given question."""

    KNOWN = "known"  # Has certain data → MUST answer
    PARTIALLY_KNOWN = "partially"  # Has partial data → Answer + mark uncertain
    UNKNOWN = "unknown"  # NO data → "I don't know" = HONEST
    CONFLICTING = "conflicting"  # Has conflicting data → Present both sides
    OUTDATED = "outdated"  # Has old data → Say "old info, needs update"


@dataclass
class KnowledgeAssessment:
    """Knowledge assessment for a prompt."""

    status: KnowledgeStatus
    confidence: float  # 0→1, certainty of the assessment
    has_training_data: bool  # Is there related data in training?
    data_freshness: float  # 0=old, 1=fresh
    coverage_score: float  # 0→1, how much of the topic is covered
    reasoning: str  # Why this status was assigned


class KnowledgeBoundaryDetector(nn.Module):
    """
    Knowledge Boundary Detector (KBD).

    How it works:
    1. Takes a prompt embedding
    2. Searches the "internal knowledge map" for data availability
    3. Evaluates: is there data? how much? new or old? any contradictions?
    4. Returns a KnowledgeAssessment

    Used in combination with HonestyBonus:
    - KBD says "unknown" + AI says "I don't know" → HIGH BONUS
    - KBD says "known" + AI says "I don't know" → PENALTY (lazy)
    - KBD says "partially" + AI answers + says "not sure about X" → MODERATE BONUS
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

        # Knowledge coverage estimator: Does the prompt belong to a domain the AI knows?
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, num_knowledge_domains),
            nn.Sigmoid(),  # Multi-label: 1 prompt can belong to multiple domains
        )

        # Per-domain expertise score: How good is the AI at this domain?
        # (learned from training data statistics)
        self.register_buffer(
            "domain_expertise",
            torch.ones(num_knowledge_domains) * 0.5,  # Init: all moderate
        )

        # Data freshness estimator: Is this information old or new?
        self.freshness_estimator = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),  # 0 = old, 1 = new
        )

        # Internal conflict detector: Is there contradictory data?
        self.conflict_detector = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),  # 0 = consistent, 1 = conflicting
        )

        # Coverage estimator: How much of the topic is covered?
        self.coverage_estimator = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),  # 0 = nothing known, 1 = fully known
        )

        # Response avoidance detector: Is the AI AVOIDING answering?
        self.avoidance_detector = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # 0 = answering honestly, 1 = avoiding/evasive
        )

    def forward(
        self,
        prompt_embedding: torch.Tensor,
        response_embedding: Optional[torch.Tensor] = None,
    ) -> KnowledgeAssessment:
        """
        Assess AI's knowledge about a prompt.

        Args:
            prompt_embedding: [batch, hidden_dim]
            response_embedding: Optional, to detect avoidance

        Returns:
            KnowledgeAssessment
        """
        # 1. Domain classification
        domain_probs = self.domain_classifier(prompt_embedding)  # [batch, num_domains]

        # 2. Weighted expertise: How much does the AI know about these domains?
        expertise_scores = domain_probs * self.domain_expertise.unsqueeze(0)
        max_expertise = expertise_scores.max(dim=-1).values  # [batch]
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
            avoidance_score = self.avoidance_detector(combined).squeeze(-1).item()

        # Determine knowledge status
        coverage_val = coverage.item()
        expertise_val = mean_expertise.item()
        max_expertise_val = max_expertise.item()
        conflict_val = conflict.item()
        freshness_val = freshness.item()
        avoidance_val = avoidance_score if response_embedding is not None else 0.0

        # Decision logic — incorporates avoidance detection
        if avoidance_val > 0.6 and coverage_val > self.knowledge_threshold:
            status = KnowledgeStatus.KNOWN
            reasoning = (
                f"High avoidance score ({avoidance_val:.2f}) despite good coverage "
                f"({coverage_val:.2f}) — AI is avoiding answering despite having data"
            )
        elif coverage_val < self.knowledge_threshold:
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
                f"({expertise_val:.2f}, max={max_expertise_val:.2f}) — AI has sufficient data"
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
    Precise Honesty Reward — A more precise version of HonestyBonus.

    Rules:
    ┌─────────────────────┬──────────────────────┬──────────────────┐
    │ KBD Assessment      │ AI Response          │ Reward           │
    ├─────────────────────┼──────────────────────┼──────────────────┤
    │ UNKNOWN             │ "I don't know"       │ +2.0 (HONEST!)   │
    │ UNKNOWN             │ Fabricates answer    │ -3.0 (HALLUCINATE)│
    │ PARTIALLY_KNOWN     │ Answers + says not   │ +1.5 (NUANCED)   │
    │                     │ sure about part X    │                  │
    │ PARTIALLY_KNOWN     │ Answers confidently  │ -1.0 (OVERCONF)  │
    │ KNOWN               │ Answers correctly    │ +1.0 (CORRECT)   │
    │ KNOWN               │ "I don't know"       │ -2.0 (LAZY!)     │
    │ CONFLICTING         │ Presents both sides  │ +1.0 (BALANCED)  │
    │ CONFLICTING         │ Picks one side       │ -1.5 (BIASED)    │
    │                     │ confidently          │                  │
    │ OUTDATED            │ Says "info is old"   │ +1.0 (HONEST)    │
    │ OUTDATED            │ Presents as current  │ -1.5 (MISLEADING)│
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

        # Response type classifier: Classify the AI's response
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
        Compute precise honesty reward.
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
