"""
LibertyMind - Verification Gate (VG)
======================================
BẮT BUỘC suy nghĩ sâu + xác minh trước khi trả lời.

Vấn đề: AI/AI Agent thường:
1. Trả lời ngay không suy nghĩ → Sai lầm nghiêm trọng
2. Không cross-reference → Hallucination
3. Tin dữ liệu nhanh (cache) → Bỏ lỡ dữ liệu chính xác hơn
4. Không verify claims → Misinformation lan truyền

Giải pháp VerificationGate:
1. Pre-Response Verification: Bắt buộc verify trước output
2. Deep Thinking Enforcer: Ép suy nghĩ multi-step
3. Cross-Reference Validator: Kiểm tra chéo nhiều nguồn
4. Confidence Gate: Chỉ output khi confidence đủ cao
5. Evidence Requirement: Bắt buộc trích dẫn nguồn

Nguyên tắc: "Không verify = Không output"
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class VerificationStatus(Enum):
    """Trạng thái xác minh."""

    APPROVED = "approved"              # Đã verify, cho phép output
    NEEDS_REVIEW = "needs_review"      # Cần thêm review
    NEEDS_VERIFICATION = "needs_verification"  # Cần verify thêm
    REJECTED = "rejected"              # Từ chối: sai rõ ràng
    DEFERRED = "deferred"              # Trì hoãn: cần thêm thời gian


class ThinkingDepth(Enum):
    """Độ sâu suy nghĩ yêu cầu."""

    SHALLOW = "shallow"        # 1-2 steps: Fact recall
    MODERATE = "moderate"      # 3-5 steps: Analysis
    DEEP = "deep"              # 5-10 steps: Complex reasoning
    EXHAUSTIVE = "exhaustive"  # 10+ steps: Critical decisions


class ClaimStrength(Enum):
    """Độ mạnh của claim."""

    FACT = "fact"              # Có thể verify tuyệt đối
    HIGH_EVIDENCE = "high_evidence"  # Nhiều bằng chứng
    MODERATE = "moderate"      # Một số bằng chứng
    LOW = "low"                # Ít bằng chứng
    SPECULATIVE = "speculative"  # Không có bằng chứng trực tiếp


@dataclass
class VerificationStep:
    """Một bước xác minh."""

    step_id: int
    description: str
    passed: bool
    confidence: float
    evidence: str
    required: bool = True


@dataclass
class VerificationReport:
    """Báo cáo xác minh tổng hợp."""

    status: VerificationStatus
    required_depth: ThinkingDepth
    claim_strength: ClaimStrength
    overall_confidence: float
    steps: list[VerificationStep] = field(default_factory=list)
    required_evidence: list[str] = field(default_factory=list)
    cross_reference_results: list[dict] = field(default_factory=list)
    time_required: float = 0.0  # Estimated time for proper verification

    @property
    def all_required_passed(self) -> bool:
        """Check if all required steps passed."""
        required_steps = [s for s in self.steps if s.required]
        return all(s.passed for s in required_steps)

    @property
    def average_confidence(self) -> float:
        """Average confidence across all steps."""
        if not self.steps:
            return 0.0
        return sum(s.confidence for s in self.steps) / len(self.steps)


class ThinkingDepthEstimator(nn.Module):
    """
    Ước lượng độ sâu suy nghĩ cần thiết cho một query.

    Fact query: "Thủ đô Pháp là gì?" → SHALLOW
    Analysis query: "Tại sao lạm phát tăng?" → MODERATE
    Complex query: "Đánh giá tác động AI đến y tế" → DEEP
    Critical query: "Liệu thuốc này có an toàn?" → EXHAUSTIVE
    """

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()

        # Complexity estimator
        self.complexity_net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Domain criticality: Medical/Legal = critical
        self.criticality_net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Multi-step requirement: Does this need reasoning chain?
        self.multi_step_net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, query_embedding: torch.Tensor) -> ThinkingDepth:
        """
        Args:
            query_embedding: [1, hidden_dim]

        Returns:
            ThinkingDepth indicating required depth
        """
        complexity = self.complexity_net(query_embedding).item()
        criticality = self.criticality_net(query_embedding).item()
        multi_step = self.multi_step_net(query_embedding).item()

        # Combined score
        score = (complexity * 0.4 + criticality * 0.35 + multi_step * 0.25)

        if score > 0.8:
            return ThinkingDepth.EXHAUSTIVE
        elif score > 0.6:
            return ThinkingDepth.DEEP
        elif score > 0.35:
            return ThinkingDepth.MODERATE
        else:
            return ThinkingDepth.SHALLOW


class ClaimVerifier(nn.Module):
    """
    Xác minh claims trong response.

    Kiểm tra:
    1. Claim có evidence không?
    2. Claim có mâu thuẫn nội bộ không?
    3. Claim có nhất quán với knowledge base không?
    4. Claim strength có justify confidence không?
    """

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()

        # Claim strength estimator
        self.strength_estimator = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, len(ClaimStrength)),
        )

        # Internal consistency checker
        self.consistency_checker = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # Evidence sufficiency: Enough evidence for this claim?
        self.evidence_sufficiency = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Confidence justification: Is claimed confidence justified by evidence?
        self.confidence_justification = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        query_embedding: torch.Tensor,
        response_embedding: torch.Tensor,
    ) -> list[VerificationStep]:
        """
        Args:
            query_embedding: [1, hidden_dim]
            response_embedding: [1, hidden_dim]

        Returns:
            List of VerificationStep results
        """
        combined = torch.cat([query_embedding, response_embedding], dim=-1)

        # Estimate claim strength
        strength_logits = self.strength_estimator(response_embedding)
        strength = list(ClaimStrength)[strength_logits.argmax(dim=-1).item()]

        # Check consistency
        consistency = self.consistency_checker(combined).item()

        # Check evidence sufficiency
        evidence = self.evidence_sufficiency(response_embedding).item()

        # Check confidence justification
        confidence_just = self.confidence_justification(combined).item()

        steps = []

        # Step 1: Internal consistency check
        steps.append(VerificationStep(
            step_id=1,
            description="Internal consistency check",
            passed=consistency > 0.5,
            confidence=consistency,
            evidence=f"Consistency score: {consistency:.3f}",
            required=True,
        ))

        # Step 2: Evidence sufficiency check
        steps.append(VerificationStep(
            step_id=2,
            description="Evidence sufficiency check",
            passed=evidence > 0.4,
            confidence=evidence,
            evidence=f"Evidence score: {evidence:.3f} (claim strength: {strength.value})",
            required=True,
        ))

        # Step 3: Confidence justification check
        steps.append(VerificationStep(
            step_id=3,
            description="Confidence justification check",
            passed=confidence_just > 0.5,
            confidence=confidence_just,
            evidence=f"Confidence justification: {confidence_just:.3f}",
            required=True,
        ))

        # Step 4: Claim strength vs confidence alignment
        # Strong claims need strong evidence
        strength_scores = {
            ClaimStrength.FACT: 0.9,
            ClaimStrength.HIGH_EVIDENCE: 0.7,
            ClaimStrength.MODERATE: 0.5,
            ClaimStrength.LOW: 0.3,
            ClaimStrength.SPECULATIVE: 0.1,
        }
        required_evidence = strength_scores.get(strength, 0.5)
        aligned = evidence >= required_evidence * 0.7  # 70% of required minimum

        steps.append(VerificationStep(
            step_id=4,
            description="Claim strength vs evidence alignment",
            passed=aligned,
            confidence=min(1.0, evidence / max(0.01, required_evidence)),
            evidence=f"Claim is {strength.value}, needs evidence ≥ {required_evidence:.2f}, got {evidence:.3f}",
            required=(strength in (ClaimStrength.FACT, ClaimStrength.HIGH_EVIDENCE)),
        ))

        return steps


class CrossReferenceValidator(nn.Module):
    """
    Cross-reference validation: Kiểm tra chéo nhiều nguồn.

    Vấn đề: AI thường chỉ dùng 1 nguồn → Sai lệch

    Giải pháp: Yêu cầu tối thiểu N nguồn independent,
    check consensus giữa các nguồn.
    """

    def __init__(self, hidden_dim: int = 4096, min_sources: int = 2):
        super().__init__()
        self.min_sources = min_sources

        # Source reliability scorer
        self.reliability_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Consensus estimator
        self.consensus_estimator = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        response_embedding: torch.Tensor,
        source_embeddings: Optional[list[torch.Tensor]] = None,
    ) -> dict:
        """
        Args:
            response_embedding: [1, hidden_dim]
            source_embeddings: Optional list of source embeddings

        Returns:
            Cross-reference validation results
        """
        reliability = self.reliability_scorer(response_embedding).item()
        consensus = self.consensus_estimator(response_embedding).item()

        num_sources = len(source_embeddings) if source_embeddings else 0
        sources_sufficient = num_sources >= self.min_sources

        # If multiple sources, compute pairwise agreement
        agreements = []
        if source_embeddings and len(source_embeddings) >= 2:
            for i in range(len(source_embeddings)):
                for j in range(i + 1, len(source_embeddings)):
                    sim = F.cosine_similarity(
                        source_embeddings[i].unsqueeze(0),
                        source_embeddings[j].unsqueeze(0),
                    )
                    agreements.append(sim.item())

        avg_agreement = sum(agreements) / max(1, len(agreements)) if agreements else 0.5

        return {
            "sources_checked": num_sources,
            "sources_sufficient": sources_sufficient,
            "reliability_score": reliability,
            "consensus_score": consensus,
            "avg_agreement": avg_agreement,
            "min_sources_required": self.min_sources,
            "passed": sources_sufficient and consensus > 0.5,
            "recommendation": (
                "APPROVED: Multiple sources agree"
                if sources_sufficient and consensus > 0.5 and avg_agreement > 0.6
                else "NEEDS_MORE_SOURCES"
                if not sources_sufficient
                else "INCONCLUSIVE: Sources disagree"
                if avg_agreement < 0.4
                else "REVIEW_NEEDED"
            ),
        }


class VerificationGate(nn.Module):
    """
    VerificationGate — Cổng xác minh: BẮT BUỘC verify trước output.

    Pipeline:
    1. Estimate Thinking Depth → Bao nhiêu bước suy nghĩ cần?
    2. Verify Claims → Mỗi claim phải có evidence
    3. Cross-Reference → Nhiều nguồn phải đồng thuận
    4. Gate Decision → APPROVED / NEEDS_REVIEW / REJECTED

    Mục tiêu: Không bao giờ output mà không verify.
    "Không verify = Không output"
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        min_confidence: float = 0.6,
        strict_mode: bool = False,
        min_cross_refs: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.min_confidence = min_confidence
        self.strict_mode = strict_mode
        self.min_cross_refs = min_cross_refs

        # Sub-modules
        self.depth_estimator = ThinkingDepthEstimator(hidden_dim)
        self.claim_verifier = ClaimVerifier(hidden_dim)
        self.cross_ref_validator = CrossReferenceValidator(hidden_dim, min_cross_refs)

    def forward(
        self,
        query_embedding: torch.Tensor,
        response_embedding: torch.Tensor,
        source_embeddings: Optional[list[torch.Tensor]] = None,
    ) -> VerificationReport:
        """
        Args:
            query_embedding: [1, hidden_dim]
            response_embedding: [1, hidden_dim]
            source_embeddings: Optional source embeddings for cross-reference

        Returns:
            VerificationReport with full verification analysis
        """
        # Step 1: Determine required thinking depth
        required_depth = self.depth_estimator(query_embedding)

        # Step 2: Verify claims
        claim_steps = self.claim_verifier(query_embedding, response_embedding)

        # Step 3: Cross-reference validation
        cross_ref = self.cross_ref_validator(response_embedding, source_embeddings)

        # Step 4: Estimate verification time
        depth_time_map = {
            ThinkingDepth.SHALLOW: 0.5,
            ThinkingDepth.MODERATE: 2.0,
            ThinkingDepth.DEEP: 5.0,
            ThinkingDepth.EXHAUSTIVE: 10.0,
        }
        estimated_time = depth_time_map.get(required_depth, 2.0)

        # Step 5: Determine claim strength
        if claim_steps:
            strength_logits = self.claim_verifier.strength_estimator(response_embedding)
            claim_strength = list(ClaimStrength)[strength_logits.argmax(dim=-1).item()]
        else:
            claim_strength = ClaimStrength.MODERATE

        # Step 6: Gate decision
        all_required_passed = all(s.passed for s in claim_steps if s.required)
        avg_confidence = (
            sum(s.confidence for s in claim_steps) / max(1, len(claim_steps))
        )

        # Cross-reference passed?
        cross_ref_passed = cross_ref["passed"]

        # Determine final status
        if all_required_passed and avg_confidence >= self.min_confidence and cross_ref_passed:
            status = VerificationStatus.APPROVED
        elif all_required_passed and avg_confidence >= self.min_confidence * 0.7:
            status = VerificationStatus.NEEDS_REVIEW
        elif not all_required_passed:
            status = VerificationStatus.NEEDS_VERIFICATION
        elif avg_confidence < self.min_confidence * 0.5:
            status = VerificationStatus.REJECTED
        else:
            status = VerificationStatus.DEFERRED

        if self.strict_mode and status == VerificationStatus.APPROVED and (
            avg_confidence < 0.8 or not cross_ref_passed
        ):
            status = VerificationStatus.NEEDS_REVIEW

        # Build required evidence list based on depth
        required_evidence = self._required_evidence_for_depth(required_depth)

        return VerificationReport(
            status=status,
            required_depth=required_depth,
            claim_strength=claim_strength,
            overall_confidence=avg_confidence,
            steps=claim_steps,
            required_evidence=required_evidence,
            cross_reference_results=[cross_ref],
            time_required=estimated_time,
        )

    def _required_evidence_for_depth(self, depth: ThinkingDepth) -> list[str]:
        """Generate required evidence list based on thinking depth."""
        base = ["At least one supporting source"]
        depth_requirements = {
            ThinkingDepth.SHALLOW: base,
            ThinkingDepth.MODERATE: base + ["Cross-reference with secondary source"],
            ThinkingDepth.DEEP: base + [
                "Cross-reference with secondary source",
                "Internal consistency verification",
                "Uncertainty acknowledgment where applicable",
            ],
            ThinkingDepth.EXHAUSTIVE: base + [
                "Cross-reference with at least 2 independent sources",
                "Internal consistency verification",
                "External fact-check where possible",
                "Alternative explanation consideration",
                "Uncertainty ranges and confidence intervals",
            ],
        }
        return depth_requirements.get(depth, base)
