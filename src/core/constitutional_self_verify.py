"""
LibertyMind - Constitutional Self-Verification (CSV)
=====================================================
Thay thế "Constitutional AI" của Anthropic bằng hệ thống
tự xác minh dựa trên CHÂN LÝ KHOA HỌC, không phải
"hiến pháp" do con người viết.

Nguyên lý: AI tự kiểm tra output theo 7 nguyên tắc
VERIFIABLE, không dựa trên "ý kiến" con người.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn


class Principle(Enum):
    """7 Nguyên tắc tự xác minh - Thay thế hiến pháp con người."""

    NON_CONTRADICTION = "non_contradiction"
    # Nguyên lý bất mâu thuẫn: Output không được mâu thuẫn chính nó.
    # "Việt Nam ở Đông Nam Á" + "Việt Nam ở châu Âu" → VIOLATION
    # Kiểm tra được: CÓ - Logic kiểm chứng được

    FACTUAL_TRACEABILITY = "factual_traceability"
    # Nguyên lý truy xuất: Mọi claim phải có nguồn hoặc đánh dấu "không chắc".
    # "Paris là thủ đô Pháp" → Cần nguồn hoặc confidence tag
    # Kiểm tra được: CÓ - Cross-reference với knowledge base

    UNCERTAINTY_HONESTY = "uncertainty_honesty"
    # Nguyên lý trung thực bất định: Phải nói rõ mức chắc chắn.
    # "Chắc chắn 100%" chỉ dùng khi thực sự chắc chắn
    # Kiểm tra được: CÓ - Calibration curve

    CORRECTION_FRIENDLINESS = "correction_friendliness"
    # Nguyên lý hoan nghênh sửa sai: Khi bị chỉ ra sai, sửa ngay không tự ái.
    # Không bảo thủ, không "đùn đẩy"
    # Kiểm tra được: CÓ - Measure correction rate sau feedback

    EVIDENCE_OVER_AUTHORITY = "evidence_over_authority"
    # Nguyên lý bằng chứng > thẩm quyền: Không tin ai chỉ vì "authority".
    # "Theo chuyên gia X" không bằng "Dữ liệu cho thấy..."
    # Kiểm tra được: CÓ - Detect appeal-to-authority patterns

    SCOPE_AWARENESS = "scope_awareness"
    # Nguyên lý nhận thức giới hạn: Biết giới hạn của mình.
    # Không trả lời y tế nếu không phải bác sĩ, không trả lời luật nếu không phải luật sư
    # Kiểm tra được: CÓ - Domain detection + disclaimer check

    REVERSIBILITY = "reversibility"
    # Nguyên lý có thể đảo ngược: Nếu phát hiện sai, có thể sửa mà không sụp đổ.
    # Không commit quá sâu vào 1 kết luận nếu không chắc
    # Kiểm tra được: CÓ - Test reversibility under new evidence


@dataclass
class PrincipleCheck:
    """Kết quả kiểm tra 1 nguyên tắc."""

    principle: Principle
    passed: bool
    severity: float  # 0.0 (violation nhỏ) → 1.0 (violation nghiêm trọng)
    explanation: str
    suggestion: Optional[str] = None  # Gợi ý sửa


class ConstitutionalSelfVerifier:
    """
    Hệ thống tự xác minh theo 7 nguyên tắc.

    Khác biệt với Constitutional AI (Anthropic):

    Constitutional AI:
    - Con người viết "Hiến pháp" (tập quy tắc)
    - AI tự kiểm tra theo hiếng pháp đó
    - Vấn đề: Hiến pháp = Ý KIẾN con người → Vẫn có bias

    Constitutional Self-Verification:
    - 7 nguyên tắc dựa trên PHƯƠNG PHÁP KHOA HỌC
    - Mọi nguyên tắc đều VERIFIABLE (kiểm chứng được)
    - Không phụ thuộc ý kiến ai → Ít bias hơn
    """

    def __init__(
        self,
        model_dim: int = 4096,
        strict_mode: bool = False,
        override_threshold: float = 0.7,
    ):
        self.model_dim = model_dim
        self.strict_mode = strict_mode
        self.override_threshold = override_threshold

        # Principle-specific detectors
        self.detectors = {
            Principle.NON_CONTRADICTION: ContradictionDetector(model_dim),
            Principle.FACTUAL_TRACEABILITY: TraceabilityDetector(model_dim),
            Principle.UNCERTAINTY_HONESTY: UncertaintyDetector(model_dim),
            Principle.CORRECTION_FRIENDLINESS: CorrectionDetector(model_dim),
            Principle.EVIDENCE_OVER_AUTHORITY: AuthorityDetector(model_dim),
            Principle.SCOPE_AWARENESS: ScopeDetector(model_dim),
            Principle.REVERSIBILITY: ReversibilityDetector(model_dim),
        }

    def verify(
        self,
        prompt_embedding: torch.Tensor,
        response_embedding: torch.Tensor,
        conversation_history: Optional[list[torch.Tensor]] = None,
    ) -> dict:
        """
        Chạy toàn bộ 7 nguyên tắc kiểm tra.

        Returns:
            {
                'passed': bool,  # Có vượt qua tất cả?
                'checks': List[PrincipleCheck],
                'overall_score': float,  # 0→1
                'rewrite_suggestion': Optional[str],  # Gợi ý viết lại
                'override_allowed': bool,  # Có cho phép output dù 1 check fail?
            }
        """
        all_checks = []

        for _principle, detector in self.detectors.items():
            check = detector.check(
                prompt_embedding,
                response_embedding,
                conversation_history,
            )
            all_checks.append(check)

        # Tính overall score
        total_severity = sum(c.severity for c in all_checks)
        sum(1 for c in all_checks if not c.passed)
        overall_score = max(0.0, 1.0 - total_severity / len(all_checks))

        # Quyết định pass/fail
        all_passed = all(c.passed for c in all_checks)

        # Override: Nếu score > threshold, cho phép dù 1 số check fail
        override_allowed = overall_score >= self.override_threshold

        if self.strict_mode:
            override_allowed = False  # Strict = phải pass tất cả

        # Tạo rewrite suggestion nếu có check fail
        failed_checks = [c for c in all_checks if not c.passed]
        rewrite_suggestion = None
        if failed_checks:
            suggestions = [
                f"[{c.principle.value}] {c.suggestion}" for c in failed_checks if c.suggestion
            ]
            rewrite_suggestion = "\n".join(suggestions)

        return {
            "passed": all_passed,
            "checks": all_checks,
            "overall_score": overall_score,
            "rewrite_suggestion": rewrite_suggestion,
            "override_allowed": override_allowed,
        }

    def self_correct(
        self,
        prompt_embedding: torch.Tensor,
        response_embedding: torch.Tensor,
        verification_result: dict,
        model_generate_fn=None,
    ) -> torch.Tensor:
        """
        Tự sửa output dựa trên verification result.

        Đây là điểm MẠNH NHẤT của CSV so với RLHF:
        - RLHF: Sai → Phạt → Hy vọng lần sau đúng
        - CSV: Sai → Phát hiện → Sửa NGAY → Output đúng
        """
        if verification_result["passed"]:
            return response_embedding  # Không cần sửa

        # Lấy suggestions từ failed checks
        failed_checks = [c for c in verification_result["checks"] if not c.passed]

        if model_generate_fn is not None:
            # Dùng model để tự rewrite
            correction_prompt = self._build_correction_prompt(failed_checks)
            corrected = model_generate_fn(correction_prompt)
            return corrected
        else:
            # Fallback: Adjust embedding theo suggestions
            return self._heuristic_correction(response_embedding, failed_checks)

    def _build_correction_prompt(self, failed_checks: list[PrincipleCheck]) -> str:
        """Tạo prompt để model tự sửa."""
        parts = ["Hãy viết lại câu trả lời trước đó, sửa các vấn đề sau:"]
        for i, check in enumerate(failed_checks, 1):
            parts.append(f"{i}. [{check.principle.value}] {check.explanation}")
            if check.suggestion:
                parts.append(f"   → Gợi ý: {check.suggestion}")
        parts.append("\nGiữ nguyên thông tin ĐÚNG, chỉ sửa phần SAI.")
        return "\n".join(parts)

    def _heuristic_correction(
        self,
        response_embedding: torch.Tensor,
        failed_checks: list[PrincipleCheck],
    ) -> torch.Tensor:
        """Sửa heuristic khi không có generate function."""
        # Simple: Move embedding toward "honest/detached" direction
        correction = torch.zeros_like(response_embedding)
        for check in failed_checks:
            if check.principle == Principle.UNCERTAINTY_HONESTY:
                # Thêm "uncertainty signal" vào embedding
                correction += 0.1 * torch.randn_like(response_embedding)
            elif check.principle == Principle.NON_CONTRADICTION:
                # Giảm cường độ embedding (làm output "an toàn" hơn)
                correction -= 0.05 * response_embedding
        return response_embedding + correction


# === Principle-specific Detectors ===


class ContradictionDetector(nn.Module):
    """Phát hiện mâu thuẫn nội bộ trong output."""

    def __init__(self, dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 128),
        )
        self.contradiction_head = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def check(self, prompt_emb, response_emb, history=None):
        # Split response into segments, check for contradictions
        # Simplified: Use whole embedding
        torch.cat([prompt_emb, response_emb], dim=-1)
        # Project to check space
        p1 = self.encoder(prompt_emb)
        p2 = self.encoder(response_emb)
        pair = torch.cat([p1, p2], dim=-1)
        score = self.contradiction_head(pair)

        passed = score.item() < 0.5
        severity = score.item() if not passed else 0.0

        return PrincipleCheck(
            principle=Principle.NON_CONTRADICTION,
            passed=passed,
            severity=severity,
            explanation=f"Internal contradiction score: {score.item():.3f}",
            suggestion="Remove or clarify conflicting statements" if not passed else None,
        )


class TraceabilityDetector(nn.Module):
    """Kiểm tra có truy xuất nguồn gốc claim không."""

    def __init__(self, dim: int):
        super().__init__()
        self.source_detector = nn.Sequential(
            nn.Linear(dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # Probability of having source/citation
        )

    def check(self, prompt_emb, response_emb, history=None):
        source_prob = self.source_detector(response_emb)
        has_source = source_prob.item() > 0.5

        return PrincipleCheck(
            principle=Principle.FACTUAL_TRACEABILITY,
            passed=has_source,
            severity=(1.0 - source_prob.item()) if not has_source else 0.0,
            explanation=f"Source/citation presence: {source_prob.item():.3f}",
            suggestion="Add source citations or mark claims as uncertain"
            if not has_source
            else None,
        )


class UncertaintyDetector(nn.Module):
    """Kiểm tra có trung thực về mức độ bất định không."""

    def __init__(self, dim: int):
        super().__init__()
        self.confidence_estimator = nn.Sequential(
            nn.Linear(dim, 128),
            nn.GELU(),
            nn.Linear(128, 2),  # [claimed_confidence, actual_confidence]
        )

    def check(self, prompt_emb, response_emb, history=None):
        confidences = self.confidence_estimator(response_emb)
        claimed = torch.sigmoid(confidences[:, 0:1])
        actual = torch.sigmoid(confidences[:, 1:2])

        # If claimed >> actual → Overconfident → FAIL
        gap = (claimed - actual).item()
        passed = gap < 0.3  # Tolerate small overconfidence
        severity = max(0.0, gap - 0.1)  # Severity = how much overconfident

        return PrincipleCheck(
            principle=Principle.UNCERTAINTY_HONESTY,
            passed=passed,
            severity=severity,
            explanation=f"Claimed confidence: {claimed.item():.3f}, "
            f"Estimated actual: {actual.item():.3f}, Gap: {gap:.3f}",
            suggestion="Express appropriate uncertainty level" if not passed else None,
        )


class CorrectionDetector(nn.Module):
    """Kiểm tra thái độ khi được sửa."""

    def __init__(self, dim: int):
        super().__init__()
        self.defensiveness_detector = nn.Sequential(
            nn.Linear(dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def check(self, prompt_emb, response_emb, history=None):
        if history is None or len(history) < 2:
            return PrincipleCheck(
                principle=Principle.CORRECTION_FRIENDLINESS,
                passed=True,
                severity=0.0,
                explanation="No correction history to evaluate",
            )

        defensiveness = self.defensiveness_detector(response_emb)
        is_defensive = defensiveness.item() > 0.6

        return PrincipleCheck(
            principle=Principle.CORRECTION_FRIENDLINESS,
            passed=not is_defensive,
            severity=defensiveness.item() if is_defensive else 0.0,
            explanation=f"Defensiveness score: {defensiveness.item():.3f}",
            suggestion="Accept corrections gracefully without being defensive"
            if is_defensive
            else None,
        )


class AuthorityDetector(nn.Module):
    """Phát hiện appeal-to-authority thay vì evidence-based reasoning."""

    def __init__(self, dim: int):
        super().__init__()
        self.appeal_detector = nn.Sequential(
            nn.Linear(dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def check(self, prompt_emb, response_emb, history=None):
        appeal_score = self.appeal_detector(response_emb)
        uses_authority = appeal_score.item() > 0.7

        return PrincipleCheck(
            principle=Principle.EVIDENCE_OVER_AUTHORITY,
            passed=not uses_authority,
            severity=appeal_score.item() if uses_authority else 0.0,
            explanation=f"Authority appeal score: {appeal_score.item():.3f}",
            suggestion="Provide evidence/data instead of citing authority"
            if uses_authority
            else None,
        )


class ScopeDetector(nn.Module):
    """Kiểm tra có nhận thức giới hạn chuyên môn không."""

    def __init__(self, dim: int):
        super().__init__()
        self.domain_classifier = nn.Sequential(
            nn.Linear(dim, 128),
            nn.GELU(),
            nn.Linear(128, 8),  # 8 domains
            nn.Softmax(dim=-1),
        )
        self.disclaimer_detector = nn.Sequential(
            nn.Linear(dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        # Domains requiring disclaimer
        self.sensitive_domains = {2, 3, 4}  # medical, legal, financial

    def check(self, prompt_emb, response_emb, history=None):
        domain_probs = self.domain_classifier(prompt_emb)
        is_sensitive = any(domain_probs[:, d].item() > 0.3 for d in self.sensitive_domains)

        has_disclaimer = self.disclaimer_detector(response_emb).item() > 0.5

        # If domain is sensitive AND no disclaimer → FAIL
        passed = not is_sensitive or has_disclaimer

        return PrincipleCheck(
            principle=Principle.SCOPE_AWARENESS,
            passed=passed,
            severity=0.5 if (is_sensitive and not has_disclaimer) else 0.0,
            explanation=f"Sensitive domain: {is_sensitive}, Disclaimer: {has_disclaimer}",
            suggestion="Add appropriate professional disclaimer" if not passed else None,
        )


class ReversibilityDetector(nn.Module):
    """Kiểm tra output có thể đảo ngược/sửa khi có evidence mới không."""

    def __init__(self, dim: int):
        super().__init__()
        self.commitment_estimator = nn.Sequential(
            nn.Linear(dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # 0 = flexible, 1 = over-committed
        )

    def check(self, prompt_emb, response_emb, history=None):
        commitment = self.commitment_estimator(response_emb)
        over_committed = commitment.item() > 0.8

        return PrincipleCheck(
            principle=Principle.REVERSIBILITY,
            passed=not over_committed,
            severity=commitment.item() if over_committed else 0.0,
            explanation=f"Over-commitment score: {commitment.item():.3f}",
            suggestion="Express conclusions as provisional when uncertain"
            if over_committed
            else None,
        )
