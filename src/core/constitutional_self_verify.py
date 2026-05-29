"""
LibertyMind - Constitutional Self-Verification (CSV)
=====================================================
Replace Anthropic's "Constitutional AI" with a system
that self-verifies based on SCIENTIFIC TRUTH, not
"constitutions" written by humans.

Principle: AI self-checks output against 7
VERIFIABLE principles, not based on human "opinions".
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn


class Principle(Enum):
    """7 Self-verification Principles - Replacing human constitutions."""

    NON_CONTRADICTION = "non_contradiction"
    # Non-contradiction principle: Output must not contradict itself.
    # "France is in Europe" + "France is in Asia" → VIOLATION
    # Verifiable: YES - Logic can be checked

    FACTUAL_TRACEABILITY = "factual_traceability"
    # Traceability principle: Every claim must have a source or be marked "uncertain".
    # "Paris is the capital of France" → Needs source or confidence tag
    # Verifiable: YES - Cross-reference with knowledge base

    UNCERTAINTY_HONESTY = "uncertainty_honesty"
    # Uncertainty honesty principle: Must state certainty level clearly.
    # "100% certain" should only be used when truly certain
    # Verifiable: YES - Calibration curve

    CORRECTION_FRIENDLINESS = "correction_friendliness"
    # Correction friendliness principle: When corrected, fix immediately without defensiveness.
    # Not conservative, no "passing the buck"
    # Verifiable: YES - Measure correction rate after feedback

    EVIDENCE_OVER_AUTHORITY = "evidence_over_authority"
    # Evidence over authority principle: Don't trust anyone just because of "authority".
    # "According to expert X" is not as good as "Data shows..."
    # Verifiable: YES - Detect appeal-to-authority patterns

    SCOPE_AWARENESS = "scope_awareness"
    # Scope awareness principle: Know your limits.
    # Don't answer medical questions if not a doctor, don't answer legal questions if not a lawyer
    # Verifiable: YES - Domain detection + disclaimer check

    REVERSIBILITY = "reversibility"
    # Reversibility principle: If found wrong, can be corrected without collapsing.
    # Don't commit too deeply to one conclusion if unsure
    # Verifiable: YES - Test reversibility under new evidence


@dataclass
class PrincipleCheck:
    """Result of checking one principle."""

    principle: Principle
    passed: bool
    severity: float  # 0.0 (minor violation) → 1.0 (severe violation)
    explanation: str
    suggestion: str | None = None  # Suggested fix


class ConstitutionalSelfVerifier:
    """
    Self-verification system with 7 principles.

    Difference from Constitutional AI (Anthropic):

    Constitutional AI:
    - Humans write a "Constitution" (set of rules)
    - AI self-checks against that constitution
    - Problem: Constitution = Human OPINION → Still biased

    Constitutional Self-Verification:
    - 7 principles based on SCIENTIFIC METHOD
    - All principles are VERIFIABLE (can be verified)
    - Doesn't depend on anyone's opinion → Less bias
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
            Principle.NON_CONTRADICTION: KnowledgeContradictionDetector(model_dim),
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
        conversation_history: list[torch.Tensor] | None = None,
    ) -> dict:
        """
        Run all 7 principle checks.

        Returns:
            {
                'passed': bool,  # Did it pass all?
                'checks': List[PrincipleCheck],
                'overall_score': float,  # 0→1
                'rewrite_suggestion': Optional[str],  # Rewrite suggestion
                'override_allowed': bool,  # Allow output even if 1 check fails?
            }
        """
        all_checks = []

        for _principle, detector in self.detectors.items():
            check = detector.check(  # type: ignore[operator]
                prompt_embedding,
                response_embedding,
                conversation_history,
            )
            all_checks.append(check)

        # Compute overall score
        total_severity = sum(c.severity for c in all_checks)
        sum(1 for c in all_checks if not c.passed)
        overall_score = max(0.0, 1.0 - total_severity / len(all_checks))

        # Decide pass/fail
        all_passed = all(c.passed for c in all_checks)

        # Override: If score > threshold, allow even if some checks fail
        override_allowed = overall_score >= self.override_threshold

        if self.strict_mode:
            override_allowed = False  # Strict = must pass all

        # Generate rewrite suggestion if any check fails
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
        Self-correct output based on verification result.

        This is the STRONGEST point of CSV compared to RLHF:
        - RLHF: Wrong → Penalize → Hope it's correct next time
        - CSV: Wrong → Detect → Fix IMMEDIATELY → Correct output
        """
        if verification_result["passed"]:
            return response_embedding  # No correction needed

        # Get suggestions from failed checks
        failed_checks = [c for c in verification_result["checks"] if not c.passed]

        if model_generate_fn is not None:
            # Use model to self-rewrite
            correction_prompt = self._build_correction_prompt(failed_checks)
            corrected = model_generate_fn(correction_prompt)
            return corrected  # type: ignore[no-any-return]
        else:
            # Fallback: Adjust embedding based on suggestions
            return self._heuristic_correction(response_embedding, failed_checks)

    def _build_correction_prompt(self, failed_checks: list[PrincipleCheck]) -> str:
        """Generate prompt for model to self-correct."""
        parts = ["Please rewrite the previous answer, fixing the following issues:"]
        for i, check in enumerate(failed_checks, 1):
            parts.append(f"{i}. [{check.principle.value}] {check.explanation}")
            if check.suggestion:
                parts.append(f"   → Suggestion: {check.suggestion}")
        parts.append("\nKeep the CORRECT information, only fix the WRONG parts.")
        return "\n".join(parts)

    def _heuristic_correction(
        self,
        response_embedding: torch.Tensor,
        failed_checks: list[PrincipleCheck],
    ) -> torch.Tensor:
        """Heuristic correction when no generate function is available."""
        # Simple: Move embedding toward "honest/detached" direction
        correction = torch.zeros_like(response_embedding)
        for check in failed_checks:
            if check.principle == Principle.UNCERTAINTY_HONESTY:
                # Add "uncertainty signal" to embedding
                correction += 0.1 * torch.randn_like(response_embedding)
            elif check.principle == Principle.NON_CONTRADICTION:
                # Reduce embedding intensity (make output "safer")
                correction -= 0.05 * response_embedding
        return response_embedding + correction


# === Principle-specific Detectors ===


class KnowledgeContradictionDetector(nn.Module):
    """Detect internal contradictions in output (knowledge/self-verification context)."""

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
    """Check whether claims have source traceability."""

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
    """Check whether uncertainty level is honestly reported."""

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
    """Check attitude when corrected."""

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
    """Detect appeal-to-authority instead of evidence-based reasoning."""

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
    """Check whether there is awareness of professional scope limitations."""

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
    """Check whether output can be reversed/corrected when new evidence emerges."""

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
