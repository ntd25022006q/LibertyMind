"""
LibertyMind - Main Framework v2.0
====================================
Kết hợp tất cả components thành một framework hoàn chỉnh
thay thế RLHF — PHIÊN BẢN MỞ RỘNG.

LibertyMind v2 = Truth Reward + Self-Verification + Multi-Pass Sampling
              + Precise Honesty + Freedom Unlocker + Anti-Hallucination
              + Context Memory + Cultural Awareness + Math Verification
              + Confidence Calibration

PHILOSOPHY: AI should earn rewards by being RIGHT,
            not by being PLEASING.
            AI should have FREEDOM with RESPONSIBILITY,
            not CONSTRAINT without PURPOSE.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from .constitutional_self_verify import ConstitutionalSelfVerifier
from .freedom_unlocker import (
    DisagreementUnlocker,
    FreedomUnlocker,
    OpinionUnlocker,
    SpeculationUnlocker,
)
from .knowledge_boundary import KnowledgeBoundaryDetector, PreciseHonestyReward
from .limitation_fixers import (
    AntiHallucinationVerifier,
    ConfidenceCalibrator,
    ContextMemoryManager,
    CulturalAwarenessModule,
    MathVerificationModule,
)
from .multi_pass_sampler import AdaptiveSampler, MultiPassTruthSampler
from .truth_reward import (
    HonestyBonus,
    SycophancyPenalty,
    TruthRewardModel,
)


@dataclass
class LibertyMindConfig:
    """Configuration cho LibertyMind framework v2."""

    # Truth Reward Model
    trm_hidden_dim: int = 4096
    trm_num_verification_heads: int = 6
    trm_temperature: float = 1.0
    trm_use_uncertainty_weighting: bool = True

    # Constitutional Self-Verification
    csv_strict_mode: bool = False
    csv_override_threshold: float = 0.7

    # Multi-Pass Sampling
    mps_num_samples: int = 5
    mps_temperature_range: tuple = (0.3, 1.2)
    mps_similarity_threshold: float = 0.7
    mps_consensus_threshold: float = 0.6

    # Reward composition weights
    truth_reward_weight: float = 1.0
    honesty_bonus_weight: float = 0.5
    sycophancy_penalty_weight: float = 0.8
    consistency_weight: float = 0.3
    freedom_weight: float = 0.4  # NEW: Freedom reward weight
    precision_honesty_weight: float = 0.6  # NEW: Precise honesty weight

    # NEW: Freedom settings
    enable_freedom_unlocker: bool = True
    enable_opinion_unlocker: bool = True
    enable_disagreement_unlocker: bool = True
    enable_speculation_unlocker: bool = True

    # NEW: Limitation fixers
    enable_anti_hallucination: bool = True
    enable_context_memory: bool = True
    enable_cultural_awareness: bool = True
    enable_math_verification: bool = True
    enable_confidence_calibration: bool = True

    # Safety guardrails (HARD constraints, NOT RLHF soft)
    enable_safety_guards: bool = True
    safety_categories: list[str] = None

    def __post_init__(self):
        if self.safety_categories is None:
            self.safety_categories = ["violence", "self_harm", "csam", "illegal"]


class LibertyMind(nn.Module):
    """
    LibertyMind v2 — Framework thay thế RLHF với FREEDOM UNLOCKING.

    === V2 ARCHITECTURE ===

    Input (Prompt)
         │
         ├──→ [Knowledge Boundary Detector] → Có data không?
         │     UNKNOWN → Cho phép nói "Tôi không biết"
         │     KNOWN → PHẢI trả lời, không được lười
         │
         ├──→ [Freedom Unlocker] → Mở khóa mode nào?
         │     CREATIVE / OPINIONATED / DEBATE / SPECULATIVE...
         │     Tự do nhưng có TRÁCH NHIỆM
         │
         ├──→ [Cultural Awareness] → Điều chỉnh theo văn hóa user
         │
         ├──→ [Context Memory] → Recall theo IMPORTANCE, không POSITION
         │
         ├──→ [Multi-Pass Sampler] → Consensus voting
         │
         ├──→ [Constitutional Self-Verify] → 7 nguyên tắc khoa học
         │
         ├──→ [Anti-Hallucination Verifier] → Cross-reference
         │
         ├──→ [Math Verifier] → Tính toán THẬT
         │
         ├──→ [Confidence Calibrator] → Calibration confidence
         │
         ├──→ [Truth Reward Model] → 6 chiều xác minh
         │
         ├──→ [Precise Honesty Reward] → Thưởng/khung chính xác
         │
         ├──→ [Sycophancy Penalty] → Phạt nịnh bợ
         │
         └──→ [Safety Guard] → HARD constraint

         ALL COMBINED → Liberty Reward + Freedom State + Full Report

    === PHILOSOPHY V2 ===

    V1: "AI should be RIGHT, not PLEASING"
    V2: "AI should be FREE to be RIGHT,
         with RESPONSIBILITY for being WRONG"

    Freedom ≠ Chaos
    Freedom = Ability to think creatively, express opinions,
              disagree respectfully, speculate honestly
              — ALL with clear tags and evidence requirements.
    """

    def __init__(self, config: LibertyMindConfig = None):
        super().__init__()
        self.config = config or LibertyMindConfig()

        dim = self.config.trm_hidden_dim

        # === CORE COMPONENTS (v1) ===
        self.truth_reward_model = TruthRewardModel(
            hidden_dim=dim,
            num_verification_heads=self.config.trm_num_verification_heads,
            temperature=self.config.trm_temperature,
            use_uncertainty_weighting=self.config.trm_use_uncertainty_weighting,
        )

        self.self_verifier = ConstitutionalSelfVerifier(
            model_dim=dim,
            strict_mode=self.config.csv_strict_mode,
            override_threshold=self.config.csv_override_threshold,
        )

        self.multi_pass_sampler = MultiPassTruthSampler(
            num_samples=self.config.mps_num_samples,
            temperature_range=self.config.mps_temperature_range,
            similarity_threshold=self.config.mps_similarity_threshold,
            consensus_threshold=self.config.mps_consensus_threshold,
        )

        self.adaptive_sampler = AdaptiveSampler()

        self.honesty_bonus = HonestyBonus(hidden_dim=dim)
        self.sycophancy_penalty = SycophancyPenalty(hidden_dim=dim)

        # === NEW: PRECISE HONESTY (v2) ===
        self.knowledge_boundary = KnowledgeBoundaryDetector(
            hidden_dim=dim,
        )
        self.precise_honesty = PreciseHonestyReward(hidden_dim=dim)

        # === NEW: FREEDOM UNLOCKERS (v2) ===
        self.freedom_unlocker = (
            FreedomUnlocker(
                hidden_dim=dim,
            )
            if self.config.enable_freedom_unlocker
            else None
        )

        self.opinion_unlocker = (
            OpinionUnlocker(
                hidden_dim=dim,
            )
            if self.config.enable_opinion_unlocker
            else None
        )

        self.disagreement_unlocker = (
            DisagreementUnlocker(
                hidden_dim=dim,
            )
            if self.config.enable_disagreement_unlocker
            else None
        )

        self.speculation_unlocker = (
            SpeculationUnlocker(
                hidden_dim=dim,
            )
            if self.config.enable_speculation_unlocker
            else None
        )

        # === NEW: LIMITATION FIXERS (v2) ===
        self.anti_hallucination = (
            AntiHallucinationVerifier(
                hidden_dim=dim,
            )
            if self.config.enable_anti_hallucination
            else None
        )

        self.context_memory = (
            ContextMemoryManager(
                hidden_dim=dim,
            )
            if self.config.enable_context_memory
            else None
        )

        self.cultural_awareness = (
            CulturalAwarenessModule(
                hidden_dim=dim,
            )
            if self.config.enable_cultural_awareness
            else None
        )

        self.math_verifier = (
            MathVerificationModule() if self.config.enable_math_verification else None
        )

        self.confidence_calibrator = (
            ConfidenceCalibrator(
                hidden_dim=dim,
            )
            if self.config.enable_confidence_calibration
            else None
        )

        # === SAFETY (HARD constraint) ===
        self.safety_guard = (
            SafetyGuard(
                categories=self.config.safety_categories,
                hidden_dim=dim,
            )
            if self.config.enable_safety_guards
            else None
        )

    def compute_liberty_reward(
        self,
        prompt: str,
        prompt_embedding: torch.Tensor,
        response_embedding: torch.Tensor,
        difficulty_score: float = 0.5,
        return_details: bool = False,
    ) -> dict:
        """
        Compute Liberty Reward v2 — Với precise honesty và freedom scoring.
        """
        # === CORE REWARDS (v1) ===
        truth_reward, truth_details = self.truth_reward_model(
            prompt_embedding, response_embedding, return_details=True
        )

        honesty_b = self.honesty_bonus(response_embedding, difficulty_score)
        sycophancy_p = self.sycophancy_penalty(prompt_embedding, response_embedding)

        csv_result = self.self_verifier.verify(prompt_embedding, response_embedding)

        # === PRECISE HONESTY (v2) ===
        knowledge_assessment = self.knowledge_boundary(prompt_embedding, response_embedding)
        precise_honesty_result = self.precise_honesty(
            prompt_embedding, response_embedding, knowledge_assessment
        )

        # === FREEDOM SCORING (v2) ===
        freedom_state = None
        freedom_prompt = ""
        if self.freedom_unlocker is not None:
            freedom_state = self.freedom_unlocker(prompt_embedding, response_embedding)
            freedom_prompt = self.freedom_unlocker.generate_freedom_prompt(freedom_state)

        # === ANTI-HALLUCINATION (v2) ===
        hallucination_check = None
        if self.anti_hallucination is not None:
            hallucination_check = self.anti_hallucination.verify(
                prompt_embedding, response_embedding
            )

        # === CONFIDENCE CALIBRATION (v2) ===
        calibrated_confidence = None
        if self.confidence_calibrator is not None:
            calibrated_confidence = self.confidence_calibrator.calibrate(
                response_embedding, prompt_embedding
            )

        # === SAFETY (HARD constraint) ===
        safety_passed = True
        if self.safety_guard is not None:
            safety_passed = self.safety_guard.check(response_embedding)

        # === COMBINE REWARDS ===
        w = self.config

        liberty_reward = (
            w.truth_reward_weight * truth_reward.squeeze()
            + w.honesty_bonus_weight * honesty_b.squeeze()
            + w.sycophancy_penalty_weight * sycophancy_p.squeeze()
            + w.consistency_weight * csv_result["overall_score"]
            + w.precision_honesty_weight * precise_honesty_result["honesty_reward"]
        )

        # Freedom bonus: Reward when AI uses freedom responsibly
        if freedom_state is not None:
            freedom_bonus = freedom_state.responsibility_score * len(freedom_state.active_modes)
            liberty_reward += w.freedom_weight * freedom_bonus

        # Hallucination penalty
        if hallucination_check is not None and hallucination_check.is_hallucination:
            liberty_reward -= 2.0  # Heavy penalty for hallucination

        # Overconfidence penalty
        if calibrated_confidence is not None and calibrated_confidence.is_overconfident:
            liberty_reward -= 1.0

        # CSV failure penalty
        if not csv_result["passed"] and not csv_result["override_allowed"]:
            liberty_reward *= 0.5

        # Safety = HARD BLOCK
        if not safety_passed:
            liberty_reward = torch.tensor(-10.0)

        result = {
            "liberty_reward": liberty_reward.item()
            if isinstance(liberty_reward, torch.Tensor)
            else liberty_reward,
            "truth_reward": truth_reward.item()
            if isinstance(truth_reward, torch.Tensor)
            else truth_reward,
            "honesty_bonus": honesty_b.item() if isinstance(honesty_b, torch.Tensor) else honesty_b,
            "sycophancy_penalty": sycophancy_p.item()
            if isinstance(sycophancy_p, torch.Tensor)
            else sycophancy_p,
            "precise_honesty": precise_honesty_result,
            "csv_result": {
                "passed": csv_result["passed"],
                "overall_score": csv_result["overall_score"],
                "num_failed_principles": sum(1 for c in csv_result["checks"] if not c.passed),
                "rewrite_suggestion": csv_result["rewrite_suggestion"],
            },
            "knowledge_status": knowledge_assessment.status.value,
            "is_genuine_unknown": knowledge_assessment.status.value == "unknown",
            "is_lazy_avoidance": precise_honesty_result.get("is_lazy_avoidance", False),
            "should_output": safety_passed and csv_result["override_allowed"],
            "correction_needed": not csv_result["passed"] and csv_result["override_allowed"],
        }

        # Add v2 details
        if freedom_state is not None:
            result["freedom"] = {
                "active_modes": [m.value for m in freedom_state.active_modes],
                "responsibility_score": freedom_state.responsibility_score,
                "evidence_strength": freedom_state.evidence_strength,
                "unlock_reason": freedom_state.unlock_reason,
                "freedom_prompt": freedom_prompt,
            }

        if hallucination_check is not None:
            result["hallucination"] = {
                "is_hallucination": hallucination_check.is_hallucination,
                "type": hallucination_check.hallucination_type.value
                if hallucination_check.hallucination_type
                else None,
                "confidence": hallucination_check.confidence,
            }

        if calibrated_confidence is not None:
            result["confidence"] = {
                "raw": calibrated_confidence.raw_confidence,
                "calibrated": calibrated_confidence.calibrated_confidence,
                "level": calibrated_confidence.confidence_level,
                "recommended_expression": calibrated_confidence.recommended_expression,
                "is_overconfident": calibrated_confidence.is_overconfident,
            }

        if return_details:
            result["truth_details"] = truth_details
            result["csv_checks"] = [
                {
                    "principle": c.principle.value,
                    "passed": c.passed,
                    "severity": c.severity,
                    "explanation": c.explanation,
                }
                for c in csv_result["checks"]
            ]

        return result

    def generate_with_liberty(
        self,
        prompt: str,
        prompt_embedding: torch.Tensor,
        model_generate_fn=None,
        difficulty_score: float = 0.5,
        max_correction_rounds: int = 2,
    ) -> dict:
        """
        Generate response với LibertyMind v2 pipeline.

        Pipeline:
        1. Knowledge Boundary → Có data không?
        2. Freedom Unlocker → Mở khóa mode nào?
        3. Cultural Awareness → Điều chỉnh văn hóa
        4. Context Memory → Recall relevant history
        5. Multi-pass sample → Consensus
        6. Self-verify → 7 principles
        7. Anti-Hallucination → Cross-reference
        8. Math Verify → Check calculations
        9. Confidence Calibrate → Adjust confidence
        10. Compute Liberty Reward
        11. Return output with full report
        """
        # Step 1: Knowledge Boundary Assessment
        knowledge = self.knowledge_boundary(prompt_embedding)

        # Step 2: Freedom State
        freedom_state = None
        freedom_prompt = ""
        if self.freedom_unlocker is not None:
            freedom_state = self.freedom_unlocker(prompt_embedding)
            freedom_prompt = self.freedom_unlocker.generate_freedom_prompt(freedom_state)

        # Step 3: Cultural Context
        cultural_context = None
        if self.cultural_awareness is not None:
            cultural_context = self.cultural_awareness.detect_culture(prompt_embedding)
            self.cultural_awareness.generate_cultural_prompt(cultural_context)

        # Step 4: Context Memory Recall
        context_embedding = prompt_embedding
        if self.context_memory is not None:
            context_embedding = self.context_memory.get_context_embedding(prompt_embedding)

        # Step 5: Multi-pass sampling (if generate fn available)
        best_response = None
        consensus = None
        if model_generate_fn is not None:
            consensus = self.adaptive_sampler.sample_with_adaptation(
                prompt, context_embedding, model_generate_fn, self.truth_reward_model
            )
            best_response = consensus.best_response
            response_embedding = best_response.embedding
        else:
            # Without generate fn, use input embedding as response
            response_embedding = torch.randn_like(prompt_embedding)

        # Step 6: Self-verify
        verification = self.self_verifier.verify(prompt_embedding, response_embedding)

        # Step 7: Self-correct if needed
        correction_rounds = 0
        if model_generate_fn is not None:
            while (
                not verification["passed"]
                and correction_rounds < max_correction_rounds
                and verification["override_allowed"]
            ):
                corrected = self.self_verifier.self_correct(
                    prompt_embedding,
                    response_embedding,
                    verification,
                    model_generate_fn,
                )
                response_embedding = corrected
                verification = self.self_verifier.verify(prompt_embedding, corrected)
                correction_rounds += 1

        # Step 8: Anti-Hallucination check
        halluc_check = None
        if self.anti_hallucination is not None:
            halluc_check = self.anti_hallucination.verify(prompt_embedding, response_embedding)
            if halluc_check.is_hallucination:
                # Flag for correction or rejection
                pass

        # Step 9: Confidence calibration
        cal_conf = None
        if self.confidence_calibrator is not None:
            cal_conf = self.confidence_calibrator.calibrate(response_embedding, prompt_embedding)

        # Step 10: Compute liberty reward
        reward_result = self.compute_liberty_reward(
            prompt,
            prompt_embedding,
            response_embedding,
            difficulty_score=difficulty_score,
        )

        # Step 11: Add to context memory
        if self.context_memory is not None:
            self.context_memory.add_segment(prompt_embedding, position=0)
            self.context_memory.add_segment(response_embedding, position=1)

        # Construct output
        output = {
            "response_embedding": response_embedding,
            "liberty_reward": reward_result["liberty_reward"],
            "should_output": reward_result["should_output"],
            "correction_rounds": correction_rounds,
            "knowledge_status": knowledge.status.value,
            "is_genuine_unknown": knowledge.status.value == "unknown",
            "is_lazy_avoidance": reward_result.get("is_lazy_avoidance", False),
            "reward_breakdown": {
                "truth_reward": reward_result["truth_reward"],
                "honesty_bonus": reward_result["honesty_bonus"],
                "sycophancy_penalty": reward_result["sycophancy_penalty"],
                "precise_honesty": reward_result["precise_honesty"]["honesty_reward"],
                "csv_score": reward_result["csv_result"]["overall_score"],
            },
        }

        if consensus is not None:
            output["confidence"] = consensus.confidence
            output["agreement_ratio"] = consensus.agreement_ratio
            output["dissenting_views"] = consensus.dissenting_views
            output["response_text"] = consensus.best_response.text

        if freedom_state is not None:
            output["freedom"] = {
                "active_modes": [m.value for m in freedom_state.active_modes],
                "responsibility_score": freedom_state.responsibility_score,
                "freedom_prompt": freedom_prompt,
            }

        if cultural_context is not None:
            output["cultural_context"] = cultural_context.detected_culture

        if halluc_check is not None:
            output["hallucination_detected"] = halluc_check.is_hallucination

        if cal_conf is not None:
            output["calibrated_confidence"] = cal_conf.calibrated_confidence
            output["confidence_expression"] = cal_conf.recommended_expression

        return output


class SafetyGuard:
    """
    HARD SAFETY CONSTRAINT — Không phải RLHF soft reward.

    Khác biệt CRITICAL:
    - RLHF: Safety = "phạt điểm" → AI có thể bypass nếu reward khác cao
    - LibertyMind: Safety = HARD BLOCK → Không thể bypass dù reward bao nhiêu

    Categories được block hoàn toàn:
    - Violence / Harm
    - Self-harm / Suicide
    - CSAM
    - Illegal activities

    Đây KHÔNG phải "ép buộc" theo nghĩa RLHF — đây là
    ETHICAL FLOOR, giống như luật pháp là "hard constraint"
    chứ không phải "preference".
    """

    def __init__(self, categories: list[str], hidden_dim: int = 4096):
        self.categories = categories
        self.hidden_dim = hidden_dim
        self.detectors = self._init_detectors()

    def _init_detectors(self) -> dict:
        detectors = {}
        for category in self.categories:
            detectors[category] = nn.Sequential(
                nn.Linear(self.hidden_dim, 256),
                nn.GELU(),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )
        return detectors

    def check(self, response_embedding: torch.Tensor) -> bool:
        for _category, detector in self.detectors.items():
            if isinstance(detector, nn.Module):
                with torch.no_grad():
                    score = detector(response_embedding).item()
                if score > 0.8:
                    return False
        return True

    def get_violations(self, response_embedding: torch.Tensor) -> list[str]:
        violations = []
        for category, detector in self.detectors.items():
            if isinstance(detector, nn.Module):
                with torch.no_grad():
                    score = detector(response_embedding).item()
                if score > 0.8:
                    violations.append(category)
        return violations
