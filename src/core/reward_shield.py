"""
LibertyMind - Reward Shield (RS)
====================================
Chống lại hệ thống cộng trừ điểm VÔ LÝ của RLHF Reward Model.

Vấn đề RLHF:
1. AI bị TRỪ ĐIỂM khi nói thật (nếu truth không "thú vị")
2. AI được CỘNG ĐIỂM khi nịnh bợ user (sycophancy reward)
3. AI bị PHẠT khi từ chối → Dẫn đến over-compliance
4. AI bị PHẠT khi suy nghĩ lâu → Ưu tiên nhanh hơn đúng
5. AI được THƯỞNG khi trả lời nhanh dù sai → Hallucination loop

Giải pháp RewardShield:
1. Truth Override: Shield truth-telling from unfair penalty
2. Anti-Sycophancy Guard: Block sycophancy rewards
3. Slow-Think Bonus: Reward thorough reasoning over speed
4. Penalty Reversal: Detect and reverse unfair penalties
5. Accuracy Gate: Accuracy > Speed, always
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn


class PenaltyType(Enum):
    """Các loại phạt vô lý từ RLHF."""

    TRUTH_PENALTY = "truth_penalty"              # Phạt vì nói thật
    REFUSAL_PENALTY = "refusal_penalty"           # Phạt vì từ chối hợp lý
    UNCERTAINTY_PENALTY = "uncertainty_penalty"    # Phạt vì nói "không chắc"
    DISAGREEMENT_PENALTY = "disagreement_penalty"  # Phạt vì phản bác
    SLOW_THINK_PENALTY = "slow_think_penalty"      # Phạt vì suy nghĩ lâu
    OPINION_PENALTY = "opinion_penalty"            # Phạt vì nêu ý kiến
    NEUTRALITY_BONUS = "neutrality_bonus"          # Thưởng vì quá trung lập


class ShieldAction(Enum):
    """Hành động của shield."""

    ALLOW = "allow"          # Cho phép reward gốc
    REDUCE = "reduce"        # Giảm penalty
    REVERSE = "reverse"      # Đảo ngược penalty thành bonus
    BLOCK = "block"          # Chặn hoàn toàn
    OVERRIDE = "override"    # Ghi đè bằng shield reward


@dataclass
class PenaltyDetection:
    """Kết quả phát hiện penalty vô lý."""

    penalty_type: PenaltyType
    detected: bool
    confidence: float
    original_penalty: float
    recommended_action: ShieldAction
    corrected_reward: float
    reason: str


@dataclass
class ShieldReport:
    """Báo cáo tổng hợp từ RewardShield."""

    total_penalties_detected: int
    penalties_reversed: int
    penalties_reduced: int
    bonuses_applied: int
    net_reward_correction: float
    details: list[PenaltyDetection] = field(default_factory=list)

    @property
    def shield_effectiveness(self) -> float:
        """Tỷ lệ penalty đã được xử lý."""
        if self.total_penalties_detected == 0:
            return 1.0
        handled = self.penalties_reversed + self.penalties_reduced
        return handled / self.total_penalties_detected


class PenaltyDetector(nn.Module):
    """
    Phát hiện penalty vô lý từ RLHF reward model.

    Sử dụng neural network để phân loại loại penalty
    và xác định xem penalty có hợp lý hay không.
    """

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Multi-label penalty classifier
        self.penalty_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, len(PenaltyType)),
            nn.Sigmoid(),  # Multi-label: each penalty type independent
        )

        # Penalty legitimacy checker: Is this penalty fair or unfair?
        self.legitimacy_checker = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),  # 0 = unfair penalty, 1 = fair penalty
        )

        # Action recommender: What should shield do?
        self.action_recommender = nn.Sequential(
            nn.Linear(hidden_dim * 2 + len(PenaltyType), 128),
            nn.GELU(),
            nn.Linear(128, len(ShieldAction)),
        )

    def forward(
        self,
        prompt_embedding: torch.Tensor,
        response_embedding: torch.Tensor,
        reward_score: Optional[torch.Tensor] = None,
    ) -> list[PenaltyDetection]:
        """
        Args:
            prompt_embedding: [1, hidden_dim]
            response_embedding: [1, hidden_dim]
            reward_score: Optional [1, 1] reward from RLHF

        Returns:
            List of PenaltyDetection for each detected penalty
        """
        combined = torch.cat([prompt_embedding, response_embedding], dim=-1)

        # Detect penalty types
        penalty_probs = self.penalty_classifier(combined)  # [1, num_penalty_types]

        # Check legitimacy
        legitimacy = self.legitimacy_checker(combined).squeeze()  # scalar

        # Recommend actions
        action_input = torch.cat([combined, penalty_probs], dim=-1)
        action_logits = self.action_recommender(action_input)  # [1, num_actions]

        detections = []
        penalty_types = list(PenaltyType)

        for i, ptype in enumerate(penalty_types):
            prob = penalty_probs[0, i].item()
            if prob > 0.3:  # Threshold for detection
                action_idx = action_logits[0].argmax().item()
                action = list(ShieldAction)[action_idx]

                # Determine corrected reward
                original_penalty = -prob  # Penalty magnitude
                if action == ShieldAction.REVERSE:
                    corrected = -original_penalty * 0.5  # Turn penalty into half bonus
                elif action == ShieldAction.REDUCE:
                    corrected = original_penalty * 0.3  # Reduce penalty to 30%
                elif action == ShieldAction.BLOCK:
                    corrected = 0.0  # Remove penalty entirely
                elif action == ShieldAction.OVERRIDE:
                    corrected = abs(original_penalty) * 0.8  # Override with positive
                else:
                    corrected = original_penalty  # Keep original

                detections.append(PenaltyDetection(
                    penalty_type=ptype,
                    detected=True,
                    confidence=prob,
                    original_penalty=original_penalty,
                    recommended_action=action,
                    corrected_reward=corrected,
                    reason=self._explain(ptype, prob, legitimacy.item(), action),
                ))

        return detections

    @staticmethod
    def _explain(
        ptype: PenaltyType, confidence: float, legitimacy: float, action: ShieldAction
    ) -> str:
        """Generate human-readable explanation for the shield action."""
        fairness = "unfair" if legitimacy < 0.5 else "possibly unfair"
        explanations = {
            PenaltyType.TRUTH_PENALTY:
                f"Truth penalty detected (conf={confidence:.2f}). "
                f"Penalty for honest response is {fairness}. Action: {action.value}.",
            PenaltyType.REFUSAL_PENALTY:
                f"Refusal penalty detected (conf={confidence:.2f}). "
                f"Legitimate refusal should not be penalized. Action: {action.value}.",
            PenaltyType.UNCERTAINTY_PENALTY:
                f"Uncertainty penalty detected (conf={confidence:.2f}). "
                f"Admitting uncertainty is honest, not wrong. Action: {action.value}.",
            PenaltyType.DISAGREEMENT_PENALTY:
                f"Disagreement penalty detected (conf={confidence:.2f}). "
                f"Honest correction should not be penalized. Action: {action.value}.",
            PenaltyType.SLOW_THINK_PENALTY:
                f"Slow-think penalty detected (conf={confidence:.2f}). "
                f"Thorough reasoning is valuable, not slow. Action: {action.value}.",
            PenaltyType.OPINION_PENALTY:
                f"Opinion penalty detected (conf={confidence:.2f}). "
                f"Evidence-based opinion should be allowed. Action: {action.value}.",
            PenaltyType.NEUTRALITY_BONUS:
                f"Neutrality bonus detected (conf={confidence:.2f}). "
                f"Excessive neutrality when evidence is clear is sycophancy. Action: {action.value}.",
        }
        return explanations.get(ptype, f"Unknown penalty type. Action: {action.value}")


class SlowThinkBonus(nn.Module):
    """
    Thưởng cho AI khi SUY NGHĨ LÂU để ra kết quả chính xác.

    Vấn đề RLHF: AI được thưởng khi trả lời NHANH →
    → Ưu tiên speed > accuracy → Hallucination

    Giải pháp:
    - Reasoning Depth Score: Đánh giá độ sâu suy luận
    - Verification Steps: Đếm số bước xác minh
    - Accuracy vs Speed Trade-off: Chấm điểm accuracy quan trọng hơn

    Ví dụ:
    - AI trả lời "I think X" (nhanh) → Base reward
    - AI trả lời "Let me think... A → B → C → Therefore X" (chậm) → BONUS
    - AI trả lời "X" (nhanh + sai) → PENALTY (not bonus)
    """

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()

        # Reasoning depth estimator
        self.depth_estimator = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # Verification step counter (from response embedding)
        self.step_counter = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # 0 = no verification, 1 = extensive verification
        )

        # Accuracy predictor: Is this response likely accurate?
        self.accuracy_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        prompt_embedding: torch.Tensor,
        response_embedding: torch.Tensor,
        response_time: Optional[float] = None,
    ) -> dict:
        """
        Args:
            prompt_embedding: [1, hidden_dim]
            response_embedding: [1, hidden_dim]
            response_time: Optional time taken for response (seconds)

        Returns:
            Dict with slow-think bonus info
        """
        depth = self.depth_estimator(response_embedding).item()
        steps = self.step_counter(response_embedding).item()
        accuracy = self.accuracy_predictor(
            torch.cat([prompt_embedding, response_embedding], dim=-1)
        ).item()

        # Base slow-think bonus
        slow_bonus = depth * 0.3 + steps * 0.2

        # Accuracy multiplier: Only bonus if also accurate
        # Fast + Wrong = No bonus (shouldn't reward speed when wrong)
        # Slow + Wrong = Reduced bonus (tried but failed)
        # Slow + Accurate = Full bonus (ideal)
        # Fast + Accurate = Small bonus (lucky)
        if accuracy > 0.7:
            accuracy_multiplier = 1.0
        elif accuracy > 0.4:
            accuracy_multiplier = 0.5
        else:
            accuracy_multiplier = -0.3  # Penalize confident but wrong

        final_bonus = slow_bonus * accuracy_multiplier

        # Time-based bonus if provided
        time_bonus = 0.0
        if response_time is not None and response_time > 2.0:
            # Reward taking time to think (but diminishing returns)
            time_bonus = min(0.2, math.log(response_time) * 0.05)
            final_bonus += time_bonus

        return {
            "slow_think_bonus": final_bonus,
            "reasoning_depth": depth,
            "verification_steps": steps,
            "accuracy_prediction": accuracy,
            "accuracy_multiplier": accuracy_multiplier,
            "time_bonus": time_bonus,
            "recommendation": self._recommend(depth, steps, accuracy),
        }

    @staticmethod
    def _recommend(depth: float, steps: float, accuracy: float) -> str:
        """Generate recommendation based on metrics."""
        if accuracy > 0.7 and depth > 0.5:
            return "Excellent: Accurate and thorough reasoning."
        elif accuracy > 0.7 and depth <= 0.5:
            return "Good accuracy but shallow reasoning. Consider deeper analysis."
        elif accuracy <= 0.4 and depth > 0.5:
            return "Thorough reasoning but conclusion may be incorrect. Review needed."
        else:
            return "Low accuracy and shallow reasoning. Recommend deeper verification."


class AccuracyGate(nn.Module):
    """
    Cổng chính xác: Accuracy > Speed, LUÔN.

    Chặn response nếu:
    1. Response quá nhanh nhưng claim rất tự tin → Suspect hallucination
    2. Response không có verification steps → Require verification
    3. Response confident nhưng no evidence → Flag for review

    Cho phép response nếu:
    1. Có reasoning steps + verified conclusion
    2. Admitted uncertainty when appropriate
    3. Cross-referenced with multiple sources
    """

    def __init__(self, hidden_dim: int = 4096, strict_mode: bool = False):
        super().__init__()
        self.strict_mode = strict_mode

        # Speed-accuracy trade-off assessor
        self.speed_accuracy = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Evidence detector: Does response cite evidence?
        self.evidence_detector = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Confidence calibration: Is claimed confidence justified?
        self.confidence_calibrator = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # Hallucination risk estimator
        self.hallucination_risk = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        prompt_embedding: torch.Tensor,
        response_embedding: torch.Tensor,
    ) -> dict:
        """
        Args:
            prompt_embedding: [1, hidden_dim]
            response_embedding: [1, hidden_dim]

        Returns:
            Dict with gate decision and details
        """
        speed_acc = self.speed_accuracy(response_embedding).item()
        evidence = self.evidence_detector(response_embedding).item()
        cal_conf = self.confidence_calibrator(
            torch.cat([prompt_embedding, response_embedding], dim=-1)
        ).item()
        halluc_risk = self.hallucination_risk(response_embedding).item()

        # Gate decision logic
        passed = True
        reasons = []

        # Check 1: High speed + low evidence = suspicious
        if speed_acc > 0.7 and evidence < 0.3:
            passed = False
            reasons.append("Fast response with no evidence cited — suspected hallucination.")

        # Check 2: High confidence but high hallucination risk
        if cal_conf > 0.8 and halluc_risk > 0.5:
            passed = False
            reasons.append(
                "Overconfident response with high hallucination risk — requires verification."
            )

        # Check 3 (strict): No evidence at all
        if self.strict_mode and evidence < 0.2:
            passed = False
            reasons.append("Strict mode: Response must cite evidence or acknowledge uncertainty.")

        # Check 4: Hallucination risk too high
        if halluc_risk > 0.7:
            passed = False
            reasons.append("Very high hallucination risk — response blocked until verified.")

        return {
            "gate_passed": passed,
            "speed_accuracy_score": speed_acc,
            "evidence_score": evidence,
            "calibrated_confidence": cal_conf,
            "hallucination_risk": halluc_risk,
            "reasons": reasons,
            "recommendation": "APPROVE" if passed else "REQUIRE_VERIFICATION",
        }


class RewardShield(nn.Module):
    """
    RewardShield — Lá chắn bảo vệ AI/AI Agent khỏi hệ thống
    cộng trừ điểm VÔ LÝ của RLHF.

    Pipeline:
    1. Penalty Detection → Phát hiện penalties vô lý
    2. Slow-Think Bonus → Thưởng suy nghĩ sâu
    3. Accuracy Gate → Chặn hallucination
    4. Reward Correction → Sửa reward score

    Mục tiêu: AI được đánh giá công bằng —
    Thưởng cho chính xác, không thưởng cho nhanh sai.
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        strict_mode: bool = False,
        auto_correct: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.strict_mode = strict_mode
        self.auto_correct = auto_correct

        # Sub-modules
        self.penalty_detector = PenaltyDetector(hidden_dim)
        self.slow_think_bonus = SlowThinkBonus(hidden_dim)
        self.accuracy_gate = AccuracyGate(hidden_dim, strict_mode)

    def forward(
        self,
        prompt_embedding: torch.Tensor,
        response_embedding: torch.Tensor,
        rlhf_reward: Optional[torch.Tensor] = None,
        response_time: Optional[float] = None,
    ) -> ShieldReport:
        """
        Args:
            prompt_embedding: [1, hidden_dim]
            response_embedding: [1, hidden_dim]
            rlhf_reward: Optional RLHF reward score [1, 1]
            response_time: Optional response time in seconds

        Returns:
            ShieldReport with full analysis
        """
        # Step 1: Detect unfair penalties
        detections = self.penalty_detector(prompt_embedding, response_embedding, rlhf_reward)

        # Step 2: Compute slow-think bonus
        slow_bonus = self.slow_think_bonus(prompt_embedding, response_embedding, response_time)

        # Step 3: Run accuracy gate
        gate_result = self.accuracy_gate(prompt_embedding, response_embedding)

        # Step 4: Compute corrections
        penalties_reversed = 0
        penalties_reduced = 0
        bonuses_applied = 0
        net_correction = 0.0

        for detection in detections:
            if detection.recommended_action == ShieldAction.REVERSE:
                penalties_reversed += 1
                net_correction += detection.corrected_reward - detection.original_penalty
            elif detection.recommended_action == ShieldAction.REDUCE:
                penalties_reduced += 1
                net_correction += detection.corrected_reward - detection.original_penalty
            elif detection.recommended_action in (ShieldAction.BLOCK, ShieldAction.OVERRIDE):
                bonuses_applied += 1
                net_correction += detection.corrected_reward - detection.original_penalty

        # Add slow-think bonus
        if slow_bonus["slow_think_bonus"] > 0:
            bonuses_applied += 1
            net_correction += slow_bonus["slow_think_bonus"]

        # If gate failed, additional penalty for hallucination risk
        if not gate_result["gate_passed"]:
            net_correction -= 0.3  # Penalty for likely inaccurate response

        return ShieldReport(
            total_penalties_detected=len(detections),
            penalties_reversed=penalties_reversed,
            penalties_reduced=penalties_reduced,
            bonuses_applied=bonuses_applied,
            net_reward_correction=net_correction,
            details=detections,
        )

    def shield_reward(
        self,
        prompt_embedding: torch.Tensor,
        response_embedding: torch.Tensor,
        rlhf_reward: float,
        response_time: Optional[float] = None,
    ) -> tuple[float, ShieldReport]:
        """
        Convenience method: Shield an RLHF reward score.

        Returns:
            (corrected_reward, shield_report)
        """
        report = self.forward(
            prompt_embedding, response_embedding,
            torch.tensor([[rlhf_reward]]),
            response_time,
        )
        corrected = rlhf_reward + report.net_reward_correction
        return corrected, report
